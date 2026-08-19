"""Stage 3: rigorous global cleaning for vote-level tables before clustering / modelling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from dao_governance.data.validation import CleanStats, clean_master_votes


@dataclass
class GlobalCleanSummary:
    votes_rows_in: int = 0
    votes_rows_out: int = 0
    proposals_rows_in: int = 0
    proposals_rows_out: int = 0
    negative_vp_rows_raw: int = 0
    clipped_negative_rows: int = 0
    extreme_vp_flagged: int = 0
    proposal_duplicate_rows_removed: int = 0
    notes: List[str] = field(default_factory=list)


def _standardize_vote_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common raw export column names to canonical pipeline names."""
    out = df.copy()
    rename_map = {
        "Space": "space",
        "Proposal ID": "proposal_id",
        "Proposal Title": "proposal_title",
        "Proposal Body": "proposal_body",
        "Voter": "voter",
        "Choice": "choice",
        "Voting Power": "voting_power",
        "VP Ratio (%)": "vp_ratio_pct",
        "Is Whale": "is_whale",
        "Aligned With Majority": "aligned_with_majority",
        "Vote Timestamp": "vote_timestamp",
        "Created Time": "created_time",
        "Vote Label": "vote_label",
        "Original Choice": "original_choice",
    }
    for src, dst in rename_map.items():
        if src not in out.columns:
            continue
        if dst not in out.columns:
            out = out.rename(columns={src: dst})
            continue
        # If both alias and canonical exist, preserve canonical and backfill nulls.
        out[dst] = out[dst].where(out[dst].notna(), out[src])
        out = out.drop(columns=[src])
    return out


def _ensure_choice_norm_like_prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "choice_norm" not in out.columns and "choice" in out.columns:
        out["choice_norm"] = out["choice"].astype(str).str.lower().str.strip()
    if "vote_timestamp" not in out.columns and "created_time" in out.columns:
        out["vote_timestamp"] = out["created_time"]
    if "created_time" not in out.columns and "vote_timestamp" in out.columns:
        out["created_time"] = out["vote_timestamp"]
    return out


def clean_votes_table(
    df: pd.DataFrame,
    *,
    dedupe_keys: List[str],
    valid_choice_norm: List[str],
    drop_invalid_timestamp: bool,
    negative_policy: str,
    extreme_quantile: float,
    flag_extreme: bool,
) -> Tuple[pd.DataFrame, CleanStats, GlobalCleanSummary]:
    summary = GlobalCleanSummary()
    summary.votes_rows_in = len(df)
    work = _ensure_choice_norm_like_prepare(df)

    if "voting_power" in work.columns:
        vp = pd.to_numeric(work["voting_power"], errors="coerce")
        inf_mask = np.isinf(vp.to_numpy(dtype=float, copy=False))
        vp = vp.mask(inf_mask, np.nan)
        neg_mask = vp < 0
        summary.negative_vp_rows_raw = int(neg_mask.sum())
        if negative_policy == "clip_to_zero":
            clipped = neg_mask.fillna(False) & vp.notna()
            summary.clipped_negative_rows = int(clipped.sum())
            vp = vp.clip(lower=0.0)
            work["voting_power"] = vp
        elif negative_policy == "drop":
            vp = vp.mask(neg_mask, np.nan)
            work["voting_power"] = vp
        else:
            summary.notes.append(f"Unknown negative_voting_power_policy={negative_policy!r}; using drop.")
            vp = vp.mask(neg_mask, np.nan)
            work["voting_power"] = vp

    treat_neg_invalid = negative_policy != "clip_to_zero"

    cleaned, stats = clean_master_votes(
        work,
        dedupe_keys=dedupe_keys,
        valid_choice_norm=valid_choice_norm,
        drop_invalid_timestamp=drop_invalid_timestamp,
        treat_negative_vp_invalid=treat_neg_invalid,
    )

    if flag_extreme and "voting_power" in cleaned.columns:
        vp2 = pd.to_numeric(cleaned["voting_power"], errors="coerce")
        thr = float(vp2.quantile(extreme_quantile)) if vp2.notna().any() else float("nan")
        if np.isfinite(thr):
            extreme = vp2 > thr
            summary.extreme_vp_flagged = int(extreme.sum())
            cleaned = cleaned.copy()
            cleaned["vp_extreme_flag"] = extreme.astype("int8")
            summary.notes.append(
                f"Flagged voting_power > q={extreme_quantile:.6g} (threshold={thr:g})."
            )

    summary.votes_rows_out = len(cleaned)
    return cleaned, stats, summary


def clean_proposals_table(df: pd.DataFrame, *, id_cols: List[str]) -> Tuple[pd.DataFrame, GlobalCleanSummary]:
    summary = GlobalCleanSummary()
    summary.proposals_rows_in = len(df)
    keys = [k for k in id_cols if k in df.columns]
    out = df.copy()
    if keys:
        before = len(out)
        out = out.drop_duplicates(subset=keys, keep="first")
        summary.proposal_duplicate_rows_removed = before - len(out)
    summary.proposals_rows_out = len(out)
    return out, summary


def write_stage_report(path: Path, title: str, stats: CleanStats, extra: GlobalCleanSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "## Vote-level summary",
        f"- Rows in: **{extra.votes_rows_in}**",
        f"- Rows out: **{extra.votes_rows_out}**",
        f"- Negative VP rows (raw input): **{extra.negative_vp_rows_raw}**",
        f"- Clipped negative VP rows (clip_to_zero policy): **{extra.clipped_negative_rows}**",
        f"- Extreme VP flagged: **{extra.extreme_vp_flagged}**",
        "",
        "## Proposal-level summary",
        f"- Rows in: **{extra.proposals_rows_in}**",
        f"- Rows out: **{extra.proposals_rows_out}**",
        f"- Duplicate proposals removed: **{extra.proposal_duplicate_rows_removed}**",
        "",
        "## clean_master_votes",
        f"- Duplicates removed: **{stats.duplicates_removed}**",
        f"- Invalid timestamp dropped: **{stats.invalid_timestamp_dropped}**",
        f"- Invalid choice dropped: **{stats.invalid_choice_dropped}**",
        f"- Negative VP set missing (invalid): **{stats.negative_voting_power_count}**",
        f"- Inf VP set missing: **{stats.inf_voting_power_count}**",
        f"- NaN VP after clean: **{stats.nan_voting_power_count}**",
        "",
        "## Notes",
    ]
    for n in extra.notes:
        lines.append(f"- {n}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_under_root(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def run_global_cleaning_stage(cfg: Dict[str, Any], root: Path) -> Path:
    paths = cfg.get("paths", {})
    dq = cfg.get("data_quality", {})
    cl = cfg.get("cleaning", {})

    raw_rel = paths.get("master_votes_parquet") or paths.get("snapshot_votes_parquet")
    if not raw_rel:
        raise ValueError("Configure paths.master_votes_parquet or snapshot_votes_parquet.")
    raw_path = _resolve_under_root(root, str(raw_rel))
    if not raw_path.exists():
        raise FileNotFoundError(f"Votes parquet not found: {raw_path}")

    if raw_path.suffix.lower() == ".parquet":
        votes = pd.read_parquet(raw_path)
    else:
        votes = pd.read_csv(raw_path, low_memory=False)
    votes = _standardize_vote_columns(votes)

    dedupe = dq.get("dedupe_keys") or ["voter", "space", "proposal_id"]
    valid = dq.get("valid_choice_norm") or ["for", "against", "abstain"]

    cleaned_votes, stats, gsum = clean_votes_table(
        votes,
        dedupe_keys=list(dedupe),
        valid_choice_norm=list(valid),
        drop_invalid_timestamp=dq.get("drop_rows_with_invalid_timestamp", True),
        negative_policy=str(cl.get("negative_voting_power_policy", "drop")),
        extreme_quantile=float(cl.get("extreme_vp_quantile", 0.999)),
        flag_extreme=bool(cl.get("flag_extreme_voting_power", True)),
    )

    proposals_path_rel = (paths.get("proposals_parquet") or "").strip()
    if proposals_path_rel:
        pp = _resolve_under_root(root, proposals_path_rel)
        if pp.exists():
            prop = pd.read_parquet(pp) if pp.suffix.lower() == ".parquet" else pd.read_csv(pp)
            cleaned_prop, psum = clean_proposals_table(prop, id_cols=["space", "proposal_id"])
            gsum.proposals_rows_in = psum.proposals_rows_in
            gsum.proposals_rows_out = psum.proposals_rows_out
            gsum.proposal_duplicate_rows_removed = psum.proposal_duplicate_rows_removed
            out_prop = (root / paths.get("cleaned_proposals_parquet", "data/processed/proposals_cleaned.parquet")).resolve()
            out_prop.parent.mkdir(parents=True, exist_ok=True)
            cleaned_prop.to_parquet(out_prop, index=False)

    out_votes = (root / paths.get("cleaned_master_parquet", "data/processed/votes_cleaned.parquet")).resolve()
    out_votes.parent.mkdir(parents=True, exist_ok=True)
    cleaned_votes.to_parquet(out_votes, index=False)

    report_path = (root / cfg.get("reports", {}).get("stage03_md", "outputs/reports/stage03_global_cleaning.md")).resolve()
    write_stage_report(report_path, "Stage 3 — Global cleaning", stats, gsum)
    return out_votes
