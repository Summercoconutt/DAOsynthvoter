"""Robust validation and cleaning for governance vote tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class CleanStats:
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    invalid_timestamp_dropped: int = 0
    invalid_choice_dropped: int = 0
    negative_voting_power_count: int = 0
    inf_voting_power_count: int = 0
    nan_voting_power_count: int = 0
    extreme_voting_power_winsorised: int = 0
    empty_text_filled: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    stats: CleanStats
    class_balance: Dict[str, float] = field(default_factory=dict)
    zero_variance_features: List[str] = field(default_factory=list)

    def to_markdown(self, title: str = "Data quality report") -> str:
        s = self.stats
        lines = [
            f"# {title}",
            "",
            "## Row counts",
            f"- Rows in: **{s.rows_in}**",
            f"- Rows out: **{s.rows_out}**",
            f"- Duplicates removed (by key): **{s.duplicates_removed}**",
            f"- Dropped (invalid timestamp): **{s.invalid_timestamp_dropped}**",
            f"- Dropped (invalid vote choice): **{s.invalid_choice_dropped}**",
            "",
            "## Voting power",
            f"- Negative (invalid, set missing): **{s.negative_voting_power_count}**",
            f"- Infinite: **{s.inf_voting_power_count}**",
            f"- NaN (before imputation in modelling): **{s.nan_voting_power_count}**",
            f"- Extreme values winsorised at upper quantile (preprocessor): **{s.extreme_voting_power_winsorised}**",
            "",
            "## Class balance (label proportions)",
        ]
        for k, v in self.class_balance.items():
            lines.append(f"- {k}: **{v:.4f}**")
        if self.zero_variance_features:
            lines.extend(["", "## Zero-variance numeric columns (after cleaning)", ", ".join(self.zero_variance_features)])
        if s.notes:
            lines.extend(["", "## Notes"])
            for n in s.notes:
                lines.append(f"- {n}")
        lines.append("")
        return "\n".join(lines)


def _norm_choice_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()


def clean_master_votes(
    df: pd.DataFrame,
    *,
    dedupe_keys: Sequence[str],
    valid_choice_norm: Sequence[str],
    drop_invalid_timestamp: bool,
    treat_negative_vp_invalid: bool,
) -> Tuple[pd.DataFrame, CleanStats]:
    """
    Clean raw / semi-standard master vote rows. Does not apply modelling preprocessor;
    only structural hygiene and explicit invalid handling.
    """
    stats = CleanStats()
    stats.rows_in = len(df)
    out = df.copy()

    keys = [k for k in dedupe_keys if k in out.columns]
    if keys:
        before = len(out)
        out = out.drop_duplicates(subset=keys, keep="first")
        stats.duplicates_removed = before - len(out)

    if "vote_timestamp" in out.columns:
        ts = pd.to_datetime(out["vote_timestamp"], utc=True, errors="coerce")
        bad = ts.isna()
        n_bad = int(bad.sum())
        if drop_invalid_timestamp and n_bad:
            out = out.loc[~bad].copy()
            stats.invalid_timestamp_dropped = n_bad
        else:
            out["vote_timestamp"] = ts
            stats.notes.append(f"Invalid timestamps kept as NaT: {n_bad} rows")
    elif "vote_ts" in out.columns:
        out["vote_timestamp"] = pd.to_datetime(out["vote_ts"], utc=True, errors="coerce")

    if "choice_norm" in out.columns:
        cn = _norm_choice_series(out["choice_norm"])
        valid_set = set(x.lower() for x in valid_choice_norm)
        mask = cn.isin(valid_set)
        stats.invalid_choice_dropped = int((~mask).sum())
        out = out.loc[mask].copy()
        out["choice_norm"] = _norm_choice_series(out["choice_norm"])

    if "voting_power" in out.columns:
        vp = pd.to_numeric(out["voting_power"], errors="coerce")
        stats.nan_voting_power_count = int(vp.isna().sum())
        inf_mask = np.isinf(vp)
        stats.inf_voting_power_count = int(inf_mask.sum())
        vp = vp.mask(inf_mask, np.nan)
        if treat_negative_vp_invalid:
            neg = vp < 0
            stats.negative_voting_power_count = int(neg.sum())
            vp = vp.mask(neg, np.nan)
        out["voting_power"] = vp

    for text_col in ("proposal_title", "text"):
        if text_col in out.columns:
            empty = out[text_col].isna() | (out[text_col].astype(str).str.strip() == "")
            stats.empty_text_filled += int(empty.sum())
            out[text_col] = out[text_col].fillna("").astype(str)

    stats.rows_out = len(out)
    return out, stats


def behaviour_dataset_quality_report(df: pd.DataFrame, numeric_cols: Sequence[str]) -> DataQualityReport:
    """Summaries for behaviour modelling CSV after label mapping."""
    stats = CleanStats(rows_in=len(df), rows_out=len(df))
    bal: Dict[str, float] = {}
    if "label_id" in df.columns and df["label_id"].notna().any():
        vc = df["label_id"].value_counts(normalize=True)
        for i, name in enumerate(["FOR", "AGAINST", "ABSTAIN"]):
            bal[name] = float(vc.get(i, 0.0))
    zv: List[str] = []
    for c in numeric_cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() > 1 and float(x.std(ddof=0) or 0.0) == 0.0:
            zv.append(c)
    return DataQualityReport(stats=stats, class_balance=bal, zero_variance_features=zv)


def validate_finite_features(arr: np.ndarray, *, context: str = "") -> None:
    """Raise if array contains NaN or Inf (model inputs)."""
    if not np.isfinite(arr).all():
        bad = ~np.isfinite(arr)
        idx = np.argwhere(bad)
        raise ValueError(
            f"Non-finite values in numeric features{(': ' + context) if context else ''}. "
            f"First bad positions (flat index): {idx[:10].tolist()}"
        )
