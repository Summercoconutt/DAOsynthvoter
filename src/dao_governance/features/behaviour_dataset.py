"""Build behaviour modelling CSV from master votes (no pre-split cluster merge)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dao_governance.data.validation import DataQualityReport, behaviour_dataset_quality_report

BEHAVIOUR_REQUIRED_COLUMNS = [
    "voter",
    "space",
    "proposal_id",
    "choice_norm",
    "vote_timestamp",
    "vote_ts",
    "proposal_title",
    "proposal_body",
    "Proposal Body",
    "voting_power",
    "is_whale",
]

LABEL_MAP = {"for": 0, "against": 1, "abstain": 2}
TEXT_MODES = frozenset({"title", "title_body"})


def _normalise_bool(col: pd.Series) -> pd.Series:
    def to_bool(x):
        if isinstance(x, str):
            return x.strip().lower() in {"1", "true", "yes", "y", "t"}
        return bool(x)

    return col.apply(to_bool)


def _text_series(df: pd.DataFrame, text_mode: str) -> pd.Series:
    if text_mode not in TEXT_MODES:
        raise ValueError(f"Unsupported text_mode={text_mode!r}; expected one of {sorted(TEXT_MODES)}")

    title = df.get("proposal_title", pd.Series("", index=df.index)).fillna("").astype(str)
    if text_mode == "title":
        return title

    body_col = "proposal_body" if "proposal_body" in df.columns else "Proposal Body"
    if body_col not in df.columns:
        raise ValueError(
            "text_mode='title_body' requires 'proposal_body' or 'Proposal Body' in the master votes parquet."
        )
    body = df[body_col].fillna("").astype(str)
    return "[TITLE] " + title + " [BODY] " + body


def build_behaviour_dataset(
    master_votes_parquet: Path,
    voter_cluster_assignments_csv: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    *,
    include_legacy_clusters: bool = False,
    text_mode: str = "title",
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Vote-level behaviour table without cluster IDs.

    Cluster assignment is deferred to Stage 8 (train-only fit via causal_clusters).
    Set include_legacy_clusters=True only for deprecated / exploratory runs.
    """
    available_columns = set(pq.ParquetFile(master_votes_parquet).schema.names)
    parquet_columns = [c for c in BEHAVIOUR_REQUIRED_COLUMNS if c in available_columns]
    if not parquet_columns:
        raise ValueError(f"No supported behaviour columns found in parquet schema: {sorted(available_columns)}")
    votes = pd.read_parquet(master_votes_parquet, columns=parquet_columns)
    df = votes.copy()

    if not include_legacy_clusters:
        for col in ("dao_cluster", "voter_cluster"):
            if col in df.columns:
                df = df.drop(columns=[col])
    elif voter_cluster_assignments_csv is not None and voter_cluster_assignments_csv.exists():
        assign = pd.read_csv(voter_cluster_assignments_csv)
        if "clustering_method" in assign.columns:
            kmeans = assign[assign["clustering_method"] == "kmeans"].copy()
            if not kmeans.empty:
                assign = kmeans
        keep_assign_cols = ["voter", "space", "voter_cluster"]
        assign = assign[keep_assign_cols].drop_duplicates(subset=["voter", "space"])
        df = df.merge(assign, on=["voter", "space"], how="left")
        df["voter_cluster"] = pd.to_numeric(df["voter_cluster"], errors="coerce").fillna(-1).astype(int)
        if "dao_cluster" not in df.columns:
            df["dao_cluster"] = -1
        else:
            df["dao_cluster"] = pd.to_numeric(df["dao_cluster"], errors="coerce").fillna(-1).astype(int)

    df["choice_norm"] = df["choice_norm"].astype(str).str.lower().str.strip()
    df["label_id"] = df["choice_norm"].map(LABEL_MAP).astype("Int64")
    before_labels = len(df)
    df = df[df["label_id"].isin([0, 1, 2])].copy()
    dropped_bad_label = before_labels - len(df)

    df["vote_ts"] = pd.to_datetime(
        df["vote_timestamp"] if "vote_timestamp" in df.columns else df.get("vote_ts"),
        utc=True,
        errors="coerce",
    )
    df["text"] = _text_series(df, text_mode)
    df["voting_power"] = pd.to_numeric(df.get("voting_power", np.nan), errors="coerce")
    df["is_whale"] = _normalise_bool(df.get("is_whale", False))

    numeric_cols = ["voting_power", "is_whale"]
    report = behaviour_dataset_quality_report(df, numeric_cols)
    if dropped_bad_label:
        report.stats.notes.append(f"Dropped rows with non-{list(LABEL_MAP.keys())} choice_norm: {dropped_bad_label}")
    report.stats.notes.append(
        "Clusters not merged at build time; Stage 8 fits train-only structural clusters."
    )
    report.stats.notes.append(f"Text mode: {text_mode}")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df, report


def write_behaviour_dataset_with_report(
    master_parquet: Path,
    assignments_csv: Optional[Path],
    output_csv: Path,
    report_md_path: Optional[Path] = None,
    *,
    include_legacy_clusters: bool = False,
    text_mode: str = "title",
) -> pd.DataFrame:
    df, rep = build_behaviour_dataset(
        master_parquet,
        assignments_csv,
        output_csv=output_csv,
        include_legacy_clusters=include_legacy_clusters,
        text_mode=text_mode,
    )
    if report_md_path is not None:
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_md_path.write_text(rep.to_markdown("Behaviour dataset quality"), encoding="utf-8")
    return df
