"""Build behaviour modelling CSV from master votes + voter cluster assignments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from dao_governance.data.validation import DataQualityReport, behaviour_dataset_quality_report

LABEL_MAP = {"for": 0, "against": 1, "abstain": 2}


def _normalise_bool(col: pd.Series) -> pd.Series:
    def to_bool(x):
        if isinstance(x, str):
            return x.strip().lower() in {"1", "true", "yes", "y", "t"}
        return bool(x)

    return col.apply(to_bool)


def build_behaviour_dataset(
    master_votes_with_cluster_parquet: Path,
    voter_cluster_assignments_csv: Path,
    output_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    votes = pd.read_parquet(master_votes_with_cluster_parquet)
    assign = pd.read_csv(voter_cluster_assignments_csv)

    if "clustering_method" in assign.columns:
        kmeans = assign[assign["clustering_method"] == "kmeans"].copy()
        if not kmeans.empty:
            assign = kmeans

    keep_assign_cols = ["voter", "space", "voter_cluster"]
    missing = [c for c in keep_assign_cols if c not in assign.columns]
    if missing:
        raise ValueError(f"Missing columns in assignments file: {missing}")
    assign = assign[keep_assign_cols].drop_duplicates(subset=["voter", "space"])

    df = votes.merge(assign, on=["voter", "space"], how="left")
    df["voter_cluster"] = pd.to_numeric(df["voter_cluster"], errors="coerce").fillna(-1).astype(int)
    if "dao_cluster" in df.columns:
        df["dao_cluster"] = pd.to_numeric(df["dao_cluster"], errors="coerce").fillna(-1).astype(int)
    else:
        df["dao_cluster"] = -1

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
    df["text"] = df.get("proposal_title", "").fillna("").astype(str)
    df["voting_power"] = pd.to_numeric(df.get("voting_power", np.nan), errors="coerce")
    df["vp_ratio_pct"] = pd.to_numeric(df.get("vp_ratio_pct", np.nan), errors="coerce")
    df["vp_share"] = df["vp_ratio_pct"] / 100.0
    df["is_whale"] = _normalise_bool(df.get("is_whale", False))
    df["aligned_with_majority"] = _normalise_bool(df.get("aligned_with_majority", False))

    numeric_cols = ["voting_power", "vp_share", "is_whale", "aligned_with_majority", "dao_cluster", "voter_cluster"]
    report = behaviour_dataset_quality_report(df, numeric_cols)
    if dropped_bad_label:
        report.stats.notes.append(f"Dropped rows with non-{list(LABEL_MAP.keys())} choice_norm: {dropped_bad_label}")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df, report


def write_behaviour_dataset_with_report(
    master_parquet: Path,
    assignments_csv: Path,
    output_csv: Path,
    report_md_path: Optional[Path] = None,
) -> pd.DataFrame:
    df, rep = build_behaviour_dataset(master_parquet, assignments_csv, output_csv=output_csv)
    if report_md_path is not None:
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_md_path.write_text(rep.to_markdown("Behaviour dataset quality"), encoding="utf-8")
    return df
