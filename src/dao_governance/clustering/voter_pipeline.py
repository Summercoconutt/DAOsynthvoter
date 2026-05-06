"""
Thin wrappers around `Voter_Clustering` package (import path injected by scripts).
Caller must add `Voter_Clustering/src` to `sys.path` before importing this module's targets,
or import `voter_clustering` directly from scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def run_voter_feature_stage(
    master_parquet: Path,
    dao_assignments_csv: Path,
    out_with_cluster_parquet: Path,
    out_features_csv: Path,
    out_features_filtered_csv: Path,
    min_votes: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from voter_clustering.build_voter_space_features import build_voter_space_features

    return build_voter_space_features(
        master_parquet=master_parquet,
        assignments_csv=dao_assignments_csv,
        out_with_cluster_parquet=out_with_cluster_parquet,
        out_features_csv=out_features_csv,
        out_features_filtered_csv=out_features_filtered_csv,
        min_votes=min_votes,
    )


def run_voter_clustering_stage(
    filtered_features_csv: Path,
    assignments_out_csv: Path,
    summary_out_csv: Path,
    figures_dir: Path,
) -> None:
    from voter_clustering.run_voter_clustering import run_clustering_and_save

    run_clustering_and_save(
        filtered_features_csv=filtered_features_csv,
        assignments_out_csv=assignments_out_csv,
        summary_out_csv=summary_out_csv,
        figures_dir=figures_dir,
    )
