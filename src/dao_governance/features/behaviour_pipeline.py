"""
Orchestration for leakage-safe behaviour modelling (split → fit clusters → assign).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from dao_governance.features.causal_clusters import (
    ClusterBundle,
    assign_clusters_to_votes,
    fit_cluster_bundle,
    save_cluster_report,
)
from dao_governance.modelling.preprocess import (
    fit_numeric_preprocessor,
    normalise_columns,
    split_by_voter_three_way,
)


def prepare_behaviour_splits(
    raw_df: pd.DataFrame,
    *,
    dao_feature_table_path: Path,
    train_frac: float,
    val_frac: float,
    seed: int,
    upper_quantile_cap: float,
    absolute_cap: float,
    use_dao_clusters: bool = True,
    use_voter_clusters: bool = True,
    min_votes_per_pair: int = 5,
    cluster_artifacts_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], ClusterBundle, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leakage-safe pipeline:
      1. voter split (before any cluster fit)
      2. fit numeric preprocessor on train
      3. fit structural clusters on train only
      4. assign clusters to all splits (transform only)
      5. normalise numeric columns
    """
    train_raw, val_raw, test_raw = split_by_voter_three_way(
        raw_df, train_frac=train_frac, val_frac=val_frac, seed=seed
    )

    preprocessor = fit_numeric_preprocessor(
        train_raw, upper_quantile_cap=upper_quantile_cap, absolute_cap=absolute_cap
    )

    bundle = fit_cluster_bundle(
        train_raw,
        dao_feature_table_path,
        use_dao_clusters=use_dao_clusters,
        use_voter_clusters=use_voter_clusters,
        min_votes_per_pair=min_votes_per_pair,
    )

    if cluster_artifacts_dir is not None:
        cluster_artifacts_dir.mkdir(parents=True, exist_ok=True)
        bundle.save(cluster_artifacts_dir / "cluster_bundle.pkl")
        save_cluster_report(cluster_artifacts_dir / "cluster_report.md", bundle)

    train_assigned = assign_clusters_to_votes(
        train_raw, bundle, dao_feature_table_path, min_votes_per_pair=min_votes_per_pair
    )
    val_assigned = assign_clusters_to_votes(
        val_raw, bundle, dao_feature_table_path, min_votes_per_pair=min_votes_per_pair
    )
    test_assigned = assign_clusters_to_votes(
        test_raw, bundle, dao_feature_table_path, min_votes_per_pair=min_votes_per_pair
    )

    train_df = normalise_columns(train_assigned, preprocessor=preprocessor)
    val_df = normalise_columns(val_assigned, preprocessor=preprocessor)
    test_df = normalise_columns(test_assigned, preprocessor=preprocessor)

    return train_df, val_df, test_df, preprocessor, bundle, train_assigned, val_assigned, test_assigned


def load_behaviour_votes(csv_path: Path) -> pd.DataFrame:
    """Load behaviour CSV and strip legacy cluster / leaky columns if present."""
    from dao_governance.modelling.preprocess import load_dataset

    df = load_dataset(csv_path)
    drop = [c for c in ("dao_cluster", "voter_cluster", "aligned_with_majority", "vp_share", "vp_ratio_pct") if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    return df


def write_enriched_behaviour_csv(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    path: Path,
) -> None:
    """Write vote-level CSV with train-only cluster assignments (for window cache)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([train_raw, val_raw, test_raw], ignore_index=True).to_csv(path, index=False)
