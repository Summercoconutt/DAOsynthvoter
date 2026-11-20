#!/usr/bin/env python3
"""Stage 7: Voter-space features + voter clustering (uses cleaned votes + DAO assignments)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.clustering.voter_pipeline import run_voter_clustering_stage, run_voter_feature_stage
from dao_governance.settings import load_config, project_root, require_local_data_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--extra-config", type=str, default="", help="Optional YAML merged over --config (e.g. server raw paths).")
    ap.add_argument("--predictive-profiles", type=str, default="")
    ap.add_argument("--split-manifest", type=str, default="")
    ap.add_argument("--cluster-output", type=str, default="")
    ap.add_argument("--cluster-feature", action="append", default=[])
    ap.add_argument("--cluster-scope", choices=("global", "dao_cluster", "individual_dao"))
    ap.add_argument("--dao-bundle", type=str, default="")
    ap.add_argument("--cluster-count", type=int, default=2)
    ap.add_argument("--min-group-size", type=int, default=5)
    ap.add_argument("--min-history", type=int, default=1)
    ap.add_argument("--min-cluster-size", type=int, default=1)
    ap.add_argument("--cluster-algorithm", choices=("kmeans", "gaussian_mixture"), default="kmeans")
    args = ap.parse_args()
    base = project_root()
    cfg_path = (base / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=cfg_path, extra_path=extra)
    local_only = bool((cfg.get("run") or {}).get("local_data_only", False))
    if args.predictive_profiles:
        if not args.split_manifest or not args.cluster_output or not args.cluster_feature or not args.cluster_scope:
            ap.error("Predictive clustering requires --split-manifest, --cluster-output, --cluster-feature, and --cluster-scope.")
        from dao_governance.clustering.predictive import fit_predictive_stage

        profiles, split, output = Path(args.predictive_profiles), Path(args.split_manifest), Path(args.cluster_output)
        dao_bundle = Path(args.dao_bundle) if args.dao_bundle else None
        if local_only:
            profiles = require_local_data_path(profiles, root=base)
            split = require_local_data_path(split, root=base)
            output = require_local_data_path(output, root=base)
            dao_bundle = require_local_data_path(dao_bundle, root=base) if dao_bundle is not None else None
        result = fit_predictive_stage(
            profiles_path=profiles,
            split_manifest=split,
            output=output,
            kind="voter",
            scope=args.cluster_scope,
            feature_columns=args.cluster_feature,
            history_column="voter_dao_prior_count",
            n_clusters=args.cluster_count,
            min_group_size=args.min_group_size,
            min_history=args.min_history,
            min_cluster_size=args.min_cluster_size,
            algorithm=args.cluster_algorithm,
            dao_bundle_path=dao_bundle,
        )
        print(f"[07] Predictive voter clustering: {result['fit_summary']}")
        return
    if args.split_manifest or args.cluster_output or args.cluster_feature or args.cluster_scope or args.dao_bundle:
        ap.error("Predictive clustering options require --predictive-profiles.")
    paths = cfg.get("paths", {})
    vc = cfg.get("voter_clustering", {})
    rep_path = (base / cfg.get("reports", {}).get("stage07_md", "outputs/reports/stage07_voter_clustering.md")).resolve()

    cleaned = (base / paths.get("cleaned_master_parquet", "data/processed/votes_cleaned.parquet")).resolve()
    master_parquet = cleaned if cleaned.exists() else (base / paths.get("master_votes_parquet")).resolve()
    if local_only:
        master_parquet = require_local_data_path(master_parquet, root=base)
    if not master_parquet.exists():
        raise FileNotFoundError(f"Need cleaned or raw master votes parquet: {master_parquet}")

    dao_csv = (base / paths.get("dao_cluster_assignments_csv", "outputs/cluster_results/no_outliers/cluster_assignments.csv")).resolve()
    if local_only:
        dao_csv = require_local_data_path(dao_csv, root=base)
    if not dao_csv.exists():
        raise FileNotFoundError(f"DAO cluster assignments missing: {dao_csv}. Run stage 6.")

    out_merged = (base / paths.get("master_with_dao_parquet", "data/processed/master_votes_with_dao_cluster.parquet")).resolve()
    feat = (base / paths.get("voter_features_csv", "data/processed/voter_space_features.csv")).resolve()
    feat_f = (base / paths.get("voter_features_filtered_csv", "data/processed/voter_space_features_filtered.csv")).resolve()
    assign = (base / paths.get("voter_cluster_assignments_csv", "outputs/voter_clustering/voter_cluster_assignments.csv")).resolve()
    summary = (base / paths.get("voter_clustering_summary_csv", "outputs/voter_clustering/cluster_summary_statistics.csv")).resolve()
    figures = (base / paths.get("voter_clustering_figures_dir", "outputs/voter_clustering/figures")).resolve()

    min_v = int(vc.get("min_votes_per_voter_space", 5))
    env_min = (vc.get("env_min_rows_mode_a") or "").strip()
    if env_min:
        os.environ["VOTER_CLUSTER_MIN_ROWS_MODE_A"] = env_min

    for d in (out_merged.parent, feat.parent, assign.parent, figures):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[07] Building voter features from {master_parquet}")
    run_voter_feature_stage(
        master_parquet=master_parquet,
        dao_assignments_csv=dao_csv,
        out_with_cluster_parquet=out_merged,
        out_features_csv=feat,
        out_features_filtered_csv=feat_f,
        min_votes=min_v,
    )

    print(f"[07] Clustering voters from {feat_f}")
    run_voter_clustering_stage(
        filtered_features_csv=feat_f,
        assignments_out_csv=assign,
        summary_out_csv=summary,
        figures_dir=figures,
    )

    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(
        "\n".join(
            [
                "# Stage 7 — Voter clustering (exploratory)",
                "",
                "Stage 7 outputs are for **EDA / dissertation figures only**.",
                "Behaviour modelling (Stage 8) fits **train-only structural clusters** internally.",
                "",
                f"- Master with DAO labels: `{out_merged}`",
                f"- Voter features (filtered): `{feat_f}`",
                f"- Exploratory assignments: `{assign}`",
                f"- Summary: `{summary}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[07] Report: {rep_path}")


if __name__ == "__main__":
    main()
