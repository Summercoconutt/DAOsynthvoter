#!/usr/bin/env python3
"""Stage 6: DAO clustering validation & exports (baseline = no-outliers assignments)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.settings import load_config, project_root, require_local_data_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--extra-config", type=str, default="", help="Optional YAML merged over --config (e.g. server raw paths).")
    ap.add_argument("--predictive-profiles", type=str, default="")
    ap.add_argument("--split-manifest", type=str, default="")
    ap.add_argument("--cluster-output", type=str, default="")
    ap.add_argument("--cluster-feature", action="append", default=[])
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
        if not args.split_manifest or not args.cluster_output or not args.cluster_feature:
            ap.error("Predictive clustering requires --split-manifest, --cluster-output, and --cluster-feature.")
        from dao_governance.clustering.predictive import fit_predictive_stage

        profiles, split, output = Path(args.predictive_profiles), Path(args.split_manifest), Path(args.cluster_output)
        if local_only:
            profiles = require_local_data_path(profiles, root=base)
            split = require_local_data_path(split, root=base)
            output = require_local_data_path(output, root=base)
        result = fit_predictive_stage(
            profiles_path=profiles,
            split_manifest=split,
            output=output,
            kind="dao",
            scope=None,
            feature_columns=args.cluster_feature,
            history_column="dao_prior_count",
            n_clusters=args.cluster_count,
            min_group_size=args.min_group_size,
            min_history=args.min_history,
            min_cluster_size=args.min_cluster_size,
            algorithm=args.cluster_algorithm,
        )
        print(f"[06] Predictive DAO clustering: {result['fit_summary']}")
        return
    if args.split_manifest or args.cluster_output or args.cluster_feature:
        ap.error("Predictive clustering options require --predictive-profiles.")
    paths = cfg.get("paths", {})
    rep_path = (base / cfg.get("reports", {}).get("stage06_md", "outputs/reports/stage06_dao_clustering.md")).resolve()

    feat_csv = (base / paths.get("dao_feature_table_csv", "data/processed/dao_feature_table.csv")).resolve()
    feat_xlsx = (base / "data" / "processed" / "dao_feature_table.xlsx")
    if feat_csv.exists():
        primary = feat_csv
    elif feat_xlsx.exists():
        primary = feat_xlsx
    else:
        primary = feat_csv
    if local_only:
        primary = require_local_data_path(primary, root=base)
        feat_csv = require_local_data_path(feat_csv, root=base)

    val_out = (base / paths.get("clustering_validation_out_dir", "outputs/clustering_validation")).resolve()
    cr_out = (base / paths.get("cluster_results_dir", "outputs/cluster_results")).resolve()

    env = os.environ.copy()
    env["PIPELINE_DAO_FEATURES_INPUT"] = str(primary)
    env["PIPELINE_DAO_FEATURES_FALLBACK_CSV"] = str(feat_csv)
    env["PIPELINE_CLUSTERING_VALIDATION_OUT"] = str(val_out)
    env["PIPELINE_CLUSTER_RESULTS_DIR"] = str(cr_out)

    script = SRC / "dao_clustering_validation" / "clustering_validation_pipeline.py"
    subprocess.run([sys.executable, str(script)], check=True, env=env, cwd=str(ROOT))

    baseline = cr_out / "no_outliers" / "cluster_assignments.csv"
    lines = [
        "# Stage 6 — DAO clustering",
        "",
        f"- Feature input used: `{primary}`",
        f"- Validation bundle: `{val_out}`",
        f"- Cluster exports: `{cr_out}`",
        "",
        "## Baseline for downstream voter modelling",
        f"- Preferred assignments CSV: `{baseline}`",
        "",
        "Align `paths.dao_cluster_assignments_csv` in `configs/default.yaml` with this file.",
        "",
    ]
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[06] Report: {rep_path}")


if __name__ == "__main__":
    main()
