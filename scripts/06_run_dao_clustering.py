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

from dao_governance.settings import load_config, project_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--extra-config", type=str, default="", help="Optional YAML merged over --config (e.g. server raw paths).")
    args = ap.parse_args()
    base = project_root()
    cfg_path = (base / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=cfg_path, extra_path=extra)
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
