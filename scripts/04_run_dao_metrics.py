#!/usr/bin/env python3
"""Stage 4: DAO metrics — load votes, compute metrics, participation, merge feature table."""

from __future__ import annotations

import argparse
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
    ap.add_argument("--skip-load", action="store_true", help="Skip 01_load_data if votes parquet already exists.")
    args = ap.parse_args()
    base = project_root()
    cfg_path = (base / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=cfg_path, extra_path=extra)
    paths = cfg.get("paths", {})
    report = (base / cfg.get("reports", {}).get("stage04_md", "outputs/reports/stage04_dao_metrics.md")).resolve()

    snap_root = (paths.get("snapshot_spaces_root") or "").strip()
    votes_in = (base / paths.get("cleaned_master_parquet", "data/processed/votes_cleaned.parquet")).resolve()
    if not votes_in.exists():
        votes_in = (base / paths.get("snapshot_votes_parquet", "data/interim/snapshot_votes.parquet")).resolve()
    if not votes_in.exists():
        raise FileNotFoundError(
            f"No votes parquet at cleaned or snapshot path. Run stages 2–3 or set paths. Tried: {votes_in}"
        )

    load_script = SRC / "dao_clustering_scripts" / "01_load_data.py"
    metrics_script = SRC / "dao_clustering_scripts" / "02_compute_metrics.py"
    merge_script = SRC / "dao_clustering_scripts" / "03_build_dao_feature_table.py"
    part_script = SRC / "participation_rate" / "participation_rate_pipeline.py"

    if snap_root and Path(snap_root).is_dir() and not args.skip_load:
        out_raw = (base / paths.get("snapshot_votes_parquet", "data/interim/snapshot_votes.parquet")).resolve()
        out_raw.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(load_script),
                "--base-spaces-dir",
                snap_root,
                "--out",
                str(out_raw),
            ],
            check=True,
        )
        votes_in = out_raw

    dao_metrics = (base / paths.get("dao_metrics_parquet", "data/interim/dao_metrics.parquet")).resolve()
    dao_metrics.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(metrics_script),
            "--in",
            str(votes_in),
            "--out-dao",
            str(dao_metrics),
            "--out-proposal",
            str((base / paths.get("proposal_metrics_parquet", "data/interim/proposal_metrics.parquet")).resolve()),
        ],
        check=True,
    )

    part_out = (base / paths.get("participation_out_dir", "data/interim/participation_rate")).resolve()
    part_out.mkdir(parents=True, exist_ok=True)
    if snap_root and Path(snap_root).is_dir():
        subprocess.run(
            [
                sys.executable,
                str(part_script),
                "--base-spaces-dir",
                snap_root,
                "--out-dir",
                str(part_out),
            ],
            check=True,
        )
    else:
        print("[04] WARN: snapshot_spaces_root empty — skipping participation_rate_pipeline (provide CSV paths manually).")

    rep_csv = (base / paths.get("representative_csv", "data/interim/dao_representative_voters.csv")).resolve()
    if not rep_csv.exists():
        print(
            f"[04] WARN: Representative voter CSV not found at {rep_csv}. "
            "Run `src/representative_voter/dao_representative_voter_pipeline.py` with your Snapshot layout, "
            "then copy/link the output CSV to this path."
        )

    feat_parquet = (base / paths.get("dao_feature_table_parquet", "data/processed/dao_feature_table.parquet")).resolve()
    feat_parquet.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(merge_script),
            "--dao-metrics",
            str(dao_metrics),
            "--participation-summary",
            str(part_out / "dao_level_participation_summary.csv"),
            "--representative-csv",
            str(rep_csv),
            "--out",
            str(feat_parquet),
        ],
        check=True,
    )

    lines = [
        "# Stage 4 — DAO metrics",
        "",
        "## Outputs",
        f"- DAO metrics: `{dao_metrics}`",
        f"- Participation dir: `{part_out}`",
        f"- DAO feature table: `{feat_parquet}`",
        "",
        "## Validation",
        "- Confirm `dao_level_participation_summary.csv` exists under participation dir when participation stage ran.",
        f"- Representative CSV used: `{rep_csv}` (must exist for full merges).",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[04] Report: {report}")


if __name__ == "__main__":
    main()
