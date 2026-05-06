#!/usr/bin/env python3
"""Stage 5: DAO feature selection (correlation, variance, missingness)."""

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
    args = ap.parse_args()
    base = project_root()
    cfg_path = (base / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(config_path=cfg_path)
    paths = cfg.get("paths", {})
    cl = cfg.get("clustering", {})
    rep_path = (base / cfg.get("reports", {}).get("stage05_md", "outputs/reports/stage05_feature_selection.md")).resolve()

    inp = (base / paths.get("dao_feature_table_csv", "data/processed/dao_feature_table.csv")).resolve()
    if not inp.exists():
        pq = (base / paths.get("dao_feature_table_parquet", "data/processed/dao_feature_table.parquet")).resolve()
        if pq.exists():
            import pandas as pd

            pd.read_parquet(pq).to_csv(inp, index=False)

    out_dir = (base / paths.get("feature_selection_out_dir", "outputs/feature_screening")).resolve()
    script = SRC / "feature_selection" / "feature_selection_pipeline.py"
    corr = float(cl.get("corr_threshold_feature_selection", 0.85))

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(inp),
            "--out-dir",
            str(out_dir),
            "--corr-threshold",
            str(corr),
        ],
        check=True,
        cwd=str(base),
    )

    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(
        "\n".join(
            [
                "# Stage 5 — Feature selection",
                "",
                f"- Input table: `{inp}`",
                f"- Output directory: `{out_dir}`",
                f"- Correlation threshold: **{corr}**",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[05] Report: {rep_path}")


if __name__ == "__main__":
    main()
