#!/usr/bin/env python3
"""Run dissertation stages 01–08 sequentially (stop on first failure)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "01_run_data_selection.py",
    "02_run_data_expansion.py",
    "03_run_global_cleaning.py",
    "04_run_dao_metrics.py",
    "05_run_feature_selection.py",
    "06_run_dao_clustering.py",
    "07_run_voter_clustering.py",
    "08_run_behaviour_modelling.py",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument(
        "--extra-config",
        type=str,
        default="",
        help="Optional Stage 08 override; for stages 01–07, combine overrides into the data-source YAML.",
    )
    ap.add_argument("--data-sources-config", type=str, default="configs/data_sources.yaml")
    ap.add_argument("--run-id", type=str, default="", help="Unique Stage 08 output ID; generated if omitted.")
    ap.add_argument("--from-stage", type=int, default=1, help="Start at stage N (1–8).")
    ap.add_argument("--to-stage", type=int, default=8, help="End at stage N (1–8).")
    args = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    data_sources_path = Path(args.data_sources_config)
    if not data_sources_path.is_absolute():
        data_sources_path = ROOT / data_sources_path
    uses_default_sources = data_sources_path.resolve() == (ROOT / "configs" / "data_sources.yaml").resolve()

    lo = max(1, min(8, args.from_stage))
    hi = max(1, min(8, args.to_stage))
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    print(f"Stage 08 run ID: {run_id}")
    if hi < lo:
        lo, hi = hi, lo
    if lo < 8 and args.extra_config:
        ap.error(
            "Stages 01–07 accept one extra YAML only; combine source and other overrides into one file, "
            "or run from Stage 08 with separate --data-sources-config and --extra-config options."
        )

    for i in range(lo, hi + 1):
        script = ROOT / "scripts" / SCRIPTS[i - 1]
        print(f"\n=== Stage {i}: {script.name} ===\n")
        cmd = [sys.executable, str(script), "--config", args.config]
        if i < 8 and args.extra_config:
            cmd.extend(["--extra-config", args.extra_config])
        elif i < 8 and not uses_default_sources:
            cmd.extend(["--extra-config", args.data_sources_config])
        if i == 8:
            cmd.extend(["--data-sources-config", args.data_sources_config, "--run-id", run_id])
            if args.extra_config:
                cmd.extend(["--extra-config", args.extra_config])
        subprocess.run(
            cmd,
            check=True,
            cwd=str(ROOT),
            env=env,
        )


if __name__ == "__main__":
    main()
