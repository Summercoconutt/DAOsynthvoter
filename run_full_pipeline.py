#!/usr/bin/env python3
"""Run dissertation stages 01–08 sequentially (stop on first failure)."""

from __future__ import annotations

import argparse
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
    ap.add_argument("--extra-config", type=str, default="", help="Optional YAML merged over --config (e.g. server raw paths).")
    ap.add_argument("--from-stage", type=int, default=1, help="Start at stage N (1–8).")
    ap.add_argument("--to-stage", type=int, default=8, help="End at stage N (1–8).")
    args = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    lo = max(1, min(8, args.from_stage))
    hi = max(1, min(8, args.to_stage))
    if hi < lo:
        lo, hi = hi, lo

    for i in range(lo, hi + 1):
        script = ROOT / "scripts" / SCRIPTS[i - 1]
        print(f"\n=== Stage {i}: {script.name} ===\n")
        cmd = [sys.executable, str(script), "--config", args.config]
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
