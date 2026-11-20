#!/usr/bin/env python3
"""Explain frozen held-out predictions with training-only backgrounds."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.evaluation.chronological_runs import parse_named_runs
from dao_governance.evaluation.explain_chronological import explain_frozen_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--selected-run", required=True)
    parser.add_argument("--additional-run", action="append", default=[], help="Additional model NAME=RUN_DIRECTORY.")
    parser.add_argument("--test-evaluation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=9)
    parser.add_argument("--global-sample-size", type=int, default=20)
    parser.add_argument("--background-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--completeness-tolerance", type=float, default=1e-3)
    parser.add_argument("--lean-membership", default="")
    args = parser.parse_args()
    result = explain_frozen_test(
        selection_path=Path(args.selection),
        validation_report=Path(args.validation_report),
        checkpoint_root=Path(args.selected_run),
        additional_checkpoints=parse_named_runs(args.additional_run),
        test_evaluation_dir=Path(args.test_evaluation_dir),
        output_dir=Path(args.output_dir),
        max_cases=args.max_cases,
        global_sample_size=args.global_sample_size,
        background_size=args.background_size,
        steps=args.steps,
        seed=args.seed,
        completeness_tolerance=args.completeness_tolerance,
        lean_membership=Path(args.lean_membership) if args.lean_membership else None,
    )
    print(
        f"Explained {result['explained_examples']} held-out examples; "
        f"completeness failures: {result['cases_with_completeness_failure']}"
    )


if __name__ == "__main__":
    main()
