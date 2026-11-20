#!/usr/bin/env python3
"""Write a report from frozen chronological evaluation and explanation artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.evaluation.chronological_runs import parse_named_runs
from dao_governance.evaluation.final_report import write_final_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--selected-run", required=True)
    parser.add_argument("--additional-run", action="append", default=[], help="Additional model NAME=RUN_DIRECTORY.")
    parser.add_argument("--test-evaluation-dir", required=True)
    parser.add_argument("--explanation-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lean-membership", default="")
    args = parser.parse_args()
    write_final_report(
        selection_path=Path(args.selection),
        validation_report=Path(args.validation_report),
        checkpoint_root=Path(args.selected_run),
        additional_checkpoints=parse_named_runs(args.additional_run),
        test_evaluation_dir=Path(args.test_evaluation_dir),
        explanation_dir=Path(args.explanation_dir),
        output=Path(args.output),
        lean_membership=Path(args.lean_membership) if args.lean_membership else None,
    )
    print(f"Chronological report written: {args.output}")


if __name__ == "__main__":
    main()
