#!/usr/bin/env python3
"""Export a verified chronological run to a local ZIP archive."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.evaluation.chronological_runs import parse_named_runs
from dao_governance.modelling.chronological_export import export_chronological_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("destination")
    parser.add_argument("--include-sensitive-inputs", action="store_true")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--selection", default="")
    parser.add_argument("--validation-report", default="")
    parser.add_argument("--additional-run", action="append", default=[], help="Additional model NAME=RUN_DIRECTORY.")
    parser.add_argument("--test-evaluation-dir", default="")
    parser.add_argument("--explanation-dir", default="")
    parser.add_argument("--final-report", default="")
    args = parser.parse_args()
    result = export_chronological_run(
        Path(args.run_dir), Path(args.destination),
        include_sensitive_inputs=args.include_sensitive_inputs, include_test=args.include_test,
        selection_path=Path(args.selection) if args.selection else None,
        validation_report=Path(args.validation_report) if args.validation_report else None,
        additional_checkpoints=parse_named_runs(args.additional_run),
        test_evaluation_dir=Path(args.test_evaluation_dir) if args.test_evaluation_dir else None,
        explanation_dir=Path(args.explanation_dir) if args.explanation_dir else None,
        final_report=Path(args.final_report) if args.final_report else None,
    )
    print(f"Chronological export created: {result}")


if __name__ == "__main__":
    main()
