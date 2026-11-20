#!/usr/bin/env python3
"""Compare validation models, freeze a selection, and evaluate held-out test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.evaluation.chronological_runs import compare_validation_runs, evaluate_frozen_test, parse_named_runs
from dao_governance.evaluation.point_in_time import freeze_validation_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare-validation")
    compare.add_argument("--run", action="append", required=True, help="Model NAME=RUN_DIRECTORY.")
    compare.add_argument("--output", required=True)
    compare.add_argument("--bootstrap-samples", type=int, default=1000)
    compare.add_argument("--compare-history-sensitivity", action="store_true")
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--validation-report", required=True)
    freeze.add_argument("--selected-model", required=True)
    freeze.add_argument("--selected-run", required=True)
    freeze.add_argument("--additional-run", action="append", default=[], help="Additional model NAME=RUN_DIRECTORY.")
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--lean-membership", default="")
    test = commands.add_parser("test")
    test.add_argument("--selection", required=True)
    test.add_argument("--validation-report", required=True)
    test.add_argument("--selected-run", required=True)
    test.add_argument("--additional-run", action="append", default=[], help="Additional model NAME=RUN_DIRECTORY.")
    test.add_argument("--output-dir", required=True)
    test.add_argument("--bootstrap-samples", type=int, default=1000)
    test.add_argument("--batch-size", type=int, default=32)
    test.add_argument("--device", default="cpu")
    test.add_argument("--content-disjoint-sensitivity", action="store_true")
    test.add_argument("--lean-membership", default="")
    args = parser.parse_args()

    if args.command == "compare-validation":
        result = compare_validation_runs(
            parse_named_runs(args.run), Path(args.output), bootstrap_samples=args.bootstrap_samples,
            allow_history_sensitivity=args.compare_history_sensitivity,
        )
        print(f"Compared validation examples: {result['n_examples']}")
    elif args.command == "freeze":
        additional = parse_named_runs(args.additional_run)
        selected = Path(args.selected_run)
        if args.selected_model in additional:
            parser.error("The selected model cannot also be an additional run.")
        allowed = [args.selected_model, *additional, "majority", "historical_choice"]
        result = freeze_validation_selection(
            Path(args.output), split_manifest=selected / "shared/split_manifest.json",
            validation_report=Path(args.validation_report), checkpoint_root=selected,
            selected_model=args.selected_model, allowed_models=allowed, additional_checkpoints=additional,
            lean_membership=Path(args.lean_membership) if args.lean_membership else None,
        )
        print(f"Frozen validation selection: {result['selected_model']}")
    else:
        result = evaluate_frozen_test(
            selection_path=Path(args.selection), validation_report=Path(args.validation_report),
            checkpoint_root=Path(args.selected_run), additional_checkpoints=parse_named_runs(args.additional_run),
            output_dir=Path(args.output_dir), bootstrap_samples=args.bootstrap_samples,
            batch_size=args.batch_size, device=args.device,
            content_disjoint_sensitivity=args.content_disjoint_sensitivity,
            lean_membership=Path(args.lean_membership) if args.lean_membership else None,
        )
        print(f"Evaluated held-out test examples: {result['n_examples']}")


if __name__ == "__main__":
    main()
