#!/usr/bin/env python3
"""
9) Comprehensive behaviour-model evaluation (held-out test / val from split_manifest).

  cd whole_pipeline
  set PYTHONPATH=src
    python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --run-id full_trial_01 --split test val
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.evaluation.evaluate import run_comprehensive_evaluation
from dao_governance.modelling.preprocess import load_split_manifest
from dao_governance.modelling.run_artifacts import (
    record_stage,
    resolve_artifact_path,
    resolve_run_root,
    verify_recorded_artifacts,
    verify_run_identity,
)
from dao_governance.settings import load_config, project_root


def main() -> None:
    ap = argparse.ArgumentParser(description="Comprehensive evaluation for stage-08 behaviour model.")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--data-sources-config", type=str, default="configs/data_sources.yaml")
    ap.add_argument("--extra-config", type=str, default="")
    ap.add_argument("--run-id", required=True, help="Completed paired Stage 08/08b run to evaluate.")
    ap.add_argument(
        "--text-encoder-mode",
        choices=("fine-tune", "frozen"),
        default=None,
        help="Match a Stage 08 command-line override when verifying the paired run config.",
    )
    ap.add_argument("--resume-stage", action="store_true", help="Resume an incomplete evaluation within this run.")
    ap.add_argument(
        "--model",
        nargs="+",
        choices=["08", "08b"],
        default=["08", "08b"],
        help="Which paired model(s) to evaluate.",
    )
    ap.add_argument(
        "--split",
        type=str,
        nargs="+",
        choices=["test", "val", "all"],
        default=["test"],
        help="One or more splits to evaluate (uses split_manifest except 'all').",
    )
    ap.add_argument("--min-group-n", type=int, default=5, help="Min windows per cluster for stratified metrics.")
    ap.add_argument("--calibration-bins", type=int, default=15)
    args = ap.parse_args()
    if set(args.model) != {"08", "08b"}:
        raise ValueError("Paired Stage 09 evaluation requires both --model 08 and --model 08b")

    base = project_root()
    cfg_path = (base / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    data_sources_cfg_path = (
        (base / args.data_sources_config).resolve()
        if not Path(args.data_sources_config).is_absolute()
        else Path(args.data_sources_config).resolve()
    )
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=cfg_path, extra_path=extra, data_sources_path=data_sources_cfg_path)
    bm = cfg.get("behaviour_model", {})
    if args.text_encoder_mode is not None:
        bm["text_encoder_mode"] = args.text_encoder_mode
    run_id = args.run_id.strip()
    sample_variant = str((cfg.get("run") or {}).get("sample_variant", "full")).strip().lower()
    run_root = resolve_run_root(base, cfg, run_id)
    manifest = verify_run_identity(
        run_root,
        run_id=run_id,
        sample_variant=sample_variant,
        config=cfg,
    )
    required_stages = {"08", "08b"}
    if not required_stages.issubset(manifest.get("stages", {})):
        raise RuntimeError("Stage 09 requires completed Stage 08 and Stage 08b artifacts from the same run")
    if "evaluation" in manifest.get("stages", {}):
        raise RuntimeError(f"Stage 09 is already recorded for run {run_id!r}; use a new run ID")
    verify_recorded_artifacts(run_root, manifest)

    model_dirs = {"08": run_root / "08", "08b": run_root / "08b"}
    model_configs = {
        name: json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        for name, model_dir in model_dirs.items()
    }
    for name, model_config in model_configs.items():
        if model_config.get("run_id") != run_id or model_config.get("sample_variant") != sample_variant:
            raise RuntimeError(f"Stage {name} config does not identify this run and sample variant")
        if not model_config.get("numeric_preprocessor"):
            raise RuntimeError(f"Stage {name} config is missing its fitted numeric preprocessor")
    stage08_cfg, stage08b_cfg = model_configs["08"], model_configs["08b"]
    if stage08_cfg.get("input_hashes") != stage08b_cfg.get("input_hashes"):
        raise RuntimeError("Stage 08 and 08b were not trained from identical source inputs")
    split_path = run_root / "shared" / "split_manifest.json"
    stage08_split = resolve_artifact_path(run_root, stage08_cfg.get("split_manifest", ""))
    stage08b_split = resolve_artifact_path(run_root, stage08b_cfg.get("split_manifest", ""))
    if stage08_split != split_path or stage08b_split != split_path:
        raise RuntimeError("Stage 08 and 08b do not reference the same run-scoped split manifest")
    split_data = load_split_manifest(split_path)
    max_w = int(bm.get("max_test_windows", 0))
    batch_size = int(bm.get("eval_batch_size", bm.get("batch_size", 32)))
    summaries = []
    evaluation_root = run_root / "evaluation"
    if evaluation_root.exists() and not args.resume_stage:
        raise RuntimeError(f"Stage 09 output directory already exists for run {run_id!r}; use a new run ID")
    for model_name in args.model:
        model_dir = model_dirs[model_name]
        model_config = model_configs[model_name]
        dataset_csv = resolve_artifact_path(run_root, model_config.get("enriched_behaviour_csv", ""))
        if not dataset_csv.is_file():
            raise FileNotFoundError(f"Stage {model_name} evaluation dataset is missing: {dataset_csv}")
        for split_label in args.split:
            if split_label == "all":
                voter_ids = None
            else:
                voter_ids = split_data.get(f"{split_label}_voters") or []
                if not voter_ids:
                    raise RuntimeError(f"No voters listed under '{split_label}_voters' in {split_path}")
            print(f"[09] Evaluating model={model_name} split={split_label} ...")
            runtime_model_config = dict(model_config)
            runtime_model_config["enriched_behaviour_csv"] = str(dataset_csv)
            summary = run_comprehensive_evaluation(
                dataset_csv=dataset_csv,
                artifacts_dir=model_dir,
                output_dir=evaluation_root / model_name,
                preprocessor=model_config["numeric_preprocessor"],
                cfg_train=runtime_model_config,
                voter_ids=voter_ids,
                seed=int(model_config.get("seed", cfg.get("project", {}).get("seed", 42))),
                batch_size=batch_size,
                max_windows=max_w,
                split_label=split_label,
                min_group_n=args.min_group_n,
                calibration_bins=args.calibration_bins,
            )
            summaries.append({"model": model_name, **summary})
            print(
                f"[09] {model_name}/{split_label}: macro_f1={summary['macro_f1']:.4f}, "
                f"balanced_acc={summary['balanced_accuracy']:.4f}, mcc={summary['matthews_corrcoef']:.4f}, "
                f"ece={summary.get('ece')}"
            )

    report_dir = evaluation_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = report_dir / "within_run_model_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["run_id", "sample_variant", "model", "split", "n_windows", "macro_f1", "balanced_accuracy", "matthews_corrcoef", "ece"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({"run_id": run_id, "sample_variant": sample_variant, **{key: summary.get(key) for key in fields if key not in {"run_id", "sample_variant"}}})

    lines = [f"# Stage 09 — Evaluation: {sample_variant}/{run_id}", ""]
    for summary in summaries:
        lines.extend(
            [
                f"## Model {summary['model']} / split `{summary['split']}`",
                "",
                f"- Windows: **{summary['n_windows']}**",
                f"- Macro F1: **{summary['macro_f1']:.4f}**",
                f"- Balanced accuracy: **{summary['balanced_accuracy']:.4f}**",
                f"- MCC: **{summary['matthews_corrcoef']:.4f}**",
                f"- ECE: **{summary.get('ece', 'n/a')}**",
                f"- Output: `{evaluation_root / summary['model'] / summary['split']}`",
                "",
            ]
        )
    report_path = report_dir / "evaluation_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    stage_dirs = [evaluation_root / model_name for model_name in args.model]
    record_stage(
        run_root,
        manifest,
        "evaluation",
        [*stage_dirs, comparison_path, report_path],
        details={"models": args.model, "splits": args.split},
    )
    print(f"[09] Run-scoped reports: {report_dir}")


if __name__ == "__main__":
    main()
