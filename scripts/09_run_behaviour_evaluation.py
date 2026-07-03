#!/usr/bin/env python3
"""
9) Comprehensive behaviour-model evaluation (held-out test / val from split_manifest).

  cd whole_pipeline
  set PYTHONPATH=src
  python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --split test
  python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --split val
  python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --split test val
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.evaluation.evaluate import run_comprehensive_evaluation
from dao_governance.modelling.preprocess import load_split_manifest
from dao_governance.settings import load_config, project_root


def main() -> None:
    ap = argparse.ArgumentParser(description="Comprehensive evaluation for stage-08 behaviour model.")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--extra-config", type=str, default="")
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

    base = project_root()
    cfg_path = (base / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=cfg_path, extra_path=extra)
    paths = cfg.get("paths", {})
    bm = cfg.get("behaviour_model", {})

    behaviour_csv = (base / paths.get("behaviour_dataset_csv", "outputs/tables/behaviour_dataset.csv")).resolve()
    artifacts = (base / paths.get("model_artifacts_dir", "outputs/models/behaviour_agent2")).resolve()
    eval_out = (base / paths.get("eval_output_dir", "outputs/tables/eval")).resolve()
    split_path = (base / paths.get("split_manifest_json", "outputs/processed/split_manifest.json")).resolve()
    report_path = (base / (cfg.get("reports") or {}).get("stage09_md", "outputs/reports/stage09_behaviour_evaluation.md")).resolve()

    cfg_train = json.loads((artifacts / "config.json").read_text(encoding="utf-8"))
    preprocessor = cfg_train.get("numeric_preprocessor")
    if preprocessor is None:
        raise RuntimeError("config.json missing numeric_preprocessor; run scripts/08 first.")

    manifest = load_split_manifest(split_path) if split_path.exists() else {}
    max_w = int(bm.get("max_test_windows", 0))
    batch_size = int(bm.get("eval_batch_size", bm.get("batch_size", 32)))
    seed = int(cfg_train.get("seed", cfg.get("project", {}).get("seed", 42)))

    cfg_model = {
        "pretrained": cfg_train["pretrained"],
        "feat_dim": cfg_train["feat_dim"],
        "window": cfg_train["window"],
        "max_length": cfg_train["max_length"],
    }
    if cfg_train.get("enriched_behaviour_csv"):
        cfg_model["enriched_behaviour_csv"] = cfg_train["enriched_behaviour_csv"]

    summaries = []
    for split_label in args.split:
        if split_label == "all":
            voter_ids = None
        else:
            if not split_path.exists():
                raise FileNotFoundError(f"Split manifest not found: {split_path}. Run scripts/08 first.")
            key = f"{split_label}_voters"
            voter_ids = manifest.get(key) or []
            if not voter_ids:
                raise RuntimeError(f"No voters listed under '{key}' in {split_path}")

        print(f"[09] Evaluating split={split_label} ...")
        summary = run_comprehensive_evaluation(
            dataset_csv=behaviour_csv,
            artifacts_dir=artifacts,
            output_dir=eval_out,
            preprocessor=preprocessor,
            cfg_train=cfg_model,
            voter_ids=voter_ids,
            seed=seed,
            batch_size=batch_size,
            max_windows=max_w,
            split_label=split_label,
            min_group_n=args.min_group_n,
            calibration_bins=args.calibration_bins,
        )
        summaries.append(summary)
        print(
            f"[09] {split_label}: macro_f1={summary['macro_f1']:.4f}, "
            f"balanced_acc={summary['balanced_accuracy']:.4f}, mcc={summary['matthews_corrcoef']:.4f}, "
            f"ece={summary.get('ece')}"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Stage 9 — Behaviour model evaluation", ""]
    for s in summaries:
        lines.extend(
            [
                f"## Split: `{s['split']}`",
                "",
                f"- Output dir: `{eval_out / s['split']}`",
                f"- Windows: **{s['n_windows']}**",
                f"- Macro F1: **{s['macro_f1']:.4f}**",
                f"- Balanced accuracy: **{s['balanced_accuracy']:.4f}**",
                f"- MCC: **{s['matthews_corrcoef']:.4f}**",
                f"- ECE: **{s.get('ece', 'n/a')}**",
                "",
                "See `metrics_report.md`, `figures/`, and cluster CSVs in the split folder.",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[09] Reports under {eval_out}")
    print(f"[09] Stage summary: {report_path}")


if __name__ == "__main__":
    main()
