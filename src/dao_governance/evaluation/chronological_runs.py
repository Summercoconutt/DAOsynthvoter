"""Compare validation runs and evaluate frozen chronological test models."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from dao_governance.data.canonical_dataset import load_proposal_catalog
from dao_governance.data.chronological_split import validate_chronological_manifest, validate_proposal_metadata
from dao_governance.data.lean_sensitivity import load_lean_membership
from dao_governance.data.reporting import json_safe_report
from dao_governance.evaluation.point_in_time import (
    baseline_predictions,
    evaluate_matched_predictions,
    verify_frozen_selection,
)
from dao_governance.modelling.point_in_time_artifacts import (
    load_validation_checkpoint,
    verify_validation_checkpoint,
)
from dao_governance.modelling.point_in_time_model import NUMERIC_FEATURES, ProposalTextCache, VoteDataset, validate_target_examples
from dao_governance.modelling.point_in_time_training import predict_vote_classifier
from dao_governance.modelling.point_in_time_runner import _validate_partition_coverage
from dao_governance.modelling.run_artifacts import resolve_artifact_path, sha256_file, validate_run_id

_BASELINES = {"majority", "historical_choice"}


def parse_named_runs(values: Sequence[str]) -> Dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("Runs must be unique NAME=PATH entries.")
        result[name] = Path(path)
    return result


def _prediction_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"example_id": "string", "space": "string", "proposal_id": "string"})


def _comparison_examples(frame: pd.DataFrame, *, include_history: bool) -> pd.DataFrame:
    identity = ["example_id", "space", "proposal_id", "vote_ts", "label_id"]
    columns = identity + (sorted(NUMERIC_FEATURES) if include_history else [])
    if not set(columns).issubset(frame.columns):
        raise ValueError("Compared runs are missing canonical example or history columns.")
    result = frame[columns].copy()
    for column in ("example_id", "space", "proposal_id"):
        result[column] = result[column].astype("string")
    result["vote_ts"] = pd.to_datetime(result["vote_ts"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    labels = pd.to_numeric(result["label_id"], errors="raise")
    if not labels.isin([0, 1, 2]).all() or labels.map(lambda value: isinstance(value, bool)).any():
        raise ValueError("Compared examples must use the three canonical labels.")
    result["label_id"] = labels.astype("int64")
    for column in columns[len(identity):]:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    return result.sort_values("example_id").reset_index(drop=True)


def compare_validation_runs(
    runs: Mapping[str, Path],
    output: Path,
    *,
    bootstrap_samples: int,
    seed: int = 42,
    allow_history_sensitivity: bool = False,
) -> Dict[str, Any]:
    if not runs or _BASELINES.intersection(runs):
        raise ValueError("Provide at least one uniquely named trained model, excluding baseline names.")
    if len({root.resolve() for root in runs.values()}) != len(runs):
        raise ValueError("Each compared model must use a distinct run.")
    if output.exists():
        raise FileExistsError("Validation comparison output already exists.")
    predictions: Dict[str, pd.DataFrame] = {}
    run_records = {}
    run_details = {}
    reference_split = None
    reference_hash = None
    reference_history = None
    history_modes = {}
    reference_baselines = {}
    strata = None
    reference_train = None
    reference_validation = None
    for name, root in runs.items():
        validate_run_id(name)
        split_path = root / "shared" / "split_manifest.json"
        manifest, config = verify_validation_checkpoint(root, expected_run_id=root.name, expected_split=split_path)
        recorded = {record["relative_path"] for record in manifest["artifacts"]}
        required = {"validation/model.csv", "validation/majority.csv", "validation/historical_choice.csv", "validation/report.json"}
        if not required.issubset(recorded):
            raise RuntimeError("A run is missing recorded validation predictions.")
        if reference_hash is None:
            reference_hash = manifest["split_sha256"]
            reference_split = json.loads(split_path.read_text(encoding="utf-8"))
            reference_history = config["history_mode"]
            for baseline in _BASELINES:
                reference_baselines[baseline] = _prediction_csv(root / "validation" / f"{baseline}.csv")
            validation_record = (manifest.get("inputs") or {}).get("validation_examples")
            train_record = (manifest.get("inputs") or {}).get("train_examples")
            if not isinstance(validation_record, dict) or not isinstance(train_record, dict):
                raise RuntimeError("A run is missing training or validation example provenance.")
            strata = pd.read_parquet(validation_record["path"])
            reference_train = pd.read_parquet(train_record["path"])
            reference_validation = strata
        elif manifest["split_sha256"] != reference_hash:
            raise ValueError("Compared runs must use the same chronological split.")
        elif config["history_mode"] != reference_history and not allow_history_sensitivity:
            raise ValueError("Different history modes require an explicit sensitivity comparison.")
        train_record = (manifest.get("inputs") or {}).get("train_examples")
        validation_record = (manifest.get("inputs") or {}).get("validation_examples")
        if not isinstance(train_record, dict) or not isinstance(validation_record, dict):
            raise RuntimeError("A run is missing training or validation example provenance.")
        current_train = pd.read_parquet(train_record["path"])
        current_validation = pd.read_parquet(validation_record["path"])
        if reference_train is None or reference_validation is None:
            raise RuntimeError("No reference training or validation examples were loaded.")
        compare_history = config["history_mode"] == reference_history
        if (
            not _comparison_examples(current_train, include_history=compare_history).equals(
                _comparison_examples(reference_train, include_history=compare_history)
            )
            or not _comparison_examples(current_validation, include_history=compare_history).equals(
                _comparison_examples(reference_validation, include_history=compare_history)
            )
        ):
            raise ValueError("Compared models must use identical training/validation targets and matching history.")
        for baseline in _BASELINES:
            actual = _prediction_csv(root / "validation" / f"{baseline}.csv").sort_values("example_id").reset_index(drop=True)
            expected = reference_baselines[baseline].sort_values("example_id").reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
            except AssertionError as exc:
                raise ValueError("Compared runs have different matched baseline predictions.") from exc
        predictions[name] = _prediction_csv(root / "validation" / "model.csv")
        history_modes[name] = config["history_mode"]
        run_records[name] = {"run_id": root.name, "manifest_sha256": sha256_file(root / "run_manifest.json")}
        run_report = json.loads((root / "validation/report.json").read_text(encoding="utf-8"))
        run_details[name] = {
            "model_kind": run_report.get("model_kind"),
            "tokenized_proposals": run_report.get("tokenized_proposals"),
            "truncated_proposals": run_report.get("truncated_proposals"),
            "voter_cluster_scope": config.get("voter_cluster_scope"),
        }
    predictions.update(reference_baselines)
    if reference_split is None or reference_hash is None:
        raise RuntimeError("No validation split was loaded.")
    report = evaluate_matched_predictions(
        predictions, manifest=reference_split, split="validation", split_fingerprint=reference_hash,
        bootstrap_samples=bootstrap_samples, seed=seed, strata=strata,
    )
    report["run_manifests"] = run_records
    report["run_details"] = run_details
    report["history_modes"] = history_modes
    report["comparison_type"] = "history_sensitivity" if len(set(history_modes.values())) > 1 else "matched_models"
    payload = json_safe_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return payload


def _test_examples(root: Path, manifest: Mapping[str, Any]) -> pd.DataFrame:
    record = (manifest.get("inputs") or {}).get("examples_manifest")
    if not isinstance(record, dict):
        raise RuntimeError("The model run is missing its example-artifact provenance.")
    path = Path(record["path"])
    examples = json.loads(path.read_text(encoding="utf-8"))
    config = json.loads((root / "model/config.json").read_text(encoding="utf-8"))
    if (
        examples.get("split_sha256") != manifest["split_sha256"]
        or examples.get("dataset_sha256") != manifest["dataset_sha256"]
        or examples.get("history_mode") != config.get("history_mode")
        or examples.get("provenance_mode", "verified") != config.get("provenance_mode", "verified")
    ):
        raise RuntimeError("Test examples do not match the model run.")
    test_record = (examples.get("partitions") or {}).get("test")
    if not isinstance(test_record, dict) or not isinstance(test_record.get("path"), str):
        raise RuntimeError("The example artifacts do not record a test partition.")
    test_path = resolve_artifact_path(path.parent, test_record["path"])
    if not test_path.is_file() or sha256_file(test_path) != test_record.get("sha256"):
        raise RuntimeError("The held-out test partition is missing or changed.")
    frame = pd.read_parquet(test_path)
    if len(frame) != test_record.get("rows"):
        raise RuntimeError("The held-out test partition row count changed.")
    return frame


def filter_content_disjoint_test(frame: pd.DataFrame, split: Mapping[str, Any]) -> pd.DataFrame:
    validate_chronological_manifest(split)
    excluded = {
        (row["space"], row["proposal_id"])
        for row in split["proposals"]
        if row["split"] == "test" and row.get("content_sensitivity_excluded") is True
    }
    mask = [(space, proposal_id) not in excluded for space, proposal_id in zip(frame["space"], frame["proposal_id"])]
    result = frame.loc[mask].reset_index(drop=True)
    if result.empty:
        raise ValueError("Content-disjoint sensitivity leaves no held-out test examples.")
    return result


def filter_lean_test(frame: pd.DataFrame, membership: set[tuple[str, str]]) -> pd.DataFrame:
    mask = [(space, proposal_id) in membership for space, proposal_id in zip(frame["space"], frame["proposal_id"])]
    result = frame.loc[mask].reset_index(drop=True)
    if result.empty:
        raise ValueError("Outcome-selected lean sensitivity leaves no held-out test examples.")
    return result


def evaluate_frozen_test(
    *,
    selection_path: Path,
    validation_report: Path,
    checkpoint_root: Path,
    additional_checkpoints: Mapping[str, Path],
    output_dir: Path,
    bootstrap_samples: int,
    seed: int = 42,
    batch_size: int = 32,
    device: str = "cpu",
    content_disjoint_sensitivity: bool = False,
    lean_membership: Path | None = None,
) -> Dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("Test evaluation output directory already exists.")
    split_path = checkpoint_root / "shared" / "split_manifest.json"
    selection = verify_frozen_selection(
        selection_path, split_manifest=split_path, validation_report=validation_report,
        checkpoint_root=checkpoint_root, additional_checkpoints=additional_checkpoints, lean_membership=lean_membership,
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    selected_name = selection["selected_model"]
    runs = {selected_name: checkpoint_root, **dict(additional_checkpoints)}
    if set(runs) != set(selection["allowed_models"]) - _BASELINES:
        raise ValueError("Test model runs do not match the frozen comparison.")
    lean_keys = None
    if lean_membership is not None:
        checkpoint = json.loads((checkpoint_root / "run_manifest.json").read_text(encoding="utf-8"))
        raw_record = (checkpoint.get("inputs") or {}).get("raw_votes")
        if not isinstance(raw_record, dict) or not raw_record.get("sha256"):
            raise RuntimeError("The selected run lacks raw vote provenance for lean sensitivity.")
        lean_keys = load_lean_membership(lean_membership, raw_votes_sha256=raw_record["sha256"])

    predictions = {}
    text_truncation = {}
    selected_examples = None
    for name, root in runs.items():
        model, encoder, tokenizer, manifest = load_validation_checkpoint(
            root, expected_run_id=root.name, expected_split=split_path, device=device
        )
        config = json.loads((root / "model/config.json").read_text(encoding="utf-8"))
        test = _test_examples(root, manifest)
        validate_target_examples(test, split, "test")
        canonical_record = (manifest.get("inputs") or {}).get("canonical_events")
        if not isinstance(canonical_record, dict):
            raise RuntimeError("The model run is missing canonical event provenance.")
        _validate_partition_coverage(Path(canonical_record["path"]), split, {"test": test})
        if content_disjoint_sensitivity:
            test = filter_content_disjoint_test(test, split)
        if lean_keys is not None:
            test = filter_lean_test(test, lean_keys)
        inputs = manifest.get("inputs") or {}
        required = {"raw_votes", "proposal_metadata", "train_examples"}
        local_proxy = manifest.get("provenance_mode") == "local_proxy"
        required.add("local_metadata_binding" if local_proxy else "historical_provenance")
        if not required.issubset(inputs):
            raise RuntimeError("The model run lacks required dataset provenance.")
        historical = Path(inputs["historical_provenance"]["path"]) if not local_proxy else None
        catalog, fingerprint, _ = load_proposal_catalog(
            Path(inputs["raw_votes"]["path"]),
            Path(inputs["proposal_metadata"]["path"]),
            historical,
            local_proxy=local_proxy,
        )
        if fingerprint != manifest["dataset_sha256"]:
            raise RuntimeError("The eligible proposal catalog differs from the model run.")
        membership = {(row["space"], row["proposal_id"]): row for row in split["proposals"]}
        proposals = []
        for key in sorted(set(zip(test["space"], test["proposal_id"]))):
            proposal = catalog.eligible.get(key)
            if proposal is None:
                raise ValueError("A test proposal lacks verified eligible metadata.")
            validate_proposal_metadata(proposal, membership[key])
            proposals.append(proposal)
        cache = ProposalTextCache(proposals, tokenizer, max_length=int(config["text_max_length"])) if tokenizer is not None else None
        text_truncation[name] = {
            "proposals": len(cache.tokens) if cache is not None else 0,
            "truncated": cache.truncated if cache is not None else 0,
        }
        dataset = VoteDataset(test, encoder.transform(test), cache)
        predictions[name] = predict_vote_classifier(model, dataset, batch_size=batch_size, device=device)
        if name == selected_name:
            selected_examples = test
            train = pd.read_parquet(inputs["train_examples"]["path"])
            baselines = baseline_predictions(train, test)
            predictions.update({key: value for key, value in baselines.items() if key in selection["allowed_models"]})
    if selected_examples is None:
        raise RuntimeError("The selected model did not produce test examples.")
    report = evaluate_matched_predictions(
        predictions, manifest=split, split="test", split_fingerprint=sha256_file(split_path),
        bootstrap_samples=bootstrap_samples, seed=seed, strata=selected_examples,
        selection_path=selection_path, selection_split_manifest=split_path,
        validation_report=validation_report, checkpoint_root=checkpoint_root,
        additional_checkpoints=additional_checkpoints, lean_membership=lean_membership,
    )
    report["selection_sha256"] = sha256_file(selection_path)
    report["validation_report_sha256"] = sha256_file(validation_report)
    report["content_mode"] = "content_disjoint_sensitivity" if content_disjoint_sensitivity else "primary"
    report["population"] = "outcome_selected_lean_sensitivity" if lean_keys is not None else "full_eligible"
    report["lean_membership_sha256"] = sha256_file(lean_membership) if lean_membership is not None else None
    report["text_truncation"] = text_truncation
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name, frame in predictions.items():
        validate_run_id(name)
        path = output_dir / f"{name}_predictions.csv"
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            frame.to_csv(stream, index=False)
    payload = json_safe_report(report)
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Chronological held-out evaluation", "",
        f"- Provenance mode: {report['provenance_mode']}",
        f"- Population: {report['population']}",
        f"- Examples: {report['n_examples']}", f"- Proposals: {report['n_proposals']}", "",
    ]
    for name, data in report["models"].items():
        metrics = data["metrics"]
        interval = data["proposal_bootstrap_macro_f1"]
        ece = metrics.get("ece")
        log_loss = metrics.get("log_loss")
        ece_text = f"{ece:.4f}" if isinstance(ece, (int, float)) and math.isfinite(ece) else "undefined"
        log_loss_text = f"{log_loss:.4f}" if isinstance(log_loss, (int, float)) and math.isfinite(log_loss) else "undefined"
        lines.extend(
            [
                f"## {name}",
                f"- Macro-F1: {metrics['macro_f1']:.4f}",
                f"- Proposal bootstrap 95% interval: {interval['lower']:.4f} - {interval['upper']:.4f}",
                f"- Accuracy: {metrics['accuracy']:.4f}",
                f"- ECE: {ece_text}; log loss: {log_loss_text}",
                f"- Confusion matrix (FOR, AGAINST, ABSTAIN): {data['confusion_matrix']}",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label in ("FOR", "AGAINST", "ABSTAIN"):
            item = metrics["per_class"][label]
            lines.append(f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |")
        lines.extend(["", "### History strata", ""])
        for group, values in data.get("strata", {}).items():
            for label, item in values.items():
                lines.append(f"- {group}/{label}: n={item['n']}, macro-F1={item['macro_f1']:.4f}")
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    artifacts = [
        {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir())
    ]
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps({"version": 1, "selection_sha256": report["selection_sha256"], "artifacts": artifacts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
