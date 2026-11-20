"""Matched model evaluation with proposal-level uncertainty."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from dao_governance.data.lean_sensitivity import load_lean_membership
from dao_governance.evaluation.metrics import compute_classification_metrics
from dao_governance.modelling.point_in_time_model import (
    historical_choice_baseline,
    majority_label,
    validate_target_examples,
)
from dao_governance.modelling.point_in_time_artifacts import verify_validation_checkpoint
from dao_governance.modelling.run_artifacts import sha256_file

_PREDICTION_COLUMNS = (
    "example_id", "space", "proposal_id", "vote_ts", "y_true", "y_pred",
    "prob_FOR", "prob_AGAINST", "prob_ABSTAIN",
)


def baseline_predictions(train: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    required = {"example_id", "space", "proposal_id", "vote_ts", "label_id"}
    if "label_id" not in train.columns or not required.issubset(targets.columns):
        raise ValueError("Baseline inputs are missing labels or target identities.")
    if targets["example_id"].isna().any() or targets["example_id"].duplicated().any():
        raise ValueError("Baseline targets require unique example identifiers.")
    fallback = majority_label(train["label_id"])
    choices = {
        "majority": np.full(len(targets), fallback, dtype=int),
        "historical_choice": historical_choice_baseline(targets, fallback_label=fallback),
    }
    results = {}
    for name, predicted in choices.items():
        probabilities = np.eye(3, dtype=float)[predicted]
        results[name] = pd.DataFrame(
            {
                "example_id": targets["example_id"].to_numpy(),
                "space": targets["space"].to_numpy(),
                "proposal_id": targets["proposal_id"].to_numpy(),
                "vote_ts": targets["vote_ts"].to_numpy(),
                "y_true": targets["label_id"].to_numpy(),
                "y_pred": predicted,
                "prob_FOR": probabilities[:, 0],
                "prob_AGAINST": probabilities[:, 1],
                "prob_ABSTAIN": probabilities[:, 2],
            }
        )
    return results


def _proposal_bootstrap(frame: pd.DataFrame, *, samples: int, seed: int) -> Dict[str, float]:
    groups = list(frame.groupby(["space", "proposal_id"], sort=True).indices.values())
    if not groups or samples < 1:
        raise ValueError("Proposal bootstrap requires proposals and a positive sample count.")
    y_true = frame["y_true"].to_numpy()
    y_pred = frame["y_pred"].to_numpy()
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(samples):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        scores.append(f1_score(y_true[indices], y_pred[indices], labels=[0, 1, 2], average="macro", zero_division=0))
    low, high = np.quantile(scores, [0.025, 0.975])
    return {"lower": float(low), "upper": float(high), "samples": samples}


def _strata(frame: pd.DataFrame, features: pd.DataFrame, *, sparse_threshold: int) -> Dict[str, Any]:
    required = {"example_id", "voter_has_history", "voter_dao_prior_count", "dao_prior_count"}
    if not required.issubset(features.columns):
        raise ValueError("Stratification requires voter and DAO history features.")
    if features["example_id"].isna().any() or features["example_id"].duplicated().any():
        raise ValueError("Stratification examples require unique identifiers.")
    joined = frame.merge(features[list(sorted(required))], on="example_id", how="left", validate="one_to_one")
    for column in ("voter_has_history", "voter_dao_prior_count", "dao_prior_count"):
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
        if joined[column].isna().any() or (joined[column] < 0).any():
            raise ValueError("Stratification history values must be present and nonnegative.")
    if not joined["voter_has_history"].isin([0, 1]).all():
        raise ValueError("Voter history indicators must be zero or one.")
    joined["voter_history"] = np.where(joined["voter_has_history"] == 0, "new", "known")
    for source, target in (("voter_dao_prior_count", "voter_dao_history"), ("dao_prior_count", "dao_history")):
        joined[target] = np.where(
            joined[source] == 0,
            "none",
            np.where(joined[source] < sparse_threshold, "sparse", "established"),
        )
    result: Dict[str, Any] = {}
    for column in ("voter_history", "voter_dao_history", "dao_history"):
        result[column] = {}
        for value, group in joined.groupby(column, sort=True):
            result[column][str(value)] = {
                "n": int(len(group)),
                "macro_f1": float(f1_score(group["y_true"], group["y_pred"], labels=[0, 1, 2], average="macro", zero_division=0)),
            }
    return result


def evaluate_matched_predictions(
    predictions: Mapping[str, pd.DataFrame],
    *,
    manifest: Mapping[str, Any],
    split: str,
    split_fingerprint: str,
    bootstrap_samples: int,
    seed: int = 42,
    strata: Optional[pd.DataFrame] = None,
    sparse_threshold: int = 5,
    selection_path: Optional[Path] = None,
    selection_split_manifest: Optional[Path] = None,
    validation_report: Optional[Path] = None,
    checkpoint_root: Optional[Path] = None,
    additional_checkpoints: Optional[Mapping[str, Path]] = None,
    lean_membership: Optional[Path] = None,
) -> Dict[str, Any]:
    if not predictions or bootstrap_samples < 1 or sparse_threshold < 1:
        raise ValueError("Matched evaluation requires models and positive bootstrap/strata settings.")
    if not re.fullmatch(r"[0-9a-f]{64}", split_fingerprint):
        raise ValueError("A SHA-256 split fingerprint is required.")
    if split == "test":
        if selection_path is None or selection_split_manifest is None or validation_report is None or checkpoint_root is None:
            raise ValueError("Test evaluation requires a matching, frozen validation selection.")
        if (
            sha256_file(selection_split_manifest) != split_fingerprint
            or json.loads(selection_split_manifest.read_text(encoding="utf-8")) != manifest
        ):
            raise ValueError("Test evaluation does not match the recorded split manifest.")
        frozen_selection = verify_frozen_selection(
            selection_path,
            split_manifest=selection_split_manifest,
            validation_report=validation_report,
            checkpoint_root=checkpoint_root,
            additional_checkpoints=additional_checkpoints,
            lean_membership=lean_membership,
        )
        if (
            frozen_selection.get("status") != "frozen"
            or frozen_selection.get("split_sha256") != split_fingerprint
            or frozen_selection.get("dataset_sha256") != manifest.get("dataset_sha256")
            or frozen_selection.get("provenance_mode") != manifest.get("provenance_mode", "verified")
            or set(frozen_selection.get("allowed_models", [])) != set(predictions)
            or frozen_selection.get("selected_model") not in predictions
        ):
            raise ValueError("Test evaluation requires a matching, frozen validation selection.")
    reference = None
    results: Dict[str, Any] = {}
    for name, frame in predictions.items():
        if not name or not set(_PREDICTION_COLUMNS).issubset(frame.columns) or frame.empty:
            raise ValueError("Each model requires nonempty, fully identified predictions.")
        if frame["example_id"].isna().any() or frame["example_id"].duplicated().any():
            raise ValueError("Prediction example identifiers must be unique.")
        validate_target_examples(frame, manifest, split)
        ordered = frame.copy()
        if not ordered["y_true"].isin([0, 1, 2]).all() or not ordered["y_pred"].isin([0, 1, 2]).all():
            raise ValueError("Predictions must use the three canonical labels.")
        for column in ("example_id", "space", "proposal_id"):
            ordered[column] = ordered[column].astype("string")
        ordered["y_true"] = ordered["y_true"].astype("int64")
        ordered["y_pred"] = ordered["y_pred"].astype("int64")
        ordered["vote_ts"] = pd.to_datetime(ordered["vote_ts"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
        ordered = ordered.set_index("example_id", drop=False).sort_index()
        identity = ordered[["space", "proposal_id", "vote_ts", "y_true"]]
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            raise ValueError("Compared models must use identical target IDs, proposals, and labels.")
        probabilities = ordered[["prob_FOR", "prob_AGAINST", "prob_ABSTAIN"]].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError("Predicted class probabilities must be finite and sum to one.")
        evaluation = {
            "metrics": compute_classification_metrics(ordered["y_true"], ordered["y_pred"], probabilities),
            "confusion_matrix": confusion_matrix(ordered["y_true"], ordered["y_pred"], labels=[0, 1, 2]).tolist(),
            "proposal_bootstrap_macro_f1": _proposal_bootstrap(ordered, samples=bootstrap_samples, seed=seed),
        }
        if strata is not None:
            evaluation["strata"] = _strata(ordered.reset_index(drop=True), strata, sparse_threshold=sparse_threshold)
        results[name] = evaluation
    if reference is None:
        raise RuntimeError("Matched evaluation has no reference predictions.")
    return {
        "split": split,
        "split_sha256": split_fingerprint,
        "dataset_sha256": manifest["dataset_sha256"],
        "provenance_mode": manifest.get("provenance_mode", "verified"),
        "n_examples": int(len(reference)),
        "n_proposals": int(reference[["space", "proposal_id"]].drop_duplicates().shape[0]),
        "models": results,
    }


def freeze_validation_selection(
    path: Path,
    *,
    split_manifest: Path,
    validation_report: Path,
    checkpoint_root: Path,
    selected_model: str,
    allowed_models: list[str],
    additional_checkpoints: Optional[Mapping[str, Path]] = None,
    lean_membership: Optional[Path] = None,
) -> Dict[str, Any]:
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    split_hash = sha256_file(split_manifest)
    if (
        report.get("split") != "validation"
        or report.get("split_sha256") != split_hash
        or not selected_model
        or selected_model not in allowed_models
        or len(set(allowed_models)) != len(allowed_models)
        or set(allowed_models) - set(report.get("models", {}))
    ):
        raise ValueError("A matching validation report and prespecified model list are required.")
    checkpoint, _ = verify_validation_checkpoint(
        checkpoint_root, expected_run_id=checkpoint_root.name, expected_split=split_manifest
    )
    if (
        report.get("dataset_sha256") != checkpoint.get("dataset_sha256")
        or report.get("provenance_mode") != checkpoint.get("provenance_mode", "verified")
    ):
        raise ValueError("Validation report and checkpoint datasets do not match.")
    if lean_membership is not None:
        raw_record = (checkpoint.get("inputs") or {}).get("raw_votes")
        if not isinstance(raw_record, dict) or not raw_record.get("sha256"):
            raise ValueError("Lean sensitivity requires recorded raw vote provenance.")
        load_lean_membership(lean_membership, raw_votes_sha256=raw_record["sha256"])
    run_records = report.get("run_manifests")
    if not isinstance(run_records, dict):
        raise ValueError("Validation report must record the compared model checkpoints.")
    selected_record = run_records.get(selected_model)
    if (
        not isinstance(selected_record, dict)
        or selected_record.get("run_id") != checkpoint_root.name
        or selected_record.get("manifest_sha256") != sha256_file(checkpoint_root / "run_manifest.json")
    ):
        raise ValueError("The selected checkpoint does not match the validation report.")
    additional = dict(additional_checkpoints or {})
    required_checkpoints = set(allowed_models) - {selected_model, "majority", "historical_choice"}
    if set(additional) != required_checkpoints:
        raise ValueError("Each additional trained comparison requires a frozen checkpoint.")
    additional_records = {}
    for name, root in additional.items():
        other, _ = verify_validation_checkpoint(root, expected_run_id=root.name, expected_split=split_manifest)
        if other.get("dataset_sha256") != checkpoint["dataset_sha256"]:
            raise ValueError("Compared checkpoints must use the same dataset.")
        report_record = run_records.get(name)
        if (
            not isinstance(report_record, dict)
            or report_record.get("run_id") != root.name
            or report_record.get("manifest_sha256") != sha256_file(root / "run_manifest.json")
        ):
            raise ValueError("A comparison checkpoint does not match the validation report.")
        additional_records[name] = {
            "run_id": root.name,
            "manifest_sha256": sha256_file(root / "run_manifest.json"),
        }
    selection = {
        "version": 1,
        "status": "frozen",
        "selected_model": selected_model,
        "allowed_models": allowed_models,
        "run_id": checkpoint_root.name,
        "dataset_sha256": checkpoint["dataset_sha256"],
        "provenance_mode": checkpoint.get("provenance_mode", "verified"),
        "split_sha256": split_hash,
        "validation_report_sha256": sha256_file(validation_report),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_root / "run_manifest.json"),
        "additional_checkpoints": additional_records,
        "lean_membership_sha256": sha256_file(lean_membership) if lean_membership is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return selection


def verify_frozen_selection(
    path: Path,
    *,
    split_manifest: Path,
    validation_report: Path,
    checkpoint_root: Path,
    additional_checkpoints: Optional[Mapping[str, Path]] = None,
    lean_membership: Optional[Path] = None,
) -> Dict[str, Any]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if (
        selection.get("version") != 1
        or selection.get("status") != "frozen"
        or selection.get("run_id") != checkpoint_root.name
        or selection.get("split_sha256") != sha256_file(split_manifest)
        or selection.get("validation_report_sha256") != sha256_file(validation_report)
        or selection.get("checkpoint_manifest_sha256") != sha256_file(checkpoint_root / "run_manifest.json")
    ):
        raise RuntimeError("Frozen selection does not match the recorded validation artifacts.")
    checkpoint, _ = verify_validation_checkpoint(
        checkpoint_root, expected_run_id=checkpoint_root.name, expected_split=split_manifest
    )
    if lean_membership is not None:
        raw_record = (checkpoint.get("inputs") or {}).get("raw_votes")
        if (
            selection.get("lean_membership_sha256") != sha256_file(lean_membership)
            or not isinstance(raw_record, dict)
            or not raw_record.get("sha256")
        ):
            raise RuntimeError("Lean sensitivity membership was not frozen with this selection.")
        load_lean_membership(lean_membership, raw_votes_sha256=raw_record["sha256"])
    additional = dict(additional_checkpoints or {})
    recorded = selection.get("additional_checkpoints")
    if not isinstance(recorded, dict) or set(recorded) != set(additional):
        raise RuntimeError("Frozen comparison checkpoints are incomplete.")
    for name, root in additional.items():
        record = recorded[name]
        if record.get("run_id") != root.name or record.get("manifest_sha256") != sha256_file(root / "run_manifest.json"):
            raise RuntimeError("A frozen comparison checkpoint has changed.")
        verify_validation_checkpoint(root, expected_run_id=root.name, expected_split=split_manifest)
    return selection
