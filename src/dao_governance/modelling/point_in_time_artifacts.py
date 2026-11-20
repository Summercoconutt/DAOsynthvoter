"""Immutable, validation-selected artifacts for chronological vote models."""
from __future__ import annotations

import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import pandas as pd
import torch

from dao_governance.data.chronological_split import validate_chronological_manifest
from dao_governance.data.reporting import json_safe_report
from dao_governance.clustering.predictive import load_verified_cluster_bundle
from dao_governance.features.asof import ObservationProvenance
from dao_governance.features.point_in_time_clusters import PointInTimeClusterBundle
from dao_governance.modelling.model import copy_state_dict_to_cpu
from dao_governance.modelling.point_in_time_model import FeatureEncoder, VoteChoiceClassifier
from dao_governance.modelling.point_in_time_training import TrainingResult
from dao_governance.modelling.run_artifacts import resolve_artifact_path, sha256_file, validate_run_id


def save_validation_checkpoint(
    run_root: Path,
    *,
    run_id: str,
    model: VoteChoiceClassifier,
    feature_encoder: FeatureEncoder,
    split_manifest: Path,
    training: TrainingResult,
    observation_provenance: ObservationProvenance,
    history_mode: Literal["other_proposals_only", "target_proposal_sensitivity"],
    tokenizer: Optional[Any] = None,
    dao_bundle_path: Optional[Path] = None,
    voter_bundle_path: Optional[Path] = None,
    text_max_length: Optional[int] = None,
    input_artifacts: Optional[Mapping[str, Path]] = None,
) -> Dict[str, Any]:
    run_id = validate_run_id(run_id)
    if run_root.name != run_id:
        raise ValueError("Run directory name must match the run ID.")
    if not feature_encoder.fitted or feature_encoder.dimension != model.numeric_dim:
        raise ValueError("The fitted feature encoder must match the model.")
    if (model.text_encoder is None) != (tokenizer is None):
        raise ValueError("Text models require the matching tokenizer.")
    if model.text_encoder is not None and (text_max_length is None or text_max_length < 1):
        raise ValueError("Text models require a positive tokenizer length.")
    if training.best_epoch < 1 or not 0 <= training.best_validation_macro_f1 <= 1:
        raise ValueError("A valid validation-selected checkpoint is required.")
    if not isinstance(observation_provenance, ObservationProvenance):
        raise ValueError("Verified vote observation-time provenance is required.")
    observation_provenance.validate()
    if history_mode not in {"other_proposals_only", "target_proposal_sensitivity"}:
        raise ValueError("A valid point-in-time history mode is required.")
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    validate_chronological_manifest(split)
    provenance_mode = split.get("provenance_mode", "verified")
    if observation_provenance.local_proxy and provenance_mode != "local_proxy":
        raise ValueError("A local vote-time proxy requires a local-proxy split.")
    split_hash = sha256_file(split_manifest)
    inputs = {}
    for name, path in (input_artifacts or {}).items():
        if not name or not path.is_file():
            raise ValueError("Checkpoint inputs require named existing files.")
        inputs[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    bundle_paths = {"dao": dao_bundle_path, "voter": voter_bundle_path}
    bundle_metadata = {}
    for kind, path in bundle_paths.items():
        required = f"{kind}_cluster" in feature_encoder.categorical_columns
        if required != (path is not None):
            raise ValueError(f"{kind.capitalize()} cluster inputs require exactly one matching bundle.")
        if path is not None:
            bundle, metadata = load_verified_cluster_bundle(path, kind=kind, split_fingerprint=split_hash)
            bundle_metadata[kind] = (bundle, metadata)
    if "dao" in bundle_metadata and "voter" in bundle_metadata:
        voter_bundle, voter_metadata = bundle_metadata["voter"]
        if dao_bundle_path is None:
            raise RuntimeError("The DAO cluster bundle path is missing.")
        if voter_bundle.scope == "dao_cluster" and voter_metadata.get("dao_bundle_sha256") != sha256_file(dao_bundle_path):
            raise ValueError("DAO classifier input and voter grouping must use the same DAO bundle.")
    text_enabled = model.text_encoder is not None
    encoder_config = getattr(model.text_encoder, "config", None)
    save_config = getattr(encoder_config, "save_pretrained", None)
    save_tokenizer = getattr(tokenizer, "save_pretrained", None)
    if text_enabled and (not callable(save_config) or not callable(save_tokenizer)):
        raise ValueError("Text encoder and tokenizer must support local serialization.")
    first_layer, dropout = model.head[0], model.head[2]
    if not isinstance(first_layer, torch.nn.Linear) or not isinstance(dropout, torch.nn.Dropout):
        raise ValueError("The classifier head has an unsupported structure.")
    run_root.mkdir(parents=True, exist_ok=False)
    shared = run_root / "shared"
    model_dir = run_root / "model"
    shared.mkdir()
    model_dir.mkdir()
    split_copy = shared / "split_manifest.json"
    with split_manifest.open("rb") as source, split_copy.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    if json.loads(split_copy.read_text(encoding="utf-8")) != split:
        raise RuntimeError("The split manifest changed while the checkpoint was being saved.")
    torch.save(copy_state_dict_to_cpu(model), model_dir / "model.pt")
    with (model_dir / "feature_encoder.pkl").open("xb") as stream:
        pickle.dump(feature_encoder, stream)
    if bundle_metadata:
        cluster_dir = model_dir / "clusters"
        cluster_dir.mkdir()
        for kind, path in bundle_paths.items():
            if path is not None:
                for source in (path, path.with_name(path.name + ".meta.json")):
                    suffix = ".pkl" if source == path else ".pkl.meta.json"
                    with source.open("rb") as src, (cluster_dir / f"{kind}{suffix}").open("xb") as dest:
                        shutil.copyfileobj(src, dest)

    if text_enabled:
        save_config(model_dir / "text_encoder_config")
        save_tokenizer(model_dir / "tokenizer")

    config = {
        "version": 1,
        "protocol": "chronological_proposal_disjoint",
        "provenance_mode": provenance_mode,
        "numeric_columns": list(feature_encoder.numeric_columns),
        "categorical_columns": list(feature_encoder.categorical_columns),
        "voter_cluster_scope": bundle_metadata["voter"][0].scope if "voter" in bundle_metadata else None,
        "numeric_dim": model.numeric_dim,
        "text_enabled": text_enabled,
        "text_encoder_frozen": model.text_encoder_frozen,
        "text_max_length": text_max_length,
        "history_mode": history_mode,
        "observation_provenance": {
            "verified": observation_provenance.verified,
            "local_proxy": observation_provenance.local_proxy,
            "evidence": observation_provenance.evidence,
        },
        "hidden_dim": first_layer.out_features,
        "dropout": dropout.p,
        "label_map": {"FOR": 0, "AGAINST": 1, "ABSTAIN": 2},
        "best_epoch": training.best_epoch,
        "best_validation_macro_f1": training.best_validation_macro_f1,
        "training_history": training.epochs,
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = []
    for path in sorted(p for p in run_root.rglob("*") if p.is_file()):
        artifacts.append(
            {"relative_path": path.relative_to(run_root).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    manifest = {
        "version": 1,
        "protocol": "chronological_proposal_disjoint",
        "provenance_mode": provenance_mode,
        "run_id": run_id,
        "status": "validation_checkpoint",
        "dataset_sha256": split["dataset_sha256"],
        "split_sha256": sha256_file(split_copy),
        "inputs": inputs,
        "artifacts": artifacts,
    }
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_validation_checkpoint(
    run_root: Path,
    *,
    expected_run_id: str,
    expected_split: Path,
    input_overrides: Optional[Mapping[str, Path]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("version") != 1
        or manifest.get("protocol") != "chronological_proposal_disjoint"
        or manifest.get("run_id") != validate_run_id(expected_run_id)
        or run_root.name != expected_run_id
        or manifest.get("status") != "validation_checkpoint"
        or manifest.get("split_sha256") != sha256_file(expected_split)
    ):
        raise RuntimeError("Chronological run identity or split does not match.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("Chronological run is missing its artifact records.")
    if not all(isinstance(record, dict) for record in artifacts):
        raise RuntimeError("Chronological run artifact records are invalid.")
    paths = [record.get("relative_path") for record in artifacts]
    required = {"shared/split_manifest.json", "model/model.pt", "model/config.json", "model/feature_encoder.pkl"}
    if not all(isinstance(path, str) and path for path in paths) or len(set(paths)) != len(paths) or not required.issubset(paths):
        raise RuntimeError("Chronological run is missing required artifact records.")
    for record in artifacts:
        path = resolve_artifact_path(run_root, record["relative_path"])
        if not path.is_file() or path.stat().st_size != record.get("size_bytes") or sha256_file(path) != record.get("sha256"):
            raise RuntimeError("A chronological run artifact is missing or changed.")
    inputs = manifest.get("inputs") or {}
    overrides = dict(input_overrides or {})
    if set(overrides) - set(inputs):
        raise ValueError("Input overrides contain names not recorded in the run.")
    for name, record in inputs.items():
        path = overrides.get(name, Path(record["path"]))
        if not path.is_file() or path.stat().st_size != record.get("size_bytes") or sha256_file(path) != record.get("sha256"):
            raise RuntimeError("A chronological run input is missing or changed.")
        if name in overrides:
            record["path"] = str(path.resolve())

    model_dir = run_root / "model"
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("version") != 1 or config.get("protocol") != "chronological_proposal_disjoint":
        raise RuntimeError("Chronological model configuration is invalid.")
    observation = config.get("observation_provenance")
    if (
        config.get("history_mode") not in {"other_proposals_only", "target_proposal_sensitivity"}
        or not isinstance(observation, dict)
        or not isinstance(observation.get("evidence"), str)
        or not observation["evidence"].strip()
        or config.get("provenance_mode", "verified") != manifest.get("provenance_mode", "verified")
        or not (
            (observation.get("verified") is True and not observation.get("local_proxy", False))
            or (observation.get("verified") is False and observation.get("local_proxy") is True
                and config.get("provenance_mode") == "local_proxy")
        )
    ):
        raise RuntimeError("Chronological model observation provenance is invalid.")
    if config.get("text_enabled"):
        for directory in ("text_encoder_config", "tokenizer"):
            actual = {
                path.relative_to(run_root).as_posix()
                for path in (model_dir / directory).rglob("*")
                if path.is_file()
            }
            recorded = {path for path in paths if path.startswith(f"model/{directory}/")}
            if not actual or actual != recorded:
                raise RuntimeError("Chronological text artifacts are incomplete or unrecorded.")
    for kind in ("dao", "voter"):
        if f"{kind}_cluster" in config.get("categorical_columns", []):
            expected = {f"model/clusters/{kind}.pkl", f"model/clusters/{kind}.pkl.meta.json"}
            if not expected.issubset(paths):
                raise RuntimeError("Chronological cluster artifacts are incomplete or unrecorded.")
    return manifest, config


def load_validation_checkpoint(
    run_root: Path,
    *,
    expected_run_id: str,
    expected_split: Path,
    device: str = "cpu",
    input_overrides: Optional[Mapping[str, Path]] = None,
) -> Tuple[VoteChoiceClassifier, FeatureEncoder, Optional[Any], Dict[str, Any]]:
    manifest, config = verify_validation_checkpoint(
        run_root, expected_run_id=expected_run_id, expected_split=expected_split,
        input_overrides=input_overrides,
    )
    model_dir = run_root / "model"
    with (model_dir / "feature_encoder.pkl").open("rb") as stream:
        feature_encoder = pickle.load(stream)
    if not isinstance(feature_encoder, FeatureEncoder) or not feature_encoder.fitted:
        raise RuntimeError("The chronological run has an invalid feature encoder.")
    if (
        list(feature_encoder.numeric_columns) != config.get("numeric_columns")
        or list(feature_encoder.categorical_columns) != config.get("categorical_columns")
    ):
        raise RuntimeError("The saved feature schema does not match the model configuration.")
    text_encoder = None
    tokenizer = None
    if config["text_enabled"]:
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        text_config = AutoConfig.from_pretrained(model_dir / "text_encoder_config", local_files_only=True)
        text_encoder = AutoModel.from_config(text_config)
        tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    model = VoteChoiceClassifier(
        numeric_dim=int(config["numeric_dim"]),
        text_encoder=text_encoder,
        freeze_text_encoder=bool(config["text_encoder_frozen"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    )
    if feature_encoder.dimension != model.numeric_dim:
        raise RuntimeError("The saved feature encoder does not match the model.")
    state = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device), feature_encoder, tokenizer, manifest


def load_checkpoint_clusters(
    run_root: Path,
    *,
    expected_run_id: str,
    expected_split: Path,
    input_overrides: Optional[Mapping[str, Path]] = None,
) -> Dict[str, PointInTimeClusterBundle]:
    manifest, config = verify_validation_checkpoint(
        run_root, expected_run_id=expected_run_id, expected_split=expected_split,
        input_overrides=input_overrides,
    )
    bundles = {}
    for kind in ("dao", "voter"):
        if f"{kind}_cluster" in config.get("categorical_columns", []):
            path = run_root / "model" / "clusters" / f"{kind}.pkl"
            bundles[kind], _ = load_verified_cluster_bundle(
                path, kind=kind, split_fingerprint=manifest["split_sha256"]
            )
    return bundles


def record_validation_evaluation(
    run_root: Path,
    *,
    predictions: Mapping[str, pd.DataFrame],
    report: Mapping[str, Any],
) -> Path:
    split = run_root / "shared" / "split_manifest.json"
    manifest, _ = verify_validation_checkpoint(
        run_root, expected_run_id=run_root.name, expected_split=split
    )
    if (
        report.get("split") != "validation"
        or report.get("split_sha256") != manifest["split_sha256"]
        or report.get("dataset_sha256") != manifest["dataset_sha256"]
        or set(predictions) != set(report.get("models", {}))
    ):
        raise ValueError("Validation predictions and report do not match the checkpoint.")
    output = run_root / "validation"
    output.mkdir(exist_ok=False)
    for name, frame in predictions.items():
        validate_run_id(name)
        frame.to_csv(output / f"{name}.csv", index=False)
    payload = json_safe_report(report)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    for path in sorted(output.iterdir()):
        manifest["artifacts"].append(
            {"relative_path": path.relative_to(run_root).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    manifest_path = run_root / "run_manifest.json"
    temporary = run_root / ".run_manifest.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, manifest_path)
    return report_path
