"""Train one chronological model from verified example artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from dao_governance.clustering.predictive import load_verified_cluster_bundle
from dao_governance.data.canonical_dataset import load_proposal_catalog
from dao_governance.data.chronological_split import validate_chronological_manifest
from dao_governance.evaluation.point_in_time import baseline_predictions, evaluate_matched_predictions
from dao_governance.features.asof import ObservationProvenance
from dao_governance.modelling.point_in_time_artifacts import record_validation_evaluation, save_validation_checkpoint
from dao_governance.modelling.point_in_time_model import VoteChoiceClassifier, prepare_training_datasets
from dao_governance.modelling.point_in_time_training import fit_vote_classifier, predict_vote_classifier
from dao_governance.modelling.run_artifacts import resolve_artifact_path, sha256_file, validate_run_id


def _validate_partition_coverage(
    canonical_events: Path,
    manifest: Dict[str, Any],
    frames: Dict[str, pd.DataFrame],
) -> None:
    if not frames or not set(frames).issubset({"train", "validation", "test"}):
        raise ValueError("Coverage verification requires recognized target splits.")
    membership = {
        (row["space"], row["proposal_id"]): row["split"]
        for row in manifest["proposals"]
        if row["split"] in frames
    }
    parquet = pq.ParquetFile(canonical_events)
    id_type = parquet.schema_arrow.field("proposal_id").type
    proposal_ids = pa.array(sorted({key[1] for key in membership}), type=id_type)
    expected: Dict[str, list[Dict[str, Any]]] = {name: [] for name in frames}
    seen = set()
    columns = ["example_id", "space", "proposal_id", "vote_ts", "label_id"]
    for batch in parquet.iter_batches(batch_size=65536, columns=columns, use_threads=False):
        table = pa.Table.from_batches([batch])
        selected = table.filter(pc.is_in(table["proposal_id"], value_set=proposal_ids))
        for row in selected.to_pylist():
            split = membership.get((row["space"], row["proposal_id"]))
            if split is None:
                continue
            if row["example_id"] in seen:
                raise ValueError("Canonical target examples contain duplicate IDs.")
            seen.add(row["example_id"])
            expected[split].append(row)
    for split, frame in frames.items():
        if not set(columns).issubset(frame.columns):
            raise ValueError("Target examples lack canonical identity columns.")
        expected_frame = pd.DataFrame(expected[split], columns=columns)
        actual = frame[columns].copy()
        for item in (expected_frame, actual):
            for column in ("example_id", "space", "proposal_id"):
                item[column] = item[column].astype("string")
            item["vote_ts"] = pd.to_datetime(item["vote_ts"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
            labels = pd.to_numeric(item["label_id"], errors="raise")
            if not labels.isin([0, 1, 2]).all() or labels.map(lambda value: isinstance(value, bool)).any():
                raise ValueError("Target labels must use the three canonical classes.")
            item["label_id"] = labels.astype("int64")
            item.sort_values("example_id", inplace=True)
            item.reset_index(drop=True, inplace=True)
        if not actual.equals(expected_frame):
            raise ValueError(f"{split.capitalize()} examples do not exactly cover their canonical vote events.")


def train_chronological_run(
    *,
    examples_dir: Path,
    raw_source: Path,
    metadata_cache: Path,
    historical_provenance: Optional[Path],
    canonical_events: Path,
    observations: Optional[Path],
    observation_provenance: Optional[Path],
    run_root: Path,
    run_id: str,
    model_kind: Literal["numeric", "text", "fused"],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    text_loader: Optional[Callable[[], Tuple[Any, torch.nn.Module]]],
    text_max_length: int,
    freeze_text_encoder: bool,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    bootstrap_samples: int,
    seed: int = 42,
    device: str = "cpu",
    dao_bundle_path: Optional[Path] = None,
    voter_bundle_path: Optional[Path] = None,
    local_proxy: bool = False,
) -> Dict[str, Any]:
    if run_root.name != validate_run_id(run_id):
        raise ValueError("Run directory name must match the run ID.")
    if run_root.exists():
        raise FileExistsError("Chronological run directory already exists.")
    if model_kind not in {"numeric", "text", "fused"}:
        raise ValueError("Choose a numeric, text, or fused model.")
    if epochs < 1 or learning_rate <= 0 or batch_size < 1 or bootstrap_samples < 1:
        raise ValueError("Training and bootstrap settings must be positive.")
    use_text = model_kind in {"text", "fused"}
    use_numeric = model_kind in {"numeric", "fused"}
    if use_text != (text_loader is not None):
        raise ValueError("The selected model requires a matching text-loader configuration.")
    if (use_text and text_max_length < 1) or (not use_text and freeze_text_encoder):
        raise ValueError("The selected model has invalid text-encoder settings.")
    if not use_numeric and (numeric_columns or categorical_columns):
        raise ValueError("Text-only models cannot receive numeric or categorical inputs.")
    if use_numeric and not (numeric_columns or categorical_columns):
        raise ValueError("Numeric and fused models require at least one non-text feature.")
    if local_proxy and (historical_provenance is not None or observations is not None or observation_provenance is not None):
        raise ValueError("Local-proxy mode does not accept verified provenance or observation files.")
    if not local_proxy and (historical_provenance is None or observations is None or observation_provenance is None):
        raise ValueError("Verified mode requires historical provenance and observation files.")

    examples_manifest_path = examples_dir / "examples_manifest.json"
    examples_manifest = json.loads(examples_manifest_path.read_text(encoding="utf-8"))
    split_path = examples_dir / "split_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_chronological_manifest(split)
    split_hash = sha256_file(split_path)
    catalog, fingerprint, _ = load_proposal_catalog(
        raw_source, metadata_cache, historical_provenance, local_proxy=local_proxy
    )
    audit_path = canonical_events.with_name(canonical_events.name + ".audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    events_hash = sha256_file(canonical_events)
    if (
        examples_manifest.get("version") != 1
        or examples_manifest.get("dataset_sha256") != fingerprint
        or split["dataset_sha256"] != fingerprint
        or examples_manifest.get("split_sha256") != split_hash
        or audit.get("dataset_sha256") != fingerprint
        or audit.get("canonical_events_sha256") != events_hash
        or examples_manifest.get("canonical_events_sha256") != events_hash
        or examples_manifest.get("provenance_mode") != ("local_proxy" if local_proxy else "verified")
        or split.get("provenance_mode", "verified") != ("local_proxy" if local_proxy else "verified")
        or audit.get("provenance_mode") != ("local_proxy" if local_proxy else "verified")
    ):
        raise ValueError("Chronological examples or their source provenance do not match.")
    recorded_observation = examples_manifest.get("observation_provenance")
    if not isinstance(recorded_observation, dict) or not isinstance(recorded_observation.get("evidence"), str):
        raise ValueError("Chronological examples lack observation-time provenance.")
    if local_proxy:
        if (
            recorded_observation.get("verified") is not False
            or recorded_observation.get("local_proxy") is not True
            or examples_manifest.get("observations_sha256") is not None
            or examples_manifest.get("observation_provenance_sha256") is not None
        ):
            raise ValueError("Local-proxy examples must explicitly record unverified vote-time assumptions.")
        provenance = ObservationProvenance(False, recorded_observation["evidence"], local_proxy=True)
    else:
        if historical_provenance is None or observations is None or observation_provenance is None:
            raise RuntimeError("Verified provenance inputs are missing.")
        observation = json.loads(observation_provenance.read_text(encoding="utf-8"))
        observations_hash = sha256_file(observations)
        if (
            examples_manifest.get("observations_sha256") != observations_hash
            or examples_manifest.get("observation_provenance_sha256") != sha256_file(observation_provenance)
            or observation.get("version") != 1
            or observation.get("verified") is not True
            or observation.get("canonical_events_sha256") != events_hash
            or observation.get("observations_sha256") != observations_hash
            or recorded_observation != {"verified": True, "local_proxy": False, "evidence": observation.get("evidence")}
        ):
            raise ValueError("Chronological examples or their observation provenance do not match.")
        provenance = ObservationProvenance(True, recorded_observation["evidence"])
    provenance.validate()
    history_mode = examples_manifest.get("history_mode")
    if history_mode not in {"other_proposals_only", "target_proposal_sensitivity"}:
        raise ValueError("Chronological example history mode is invalid.")

    partitions = examples_manifest.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError("Chronological examples are missing partition records.")
    frames = {}
    input_artifacts = {
        "raw_votes": raw_source,
        "proposal_metadata": metadata_cache,
        "canonical_events": canonical_events,
        "canonical_audit": audit_path,
        "examples_manifest": examples_manifest_path,
    }
    if local_proxy:
        input_artifacts["local_metadata_binding"] = metadata_cache.with_name(metadata_cache.name + ".binding.json")
    else:
        if historical_provenance is None or observations is None or observation_provenance is None:
            raise RuntimeError("Verified provenance inputs are missing.")
        input_artifacts.update(
            historical_provenance=historical_provenance,
            observations=observations,
            observation_provenance=observation_provenance,
        )
    for split_name in ("train", "validation"):
        record = partitions.get(split_name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("Chronological examples are missing a training or validation partition.")
        path = resolve_artifact_path(examples_dir, record["path"])
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError("A chronological training or validation partition changed.")
        frame = pd.read_parquet(path)
        if len(frame) != record.get("rows"):
            raise ValueError("A chronological partition row count changed.")
        frames[split_name] = frame
        input_artifacts[f"{split_name}_examples"] = path
    _validate_partition_coverage(canonical_events, split, frames)
    for kind, path in (("dao", dao_bundle_path), ("voter", voter_bundle_path)):
        if (f"{kind}_cluster" in categorical_columns) != (path is not None):
            raise ValueError(f"{kind.capitalize()} cluster inputs require exactly one matching bundle.")
        if path is not None:
            if sha256_file(path) != examples_manifest.get(f"{kind}_bundle_sha256"):
                raise ValueError("A cluster bundle differs from the one used to create the examples.")
            bundle, _ = load_verified_cluster_bundle(path, kind=kind, split_fingerprint=split_hash)
            if kind == "voter" and bundle.scope != examples_manifest.get("voter_cluster_scope"):
                raise ValueError("Voter-clustering scope differs from the example artifacts.")

    torch.manual_seed(seed)
    tokenizer, text_encoder = text_loader() if text_loader is not None else (None, None)
    train, validation, feature_encoder, text_cache = prepare_training_datasets(
        frames["train"],
        frames["validation"],
        catalog.eligible,
        split,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        tokenizer=tokenizer,
        max_length=text_max_length,
    )
    model = VoteChoiceClassifier(
        numeric_dim=feature_encoder.dimension,
        text_encoder=text_encoder,
        freeze_text_encoder=freeze_text_encoder,
    )
    training = fit_vote_classifier(
        model, train, validation, manifest=split, epochs=epochs, learning_rate=learning_rate,
        batch_size=batch_size, seed=seed, device=device,
    )
    predictions = baseline_predictions(frames["train"], frames["validation"])
    predictions["model"] = predict_vote_classifier(model, validation, batch_size=batch_size, device=device)
    report = evaluate_matched_predictions(
        predictions, manifest=split, split="validation", split_fingerprint=split_hash,
        bootstrap_samples=bootstrap_samples, seed=seed, strata=frames["validation"],
    )
    report["model_kind"] = model_kind
    report["provenance_mode"] = "local_proxy" if local_proxy else "verified"
    report["history_mode"] = history_mode
    report["tokenized_proposals"] = len(text_cache.tokens) if text_cache is not None else 0
    report["truncated_proposals"] = text_cache.truncated if text_cache is not None else 0
    save_validation_checkpoint(
        run_root,
        run_id=run_id,
        model=model,
        feature_encoder=feature_encoder,
        split_manifest=split_path,
        training=training,
        observation_provenance=provenance,
        history_mode=history_mode,
        tokenizer=tokenizer,
        dao_bundle_path=dao_bundle_path,
        voter_bundle_path=voter_bundle_path,
        text_max_length=text_max_length if use_text else None,
        input_artifacts=input_artifacts,
    )
    report_path = record_validation_evaluation(run_root, predictions=predictions, report=report)
    return {
        "run_id": run_id,
        "model_kind": model_kind,
        "best_validation_macro_f1": training.best_validation_macro_f1,
        "validation_report": str(report_path),
    }
