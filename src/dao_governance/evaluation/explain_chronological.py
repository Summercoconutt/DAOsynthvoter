"""Reproducible, training-background explanations for a frozen test model."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import torch

from dao_governance.data.canonical_dataset import load_proposal_catalog
from dao_governance.data.chronological_split import validate_proposal_metadata
from dao_governance.data.lean_sensitivity import load_lean_membership
from dao_governance.data.reporting import json_safe_report
from dao_governance.evaluation.chronological_runs import (
    _prediction_csv, _test_examples, filter_content_disjoint_test, filter_lean_test,
)
from dao_governance.evaluation.point_in_time import verify_frozen_selection
from dao_governance.modelling.point_in_time_artifacts import load_validation_checkpoint
from dao_governance.modelling.point_in_time_explain import integrated_gradients_numeric, integrated_gradients_text
from dao_governance.modelling.point_in_time_model import (
    ProposalTextCache, VoteDataset, collate_vote_examples, validate_target_examples,
)
from dao_governance.modelling.run_artifacts import resolve_artifact_path, sha256_file, validate_run_id


def _verify_evaluation(path: Path, selection_path: Path, selected_model: str) -> pd.DataFrame:
    validate_run_id(selected_model)
    manifest = json.loads((path / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("selection_sha256") != sha256_file(selection_path):
        raise RuntimeError("Test evaluation does not match the frozen selection.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("Test evaluation is missing artifact records.")
    recorded = set()
    for record in artifacts:
        relative = record["path"]
        if relative in recorded:
            raise RuntimeError("Test evaluation contains duplicate artifact records.")
        recorded.add(relative)
        artifact = resolve_artifact_path(path, relative)
        if not artifact.is_file() or artifact.stat().st_size != record.get("size_bytes") or sha256_file(artifact) != record.get("sha256"):
            raise RuntimeError("A test evaluation artifact is missing or changed.")
    prediction_name = f"{selected_model}_predictions.csv"
    if prediction_name not in recorded or "metrics.json" not in recorded:
        raise RuntimeError("Selected model predictions or metrics are not recorded.")
    return _prediction_csv(path / prediction_name)


def _cases(predictions: pd.DataFrame, max_cases: int) -> list[str]:
    if max_cases < 1:
        raise ValueError("At least one explanation case is required.")
    selected = []
    for label in (0, 1, 2):
        group = predictions[predictions["y_true"] == label].copy()
        if group.empty:
            continue
        group["confidence"] = group[["prob_FOR", "prob_AGAINST", "prob_ABSTAIN"]].max(axis=1)
        candidates = [
            group[group["y_pred"] != label].sort_values("example_id"),
            group[group["y_pred"] == label].sort_values("example_id"),
            group.sort_values(["confidence", "example_id"]),
        ]
        for candidate in candidates:
            if not candidate.empty:
                example_id = str(candidate.iloc[0]["example_id"])
                if example_id not in selected:
                    selected.append(example_id)
                    if len(selected) == max_cases:
                        return selected
    return selected


def _global_sample(predictions: pd.DataFrame, size: int, seed: int) -> list[str]:
    if size < 1:
        raise ValueError("The global explanation sample must contain at least one proposal.")
    groups = list(predictions.groupby(["space", "proposal_id"], sort=True).indices.values())
    if not groups:
        raise ValueError("No test proposals are available for the global explanation sample.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(groups), size=min(size, len(groups)), replace=False)
    return [str(predictions.iloc[int(rng.choice(groups[index]))]["example_id"]) for index in chosen]


def _feature_groups(encoder) -> list[tuple[str, list[int]]]:
    groups = [(name, [index]) for index, name in enumerate(encoder.numeric_columns)]
    offset = len(encoder.numeric_columns)
    for column in encoder.categorical_columns:
        count = len(encoder.categories[column])
        groups.append((column, list(range(offset, offset + count))))
        offset += count
    return groups


def _cosine(left: np.ndarray, right: np.ndarray) -> Any:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else None


def explain_frozen_test(
    *,
    selection_path: Path,
    validation_report: Path,
    checkpoint_root: Path,
    additional_checkpoints: Mapping[str, Path],
    test_evaluation_dir: Path,
    output_dir: Path,
    max_cases: int = 9,
    global_sample_size: int = 20,
    background_size: int = 2,
    steps: int = 64,
    seed: int = 42,
    completeness_tolerance: float = 1e-3,
    lean_membership: Path | None = None,
) -> Dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("Explanation output directory already exists.")
    if background_size < 1 or steps < 1 or completeness_tolerance <= 0:
        raise ValueError("Explanation background size, integration steps, and tolerance must be positive.")
    split_path = checkpoint_root / "shared" / "split_manifest.json"
    selection = verify_frozen_selection(
        selection_path, split_manifest=split_path, validation_report=validation_report,
        checkpoint_root=checkpoint_root, additional_checkpoints=additional_checkpoints, lean_membership=lean_membership,
    )
    predictions = _verify_evaluation(test_evaluation_dir, selection_path, selection["selected_model"])
    model, encoder, tokenizer, manifest = load_validation_checkpoint(
        checkpoint_root, expected_run_id=checkpoint_root.name, expected_split=split_path, device="cpu"
    )
    model.eval()
    config = json.loads((checkpoint_root / "model/config.json").read_text(encoding="utf-8"))
    test = _test_examples(checkpoint_root, manifest)
    inputs = manifest.get("inputs") or {}
    local_proxy = manifest.get("provenance_mode") == "local_proxy"
    required = {"train_examples", "raw_votes", "proposal_metadata"}
    required.add("local_metadata_binding" if local_proxy else "historical_provenance")
    if not required.issubset(inputs):
        raise RuntimeError("The selected run lacks training-background or proposal provenance.")
    train = pd.read_parquet(inputs["train_examples"]["path"])
    if train.empty:
        raise ValueError("Explanation backgrounds require training examples.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_metrics = json.loads((test_evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    if test_metrics.get("content_mode") == "content_disjoint_sensitivity":
        test = filter_content_disjoint_test(test, split)
    elif test_metrics.get("content_mode") != "primary":
        raise ValueError("Test evaluation content mode is invalid.")
    if test_metrics.get("population") == "outcome_selected_lean_sensitivity":
        if lean_membership is None or test_metrics.get("lean_membership_sha256") != sha256_file(lean_membership):
            raise ValueError("Lean explanations require the frozen lean membership.")
        raw_record = (manifest.get("inputs") or {}).get("raw_votes")
        if not isinstance(raw_record, dict) or not raw_record.get("sha256"):
            raise RuntimeError("The selected run lacks raw vote provenance for lean sensitivity.")
        test = filter_lean_test(test, load_lean_membership(lean_membership, raw_votes_sha256=raw_record["sha256"]))
    elif test_metrics.get("population") != "full_eligible":
        raise ValueError("Test evaluation population is invalid.")
    validate_target_examples(train, split, "train")
    validate_target_examples(test, split, "test")
    if predictions["example_id"].duplicated().any() or set(predictions["example_id"]) != set(test["example_id"]):
        raise ValueError("Test predictions do not match the held-out examples.")
    aligned = predictions.merge(
        test[["example_id", "space", "proposal_id", "vote_ts", "label_id"]],
        on="example_id", suffixes=("_prediction", "_example"), validate="one_to_one",
    )
    if (
        (aligned["space_prediction"] != aligned["space_example"]).any()
        or (aligned["proposal_id_prediction"] != aligned["proposal_id_example"]).any()
        or (
            pd.to_datetime(aligned["vote_ts_prediction"], utc=True)
            != pd.to_datetime(aligned["vote_ts_example"], utc=True)
        ).any()
        or (aligned["y_true"] != aligned["label_id"]).any()
    ):
        raise ValueError("Test predictions and held-out examples are not aligned.")
    case_ids = _cases(predictions, max_cases)
    if not case_ids:
        raise ValueError("No test cases are available for explanation.")
    global_ids = _global_sample(predictions, global_sample_size, seed + 1)
    all_ids = list(dict.fromkeys([*case_ids, *global_ids]))
    global_set = set(global_ids)
    local_set = set(case_ids)
    rng = np.random.default_rng(seed)
    background_indices = rng.choice(len(train), size=min(background_size, len(train)), replace=False)
    backgrounds = train.iloc[background_indices].reset_index(drop=True)
    background_numeric = encoder.transform(backgrounds)
    selected = test.set_index("example_id").loc[all_ids].reset_index()
    selected_numeric = encoder.transform(selected)
    historical = Path(inputs["historical_provenance"]["path"]) if not local_proxy else None
    catalog, fingerprint, _ = load_proposal_catalog(
        Path(inputs["raw_votes"]["path"]), Path(inputs["proposal_metadata"]["path"]),
        historical, local_proxy=local_proxy,
    )
    if fingerprint != manifest["dataset_sha256"]:
        raise RuntimeError("Explanation proposal metadata differs from the selected run.")
    membership = {(row["space"], row["proposal_id"]): row for row in split["proposals"]}
    keys = set(zip(backgrounds["space"], backgrounds["proposal_id"])) | set(zip(selected["space"], selected["proposal_id"]))
    proposals = []
    for key in sorted(keys):
        proposal = catalog.eligible.get(key)
        if proposal is None:
            raise ValueError("An explanation proposal lacks verified metadata.")
        validate_proposal_metadata(proposal, membership[key])
        proposals.append(proposal)
    cache = ProposalTextCache(proposals, tokenizer, max_length=int(config["text_max_length"])) if tokenizer is not None else None
    dataset = VoteDataset(selected, selected_numeric, cache)
    prediction_by_id = predictions.set_index("example_id")
    randomized = copy.deepcopy(model)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for layer in randomized.head:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    randomized.eval()
    groups = _feature_groups(encoder)
    cases = []
    aggregate: Dict[str, list[float]] = {name: [] for name, _ in groups}
    text_aggregate = []
    completeness_failures = 0
    for index, example_id in enumerate(all_ids):
        item = dataset[index]
        batch = collate_vote_examples([item], pad_token_id=cache.pad_token_id if cache else 0)
        tensor_batch = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
        predicted = int(prediction_by_id.loc[example_id, "y_pred"])
        with torch.no_grad():
            logits = model(tensor_batch)
            probabilities = torch.softmax(logits, dim=-1)[0].numpy()
        expected = prediction_by_id.loc[example_id, ["prob_FOR", "prob_AGAINST", "prob_ABSTAIN"]].to_numpy(dtype=float)
        if not np.allclose(probabilities, expected, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Reconstructed explanation inputs do not reproduce the saved prediction.")
        case: Dict[str, Any] = {
            "example_id": example_id,
            "space": item["space"],
            "proposal_id": item["proposal_id"],
            "true_class": int(prediction_by_id.loc[example_id, "y_true"]),
            "predicted_class": predicted,
            "confidence": float(probabilities[predicted]),
            "background_example_ids": backgrounds["example_id"].astype(str).tolist(),
        }
        if model.numeric_dim:
            numeric_results = [
                integrated_gradients_numeric(
                    model, tensor_batch, torch.from_numpy(row[None, :]), class_index=predicted,
                    steps=steps, background_split="train",
                )
                for row in background_numeric
            ]
            numeric_values = np.stack([result.attributions[0].numpy() for result in numeric_results])
            mean_numeric = numeric_values.mean(axis=0)
            group_scores = {name: float(mean_numeric[indices].sum()) for name, indices in groups}
            if example_id in global_set:
                for name, value in group_scores.items():
                    aggregate[name].append(abs(value))
            ranked = sorted(groups, key=lambda group: abs(group_scores[group[0]]), reverse=True)
            baseline = torch.from_numpy(background_numeric[0:1])
            ablation = {}
            for label, (name, indices) in (("high", ranked[0]), ("low", ranked[-1])):
                altered = dict(tensor_batch)
                altered["numeric_features"] = tensor_batch["numeric_features"].clone()
                altered["numeric_features"][:, indices] = baseline[:, indices]
                with torch.no_grad():
                    changed = torch.softmax(model(altered), dim=-1)[0, predicted].item()
                ablation[label] = {"feature": name, "probability_change": float(changed - probabilities[predicted])}
            random_result = integrated_gradients_numeric(
                randomized, tensor_batch, baseline, class_index=predicted, steps=steps, background_split="train"
            )
            case["numeric"] = {
                "group_attributions": group_scores,
                "completeness_deltas": [float(result.completeness_delta.item()) for result in numeric_results],
                "completeness_passed": all(
                    abs(result.completeness_delta.item()) <= completeness_tolerance for result in numeric_results
                ),
                "background_mean_absolute_difference": float(np.abs(numeric_values[0] - numeric_values[1]).mean()) if len(numeric_values) > 1 else None,
                "head_randomization_cosine": _cosine(numeric_values[0], random_result.attributions[0].numpy()),
                "ablation": ablation,
            }
        if model.text_encoder is not None and cache is not None:
            embedding = model.text_encoder.get_input_embeddings()
            target_length = tensor_batch["input_ids"].shape[1]
            text_results = []
            text_baselines = []
            for background in backgrounds.itertuples(index=False):
                token_ids = cache.tokens[(background.space, background.proposal_id)]["input_ids"][:target_length]
                token_ids = token_ids + [cache.pad_token_id] * (target_length - len(token_ids))
                baseline_embeddings = embedding(torch.tensor([token_ids], dtype=torch.long)).detach()
                text_baselines.append(baseline_embeddings)
                text_results.append(
                    integrated_gradients_text(
                        model, tensor_batch, baseline_embeddings, class_index=predicted,
                        steps=steps, background_split="train",
                    )
                )
            text_values = np.stack([result.attributions[0].sum(dim=-1).numpy() for result in text_results])
            if any(not np.isclose(result.input_score.item(), logits[0, predicted].item(), atol=1e-5) for result in text_results):
                raise RuntimeError("Embedding attribution does not reproduce the saved model logits.")
            mean_text = text_values.mean(axis=0)
            active = np.flatnonzero(tensor_batch["attention_mask"][0].numpy())
            if not len(active):
                raise ValueError("Text explanations require at least one unmasked token.")
            if example_id in global_set:
                text_aggregate.append(float(np.abs(mean_text[active]).mean()))
            ranked = active[np.argsort(-np.abs(mean_text[active]))]
            original_embeddings = embedding(tensor_batch["input_ids"]).detach()
            ablation = {}
            for label, position in (("high", int(ranked[0])), ("low", int(ranked[-1]))):
                altered = original_embeddings.clone()
                altered[:, position, :] = text_baselines[0][:, position, :]
                with torch.no_grad():
                    changed = torch.softmax(
                        model.forward_from_embeddings(tensor_batch["numeric_features"], altered, tensor_batch["attention_mask"]),
                        dim=-1,
                    )[0, predicted].item()
                ablation[label] = {"token_position": position, "probability_change": float(changed - probabilities[predicted])}
            random_text = integrated_gradients_text(
                randomized, tensor_batch, text_baselines[0], class_index=predicted, steps=steps, background_split="train"
            )
            case["text"] = {
                "token_count": int(tensor_batch["attention_mask"][0].sum().item()),
                "token_attributions": mean_text.tolist(),
                "completeness_deltas": [float(result.completeness_delta.item()) for result in text_results],
                "completeness_passed": all(
                    abs(result.completeness_delta.item()) <= completeness_tolerance for result in text_results
                ),
                "background_mean_absolute_difference": float(np.abs(text_values[0] - text_values[1]).mean()) if len(text_values) > 1 else None,
                "head_randomization_cosine": _cosine(text_values[0], random_text.attributions[0].sum(dim=-1).numpy()),
                "ablation": ablation,
            }
        if any(not case[modality]["completeness_passed"] for modality in ("numeric", "text") if modality in case):
            completeness_failures += 1
        if example_id in local_set:
            cases.append(case)

    report: Dict[str, Any] = {
        "version": 1,
        "selection_sha256": sha256_file(selection_path),
        "test_evaluation_manifest_sha256": sha256_file(test_evaluation_dir / "evaluation_manifest.json"),
        "selected_model": selection["selected_model"],
        "provenance_mode": manifest.get("provenance_mode", "verified"),
        "population": test_metrics["population"],
        "content_mode": test_metrics["content_mode"],
        "test_examples": len(test),
        "explained_examples": len(all_ids),
        "local_cases": len(cases),
        "global_sample_examples": len(global_ids),
        "global_sample_example_ids": global_ids,
        "global_sampling": "uniform proposal sample with one randomly selected vote per proposal",
        "background_source": "training examples only",
        "background_size": len(backgrounds),
        "integration_steps": steps,
        "completeness_tolerance": completeness_tolerance,
        "cases_with_completeness_failure": completeness_failures,
        "score_type": "class logit",
        "sampling_note": "Global summaries are proposal-weighted samples, not exact population-wide importance.",
        "text_baseline_note": "Training proposal token embeddings are padded or truncated to each target length.",
        "numeric_mean_absolute_group_attribution": {name: float(np.mean(values)) for name, values in aggregate.items() if values},
        "text_mean_absolute_token_attribution": float(np.mean(text_aggregate)) if text_aggregate else None,
        "cases": cases,
    }
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    output = output_dir / "explanations.json"
    output.write_text(json.dumps(json_safe_report(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "explanation_manifest.json").write_text(
        json.dumps({"version": 1, "selection_sha256": report["selection_sha256"],
                    "explanations_sha256": sha256_file(output)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
