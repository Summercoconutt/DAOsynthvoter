"""Generate a traceable research report from frozen evaluation artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from dao_governance.evaluation.explain_chronological import _verify_evaluation
from dao_governance.evaluation.point_in_time import verify_frozen_selection
from dao_governance.modelling.point_in_time_artifacts import verify_validation_checkpoint
from dao_governance.modelling.run_artifacts import sha256_file


def _number(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) and math.isfinite(value) else "undefined"


def write_final_report(
    *,
    selection_path: Path,
    validation_report: Path,
    checkpoint_root: Path,
    additional_checkpoints: Mapping[str, Path],
    test_evaluation_dir: Path,
    explanation_dir: Path,
    output: Path,
    lean_membership: Path | None = None,
) -> None:
    if output.exists():
        raise FileExistsError("Final report output already exists.")
    split_path = checkpoint_root / "shared/split_manifest.json"
    selection = verify_frozen_selection(
        selection_path, split_manifest=split_path, validation_report=validation_report,
        checkpoint_root=checkpoint_root, additional_checkpoints=additional_checkpoints, lean_membership=lean_membership,
    )
    _verify_evaluation(test_evaluation_dir, selection_path, selection["selected_model"])
    checkpoint, config = verify_validation_checkpoint(
        checkpoint_root, expected_run_id=checkpoint_root.name, expected_split=split_path
    )
    explanation_manifest = json.loads((explanation_dir / "explanation_manifest.json").read_text(encoding="utf-8"))
    explanation_path = explanation_dir / "explanations.json"
    if (
        explanation_manifest.get("version") != 1
        or explanation_manifest.get("selection_sha256") != sha256_file(selection_path)
        or explanation_manifest.get("explanations_sha256") != sha256_file(explanation_path)
    ):
        raise RuntimeError("Explanation artifacts do not match the frozen selection.")
    explanations = json.loads(explanation_path.read_text(encoding="utf-8"))
    if (
        explanations.get("selected_model") != selection["selected_model"]
        or explanations.get("test_evaluation_manifest_sha256") != sha256_file(test_evaluation_dir / "evaluation_manifest.json")
    ):
        raise RuntimeError("Explanations do not match the held-out evaluation.")
    metrics = json.loads((test_evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    provenance_mode = checkpoint.get("provenance_mode", "verified")
    if metrics.get("provenance_mode", "verified") != provenance_mode or explanations.get("provenance_mode", "verified") != provenance_mode:
        raise RuntimeError("Model, evaluation, and explanation provenance modes differ.")
    if metrics.get("population") == "outcome_selected_lean_sensitivity":
        if lean_membership is None or metrics.get("lean_membership_sha256") != sha256_file(lean_membership):
            raise ValueError("Lean reporting requires the frozen lean membership.")
    elif metrics.get("population") != "full_eligible":
        raise ValueError("Test evaluation population is invalid.")
    if explanations.get("population") != metrics["population"] or explanations.get("content_mode") != metrics["content_mode"]:
        raise RuntimeError("Explanation and evaluation populations differ.")
    validation = json.loads(validation_report.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    audit_record = (checkpoint.get("inputs") or {}).get("canonical_audit")
    if not isinstance(audit_record, dict):
        raise RuntimeError("The selected run lacks its canonical data audit.")
    audit = json.loads(Path(audit_record["path"]).read_text(encoding="utf-8"))
    if audit.get("dataset_sha256") != checkpoint.get("dataset_sha256"):
        raise RuntimeError("The data audit does not match the selected run.")
    lines = [
        "# Chronological DAO vote-choice report",
        "",
        "## Question and scope",
        "",
        "Predict FOR, AGAINST, or ABSTAIN immediately before a recorded vote, conditional on voting. "
        "This is not a turnout prediction or a causal explanation.",
        "",
        "## Data and protocol",
        "",
        f"- Eligible proposals with votes: {audit.get('eligible_proposals_with_votes')}",
        f"- Eligible vote events: {(audit.get('vote_counts_and_exclusions') or {}).get('eligible_votes')}",
        f"- Proposal counts: {split['proposal_counts']}",
        f"- Chronological cutoffs (UTC): {split['train_cutoff']} and {split['test_cutoff']}",
        f"- History mode: {config['history_mode']}",
        f"- Numeric inputs: {', '.join(config['numeric_columns']) or 'none'}",
        f"- Categorical cluster inputs: {', '.join(config['categorical_columns']) or 'none'}",
        f"- Voter-clustering scope: {config.get('voter_cluster_scope') or 'disabled'}.",
        f"- Training/validation tokenization: {(validation.get('run_details') or {}).get(selection['selected_model'], {})}.",
        f"- Test tokenization: {(metrics.get('text_truncation') or {}).get(selection['selected_model'], {})}.",
        (
            "- **Unverified local proxy:** cached proposal metadata is assumed historical, and recorded vote time "
            "is assumed to equal observation time. These results are not verified point-in-time predictions."
            if provenance_mode == "local_proxy"
            else "- Historical proposal and observation-time evidence was supplied as explicit local attestations; "
                 "its underlying sources must be reviewed independently."
        ),
        "- Validation selected the checkpoint and comparison set before held-out test evaluation.",
        "",
        "## Held-out results",
        "",
        f"- Test examples: {metrics['n_examples']}; proposals: {metrics['n_proposals']}.",
        f"- Population: {metrics['population']}.",
        f"- Content evaluation: {metrics.get('content_mode', 'unknown')}.",
        f"- Selected model: {selection['selected_model']}.",
        "",
    ]
    for name, result in metrics["models"].items():
        values = result["metrics"]
        interval = result["proposal_bootstrap_macro_f1"]
        lines.extend(
            [
                f"### {name}",
                f"- Macro-F1: {_number(values['macro_f1'])}; accuracy: {_number(values['accuracy'])}.",
                f"- Proposal bootstrap 95% interval: {_number(interval['lower'])} - {_number(interval['upper'])}.",
                f"- ECE: {_number(values.get('ece'))}; log loss: {_number(values.get('log_loss'))}.",
                f"- Confusion matrix (FOR, AGAINST, ABSTAIN): {result['confusion_matrix']}.",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label in ("FOR", "AGAINST", "ABSTAIN"):
            item = values["per_class"][label]
            lines.append(
                f"| {label} | {_number(item['precision'])} | {_number(item['recall'])} | "
                f"{_number(item['f1'])} | {item['support']} |"
            )
        lines.append("")
        for group, values_by_group in result.get("strata", {}).items():
            for label, item in values_by_group.items():
                lines.append(f"- {group}/{label}: n={item['n']}, macro-F1={_number(item['macro_f1'])}.")
        lines.append("")
    lines.extend(
        [
            "## Explanation checks",
            "",
            f"- Explained examples: {explanations['explained_examples']} of {explanations['test_examples']}; "
            f"local cases: {explanations['local_cases']}.",
            f"- Global sample: {explanations['global_sample_examples']} proposals; {explanations['global_sampling']}.",
            f"- Background: {explanations['background_source']}; size: {explanations['background_size']}.",
            f"- Integrated Gradients steps: {explanations['integration_steps']}; "
            f"cases exceeding the completeness tolerance: {explanations['cases_with_completeness_failure']}.",
            "- The explanation artifact includes high/low attribution ablations, background comparisons, "
            "and classification-head randomization checks.",
            "- Attribution summaries describe model behavior, not causes; global summaries are proposal-weighted samples.",
            "",
            "### Sampled numeric attribution",
            "",
        ]
    )
    attributions = explanations.get("numeric_mean_absolute_group_attribution") or {}
    if attributions:
        for feature, value in sorted(attributions.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {feature}: mean absolute attribution {_number(value)}.")
    else:
        lines.append("- No numeric attribution was available.")
    lines.append(
        f"- Mean absolute text-token attribution in the proposal-weighted sample: "
        f"{_number(explanations.get('text_mean_absolute_token_attribution'))}."
    )
    for modality in ("numeric", "text"):
        checks = [case[modality] for case in explanations.get("cases", []) if modality in case]
        cosine = [item["head_randomization_cosine"] for item in checks if item.get("head_randomization_cosine") is not None]
        variation = [
            item["background_mean_absolute_difference"] for item in checks
            if item.get("background_mean_absolute_difference") is not None
        ]
        if checks:
            lines.append(
                f"- {modality} local checks: mean head-randomization cosine "
                f"{_number(sum(cosine) / len(cosine) if cosine else None)}; mean background difference "
                f"{_number(sum(variation) / len(variation) if variation else None)}."
            )
    lines.extend(["", "### Sampled cases", ""])
    labels = ("FOR", "AGAINST", "ABSTAIN")
    for case in explanations.get("cases", []):
        lines.append(
            f"- Example `{case['example_id'][:12]}`: true class {labels[case['true_class']]}, "
            f"predicted class {labels[case['predicted_class']]}, confidence {_number(case['confidence'])}."
        )
    lines.extend(
        [
            "",
            "## Limitations and reproducibility",
            "",
            "- Historical availability and vote-observation evidence remain assumptions in local-proxy mode; "
            "supplied attestations in verified mode still require independent review.",
            "- The full eligible population is primary. Outcome-selected lean data and same-proposal vote history "
            "are separate sensitivity analyses, if run.",
            "- Individual votes on the same proposal are correlated; uncertainty uses proposal-level bootstrap resampling.",
            f"- Dataset SHA-256: `{checkpoint['dataset_sha256']}`.",
            f"- Split SHA-256: `{checkpoint['split_sha256']}`.",
            f"- Frozen selection SHA-256: `{sha256_file(selection_path)}`.",
            f"- Test evaluation manifest SHA-256: `{sha256_file(test_evaluation_dir / 'evaluation_manifest.json')}`.",
            f"- Explanation artifact SHA-256: `{sha256_file(explanation_path)}`.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
