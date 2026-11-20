"""Export verified chronological runs without silently including private inputs."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional
from zipfile import ZIP_DEFLATED, ZipFile

from dao_governance.evaluation.explain_chronological import _verify_evaluation
from dao_governance.evaluation.point_in_time import verify_frozen_selection
from dao_governance.modelling.point_in_time_artifacts import verify_validation_checkpoint
from dao_governance.modelling.run_artifacts import resolve_artifact_path, sha256_file, validate_run_id


def export_chronological_run(
    run_root: Path,
    destination: Path,
    *,
    include_sensitive_inputs: bool = False,
    include_test: bool = False,
    selection_path: Optional[Path] = None,
    validation_report: Optional[Path] = None,
    additional_checkpoints: Optional[Mapping[str, Path]] = None,
    test_evaluation_dir: Optional[Path] = None,
    explanation_dir: Optional[Path] = None,
    final_report: Optional[Path] = None,
) -> Path:
    if destination.exists():
        raise FileExistsError("Chronological export destination already exists.")
    if destination.suffix.lower() != ".zip":
        raise ValueError("Chronological export destination must be a ZIP file.")
    if include_test and (not include_sensitive_inputs or selection_path is None or validation_report is None):
        raise ValueError("Including test data requires sensitive inputs and a frozen validation selection.")
    if not include_test and (
        selection_path is not None or validation_report is not None or additional_checkpoints
        or test_evaluation_dir is not None or explanation_dir is not None or final_report is not None
    ):
        raise ValueError("Selection context is only included when explicitly exporting the test partition.")
    if explanation_dir is not None and test_evaluation_dir is None:
        raise ValueError("Explanation export requires the matching test evaluation.")
    if final_report is not None and explanation_dir is None:
        raise ValueError("Final report export requires the matching explanations.")
    split = run_root / "shared/split_manifest.json"
    manifest, _ = verify_validation_checkpoint(run_root, expected_run_id=run_root.name, expected_split=split)
    selection = None
    if include_test:
        if selection_path is None or validation_report is None:
            raise ValueError("Including test data requires a frozen validation selection.")
        selection = verify_frozen_selection(
            selection_path, split_manifest=split, validation_report=validation_report,
            checkpoint_root=run_root, additional_checkpoints=additional_checkpoints,
        )
    run_id = validate_run_id(manifest["run_id"])
    recorded = {record["relative_path"] for record in manifest["artifacts"]} | {"run_manifest.json"}
    actual = {path.relative_to(run_root).as_posix() for path in run_root.rglob("*") if path.is_file()}
    if actual != recorded:
        raise RuntimeError("Run directory contains missing or unrecorded files.")

    input_records = manifest.get("inputs") or {}
    if include_sensitive_inputs and not input_records:
        raise ValueError("The run has no recorded inputs to include.")
    sources: Dict[str, tuple[Path, str]] = {}
    examples_dir = None
    examples_manifest = None
    if include_sensitive_inputs:
        for name, record in input_records.items():
            validate_run_id(name)
            source = Path(record["path"])
            if name in {"examples_manifest", "train_examples", "validation_examples"}:
                relative = f"_export_inputs/examples/{source.name}"
                if name == "examples_manifest":
                    examples_dir = source.parent
                    examples_manifest = json.loads(source.read_text(encoding="utf-8"))
            elif name in {"proposal_metadata", "local_metadata_binding"}:
                relative = f"_export_inputs/proposal_metadata/{source.name}"
            else:
                relative = f"_export_inputs/{name}/{source.name}"
            sources[name] = (source, relative)
        if examples_dir is not None and examples_manifest is not None:
            split_source = examples_dir / "split_manifest.json"
            if sha256_file(split_source) != manifest["split_sha256"]:
                raise RuntimeError("Example split manifest differs from the model run.")
            sources["_examples_split"] = (split_source, "_export_inputs/examples/split_manifest.json")
            if include_test:
                test_record = (examples_manifest.get("partitions") or {}).get("test")
                if not isinstance(test_record, dict) or not isinstance(test_record.get("path"), str):
                    raise ValueError("The example manifest does not record a test partition.")
                test_path = resolve_artifact_path(examples_dir, test_record["path"])
                if sha256_file(test_path) != test_record.get("sha256"):
                    raise RuntimeError("The held-out test partition has changed.")
                test_relative = test_path.relative_to(examples_dir.resolve()).as_posix()
                sources["_test_examples"] = (test_path, f"_export_inputs/examples/{test_relative}")
        elif include_test:
            raise ValueError("The run lacks example-artifact provenance needed to export test data.")

    paths = [relative for _, relative in sources.values()]
    if len(set(paths)) != len(paths):
        raise ValueError("Exported input paths would collide.")
    context_sources: Dict[str, tuple[Path, str]] = {}
    if include_test and selection_path is not None and validation_report is not None:
        context_sources["selection"] = (selection_path, "_export_context/selection.json")
        context_sources["validation_report"] = (validation_report, "_export_context/validation_report.json")
    if test_evaluation_dir is not None:
        if selection is None or selection_path is None:
            raise ValueError("Test evaluation export requires a frozen selection.")
        _verify_evaluation(test_evaluation_dir, selection_path, selection["selected_model"])
        evaluation_manifest = json.loads((test_evaluation_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
        for record in evaluation_manifest["artifacts"]:
            source = resolve_artifact_path(test_evaluation_dir, record["path"])
            relative = source.relative_to(test_evaluation_dir.resolve()).as_posix()
            context_sources[f"evaluation/{relative}"] = (source, f"_export_context/test_evaluation/{relative}")
        context_sources["evaluation/manifest"] = (
            test_evaluation_dir / "evaluation_manifest.json", "_export_context/test_evaluation/evaluation_manifest.json"
        )
    if explanation_dir is not None:
        if test_evaluation_dir is None:
            raise ValueError("Explanation export requires the matching test evaluation.")
        explanation_manifest = json.loads((explanation_dir / "explanation_manifest.json").read_text(encoding="utf-8"))
        explanations = explanation_dir / "explanations.json"
        if (
            selection_path is None
            or explanation_manifest.get("selection_sha256") != sha256_file(selection_path)
            or explanation_manifest.get("explanations_sha256") != sha256_file(explanations)
            or json.loads(explanations.read_text(encoding="utf-8")).get("test_evaluation_manifest_sha256")
            != sha256_file(test_evaluation_dir / "evaluation_manifest.json")
        ):
            raise RuntimeError("Explanation artifacts do not match the frozen test evaluation.")
        context_sources["explanations"] = (explanations, "_export_context/explanations/explanations.json")
        context_sources["explanation_manifest"] = (
            explanation_dir / "explanation_manifest.json", "_export_context/explanations/explanation_manifest.json"
        )
    if final_report is not None:
        if not final_report.is_file():
            raise FileNotFoundError("Final report does not exist.")
        context_sources["final_report"] = (final_report, "_export_context/final_report.md")
    all_paths = paths + [relative for _, relative in context_sources.values()]
    if len(set(all_paths)) != len(all_paths) or set(all_paths).intersection(recorded) or "export_manifest.json" in recorded:
        raise ValueError("Exported input and context paths would collide.")
    export_manifest = {
        "version": 1,
        "run_id": run_id,
        "protocol": "chronological_proposal_disjoint",
        "sensitive_inputs_included": include_sensitive_inputs,
        "test_partition_included": include_test,
        "inputs": {
            name: {"relative_path": relative, "sha256": sha256_file(source), "size_bytes": source.stat().st_size}
            for name, (source, relative) in sources.items()
        },
        "context": {
            name: {"relative_path": relative, "sha256": sha256_file(source)}
            for name, (source, relative) in context_sources.items()
        },
        "note": "The ZIP may contain voter data and local provenance; share it only with authorized recipients.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f"{run_id}.", suffix=".zip", dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for relative in sorted(recorded):
                source = resolve_artifact_path(run_root, relative)
                normalized = source.relative_to(run_root.resolve()).as_posix()
                if normalized != relative:
                    raise ValueError("Run artifact paths must be normalized before export.")
                archive.write(source, f"{run_id}/{normalized}")
            for source, relative in sources.values():
                archive.write(source, f"{run_id}/{relative}")
            for source, relative in context_sources.values():
                archive.write(source, f"{run_id}/{relative}")
            archive.writestr(f"{run_id}/export_manifest.json", json.dumps(export_manifest, indent=2, sort_keys=True))
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def exported_input_overrides(run_root: Path) -> Dict[str, Path]:
    export = json.loads((run_root / "export_manifest.json").read_text(encoding="utf-8"))
    if export.get("version") != 1 or export.get("run_id") != run_root.name or not export.get("sensitive_inputs_included"):
        raise ValueError("Export does not contain recorded input files.")
    run_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    records = export.get("inputs")
    if not isinstance(records, dict):
        raise ValueError("Export input records are invalid.")
    overrides = {}
    for name, record in records.items():
        if not isinstance(record, dict):
            raise ValueError("Export input records are invalid.")
        path = resolve_artifact_path(run_root, record["relative_path"])
        if not path.is_file() or path.stat().st_size != record.get("size_bytes") or sha256_file(path) != record.get("sha256"):
            raise RuntimeError("An exported input is missing or changed.")
        if name in (run_manifest.get("inputs") or {}):
            overrides[name] = path
    if set(overrides) != set(run_manifest.get("inputs") or {}):
        raise ValueError("Export is missing a recorded run input.")
    for record in (export.get("context") or {}).values():
        path = resolve_artifact_path(run_root, record["relative_path"])
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError("Exported selection context is missing or changed.")
    return overrides
