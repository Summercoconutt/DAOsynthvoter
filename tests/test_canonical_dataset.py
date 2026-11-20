import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

import pandas as pd
import pytest
import yaml

from dao_governance.data.canonical_dataset import build_canonical_vote_dataset, create_split_from_canonical, load_verified_catalog
from dao_governance.data.lean_sensitivity import build_lean_membership
from dao_governance.data.eligibility import historical_metadata_hash
from dao_governance.evaluation.chronological_runs import compare_validation_runs, evaluate_frozen_test
from dao_governance.evaluation.explain_chronological import explain_frozen_test
from dao_governance.evaluation.final_report import write_final_report
from dao_governance.evaluation.point_in_time import freeze_validation_selection
from dao_governance.features.point_in_time_examples import prepare_example_artifacts
from dao_governance.features.asof import ObservationProvenance, VoteEvent, build_cluster_fit_profiles
from dao_governance.clustering.predictive import fit_predictive_stage, save_fit_profiles
from dao_governance.modelling.point_in_time_runner import _validate_partition_coverage, train_chronological_run
from dao_governance.modelling.chronological_export import export_chronological_run, exported_input_overrides
from dao_governance.modelling.point_in_time_artifacts import load_validation_checkpoint
from dao_governance.modelling.run_artifacts import sha256_file


def _inputs(tmp_path: Path, *, repeated=False):
    source = tmp_path / "raw.parquet"
    rows = [
        {"Space": "dao", "Proposal ID": "p1", "Voter": "v1", "Choice": "1", "Vote Timestamp": "2023-01-02"},
        {"Space": "dao", "Proposal ID": "p1", "Voter": "v2", "Choice": "2", "Vote Timestamp": "2023-01-02"},
        {"Space": "dao", "Proposal ID": "p2", "Voter": "v3", "Choice": "1", "Vote Timestamp": "2023-01-02"},
    ]
    if repeated:
        rows.append(rows[0].copy())
    pd.DataFrame(rows).to_parquet(source, index=False)
    metadata = tmp_path / "metadata.jsonl"
    header = {"kind": "snapshot_proposal_metadata", "version": 1, "source_sha256": sha256_file(source),
              "source_size": source.stat().st_size, "proposal_count": 2}
    proposal = {"type": "single-choice", "choices": ["Against", "For", "Abstain"], "title": "Title",
                "body": "Body", "start": 1672531200, "end": 1672790400}
    records = [
        {"space": "dao", "proposal_id": "p1", "status": "retrieved",
         "metadata": {**proposal, "id": "p1", "space": {"id": "dao"}}},
        {"space": "dao", "proposal_id": "p2", "status": "retrieved",
         "metadata": {**proposal, "id": "p2", "space": {"id": "dao"}}},
    ]
    metadata.write_text("\n".join(json.dumps(item) for item in [header, *records]) + "\n", encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps({"version": 1, "metadata_sha256": sha256_file(metadata), "proposals": [
            {"space": "dao", "proposal_id": "p1", "verified": True, "evidence": "synthetic evidence",
             "available_at": 1672531199, "metadata_sha256": historical_metadata_hash(records[0]["metadata"])}
        ]}),
        encoding="utf-8",
    )
    return source, metadata, provenance


def test_canonical_dataset_uses_verified_options_and_reports_exclusions(tmp_path: Path):
    source, metadata, provenance = _inputs(tmp_path)
    output = tmp_path / "events.parquet"
    audit = build_canonical_vote_dataset(source, metadata, provenance, output, batch_size=1)
    events = pd.read_parquet(output)
    assert events["choice_norm"].tolist() == ["against", "for"]
    assert audit["vote_counts_and_exclusions"]["eligible_votes"] == 2
    assert audit["canonical_events_sha256"] == sha256_file(output)
    assert audit["vote_counts_and_exclusions"]["excluded_unverified_pre_vote_metadata"] == 1
    assert not audit["observation_times_verified"]
    assert not output.with_name(output.name + ".dedupe.sqlite").exists()
    with pytest.raises(FileExistsError):
        build_canonical_vote_dataset(source, metadata, provenance, output)


def test_canonical_dataset_rejects_revisions_across_batches(tmp_path: Path):
    source, metadata, provenance = _inputs(tmp_path, repeated=True)
    output = tmp_path / "events.parquet"
    with pytest.raises(ValueError, match="revision policy"):
        build_canonical_vote_dataset(source, metadata, provenance, output, batch_size=1)
    assert not output.exists()
    assert not output.with_name(output.name + ".dedupe.sqlite").exists()


def _three_proposal_inputs(tmp_path: Path, *, include_warmup=False):
    source = tmp_path / "raw.parquet"
    schedule = [
        ("p1", "2023-01-01", "2023-01-03", "2023-01-02"),
        ("p2", "2023-01-04", "2023-01-06", "2023-01-05"),
        ("p3", "2023-01-07", "2023-01-09", "2023-01-08"),
    ]
    if include_warmup:
        schedule.insert(0, ("p0", "2022-12-27", "2022-12-30", "2022-12-28"))
    rows = [
        {"Space": "dao", "space": "dao", "Proposal ID": name, "proposal_id": f"{name}_1700000000000",
         "Voter": f"v{choice}", "Choice": str(choice), "Vote Timestamp": vote}
        for name, _, _, vote in schedule
        for choice in (1, 2, 3)
    ]
    pd.DataFrame(rows).to_parquet(source, index=False)
    metadata = tmp_path / "metadata.jsonl"
    header = {"kind": "snapshot_proposal_metadata", "version": 1, "source_sha256": sha256_file(source),
              "source_size": source.stat().st_size, "proposal_count": len(schedule)}
    records = []
    for name, start, end, _ in schedule:
        records.append({"space": "dao", "proposal_id": name, "status": "retrieved", "metadata": {
            "id": name, "space": {"id": "dao"},
            "type": "single-choice", "choices": ["For", "Against", "Abstain"], "title": name,
            "body": "Body", "start": int(pd.Timestamp(start, tz="UTC").timestamp()),
            "end": int(pd.Timestamp(end, tz="UTC").timestamp())}})
    metadata.write_text("\n".join(json.dumps(item) for item in [header, *records]) + "\n", encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "version": 1, "metadata_sha256": sha256_file(metadata),
        "proposals": [
            {"space": "dao", "proposal_id": row["proposal_id"], "verified": True, "evidence": "synthetic",
             "available_at": row["metadata"]["start"] - 1, "metadata_sha256": historical_metadata_hash(row["metadata"])}
            for row in records
        ],
    }), encoding="utf-8")
    return source, metadata, provenance


def _tiny_text_loader():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import AutoModel, PreTrainedTokenizerFast, RobertaConfig

    vocabulary = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "Body": 4}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> $B </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>"
    )
    encoder = AutoModel.from_config(
        RobertaConfig(vocab_size=len(vocabulary), hidden_size=8, num_hidden_layers=1,
                      num_attention_heads=2, intermediate_size=16, max_position_embeddings=32, pad_token_id=0)
    )
    return tokenizer, encoder


def test_split_is_bound_to_canonical_data_and_verified_metadata(tmp_path: Path):
    source, metadata, provenance = _three_proposal_inputs(tmp_path)
    events = tmp_path / "events.parquet"
    audit = build_canonical_vote_dataset(source, metadata, provenance, events)
    split = tmp_path / "split.json"
    result = create_split_from_canonical(
        source, metadata, provenance, events, split, train_cutoff="2023-01-04", test_cutoff="2023-01-07"
    )
    assert result["dataset_sha256"] == audit["dataset_sha256"]
    assert result["proposal_counts"] == {"train": 1, "validation": 1, "test": 1, "purged": 0}
    observations = tmp_path / "observations.parquet"
    pd.read_parquet(events)[["example_id", "vote_ts"]].rename(columns={"vote_ts": "observed_at"}).to_parquet(observations)
    observation_provenance = tmp_path / "observations.json"
    observation_provenance.write_text(json.dumps({
        "version": 1, "verified": False, "evidence": "synthetic observation data",
        "canonical_events_sha256": sha256_file(events), "observations_sha256": sha256_file(observations),
    }), encoding="utf-8")
    examples = tmp_path / "examples"
    with pytest.raises(ValueError, match="Verified observation-time evidence"):
        prepare_example_artifacts(
            raw_source=source, metadata_cache=metadata, historical_provenance=provenance,
            canonical_events=events, observations=observations, observation_provenance=observation_provenance,
            split_manifest=split, output_dir=examples,
        )
    info = json.loads(observation_provenance.read_text(encoding="utf-8"))
    info["verified"] = True
    observation_provenance.write_text(json.dumps(info), encoding="utf-8")
    prepared = prepare_example_artifacts(
        raw_source=source, metadata_cache=metadata, historical_provenance=provenance,
        canonical_events=events, observations=observations, observation_provenance=observation_provenance,
        split_manifest=split, output_dir=examples,
    )
    assert {name: info["rows"] for name, info in prepared["partitions"].items()} == {
        "train": 3, "validation": 3, "test": 3
    }
    assert pd.read_parquet(examples / "test.parquet")["voter_dao_prior_count"].tolist() == [2, 2, 2]
    with pytest.raises(ValueError, match="exactly cover"):
        _validate_partition_coverage(events, result, {"test": pd.read_parquet(examples / "test.parquet").iloc[:-1]})
    incomplete = tmp_path / "incomplete-examples"
    shutil.copytree(examples, incomplete)
    train_path = incomplete / "train.parquet"
    pd.read_parquet(train_path).iloc[:-1].to_parquet(train_path, index=False)
    incomplete_manifest = incomplete / "examples_manifest.json"
    incomplete_info = json.loads(incomplete_manifest.read_text(encoding="utf-8"))
    incomplete_info["partitions"]["train"].update({"rows": 2, "sha256": sha256_file(train_path)})
    incomplete_manifest.write_text(json.dumps(incomplete_info), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover"):
        train_chronological_run(
            examples_dir=incomplete, raw_source=source, metadata_cache=metadata, historical_provenance=provenance,
            canonical_events=events, observations=observations, observation_provenance=observation_provenance,
            run_root=tmp_path / "incomplete-run", run_id="incomplete-run", model_kind="numeric",
            numeric_columns=["voter_dao_prior_count"], categorical_columns=[], text_loader=None,
            text_max_length=128, freeze_text_encoder=False, epochs=1, learning_rate=0.01,
            batch_size=3, bootstrap_samples=2,
        )
    test_bytes = (examples / "test.parquet").read_bytes()
    (examples / "test.parquet").write_bytes(b"untouched test partition")
    run = tmp_path / "numeric-run"
    trained = train_chronological_run(
        examples_dir=examples, raw_source=source, metadata_cache=metadata, historical_provenance=provenance,
        canonical_events=events, observations=observations, observation_provenance=observation_provenance,
        run_root=run, run_id="numeric-run", model_kind="numeric",
        numeric_columns=["voter_dao_prior_count"], categorical_columns=[], text_loader=None,
        text_max_length=128, freeze_text_encoder=False, epochs=1, learning_rate=0.01,
        batch_size=1, bootstrap_samples=2,
    )
    assert Path(trained["validation_report"]).is_file()
    assert (run / "validation/model.csv").is_file()
    (examples / "test.parquet").write_bytes(test_bytes)
    fused = tmp_path / "fused-run"
    fused_result = train_chronological_run(
        examples_dir=examples, raw_source=source, metadata_cache=metadata, historical_provenance=provenance,
        canonical_events=events, observations=observations, observation_provenance=observation_provenance,
        run_root=fused, run_id="fused-run", model_kind="fused",
        numeric_columns=["voter_dao_prior_count"], categorical_columns=[], text_loader=_tiny_text_loader,
        text_max_length=16, freeze_text_encoder=False, epochs=1, learning_rate=0.001,
        batch_size=1, bootstrap_samples=2,
    )
    assert Path(fused_result["validation_report"]).is_file()
    assert (fused / "model/tokenizer").is_dir()
    comparison = tmp_path / "validation-comparison.json"
    compared = compare_validation_runs({"numeric": run, "fused": fused}, comparison, bootstrap_samples=2)
    assert set(compared["models"]) == {"numeric", "fused", "majority", "historical_choice"}
    sensitivity_examples = tmp_path / "sensitivity-examples"
    prepare_example_artifacts(
        raw_source=source, metadata_cache=metadata, historical_provenance=provenance,
        canonical_events=events, observations=observations, observation_provenance=observation_provenance,
        split_manifest=split, output_dir=sensitivity_examples, include_target_proposal=True,
    )
    sensitivity_run = tmp_path / "sensitivity-run"
    train_chronological_run(
        examples_dir=sensitivity_examples, raw_source=source, metadata_cache=metadata,
        historical_provenance=provenance, canonical_events=events, observations=observations,
        observation_provenance=observation_provenance, run_root=sensitivity_run, run_id="sensitivity-run",
        model_kind="numeric", numeric_columns=["voter_dao_prior_count"], categorical_columns=[],
        text_loader=None, text_max_length=128, freeze_text_encoder=False, epochs=1,
        learning_rate=0.01, batch_size=1, bootstrap_samples=2,
    )
    with pytest.raises(ValueError, match="explicit sensitivity"):
        compare_validation_runs({"primary": run, "sensitivity": sensitivity_run}, tmp_path / "mixed.json", bootstrap_samples=2)
    mixed = compare_validation_runs(
        {"primary": run, "sensitivity": sensitivity_run}, tmp_path / "mixed.json",
        bootstrap_samples=2, allow_history_sensitivity=True,
    )
    assert mixed["comparison_type"] == "history_sensitivity"
    lean_audit = tmp_path / "lean-audit.csv"
    pd.DataFrame(
        {"space": ["dao", "dao", "dao"], "proposal_id": [f"{name}_1700000000000" for name in ("p1", "p2", "p3")],
         "retained": [False, False, True]}
    ).to_csv(lean_audit, index=False)
    lean_membership = tmp_path / "lean-membership.parquet"
    build_lean_membership(source, lean_audit, lean_membership)
    selection = tmp_path / "selection.json"
    freeze_validation_selection(
        selection, split_manifest=examples / "split_manifest.json", validation_report=comparison,
        checkpoint_root=fused, selected_model="fused",
        allowed_models=["fused", "numeric", "majority", "historical_choice"],
        additional_checkpoints={"numeric": run},
        lean_membership=lean_membership,
    )
    test_output = tmp_path / "test-evaluation"
    test_report = evaluate_frozen_test(
        selection_path=selection, validation_report=comparison, checkpoint_root=fused,
        additional_checkpoints={"numeric": run}, output_dir=test_output, bootstrap_samples=2,
    )
    assert test_report["n_examples"] == 3
    assert (test_output / "evaluation_manifest.json").is_file()
    assert "voter" not in pd.read_csv(test_output / "fused_predictions.csv").columns
    lean_output = tmp_path / "lean-evaluation"
    lean_report = evaluate_frozen_test(
        selection_path=selection, validation_report=comparison, checkpoint_root=fused,
        additional_checkpoints={"numeric": run}, output_dir=lean_output, bootstrap_samples=2,
        lean_membership=lean_membership,
    )
    assert lean_report["population"] == "outcome_selected_lean_sensitivity"
    explanation_dir = tmp_path / "explanations"
    explanations = explain_frozen_test(
        selection_path=selection, validation_report=comparison, checkpoint_root=fused,
        additional_checkpoints={"numeric": run}, test_evaluation_dir=test_output,
        output_dir=explanation_dir, max_cases=3, background_size=2, steps=16,
    )
    assert explanations["explained_examples"] == 3
    assert explanations["background_source"] == "training examples only"
    assert all("numeric" in case and "text" in case and "voter" not in case for case in explanations["cases"])
    final_report = tmp_path / "final-report.md"
    write_final_report(
        selection_path=selection, validation_report=comparison, checkpoint_root=fused,
        additional_checkpoints={"numeric": run}, test_evaluation_dir=test_output,
        explanation_dir=explanation_dir, output=final_report,
    )
    report_text = final_report.read_text(encoding="utf-8")
    assert "## Held-out results" in report_text and "## Explanation checks" in report_text
    assert "v1" not in report_text
    archive = tmp_path / "fused-export.zip"
    export_chronological_run(
        fused, archive, include_sensitive_inputs=True, include_test=True,
        selection_path=selection, validation_report=comparison, additional_checkpoints={"numeric": run},
        test_evaluation_dir=test_output, explanation_dir=explanation_dir, final_report=final_report,
    )
    extracted = tmp_path / "extracted"
    with ZipFile(archive) as bundle:
        bundle.extractall(extracted)
    restored_run = extracted / "fused-run"
    overrides = exported_input_overrides(restored_run)
    _, _, _, restored_manifest = load_validation_checkpoint(
        restored_run, expected_run_id="fused-run", expected_split=restored_run / "shared/split_manifest.json",
        input_overrides=overrides,
    )
    assert Path(restored_manifest["inputs"]["raw_votes"]["path"]).is_relative_to(restored_run)
    exported_test = restored_run / "_export_inputs/examples/test.parquet"
    assert exported_test.is_file()
    assert (restored_run / "_export_context/final_report.md").is_file()
    exported_test.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="exported input"):
        exported_input_overrides(restored_run)
    with pytest.raises(FileExistsError):
        prepare_example_artifacts(
            raw_source=source, metadata_cache=metadata, historical_provenance=provenance,
            canonical_events=events, observations=observations, observation_provenance=observation_provenance,
            split_manifest=split, output_dir=examples,
        )
    events.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="do not match"):
        create_split_from_canonical(source, metadata, provenance, events, tmp_path / "another.json")


def test_chronological_cli_workflow_runs_offline_on_synthetic_data(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    source, metadata, provenance = _three_proposal_inputs(tmp_path)
    events = tmp_path / "events.parquet"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"paths": {
        "master_votes_parquet": str(source), "proposal_metadata_jsonl": str(metadata),
        "canonical_vote_events_parquet": str(events),
    }}), encoding="utf-8")
    extra = tmp_path / "extra.yaml"
    extra.write_text("{}\n", encoding="utf-8")
    common = ["--config", str(config), "--extra-config", str(extra)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}

    def run(script, *args):
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / script), *map(str, args)],
            cwd=root, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"{script} failed: {result.stderr}"

    run("03_run_global_cleaning.py", *common, "--point-in-time", "--historical-provenance", provenance)
    split = tmp_path / "split.json"
    run("create_chronological_split.py", *common, "--historical-provenance", provenance,
        "--output", split, "--train-cutoff", "2023-01-04", "--test-cutoff", "2023-01-07")
    observations = tmp_path / "observations.parquet"
    pd.read_parquet(events)[["example_id", "vote_ts"]].rename(columns={"vote_ts": "observed_at"}).to_parquet(observations)
    observation_provenance = tmp_path / "observations.json"
    observation_provenance.write_text(json.dumps({
        "version": 1, "verified": True, "evidence": "synthetic observation data",
        "canonical_events_sha256": sha256_file(events), "observations_sha256": sha256_file(observations),
    }), encoding="utf-8")
    examples = tmp_path / "examples"
    run("create_point_in_time_examples.py", *common, "--historical-provenance", provenance,
        "--observations", observations, "--observation-provenance", observation_provenance,
        "--split-manifest", split, "--output-dir", examples)
    local_model = tmp_path / "local-roberta"
    tokenizer, encoder = _tiny_text_loader()
    tokenizer.save_pretrained(local_model)
    encoder.save_pretrained(local_model)
    runs = tmp_path / "runs"
    for name in ("numeric", "fused"):
        command = [
            *common, "--historical-provenance", provenance, "--observations", observations,
            "--observation-provenance", observation_provenance, "--examples-dir", examples,
            "--output-root", runs, "--run-id", name, "--model-kind", name,
            "--numeric-feature", "voter_dao_prior_count", "--epochs", "1",
            "--learning-rate", "0.001", "--batch-size", "3", "--bootstrap-samples", "2",
            "--confirm-full-training",
        ]
        if name == "fused":
            command.extend(["--pretrained", local_model, "--max-length", "16"])
        run("08_run_chronological_model.py", *command)
    comparison = tmp_path / "comparison.json"
    run("09_run_chronological_evaluation.py", "compare-validation",
        "--run", f"numeric={runs / 'numeric'}", "--run", f"fused={runs / 'fused'}",
        "--output", comparison, "--bootstrap-samples", "2")
    selection = tmp_path / "selection.json"
    common_selection = [
        "--selection", selection, "--validation-report", comparison, "--selected-run", runs / "fused",
        "--additional-run", f"numeric={runs / 'numeric'}",
    ]
    run("09_run_chronological_evaluation.py", "freeze", "--validation-report", comparison,
        "--selected-model", "fused", "--selected-run", runs / "fused",
        "--additional-run", f"numeric={runs / 'numeric'}", "--output", selection)
    evaluation = tmp_path / "evaluation"
    run("09_run_chronological_evaluation.py", "test", *common_selection,
        "--output-dir", evaluation, "--bootstrap-samples", "2")
    explanation = tmp_path / "explanation"
    run("11_explain_chronological_model.py", *common_selection, "--test-evaluation-dir", evaluation,
        "--output-dir", explanation, "--max-cases", "1", "--background-size", "1", "--steps", "8")
    report = tmp_path / "report.md"
    run("12_write_chronological_report.py", *common_selection, "--test-evaluation-dir", evaluation,
        "--explanation-dir", explanation, "--output", report)
    assert "## Held-out results" in report.read_text(encoding="utf-8")
    archive = tmp_path / "export.zip"
    run("export_chronological_run.py", runs / "fused", archive, "--include-sensitive-inputs", "--include-test",
        "--selection", selection, "--validation-report", comparison, "--additional-run", f"numeric={runs / 'numeric'}",
        "--test-evaluation-dir", evaluation, "--explanation-dir", explanation, "--final-report", report)
    assert archive.is_file()


def test_global_voter_clusters_flow_into_matched_chronological_training(tmp_path: Path):
    source, metadata, provenance = _three_proposal_inputs(tmp_path, include_warmup=True)
    events_path = tmp_path / "events.parquet"
    build_canonical_vote_dataset(source, metadata, provenance, events_path)
    split = tmp_path / "split.json"
    create_split_from_canonical(
        source, metadata, provenance, events_path, split,
        train_cutoff="2023-01-04", test_cutoff="2023-01-07",
    )
    events = pd.read_parquet(events_path)
    observations = tmp_path / "observations.parquet"
    events[["example_id", "vote_ts"]].rename(columns={"vote_ts": "observed_at"}).to_parquet(observations)
    observation_provenance = tmp_path / "observations.json"
    observation_provenance.write_text(json.dumps({
        "version": 1, "verified": True, "evidence": "synthetic observation data",
        "canonical_events_sha256": sha256_file(events_path), "observations_sha256": sha256_file(observations),
    }), encoding="utf-8")
    catalog, _, _ = load_verified_catalog(source, metadata, provenance)
    warmup = events[events["proposal_id"] == "p0"]
    history = [
        VoteEvent(row.example_id, row.voter, row.space, row.proposal_id, row.vote_ts, row.vote_ts,
                  catalog.eligible[(row.space, row.proposal_id)].end, int(row.label_id))
        for row in warmup.itertuples(index=False)
    ]
    evidence = ObservationProvenance(True, "synthetic observation data")
    _, voter_profiles = build_cluster_fit_profiles(
        history, cutoff=pd.Timestamp("2022-12-31", tz="UTC"), availability_provenance=evidence
    )
    profiles_path = tmp_path / "voter-profiles.parquet"
    save_fit_profiles(
        voter_profiles, profiles_path, kind="voter", split_manifest=split,
        fit_cutoff="2022-12-31", availability_provenance=evidence,
    )
    bundle = tmp_path / "global-voters.pkl"
    fit_predictive_stage(
        profiles_path=profiles_path, split_manifest=split, output=bundle, kind="voter", scope="global",
        feature_columns=["voter_dao_prior_for"], history_column="voter_dao_prior_count",
        n_clusters=2, min_group_size=3, min_history=1, algorithm="kmeans",
    )
    examples = tmp_path / "clustered-examples"
    prepared = prepare_example_artifacts(
        raw_source=source, metadata_cache=metadata, historical_provenance=provenance,
        canonical_events=events_path, observations=observations, observation_provenance=observation_provenance,
        split_manifest=split, output_dir=examples, voter_bundle_path=bundle,
    )
    assert prepared["voter_cluster_scope"] == "global"
    train_examples = pd.read_parquet(examples / "train.parquet")
    assert (train_examples["voter_cluster"] != "unknown").any()
    run_root = tmp_path / "clustered-run"
    result = train_chronological_run(
        examples_dir=examples, raw_source=source, metadata_cache=metadata, historical_provenance=provenance,
        canonical_events=events_path, observations=observations, observation_provenance=observation_provenance,
        run_root=run_root, run_id="clustered-run", model_kind="numeric",
        numeric_columns=["voter_dao_prior_count"], categorical_columns=["voter_cluster"], text_loader=None,
        text_max_length=128, freeze_text_encoder=False, epochs=1, learning_rate=0.01,
        batch_size=3, bootstrap_samples=2, voter_bundle_path=bundle,
    )
    assert Path(result["validation_report"]).is_file()
    assert (run_root / "model/clusters/voter.pkl").is_file()
