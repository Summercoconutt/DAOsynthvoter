import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from dao_governance.data.local_metadata import bind_local_metadata_cache
from dao_governance.data.canonical_dataset import build_canonical_vote_dataset, create_split_from_canonical, load_proposal_catalog
from dao_governance.features.point_in_time_examples import prepare_example_artifacts
from dao_governance.modelling.point_in_time_runner import train_chronological_run
from dao_governance.evaluation.chronological_runs import compare_validation_runs, evaluate_frozen_test
from dao_governance.evaluation.point_in_time import freeze_validation_selection
from dao_governance.evaluation.explain_chronological import explain_frozen_test
from dao_governance.evaluation.final_report import write_final_report
from dao_governance.clustering.local_profiles import prepare_local_cluster_profiles
from dao_governance.modelling.run_artifacts import sha256_file
from dao_governance.modelling.chronological_export import export_chronological_run, exported_input_overrides
from dao_governance.modelling.point_in_time_artifacts import load_validation_checkpoint


def _inputs(data: Path, *, body="Body"):
    data.mkdir()
    votes = data / "votes.parquet"
    pd.DataFrame(
        {"Space": ["dao", "dao"], "Proposal ID": ["p1", "p2"],
         "Proposal Title": ["Title", "Other"], "Proposal Body": [body, "Other body"]}
    ).to_parquet(votes, index=False)
    cache = data / "cache.jsonl"
    header = {"kind": "snapshot_proposal_metadata", "version": 1, "proposal_count": 1}
    record = {"space": "dao", "proposal_id": "p1", "status": "retrieved", "metadata": {
        "id": "p1", "space": {"id": "dao"}, "title": "Title", "body": "Body",
        "type": "single-choice", "choices": ["For", "Against", "Abstain"],
    }}
    cache.write_text("\n".join(json.dumps(value) for value in (header, record)) + "\n", encoding="utf-8")
    return votes, cache


def test_local_binding_records_matching_and_missing_proposals(tmp_path: Path):
    data = tmp_path / "data"
    votes, cache = _inputs(data)
    output = data / "bound.jsonl"
    audit = bind_local_metadata_cache(votes, cache, output, data_root=data, batch_size=1)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["source_sha256"] == sha256_file(votes)
    assert rows[0]["origin_cache_sha256"] == sha256_file(cache)
    assert rows[0]["historical_availability"] == "unverified"
    assert audit["statuses"] == {"retrieved": 1, "missing": 1}
    assert [row["status"] for row in rows[1:]] == ["retrieved", "missing"]


def test_local_binding_excludes_mismatches_and_rejects_external_paths(tmp_path: Path):
    data = tmp_path / "data"
    votes, cache = _inputs(data, body="Changed")
    output = data / "bound.jsonl"
    audit = bind_local_metadata_cache(votes, cache, output, data_root=data)
    assert audit["statuses"]["local_text_mismatch"] == 1
    with pytest.raises(ValueError, match="under the data directory"):
        bind_local_metadata_cache(votes, cache, tmp_path / "outside.jsonl", data_root=data)


def test_local_binding_cleans_its_temporary_file_on_write_failure(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    votes, cache = _inputs(data)
    output = data / "bound.jsonl"

    def fail_sync(_):
        raise OSError("synthetic write failure")

    monkeypatch.setattr("dao_governance.data.local_metadata.os.fsync", fail_sync)
    with pytest.raises(OSError, match="synthetic write failure"):
        bind_local_metadata_cache(votes, cache, output, data_root=data)
    assert not output.exists()
    assert not list(data.glob(".local-metadata-*.tmp"))


def test_local_proxy_candidates_and_split_are_explicitly_unverified(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    votes = data / "votes.parquet"
    schedule = [
        ("p1", "2023-01-01", "2023-01-03", "2023-01-02"),
        ("p2", "2023-01-04", "2023-01-06", "2023-01-05"),
        ("p3", "2023-01-07", "2023-01-09", "2023-01-08"),
    ]
    pd.DataFrame(
        [{"Space": "dao", "Proposal ID": name, "Proposal Title": name, "Proposal Body": "Body",
          "Voter": f"v{choice}", "Choice": str(choice), "Vote Timestamp": vote}
         for name, _, _, vote in schedule for choice in (1, 2, 3)]
    ).to_parquet(votes, index=False)
    cache = data / "cache.jsonl"
    header = {"kind": "snapshot_proposal_metadata", "version": 1, "proposal_count": 3}
    records = [
        {"space": "dao", "proposal_id": name, "status": "retrieved", "metadata": {
            "id": name, "space": {"id": "dao"}, "title": name, "body": "Body",
            "type": "single-choice", "choices": ["Against", "For", "Abstain"],
            "start": int(pd.Timestamp(start, tz="UTC").timestamp()),
            "end": int(pd.Timestamp(end, tz="UTC").timestamp())}}
        for name, start, end, _ in schedule
    ]
    cache.write_text("\n".join(json.dumps(value) for value in (header, *records)) + "\n", encoding="utf-8")
    bound = data / "bound.jsonl"
    bind_local_metadata_cache(votes, cache, bound, data_root=data)
    events = data / "events.parquet"
    audit = build_canonical_vote_dataset(votes, bound, None, events, local_proxy=True)
    assert audit["provenance_mode"] == "local_proxy"
    assert audit["historical_provenance_verified"] is False
    assert audit["vote_counts_and_exclusions"]["unverified_local_proxy_votes"] == 9
    assert pd.read_parquet(events)["choice_norm"].tolist() == ["against", "for", "abstain"] * 3
    split = data / "split.json"
    manifest = create_split_from_canonical(
        votes, bound, None, events, split, train_cutoff="2023-01-04", test_cutoff="2023-01-07", local_proxy=True
    )
    assert manifest["provenance_mode"] == "local_proxy"
    assert all(row["metadata_available_at"] is None for row in manifest["proposals"])
    profiles = data / "profiles"
    profile_counts = prepare_local_cluster_profiles(
        raw_source=votes, metadata_cache=bound, canonical_events=events, split_manifest=split,
        fit_cutoff="2023-01-03T12:00:00Z", output_dir=profiles,
    )
    assert profile_counts == {"warmup_votes": 3, "dao_profiles": 1, "voter_dao_profiles": 3}
    profile_meta = json.loads((profiles / "voters.parquet.meta.json").read_text(encoding="utf-8"))
    assert profile_meta["availability_provenance"]["verified"] is False
    assert profile_meta["availability_provenance"]["local_proxy"] is True
    examples = data / "examples"
    prepared = prepare_example_artifacts(
        raw_source=votes, metadata_cache=bound, historical_provenance=None, canonical_events=events,
        observations=None, observation_provenance=None, split_manifest=split, output_dir=examples,
        local_proxy=True,
    )
    assert prepared["provenance_mode"] == "local_proxy"
    assert prepared["observation_provenance"]["verified"] is False
    assert prepared["observation_provenance"]["local_proxy"] is True
    assert pd.read_parquet(examples / "test.parquet")["voter_dao_prior_count"].tolist() == [2, 2, 2]
    run = tmp_path / "local-proxy-run"
    train_chronological_run(
        examples_dir=examples, raw_source=votes, metadata_cache=bound, historical_provenance=None,
        canonical_events=events, observations=None, observation_provenance=None,
        run_root=run, run_id="local-proxy-run", model_kind="numeric",
        numeric_columns=["voter_dao_prior_count"], categorical_columns=[], text_loader=None,
        text_max_length=128, freeze_text_encoder=False, epochs=1, learning_rate=0.01,
        batch_size=3, bootstrap_samples=2, local_proxy=True,
    )
    run_manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["provenance_mode"] == "local_proxy"
    model_config = json.loads((run / "model/config.json").read_text(encoding="utf-8"))
    assert model_config["observation_provenance"]["verified"] is False
    comparison = tmp_path / "comparison.json"
    compare_validation_runs({"numeric": run}, comparison, bootstrap_samples=2)
    selection = tmp_path / "selection.json"
    freeze_validation_selection(
        selection, split_manifest=examples / "split_manifest.json", validation_report=comparison,
        checkpoint_root=run, selected_model="numeric",
        allowed_models=["numeric", "majority", "historical_choice"],
    )
    evaluation = tmp_path / "evaluation"
    result = evaluate_frozen_test(
        selection_path=selection, validation_report=comparison, checkpoint_root=run,
        additional_checkpoints={}, output_dir=evaluation, bootstrap_samples=2,
    )
    assert result["provenance_mode"] == "local_proxy"
    explanations = tmp_path / "explanations"
    explain_frozen_test(
        selection_path=selection, validation_report=comparison, checkpoint_root=run, additional_checkpoints={},
        test_evaluation_dir=evaluation, output_dir=explanations, max_cases=1, global_sample_size=1,
        background_size=1, steps=8,
    )
    report = tmp_path / "report.md"
    write_final_report(
        selection_path=selection, validation_report=comparison, checkpoint_root=run, additional_checkpoints={},
        test_evaluation_dir=evaluation, explanation_dir=explanations, output=report,
    )
    assert "**Unverified local proxy:**" in report.read_text(encoding="utf-8")
    archive = tmp_path / "local-export.zip"
    export_chronological_run(
        run, archive, include_sensitive_inputs=True, include_test=True,
        selection_path=selection, validation_report=comparison,
        test_evaluation_dir=evaluation, explanation_dir=explanations, final_report=report,
    )
    extracted = tmp_path / "extracted"
    with ZipFile(archive) as bundle:
        bundle.extractall(extracted)
    restored = extracted / "local-proxy-run"
    overrides = exported_input_overrides(restored)
    _, _, _, restored_manifest = load_validation_checkpoint(
        restored, expected_run_id="local-proxy-run", expected_split=restored / "shared/split_manifest.json",
        input_overrides=overrides,
    )
    _, fingerprint, _ = load_proposal_catalog(
        overrides["raw_votes"], overrides["proposal_metadata"], None, local_proxy=True
    )
    assert fingerprint == restored_manifest["dataset_sha256"]
    assert restored_manifest["provenance_mode"] == "local_proxy"
    assert all(path.is_relative_to(restored) for path in overrides.values())
