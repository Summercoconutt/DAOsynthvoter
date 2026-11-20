from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from dao_governance.data.eligibility import EligibleProposal
from dao_governance.data.chronological_split import build_chronological_manifest
from dao_governance.data.chronological_split import write_chronological_manifest
from dao_governance.features.point_in_time_examples import build_example_partitions
from dao_governance.features.asof import ObservationProvenance
from dao_governance.features.point_in_time_clusters import fit_point_in_time_clusters
from dao_governance.evaluation.point_in_time import (
    baseline_predictions,
    evaluate_matched_predictions,
    freeze_validation_selection,
    verify_frozen_selection,
)
from dao_governance.modelling.point_in_time_model import (
    FeatureEncoder,
    ProposalTextCache,
    VoteChoiceClassifier,
    VoteDataset,
    collate_vote_examples,
    historical_choice_baseline,
    majority_label,
    prepare_training_datasets,
    validate_target_examples,
)
from dao_governance.modelling.point_in_time_training import fit_vote_classifier, predict_vote_classifier
from dao_governance.modelling.point_in_time_artifacts import load_validation_checkpoint, save_validation_checkpoint
from dao_governance.modelling.run_artifacts import sha256_file
from dao_governance.modelling.point_in_time_explain import integrated_gradients_numeric, integrated_gradients_text


class DummyTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.seen = []

    def __call__(self, title, body, *, truncation, max_length=None):
        self.seen.append((title, body))
        ids = [1] + [2] * len(title) + [3] + [4] * len(body) + [5]
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


OBSERVATION_PROVENANCE = ObservationProvenance(True, "synthetic timestamps")


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(6, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids=None, inputs_embeds=None, attention_mask):
        values = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=values)


def _proposal():
    return EligibleProposal(
        space="dao.eth", proposal_id="p1", title="title", body="body",
        start=pd.Timestamp("2023-01-01", tz="UTC"), end=pd.Timestamp("2023-01-03", tz="UTC"),
        labels_by_position=(0, 1, 2), historical_evidence="synthetic fixture",
        metadata_available_at=pd.Timestamp("2022-12-31T23:59:59Z"),
    )


def test_encoder_fits_only_training_and_uses_explicit_unknown_category():
    train = pd.DataFrame({"voter_prior_count": [1.0, 3.0], "voter_cluster": ["a", "b"]})
    encoder = FeatureEncoder(("voter_prior_count",), ("voter_cluster",)).fit(train)
    valid = pd.DataFrame({"voter_prior_count": [np.nan, 1000.0], "voter_cluster": ["new", "a"]})
    transformed = encoder.transform(valid)
    assert transformed.shape == (2, 4)
    assert transformed[0, 0] == pytest.approx(0.0)
    assert transformed[0, 1:].tolist() == [1, 0, 0]
    with pytest.raises(RuntimeError, match="cannot be refitted"):
        encoder.fit(valid)
    with pytest.raises(ValueError, match="approved point-in-time"):
        FeatureEncoder(("label_id",))
    with pytest.raises(ValueError, match="earlier causal warm-up"):
        FeatureEncoder((), ("voter_cluster",)).fit(pd.DataFrame({"voter_cluster": ["unknown", "unknown"]}))


def test_title_and_body_reach_fused_classifier_without_downloads():
    tokenizer = DummyTokenizer()
    cache = ProposalTextCache([_proposal()], tokenizer, max_length=8)
    assert tokenizer.seen == [("title", "body"), ("title", "body")]
    assert cache.truncated == 1
    events = pd.DataFrame(
        [{"example_id": "e1", "space": "dao.eth", "proposal_id": "p1",
          "vote_ts": "2023-01-02", "label_id": 1}]
    )
    dataset = VoteDataset(events, np.array([[0.5]], dtype=np.float32), cache)
    batch = collate_vote_examples([dataset[0]], pad_token_id=cache.pad_token_id)
    assert 4 in batch["input_ids"][0].tolist()
    model = VoteChoiceClassifier(numeric_dim=1, text_encoder=DummyEncoder(), dropout=0.0)
    logits = model(batch)
    assert logits.shape == (1, 3)
    nn.CrossEntropyLoss()(logits, batch["labels"]).backward()
    assert model.text_encoder.embedding.weight.grad is not None
    assert batch["example_ids"] == ["e1"]


def test_matched_baselines_and_manifest_membership():
    assert majority_label([0, 1, 1]) == 1
    history = pd.DataFrame(
        [
            {"voter_dao_prior_count": 0, "voter_dao_prior_for": np.nan,
             "voter_dao_prior_against": np.nan, "voter_dao_prior_abstain": np.nan},
            {"voter_dao_prior_count": 2, "voter_dao_prior_for": 0.5,
             "voter_dao_prior_against": 0.5, "voter_dao_prior_abstain": 0.0},
        ]
    )
    assert historical_choice_baseline(history, fallback_label=1).tolist() == [1, 1]
    events = pd.DataFrame({"example_id": ["e1"], "space": ["dao.eth"], "proposal_id": ["p1"], "vote_ts": ["2023-01-02"]})
    manifest = {
        "version": 1, "dataset_sha256": "a" * 64, "train_cutoff": "2023-01-04", "test_cutoff": "2023-01-07",
        "proposals": [
            {"space": "dao.eth", "proposal_id": "p1", "split": "train", "start": "2023-01-01", "end": "2023-01-03"},
            {"space": "dao.eth", "proposal_id": "p2", "split": "validation", "start": "2023-01-04", "end": "2023-01-06"},
            {"space": "dao.eth", "proposal_id": "p3", "split": "test", "start": "2023-01-07", "end": "2023-01-09"},
        ],
    }
    for row in manifest["proposals"]:
        row.update({"labels_by_position": [0, 1, 2],
                    "content_sha256": hashlib.sha256(row["proposal_id"].encode()).hexdigest(),
                    "provenance_sha256": "d" * 64, "content_sensitivity_excluded": False,
                    "metadata_available_at": (pd.Timestamp(row["start"], tz="UTC") - pd.Timedelta(seconds=1)).isoformat()})
    validate_target_examples(events, manifest, "train")
    with pytest.raises(ValueError, match="split manifest"):
        validate_target_examples(events, manifest, "test")


def test_training_selects_validation_checkpoint_and_aligns_predictions(tmp_path):
    manifest = {
        "version": 1, "dataset_sha256": "a" * 64, "train_cutoff": "2023-01-04", "test_cutoff": "2023-01-07",
        "proposals": [
            {"space": "dao.eth", "proposal_id": "p1", "split": "train", "start": "2023-01-01", "end": "2023-01-03"},
            {"space": "dao.eth", "proposal_id": "p2", "split": "validation", "start": "2023-01-04", "end": "2023-01-06"},
            {"space": "dao.eth", "proposal_id": "p3", "split": "test", "start": "2023-01-07", "end": "2023-01-09"},
        ],
    }
    for row in manifest["proposals"]:
        row.update({"labels_by_position": [0, 1, 2],
                    "content_sha256": hashlib.sha256(row["proposal_id"].encode()).hexdigest(),
                    "provenance_sha256": "d" * 64, "content_sensitivity_excluded": False,
                    "metadata_available_at": (pd.Timestamp(row["start"], tz="UTC") - pd.Timedelta(seconds=1)).isoformat()})
    train_events = pd.DataFrame(
        [{"example_id": "t1", "space": "dao.eth", "proposal_id": "p1", "vote_ts": "2023-01-02", "label_id": 0},
         {"example_id": "t2", "space": "dao.eth", "proposal_id": "p1", "vote_ts": "2023-01-02", "label_id": 1}]
    )
    valid_events = pd.DataFrame(
        [{"example_id": "v1", "space": "dao.eth", "proposal_id": "p2", "vote_ts": "2023-01-05", "label_id": 2}]
    )
    train = VoteDataset(train_events, np.array([[0.0], [1.0]], dtype=np.float32), None)
    valid = VoteDataset(valid_events, np.array([[0.5]], dtype=np.float32), None)
    model = VoteChoiceClassifier(numeric_dim=1, dropout=0.0)
    result = fit_vote_classifier(
        model, train, valid, manifest=manifest, epochs=2, learning_rate=0.01, batch_size=2, seed=1
    )
    predictions = predict_vote_classifier(model, valid, batch_size=1)
    assert 1 <= result.best_epoch <= 2
    assert predictions["example_id"].tolist() == ["v1"]
    assert predictions["proposal_id"].tolist() == ["p2"]
    assert predictions[["prob_FOR", "prob_AGAINST", "prob_ABSTAIN"]].sum(axis=1).iloc[0] == pytest.approx(1.0)
    split_path = tmp_path / "split.json"
    write_chronological_manifest(manifest, split_path)
    encoder = FeatureEncoder(("voter_prior_count",)).fit(pd.DataFrame({"voter_prior_count": [0.0, 1.0]}))
    run = tmp_path / "run-a"
    save_validation_checkpoint(
        run, run_id="run-a", model=model, feature_encoder=encoder, split_manifest=split_path, training=result,
        observation_provenance=OBSERVATION_PROVENANCE, history_mode="other_proposals_only",
    )
    restored, restored_encoder, tokenizer, _ = load_validation_checkpoint(
        run, expected_run_id="run-a", expected_split=split_path
    )
    assert tokenizer is None and restored_encoder.dimension == 1
    pd.testing.assert_frame_equal(predict_vote_classifier(restored, valid, batch_size=1), predictions)
    split_hash = sha256_file(split_path)
    validation_report = tmp_path / "validation.json"
    report = evaluate_matched_predictions(
        {"numeric": predictions, "majority": predictions},
        manifest=manifest, split="validation", split_fingerprint=split_hash, bootstrap_samples=2,
    )
    report["run_manifests"] = {
        "numeric": {"run_id": "run-a", "manifest_sha256": sha256_file(run / "run_manifest.json")}
    }
    validation_report.write_text(json.dumps(report), encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    freeze_validation_selection(
        selection_path, split_manifest=split_path, validation_report=validation_report,
        checkpoint_root=run, selected_model="numeric", allowed_models=["numeric", "majority"],
    )
    selection = verify_frozen_selection(
        selection_path, split_manifest=split_path, validation_report=validation_report, checkpoint_root=run
    )
    assert selection["status"] == "frozen"
    test_predictions = predictions.copy().assign(
        example_id="test-id", proposal_id="p3", vote_ts="2023-01-08"
    )
    test_report = evaluate_matched_predictions(
        {"numeric": test_predictions, "majority": test_predictions},
        manifest=manifest, split="test", split_fingerprint=split_hash, bootstrap_samples=2,
        selection_path=selection_path, selection_split_manifest=split_path,
        validation_report=validation_report, checkpoint_root=run,
    )
    assert test_report["n_examples"] == 1
    validation_report.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen selection"):
        verify_frozen_selection(
            selection_path, split_manifest=split_path, validation_report=validation_report, checkpoint_root=run
        )
    manifest_path = run / "run_manifest.json"
    saved_manifest = manifest_path.read_text(encoding="utf-8")
    incomplete = json.loads(saved_manifest)
    incomplete["artifacts"] = [row for row in incomplete["artifacts"] if row["relative_path"] != "model/model.pt"]
    manifest_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(RuntimeError, match="required artifact records"):
        load_validation_checkpoint(run, expected_run_id="run-a", expected_split=split_path)
    manifest_path.write_text(saved_manifest, encoding="utf-8")
    (run / "model/model.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="artifact is missing or changed"):
        load_validation_checkpoint(run, expected_run_id="run-a", expected_split=split_path)


def test_example_partitions_update_history_without_using_future_choices():
    proposals = {
        ("dao.eth", name): EligibleProposal(
            "dao.eth", name, name, "body", pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC"), (0, 1, 2), "synthetic fixture",
            pd.Timestamp(start, tz="UTC") - pd.Timedelta(seconds=1)
        )
        for name, start, end in (
            ("p1", "2023-01-01", "2023-01-03"),
            ("p2", "2023-01-04", "2023-01-06"),
            ("p3", "2023-01-07", "2023-01-09"),
        )
    }
    manifest = build_chronological_manifest(
        list(proposals.values()), dataset_fingerprint="a" * 64,
        train_cutoff="2023-01-04", test_cutoff="2023-01-07",
    )
    events = pd.DataFrame(
        [
            {"example_id": name, "voter": "v1", "space": "dao.eth", "proposal_id": proposal,
             "vote_ts": time, "observed_at": time, "label_id": label}
            for name, proposal, time, label in (
                ("e1", "p1", "2023-01-02", 0), ("e2", "p2", "2023-01-05", 1),
                ("e3", "p3", "2023-01-08", 2),
            )
        ]
    )
    parts, _ = build_example_partitions(
        events, proposals, manifest, split_fingerprint="b" * 64, availability_provenance=OBSERVATION_PROVENANCE
    )
    assert parts["train"]["voter_dao_prior_count"].tolist() == [0]
    assert parts["validation"]["voter_dao_prior_count"].tolist() == [1]
    assert parts["test"]["voter_dao_prior_count"].tolist() == [2]
    changed = events.copy()
    changed.loc[2, "label_id"] = 0
    after, _ = build_example_partitions(
        changed, proposals, manifest, split_fingerprint="b" * 64, availability_provenance=OBSERVATION_PROVENANCE
    )
    pd.testing.assert_frame_equal(parts["validation"], after["validation"])
    changed_proposals = dict(proposals)
    changed_proposals[("dao.eth", "p1")] = replace(proposals[("dao.eth", "p1")], labels_by_position=(1, 0, 2))
    with pytest.raises(ValueError, match="metadata differs"):
        build_example_partitions(
            events, changed_proposals, manifest, split_fingerprint="b" * 64, availability_provenance=OBSERVATION_PROVENANCE
        )

    dao_profiles = pd.DataFrame(
        [{"space": space, "asof_ts": "2022-12-28T00:00:00Z", "dao_prior_count": count}
         for space, count in (("dao.eth", 1), ("b.eth", 2), ("c.eth", 9))]
    )
    dao_bundle = fit_point_in_time_clusters(
        dao_profiles, kind="dao", scope=None, feature_columns=["dao_prior_count"],
        history_column="dao_prior_count", fit_cutoff="2022-12-29", train_cutoff="2023-01-04",
        split_fingerprint="b" * 64, n_clusters=2, min_group_size=3,
    )
    voter_profiles = pd.DataFrame(
        [{"voter": f"v{i}", "space": space, "asof_ts": "2022-12-30T00:00:00Z",
          "dao_prior_count": count, "voter_dao_prior_count": i + 1}
         for space, count in (("dao.eth", 1), ("b.eth", 2), ("c.eth", 9))
         for i in range(4)]
    )
    voter_bundle = fit_point_in_time_clusters(
        voter_profiles, kind="voter", scope="dao_cluster", feature_columns=["voter_dao_prior_count"],
        history_column="voter_dao_prior_count", fit_cutoff="2022-12-31", train_cutoff="2023-01-04",
        split_fingerprint="b" * 64, n_clusters=2, min_group_size=3, dao_bundle=dao_bundle,
    )
    clustered, _ = build_example_partitions(
        events, proposals, manifest, split_fingerprint="b" * 64, availability_provenance=OBSERVATION_PROVENANCE,
        dao_bundle=dao_bundle, voter_bundle=voter_bundle,
    )
    assert "dao_cluster" in clustered["validation"]
    assert "voter_cluster" in clustered["validation"]


def test_integrated_gradients_uses_training_baselines_and_reports_completeness():
    torch.manual_seed(2)
    model = VoteChoiceClassifier(numeric_dim=1, text_encoder=DummyEncoder(), dropout=0.0)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "numeric_features": torch.tensor([[0.5]]),
    }
    numeric = integrated_gradients_numeric(
        model, batch, torch.zeros((1, 1)), class_index=1, steps=256, background_split="train"
    )
    text = integrated_gradients_text(
        model, batch, torch.zeros((1, 3, 4)), class_index=1, steps=256, background_split="train"
    )
    assert numeric.attributions.shape == (1, 1)
    assert text.attributions.shape == (1, 3, 4)
    assert abs(numeric.completeness_delta.item()) < 1e-3
    assert abs(text.completeness_delta.item()) < 1e-3
    with pytest.raises(ValueError, match="training data"):
        integrated_gradients_numeric(
            model, batch, torch.zeros((1, 1)), class_index=1, steps=1, background_split="test"
        )


def test_synthetic_matched_models_share_chronological_examples():
    proposals = {
        ("dao.eth", name): EligibleProposal(
            "dao.eth", name, name, "full body", pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC"), (0, 1, 2), "synthetic evidence",
            pd.Timestamp(start, tz="UTC") - pd.Timedelta(seconds=1)
        )
        for name, start, end in (
            ("p1", "2023-01-01", "2023-01-03"),
            ("p2", "2023-01-04", "2023-01-06"),
            ("p3", "2023-01-07", "2023-01-09"),
        )
    }
    manifest = build_chronological_manifest(
        list(proposals.values()), dataset_fingerprint="a" * 64,
        train_cutoff="2023-01-04", test_cutoff="2023-01-07",
    )
    events = pd.DataFrame(
        [
            {"example_id": f"{proposal}-{i}", "voter": f"v{i}", "space": "dao.eth", "proposal_id": proposal,
             "vote_ts": time, "observed_at": time, "label_id": i}
            for proposal, time in (("p1", "2023-01-02"), ("p2", "2023-01-05"), ("p3", "2023-01-08"))
            for i in range(3)
        ]
    )
    partitions, _ = build_example_partitions(
        events, proposals, manifest, split_fingerprint="b" * 64, availability_provenance=OBSERVATION_PROVENANCE
    )
    train, validation = partitions["train"], partitions["validation"]
    predictions = baseline_predictions(train, validation)
    for mode in ("numeric", "text", "fused"):
        use_text = mode in {"text", "fused"}
        use_numeric = mode in {"numeric", "fused"}
        train_data, validation_data, encoder, _ = prepare_training_datasets(
            train, validation, proposals, manifest,
            numeric_columns=["voter_dao_prior_count"] if use_numeric else [],
            tokenizer=DummyTokenizer() if use_text else None, max_length=32,
        )
        model = VoteChoiceClassifier(numeric_dim=encoder.dimension, text_encoder=DummyEncoder() if use_text else None, dropout=0.0)
        fit_vote_classifier(model, train_data, validation_data, manifest=manifest, epochs=1, learning_rate=0.01, batch_size=3)
        predictions[mode] = predict_vote_classifier(model, validation_data, batch_size=3)
    result = evaluate_matched_predictions(
        predictions, manifest=manifest, split="validation", split_fingerprint="b" * 64,
        bootstrap_samples=5, strata=validation
    )
    assert set(result["models"]) == {"majority", "historical_choice", "numeric", "text", "fused"}
    assert result["n_examples"] == 3


def test_tiny_roberta_finetunes_and_restores_offline(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import AutoModel, PreTrainedTokenizerFast, RobertaConfig

    vocabulary = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "title": 4, "body": 5}
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
                      num_attention_heads=2, intermediate_size=16, max_position_embeddings=32,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2)
    )
    proposals = {
        ("dao.eth", name): EligibleProposal(
            "dao.eth", name, "title", "body", pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC"), (0, 1, 2), "synthetic evidence",
            pd.Timestamp(start, tz="UTC") - pd.Timedelta(seconds=1)
        )
        for name, start, end in (
            ("p1", "2023-01-01", "2023-01-03"), ("p2", "2023-01-04", "2023-01-06"),
            ("p3", "2023-01-07", "2023-01-09"),
        )
    }
    manifest = build_chronological_manifest(
        list(proposals.values()), dataset_fingerprint="a" * 64,
        train_cutoff="2023-01-04", test_cutoff="2023-01-07",
    )
    train = pd.DataFrame(
        [{"example_id": f"t{i}", "space": "dao.eth", "proposal_id": "p1",
          "vote_ts": "2023-01-02", "label_id": i, "voter_dao_prior_count": float(i)} for i in range(3)]
    )
    validation = pd.DataFrame(
        [{"example_id": f"v{i}", "space": "dao.eth", "proposal_id": "p2",
          "vote_ts": "2023-01-05", "label_id": i, "voter_dao_prior_count": float(i)} for i in range(3)]
    )
    train_data, valid_data, feature_encoder, cache = prepare_training_datasets(
        train, validation, proposals, manifest, numeric_columns=["voter_dao_prior_count"],
        tokenizer=tokenizer, max_length=16,
    )
    model = VoteChoiceClassifier(numeric_dim=feature_encoder.dimension, text_encoder=encoder, dropout=0.0)
    before = encoder.embeddings.word_embeddings.weight.detach().clone()
    training = fit_vote_classifier(
        model, train_data, valid_data, manifest=manifest, epochs=1, learning_rate=0.001, batch_size=3
    )
    assert not torch.equal(before, encoder.embeddings.word_embeddings.weight.detach())
    expected = predict_vote_classifier(model, valid_data, batch_size=3)
    split = tmp_path / "split.json"
    write_chronological_manifest(manifest, split)
    run = tmp_path / "roberta-run"
    save_validation_checkpoint(
        run, run_id="roberta-run", model=model, feature_encoder=feature_encoder, split_manifest=split,
        training=training, observation_provenance=OBSERVATION_PROVENANCE,
        history_mode="other_proposals_only", tokenizer=tokenizer, text_max_length=16,
    )
    restored, restored_features, restored_tokenizer, _ = load_validation_checkpoint(
        run, expected_run_id="roberta-run", expected_split=split
    )
    assert restored_tokenizer is not None and cache is not None
    restored_cache = ProposalTextCache([proposals[("dao.eth", "p2")]], restored_tokenizer, max_length=16)
    restored_data = VoteDataset(validation, restored_features.transform(validation), restored_cache)
    actual = predict_vote_classifier(restored, restored_data, batch_size=3)
    np.testing.assert_allclose(
        actual[["prob_FOR", "prob_AGAINST", "prob_ABSTAIN"]].to_numpy(),
        expected[["prob_FOR", "prob_AGAINST", "prob_ABSTAIN"]].to_numpy(),
        atol=1e-6,
    )
