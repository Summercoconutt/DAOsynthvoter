from __future__ import annotations

import pandas as pd
import pytest

from dao_governance.data.chronological_split import build_chronological_manifest
from dao_governance.data.eligibility import (
    EligibleProposal,
    assess_proposal,
    build_proposal_catalog,
    canonicalize_vote_partition,
    historical_metadata_hash,
)
from dao_governance.features.asof import ObservationProvenance, VoteEvent, build_cluster_fit_profiles, iter_asof_features

FINGERPRINT = "a" * 64
OBSERVATION_PROVENANCE = ObservationProvenance(True, "synthetic observation timestamps")


def _record(proposal_id="p1", choices=None):
    record = {
        "space": "dao.eth",
        "proposal_id": proposal_id,
        "status": "retrieved",
        "metadata": {
            "id": proposal_id,
            "space": {"id": "dao.eth"},
            "type": "single-choice",
            "choices": choices or ["Against", "For", "Abstain"],
            "title": "Title",
            "body": "",
            "start": 1672531200,
            "end": 1672790400,
        },
        "historical_provenance": {"verified": True, "evidence": "synthetic fixture", "available_at": 1672531199},
    }
    record["historical_provenance"]["metadata_sha256"] = historical_metadata_hash(record["metadata"])
    return record


def test_reversed_options_map_vote_positions_and_keep_empty_body():
    catalog = build_proposal_catalog([_record()])
    votes = pd.DataFrame(
        [
            {"voter": "v1", "space": "dao.eth", "proposal_id": "p1", "choice": "1", "vote_timestamp": "2023-01-02"},
            {"voter": "v2", "space": "dao.eth", "proposal_id": "p1", "choice": 2, "vote_timestamp": "2023-01-02"},
            {"voter": "v3", "space": "dao.eth", "proposal_id": "p1", "choice": 3.0, "vote_timestamp": "2023-01-02"},
        ]
    )
    out, counts = canonicalize_vote_partition(votes, catalog, source_fingerprint=FINGERPRINT)
    assert out["choice_norm"].tolist() == ["against", "for", "abstain"]
    assert counts["eligible_votes"] == 3
    assert catalog.eligible[("dao.eth", "p1")].body == ""
    changed = votes.copy()
    changed.loc[0, "choice"] = "2"
    changed_out, _ = canonicalize_vote_partition(changed, catalog, source_fingerprint=FINGERPRINT)
    assert changed_out.loc[0, "example_id"] == out.loc[0, "example_id"]


def test_missing_ambiguous_and_unverified_metadata_are_excluded():
    ambiguous = _record(choices=["For", "For", "Abstain"])
    assert "ambiguous_options" in assess_proposal(ambiguous)[1]
    unavailable = _record()
    unavailable["metadata"]["body"] = None
    unavailable.pop("historical_provenance")
    _, reasons = assess_proposal(unavailable)
    assert "text_unavailable" in reasons
    assert "unverified_pre_vote_metadata" in reasons
    assert assess_proposal({"space": "dao.eth", "proposal_id": "p1", "status": "missing"})[1] == (
        "missing_or_unmatched_metadata",
    )
    mismatched = _record()
    mismatched["metadata"]["id"] = "different"
    assert assess_proposal(mismatched)[1] == ("missing_or_unmatched_metadata",)
    changed = _record()
    changed["historical_provenance"]["metadata_sha256"] = "0" * 64
    assert "unverified_pre_vote_metadata" in assess_proposal(changed)[1]
    late = _record()
    late["historical_provenance"]["available_at"] = late["metadata"]["start"] + 1
    assert "metadata_not_available_at_open" in assess_proposal(late)[1]


def test_vote_at_metadata_availability_time_is_excluded():
    record = _record()
    record["historical_provenance"]["available_at"] = record["metadata"]["start"]
    catalog = build_proposal_catalog([record])
    votes = pd.DataFrame(
        [{"voter": "v1", "space": "dao.eth", "proposal_id": "p1",
          "choice": 1, "vote_timestamp": "2023-01-01"}]
    )
    out, counts = canonicalize_vote_partition(votes, catalog, source_fingerprint=FINGERPRINT)
    assert out.empty
    assert counts["metadata_not_observable_before_vote"] == 1


def test_weighted_multi_choice_and_revisions_are_not_silently_relabelled():
    catalog = build_proposal_catalog([_record()])
    votes = pd.DataFrame(
        [
            {"voter": "v1", "space": "dao.eth", "proposal_id": "p1", "choice": '{"1": 60, "2": 40}', "vote_timestamp": "2023-01-02"},
            {"voter": "v2", "space": "dao.eth", "proposal_id": "p1", "choice": "[1, 2]", "vote_timestamp": "2023-01-02"},
        ]
    )
    out, counts = canonicalize_vote_partition(votes, catalog, source_fingerprint=FINGERPRINT)
    assert out.empty
    assert counts == {"weighted_choice": 1, "multi_choice": 1}
    revision = pd.concat([votes.iloc[[0]], votes.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="revision policy"):
        canonicalize_vote_partition(revision, catalog, source_fingerprint=FINGERPRINT)


def _proposal(name: str, start: str, end: str, title: str) -> EligibleProposal:
    return EligibleProposal(
        space="dao.eth",
        proposal_id=name,
        title=title,
        body="body",
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        labels_by_position=(0, 1, 2),
        historical_evidence="synthetic fixture",
        metadata_available_at=pd.Timestamp(start, tz="UTC") - pd.Timedelta(seconds=1),
    )


def test_split_purges_crossing_proposals_and_marks_cross_split_text_duplicates():
    proposals = [
        _proposal("train", "2023-01-01", "2023-01-02", "Same text"),
        _proposal("cross", "2023-01-02", "2023-01-04", "Cross"),
        _proposal("val", "2023-01-04", "2023-01-05", "  SAME   text "),
        _proposal("test", "2023-01-07", "2023-01-08", "Test"),
    ]
    manifest = build_chronological_manifest(
        proposals, dataset_fingerprint=FINGERPRINT, train_cutoff="2023-01-03", test_cutoff="2023-01-06"
    )
    rows = {row["proposal_id"]: row for row in manifest["proposals"]}
    assert {name: row["split"] for name, row in rows.items()} == {
        "train": "train", "cross": "purged", "val": "validation", "test": "test"
    }
    assert rows["val"]["content_sensitivity_excluded"]
    assert not rows["train"]["content_sensitivity_excluded"]
    assert manifest["proposal_counts"] == {"train": 1, "validation": 1, "test": 1, "purged": 1}


def test_split_rejects_duplicate_proposals_and_empty_periods():
    proposals = [
        _proposal("p1", "2023-01-01", "2023-01-02", "A"),
        _proposal("p2", "2023-01-03", "2023-01-04", "B"),
        _proposal("p3", "2023-01-05", "2023-01-06", "C"),
    ]
    with pytest.raises(ValueError, match="unique"):
        build_chronological_manifest(proposals + proposals[:1], dataset_fingerprint=FINGERPRINT)
    with pytest.raises(ValueError, match="empty target split"):
        build_chronological_manifest(
            proposals, dataset_fingerprint=FINGERPRINT, train_cutoff="2023-01-01", test_cutoff="2023-01-04"
        )


def _event(name, voter, proposal, time, label, observed=None):
    vote_time = pd.Timestamp(time, tz="UTC")
    return VoteEvent(
        example_id=name,
        voter=voter,
        space="dao.eth",
        proposal_id=proposal,
        vote_ts=vote_time,
        observed_at=pd.Timestamp(observed, tz="UTC") if observed else vote_time,
        proposal_end=pd.Timestamp("2023-02-01", tz="UTC"),
        label_id=label,
    )


def _asof(events, *, include_target_proposal=False):
    targets = sorted(events, key=lambda event: event.vote_ts)
    observed = sorted(events, key=lambda event: event.observed_at)
    return list(
        iter_asof_features(
            targets,
            observed,
            availability_provenance=OBSERVATION_PROVENANCE,
            include_target_proposal=include_target_proposal,
        )
    )


def test_asof_features_exclude_ties_and_retain_cold_starts():
    events = [
        _event("a", "v1", "p1", "2023-01-01", 0),
        _event("b", "v2", "p2", "2023-01-01", 1),
        _event("c", "v1", "p3", "2023-01-02", 2),
    ]
    features = _asof(events)
    assert features[0]["dao_prior_count"] == features[1]["dao_prior_count"] == 0
    assert pd.isna(features[0]["voter_prior_for"])
    assert features[0]["voter_has_history"] == 0
    assert features[2]["dao_prior_count"] == 2
    assert features[2]["voter_dao_prior_count"] == 1
    assert features[2]["voter_dao_prior_for"] == 1


def test_target_proposal_history_is_separate_and_requires_prior_visibility():
    events = [
        _event("a", "v1", "p1", "2023-01-01", 0, observed="2023-01-02"),
        _event("b", "v2", "p1", "2023-01-02", 1),
        _event("c", "v3", "p1", "2023-01-03", 2),
    ]
    primary = _asof(events)
    sensitivity = _asof(events, include_target_proposal=True)
    assert [row["dao_prior_count"] for row in primary] == [0, 0, 0]
    assert [row["dao_prior_count"] for row in sensitivity] == [0, 0, 2]


def test_future_or_current_choices_do_not_change_earlier_asof_features():
    events = [
        _event("a", "v1", "p1", "2023-01-01", 0),
        _event("b", "v1", "p2", "2023-01-02", 1),
        _event("c", "v1", "p3", "2023-01-03", 2),
    ]
    changed = [events[0], _event("b", "v1", "p2", "2023-01-02", 2), _event("c", "v1", "p3", "2023-01-03", 0)]
    pd.testing.assert_frame_equal(pd.DataFrame(_asof(events)[:2]), pd.DataFrame(_asof(changed)[:2]))
    assert _asof(changed)[2]["voter_prior_abstain"] == 0.5


def test_asof_rejects_unverified_or_unsorted_observations():
    events = [
        _event("a", "v1", "p1", "2023-01-01", 0),
        _event("b", "v2", "p2", "2023-01-02", 1),
    ]
    with pytest.raises(ValueError, match="provenance"):
        list(iter_asof_features(events, events, availability_provenance=ObservationProvenance(False, "unverified")))
    with pytest.raises(ValueError, match="sorted by observation"):
        list(iter_asof_features(events, reversed(events), availability_provenance=OBSERVATION_PROVENANCE))


def test_cluster_fit_profiles_exclude_open_proposals_and_cutoff_ties():
    closed = VoteEvent(
        example_id="closed", voter="v1", space="dao.eth", proposal_id="p1",
        vote_ts=pd.Timestamp("2023-01-01", tz="UTC"), observed_at=pd.Timestamp("2023-01-01", tz="UTC"),
        proposal_end=pd.Timestamp("2023-01-02", tz="UTC"), label_id=0,
    )
    still_open = VoteEvent(
        example_id="open", voter="v1", space="dao.eth", proposal_id="p2",
        vote_ts=pd.Timestamp("2023-01-01", tz="UTC"), observed_at=pd.Timestamp("2023-01-01", tz="UTC"),
        proposal_end=pd.Timestamp("2023-01-04", tz="UTC"), label_id=1,
    )
    at_cutoff = VoteEvent(
        example_id="late", voter="v2", space="dao.eth", proposal_id="p1",
        vote_ts=pd.Timestamp("2023-01-01", tz="UTC"), observed_at=pd.Timestamp("2023-01-03", tz="UTC"),
        proposal_end=pd.Timestamp("2023-01-02", tz="UTC"), label_id=2,
    )
    dao, voter = build_cluster_fit_profiles(
        [closed, still_open, at_cutoff], cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        availability_provenance=OBSERVATION_PROVENANCE,
    )
    assert dao["dao_prior_count"].tolist() == [1]
    assert voter["voter_dao_prior_count"].tolist() == [1]
    assert voter["voter_dao_prior_for"].tolist() == [1]
