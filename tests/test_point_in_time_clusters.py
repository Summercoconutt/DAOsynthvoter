from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from dao_governance.features.point_in_time_clusters import (
    UNKNOWN_CLUSTER,
    PointInTimeClusterBundle,
    diagnose_cluster_fit,
    fit_point_in_time_clusters,
)
from dao_governance.clustering.predictive import fit_predictive_stage, save_fit_profiles
from dao_governance.features.asof import ObservationProvenance, VoteEvent, iter_asof_features
from dao_governance.modelling.point_in_time_artifacts import load_checkpoint_clusters, save_validation_checkpoint
from dao_governance.modelling.point_in_time_model import FeatureEncoder, VoteChoiceClassifier
from dao_governance.modelling.point_in_time_training import TrainingResult

FINGERPRINT = "b" * 64
OBSERVATION_PROVENANCE = ObservationProvenance(True, "synthetic timestamps")


def _write_split(path: Path):
    proposals = [
        {"space": space, "proposal_id": f"train-{space}", "split": "train",
         "start": "2023-01-01", "end": "2023-01-02"}
        for space in ("a.eth", "b.eth", "c.eth")
    ]
    proposals.extend(
        [
            {"space": "a.eth", "proposal_id": "validation", "split": "validation",
             "start": "2023-01-11", "end": "2023-01-12"},
            {"space": "a.eth", "proposal_id": "test", "split": "test",
             "start": "2023-01-21", "end": "2023-01-22"},
        ]
    )
    for proposal in proposals:
        proposal.update({"labels_by_position": [0, 1, 2],
                         "content_sha256": hashlib.sha256(proposal["proposal_id"].encode()).hexdigest(),
                         "provenance_sha256": "d" * 64, "content_sensitivity_excluded": False,
                         "metadata_available_at": (pd.Timestamp(proposal["start"], tz="UTC") - pd.Timedelta(seconds=1)).isoformat()})
    path.write_text(
        json.dumps(
            {"version": 1, "dataset_sha256": FINGERPRINT, "train_cutoff": "2023-01-10T00:00:00Z",
             "test_cutoff": "2023-01-20T00:00:00Z", "proposals": proposals}
        ),
        encoding="utf-8",
    )


def _profiles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"voter": f"v{i}", "space": space, "asof_ts": "2023-01-03T00:00:00Z",
             "voter_dao_prior_count": 3, "voter_dao_prior_for": value,
             "dao_prior_count": 10, "dao_prior_for": dao_value}
            for space, dao_value in (("a.eth", 0.1), ("b.eth", 0.2), ("c.eth", 0.9))
            for i, value in enumerate((0.0, 0.1, 0.9, 1.0))
        ]
    )


def _dao_bundle(profiles):
    daos = profiles.drop_duplicates("space").assign(asof_ts="2023-01-01T00:00:00Z")
    return fit_point_in_time_clusters(
        daos, kind="dao", scope=None, feature_columns=["dao_prior_for"],
        history_column="dao_prior_count", fit_cutoff="2023-01-02T00:00:00Z",
        train_cutoff="2023-01-10T00:00:00Z", split_fingerprint=FINGERPRINT,
        n_clusters=2, min_group_size=3,
    )


def _fit(profiles, scope, **kwargs):
    dao_bundle = _dao_bundle(profiles) if scope == "dao_cluster" else None
    return fit_point_in_time_clusters(
        profiles,
        kind="voter",
        scope=scope,
        feature_columns=["voter_dao_prior_for"],
        history_column="voter_dao_prior_count",
        fit_cutoff="2023-01-04T00:00:00Z",
        train_cutoff="2023-01-10T00:00:00Z",
        split_fingerprint=FINGERPRINT,
        n_clusters=2,
        min_group_size=3,
        dao_bundle=dao_bundle,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("scope", "expected_groups"),
    [("global", {"all"}), ("dao_cluster", None), ("individual_dao", {"a.eth", "b.eth", "c.eth"})],
)
def test_voter_clustering_scopes_fit_separate_voter_dao_profiles(scope, expected_groups):
    profiles = _profiles()
    bundle = _fit(profiles, scope)
    if scope == "dao_cluster":
        expected_groups = set(bundle.dao_bundle.assign(profiles)["cluster"])
        assert len(expected_groups) == 2
    assert set(bundle.models) == expected_groups
    later = profiles.assign(asof_ts="2023-01-05T00:00:00Z")
    assigned = bundle.assign(later)
    assert (assigned["cluster_status"] == "assigned").all()
    assert UNKNOWN_CLUSTER not in assigned["cluster"].tolist()


def test_group_identity_unknown_cases_and_persisted_model(tmp_path: Path):
    profiles = _profiles()
    bundle = _fit(profiles, "individual_dao")
    later = profiles.iloc[[0, 4]].copy().assign(asof_ts="2023-01-05T00:00:00Z")
    assigned = bundle.assign(later)
    assert assigned.iloc[0]["cluster"] != assigned.iloc[1]["cluster"]
    artifact = tmp_path / "voter-clusters.pkl"
    bundle.save(artifact)
    restored = PointInTimeClusterBundle.load(artifact)
    pd.testing.assert_frame_equal(restored.assign(later), assigned)
    with pytest.raises(FileExistsError):
        bundle.save(artifact)

    cold = later.iloc[[0]].copy()
    cold["voter_dao_prior_count"] = 0
    assert restored.assign(cold).iloc[0].to_dict() == {
        "cluster": UNKNOWN_CLUSTER, "cluster_status": "insufficient_history"
    }
    unseen = later.iloc[[0]].copy()
    unseen["space"] = "new.eth"
    assert restored.assign(unseen).iloc[0]["cluster_status"] == "unseen_group"


def test_dao_group_scope_never_falls_back_to_global():
    bundle = _fit(_profiles(), "dao_cluster")
    target = _profiles().iloc[[0]].copy().assign(asof_ts="2023-01-05T00:00:00Z", dao_prior_count=0)
    assert bundle.assign(target).iloc[0].to_dict() == {
        "cluster": UNKNOWN_CLUSTER, "cluster_status": "unknown_group"
    }
    with pytest.raises(ValueError, match="external column"):
        bundle.assign(target.assign(dao_cluster="forged"))


def test_future_profiles_cannot_enter_fit_or_change_earlier_assignments():
    profiles = _profiles()
    bundle = _fit(profiles, "global")
    before_fit = profiles.iloc[[0]].copy()
    assert bundle.assign(before_fit).iloc[0]["cluster_status"] == "before_fit"
    future = profiles.copy()
    future.loc[0, "asof_ts"] = "2023-01-05T00:00:00Z"
    with pytest.raises(ValueError, match="after the fit cutoff"):
        _fit(future, "global")
    with pytest.raises(ValueError, match="training cutoff"):
        fit_point_in_time_clusters(
            profiles, kind="voter", scope="global", feature_columns=["voter_dao_prior_for"],
            history_column="voter_dao_prior_count", fit_cutoff="2023-01-11",
            train_cutoff="2023-01-10", split_fingerprint=FINGERPRINT, min_group_size=3,
        )


def test_cluster_diagnostics_are_train_only_and_report_stability():
    profiles = _profiles()
    bundle = _fit(profiles, "global")
    diagnostics = diagnose_cluster_fit(profiles, bundle, max_samples=6)
    group = diagnostics["groups"]["all"]
    assert group["training_profiles"] == len(profiles)
    assert sum(group["cluster_sizes"]) == len(profiles)
    assert group["sampled_profiles"] == 6
    assert -1 <= group["seed_stability_adjusted_rand"] <= 1
    future = profiles.copy()
    future.loc[0, "asof_ts"] = "2023-01-05T00:00:00Z"
    with pytest.raises(ValueError, match="fit-time"):
        diagnose_cluster_fit(future, bundle)


def test_alternative_clusterer_and_minimum_cluster_size():
    profiles = _profiles()
    mixture = _fit(profiles, "global", algorithm="gaussian_mixture")
    assigned = mixture.assign(profiles.assign(asof_ts="2023-01-05T00:00:00Z"))
    assert (assigned["cluster_status"] == "assigned").all()
    with pytest.raises(ValueError, match="No training groups"):
        _fit(profiles, "global", min_cluster_size=7)
    invalid = profiles.copy()
    invalid.loc[0, "voter_dao_prior_for"] = 1.5
    with pytest.raises(ValueError, match="fractions"):
        _fit(invalid, "global")


def test_target_and_future_structural_changes_do_not_change_earlier_cluster():
    bundle = _fit(_profiles(), "global")

    def event(example, voter, proposal, time, label):
        timestamp = pd.Timestamp(time, tz="UTC")
        return VoteEvent(example, voter, "a.eth", proposal, timestamp, timestamp,
                         pd.Timestamp("2023-02-01", tz="UTC"), label)

    original = [
        event("prior", "v0", "p1", "2023-01-03", 0),
        event("target", "v0", "p2", "2023-01-05", 1),
        event("future", "v0", "p3", "2023-01-06", 2),
    ]
    changed = [
        original[0],
        event("target", "v0", "p2", "2023-01-05", 2),
        event("future", "different-voter", "p3", "2023-01-06", 0),
        event("extra", "v0", "p4", "2023-01-07", 1),
    ]

    def target_profile(events):
        features = pd.DataFrame(
            iter_asof_features(events, events, availability_provenance=OBSERVATION_PROVENANCE)
        )
        return features.loc[features["example_id"] == "target"].reset_index(drop=True)

    before, after = target_profile(original), target_profile(changed)
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(bundle.assign(before), bundle.assign(after))


def test_predictive_stage_commands_preserve_scope_and_dao_dependency(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    split = tmp_path / "split.json"
    _write_split(split)
    profiles = _profiles()
    dao_profiles = tmp_path / "dao.parquet"
    voter_profiles = tmp_path / "voters.parquet"
    save_fit_profiles(
        profiles.drop_duplicates("space").assign(asof_ts="2023-01-01T00:00:00Z"),
        dao_profiles, kind="dao", split_manifest=split, fit_cutoff="2023-01-02T00:00:00Z",
        availability_provenance=OBSERVATION_PROVENANCE,
    )
    save_fit_profiles(
        profiles, voter_profiles, kind="voter", split_manifest=split,
        fit_cutoff="2023-01-04T00:00:00Z", availability_provenance=OBSERVATION_PROVENANCE,
    )
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    dao_output = tmp_path / "dao.pkl"
    subprocess.run(
        [sys.executable, str(root / "scripts/06_run_dao_clustering.py"),
         "--predictive-profiles", str(dao_profiles), "--split-manifest", str(split),
         "--cluster-output", str(dao_output), "--cluster-feature", "dao_prior_for", "--min-group-size", "3"],
        check=True, capture_output=True, text=True, env=env,
    )
    for scope in ("global", "dao_cluster", "individual_dao"):
        output = tmp_path / f"{scope}.pkl"
        command = [
            sys.executable, str(root / "scripts/07_run_voter_clustering.py"),
            "--predictive-profiles", str(voter_profiles), "--split-manifest", str(split),
            "--cluster-output", str(output), "--cluster-feature", "voter_dao_prior_for",
            "--cluster-scope", scope, "--min-group-size", "3",
        ]
        if scope == "dao_cluster":
            command.extend(["--dao-bundle", str(dao_output)])
        subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        assert PointInTimeClusterBundle.load(output).scope == scope

    dao_bundle = PointInTimeClusterBundle.load(dao_output)
    voter_bundle = PointInTimeClusterBundle.load(tmp_path / "dao_cluster.pkl")
    later = profiles.assign(asof_ts="2023-01-05T00:00:00Z")
    cluster_features = pd.DataFrame(
        {"dao_cluster": dao_bundle.assign(later)["cluster"], "voter_cluster": voter_bundle.assign(later)["cluster"]}
    )
    encoder = FeatureEncoder((), ("dao_cluster", "voter_cluster")).fit(cluster_features)
    run = tmp_path / "clustered"
    save_validation_checkpoint(
        run, run_id="clustered", model=VoteChoiceClassifier(numeric_dim=encoder.dimension),
        feature_encoder=encoder, split_manifest=split,
        training=TrainingResult(best_epoch=1, best_validation_macro_f1=0.5, epochs=[]),
        observation_provenance=OBSERVATION_PROVENANCE, history_mode="other_proposals_only",
        dao_bundle_path=dao_output, voter_bundle_path=tmp_path / "dao_cluster.pkl",
    )
    saved = load_checkpoint_clusters(run, expected_run_id="clustered", expected_split=split)
    assert saved["voter"].scope == "dao_cluster"
    pd.testing.assert_frame_equal(saved["voter"].assign(later), voter_bundle.assign(later))

    with pytest.raises(ValueError, match="requires a persisted DAO bundle"):
        fit_predictive_stage(
            profiles_path=voter_profiles, split_manifest=split, output=tmp_path / "missing.pkl",
            kind="voter", scope="dao_cluster", feature_columns=["voter_dao_prior_for"],
            history_column="voter_dao_prior_count", n_clusters=2, min_group_size=3,
            min_history=1, algorithm="kmeans",
        )


def test_predictive_stage_rejects_modified_profile(tmp_path: Path):
    split = tmp_path / "split.json"
    _write_split(split)
    profiles = tmp_path / "profiles.parquet"
    save_fit_profiles(
        _profiles(), profiles, kind="voter", split_manifest=split, fit_cutoff="2023-01-04",
        availability_provenance=OBSERVATION_PROVENANCE,
    )
    profiles.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="provenance"):
        fit_predictive_stage(
            profiles_path=profiles, split_manifest=split, output=tmp_path / "out.pkl",
            kind="voter", scope="global", feature_columns=["voter_dao_prior_for"],
            history_column="voter_dao_prior_count", n_clusters=2, min_group_size=3,
            min_history=1, algorithm="kmeans",
        )
