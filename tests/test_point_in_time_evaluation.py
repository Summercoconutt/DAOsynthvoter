import numpy as np
import pandas as pd
import pytest

from dao_governance.evaluation.point_in_time import baseline_predictions, evaluate_matched_predictions
from dao_governance.evaluation.chronological_runs import filter_content_disjoint_test


def _targets():
    return pd.DataFrame(
        [
            {"example_id": "e1", "space": "dao", "proposal_id": "p1", "vote_ts": "2023-01-05", "label_id": 0,
             "voter_has_history": 0, "voter_dao_prior_count": 0, "dao_prior_count": 0,
             "voter_dao_prior_for": np.nan, "voter_dao_prior_against": np.nan, "voter_dao_prior_abstain": np.nan},
            {"example_id": "e2", "space": "dao", "proposal_id": "p2", "vote_ts": "2023-01-05", "label_id": 1,
             "voter_has_history": 1, "voter_dao_prior_count": 2, "dao_prior_count": 8,
             "voter_dao_prior_for": 0.0, "voter_dao_prior_against": 1.0, "voter_dao_prior_abstain": 0.0},
            {"example_id": "e3", "space": "dao", "proposal_id": "p2", "vote_ts": "2023-01-05", "label_id": 2,
             "voter_has_history": 1, "voter_dao_prior_count": 5, "dao_prior_count": 8,
             "voter_dao_prior_for": 0.0, "voter_dao_prior_against": 0.0, "voter_dao_prior_abstain": 1.0},
        ]
    )


def _manifest():
    rows = [
        {"space": "dao", "proposal_id": "p0", "split": "train", "start": "2023-01-01", "end": "2023-01-03"},
        {"space": "dao", "proposal_id": "p1", "split": "validation", "start": "2023-01-04", "end": "2023-01-06"},
        {"space": "dao", "proposal_id": "p2", "split": "validation", "start": "2023-01-04", "end": "2023-01-06"},
        {"space": "dao", "proposal_id": "p3", "split": "test", "start": "2023-01-07", "end": "2023-01-09"},
    ]
    for row in rows:
        row.update({"labels_by_position": [0, 1, 2],
                    "content_sha256": hashlib.sha256(row["proposal_id"].encode()).hexdigest(),
                    "provenance_sha256": "d" * 64, "content_sensitivity_excluded": False,
                    "metadata_available_at": (pd.Timestamp(row["start"], tz="UTC") - pd.Timedelta(seconds=1)).isoformat()})
    return {"version": 1, "dataset_sha256": "a" * 64, "train_cutoff": "2023-01-04", "test_cutoff": "2023-01-07", "proposals": rows}


def test_matched_baselines_report_fixed_classes_and_proposal_bootstrap():
    targets = _targets()
    predictions = baseline_predictions(pd.DataFrame({"label_id": [0, 0, 1]}), targets)
    result = evaluate_matched_predictions(
        predictions, manifest=_manifest(), split="validation", split_fingerprint="b" * 64,
        bootstrap_samples=20, seed=3, strata=targets
    )
    assert result["n_examples"] == 3
    assert result["n_proposals"] == 2
    assert result["models"]["historical_choice"]["metrics"]["macro_f1"] == pytest.approx(1.0)
    assert result["models"]["majority"]["proposal_bootstrap_macro_f1"]["samples"] == 20
    assert result["models"]["majority"]["strata"]["voter_history"]["new"]["n"] == 1


def test_comparison_rejects_mismatched_target_rows_or_labels():
    predictions = baseline_predictions(pd.DataFrame({"label_id": [0]}), _targets())
    predictions["historical_choice"].loc[0, "y_true"] = 2
    with pytest.raises(ValueError, match="identical target IDs"):
        evaluate_matched_predictions(
            predictions, manifest=_manifest(), split="validation", split_fingerprint="b" * 64, bootstrap_samples=2
        )


def test_test_split_requires_frozen_selection():
    predictions = baseline_predictions(pd.DataFrame({"label_id": [0]}), _targets())
    with pytest.raises(ValueError, match="frozen validation selection"):
        evaluate_matched_predictions(
            predictions, manifest=_manifest(), split="test", split_fingerprint="b" * 64, bootstrap_samples=2
        )


def test_content_disjoint_sensitivity_filters_later_duplicate_proposals():
    frame = pd.DataFrame({"space": ["dao", "dao"], "proposal_id": ["duplicate", "novel"]})
    manifest = _manifest()
    duplicate_hash = manifest["proposals"][0]["content_sha256"]
    for proposal_id, content_hash, excluded in (
        ("duplicate", duplicate_hash, True), ("novel", "e" * 64, False)
    ):
        manifest["proposals"].append(
            {"space": "dao", "proposal_id": proposal_id, "split": "test", "start": "2023-01-07",
             "end": "2023-01-09", "labels_by_position": [0, 1, 2],
             "content_sha256": content_hash, "provenance_sha256": "d" * 64,
             "content_sensitivity_excluded": excluded, "metadata_available_at": "2023-01-06T23:59:59Z"}
        )
    assert filter_content_disjoint_test(frame, manifest)["proposal_id"].tolist() == ["novel"]
    with pytest.raises(ValueError, match="no held-out test"):
        filter_content_disjoint_test(frame.iloc[[0]], manifest)
import hashlib

