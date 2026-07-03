"""
Leakage-safety and pipeline smoke tests (synthetic data; no server files required).
Run: python -m pytest tests/test_leakage_safe_pipeline.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.features.behaviour_dataset import build_behaviour_dataset
from dao_governance.features.behaviour_pipeline import load_behaviour_votes, prepare_behaviour_splits
from dao_governance.features.causal_clusters import (
    DAO_CLUSTER_FEATURES,
    ClusterBundle,
    assign_clusters_to_votes,
    fit_cluster_bundle,
)
from dao_governance.modelling.preprocess import select_numeric_columns
from dao_governance.modelling.windows import WINDOW_GROUP_COLS, build_windows

DAO_FIXTURE = ROOT / "src" / "dao_clustering_scripts" / "data" / "processed" / "dao_feature_table.csv"
SPACES = ["ens.eth", "uniswapgovernance.eth"]


def _synthetic_votes(n_voters: int = 24, votes_per_pair: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base_ts = pd.Timestamp("2023-01-01", tz="UTC")
    for vi in range(n_voters):
        voter = f"voter_{vi:03d}"
        for space in SPACES:
            for j in range(votes_per_pair):
                choice = ["for", "against", "abstain"][j % 3]
                rows.append(
                    {
                        "voter": voter,
                        "space": space,
                        "proposal_id": f"p_{space}_{j % 4}",
                        "choice_norm": choice,
                        "vote_timestamp": base_ts + pd.Timedelta(days=vi, hours=j),
                        "proposal_title": f"Proposal {j} in {space}",
                        "voting_power": float(100 + vi * 10 + j),
                        "is_whale": bool(vi % 10 == 0),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_parquet(tmp_path: Path) -> Path:
    pq = tmp_path / "votes.parquet"
    _synthetic_votes().to_parquet(pq, index=False)
    return pq


class TestNoLeakyModelInputs:
    def test_select_numeric_columns_excludes_label_derived(self):
        cols = select_numeric_columns()
        assert "aligned_with_majority" not in cols
        assert "vp_share" not in cols
        assert "label_id" not in cols
        assert set(cols) == {"voting_power", "is_whale", "dao_cluster", "voter_cluster"}

    def test_current_window_step_zeros_smuggled_alignment(self):
        """Even if aligned_with_majority is present in df, it must not reach current step."""
        cols = list(select_numeric_columns()) + ["aligned_with_majority"]
        df = _synthetic_votes(n_voters=4, votes_per_pair=6)
        df["vote_ts"] = pd.to_datetime(df["vote_timestamp"], utc=True)
        df["label_id"] = df["choice_norm"].map({"for": 0, "against": 1, "abstain": 2})
        df["text"] = df["proposal_title"]
        df["aligned_with_majority"] = (df["label_id"] == 0).astype(float)
        df["dao_cluster"] = 0
        df["voter_cluster"] = 1

        windows = build_windows(df, window_size=3, numeric_cols=cols)
        assert windows
        w = windows[0]
        cur_feat = w.window_features[-1]
        align_idx = cols.index("aligned_with_majority")
        assert cur_feat[align_idx] == 0.0


class TestClusterSafety:
    @pytest.mark.skipif(not DAO_FIXTURE.exists(), reason="DAO fixture CSV missing")
    def test_dao_features_are_structural_only(self):
        tbl = pd.read_csv(DAO_FIXTURE, nrows=1)
        for bad in ("z_rep_pct_for_votes", "z_rep_choice_entropy", "pct_for_votes"):
            assert bad not in DAO_CLUSTER_FEATURES

    @pytest.mark.skipif(not DAO_FIXTURE.exists(), reason="DAO fixture CSV missing")
    def test_cluster_fit_uses_train_voters_only(self, synthetic_parquet: Path):
        raw = _synthetic_votes()
        raw["vote_ts"] = pd.to_datetime(raw["vote_timestamp"], utc=True)
        raw["label_id"] = raw["choice_norm"].map({"for": 0, "against": 1, "abstain": 2})
        raw["text"] = raw["proposal_title"]

        from dao_governance.modelling.preprocess import split_by_voter_three_way

        train_raw, val_raw, _ = split_by_voter_three_way(raw, train_frac=0.7, val_frac=0.15, seed=42)
        bundle = fit_cluster_bundle(
            train_raw, DAO_FIXTURE, use_dao_clusters=True, use_voter_clusters=True, min_votes_per_pair=5
        )
        train_voters = set(train_raw["voter"].astype(str))
        val_voters = set(val_raw["voter"].astype(str))
        assert train_voters.isdisjoint(val_voters)
        for v, _s in bundle.voter_pairs:
            assert v in train_voters

    @pytest.mark.skipif(not DAO_FIXTURE.exists(), reason="DAO fixture CSV missing")
    def test_changing_test_labels_does_not_change_cluster_assign(self):
        raw = _synthetic_votes()
        raw["vote_ts"] = pd.to_datetime(raw["vote_timestamp"], utc=True)
        raw["label_id"] = raw["choice_norm"].map({"for": 0, "against": 1, "abstain": 2})
        raw["text"] = raw["proposal_title"]

        from dao_governance.modelling.preprocess import split_by_voter_three_way

        train_raw, val_raw, test_raw = split_by_voter_three_way(
            raw, train_frac=0.7, val_frac=0.15, seed=42
        )
        bundle = fit_cluster_bundle(
            train_raw, DAO_FIXTURE, min_votes_per_pair=5
        )
        val_a = assign_clusters_to_votes(val_raw, bundle, DAO_FIXTURE, min_votes_per_pair=5)

        val_corrupt = val_raw.copy()
        val_corrupt["choice_norm"] = "against"
        val_corrupt["label_id"] = 1
        val_b = assign_clusters_to_votes(val_corrupt, bundle, DAO_FIXTURE, min_votes_per_pair=5)

        merge_keys = ["voter", "space", "proposal_id"]
        a = val_a[merge_keys + ["dao_cluster", "voter_cluster"]].sort_values(merge_keys).reset_index(drop=True)
        b = val_b[merge_keys + ["dao_cluster", "voter_cluster"]].sort_values(merge_keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


class TestWindowGrouping:
    def test_voter_space_separate_sequences(self):
        df = pd.DataFrame(
            [
                {"voter": "v1", "space": "a.eth", "vote_ts": pd.Timestamp("2023-01-01", tz="UTC"), "label_id": 0, "text": "t", "voting_power": 1.0, "is_whale": 0, "dao_cluster": 0, "voter_cluster": 0},
                {"voter": "v1", "space": "b.eth", "vote_ts": pd.Timestamp("2023-01-02", tz="UTC"), "label_id": 1, "text": "t", "voting_power": 1.0, "is_whale": 0, "dao_cluster": 0, "voter_cluster": 0},
                {"voter": "v1", "space": "a.eth", "vote_ts": pd.Timestamp("2023-01-03", tz="UTC"), "label_id": 2, "text": "t", "voting_power": 1.0, "is_whale": 0, "dao_cluster": 0, "voter_cluster": 0},
                {"voter": "v1", "space": "a.eth", "vote_ts": pd.Timestamp("2023-01-04", tz="UTC"), "label_id": 0, "text": "t", "voting_power": 1.0, "is_whale": 0, "dao_cluster": 0, "voter_cluster": 0},
            ]
        )
        cols = select_numeric_columns()
        windows = build_windows(df, window_size=3, numeric_cols=cols)
        assert len(windows) == 1
        labels_in_text = [t.split()[0] for t in windows[0].window_texts[:-1]]
        assert labels_in_text == ["[LABEL_0]", "[LABEL_2]"]


class TestBehaviourDataset:
    def test_build_strips_legacy_clusters(self, synthetic_parquet: Path):
        votes = pd.read_parquet(synthetic_parquet)
        votes["dao_cluster"] = 99
        votes["voter_cluster"] = 88
        votes.to_parquet(synthetic_parquet, index=False)

        df, _ = build_behaviour_dataset(synthetic_parquet, include_legacy_clusters=False)
        assert "dao_cluster" not in df.columns
        assert "voter_cluster" not in df.columns
        assert "aligned_with_majority" not in df.columns

    def test_load_behaviour_votes_strips_leaky_cols(self, tmp_path: Path):
        p = tmp_path / "b.csv"
        pd.DataFrame(
            {
                "voter": ["v1"],
                "aligned_with_majority": [True],
                "vp_share": [0.5],
                "dao_cluster": [1],
            }
        ).to_csv(p, index=False)
        out = load_behaviour_votes(p)
        assert "aligned_with_majority" not in out.columns
        assert "vp_share" not in out.columns
        assert "dao_cluster" not in out.columns


@pytest.mark.skipif(not DAO_FIXTURE.exists(), reason="DAO fixture CSV missing")
class TestIntegrationPrepareSplits:
    def test_end_to_end_prepare(self, tmp_path: Path):
        raw = _synthetic_votes(n_voters=30, votes_per_pair=8)
        raw["vote_ts"] = pd.to_datetime(raw["vote_timestamp"], utc=True)
        raw["label_id"] = raw["choice_norm"].map({"for": 0, "against": 1, "abstain": 2})
        raw["text"] = raw["proposal_title"]
        csv_path = tmp_path / "behaviour.csv"
        raw.to_csv(csv_path, index=False)

        train_df, val_df, test_df, pre, bundle, *_ = prepare_behaviour_splits(
            load_behaviour_votes(csv_path),
            dao_feature_table_path=DAO_FIXTURE,
            train_frac=0.7,
            val_frac=0.15,
            seed=42,
            upper_quantile_cap=0.999,
            absolute_cap=1e18,
            min_votes_per_pair=5,
            cluster_artifacts_dir=tmp_path / "clusters",
        )
        cols = select_numeric_columns()
        for part in (train_df, val_df, test_df):
            for c in cols:
                assert c in part.columns
            assert part[cols].notna().all().all()

        windows = build_windows(train_df, window_size=3, numeric_cols=cols)
        assert len(windows) > 0
        assert len(windows[0].window_features[0]) == len(cols) + 4
        assert bundle.meta.get("clustering_mode") == "train_only_structural"
