"""
Train-only structural clustering for behaviour modelling (leakage-safe).

DAO clusters: structural DAO features only (no z_rep label centroids).
Voter clusters: structural (voter, space) features from train votes only.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

# Structural DAO features — no label-derived z_rep_* columns.
DAO_CLUSTER_FEATURES: List[str] = [
    "log_n_unique_voters",
    "mean_robust_participation_vp",
    "std_robust_participation_vp",
    "gini_voting_power",
    "whale_ratio_top1pct",
    "proposal_frequency_per_30d",
    "repeat_voter_rate",
]

# Fallback if log_n_unique_voters missing
DAO_LOG_FALLBACK = "n_unique_voters"

VOTER_CLUSTER_FEATURES: List[str] = [
    "log_total_votes",
    "avg_voting_power",
    "std_voting_power",
    "active_span_days",
    "vote_frequency",
    "participation_rate",
]

KMEANS_K_MIN = 2
KMEANS_K_MAX = 6
RANDOM_STATE = 42
N_INIT = 15


@dataclass
class ClusterBundle:
    """Fitted cluster models (train-only)."""

    dao_scaler: Optional[RobustScaler] = None
    dao_kmeans: Optional[KMeans] = None
    dao_feature_cols: List[str] = field(default_factory=list)
    dao_spaces: List[str] = field(default_factory=list)

    voter_scaler: Optional[RobustScaler] = None
    voter_kmeans: Optional[KMeans] = None
    voter_feature_cols: List[str] = field(default_factory=list)
    voter_pairs: List[Tuple[str, str]] = field(default_factory=list)

    use_dao_clusters: bool = True
    use_voter_clusters: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "ClusterBundle":
        with open(path, "rb") as f:
            return pickle.load(f)


def _ensure_log_n_unique_voters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "log_n_unique_voters" not in out.columns and DAO_LOG_FALLBACK in out.columns:
        out["log_n_unique_voters"] = np.log1p(
            pd.to_numeric(out[DAO_LOG_FALLBACK], errors="coerce").clip(lower=0)
        )
    return out


def _pick_kmeans_k(X: np.ndarray, k_min: int = KMEANS_K_MIN, k_max: int = KMEANS_K_MAX) -> int:
    n = X.shape[0]
    if n <= k_min:
        return max(2, min(n, 2))
    best_k, best_inertia = k_min, float("inf")
    for k in range(k_min, min(k_max, n - 1) + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        km.fit(X)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            best_k = k
    return best_k


def _fill_median(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        med = float(out[c].median()) if out[c].notna().any() else 0.0
        out[c] = out[c].fillna(med)
    return out


def fit_dao_clusters(
    train_df: pd.DataFrame,
    dao_feature_table_path: Path,
    *,
    use_clusters: bool = True,
) -> Tuple[Optional[RobustScaler], Optional[KMeans], List[str], pd.DataFrame]:
    """
    Fit DAO KMeans on structural features for spaces seen in train_df only.
    Returns (scaler, kmeans, feature_cols, assignments_df with columns space, dao_cluster).
    """
    if not use_clusters or train_df.empty:
        return None, None, [], pd.DataFrame(columns=["space", "dao_cluster"])

    if not dao_feature_table_path.exists():
        raise FileNotFoundError(f"DAO feature table not found: {dao_feature_table_path}")

    if dao_feature_table_path.suffix.lower() == ".parquet":
        dao_tbl = pd.read_parquet(dao_feature_table_path)
    else:
        dao_tbl = pd.read_csv(dao_feature_table_path)

    dao_tbl = _ensure_log_n_unique_voters(dao_tbl)
    feat_cols = [c for c in DAO_CLUSTER_FEATURES if c in dao_tbl.columns]
    if len(feat_cols) < 2:
        raise ValueError(
            f"Need at least 2 DAO structural features; found {feat_cols}. "
            f"Check {dao_feature_table_path}"
        )

    train_spaces = train_df["space"].dropna().astype(str).unique().tolist()
    sub = dao_tbl[dao_tbl["space"].astype(str).isin(train_spaces)].copy()
    if sub.empty:
        raise ValueError("No DAO feature rows for train spaces.")

    sub = _fill_median(sub, feat_cols)
    X = sub[feat_cols].to_numpy(dtype=float)

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    k = _pick_kmeans_k(Xs)
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = kmeans.fit_predict(Xs)

    assign = pd.DataFrame({"space": sub["space"].astype(str), "dao_cluster": labels.astype(int)})
    return scaler, kmeans, feat_cols, assign


def build_voter_structural_features(votes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Structural (voter, space) features from vote rows — no label proportions.
    Uses only votes present in votes_df (caller passes train-only for fit).
    """
    v = votes_df.copy()
    v["vote_ts"] = pd.to_datetime(
        v.get("vote_ts", v.get("vote_timestamp", pd.NaT)), utc=True, errors="coerce"
    )
    v["voting_power"] = pd.to_numeric(v.get("voting_power", np.nan), errors="coerce")
    gcols = ["voter", "space"]
    G = v.groupby(gcols, dropna=False)

    feat = G.size().rename("total_votes").reset_index()
    feat = feat.merge(G["voting_power"].mean().rename("avg_voting_power").reset_index(), on=gcols)
    feat = feat.merge(G["voting_power"].std(ddof=1).rename("std_voting_power").reset_index(), on=gcols)
    feat["std_voting_power"] = feat["std_voting_power"].fillna(0.0)

    t0 = G["vote_ts"].min().reset_index(name="t0")
    t1 = G["vote_ts"].max().reset_index(name="t1")
    feat = feat.merge(t0, on=gcols).merge(t1, on=gcols)
    feat["active_span_days"] = ((feat["t1"] - feat["t0"]).dt.days.fillna(0).astype(int) + 1).clip(lower=1)
    feat["vote_frequency"] = feat["total_votes"] / feat["active_span_days"]
    feat["log_total_votes"] = np.log1p(feat["total_votes"].astype(float))

    np_ = (
        v.dropna(subset=["proposal_id"])
        .groupby("space")["proposal_id"]
        .nunique()
        .rename("n_proposals_space")
        .reset_index()
    )
    feat = feat.merge(np_, on="space", how="left")
    feat["participation_rate"] = feat["total_votes"] / feat["n_proposals_space"].replace(0, np.nan)
    feat["participation_rate"] = feat["participation_rate"].fillna(0.0)
    feat.drop(columns=["t0", "t1"], inplace=True, errors="ignore")
    return feat


def fit_voter_clusters(
    train_df: pd.DataFrame,
    *,
    use_clusters: bool = True,
    min_votes_per_pair: int = 5,
) -> Tuple[Optional[RobustScaler], Optional[KMeans], List[str], pd.DataFrame]:
    """
    Fit voter KMeans on structural (voter, space) features from train votes only.
    """
    if not use_clusters or train_df.empty:
        return None, None, [], pd.DataFrame(columns=["voter", "space", "voter_cluster"])

    feat = build_voter_structural_features(train_df)
    feat = feat[feat["total_votes"] >= min_votes_per_pair].copy()
    if len(feat) < KMEANS_K_MIN + 1:
        raise ValueError(
            f"Too few (voter, space) pairs for clustering after min_votes={min_votes_per_pair}: {len(feat)}"
        )

    feat_cols = [c for c in VOTER_CLUSTER_FEATURES if c in feat.columns]
    feat = _fill_median(feat, feat_cols)
    X = feat[feat_cols].to_numpy(dtype=float)

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    k = _pick_kmeans_k(Xs)
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = kmeans.fit_predict(Xs)

    assign = feat[["voter", "space"]].copy()
    assign["voter"] = assign["voter"].astype(str)
    assign["space"] = assign["space"].astype(str)
    assign["voter_cluster"] = labels.astype(int)
    return scaler, kmeans, feat_cols, assign


def predict_dao_clusters(
    spaces: Sequence[str],
    dao_tbl: pd.DataFrame,
    scaler: RobustScaler,
    kmeans: KMeans,
    feat_cols: List[str],
) -> pd.DataFrame:
    """Assign dao_cluster; unseen spaces get -1."""
    dao_tbl = _ensure_log_n_unique_voters(dao_tbl)
    sub = dao_tbl[dao_tbl["space"].astype(str).isin([str(s) for s in spaces])].copy()
    out_rows = []
    for space in spaces:
        space = str(space)
        row = sub[sub["space"].astype(str) == space]
        if row.empty:
            out_rows.append({"space": space, "dao_cluster": -1})
            continue
        r = _fill_median(row, feat_cols)
        X = r[feat_cols].to_numpy(dtype=float)
        Xs = scaler.transform(X)
        out_rows.append({"space": space, "dao_cluster": int(kmeans.predict(Xs)[0])})
    return pd.DataFrame(out_rows)


def predict_voter_clusters(
    votes_df: pd.DataFrame,
    scaler: RobustScaler,
    kmeans: KMeans,
    feat_cols: List[str],
    *,
    min_votes_per_pair: int = 5,
) -> pd.DataFrame:
    """Assign voter_cluster from structural features; low-activity pairs get -1."""
    feat = build_voter_structural_features(votes_df)
    feat["voter"] = feat["voter"].astype(str)
    feat["space"] = feat["space"].astype(str)
    labels = []
    for _, row in feat.iterrows():
        if int(row["total_votes"]) < min_votes_per_pair:
            labels.append(-1)
            continue
        r = _fill_median(pd.DataFrame([row]), feat_cols)
        X = r[feat_cols].to_numpy(dtype=float)
        Xs = scaler.transform(X)
        labels.append(int(kmeans.predict(Xs)[0]))
    feat["voter_cluster"] = labels
    return feat[["voter", "space", "voter_cluster"]]


def fit_cluster_bundle(
    train_df: pd.DataFrame,
    dao_feature_table_path: Path,
    *,
    use_dao_clusters: bool = True,
    use_voter_clusters: bool = True,
    min_votes_per_pair: int = 5,
) -> ClusterBundle:
    bundle = ClusterBundle(use_dao_clusters=use_dao_clusters, use_voter_clusters=use_voter_clusters)

    if use_dao_clusters:
        scaler, km, cols, dao_assign = fit_dao_clusters(
            train_df, dao_feature_table_path, use_clusters=True
        )
        bundle.dao_scaler = scaler
        bundle.dao_kmeans = km
        bundle.dao_feature_cols = cols
        bundle.dao_spaces = dao_assign["space"].astype(str).tolist()
        bundle.meta["dao_k"] = int(km.n_clusters) if km else 0
        bundle.meta["n_dao_train_spaces"] = len(dao_assign)

    if use_voter_clusters:
        scaler, km, cols, voter_assign = fit_voter_clusters(
            train_df, use_clusters=True, min_votes_per_pair=min_votes_per_pair
        )
        bundle.voter_scaler = scaler
        bundle.voter_kmeans = km
        bundle.voter_feature_cols = cols
        bundle.voter_pairs = list(
            zip(voter_assign["voter"].astype(str), voter_assign["space"].astype(str))
        )
        bundle.meta["voter_k"] = int(km.n_clusters) if km else 0
        bundle.meta["n_voter_train_pairs"] = len(voter_assign)

    bundle.meta["clustering_mode"] = "train_only_structural"
    return bundle


def assign_clusters_to_votes(
    df: pd.DataFrame,
    bundle: ClusterBundle,
    dao_feature_table_path: Path,
    *,
    min_votes_per_pair: int = 5,
) -> pd.DataFrame:
    """Merge dao_cluster and voter_cluster onto vote-level dataframe."""
    out = df.copy()
    out["voter"] = out["voter"].astype(str)
    out["space"] = out["space"].astype(str)

    if "dao_cluster" in out.columns:
        out = out.drop(columns=["dao_cluster"])
    if "voter_cluster" in out.columns:
        out = out.drop(columns=["voter_cluster"])

    if bundle.use_dao_clusters and bundle.dao_kmeans is not None:
        if dao_feature_table_path.suffix.lower() == ".parquet":
            dao_tbl = pd.read_parquet(dao_feature_table_path)
        else:
            dao_tbl = pd.read_csv(dao_feature_table_path)
        spaces = out["space"].unique().tolist()
        dao_assign = predict_dao_clusters(
            spaces,
            dao_tbl,
            bundle.dao_scaler,
            bundle.dao_kmeans,
            bundle.dao_feature_cols,
        )
        out = out.merge(dao_assign, on="space", how="left")
    else:
        out["dao_cluster"] = -1

    if bundle.use_voter_clusters and bundle.voter_kmeans is not None:
        voter_assign = predict_voter_clusters(
            out,
            bundle.voter_scaler,
            bundle.voter_kmeans,
            bundle.voter_feature_cols,
            min_votes_per_pair=min_votes_per_pair,
        )
        out = out.merge(voter_assign, on=["voter", "space"], how="left")
    else:
        out["voter_cluster"] = -1

    out["dao_cluster"] = pd.to_numeric(out["dao_cluster"], errors="coerce").fillna(-1).astype(int)
    out["voter_cluster"] = pd.to_numeric(out["voter_cluster"], errors="coerce").fillna(-1).astype(int)
    return out


def save_cluster_report(path: Path, bundle: ClusterBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Causal cluster report (train-only structural)\n\n"
        + json.dumps(bundle.meta, indent=2)
        + "\n",
        encoding="utf-8",
    )
