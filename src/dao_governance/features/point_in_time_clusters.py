"""Train-fitted clustering of point-in-time DAO and voter/DAO profiles."""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

VoterScope = Literal["global", "dao_cluster", "individual_dao"]
Algorithm = Literal["kmeans", "gaussian_mixture"]
UNKNOWN_CLUSTER = "unknown"
_FORBIDDEN_FEATURES = {
    "label_id",
    "choice_norm",
    "target_label",
    "voting_power",
    "is_whale",
    "vp_share",
    "aligned_with_majority",
    "dao_cluster",
    "voter_cluster",
}


@dataclass
class _GroupModel:
    imputer: SimpleImputer
    scaler: RobustScaler
    estimator: Union[KMeans, GaussianMixture]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        values = self.imputer.transform(features)
        scaled = self.scaler.transform(values)
        return self.estimator.predict(scaled)


@dataclass
class PointInTimeClusterBundle:
    kind: Literal["dao", "voter"]
    scope: Optional[VoterScope]
    feature_columns: Tuple[str, ...]
    history_column: str
    min_history: int
    fit_cutoff: pd.Timestamp
    train_cutoff: pd.Timestamp
    split_fingerprint: str
    algorithm: Algorithm
    min_cluster_size: int
    models: Dict[str, _GroupModel]
    skipped_groups: Dict[str, str]
    fit_summary: Dict[str, int]
    dao_bundle: Optional["PointInTimeClusterBundle"] = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, path: Path) -> "PointInTimeClusterBundle":
        with path.open("rb") as stream:
            bundle = pickle.load(stream)
        if not isinstance(bundle, cls):
            raise ValueError("Cluster artifact has an unexpected type.")
        return bundle

    def assign(self, profiles: pd.DataFrame) -> pd.DataFrame:
        profiles_for_grouping = _attach_dao_groups(profiles, self.scope, self.dao_bundle)
        frame = _prepare_profiles(
            profiles_for_grouping,
            kind=self.kind,
            scope=self.scope,
            feature_columns=self.feature_columns,
            history_column=self.history_column,
        )
        categories = [UNKNOWN_CLUSTER] * len(frame)
        statuses = ["unknown_group"] * len(frame)
        eligible_by_group: Dict[str, list[int]] = {}
        for index, row in frame.iterrows():
            if row["asof_ts"] <= self.fit_cutoff:
                statuses[index] = "before_fit"
            elif pd.isna(row[self.history_column]) or row[self.history_column] < self.min_history:
                statuses[index] = "insufficient_history"
            else:
                group = _group_key(row, self.kind, self.scope)
                if group is None:
                    statuses[index] = "unknown_group"
                elif group not in self.models:
                    statuses[index] = "insufficient_training_group" if group in self.skipped_groups else "unseen_group"
                else:
                    eligible_by_group.setdefault(group, []).append(index)

        for group, indices in eligible_by_group.items():
            model = self.models[group]
            labels = model.predict(frame.loc[indices, list(self.feature_columns)])
            for index, label in zip(indices, labels):
                identity = [self.kind, self.scope, group, int(label)]
                categories[index] = json.dumps(identity, ensure_ascii=True, separators=(",", ":"))
                statuses[index] = "assigned"
        return pd.DataFrame(
            {"cluster": categories, "cluster_status": statuses},
            index=profiles.index,
        )


def _time(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid UTC timestamp.")
    return timestamp


def _attach_dao_groups(
    profiles: pd.DataFrame,
    scope: Optional[VoterScope],
    dao_bundle: Optional[PointInTimeClusterBundle],
) -> pd.DataFrame:
    if scope != "dao_cluster":
        return profiles
    if dao_bundle is None or dao_bundle.kind != "dao":
        raise ValueError("DAO-cluster voter scope requires a train-fitted DAO cluster bundle.")
    if "dao_cluster" in profiles.columns:
        raise ValueError("DAO-cluster groups must come from the fitted DAO bundle, not an external column.")
    result = profiles.copy()
    result["dao_cluster"] = dao_bundle.assign(profiles)["cluster"].to_numpy()
    return result


def _prepare_profiles(
    profiles: pd.DataFrame,
    *,
    kind: Literal["dao", "voter"],
    scope: Optional[VoterScope],
    feature_columns: Sequence[str],
    history_column: str,
) -> pd.DataFrame:
    required = {"space", "asof_ts", history_column, *feature_columns}
    if kind == "voter":
        required.add("voter")
        if scope == "dao_cluster":
            required.add("dao_cluster")
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Cluster profiles are missing columns: {sorted(missing)}")
    frame = profiles.reset_index(drop=True).copy()
    frame["asof_ts"] = pd.to_datetime(frame["asof_ts"], utc=True, errors="coerce")
    if frame["asof_ts"].isna().any():
        raise ValueError("Cluster profiles require valid as-of timestamps.")
    for column in ("space", "voter") if kind == "voter" else ("space",):
        invalid = frame[column].isna() | frame[column].astype("string").str.strip().eq("")
        if invalid.any():
            raise ValueError(f"Cluster profiles require nonempty {column} identifiers.")
    for column in set(feature_columns) | {history_column}:
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & numeric.isna()).any() or np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Cluster profile column {column!r} contains invalid numeric values.")
        values = numeric.dropna()
        if column.endswith("_prior_count") and (values < 0).any():
            raise ValueError("Cluster history counts must be nonnegative.")
        if column.endswith("_has_history") and not values.isin([0, 1]).all():
            raise ValueError("Cluster history indicators must be zero or one.")
        if column.endswith(("_prior_for", "_prior_against", "_prior_abstain")) and not values.between(0, 1).all():
            raise ValueError("Cluster prior choice fractions must be between zero and one.")
        frame[column] = numeric
    return frame


def _group_key(row: pd.Series, kind: str, scope: Optional[VoterScope]) -> Optional[str]:
    if kind == "dao" or scope == "global":
        return "all"
    if scope == "individual_dao":
        return str(row["space"])
    value = row["dao_cluster"]
    if pd.isna(value) or str(value).strip().lower() in {"", "-1", UNKNOWN_CLUSTER}:
        return None
    return str(value)


def _fit_group(
    features: pd.DataFrame,
    *,
    algorithm: Algorithm,
    n_clusters: int,
    seed: int,
) -> Optional[_GroupModel]:
    if any(features[column].notna().sum() == 0 for column in features.columns):
        return None
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(features)
    seen = set()
    for row in values:
        seen.add(tuple(row))
        if len(seen) >= n_clusters:
            break
    if len(seen) < n_clusters:
        return None
    scaler = RobustScaler()
    scaled = scaler.fit_transform(values)
    estimator: Union[KMeans, GaussianMixture]
    if algorithm == "kmeans":
        estimator = KMeans(n_clusters=n_clusters, random_state=seed, n_init=15)
    else:
        estimator = GaussianMixture(n_components=n_clusters, random_state=seed, reg_covar=1e-6)
    estimator.fit(scaled)
    return _GroupModel(imputer=imputer, scaler=scaler, estimator=estimator)


def fit_point_in_time_clusters(
    profiles: pd.DataFrame,
    *,
    kind: Literal["dao", "voter"],
    scope: Optional[VoterScope],
    feature_columns: Sequence[str],
    history_column: str,
    fit_cutoff: object,
    train_cutoff: object,
    split_fingerprint: str,
    n_clusters: int = 2,
    min_group_size: int = 5,
    min_history: int = 1,
    min_cluster_size: int = 1,
    algorithm: Algorithm = "kmeans",
    seed: int = 42,
    dao_bundle: Optional[PointInTimeClusterBundle] = None,
) -> PointInTimeClusterBundle:
    if kind not in {"dao", "voter"}:
        raise ValueError("Cluster kind must be 'dao' or 'voter'.")
    if kind == "dao" and scope is not None:
        raise ValueError("DAO clustering does not use a voter-clustering scope.")
    if kind == "voter" and scope not in {"global", "dao_cluster", "individual_dao"}:
        raise ValueError("Choose exactly one voter-clustering scope.")
    if algorithm not in {"kmeans", "gaussian_mixture"}:
        raise ValueError("Unsupported clustering algorithm.")
    columns = tuple(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Cluster feature columns must be nonempty and unique.")
    if (set(columns) | {history_column}) & _FORBIDDEN_FEATURES:
        raise ValueError("Unverified current-vote or cluster-ID columns cannot be cluster features.")
    if n_clusters < 2 or min_group_size < n_clusters or min_history < 1 or min_cluster_size < 1:
        raise ValueError("Cluster counts and minimum group/history sizes are invalid.")
    if not re.fullmatch(r"[0-9a-f]{64}", split_fingerprint):
        raise ValueError("A SHA-256 split fingerprint is required.")
    fit_time, train_time = _time(fit_cutoff, "Fit cutoff"), _time(train_cutoff, "Training cutoff")
    if fit_time > train_time:
        raise ValueError("Cluster fitting must finish no later than the training cutoff.")

    if scope == "dao_cluster":
        if dao_bundle is None or dao_bundle.kind != "dao":
            raise ValueError("DAO-cluster voter scope requires a train-fitted DAO cluster bundle.")
        if dao_bundle.split_fingerprint != split_fingerprint or dao_bundle.train_cutoff != train_time:
            raise ValueError("DAO and voter cluster bundles must use the same training split.")
        if dao_bundle.fit_cutoff >= fit_time:
            raise ValueError("DAO clustering must be fitted before DAO-cluster voter clustering.")
    elif dao_bundle is not None:
        raise ValueError("A DAO cluster bundle is only used by DAO-cluster voter scope.")

    profiles_for_grouping = _attach_dao_groups(profiles, scope, dao_bundle)
    frame = _prepare_profiles(
        profiles_for_grouping, kind=kind, scope=scope, feature_columns=columns, history_column=history_column
    )
    if (frame["asof_ts"] > fit_time).any():
        raise ValueError("Cluster fitting profiles include information after the fit cutoff.")
    keys = ["space"] if kind == "dao" else ["voter", "space"]
    if frame.duplicated(keys).any():
        raise ValueError("Cluster fitting requires one as-of profile per entity.")
    groups: Dict[str, list[int]] = {}
    summary = {
        "profiles": len(frame), "insufficient_history": 0, "unknown_groups": 0,
        "fitted_groups": 0, "skipped_groups": 0, "fitted_profiles": 0,
    }
    for index, row in frame.iterrows():
        if pd.isna(row[history_column]) or row[history_column] < min_history:
            summary["insufficient_history"] += 1
            continue
        group = _group_key(row, kind, scope)
        if group is None:
            summary["unknown_groups"] += 1
            continue
        groups.setdefault(group, []).append(index)
    models: Dict[str, _GroupModel] = {}
    skipped: Dict[str, str] = {}
    for group, indices in groups.items():
        if len(indices) < min_group_size:
            skipped[group] = "insufficient_group_size"
            continue
        model = _fit_group(
            frame.loc[indices, list(columns)], algorithm=algorithm, n_clusters=n_clusters, seed=seed
        )
        if model is None:
            skipped[group] = "missing_or_indistinct_features"
        else:
            labels = model.predict(frame.loc[indices, list(columns)])
            sizes = np.bincount(labels, minlength=n_clusters)
            if (sizes < min_cluster_size).any():
                skipped[group] = "insufficient_cluster_size"
            else:
                models[group] = model
                summary["fitted_profiles"] += len(indices)
    summary["fitted_groups"] = len(models)
    summary["skipped_groups"] = len(skipped)
    if not models:
        raise ValueError("No training groups have sufficient history and distinct profiles for clustering.")
    return PointInTimeClusterBundle(
        kind=kind,
        scope=scope,
        feature_columns=columns,
        history_column=history_column,
        min_history=min_history,
        fit_cutoff=fit_time,
        train_cutoff=train_time,
        split_fingerprint=split_fingerprint,
        algorithm=algorithm,
        min_cluster_size=min_cluster_size,
        models=models,
        skipped_groups=skipped,
        fit_summary=summary,
        dao_bundle=dao_bundle,
    )


def diagnose_cluster_fit(
    profiles: pd.DataFrame,
    bundle: PointInTimeClusterBundle,
    *,
    max_samples: int = 500,
    seed: int = 42,
) -> Dict[str, object]:
    if max_samples < 3:
        raise ValueError("Cluster diagnostics require a sample limit of at least three.")
    grouped = _attach_dao_groups(profiles, bundle.scope, bundle.dao_bundle)
    frame = _prepare_profiles(
        grouped, kind=bundle.kind, scope=bundle.scope,
        feature_columns=bundle.feature_columns, history_column=bundle.history_column,
    )
    if (frame["asof_ts"] > bundle.fit_cutoff).any():
        raise ValueError("Cluster diagnostics may only use fit-time training profiles.")
    indices_by_group: Dict[str, list[int]] = {}
    for index, row in frame.iterrows():
        if pd.isna(row[bundle.history_column]) or row[bundle.history_column] < bundle.min_history:
            continue
        group = _group_key(row, bundle.kind, bundle.scope)
        if group in bundle.models:
            indices_by_group.setdefault(group, []).append(index)
    diagnostics: Dict[str, object] = {}
    rng = np.random.default_rng(seed)
    for group, model in bundle.models.items():
        indices = indices_by_group.get(group, [])
        if not indices:
            raise ValueError("Cluster diagnostics are missing fitted training groups.")
        values = model.imputer.transform(frame.loc[indices, list(bundle.feature_columns)])
        scaled = model.scaler.transform(values)
        labels = model.estimator.predict(scaled)
        n_clusters = model.estimator.n_clusters if isinstance(model.estimator, KMeans) else model.estimator.n_components
        sizes = np.bincount(labels, minlength=n_clusters).tolist()
        selected = rng.choice(len(indices), size=min(max_samples, len(indices)), replace=False)
        sample, sample_labels = scaled[selected], labels[selected]
        unique_labels = np.unique(sample_labels)
        silhouette = None
        stability = None
        if 1 < len(unique_labels) < len(sample_labels):
            silhouette = float(silhouette_score(sample, sample_labels))
        if len(sample) >= n_clusters and len(np.unique(sample, axis=0)) >= n_clusters:
            if bundle.algorithm == "kmeans":
                alternate = KMeans(n_clusters=n_clusters, random_state=seed + 1, n_init=15)
            else:
                alternate = GaussianMixture(n_components=n_clusters, random_state=seed + 1, reg_covar=1e-6)
            stability = float(adjusted_rand_score(sample_labels, alternate.fit_predict(sample)))
        centers = model.estimator.cluster_centers_ if isinstance(model.estimator, KMeans) else model.estimator.means_
        diagnostics[group] = {
            "training_profiles": len(indices),
            "cluster_sizes": sizes,
            "sampled_profiles": len(selected),
            "silhouette": silhouette,
            "seed_stability_adjusted_rand": stability,
            "centers_original_scale": model.scaler.inverse_transform(centers).tolist(),
        }
    return {"feature_columns": list(bundle.feature_columns), "groups": diagnostics, "sample_limit": max_samples}
