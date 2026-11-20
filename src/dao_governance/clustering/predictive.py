"""Persist and fit point-in-time clustering inputs for stages 6 and 7."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence

import pandas as pd

from dao_governance.data.chronological_split import validate_chronological_manifest
from dao_governance.features.asof import ObservationProvenance
from dao_governance.features.point_in_time_clusters import (
    Algorithm,
    PointInTimeClusterBundle,
    VoterScope,
    diagnose_cluster_fit,
    fit_point_in_time_clusters,
)
from dao_governance.modelling.run_artifacts import sha256_file


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _split(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_chronological_manifest(data)
    return data


def load_verified_cluster_bundle(
    path: Path,
    *,
    kind: Literal["dao", "voter"],
    split_fingerprint: str,
) -> tuple[PointInTimeClusterBundle, Dict[str, Any]]:
    metadata = json.loads(_sidecar(path).read_text(encoding="utf-8"))
    if (
        metadata.get("version") != 1
        or metadata.get("kind") != kind
        or metadata.get("split_sha256") != split_fingerprint
        or metadata.get("bundle_sha256") != sha256_file(path)
    ):
        raise RuntimeError("Cluster bundle provenance does not match the chronological split.")
    bundle = PointInTimeClusterBundle.load(path)
    if bundle.kind != kind or bundle.split_fingerprint != split_fingerprint:
        raise RuntimeError("Cluster bundle contents do not match their recorded provenance.")
    return bundle, metadata


def save_fit_profiles(
    profiles: pd.DataFrame,
    path: Path,
    *,
    kind: Literal["dao", "voter"],
    split_manifest: Path,
    fit_cutoff: object,
    availability_provenance: ObservationProvenance,
) -> None:
    if kind not in {"dao", "voter"}:
        raise ValueError("Profile kind must be 'dao' or 'voter'.")
    if not isinstance(availability_provenance, ObservationProvenance):
        raise ValueError("Verified vote observation-time provenance is required.")
    availability_provenance.validate()
    split = _split(split_manifest)
    if availability_provenance.local_proxy and split.get("provenance_mode") != "local_proxy":
        raise ValueError("A local vote-time proxy requires a local-proxy split manifest.")
    cutoff = pd.to_datetime(fit_cutoff, utc=True, errors="coerce")
    train_cutoff = pd.to_datetime(split["train_cutoff"], utc=True, errors="coerce")
    if pd.isna(cutoff) or pd.isna(train_cutoff) or cutoff > train_cutoff:
        raise ValueError("The profile fit cutoff must not be later than the training cutoff.")
    if "asof_ts" not in profiles.columns:
        raise ValueError("Cluster fit profiles require as-of timestamps.")
    if "space" not in profiles.columns:
        raise ValueError("Cluster fit profiles require DAO spaces.")
    train_spaces = {row["space"] for row in split["proposals"] if row["split"] == "train"}
    if profiles["space"].isna().any() or not set(profiles["space"]).issubset(train_spaces):
        raise ValueError("Cluster fit profiles must belong to training DAO spaces.")
    timestamps = pd.to_datetime(profiles["asof_ts"], utc=True, errors="coerce")
    if timestamps.isna().any() or (timestamps > cutoff).any():
        raise ValueError("Cluster fit profiles include invalid or future timestamps.")
    sidecar = _sidecar(path)
    if path.exists() or sidecar.exists():
        raise FileExistsError("Cluster profile output or its metadata already exists.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        profiles.to_parquet(stream, index=False)
    metadata = {
        "version": 1,
        "kind": kind,
        "dataset_sha256": split["dataset_sha256"],
        "split_sha256": sha256_file(split_manifest),
        "profiles_sha256": sha256_file(path),
        "fit_cutoff": cutoff.isoformat(),
        "availability_provenance": {
            "verified": availability_provenance.verified,
            "local_proxy": availability_provenance.local_proxy,
            "evidence": availability_provenance.evidence,
        },
    }
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")


def fit_predictive_stage(
    *,
    profiles_path: Path,
    split_manifest: Path,
    output: Path,
    kind: Literal["dao", "voter"],
    scope: Optional[VoterScope],
    feature_columns: Sequence[str],
    history_column: str,
    n_clusters: int,
    min_group_size: int,
    min_history: int,
    algorithm: Algorithm,
    dao_bundle_path: Optional[Path] = None,
    min_cluster_size: int = 1,
) -> Dict[str, Any]:
    split = _split(split_manifest)
    split_hash = sha256_file(split_manifest)
    profile_meta = json.loads(_sidecar(profiles_path).read_text(encoding="utf-8"))
    availability = profile_meta.get("availability_provenance")
    if (
        profile_meta.get("version") != 1
        or profile_meta.get("kind") != kind
        or profile_meta.get("dataset_sha256") != split["dataset_sha256"]
        or profile_meta.get("split_sha256") != split_hash
        or profile_meta.get("profiles_sha256") != sha256_file(profiles_path)
        or not isinstance(availability, dict)
        or not (
            (availability.get("verified") is True and not availability.get("local_proxy", False))
            or (availability.get("verified") is False and availability.get("local_proxy") is True
                and split.get("provenance_mode") == "local_proxy")
        )
        or not isinstance(availability.get("evidence"), str)
        or not availability["evidence"].strip()
        or not profile_meta.get("fit_cutoff")
    ):
        raise RuntimeError("Cluster profile provenance does not match the chronological split.")

    dao_bundle = None
    if scope == "dao_cluster":
        if dao_bundle_path is None:
            raise ValueError("DAO-cluster voter scope requires a persisted DAO bundle.")
        dao_bundle, _ = load_verified_cluster_bundle(dao_bundle_path, kind="dao", split_fingerprint=split_hash)
    elif dao_bundle_path is not None:
        raise ValueError("A DAO bundle is only used for DAO-cluster voter scope.")

    if output.exists() or _sidecar(output).exists():
        raise FileExistsError("Cluster bundle output or its metadata already exists.")
    profiles = pd.read_parquet(profiles_path)
    bundle = fit_point_in_time_clusters(
        profiles,
        kind=kind,
        scope=scope,
        feature_columns=feature_columns,
        history_column=history_column,
        fit_cutoff=profile_meta["fit_cutoff"],
        train_cutoff=split["train_cutoff"],
        split_fingerprint=split_hash,
        n_clusters=n_clusters,
        min_group_size=min_group_size,
        min_history=min_history,
        min_cluster_size=min_cluster_size,
        algorithm=algorithm,
        dao_bundle=dao_bundle,
    )
    bundle.save(output)
    metadata = {
        "version": 1,
        "kind": kind,
        "scope": scope,
        "split_sha256": split_hash,
        "profiles_sha256": profile_meta["profiles_sha256"],
        "bundle_sha256": sha256_file(output),
        "fit_cutoff": bundle.fit_cutoff.isoformat(),
        "algorithm": algorithm,
        "features": list(feature_columns),
        "fit_summary": bundle.fit_summary,
        "min_cluster_size": min_cluster_size,
        "diagnostics": diagnose_cluster_fit(profiles, bundle),
        "dao_bundle_sha256": sha256_file(dao_bundle_path) if dao_bundle_path is not None else None,
    }
    with _sidecar(output).open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return metadata
