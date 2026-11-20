"""Assemble matched chronological examples from verified vote events."""
from __future__ import annotations

import re
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

from dao_governance.clustering.predictive import load_verified_cluster_bundle
from dao_governance.data.canonical_dataset import load_proposal_catalog
from dao_governance.data.chronological_split import validate_chronological_manifest, validate_proposal_metadata
from dao_governance.data.eligibility import EligibleProposal
from dao_governance.features.asof import ObservationProvenance, VoteEvent, iter_asof_features
from dao_governance.features.point_in_time_clusters import PointInTimeClusterBundle
from dao_governance.modelling.point_in_time_model import validate_target_examples
from dao_governance.modelling.run_artifacts import sha256_file


def build_example_partitions(
    events: pd.DataFrame,
    proposals: Mapping[Tuple[str, str], EligibleProposal],
    manifest: Mapping[str, object],
    *,
    split_fingerprint: str,
    availability_provenance: ObservationProvenance,
    include_target_proposal: bool = False,
    dao_bundle: Optional[PointInTimeClusterBundle] = None,
    voter_bundle: Optional[PointInTimeClusterBundle] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    validate_chronological_manifest(manifest)
    if not re.fullmatch(r"[0-9a-f]{64}", split_fingerprint):
        raise ValueError("A SHA-256 split fingerprint is required.")
    if not isinstance(availability_provenance, ObservationProvenance):
        raise ValueError("Verified vote observation-time provenance is required.")
    availability_provenance.validate()
    if availability_provenance.local_proxy and manifest.get("provenance_mode") != "local_proxy":
        raise ValueError("A local vote-time proxy requires a local-proxy split manifest.")
    required = {"example_id", "voter", "space", "proposal_id", "vote_ts", "observed_at", "label_id"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Vote events are missing columns: {sorted(missing)}")
    if events.empty:
        raise ValueError("Vote events cannot be empty.")
    frame = events[list(sorted(required))].copy()
    if frame["example_id"].isna().any() or frame["example_id"].duplicated().any():
        raise ValueError("Vote events require unique example identifiers.")
    if frame.duplicated(["voter", "space", "proposal_id"]).any():
        raise ValueError("Repeated voter/proposal records require an explicit revision policy.")
    for column in ("vote_ts", "observed_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"Vote events require valid {column} timestamps.")
    if not frame["label_id"].isin([0, 1, 2]).all():
        raise ValueError("Vote events require canonical three-class labels.")

    membership = {
        (row["space"], row["proposal_id"]): row
        for row in manifest["proposals"]
    }
    vote_events = []
    validated_proposals = set()
    for row in frame.itertuples(index=False):
        key = (row.space, row.proposal_id)
        proposal = proposals.get(key)
        if proposal is None or key not in membership:
            raise ValueError("A vote event lacks an eligible proposal or split membership.")
        split_proposal = membership[key]
        if key not in validated_proposals:
            validate_proposal_metadata(proposal, split_proposal)
            validated_proposals.add(key)
        if not proposal.start <= row.vote_ts < proposal.end:
            raise ValueError("A vote event is outside its proposal window.")
        vote_events.append(
            VoteEvent(
                example_id=row.example_id,
                voter=row.voter,
                space=row.space,
                proposal_id=row.proposal_id,
                vote_ts=row.vote_ts,
                observed_at=row.observed_at,
                proposal_end=proposal.end,
                label_id=int(row.label_id),
            )
        )
    targets = sorted(vote_events, key=lambda event: event.vote_ts)
    observations = sorted(vote_events, key=lambda event: event.observed_at)
    features = pd.DataFrame(
        iter_asof_features(
            targets,
            observations,
            availability_provenance=availability_provenance,
            include_target_proposal=include_target_proposal,
        )
    )
    if len(features) != len(frame):
        raise RuntimeError("Point-in-time features do not cover every vote event.")
    labels = frame[["example_id", "label_id"]]
    examples = features.merge(labels, on="example_id", how="inner", validate="one_to_one")
    examples["vote_ts"] = examples["asof_ts"]

    cluster_counts: Counter[str] = Counter()
    for name, bundle in (("dao", dao_bundle), ("voter", voter_bundle)):
        if bundle is None:
            continue
        if (
            bundle.kind != name
            or bundle.split_fingerprint != split_fingerprint
            or bundle.train_cutoff != pd.to_datetime(manifest["train_cutoff"], utc=True)
        ):
            raise ValueError("Cluster bundle kind or split fingerprint does not match the examples.")
        cluster_inputs = examples
        if name == "voter" and bundle.scope == "dao_cluster" and "dao_cluster" in examples.columns:
            if bundle.dao_bundle is None:
                raise ValueError("DAO-cluster voter scope is missing its fitted DAO bundle.")
            cluster_inputs = examples.drop(columns="dao_cluster")
            grouped = bundle.dao_bundle.assign(cluster_inputs)["cluster"].to_numpy()
            if not (grouped == examples["dao_cluster"].to_numpy()).all():
                raise ValueError("DAO feature and voter-group assignments use different DAO clusterers.")
        assignments = bundle.assign(cluster_inputs)
        examples[f"{name}_cluster"] = assignments["cluster"].to_numpy()
        for status, count in assignments["cluster_status"].value_counts().items():
            cluster_counts[f"{name}_{status}"] += int(count)
    examples["split"] = [
        membership[(space, proposal_id)]["split"]
        for space, proposal_id in zip(examples["space"], examples["proposal_id"])
    ]
    partitions = {
        split: examples.loc[examples["split"] == split].drop(columns="split").reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    for split, partition in partitions.items():
        validate_target_examples(partition, manifest, split)
    cluster_counts["purged_events"] = int((examples["split"] == "purged").sum())
    return partitions, dict(cluster_counts)


def prepare_example_artifacts(
    *,
    raw_source: Path,
    metadata_cache: Path,
    historical_provenance: Optional[Path],
    canonical_events: Path,
    observations: Optional[Path],
    observation_provenance: Optional[Path],
    split_manifest: Path,
    output_dir: Path,
    include_target_proposal: bool = False,
    dao_bundle_path: Optional[Path] = None,
    voter_bundle_path: Optional[Path] = None,
    max_events: int = 2_000_000,
    local_proxy: bool = False,
) -> Dict[str, object]:
    if max_events < 1:
        raise ValueError("The maximum in-memory event count must be positive.")
    if output_dir.exists():
        raise FileExistsError("Example artifact directory already exists.")
    if local_proxy and (historical_provenance is not None or observations is not None or observation_provenance is not None):
        raise ValueError("Local-proxy mode derives its observation proxy and does not accept verified provenance files.")
    if not local_proxy and (historical_provenance is None or observations is None or observation_provenance is None):
        raise ValueError("Verified mode requires historical provenance and observation files.")
    catalog, fingerprint, _ = load_proposal_catalog(
        raw_source, metadata_cache, historical_provenance, local_proxy=local_proxy
    )
    audit_path = canonical_events.with_name(canonical_events.name + ".audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    events_hash = sha256_file(canonical_events)
    if (
        audit.get("dataset_sha256") != fingerprint
        or audit.get("canonical_events_sha256") != events_hash
        or audit.get("provenance_mode") != ("local_proxy" if local_proxy else "verified")
    ):
        raise ValueError("Canonical vote events do not match their source and audit.")
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    validate_chronological_manifest(manifest)
    if manifest["dataset_sha256"] != fingerprint:
        raise ValueError("The split manifest does not match the canonical vote dataset.")
    if manifest.get("provenance_mode", "verified") != ("local_proxy" if local_proxy else "verified"):
        raise ValueError("The split provenance mode does not match the requested example mode.")
    split_hash = sha256_file(split_manifest)
    observations_hash = None
    observation_provenance_hash = None
    if local_proxy:
        provenance = ObservationProvenance(
            False, "Recorded vote time is assumed to equal observation time; this is unverified.", local_proxy=True
        )
    else:
        if observation_provenance is None or observations is None:
            raise RuntimeError("Verified observation inputs are missing.")
        observation = json.loads(observation_provenance.read_text(encoding="utf-8"))
        observations_hash = sha256_file(observations)
        observation_provenance_hash = sha256_file(observation_provenance)
        if (
            observation.get("version") != 1
            or observation.get("verified") is not True
            or not isinstance(observation.get("evidence"), str)
            or not observation["evidence"].strip()
            or observation.get("canonical_events_sha256") != events_hash
            or observation.get("observations_sha256") != observations_hash
        ):
            raise ValueError("Verified observation-time evidence must match the vote and observation files.")
        provenance = ObservationProvenance(True, observation["evidence"])
    event_count = pq.ParquetFile(canonical_events).metadata.num_rows
    if event_count > max_events:
        raise ValueError("Canonical event count exceeds the configured in-memory preparation limit.")
    events = pd.read_parquet(canonical_events)
    if local_proxy:
        events["observed_at"] = events["vote_ts"]
    else:
        if observations is None:
            raise RuntimeError("Verified observation data are missing.")
        if pq.ParquetFile(observations).metadata.num_rows != event_count:
            raise ValueError("Observation data must cover exactly the canonical vote events.")
        observed = pd.read_parquet(observations)
        if sha256_file(observations) != observations_hash:
            raise RuntimeError("Observation data changed during example preparation.")
        if not {"example_id", "observed_at"}.issubset(observed.columns):
            raise ValueError("Observation data require example_id and observed_at columns.")
        if observed["example_id"].isna().any() or observed["example_id"].duplicated().any():
            raise ValueError("Observation data require unique example identifiers.")
        if set(events["example_id"]) != set(observed["example_id"]):
            raise ValueError("Observation data must cover exactly the canonical vote events.")
        events = events.merge(observed[["example_id", "observed_at"]], on="example_id", validate="one_to_one")
    if sha256_file(canonical_events) != events_hash:
        raise RuntimeError("Vote data changed during example preparation.")

    dao_bundle = None
    voter_bundle = None
    if dao_bundle_path is not None:
        dao_bundle, _ = load_verified_cluster_bundle(dao_bundle_path, kind="dao", split_fingerprint=split_hash)
    if voter_bundle_path is not None:
        voter_bundle, _ = load_verified_cluster_bundle(voter_bundle_path, kind="voter", split_fingerprint=split_hash)
    partitions, cluster_counts = build_example_partitions(
        events,
        catalog.eligible,
        manifest,
        split_fingerprint=split_hash,
        availability_provenance=provenance,
        include_target_proposal=include_target_proposal,
        dao_bundle=dao_bundle,
        voter_bundle=voter_bundle,
    )
    if any(partitions[name].empty for name in ("train", "validation", "test")):
        raise ValueError("Chronological example partitions must all be nonempty.")
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    split_copy = output_dir / "split_manifest.json"
    with split_manifest.open("rb") as source, split_copy.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    if sha256_file(split_copy) != split_hash:
        raise RuntimeError("The split manifest changed during example preparation.")
    files = {}
    for split, frame in partitions.items():
        path = output_dir / f"{split}.parquet"
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            frame.to_parquet(stream, index=False)
        files[split] = {"path": path.name, "sha256": sha256_file(path), "rows": len(frame)}
    result: Dict[str, object] = {
        "version": 1,
        "dataset_sha256": fingerprint,
        "provenance_mode": "local_proxy" if local_proxy else "verified",
        "split_sha256": split_hash,
        "canonical_events_sha256": events_hash,
        "observations_sha256": observations_hash,
        "observation_provenance_sha256": observation_provenance_hash,
        "observation_provenance": {
            "verified": provenance.verified, "local_proxy": provenance.local_proxy, "evidence": provenance.evidence,
        },
        "history_mode": "target_proposal_sensitivity" if include_target_proposal else "other_proposals_only",
        "dao_bundle_sha256": sha256_file(dao_bundle_path) if dao_bundle_path is not None else None,
        "voter_bundle_sha256": sha256_file(voter_bundle_path) if voter_bundle_path is not None else None,
        "voter_cluster_scope": voter_bundle.scope if voter_bundle is not None else None,
        "cluster_counts": cluster_counts,
        "partitions": files,
    }
    manifest_path = output_dir / "examples_manifest.json"
    descriptor = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result
