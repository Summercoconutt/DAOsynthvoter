"""Build a provenance-gated canonical vote dataset from a raw Parquet export."""
from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dao_governance.data.eligibility import ProposalCatalog, build_proposal_catalog, canonicalize_vote_partition
from dao_governance.data.chronological_split import build_chronological_manifest, write_chronological_manifest
from dao_governance.modelling.run_artifacts import sha256_file, sha256_json


def load_proposal_catalog(
    raw_source: Path,
    metadata_cache: Path,
    provenance_file: Optional[Path],
    *,
    local_proxy: bool = False,
) -> Tuple[ProposalCatalog, str, Dict[str, int]]:
    metadata_hash = sha256_file(metadata_cache)
    raw_hash = sha256_file(raw_source)
    evidence: Dict[Tuple[str, str], Dict[str, Any]] = {}
    provenance_hash = None
    binding_hash = None
    if local_proxy:
        if provenance_file is not None:
            raise ValueError("Local-proxy mode does not accept a verified provenance file.")
        binding_path = metadata_cache.with_name(metadata_cache.name + ".binding.json")
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if (
            binding.get("version") != 1
            or binding.get("local_votes_sha256") != raw_hash
            or binding.get("bound_cache_sha256") != metadata_hash
            or binding.get("historical_availability") != "unverified"
        ):
            raise ValueError("The local metadata binding does not match the local vote and metadata files.")
        binding_hash = sha256_file(binding_path)
    else:
        if provenance_file is None:
            raise ValueError("Verified historical proposal provenance is required.")
        provenance_hash = sha256_file(provenance_file)
        provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
        if provenance.get("version") != 1 or provenance.get("metadata_sha256") != metadata_hash:
            raise ValueError("Historical provenance does not match the proposal metadata cache.")
        attestations = provenance.get("proposals")
        if not isinstance(attestations, list):
            raise ValueError("Historical provenance must contain proposal attestations.")
        for entry in attestations:
            if not isinstance(entry, dict):
                raise ValueError("Historical provenance entries must be objects.")
            space, proposal_id = entry.get("space"), entry.get("proposal_id")
            if not isinstance(space, str) or not space or not isinstance(proposal_id, str) or not proposal_id:
                raise ValueError("Historical provenance proposal keys must be nonempty and unique.")
            key = (space, proposal_id)
            if key in evidence:
                raise ValueError("Historical provenance proposal keys must be nonempty and unique.")
            if entry.get("verified") is not True or not isinstance(entry.get("evidence"), str) or not entry["evidence"].strip():
                raise ValueError("Each historical provenance entry requires verified evidence.")
            if not isinstance(entry.get("metadata_sha256"), str) or not isinstance(entry.get("available_at"), (int, float)):
                raise ValueError("Historical provenance must bind metadata content and its availability time.")
            evidence[key] = {
                "verified": True,
                "evidence": entry["evidence"],
                "metadata_sha256": entry["metadata_sha256"],
                "available_at": entry["available_at"],
            }

    records = []
    seen = set()
    with metadata_cache.open(encoding="utf-8") as stream:
        first = stream.readline()
        if not first:
            raise ValueError("Proposal metadata cache is empty.")
        header = json.loads(first)
        if header.get("kind") != "snapshot_proposal_metadata" or header.get("version") != 1:
            raise ValueError("Proposal metadata cache header is invalid.")
        if header.get("source_sha256") != raw_hash or header.get("source_size") != raw_source.stat().st_size:
            raise ValueError("Proposal metadata cache does not match the raw vote source.")
        if local_proxy and (
            header.get("binding") != "local_id_space_title_body_prefix"
            or header.get("historical_availability") != "unverified"
        ):
            raise ValueError("Local-proxy mode requires a locally reconciled, unverified metadata cache.")
        for line in stream:
            record = json.loads(line)
            key = (record.get("space"), record.get("proposal_id"))
            if key in seen:
                raise ValueError("Proposal metadata cache contains duplicate keys.")
            seen.add(key)
            if key in evidence:
                record["historical_provenance"] = evidence[key]
            records.append(record)
    if len(seen) != header.get("proposal_count") or set(evidence) - seen:
        raise ValueError("Proposal metadata cache is incomplete or provenance contains unknown keys.")
    catalog = build_proposal_catalog(records, allow_local_proxy=local_proxy)
    reasons: Counter[str] = Counter()
    for excluded in catalog.excluded.values():
        reasons.update(excluded)
    fingerprint = sha256_json(
        {
            "raw_votes_sha256": raw_hash,
            "proposal_metadata_sha256": metadata_hash,
            "historical_provenance_sha256": provenance_hash,
            "local_binding_sha256": binding_hash,
            "provenance_mode": "local_proxy" if local_proxy else "verified",
        }
    )
    return catalog, fingerprint, dict(reasons)


def load_verified_catalog(
    raw_source: Path, metadata_cache: Path, provenance_file: Path
) -> Tuple[ProposalCatalog, str, Dict[str, int]]:
    return load_proposal_catalog(raw_source, metadata_cache, provenance_file)


def _column(names: set[str], alternatives: tuple[str, ...], label: str) -> str:
    for name in alternatives:
        if name in names:
            return name
    raise ValueError(f"Raw vote source is missing {label}.")


def build_canonical_vote_dataset(
    raw_source: Path,
    metadata_cache: Path,
    provenance_file: Optional[Path],
    output: Path,
    *,
    batch_size: int = 65536,
    local_proxy: bool = False,
) -> Dict[str, Any]:
    if batch_size < 1:
        raise ValueError("Canonical vote batch size must be positive.")
    raw_source, metadata_cache, output = raw_source.resolve(), metadata_cache.resolve(), output.resolve()
    provenance_file = provenance_file.resolve() if provenance_file is not None else None
    if output in {raw_source, metadata_cache, provenance_file}:
        raise ValueError("Canonical output must not overwrite an input.")
    audit_path = output.with_name(output.name + ".audit.json")
    scratch = output.with_name(output.name + ".dedupe.sqlite")
    if output.exists() or audit_path.exists() or scratch.exists():
        raise FileExistsError("Canonical output, audit, or scratch file already exists.")
    catalog, fingerprint, proposal_reasons = load_proposal_catalog(
        raw_source, metadata_cache, provenance_file, local_proxy=local_proxy
    )
    if not catalog.eligible:
        raise ValueError("No proposals have usable metadata under the selected provenance policy.")

    parquet = pq.ParquetFile(raw_source)
    names = set(parquet.schema_arrow.names)
    space_col = _column(names, ("Space", "space"), "DAO space")
    id_col = _column(names, ("Proposal ID", "proposal_id"), "original proposal ID")
    voter_col = _column(names, ("Voter", "voter"), "voter")
    choice_col = _column(names, ("Choice", "choice"), "raw vote choice")
    time_col = _column(names, ("Vote Timestamp", "vote_timestamp", "timestamp"), "vote timestamp")
    columns = [space_col, id_col, voter_col, choice_col, time_col]
    id_type = parquet.schema_arrow.field(id_col).type
    eligible_ids = pa.array([key[1] for key in catalog.eligible], type=id_type)
    schema = pa.schema(
        [
            ("example_id", pa.string()), ("voter", pa.string()), ("space", pa.string()),
            ("proposal_id", pa.string()), ("vote_ts", pa.timestamp("us", tz="UTC")),
            ("label_id", pa.int8()), ("choice_norm", pa.string()),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(scratch, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    connection = None
    counts: Counter[str] = Counter()
    created_output = False
    created_audit = False
    complete = False
    proposals_with_votes = set()
    try:
        connection = sqlite3.connect(scratch)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("CREATE TABLE seen (space TEXT, proposal_id TEXT, voter TEXT, PRIMARY KEY (space, proposal_id, voter)) WITHOUT ROWID")
        output_fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        created_output = True
        with os.fdopen(output_fd, "wb") as stream, pq.ParquetWriter(stream, schema) as writer:
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
                table = pa.Table.from_batches([batch])
                counts["raw_votes"] += len(table)
                counts["excluded_missing_or_unmatched_metadata"] += table[id_col].null_count
                groups = table.group_by([space_col, id_col], use_threads=False).aggregate([(id_col, "count")])
                for group in groups.to_pylist():
                    key = (group[space_col], group[id_col])
                    if key not in catalog.eligible:
                        for reason in catalog.excluded.get(key, ("missing_or_unmatched_metadata",)):
                            counts[f"excluded_{reason}"] += int(group[f"{id_col}_count"])
                selected = table.filter(pc.is_in(table[id_col], value_set=eligible_ids))
                if not len(selected):
                    continue
                votes = selected.to_pandas().rename(
                    columns={
                        space_col: "space", id_col: "proposal_id", voter_col: "voter",
                        choice_col: "choice", time_col: "vote_timestamp",
                    }
                )
                keys = [
                    (row.space, row.proposal_id, str(row.voter).strip().lower())
                    for row in votes[["space", "proposal_id", "voter"]].itertuples(index=False)
                    if pd.notna(row.space) and pd.notna(row.proposal_id) and pd.notna(row.voter) and str(row.voter).strip()
                ]
                try:
                    connection.executemany("INSERT INTO seen (space, proposal_id, voter) VALUES (?, ?, ?)", keys)
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Repeated voter/proposal records require an explicit revision policy.") from exc
                canonical, batch_counts = canonicalize_vote_partition(votes, catalog, source_fingerprint=fingerprint)
                counts.update(batch_counts)
                if not canonical.empty:
                    proposals_with_votes.update(zip(canonical["space"], canonical["proposal_id"]))
                    writer.write_table(pa.Table.from_pandas(canonical, schema=schema, preserve_index=False))
        if not counts["eligible_votes"]:
            raise ValueError("No eligible vote events remain after canonicalization.")
        audit = {
            "version": 1,
            "dataset_sha256": fingerprint,
            "canonical_events_sha256": sha256_file(output),
            "raw_votes_sha256": sha256_file(raw_source),
            "proposal_metadata_sha256": sha256_file(metadata_cache),
            "historical_provenance_sha256": sha256_file(provenance_file) if provenance_file is not None else None,
            "provenance_mode": "local_proxy" if local_proxy else "verified",
            "historical_provenance_verified": not local_proxy,
            "eligible_proposals": len(catalog.eligible),
            "eligible_proposals_with_votes": len(proposals_with_votes),
            "proposal_exclusions_by_reason": proposal_reasons,
            "vote_counts_and_exclusions": dict(counts),
            "exclusion_reasons_may_overlap": True,
            "observation_times_verified": False,
        }
        with audit_path.open("x", encoding="utf-8") as stream:
            created_audit = True
            json.dump(audit, stream, indent=2, sort_keys=True)
            stream.write("\n")
        complete = True
        return audit
    finally:
        if connection is not None:
            connection.close()
        scratch.unlink(missing_ok=True)
        if not complete:
            if created_output:
                output.unlink(missing_ok=True)
            if created_audit:
                audit_path.unlink(missing_ok=True)


def create_split_from_canonical(
    raw_source: Path,
    metadata_cache: Path,
    provenance_file: Optional[Path],
    canonical_events: Path,
    output: Path,
    *,
    train_cutoff: Any = None,
    test_cutoff: Any = None,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    local_proxy: bool = False,
) -> Dict[str, Any]:
    catalog, fingerprint, _ = load_proposal_catalog(
        raw_source, metadata_cache, provenance_file, local_proxy=local_proxy
    )
    audit_path = canonical_events.with_name(canonical_events.name + ".audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("dataset_sha256") != fingerprint
        or audit.get("canonical_events_sha256") != sha256_file(canonical_events)
        or audit.get("provenance_mode") != ("local_proxy" if local_proxy else "verified")
    ):
        raise ValueError("Canonical vote events do not match their source and audit.")
    parquet = pq.ParquetFile(canonical_events)
    keys = set()
    for batch in parquet.iter_batches(batch_size=65536, columns=["space", "proposal_id"], use_threads=False):
        distinct = pa.Table.from_batches([batch]).group_by(["space", "proposal_id"], use_threads=False).aggregate([])
        keys.update((row["space"], row["proposal_id"]) for row in distinct.to_pylist())
    if not keys or keys - set(catalog.eligible):
        raise ValueError("Canonical vote events reference missing or unverified proposals.")
    proposals = [catalog.eligible[key] for key in sorted(keys)]
    manifest = build_chronological_manifest(
        proposals,
        dataset_fingerprint=fingerprint,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        train_cutoff=train_cutoff,
        test_cutoff=test_cutoff,
    )
    write_chronological_manifest(manifest, output)
    return manifest
