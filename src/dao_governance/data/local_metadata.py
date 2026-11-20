"""Bind an existing proposal metadata cache to a local vote export."""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from dao_governance.modelling.run_artifacts import sha256_file


def _under_data(path: Path, data_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(data_root.resolve())
    except ValueError as exc:
        raise ValueError("Local-only inputs and metadata outputs must remain under the data directory.") from exc
    return resolved


def _field(names: set[str], alternatives: tuple[str, ...], description: str) -> str:
    for name in alternatives:
        if name in names:
            return name
    raise ValueError(f"Local vote data are missing {description}.")


def bind_local_metadata_cache(
    votes: Path,
    source_cache: Path,
    output: Path,
    *,
    data_root: Path,
    batch_size: int = 65536,
) -> Dict[str, Any]:
    if batch_size < 1:
        raise ValueError("Local metadata batch size must be positive.")
    votes = _under_data(votes, data_root)
    source_cache = _under_data(source_cache, data_root)
    output = _under_data(output, data_root)
    audit_path = output.with_name(output.name + ".binding.json")
    if output in {votes, source_cache} or output.exists() or audit_path.exists():
        raise FileExistsError("Local metadata output or its audit already exists.")

    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with source_cache.open(encoding="utf-8") as stream:
        first = stream.readline()
        if not first:
            raise ValueError("The local metadata cache is empty.")
        header = json.loads(first)
        if header.get("kind") != "snapshot_proposal_metadata" or header.get("version") != 1:
            raise ValueError("The local metadata cache header is invalid.")
        for line in stream:
            record = json.loads(line)
            space, proposal_id = record.get("space"), record.get("proposal_id")
            if not isinstance(space, str) or not space or not isinstance(proposal_id, str) or not proposal_id:
                raise ValueError("Local metadata contains an invalid proposal key.")
            key = (space, proposal_id)
            if key in records:
                raise ValueError("Local metadata contains duplicate proposal keys.")
            if record.get("status") not in {"retrieved", "missing", "space_mismatch"}:
                raise ValueError("Local metadata contains an unsupported collection status.")
            records[key] = record
    if len(records) != header.get("proposal_count"):
        raise ValueError("The local metadata cache is incomplete.")

    parquet = pq.ParquetFile(votes)
    names = set(parquet.schema_arrow.names)
    space_col = _field(names, ("Space", "space"), "DAO spaces")
    id_col = _field(names, ("Proposal ID", "proposal_id"), "original proposal IDs")
    title_col = _field(names, ("Proposal Title", "proposal_title"), "proposal titles")
    body_col = _field(names, ("Proposal Body", "proposal_body"), "proposal bodies")
    columns = [space_col, id_col, title_col, body_col]
    states: Dict[Tuple[str, str], str] = {}
    rows_scanned = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
        table = pa.Table.from_batches([batch])
        rows_scanned += len(table)
        groups = table.group_by(columns, use_threads=False).aggregate([]).to_pylist()
        for row in groups:
            space, proposal_id = row[space_col], row[id_col]
            if not isinstance(space, str) or not space or not isinstance(proposal_id, str) or not proposal_id:
                raise ValueError("Local vote data contain a missing proposal key.")
            key = (space, proposal_id)
            record = records.get(key)
            if record is None or record.get("status") == "missing":
                status = "missing"
            elif record.get("status") != "retrieved":
                status = "space_mismatch"
            else:
                metadata = record.get("metadata")
                metadata_space = metadata.get("space") if isinstance(metadata, dict) else None
                title = metadata.get("title") if isinstance(metadata, dict) else None
                body = metadata.get("body") if isinstance(metadata, dict) else None
                if (
                    not isinstance(metadata, dict)
                    or metadata.get("id") != proposal_id
                    or not isinstance(metadata_space, dict)
                    or metadata_space.get("id") != space
                    or not isinstance(title, str)
                    or not isinstance(body, str)
                    or not isinstance(row[title_col], str)
                    or not isinstance(row[body_col], str)
                    or row[title_col] != title
                    or row[body_col] != body.replace("\n", " ").replace("\r", " ")[:800]
                ):
                    status = "local_text_mismatch"
                else:
                    status = "retrieved"
            previous = states.get(key)
            if previous is not None and previous != status:
                status = "local_text_mismatch"
            states[key] = status
    if not states:
        raise ValueError("Local vote data contain no proposal keys.")

    source_hash = sha256_file(votes)
    origin_hash = sha256_file(source_cache)
    local_header = {
        "kind": "snapshot_proposal_metadata",
        "version": 1,
        "source_sha256": source_hash,
        "source_size": votes.stat().st_size,
        "proposal_count": len(states),
        "origin_cache_sha256": origin_hash,
        "binding": "local_id_space_title_body_prefix",
        "historical_availability": "unverified",
    }
    counts = Counter(states.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    created_output = False
    created_audit = False
    complete = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".local-metadata-", suffix=".tmp",
            dir=output.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(local_header, sort_keys=True) + "\n")
            for key in sorted(states):
                source = records.get(key)
                if source is None:
                    record = {"space": key[0], "proposal_id": key[1], "status": "missing", "metadata": None}
                else:
                    record = dict(source)
                    record["status"] = states[key]
                record["local_binding"] = states[key]
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if temporary is None:
            raise RuntimeError("Local metadata temporary output was not created.")
        os.link(temporary, output)
        created_output = True
        audit = {
            "version": 1,
            "local_votes_sha256": source_hash,
            "origin_cache_sha256": origin_hash,
            "bound_cache_sha256": sha256_file(output),
            "rows_scanned": rows_scanned,
            "local_proposals": len(states),
            "statuses": dict(counts),
            "cache_only_proposals": len(set(records) - set(states)),
            "historical_availability": "unverified",
        }
        with audit_path.open("x", encoding="utf-8") as stream:
            created_audit = True
            json.dump(audit, stream, indent=2, sort_keys=True)
            stream.write("\n")
        complete = True
        return audit
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if not complete:
            if created_output:
                output.unlink(missing_ok=True)
            if created_audit:
                audit_path.unlink(missing_ok=True)
