"""Recover public Snapshot proposal metadata without requesting vote records."""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from dao_governance.modelling.run_artifacts import sha256_file

SNAPSHOT_URL = "https://hub.snapshot.org/graphql"
PUBLIC_PROPOSAL_ID = re.compile(
    r"(?:0x[0-9a-fA-F]{64}|Qm[1-9A-HJ-NP-Za-km-z]{44}|b[a-z2-7]{20,100})"
)
PROPOSAL_QUERY = """
query ProposalMetadata($ids: [String], $n: Int!) {
  proposals(first: $n, skip: 0, where: {id_in: $ids}) {
    id title body choices type start end created updated ipfs votes
    space { id }
  }
}
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_proposal_keys(source: Path, *, limit: int = 0) -> Dict[str, str]:
    """Read distinct public proposal IDs and spaces in bounded batches."""
    if limit < 0:
        raise ValueError("Proposal limit cannot be negative.")
    parquet = pq.ParquetFile(source)
    names = set(parquet.schema_arrow.names)
    id_col = "Proposal ID" if "Proposal ID" in names else "proposal_id"
    space_col = "Space" if "Space" in names else "space"
    missing = {id_col, space_col} - names
    if missing:
        raise ValueError(f"Vote source is missing proposal identity columns: {sorted(missing)}")

    keys: Dict[str, str] = {}
    for batch in parquet.iter_batches(
        batch_size=65536, columns=[id_col, space_col], use_threads=False
    ):
        distinct = (
            pa.Table.from_batches([batch])
            .group_by([id_col, space_col], use_threads=False)
            .aggregate([])
            .to_pylist()
        )
        for row in distinct:
            proposal_id, space = row[id_col], row[space_col]
            if not isinstance(proposal_id, str) or not PUBLIC_PROPOSAL_ID.fullmatch(proposal_id):
                raise ValueError(
                    "Vote source contains missing or unsupported public proposal IDs; "
                    "do not infer IDs from filenames."
                )
            if not isinstance(space, str) or not space.strip():
                raise ValueError("Vote source contains a missing proposal space.")
            if proposal_id in keys and keys[proposal_id] != space:
                raise ValueError("A proposal ID is associated with more than one space.")
            keys[proposal_id] = space
        if limit and len(keys) >= limit:
            break
    items = sorted(keys.items())
    return dict(items[:limit] if limit else items)


class SnapshotProposalClient:
    def __init__(
        self,
        *,
        session: Optional[requests.Session] = None,
        min_interval: float = 0.75,
        timeout: float = 30.0,
        max_attempts: int = 4,
    ) -> None:
        if min_interval < 0.6:
            raise ValueError("Snapshot requests must be limited to at most 100 per minute.")
        if timeout <= 0 or max_attempts < 1:
            raise ValueError("Timeout and maximum attempts must be positive.")
        self.session = session or requests.Session()
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._last_request: Optional[float] = None

    def fetch(self, proposal_ids: Sequence[str]) -> List[Dict[str, Any]]:
        ids = list(proposal_ids)
        if not ids or len(ids) > 100:
            raise ValueError("A metadata request must contain between 1 and 100 proposal IDs.")
        if any(not PUBLIC_PROPOSAL_ID.fullmatch(pid) for pid in ids):
            raise ValueError("Only validated public Snapshot proposal IDs may be requested.")

        payload = {"query": PROPOSAL_QUERY, "variables": {"ids": ids, "n": len(ids)}}
        response: Optional[requests.Response] = None
        for attempt in range(self.max_attempts):
            if self._last_request is not None:
                delay = self.min_interval - (time.monotonic() - self._last_request)
                if delay > 0:
                    time.sleep(delay)
            self._last_request = time.monotonic()
            try:
                response = self.session.post(SNAPSHOT_URL, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt + 1 == self.max_attempts:
                    raise RuntimeError("Snapshot metadata request failed after retries.") from exc
                time.sleep(2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 == self.max_attempts:
                    raise RuntimeError(
                        f"Snapshot metadata request failed after retries: HTTP {response.status_code}."
                    )
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            break

        if response is None:
            raise RuntimeError("Snapshot metadata request did not produce a response.")
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise RuntimeError("Snapshot returned invalid JSON.") from exc
        if not isinstance(payload, dict) or payload.get("errors"):
            raise RuntimeError("Snapshot returned GraphQL errors for the metadata request.")
        data = payload.get("data")
        proposals = data.get("proposals") if isinstance(data, dict) else None
        if not isinstance(proposals, list) or not all(isinstance(item, dict) for item in proposals):
            raise RuntimeError("Snapshot returned an invalid proposal metadata response.")
        return proposals


def _existing_records(
    path: Path, header: Mapping[str, Any], keys: Mapping[str, str]
) -> tuple[set[str], Counter[str]]:
    completed: set[str] = set()
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as stream:
        first = stream.readline()
        if not first or json.loads(first) != dict(header):
            raise RuntimeError("Metadata cache does not match the current source and collection settings.")
        for line in stream:
            record = json.loads(line)
            proposal_id = record.get("proposal_id")
            if proposal_id not in keys or record.get("space") != keys[proposal_id]:
                raise RuntimeError("Metadata cache contains an unexpected proposal or space.")
            if proposal_id in completed:
                raise RuntimeError("Metadata cache contains duplicate proposal records.")
            if record.get("status") not in {"retrieved", "missing", "space_mismatch"}:
                raise RuntimeError("Metadata cache contains an invalid collection status.")
            completed.add(proposal_id)
            counts[record["status"]] += 1
    return completed, counts


def collect_proposal_metadata(
    source: Path,
    output: Path,
    *,
    limit: int = 0,
    batch_size: int = 100,
    resume: bool = False,
    client: Optional[SnapshotProposalClient] = None,
) -> Dict[str, int]:
    """Write a resumable metadata-only JSONL cache for a raw vote source."""
    if batch_size < 1 or batch_size > 100:
        raise ValueError("Metadata batch size must be between 1 and 100.")
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Metadata output must not overwrite the vote source.")
    keys = read_proposal_keys(source, limit=limit)
    if not keys:
        raise ValueError("Vote source contains no proposal IDs.")
    header = {
        "kind": "snapshot_proposal_metadata",
        "version": 1,
        "source_sha256": sha256_file(source),
        "source_size": source.stat().st_size,
        "limit": limit,
        "proposal_count": len(keys),
        "endpoint": SNAPSHOT_URL,
    }
    if output.exists() and not resume:
        raise FileExistsError(f"Metadata cache already exists: {output}; use --resume-metadata.")
    if resume and not output.is_file():
        raise FileNotFoundError(f"Metadata cache does not exist: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".lock")
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        completed, previous = _existing_records(output, header, keys) if resume else (set(), Counter())
        pending = [pid for pid in keys if pid not in completed]
        api = client or SnapshotProposalClient()
        counts = {
            "total": len(keys),
            "already_recorded": len(completed),
            "retrieved": previous["retrieved"],
            "missing": previous["missing"],
            "space_mismatch": previous["space_mismatch"],
        }
        with output.open("a" if resume else "x", encoding="utf-8") as stream:
            if not resume:
                stream.write(json.dumps(header, sort_keys=True) + "\n")
                stream.flush()
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                found: Dict[str, Dict[str, Any]] = {}
                for proposal in api.fetch(batch):
                    proposal_id = proposal.get("id")
                    if proposal_id not in batch or proposal_id in found:
                        raise RuntimeError("Snapshot returned an unexpected or duplicate proposal ID.")
                    found[proposal_id] = proposal
                retrieved_at = _utc_now()
                for proposal_id in batch:
                    proposal = found.get(proposal_id)
                    if proposal is None:
                        status = "missing"
                    elif not isinstance(proposal.get("space"), dict) or proposal["space"].get("id") != keys[proposal_id]:
                        status = "space_mismatch"
                    else:
                        status = "retrieved"
                    counts[status] += 1
                    record = {
                        "proposal_id": proposal_id,
                        "space": keys[proposal_id],
                        "status": status,
                        "retrieved_at": retrieved_at,
                        "metadata": proposal,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                if start == 0 or start + len(batch) == len(pending) or (start // batch_size + 1) % 10 == 0:
                    recorded = len(completed) + min(start + len(batch), len(pending))
                    print(f"[02] Proposal metadata recorded: {recorded}/{len(keys)}")
        return counts
    finally:
        lock.unlink()
