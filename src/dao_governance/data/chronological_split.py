"""Proposal-disjoint chronological target splits."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from dao_governance.data.eligibility import EligibleProposal


def proposal_content_hash(title: str, body: str) -> str:
    def normalize(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

    content = json.dumps([normalize(title), normalize(body)], ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def proposal_provenance_hash(proposal: EligibleProposal) -> str:
    payload = {
        "evidence": proposal.historical_evidence,
        "available_at": proposal.metadata_available_at.isoformat() if proposal.metadata_available_at is not None else None,
        "mode": proposal.provenance_mode,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("Split cutoffs must be valid timestamps.")
    return timestamp


def validate_chronological_manifest(manifest: Mapping[str, Any]) -> None:
    fingerprint = manifest.get("dataset_sha256")
    proposals = manifest.get("proposals")
    if (
        manifest.get("version") != 1
        or not isinstance(fingerprint, str)
        or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        or not isinstance(proposals, list)
        or not proposals
    ):
        raise ValueError("A versioned chronological split manifest is required.")
    manifest_mode = manifest.get("provenance_mode", "verified")
    if manifest_mode not in {"verified", "local_proxy"}:
        raise ValueError("Chronological split provenance mode is invalid.")
    train_cutoff, test_cutoff = _utc(manifest.get("train_cutoff")), _utc(manifest.get("test_cutoff"))
    if train_cutoff >= test_cutoff:
        raise ValueError("Chronological split cutoffs are invalid.")
    seen = set()
    counts = {"train": 0, "validation": 0, "test": 0, "purged": 0}
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("Chronological split proposal entries must be objects.")
        space, proposal_id = proposal.get("space"), proposal.get("proposal_id")
        if not isinstance(space, str) or not space or not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError("Chronological split proposals require nonempty keys.")
        key = (space, proposal_id)
        if key in seen:
            raise ValueError("Chronological split proposal keys must be unique.")
        seen.add(key)
        start, end = _utc(proposal.get("start")), _utc(proposal.get("end"))
        if start >= end:
            raise ValueError("Chronological split proposal times are invalid.")
        proposal_mode = proposal.get("provenance_mode", "verified")
        if proposal_mode == "verified":
            if _utc(proposal.get("metadata_available_at")) > start:
                raise ValueError("Proposal metadata was not available at voting open.")
        elif proposal_mode == "local_proxy":
            if manifest_mode != "local_proxy" or proposal.get("metadata_available_at") is not None:
                raise ValueError("Local-proxy proposals cannot claim verified historical availability.")
        else:
            raise ValueError("Proposal provenance mode is invalid.")
        if end < train_cutoff:
            expected = "train"
        elif start >= train_cutoff and end < test_cutoff:
            expected = "validation"
        elif start >= test_cutoff:
            expected = "test"
        else:
            expected = "purged"
        if proposal.get("split") != expected:
            raise ValueError("A proposal membership conflicts with the chronological cutoffs.")
        labels = proposal.get("labels_by_position")
        if (
            not isinstance(labels, list)
            or len(labels) != 3
            or not all(isinstance(label, int) and not isinstance(label, bool) for label in labels)
            or set(labels) != {0, 1, 2}
        ):
            raise ValueError("Chronological split proposal option mappings are invalid.")
        for field in ("content_sha256", "provenance_sha256"):
            value = proposal.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Chronological split proposal provenance hashes are invalid.")
        counts[expected] += 1
    if any(counts[name] == 0 for name in ("train", "validation", "test")):
        raise ValueError("Chronological split target partitions must be nonempty.")
    if "proposal_counts" in manifest and manifest["proposal_counts"] != counts:
        raise ValueError("Chronological split proposal counts do not match its entries.")
    order = {"train": 0, "validation": 1, "test": 2}
    earliest: Dict[str, int] = {}
    for proposal in proposals:
        split = proposal["split"]
        if split != "purged":
            content = proposal["content_sha256"]
            earliest[content] = min(earliest.get(content, order[split]), order[split])
    for proposal in proposals:
        split = proposal["split"]
        expected = split != "purged" and order[split] > earliest[proposal["content_sha256"]]
        if proposal.get("content_sensitivity_excluded") is not expected:
            raise ValueError("Chronological split content-sensitivity flags are invalid.")


def validate_proposal_metadata(proposal: EligibleProposal, row: Mapping[str, Any]) -> None:
    recorded_available = row.get("metadata_available_at")
    availability_matches = (
        recorded_available is None if proposal.metadata_available_at is None
        else proposal.metadata_available_at == _utc(recorded_available)
    )
    if (
        (proposal.space, proposal.proposal_id) != (row.get("space"), row.get("proposal_id"))
        or proposal.start != _utc(row.get("start"))
        or proposal.end != _utc(row.get("end"))
        or not availability_matches
        or proposal.provenance_mode != row.get("provenance_mode", "verified")
        or list(proposal.labels_by_position) != row.get("labels_by_position")
        or proposal_content_hash(proposal.title, proposal.body) != row.get("content_sha256")
        or proposal_provenance_hash(proposal) != row.get("provenance_sha256")
    ):
        raise ValueError("Eligible proposal metadata differs from the split manifest.")


def build_chronological_manifest(
    proposals: Sequence[EligibleProposal],
    *,
    dataset_fingerprint: str,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    train_cutoff: Optional[Any] = None,
    test_cutoff: Optional[Any] = None,
) -> Dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", dataset_fingerprint):
        raise ValueError("A SHA-256 dataset fingerprint is required.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Train and validation fractions must leave a nonempty test fraction.")
    if len(proposals) < 3:
        raise ValueError("At least three proposals are required for chronological splitting.")
    keys = [(p.space, p.proposal_id) for p in proposals]
    if len(set(keys)) != len(keys):
        raise ValueError("Proposal keys must be unique before splitting.")
    if (train_cutoff is None) != (test_cutoff is None):
        raise ValueError("Provide both chronological cutoffs or neither.")

    ordered_starts = sorted(p.start for p in proposals)
    if train_cutoff is None:
        train_cutoff = ordered_starts[int(len(ordered_starts) * train_fraction)]
        test_cutoff = ordered_starts[int(len(ordered_starts) * (train_fraction + validation_fraction))]
    train_time, test_time = _utc(train_cutoff), _utc(test_cutoff)
    if train_time >= test_time:
        raise ValueError("Chronological cutoffs must be strictly increasing.")

    rows = []
    split_counts = {"train": 0, "validation": 0, "test": 0, "purged": 0}
    for proposal in sorted(proposals, key=lambda p: (p.start, p.space, p.proposal_id)):
        if proposal.start >= proposal.end:
            raise ValueError("Proposal opening must precede closing.")
        if proposal.end < train_time:
            split = "train"
        elif proposal.start >= train_time and proposal.end < test_time:
            split = "validation"
        elif proposal.start >= test_time:
            split = "test"
        else:
            split = "purged"
        split_counts[split] += 1
        rows.append(
            {
                "space": proposal.space,
                "proposal_id": proposal.proposal_id,
                "split": split,
                "start": proposal.start.isoformat(),
                "end": proposal.end.isoformat(),
                "metadata_available_at": proposal.metadata_available_at.isoformat() if proposal.metadata_available_at is not None else None,
                "provenance_mode": proposal.provenance_mode,
                "content_sha256": proposal_content_hash(proposal.title, proposal.body),
                "labels_by_position": list(proposal.labels_by_position),
                "provenance_sha256": proposal_provenance_hash(proposal),
                "content_sensitivity_excluded": False,
            }
        )
    if any(split_counts[split] == 0 for split in ("train", "validation", "test")):
        raise ValueError(f"Chronological cutoffs leave an empty target split: {split_counts}")

    earliest_split: Dict[str, int] = {}
    split_order = {"train": 0, "validation": 1, "test": 2}
    for row in rows:
        split = row["split"]
        if split == "purged":
            continue
        content = row["content_sha256"]
        order = split_order[split]
        previous = earliest_split.get(content)
        if previous is not None and previous < order:
            row["content_sensitivity_excluded"] = True
        earliest_split[content] = min(order, previous) if previous is not None else order

    return {
        "version": 1,
        "dataset_sha256": dataset_fingerprint,
        "provenance_mode": "local_proxy" if any(p.provenance_mode == "local_proxy" for p in proposals) else "verified",
        "train_cutoff": train_time.isoformat(),
        "test_cutoff": test_time.isoformat(),
        "train_fraction_requested": train_fraction,
        "validation_fraction_requested": validation_fraction,
        "proposal_counts": split_counts,
        "proposals": rows,
    }


def write_chronological_manifest(manifest: Dict[str, Any], path: Path) -> None:
    validate_chronological_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
