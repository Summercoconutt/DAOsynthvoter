"""Eligibility and label mapping for point-in-time Snapshot vote prediction."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple

import pandas as pd

from dao_governance.features.behaviour_dataset import LABEL_MAP

_OPTION_LABELS = {
    "for": LABEL_MAP["for"],
    "yes": LABEL_MAP["for"],
    "approve": LABEL_MAP["for"],
    "support": LABEL_MAP["for"],
    "in favor": LABEL_MAP["for"],
    "in favour": LABEL_MAP["for"],
    "against": LABEL_MAP["against"],
    "no": LABEL_MAP["against"],
    "reject": LABEL_MAP["against"],
    "oppose": LABEL_MAP["against"],
    "abstain": LABEL_MAP["abstain"],
}
_LABEL_NAMES = {value: key for key, value in LABEL_MAP.items()}
_INTEGER_CHOICE = re.compile(r"-?\d+(?:\.0+)?")
_FINGERPRINT = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class EligibleProposal:
    space: str
    proposal_id: str
    title: str
    body: str
    start: pd.Timestamp
    end: pd.Timestamp
    labels_by_position: Tuple[int, int, int]
    historical_evidence: str
    metadata_available_at: Optional[pd.Timestamp]
    provenance_mode: Literal["verified", "local_proxy"] = "verified"

    def __post_init__(self) -> None:
        if not self.space or not self.proposal_id:
            raise ValueError("Eligible proposals require nonempty identifiers.")
        if not isinstance(self.title, str) or not isinstance(self.body, str):
            raise ValueError("Eligible proposals require available title and body text.")
        if self.start.tzinfo is None or self.end.tzinfo is None or self.start >= self.end:
            raise ValueError("Eligible proposals require a timezone-aware voting window.")
        if set(self.labels_by_position) != {0, 1, 2} or len(self.labels_by_position) != 3:
            raise ValueError("Eligible proposal options must map once to each canonical class.")
        if not isinstance(self.historical_evidence, str) or not self.historical_evidence.strip():
            raise ValueError("Eligible proposals require explicit historical provenance.")
        if self.provenance_mode == "verified":
            if not isinstance(self.metadata_available_at, pd.Timestamp) or self.metadata_available_at.tzinfo is None or self.metadata_available_at > self.start:
                raise ValueError("Proposal metadata must be available no later than voting opens.")
        elif self.provenance_mode != "local_proxy" or self.metadata_available_at is not None:
            raise ValueError("Unverified local metadata must not claim a historical availability time.")


@dataclass
class ProposalCatalog:
    eligible: Dict[Tuple[str, str], EligibleProposal]
    excluded: Dict[Tuple[str, str], Tuple[str, ...]]


def _option_label(value: Any) -> Optional[int]:
    if not isinstance(value, str):
        return None
    return _OPTION_LABELS.get(" ".join(value.casefold().split()))


def _unix_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or int(value) != value:
        return None
    timestamp = pd.to_datetime(int(value), unit="s", utc=True, errors="coerce")
    return None if pd.isna(timestamp) else timestamp


def historical_metadata_hash(metadata: Mapping[str, Any]) -> str:
    space = metadata.get("space")
    payload = {
        "id": metadata.get("id"),
        "space": space.get("id") if isinstance(space, dict) else None,
        "type": metadata.get("type"),
        "choices": metadata.get("choices"),
        "title": metadata.get("title"),
        "body": metadata.get("body"),
        "start": metadata.get("start"),
        "end": metadata.get("end"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def assess_proposal(
    record: Mapping[str, Any], *, allow_local_proxy: bool = False
) -> Tuple[Optional[EligibleProposal], Tuple[str, ...]]:
    """Return a proposal only when options, text, timing, and provenance are usable."""
    space = record.get("space")
    proposal_id = record.get("proposal_id")
    if not isinstance(space, str) or not space or not isinstance(proposal_id, str) or not proposal_id:
        raise ValueError("Proposal records require nonempty space and proposal IDs.")
    metadata = record.get("metadata")
    if record.get("status") != "retrieved" or not isinstance(metadata, dict):
        return None, ("missing_or_unmatched_metadata",)
    metadata_space = metadata.get("space")
    if metadata.get("id") != proposal_id or not isinstance(metadata_space, dict) or metadata_space.get("id") != space:
        return None, ("missing_or_unmatched_metadata",)

    reasons = []
    voting_type = metadata.get("type")
    if voting_type != "single-choice":
        reasons.append("unsupported_voting_type")

    choices = metadata.get("choices")
    labels = [_option_label(choice) for choice in choices] if isinstance(choices, list) else []
    if len(labels) != 3:
        reasons.append("unsupported_option_count")
    elif any(label is None for label in labels) or len(set(labels)) != 3:
        reasons.append("ambiguous_options")

    title, body = metadata.get("title"), metadata.get("body")
    if not isinstance(title, str) or not isinstance(body, str):
        reasons.append("text_unavailable")

    start, end = _unix_timestamp(metadata.get("start")), _unix_timestamp(metadata.get("end"))
    if start is None or end is None or start >= end:
        reasons.append("invalid_proposal_window")

    provenance = record.get("historical_provenance")
    evidence = provenance.get("evidence") if isinstance(provenance, dict) else None
    available_at = _unix_timestamp(provenance.get("available_at")) if isinstance(provenance, dict) else None
    local_proxy = allow_local_proxy and record.get("local_binding") == "retrieved"
    if local_proxy:
        evidence = "Local metadata matched stored title and body prefix; historical availability is unverified."
        available_at = None
    else:
        if (
            not isinstance(provenance, dict)
            or provenance.get("verified") is not True
            or not isinstance(evidence, str)
            or not evidence.strip()
            or available_at is None
            or provenance.get("metadata_sha256") != historical_metadata_hash(metadata)
        ):
            reasons.append("unverified_pre_vote_metadata")
        elif start is not None and available_at > start:
            reasons.append("metadata_not_available_at_open")

    if reasons:
        return None, tuple(reasons)
    if start is None or end is None or not isinstance(title, str) or not isinstance(body, str):
        raise RuntimeError("Validated proposal has inconsistent metadata.")
    if any(label is None for label in labels):
        raise RuntimeError("Validated proposal has inconsistent option labels.")
    if not isinstance(evidence, str) or (not local_proxy and available_at is None):
        raise RuntimeError("Validated proposal has inconsistent historical provenance.")
    return (
        EligibleProposal(
            space=space,
            proposal_id=proposal_id,
            title=title,
            body=body,
            start=start,
            end=end,
            labels_by_position=(int(labels[0]), int(labels[1]), int(labels[2])),
            historical_evidence=evidence,
            metadata_available_at=available_at,
            provenance_mode="local_proxy" if local_proxy else "verified",
        ),
        (),
    )


def build_proposal_catalog(
    records: Iterable[Mapping[str, Any]], *, allow_local_proxy: bool = False
) -> ProposalCatalog:
    eligible: Dict[Tuple[str, str], EligibleProposal] = {}
    excluded: Dict[Tuple[str, str], Tuple[str, ...]] = {}
    for record in records:
        proposal, reasons = assess_proposal(record, allow_local_proxy=allow_local_proxy)
        key = (record["space"], record["proposal_id"])
        if key in eligible or key in excluded:
            raise ValueError("Duplicate proposal metadata records require resolution.")
        if proposal is None:
            excluded[key] = reasons
        else:
            eligible[key] = proposal
    return ProposalCatalog(eligible=eligible, excluded=excluded)


def map_vote_choice(choice: Any, proposal: EligibleProposal) -> Tuple[Optional[int], Optional[str]]:
    """Map a single option position using that proposal's ordered choices."""
    if isinstance(choice, dict):
        return None, "weighted_choice"
    if isinstance(choice, (list, tuple)):
        return None, "multi_choice"
    if choice is None or isinstance(choice, bool):
        return None, "invalid_choice"
    value = str(choice).strip()
    if value.startswith("{"):
        return None, "weighted_choice"
    if value.startswith("["):
        return None, "multi_choice"
    if not _INTEGER_CHOICE.fullmatch(value):
        return None, "invalid_choice"
    try:
        position = int(value.split(".", 1)[0])
    except ValueError:
        return None, "choice_out_of_range"
    if position < 1 or position > len(proposal.labels_by_position):
        return None, "choice_out_of_range"
    return proposal.labels_by_position[position - 1], None


def canonicalize_vote_partition(
    votes: pd.DataFrame,
    catalog: ProposalCatalog,
    *,
    source_fingerprint: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Canonicalize a complete proposal partition; repeated voter keys require a decision."""
    required = {"voter", "space", "proposal_id", "choice", "vote_timestamp"}
    missing = required - set(votes.columns)
    if missing:
        raise ValueError(f"Vote partition is missing columns: {sorted(missing)}")
    if not _FINGERPRINT.fullmatch(source_fingerprint):
        raise ValueError("A SHA-256 source fingerprint is required.")

    frame = votes[list(sorted(required))].copy()
    frame["voter"] = frame["voter"].astype("string").str.strip().str.lower()
    frame["vote_timestamp"] = pd.to_datetime(frame["vote_timestamp"], utc=True, errors="coerce")
    valid_keys = (
        frame["voter"].notna()
        & frame["voter"].ne("")
        & frame["space"].notna()
        & frame["proposal_id"].notna()
    )
    if frame.loc[valid_keys].duplicated(["voter", "space", "proposal_id"], keep=False).any():
        raise ValueError("Repeated voter/proposal records require an explicit revision policy.")

    output = []
    counts: Counter[str] = Counter()
    for row in frame.itertuples(index=False):
        voter, space, proposal_id, timestamp, choice = (
            row.voter,
            row.space,
            row.proposal_id,
            row.vote_timestamp,
            row.choice,
        )
        if pd.isna(voter) or not voter or pd.isna(space) or pd.isna(proposal_id):
            counts["missing_identity"] += 1
            continue
        if pd.isna(timestamp):
            counts["invalid_vote_timestamp"] += 1
            continue
        key = (space, proposal_id)
        proposal = catalog.eligible.get(key)
        if proposal is None:
            reasons = catalog.excluded.get(key, ("missing_or_unmatched_metadata",))
            for reason in reasons:
                counts[reason] += 1
            continue
        if timestamp < proposal.start or timestamp >= proposal.end:
            counts["vote_outside_proposal_window"] += 1
            continue
        if proposal.metadata_available_at is not None and timestamp <= proposal.metadata_available_at:
            counts["metadata_not_observable_before_vote"] += 1
            continue
        label, reason = map_vote_choice(choice, proposal)
        if reason is not None:
            counts[reason] += 1
            continue
        if label is None:
            raise RuntimeError("A valid vote choice did not produce a label.")
        identity = [source_fingerprint, space, proposal_id, voter, timestamp.isoformat()]
        example_id = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        output.append(
            {
                "example_id": example_id,
                "voter": voter,
                "space": space,
                "proposal_id": proposal_id,
                "vote_ts": timestamp,
                "label_id": label,
                "choice_norm": _LABEL_NAMES[label],
            }
        )
        counts["eligible_votes"] += 1
        if proposal.provenance_mode == "local_proxy":
            counts["unverified_local_proxy_votes"] += 1
    columns = ["example_id", "voter", "space", "proposal_id", "vote_ts", "label_id", "choice_norm"]
    return pd.DataFrame(output, columns=columns), dict(counts)
