"""Streaming, strictly pre-vote history features."""
from __future__ import annotations

import heapq
import itertools
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Iterator, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class ObservationProvenance:
    verified: bool
    evidence: str
    local_proxy: bool = False

    def validate(self) -> None:
        if (
            not isinstance(self.evidence, str)
            or not self.evidence.strip()
            or not ((self.verified is True and self.local_proxy is False) or (self.verified is False and self.local_proxy is True))
        ):
            raise ValueError("Verified observation provenance or an explicit local proxy is required.")


@dataclass(frozen=True)
class VoteEvent:
    example_id: str
    voter: str
    space: str
    proposal_id: str
    vote_ts: datetime
    observed_at: datetime
    proposal_end: datetime
    label_id: int


def _validate_event(event: VoteEvent) -> None:
    if not all(isinstance(value, str) and value.strip() for value in (event.example_id, event.voter, event.space, event.proposal_id)):
        raise ValueError("Vote events require nonempty identifiers.")
    for timestamp in (event.vote_ts, event.observed_at, event.proposal_end):
        if not isinstance(timestamp, datetime) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Vote event timestamps must be timezone-aware.")
    if event.observed_at < event.vote_ts:
        raise ValueError("A vote cannot be observable before it is cast.")
    if event.vote_ts >= event.proposal_end:
        raise ValueError("A vote must precede the proposal close.")
    if isinstance(event.label_id, bool) or event.label_id not in (0, 1, 2):
        raise ValueError("Vote event labels must use the three canonical classes.")


def _update(counter: Counter[int], label: int) -> None:
    counter[label] += 1


def _without(counter: Counter[int], excluded: Counter[int]) -> Counter[int]:
    if any(excluded[label] > counter[label] for label in (0, 1, 2)):
        raise RuntimeError("Proposal history exceeds the corresponding aggregate history.")
    return counter - excluded


def _history_values(prefix: str, counts: Counter[int]) -> Dict[str, float]:
    total = sum(counts.values())
    return {
        f"{prefix}_prior_count": float(total),
        f"{prefix}_has_history": float(total > 0),
        f"{prefix}_prior_for": counts[0] / total if total else float("nan"),
        f"{prefix}_prior_against": counts[1] / total if total else float("nan"),
        f"{prefix}_prior_abstain": counts[2] / total if total else float("nan"),
    }


class _History:
    def __init__(self) -> None:
        self.voters: Dict[str, Counter[int]] = {}
        self.voter_daos: Dict[Tuple[str, str], Counter[int]] = {}
        self.daos: Dict[str, Counter[int]] = {}
        self.proposals: Dict[Tuple[str, str], Counter[int]] = {}
        self.voter_proposals: Dict[Tuple[str, str], Dict[str, Counter[int]]] = {}
        self.ends: Dict[Tuple[str, str], datetime] = {}
        self.proposal_ends: Dict[Tuple[str, str], datetime] = {}
        self.expiry: list[Tuple[datetime, Tuple[str, str]]] = []

    def expire(self, current_time: datetime) -> None:
        while self.expiry and self.expiry[0][0] <= current_time:
            end, key = heapq.heappop(self.expiry)
            if self.ends.get(key) == end:
                self.proposals.pop(key, None)
                self.voter_proposals.pop(key, None)
                del self.ends[key]

    def add(self, event: VoteEvent, current_time: datetime) -> None:
        _update(self.voters.setdefault(event.voter, Counter()), event.label_id)
        _update(self.voter_daos.setdefault((event.voter, event.space), Counter()), event.label_id)
        _update(self.daos.setdefault(event.space, Counter()), event.label_id)
        key = (event.space, event.proposal_id)
        if key in self.proposal_ends and self.proposal_ends[key] != event.proposal_end:
            raise ValueError("A proposal has conflicting close times.")
        self.proposal_ends[key] = event.proposal_end
        if event.proposal_end <= current_time:
            return
        if key not in self.ends:
            self.ends[key] = event.proposal_end
            heapq.heappush(self.expiry, (event.proposal_end, key))
        _update(self.proposals.setdefault(key, Counter()), event.label_id)
        voters = self.voter_proposals.setdefault(key, {})
        _update(voters.setdefault(event.voter, Counter()), event.label_id)

    def features(self, target: VoteEvent, include_target_proposal: bool) -> Dict[str, float]:
        key = (target.space, target.proposal_id)
        voter = self.voters.get(target.voter, Counter())
        voter_dao = self.voter_daos.get((target.voter, target.space), Counter())
        dao = self.daos.get(target.space, Counter())
        if not include_target_proposal:
            voter_proposal = self.voter_proposals.get(key, {}).get(target.voter, Counter())
            voter = _without(voter, voter_proposal)
            voter_dao = _without(voter_dao, voter_proposal)
            dao = _without(dao, self.proposals.get(key, Counter()))
        return {
            **_history_values("voter", voter),
            **_history_values("voter_dao", voter_dao),
            **_history_values("dao", dao),
        }


def _ordered(events: Iterable[VoteEvent], attribute: str, message: str) -> Iterator[VoteEvent]:
    previous: Optional[datetime] = None
    for event in events:
        _validate_event(event)
        timestamp = getattr(event, attribute)
        if previous is not None and timestamp < previous:
            raise ValueError(message)
        previous = timestamp
        yield event


def iter_asof_features(
    targets: Iterable[VoteEvent],
    observations: Iterable[VoteEvent],
    *,
    availability_provenance: ObservationProvenance,
    include_target_proposal: bool = False,
) -> Iterator[Dict[str, object]]:
    """Yield features from target-time and observation-time sorted event streams."""
    if not isinstance(availability_provenance, ObservationProvenance):
        raise ValueError("Verified vote observation-time provenance is required.")
    availability_provenance.validate()
    history = _History()
    observed = _ordered(observations, "observed_at", "Observed events must be sorted by observation timestamp.")
    ordered_targets = _ordered(targets, "vote_ts", "Target events must be sorted by vote timestamp.")
    pending: Optional[VoteEvent] = next(observed, None)

    for target_time, group in itertools.groupby(ordered_targets, key=lambda event: event.vote_ts):
        history.expire(target_time)
        while pending is not None and pending.observed_at < target_time:
            history.add(pending, target_time)
            pending = next(observed, None)
        for target in group:
            yield {
                "example_id": target.example_id,
                "voter": target.voter,
                "space": target.space,
                "proposal_id": target.proposal_id,
                "asof_ts": target.vote_ts,
                **history.features(target, include_target_proposal),
            }

    while pending is not None:
        pending = next(observed, None)


def build_cluster_fit_profiles(
    observations: Iterable[VoteEvent],
    *,
    cutoff: datetime,
    availability_provenance: ObservationProvenance,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Snapshot completed-proposal history before a cluster fit cutoff."""
    if not isinstance(availability_provenance, ObservationProvenance):
        raise ValueError("Verified vote observation-time provenance is required.")
    availability_provenance.validate()
    if not isinstance(cutoff, datetime) or cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("The cluster fit cutoff must be timezone-aware.")
    history = _History()
    ordered = _ordered(observations, "observed_at", "Observed events must be sorted by observation timestamp.")
    for event in ordered:
        if event.observed_at < cutoff and event.proposal_end < cutoff:
            history.add(event, cutoff)
    dao_rows = [
        {"space": space, "asof_ts": cutoff, **_history_values("dao", counts)}
        for space, counts in history.daos.items()
    ]
    voter_rows = [
        {
            "voter": voter,
            "space": space,
            "asof_ts": cutoff,
            **_history_values("voter", history.voters[voter]),
            **_history_values("voter_dao", counts),
            **_history_values("dao", history.daos[space]),
        }
        for (voter, space), counts in history.voter_daos.items()
    ]
    return pd.DataFrame(dao_rows), pd.DataFrame(voter_rows)
