"""Proposal-level sampling utilities for optional modelling dataset variants."""
from __future__ import annotations

from typing import Tuple

import pandas as pd


PROPOSAL_KEYS = ["space", "proposal_id"]


def _normalized_length(value: object) -> int:
    return len(" ".join(str(value if value is not None else "").split()))


def build_lean_proposal_sample(
    votes: pd.DataFrame,
    *,
    max_for_fraction: float,
    max_against_fraction: float,
    min_title_chars: int,
    min_body_chars: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return retained votes and one audit row per proposal.

    Vote fractions are calculated from all valid votes, including abstentions.
    """
    missing = [col for col in PROPOSAL_KEYS + ["choice_norm"] if col not in votes.columns]
    if missing:
        raise ValueError(f"Lean proposal sampling requires columns: {missing}")
    if not 0.0 <= max_for_fraction <= 1.0 or not 0.0 <= max_against_fraction <= 1.0:
        raise ValueError("Lean consensus thresholds must be between 0 and 1.")
    if min_title_chars < 0 or min_body_chars < 0:
        raise ValueError("Lean text length thresholds must be non-negative.")
    if min_title_chars and "proposal_title" not in votes.columns:
        raise ValueError("Lean title filtering requires 'proposal_title'.")
    if min_body_chars and "proposal_body" not in votes.columns:
        raise ValueError("Lean body filtering requires 'proposal_body'.")

    rows = []
    for key, group in votes.groupby(PROPOSAL_KEYS, dropna=False, sort=False):
        counts = group["choice_norm"].astype(str).str.lower().str.strip().value_counts()
        total_votes = int(len(group))
        for_votes = int(counts.get("for", 0))
        against_votes = int(counts.get("against", 0))
        abstain_votes = int(counts.get("abstain", 0))
        for_fraction = for_votes / total_votes if total_votes else 0.0
        against_fraction = against_votes / total_votes if total_votes else 0.0
        title_chars = _normalized_length(group["proposal_title"].iloc[0]) if "proposal_title" in group else 0
        body_chars = _normalized_length(group["proposal_body"].iloc[0]) if "proposal_body" in group else 0

        reasons = []
        if for_fraction >= max_for_fraction:
            reasons.append("high_for_consensus")
        if against_fraction >= max_against_fraction:
            reasons.append("high_against_consensus")
        if min_title_chars and title_chars < min_title_chars:
            reasons.append("short_title")
        if min_body_chars and body_chars < min_body_chars:
            reasons.append("short_body")

        rows.append(
            {
                "space": key[0],
                "proposal_id": key[1],
                "total_valid_votes": total_votes,
                "for_votes": for_votes,
                "against_votes": against_votes,
                "abstain_votes": abstain_votes,
                "for_fraction": for_fraction,
                "against_fraction": against_fraction,
                "title_chars": title_chars,
                "body_chars": body_chars,
                "retained": not reasons,
                "exclusion_reasons": "|".join(reasons),
            }
        )

    audit = pd.DataFrame(rows)
    retained_keys = audit.loc[audit["retained"], PROPOSAL_KEYS]
    lean_votes = votes.merge(retained_keys, on=PROPOSAL_KEYS, how="inner")
    return lean_votes, audit