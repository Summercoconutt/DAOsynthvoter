# Point-in-time vote prediction: initial data audit

> Current direction: use existing files under `data/` only and do not query or
> verify against Snapshot. References below to the public API describe checks
> performed before that direction; the resulting cache is now a local input
> whose historical provenance remains unverified.

This audits the configured local exports and a newly collected public Snapshot
metadata cache. It does **not** establish historically verified eligibility or
validate the historical model results.

## Local coverage

| Dataset | Vote rows | DAO spaces | Proposal groups | Distinct voters |
|---|---:|---:|---:|---:|
| Raw merged votes | 39,353,331 | 436 | 57,039 actual `(space, Proposal ID)` keys | 1,845,961 |
| Existing full cleaned votes | 38,035,589 | 429 | 52,051 file-based keys | 1,790,047 |
| Existing lean votes | 25,993,599 | 370 | 11,452 file-based keys | 1,096,693 |

The raw vote timestamps range from 2020-07-18 16:12:04 UTC through
2026-02-14 19:57:39 UTC. The raw export has no missing proposal IDs, voter IDs,
or vote timestamps. No repeated `(case-normalized voter, space, original
proposal ID)` keys were found in this export. The export has no vote revision
ID, so it cannot establish whether earlier revisions were omitted upstream.

The existing cleaned files use an assumed numeric choice mapping. Their
recorded counts are:

| Recorded label | Full | Lean |
|---|---:|---:|
| for | 31,278,470 | 19,993,324 |
| against | 6,004,137 | 5,336,959 |
| abstain | 752,982 | 663,316 |

These are **not verified eligible class counts**. The raw source also has
1,317,742 rows labelled `unknown` by the old normalization, which the existing
cleaning omitted. Rebuild eligibility from the raw choice and authoritative
proposal options rather than starting from the cleaned files.

The lean sample retains approximately 68.3% of full-cleaned votes but only
22.0% of its file-keyed proposal groups. It was selected using final vote
fractions and text thresholds, so it is an outcome-selected sensitivity
population, not an unbiased replacement for the full dataset.

## Proposal identity and metadata

In the raw merged file, every one of the 39,353,331 existing `proposal_id`
values differs from the original `Proposal ID`: the former includes the
collector's filename timestamp. The original ID is the public Snapshot ID.
The cleaning code now restores the original ID when the two representations
match this known filename form and rejects other conflicts. Existing cleaned
and historical run artifacts are unchanged.

The merged vote files do not contain the ordered proposal options, voting type,
or proposal opening and closing times. In a random sample of 32 source
proposal files, all lacked those fields; 13 stored bodies were exactly 800
characters long and five were empty. The bundled collector truncates bodies at
800 characters.

A public Snapshot API lookup using the **original** IDs returned all 32
sampled proposals. Nineteen were single-choice, seven had three options, and
one satisfied both conditions; its options were not exactly the plain
FOR/AGAINST/ABSTAIN or YES/NO/ABSTAIN sets. These small-sample counts must not
be extrapolated to the corpus. All 32 titles and truncated body prefixes
matched the export; 13 current API bodies exceeded 800 characters.

All 32 sampled API records had a missing `updated` value. Snapshot's current
[editing documentation](https://docs.snapshot.box/proposals/edit-a-proposal)
states that proposals cannot be edited once voting starts, but the pilot does
not by itself verify historical full-text or option provenance for this
dataset. The metadata collector records current public metadata and its
retrieval time; it does not certify that metadata as historically available.

The completed [metadata cache](../data/interim/snapshot_proposal_metadata.jsonl)
contains 57,037 retrieved proposals and two missing proposals. Its input hash
and record count were verified. Of the retrieved records, 56,650 have no
`updated` value; the 387 with one were updated before their recorded voting
start. Missing update timestamps are not proof that historical metadata was
unchanged.

After the local-only direction, the existing cache was reconciled with
`data/interim/snapshot_votes.parquet` without network access. All 39,353,331
local vote rows resolved to 57,039 proposal keys: 57,037 matched cached
metadata and two were missing. The resulting local cache records both local
file fingerprints and explicitly marks historical availability as unverified.

The local-only retrospective proxy preparation produced 120,891 candidate vote
events and a proposal-disjoint split with 116,073 training, 1,877 validation,
and 2,333 test examples; 608 vote events belong to purged boundary-spanning
proposals. Its manifests explicitly record that historical proposal metadata
is unverified and that recorded vote time is assumed to equal observation
time. No full model training was performed.

Across all 57,037 retrieved proposals, the current title and first 800
characters of the normalized body match the corresponding raw export. This
includes all 394 structural candidates. The comparison cannot verify body
content beyond the export's truncation limit or establish historical option
ordering.

Using conservative exact option aliases and complete current metadata, 394
proposals are **structural candidates** for the three-class task. The proposal
audit also found 42,819 unsupported option counts, 27,196 unsupported voting
types, and 6,559 ambiguous option lists; these reasons can overlap. None of the
57,037 retrieved records has yet been independently verified as historically
available before the vote.

The 394 candidates correspond to 120,891 raw vote rows in 51 DAOs, from
32,264 distinct voters. Mapping the raw positions through the retrieved option
order gives 97,802 FOR, 15,368 AGAINST, and 7,721 ABSTAIN candidate votes.
All candidate vote timestamps fall within the recorded proposal windows.
These counts remain provisional until historical provenance is established.
The outcome-selected lean sample retains 197 of the candidate proposals and
69,409 candidate votes across 29 DAOs.

For 1,771 of those candidate votes (1.46%), the old normalized label differs
from the label obtained using the retrieved option order. Most of these are
AGAINST/ABSTAIN swaps. This comparison is also provisional because historical
option ordering is not independently verified.

The raw export marks 1,741 candidate votes as whales, and every candidate
whale flag agrees with `VP Ratio (%) > 5`. The bundled collector derives that
ratio from the proposal's final total, so this flag is not a safe pre-vote
feature. Candidate voting-power values were finite and nonnegative, but their
availability immediately before voting has not been established.

### Provisional chronological split

Proposal-start quantiles, without using labels, give tentative UTC cutoffs of
2025-02-19 23:56:59 and 2025-05-04 23:12:00. Purging proposals spanning either
cutoff gives the following candidate counts:

| Target split | Proposals | Votes | FOR | AGAINST | ABSTAIN |
|---|---:|---:|---:|---:|---:|
| Train | 273 | 116,073 | 93,914 | 14,713 | 7,446 |
| Validation | 40 | 1,877 | 1,521 | 283 | 73 |
| Test | 60 | 2,333 | 1,885 | 273 | 175 |
| Purged | 21 | 608 | 482 | 99 | 27 |

These are proposal-count-based cutoffs, not a 70/15/15 split of vote rows:
the training period contains most candidate votes. No normalized title/body
duplicates were found across these provisional target splits.

Using vote timestamps as an **unverified availability proxy**, 34,576 train,
420 validation, and 480 test candidate votes have no strictly earlier
voter/DAO candidate history. Equal timestamps were not counted as prior
history. Cold-start votes must remain in the model population.

## Timestamps and source completeness

Among 12,596 votes in the same 32-file sample, 1,629 shared a timestamp with
another vote in their proposal file. The collector uses a strict timestamp
pagination cursor, so a page boundary inside a tied timestamp can omit votes.
For one sampled proposal, the current API vote count exceeded the local file's
row count; a changed source or pagination loss are possible explanations, not
established causes.

The public [Snapshot API documentation](https://docs.snapshot.box/tools/api)
states a limit of 100 requests per minute. The new collector uses bounded
batches, rate limiting, resumable output, and only public proposal IDs; it does
not request or transmit voter addresses.

## Historical voter-split overlap

For the historical `full/full_20261013` run, the cleaned input and saved voter
split hashes match their run manifest. Reconstructing the five-vote windows
from that input gives these bounds on cross-split target-proposal overlap:

| Split pair | Guaranteed overlap | Possible overlap |
|---|---:|---:|
| Train / validation | 31,815 | 32,663 |
| Train / test | 31,331 | 32,201 |
| Validation / test | 28,393 | 29,237 |

Equal timestamps at a window boundary make the exact target membership depend
on tie ordering, so the table reports lower and upper bounds rather than an
invented exact count. The historical behavior CSV itself was not rehashed;
these are reconstructions from the verified source and split.

Using Unicode NFKC, case folding, and whitespace normalization on the stored
title/body pairs, the corresponding bounds on **unique text** overlap are
30,208–30,989 for train/validation, 29,696–30,469 for train/test, and
27,105–27,903 for validation/test. Stored text was consistent within each
target proposal. Bodies in this historical source may be truncated.

## Outstanding gates

- Verify historical metadata and event availability before turning the
  structural candidate counts above into an eligible population. The
  chronological counts are provisional until that gate is resolved.
- Establish historical availability of the full text and options, investigate
  source completeness, and resolve revision semantics if revisions are found.
- Verify any additional historical artifacts and hashes before using them.
- Use a compatible isolated Python environment. The existing project `.venv`
  lacks dependencies; focused tests passed with the system interpreter, which
  emitted NumPy binary-compatibility warnings for optional libraries.
- Do not train the full model until the audit is complete and the user has
  separately approved its compute and storage requirements.
