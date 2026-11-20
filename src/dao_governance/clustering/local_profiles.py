"""Build local-proxy clustering snapshots from training history only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dao_governance.clustering.predictive import save_fit_profiles
from dao_governance.data.canonical_dataset import load_proposal_catalog
from dao_governance.data.chronological_split import validate_chronological_manifest, validate_proposal_metadata
from dao_governance.features.asof import ObservationProvenance, VoteEvent, build_cluster_fit_profiles
from dao_governance.modelling.run_artifacts import sha256_file


def prepare_local_cluster_profiles(
    *,
    raw_source: Path,
    metadata_cache: Path,
    canonical_events: Path,
    split_manifest: Path,
    fit_cutoff: object,
    output_dir: Path,
    max_events: int = 2_000_000,
) -> Dict[str, int]:
    if output_dir.exists():
        raise FileExistsError("Cluster profile directory already exists.")
    if max_events < 1:
        raise ValueError("The maximum event count must be positive.")
    catalog, fingerprint, _ = load_proposal_catalog(raw_source, metadata_cache, None, local_proxy=True)
    audit = json.loads(canonical_events.with_name(canonical_events.name + ".audit.json").read_text(encoding="utf-8"))
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    validate_chronological_manifest(split)
    if (
        audit.get("dataset_sha256") != fingerprint
        or audit.get("canonical_events_sha256") != sha256_file(canonical_events)
        or audit.get("provenance_mode") != "local_proxy"
        or split.get("dataset_sha256") != fingerprint
        or split.get("provenance_mode") != "local_proxy"
    ):
        raise ValueError("Local cluster inputs do not share the same proxy dataset.")
    cutoff = pd.to_datetime(fit_cutoff, utc=True, errors="coerce")
    train_cutoff = pd.to_datetime(split["train_cutoff"], utc=True, errors="coerce")
    if pd.isna(cutoff) or cutoff >= train_cutoff:
        raise ValueError("Cluster warm-up cutoff must be strictly before the training cutoff.")
    parquet = pq.ParquetFile(canonical_events)
    if parquet.metadata.num_rows > max_events:
        raise ValueError("Canonical event count exceeds the configured profile preparation limit.")
    proposals = {
        (row["space"], row["proposal_id"]): row
        for row in split["proposals"]
        if row["split"] == "train" and pd.to_datetime(row["end"], utc=True) < cutoff
    }
    if not proposals:
        raise ValueError("No training proposals have closed before the warm-up cutoff.")
    ids = pa.array(sorted({key[1] for key in proposals}), type=parquet.schema_arrow.field("proposal_id").type)
    columns = ["example_id", "voter", "space", "proposal_id", "vote_ts", "label_id"]
    events = []
    checked = set()
    for batch in parquet.iter_batches(batch_size=65536, columns=columns, use_threads=False):
        table = pa.Table.from_batches([batch])
        selected = table.filter(pc.is_in(table["proposal_id"], value_set=ids))
        for row in selected.to_pylist():
            key = (row["space"], row["proposal_id"])
            if key not in proposals:
                continue
            proposal = catalog.eligible.get(key)
            if proposal is None:
                raise ValueError("A warm-up event lacks eligible local proposal metadata.")
            if key not in checked:
                validate_proposal_metadata(proposal, proposals[key])
                checked.add(key)
            timestamp = pd.to_datetime(row["vote_ts"], utc=True, errors="coerce")
            if pd.isna(timestamp):
                raise ValueError("Warm-up events require valid vote timestamps.")
            events.append(
                VoteEvent(
                    example_id=row["example_id"], voter=row["voter"], space=row["space"],
                    proposal_id=row["proposal_id"], vote_ts=timestamp, observed_at=timestamp,
                    proposal_end=proposal.end, label_id=int(row["label_id"]),
                )
            )
    if not events:
        raise ValueError("No eligible training votes precede the warm-up cutoff.")
    events.sort(key=lambda event: event.observed_at)
    provenance = ObservationProvenance(
        False, "Recorded vote time is assumed to equal observation time; this is unverified.", local_proxy=True
    )
    dao, voter = build_cluster_fit_profiles(events, cutoff=cutoff, availability_provenance=provenance)
    if dao.empty or voter.empty:
        raise ValueError("The warm-up period produced no clustering profiles.")
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    save_fit_profiles(
        dao, output_dir / "dao.parquet", kind="dao", split_manifest=split_manifest,
        fit_cutoff=cutoff, availability_provenance=provenance,
    )
    save_fit_profiles(
        voter, output_dir / "voters.parquet", kind="voter", split_manifest=split_manifest,
        fit_cutoff=cutoff, availability_provenance=provenance,
    )
    return {"warmup_votes": len(events), "dao_profiles": len(dao), "voter_dao_profiles": len(voter)}
