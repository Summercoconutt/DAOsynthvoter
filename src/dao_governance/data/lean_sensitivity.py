"""Map the historical outcome-selected lean sample to original proposal IDs."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dao_governance.modelling.run_artifacts import sha256_file


def build_lean_membership(raw_source: Path, lean_audit: Path, output: Path) -> Dict[str, object]:
    sidecar = output.with_name(output.name + ".meta.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Lean membership output or metadata already exists.")
    audit = pd.read_csv(lean_audit, dtype={"space": "string", "proposal_id": "string"})
    if not {"space", "proposal_id", "retained"}.issubset(audit.columns):
        raise ValueError("Lean audit requires space, proposal_id, and retained columns.")
    retained = audit["retained"].astype("string").str.strip().str.lower()
    if not retained.isin(["true", "false"]).all():
        raise ValueError("Lean audit retained values must be explicit booleans.")
    chosen = audit.loc[retained == "true", ["space", "proposal_id"]]
    if chosen.isna().any().any() or chosen.duplicated().any():
        raise ValueError("Lean audit proposal keys must be nonempty and unique.")
    selected = set(zip(chosen["space"], chosen["proposal_id"]))
    if not selected:
        raise ValueError("Lean audit contains no retained proposals.")

    parquet = pq.ParquetFile(raw_source)
    names = set(parquet.schema_arrow.names)
    if not {"space", "proposal_id", "Proposal ID"}.issubset(names):
        raise ValueError("Raw votes must contain file-based and original proposal IDs.")
    mapping: Dict[Tuple[str, str], str] = {}
    for batch in parquet.iter_batches(
        batch_size=65536, columns=["space", "proposal_id", "Proposal ID"], use_threads=False
    ):
        groups = (
            pa.Table.from_batches([batch])
            .group_by(["space", "proposal_id", "Proposal ID"], use_threads=False)
            .aggregate([])
            .to_pylist()
        )
        for row in groups:
            key = (row["space"], row["proposal_id"])
            if key not in selected:
                continue
            original = row["Proposal ID"]
            if not isinstance(original, str) or not original:
                raise ValueError("A retained proposal lacks an original Snapshot ID.")
            if key in mapping and mapping[key] != original:
                raise ValueError("A lean file key maps to multiple original proposal IDs.")
            mapping[key] = original
    if set(mapping) != selected:
        raise ValueError("Some retained lean proposal keys are absent from the raw source.")
    pairs = sorted({(space, original) for (space, _), original in mapping.items()})
    if len(pairs) != len(mapping):
        raise ValueError("Multiple retained lean file keys refer to the same original proposal.")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        pd.DataFrame(pairs, columns=["space", "proposal_id"]).to_parquet(stream, index=False)
    metadata: Dict[str, object] = {
        "version": 1,
        "selection": "legacy_outcome_selected_lean",
        "raw_votes_sha256": sha256_file(raw_source),
        "lean_audit_sha256": sha256_file(lean_audit),
        "membership_sha256": sha256_file(output),
        "retained_proposals": len(pairs),
        "source_provenance_note": "The historical lean audit is outcome-selected and its original generation is not independently verified.",
    }
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return metadata


def load_lean_membership(path: Path, *, raw_votes_sha256: str) -> set[Tuple[str, str]]:
    metadata = json.loads(path.with_name(path.name + ".meta.json").read_text(encoding="utf-8"))
    if (
        metadata.get("version") != 1
        or metadata.get("selection") != "legacy_outcome_selected_lean"
        or metadata.get("raw_votes_sha256") != raw_votes_sha256
        or metadata.get("membership_sha256") != sha256_file(path)
    ):
        raise ValueError("Lean sensitivity membership does not match the recorded raw source.")
    membership = pd.read_parquet(path)
    if not {"space", "proposal_id"}.issubset(membership.columns) or membership[["space", "proposal_id"]].isna().any().any():
        raise ValueError("Lean sensitivity membership has invalid proposal keys.")
    pairs = set(zip(membership["space"], membership["proposal_id"]))
    if len(pairs) != len(membership) or len(pairs) != metadata.get("retained_proposals"):
        raise ValueError("Lean sensitivity membership contains duplicate or missing keys.")
    return pairs
