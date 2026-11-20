#!/usr/bin/env python3
"""Create a proposal-disjoint split from provenance-gated canonical vote events."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.data.canonical_dataset import create_split_from_canonical
from dao_governance.settings import load_config, project_root, require_local_data_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--extra-config", default="")
    parser.add_argument("--historical-provenance", default="")
    parser.add_argument("--local-proxy", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-cutoff", default=None)
    parser.add_argument("--test-cutoff", default=None)
    args = parser.parse_args()
    base = project_root()
    config = (base / args.config).resolve()
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=config, extra_path=extra)
    local_only = bool((cfg.get("run") or {}).get("local_data_only", False))
    if args.local_proxy and (not local_only or args.historical_provenance):
        parser.error("Local-proxy mode requires local-data-only configuration and no verified provenance file.")
    if not args.local_proxy and not args.historical_provenance:
        parser.error("Verified mode requires --historical-provenance.")
    paths = cfg.get("paths", {})
    source = (base / paths.get("master_votes_parquet", "data/interim/master_votes.parquet")).resolve()
    metadata = (base / paths.get("proposal_metadata_jsonl", "data/interim/snapshot_proposal_metadata.jsonl")).resolve()
    events = (base / paths.get("canonical_vote_events_parquet", "data/processed/canonical_vote_events.parquet")).resolve()
    provenance = Path(args.historical_provenance) if args.historical_provenance else None
    output = Path(args.output)
    if local_only:
        source = require_local_data_path(source, root=base)
        metadata = require_local_data_path(metadata, root=base)
        events = require_local_data_path(events, root=base)
        if provenance is not None:
            provenance = require_local_data_path(provenance, root=base)
        if args.local_proxy:
            output = require_local_data_path(output, root=base)
    manifest = create_split_from_canonical(
        source,
        metadata,
        provenance,
        events,
        output,
        train_cutoff=args.train_cutoff,
        test_cutoff=args.test_cutoff,
        local_proxy=args.local_proxy,
    )
    print(f"Chronological proposal split counts: {manifest['proposal_counts']}")


if __name__ == "__main__":
    main()
