#!/usr/bin/env python3
"""Prepare chronological vote examples using verified observation timestamps."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.features.point_in_time_examples import prepare_example_artifacts
from dao_governance.settings import load_config, project_root, require_local_data_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--extra-config", default="")
    parser.add_argument("--historical-provenance", default="")
    parser.add_argument("--observations", default="")
    parser.add_argument("--observation-provenance", default="")
    parser.add_argument("--local-proxy", action="store_true")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dao-bundle", default="")
    parser.add_argument("--voter-bundle", default="")
    parser.add_argument("--include-target-proposal", action="store_true")
    parser.add_argument("--max-events", type=int, default=2_000_000)
    args = parser.parse_args()
    base = project_root()
    config = (base / args.config).resolve()
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=config, extra_path=extra)
    local_only = bool((cfg.get("run") or {}).get("local_data_only", False))
    if args.local_proxy and (not local_only or args.historical_provenance or args.observations or args.observation_provenance):
        parser.error("Local-proxy mode requires local-data-only configuration and no verified provenance files.")
    if not args.local_proxy and not (args.historical_provenance and args.observations and args.observation_provenance):
        parser.error("Verified mode requires historical provenance and observation files.")
    paths = cfg.get("paths", {})
    source = (base / paths.get("master_votes_parquet", "data/interim/master_votes.parquet")).resolve()
    metadata = (base / paths.get("proposal_metadata_jsonl", "data/interim/snapshot_proposal_metadata.jsonl")).resolve()
    canonical = (base / paths.get("canonical_vote_events_parquet", "data/processed/canonical_vote_events.parquet")).resolve()
    historical = Path(args.historical_provenance) if args.historical_provenance else None
    observations = Path(args.observations) if args.observations else None
    observation_provenance = Path(args.observation_provenance) if args.observation_provenance else None
    split_manifest = Path(args.split_manifest)
    output_dir = Path(args.output_dir)
    if local_only:
        source = require_local_data_path(source, root=base)
        metadata = require_local_data_path(metadata, root=base)
        canonical = require_local_data_path(canonical, root=base)
        historical = require_local_data_path(historical, root=base) if historical is not None else None
        observations = require_local_data_path(observations, root=base) if observations is not None else None
        observation_provenance = require_local_data_path(observation_provenance, root=base) if observation_provenance is not None else None
        split_manifest = require_local_data_path(split_manifest, root=base)
        if args.local_proxy:
            output_dir = require_local_data_path(output_dir, root=base)
    result = prepare_example_artifacts(
        raw_source=source,
        metadata_cache=metadata,
        historical_provenance=historical,
        canonical_events=canonical,
        observations=observations,
        observation_provenance=observation_provenance,
        split_manifest=split_manifest,
        output_dir=output_dir,
        include_target_proposal=args.include_target_proposal,
        dao_bundle_path=Path(args.dao_bundle) if args.dao_bundle else None,
        voter_bundle_path=Path(args.voter_bundle) if args.voter_bundle else None,
        max_events=args.max_events,
        local_proxy=args.local_proxy,
    )
    print(f"Prepared chronological example partitions: {result['partitions']}")


if __name__ == "__main__":
    main()
