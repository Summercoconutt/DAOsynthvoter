#!/usr/bin/env python3
"""Stage 3: Global cleaning — votes/proposals validation (see config/cleaning.yaml)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.settings import load_config, project_root, require_local_data_path
from pipeline.global_cleaning import run_global_cleaning_stage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--extra-config", type=str, default="", help="Optional YAML merged over --config (e.g. server raw paths).")
    ap.add_argument("--point-in-time", action="store_true")
    ap.add_argument("--historical-provenance", type=str, default="")
    ap.add_argument("--local-proxy", action="store_true")
    args = ap.parse_args()
    base = project_root()
    cfg_path = (base / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=cfg_path, extra_path=extra)

    if args.point_in_time:
        local_only = bool((cfg.get("run") or {}).get("local_data_only", False))
        if args.local_proxy and (not local_only or args.historical_provenance):
            ap.error("Local-proxy mode requires local-data-only configuration and no verified provenance file.")
        if not args.local_proxy and not args.historical_provenance:
            ap.error("Point-in-time canonicalization requires --historical-provenance.")
        from dao_governance.data.canonical_dataset import build_canonical_vote_dataset

        paths = cfg.get("paths", {})
        source = (base / paths.get("master_votes_parquet", "data/interim/master_votes.parquet")).resolve()
        metadata = (base / paths.get("proposal_metadata_jsonl", "data/interim/snapshot_proposal_metadata.jsonl")).resolve()
        output = (base / paths.get("canonical_vote_events_parquet", "data/processed/canonical_vote_events.parquet")).resolve()
        provenance = Path(args.historical_provenance) if args.historical_provenance else None
        if local_only:
            source = require_local_data_path(source, root=base)
            metadata = require_local_data_path(metadata, root=base)
            output = require_local_data_path(output, root=base)
            if provenance is not None:
                provenance = require_local_data_path(provenance, root=base)
        audit = build_canonical_vote_dataset(source, metadata, provenance, output, local_proxy=args.local_proxy)
        print(f"[03] Canonical vote events: {output}; provenance mode: {audit['provenance_mode']}; votes: {audit['vote_counts_and_exclusions']['eligible_votes']}")
        return
    out = run_global_cleaning_stage(cfg, base)
    print(f"[03] Cleaned votes: {out}")


if __name__ == "__main__":
    main()
