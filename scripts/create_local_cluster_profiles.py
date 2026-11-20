#!/usr/bin/env python3
"""Prepare local-proxy clustering profiles before a training warm-up cutoff."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.clustering.local_profiles import prepare_local_cluster_profiles
from dao_governance.settings import load_config, project_root, require_local_data_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--extra-config", default="configs/local_data_only.yaml")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--fit-cutoff", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-events", type=int, default=2_000_000)
    args = parser.parse_args()
    base = project_root()
    cfg = load_config(config_path=(base / args.config).resolve(), extra_path=(base / args.extra_config).resolve())
    if not bool((cfg.get("run") or {}).get("local_data_only", False)):
        parser.error("Local cluster profiles require local-data-only configuration.")
    paths = cfg.get("paths", {})
    result = prepare_local_cluster_profiles(
        raw_source=require_local_data_path(Path(paths["master_votes_parquet"]), root=base),
        metadata_cache=require_local_data_path(Path(paths["proposal_metadata_jsonl"]), root=base),
        canonical_events=require_local_data_path(Path(paths["canonical_vote_events_parquet"]), root=base),
        split_manifest=require_local_data_path(Path(args.split_manifest), root=base),
        fit_cutoff=args.fit_cutoff,
        output_dir=require_local_data_path(Path(args.output_dir), root=base),
        max_events=args.max_events,
    )
    print(f"Local-proxy warm-up profiles: {result}")


if __name__ == "__main__":
    main()
