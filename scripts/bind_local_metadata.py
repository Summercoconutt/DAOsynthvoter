#!/usr/bin/env python3
"""Reconcile existing proposal metadata with votes under the local data directory."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.data.local_metadata import bind_local_metadata_cache
from dao_governance.settings import load_config, project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--extra-config", default="configs/local_data_only.yaml")
    args = parser.parse_args()
    base = project_root()
    config = (base / args.config).resolve()
    extra = (base / args.extra_config).resolve()
    cfg = load_config(config_path=config, extra_path=extra)
    if not bool((cfg.get("run") or {}).get("local_data_only", False)):
        parser.error("Local metadata binding requires local-data-only mode.")
    paths = cfg.get("paths", {})
    audit = bind_local_metadata_cache(
        (base / paths.get("master_votes_parquet", "data/interim/snapshot_votes.parquet")).resolve(),
        (base / paths.get("local_metadata_source_jsonl", "data/interim/snapshot_proposal_metadata.jsonl")).resolve(),
        (base / paths.get("proposal_metadata_jsonl", "data/interim/snapshot_proposal_metadata_local.jsonl")).resolve(),
        data_root=base / "data",
    )
    print(f"Local proposal binding: {audit['statuses']}; historical availability remains unverified.")


if __name__ == "__main__":
    main()
