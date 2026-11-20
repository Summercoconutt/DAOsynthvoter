#!/usr/bin/env python3
"""Map an outcome-selected historical lean audit to original Snapshot IDs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.data.lean_sensitivity import build_lean_membership
from dao_governance.settings import load_config, project_root, require_local_data_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--extra-config", default="")
    parser.add_argument("--lean-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = project_root()
    config = (base / args.config).resolve()
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=config, extra_path=extra)
    paths = cfg.get("paths", {})
    source = (base / paths.get("master_votes_parquet", "data/interim/master_votes.parquet")).resolve()
    lean_audit = Path(args.lean_audit)
    output = Path(args.output)
    if bool((cfg.get("run") or {}).get("local_data_only", False)):
        source = require_local_data_path(source, root=base)
        lean_audit = require_local_data_path(lean_audit, root=base)
        output = require_local_data_path(output, root=base)
    result = build_lean_membership(source, lean_audit, output)
    print(f"Mapped outcome-selected lean proposals: {result['retained_proposals']}")


if __name__ == "__main__":
    main()
