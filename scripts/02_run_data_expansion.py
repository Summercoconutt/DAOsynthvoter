#!/usr/bin/env python3
"""Stage 2: Data expansion — Snapshot proposal/vote collection (see src/data_expansion)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.settings import load_config, project_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--extra-config", type=str, default="")
    ap.add_argument("--fetch-proposal-metadata", action="store_true")
    ap.add_argument("--resume-metadata", action="store_true")
    ap.add_argument("--metadata-limit", type=int, default=0, help="Limit proposal IDs for a pilot; 0 means all.")
    ap.add_argument("--metadata-batch-size", type=int, default=100)
    args = ap.parse_args()
    base = project_root()
    cfg_path = (base / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=cfg_path, extra_path=extra)
    if args.fetch_proposal_metadata and bool((cfg.get("run") or {}).get("local_data_only", False)):
        ap.error("Proposal metadata fetching is disabled in local-data-only mode.")
    paths = cfg.get("paths", {})
    rep = (base / cfg.get("reports", {}).get("stage02_md", "outputs/reports/stage02_data_expansion.md")).resolve()

    exp_dir = ROOT / "src" / "data_expansion"
    scripts = sorted(exp_dir.glob("*.py"))
    out_parquet = (base / paths.get("snapshot_votes_parquet", "data/interim/snapshot_votes.parquet")).resolve()

    lines = [
        "# Stage 2 — Data expansion",
        "",
        "Collect Snapshot proposal-level and vote-level data for selected DAOs; prefer **parquet** outputs.",
        "",
        "## Available drivers (copied into `src/data_expansion/`)",
        "",
    ]
    for p in scripts:
        lines.append(f"- `{p.relative_to(ROOT)}`")
    lines.extend(
        [
            "",
            "## Configuration",
            f"- Target merged votes path (typical): `{out_parquet}`",
            f"- Snapshot spaces root (set in config): `{paths.get('snapshot_spaces_root', '') or '(empty)'}`",
            "",
            "## External requirements",
            "- `dao_eligible_reconstruct_full.py` needs `RPC_URL` (and optionally `ETHERSCAN_API_KEY`).",
            "- Platform parquet builders may need API keys / manual paths — inspect each script header.",
            "",
            "## TODO",
            "- Wire your chosen expansion script to write **new** files under `data/interim/` without overwriting immutable raw archives.",
            "",
        ]
    )
    if args.fetch_proposal_metadata:
        from dao_governance.data.snapshot_metadata import collect_proposal_metadata

        source = (base / paths.get("master_votes_parquet", "data/interim/master_votes.parquet")).resolve()
        output = (base / paths.get("proposal_metadata_jsonl", "data/interim/snapshot_proposal_metadata.jsonl")).resolve()
        counts = collect_proposal_metadata(
            source,
            output,
            limit=args.metadata_limit,
            batch_size=args.metadata_batch_size,
            resume=args.resume_metadata,
        )
        lines.extend(
            [
                "## Public proposal metadata",
                f"- Cache: `{output}`",
                f"- Collection counts: `{counts}`",
                "- Current metadata is not, by itself, proof of historical availability.",
                "",
            ]
        )

    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"[02] Report: {rep}")


if __name__ == "__main__":
    main()
