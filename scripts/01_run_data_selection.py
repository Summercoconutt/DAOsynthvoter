#!/usr/bin/env python3
"""Stage 1: DAO selection — validate follower-based knee selection artefacts and write report."""

from __future__ import annotations

import argparse
import json
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
    args = ap.parse_args()
    base = project_root()
    cfg_path = (base / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(config_path=cfg_path)
    paths = cfg.get("paths", {})
    rep = (base / cfg.get("reports", {}).get("stage01_md", "outputs/reports/stage01_dao_selection.md")).resolve()

    sel = (base / paths.get("selected_dao_json", "data/raw/dao_selection/selected_spaces.json")).resolve()
    uni = base / "data" / "raw" / "dao_selection" / "spaces_universe.csv"
    legacy_nb = Path(__file__).resolve().parents[2] / "data_collection" / "fetch_spaces.py.ipynb"

    lines = [
        "# Stage 1 — DAO selection",
        "",
        "This stage documents the **follower-based knee** DAO universe and the final selected list.",
        "",
        "## Inputs",
        f"- Selected DAO list (JSON): `{sel}`",
        f"- Universe table (optional): `{uni}`",
        "",
    ]

    if sel.exists():
        data = json.loads(sel.read_text(encoding="utf-8"))
        if isinstance(data, list):
            spaces = data
        elif isinstance(data, dict) and "spaces" in data:
            spaces = data["spaces"]
        else:
            spaces = []
        lines.extend(
            [
                "## Validation",
                f"- Parsed **{len(spaces)}** DAO ids from JSON.",
                "",
                "### Sample (first 15)",
                "",
            ]
        )
        for s in spaces[:15]:
            lines.append(f"- `{s}`")
        lines.append("")
    else:
        lines.append("## Validation\n\n**WARN**: selected_spaces JSON missing — copy from `Dissertation/data_collection/`.\n")

    lines.extend(
        [
            "## Provenance",
            f"- Original exploratory notebook (not executed here): `{legacy_nb}`",
            "- TODO: If you re-run knee detection, update `selected_spaces.json` and archive the notebook output.",
            "",
        ]
    )

    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"[01] Report: {rep}")


if __name__ == "__main__":
    main()
