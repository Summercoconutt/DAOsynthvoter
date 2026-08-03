from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.settings import load_config


def test_data_sources_override_is_merged_by_default() -> None:
    cfg = load_config(config_path=ROOT / "configs" / "default.yaml")

    assert cfg["paths"]["master_votes_parquet"] == "/home/carlo/data/summer_voters/snapshot_votes_441/outputs/DAO representative_20260326_153850/merged_votes_10daos.parquet"
    assert cfg["paths"]["snapshot_spaces_root"] == "/home/carlo/data/summer_voters/snapshot_votes_441/spaces"
