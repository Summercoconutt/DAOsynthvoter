from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.settings import load_config, require_local_data_path


def test_data_sources_override_is_merged_by_default() -> None:
    cfg = load_config(config_path=ROOT / "configs" / "default.yaml")

    assert cfg["paths"]["master_votes_parquet"] == "/home/carlo/data/summer_voters/snapshot_votes_441/outputs/DAO representative_20260326_153850/merged_votes_10daos.parquet"
    assert cfg["paths"]["snapshot_spaces_root"] == "/home/carlo/data/summer_voters/snapshot_votes_441/spaces"


def test_explicit_lean_sources_and_separate_model_override() -> None:
    cfg = load_config(
        config_path=ROOT / "configs" / "default.yaml",
        data_sources_path=ROOT / "configs" / "data_sources_lean.yaml",
        extra_path=ROOT / "configs" / "smoke_behaviour.yaml",
    )

    assert cfg["run"]["sample_variant"] == "lean"
    assert cfg["paths"]["cleaned_master_parquet"] == "data/processed/votes_cleaned_lean.parquet"
    assert cfg["behaviour_model"]["window"] == 3


def test_early_stages_accept_overrides_and_parse_selected_space_ids(tmp_path: Path) -> None:
    selection = tmp_path / "spaces.json"
    selection.write_text(json.dumps({"selected_space_ids": ["one", "two"]}), encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text("paths: {}\nreports: {}\n", encoding="utf-8")
    override = tmp_path / "override.yaml"
    report1 = tmp_path / "stage01.md"
    report2 = tmp_path / "stage02.md"
    override.write_text(
        yaml.safe_dump(
            {
                "paths": {"selected_dao_json": str(selection)},
                "reports": {"stage01_md": str(report1), "stage02_md": str(report2)},
            }
        ),
        encoding="utf-8",
    )
    for script in ("01_run_data_selection.py", "02_run_data_expansion.py"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--config", str(base), "--extra-config", str(override)],
            check=True,
            capture_output=True,
            text=True,
        )
    assert "Parsed **2** DAO ids" in report1.read_text(encoding="utf-8")
    assert report2.is_file()


def test_legacy_runners_require_explicit_opt_in() -> None:
    env = os.environ.copy()
    env.pop("ALLOW_LEGACY_VOTER_SPLIT", None)
    shell = subprocess.run(
        ["bash", str(ROOT / "run_all.sh")], capture_output=True, text=True, env=env
    )
    assert shell.returncode != 0
    assert "non-chronological voter split" in shell.stderr
    python = subprocess.run(
        [sys.executable, str(ROOT / "run_full_pipeline.py"), "--from-stage", "8", "--to-stage", "8"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert python.returncode != 0
    assert "legacy voter split" in python.stderr
    detection = subprocess.run(
        [sys.executable, str(ROOT / "scripts/10_run_detection_mode.py"), "--smoke-train"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert detection.returncode != 0
    assert "legacy voter split" in detection.stderr


def test_chronological_training_requires_explicit_confirmation() -> None:
    command = [
        sys.executable, str(ROOT / "scripts/08_run_chronological_model.py"),
        "--historical-provenance", "missing.json", "--observations", "missing.parquet",
        "--observation-provenance", "missing-observations.json", "--examples-dir", "missing-examples",
        "--run-id", "test-run",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "--confirm-full-training" in result.stderr


def test_local_data_only_mode_rejects_proposal_fetching(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("paths: {}\nreports: {}\n", encoding="utf-8")
    local = ROOT / "configs/local_data_only.yaml"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/02_run_data_expansion.py"),
         "--config", str(config), "--extra-config", str(local), "--fetch-proposal-metadata"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "disabled in local-data-only mode" in result.stderr


def test_local_data_path_rejects_external_and_symlinked_inputs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    local = data / "votes.parquet"
    local.write_bytes(b"local")
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"outside")
    assert require_local_data_path(local, root=tmp_path) == local
    with pytest.raises(ValueError, match="under the data directory"):
        require_local_data_path(outside, root=tmp_path)
    link = data / "external-link.parquet"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="under the data directory"):
        require_local_data_path(link, root=tmp_path)


@pytest.mark.parametrize("stage", ("06_run_dao_clustering.py", "07_run_voter_clustering.py"))
def test_predictive_clustering_honors_local_only_base_config(stage: str, tmp_path: Path) -> None:
    config = tmp_path / "local.yaml"
    config.write_text("run:\n  local_data_only: true\n", encoding="utf-8")
    command = [
        sys.executable, str(ROOT / "scripts" / stage), "--config", str(config),
        "--predictive-profiles", str(tmp_path / "outside.parquet"),
        "--split-manifest", str(tmp_path / "split.json"),
        "--cluster-output", str(tmp_path / "clusters.pkl"),
        "--cluster-feature", "dao_prior_count",
    ]
    if stage.startswith("07"):
        command.extend(["--cluster-scope", "global"])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "under the data directory" in result.stderr
