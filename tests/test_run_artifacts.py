from pathlib import Path

import pytest

from dao_governance.modelling.run_artifacts import (
    create_run,
    record_stage,
    resolve_run_root,
    resolve_stage_inputs,
    verify_recorded_artifacts,
)


def test_run_roots_are_variant_and_id_scoped(tmp_path: Path) -> None:
    full = resolve_run_root(tmp_path, {"run": {"sample_variant": "full"}}, "trial-1")
    lean = resolve_run_root(tmp_path, {"run": {"sample_variant": "lean"}}, "trial-1")
    assert full != lean
    assert full == tmp_path / "outputs/stage08_runs/full/trial-1"


def test_manifest_refuses_reusing_run_root_and_verifies_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "data_sources.yaml"
    config_path.write_text("paths: {}\n", encoding="utf-8")
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source")
    run_root = tmp_path / "outputs/stage08_runs/full/run-a"
    config = {"run": {"sample_variant": "full"}}

    manifest = create_run(
        run_root,
        run_id="run-a",
        sample_variant="full",
        config=config,
        config_path=config_path,
        extra_config_path=None,
        input_paths={"master_votes": source},
    )
    artifact = run_root / "08" / "config.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    record_stage(run_root, manifest, "08", [artifact])
    verify_recorded_artifacts(run_root, manifest)

    with pytest.raises(FileExistsError):
        create_run(
            run_root,
            run_id="run-a",
            sample_variant="full",
            config=config,
            config_path=config_path,
            extra_config_path=None,
            input_paths={"master_votes": source},
        )

    artifact.write_text("!!", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Hash mismatch"):
        verify_recorded_artifacts(run_root, manifest)


def test_run_id_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_run_root(tmp_path, {"run": {"sample_variant": "full"}}, "../outside")


def test_disabled_dao_clusters_do_not_require_a_feature_table(tmp_path: Path) -> None:
    source = tmp_path / "votes.parquet"
    source.write_bytes(b"source")
    cfg = {
        "paths": {"cleaned_master_parquet": str(source)},
        "behaviour_model": {"use_dao_clusters": False, "use_voter_clusters": False},
    }
    assert resolve_stage_inputs(tmp_path, cfg) == {"cleaned_master_parquet": source}
    cfg["behaviour_model"]["use_dao_clusters"] = True
    with pytest.raises(FileNotFoundError, match="DAO feature table"):
        resolve_stage_inputs(tmp_path, cfg)