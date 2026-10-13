import os
import subprocess
import sys
import zipfile
from pathlib import Path

from dao_governance.modelling.run_artifacts import create_run, record_stage


ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = ROOT / "scripts" / "export_stage08_run.py"


def test_export_validates_and_preserves_run_tree(tmp_path: Path) -> None:
    source_config = tmp_path / "data_sources.yaml"
    source_config.write_text("run:\n  sample_variant: full\n", encoding="utf-8")
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source")
    run_root = tmp_path / "outputs/stage08_runs/full/export-test"
    config = {"run": {"sample_variant": "full"}}
    manifest = create_run(
        run_root,
        run_id="export-test",
        sample_variant="full",
        config=config,
        config_path=source_config,
        extra_config_path=None,
        data_sources_config_path=source_config,
        input_paths={"cleaned_master_parquet": source},
    )

    required_artifacts = [
        run_root / "inputs/behaviour_dataset.csv",
        run_root / "shared/split_manifest.json",
        run_root / "08/model.pt",
        run_root / "08/config.json",
        run_root / "08/tokenizer/tokenizer.json",
        run_root / "08/reports/report.md",
        run_root / "08b/model.pt",
        run_root / "08b/config.json",
        run_root / "08b/reports/report.md",
    ]
    for index, artifact in enumerate(required_artifacts):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"artifact-{index}\n", encoding="utf-8")
    record_stage(run_root, manifest, "08", [run_root / "08", run_root / "inputs", run_root / "shared"])
    record_stage(run_root, manifest, "08b", [run_root / "08b"])

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    model_only_destination = tmp_path / "model-only-export"
    model_only_command = [
        sys.executable,
        str(EXPORT_SCRIPT),
        str(run_root),
        str(model_only_destination),
        "--allow-without-evaluation",
    ]
    model_only = subprocess.run(model_only_command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert model_only.returncode == 0, model_only.stderr
    assert (model_only_destination / "stage08_08b_full_export-test.zip").is_file()

    for index, artifact in enumerate(
        [
            run_root / "evaluation/08/test/metrics_summary.json",
            run_root / "evaluation/08b/test/metrics_summary.json",
        ]
    ):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"evaluation-{index}\n", encoding="utf-8")
    record_stage(
        run_root,
        manifest,
        "evaluation",
        [run_root / "evaluation"],
    )

    destination = tmp_path / "exports"
    command = [sys.executable, str(EXPORT_SCRIPT), str(run_root), str(destination)]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    archive_path = destination / "stage08_08b_full_export-test.zip"
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "stage08_08b_full_export-test/run_manifest.json" in names
    assert "stage08_08b_full_export-test/08/reports/report.md" in names
    assert "stage08_08b_full_export-test/08b/reports/report.md" in names
    assert "stage08_08b_full_export-test/evaluation/08/test/metrics_summary.json" in names
    assert "stage08_08b_full_export-test/export_manifest.json" in names

    repeated = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "Refusing to overwrite existing export" in repeated.stderr
