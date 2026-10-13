#!/usr/bin/env python3
"""Validate and package one immutable Stage 08/08b run directory."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dao_governance.modelling.run_artifacts import (  # noqa: E402
    read_manifest,
    sha256_file,
    validate_run_id,
    verify_recorded_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory containing run_manifest.json")
    parser.add_argument("destination", help="Destination ZIP path or directory")
    parser.add_argument(
        "--allow-without-evaluation",
        action="store_true",
        help="Permit a model-only bundle before paired Stage 09 evaluation is complete.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir.resolve()
    manifest = read_manifest(run_dir)
    run_id = validate_run_id(manifest.get("run_id", ""))
    variant = manifest.get("sample_variant")
    if variant not in {"full", "lean"}:
        raise RuntimeError(f"Invalid sample variant in {run_dir / 'run_manifest.json'}")
    if run_dir.name != run_id or run_dir.parent.name != variant:
        raise RuntimeError("Run directory must be <runs-root>/<variant>/<run-id>")
    required = {"08", "08b"}
    if not args.allow_without_evaluation:
        required.add("evaluation")
    missing = required.difference(manifest.get("stages", {}))
    if missing:
        raise RuntimeError(f"Run is incomplete; missing recorded stage(s): {', '.join(sorted(missing))}")
    if manifest.get("status") != "ready":
        raise RuntimeError(f"Run is not marked ready: {manifest.get('status')!r}")
    verify_recorded_artifacts(run_dir, manifest)
    required_files = [
        run_dir / "run_manifest.json",
        run_dir / "config" / "resolved.yaml",
        run_dir / "config" / "data_sources.yaml",
        run_dir / "inputs" / "behaviour_dataset.csv",
        run_dir / "shared" / "split_manifest.json",
        run_dir / "08" / "model.pt",
        run_dir / "08" / "config.json",
        run_dir / "08" / "tokenizer" / "tokenizer.json",
        run_dir / "08b" / "model.pt",
        run_dir / "08b" / "config.json",
    ]
    if not args.allow_without_evaluation:
        required_files.extend(
            [
                run_dir / "evaluation" / "08" / "test" / "metrics_summary.json",
                run_dir / "evaluation" / "08b" / "test" / "metrics_summary.json",
            ]
        )
    missing_files = [str(path.relative_to(run_dir)) for path in required_files if not path.is_file()]
    if missing_files:
        raise RuntimeError(f"Run is missing required export files: {', '.join(missing_files)}")

    destination = Path(args.destination)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination = destination.resolve()
    bundle_name = f"stage08_08b_{variant}_{run_id}"
    if destination.suffix.lower() == ".zip":
        zip_path = destination
    else:
        zip_path = destination / f"{bundle_name}.zip"
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {zip_path}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    export_files = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        export_files.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    export_manifest = {
        "run_id": run_id,
        "sample_variant": variant,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_config_sha256": manifest.get("source_config", {}).get("resolved_config_sha256"),
        "files": export_files,
    }

    with tempfile.NamedTemporaryFile(prefix=f"{bundle_name}.", suffix=".zip", dir=zip_path.parent, delete=False) as tmp:
        temp_zip = Path(tmp.name)
    try:
        with ZipFile(temp_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
                archive.write(path, f"{bundle_name}/{path.relative_to(run_dir).as_posix()}")
            archive.writestr(
                f"{bundle_name}/export_manifest.json",
                json.dumps(export_manifest, indent=2, sort_keys=True),
            )
        os.link(temp_zip, zip_path)
    finally:
        if temp_zip.exists():
            temp_zip.unlink()
    print(f"Created validated bundle: {zip_path}")


if __name__ == "__main__":
    main()
