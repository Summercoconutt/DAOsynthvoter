"""Run-scoped artifact paths and manifests for behaviour modelling."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MANIFEST_NAME = "run_manifest.json"


def validate_run_id(run_id: str) -> str:
    value = str(run_id).strip()
    if not _RUN_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError("run_id must contain only letters, digits, '.', '_' or '-' and start alphanumeric")
    return value


def resolve_run_root(base: Path, cfg: Dict[str, Any], run_id: str) -> Path:
    run_id = validate_run_id(run_id)
    run_cfg = cfg.get("run") or {}
    variant = str(run_cfg.get("sample_variant", "full")).strip().lower()
    if variant not in {"full", "lean"}:
        raise ValueError(f"run.sample_variant must be 'full' or 'lean', got {variant!r}")
    root_value = (cfg.get("paths") or {}).get("stage08_runs_root", "outputs/stage08_runs")
    root = Path(root_value)
    if not root.is_absolute():
        root = base / root
    return root.resolve() / variant / run_id


def resolve_artifact_path(run_root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (run_root / path).resolve()
    try:
        resolved.relative_to(run_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Artifact path must remain inside the run directory: {value}") from exc
    return resolved


def resolve_stage_inputs(base: Path, cfg: Dict[str, Any]) -> Dict[str, Path]:
    paths = cfg.get("paths") or {}

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (base / path).resolve()

    cleaned_path = resolve(paths.get("cleaned_master_parquet", "data/processed/votes_cleaned.parquet"))
    if not cleaned_path.is_file():
        raise FileNotFoundError(
            "Configured paths.cleaned_master_parquet does not exist: "
            f"{cleaned_path}. Stage 08 will not fall back to another dataset."
        )

    if not bool((cfg.get("behaviour_model") or {}).get("use_dao_clusters", True)):
        return {"cleaned_master_parquet": cleaned_path}

    dao_csv = resolve(paths.get("dao_feature_table_csv", "data/processed/dao_feature_table.csv"))
    dao_parquet = resolve(paths.get("dao_feature_table_parquet", "data/processed/dao_feature_table.parquet"))
    dao_path = dao_csv if dao_csv.is_file() else dao_parquet
    if not dao_path.is_file():
        raise FileNotFoundError(f"Configured DAO feature table is missing: {dao_csv} or {dao_parquet}")
    return {"cleaned_master_parquet": cleaned_path, "dao_feature_table": dao_path}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_record(path: Path, *, root: Path | None = None) -> Dict[str, Any]:
    resolved = path.resolve()
    record: Dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if root is not None:
        try:
            record["relative_path"] = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return record


def _runtime_metadata(repo_root: Path) -> Dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def create_run(
    run_root: Path,
    *,
    run_id: str,
    sample_variant: str,
    config: Dict[str, Any],
    config_path: Path,
    extra_config_path: Path | None,
    input_paths: Dict[str, Path],
    data_sources_config_path: Path | None = None,
) -> Dict[str, Any]:
    run_id = validate_run_id(run_id)
    if sample_variant not in {"full", "lean"}:
        raise ValueError(f"Unsupported sample variant: {sample_variant!r}")
    inputs: Dict[str, Any] = {}
    for name, path in input_paths.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Configured Stage 08 input '{name}' does not exist: {resolved}")
        inputs[name] = file_record(resolved)

    source_config = data_sources_config_path or extra_config_path or config_path
    if not source_config.is_file():
        raise FileNotFoundError(f"Data-source config not found: {source_config}")
    if extra_config_path and not extra_config_path.is_file():
        raise FileNotFoundError(f"Extra config not found: {extra_config_path}")
    run_root.mkdir(parents=True, exist_ok=False)
    config_dir = run_root / "config"
    config_dir.mkdir(parents=True)
    resolved_config_path = config_dir / "resolved.yaml"
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    copied_source_config = config_dir / "data_sources.yaml"
    copied_source_config.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    config_files = [
        file_record(resolved_config_path, root=run_root),
        file_record(copied_source_config, root=run_root),
    ]
    if extra_config_path and extra_config_path.resolve() != source_config.resolve():
        extra_copy = config_dir / "extra.yaml"
        extra_copy.write_text(extra_config_path.read_text(encoding="utf-8"), encoding="utf-8")
        config_files.append(file_record(extra_copy, root=run_root))

    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "run_id": run_id,
        "sample_variant": sample_variant,
        "status": "initialized",
        "source_config": {
            "config_path": str(config_path.resolve()),
            "extra_config_path": str(extra_config_path.resolve()) if extra_config_path else None,
            "resolved_config_sha256": sha256_json(config),
            "files": config_files,
        },
        "runtime": _runtime_metadata(config_path.resolve().parents[1]),
        "inputs": inputs,
        "stages": {},
    }
    write_manifest(run_root, manifest)
    return manifest


def read_manifest(run_root: Path) -> Dict[str, Any]:
    path = run_root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(run_root: Path, manifest: Dict[str, Any]) -> None:
    (run_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def record_input(run_root: Path, manifest: Dict[str, Any], name: str, path: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Cannot record missing input: {resolved}")
    record = file_record(resolved, root=run_root)
    existing = (manifest.get("inputs") or {}).get(name)
    if existing and existing.get("sha256") != record["sha256"]:
        raise RuntimeError(f"Input '{name}' changed within run {manifest.get('run_id')!r}")
    manifest.setdefault("inputs", {})[name] = record
    write_manifest(run_root, manifest)


def verify_run_identity(
    run_root: Path,
    *,
    run_id: str,
    sample_variant: str,
    config: Dict[str, Any],
    input_paths: Dict[str, Path] | None = None,
) -> Dict[str, Any]:
    manifest = read_manifest(run_root)
    if manifest.get("run_id") != validate_run_id(run_id):
        raise RuntimeError(f"Run ID mismatch in {run_root / MANIFEST_NAME}")
    if manifest.get("sample_variant") != sample_variant:
        raise RuntimeError(
            f"Sample variant mismatch: manifest={manifest.get('sample_variant')!r}, requested={sample_variant!r}"
        )
    if manifest.get("source_config", {}).get("resolved_config_sha256") != sha256_json(config):
        raise RuntimeError("Resolved config differs from the config recorded when this run was created")
    if input_paths:
        for name, path in input_paths.items():
            recorded = (manifest.get("inputs") or {}).get(name) or {}
            if not path.is_file() or recorded.get("sha256") != sha256_file(path.resolve()):
                raise RuntimeError(f"Configured input '{name}' no longer matches this run manifest")
    return manifest


def record_stage(
    run_root: Path,
    manifest: Dict[str, Any],
    stage_name: str,
    artifacts: Iterable[Path],
    *,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if stage_name in (manifest.get("stages") or {}):
        raise RuntimeError(f"Stage {stage_name!r} is already recorded for run {manifest.get('run_id')!r}")
    records = []
    for path in artifacts:
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Cannot record missing artifact: {resolved}")
        if resolved.is_file():
            records.append(file_record(resolved, root=run_root))
        else:
            for child in sorted(p for p in resolved.rglob("*") if p.is_file()):
                records.append(file_record(child, root=run_root))
    manifest.setdefault("stages", {})[stage_name] = {"artifacts": records, **(details or {})}
    manifest["status"] = "ready" if {"08", "08b"}.issubset(manifest["stages"]) else "in_progress"
    write_manifest(run_root, manifest)
    return manifest


def verify_recorded_artifacts(run_root: Path, manifest: Dict[str, Any]) -> None:
    records = list((manifest.get("source_config") or {}).get("files", []))
    records.extend((manifest.get("inputs") or {}).values())
    for stage in (manifest.get("stages") or {}).values():
        records.extend(stage.get("artifacts", []))
    for record in records:
        relative = record.get("relative_path")
        if not relative:
            continue
        path = run_root / relative
        if not path.is_file() or path.stat().st_size != record.get("size_bytes"):
            raise RuntimeError(f"Missing or size-mismatched artifact: {relative}")
        if sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"Hash mismatch: {relative}")