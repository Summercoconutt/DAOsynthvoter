"""Load and merge YAML configuration with environment overrides."""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_DATA_SOURCES_CONFIG = "configs/data_sources.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(
    config_path: Path | None = None,
    extra_path: Path | None = None,
) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    default = root / "configs" / "default.yaml"
    path = Path(config_path) if config_path else default
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_sources_path = root / DEFAULT_DATA_SOURCES_CONFIG
    if not extra_path and data_sources_path.exists():
        with open(data_sources_path, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    if extra_path and Path(extra_path).exists():
        with open(extra_path, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    # Env overrides for common paths (optional)
    snap = os.environ.get("PIPELINE_SNAPSHOT_ROOT", "").strip()
    if snap:
        cfg.setdefault("paths", {})["snapshot_spaces_root"] = snap
    master = os.environ.get("PIPELINE_MASTER_PARQUET", "").strip()
    if master:
        cfg.setdefault("paths", {})["master_votes_parquet"] = master
    dao = os.environ.get("PIPELINE_DAO_ASSIGNMENTS_CSV", "").strip()
    if dao:
        cfg.setdefault("paths", {})["dao_cluster_assignments_csv"] = dao

    return cfg


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(cfg: Dict[str, Any], key: str, relative_to: Path | None = None) -> Path:
    """Resolve a path string from cfg['paths'][key] relative to project root."""
    base = relative_to or project_root()
    rel = (cfg.get("paths") or {}).get(key) or ""
    p = Path(rel)
    if p.is_absolute():
        return p
    return (base / p).resolve()
