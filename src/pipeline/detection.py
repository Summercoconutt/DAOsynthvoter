"""Readiness checks for behaviour modelling + optional raw-data diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dao_governance.settings import resolve_path


VOTES_NEED_MERGE = ["voter", "space", "proposal_id", "choice_norm"]
MERGED_EXTRA = ["dao_cluster", "vote_timestamp", "vote_ts", "proposal_title", "voting_power"]
ASSIGN_NEED = ["voter", "space", "voter_cluster"]


@dataclass
class ScanResult:
    path: str
    rows: int
    cols: List[str]
    vp_column: Optional[str]
    vp_neg: int = 0
    vp_nan: int = 0
    vp_inf: int = 0
    vp_min: Optional[float] = None
    vp_max: Optional[float] = None
    notes: List[str] = field(default_factory=list)


def _find_vp_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("voting_power", "vp", "Voting Power", "voting power"):
        if c in df.columns:
            return c
    return None


def scan_parquet_quick(path: Path, *, max_rows: int = 80_000) -> ScanResult:
    path = path.resolve()
    df = pd.read_parquet(path)
    n_full = len(df)
    if n_full > max_rows:
        df = df.sample(max_rows, random_state=0)
    vp_col = _find_vp_column(df)
    sr = ScanResult(path=str(path), rows=n_full, cols=list(df.columns), vp_column=vp_col)
    if vp_col is None:
        sr.notes.append("No voting_power/vp column found.")
        return sr
    vp = pd.to_numeric(df[vp_col], errors="coerce")
    sr.vp_neg = int((vp < 0).sum())
    sr.vp_nan = int(vp.isna().sum())
    sr.vp_inf = int(np.isinf(vp.to_numpy(dtype=float, copy=False)).sum())
    finite = vp.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite):
        sr.vp_min = float(finite.min())
        sr.vp_max = float(finite.max())
    return sr


def scan_csv_quick(path: Path, *, max_rows: int = 80_000) -> ScanResult:
    df = pd.read_csv(path, low_memory=False, nrows=max_rows if max_rows else None)
    vp_col = _find_vp_column(df)
    sr = ScanResult(path=str(path.resolve()), rows=len(df), cols=list(df.columns), vp_column=vp_col)
    if vp_col is None:
        sr.notes.append("No voting_power/vp column found.")
        return sr
    vp = pd.to_numeric(df[vp_col], errors="coerce")
    sr.vp_neg = int((vp < 0).sum())
    sr.vp_nan = int(vp.isna().sum())
    sr.vp_inf = int(np.isinf(vp.to_numpy(dtype=float, copy=False)).sum())
    finite = vp.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite):
        sr.vp_min = float(finite.min())
        sr.vp_max = float(finite.max())
    return sr


def iter_data_files(root: Path, glob_pat: str, *, limit: int) -> List[Path]:
    if not root.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(root.glob(glob_pat)):
        if p.is_file() and p.suffix.lower() in {".parquet", ".csv"}:
            out.append(p)
            if len(out) >= limit:
                break
    return out


def check_behaviour_prerequisites(cfg: Dict[str, Any], root: Path) -> Tuple[bool, List[str]]:
    """Returns (all_ok, lines)."""
    lines: List[str] = []
    ok = True

    merged = resolve_path(cfg, "master_with_dao_parquet", relative_to=root)
    assign = resolve_path(cfg, "voter_cluster_assignments_csv", relative_to=root)

    if not merged.exists():
        ok = False
        lines.append(f"MISSING master_with_dao_parquet: `{merged}`")
    else:
        lines.append(f"OK merged votes: `{merged}`")
        try:
            df = pd.read_parquet(merged)
            miss = [c for c in VOTES_NEED_MERGE if c not in df.columns]
            if miss:
                ok = False
                lines.append(f"  FAIL columns missing: {miss}")
            else:
                lines.append("  OK core columns for behaviour_dataset.")
            miss2 = [c for c in MERGED_EXTRA if c not in df.columns]
            if miss2:
                lines.append(f"  WARN optional columns absent (may break features): {miss2}")
            n = len(df)
            lines.append(f"  rows={n:,}")
            if n < 500:
                lines.append("  WARN very small — smoke train may fail windowing.")
        except Exception as exc:
            ok = False
            lines.append(f"  FAIL read parquet: {exc}")

    if not assign.exists():
        ok = False
        lines.append(f"MISSING voter_cluster_assignments_csv: `{assign}`")
    else:
        lines.append(f"OK voter clusters: `{assign}`")
        try:
            a = pd.read_csv(assign)
            miss = [c for c in ASSIGN_NEED if c not in a.columns]
            if miss:
                ok = False
                lines.append(f"  FAIL assignment columns missing: {miss}")
        except Exception as exc:
            ok = False
            lines.append(f"  FAIL read CSV: {exc}")

    cleaned = resolve_path(cfg, "cleaned_master_parquet", relative_to=root)
    if cleaned.exists():
        lines.append(f"OK cleaned votes exist: `{cleaned}`")
    else:
        lines.append(f"INFO cleaned_master_parquet not found yet: `{cleaned}` (run stage 03 or detection clean).")

    return ok, lines


def render_scan_results(results: List[ScanResult]) -> List[str]:
    lines: List[str] = []
    for sr in results:
        lines.append(f"### `{sr.path}`")
        lines.append(f"- rows (file): **{sr.rows:,}**")
        lines.append(f"- vp column: **{sr.vp_column or 'NOT FOUND'}**")
        if sr.vp_column:
            lines.append(f"- vp negatives: **{sr.vp_neg}** | nan: **{sr.vp_nan}** | inf: **{sr.vp_inf}**")
            lines.append(f"- vp min/max (sample): **{sr.vp_min}** / **{sr.vp_max}**")
        for n in sr.notes:
            lines.append(f"- {n}")
        lines.append("")
    return lines
