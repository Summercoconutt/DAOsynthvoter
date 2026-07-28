"""Build and load on-disk window caches for large behaviour datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

from dao_governance.modelling.preprocess import normalise_columns, select_numeric_columns
from dao_governance.modelling.windows import (
    WINDOW_GROUP_COLS,
    _eval_cluster_id,
    _row_numeric_vec,
    _time_feats,
)


READ_CSV_OPTS = "header=true, ignore_errors=true, strict_mode=false, quote='\"'"

SLIM_COLUMNS = [
    "voter",
    "space",
    "vote_ts",
    "label_id",
    "voting_power",
    "is_whale",
    "dao_cluster",
    "voter_cluster",
]

SLIM_SELECT = """
    CAST(voter AS VARCHAR) AS voter,
    CAST(space AS VARCHAR) AS space,
    vote_ts AS vote_ts,
    label_id AS label_id,
    voting_power AS voting_power,
    is_whale AS is_whale,
    dao_cluster AS dao_cluster,
    voter_cluster AS voter_cluster
"""


def _register_voters(con: duckdb.DuckDBPyConnection, voters: List[str], table_name: str) -> None:
    con.register(table_name, pd.DataFrame({"voter": voters}))


def _count_windows(
    con: duckdb.DuckDBPyConnection,
    csv_path: Path,
    voters_table: str,
    window_size: int,
) -> int:
    q = f"""
    SELECT COALESCE(SUM(
        CASE WHEN cnt >= {window_size} THEN cnt - {window_size} + 1 ELSE 0 END
    ), 0)::BIGINT AS n_windows
    FROM (
        SELECT voter, space, COUNT(*) AS cnt
        FROM read_csv(?, {READ_CSV_OPTS})
        WHERE label_id IN (0, 1, 2)
          AND voter IN (SELECT voter FROM {voters_table})
        GROUP BY voter, space
    )
    """
    return int(con.execute(q, [str(csv_path)]).fetchone()[0])


def _windows_from_voter_group(
    g: pd.DataFrame,
    window_size: int,
    numeric_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = g.sort_values("vote_ts").reset_index(drop=True)
    if len(g) < window_size:
        return (
            np.empty((0, window_size, len(numeric_cols) + 4), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    feats_list: List[np.ndarray] = []
    labels: List[int] = []
    dao_cs: List[int] = []
    voter_cs: List[int] = []

    for t in range(window_size - 1, len(g)):
        indices = list(range(t - window_size + 1, t + 1))
        step_feats: List[np.ndarray] = []
        for idx in indices:
            row = g.loc[idx]
            is_cur = idx == t
            vec = _row_numeric_vec(row, numeric_cols, is_current_step=is_cur)
            arr = np.asarray(vec, dtype=np.float32)
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite feature vector for voter={row['voter']}")
            step_feats.append(arr)

        cur = g.loc[t]
        feats_list.append(np.stack(step_feats, axis=0))
        labels.append(int(cur["label_id"]))
        dao_cs.append(_eval_cluster_id(cur, "dao_cluster"))
        voter_cs.append(_eval_cluster_id(cur, "voter_cluster"))

    return (
        np.stack(feats_list, axis=0),
        np.asarray(labels, dtype=np.int64),
        np.asarray(dao_cs, dtype=np.int64),
        np.asarray(voter_cs, dtype=np.int64),
    )


def materialize_split_cache(
    *,
    csv_path: Path,
    voters: List[str],
    split_name: str,
    cache_dir: Path,
    preprocessor: Dict[str, Any],
    window_size: int,
    con: duckdb.DuckDBPyConnection,
    chunk_rows: int = 200_000,
) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = select_numeric_columns()
    feat_dim = len(numeric_cols) + 4

    voters_table = f"voters_{split_name}"
    _register_voters(con, voters, voters_table)

    n_windows = _count_windows(con, csv_path, voters_table, window_size)
    print(f"[cache] {split_name}: {n_windows} windows (window={window_size}, feat_dim={feat_dim})")
    if n_windows == 0:
        raise RuntimeError(f"No windows for split={split_name}")

    num_path = cache_dir / f"{split_name}_num_feats.dat"
    lbl_path = cache_dir / f"{split_name}_labels.dat"
    dao_path = cache_dir / f"{split_name}_dao_clusters.dat"
    voter_path = cache_dir / f"{split_name}_voter_clusters.dat"

    num_mm = np.memmap(num_path, dtype=np.float32, mode="w+", shape=(n_windows, window_size, feat_dim))
    lbl_mm = np.memmap(lbl_path, dtype=np.int64, mode="w+", shape=(n_windows,))
    dao_mm = np.memmap(dao_path, dtype=np.int64, mode="w+", shape=(n_windows,))
    voter_mm = np.memmap(voter_path, dtype=np.int64, mode="w+", shape=(n_windows,))

    write_idx = 0
    carry_over = pd.DataFrame(columns=SLIM_COLUMNS)

    rel = con.execute(
        f"""
                SELECT
                        {SLIM_SELECT}
        FROM read_csv(?, {READ_CSV_OPTS})
        WHERE label_id IN (0, 1, 2)
          AND voter IN (SELECT voter FROM {voters_table})
        ORDER BY voter, space, vote_ts
        """,
        [str(csv_path)],
    )

    def _flush_group(parts: List[pd.DataFrame]) -> None:
        nonlocal write_idx
        if not parts:
            return
        g = normalise_columns(pd.concat(parts, ignore_index=True), preprocessor=preprocessor)
        feats, labels, dao_cs, voter_cs = _windows_from_voter_group(g, window_size, numeric_cols)
        n = len(labels)
        if n == 0:
            return
        num_mm[write_idx : write_idx + n] = feats
        lbl_mm[write_idx : write_idx + n] = labels
        dao_mm[write_idx : write_idx + n] = dao_cs
        voter_mm[write_idx : write_idx + n] = voter_cs
        write_idx += n

    while True:
        chunk = rel.fetch_df_chunk(chunk_rows)
        if chunk is None or chunk.empty:
            break

        if not carry_over.empty:
            chunk = pd.concat([carry_over, chunk], ignore_index=True)
            carry_over = pd.DataFrame(columns=SLIM_COLUMNS)

        # Defensive normalization in case mixed-case CSV labels are returned.
        chunk.columns = [str(c).strip().lower() for c in chunk.columns]
        required = ["voter", "space", "vote_ts", "label_id"]
        missing = [c for c in required if c not in chunk.columns]
        if missing:
            raise KeyError(
                f"Window cache input missing columns {missing}. "
                f"Available columns: {list(chunk.columns)}"
            )

        chunk["voter"] = chunk["voter"].astype(str)
        chunk["space"] = chunk["space"].astype(str)
        group_keys = list(WINDOW_GROUP_COLS)
        boundaries: List[Tuple[str, str]] = []
        prev = None
        for _, row in chunk.iterrows():
            key = (row["voter"], row["space"])
            if prev is not None and key != prev:
                boundaries.append(prev)
            prev = key
        boundaries.append(prev)

        for i, key in enumerate(boundaries[:-1]):
            mask = (chunk["voter"] == key[0]) & (chunk["space"] == key[1])
            _flush_group([chunk.loc[mask]])

        last_key = boundaries[-1]
        carry_over = chunk[(chunk["voter"] == last_key[0]) & (chunk["space"] == last_key[1])].copy()

    if not carry_over.empty:
        _flush_group([carry_over])

    if write_idx != n_windows:
        print(f"[cache] WARNING: expected {n_windows} windows, wrote {write_idx}")

    num_mm.flush()
    lbl_mm.flush()
    dao_mm.flush()
    voter_mm.flush()
    del num_mm, lbl_mm, dao_mm, voter_mm

    split_meta = {
        "split": split_name,
        "n_windows": int(write_idx),
        "window_size": window_size,
        "feat_dim": feat_dim,
        "num_feats": str(num_path.resolve()),
        "labels": str(lbl_path.resolve()),
        "dao_clusters": str(dao_path.resolve()),
        "voter_clusters": str(voter_path.resolve()),
    }
    return split_meta


def materialize_window_cache(
    *,
    csv_path: Path,
    split_manifest_path: Path,
    cache_dir: Path,
    preprocessor: Dict[str, Any],
    window_size: int,
    splits: Tuple[str, ...] = ("train", "val"),
) -> Dict[str, Any]:
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")

    split_meta: Dict[str, Any] = {}
    for split_name in splits:
        voters = manifest.get(f"{split_name}_voters") or []
        split_meta[split_name] = materialize_split_cache(
            csv_path=csv_path,
            voters=list(map(str, voters)),
            split_name=split_name,
            cache_dir=cache_dir,
            preprocessor=preprocessor,
            window_size=window_size,
            con=con,
        )

    con.close()
    meta = {
        "csv_path": str(csv_path.resolve()),
        "split_manifest": str(split_manifest_path.resolve()),
        "window_size": window_size,
        "splits": split_meta,
    }
    meta_path = cache_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[cache] wrote {meta_path}")
    return meta


def load_window_cache_meta(cache_dir: Path) -> Dict[str, Any]:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing window cache metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))
