"""Sliding windows for behaviour modelling (leakage-safe)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

# Label-derived columns: allowed in history steps, zeroed at prediction step.
LABEL_DERIVED_AT_PREDICT_TIME = frozenset({"aligned_with_majority"})

WINDOW_GROUP_COLS: Tuple[str, ...] = ("voter", "space")


@dataclass
class Window:
    window_texts: List[str]
    window_features: List[np.ndarray]
    target_label: int
    voter_id: str
    dao_cluster: int
    voter_cluster: int


def _eval_cluster_id(row: pd.Series, col: str) -> int:
    eval_col = f"{col}_eval"
    if eval_col in row.index and pd.notna(row.get(eval_col)):
        return int(row[eval_col])
    val = row.get(col, -1)
    if pd.isna(val):
        return -1
    return int(val)


def _time_feats(ts: pd.Timestamp) -> List[float]:
    if pd.isna(ts):
        return [0.0, 0.0, 0.0, 0.0]
    return [
        ts.hour / 23.0,
        ts.weekday() / 6.0,
        (ts.month - 1) / 11.0,
        (ts.day - 1) / 30.0,
    ]


def _eval_cluster_id(row: pd.Series, col: str) -> int:
    val = row.get(col, -1)
    if pd.isna(val):
        return -1
    try:
        return int(val)
    except (TypeError, ValueError):
        return -1


def _row_numeric_vec(
    row: pd.Series,
    numeric_cols: Sequence[str],
    *,
    is_current_step: bool,
) -> List[float]:
    vec: List[float] = []
    for c in numeric_cols:
        if is_current_step and c in LABEL_DERIVED_AT_PREDICT_TIME:
            vec.append(0.0)
            continue
        val = row.get(c, 0.0)
        if isinstance(val, (bool, np.bool_)):
            vec.append(float(val))
        else:
            try:
                vec.append(float(val))
            except (TypeError, ValueError):
                vec.append(0.0)
    vec.extend(_time_feats(pd.to_datetime(row["vote_ts"], utc=True, errors="coerce")))
    return vec


def build_windows(
    df: pd.DataFrame,
    window_size: int,
    numeric_cols: List[str],
    *,
    group_cols: Sequence[str] = WINDOW_GROUP_COLS,
) -> List[Window]:
    df = df[df["label_id"].isin([0, 1, 2])].copy()
    out: List[Window] = []

    for group_key, g in df.groupby(list(group_cols)):
        g = g.sort_values("vote_ts").reset_index(drop=True)
        if len(g) < window_size:
            continue

        voter_id = str(group_key[0]) if isinstance(group_key, tuple) else str(group_key)

        for t in range(window_size - 1, len(g)):
            hist = list(range(t - window_size + 1, t))
            cur = t
            indices = hist + [cur]

            texts: List[str] = []
            feats: List[np.ndarray] = []
            for idx in indices:
                row = g.loc[idx]
                is_cur = idx == cur
                if is_cur:
                    texts.append("[PREDICT] " + str(row["text"]))
                else:
                    texts.append(f"[LABEL_{int(row['label_id'])}] " + str(row["text"]))

                vec = _row_numeric_vec(row, numeric_cols, is_current_step=is_cur)
                arr = np.asarray(vec, dtype=np.float32)
                if not np.isfinite(arr).all():
                    raise ValueError(
                        f"Non-finite feature vector for voter={voter_id} row_idx={idx}. "
                        "Upstream preprocessing must produce finite values only."
                    )
                feats.append(arr)

            target = int(g.loc[cur, "label_id"])
            dc = _eval_cluster_id(g.loc[cur], "dao_cluster")
            vc = _eval_cluster_id(g.loc[cur], "voter_cluster")
            out.append(
                Window(
                    window_texts=texts,
                    window_features=feats,
                    target_label=target,
                    voter_id=voter_id,
                    dao_cluster=dc,
                    voter_cluster=vc,
                )
            )
    return out
