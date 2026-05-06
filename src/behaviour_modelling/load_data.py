#!/usr/bin/env python3
"""
Data loading utilities for behaviour modelling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np


def load_dataset(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def _to_bool_series(col: pd.Series) -> pd.Series:
    def _to_bool(x):
        if isinstance(x, str):
            return x.strip().lower() in {"1", "true", "yes", "y", "t"}
        if pd.isna(x):
            return False
        return bool(x)

    return col.apply(_to_bool)


def _safe_iqr(x: pd.Series) -> float:
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1
    return iqr if iqr > 1e-8 else 1.0


def fit_numeric_preprocessor(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit robust preprocessing parameters on training split only.
    """
    raw_vp = pd.to_numeric(df.get("voting_power", np.nan), errors="coerce")
    raw_vp = raw_vp.replace([np.inf, -np.inf], np.nan)
    # Voting power should be non-negative in most governance systems.
    # We treat negatives as invalid and set to NaN for robust imputation.
    raw_vp = raw_vp.mask(raw_vp < 0.0, np.nan)
    vp_med = float(raw_vp.median()) if raw_vp.notna().any() else 0.0
    vp = raw_vp.fillna(vp_med)
    vp_cap = float(vp.quantile(0.999)) if vp.notna().any() else max(vp_med, 1.0)
    vp = vp.clip(lower=0.0, upper=max(vp_cap, 1.0))
    vp_log = np.log1p(vp)
    vp_center = float(vp_log.median()) if vp_log.notna().any() else 0.0
    vp_scale = _safe_iqr(vp_log) if vp_log.notna().any() else 1.0

    raw_share = pd.to_numeric(df.get("vp_share", np.nan), errors="coerce")
    raw_share = raw_share.replace([np.inf, -np.inf], np.nan)
    # Safety: if ratio accidentally stored as percentage, map to [0,1].
    raw_share = raw_share.where(raw_share <= 1.0, raw_share / 100.0)
    share_med = float(raw_share.median()) if raw_share.notna().any() else 0.0
    share = raw_share.fillna(share_med).clip(lower=0.0, upper=1.0)
    share_center = float(share.median()) if share.notna().any() else 0.0
    share_scale = _safe_iqr(share) if share.notna().any() else 1.0

    return {
        "voting_power": {
            "median_raw_nonneg": vp_med,
            "cap_q999": float(max(vp_cap, 1.0)),
            "log_center": vp_center,
            "log_scale": float(max(vp_scale, 1e-8)),
        },
        "vp_share": {
            "median_raw": share_med,
            "center": share_center,
            "scale": float(max(share_scale, 1e-8)),
        },
    }


def normalise_columns(df: pd.DataFrame, preprocessor: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["voter"] = out["voter"].astype(str)
    out["vote_ts"] = pd.to_datetime(out["vote_ts"], utc=True, errors="coerce")
    out["label_id"] = pd.to_numeric(out["label_id"], errors="coerce").astype("Int64")
    out["label_id"] = out["label_id"].fillna(-1).astype(int)

    vp_cfg = preprocessor["voting_power"]
    vp = pd.to_numeric(out.get("voting_power", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan)
    vp = vp.mask(vp < 0.0, np.nan).fillna(float(vp_cfg["median_raw_nonneg"]))
    vp = vp.clip(lower=0.0, upper=float(vp_cfg["cap_q999"]))
    vp = (np.log1p(vp) - float(vp_cfg["log_center"])) / float(vp_cfg["log_scale"])
    out["voting_power"] = vp.astype(float)

    sh_cfg = preprocessor["vp_share"]
    share = pd.to_numeric(out.get("vp_share", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan)
    share = share.where(share <= 1.0, share / 100.0).fillna(float(sh_cfg["median_raw"]))
    share = share.clip(lower=0.0, upper=1.0)
    share = (share - float(sh_cfg["center"])) / float(sh_cfg["scale"])
    out["vp_share"] = share.astype(float)

    out["is_whale"] = _to_bool_series(out.get("is_whale", False))
    out["aligned_with_majority"] = _to_bool_series(out.get("aligned_with_majority", False))
    out["dao_cluster"] = pd.to_numeric(out.get("dao_cluster", -1), errors="coerce").fillna(-1).astype(int)
    out["voter_cluster"] = pd.to_numeric(out.get("voter_cluster", -1), errors="coerce").fillna(-1).astype(int)
    out["text"] = out.get("text", "").fillna("").astype(str)
    return out


def select_numeric_columns() -> List[str]:
    return ["voting_power", "vp_share", "is_whale", "aligned_with_majority", "dao_cluster", "voter_cluster"]


def split_by_voter(df: pd.DataFrame, train_frac: float = 0.8, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    voters = df["voter"].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(voters)
    n_train = int(len(voters) * train_frac)
    train_voters = set(voters[:n_train])
    train_df = df[df["voter"].isin(train_voters)].copy()
    valid_df = df[~df["voter"].isin(train_voters)].copy()
    return train_df, valid_df
