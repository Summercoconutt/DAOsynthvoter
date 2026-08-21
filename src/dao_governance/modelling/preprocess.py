"""Numeric preprocessing for behaviour modelling — fit on training split only."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _safe_iqr(x: pd.Series) -> float:
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1
    return iqr if iqr > 1e-8 else 1.0


def _numeric_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def fit_numeric_preprocessor(
    df: pd.DataFrame,
    *,
    upper_quantile_cap: float = 0.999,
    absolute_cap: float = 1e18,
    include_prior_vote_fractions: bool = False,
) -> Dict[str, Any]:
    """
    Fit robust preprocessing on training split only.
    Voting power: negatives invalid -> NaN -> median impute -> clip to quantile cap -> log1p -> robust scale.
    vp_share: map accidental percentages -> [0,1], robust scale.
    Optional columns get simple robust scaling if present and numeric.
    """
    report_notes: List[str] = []

    raw_vp = pd.to_numeric(df.get("voting_power", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan)
    raw_vp = raw_vp.mask(raw_vp < 0.0, np.nan)
    neg_n = int((pd.to_numeric(df.get("voting_power", np.nan), errors="coerce") < 0).sum())
    if neg_n:
        report_notes.append(f"Negative voting_power rows masked before fit: {neg_n}")

    vp_med = float(raw_vp.median()) if raw_vp.notna().any() else 0.0
    vp = raw_vp.fillna(vp_med)
    vp_cap = float(vp.quantile(upper_quantile_cap)) if vp.notna().any() else max(vp_med, 1.0)
    vp_cap = min(vp_cap, absolute_cap)
    vp = vp.clip(lower=0.0, upper=max(vp_cap, 1.0))
    wins_n = int((pd.to_numeric(df.get("voting_power", np.nan), errors="coerce").fillna(vp_med) > vp_cap).sum())
    if wins_n:
        report_notes.append(f"Voting power values clipped at cap q={upper_quantile_cap}: up to {wins_n} rows")
    vp_log = np.log1p(vp)
    vp_center = float(vp_log.median()) if vp_log.notna().any() else 0.0
    vp_scale = _safe_iqr(vp_log) if vp_log.notna().any() else 1.0

    raw_share = _numeric_series(df, "vp_share").replace([np.inf, -np.inf], np.nan)
    raw_share = raw_share.where(raw_share <= 1.0, raw_share / 100.0)
    share_med = float(raw_share.median()) if raw_share.notna().any() else 0.0
    share = raw_share.fillna(share_med).clip(lower=0.0, upper=1.0)
    share_center = float(share.median()) if share.notna().any() else 0.0
    share_scale = _safe_iqr(share) if share.notna().any() else 1.0

    extra: Dict[str, Any] = {}
    zero_var_cols: List[str] = []
    for col in ("dao_cluster", "voter_cluster"):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").fillna(-1.0)
        if float(s.std(ddof=0) or 0.0) < 1e-12:
            zero_var_cols.append(col)
        med = float(s.median())
        sc = _safe_iqr(s)
        extra[col] = {"center": med, "scale": max(sc, 1e-8)}

    if include_prior_vote_fractions:
        for col in ("prior_frac_for", "prior_frac_against"):
            s = _numeric_series(df, col, default=0.0).replace([np.inf, -np.inf], np.nan)
            s = s.fillna(0.0).clip(lower=0.0, upper=1.0)
            if float(s.std(ddof=0) or 0.0) < 1e-12:
                zero_var_cols.append(col)
            extra[col] = {"center": float(s.median()), "scale": max(_safe_iqr(s), 1e-8)}

    return {
        "voting_power": {
            "median_raw_nonneg": vp_med,
            "cap_value": float(max(vp_cap, 1.0)),
            "cap_quantile": upper_quantile_cap,
            "log_center": vp_center,
            "log_scale": float(max(vp_scale, 1e-8)),
            "transform": "clip_then_log1p_robust",
        },
        "vp_share": {
            "median_raw": share_med,
            "center": share_center,
            "scale": float(max(share_scale, 1e-8)),
        },
        "extra_robust": extra,
        "meta": {"notes": report_notes, "zero_variance_columns": zero_var_cols},
    }


def normalise_columns(df: pd.DataFrame, preprocessor: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["voter"] = out["voter"].astype(str)
    out["vote_ts"] = pd.to_datetime(out["vote_ts"], utc=True, errors="coerce")
    out["label_id"] = pd.to_numeric(out["label_id"], errors="coerce").astype("Int64")
    out["label_id"] = out["label_id"].fillna(-1).astype(int)

    vp_cfg = preprocessor["voting_power"]
    vp = _numeric_series(out, "voting_power").replace([np.inf, -np.inf], np.nan)
    vp = vp.mask(vp < 0.0, np.nan).fillna(float(vp_cfg["median_raw_nonneg"]))
    vp = vp.clip(lower=0.0, upper=float(vp_cfg["cap_value"]))
    vp = (np.log1p(vp) - float(vp_cfg["log_center"])) / float(vp_cfg["log_scale"])
    out["voting_power"] = vp.astype(float)

    if "vp_share" in out.columns:
        share = _numeric_series(out, "vp_share").replace([np.inf, -np.inf], np.nan)
        share = share.where(share <= 1.0, share / 100.0).fillna(float(sh_cfg["median_raw"]))
        share = share.clip(lower=0.0, upper=1.0)
        share = (share - float(sh_cfg["center"])) / float(sh_cfg["scale"])
        out["vp_share"] = share.astype(float)

    out["is_whale"] = _to_bool_series(out.get("is_whale", False))

    ex = preprocessor.get("extra_robust") or {}
    for col in ("dao_cluster", "voter_cluster", "prior_frac_for", "prior_frac_against"):
        if col in out.columns and col in ex:
            default = 0.0 if col.startswith("prior_frac_") else -1.0
            s = pd.to_numeric(out[col], errors="coerce").fillna(default)
            if col.startswith("prior_frac_"):
                s = s.clip(lower=0.0, upper=1.0)
            cfg = ex[col]
            out[col] = ((s - float(cfg["center"])) / float(cfg["scale"])).astype(float)

    if "text" in out.columns:
        out["text"] = out["text"].fillna("").astype(str)
    else:
        out["text"] = ""
    return out


def _to_bool_series(col: pd.Series) -> pd.Series:
    def _to_bool(x):
        if isinstance(x, str):
            return x.strip().lower() in {"1", "true", "yes", "y", "t"}
        if pd.isna(x):
            return False
        return bool(x)

    return col.apply(_to_bool)


def select_numeric_columns() -> List[str]:
    """Leakage-safe model inputs (no label-derived or unverified post-hoc fields)."""
    return ["voting_power", "is_whale", "dao_cluster", "voter_cluster"]


def select_numeric_columns_no_roberta() -> List[str]:
    """Stage 08b inputs, including causal historical-choice fractions."""
    return select_numeric_columns() + ["prior_frac_for", "prior_frac_against"]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def split_by_voter_three_way(
    df: pd.DataFrame,
    *,
    train_frac: float,
    val_frac: float,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Disjoint voter splits; remaining goes to test."""
    voters = df["voter"].dropna().astype(str).unique()
    # force a plain ndarray: shuffling arrow-backed string arrays in place is slow/unsafe
    voters = np.asarray(voters, dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(voters)
    n = len(voters)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_v = set(voters[:n_train])
    val_v = set(voters[n_train : n_train + n_val])
    test_v = set(voters[n_train + n_val :])
    train_df = df[df["voter"].astype(str).isin(train_v)].copy()
    val_df = df[df["voter"].astype(str).isin(val_v)].copy()
    test_df = df[df["voter"].astype(str).isin(test_v)].copy()
    return train_df, val_df, test_df


def save_split_manifest(
    path: Path,
    *,
    train_voters: Sequence[str],
    val_voters: Sequence[str],
    test_voters: Sequence[str],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "n_train_voters": len(train_voters),
        "n_val_voters": len(val_voters),
        "n_test_voters": len(test_voters),
        "train_voters": sorted(map(str, train_voters)),
        "val_voters": sorted(map(str, val_voters)),
        "test_voters": sorted(map(str, test_voters)),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def filter_df_to_voters(df: pd.DataFrame, voters: Sequence[str]) -> pd.DataFrame:
    s = set(map(str, voters))
    return df[df["voter"].astype(str).isin(s)].copy()


def assert_finite_feature_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Abort training if any modelling numeric column is NaN/Inf after preprocessing."""
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing expected column after preprocessing: {c}")
        x = pd.to_numeric(df[c], errors="coerce")
        if x.isna().any():
            raise ValueError(f"NaN remains in column {c} after preprocessing (must not reach the model).")
        arr = np.asarray(x, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values in column {c} after preprocessing.")
