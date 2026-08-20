#!/usr/bin/env python3
"""
8b) Train numeric-only voter-choice model (no RoBERTa / no proposal text).

Compare against the existing RoBERTa pipeline without modifying it.

  cd whole_pipeline
  set PYTHONPATH=src
  python scripts/08b_run_behaviour_modelling_no_roberta.py --config configs/default.yaml

Quick smoke test:
  python scripts/08b_run_behaviour_modelling_no_roberta.py --config configs/default.yaml \\
      --extra-config configs/smoke_no_roberta.yaml --reuse-behaviour-csv \\
      --max-train-windows 320 --max-valid-windows 96
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dao_governance.features.behaviour_dataset import build_behaviour_dataset
from dao_governance.features.behaviour_pipeline import (
    load_behaviour_votes,
    prepare_behaviour_splits,
    write_enriched_behaviour_csv,
)
from dao_governance.modelling.dataset import (
    MemmapNumericWindowDataset,
    NumericWindowDataset,
    numeric_collate_fn,
)
from dao_governance.modelling.metrics import macro_prf
from dao_governance.modelling.model import NumericOnlyTimeSeriesClassifier
from dao_governance.modelling.preprocess import (
    assert_finite_feature_columns,
    save_split_manifest,
    select_numeric_columns_no_roberta,
    split_by_voter_three_way,
)
from dao_governance.modelling.window_cache import load_window_cache_meta, materialize_window_cache
from dao_governance.modelling.windows import build_windows
from dao_governance.settings import load_config, project_root

NUMERIC_COLUMNS = select_numeric_columns_no_roberta()


def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    n = counts.sum()
    w = n / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32)


def _parse_best_f1_from_report(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(r"Best validation macro-F1:\s*\*\*([0-9.]+)\*\*", text)
    if m:
        return float(m.group(1))
    m = re.search(r"best_valid_f1[\"']?\s*[:=]\s*([0-9.]+)", text, re.I)
    if m:
        return float(m.group(1))
    return None


def _read_roberta_result(base: Path) -> Dict[str, Any]:
    report_candidates = [
        base / "outputs/tables/training_report.md",
        base / "outputs/behaviour_modelling/training_report.md",
        base / "../test_pipeline/outputs/tables/training_report.md",
    ]
    config_candidates = [
        base / "outputs/behaviour_modelling/agent2_artifacts/config.json",
        base / "outputs/models/behaviour_agent2/config.json",
        base / "../test_pipeline/outputs/behaviour_modelling/agent2_artifacts/config.json",
    ]

    best_f1: Optional[float] = None
    config_path: Optional[Path] = None
    training_report_path: Optional[Path] = None
    artifact_dir: Optional[Path] = None

    for p in config_candidates:
        p = p.resolve()
        if p.exists():
            config_path = p
            artifact_dir = p.parent
            data = json.loads(p.read_text(encoding="utf-8"))
            best_f1 = data.get("best_valid_f1")
            break

    for p in report_candidates:
        p = p.resolve()
        if p.exists():
            training_report_path = p
            if best_f1 is None:
                best_f1 = _parse_best_f1_from_report(p)
            break

    return {
        "model_name": "RoBERTa + Numeric + Clusters",
        "use_roberta": True,
        "use_text": True,
        "best_valid_macro_f1": best_f1,
        "artifact_dir": str(artifact_dir) if artifact_dir else "",
        "config_path": str(config_path) if config_path else "",
        "training_report_path": str(training_report_path) if training_report_path else "",
    }


def write_comparison_csv(base: Path, no_roberta_cfg_path: Path, no_roberta_report_path: Path) -> Path:
    out_path = (base / "outputs/tables/roberta_vs_no_roberta_comparison.csv").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    roberta_row = _read_roberta_result(base)

    no_data: Dict[str, Any] = {}
    if no_roberta_cfg_path.exists():
        no_data = json.loads(no_roberta_cfg_path.read_text(encoding="utf-8"))

    no_row = {
        "model_name": "Numeric + Clusters only",
        "use_roberta": False,
        "use_text": False,
        "best_valid_macro_f1": no_data.get("best_valid_f1"),
        "artifact_dir": str(no_roberta_cfg_path.parent),
        "config_path": str(no_roberta_cfg_path),
        "training_report_path": str(no_roberta_report_path),
    }

    fieldnames = [
        "model_name",
        "use_roberta",
        "use_text",
        "best_valid_macro_f1",
        "artifact_dir",
        "config_path",
        "training_report_path",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(roberta_row)
        writer.writerow(no_row)

    print(f"[08b] Comparison CSV: {out_path}")
    return out_path


def _load_preprocessor_from_config(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pre = data.get("numeric_preprocessor")
    if not pre:
        raise ValueError(f"No numeric_preprocessor in {path}")
    return pre


def _preprocessor_has_history_features(preprocessor: Dict[str, Any]) -> bool:
    extra = preprocessor.get("extra_robust") or {}
    return all(col in extra for col in ("prior_frac_for", "prior_frac_against"))


def _assert_csv_schema(csv_path: Path, required: list[str]) -> None:
    cols = [str(c).strip() for c in pd.read_csv(csv_path, nrows=0).columns.tolist()]
    lower_cols = [c.lower() for c in cols]

    # Accept mixed-case variants by normalizing to lowercase and keeping the
    # first canonical spelling encountered. This prevents transient CSV exports
    # from failing when they contain both 'space' and 'Space'.
    normalized = []
    seen = set()
    for c in cols:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)

    missing = [c for c in required if c not in normalized]
    if missing:
        raise ValueError(
            f"[08b] Missing required columns in {csv_path}: {missing}. "
            f"Available columns: {cols}"
        )


def _csv_has_columns(csv_path: Path, required: list[str]) -> bool:
    cols = {str(c).strip().lower() for c in pd.read_csv(csv_path, nrows=0).columns}
    return all(c.lower() in cols for c in required)


def _prepare_from_cache(
    *,
    behaviour_csv: Path,
    split_path: Path,
    cache_dir: Path,
    window_size: int,
    reuse_cache: bool,
    preprocessor: Dict[str, Any],
    numeric_cols: List[str],
) -> tuple[MemmapNumericWindowDataset, MemmapNumericWindowDataset, Dict[str, Any]]:
    meta_path = cache_dir / "meta.json"
    if reuse_cache and meta_path.exists():
        print("[08b] Reusing window cache:", cache_dir)
        meta = load_window_cache_meta(cache_dir)
        if meta.get("numeric_columns") != numeric_cols or meta.get("window_size") != window_size:
            raise RuntimeError(
                "[08b] Window cache schema does not match this run. "
                "Delete the cache directory or rerun without --reuse-window-cache."
            )
    else:
        print("[08b] Materializing window cache (this may take a while)...")
        meta = materialize_window_cache(
            csv_path=behaviour_csv,
            split_manifest_path=split_path,
            cache_dir=cache_dir,
            preprocessor=preprocessor,
            window_size=window_size,
            numeric_cols=numeric_cols,
        )
    train_ds = MemmapNumericWindowDataset(meta["splits"]["train"])
    valid_ds = MemmapNumericWindowDataset(meta["splits"]["val"])
    return train_ds, valid_ds, meta


def _resolve_master_parquet(base: Path, paths: dict) -> Path:
    cleaned = (base / paths.get("cleaned_master_parquet", "data/processed/votes_cleaned.parquet")).resolve()
    if cleaned.exists():
        return cleaned
    merged = (base / paths.get("master_with_dao_parquet", "data/processed/master_votes_with_dao_cluster.parquet")).resolve()
    if merged.exists():
        return merged
    raise FileNotFoundError(f"Need cleaned or master votes parquet under {base}")


def _log_phase(message: str) -> None:
    print(f"[08b] {message}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--extra-config", type=str, default="")
    ap.add_argument("--reuse-behaviour-csv", action="store_true", help="Skip rebuilding behaviour CSV if it exists.")
    ap.add_argument(
        "--reuse-split-manifest",
        action="store_true",
        help="Use existing split_manifest.json (required for large CSV; avoids loading full dataset).",
    )
    ap.add_argument(
        "--reuse-window-cache",
        action="store_true",
        help="Skip rebuilding on-disk window cache if meta.json exists.",
    )
    ap.add_argument(
        "--window-cache-dir",
        type=str,
        default="outputs/behaviour_modelling/window_cache_no_roberta",
        help="Directory for memmap window cache on large datasets.",
    )
    ap.add_argument(
        "--reuse-preprocessor-from",
        type=str,
        default="",
        help="Load numeric_preprocessor from an existing model config.json (default: RoBERTa artifacts).",
    )
    ap.add_argument(
        "--max-train-windows",
        type=int,
        default=None,
        help="Cap training windows (overrides config; 0 = no cap).",
    )
    ap.add_argument(
        "--max-valid-windows",
        type=int,
        default=None,
        help="Cap validation windows (overrides config; 0 = no cap).",
    )
    args = ap.parse_args()

    _log_phase("starting no-RoBERTa behaviour modelling run")
    base = project_root()
    cfg_path = (base / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    extra = (base / args.extra_config).resolve() if args.extra_config else None
    cfg = load_config(config_path=cfg_path, extra_path=extra)
    _log_phase(f"loaded config from {cfg_path}")

    seed = int(cfg.get("project", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    paths = cfg.get("paths", {})
    bm = cfg.get("behaviour_model", {})
    dq = cfg.get("data_quality", {})
    reports = cfg.get("reports") or {}

    behaviour_csv = (base / paths.get("behaviour_dataset_csv", "outputs/tables/behaviour_dataset.csv")).resolve()
    enriched_csv = behaviour_csv.with_name("behaviour_dataset_with_clusters.csv")
    master_parquet = _resolve_master_parquet(base, paths)
    dao_feat = (base / paths.get("dao_feature_table_csv", "data/processed/dao_feature_table.csv")).resolve()
    if not dao_feat.exists():
        dao_feat = (base / paths.get("dao_feature_table_parquet", "data/processed/dao_feature_table.parquet")).resolve()
    cluster_dir = (base / paths.get("predictive_cluster_artifacts_dir", "outputs/models/predictive_clusters")).resolve()
    out_dir = (
        base
        / paths.get("model_artifacts_dir_no_roberta", "outputs/behaviour_modelling/agent2_artifacts_no_roberta")
    ).resolve()
    logs_dir = (base / paths.get("logs_dir", "outputs/logs")).resolve()
    split_path = (base / paths.get("split_manifest_json", "outputs/processed/split_manifest.json")).resolve()
    prep_report = (base / reports.get("preprocessing_md", "outputs/tables/preprocessing_report_no_roberta.md")).resolve()
    if prep_report.name == "preprocessing_report.md":
        prep_report = (base / "outputs/tables/preprocessing_report_no_roberta.md").resolve()
    train_report = (base / reports.get("training_md", "outputs/tables/training_report_no_roberta.md")).resolve()
    if train_report.name == "training_report.md":
        train_report = (base / "outputs/tables/training_report_no_roberta.md").resolve()

    use_dao = bool(bm.get("use_dao_clusters", True))
    use_voter = bool(bm.get("use_voter_clusters", True))
    min_votes = int(cfg.get("voter_clustering", {}).get("min_votes_per_voter_space", 5))

    for d in (behaviour_csv.parent, out_dir, logs_dir, split_path.parent, prep_report.parent, train_report.parent, cluster_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not args.reuse_behaviour_csv or not behaviour_csv.exists():
        _log_phase("building behaviour dataset CSV")
        _, rep_b = build_behaviour_dataset(master_parquet, output_csv=behaviour_csv)
        br_path = prep_report.parent / "behaviour_dataset_quality_no_roberta.md"
        br_path.write_text(rep_b.to_markdown("Behaviour dataset (build)"), encoding="utf-8")
    else:
        print("[08b] Reusing existing behaviour CSV:", behaviour_csv)

    raw_df: pd.DataFrame | None = None
    train_frac = float(bm.get("train_frac", 0.7))
    val_frac = float(bm.get("val_frac", 0.15))
    window_size = int(bm.get("window", 5))
    upper_q = float(dq.get("upper_quantile_cap", 0.999))
    abs_cap = float(dq.get("absolute_cap", 1e18))
    cache_dir = (base / args.window_cache_dir).resolve()

    cache_csv = enriched_csv if enriched_csv.exists() else behaviour_csv
    csv_bytes = cache_csv.stat().st_size if cache_csv.exists() else 0
    use_cache = args.reuse_split_manifest or csv_bytes > 1_000_000_000
    _assert_csv_schema(cache_csv, required=["voter", "space", "vote_ts", "label_id"])

    preprocessor: Dict[str, Any] | None = None
    prep_source = args.reuse_preprocessor_from.strip()
    if prep_source:
        pre_path = Path(prep_source)
        if not pre_path.is_absolute():
            pre_path = (base / pre_path).resolve()
    else:
        for candidate in (
            base / "outputs/behaviour_modelling/agent2_artifacts_no_roberta/config.json",
            base / "outputs/models/behaviour_agent2/config.json",
            base / "outputs/behaviour_modelling/agent2_artifacts/config.json",
        ):
            if candidate.exists():
                pre_path = candidate.resolve()
                break
        else:
            pre_path = Path("")

    if use_cache:
        _log_phase("using large-dataset cache mode")
        if not split_path.exists():
            _log_phase("creating split manifest")
            _log_phase("loading behaviour dataset")
            raw_df = load_behaviour_votes(behaviour_csv)
            tr, va, te = split_by_voter_three_way(raw_df, train_frac=train_frac, val_frac=val_frac, seed=seed)
            save_split_manifest(
                split_path,
                train_voters=tr["voter"].unique(),
                val_voters=va["voter"].unique(),
                test_voters=te["voter"].unique(),
                seed=seed,
                train_frac=train_frac,
                val_frac=val_frac,
            )
            print(f"[08b] Created split manifest: {split_path}")

        has_history_features = enriched_csv.exists() and _csv_has_columns(
            enriched_csv, ["prior_frac_for", "prior_frac_against"]
        )
        existing_preprocessor = (
            _load_preprocessor_from_config(pre_path) if pre_path.exists() else None
        )
        need_prepare = (
            not has_history_features
            or existing_preprocessor is None
            or not _preprocessor_has_history_features(existing_preprocessor)
        )
        if need_prepare:
            _log_phase("building enriched CSV + preprocessor with causal prior-vote fractions")
            if raw_df is None:
                _log_phase("loading behaviour dataset")
                raw_df = load_behaviour_votes(behaviour_csv)
            _, _, _, preprocessor, _, tr_a, va_a, te_a = prepare_behaviour_splits(
                raw_df,
                dao_feature_table_path=dao_feat,
                train_frac=train_frac,
                val_frac=val_frac,
                seed=seed,
                upper_quantile_cap=upper_q,
                absolute_cap=abs_cap,
                use_dao_clusters=use_dao,
                use_voter_clusters=use_voter,
                min_votes_per_pair=min_votes,
                cluster_artifacts_dir=cluster_dir,
                include_prior_vote_fractions=True,
            )
            write_enriched_behaviour_csv(tr_a, va_a, te_a, enriched_csv)
        elif pre_path.exists():
            preprocessor = existing_preprocessor
            print(f"[08b] Loaded preprocessor from {pre_path}")

        if preprocessor is None:
            raise RuntimeError("[08b] Preprocessor not available after cache prep.")

        cache_csv = enriched_csv
        _assert_csv_schema(
            cache_csv,
            required=["voter", "space", "vote_ts", "label_id", "prior_frac_for", "prior_frac_against"],
        )
        print(f"[08b] Large-dataset mode (CSV {cache_csv.stat().st_size / 1e9:.1f} GB), split manifest: {split_path}")
        prep_report.write_text(
            "# Preprocessing report (No-RoBERTa)\n\n"
            f"- Leakage-safe cache CSV: `{cache_csv}`\n"
            f"- Model numeric columns: **{NUMERIC_COLUMNS}**\n",
            encoding="utf-8",
        )
        _log_phase("materializing window cache")
        train_ds, valid_ds, _cache_meta = _prepare_from_cache(
            behaviour_csv=cache_csv,
            split_path=split_path,
            cache_dir=cache_dir,
            window_size=window_size,
            reuse_cache=args.reuse_window_cache,
            preprocessor=preprocessor,
            numeric_cols=NUMERIC_COLUMNS,
        )
        feat_dim = int(train_ds.feat_dim)
        train_labels = np.asarray(train_ds.labels)
        valid_labels = np.asarray(valid_ds.labels)
    else:
        _log_phase("loading behaviour dataset")
        raw_df = load_behaviour_votes(behaviour_csv)
        _log_phase("splitting data by voter")
        train_raw, val_raw, test_raw = split_by_voter_three_way(
            raw_df, train_frac=train_frac, val_frac=val_frac, seed=seed
        )
        save_split_manifest(
            split_path,
            train_voters=train_raw["voter"].unique(),
            val_voters=val_raw["voter"].unique(),
            test_voters=test_raw["voter"].unique(),
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
        )
        print(f"[08b] Split manifest: {split_path}")

        _log_phase("preparing train/val/test splits and fitting structural clusters")
        train_df, val_df, test_df, preprocessor, cluster_bundle, tr_a, va_a, te_a = prepare_behaviour_splits(
            raw_df,
            dao_feature_table_path=dao_feat,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
            upper_quantile_cap=upper_q,
            absolute_cap=abs_cap,
            use_dao_clusters=use_dao,
            use_voter_clusters=use_voter,
            min_votes_per_pair=min_votes,
            cluster_artifacts_dir=cluster_dir,
            include_prior_vote_fractions=True,
        )
        write_enriched_behaviour_csv(tr_a, va_a, te_a, enriched_csv)

        prep_report.write_text(
            "# Preprocessing report (No-RoBERTa)\n\n"
            f"- Leakage-safe: **train-only structural clusters**\n"
            f"- Voting power cap (quantile {upper_q}): **{preprocessor['voting_power']['cap_value']}**\n"
            f"- Model numeric columns: **{NUMERIC_COLUMNS}**\n"
            f"- Cluster meta: **{cluster_bundle.meta}**\n",
            encoding="utf-8",
        )

        num_cols = NUMERIC_COLUMNS
        for name, dfx in [("train", train_df), ("val", val_df), ("test", test_df)]:
            assert_finite_feature_columns(dfx, num_cols)
            print(f"[08b] finite check OK: {name}")

        train_windows = build_windows(train_df, window_size=window_size, numeric_cols=num_cols)
        valid_windows = build_windows(val_df, window_size=window_size, numeric_cols=num_cols)
        if not train_windows or not valid_windows:
            raise RuntimeError("Empty windows. Increase data volume or reduce window size in config.")

        train_ds = NumericWindowDataset(train_windows)
        valid_ds = NumericWindowDataset(valid_windows)
        feat_dim = len(train_windows[0].window_features[0])
        train_labels = np.array([w.target_label for w in train_windows], dtype=int)
        valid_labels = np.array([w.target_label for w in valid_windows], dtype=int)

    max_tr = args.max_train_windows if args.max_train_windows is not None else int(bm.get("max_train_windows", 0))
    max_va = args.max_valid_windows if args.max_valid_windows is not None else int(bm.get("max_valid_windows", 0))
    if max_tr > 0 and len(train_ds) > max_tr:
        rng_sub = np.random.default_rng(seed)
        idx = rng_sub.choice(len(train_ds), size=max_tr, replace=False)
        train_ds = torch.utils.data.Subset(train_ds, sorted(idx.tolist()))
        train_labels = train_labels[sorted(idx)]
        print(f"[08b] subsampled train windows -> {max_tr}")
    if max_va > 0 and len(valid_ds) > max_va:
        rng_sub = np.random.default_rng(seed)
        idx = rng_sub.choice(len(valid_ds), size=max_va, replace=False)
        valid_ds = torch.utils.data.Subset(valid_ds, sorted(idx.tolist()))
        valid_labels = valid_labels[sorted(idx)]
        print(f"[08b] subsampled valid windows -> {max_va}")

    print(f"[08b] final windows: train={len(train_ds)} valid={len(valid_ds)} feat_dim={feat_dim}")
    _log_phase("initializing numeric-only model")

    batch_size = int(bm.get("batch_size", 16))
    epochs = int(bm.get("epochs", 4))
    lr = float(bm.get("numeric_only_lr", bm.get("lr", 2e-5)))
    dropout = float(bm.get("dropout", 0.1))
    max_grad_norm = float(bm.get("max_grad_norm", 1.0))
    dim_feedforward = int(bm.get("dim_feedforward", 1024))

    use_gpu = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=numeric_collate_fn,
        pin_memory=use_gpu,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=numeric_collate_fn,
        pin_memory=use_gpu,
    )

    device = "cuda" if use_gpu else "cpu"
    model = NumericOnlyTimeSeriesClassifier(
        feat_dim=feat_dim,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    class_weights = compute_class_weights(train_labels).to(device)

    best_f1 = -1.0
    best_state = None
    train_lines: List[str] = []

    _log_phase(f"starting training for {epochs} epochs")
    for epoch in range(epochs):
        model.train()
        tr_true, tr_pred, tr_loss = [], [], 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"train {epoch+1}/{epochs}")):
            for k in batch:
                batch[k] = batch[k].to(device)
            out = model(batch)
            loss = model.loss_fn(out["logits"], batch["labels"], class_weights=class_weights)
            if not torch.isfinite(loss):
                nf = batch["num_feats"].detach().cpu().numpy()
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch+1} batch={batch_idx}. "
                    f"num_feats finite={np.isfinite(nf).all()} min={np.nanmin(nf)} max={np.nanmax(nf)}"
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            tr_loss += float(loss.item())
            tr_true.extend(batch["labels"].detach().cpu().numpy().tolist())
            tr_pred.extend(out["logits"].argmax(dim=-1).detach().cpu().numpy().tolist())

        train_m = macro_prf(tr_true, tr_pred)

        model.eval()
        va_true, va_pred = [], []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"valid {epoch+1}/{epochs}"):
                for k in batch:
                    batch[k] = batch[k].to(device)
                out = model(batch)
                va_true.extend(batch["labels"].detach().cpu().numpy().tolist())
                va_pred.extend(out["logits"].argmax(dim=-1).detach().cpu().numpy().tolist())
        valid_m = macro_prf(va_true, va_pred)
        cm = confusion_matrix(va_true, va_pred, labels=[0, 1, 2])
        line = (
            f"epoch={epoch+1} loss={tr_loss/max(len(train_loader),1):.6f} "
            f"train_f1={train_m['f1']:.4f} valid_f1={valid_m['f1']:.4f} "
            f"acc_train={train_m['accuracy']:.4f} acc_valid={valid_m['accuracy']:.4f}"
        )
        print(line)
        train_lines.append(line)
        (logs_dir / f"no_roberta_confusion_valid_epoch{epoch+1}.csv").write_text(
            "pred_FOR,pred_AGAINST,pred_ABSTAIN\n"
            + "\n".join(",".join(map(str, row)) for row in cm),
            encoding="utf-8",
        )

        if valid_m["f1"] > best_f1:
            best_f1 = valid_m["f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model.pt")

    train_cfg: Dict[str, Any] = {
        "model_type": "numeric_only_temporal",
        "use_roberta": False,
        "use_text": False,
        "window": window_size,
        "feat_dim": int(feat_dim),
        "best_valid_f1": float(best_f1),
        "label_map": {"FOR": 0, "AGAINST": 1, "ABSTAIN": 2},
        "numeric_preprocessor": preprocessor,
        "split_manifest": str(split_path),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "seed": seed,
        "dropout": dropout,
        "dim_feedforward": dim_feedforward,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "window_cache_dir": str(cache_dir) if use_cache else "",
        "leakage_safe": True,
        "numeric_columns": NUMERIC_COLUMNS,
        "cluster_artifacts_dir": str(cluster_dir),
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(train_cfg, indent=2), encoding="utf-8")

    train_report.write_text(
        "# Model training report (No-RoBERTa)\n\n"
        + "\n".join(f"- {ln}" for ln in train_lines)
        + f"\n\nBest validation macro-F1: **{best_f1:.4f}**\n",
        encoding="utf-8",
    )

    comparison_path = write_comparison_csv(base, config_path, train_report)

    print(f"[08b] saved model to {out_dir}")
    print(f"[08b] training report: {train_report}")
    print(f"[08b] preprocessing report: {prep_report}")
    print(f"[08b] comparison: {comparison_path}")
    _log_phase("finished no-RoBERTa behaviour modelling run")


if __name__ == "__main__":
    main()
