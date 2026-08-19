#!/usr/bin/env python3
"""
8) Build behaviour dataset (optional), train voter-choice model with safeguards.

Leakage-safe flow: split -> train-only structural clusters -> causal windows.

  cd whole_pipeline
  set PYTHONPATH=src
  python scripts/08_run_behaviour_modelling.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dao_governance.features.behaviour_dataset import build_behaviour_dataset
from dao_governance.features.behaviour_pipeline import (
    load_behaviour_votes,
    prepare_behaviour_splits,
    write_enriched_behaviour_csv,
)
from dao_governance.modelling.dataset import WindowDataset, collate_fn
from dao_governance.modelling.metrics import macro_prf
from dao_governance.modelling.model import TimeSeriesClassifier
from dao_governance.modelling.preprocess import (
    assert_finite_feature_columns,
    save_split_manifest,
    select_numeric_columns,
    split_by_voter_three_way,
)
from dao_governance.modelling.windows import build_windows
from dao_governance.settings import load_config, project_root


def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    n = counts.sum()
    w = n / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32)


def _resolve_master_parquet(base: Path, paths: dict) -> Path:
    cleaned = (base / paths.get("cleaned_master_parquet", "data/processed/votes_cleaned.parquet")).resolve()
    if cleaned.exists():
        return cleaned
    merged = (base / paths.get("master_with_dao_parquet", "data/processed/master_votes_with_dao_cluster.parquet")).resolve()
    if merged.exists():
        return merged
    raise FileNotFoundError(
        f"Need cleaned or master votes parquet. Tried: {cleaned}, {merged}. Run stages 02–03 / 07."
    )


def _log_phase(message: str) -> None:
    print(f"[08] {message}")


def _behaviour_csv_meta_path(behaviour_csv: Path) -> Path:
    return behaviour_csv.with_suffix(".meta.json")


def _write_behaviour_csv_meta(path: Path, *, text_mode: str) -> None:
    path.write_text(json.dumps({"text_mode": text_mode}, indent=2), encoding="utf-8")


def _assert_reusable_behaviour_csv(behaviour_csv: Path, *, text_mode: str) -> None:
    meta_path = _behaviour_csv_meta_path(behaviour_csv)
    if not meta_path.exists():
        raise RuntimeError(
            f"Cannot reuse {behaviour_csv}: missing text-mode metadata {meta_path}. "
            "Rebuild without --reuse-behaviour-csv."
        )
    saved_mode = json.loads(meta_path.read_text(encoding="utf-8")).get("text_mode")
    if saved_mode != text_mode:
        raise RuntimeError(
            f"Cannot reuse {behaviour_csv}: saved text_mode={saved_mode!r}, requested={text_mode!r}. "
            "Rebuild without --reuse-behaviour-csv."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--extra-config", type=str, default="")
    ap.add_argument("--reuse-behaviour-csv", action="store_true", help="Skip rebuilding behaviour CSV if it exists.")
    args = ap.parse_args()

    _log_phase("starting behaviour modelling run")
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
    vc = cfg.get("voter_clustering", {})

    behaviour_csv = (base / paths.get("behaviour_dataset_csv", "outputs/tables/behaviour_dataset.csv")).resolve()
    enriched_csv = behaviour_csv.with_name("behaviour_dataset_with_clusters.csv")
    master_parquet = _resolve_master_parquet(base, paths)
    dao_feat = (base / paths.get("dao_feature_table_csv", "data/processed/dao_feature_table.csv")).resolve()
    if not dao_feat.exists():
        dao_feat = (base / paths.get("dao_feature_table_parquet", "data/processed/dao_feature_table.parquet")).resolve()

    cluster_dir = (base / paths.get("predictive_cluster_artifacts_dir", "outputs/models/predictive_clusters")).resolve()
    out_dir = (base / paths.get("model_artifacts_dir", "outputs/models/behaviour_agent2")).resolve()
    logs_dir = (base / paths.get("logs_dir", "outputs/logs")).resolve()
    split_path = (base / paths.get("split_manifest_json", "outputs/processed/split_manifest.json")).resolve()
    prep_report = (base / (cfg.get("reports") or {}).get("preprocessing_md", "outputs/tables/preprocessing_report.md")).resolve()
    train_report = (base / (cfg.get("reports") or {}).get("training_md", "outputs/tables/training_report.md")).resolve()

    use_dao = bool(bm.get("use_dao_clusters", True))
    use_voter = bool(bm.get("use_voter_clusters", True))
    text_mode = str(bm.get("text_mode", "title"))
    if text_mode not in {"title", "title_body"}:
        raise ValueError("behaviour_model.text_mode must be 'title' or 'title_body'.")
    min_votes = int(vc.get("min_votes_per_voter_space", 5))

    for d in (behaviour_csv.parent, out_dir, logs_dir, split_path.parent, prep_report.parent, cluster_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not args.reuse_behaviour_csv or not behaviour_csv.exists():
        _log_phase("building behaviour dataset CSV")
        _, rep_b = build_behaviour_dataset(
            master_parquet, output_csv=behaviour_csv, text_mode=text_mode
        )
        _write_behaviour_csv_meta(_behaviour_csv_meta_path(behaviour_csv), text_mode=text_mode)
        br_path = prep_report.parent / "behaviour_dataset_quality.md"
        br_path.write_text(rep_b.to_markdown("Behaviour dataset (build)"), encoding="utf-8")
    else:
        _assert_reusable_behaviour_csv(behaviour_csv, text_mode=text_mode)
        print("[08] Reusing existing behaviour CSV:", behaviour_csv)

    _log_phase("loading behaviour dataset")
    raw_df = load_behaviour_votes(behaviour_csv)
    train_frac = float(bm.get("train_frac", 0.7))
    val_frac = float(bm.get("val_frac", 0.15))

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
    print(f"[08] Split manifest: {split_path}")

    _log_phase("preparing train/val/test splits and fitting structural clusters")
    train_df, val_df, test_df, preprocessor, cluster_bundle, tr_a, va_a, te_a = prepare_behaviour_splits(
        raw_df,
        dao_feature_table_path=dao_feat,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        upper_quantile_cap=float(dq.get("upper_quantile_cap", 0.999)),
        absolute_cap=float(dq.get("absolute_cap", 1e18)),
        use_dao_clusters=use_dao,
        use_voter_clusters=use_voter,
        min_votes_per_pair=min_votes,
        cluster_artifacts_dir=cluster_dir,
    )
    _log_phase("writing enriched behaviour CSV")
    write_enriched_behaviour_csv(tr_a, va_a, te_a, enriched_csv)
    print(f"[08] Enriched behaviour CSV (clusters): {enriched_csv}")

    prep_report.write_text(
        "# Preprocessing report\n\n"
        f"- Leakage-safe: **train-only structural clusters**, `(voter, space)` windows\n"
        f"- Voting power cap (quantile {dq.get('upper_quantile_cap', 0.999)}): **{preprocessor['voting_power']['cap_value']}**\n"
        f"- Transform: **{preprocessor['voting_power'].get('transform', 'clip_then_log1p_robust')}**\n"
        f"- Model numeric columns: **{select_numeric_columns()}**\n"
        f"- Text mode: **{text_mode}**\n"
        f"- Cluster meta: **{cluster_bundle.meta}**\n"
        f"- Notes: {preprocessor.get('meta', {}).get('notes', [])}\n",
        encoding="utf-8",
    )

    num_cols = select_numeric_columns()
    for name, dfx in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert_finite_feature_columns(dfx, num_cols)
        print(f"[08] finite check OK: {name}")

    window_size = int(bm.get("window", 5))
    _log_phase(f"building windows with size={window_size}")
    train_windows = build_windows(train_df, window_size=window_size, numeric_cols=num_cols)
    valid_windows = build_windows(val_df, window_size=window_size, numeric_cols=num_cols)
    if not train_windows or not valid_windows:
        raise RuntimeError("Empty windows. Increase data volume or reduce --window in config.")

    max_tr = int(bm.get("max_train_windows", 0))
    max_va = int(bm.get("max_valid_windows", 0))
    rng_sub = np.random.default_rng(seed)

    def _maybe_sub(ws: list, cap: int) -> list:
        if cap <= 0 or len(ws) <= cap:
            return ws
        idx = rng_sub.choice(len(ws), size=cap, replace=False)
        return [ws[i] for i in sorted(idx)]

    train_windows = _maybe_sub(train_windows, max_tr)
    valid_windows = _maybe_sub(valid_windows, max_va)

    pretrained = bm.get("pretrained", "distilroberta-base")
    max_length = int(bm.get("max_length", 128))
    batch_size = int(bm.get("batch_size", 16))
    epochs = int(bm.get("epochs", 4))
    lr = float(bm.get("lr", 2e-5))
    warn_lr = float(bm.get("lr_warn_above", 1e-3))
    max_grad_norm = float(bm.get("max_grad_norm", 1.0))

    if lr > warn_lr:
        print(f"[08] WARNING: learning rate {lr} exceeds recommended upper bound {warn_lr} for AdamW + transformers.")

    _log_phase("initializing tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
    special_tokens = ["[PREDICT]", "[LABEL_0]", "[LABEL_1]", "[LABEL_2]"]
    if text_mode == "title_body":
        special_tokens.extend(["[TITLE]", "[BODY]"])
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    train_ds = WindowDataset(train_windows, tokenizer, max_length)
    valid_ds = WindowDataset(valid_windows, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dim = len(train_windows[0].window_features[0])
    model = TimeSeriesClassifier(pretrained_model_name=pretrained, feat_dim=feat_dim).to(device)
    model.text_encoder.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    class_weights = compute_class_weights(np.array([w.target_label for w in train_windows], dtype=int)).to(device)

    best_f1 = -1.0
    best_state = None
    train_lines = []

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
            scheduler.step()
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
        (logs_dir / f"confusion_valid_epoch{epoch+1}.csv").write_text(
            "pred_FOR,pred_AGAINST,pred_ABSTAIN\n"
            + "\n".join(",".join(map(str, row)) for row in cm),
            encoding="utf-8",
        )

        if valid_m["f1"] > best_f1:
            best_f1 = valid_m["f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    tok_dir = out_dir / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model.pt")
    tokenizer.save_pretrained(tok_dir)

    train_cfg = {
        "pretrained": pretrained,
        "text_mode": text_mode,
        "window": window_size,
        "max_length": max_length,
        "feat_dim": int(feat_dim),
        "best_valid_f1": float(best_f1),
        "label_map": {"FOR": 0, "AGAINST": 1, "ABSTAIN": 2},
        "numeric_preprocessor": preprocessor,
        "split_manifest": str(split_path),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "seed": seed,
        "leakage_safe": True,
        "numeric_columns": select_numeric_columns(),
        "cluster_artifacts_dir": str(cluster_dir),
        "use_dao_clusters": use_dao,
        "use_voter_clusters": use_voter,
        "enriched_behaviour_csv": str(enriched_csv),
    }
    (out_dir / "config.json").write_text(json.dumps(train_cfg, indent=2), encoding="utf-8")

    train_report.write_text(
        "# Model training report\n\n"
        + "\n".join(f"- {ln}" for ln in train_lines)
        + f"\n\nBest validation macro-F1: **{best_f1:.4f}**\n",
        encoding="utf-8",
    )
    st8 = (base / (cfg.get("reports") or {}).get("stage08_md", "outputs/reports/stage08_behaviour_modelling.md")).resolve()
    st8.parent.mkdir(parents=True, exist_ok=True)
    st8.write_text(
        "\n".join(
            [
                "# Stage 8 — Behaviour modelling (leakage-safe)",
                "",
                f"- Model dir: `{out_dir}`",
                f"- Cluster artifacts: `{cluster_dir}`",
                f"- Enriched CSV (cache): `{enriched_csv}`",
                f"- Text mode: `{text_mode}`",
                f"- Training report: `{train_report}`",
                f"- Preprocessing report: `{prep_report}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _log_phase("finished behaviour modelling run")
    print(f"[08] saved model to {out_dir}")


if __name__ == "__main__":
    main()
