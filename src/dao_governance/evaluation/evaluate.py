"""Evaluate behaviour model on a voter split (typically held-out test)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dao_governance.modelling.dataset import WindowDataset, collate_fn
from dao_governance.modelling.model import TimeSeriesClassifier
from dao_governance.modelling.preprocess import filter_df_to_voters, load_dataset, normalise_columns, select_numeric_columns
from dao_governance.modelling.windows import build_windows


def run_evaluation(
    dataset_csv: Path,
    artifacts_dir: Path,
    output_dir: Path,
    *,
    preprocessor: Dict[str, Any],
    cfg_train: Dict[str, Any],
    voter_ids: Optional[List[str]] = None,
    seed: int = 42,
    batch_size: int = 32,
    max_windows: int = 0,
    split_label: str = "eval",
) -> Dict[str, Any]:
    """
    If voter_ids is set, restrict to those voters (e.g. test split). Otherwise use full CSV.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(artifacts_dir / "tokenizer")
    model = TimeSeriesClassifier(pretrained_model_name=cfg_train["pretrained"], feat_dim=cfg_train["feat_dim"])
    model.text_encoder.resize_token_embeddings(len(tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(artifacts_dir / "model.pt", map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    raw_df = load_dataset(Path(dataset_csv))
    if voter_ids is not None:
        raw_df = filter_df_to_voters(raw_df, voter_ids)
    if raw_df.empty:
        raise RuntimeError(f"No rows for evaluation ({split_label}).")

    valid_df = normalise_columns(raw_df, preprocessor=preprocessor)
    windows = build_windows(valid_df, window_size=cfg_train["window"], numeric_cols=select_numeric_columns())
    if not windows:
        raise RuntimeError(f"No windows for evaluation ({split_label}).")
    if max_windows > 0 and len(windows) > max_windows:
        rng = np.random.default_rng(seed + 1)
        idx = rng.choice(len(windows), size=max_windows, replace=False)
        windows = [windows[i] for i in sorted(idx)]

    ds = WindowDataset(windows, tokenizer, max_length=cfg_train["max_length"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    y_true, y_pred, dao, voter_c = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            for k in batch:
                batch[k] = batch[k].to(device)
            logits = model(batch)["logits"]
            y_true.extend(batch["labels"].cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            dao.extend(batch["dao_clusters"].cpu().numpy().tolist())
            voter_c.extend(batch["voter_clusters"].cpu().numpy().tolist())

    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(out / f"classification_report_{split_label}.csv")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    pd.DataFrame(cm, index=["FOR", "AGAINST", "ABSTAIN"], columns=["FOR", "AGAINST", "ABSTAIN"]).to_csv(
        out / f"confusion_matrix_{split_label}.csv"
    )

    detail = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "dao_cluster": dao, "voter_cluster": voter_c})
    detail.to_csv(out / f"predictions_with_clusters_{split_label}.csv", index=False)

    rows = []
    for key, group_col in [("dao_cluster", "dao_cluster"), ("voter_cluster", "voter_cluster")]:
        for gid, g in detail.groupby(group_col):
            if len(g) < 5:
                continue
            sub = classification_report(g["y_true"], g["y_pred"], output_dict=True, zero_division=0)
            rows.append({"group_type": key, "group_id": int(gid), "macro_f1": sub["macro avg"]["f1-score"], "n": len(g)})
    pd.DataFrame(rows).to_csv(out / f"macro_f1_by_cluster_{split_label}.csv", index=False)

    summary = {
        "split": split_label,
        "n_windows": len(windows),
        "macro_f1": float(rep.get("macro avg", {}).get("f1-score", 0.0)),
        "accuracy": float(rep.get("accuracy", 0.0)),
    }
    (out / f"summary_{split_label}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
