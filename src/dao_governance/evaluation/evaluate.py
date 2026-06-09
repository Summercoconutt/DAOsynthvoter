"""Evaluate behaviour model on a voter split (typically held-out test)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dao_governance.evaluation.metrics import (
    CLASS_NAMES,
    attach_eval_cluster_columns,
    classification_report_dataframe,
    compute_classification_metrics,
    metrics_by_group,
    plot_confusion_matrix,
    plot_max_confidence_reliability,
    plot_reliability_diagram,
    reliability_bin_table,
    write_metrics_markdown,
)
from dao_governance.modelling.dataset import WindowDataset, collate_fn
from dao_governance.modelling.model import TimeSeriesClassifier
from dao_governance.modelling.preprocess import filter_df_to_voters, load_dataset, normalise_columns, select_numeric_columns
from dao_governance.modelling.windows import build_windows


def _run_inference(
    *,
    dataset_csv: Path,
    artifacts_dir: Path,
    preprocessor: Dict[str, Any],
    cfg_train: Dict[str, Any],
    voter_ids: Optional[List[str]],
    seed: int,
    batch_size: int,
    max_windows: int,
) -> tuple[list[int], list[int], np.ndarray, list[str], list[int], list[int], int]:
    tokenizer = AutoTokenizer.from_pretrained(artifacts_dir / "tokenizer")
    model = TimeSeriesClassifier(pretrained_model_name=cfg_train["pretrained"], feat_dim=cfg_train["feat_dim"])
    model.text_encoder.resize_token_embeddings(len(tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(artifacts_dir / "model.pt", map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    raw_df = attach_eval_cluster_columns(load_dataset(Path(dataset_csv)))
    if voter_ids is not None:
        raw_df = filter_df_to_voters(raw_df, voter_ids)
    if raw_df.empty:
        raise RuntimeError("No rows for evaluation after voter filter.")

    valid_df = normalise_columns(raw_df, preprocessor=preprocessor)
    windows = build_windows(valid_df, window_size=cfg_train["window"], numeric_cols=select_numeric_columns())
    if not windows:
        raise RuntimeError("No windows for evaluation.")
    if max_windows > 0 and len(windows) > max_windows:
        rng = np.random.default_rng(seed + 1)
        idx = rng.choice(len(windows), size=max_windows, replace=False)
        windows = [windows[i] for i in sorted(idx)]

    ds = WindowDataset(windows, tokenizer, max_length=cfg_train["max_length"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob_rows: list[np.ndarray] = []
    voters: list[str] = []
    dao: list[int] = []
    voter_c: list[int] = []

    model.eval()
    offset = 0
    with torch.no_grad():
        for batch in dl:
            for k in batch:
                batch[k] = batch[k].to(device)
            logits = model(batch)["logits"]
            prob = F.softmax(logits, dim=-1).cpu().numpy()
            preds = prob.argmax(axis=-1)
            labels = batch["labels"].cpu().numpy()
            bs = len(labels)
            batch_windows = windows[offset : offset + bs]
            offset += bs

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            y_prob_rows.extend(prob)
            voters.extend(w.voter_id for w in batch_windows)
            dao.extend(int(w.dao_cluster) for w in batch_windows)
            voter_c.extend(int(w.voter_cluster) for w in batch_windows)

    y_prob = np.stack(y_prob_rows, axis=0)
    return y_true, y_pred, y_prob, voters, dao, voter_c, len(windows)


def run_comprehensive_evaluation(
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
    split_label: str = "test",
    min_group_n: int = 5,
    calibration_bins: int = 15,
) -> Dict[str, Any]:
    """
    Full evaluation: classification, calibration (ECE + reliability plots),
    confusion matrices, and metrics stratified by DAO / voter cluster.
    """
    out = Path(output_dir) / split_label
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_prob, voters, dao, voter_c, n_windows = _run_inference(
        dataset_csv=dataset_csv,
        artifacts_dir=artifacts_dir,
        preprocessor=preprocessor,
        cfg_train=cfg_train,
        voter_ids=voter_ids,
        seed=seed,
        batch_size=batch_size,
        max_windows=max_windows,
    )

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    prob_cols = [f"prob_{name}" for name in CLASS_NAMES]
    detail = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "voter": voters,
            "dao_cluster": dao,
            "voter_cluster": voter_c,
            **{prob_cols[i]: y_prob[:, i] for i in range(len(CLASS_NAMES))},
        }
    )
    detail.to_csv(out / "predictions.csv", index=False)

    classification_report_dataframe(y_true, y_pred).to_csv(out / "classification_report.csv")
    pd.DataFrame(metrics["per_class"]).T.to_csv(out / "per_class_metrics.csv")

    from sklearn.metrics import confusion_matrix

    cm_arr = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    pd.DataFrame(cm_arr, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out / "confusion_matrix_counts.csv")

    plot_confusion_matrix(y_true, y_pred, fig_dir / "confusion_matrix_counts.png", title=f"Confusion matrix ({split_label})")
    plot_confusion_matrix(
        y_true,
        y_pred,
        fig_dir / "confusion_matrix_normalized_true.png",
        normalize="true",
        title=f"Row-normalized CM ({split_label})",
    )

    reliability_bin_table(y_true, y_prob, n_bins=calibration_bins).to_csv(out / "reliability_bins.csv", index=False)
    plot_reliability_diagram(
        y_true,
        y_prob,
        fig_dir / "reliability_diagram_ovr.png",
        n_bins=calibration_bins,
        title=f"Reliability diagram — one-vs-rest ({split_label})",
    )
    plot_max_confidence_reliability(
        y_true,
        y_prob,
        fig_dir / "reliability_max_confidence.png",
        n_bins=calibration_bins,
        title=f"Max-confidence reliability ({split_label})",
    )

    dao_metrics = metrics_by_group(detail, "dao_cluster", min_n=min_group_n, y_prob_cols=prob_cols)
    voter_metrics = metrics_by_group(detail, "voter_cluster", min_n=min_group_n, y_prob_cols=prob_cols)
    dao_metrics.to_csv(out / "metrics_by_dao_cluster.csv", index=False)
    voter_metrics.to_csv(out / "metrics_by_voter_cluster.csv", index=False)

    summary = {
        "split": split_label,
        "n_windows": n_windows,
        "n_voters": int(detail["voter"].nunique()),
        **{k: metrics[k] for k in metrics if k != "per_class" and k != "ece_per_class"},
        "per_class": metrics["per_class"],
        "ece_per_class": metrics.get("ece_per_class", {}),
        "outputs": {
            "dir": str(out),
            "predictions_csv": str(out / "predictions.csv"),
            "figures_dir": str(fig_dir),
        },
    }
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_metrics_markdown(metrics, out / "metrics_report.md", split_label=split_label)

    legacy_summary = {
        "split": split_label,
        "n_windows": n_windows,
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "matthews_corrcoef": metrics["matthews_corrcoef"],
        "ece": metrics.get("ece"),
        "accuracy": metrics["accuracy"],
    }
    return legacy_summary


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
    comprehensive: bool = True,
    min_group_n: int = 5,
    calibration_bins: int = 15,
) -> Dict[str, Any]:
    """
    Evaluate behaviour model. By default runs comprehensive metrics under
    ``output_dir/{split_label}/``. Set comprehensive=False for legacy flat CSV outputs.
    """
    if comprehensive:
        return run_comprehensive_evaluation(
            dataset_csv=dataset_csv,
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
            preprocessor=preprocessor,
            cfg_train=cfg_train,
            voter_ids=voter_ids,
            seed=seed,
            batch_size=batch_size,
            max_windows=max_windows,
            split_label=split_label,
            min_group_n=min_group_n,
            calibration_bins=calibration_bins,
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_prob, _, dao, voter_c, n_windows = _run_inference(
        dataset_csv=dataset_csv,
        artifacts_dir=artifacts_dir,
        preprocessor=preprocessor,
        cfg_train=cfg_train,
        voter_ids=voter_ids,
        seed=seed,
        batch_size=batch_size,
        max_windows=max_windows,
    )

    rep = classification_report_dataframe(y_true, y_pred)
    rep.to_csv(out / f"classification_report_{split_label}.csv")
    from sklearn.metrics import confusion_matrix as sk_cm

    cm = sk_cm(y_true, y_pred, labels=[0, 1, 2])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out / f"confusion_matrix_{split_label}.csv")

    detail = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "dao_cluster": dao, "voter_cluster": voter_c})
    detail.to_csv(out / f"predictions_with_clusters_{split_label}.csv", index=False)

    rows = []
    for key, group_col in [("dao_cluster", "dao_cluster"), ("voter_cluster", "voter_cluster")]:
        gdf = metrics_by_group(detail, group_col, min_n=min_group_n)
        for _, row in gdf.iterrows():
            rows.append({"group_type": key, "group_id": row["group_id"], "macro_f1": row["macro_f1"], "n": row["n"]})
    pd.DataFrame(rows).to_csv(out / f"macro_f1_by_cluster_{split_label}.csv", index=False)

    m = compute_classification_metrics(y_true, y_pred, y_prob)
    summary = {
        "split": split_label,
        "n_windows": n_windows,
        "macro_f1": m["macro_f1"],
        "accuracy": m["accuracy"],
    }
    (out / f"summary_{split_label}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
