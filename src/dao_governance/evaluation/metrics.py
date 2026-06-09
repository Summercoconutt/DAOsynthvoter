"""Classification, calibration, and grouped metrics for behaviour-model evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

CLASS_NAMES = ["FOR", "AGAINST", "ABSTAIN"]
LABELS = [0, 1, 2]


def attach_eval_cluster_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve integer cluster IDs for reporting (pre-normalisation semantics)."""
    out = df.copy()
    for col, eval_col in (("dao_cluster", "dao_cluster_eval"), ("voter_cluster", "voter_cluster_eval")):
        if eval_col not in out.columns:
            out[eval_col] = pd.to_numeric(out.get(col, -1), errors="coerce").fillna(-1).astype(int)
    return out


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[np.ndarray] = None,
    *,
    labels: Sequence[int] = LABELS,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, zero_division=0
    )

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    metrics: Dict[str, Any] = {
        "n_samples": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        "cohen_kappa": float(cohen_kappa_score(y_true_arr, y_pred_arr)),
        "per_class": per_class,
    }

    if y_prob is not None and len(y_prob) == len(y_true_arr):
        prob = np.asarray(y_prob, dtype=float)
        prob = np.clip(prob, 1e-12, 1.0 - 1e-12)
        prob = prob / prob.sum(axis=1, keepdims=True)
        metrics["log_loss"] = float(log_loss(y_true_arr, prob, labels=labels))
        metrics["brier_score"] = float(np.mean(np.sum((prob - _one_hot(y_true_arr, len(labels))) ** 2, axis=1)))
        cal = compute_calibration_metrics(y_true_arr, prob, labels=labels, class_names=class_names)
        metrics.update(cal)

    return metrics


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


def compute_ece(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
    labels: Sequence[int] = LABELS,
) -> float:
    """Multiclass ECE: average of one-vs-rest bin-wise calibration errors."""
    y_true_arr = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    prob = np.clip(prob, 1e-12, 1.0 - 1e-12)
    prob = prob / prob.sum(axis=1, keepdims=True)

    eces = []
    for j, _ in enumerate(labels):
        binary_true = (y_true_arr == j).astype(float)
        binary_prob = prob[:, j]
        eces.append(_binary_ece(binary_true, binary_prob, n_bins=n_bins))
    return float(np.mean(eces))


def _binary_ece(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


def compute_calibration_metrics(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
    labels: Sequence[int] = LABELS,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    prob = np.clip(prob, 1e-12, 1.0 - 1e-12)
    prob = prob / prob.sum(axis=1, keepdims=True)

    ece_overall = compute_ece(y_true_arr, prob, n_bins=n_bins, labels=labels)
    per_class_ece = {}
    for j, name in enumerate(class_names):
        binary_true = (y_true_arr == j).astype(float)
        per_class_ece[name] = float(_binary_ece(binary_true, prob[:, j], n_bins=n_bins))

    conf = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == y_true_arr).astype(float)
    max_conf_ece = float(_binary_ece(correct, conf, n_bins=n_bins))
    avg_confidence = float(conf.mean())
    avg_accuracy = float(correct.mean())

    return {
        "ece": ece_overall,
        "ece_max_confidence": max_conf_ece,
        "ece_per_class": per_class_ece,
        "avg_confidence": avg_confidence,
        "avg_accuracy": avg_accuracy,
        "confidence_accuracy_gap": float(avg_confidence - avg_accuracy),
    }


def reliability_bin_table(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
    labels: Sequence[int] = LABELS,
    class_names: Sequence[str] = CLASS_NAMES,
) -> pd.DataFrame:
    """One-vs-rest reliability bins for each class."""
    y_true_arr = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    prob = np.clip(prob, 1e-12, 1.0 - 1e-12)
    prob = prob / prob.sum(axis=1, keepdims=True)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    for j, name in enumerate(class_names):
        binary_true = (y_true_arr == j).astype(float)
        p = prob[:, j]
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            if i == n_bins - 1:
                mask = (p >= lo) & (p <= hi)
            else:
                mask = (p >= lo) & (p < hi)
            count = int(mask.sum())
            if count == 0:
                continue
            rows.append(
                {
                    "class": name,
                    "bin": i,
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "count": count,
                    "avg_confidence": float(p[mask].mean()),
                    "empirical_frequency": float(binary_true[mask].mean()),
                    "calibration_gap": float(abs(p[mask].mean() - binary_true[mask].mean())),
                }
            )
    return pd.DataFrame(rows)


def plot_reliability_diagram(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    out_path: Path,
    *,
    n_bins: int = 15,
    class_names: Sequence[str] = CLASS_NAMES,
    title: str = "Reliability diagram (one-vs-rest)",
) -> None:
    y_true_arr = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    prob = np.clip(prob, 1e-12, 1.0 - 1e-12)
    prob = prob / prob.sum(axis=1, keepdims=True)

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(4.5 * n_classes, 4.2), squeeze=False)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for j, name in enumerate(class_names):
        ax = axes[0, j]
        binary_true = (y_true_arr == j).astype(float)
        p = prob[:, j]
        accs, confs, counts = [], [], []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            if i == n_bins - 1:
                mask = (p >= lo) & (p <= hi)
            else:
                mask = (p >= lo) & (p < hi)
            if not mask.any():
                accs.append(np.nan)
                confs.append(np.nan)
                counts.append(0)
                continue
            accs.append(float(binary_true[mask].mean()))
            confs.append(float(p[mask].mean()))
            counts.append(int(mask.sum()))

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.plot(bin_centers, confs, marker="o", label="Avg confidence")
        ax.plot(bin_centers, accs, marker="s", label="Empirical freq")
        ax.set_title(f"{name}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_max_confidence_reliability(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    out_path: Path,
    *,
    n_bins: int = 15,
    title: str = "Max-confidence reliability",
) -> None:
    y_true_arr = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    prob = prob / prob.sum(axis=1, keepdims=True)
    conf = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == y_true_arr).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    accs, confs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            accs.append(np.nan)
            confs.append(np.nan)
            continue
        accs.append(float(correct[mask].mean()))
        confs.append(float(conf[mask].mean()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(bin_centers, confs, marker="o", label="Avg max confidence")
    ax.plot(bin_centers, accs, marker="s", label="Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy / confidence")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    out_path: Path,
    *,
    normalize: Optional[str] = None,
    title: str = "Confusion matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    if normalize == "true":
        cm_plot = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    elif normalize == "pred":
        cm_plot = cm.astype(float) / np.maximum(cm.sum(axis=0, keepdims=True), 1)
    elif normalize == "all":
        cm_plot = cm.astype(float) / max(cm.sum(), 1)
    else:
        cm_plot = cm.astype(float)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm_plot, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_plot[i, j]
            text = f"{val}\n({pct:.1%})" if normalize else str(val)
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def metrics_by_group(
    detail: pd.DataFrame,
    group_col: str,
    *,
    min_n: int = 5,
    y_prob_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    rows = []
    for gid, g in detail.groupby(group_col):
        if len(g) < min_n:
            continue
        y_true = g["y_true"].tolist()
        y_pred = g["y_pred"].tolist()
        y_prob = None
        if y_prob_cols and all(c in g.columns for c in y_prob_cols):
            y_prob = g[y_prob_cols].to_numpy(dtype=float)
        m = compute_classification_metrics(y_true, y_pred, y_prob)
        rows.append(
            {
                "group_col": group_col,
                "group_id": int(gid) if pd.notna(gid) else -1,
                "n": int(len(g)),
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "matthews_corrcoef": m["matthews_corrcoef"],
                "cohen_kappa": m["cohen_kappa"],
                "f1_FOR": m["per_class"]["FOR"]["f1"],
                "f1_AGAINST": m["per_class"]["AGAINST"]["f1"],
                "f1_ABSTAIN": m["per_class"]["ABSTAIN"]["f1"],
                **(
                    {
                        "ece": m.get("ece"),
                        "log_loss": m.get("log_loss"),
                    }
                    if y_prob is not None
                    else {}
                ),
            }
        )
    return pd.DataFrame(rows)


def write_metrics_markdown(metrics: Dict[str, Any], out_path: Path, *, split_label: str) -> None:
    lines = [
        f"# Behaviour model evaluation — `{split_label}`",
        "",
        f"- Samples: **{metrics['n_samples']}**",
        f"- Accuracy: **{metrics['accuracy']:.4f}**",
        f"- Balanced accuracy: **{metrics['balanced_accuracy']:.4f}**",
        f"- Macro F1: **{metrics['macro_f1']:.4f}**",
        f"- Weighted F1: **{metrics['weighted_f1']:.4f}**",
        f"- Matthews CC: **{metrics['matthews_corrcoef']:.4f}**",
        f"- Cohen's kappa: **{metrics['cohen_kappa']:.4f}**",
    ]
    if "log_loss" in metrics:
        lines.append(f"- Log loss: **{metrics['log_loss']:.4f}**")
    if "ece" in metrics:
        lines.extend(
            [
                f"- ECE (avg one-vs-rest): **{metrics['ece']:.4f}**",
                f"- ECE (max-confidence): **{metrics.get('ece_max_confidence', float('nan')):.4f}**",
                f"- Avg confidence: **{metrics.get('avg_confidence', float('nan')):.4f}**",
            ]
        )

    lines.extend(["", "## Per-class F1", ""])
    for name in CLASS_NAMES:
        pc = metrics["per_class"][name]
        lines.append(
            f"- **{name}**: F1={pc['f1']:.4f}, P={pc['precision']:.4f}, "
            f"R={pc['recall']:.4f}, support={pc['support']}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def classification_report_dataframe(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T
