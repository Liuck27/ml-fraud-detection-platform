"""Shared evaluation utilities for XGBoost and Autoencoder models.

Used by both training scripts to compute metrics, find optimal decision
thresholds, and produce ROC / PR curve figures for MLflow logging.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe on servers and CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return a dict of classification metrics evaluated at *threshold*.

    Metrics: auc_roc, pr_auc, f1, precision, recall, threshold.
    Reports PR-AUC (average_precision_score) alongside ROC-AUC because
    PR-AUC is more informative on highly imbalanced datasets.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
) -> float:
    """Sweep the PR curve and return the threshold that minimises total cost.

    cost_fn > cost_fp reflects the asymmetry in fraud detection: a missed
    fraud (FN) costs more than a false alarm (FP).  Default ratio is 10:1.
    """
    y_pred_proba = np.asarray(y_pred_proba)
    _, _, thresholds = precision_recall_curve(y_true, y_pred_proba)
    if thresholds.size == 0:
        return 0.5

    # Vectorised sweep: sort scores descending once, then cumulative sums
    # give the TP/FP counts at every candidate threshold. The naive version
    # re-thresholds the full array per candidate — O(n * k) on ~57k rows.
    order = np.argsort(y_pred_proba)[::-1]
    y_sorted = np.asarray(y_true)[order].astype(np.int64)
    scores_desc = y_pred_proba[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    # For each threshold t, predictions are (score >= t); the index of the
    # last qualifying sample in the descending sort gives its TP/FP counts.
    count_ge = scores_desc.size - np.searchsorted(
        scores_desc[::-1], thresholds, side="left"
    )
    last_idx = count_ge - 1
    fns = tps[-1] - tps[last_idx]
    costs = cost_fp * fps[last_idx] + cost_fn * fns

    # argmin takes the first minimum; thresholds are ascending, so ties
    # resolve to the lowest threshold — same behaviour as the loop version.
    return float(thresholds[int(np.argmin(costs))])


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "ROC Curve",
) -> plt.Figure:
    """Return a matplotlib Figure containing the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC-ROC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    """Return a matplotlib Figure containing the Precision-Recall curve."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls, precisions, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
