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
    _, _, thresholds = precision_recall_curve(y_true, y_pred_proba)

    best_threshold = 0.5
    best_cost = float("inf")

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)

    return best_threshold


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
