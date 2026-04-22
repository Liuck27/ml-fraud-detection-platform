"""Unit tests for training/evaluate.py.

Tests run with the training venv:
    make test-training

No MLflow or I/O — pure numeric functions only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the training/ directory importable without installing it as a package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate import compute_metrics, find_optimal_threshold  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_labels() -> tuple[np.ndarray, np.ndarray]:
    """y_true and y_score where score perfectly separates classes."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.1, 0.9, 0.95])
    return y_true, y_score


def _noisy_labels() -> tuple[np.ndarray, np.ndarray]:
    """y_true and y_score with realistic imbalance (20% fraud)."""
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 80 + [1] * 20)
    y_score = rng.uniform(0, 1, size=100)
    # Push fraud scores higher so AUC > 0.5
    y_score[y_true == 1] = np.clip(y_score[y_true == 1] + 0.4, 0, 1)
    return y_true, y_score


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_expected_keys() -> None:
    y_true, y_score = _perfect_labels()
    metrics = compute_metrics(y_true, y_score, threshold=0.5)

    assert set(metrics.keys()) == {
        "auc_roc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "threshold",
    }


def test_compute_metrics_perfect_classifier() -> None:
    y_true, y_score = _perfect_labels()
    metrics = compute_metrics(y_true, y_score, threshold=0.5)

    assert metrics["auc_roc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["threshold"] == pytest.approx(0.5)


def test_compute_metrics_threshold_stored() -> None:
    y_true, y_score = _noisy_labels()
    metrics = compute_metrics(y_true, y_score, threshold=0.3)
    assert metrics["threshold"] == pytest.approx(0.3)


def test_compute_metrics_values_in_valid_range() -> None:
    y_true, y_score = _noisy_labels()
    metrics = compute_metrics(y_true, y_score)

    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} is outside [0, 1]"


# ---------------------------------------------------------------------------
# find_optimal_threshold
# ---------------------------------------------------------------------------


def test_find_optimal_threshold_returns_float() -> None:
    y_true, y_score = _perfect_labels()
    t = find_optimal_threshold(y_true, y_score)
    assert isinstance(t, float)


def test_find_optimal_threshold_in_valid_range() -> None:
    y_true, y_score = _noisy_labels()
    t = find_optimal_threshold(y_true, y_score, cost_fp=1.0, cost_fn=10.0)
    assert 0.0 <= t <= 1.0


def test_find_optimal_threshold_cost_sensitivity() -> None:
    """A higher FN cost should push the threshold lower (catch more fraud)."""
    y_true, y_score = _noisy_labels()
    t_balanced = find_optimal_threshold(y_true, y_score, cost_fp=1.0, cost_fn=1.0)
    t_fn_heavy = find_optimal_threshold(y_true, y_score, cost_fp=1.0, cost_fn=50.0)
    # Higher FN cost → lower threshold (flag more transactions as fraud)
    assert t_fn_heavy <= t_balanced
