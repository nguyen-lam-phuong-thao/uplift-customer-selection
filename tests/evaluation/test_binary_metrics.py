"""Tests for reusable binary metrics."""

import numpy as np
import pytest

from uplift_modeling.evaluation.binary_metrics import calculate_binary_metrics


def test_calculate_binary_metrics_returns_expected_keys() -> None:
    """Binary metrics include model-quality and summary fields."""
    metrics = calculate_binary_metrics(
        y_true=np.array([0, 0, 1, 1]),
        scores=np.array([0.1, 0.3, 0.7, 0.9]),
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["positive_rate"] == pytest.approx(0.5)
    assert metrics["row_count"] == 4
    assert metrics["log_loss"] > 0


def test_calculate_binary_metrics_rejects_length_mismatch() -> None:
    """Labels and scores must describe the same rows."""
    with pytest.raises(ValueError, match="same length"):
        calculate_binary_metrics(
            y_true=np.array([0, 1]),
            scores=np.array([0.2]),
        )


def test_calculate_binary_metrics_rejects_single_class() -> None:
    """ROC AUC requires both binary classes."""
    with pytest.raises(ValueError, match="both classes 0 and 1"):
        calculate_binary_metrics(
            y_true=np.array([0, 0, 0]),
            scores=np.array([0.1, 0.2, 0.3]),
        )
