"""Reusable binary classification metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def calculate_binary_metrics(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    """Calculate binary classification metrics from labels and scores."""
    labels = np.asarray(y_true)
    score_values = np.asarray(scores)

    if labels.shape[0] == 0:
        raise ValueError("Cannot calculate binary metrics for zero rows.")

    if labels.shape[0] != score_values.shape[0]:
        raise ValueError(
            "y_true and scores must have the same length. "
            f"Received: {labels.shape[0]} and {score_values.shape[0]}"
        )

    unique_labels = set(np.unique(labels))
    invalid_labels = sorted(unique_labels.difference({0, 1}))

    if invalid_labels:
        raise ValueError(
            "y_true must contain only binary labels 0 and 1. "
            f"Received invalid labels: {invalid_labels}"
        )

    if unique_labels != {0, 1}:
        raise ValueError(
            "y_true must contain both classes 0 and 1 to calculate "
            f"ROC AUC. Received classes: {sorted(unique_labels)}"
        )

    if not np.isfinite(score_values).all():
        raise ValueError("scores must contain only finite values.")

    return {
        "roc_auc": float(roc_auc_score(labels, score_values)),
        "average_precision": float(
            average_precision_score(labels, score_values)
        ),
        "log_loss": float(log_loss(labels, score_values, labels=[0, 1])),
        "positive_rate": float(labels.mean()),
        "row_count": int(labels.shape[0]),
    }