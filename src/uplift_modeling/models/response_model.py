"""LightGBM response-model helpers for binary outcome prediction."""

from typing import Any
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


FeatureMatrix = pd.DataFrame | np.ndarray
TargetVector = pd.Series | np.ndarray


def build_response_model(params: dict[str, Any]) -> LGBMClassifier:
    """Build a LightGBM binary response model from configuration params."""
    return LGBMClassifier(**params)


def fit_response_model(
    model: LGBMClassifier,
    X_train: FeatureMatrix,
    y_train: TargetVector,
    X_valid: FeatureMatrix,
    y_valid: TargetVector,
    early_stopping_rounds: int,
    log_evaluation_period: int,
) -> LGBMClassifier:
    """Fit a response model with validation callbacks."""
    callbacks = []

    if early_stopping_rounds > 0:
        callbacks.append(early_stopping(early_stopping_rounds))

    if log_evaluation_period > 0:
        callbacks.append(log_evaluation(log_evaluation_period))

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=callbacks,
    )
    return model
