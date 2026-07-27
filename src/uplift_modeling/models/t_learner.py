"""T-Learner helpers built from two response models."""

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from uplift_modeling.models.response_model import (
    build_response_model,
    fit_response_model,
)


def fit_t_learner(
    treatment_train: pd.DataFrame,
    control_train: pd.DataFrame,
    treatment_valid: pd.DataFrame,
    control_valid: pd.DataFrame,
    feature_columns: tuple[str, ...],
    outcome_column: str,
    model_params: dict[str, Any],
    early_stopping_rounds: int,
    log_evaluation_period: int,
) -> tuple[LGBMClassifier, LGBMClassifier]:
    """Fit treatment and control response models."""
    treatment_model = build_response_model(dict(model_params))
    control_model = build_response_model(dict(model_params))

    fit_response_model(
        model=treatment_model,
        X_train=treatment_train.loc[:, feature_columns],
        y_train=treatment_train[outcome_column],
        X_valid=treatment_valid.loc[:, feature_columns],
        y_valid=treatment_valid[outcome_column],
        early_stopping_rounds=early_stopping_rounds,
        log_evaluation_period=log_evaluation_period,
    )
    fit_response_model(
        model=control_model,
        X_train=control_train.loc[:, feature_columns],
        y_train=control_train[outcome_column],
        X_valid=control_valid.loc[:, feature_columns],
        y_valid=control_valid[outcome_column],
        early_stopping_rounds=early_stopping_rounds,
        log_evaluation_period=log_evaluation_period,
    )

    return treatment_model, control_model


def predict_t_learner_scores(
    treatment_model: LGBMClassifier,
    control_model: LGBMClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    """Predict uplift scores as mu1(x) minus mu0(x)."""
    treatment_scores = treatment_model.predict_proba(features)[:, 1]
    control_scores = control_model.predict_proba(features)[:, 1]
    return treatment_scores - control_scores
