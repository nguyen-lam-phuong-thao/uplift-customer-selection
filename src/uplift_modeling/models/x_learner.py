"""X-Learner helpers built from outcome and effect models."""

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import (
    LGBMClassifier,
    LGBMRegressor,
    early_stopping,
    log_evaluation,
)

from uplift_modeling.models.response_model import (
    build_response_model,
    fit_response_model,
)


XLearnerModels = tuple[
    LGBMClassifier,
    LGBMClassifier,
    LGBMRegressor,
    LGBMRegressor,
    float,
]


def build_effect_model(params: dict[str, Any]) -> LGBMRegressor:
    """Build a LightGBM regression model for imputed treatment effects."""
    effect_params = dict(params)
    effect_params["objective"] = "regression"
    return LGBMRegressor(**effect_params)


def fit_effect_model(
    model: LGBMRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series | np.ndarray,
    early_stopping_rounds: int,
    log_evaluation_period: int,
) -> LGBMRegressor:
    """Fit an effect model with validation callbacks."""
    callbacks = []

    if early_stopping_rounds > 0:
        callbacks.append(early_stopping(early_stopping_rounds))

    if log_evaluation_period > 0:
        callbacks.append(log_evaluation(log_evaluation_period))

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
        callbacks=callbacks,
    )
    return model


def fit_x_learner(
    treatment_train: pd.DataFrame,
    control_train: pd.DataFrame,
    treatment_valid: pd.DataFrame,
    control_valid: pd.DataFrame,
    feature_columns: tuple[str, ...],
    outcome_column: str,
    model_params: dict[str, Any],
    early_stopping_rounds: int,
    log_evaluation_period: int,
) -> XLearnerModels:
    """Fit X-Learner outcome and effect models."""
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

    treatment_features = treatment_train.loc[:, feature_columns]
    control_features = control_train.loc[:, feature_columns]
    treatment_valid_features = treatment_valid.loc[:, feature_columns]
    control_valid_features = control_valid.loc[:, feature_columns]

    treatment_effect_train = (
        treatment_train[outcome_column].to_numpy()
        - control_model.predict_proba(treatment_features)[:, 1]
    )
    control_effect_train = (
        treatment_model.predict_proba(control_features)[:, 1]
        - control_train[outcome_column].to_numpy()
    )
    treatment_effect_valid = (
        treatment_valid[outcome_column].to_numpy()
        - control_model.predict_proba(treatment_valid_features)[:, 1]
    )
    control_effect_valid = (
        treatment_model.predict_proba(control_valid_features)[:, 1]
        - control_valid[outcome_column].to_numpy()
    )

    treatment_effect_model = build_effect_model(dict(model_params))
    control_effect_model = build_effect_model(dict(model_params))

    fit_effect_model(
        model=treatment_effect_model,
        X_train=treatment_features,
        y_train=treatment_effect_train,
        X_valid=treatment_valid_features,
        y_valid=treatment_effect_valid,
        early_stopping_rounds=early_stopping_rounds,
        log_evaluation_period=log_evaluation_period,
    )
    fit_effect_model(
        model=control_effect_model,
        X_train=control_features,
        y_train=control_effect_train,
        X_valid=control_valid_features,
        y_valid=control_effect_valid,
        early_stopping_rounds=early_stopping_rounds,
        log_evaluation_period=log_evaluation_period,
    )

    treatment_count = len(treatment_train)
    control_count = len(control_train)
    treatment_weight = treatment_count / (treatment_count + control_count)

    return (
        treatment_model,
        control_model,
        treatment_effect_model,
        control_effect_model,
        treatment_weight,
    )


def predict_x_learner_scores(
    treatment_effect_model: LGBMRegressor,
    control_effect_model: LGBMRegressor,
    treatment_weight: float,
    features: pd.DataFrame,
) -> np.ndarray:
    """Predict uplift scores with a constant treated-share weight."""
    treatment_effect_scores = treatment_effect_model.predict(features)
    control_effect_scores = control_effect_model.predict(features)
    return (
        treatment_weight * control_effect_scores
        + (1 - treatment_weight) * treatment_effect_scores
    )
