"""X-Learner helpers built from outcome and effect models."""

from typing import Any, NamedTuple

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


CLASSIFIER_ONLY_PARAMS = ("class_weight", "is_unbalance", "scale_pos_weight")


class XLearnerFitResult(NamedTuple):
    """Fitted X-Learner models and training diagnostics."""

    treatment_model: LGBMClassifier
    control_model: LGBMClassifier
    treatment_effect_model: LGBMRegressor
    control_effect_model: LGBMRegressor
    constant_treatment_rate_weight: float
    treatment_effect_summary: dict[str, float | int]
    control_effect_summary: dict[str, float | int]


def build_effect_model(params: dict[str, Any]) -> LGBMRegressor:
    """Build a LightGBM regression model for imputed treatment effects."""
    effect_params = dict(params)
    for classifier_param in CLASSIFIER_ONLY_PARAMS:
        effect_params.pop(classifier_param, None)

    effect_params["objective"] = "regression"
    return LGBMRegressor(**effect_params)


def validate_finite_values(values: np.ndarray, name: str) -> None:
    """Raise when an array contains NaN or infinite values."""
    if not np.isfinite(values).all():
        raise ValueError(f"{name} values must be finite.")


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    """Return compact distribution diagnostics for finite numeric values."""
    values = np.asarray(values, dtype=float)
    validate_finite_values(values, "summary")

    if values.size == 0:
        raise ValueError("summary values must contain at least one value.")

    percentiles = np.percentile(values, [1, 5, 50, 95, 99])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "max": float(np.max(values)),
        "unique_count": int(np.unique(values).size),
    }


def calculate_treatment_pseudo_effects(
    treatment_outcomes: pd.Series | np.ndarray,
    control_predictions: np.ndarray,
) -> np.ndarray:
    """Calculate treated pseudo-effects: D1 = Y_treated - mu0(X_treated)."""
    outcomes = np.asarray(treatment_outcomes, dtype=float)
    predictions = np.asarray(control_predictions, dtype=float)
    if len(outcomes) != len(predictions):
        raise ValueError("D1 outcomes and predictions must have equal length.")

    pseudo_effects = outcomes - predictions
    validate_finite_values(pseudo_effects, "D1 pseudo-effect")
    return pseudo_effects


def calculate_control_pseudo_effects(
    treatment_predictions: np.ndarray,
    control_outcomes: pd.Series | np.ndarray,
) -> np.ndarray:
    """Calculate control pseudo-effects: D0 = mu1(X_control) - Y_control."""
    predictions = np.asarray(treatment_predictions, dtype=float)
    outcomes = np.asarray(control_outcomes, dtype=float)
    if len(predictions) != len(outcomes):
        raise ValueError("D0 predictions and outcomes must have equal length.")

    pseudo_effects = predictions - outcomes
    validate_finite_values(pseudo_effects, "D0 pseudo-effect")
    return pseudo_effects


def combine_x_learner_scores(
    treatment_effect_scores: np.ndarray,
    control_effect_scores: np.ndarray,
    constant_treatment_rate_weight: float,
) -> np.ndarray:
    """Combine effect scores as g * tau0(x) + (1 - g) * tau1(x)."""
    if (
        constant_treatment_rate_weight <= 0.0
        or constant_treatment_rate_weight >= 1.0
    ):
        raise ValueError(
            "constant_treatment_rate_weight must be strictly between 0 and 1."
        )

    treatment_effect_scores = np.asarray(treatment_effect_scores, dtype=float)
    control_effect_scores = np.asarray(control_effect_scores, dtype=float)
    if len(treatment_effect_scores) != len(control_effect_scores):
        raise ValueError("tau1 and tau0 scores must have equal length.")

    scores = (
        constant_treatment_rate_weight
        * control_effect_scores
        + (1.0 - constant_treatment_rate_weight)
        * treatment_effect_scores
    )
    validate_finite_values(scores, "final uplift score")
    return scores


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
) -> XLearnerFitResult:
    """Fit X-Learner outcome and effect models."""
    if treatment_train.empty:
        raise ValueError("X-Learner treatment training rows must be > 0.")
    if control_train.empty:
        raise ValueError("X-Learner control training rows must be > 0.")
    if treatment_valid.empty:
        raise ValueError("X-Learner treatment validation rows must be > 0.")
    if control_valid.empty:
        raise ValueError("X-Learner control validation rows must be > 0.")

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

    treatment_effect_train = calculate_treatment_pseudo_effects(
        treatment_outcomes=treatment_train[outcome_column],
        control_predictions=control_model.predict_proba(treatment_features)[:, 1],
    )
    control_effect_train = calculate_control_pseudo_effects(
        treatment_predictions=treatment_model.predict_proba(control_features)[
            :, 1
        ],
        control_outcomes=control_train[outcome_column],
    )
    treatment_effect_valid = calculate_treatment_pseudo_effects(
        treatment_outcomes=treatment_valid[outcome_column],
        control_predictions=control_model.predict_proba(
            treatment_valid_features
        )[:, 1],
    )
    control_effect_valid = calculate_control_pseudo_effects(
        treatment_predictions=treatment_model.predict_proba(
            control_valid_features
        )[:, 1],
        control_outcomes=control_valid[outcome_column],
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
    constant_treatment_rate_weight = treatment_count / (
        treatment_count + control_count
    )

    return XLearnerFitResult(
        treatment_model=treatment_model,
        control_model=control_model,
        treatment_effect_model=treatment_effect_model,
        control_effect_model=control_effect_model,
        constant_treatment_rate_weight=constant_treatment_rate_weight,
        treatment_effect_summary=summarize_values(treatment_effect_train),
        control_effect_summary=summarize_values(control_effect_train),
    )


def predict_x_learner_scores(
    treatment_effect_model: LGBMRegressor,
    control_effect_model: LGBMRegressor,
    constant_treatment_rate_weight: float,
    features: pd.DataFrame,
) -> np.ndarray:
    """Predict uplift scores with a constant treated-share weight."""
    treatment_effect_scores = treatment_effect_model.predict(features)
    control_effect_scores = control_effect_model.predict(features)
    return combine_x_learner_scores(
        treatment_effect_scores=treatment_effect_scores,
        control_effect_scores=control_effect_scores,
        constant_treatment_rate_weight=constant_treatment_rate_weight,
    )
