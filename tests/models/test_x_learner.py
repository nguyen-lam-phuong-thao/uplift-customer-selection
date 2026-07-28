"""Tests for X-Learner model helpers."""

import numpy as np
import pandas as pd
import pytest

from uplift_modeling.models.x_learner import (
    build_effect_model,
    calculate_control_pseudo_effects,
    calculate_treatment_pseudo_effects,
    combine_x_learner_scores,
    predict_x_learner_scores,
)


class DummyEffectModel:
    """Minimal effect model with a predict contract."""

    def __init__(self, scores: np.ndarray) -> None:
        self.scores = scores

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if len(features) != len(self.scores):
            raise ValueError("features and scores must have the same length.")

        return self.scores


def test_predict_x_learner_scores_uses_constant_weight() -> None:
    """X-Learner scores are g * tau0(x) + (1 - g) * tau1(x)."""
    features = pd.DataFrame({"f0": [0.1, 0.2]})
    treatment_effect_model = DummyEffectModel(np.array([0.6, -0.2]))
    control_effect_model = DummyEffectModel(np.array([0.2, 0.4]))

    scores = predict_x_learner_scores(
        treatment_effect_model=treatment_effect_model,
        control_effect_model=control_effect_model,
        constant_treatment_rate_weight=0.75,
        features=features,
    )

    assert scores == pytest.approx([0.3, 0.25])


def test_calculate_treatment_pseudo_effects_uses_d1_formula() -> None:
    """D1 is exactly Y_treated - mu0(X_treated)."""
    outcomes = np.array([1.0, 0.0, 1.0])
    control_predictions = np.array([0.2, 0.3, 0.8])

    pseudo_effects = calculate_treatment_pseudo_effects(
        treatment_outcomes=outcomes,
        control_predictions=control_predictions,
    )

    assert pseudo_effects == pytest.approx([0.8, -0.3, 0.2])


def test_calculate_control_pseudo_effects_uses_d0_formula() -> None:
    """D0 is exactly mu1(X_control) - Y_control."""
    treatment_predictions = np.array([0.7, 0.1, 0.4])
    outcomes = np.array([0.0, 1.0, 0.0])

    pseudo_effects = calculate_control_pseudo_effects(
        treatment_predictions=treatment_predictions,
        control_outcomes=outcomes,
    )

    assert pseudo_effects == pytest.approx([0.7, -0.9, 0.4])


def test_combine_x_learner_scores_uses_constant_weight_formula() -> None:
    """Combined uplift is exactly g * tau0 + (1 - g) * tau1."""
    scores = combine_x_learner_scores(
        treatment_effect_scores=np.array([0.6, -0.2]),
        control_effect_scores=np.array([0.2, 0.4]),
        constant_treatment_rate_weight=0.75,
    )

    assert scores == pytest.approx([0.3, 0.25])


def test_non_finite_treatment_pseudo_effects_raise_error() -> None:
    """Pseudo-effects must be finite before effect-model fitting."""
    with pytest.raises(ValueError, match="D1 pseudo-effect"):
        calculate_treatment_pseudo_effects(
            treatment_outcomes=np.array([1.0]),
            control_predictions=np.array([np.inf]),
        )


def test_build_effect_model_uses_regression_compatible_params() -> None:
    """Effect regressors drop classifier-only params and use regression."""
    model = build_effect_model(
        {
            "objective": "binary",
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "is_unbalance": True,
            "class_weight": "balanced",
            "scale_pos_weight": 2,
            "random_state": 42,
        }
    )
    params = model.get_params()

    assert params["objective"] == "regression"
    assert params["n_estimators"] == 10
    assert params["learning_rate"] == 0.1
    assert params["num_leaves"] == 7
    assert params["random_state"] == 42
    assert "is_unbalance" not in params
    assert params["class_weight"] is None
    assert "scale_pos_weight" not in params
