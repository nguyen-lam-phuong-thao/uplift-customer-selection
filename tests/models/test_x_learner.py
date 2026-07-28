"""Tests for X-Learner model helpers."""

import numpy as np
import pandas as pd
import pytest

from uplift_modeling.models.x_learner import predict_x_learner_scores


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
        treatment_weight=0.75,
        features=features,
    )

    assert scores == pytest.approx([0.3, 0.25])
