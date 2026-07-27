"""Tests for T-Learner model helpers."""

import numpy as np
import pandas as pd
import pytest

from uplift_modeling.models.t_learner import predict_t_learner_scores


class DummyResponseModel:
    """Minimal response model with a predict_proba contract."""

    def __init__(self, positive_scores: np.ndarray) -> None:
        self.positive_scores = positive_scores

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if len(features) != len(self.positive_scores):
            raise ValueError("features and scores must have the same length.")

        return np.column_stack(
            [1.0 - self.positive_scores, self.positive_scores]
        )


def test_predict_t_learner_scores_returns_treatment_minus_control() -> None:
    """T-Learner scores are mu1(x) minus mu0(x)."""
    features = pd.DataFrame({"f0": [0.1, 0.2]})
    treatment_model = DummyResponseModel(np.array([0.8, 0.2]))
    control_model = DummyResponseModel(np.array([0.3, 0.4]))

    scores = predict_t_learner_scores(
        treatment_model,
        control_model,
        features,
    )

    assert scores == pytest.approx([0.5, -0.2])
