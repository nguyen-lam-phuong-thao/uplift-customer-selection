"""Tests for saved model scoring metadata validation."""

import numpy as np
import pandas as pd
import pytest

from uplift_modeling.models import scoring


class _ProbabilityModel:
    """Small predict_proba test double."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return a two-column probability matrix."""
        probabilities = np.full(len(features), self.probability)
        return np.column_stack([1.0 - probabilities, probabilities])


class _RegressionModel:
    """Small predict test double."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return a constant regression score."""
        return np.full(len(features), self.value)


def test_response_model_scoring_uses_exact_model_uri(monkeypatch) -> None:
    """Response scoring loads the configured model URI."""
    loaded_uris = []

    def load_model(model_uri: str):
        loaded_uris.append(model_uri)
        return _ProbabilityModel(0.7)

    monkeypatch.setattr(scoring.mlflow.lightgbm, "load_model", load_model)
    score_batch = scoring.build_policy_score_batch(
        policy="treated_response_lgbm",
        model_artifact={
            "artifact_type": "model_provenance",
            "policy_name": "treated_response_lgbm",
            "model_kind": "response",
            "mlflow_run_id": "run-001",
            "model_uri": "runs:/run-001/model",
        },
    )

    scores = score_batch(pd.DataFrame({"f0": [1.0, 2.0]}))

    assert loaded_uris == ["runs:/run-001/model"]
    assert scores.tolist() == [0.7, 0.7]


def test_t_learner_scoring_uses_exact_model_uris(monkeypatch) -> None:
    """T-Learner scoring loads treatment and control model URIs."""
    models = {
        "runs:/run-002/treatment_model": _ProbabilityModel(0.8),
        "runs:/run-002/control_model": _ProbabilityModel(0.3),
    }
    loaded_uris = []

    def load_model(model_uri: str):
        loaded_uris.append(model_uri)
        return models[model_uri]

    monkeypatch.setattr(scoring.mlflow.lightgbm, "load_model", load_model)
    score_batch = scoring.build_policy_score_batch(
        policy="t_learner_lgbm",
        model_artifact={
            "artifact_type": "model_provenance",
            "policy_name": "t_learner_lgbm",
            "model_kind": "t_learner",
            "mlflow_run_id": "run-002",
            "treatment_model_uri": "runs:/run-002/treatment_model",
            "control_model_uri": "runs:/run-002/control_model",
        },
    )

    scores = score_batch(pd.DataFrame({"f0": [1.0, 2.0]}))

    assert loaded_uris == [
        "runs:/run-002/treatment_model",
        "runs:/run-002/control_model",
    ]
    assert scores.tolist() == pytest.approx([0.5, 0.5])


def test_x_learner_scoring_uses_exact_model_uris(monkeypatch) -> None:
    """X-Learner scoring loads treatment-effect and control-effect URIs."""
    models = {
        "runs:/run-003/tau1_model": _RegressionModel(0.6),
        "runs:/run-003/tau0_model": _RegressionModel(0.2),
    }
    loaded_uris = []

    def load_model(model_uri: str):
        loaded_uris.append(model_uri)
        return models[model_uri]

    monkeypatch.setattr(scoring.mlflow.lightgbm, "load_model", load_model)
    score_batch = scoring.build_policy_score_batch(
        policy="x_learner_lgbm",
        model_artifact={
            "artifact_type": "model_provenance",
            "policy_name": "x_learner_lgbm",
            "model_kind": "x_learner",
            "mlflow_run_id": "run-003",
            "treatment_effect_model_uri": "runs:/run-003/tau1_model",
            "control_effect_model_uri": "runs:/run-003/tau0_model",
            "constant_treatment_rate_weight": 0.25,
        },
    )

    scores = score_batch(pd.DataFrame({"f0": [1.0, 2.0]}))

    assert loaded_uris == ["runs:/run-003/tau1_model", "runs:/run-003/tau0_model"]
    assert scores.tolist() == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize(
    ("policy", "model_artifact"),
    [
        (
            "treated_response_lgbm",
            {
                "artifact_type": "model_provenance",
                "policy_name": "treated_response_lgbm",
                "model_kind": "response",
                "mlflow_run_id": "run-001",
                "model_uri": "runs:/run-999/model",
            },
        ),
        (
            "t_learner_lgbm",
            {
                "artifact_type": "model_provenance",
                "policy_name": "t_learner_lgbm",
                "model_kind": "t_learner",
                "mlflow_run_id": "run-002",
                "treatment_model_uri": (
                    "runs:/run-002/treatment_model"
                ),
                "control_model_uri": (
                    "runs:/run-999/control_model"
                ),
            },
        ),
        (
            "x_learner_lgbm",
            {
                "artifact_type": "model_provenance",
                "policy_name": "x_learner_lgbm",
                "model_kind": "x_learner",
                "mlflow_run_id": "run-003",
                "treatment_effect_model_uri": (
                    "runs:/run-003/tau1_model"
                ),
                "control_effect_model_uri": (
                    "runs:/run-999/tau0_model"
                ),
                "constant_treatment_rate_weight": 0.25,
            },
        ),
    ],
)
def test_scoring_rejects_model_uri_from_different_run(
    policy,
    model_artifact,
    monkeypatch,
) -> None:
    """Every model URI must belong to the declared MLflow run."""

    def fail_load_model(model_uri: str):
        raise AssertionError(
            "MLflow must not load a model with mismatched provenance."
        )

    monkeypatch.setattr(
        scoring.mlflow.lightgbm,
        "load_model",
        fail_load_model,
    )

    with pytest.raises(
        ValueError,
        match="must belong to mlflow_run_id",
    ):
        scoring.build_policy_score_batch(
            policy=policy,
            model_artifact=model_artifact,
        )
        

def test_scoring_fails_without_mlflow_run_id() -> None:
    """Missing run provenance fails before model loading."""
    with pytest.raises(ValueError, match="mlflow_run_id"):
        scoring.build_policy_score_batch(
            policy="treated_response_lgbm",
            model_artifact={
                "artifact_type": "model_provenance",
                "policy_name": "treated_response_lgbm",
                "model_kind": "response",
                "model_uri": "runs:/run-001/model",
            },
        )


def test_scoring_fails_for_unsupported_model_kind() -> None:
    """Unsupported policy metadata fails clearly."""
    with pytest.raises(ValueError, match="Unsupported model artifact kind"):
        scoring.build_policy_score_batch(
            policy="unknown_policy",
            model_artifact={
                "artifact_type": "model_provenance",
                "policy_name": "unknown_policy",
                "model_kind": "unknown_learner",
                "mlflow_run_id": "run-004",
            },
        )


def test_scoring_fails_for_incomplete_model_metadata() -> None:
    """Incomplete policy metadata fails clearly."""
    with pytest.raises(ValueError, match="control_model_uri"):
        scoring.build_policy_score_batch(
            policy="t_learner_lgbm",
            model_artifact={
                "artifact_type": "model_provenance",
                "policy_name": "t_learner_lgbm",
                "model_kind": "t_learner",
                "mlflow_run_id": "run-005",
                "treatment_model_uri": "runs:/run-005/treatment_model",
            },
        )


def test_x_learner_scoring_fails_for_incomplete_model_metadata() -> None:
    """Incomplete X-Learner metadata fails before model loading."""
    with pytest.raises(ValueError, match="control_effect_model_uri"):
        scoring.build_policy_score_batch(
            policy="x_learner_lgbm",
            model_artifact={
                "artifact_type": "model_provenance",
                "policy_name": "x_learner_lgbm",
                "model_kind": "x_learner",
                "mlflow_run_id": "run-006",
                "treatment_effect_model_uri": "runs:/run-006/tau1_model",
            },
        )


def test_x_learner_scoring_accepts_mlflow3_model_uris(
    monkeypatch,
) -> None:
    """X-Learner scoring accepts MLflow 3 logged models."""
    models = {
        "models:/m-tau1": _RegressionModel(0.6),
        "models:/m-tau0": _RegressionModel(0.2),
    }
    loaded_uris = []

    def load_model(model_uri: str):
        loaded_uris.append(model_uri)
        return models[model_uri]

    monkeypatch.setattr(
        scoring.mlflow.lightgbm,
        "load_model",
        load_model,
    )

    score_batch = scoring.build_policy_score_batch(
        policy="x_learner_lgbm",
        model_artifact={
            "artifact_type": "model_provenance",
            "policy_name": "x_learner_lgbm",
            "model_kind": "x_learner",
            "mlflow_run_id": "run-003",
            "treatment_effect_model_uri": (
                "models:/m-tau1"
            ),
            "control_effect_model_uri": (
                "models:/m-tau0"
            ),
            "constant_treatment_rate_weight": 0.25,
        },
    )

    scores = score_batch(
        pd.DataFrame({"f0": [1.0, 2.0]})
    )

    assert loaded_uris == [
        "models:/m-tau1",
        "models:/m-tau0",
    ]
    assert scores.tolist() == pytest.approx(
        [0.5, 0.5]
    )        
