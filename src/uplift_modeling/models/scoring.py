"""Shared scoring helpers for saved uplift model artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import mlflow.lightgbm
import numpy as np
import pandas as pd

from uplift_modeling.models.t_learner import predict_t_learner_scores
from uplift_modeling.models.x_learner import predict_x_learner_scores


RESPONSE_MODEL_KIND = "response"
T_LEARNER_MODEL_KIND = "t_learner"
X_LEARNER_MODEL_KIND = "x_learner"
SUPPORTED_MODEL_KINDS = (
    RESPONSE_MODEL_KIND,
    T_LEARNER_MODEL_KIND,
    X_LEARNER_MODEL_KIND,
)

ScoreBatch = Callable[[pd.DataFrame], np.ndarray]


def build_policy_score_batch(
    policy: str,
    model_artifact: Mapping[str, Any],
) -> ScoreBatch:
    """Load one saved policy model and return its batch scoring function."""
    model_kind = validate_model_artifact_identity(
        policy=policy,
        model_artifact=model_artifact,
    )
    if model_kind == RESPONSE_MODEL_KIND:
        return _build_response_score_batch(policy, model_artifact)
    if model_kind == T_LEARNER_MODEL_KIND:
        return _build_t_learner_score_batch(policy, model_artifact)

    return _build_x_learner_score_batch(policy, model_artifact)


def validate_model_artifact_identity(
    policy: str,
    model_artifact: Mapping[str, Any],
) -> str:
    """Validate model provenance identity and return its supported model kind."""
    artifact_type = model_artifact.get("artifact_type")
    if artifact_type != "model_provenance":
        raise ValueError(
            "Model artifact metadata for policy "
            f"'{policy}' has missing or unsupported artifact_type "
            f"{artifact_type!r}."
        )
    policy_name = _require_string(model_artifact, "policy_name", policy)
    if policy_name != policy:
        raise ValueError(
            "Model artifact metadata policy_name does not match selected "
            f"policy '{policy}'. Received: '{policy_name}'."
        )
    _require_string(model_artifact, "mlflow_run_id", policy)
    model_kind = _require_string(model_artifact, "model_kind", policy)
    if model_kind not in SUPPORTED_MODEL_KINDS:
        supported_text = ", ".join(SUPPORTED_MODEL_KINDS)
        raise ValueError(
            f"Unsupported model artifact kind '{model_kind}' for policy "
            f"'{policy}'. Supported kinds: {supported_text}."
        )

    if model_kind == RESPONSE_MODEL_KIND:
        _require_model_uri(model_artifact, "model_uri", policy)
    if model_kind == T_LEARNER_MODEL_KIND:
        _require_model_uri(model_artifact, "treatment_model_uri", policy)
        _require_model_uri(model_artifact, "control_model_uri", policy)
    if model_kind == X_LEARNER_MODEL_KIND:
        _require_model_uri(
            model_artifact,
            "treatment_effect_model_uri",
            policy,
        )
        _require_model_uri(
            model_artifact,
            "control_effect_model_uri",
            policy,
        )
        constant_treatment_rate_weight = _require_float(
            model_artifact,
            "constant_treatment_rate_weight",
            policy,
        )
        if not 0.0 <= constant_treatment_rate_weight <= 1.0:
            raise ValueError(
                "Model artifact metadata for policy "
                f"'{policy}' must contain "
                "'constant_treatment_rate_weight' between 0 and 1. "
                f"Received: {constant_treatment_rate_weight}."
            )

    return model_kind


def _build_response_score_batch(
    policy: str,
    model_artifact: Mapping[str, Any],
) -> ScoreBatch:
    model_uri = _require_model_uri(model_artifact, "model_uri", policy)
    model = mlflow.lightgbm.load_model(model_uri)

    def score_batch(features: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError(
                f"Response model for policy '{policy}' returned an invalid "
                f"predict_proba shape: {probabilities.shape}."
            )
        return _validate_scores(probabilities[:, 1], len(features), policy)

    return score_batch


def _build_t_learner_score_batch(
    policy: str,
    model_artifact: Mapping[str, Any],
) -> ScoreBatch:
    treatment_model_uri = _require_model_uri(
        model_artifact,
        "treatment_model_uri",
        policy,
    )
    control_model_uri = _require_model_uri(
        model_artifact,
        "control_model_uri",
        policy,
    )

    treatment_model = mlflow.lightgbm.load_model(treatment_model_uri)
    control_model = mlflow.lightgbm.load_model(control_model_uri)

    def score_batch(features: pd.DataFrame) -> np.ndarray:
        scores = predict_t_learner_scores(
            treatment_model=treatment_model,
            control_model=control_model,
            features=features,
        )
        return _validate_scores(scores, len(features), policy)

    return score_batch


def _build_x_learner_score_batch(
    policy: str,
    model_artifact: Mapping[str, Any],
) -> ScoreBatch:
    treatment_effect_model_uri = _require_model_uri(
        model_artifact,
        "treatment_effect_model_uri",
        policy,
    )
    control_effect_model_uri = _require_model_uri(
        model_artifact,
        "control_effect_model_uri",
        policy,
    )
    constant_treatment_rate_weight = _require_float(
        model_artifact,
        "constant_treatment_rate_weight",
        policy,
    )

    if not 0.0 <= constant_treatment_rate_weight <= 1.0:
        raise ValueError(
            "Model artifact metadata for policy "
            f"'{policy}' must contain "
            "'constant_treatment_rate_weight' between 0 and 1. "
            f"Received: {constant_treatment_rate_weight}."
        )

    treatment_effect_model = mlflow.lightgbm.load_model(
        treatment_effect_model_uri
    )
    control_effect_model = mlflow.lightgbm.load_model(
        control_effect_model_uri
    )

    def score_batch(features: pd.DataFrame) -> np.ndarray:
        scores = predict_x_learner_scores(
            treatment_effect_model=treatment_effect_model,
            control_effect_model=control_effect_model,
            constant_treatment_rate_weight=constant_treatment_rate_weight,
            features=features,
        )
        return _validate_scores(scores, len(features), policy)

    return score_batch


def _require_string(
    model_artifact: Mapping[str, Any],
    key: str,
    policy: str,
) -> str:
    value = model_artifact.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Model artifact metadata for policy '{policy}' must contain a "
            f"non-empty '{key}' string."
        )
    return value


def _require_model_uri(
    model_artifact: Mapping[str, Any],
    key: str,
    policy: str,
) -> str:
    """Return a model URI belonging to the declared MLflow run."""
    model_uri = _require_string(
        model_artifact,
        key,
        policy,
    )
    mlflow_run_id = _require_string(
        model_artifact,
        "mlflow_run_id",
        policy,
    )
    expected_prefix = f"runs:/{mlflow_run_id}/"

    if not model_uri.startswith(expected_prefix):
        raise ValueError(
            f"Model artifact metadata for policy '{policy}' has "
            f"'{key}' that must belong to mlflow_run_id "
            f"'{mlflow_run_id}'. Received: '{model_uri}'."
        )

    return model_uri


def _require_float(
    model_artifact: Mapping[str, Any],
    key: str,
    policy: str,
) -> float:
    value = model_artifact.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"Model artifact metadata for policy '{policy}' must contain a "
            f"numeric '{key}'."
        )

    number = float(value)
    if not np.isfinite(number):
        raise ValueError(
            f"Model artifact metadata for policy '{policy}' must contain a "
            f"finite '{key}'."
        )
    return number


def _validate_scores(scores: Any, row_count: int, policy: str) -> np.ndarray:
    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim != 1:
        raise ValueError(
            f"Scores for policy '{policy}' must be one-dimensional. "
            f"Received shape: {score_array.shape}."
        )
    if len(score_array) != row_count:
        raise ValueError(
            f"Scores for policy '{policy}' must match the input row count. "
            f"Received {len(score_array)} scores for {row_count} rows."
        )
    if not np.isfinite(score_array).all():
        raise ValueError(f"Scores for policy '{policy}' must be finite.")

    return score_array
