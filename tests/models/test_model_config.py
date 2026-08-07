"""Tests for shared model defaults and candidate overrides."""

import pytest

from uplift_modeling.models.config import (
    resolve_model_candidate,
    resolve_model_candidates,
)
from uplift_modeling.models.scoring import (
    RESPONSE_MODEL_KIND,
    T_LEARNER_MODEL_KIND,
    X_LEARNER_MODEL_KIND,
)


def make_config() -> dict:
    return {
        "models": {
            "model_defaults": {
                "learning_rate": 0.03,
                "num_leaves": 31,
                "random_state": 42,
            },
            "candidates": [
                {
                    "name": "treated_response_lgbm",
                    "kind": RESPONSE_MODEL_KIND,
                    "params": {},
                },
                {
                    "name": "t_learner_lgbm",
                    "kind": T_LEARNER_MODEL_KIND,
                    "params": {
                        "num_leaves": 63,
                    },
                },
                {
                    "name": "x_learner_lgbm",
                    "kind": X_LEARNER_MODEL_KIND,
                    "params": {},
                },
            ],
        }
    }


def test_resolve_model_candidates_merges_defaults() -> None:
    candidates = resolve_model_candidates(make_config())

    assert len(candidates) == 3

    response = candidates[0]
    assert response.params == {
        "learning_rate": 0.03,
        "num_leaves": 31,
        "random_state": 42,
    }


def test_candidate_params_override_defaults() -> None:
    candidate = resolve_model_candidate(
        make_config(),
        T_LEARNER_MODEL_KIND,
    )

    assert candidate.name == "t_learner_lgbm"
    assert candidate.kind == T_LEARNER_MODEL_KIND
    assert candidate.params["learning_rate"] == 0.03
    assert candidate.params["num_leaves"] == 63
    assert candidate.params["random_state"] == 42


def test_resolve_model_candidate_rejects_missing_kind() -> None:
    with pytest.raises(ValueError, match="No model candidate"):
        resolve_model_candidate(
            make_config(),
            "unknown_kind",
        )


def test_resolve_model_candidates_rejects_duplicate_names() -> None:
    config = make_config()
    config["models"]["candidates"][1]["name"] = "treated_response_lgbm"

    with pytest.raises(ValueError, match="Duplicate candidate"):
        resolve_model_candidates(config)


def test_resolve_model_candidates_rejects_unknown_kind() -> None:
    config = make_config()
    config["models"]["candidates"][0]["kind"] = "unsupported"

    with pytest.raises(ValueError, match="Unsupported model kind"):
        resolve_model_candidates(config)