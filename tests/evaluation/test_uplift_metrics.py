"""Tests for reusable uplift metrics."""

import pandas as pd
import pytest

from uplift_modeling.evaluation.uplift_metrics import (
    build_uplift_curve,
    calculate_policy_value,
    calculate_selected_incremental_outcome,
    calculate_uplift_metrics,
    validate_prediction_frame,
)


def _prediction_frame() -> pd.DataFrame:
    """Return a small prediction frame with treated and control rows."""
    return pd.DataFrame(
        {
            "treatment": [1, 0, 1, 0],
            "outcome": [1, 0, 0, 0],
            "split": ["validation"] * 4,
            "score": [0.9, 0.8, 0.7, 0.6],
            "model_name": ["response_model_lgbm"] * 4,
        }
    )


def test_build_uplift_curve_uses_score_as_ranking() -> None:
    """The uplift curve is cumulative and sorted by descending score."""
    curve = build_uplift_curve(_prediction_frame())

    assert curve["rank"].tolist() == [1, 2, 3, 4]
    assert curve["population_fraction"].iloc[-1] == pytest.approx(1.0)
    assert curve["cumulative_incremental_outcome"].iloc[-1] == pytest.approx(
        1.0
    )


def test_build_uplift_curve_supports_num_points() -> None:
    """The uplift curve can be sampled to a configured number of points."""
    frame = pd.DataFrame(
        {
            "treatment": [1, 0] * 10,
            "outcome": [1, 0] * 10,
            "split": ["validation"] * 20,
            "score": list(range(20, 0, -1)),
            "model_name": ["response_model_lgbm"] * 20,
        }
    )
    curve = build_uplift_curve(frame, num_points=5)

    assert len(curve) == 5
    assert curve["rank"].iloc[-1] == 20
    assert curve["population_fraction"].iloc[-1] == pytest.approx(1.0)


def test_build_uplift_curve_defaults_to_at_most_num_points() -> None:
    """Default curve output is sampled instead of one row per customer."""
    frame = pd.DataFrame(
        {
            "treatment": [1, 0] * 75,
            "outcome": [1, 0] * 75,
            "split": ["validation"] * 150,
            "score": list(range(150, 0, -1)),
            "model_name": ["response_model_lgbm"] * 150,
        }
    )
    curve = build_uplift_curve(frame)

    assert len(curve) <= 100
    assert curve["rank"].iloc[-1] == 150


def test_calculate_uplift_metrics_returns_business_metrics() -> None:
    """Uplift metrics include Qini, AUUC, incremental outcome, and value."""
    metrics = calculate_uplift_metrics(
        _prediction_frame(),
        top_fraction=0.5,
    )

    assert metrics["row_count"] == 4
    assert metrics["positive_rate"] == pytest.approx(0.25)
    assert metrics["treatment_rate"] == pytest.approx(0.5)
    assert metrics["cumulative_incremental_outcome"] == pytest.approx(1.0)
    assert metrics["policy_value"] == pytest.approx(0.5)
    assert "qini" in metrics
    assert "auuc" in metrics


def test_validate_prediction_frame_rejects_missing_contract_columns() -> None:
    """Prediction frames must follow the shared artifact contract."""
    with pytest.raises(ValueError, match="model_name"):
        validate_prediction_frame(_prediction_frame().drop(columns="model_name"))


def test_validate_prediction_frame_rejects_non_binary_outcome() -> None:
    """Outcome values must be binary for uplift evaluation."""
    frame = _prediction_frame()
    frame.loc[0, "outcome"] = 2

    with pytest.raises(ValueError, match="outcome values"):
        validate_prediction_frame(frame)


def test_validate_prediction_frame_rejects_only_treated_rows() -> None:
    """Evaluated frames must include both treatment and control rows."""
    frame = _prediction_frame()
    frame["treatment"] = 1

    with pytest.raises(ValueError, match="both treatment and control"):
        validate_prediction_frame(frame)


def test_validate_prediction_frame_rejects_only_control_rows() -> None:
    """Evaluated frames must include both treatment and control rows."""
    frame = _prediction_frame()
    frame["treatment"] = 0

    with pytest.raises(ValueError, match="both treatment and control"):
        validate_prediction_frame(frame)


def test_validate_prediction_frame_rejects_invalid_score() -> None:
    """Scores must be numeric, non-null, and finite."""
    frame = _prediction_frame()
    frame.loc[0, "score"] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        validate_prediction_frame(frame)


def test_calculate_policy_value_rejects_invalid_top_fraction() -> None:
    """Policy value requires a valid selected population fraction."""
    with pytest.raises(ValueError, match="top_fraction"):
        calculate_policy_value(_prediction_frame(), top_fraction=0)


def test_calculate_selected_incremental_outcome_uses_top_subset_formula() -> None:
    """Selected incremental outcome is rate difference times selected count."""
    value = calculate_selected_incremental_outcome(
        _prediction_frame(),
        top_fraction=0.5,
    )

    assert value == pytest.approx(2.0)
