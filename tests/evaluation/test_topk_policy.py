"""Tests for Top-K policy artifact evaluation."""

import pandas as pd
import pytest

from uplift_modeling.evaluation.uplift_metrics import calculate_policy_value
from uplift_modeling.evaluation.topk_policy import (
    build_random_policy_frame,
    calculate_topk_policy_metrics,
    save_topk_policy_evaluation,
)


def _prediction_frame() -> pd.DataFrame:
    """Return a tiny prediction artifact frame."""
    return pd.DataFrame(
        {
            "treatment": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome": [1, 0, 0, 1, 1, 0, 0, 0],
            "split": ["validation"] * 4 + ["test"] * 4,
            "score": [0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6],
            "model_name": ["t_learner_lgbm"] * 8,
        }
    )


def test_random_policy_frame_is_deterministic() -> None:
    """Random targeting scores are deterministic for a fixed seed."""
    frame = _prediction_frame()

    first = build_random_policy_frame(frame, random_seed=42)
    second = build_random_policy_frame(frame, random_seed=42)

    assert first["score"].tolist() == second["score"].tolist()
    assert first["model_name"].unique().tolist() == ["random_targeting"]


def test_calculate_topk_policy_metrics_selects_budget_count() -> None:
    """Top-K policy metrics select ceil(budget * row count) rows."""
    metrics = calculate_topk_policy_metrics(
        _prediction_frame().loc[lambda frame: frame["split"] == "validation"],
        budget_fraction=0.5,
    )

    assert metrics["n_selected"] == 2
    assert metrics["treated_selected"] == 1
    assert metrics["control_selected"] == 1


def test_calculate_topk_policy_metrics_uses_incremental_formula() -> None:
    """Top-K incremental outcome uses selected-group uplift times count."""
    frame = _prediction_frame().loc[lambda data: data["split"] == "validation"]

    metrics = calculate_topk_policy_metrics(frame, budget_fraction=0.5)

    assert metrics["treated_outcome_rate"] == pytest.approx(1.0)
    assert metrics["control_outcome_rate"] == pytest.approx(0.0)
    assert metrics["uplift_rate"] == pytest.approx(1.0)
    assert metrics["incremental_outcome"] == pytest.approx(2.0)
    assert metrics["incremental_outcome_per_1k"] == pytest.approx(1000.0)


def test_calculate_topk_policy_metrics_computes_policy_value() -> None:
    """Top-K policy value is computed at the same budget fraction."""
    frame = _prediction_frame().loc[lambda data: data["split"] == "validation"]

    metrics = calculate_topk_policy_metrics(frame, budget_fraction=0.5)

    assert metrics["policy_value"] == pytest.approx(
        calculate_policy_value(frame, top_fraction=0.5)
    )


def test_calculate_topk_policy_metrics_uses_none_when_one_arm_selected() -> None:
    """Selected-group uplift is null when selected rows miss one arm."""
    metrics = calculate_topk_policy_metrics(
        _prediction_frame().loc[lambda frame: frame["split"] == "validation"],
        budget_fraction=0.25,
    )

    assert metrics["n_selected"] == 1
    assert metrics["control_selected"] == 0
    assert metrics["control_outcome_rate"] is None
    assert metrics["uplift_rate"] is None
    assert metrics["incremental_outcome"] is None
    assert metrics["incremental_outcome_per_1k"] is None


def test_missing_optional_policy_artifacts_do_not_crash(tmp_path) -> None:
    """Top-K evaluation runs when only one expected policy artifact exists."""
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    prediction_dir.mkdir()
    prediction_path = (
        prediction_dir
        / "criteo_visit_t_learner_lgbm_run01_predictions.parquet"
    )
    _prediction_frame().to_parquet(prediction_path, index=False)

    json_path, payload = save_topk_policy_evaluation(
        prediction_dir=prediction_dir,
        metric_dir=metric_dir,
        dataset_name="criteo",
        outcome="visit",
        random_seed=42,
    )

    assert json_path.exists()
    assert not list(metric_dir.glob("*.csv"))
    assert "t_learner_lgbm" in payload["evaluated_policy_names"]
    assert "random_targeting" in payload["evaluated_policy_names"]
    assert payload["warnings"]
