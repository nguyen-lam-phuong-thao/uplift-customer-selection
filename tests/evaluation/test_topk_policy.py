"""Tests for Top-K policy artifact evaluation."""

import pandas as pd
import pytest

from uplift_modeling.evaluation.uplift_metrics import calculate_policy_value
from uplift_modeling.evaluation.topk_policy import (
    build_random_policy_frame,
    calculate_topk_policy_metrics,
    calculate_topk_policy_rows,
    load_policy_prediction_frames,
    resolve_expected_policy_paths,
    save_topk_policy_evaluation,
)


def _prediction_frame() -> pd.DataFrame:
    """Return a tiny prediction artifact frame."""
    return pd.DataFrame(
        {
            "row_id": list(range(4)),
            "treatment": [1, 0, 1, 0],
            "outcome": [1, 0, 0, 1],
            "split": ["validation"] * 4,
            "score": [0.9, 0.8, 0.7, 0.6],
            "model_name": ["t_learner_lgbm"] * 4,
        }
    )


def test_random_policy_frame_is_deterministic() -> None:
    """Random targeting scores are deterministic for a fixed seed."""
    frame = _prediction_frame().sort_values("row_id", ascending=False)

    first = build_random_policy_frame(frame, random_seed=42)
    second = build_random_policy_frame(frame, random_seed=42)

    assert first["score"].tolist() == second["score"].tolist()
    assert first["row_id"].tolist() == list(range(4))
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


def test_topk_evaluation_runs_with_one_manifest_policy(tmp_path) -> None:
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
        manifest_prediction_paths={"t_learner_lgbm": prediction_path},
        metric_dir=metric_dir,
        dataset_name="criteo",
        outcome="visit",
        random_seed=42,
    )

    assert json_path.exists()
    assert not list(metric_dir.glob("*.csv"))
    assert payload["evaluated_splits"] == ["validation"]
    assert {row["split"] for row in payload["table_rows"]} == {"validation"}
    assert "t_learner_lgbm" in payload["evaluated_policy_names"]
    assert "random_targeting" in payload["evaluated_policy_names"]
    assert payload["warnings"] == []


def test_topk_policy_evaluation_rejects_test_split_request(tmp_path) -> None:
    """Standard Top-K model comparison cannot evaluate test rows."""
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    prediction_dir.mkdir()
    prediction_path = (
        prediction_dir
        / "criteo_visit_t_learner_lgbm_run01_predictions.parquet"
    )
    _prediction_frame().to_parquet(prediction_path, index=False)

    with pytest.raises(ValueError, match="test split is reserved"):
        save_topk_policy_evaluation(
            manifest_prediction_paths={"t_learner_lgbm": prediction_path},
            metric_dir=metric_dir,
            dataset_name="criteo",
            outcome="visit",
            random_seed=42,
            evaluation_splits=("test",),
        )


def test_topk_policy_evaluation_ignores_newer_unlisted_prediction(tmp_path) -> None:
    """A newer prediction file outside the manifest does not affect Top-K."""
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    prediction_dir.mkdir()
    manifest_path = (
        prediction_dir
        / "criteo_visit_t_learner_lgbm_run01_predictions.parquet"
    )
    newer_path = (
        prediction_dir
        / "criteo_visit_t_learner_lgbm_run99_predictions.parquet"
    )
    _prediction_frame().to_parquet(manifest_path, index=False)
    newer_frame = _prediction_frame()
    newer_frame["score"] = 0.0
    newer_frame.to_parquet(newer_path, index=False)

    _, payload = save_topk_policy_evaluation(
        manifest_prediction_paths={"t_learner_lgbm": manifest_path},
        metric_dir=metric_dir,
        dataset_name="criteo",
        outcome="visit",
        random_seed=42,
    )

    assert payload["prediction_artifacts"] == {
        "t_learner_lgbm": manifest_path.name,
    }


def test_topk_policy_uses_exact_manifest_policy_mapping(tmp_path) -> None:
    """Evaluation accepts the exact policies listed in the manifest."""
    custom_model_path = tmp_path / "custom_model.parquet"
    t_learner_path = tmp_path / "t_learner.parquet"

    policy_paths, warnings = resolve_expected_policy_paths(
        manifest_prediction_paths={
            "custom_uplift_model": custom_model_path,
            "t_learner_lgbm": t_learner_path,
        },
        outcome="visit",
    )

    assert policy_paths == {
        "custom_uplift_model": custom_model_path,
        "t_learner_lgbm": t_learner_path,
    }
    assert warnings == []


def test_calculate_topk_policy_rows_rejects_test_split() -> None:
    """Direct standard Top-K row calculation cannot evaluate test rows."""
    frame = _prediction_frame()
    frame.loc[0, "split"] = "test"

    with pytest.raises(ValueError, match="test split is reserved"):
        calculate_topk_policy_rows({"t_learner_lgbm": frame})

def test_topk_loader_checks_split_before_full_read(tmp_path, monkeypatch) -> None:
    """Standard Top-K checks split before loading full prediction rows."""
    prediction_path = tmp_path / "predictions.parquet"
    frame = _prediction_frame()
    frame.loc[0, "split"] = "test"
    frame.to_parquet(prediction_path, index=False)

    import uplift_modeling.evaluation.topk_policy as topk_policy

    original_read_parquet = topk_policy.pd.read_parquet
    read_columns = []

    def tracking_read_parquet(*args, **kwargs):
        read_columns.append(kwargs.get("columns"))
        if kwargs.get("columns") is None:
            raise AssertionError("Full artifact should not be loaded before rejection.")
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(topk_policy.pd, "read_parquet", tracking_read_parquet)

    with pytest.raises(ValueError, match="test split is reserved"):
        load_policy_prediction_frames({"t_learner_lgbm": prediction_path})

    assert read_columns == [["split"]]