"""Tests for locked test-set reporting."""

import json
from pathlib import Path

import pandas as pd
import pytest

from uplift_modeling.evaluation.locked_test import (
    get_champion_policy,
    save_locked_test_evaluation,
)
from uplift_modeling.evaluation.topk_policy import TOPK_BUDGET_FRACTIONS


def _prediction_frame(
    model_name: str,
    splits: list[str] | None = None,
) -> pd.DataFrame:
    """Return a small prediction frame with both treatment arms."""
    split_values = splits or ["validation"] * 4 + ["test"] * 8
    row_count = len(split_values)
    return pd.DataFrame(
        {
            "row_id": list(range(row_count)),
            "treatment": ([1, 0] * (row_count // 2 + 1))[:row_count],
            "outcome": ([1, 0, 0, 1, 1, 0] * (row_count // 6 + 1))[
                :row_count
            ],
            "split": split_values,
            "score": list(
                reversed([index / row_count for index in range(row_count)])
            ),
            "model_name": [model_name] * row_count,
        }
    )


def _write_predictions(
    prediction_dir: Path,
    policy: str,
    splits: list[str] | None = None,
) -> Path:
    """Write one run-numbered prediction artifact."""
    prediction_path = (
        prediction_dir / f"criteo_visit_{policy}_run01_predictions.parquet"
    )
    _prediction_frame(policy, splits=splits).to_parquet(
        prediction_path,
        index=False,
    )
    return prediction_path


def _manifest_paths(*prediction_paths: Path) -> dict[str, Path]:
    """Build a policy-to-path mapping from test prediction file names."""
    paths: dict[str, Path] = {}
    for prediction_path in prediction_paths:
        policy = prediction_path.name.removeprefix(
            "criteo_visit_"
        ).removesuffix("_run01_predictions.parquet")
        paths[policy] = prediction_path

    return paths


def _write_selection_artifact(tmp_path: Path, champion_policy: str) -> Path:
    """Write a minimal Selection Gate artifact."""
    selection_path = tmp_path / "criteo_visit_model_selection_gate_run01.json"
    selection_path.write_text(
        json.dumps(
            {
                "artifact_type": "model_selection_gate",
                "dataset_name": "criteo",
                "outcome": "visit",
                "champion_policy": champion_policy,
            }
        ),
        encoding="utf-8",
    )
    return selection_path


def test_champion_is_loaded_from_selection_gate_artifact(tmp_path) -> None:
    """Locked-test evaluation uses the champion fixed by Selection Gate."""
    selection_path = _write_selection_artifact(tmp_path, "t_learner_lgbm")
    payload = json.loads(selection_path.read_text(encoding="utf-8"))

    assert get_champion_policy(payload) == "t_learner_lgbm"


def test_locked_test_evaluates_only_champion_and_baseline(tmp_path) -> None:
    """Locked-test rows include the champion and baseline, not extra policies."""
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    prediction_dir.mkdir()
    selection_path = _write_selection_artifact(tmp_path, "t_learner_lgbm")
    baseline_path = _write_predictions(prediction_dir, "treated_response_lgbm")
    champion_path = _write_predictions(prediction_dir, "t_learner_lgbm")
    extra_path = _write_predictions(prediction_dir, "x_learner_lgbm")

    output_path, payload = save_locked_test_evaluation(
        manifest_prediction_paths=_manifest_paths(
            baseline_path,
            champion_path,
            extra_path,
        ),
        metric_dir=metric_dir,
        dataset_name="criteo",
        outcome="visit",
        selection_artifact_path=selection_path,
    )

    row_policies = {row["policy"] for row in payload["locked_test_rows"]}
    assert output_path.exists()
    assert output_path.name == "criteo_visit_locked_test_evaluation_run01.json"
    assert payload["champion_policy"] == "t_learner_lgbm"
    assert payload["baseline_policy"] == "treated_response_lgbm"
    assert set(payload["prediction_artifacts"]) == {
        "t_learner_lgbm",
        "treated_response_lgbm",
    }
    assert row_policies == {"t_learner_lgbm", "treated_response_lgbm"}
    assert len(payload["locked_test_rows"]) == 2 * len(TOPK_BUDGET_FRACTIONS)
    assert {row["split"] for row in payload["locked_test_rows"]} == {"test"}


def test_locked_test_aligns_reordered_prediction_rows_by_row_id(tmp_path) -> None:
    """Locked-test evaluation tolerates reordered champion artifacts."""
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    prediction_dir.mkdir()
    selection_path = _write_selection_artifact(tmp_path, "t_learner_lgbm")
    baseline_path = _write_predictions(prediction_dir, "treated_response_lgbm")
    champion_path = (
        prediction_dir / "criteo_visit_t_learner_lgbm_run01_predictions.parquet"
    )
    _prediction_frame("t_learner_lgbm").sort_values(
        "row_id",
        ascending=False,
    ).to_parquet(champion_path, index=False)

    _, payload = save_locked_test_evaluation(
        manifest_prediction_paths=_manifest_paths(
            baseline_path,
            champion_path,
        ),
        metric_dir=metric_dir,
        dataset_name="criteo",
        outcome="visit",
        selection_artifact_path=selection_path,
    )

    assert {row["split"] for row in payload["locked_test_rows"]} == {"test"}
    assert {row["policy"] for row in payload["locked_test_rows"]} == {
        "t_learner_lgbm",
        "treated_response_lgbm",
    }


def test_locked_test_fails_if_champion_test_predictions_are_missing(tmp_path) -> None:
    """A champion artifact without test rows raises a clear error."""
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    prediction_dir.mkdir()
    selection_path = _write_selection_artifact(tmp_path, "t_learner_lgbm")
    baseline_path = _write_predictions(prediction_dir, "treated_response_lgbm")
    champion_path = _write_predictions(
        prediction_dir,
        "t_learner_lgbm",
        splits=["validation"] * 12,
    )

    with pytest.raises(ValueError, match="t_learner_lgbm test predictions"):
        save_locked_test_evaluation(
            manifest_prediction_paths=_manifest_paths(
                baseline_path,
                champion_path,
            ),
            metric_dir=metric_dir,
            dataset_name="criteo",
            outcome="visit",
            selection_artifact_path=selection_path,
        )


def test_locked_test_fails_if_baseline_test_predictions_are_missing(tmp_path) -> None:
    """A baseline artifact without test rows raises a clear error."""
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    prediction_dir.mkdir()
    selection_path = _write_selection_artifact(tmp_path, "t_learner_lgbm")
    champion_path = _write_predictions(prediction_dir, "t_learner_lgbm")
    baseline_path = _write_predictions(
        prediction_dir,
        "treated_response_lgbm",
        splits=["validation"] * 12,
    )

    with pytest.raises(
        ValueError,
        match="treated_response_lgbm test predictions",
    ):
        save_locked_test_evaluation(
            manifest_prediction_paths=_manifest_paths(
                champion_path,
                baseline_path,
            ),
            metric_dir=metric_dir,
            dataset_name="criteo",
            outcome="visit",
            selection_artifact_path=selection_path,
        )


def test_locked_test_fails_if_champion_prediction_artifact_is_missing(
    tmp_path,
) -> None:
    """A missing champion prediction artifact raises a clear ValueError."""
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    prediction_dir.mkdir()
    selection_path = _write_selection_artifact(tmp_path, "t_learner_lgbm")
    baseline_path = _write_predictions(prediction_dir, "treated_response_lgbm")

    with pytest.raises(ValueError, match="Champion prediction artifact"):
        save_locked_test_evaluation(
            manifest_prediction_paths=_manifest_paths(baseline_path),
            metric_dir=metric_dir,
            dataset_name="criteo",
            outcome="visit",
            selection_artifact_path=selection_path,
        )


def test_locked_test_fails_if_baseline_prediction_artifact_is_missing(
    tmp_path,
) -> None:
    """A missing baseline prediction artifact raises a clear ValueError."""
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    prediction_dir.mkdir()
    selection_path = _write_selection_artifact(tmp_path, "t_learner_lgbm")
    champion_path = _write_predictions(prediction_dir, "t_learner_lgbm")

    with pytest.raises(ValueError, match="Baseline prediction artifact"):
        save_locked_test_evaluation(
            manifest_prediction_paths=_manifest_paths(champion_path),
            metric_dir=metric_dir,
            dataset_name="criteo",
            outcome="visit",
            selection_artifact_path=selection_path,
        )


def test_locked_test_ignores_newer_unlisted_prediction_artifact(tmp_path) -> None:
    """A newer champion file outside the manifest does not affect locked test."""
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    prediction_dir.mkdir()
    selection_path = _write_selection_artifact(tmp_path, "t_learner_lgbm")
    baseline_path = _write_predictions(prediction_dir, "treated_response_lgbm")
    champion_path = _write_predictions(prediction_dir, "t_learner_lgbm")
    newer_champion_path = (
        prediction_dir / "criteo_visit_t_learner_lgbm_run99_predictions.parquet"
    )
    newer_frame = _prediction_frame("t_learner_lgbm")
    newer_frame["score"] = 0.0
    newer_frame.to_parquet(newer_champion_path, index=False)

    _, payload = save_locked_test_evaluation(
        manifest_prediction_paths=_manifest_paths(
            baseline_path,
            champion_path,
        ),
        metric_dir=metric_dir,
        dataset_name="criteo",
        outcome="visit",
        selection_artifact_path=selection_path,
    )

    assert payload["prediction_artifacts"]["t_learner_lgbm"] == (
        champion_path.name
    )
