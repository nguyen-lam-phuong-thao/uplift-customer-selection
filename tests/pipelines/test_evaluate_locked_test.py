"""Tests for locked-test pipeline manifest wiring."""

import json
from pathlib import Path

import pandas as pd

from uplift_modeling.pipelines.evaluate_locked_test import evaluate_locked_test


def _prediction_frame(model_name: str) -> pd.DataFrame:
    """Return a small prediction frame with test rows."""
    return pd.DataFrame(
        {
            "row_id": list(range(8)),
            "treatment": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome": [1, 0, 0, 1, 1, 0, 0, 0],
            "split": ["validation"] * 4 + ["test"] * 4,
            "score": [0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6],
            "model_name": [model_name] * 8,
        }
    )


def test_locked_test_pipeline_loads_manifest_prediction_mapping(tmp_path) -> None:
    """Locked-test pipeline resolves champion and baseline from a manifest."""
    config_path = tmp_path / "config.yaml"
    metric_dir = tmp_path / "metrics"
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_name: criteo",
                "outputs:",
                f"  prediction_dir: {prediction_dir}",
                f"  metric_dir: {metric_dir}",
                "selection:",
                "  baseline_policy: treated_response_lgbm",
            ]
        ),
        encoding="utf-8",
    )
    champion_path = (
        prediction_dir
        / "criteo_visit_t_learner_lgbm_run01_predictions.parquet"
    )
    baseline_path = (
        prediction_dir
        / "criteo_visit_treated_response_lgbm_run01_predictions.parquet"
    )
    _prediction_frame("t_learner_lgbm").to_parquet(champion_path, index=False)
    _prediction_frame("treated_response_lgbm").to_parquet(
        baseline_path,
        index=False,
    )
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "experiment_manifest",
                "experiment_id": "exp-001",
                "dataset_name": "criteo",
                "outcome": "visit",
                "config_path": str(config_path),
                "prediction_artifacts": {
                    "t_learner_lgbm": str(champion_path),
                    "treated_response_lgbm": str(baseline_path),
                },
            }
        ),
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "artifact_type": "model_selection_gate",
                "dataset_name": "criteo",
                "outcome": "visit",
                "champion_policy": "t_learner_lgbm",
            }
        ),
        encoding="utf-8",
    )

    output_path = evaluate_locked_test(
        config_path=config_path,
        manifest_path=manifest_path,
        selection_artifact_path=selection_path,
        outcome="visit",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.exists()
    assert payload["prediction_artifacts"] == {
        "t_learner_lgbm": champion_path.name,
        "treated_response_lgbm": baseline_path.name,
    }
