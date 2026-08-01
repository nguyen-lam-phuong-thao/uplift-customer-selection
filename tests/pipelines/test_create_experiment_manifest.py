"""Tests for automatic experiment manifest creation."""

import json
from pathlib import Path

import pandas as pd

from uplift_modeling.pipelines.create_experiment_manifest import (
    create_experiment_manifest,
)


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal manifest-creation config."""
    config_path = tmp_path / "config.yaml"
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    prediction_dir.mkdir()
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_name: criteo",
                "outputs:",
                f"  prediction_dir: {prediction_dir}",
                f"  metric_dir: {metric_dir}",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_prediction_artifact(prediction_path: Path) -> None:
    """Write an empty parquet prediction artifact with the current schema."""
    pd.DataFrame(
        {
            "row_id": [],
            "treatment": [],
            "outcome": [],
            "split": [],
            "score": [],
            "model_name": [],
        }
    ).to_parquet(prediction_path, index=False)


def test_create_experiment_manifest_writes_latest_prediction_mapping(
    tmp_path,
) -> None:
    """Pipeline writes a manifest from latest prediction artifacts."""
    config_path = _write_config(tmp_path)
    prediction_dir = tmp_path / "predictions"
    _write_prediction_artifact(
        prediction_dir
        / "criteo_conversion_response_lgbm_run01_predictions.parquet"
    )
    _write_prediction_artifact(
        prediction_dir
        / "criteo_conversion_t_learner_lgbm_run01_predictions.parquet"
    )
    latest_x_learner = (
        prediction_dir
        / "criteo_conversion_x_learner_lgbm_run02_predictions.parquet"
    )
    _write_prediction_artifact(latest_x_learner)
    _write_prediction_artifact(
        prediction_dir
        / "criteo_conversion_x_learner_lgbm_run01_predictions.parquet"
    )

    output_path = create_experiment_manifest(
        config_path=config_path,
        outcome="conversion",
        experiment_id="exp-001",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path == (
        tmp_path / "metrics" / "criteo_conversion_experiment_manifest.json"
    )
    assert payload["artifact_type"] == "experiment_manifest"
    assert payload["experiment_id"] == "exp-001"
    assert payload["dataset_name"] == "criteo"
    assert payload["outcome"] == "conversion"
    assert payload["prediction_artifacts"] == {
        "pooled_response_lgbm": (
            prediction_dir
            / "criteo_conversion_response_lgbm_run01_predictions.parquet"
        ).resolve().as_posix(),
        "t_learner_lgbm": (
            prediction_dir
            / "criteo_conversion_t_learner_lgbm_run01_predictions.parquet"
        ).resolve().as_posix(),
        "x_learner_lgbm": latest_x_learner.resolve().as_posix(),
    }
