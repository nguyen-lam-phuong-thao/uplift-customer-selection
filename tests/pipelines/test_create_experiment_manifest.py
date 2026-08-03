"""Tests for automatic experiment manifest creation."""

import json
from pathlib import Path
import pytest

import pandas as pd

from uplift_modeling.pipelines.create_experiment_manifest import (
    create_experiment_manifest,
)
from uplift_modeling.artifacts.model_provenance import (
    get_model_provenance_path,
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


def _write_prediction_artifact(
    prediction_path: Path,
    policy_name: str,
    outcome: str = "conversion",
    split_name: str = "validation",
) -> None:
    """Write a prediction artifact and its model provenance sidecar."""
    pd.DataFrame(
        {
            "row_id": [1],
            "treatment": [1],
            "outcome": [1],
            "split": [split_name],
            "score": [0.8],
            "model_name": [policy_name],
        }
    ).to_parquet(prediction_path, index=False)

    provenance = {
        "artifact_type": "model_provenance",
        "dataset_name": "criteo",
        "outcome": outcome,
        "policy_name": policy_name,
        "prediction_artifact": prediction_path.name,
        "model_kind": "test_model",
        "mlflow_run_id": f"run-{policy_name}",
        "model_uri": f"runs:/run-{policy_name}/model",
    }
    get_model_provenance_path(prediction_path).write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )


def test_create_experiment_manifest_writes_exact_prediction_mapping(
    tmp_path,
) -> None:
    """The manifest uses exactly the supplied prediction artifacts."""
    config_path = _write_config(tmp_path)
    prediction_dir = tmp_path / "predictions"

    treated_response = (
        prediction_dir
        / "criteo_conversion_treated_response_lgbm_run01_predictions.parquet"
    )
    t_learner = (
        prediction_dir
        / "criteo_conversion_t_learner_lgbm_run01_predictions.parquet"
    )
    selected_x_learner = (
        prediction_dir
        / "criteo_conversion_x_learner_lgbm_run01_predictions.parquet"
    )
    newer_x_learner = (
        prediction_dir
        / "criteo_conversion_x_learner_lgbm_run02_predictions.parquet"
    )

    _write_prediction_artifact(
        treated_response,
        policy_name="treated_response_lgbm",
    )
    _write_prediction_artifact(
        t_learner,
        policy_name="t_learner_lgbm",
    )
    _write_prediction_artifact(
        selected_x_learner,
        policy_name="x_learner_lgbm",
    )
    _write_prediction_artifact(
        newer_x_learner,
        policy_name="x_learner_lgbm",
    )

    prediction_artifacts = {
        "treated_response_lgbm": treated_response,
        "t_learner_lgbm": t_learner,
        "x_learner_lgbm": selected_x_learner,
    }

    output_path = create_experiment_manifest(
        config_path=config_path,
        outcome="conversion",
        experiment_id="exp-001",
        prediction_artifacts=prediction_artifacts,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        tmp_path
        / "metrics"
        / "criteo_conversion_exp-001_experiment_manifest.json"
    )
    assert payload["artifact_type"] == "experiment_manifest"
    assert payload["experiment_id"] == "exp-001"
    assert payload["dataset_name"] == "criteo"
    assert payload["outcome"] == "conversion"
    assert payload["prediction_artifacts"] == {
        policy_name: path.resolve().as_posix()
        for policy_name, path in sorted(prediction_artifacts.items())
    }


def test_create_experiment_manifest_does_not_overwrite_existing_manifest(
    tmp_path,
) -> None:
    """An immutable experiment manifest cannot be overwritten."""
    config_path = _write_config(tmp_path)
    prediction_path = (
        tmp_path
        / "predictions"
        / "criteo_conversion_t_learner_lgbm_run01_predictions.parquet"
    )
    _write_prediction_artifact(
        prediction_path,
        policy_name="t_learner_lgbm",
    )
    prediction_artifacts = {
        "t_learner_lgbm": prediction_path,
    }
    create_experiment_manifest(
        config_path=config_path,
        outcome="conversion",
        experiment_id="exp-001",
        prediction_artifacts=prediction_artifacts,
    )

    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        create_experiment_manifest(
            config_path=config_path,
            outcome="conversion",
            experiment_id="exp-001",
            prediction_artifacts=prediction_artifacts,
        )


def test_create_experiment_manifest_requires_model_provenance(
    tmp_path,
) -> None:
    """Every selected prediction artifact must have model provenance."""
    config_path = _write_config(tmp_path)
    prediction_path = (
        tmp_path
        / "predictions"
        / "criteo_conversion_t_learner_lgbm_run01_predictions.parquet"
    )
    pd.DataFrame(
        {
            "row_id": [1],
            "treatment": [1],
            "outcome": [1],
            "split": ["validation"],
            "score": [0.8],
            "model_name": ["t_learner_lgbm"],
        }
    ).to_parquet(prediction_path, index=False)

    with pytest.raises(FileNotFoundError, match="Missing model provenance"):
        create_experiment_manifest(
            config_path=config_path,
            outcome="conversion",
            experiment_id="exp-001",
            prediction_artifacts={
                "t_learner_lgbm": prediction_path,
            },
        )


def test_create_experiment_manifest_rejects_non_validation_rows(
    tmp_path,
) -> None:
    """Manifest inputs must contain validation rows only."""
    config_path = _write_config(tmp_path)
    prediction_path = (
        tmp_path
        / "predictions"
        / "criteo_conversion_t_learner_lgbm_run01_predictions.parquet"
    )

    _write_prediction_artifact(
        prediction_path,
        policy_name="t_learner_lgbm",
        split_name="test",
    )

    with pytest.raises(ValueError, match="only validation rows"):
        create_experiment_manifest(
            config_path=config_path,
            outcome="conversion",
            experiment_id="exp-001",
            prediction_artifacts={
                "t_learner_lgbm": prediction_path,
            },
        )