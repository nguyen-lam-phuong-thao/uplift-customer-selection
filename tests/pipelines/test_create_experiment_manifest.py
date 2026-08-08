"""Focused tests for automatic experiment manifest creation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from uplift_modeling.artifacts.model_provenance import get_model_provenance_path
from uplift_modeling.pipelines.create_experiment_manifest import (
    create_experiment_manifest,
)


def _write_dataset_config(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: synthetic",
                f"  prepared_path: {(tmp_path / 'prepared.parquet').as_posix()}",
                "schema:",
                "  treatment_column: treatment",
                "  split_column: split",
                "  feature_columns:",
                "    - f0",
                "  outcome_columns:",
                "    - visit",
                "split:",
                "  assign_if_missing: false",
                "  train_size: 0.6",
                "  validation_size: 0.2",
                "  test_size: 0.2",
                "  random_state: 42",
                "outputs:",
                "  processed_paths:",
                f"    visit: {(tmp_path / 'decision_visit.parquet').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_modeling_config(tmp_path: Path) -> Path:
    path = tmp_path / "modeling.yaml"
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    path.write_text(
        "\n".join(
            [
                "project:",
                "  experiment_name: uplift-test",
                "outputs:",
                f"  prediction_dir: {prediction_dir.as_posix()}",
                f"  metric_dir: {metric_dir.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_prediction(
    path: Path,
    *,
    split: str = "validation",
    write_provenance: bool = True,
) -> None:
    policy = "treated_response_lgbm"
    pd.DataFrame(
        {
            "row_id": [1],
            "treatment": [1],
            "outcome": [1],
            "split": [split],
            "score": [0.8],
            "model_name": [policy],
        }
    ).to_parquet(path, index=False)

    if not write_provenance:
        return

    provenance = {
        "artifact_type": "model_provenance",
        "dataset_name": "synthetic",
        "outcome": "visit",
        "policy_name": policy,
        "prediction_artifact": path.name,
        "model_kind": "response",
        "mlflow_run_id": "run-response",
        "model_uri": "runs:/run-response/model",
    }
    get_model_provenance_path(path).write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_config_path = _write_dataset_config(tmp_path)
    modeling_config_path = _write_modeling_config(tmp_path)
    prediction_path = (
        tmp_path
        / "predictions"
        / "synthetic_visit_treated_response_lgbm_run01_predictions.parquet"
    )
    return dataset_config_path, modeling_config_path, prediction_path


def test_create_manifest_records_new_contract(tmp_path: Path) -> None:
    dataset_config_path, modeling_config_path, prediction_path = _setup(tmp_path)
    _write_prediction(prediction_path)

    output_path = create_experiment_manifest(
        dataset_config_path=dataset_config_path,
        modeling_config_path=modeling_config_path,
        outcome="visit",
        experiment_id="exp-001",
        prediction_artifacts={"treated_response_lgbm": prediction_path},
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["dataset_name"] == "synthetic"
    assert payload["outcome"] == "visit"
    assert payload["experiment_id"] == "exp-001"
    assert payload["dataset_config_path"] == dataset_config_path.resolve().as_posix()
    assert payload["modeling_config_path"] == modeling_config_path.resolve().as_posix()
    assert "config_path" not in payload
    assert payload["prediction_artifacts"] == {
        "treated_response_lgbm": prediction_path.resolve().as_posix()
    }


def test_create_manifest_does_not_overwrite(tmp_path: Path) -> None:
    dataset_config_path, modeling_config_path, prediction_path = _setup(tmp_path)
    _write_prediction(prediction_path)
    kwargs = {
        "dataset_config_path": dataset_config_path,
        "modeling_config_path": modeling_config_path,
        "outcome": "visit",
        "experiment_id": "exp-001",
        "prediction_artifacts": {"treated_response_lgbm": prediction_path},
    }

    create_experiment_manifest(**kwargs)

    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        create_experiment_manifest(**kwargs)


def test_create_manifest_requires_provenance(tmp_path: Path) -> None:
    dataset_config_path, modeling_config_path, prediction_path = _setup(tmp_path)
    _write_prediction(prediction_path, write_provenance=False)

    with pytest.raises(FileNotFoundError, match="Missing model provenance"):
        create_experiment_manifest(
            dataset_config_path=dataset_config_path,
            modeling_config_path=modeling_config_path,
            outcome="visit",
            experiment_id="exp-001",
            prediction_artifacts={"treated_response_lgbm": prediction_path},
        )


def test_create_manifest_accepts_validation_only(tmp_path: Path) -> None:
    dataset_config_path, modeling_config_path, prediction_path = _setup(tmp_path)
    _write_prediction(prediction_path, split="test")

    with pytest.raises(ValueError, match="only validation rows"):
        create_experiment_manifest(
            dataset_config_path=dataset_config_path,
            modeling_config_path=modeling_config_path,
            outcome="visit",
            experiment_id="exp-001",
            prediction_artifacts={"treated_response_lgbm": prediction_path},
        )