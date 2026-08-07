"""Focused tests for experiment manifest identity and artifact resolution."""

import json
from pathlib import Path

import pandas as pd
import pytest

from uplift_modeling.artifacts.manifest import (
    build_experiment_manifest,
    build_experiment_manifest_model_artifacts,
    load_experiment_manifest,
    resolve_model_artifacts,
    resolve_prediction_paths,
    validate_model_artifacts_match_predictions,
)


def _write_prediction_artifact(
    prediction_path: Path,
    model_name: str = "t_learner_lgbm",
) -> None:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": [1],
            "treatment": [1],
            "outcome": [1],
            "split": ["validation"],
            "score": [0.8],
            "model_name": [model_name],
        }
    ).to_parquet(prediction_path, index=False)


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _base_payload(prediction_path: Path) -> dict:
    return {
        "artifact_type": "experiment_manifest",
        "experiment_id": "exp-001",
        "dataset_name": "synthetic",
        "outcome": "visit",
        "dataset_config_path": "configs/datasets/synthetic.yaml",
        "modeling_config_path": "configs/modeling/uplift_lgbm.yaml",
        "prediction_artifacts": {
            "t_learner_lgbm": str(prediction_path),
        },
    }


def test_manifest_resolves_exact_prediction_artifact(tmp_path: Path) -> None:
    prediction_path = tmp_path / "run01_predictions.parquet"
    _write_prediction_artifact(prediction_path)
    manifest_path = _write_manifest(tmp_path, _base_payload(prediction_path))

    resolved = resolve_prediction_paths(
        manifest=load_experiment_manifest(manifest_path),
        manifest_path=manifest_path,
        dataset_name="synthetic",
        outcome="visit",
    )

    assert resolved == {"t_learner_lgbm": prediction_path}


@pytest.mark.parametrize(
    "missing_key",
    ["dataset_config_path", "modeling_config_path"],
)
def test_manifest_requires_both_config_paths(
    tmp_path: Path,
    missing_key: str,
) -> None:
    prediction_path = tmp_path / "run01_predictions.parquet"
    _write_prediction_artifact(prediction_path)
    payload = _base_payload(prediction_path)
    payload.pop(missing_key)
    manifest_path = _write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match=missing_key):
        resolve_prediction_paths(
            manifest=load_experiment_manifest(manifest_path),
            manifest_path=manifest_path,
            dataset_name="synthetic",
            outcome="visit",
        )


def test_manifest_rejects_identity_mismatch(tmp_path: Path) -> None:
    prediction_path = tmp_path / "run01_predictions.parquet"
    _write_prediction_artifact(prediction_path)
    manifest_path = _write_manifest(tmp_path, _base_payload(prediction_path))

    with pytest.raises(ValueError, match="dataset_name"):
        resolve_prediction_paths(
            manifest=load_experiment_manifest(manifest_path),
            manifest_path=manifest_path,
            dataset_name="other_dataset",
            outcome="visit",
        )

    with pytest.raises(ValueError, match="outcome"):
        resolve_prediction_paths(
            manifest=load_experiment_manifest(manifest_path),
            manifest_path=manifest_path,
            dataset_name="synthetic",
            outcome="conversion",
        )


def test_build_manifest_records_new_config_contract(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    dataset_config_path = (
        project_root / "configs" / "datasets" / "synthetic.yaml"
    )
    modeling_config_path = (
        project_root / "configs" / "modeling" / "uplift_lgbm.yaml"
    )
    prediction_path = project_root / "artifacts" / "predictions.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.touch()

    manifest = build_experiment_manifest(
        experiment_id="exp-001",
        dataset_name="synthetic",
        outcome="visit",
        dataset_config_path=dataset_config_path,
        modeling_config_path=modeling_config_path,
        prediction_artifacts={"t_learner_lgbm": prediction_path},
        project_root=project_root,
    )

    assert manifest["dataset_config_path"] == (
        "configs/datasets/synthetic.yaml"
    )
    assert manifest["modeling_config_path"] == (
        "configs/modeling/uplift_lgbm.yaml"
    )
    assert "config_path" not in manifest
    assert manifest["prediction_artifacts"] == {
        "t_learner_lgbm": "artifacts/predictions.parquet"
    }


def test_manifest_loads_and_validates_model_provenance(tmp_path: Path) -> None:
    prediction_path = (
        tmp_path / "synthetic_visit_t_learner_lgbm_run01_predictions.parquet"
    )
    _write_prediction_artifact(prediction_path)

    provenance = {
        "artifact_type": "model_provenance",
        "dataset_name": "synthetic",
        "outcome": "visit",
        "policy_name": "t_learner_lgbm",
        "prediction_artifact": prediction_path.name,
        "model_kind": "t_learner",
        "mlflow_run_id": "run-001",
        "treatment_model_uri": "runs:/run-001/treatment_model",
        "control_model_uri": "runs:/run-001/control_model",
    }
    provenance_path = (
        tmp_path
        / "synthetic_visit_t_learner_lgbm_run01_model_provenance.json"
    )
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    model_artifacts = build_experiment_manifest_model_artifacts(
        {"t_learner_lgbm": prediction_path}
    )

    assert model_artifacts == {"t_learner_lgbm": provenance}

    validate_model_artifacts_match_predictions(
        model_artifacts=model_artifacts,
        prediction_paths={"t_learner_lgbm": prediction_path},
    )


def test_locked_test_model_resolution_requires_provenance(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "run01_predictions.parquet"
    prediction_path.touch()
    manifest = _base_payload(prediction_path)

    with pytest.raises(ValueError, match="model_artifacts"):
        resolve_model_artifacts(
            manifest=manifest,
            required_policies=("t_learner_lgbm",),
        )