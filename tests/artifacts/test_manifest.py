"""Tests for experiment manifest artifact resolution."""

import json
from pathlib import Path

import pandas as pd
import pytest

from uplift_modeling.artifacts.manifest import (
    build_experiment_manifest,
    build_experiment_manifest_model_artifacts,
    find_latest_prediction_artifacts,
    load_experiment_manifest,
    resolve_model_artifacts,
    resolve_prediction_paths,
    validate_model_artifacts_match_predictions,
)


def _write_prediction_artifact(
    prediction_path: Path,
    columns: tuple[str, ...] = (
        "row_id",
        "treatment",
        "outcome",
        "split",
        "score",
        "model_name",
    ),
) -> None:
    """Write an empty parquet prediction artifact with the requested schema."""
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({column: [] for column in columns}).to_parquet(
        prediction_path,
        index=False,
    )


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    """Write a manifest payload."""
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _base_payload(prediction_path: Path) -> dict:
    """Return a minimal valid manifest payload."""
    return {
        "artifact_type": "experiment_manifest",
        "experiment_id": "exp-001",
        "dataset_name": "criteo",
        "outcome": "visit",
        "config_path": "configs/modeling/criteo_response_lgbm.yaml",
        "prediction_artifacts": {
            "t_learner_lgbm": str(prediction_path),
        },
    }


def test_valid_manifest_resolves_exact_prediction_artifacts(tmp_path) -> None:
    """A valid manifest resolves the configured prediction file."""
    prediction_path = tmp_path / "run01_predictions.parquet"
    prediction_path.touch()
    manifest_path = _write_manifest(tmp_path, _base_payload(prediction_path))

    manifest = load_experiment_manifest(manifest_path)
    resolved_paths = resolve_prediction_paths(
        manifest=manifest,
        manifest_path=manifest_path,
        dataset_name="criteo",
        outcome="visit",
    )

    assert resolved_paths == {"t_learner_lgbm": prediction_path}


def test_manifest_path_that_does_not_exist_is_rejected(tmp_path) -> None:
    """A missing manifest raises a clear error."""
    with pytest.raises(FileNotFoundError, match="Experiment manifest"):
        load_experiment_manifest(tmp_path / "missing.json")


def test_missing_required_manifest_entry_is_rejected(tmp_path) -> None:
    """Required policy resolution rejects absent manifest entries."""
    prediction_path = tmp_path / "run01_predictions.parquet"
    prediction_path.touch()
    manifest_path = _write_manifest(tmp_path, _base_payload(prediction_path))

    with pytest.raises(ValueError, match="x_learner_lgbm"):
        resolve_prediction_paths(
            manifest=load_experiment_manifest(manifest_path),
            manifest_path=manifest_path,
            dataset_name="criteo",
            outcome="visit",
            required_policies=("x_learner_lgbm",),
        )


def test_missing_referenced_prediction_file_is_rejected(tmp_path) -> None:
    """Manifest-listed prediction paths must exist."""
    prediction_path = tmp_path / "missing_predictions.parquet"
    manifest_path = _write_manifest(tmp_path, _base_payload(prediction_path))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_prediction_paths(
            manifest=load_experiment_manifest(manifest_path),
            manifest_path=manifest_path,
            dataset_name="criteo",
            outcome="visit",
        )


def test_manifest_dataset_mismatch_is_rejected(tmp_path) -> None:
    """Manifest dataset must match the requested evaluation dataset."""
    prediction_path = tmp_path / "run01_predictions.parquet"
    prediction_path.touch()
    payload = _base_payload(prediction_path)
    payload["dataset_name"] = "retailhero"
    manifest_path = _write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match="dataset_name"):
        resolve_prediction_paths(
            manifest=load_experiment_manifest(manifest_path),
            manifest_path=manifest_path,
            dataset_name="criteo",
            outcome="visit",
        )


def test_manifest_outcome_mismatch_is_rejected(tmp_path) -> None:
    """Manifest outcome must match the requested evaluation outcome."""
    prediction_path = tmp_path / "run01_predictions.parquet"
    prediction_path.touch()
    payload = _base_payload(prediction_path)
    payload["outcome"] = "conversion"
    manifest_path = _write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match="outcome"):
        resolve_prediction_paths(
            manifest=load_experiment_manifest(manifest_path),
            manifest_path=manifest_path,
            dataset_name="criteo",
            outcome="visit",
        )


def test_duplicate_policy_entries_are_rejected(tmp_path) -> None:
    """Duplicate JSON policy keys are rejected while loading."""
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(
        """
        {
          "artifact_type": "experiment_manifest",
          "experiment_id": "exp-001",
          "dataset_name": "criteo",
          "outcome": "visit",
          "config_path": "configs/modeling/criteo_response_lgbm.yaml",
          "prediction_artifacts": {
            "t_learner_lgbm": "first.parquet",
            "t_learner_lgbm": "second.parquet"
          }
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        load_experiment_manifest(manifest_path)


def test_duplicate_artifact_paths_are_rejected(tmp_path) -> None:
    """Two policy entries cannot point to the same artifact file."""
    prediction_path = tmp_path / "run01_predictions.parquet"
    prediction_path.touch()
    payload = _base_payload(prediction_path)
    payload["prediction_artifacts"]["x_learner_lgbm"] = str(prediction_path)
    manifest_path = _write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match="duplicate prediction artifact path"):
        resolve_prediction_paths(
            manifest=load_experiment_manifest(manifest_path),
            manifest_path=manifest_path,
            dataset_name="criteo",
            outcome="visit",
        )


def test_build_manifest_uses_repository_relative_prediction_paths(tmp_path) -> None:
    """Manifest creation stores project-relative artifact paths when possible."""
    project_root = tmp_path / "project"
    prediction_path = project_root / "artifacts" / "predictions.parquet"
    prediction_path.parent.mkdir(parents=True)
    prediction_path.touch()

    manifest = build_experiment_manifest(
        experiment_id="exp-001",
        dataset_name="criteo",
        outcome="visit",
        config_path=project_root / "configs" / "model.yaml",
        prediction_artifacts={"t_learner_lgbm": prediction_path},
        project_root=project_root,
    )

    assert manifest["config_path"] == "configs/model.yaml"
    assert manifest["prediction_artifacts"] == {
        "t_learner_lgbm": "artifacts/predictions.parquet",
    }


def test_manifest_creation_loads_prediction_model_provenance(tmp_path) -> None:
    """Manifest creation picks up sidecar provenance next to predictions."""
    prediction_path = (
        tmp_path / "criteo_visit_t_learner_lgbm_run01_predictions.parquet"
    )
    _write_prediction_artifact(prediction_path)
    provenance_path = (
        tmp_path / "criteo_visit_t_learner_lgbm_run01_model_provenance.json"
    )
    provenance = {
        "artifact_type": "model_provenance",
        "dataset_name": "criteo",
        "outcome": "visit",
        "policy_name": "t_learner_lgbm",
        "prediction_artifact": prediction_path.name,
        "model_kind": "t_learner",
        "mlflow_run_id": "run-001",
        "treatment_model_uri": "runs:/run-001/treatment_model",
        "control_model_uri": "runs:/run-001/control_model",
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    model_artifacts = build_experiment_manifest_model_artifacts(
        {"t_learner_lgbm": prediction_path}
    )

    assert model_artifacts == {"t_learner_lgbm": provenance}


def test_locked_test_model_artifact_resolution_requires_provenance(
    tmp_path,
) -> None:
    """Locked-test model resolution rejects manifests without provenance."""
    prediction_path = tmp_path / "run01_predictions.parquet"
    prediction_path.touch()
    manifest = _base_payload(prediction_path)

    with pytest.raises(ValueError, match="model_artifacts"):
        resolve_model_artifacts(
            manifest=manifest,
            required_policies=("t_learner_lgbm",),
        )


def test_model_provenance_must_match_manifest_prediction_artifact(
    tmp_path,
) -> None:
    """Model provenance must be tied to the validation prediction artifact."""
    prediction_path = tmp_path / "run01_predictions.parquet"
    prediction_path.touch()
    model_artifacts = {
        "t_learner_lgbm": {
            "prediction_artifact": "different_predictions.parquet",
        },
    }

    with pytest.raises(ValueError, match="different_predictions.parquet"):
        validate_model_artifacts_match_predictions(
            model_artifacts=model_artifacts,
            prediction_paths={"t_learner_lgbm": prediction_path},
        )


def test_find_latest_prediction_artifacts_uses_latest_run_per_model(
    tmp_path,
) -> None:
    """Manifest discovery selects latest run-numbered prediction files."""
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    latest_t_learner = (
        prediction_dir
        / "criteo_conversion_t_learner_lgbm_run03_predictions.parquet"
    )
    latest_pooled_response = (
        prediction_dir
        / "criteo_conversion_response_lgbm_run02_predictions.parquet"
    )
    for filename in (
        "criteo_conversion_t_learner_lgbm_run01_predictions.parquet",
        latest_t_learner.name,
        "criteo_conversion_response_lgbm_run01_predictions.parquet",
        latest_pooled_response.name,
    ):
        _write_prediction_artifact(prediction_dir / filename)
    (
        prediction_dir
        / "criteo_visit_t_learner_lgbm_run99_predictions.parquet"
    ).touch()
    (prediction_dir / "conversion_response_model_predictions.parquet").touch()

    artifacts = find_latest_prediction_artifacts(
        prediction_dir=prediction_dir,
        dataset_name="criteo",
        outcome="conversion",
    )

    assert artifacts == {
        "pooled_response_lgbm": latest_pooled_response,
        "t_learner_lgbm": latest_t_learner,
    }


def test_find_latest_prediction_artifacts_skips_schema_incompatible_files(
    tmp_path,
) -> None:
    """Manifest discovery ignores run artifacts missing row_id."""
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    invalid_latest = (
        prediction_dir
        / "criteo_conversion_t_learner_lgbm_run03_predictions.parquet"
    )
    valid_previous = (
        prediction_dir
        / "criteo_conversion_t_learner_lgbm_run02_predictions.parquet"
    )
    _write_prediction_artifact(
        invalid_latest,
        columns=("treatment", "outcome", "split", "score", "model_name"),
    )
    _write_prediction_artifact(valid_previous)

    artifacts = find_latest_prediction_artifacts(
        prediction_dir=prediction_dir,
        dataset_name="criteo",
        outcome="conversion",
    )

    assert artifacts == {"t_learner_lgbm": valid_previous}


def test_find_latest_prediction_artifacts_skips_test_split_files(
    tmp_path,
) -> None:
    """Manifest discovery ignores normal artifacts containing test rows."""
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    invalid_latest = (
        prediction_dir
        / "criteo_conversion_t_learner_lgbm_run03_predictions.parquet"
    )
    valid_previous = (
        prediction_dir
        / "criteo_conversion_t_learner_lgbm_run02_predictions.parquet"
    )
    pd.DataFrame(
        {
            "row_id": [1, 2],
            "treatment": [1, 0],
            "outcome": [1, 0],
            "split": ["validation", "test"],
            "score": [0.5, 0.4],
            "model_name": ["t_learner_lgbm"] * 2,
        }
    ).to_parquet(invalid_latest, index=False)
    pd.DataFrame(
        {
            "row_id": [1],
            "treatment": [1],
            "outcome": [1],
            "split": ["validation"],
            "score": [0.5],
            "model_name": ["t_learner_lgbm"],
        }
    ).to_parquet(valid_previous, index=False)

    artifacts = find_latest_prediction_artifacts(
        prediction_dir=prediction_dir,
        dataset_name="criteo",
        outcome="conversion",
    )

    assert artifacts == {"t_learner_lgbm": valid_previous}


def test_find_latest_prediction_artifacts_rejects_missing_requested_model(
    tmp_path,
) -> None:
    """Requested model names must exist after artifact discovery."""
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_prediction_artifact(
        prediction_dir
        / "criteo_conversion_t_learner_lgbm_run01_predictions.parquet"
    )

    with pytest.raises(FileNotFoundError, match="x_learner_lgbm"):
        find_latest_prediction_artifacts(
            prediction_dir=prediction_dir,
            dataset_name="criteo",
            outcome="conversion",
            model_names=("t_learner_lgbm", "x_learner_lgbm"),
        )
