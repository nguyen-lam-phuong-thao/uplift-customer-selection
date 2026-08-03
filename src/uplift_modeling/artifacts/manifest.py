"""Experiment manifest loading and prediction artifact resolution."""

from __future__ import annotations

import glob
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from uplift_modeling.artifacts.json import save_json_artifact
from uplift_modeling.artifacts.model_provenance import (
    load_prediction_model_provenance,
)

EXPERIMENT_MANIFEST_ARTIFACT_TYPE = "experiment_manifest"
REQUIRED_PREDICTION_COLUMNS = (
    "row_id",
    "treatment",
    "outcome",
    "split",
    "score",
    "model_name",
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    payload: dict[str, Any] = {}

    for key, value in pairs:
        if key in payload:
            raise ValueError(
                f"Experiment manifest contains duplicate key: {key}"
            )
        payload[key] = value

    return payload


def load_experiment_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load an experiment manifest JSON artifact."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Experiment manifest does not exist: {manifest_path}"
        )

    with manifest_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file, object_pairs_hook=_reject_duplicate_keys)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Experiment manifest must contain a JSON object: {manifest_path}"
        )

    return payload


def build_experiment_manifest(
    experiment_id: str,
    dataset_name: str,
    outcome: str,
    config_path: str | Path,
    prediction_artifacts: dict[str, str | Path],
    model_artifacts: dict[str, dict[str, Any]] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Build an experiment manifest payload."""
    payload = {
        "artifact_type": EXPERIMENT_MANIFEST_ARTIFACT_TYPE,
        "experiment_id": experiment_id,
        "dataset_name": dataset_name,
        "outcome": outcome,
        "config_path": _format_manifest_path(config_path, project_root),
        "prediction_artifacts": {
            policy_name: _format_manifest_path(prediction_path, project_root)
            for policy_name, prediction_path in sorted(
                prediction_artifacts.items()
            )
        },
    }
    if model_artifacts:
        payload["model_artifacts"] = model_artifacts

    validate_experiment_manifest(
        manifest=payload,
        dataset_name=dataset_name,
        outcome=outcome,
    )
    return payload


def build_experiment_manifest_model_artifacts(
    prediction_artifacts: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    """Load model provenance sidecars for discovered prediction artifacts."""
    return load_prediction_model_provenance(prediction_artifacts)


def save_experiment_manifest(
    manifest: dict[str, Any],
    output_path: Path,
    dataset_name: str,
    outcome: str,
) -> Path:
    """Validate and save an experiment manifest JSON artifact."""
    validate_experiment_manifest(
        manifest=manifest,
        dataset_name=dataset_name,
        outcome=outcome,
    )
    return save_json_artifact(manifest, output_path)


def get_missing_prediction_columns(prediction_path: Path) -> tuple[str, ...]:
    """Return required prediction columns missing from a parquet artifact."""
    schema_columns = set(pq.read_schema(prediction_path).names)
    return tuple(
        sorted(set(REQUIRED_PREDICTION_COLUMNS).difference(schema_columns))
    )


def validate_experiment_manifest(
    manifest: dict[str, Any],
    dataset_name: str,
    outcome: str,
) -> dict[str, str]:
    """Validate the manifest contract and return raw prediction references."""
    artifact_type = manifest.get("artifact_type")
    if artifact_type != EXPERIMENT_MANIFEST_ARTIFACT_TYPE:
        raise ValueError(
            "Experiment manifest artifact_type must be "
            f"'{EXPERIMENT_MANIFEST_ARTIFACT_TYPE}'. "
            f"Received: {artifact_type}"
        )

    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError(
            "Experiment manifest must contain a non-empty 'experiment_id'."
        )

    manifest_dataset_name = manifest.get("dataset_name")
    if manifest_dataset_name != dataset_name:
        raise ValueError(
            "Experiment manifest dataset_name does not match requested "
            f"dataset. Expected '{dataset_name}', received "
            f"'{manifest_dataset_name}'."
        )

    manifest_outcome = manifest.get("outcome")
    if manifest_outcome != outcome:
        raise ValueError(
            "Experiment manifest outcome does not match requested outcome. "
            f"Expected '{outcome}', received '{manifest_outcome}'."
        )

    if not _has_config_identity(manifest):
        raise ValueError(
            "Experiment manifest must contain a non-empty 'config_path' "
            "or 'config_identity'."
        )

    prediction_artifacts = manifest.get("prediction_artifacts")
    if not isinstance(prediction_artifacts, dict) or not prediction_artifacts:
        raise ValueError(
            "Experiment manifest must contain a non-empty "
            "'prediction_artifacts' mapping."
        )

    raw_paths: dict[str, str] = {}
    observed_references: set[str] = set()
    for policy_name, path_value in prediction_artifacts.items():
        if not isinstance(policy_name, str) or not policy_name:
            raise ValueError(
                "Experiment manifest prediction artifact keys must be "
                "non-empty policy/model names."
            )
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(
                "Experiment manifest prediction artifact values must be "
                f"non-empty file paths. Policy/model: {policy_name}"
            )
        if glob.has_magic(path_value):
            raise ValueError(
                "Experiment manifest prediction artifact paths must be "
                f"explicit files, not glob patterns. Policy/model "
                f"'{policy_name}' uses: {path_value}"
            )
        if path_value in observed_references:
            raise ValueError(
                "Experiment manifest contains duplicate prediction artifact "
                f"path: {path_value}"
            )

        observed_references.add(path_value)
        raw_paths[policy_name] = path_value

    return raw_paths


def resolve_prediction_paths(
    manifest: dict[str, Any],
    manifest_path: Path,
    dataset_name: str,
    outcome: str,
    project_root: Path | None = None,
    required_policies: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Resolve manifest prediction references to explicit files."""
    raw_paths = validate_experiment_manifest(
        manifest=manifest,
        dataset_name=dataset_name,
        outcome=outcome,
    )
    policies = tuple(raw_paths) if required_policies is None else required_policies
    resolved_paths: dict[str, Path] = {}

    for policy_name in policies:
        path_value = raw_paths.get(policy_name)
        if path_value is None:
            raise ValueError(
                "Experiment manifest is missing prediction artifact for "
                f"policy/model '{policy_name}'."
            )

        prediction_path = _resolve_manifest_file_path(
            path_value=path_value,
            manifest_path=manifest_path,
            project_root=project_root,
        )
        if not prediction_path.exists():
            raise FileNotFoundError(
                "Experiment manifest references a prediction artifact that "
                f"does not exist for policy/model '{policy_name}': "
                f"{prediction_path}"
            )
        if not prediction_path.is_file():
            raise ValueError(
                "Experiment manifest prediction artifact must be a file for "
                f"policy/model '{policy_name}': {prediction_path}"
            )
        validate_prediction_artifact_model_name(
            prediction_path=prediction_path,
            manifest_policy=policy_name,
        )

        resolved_paths[policy_name] = prediction_path

    return resolved_paths


def validate_prediction_artifact_model_name(
    prediction_path: Path,
    manifest_policy: str,
) -> None:
    """Validate one prediction artifact declares the manifest policy identity."""
    schema_columns = set(pq.read_schema(prediction_path).names)
    if "model_name" not in schema_columns:
        raise ValueError(
            f"Prediction artifact {prediction_path} is missing required "
            "'model_name' column."
        )

    model_name_values = (
        pq.read_table(prediction_path, columns=["model_name"])
        .column("model_name")
        .to_pylist()
    )
    model_names = [
        value.strip()
        for value in model_name_values
        if isinstance(value, str) and value.strip()
    ]
    invalid_count = len(model_name_values) - len(model_names)
    unique_model_names = sorted(set(model_names))

    if invalid_count or len(unique_model_names) != 1:
        raise ValueError(
            "Prediction artifact model_name must contain exactly one "
            f"non-null non-empty value for manifest policy "
            f"'{manifest_policy}'. Received: {unique_model_names}."
        )

    model_name = unique_model_names[0]
    if model_name != manifest_policy:
        raise ValueError(
            f"Prediction artifact model_name '{model_name}' does not match "
            f"manifest policy '{manifest_policy}'."
        )


def resolve_model_artifacts(
    manifest: dict[str, Any],
    required_policies: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Resolve locked-test model artifact metadata from a manifest."""
    model_artifacts = manifest.get("model_artifacts")
    if not isinstance(model_artifacts, dict):
        raise ValueError(
            "Experiment manifest must contain a 'model_artifacts' mapping "
            "for locked-test scoring."
        )

    resolved_artifacts: dict[str, dict[str, Any]] = {}
    for policy_name in required_policies:
        model_artifact = model_artifacts.get(policy_name)
        if not isinstance(model_artifact, dict):
            raise ValueError(
                "Experiment manifest is missing model provenance for "
                f"policy/model '{policy_name}'."
            )
        resolved_artifacts[policy_name] = model_artifact

    return resolved_artifacts


def validate_model_artifacts_match_predictions(
    model_artifacts: Mapping[str, Mapping[str, Any]],
    prediction_paths: Mapping[str, Path],
) -> None:
    """Validate model provenance is tied to the manifest prediction files."""
    for policy_name, model_artifact in model_artifacts.items():
        prediction_path = prediction_paths.get(policy_name)
        if prediction_path is None:
            raise ValueError(
                "Cannot validate model provenance without a prediction "
                f"artifact for policy/model '{policy_name}'."
            )

        prediction_artifact = model_artifact.get("prediction_artifact")
        if not isinstance(prediction_artifact, str) or not prediction_artifact:
            raise ValueError(
                "Model provenance for policy/model "
                f"'{policy_name}' must contain a non-empty "
                "'prediction_artifact' string."
            )
        if prediction_artifact != prediction_path.name:
            raise ValueError(
                "Model provenance for policy/model "
                f"'{policy_name}' is tied to prediction artifact "
                f"'{prediction_artifact}', but the manifest references "
                f"'{prediction_path.name}'."
            )


def _has_config_identity(manifest: dict[str, Any]) -> bool:
    """Return whether the manifest has a config path or identity."""
    for key in ("config_path", "config_identity"):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return True

    return False


def _resolve_manifest_file_path(
    path_value: str,
    manifest_path: Path,
    project_root: Path | None,
) -> Path:
    """Resolve one manifest file path."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    base_dir = project_root if project_root is not None else manifest_path.parent
    return (base_dir / path).resolve()


def _format_manifest_path(
    path_value: str | Path,
    project_root: Path | None,
) -> str:
    """Format a path for portable manifest storage."""
    path = Path(path_value)

    if project_root is None:
        return path.as_posix()

    try:
        relative_path = path.resolve().relative_to(project_root.resolve())
        return relative_path.as_posix()
    except ValueError:
        return path.resolve().as_posix()
