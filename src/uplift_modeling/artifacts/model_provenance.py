"""Model provenance artifacts for prediction files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from uplift_modeling.artifacts.json import save_json_artifact


MODEL_PROVENANCE_ARTIFACT_TYPE = "model_provenance"
PREDICTION_ARTIFACT_SUFFIX = "_predictions.parquet"
MODEL_PROVENANCE_SUFFIX = "_model_provenance.json"


def get_model_provenance_path(prediction_path: Path) -> Path:
    """Return the sidecar provenance path for a prediction artifact."""
    if not prediction_path.name.endswith(PREDICTION_ARTIFACT_SUFFIX):
        raise ValueError(
            "Prediction artifact must end with "
            f"'{PREDICTION_ARTIFACT_SUFFIX}': {prediction_path}"
        )

    basename = prediction_path.name.removesuffix(PREDICTION_ARTIFACT_SUFFIX)
    return prediction_path.with_name(f"{basename}{MODEL_PROVENANCE_SUFFIX}")


def build_model_provenance_payload(
    dataset_name: str,
    outcome: str,
    policy_name: str,
    prediction_path: Path,
    model_kind: str,
    mlflow_run_id: str,
    model_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Build model provenance tied to one validation prediction artifact."""
    if not mlflow_run_id:
        raise ValueError("mlflow_run_id must be a non-empty string.")
    if not model_artifacts:
        raise ValueError("model_artifacts must not be empty.")

    return {
        "artifact_type": MODEL_PROVENANCE_ARTIFACT_TYPE,
        "dataset_name": dataset_name,
        "outcome": outcome,
        "policy_name": policy_name,
        "prediction_artifact": prediction_path.name,
        "model_kind": model_kind,
        "mlflow_run_id": mlflow_run_id,
        **dict(model_artifacts),
    }


def save_model_provenance_payload(
    payload: dict[str, Any],
    output_path: Path,
) -> Path:
    """Save model provenance as a formatted JSON artifact."""
    return save_json_artifact(payload, output_path)


def load_model_provenance_payload(provenance_path: Path) -> dict[str, Any]:
    """Load one model provenance JSON artifact."""
    if not provenance_path.exists():
        raise FileNotFoundError(
            f"Model provenance artifact does not exist: {provenance_path}"
        )

    with provenance_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise ValueError(
            "Model provenance artifact must contain a JSON object: "
            f"{provenance_path}"
        )

    return payload


def load_prediction_model_provenance(
    prediction_artifacts: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    """Load available sidecar model provenance for prediction artifacts."""
    model_artifacts: dict[str, dict[str, Any]] = {}

    for policy_name, prediction_path in prediction_artifacts.items():
        provenance_path = get_model_provenance_path(prediction_path)
        if provenance_path.exists():
            model_artifacts[policy_name] = load_model_provenance_payload(
                provenance_path
            )

    return model_artifacts
