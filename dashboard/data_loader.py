"""Load framework artifacts for the customer-targeting dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dashboard_data"
DEFAULT_CONFIG_ROOT = PROJECT_ROOT / "configs" / "dashboard"


@dataclass(frozen=True)
class DashboardData:
    """Artifacts and optional display configuration for one dashboard dataset."""

    dataset_key: str
    selection: dict[str, Any]
    locked_test: dict[str, Any]
    decision_data: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    entity_id_column: str
    dashboard_config: dict[str, Any]


def discover_datasets(
    data_root: Path = DEFAULT_DATA_ROOT,
) -> list[str]:
    """Return dataset folders that contain a Selection Gate artifact."""

    if not data_root.exists():
        return []

    dataset_keys: list[str] = []

    for dataset_dir in sorted(data_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        metrics_dir = dataset_dir / "metrics"

        if metrics_dir.exists() and any(
            metrics_dir.glob("*_model_selection_gate.json")
        ):
            dataset_keys.append(dataset_dir.name)

    return dataset_keys


def load_dashboard_config(
    dataset_key: str,
    config_root: Path = DEFAULT_CONFIG_ROOT,
) -> dict[str, Any]:
    """Load optional dataset-specific dashboard presentation settings."""

    config_path = config_root / f"{dataset_key}.json"

    if not config_path.exists():
        return {}

    payload = _load_json(config_path)

    if payload.get("dataset_key") not in (None, dataset_key):
        raise ValueError(
            f"Dashboard config {config_path} belongs to a different dataset."
        )

    return payload


def _find_single_file(
    directory: Path,
    pattern: str,
    label: str,
) -> Path:
    paths = sorted(directory.glob(pattern))

    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one {label} matching {pattern!r} "
            f"in {directory}, found {len(paths)}: {paths}"
        )

    return paths[0]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, "
            f"got {type(payload).__name__}."
        )

    return payload


def _require_keys(
    payload: dict[str, Any],
    required_keys: set[str],
    label: str,
) -> None:
    missing = required_keys.difference(payload)

    if missing:
        raise ValueError(
            f"{label} is missing required fields: {sorted(missing)}"
        )


def load_dashboard_data(
    dataset_key: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    config_root: Path = DEFAULT_CONFIG_ROOT,
) -> DashboardData:
    """Load and validate one dashboard dataset."""

    dataset_dir = data_root / dataset_key
    metrics_dir = dataset_dir / "metrics"
    predictions_dir = dataset_dir / "predictions"
    processed_dir = dataset_dir / "processed"

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dashboard dataset does not exist: {dataset_dir}"
        )

    dashboard_config = load_dashboard_config(
        dataset_key,
        config_root=config_root,
    )

    selection_path = _find_single_file(
        metrics_dir,
        "*_model_selection_gate.json",
        "Selection Gate artifact",
    )

    selection = _load_json(selection_path)

    _require_keys(
        selection,
        {
            "experiment_id",
            "dataset_name",
            "baseline_policy",
            "recommended_deployment_policy",
            "replacement_gate_passed",
        },
        "Selection Gate artifact",
    )

    experiment_id = str(selection["experiment_id"])

    locked_test_path = _find_single_file(
        metrics_dir,
        f"*{experiment_id}*_locked_test_evaluation.json",
        "locked-test evaluation artifact",
    )

    locked_test = _load_json(locked_test_path)

    _require_keys(
        locked_test,
        {
            "locked_test_rows",
            "prediction_artifacts",
        },
        "Locked-test evaluation artifact",
    )

    selection_artifact = locked_test.get("selection_artifact")

    if (
        selection_artifact is not None
        and selection_artifact != selection_path.name
    ):
        raise ValueError(
            "Locked-test evaluation references a different "
            f"Selection Gate: {selection_artifact!r}"
        )

    decision_path = _find_single_file(
        processed_dir,
        "decision_*.parquet",
        "decision dataset",
    )

    decision_data = pd.read_parquet(decision_path)

    required_decision_columns = {"row_id", "split"}
    missing_decision_columns = required_decision_columns.difference(
        decision_data.columns
    )

    if missing_decision_columns:
        raise ValueError(
            "Decision dataset is missing required columns: "
            f"{sorted(missing_decision_columns)}"
        )

    if not decision_data["row_id"].is_unique:
        raise ValueError("Decision dataset row_id must be unique.")

    entity_id_column = str(
        dashboard_config.get("entity_id_column", "row_id")
    )

    if entity_id_column not in decision_data.columns:
        raise ValueError(
            f"Configured entity_id_column {entity_id_column!r} "
            "does not exist in the decision dataset."
        )

    if not decision_data[entity_id_column].notna().all():
        raise ValueError(
            f"Entity ID column {entity_id_column!r} contains null values."
        )

    if not decision_data[entity_id_column].is_unique:
        raise ValueError(
            f"Entity ID column {entity_id_column!r} must be unique."
        )

    prediction_artifacts = locked_test["prediction_artifacts"]

    if not isinstance(prediction_artifacts, dict):
        raise ValueError(
            "locked_test.prediction_artifacts must be a mapping."
        )

    predictions: dict[str, pd.DataFrame] = {}

    for policy_name, filename in prediction_artifacts.items():
        prediction_path = predictions_dir / str(filename)

        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Missing prediction artifact for {policy_name}: "
                f"{prediction_path}"
            )

        prediction_data = pd.read_parquet(prediction_path)

        required_prediction_columns = {"row_id", "score"}
        missing_prediction_columns = required_prediction_columns.difference(
            prediction_data.columns
        )

        if missing_prediction_columns:
            raise ValueError(
                f"Prediction artifact {prediction_path.name} "
                f"is missing columns: {sorted(missing_prediction_columns)}"
            )

        if not prediction_data["row_id"].is_unique:
            raise ValueError(
                f"Prediction row_id must be unique: {prediction_path.name}"
            )

        predictions[str(policy_name)] = prediction_data

    recommended_policy = str(
        selection["recommended_deployment_policy"]
    )

    if recommended_policy not in predictions:
        raise ValueError(
            "No locked-test prediction artifact exists for recommended "
            f"policy {recommended_policy!r}."
        )

    return DashboardData(
        dataset_key=dataset_key,
        selection=selection,
        locked_test=locked_test,
        decision_data=decision_data,
        predictions=predictions,
        entity_id_column=entity_id_column,
        dashboard_config=dashboard_config,
    )
