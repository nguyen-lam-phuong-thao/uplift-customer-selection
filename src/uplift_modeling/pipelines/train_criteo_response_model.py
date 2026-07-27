"""Train the Criteo LightGBM response model for visit or conversion."""

import argparse
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import pandas as pd
import pyarrow.parquet as pq

from uplift_modeling.artifacts.json import save_json_artifact
from uplift_modeling.artifacts.predictions import (
    save_prediction_parquet_in_batches,
)
from uplift_modeling.data.criteo import FEATURE_COLUMNS
from uplift_modeling.evaluation.binary_metrics import calculate_binary_metrics
from uplift_modeling.models.response_model import (
    build_response_model,
    fit_response_model,
)
from uplift_modeling.tracking.mlflow_tracking import setup_mlflow
from uplift_modeling.utils.config import (
    get_config_section,
    get_project_root,
    load_yaml_config,
    resolve_project_path,
)


LOGGER = logging.getLogger(__name__)
VALID_OUTCOMES = ("visit", "conversion")
ALLOWED_PREDICTION_SPLITS = ("train", "validation", "test")
DEFAULT_PREDICTION_SPLITS = ("validation", "test")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Criteo response-model training."""
    parser = argparse.ArgumentParser(
        description="Train a Criteo LightGBM response model."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the response-model YAML config.",
    )
    parser.add_argument(
        "--outcome",
        default="visit",
        choices=VALID_OUTCOMES,
        help="Outcome to model. Defaults to visit.",
    )
    return parser.parse_args()


def validate_outcome(outcome: str) -> None:
    """Validate the requested response outcome."""
    if outcome not in VALID_OUTCOMES:
        valid_outcomes_text = ", ".join(VALID_OUTCOMES)
        raise ValueError(
            f"outcome must be one of: {valid_outcomes_text}. "
            f"Received: {outcome}"
        )


def get_prediction_splits(output_config: dict[str, Any]) -> tuple[str, ...]:
    """Return validated prediction splits from config."""
    configured_splits = output_config.get(
        "prediction_splits",
        list(DEFAULT_PREDICTION_SPLITS),
    )
    if not isinstance(configured_splits, list):
        raise ValueError("outputs.prediction_splits must be a list of strings.")

    prediction_splits = tuple(configured_splits)
    if not prediction_splits:
        raise ValueError("outputs.prediction_splits must not be empty.")

    if not all(isinstance(split, str) for split in prediction_splits):
        raise ValueError("outputs.prediction_splits must contain only strings.")

    invalid_splits = sorted(
        set(prediction_splits).difference(ALLOWED_PREDICTION_SPLITS)
    )
    if invalid_splits:
        allowed_text = ", ".join(ALLOWED_PREDICTION_SPLITS)
        invalid_text = ", ".join(invalid_splits)
        raise ValueError(
            "outputs.prediction_splits contains invalid values. "
            f"Allowed values: {allowed_text}. Received: {invalid_text}"
        )

    if len(set(prediction_splits)) != len(prediction_splits):
        raise ValueError("outputs.prediction_splits must not contain duplicates.")

    return prediction_splits


def get_tracking_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return tracking config with local-safe defaults."""
    tracking_config = config.get("tracking", {})

    if tracking_config is None:
        return {"log_predictions": False}

    if not isinstance(tracking_config, dict):
        raise ValueError("Config section 'tracking' must be a mapping.")

    return {"log_predictions": False, **tracking_config}


def validate_feature_columns(feature_columns: list[str]) -> tuple[str, ...]:
    """Validate that the response model uses only f0-f11 features."""
    expected_features = tuple(FEATURE_COLUMNS)
    configured_features = tuple(feature_columns)

    if configured_features != expected_features:
        expected_text = ", ".join(expected_features)
        received_text = ", ".join(configured_features)
        raise ValueError(
            "Response model feature_columns must be exactly f0-f11 in order. "
            f"Expected: {expected_text}. Received: {received_text}"
        )

    return configured_features


def get_parquet_columns(parquet_path: Path) -> set[str]:
    """Read parquet schema columns without loading the full dataset."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input parquet does not exist: {parquet_path}")

    schema = pq.read_schema(parquet_path)
    return set(schema.names)


def resolve_target_column(
    requested_outcome: str,
    available_columns: set[str],
) -> str:
    """Choose the requested outcome column or generic outcome fallback."""
    if requested_outcome in available_columns:
        return requested_outcome

    if "outcome" in available_columns:
        return "outcome"

    raise ValueError(
        "Input parquet is missing the requested outcome column "
        f"'{requested_outcome}' and fallback column 'outcome'."
    )


def validate_required_columns(
    available_columns: set[str],
    required_columns: tuple[str, ...],
) -> None:
    """Raise a clear error when required parquet columns are missing."""
    missing_columns = sorted(set(required_columns).difference(available_columns))

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Input parquet is missing columns: {missing_text}")


def load_training_frame(
    parquet_path: Path,
    feature_columns: tuple[str, ...],
    treatment_column: str,
    split_column: str,
    requested_outcome: str,
) -> tuple[pd.DataFrame, str]:
    """Load only columns required for Criteo response-model training."""
    available_columns = get_parquet_columns(parquet_path)
    target_column = resolve_target_column(requested_outcome, available_columns)
    required_columns = (
        *feature_columns,
        treatment_column,
        split_column,
        target_column,
    )
    validate_required_columns(available_columns, required_columns)

    LOGGER.info("Loading training data from %s", parquet_path)
    dataframe = pd.read_parquet(parquet_path, columns=list(required_columns))
    return dataframe, target_column


def get_split_frame(
    dataframe: pd.DataFrame,
    split_column: str,
    split_value: str,
) -> pd.DataFrame:
    """Return rows for one split label."""
    split_frame = dataframe.loc[dataframe[split_column] == split_value].copy()

    if split_frame.empty:
        raise ValueError(f"Split '{split_value}' has no rows.")

    return split_frame


def train_response_pipeline(config_path: Path, outcome: str) -> None:
    """Run the configured Criteo response-model training pipeline."""
    validate_outcome(outcome)
    project_root = get_project_root(Path(__file__))
    config = load_yaml_config(config_path)

    data_config = get_config_section(config, "data")
    training_config = get_config_section(config, "training")
    model_config = get_config_section(config, "model")
    output_config = get_config_section(config, "outputs")
    tracking_config = get_tracking_config(config)
    prediction_splits = get_prediction_splits(output_config)

    dataset_name = str(data_config["dataset_name"])
    processed_paths = data_config.get("processed_paths")
    if not isinstance(processed_paths, dict) or outcome not in processed_paths:
        raise ValueError(
            "Config data.processed_paths must define paths for visit and "
            "conversion."
        )

    feature_columns = validate_feature_columns(data_config["feature_columns"])
    treatment_column = str(data_config["treatment_column"])
    split_column = str(data_config["split_column"])
    train_split = str(training_config.get("train_split", "train"))
    validation_split = str(
        training_config.get("validation_split", "validation")
    )
    test_split = str(training_config.get("test_split", "test"))
    prediction_batch_size = int(training_config["prediction_batch_size"])
    early_stopping_rounds = int(training_config["early_stopping_rounds"])
    log_evaluation_period = int(training_config["log_evaluation_period"])
    model_name = str(model_config["name"])
    model_params = dict(model_config["params"])

    data_path = resolve_project_path(processed_paths[outcome], project_root)
    prediction_path = resolve_project_path(
        Path(output_config["prediction_dir"])
        / f"{outcome}_response_model_predictions.parquet",
        project_root,
    )
    metrics_path = resolve_project_path(
        Path(output_config["metric_dir"])
        / f"{outcome}_response_model_metrics.json",
        project_root,
    )

    dataframe, target_column = load_training_frame(
        parquet_path=data_path,
        feature_columns=feature_columns,
        treatment_column=treatment_column,
        split_column=split_column,
        requested_outcome=outcome,
    )

    train_frame = get_split_frame(dataframe, split_column, train_split)
    validation_frame = get_split_frame(
        dataframe,
        split_column,
        validation_split,
    )
    test_frame = get_split_frame(dataframe, split_column, test_split)
    split_frames = {
        "train": train_frame,
        "validation": validation_frame,
        "test": test_frame,
    }

    LOGGER.info("Training %s for %s", model_name, outcome)
    model = build_response_model(model_params)
    fit_response_model(
        model=model,
        X_train=train_frame.loc[:, feature_columns],
        y_train=train_frame[target_column],
        X_valid=validation_frame.loc[:, feature_columns],
        y_valid=validation_frame[target_column],
        early_stopping_rounds=early_stopping_rounds,
        log_evaluation_period=log_evaluation_period,
    )

    validation_scores = model.predict_proba(
        validation_frame.loc[:, feature_columns],
    )[:, 1]
    test_scores = model.predict_proba(
        test_frame.loc[:, feature_columns],
    )[:, 1]

    metrics = {
        "dataset_name": dataset_name,
        "outcome": outcome,
        "target_column": target_column,
        "model_name": model_name,
        "validation": calculate_binary_metrics(
            validation_frame[target_column],
            validation_scores,
        ),
        "test": calculate_binary_metrics(
            test_frame[target_column],
            test_scores,
        ),
    }

    save_prediction_parquet_in_batches(
        dataframes=tuple(split_frames[split] for split in prediction_splits),
        output_path=prediction_path,
        feature_columns=feature_columns,
        treatment_column=treatment_column,
        split_column=split_column,
        outcome_column=target_column,
        model_name=model_name,
        batch_size=prediction_batch_size,
        score_batch=lambda X_batch: model.predict_proba(X_batch)[:, 1],
    )

    save_json_artifact(metrics, metrics_path)

    project_config = config.get("project", {})
    experiment_name = (
        project_config.get("experiment_name", "criteo-uplift-modeling")
        if isinstance(project_config, dict)
        else "criteo-uplift-modeling"
    )
    setup_mlflow(str(experiment_name))
    mlflow_params = {
        "dataset_name": dataset_name,
        "outcome": outcome,
        "model_name": model_name,
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        **model_params,
    }
    mlflow_metrics = {
        "validation_roc_auc": metrics["validation"]["roc_auc"],
        "validation_average_precision": metrics["validation"][
            "average_precision"
        ],
        "validation_log_loss": metrics["validation"]["log_loss"],
        "test_roc_auc": metrics["test"]["roc_auc"],
        "test_average_precision": metrics["test"]["average_precision"],
        "test_log_loss": metrics["test"]["log_loss"],
    }

    with mlflow.start_run(run_name=f"{outcome}_{model_name}"):
        mlflow.log_params(mlflow_params)
        mlflow.log_metrics(mlflow_metrics)
        mlflow.log_artifact(str(metrics_path))

        if bool(tracking_config["log_predictions"]):
            mlflow.log_artifact(str(prediction_path))

        mlflow.lightgbm.log_model(model, artifact_path="model")

    LOGGER.info("Saved predictions to %s", prediction_path)
    LOGGER.info("Saved metrics to %s", metrics_path)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    project_root = get_project_root(Path(__file__))
    config_path = resolve_project_path(args.config, project_root)
    train_response_pipeline(config_path=config_path, outcome=args.outcome)


if __name__ == "__main__":
    main()
