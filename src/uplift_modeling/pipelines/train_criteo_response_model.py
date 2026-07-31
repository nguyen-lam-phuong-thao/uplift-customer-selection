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
from uplift_modeling.artifacts.naming import (
    build_artifact_filename,
    find_next_run_number,
    get_artifact_model_name,
)
from uplift_modeling.artifacts.predictions import (
    save_prediction_parquet_in_batches,
)
from uplift_modeling.data.dataset_spec import (
    DatasetSpec,
    get_dataset_spec,
    validate_supported_outcome,
)
from uplift_modeling.data.row_id import validate_row_id_column
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
        help="Outcome to model. Defaults to visit.",
    )
    return parser.parse_args()


def validate_outcome(outcome: str, dataset_spec: DatasetSpec) -> None:
    """Validate the requested response outcome."""
    validate_supported_outcome(dataset_spec, outcome)


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


def get_processed_data_path(
    data_config: dict[str, Any],
    outcome: str,
    project_root: Path,
) -> Path:
    """Return the configured processed parquet path for one outcome."""
    processed_paths = data_config.get("processed_paths")
    if not isinstance(processed_paths, dict) or outcome not in processed_paths:
        raise ValueError(
            "Config data.processed_paths must define a path for the requested "
            f"outcome: {outcome}."
        )

    return resolve_project_path(processed_paths[outcome], project_root)


def resolve_dataset_spec(data_config: dict[str, Any]) -> DatasetSpec:
    """Resolve the stable dataset schema selected by config."""
    return get_dataset_spec(str(data_config["dataset_name"]))


def get_debug_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return debug config with safe defaults."""
    debug_config = config.get("debug", {})

    if debug_config is None:
        return {"sample_rows": None, "random_state": 42}

    if not isinstance(debug_config, dict):
        raise ValueError("Config section 'debug' must be a mapping.")

    sample_rows = debug_config.get("sample_rows")
    if sample_rows is not None:
        sample_rows = int(sample_rows)
        if sample_rows <= 0:
            raise ValueError(
                "debug.sample_rows must be a positive integer or null."
            )

    return {
        "sample_rows": sample_rows,
        "random_state": int(debug_config.get("random_state", 42)),
    }


def apply_debug_sample(
    dataframe: pd.DataFrame,
    sample_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Optionally sample rows for faster debug runs."""
    if sample_rows is None:
        return dataframe

    if sample_rows >= len(dataframe):
        LOGGER.info(
            "debug.sample_rows >= dataframe rows, using full dataframe."
        )
        return dataframe

    LOGGER.info(
        "Using debug sample: %s rows from %s rows",
        sample_rows,
        len(dataframe),
    )
    return dataframe.sample(
        n=sample_rows,
        random_state=random_state,
    ).reset_index(drop=True)


def load_training_frame(
    parquet_path: Path,
    dataset_spec: DatasetSpec,
    requested_outcome: str,
) -> tuple[pd.DataFrame, str]:
    """Load only columns required for the selected dataset training frame."""
    validate_supported_outcome(dataset_spec, requested_outcome)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Input parquet does not exist: {parquet_path}")

    available_columns = set(pq.read_schema(parquet_path).names)

    if requested_outcome not in available_columns:
        raise ValueError(
            "Input parquet is missing the requested outcome column "
            f"'{requested_outcome}'."
        )

    required_columns = (
        dataset_spec.row_id_column,
        *dataset_spec.feature_columns,
        dataset_spec.treatment_column,
        dataset_spec.split_column,
        requested_outcome,
    )
    missing_columns = sorted(set(required_columns).difference(available_columns))

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Input parquet is missing columns: {missing_text}")

    LOGGER.info("Loading training data from %s", parquet_path)
    dataframe = pd.read_parquet(parquet_path, columns=list(required_columns))
    validate_row_id_column(
        dataframe,
        row_id_column=dataset_spec.row_id_column,
        context="Training dataframe",
    )
    return dataframe, requested_outcome


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
    project_root = get_project_root(Path(__file__))
    config = load_yaml_config(config_path)

    data_config = get_config_section(config, "data")
    training_config = get_config_section(config, "training")
    model_config = get_config_section(config, "model")
    output_config = get_config_section(config, "outputs")
    tracking_config = get_tracking_config(config)
    debug_config = get_debug_config(config)
    prediction_splits = get_prediction_splits(output_config)

    dataset_spec = resolve_dataset_spec(data_config)
    dataset_name = dataset_spec.name
    validate_outcome(outcome, dataset_spec)
    feature_columns = dataset_spec.feature_columns
    treatment_column = dataset_spec.treatment_column
    split_column = dataset_spec.split_column
    train_split = str(training_config.get("train_split", "train"))
    validation_split = str(
        training_config.get("validation_split", "validation")
    )
    test_split = str(training_config.get("test_split", "test"))
    prediction_batch_size = int(training_config["prediction_batch_size"])
    early_stopping_rounds = int(training_config["early_stopping_rounds"])
    log_evaluation_period = int(training_config["log_evaluation_period"])
    model_name = str(model_config["name"])
    artifact_model_name = get_artifact_model_name(model_name)
    model_params = dict(model_config["params"])

    data_path = get_processed_data_path(data_config, outcome, project_root)
    prediction_dir = resolve_project_path(
        output_config["prediction_dir"],
        project_root,
    )
    metric_dir = resolve_project_path(
        output_config["metric_dir"],
        project_root,
    )
    run_number = find_next_run_number(
        artifact_dirs=(prediction_dir, metric_dir),
        db_name=dataset_name,
        outcome=outcome,
        model_name=artifact_model_name,
    )
    prediction_path = resolve_project_path(
        prediction_dir
        / build_artifact_filename(
            db_name=dataset_name,
            outcome=outcome,
            model_name=artifact_model_name,
            run_number=run_number,
            artifact_name="predictions",
            extension="parquet",
        ),
        project_root,
    )
    metrics_path = resolve_project_path(
        metric_dir
        / build_artifact_filename(
            db_name=dataset_name,
            outcome=outcome,
            model_name=artifact_model_name,
            run_number=run_number,
            artifact_name="metrics",
            extension="json",
        ),
        project_root,
    )

    dataframe, target_column = load_training_frame(
        parquet_path=data_path,
        dataset_spec=dataset_spec,
        requested_outcome=outcome,
    )

    dataframe = apply_debug_sample(
        dataframe=dataframe,
        sample_rows=debug_config["sample_rows"],
        random_state=debug_config["random_state"],
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
    treated_train_frame = train_frame.loc[
        train_frame[treatment_column] == 1
    ].copy()
    treated_validation_frame = validation_frame.loc[
        validation_frame[treatment_column] == 1
    ].copy()

    assert len(treated_train_frame) > 0, (
        f"Split '{train_split}' has no treated rows for response-model fitting."
    )
    assert len(treated_validation_frame) > 0, (
        f"Split '{validation_split}' has no treated rows for early stopping."
    )
    assert treatment_column not in feature_columns

    treatment_fit_ratio = len(treated_train_frame) / len(train_frame)

    LOGGER.info("Full training rows: %s", len(train_frame))
    LOGGER.info(
        "Treated training rows used for fitting: %s",
        len(treated_train_frame),
    )
    LOGGER.info("Treatment ratio used for fitting: %.6f", treatment_fit_ratio)
    LOGGER.info("Full validation rows: %s", len(validation_frame))
    LOGGER.info(
        "Treated validation rows used for early stopping: %s",
        len(treated_validation_frame),
    )
    LOGGER.info("Full test rows: %s", len(test_frame))

    LOGGER.info("Training %s for %s", model_name, outcome)
    model = build_response_model(model_params)
    fit_response_model(
        model=model,
        X_train=treated_train_frame.loc[:, feature_columns],
        y_train=treated_train_frame[target_column],
        X_valid=treated_validation_frame.loc[:, feature_columns],
        y_valid=treated_validation_frame[target_column],
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
        "debug_sample_rows": (
            "full"
            if debug_config["sample_rows"] is None
            else int(debug_config["sample_rows"])
        ),
        "debug_random_state": int(debug_config["random_state"]),
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
