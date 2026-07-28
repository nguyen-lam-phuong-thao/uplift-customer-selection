"""Train the Criteo LightGBM X-Learner for visit or conversion."""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.lightgbm

from uplift_modeling.artifacts.naming import (
    build_artifact_filename,
    find_next_run_number,
    get_artifact_model_name,
)
from uplift_modeling.artifacts.predictions import (
    save_prediction_parquet_in_batches,
)
from uplift_modeling.models.x_learner import (
    fit_x_learner,
    predict_x_learner_scores,
)
from uplift_modeling.pipelines.train_criteo_response_model import (
    VALID_OUTCOMES,
    apply_debug_sample,
    get_debug_config,
    get_prediction_splits,
    get_processed_data_path,
    get_split_frame,
    get_tracking_config,
    load_training_frame,
    validate_feature_columns,
    validate_outcome,
)
from uplift_modeling.pipelines.train_criteo_t_learner import (
    CONTROL_VALUE,
    TREATMENT_VALUE,
    get_treatment_group,
)
from uplift_modeling.tracking.mlflow_tracking import setup_mlflow
from uplift_modeling.utils.config import (
    get_config_section,
    get_project_root,
    load_yaml_config,
    resolve_project_path,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Criteo X-Learner training."""
    parser = argparse.ArgumentParser(
        description="Train a Criteo LightGBM X-Learner."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the X-Learner YAML config.",
    )
    parser.add_argument(
        "--outcome",
        default="visit",
        choices=VALID_OUTCOMES,
        help="Outcome to model. Defaults to visit.",
    )
    return parser.parse_args()


def train_x_learner_pipeline(config_path: Path, outcome: str) -> None:
    """Run the configured Criteo X-Learner training pipeline."""
    validate_outcome(outcome)
    project_root = get_project_root(Path(__file__))
    config = load_yaml_config(config_path)

    data_config = get_config_section(config, "data")
    training_config = get_config_section(config, "training")
    model_config = get_config_section(config, "model")
    output_config = get_config_section(config, "outputs")
    tracking_config = get_tracking_config(config)
    debug_config = get_debug_config(config)
    prediction_splits = get_prediction_splits(output_config)

    dataset_name = str(data_config["dataset_name"])
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
    artifact_model_name = get_artifact_model_name(model_name)
    model_params = dict(model_config["params"])

    data_path = get_processed_data_path(data_config, outcome, project_root)
    prediction_dir = resolve_project_path(
        output_config["prediction_dir"],
        project_root,
    )
    run_number = find_next_run_number(
        artifact_dirs=(prediction_dir,),
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

    dataframe, target_column = load_training_frame(
        parquet_path=data_path,
        feature_columns=feature_columns,
        treatment_column=treatment_column,
        split_column=split_column,
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

    treatment_train = get_treatment_group(
        train_frame,
        treatment_column,
        TREATMENT_VALUE,
        train_split,
    )
    control_train = get_treatment_group(
        train_frame,
        treatment_column,
        CONTROL_VALUE,
        train_split,
    )
    treatment_valid = get_treatment_group(
        validation_frame,
        treatment_column,
        TREATMENT_VALUE,
        validation_split,
    )
    control_valid = get_treatment_group(
        validation_frame,
        treatment_column,
        CONTROL_VALUE,
        validation_split,
    )

    LOGGER.info("Training %s for %s", model_name, outcome)
    (
        treatment_model,
        control_model,
        treatment_effect_model,
        control_effect_model,
        treatment_weight,
    ) = fit_x_learner(
        treatment_train=treatment_train,
        control_train=control_train,
        treatment_valid=treatment_valid,
        control_valid=control_valid,
        feature_columns=feature_columns,
        outcome_column=target_column,
        model_params=model_params,
        early_stopping_rounds=early_stopping_rounds,
        log_evaluation_period=log_evaluation_period,
    )

    save_prediction_parquet_in_batches(
        dataframes=tuple(split_frames[split] for split in prediction_splits),
        output_path=prediction_path,
        feature_columns=feature_columns,
        treatment_column=treatment_column,
        split_column=split_column,
        outcome_column=target_column,
        model_name=model_name,
        batch_size=prediction_batch_size,
        score_batch=lambda X_batch: predict_x_learner_scores(
            treatment_effect_model=treatment_effect_model,
            control_effect_model=control_effect_model,
            treatment_weight=treatment_weight,
            features=X_batch,
        ),
    )

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
        "target_column": target_column,
        "debug_sample_rows": (
            "full"
            if debug_config["sample_rows"] is None
            else int(debug_config["sample_rows"])
        ),
        "debug_random_state": int(debug_config["random_state"]),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "treatment_train_rows": int(len(treatment_train)),
        "control_train_rows": int(len(control_train)),
        "treatment_weight": float(treatment_weight),
        **model_params,
    }

    with mlflow.start_run(run_name=f"{outcome}_{model_name}"):
        mlflow.log_params(mlflow_params)

        if bool(tracking_config["log_predictions"]):
            mlflow.log_artifact(str(prediction_path))

        mlflow.lightgbm.log_model(
            treatment_model,
            artifact_path="mu1_model",
        )
        mlflow.lightgbm.log_model(
            control_model,
            artifact_path="mu0_model",
        )
        mlflow.lightgbm.log_model(
            treatment_effect_model,
            artifact_path="tau1_model",
        )
        mlflow.lightgbm.log_model(
            control_effect_model,
            artifact_path="tau0_model",
        )

    LOGGER.info("Saved X-Learner predictions to %s", prediction_path)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    project_root = get_project_root(Path(__file__))
    config_path = resolve_project_path(args.config, project_root)
    train_x_learner_pipeline(config_path=config_path, outcome=args.outcome)


if __name__ == "__main__":
    main()
