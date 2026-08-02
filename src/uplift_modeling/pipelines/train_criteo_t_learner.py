"""Train the Criteo LightGBM T-Learner for visit or conversion."""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd

from uplift_modeling.artifacts.model_provenance import (
    build_model_provenance_payload,
    get_model_provenance_path,
    save_model_provenance_payload,
)
from uplift_modeling.artifacts.naming import (
    build_artifact_filename,
    find_next_run_number,
    get_artifact_model_name,
)
from uplift_modeling.artifacts.predictions import (
    save_prediction_parquet_in_batches,
)
from uplift_modeling.models.scoring import T_LEARNER_MODEL_KIND
from uplift_modeling.models.t_learner import (
    fit_t_learner,
    predict_t_learner_scores,
)
from uplift_modeling.pipelines.train_criteo_response_model import (
    apply_debug_sample,
    get_debug_config,
    get_prediction_splits,
    get_processed_data_path,
    get_split_frame,
    get_tracking_config,
    load_training_frame,
    resolve_dataset_spec,
    validate_outcome,
    get_training_splits,
)
from uplift_modeling.tracking.mlflow_tracking import setup_mlflow
from uplift_modeling.utils.config import (
    get_config_section,
    get_project_root,
    load_yaml_config,
    resolve_project_path,
)


LOGGER = logging.getLogger(__name__)
TREATMENT_VALUE = 1
CONTROL_VALUE = 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Criteo T-Learner training."""
    parser = argparse.ArgumentParser(
        description="Train a Criteo LightGBM T-Learner."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the T-Learner YAML config.",
    )
    parser.add_argument(
        "--outcome",
        default="visit",
        help="Outcome to model. Defaults to visit.",
    )
    return parser.parse_args()


def get_treatment_group(
    dataframe: pd.DataFrame,
    treatment_column: str,
    treatment_value: int,
    split_name: str,
) -> pd.DataFrame:
    """Return treatment or control rows for one split."""
    group = dataframe.loc[dataframe[treatment_column] == treatment_value]

    if group.empty:
        label = "treatment" if treatment_value == TREATMENT_VALUE else "control"
        raise ValueError(f"Split '{split_name}' has no {label} rows.")

    return group


def train_t_learner_pipeline(config_path: Path, outcome: str) -> None:
    """Run the configured Criteo T-Learner training pipeline."""
    project_root = get_project_root(Path(__file__))
    config = load_yaml_config(config_path)

    data_config = get_config_section(config, "data")
    training_config = get_config_section(config, "training")
    model_config = get_config_section(config, "model")
    output_config = get_config_section(config, "outputs")
    tracking_config = get_tracking_config(config)
    debug_config = get_debug_config(config)
    get_prediction_splits(output_config)

    dataset_spec = resolve_dataset_spec(data_config)
    dataset_name = dataset_spec.name
    validate_outcome(outcome, dataset_spec)
    feature_columns = dataset_spec.feature_columns
    treatment_column = dataset_spec.treatment_column
    split_column = dataset_spec.split_column
    train_split, validation_split = get_training_splits(
        training_config
    )
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
    treatment_model, control_model = fit_t_learner(
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
        dataframes=(validation_frame,),
        output_path=prediction_path,
        feature_columns=feature_columns,
        treatment_column=treatment_column,
        split_column=split_column,
        outcome_column=target_column,
        model_name=model_name,
        batch_size=prediction_batch_size,
        score_batch=lambda X_batch: predict_t_learner_scores(
            treatment_model,
            control_model,
            X_batch,
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
        "treatment_train_rows": int(len(treatment_train)),
        "control_train_rows": int(len(control_train)),
        **model_params,
    }
    mlflow_metrics = {
        "treatment_train_positive_rate": float(
            treatment_train[target_column].mean()
        ),
        "control_train_positive_rate": float(
            control_train[target_column].mean()
        ),
        "treatment_validation_positive_rate": float(
            treatment_valid[target_column].mean()
        ),
        "control_validation_positive_rate": float(
            control_valid[target_column].mean()
        ),
    }

    with mlflow.start_run(run_name=f"{outcome}_{model_name}") as run:
        mlflow.log_params(mlflow_params)
        mlflow.log_metrics(mlflow_metrics)

        if bool(tracking_config["log_predictions"]):
            mlflow.log_artifact(str(prediction_path))

        mlflow.lightgbm.log_model(
            treatment_model,
            artifact_path="treatment_model",
        )
        mlflow.lightgbm.log_model(
            control_model,
            artifact_path="control_model",
        )

    provenance_path = get_model_provenance_path(prediction_path)
    save_model_provenance_payload(
        build_model_provenance_payload(
            dataset_name=dataset_name,
            outcome=outcome,
            policy_name=model_name,
            prediction_path=prediction_path,
            model_kind=T_LEARNER_MODEL_KIND,
            mlflow_run_id=run.info.run_id,
            model_artifacts={
                "treatment_model_uri": (
                    f"runs:/{run.info.run_id}/treatment_model"
                ),
                "control_model_uri": f"runs:/{run.info.run_id}/control_model",
            },
        ),
        provenance_path,
    )

    LOGGER.info("Saved T-Learner predictions to %s", prediction_path)
    LOGGER.info("Saved T-Learner model provenance to %s", provenance_path)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    project_root = get_project_root(Path(__file__))
    config_path = resolve_project_path(args.config, project_root)
    train_t_learner_pipeline(config_path=config_path, outcome=args.outcome)


if __name__ == "__main__":
    main()
