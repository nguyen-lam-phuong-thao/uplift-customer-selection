"""Train the LightGBM X-Learner for visit or conversion."""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.lightgbm

from uplift_modeling.artifacts.model_provenance import (
    build_model_provenance_payload,
    get_model_provenance_path,
    save_model_provenance_payload,
)
from uplift_modeling.artifacts.naming import (
    build_artifact_filename,
    find_next_run_number,
)
from uplift_modeling.artifacts.predictions import (
    save_prediction_parquet_in_batches,
)
from uplift_modeling.models.scoring import X_LEARNER_MODEL_KIND
from uplift_modeling.models.x_learner import (
    fit_x_learner,
    predict_x_learner_scores,
    summarize_values,
)
from uplift_modeling.pipelines.train_response_model import (
    apply_debug_sample,
    filter_training_and_validation_splits,
    get_debug_config,
    get_split_frame,
    get_tracking_config,
    load_training_frame,
    validate_outcome,
    get_training_splits,
)
from uplift_modeling.pipelines.train_t_learner import (
    CONTROL_VALUE,
    TREATMENT_VALUE,
    get_treatment_group,
)
from uplift_modeling.data.dataset_spec import load_dataset_config
from uplift_modeling.models.config import (
    ModelCandidateConfig,
    resolve_model_candidate,
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
    """Parse command-line arguments for X-Learner training."""
    parser = argparse.ArgumentParser(
        description="Train a LightGBM X-Learner."
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        help="Path to the dataset YAML config.",
    )
    parser.add_argument(
        "--modeling-config",
        required=True,
        help="Path to the shared modeling YAML config.",
    )
    parser.add_argument(
        "--outcome",
        default="visit",
        help="Outcome to model. Defaults to visit.",
    )
    return parser.parse_args()


def train_x_learner_pipeline(
        dataset_config_path: Path, 
        modeling_config_path: Path, 
        outcome: str,
        model_candidate: ModelCandidateConfig,
) -> tuple[str, Path]:
    """Run the configured X-Learner training pipeline."""
    project_root = get_project_root(Path(__file__))

    dataset_config = load_dataset_config(
        dataset_config_path,
        project_root=project_root,
    )
    modeling_config = load_yaml_config(modeling_config_path)

    if model_candidate.kind != X_LEARNER_MODEL_KIND:
        raise ValueError(
            f"X-Learner trainer received candidate kind "
            f"'{model_candidate.kind}'."
        )

    training_config = get_config_section(
        modeling_config,
        "training",
    )
    output_config = get_config_section(
        modeling_config,
        "outputs",
    )
    tracking_config = get_tracking_config(modeling_config)
    debug_config = get_debug_config(modeling_config)

    dataset_spec = dataset_config.spec
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
    model_name = model_candidate.name
    artifact_model_name = model_name
    model_params = dict(model_candidate.params)

    data_path = dataset_config.processed_paths[outcome]
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
    dataframe = filter_training_and_validation_splits(
        dataframe=dataframe,
        split_column=split_column,
        train_split=train_split,
        validation_split=validation_split,
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

    LOGGER.info("Train rows: %s", len(train_frame))
    LOGGER.info("Validation rows: %s", len(validation_frame))
    LOGGER.info("Treatment train rows: %s", len(treatment_train))
    LOGGER.info("Control train rows: %s", len(control_train))
    LOGGER.info("Treatment validation rows: %s", len(treatment_valid))
    LOGGER.info("Control validation rows: %s", len(control_valid))
    LOGGER.info(
        "Treatment train positive rate: %.6f",
        treatment_train[target_column].mean(),
    )
    LOGGER.info(
        "Control train positive rate: %.6f",
        control_train[target_column].mean(),
    )
    LOGGER.info(
        "Treatment validation positive rate: %.6f",
        treatment_valid[target_column].mean(),
    )
    LOGGER.info(
        "Control validation positive rate: %.6f",
        control_valid[target_column].mean(),
    )

    LOGGER.info("Training %s for %s", model_name, outcome)
    x_learner_result = fit_x_learner(
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
    LOGGER.info(
        "Constant treatment-rate weight: %.6f",
        x_learner_result.constant_treatment_rate_weight,
    )
    LOGGER.info("D1 summary: %s", x_learner_result.treatment_effect_summary)
    LOGGER.info("D0 summary: %s", x_learner_result.control_effect_summary)

    validation_scores = predict_x_learner_scores(
        treatment_effect_model=x_learner_result.treatment_effect_model,
        control_effect_model=x_learner_result.control_effect_model,
        constant_treatment_rate_weight=(
            x_learner_result.constant_treatment_rate_weight
        ),
        features=validation_frame.loc[:, feature_columns],
    )
    validation_score_summary = summarize_values(validation_scores)
    LOGGER.info("Validation score summary: %s", validation_score_summary)

    save_prediction_parquet_in_batches(
        dataframes=(validation_frame,),
        output_path=prediction_path,
        feature_columns=feature_columns,
        treatment_column=treatment_column,
        split_column=split_column,
        outcome_column=target_column,
        model_name=model_name,
        batch_size=prediction_batch_size,
        score_batch=lambda X_batch: predict_x_learner_scores(
            treatment_effect_model=x_learner_result.treatment_effect_model,
            control_effect_model=x_learner_result.control_effect_model,
            constant_treatment_rate_weight=(
                x_learner_result.constant_treatment_rate_weight
            ),
            features=X_batch,
        ),
    )

    project_config = modeling_config.get("project", {})
    experiment_name = (
        project_config.get("experiment_name", "uplift-modeling")
        if isinstance(project_config, dict)
        else "uplift-modeling"
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
        "constant_treatment_rate_weight": float(
            x_learner_result.constant_treatment_rate_weight
        ),
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
        "constant_treatment_rate_weight": float(
            x_learner_result.constant_treatment_rate_weight
        ),
        **{
            f"d1_{key}": float(value)
            for key, value in x_learner_result.treatment_effect_summary.items()
        },
        **{
            f"d0_{key}": float(value)
            for key, value in x_learner_result.control_effect_summary.items()
        },
        **{
            f"validation_score_{key}": float(value)
            for key, value in validation_score_summary.items()
        },
    }

    with mlflow.start_run(run_name=f"{outcome}_{model_name}") as run:
        mlflow.log_params(mlflow_params)
        mlflow.log_metrics(mlflow_metrics)

        if bool(tracking_config["log_predictions"]):
            mlflow.log_artifact(str(prediction_path))

        treatment_effect_model_info = mlflow.lightgbm.log_model(
            x_learner_result.treatment_effect_model,
            name="tau1_model",
        )
        control_effect_model_info = mlflow.lightgbm.log_model(
            x_learner_result.control_effect_model,
            name="tau0_model",
        )

    provenance_path = get_model_provenance_path(prediction_path)
    save_model_provenance_payload(
        build_model_provenance_payload(
            dataset_name=dataset_name,
            outcome=outcome,
            policy_name=model_name,
            prediction_path=prediction_path,
            model_kind=X_LEARNER_MODEL_KIND,
            mlflow_run_id=run.info.run_id,
            model_artifacts={
                "treatment_effect_model_uri": (
                    treatment_effect_model_info.model_uri
                ),
                "control_effect_model_uri": (
                    control_effect_model_info.model_uri
                ),
                "constant_treatment_rate_weight": float(
                    x_learner_result.constant_treatment_rate_weight
                ),
            },
        ),
        provenance_path,
    )

    LOGGER.info("Saved X-Learner predictions to %s", prediction_path)
    LOGGER.info("Saved X-Learner model provenance to %s", provenance_path)
    return model_name, prediction_path

def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    project_root = get_project_root(Path(__file__))
    dataset_config_path = resolve_project_path(
        args.dataset_config,
        project_root,
    )
    modeling_config_path = resolve_project_path(
        args.modeling_config,
        project_root,
    )
    modeling_config = load_yaml_config(modeling_config_path)
    model_candidate = resolve_model_candidate(
        modeling_config,
        X_LEARNER_MODEL_KIND,
    )
    train_x_learner_pipeline(
        dataset_config_path=dataset_config_path,
        modeling_config_path=modeling_config_path,
        outcome=args.outcome,
        model_candidate=model_candidate,
    )


if __name__ == "__main__":
    main()
