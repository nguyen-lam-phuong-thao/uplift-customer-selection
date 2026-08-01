"""Run locked test-set scoring and reporting for a fixed champion policy."""

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from uplift_modeling.artifacts.manifest import (
    load_experiment_manifest,
    resolve_model_artifacts,
    resolve_prediction_paths,
    validate_model_artifacts_match_predictions,
)
from uplift_modeling.artifacts.model_provenance import (
    get_model_provenance_path,
    save_model_provenance_payload,
)
from uplift_modeling.artifacts.naming import (
    build_artifact_filename,
    find_next_run_number,
)
from uplift_modeling.artifacts.predictions import save_prediction_parquet_in_batches
from uplift_modeling.data.dataset_spec import (
    DatasetSpec,
    get_dataset_spec,
    validate_supported_outcome,
)
from uplift_modeling.evaluation.bootstrap_config import (
    DEFAULT_BOOTSTRAP_RANDOM_SEED,
    DEFAULT_N_BOOTSTRAP,
)
from uplift_modeling.evaluation.locked_test import (
    get_champion_policy,
    load_selection_gate_payload,
    save_locked_test_evaluation,
)
from uplift_modeling.models.scoring import build_policy_score_batch
from uplift_modeling.pipelines.train_criteo_response_model import (
    get_processed_data_path,
    get_split_frame,
    load_training_frame,
)
from uplift_modeling.tracking.mlflow_tracking import setup_mlflow
from uplift_modeling.utils.config import (
    get_config_section,
    get_project_root,
    load_yaml_config,
    resolve_project_path,
)


LOGGER = logging.getLogger(__name__)
LOCKED_TEST_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for locked-test evaluation."""
    parser = argparse.ArgumentParser(
        description="Score and evaluate the fixed Selection Gate champion on test."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config.",
    )
    parser.add_argument(
        "--selection-artifact",
        required=True,
        help="Path to the model-selection gate JSON artifact.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the validation experiment manifest JSON artifact.",
    )
    parser.add_argument(
        "--outcome",
        default="visit",
        help="Outcome to evaluate. Defaults to visit.",
    )
    parser.add_argument(
        "--n-bootstrap",
        default=DEFAULT_N_BOOTSTRAP,
        type=int,
        help="Number of bootstrap resamples for locked-test Top-K CIs.",
    )
    parser.add_argument(
        "--random-seed",
        default=DEFAULT_BOOTSTRAP_RANDOM_SEED,
        type=int,
        help="Random seed for locked-test Top-K bootstrap CIs.",
    )
    return parser.parse_args()


def get_locked_test_split(config: dict[str, Any]) -> str:
    """Return and validate the locked-test split name."""
    locked_test_config = config.get("locked_test", {})
    if locked_test_config is None:
        locked_test_config = {}
    if not isinstance(locked_test_config, dict):
        raise ValueError("Config section 'locked_test' must be a mapping.")

    test_split = locked_test_config.get(
        "split",
        LOCKED_TEST_SPLIT,
    )
    if test_split != LOCKED_TEST_SPLIT:
        raise ValueError(
            "Locked-test evaluation may use the test split only. "
            f"Expected '{LOCKED_TEST_SPLIT}', received '{test_split}'."
        )

    return LOCKED_TEST_SPLIT


def get_experiment_name(config: dict[str, Any]) -> str:
    """Return the MLflow experiment name used by model training."""
    project_config = config.get("project", {})
    if project_config is None:
        project_config = {}
    if not isinstance(project_config, dict):
        raise ValueError("Config section 'project' must be a mapping.")

    return str(project_config.get("experiment_name", "criteo-uplift-modeling"))


def load_locked_test_source_frame(
    parquet_path: Path,
    dataset_spec: DatasetSpec,
    outcome: str,
    test_split: str = LOCKED_TEST_SPLIT,
) -> tuple[pd.DataFrame, str]:
    """Load the prepared dataset and return the existing fixed test rows."""
    dataframe, target_column = load_training_frame(
        parquet_path=parquet_path,
        dataset_spec=dataset_spec,
        requested_outcome=outcome,
    )
    test_frame = get_split_frame(
        dataframe=dataframe,
        split_column=dataset_spec.split_column,
        split_value=test_split,
    )

    observed_splits = {
        str(split) for split in test_frame[dataset_spec.split_column].unique()
    }
    if observed_splits != {test_split}:
        raise ValueError(
            "Locked-test source frame must contain only split "
            f"'{test_split}'. Received: {sorted(observed_splits)}."
        )

    return test_frame, target_column


def save_locked_test_predictions(
    test_frame: pd.DataFrame,
    dataset_spec: DatasetSpec,
    target_column: str,
    dataset_name: str,
    outcome: str,
    model_artifacts: Mapping[str, Mapping[str, Any]],
    prediction_dir: Path,
    batch_size: int,
) -> dict[str, Path]:
    """Score test rows for required policies and save test-only parquets."""
    prediction_paths: dict[str, Path] = {}
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for policy_name in sorted(model_artifacts):
        score_batch = build_policy_score_batch(
            policy=policy_name,
            model_artifact=model_artifacts[policy_name],
        )
        run_number = find_next_run_number(
            artifact_dirs=(prediction_dir,),
            db_name=dataset_name,
            outcome=outcome,
            model_name=policy_name,
        )
        prediction_path = prediction_dir / build_artifact_filename(
            db_name=dataset_name,
            outcome=outcome,
            model_name=policy_name,
            run_number=run_number,
            artifact_name="locked_test_predictions",
            extension="parquet",
        )

        LOGGER.info(
            "Scoring locked-test split for %s and saving to %s",
            policy_name,
            prediction_path,
        )
        save_prediction_parquet_in_batches(
            dataframes=(test_frame,),
            output_path=prediction_path,
            feature_columns=dataset_spec.feature_columns,
            treatment_column=dataset_spec.treatment_column,
            split_column=dataset_spec.split_column,
            outcome_column=target_column,
            model_name=policy_name,
            batch_size=batch_size,
            score_batch=score_batch,
            row_id_column=dataset_spec.row_id_column,
        )
        save_locked_test_model_provenance(
            model_artifact=model_artifacts[policy_name],
            prediction_path=prediction_path,
        )
        prediction_paths[policy_name] = prediction_path

    return prediction_paths


def save_locked_test_model_provenance(
    model_artifact: Mapping[str, Any],
    prediction_path: Path,
) -> Path:
    """Save model provenance sidecar for one locked-test prediction artifact."""
    payload = dict(model_artifact)
    payload["source_prediction_artifact"] = payload["prediction_artifact"]
    payload["prediction_artifact"] = prediction_path.name
    provenance_path = get_model_provenance_path(prediction_path)
    return save_model_provenance_payload(payload, provenance_path)


def evaluate_locked_test(
    config_path: Path,
    manifest_path: Path,
    selection_artifact_path: Path,
    outcome: str,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = DEFAULT_BOOTSTRAP_RANDOM_SEED,
) -> Path:
    """Score the fixed champion on test, then save reporting."""
    project_root = get_project_root(Path(__file__))
    config = load_yaml_config(config_path)
    data_config = get_config_section(config, "data")
    training_config = get_config_section(config, "training")
    output_config = get_config_section(config, "outputs")
    dataset_spec = get_dataset_spec(str(data_config["dataset_name"]))
    dataset_name = dataset_spec.name
    validate_supported_outcome(dataset_spec, outcome)

    selection_payload = load_selection_gate_payload(selection_artifact_path)
    champion_policy = get_champion_policy(selection_payload)
    required_policies = (champion_policy,)

    manifest = load_experiment_manifest(manifest_path)
    validation_prediction_paths = resolve_prediction_paths(
        manifest=manifest,
        manifest_path=manifest_path,
        dataset_name=dataset_name,
        outcome=outcome,
        project_root=project_root,
        required_policies=required_policies,
    )
    model_artifacts = resolve_model_artifacts(
        manifest=manifest,
        required_policies=required_policies,
    )
    validate_model_artifacts_match_predictions(
        model_artifacts=model_artifacts,
        prediction_paths=validation_prediction_paths,
    )

    processed_data_path = get_processed_data_path(
        data_config=data_config,
        outcome=outcome,
        project_root=project_root,
    )
    test_frame, target_column = load_locked_test_source_frame(
        parquet_path=processed_data_path,
        dataset_spec=dataset_spec,
        outcome=outcome,
        test_split=get_locked_test_split(config),
    )

    setup_mlflow(get_experiment_name(config))
    locked_test_prediction_paths = save_locked_test_predictions(
        test_frame=test_frame,
        dataset_spec=dataset_spec,
        target_column=target_column,
        dataset_name=dataset_name,
        outcome=outcome,
        model_artifacts=model_artifacts,
        prediction_dir=resolve_project_path(
            output_config["prediction_dir"],
            project_root,
        ),
        batch_size=int(training_config["prediction_batch_size"]),
    )

    output_path, _ = save_locked_test_evaluation(
        manifest_prediction_paths=locked_test_prediction_paths,
        metric_dir=resolve_project_path(output_config["metric_dir"], project_root),
        dataset_name=dataset_name,
        outcome=outcome,
        selection_artifact_path=selection_artifact_path,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    return output_path


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    project_root = get_project_root(Path(__file__))
    config_path = resolve_project_path(args.config, project_root)
    manifest_path = resolve_project_path(args.manifest, project_root)
    selection_artifact_path = resolve_project_path(
        args.selection_artifact,
        project_root,
    )
    output_path = evaluate_locked_test(
        config_path=config_path,
        manifest_path=manifest_path,
        selection_artifact_path=selection_artifact_path,
        outcome=args.outcome,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
    )
    LOGGER.info("Locked-test evaluation artifact: %s", output_path)


if __name__ == "__main__":
    main()
