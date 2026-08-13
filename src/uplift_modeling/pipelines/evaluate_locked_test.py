"""Run locked test-set scoring for the selected uplift policy and response baseline."""

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from uplift_modeling.artifacts.manifest import (
    load_experiment_manifest,
    resolve_manifest_config_paths,
    resolve_prediction_paths,
    resolve_model_artifacts,
    validate_model_artifacts_match_predictions,
)
from uplift_modeling.artifacts.model_provenance import (
    get_model_provenance_path,
    save_model_provenance_payload,
)
from uplift_modeling.artifacts.predictions import save_prediction_parquet_in_batches
from uplift_modeling.data.dataset_spec import (
    DatasetSpec,
    validate_supported_outcome,
    load_dataset_config,
)
from uplift_modeling.evaluation.bootstrap_config import (
    DEFAULT_BOOTSTRAP_RANDOM_SEED,
    DEFAULT_N_BOOTSTRAP,
)
from uplift_modeling.evaluation.locked_test import (
    _save_locked_test_evaluation,
    build_locked_test_evaluation_path,
    build_locked_test_prediction_path,
    load_selection_gate_payload,
    load_existing_locked_test_evaluation,
    validate_selection_gate_payload,
)
from uplift_modeling.models.scoring import build_policy_score_batch
from uplift_modeling.pipelines.train_response_model import (
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
        description=(
            "Score and evaluate the validation-selected uplift policy "
            "and response baseline on the locked test split."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the validation experiment manifest JSON artifact.",
    )
    parser.add_argument(
        "--selection-artifact",
        required=True,
        help="Path to the model-selection gate JSON artifact.",
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


def get_experiment_name(config: dict[str, Any]) -> str:
    """Return the MLflow experiment name used by model training."""
    project_config = config.get("project", {})
    if project_config is None:
        project_config = {}
    if not isinstance(project_config, dict):
        raise ValueError("Config section 'project' must be a mapping.")

    return str(project_config.get("experiment_name", "uplift-modeling"))


def validate_selection_source_manifest(
    selection_payload: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Validate Selection Gate artifact was built from the supplied manifest."""
    source_manifest_path = str(selection_payload["source_manifest_path"])
    if Path(source_manifest_path).resolve() != manifest_path.resolve():
        raise ValueError(
            "Selection artifact does not reference the supplied experiment "
            "manifest."
        )


def load_locked_test_source_frame(
    parquet_path: Path,
    dataset_spec: DatasetSpec,
    outcome: str,
) -> tuple[pd.DataFrame, str]:
    """Load the existing fixed test split without resplitting."""
    dataframe, target_column = load_training_frame(
        parquet_path=parquet_path,
        dataset_spec=dataset_spec,
        requested_outcome=outcome,
    )

    test_frame = get_split_frame(
        dataframe=dataframe,
        split_column=dataset_spec.split_column,
        split_value=LOCKED_TEST_SPLIT,
    )

    observed_splits = {
        str(split)
        for split in test_frame[
            dataset_spec.split_column
        ].unique()
    }

    if observed_splits != {LOCKED_TEST_SPLIT}:
        raise ValueError(
            "Locked-test source frame must contain only split "
            f"'{LOCKED_TEST_SPLIT}'. "
            f"Received: {sorted(observed_splits)}."
        )

    return test_frame, target_column


def save_locked_test_predictions(
    test_frame: pd.DataFrame,
    dataset_spec: DatasetSpec,
    target_column: str,
    dataset_name: str,
    outcome: str,
    experiment_id: str,
    model_artifacts: Mapping[str, Mapping[str, Any]],
    prediction_dir: Path,
    batch_size: int,
) -> dict[str, Path]:
    """Score test rows for required policies and save test-only parquets."""
    prediction_paths: dict[str, Path] = {}
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for policy_name in sorted(model_artifacts):
        prediction_path = build_locked_test_prediction_path(
            prediction_dir=prediction_dir,
            dataset_name=dataset_name,
            outcome=outcome,
            policy_name=policy_name,
            experiment_id=experiment_id,
        )
        if prediction_path.exists():
            raise ValueError(
                "Incomplete Final Evaluation state: prediction exists "
                "without final report."
            )
        score_batch = build_policy_score_batch(
            policy=policy_name,
            model_artifact=model_artifacts[policy_name],
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
    manifest_path: Path,
    selection_artifact_path: Path,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = DEFAULT_BOOTSTRAP_RANDOM_SEED,
) -> Path:
    """Score the exact validation champion on the fixed test split."""
    project_root = get_project_root(Path(__file__))
    manifest = load_experiment_manifest(manifest_path)

    dataset_name = str(manifest["dataset_name"])
    outcome = str(manifest["outcome"])
    experiment_id = manifest.get("experiment_id")

    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError(
            "Experiment manifest must contain a non-empty "
            "'experiment_id'."
        )

    dataset_config_path, modeling_config_path = (
        resolve_manifest_config_paths(
            manifest=manifest,
            manifest_path=manifest_path,
            project_root=project_root,
        )
    )

    dataset_config = load_dataset_config(
        dataset_config_path,
        project_root=project_root,
    )

    modeling_config = load_yaml_config(
        modeling_config_path,
    )

    dataset_spec = dataset_config.spec

    validate_supported_outcome(
        dataset_spec,
        outcome,
    )

    if dataset_spec.name != dataset_name:
        raise ValueError(
            "Manifest dataset_name does not match dataset config. "
            f"Manifest: '{dataset_name}', "
            f"dataset config: '{dataset_spec.name}'."
        )

    training_config = get_config_section(
        modeling_config,
        "training",
    )
    output_config = get_config_section(
        modeling_config,
        "outputs",
    )

    selection_payload = load_selection_gate_payload(
        selection_artifact_path
    )

    uplift_champion_policy, baseline_policy = validate_selection_gate_payload(
        selection_payload=selection_payload,
        outcome=outcome,
    )

    validate_selection_source_manifest(
        selection_payload=selection_payload,
        manifest_path=manifest_path,
    )

    # Selection and manifest must belong to the same experiment.
    if selection_payload.get("experiment_id") != experiment_id:
        raise ValueError(
            "Selection Gate artifact and experiment manifest "
            "must have the same experiment_id."
        )

    if selection_payload.get("dataset_name") != dataset_name:
        raise ValueError(
            "Selection Gate artifact dataset_name does not "
            "match experiment manifest."
        )

    uplift_champion_model_artifact = selection_payload.get(
        "uplift_champion_model_artifact"
    )

    if not isinstance(uplift_champion_model_artifact, dict):
        raise ValueError(
            "Selection Gate artifact must contain "
            "'uplift_champion_model_artifact'."
        )

    if (
        uplift_champion_model_artifact.get("policy_name")
        != uplift_champion_policy
    ):
        raise ValueError(
            "Uplift champion model provenance policy_name does not "
            "match uplift_champion_policy."
        )

    if uplift_champion_model_artifact.get("dataset_name") != dataset_name:
        raise ValueError(
            "Uplift champion model provenance dataset_name does not "
            "match experiment dataset."
        )

    if uplift_champion_model_artifact.get("outcome") != outcome:
        raise ValueError(
            "Uplift champion model provenance outcome does not "
            "match experiment outcome."
        )

    required_policies = (
        uplift_champion_policy,
        baseline_policy,
    )

    model_artifacts = resolve_model_artifacts(
        manifest=manifest,
        required_policies=required_policies,
    )

    manifest_uplift_champion_artifact = model_artifacts[
        uplift_champion_policy
    ]

    if uplift_champion_model_artifact != manifest_uplift_champion_artifact:
        raise ValueError(
            "Selection Gate uplift_champion_model_artifact does not "
            "match the exact model recorded in the experiment manifest."
        )

    validation_prediction_paths = resolve_prediction_paths(
        manifest=manifest,
        manifest_path=manifest_path,
        dataset_name=dataset_name,
        outcome=outcome,
        project_root=project_root,
        required_policies=required_policies,
    )

    validate_model_artifacts_match_predictions(
        model_artifacts=model_artifacts,
        prediction_paths=validation_prediction_paths,
    )

    metric_dir = resolve_project_path(
        output_config["metric_dir"],
        project_root,
    )

    prediction_dir = resolve_project_path(
        output_config["prediction_dir"],
        project_root,
    )

    final_evaluation_path = build_locked_test_evaluation_path(
        metric_dir=metric_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        experiment_id=experiment_id,
    )

    if final_evaluation_path.exists():
        output_path, payload = load_existing_locked_test_evaluation(
            output_path=final_evaluation_path,
            experiment_id=experiment_id,
            uplift_champion_policy=uplift_champion_policy,
            baseline_policy=baseline_policy,
            uplift_champion_model_artifact=uplift_champion_model_artifact,
        )

        LOGGER.info(
            "Reusing locked-test evaluation JSON: %s",
            output_path,
        )
        LOGGER.info(
            "Locked-test evaluation result:\n%s",
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            ),
        )

        return output_path

    processed_data_path = (
        dataset_config.processed_paths[outcome]
    )

    test_frame, target_column = load_locked_test_source_frame(
        parquet_path=processed_data_path,
        dataset_spec=dataset_spec,
        outcome=outcome,
    )
    setup_mlflow(
        get_experiment_name(modeling_config)
    )

    locked_test_prediction_paths = save_locked_test_predictions(
        test_frame=test_frame,
        dataset_spec=dataset_spec,
        target_column=target_column,
        dataset_name=dataset_name,
        outcome=outcome,
        experiment_id=experiment_id,
        model_artifacts=model_artifacts,
        prediction_dir=prediction_dir,
        batch_size=int(
            training_config["prediction_batch_size"]
        ),
    )
    output_path, _ = _save_locked_test_evaluation(
        manifest_prediction_paths=locked_test_prediction_paths,
        metric_dir=metric_dir,
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
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s - %(message)s"
        ),
    )

    args = parse_args()
    project_root = get_project_root(Path(__file__))

    manifest_path = resolve_project_path(
        args.manifest,
        project_root,
    )

    selection_artifact_path = resolve_project_path(
        args.selection_artifact,
        project_root,
    )

    output_path = evaluate_locked_test(
        manifest_path=manifest_path,
        selection_artifact_path=selection_artifact_path,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
    )

    LOGGER.info(
        "Locked-test evaluation artifact: %s",
        output_path,
    )


if __name__ == "__main__":
    main()
