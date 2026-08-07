import argparse
import logging
from pathlib import Path

from uplift_modeling.evaluation.bootstrap_config import DEFAULT_N_BOOTSTRAP
from uplift_modeling.pipelines.create_experiment_manifest import (
    create_experiment_manifest,
)
from uplift_modeling.pipelines.evaluate_predictions import (
    evaluate_predictions,
)
from uplift_modeling.pipelines.train_response_model import (
    train_response_pipeline,
)
from uplift_modeling.pipelines.train_t_learner import (
    train_t_learner_pipeline,
)
from uplift_modeling.pipelines.train_x_learner import (
    train_x_learner_pipeline,
)
from uplift_modeling.utils.config import (
    get_project_root,
    load_yaml_config,
    resolve_project_path,
)
from uplift_modeling.data.dataset_spec import(
    load_dataset_config,
    validate_supported_outcome
)
from uplift_modeling.models.config import resolve_model_candidates
from uplift_modeling.models.scoring import (
    RESPONSE_MODEL_KIND,
    T_LEARNER_MODEL_KIND,
    X_LEARNER_MODEL_KIND,
)
TRAIN_PIPELINES = {
    RESPONSE_MODEL_KIND: train_response_pipeline,
    T_LEARNER_MODEL_KIND: train_t_learner_pipeline,
    X_LEARNER_MODEL_KIND: train_x_learner_pipeline,
}
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train configured candidates, create an experiment manifest,"
            "evaluate validation predictions, and select a champion."
        )
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
        "--experiment-id",
        required=True,
        help="Unique immutable experiment identifier.",
    )
    parser.add_argument(
        "--outcome",
        required=True,
        help="Outcome column to train and evaluate.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--curve-num-points",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
    )
    return parser.parse_args()


def train_configured_candidates(
    dataset_config_path: Path,
    modeling_config_path: Path,
    modeling_config: dict,
    outcome: str,
) -> dict[str, Path]:
    """Train every candidate declared in the shared modeling config."""
    candidates = resolve_model_candidates(modeling_config)

    prediction_artifacts: dict[str, Path] = {}

    for candidate in candidates:
        train_pipeline = TRAIN_PIPELINES.get(candidate.kind)

        if train_pipeline is None:
            raise ValueError(
                f"No training pipeline registered for model kind "
                f"'{candidate.kind}'."
            )

        model_name, prediction_path = train_pipeline(
            dataset_config_path=dataset_config_path,
            modeling_config_path=modeling_config_path,
            outcome=outcome,
        )

        if model_name != candidate.name:
            raise ValueError(
                f"Training pipeline for kind '{candidate.kind}' returned "
                f"model name '{model_name}', but modeling config declares "
                f"'{candidate.name}'."
            )

        if model_name in prediction_artifacts:
            raise ValueError(
                f"Duplicate prediction artifact policy name: {model_name}"
            )

        prediction_artifacts[model_name] = prediction_path

    return prediction_artifacts


def run_experiment(
    dataset_config_path: Path,
    modeling_config_path: Path,
    experiment_id: str,
    outcome: str,
    top_fraction: float = 0.3,
    curve_num_points: int = 100,
    random_seed: int = 42,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> tuple[Path, Path]:
    """Train candidates and complete validation-stage champion selection."""
    project_root = get_project_root(Path(__file__))

    dataset_config = load_dataset_config(
        dataset_config_path,
        project_root=project_root,
    )

    validate_supported_outcome(
        dataset_config.spec,
        outcome,
    )

    modeling_config = load_yaml_config(
        modeling_config_path,
    )

    prediction_artifacts = train_configured_candidates(
        dataset_config_path=dataset_config_path,
        modeling_config_path=modeling_config_path,
        modeling_config=modeling_config,
        outcome=outcome,
    )

    manifest_path = create_experiment_manifest(
        dataset_config_path=dataset_config_path,
        modeling_config_path=modeling_config_path,
        outcome=outcome,
        experiment_id=experiment_id,
        prediction_artifacts=prediction_artifacts,
    )

    selection_artifact_path = evaluate_predictions(
        dataset_config_path=dataset_config_path,
        modeling_config_path=modeling_config_path,
        manifest_path=manifest_path,
        outcome=outcome,
        top_fraction=top_fraction,
        curve_num_points=curve_num_points,
        random_seed=random_seed,
        n_bootstrap=n_bootstrap,
        topk_only=False,
        skip_bootstrap=False,
    )

    if selection_artifact_path is None:
        raise RuntimeError(
            "Validation experiment completed without a Selection Gate artifact."
        )

    return manifest_path, selection_artifact_path

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

    manifest_path, selection_artifact_path = run_experiment(
        dataset_config_path=dataset_config_path,
        modeling_config_path=modeling_config_path,
        experiment_id=args.experiment_id,
        outcome=args.outcome,
        top_fraction=args.top_fraction,
        curve_num_points=args.curve_num_points,
        random_seed=args.random_seed,
        n_bootstrap=args.n_bootstrap,
    )

    LOGGER.info("Experiment manifest: %s", manifest_path)
    LOGGER.info("Selection Gate artifact: %s", selection_artifact_path)
    LOGGER.info(
        "Locked test was not run. Finalize it separately after reviewing "
        "the locked champion."
    )