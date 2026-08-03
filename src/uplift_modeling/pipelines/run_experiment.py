import argparse
import logging
from pathlib import Path

from uplift_modeling.evaluation.bootstrap_config import DEFAULT_N_BOOTSTRAP
from uplift_modeling.pipelines.create_experiment_manifest import (
    create_experiment_manifest,
)
from uplift_modeling.pipelines.evaluate_criteo_predictions import (
    evaluate_predictions,
)
from uplift_modeling.pipelines.train_criteo_response_model import (
    train_response_pipeline,
)
from uplift_modeling.pipelines.train_criteo_t_learner import (
    train_t_learner_pipeline,
)
from uplift_modeling.pipelines.train_criteo_x_learner import (
    train_x_learner_pipeline,
)
from uplift_modeling.utils.config import (
    get_project_root,
    load_yaml_config,
    resolve_project_path,
)

LOGGER = logging.getLogger(__name__)

SHARED_CONFIG_SECTIONS = (
    "project",
    "data",
    "training",
    "outputs",
    "selection",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train candidates, create an exact experiment manifest, "
            "evaluate validation predictions, and select a champion."
        )
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Unique immutable experiment identifier.",
    )
    parser.add_argument(
        "--outcome",
        choices=("visit", "conversion"),
        default= "visit",
        help="Outcome to model and evaluate.",
    )
    parser.add_argument(
        "--response-config",
        default="configs/modeling/criteo_response_lgbm.yaml",
    )
    parser.add_argument(
        "--t-learner-config",
        default="configs/modeling/t_learner.yaml",
    )
    parser.add_argument(
        "--x-learner-config",
        default="configs/modeling/x_learner.yaml",
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


def validate_shared_config_sections(config_paths: tuple[Path, ...],) -> None:
    """Require candidate configs to describe the same experiment context."""
    configs = [(config_path, load_yaml_config(config_path)) for config_path in config_paths]
    reference_path, reference_config = configs[0]

    for config_path, config in configs[1:]:
        mismatched_sections = [
            section
            for section in SHARED_CONFIG_SECTIONS
            if config.get(section) != reference_config.get(section)
        ]

        if mismatched_sections:
            raise ValueError(
                "Candidate configs must use identical shared experiment"
                f"sections. Reference: {reference_path};"
                f"mismatched config: {config_path};"
                f"sections: {mismatched_sections}."
            )


def run_experiment(
    experiment_id: str,
    outcome: str,
    response_config_path: Path,
    t_learner_config_path: Path,
    x_learner_config_path: Path,
    top_fraction: float = 0.3,
    curve_num_points: int = 100,
    random_seed: int = 42,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> tuple[Path, Path]:
    """Train candidates and complete validation-stage champion selection."""
    config_paths = (
        response_config_path,
        t_learner_config_path,
        x_learner_config_path,
    )
    validate_shared_config_sections(config_paths)

    training_results = (
        train_response_pipeline(
            config_path=response_config_path,
            outcome=outcome,
        ),
        train_t_learner_pipeline(
            config_path=t_learner_config_path,
            outcome=outcome,
        ),
        train_x_learner_pipeline(
            config_path=x_learner_config_path,
            outcome=outcome,
        ),
    )

    prediction_artifacts = dict(training_results)

    if len(prediction_artifacts) != len(training_results):
        raise ValueError(
            "Candidate training pipelines returned duplicate policy names."
        )

    manifest_path = create_experiment_manifest(
        config_path=response_config_path,
        outcome=outcome,
        experiment_id=experiment_id,
        prediction_artifacts=prediction_artifacts,
    )

    selection_artifact_path = evaluate_predictions(
        config_path=response_config_path,
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

    manifest_path, selection_artifact_path = run_experiment(
        experiment_id=args.experiment_id,
        outcome=args.outcome,
        response_config_path=resolve_project_path(args.response_config, project_root),
        t_learner_config_path=resolve_project_path(args.t_learner_config, project_root),
        x_learner_config_path=resolve_project_path(args.x_learner_config, project_root),
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

if __name__ == "__main__":
    main()
