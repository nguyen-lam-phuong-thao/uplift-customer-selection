"""Run locked test-set reporting for a fixed champion policy."""

import argparse
import logging
from pathlib import Path
from typing import Any

from uplift_modeling.evaluation.bootstrap_config import DEFAULT_BASELINE_POLICY
from uplift_modeling.evaluation.locked_test import save_locked_test_evaluation
from uplift_modeling.utils.config import (
    get_config_section,
    get_project_root,
    load_yaml_config,
    resolve_project_path,
)


LOGGER = logging.getLogger(__name__)
VALID_OUTCOMES = ("visit", "conversion")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for locked-test evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed Selection Gate champion on test only."
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
        "--outcome",
        default="visit",
        choices=VALID_OUTCOMES,
        help="Outcome to evaluate. Defaults to visit.",
    )
    return parser.parse_args()


def get_locked_test_baseline_policy(config: dict[str, Any]) -> str:
    """Return the configured baseline policy for locked-test comparison."""
    selection_config = config.get("selection", {})
    if selection_config is None:
        selection_config = {}
    if not isinstance(selection_config, dict):
        raise ValueError(
            "Config section 'selection' must be a mapping when present."
        )

    return str(selection_config.get("baseline_policy", DEFAULT_BASELINE_POLICY))


def evaluate_locked_test(
    config_path: Path,
    selection_artifact_path: Path,
    outcome: str,
) -> Path:
    """Run locked-test reporting from existing prediction artifacts."""
    project_root = get_project_root(Path(__file__))
    config = load_yaml_config(config_path)
    data_config = get_config_section(config, "data")
    output_config = get_config_section(config, "outputs")
    dataset_name = str(data_config["dataset_name"])
    prediction_dir = resolve_project_path(
        output_config["prediction_dir"],
        project_root,
    )
    metric_dir = resolve_project_path(
        output_config["metric_dir"],
        project_root,
    )

    output_path, _ = save_locked_test_evaluation(
        prediction_dir=prediction_dir,
        metric_dir=metric_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        selection_artifact_path=selection_artifact_path,
        baseline_policy=get_locked_test_baseline_policy(config),
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
    selection_artifact_path = resolve_project_path(
        args.selection_artifact,
        project_root,
    )
    output_path = evaluate_locked_test(
        config_path=config_path,
        selection_artifact_path=selection_artifact_path,
        outcome=args.outcome,
    )
    LOGGER.info("Locked-test evaluation artifact: %s", output_path)


if __name__ == "__main__":
    main()
