"""Create an experiment manifest from run-numbered prediction artifacts."""

import argparse
import logging
from pathlib import Path

from uplift_modeling.artifacts.manifest import (
    build_experiment_manifest,
    build_experiment_manifest_model_artifacts,
    find_latest_prediction_artifacts,
    save_experiment_manifest,
)
from uplift_modeling.data.dataset_spec import (
    get_dataset_spec,
    validate_supported_outcome,
)
from uplift_modeling.utils.config import (
    get_config_section,
    get_project_root,
    load_yaml_config,
    resolve_project_path,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for manifest creation."""
    parser = argparse.ArgumentParser(
        description="Create an experiment manifest from prediction artifacts."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config with data and outputs sections.",
    )
    parser.add_argument(
        "--outcome",
        default="visit",
        help="Outcome to include in the manifest. Defaults to visit.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional experiment ID. Defaults to '<dataset>-<outcome>-latest'.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional artifact model names to include. When omitted, every "
            "matching run-numbered prediction artifact is included."
        ),
    )
    parser.add_argument(
        "--prediction-dir",
        default=None,
        help="Optional prediction directory override.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional manifest output path. Defaults to the configured "
            "metric_dir with '<dataset>_<outcome>_experiment_manifest.json'."
        ),
    )
    return parser.parse_args()


def get_manifest_output_path(
    output_config: dict,
    project_root: Path,
    dataset_name: str,
    outcome: str,
    output_override: str | None,
) -> Path:
    """Return the requested or default manifest output path."""
    if output_override is not None:
        return resolve_project_path(output_override, project_root)

    metric_dir = resolve_project_path(
        output_config["metric_dir"],
        project_root,
    )
    return metric_dir / f"{dataset_name}_{outcome}_experiment_manifest.json"


def create_experiment_manifest(
    config_path: Path,
    outcome: str,
    experiment_id: str | None = None,
    model_names: tuple[str, ...] | None = None,
    prediction_dir_override: str | None = None,
    output_override: str | None = None,
) -> Path:
    """Create and save a manifest using latest prediction artifacts."""
    project_root = get_project_root(Path(__file__))
    config = load_yaml_config(config_path)
    data_config = get_config_section(config, "data")
    output_config = get_config_section(config, "outputs")
    dataset_spec = get_dataset_spec(str(data_config["dataset_name"]))
    dataset_name = dataset_spec.name
    validate_supported_outcome(dataset_spec, outcome)

    prediction_dir = resolve_project_path(
        (
            prediction_dir_override
            if prediction_dir_override is not None
            else output_config["prediction_dir"]
        ),
        project_root,
    )
    prediction_artifacts = find_latest_prediction_artifacts(
        prediction_dir=prediction_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        model_names=model_names,
    )
    manifest = build_experiment_manifest(
        experiment_id=(
            experiment_id
            if experiment_id is not None
            else f"{dataset_name}-{outcome}-latest"
        ),
        dataset_name=dataset_name,
        outcome=outcome,
        config_path=config_path,
        prediction_artifacts=prediction_artifacts,
        model_artifacts=build_experiment_manifest_model_artifacts(
            prediction_artifacts
        ),
        project_root=project_root,
    )
    output_path = get_manifest_output_path(
        output_config=output_config,
        project_root=project_root,
        dataset_name=dataset_name,
        outcome=outcome,
        output_override=output_override,
    )
    save_experiment_manifest(
        manifest=manifest,
        output_path=output_path,
        dataset_name=dataset_name,
        outcome=outcome,
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
    output_path = create_experiment_manifest(
        config_path=config_path,
        outcome=args.outcome,
        experiment_id=args.experiment_id,
        model_names=(
            tuple(args.models)
            if args.models is not None
            else None
        ),
        prediction_dir_override=args.prediction_dir,
        output_override=args.output,
    )
    LOGGER.info("Saved experiment manifest to %s", output_path)


if __name__ == "__main__":
    main()
