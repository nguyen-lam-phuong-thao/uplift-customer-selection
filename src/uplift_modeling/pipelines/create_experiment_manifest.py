"""Create an experiment manifest from run-numbered prediction artifacts."""

import argparse
import logging
from pathlib import Path
import re
import pyarrow.parquet as pq
from uplift_modeling.artifacts.manifest import (
    build_experiment_manifest,
    build_experiment_manifest_model_artifacts,
    get_missing_prediction_columns,
    save_experiment_manifest,
    validate_model_artifacts_match_predictions,
    validate_prediction_artifact_model_name,
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
        required=True,
        help="Unique immutable ID for this experiment manifest.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional manifest output path. Defaults to the configured "
            "metric_dir with '<dataset>_<outcome>_experiment_manifest.json'."
        ),
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="POLICY=PATH",
        help=(
            "Exact validation prediction artifact in POLICY=PATH format. "
            "Repeat once per policy."
        ),
    )
    return parser.parse_args()


def get_manifest_output_path(
    output_config: dict,
    project_root: Path,
    dataset_name: str,
    outcome: str,
    experiment_id: str,
    output_override: str | None,
) -> Path:
    """Return the requested or default immutable manifest output path."""
    if output_override is not None:
        return resolve_project_path(output_override, project_root)

    metric_dir = resolve_project_path(
        output_config["metric_dir"],
        project_root,
    )
    return metric_dir / (
        f"{dataset_name}_{outcome}_{experiment_id}_experiment_manifest.json"
    )


def create_experiment_manifest(
    config_path: Path,
    outcome: str,
    experiment_id: str,
    prediction_artifacts: dict[str, Path],
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

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id) is None:
        raise ValueError(
            "experiment_id must start with a letter or number and contain "
            "only letters, numbers, '.', '_' or '-'."
        )
    if not prediction_artifacts:
        raise ValueError("prediction_artifacts must not be empty.")

    resolved_prediction_artifacts = {
        policy_name: resolve_project_path(
            prediction_path,
            project_root,
        )
        for policy_name, prediction_path in prediction_artifacts.items()
    }

    for policy_name, prediction_path in resolved_prediction_artifacts.items():
        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Prediction artifact does not exist for policy "
                f"'{policy_name}': {prediction_path}"
            )

        if not prediction_path.is_file():
            raise ValueError(
                f"Prediction artifact must be a file for policy "
                f"'{policy_name}': {prediction_path}"
            )

        missing_columns = get_missing_prediction_columns(prediction_path)
        if missing_columns:
            raise ValueError(
                f"Prediction artifact for policy '{policy_name}' is missing "
                f"required columns: {', '.join(missing_columns)}."
            )

        split_values = {
            str(value)
            for value in pq.read_table(
                prediction_path,
                columns=["split"],
            ).column("split").to_pylist()
        }

        if split_values != {"validation"}:
            raise ValueError(
                f"Prediction artifact for policy '{policy_name}' must contain "
                f"only validation rows. Received splits: "
                f"{sorted(split_values)}."
            )

        validate_prediction_artifact_model_name(
            prediction_path=prediction_path,
            manifest_policy=policy_name,
        )

    model_artifacts = build_experiment_manifest_model_artifacts(
        resolved_prediction_artifacts
    )

    missing_provenance = sorted(
        set(resolved_prediction_artifacts).difference(model_artifacts)
    )
    if missing_provenance:
        raise FileNotFoundError(
            "Missing model provenance for prediction artifact(s): "
            f"{missing_provenance}."
        )

    validate_model_artifacts_match_predictions(
        model_artifacts=model_artifacts,
        prediction_paths=resolved_prediction_artifacts,
    )

    for policy_name, model_artifact in model_artifacts.items():
        if model_artifact.get("dataset_name") != dataset_name:
            raise ValueError(
                f"Model provenance dataset_name does not match for "
                f"policy '{policy_name}'."
            )

        if model_artifact.get("outcome") != outcome:
            raise ValueError(
                f"Model provenance outcome does not match for "
                f"policy '{policy_name}'."
            )

        if model_artifact.get("policy_name") != policy_name:
            raise ValueError(
                f"Model provenance policy_name does not match manifest key "
                f"'{policy_name}'."
            )

    manifest = build_experiment_manifest(
        experiment_id=experiment_id,
        dataset_name=dataset_name,
        outcome=outcome,
        config_path=config_path,
        prediction_artifacts=resolved_prediction_artifacts,
        model_artifacts=model_artifacts,
        project_root=project_root,
    )
    output_path = get_manifest_output_path(
        output_config=output_config,
        project_root=project_root,
        dataset_name=dataset_name,
        outcome=outcome,
        experiment_id=experiment_id,
        output_override=output_override,
    )

    if output_path.exists():
        raise FileExistsError(
            "Experiment manifest already exists and cannot be overwritten: "
            f"{output_path}"
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
    prediction_artifacts: dict[str, Path] = {}
    for prediction_value in args.prediction:
        policy_name, separator, path_value = prediction_value.partition("=")
        policy_name = policy_name.strip()
        path_value = path_value.strip()

        if not separator or not policy_name or not path_value:
            raise ValueError(
                "Each --prediction value must use non-empty "
                "POLICY=PATH format."
            )

        if policy_name in prediction_artifacts:
            raise ValueError(
                f"Duplicate --prediction policy name: {policy_name}"
            )

        prediction_artifacts[policy_name] = resolve_project_path(
            path_value,
            project_root,
        )
    output_path = create_experiment_manifest(
        config_path=config_path,
        outcome=args.outcome,
        experiment_id=args.experiment_id,
        prediction_artifacts=prediction_artifacts,
        output_override=args.output,
    )
    LOGGER.info("Saved experiment manifest to %s", output_path)


if __name__ == "__main__":
    main()
