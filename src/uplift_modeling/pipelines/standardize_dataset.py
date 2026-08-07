"""CLI pipeline for standardizing prepared modeling datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from uplift_modeling.data.dataset_spec import load_dataset_config
from uplift_modeling.data.standardization import standardize_prepared_dataset
from uplift_modeling.utils.config import get_project_root, resolve_project_path


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize a prepared modeling table into outcome-specific "
            "decision datasets."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to dataset YAML config.",
    )
    return parser.parse_args()


def standardize_dataset_pipeline(config_path: str | Path) -> dict[str, Path]:
    project_root = get_project_root(Path(__file__))
    resolved_config_path = resolve_project_path(config_path, project_root)

    dataset_config = load_dataset_config(
        resolved_config_path,
        project_root=project_root,
    )

    output_paths = standardize_prepared_dataset(dataset_config)

    for outcome, output_path in output_paths.items():
        LOGGER.info("Saved %s decision dataset to %s", outcome, output_path)

    return output_paths


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    standardize_dataset_pipeline(args.config)


if __name__ == "__main__":
    main()