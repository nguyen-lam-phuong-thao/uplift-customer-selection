"""Helpers for run-numbered artifact filenames."""

import re
from pathlib import Path


RUN_NUMBER_WIDTH = 2
MODEL_NAME_ALIASES = {
    "response_model_lgbm": "response_lgbm",
}


def get_artifact_model_name(model_name: str) -> str:
    """Return the model name used in artifact filenames."""
    return MODEL_NAME_ALIASES.get(model_name, model_name)


def build_artifact_filename(
    db_name: str,
    outcome: str,
    model_name: str,
    run_number: int,
    artifact_name: str,
    extension: str,
) -> str:
    """Build a run-numbered artifact filename."""
    if run_number <= 0:
        raise ValueError(f"run_number must be positive. Received: {run_number}")

    normalized_extension = extension.lstrip(".")
    return (
        f"{db_name}_{outcome}_{model_name}_"
        f"run{run_number:0{RUN_NUMBER_WIDTH}d}_"
        f"{artifact_name}.{normalized_extension}"
    )


def find_next_run_number(
    artifact_dirs: tuple[Path, ...],
    db_name: str,
    outcome: str,
    model_name: str,
) -> int:
    """Return the next run number for an artifact filename prefix."""
    prefix = f"{db_name}_{outcome}_{model_name}_run"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)_")
    run_numbers = []

    for artifact_dir in artifact_dirs:
        if not artifact_dir.exists():
            continue

        for artifact_path in artifact_dir.iterdir():
            if not artifact_path.is_file():
                continue

            match = pattern.match(artifact_path.name)
            if match:
                run_numbers.append(int(match.group(1)))

    if not run_numbers:
        return 1

    return max(run_numbers) + 1


def build_model_comparison_name(prediction_paths: list[Path]) -> str:
    """Build the model-name segment for an evaluation artifact."""
    model_names = []

    for prediction_path in prediction_paths:
        name_without_suffix = prediction_path.stem.removesuffix("_predictions")
        model_names.append(name_without_suffix.rsplit("_run", maxsplit=1)[0])

    unique_model_names = sorted(set(model_names))
    if not unique_model_names:
        raise ValueError("At least one prediction path is required.")

    return "_vs_".join(
        model_name.split("_", maxsplit=2)[2] for model_name in unique_model_names
    )
