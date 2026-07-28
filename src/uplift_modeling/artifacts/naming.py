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


def find_latest_prediction_paths(
    prediction_dir: Path,
    db_name: str,
    outcome: str,
) -> list[Path]:
    """Return the latest prediction file for each model and outcome."""
    prefix = f"{db_name}_{outcome}_"
    suffix = "_predictions.parquet"
    pattern = re.compile(
        rf"^{re.escape(prefix)}(.+)_run(\d+){re.escape(suffix)}$"
    )
    latest_by_model: dict[str, tuple[int, Path]] = {}

    if not prediction_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory does not exist: {prediction_dir}"
        )

    for prediction_path in prediction_dir.iterdir():
        if not prediction_path.is_file():
            continue

        match = pattern.match(prediction_path.name)
        if not match:
            continue

        model_name = match.group(1)
        run_number = int(match.group(2))
        latest_run = latest_by_model.get(model_name)

        if latest_run is None or run_number > latest_run[0]:
            latest_by_model[model_name] = (run_number, prediction_path)

    if not latest_by_model:
        raise FileNotFoundError(
            "No run-numbered prediction parquet files found for outcome "
            f"'{outcome}' in {prediction_dir}"
        )

    return [
        prediction_path
        for _, prediction_path in sorted(
            latest_by_model.values(),
            key=lambda item: item[1].name,
        )
    ]


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
