"""Top-K targeting policy evaluation for prediction artifacts."""

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uplift_modeling.artifacts.json import save_json_artifact
from uplift_modeling.evaluation.uplift_metrics import (
    PREDICTION_COLUMNS,
    ROW_ID_COLUMN,
    calculate_policy_value,
    rank_predictions,
    validate_prediction_frame,
)


LOGGER = logging.getLogger(__name__)
TOPK_ARTIFACT_MODEL_NAME = "topk_policy_evaluation"
TOPK_BUDGET_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.30)
DEFAULT_EVALUATION_SPLITS = ("validation", "test")
EXPECTED_POLICY_ARTIFACTS = {
    "pooled_response_lgbm": ("pooled_response_lgbm", "response_lgbm"),
    "treated_response_lgbm": ("treated_response_lgbm",),
    "t_learner_lgbm": ("t_learner_lgbm",),
    "x_learner_lgbm": ("x_learner_lgbm",),
}


def parse_prediction_artifact_name(
    prediction_path: Path,
    db_name: str,
    outcome: str,
) -> tuple[str, int] | None:
    """Return model name and run number from a prediction artifact name."""
    prefix = f"{db_name}_{outcome}_"
    suffix = "_predictions.parquet"
    pattern = re.compile(
        rf"^{re.escape(prefix)}(.+)_run(\d+){re.escape(suffix)}$"
    )
    match = pattern.match(prediction_path.name)

    if match is None:
        return None

    return match.group(1), int(match.group(2))


def find_latest_expected_policy_paths(
    prediction_dir: Path,
    db_name: str,
    outcome: str,
) -> tuple[dict[str, Path], list[str]]:
    """Find latest available prediction artifacts for expected policies."""
    if not prediction_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory does not exist: {prediction_dir}"
        )

    warnings: list[str] = []
    latest_by_artifact_model: dict[str, tuple[int, Path]] = {}

    for prediction_path in prediction_dir.iterdir():
        if not prediction_path.is_file():
            continue

        parsed_name = parse_prediction_artifact_name(
            prediction_path=prediction_path,
            db_name=db_name,
            outcome=outcome,
        )
        if parsed_name is None:
            continue

        artifact_model_name, run_number = parsed_name
        latest_run = latest_by_artifact_model.get(artifact_model_name)
        if latest_run is None or run_number > latest_run[0]:
            latest_by_artifact_model[artifact_model_name] = (
                run_number,
                prediction_path,
            )

    policy_paths: dict[str, Path] = {}
    for policy_name, artifact_candidates in EXPECTED_POLICY_ARTIFACTS.items():
        matched_candidate = None
        for artifact_model_name in artifact_candidates:
            if artifact_model_name in latest_by_artifact_model:
                matched_candidate = artifact_model_name
                policy_paths[policy_name] = latest_by_artifact_model[
                    artifact_model_name
                ][1]
                break

        if matched_candidate is None:
            warning = (
                f"Missing optional prediction artifact for policy "
                f"'{policy_name}'. Checked artifact model names: "
                f"{', '.join(artifact_candidates)}."
            )
            LOGGER.warning(warning)
            warnings.append(warning)
        elif matched_candidate != policy_name:
            warning = (
                f"Using artifact model '{matched_candidate}' as policy "
                f"'{policy_name}'."
            )
            LOGGER.warning(warning)
            warnings.append(warning)

    if not policy_paths:
        raise FileNotFoundError(
            f"No expected prediction artifacts found for outcome '{outcome}' "
            f"in {prediction_dir}."
        )

    return policy_paths, warnings


def filter_evaluation_splits(
    predictions: pd.DataFrame,
    evaluation_splits: tuple[str, ...] = DEFAULT_EVALUATION_SPLITS,
) -> pd.DataFrame:
    """Keep only configured split rows for policy evaluation."""
    evaluation_predictions = predictions.loc[
        predictions["split"].isin(evaluation_splits)
    ].copy()

    if evaluation_predictions.empty:
        split_text = ", ".join(evaluation_splits)
        raise ValueError(
            f"No prediction rows found for evaluation splits: {split_text}"
        )

    return evaluation_predictions


def load_policy_prediction_frames(
    policy_paths: dict[str, Path],
    evaluation_splits: tuple[str, ...] = DEFAULT_EVALUATION_SPLITS,
) -> dict[str, pd.DataFrame]:
    """Load prediction artifacts keyed by targeting policy name."""
    policy_frames = {}

    for policy_name, prediction_path in policy_paths.items():
        LOGGER.info("Loading %s predictions from %s", policy_name, prediction_path)
        frame = pd.read_parquet(prediction_path)
        missing_columns = sorted(
            set(PREDICTION_COLUMNS).difference(frame.columns)
        )
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(
                f"Prediction artifact {prediction_path} is missing columns: "
                f"{missing_text}"
            )

        frame = frame.copy()
        frame["artifact_name"] = prediction_path.name
        frame["policy_name"] = policy_name
        policy_frames[policy_name] = filter_evaluation_splits(
            frame,
            evaluation_splits=evaluation_splits,
        )

    return policy_frames


def build_random_policy_frame(
    base_predictions: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    """Build deterministic random targeting from prediction labels."""
    rng = np.random.default_rng(random_seed)
    label_columns = ["treatment", "outcome", "split"]
    if ROW_ID_COLUMN in base_predictions.columns:
        label_columns.append(ROW_ID_COLUMN)

    random_frame = base_predictions.loc[
        :,
        label_columns,
    ].copy()
    random_frame["score"] = rng.random(len(random_frame))
    random_frame["model_name"] = "random_targeting"
    random_frame["artifact_name"] = "generated_from_prediction_labels"
    random_frame["policy_name"] = "random_targeting"
    return random_frame


def calculate_topk_policy_metrics(
    predictions: pd.DataFrame,
    budget_fraction: float,
) -> dict[str, float | int | None]:
    """Calculate Top-K targeting metrics for one policy and split."""
    validate_prediction_frame(predictions)

    if budget_fraction <= 0 or budget_fraction > 1:
        raise ValueError(
            "budget_fraction must be greater than 0 and at most 1. "
            f"Received: {budget_fraction}"
        )

    ranked = rank_predictions(predictions)
    selected_count = max(1, int(np.ceil(len(ranked) * budget_fraction)))
    selected = ranked.iloc[:selected_count]
    treated = selected.loc[selected["treatment"] == 1, "outcome"]
    control = selected.loc[selected["treatment"] == 0, "outcome"]
    treated_outcome_rate = None
    control_outcome_rate = None
    uplift_rate = None
    incremental_outcome = None
    incremental_outcome_per_1k = None

    if not treated.empty and not control.empty:
        treated_outcome_rate = float(treated.mean())
        control_outcome_rate = float(control.mean())
        uplift_rate = treated_outcome_rate - control_outcome_rate
        incremental_outcome = uplift_rate * selected_count
        incremental_outcome_per_1k = incremental_outcome / selected_count * 1000

    return {
        "budget_fraction": float(budget_fraction),
        "budget_pct": float(budget_fraction * 100),
        "n_selected": int(selected_count),
        "treated_selected": int(len(treated)),
        "control_selected": int(len(control)),
        "treated_outcome_rate": treated_outcome_rate,
        "control_outcome_rate": control_outcome_rate,
        "uplift_rate": uplift_rate,
        "incremental_outcome": incremental_outcome,
        "incremental_outcome_per_1k": incremental_outcome_per_1k,
        "policy_value": calculate_policy_value(
            predictions,
            top_fraction=budget_fraction,
        ),
    }


def calculate_topk_policy_rows(
    policy_frames: dict[str, pd.DataFrame],
    budget_fractions: tuple[float, ...] = TOPK_BUDGET_FRACTIONS,
) -> list[dict[str, Any]]:
    """Calculate Top-K policy rows for each policy, split, and budget."""
    rows: list[dict[str, Any]] = []

    for policy_name, policy_frame in policy_frames.items():
        for split, split_frame in policy_frame.groupby("split", sort=True):
            for budget_fraction in budget_fractions:
                metrics = calculate_topk_policy_metrics(
                    split_frame,
                    budget_fraction=budget_fraction,
                )
                rows.append(
                    {
                        "policy_name": policy_name,
                        "model_name": str(split_frame["model_name"].iloc[0]),
                        "split": str(split),
                        **metrics,
                    }
                )

    return rows


def prepare_topk_policy_frames(
    prediction_dir: Path,
    dataset_name: str,
    outcome: str,
    random_seed: int,
    evaluation_splits: tuple[str, ...] = DEFAULT_EVALUATION_SPLITS,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path], list[str]]:
    """Load available policy frames and add deterministic random targeting."""
    policy_paths, warnings = find_latest_expected_policy_paths(
        prediction_dir=prediction_dir,
        db_name=dataset_name,
        outcome=outcome,
    )
    policy_frames = load_policy_prediction_frames(
        policy_paths,
        evaluation_splits=evaluation_splits,
    )
    first_policy_name = sorted(policy_frames)[0]
    random_policy_frame = build_random_policy_frame(
        base_predictions=policy_frames[first_policy_name],
        random_seed=random_seed,
    )
    policy_frames = {
        "random_targeting": random_policy_frame,
        **policy_frames,
    }

    return policy_frames, policy_paths, warnings


def find_next_topk_run_number(metric_dir: Path, db_name: str, outcome: str) -> int:
    """Return next run number for Top-K policy evaluation artifacts."""
    prefix = f"{db_name}_{outcome}_{TOPK_ARTIFACT_MODEL_NAME}_run"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.json$")
    run_numbers = []

    if metric_dir.exists():
        for artifact_path in metric_dir.iterdir():
            if not artifact_path.is_file():
                continue

            match = pattern.match(artifact_path.name)
            if match:
                run_numbers.append(int(match.group(1)))

    if not run_numbers:
        return 1

    return max(run_numbers) + 1


def save_topk_policy_evaluation_artifacts(
    policy_frames: dict[str, pd.DataFrame],
    policy_paths: dict[str, Path],
    warnings: list[str],
    metric_dir: Path,
    dataset_name: str,
    outcome: str,
    random_seed: int,
) -> tuple[Path, dict[str, Any]]:
    """Save Top-K targeting policy evaluation JSON from loaded frames."""
    table_rows = calculate_topk_policy_rows(policy_frames)
    evaluated_splits = sorted({str(row["split"]) for row in table_rows})
    evaluated_policy_names = sorted(
        {str(row["policy_name"]) for row in table_rows}
    )
    run_number = find_next_topk_run_number(
        metric_dir=metric_dir,
        db_name=dataset_name,
        outcome=outcome,
    )
    metric_dir.mkdir(parents=True, exist_ok=True)
    json_path = metric_dir / (
        f"{dataset_name}_{outcome}_{TOPK_ARTIFACT_MODEL_NAME}_"
        f"run{run_number:02d}.json"
    )
    payload = {
        "dataset_name": dataset_name,
        "outcome": outcome,
        "evaluated_splits": evaluated_splits,
        "budget_fractions": list(TOPK_BUDGET_FRACTIONS),
        "random_seed": int(random_seed),
        "evaluated_policy_names": evaluated_policy_names,
        "prediction_artifacts": {
            policy_name: prediction_path.name
            for policy_name, prediction_path in sorted(policy_paths.items())
        },
        "warnings": warnings,
        "table_rows": table_rows,
    }

    save_json_artifact(payload, json_path)
    LOGGER.info("Saved Top-K policy evaluation JSON to %s", json_path)
    return json_path, payload


def save_topk_policy_evaluation(
    prediction_dir: Path,
    metric_dir: Path,
    dataset_name: str,
    outcome: str,
    random_seed: int,
    evaluation_splits: tuple[str, ...] = DEFAULT_EVALUATION_SPLITS,
) -> tuple[Path, dict[str, Any]]:
    """Load predictions and save Top-K targeting policy JSON."""
    policy_frames, policy_paths, warnings = prepare_topk_policy_frames(
        prediction_dir=prediction_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        random_seed=random_seed,
        evaluation_splits=evaluation_splits,
    )
    return save_topk_policy_evaluation_artifacts(
        policy_frames=policy_frames,
        policy_paths=policy_paths,
        warnings=warnings,
        metric_dir=metric_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        random_seed=random_seed,
    )
