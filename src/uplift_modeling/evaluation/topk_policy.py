"""Top-K targeting policy evaluation for prediction artifacts."""

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from uplift_modeling.artifacts.json import save_json_artifact
from uplift_modeling.data.row_id import (
    ROW_ID_COLUMN,
    align_frames_by_row_id,
)
from uplift_modeling.evaluation.uplift_metrics import (
    PREDICTION_COLUMNS,
    calculate_policy_value,
    rank_predictions,
    validate_prediction_frame,
)


LOGGER = logging.getLogger(__name__)
TOPK_ARTIFACT_MODEL_NAME = "topk_policy_evaluation"
TOPK_BUDGET_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.30)
SELECTION_SPLIT = "validation"
DEFAULT_EVALUATION_SPLITS = (SELECTION_SPLIT,)
EXPECTED_POLICY_ARTIFACTS = {
    "pooled_response_lgbm": ("pooled_response_lgbm", "response_lgbm"),
    "treated_response_lgbm": ("treated_response_lgbm",),
    "t_learner_lgbm": ("t_learner_lgbm",),
    "x_learner_lgbm": ("x_learner_lgbm",),
}


def validate_standard_evaluation_splits(
    evaluation_splits: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate that standard evaluation uses validation rows only."""
    requested_splits = tuple(str(split) for split in evaluation_splits)
    invalid_splits = sorted(set(requested_splits).difference({SELECTION_SPLIT}))
    if invalid_splits:
        invalid_text = ", ".join(invalid_splits)
        raise ValueError(
            "Standard evaluation may use the validation split only. "
            "The test split is reserved for locked-test evaluation. "
            f"Received invalid split(s): {invalid_text}."
        )

    if not requested_splits:
        raise ValueError("At least one standard evaluation split is required.")

    return requested_splits


def resolve_expected_policy_paths(
    manifest_prediction_paths: dict[str, Path],
    outcome: str,
) -> tuple[dict[str, Path], list[str]]:
    """Resolve expected Top-K policies from manifest-listed artifacts."""
    warnings: list[str] = []
    policy_paths: dict[str, Path] = {}
    for policy_name, artifact_candidates in EXPECTED_POLICY_ARTIFACTS.items():
        matched_candidates = [
            artifact_model_name
            for artifact_model_name in artifact_candidates
            if artifact_model_name in manifest_prediction_paths
        ]

        if len(matched_candidates) > 1:
            matched_text = ", ".join(matched_candidates)
            raise ValueError(
                "Experiment manifest contains ambiguous prediction artifacts "
                f"for policy '{policy_name}': {matched_text}."
            )

        if not matched_candidates:
            warning = (
                f"Missing optional prediction artifact for policy "
                f"'{policy_name}'. Checked artifact model names: "
                f"{', '.join(artifact_candidates)}."
            )
            LOGGER.warning(warning)
            warnings.append(warning)
            continue

        matched_candidate = matched_candidates[0]
        policy_paths[policy_name] = manifest_prediction_paths[matched_candidate]
        if matched_candidate != policy_name:
            warning = (
                f"Using artifact model '{matched_candidate}' as policy "
                f"'{policy_name}'."
            )
            LOGGER.warning(warning)
            warnings.append(warning)

    if not policy_paths:
        raise ValueError(
            "Experiment manifest does not contain any expected prediction "
            f"artifacts for outcome '{outcome}'."
        )

    return policy_paths, warnings


def filter_evaluation_splits(
    predictions: pd.DataFrame,
    evaluation_splits: tuple[str, ...] = DEFAULT_EVALUATION_SPLITS,
) -> pd.DataFrame:
    """Keep only configured split rows for policy evaluation."""
    evaluation_splits = validate_standard_evaluation_splits(evaluation_splits)
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
        schema_columns = set(pq.read_schema(prediction_path).names)
        missing_columns = sorted(
            set(PREDICTION_COLUMNS).difference(schema_columns)
        )
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(
                f"Prediction artifact {prediction_path} is missing columns: "
                f"{missing_text}"
            )

        split_frame = pd.read_parquet(prediction_path, columns=["split"])
        if "test" in {str(split) for split in split_frame["split"].unique()}:
            raise ValueError(
                "Standard Top-K evaluation cannot load prediction artifacts "
                "containing the test split. The test split is reserved for "
                f"locked-test evaluation: {prediction_path}"
            )

        frame = pd.read_parquet(prediction_path)
        frame = frame.copy()
        frame["artifact_name"] = prediction_path.name
        frame["policy_name"] = policy_name
        evaluation_frame = filter_evaluation_splits(
            frame,
            evaluation_splits=evaluation_splits,
        )
        validate_prediction_frame(evaluation_frame)
        policy_frames[policy_name] = evaluation_frame

    return policy_frames


def build_random_policy_frame(
    base_predictions: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    """Build deterministic random targeting from prediction labels."""
    validate_prediction_frame(base_predictions)
    rng = np.random.default_rng(random_seed)
    label_columns = [ROW_ID_COLUMN, "treatment", "outcome", "split"]

    random_frame = base_predictions.sort_values(
        ROW_ID_COLUMN,
        kind="mergesort",
    ).loc[:, label_columns].copy()
    random_frame["score"] = rng.random(len(random_frame))
    random_frame["model_name"] = "random_targeting"
    random_frame["artifact_name"] = "generated_from_prediction_labels"
    random_frame["policy_name"] = "random_targeting"
    return random_frame


def calculate_topk_policy_metrics(
    predictions: pd.DataFrame,
    budget_fraction: float,
    *,
    require_unique_row_id: bool = True,
) -> dict[str, float | int | None]:
    """Calculate Top-K targeting metrics for one policy and split."""
    validate_prediction_frame(
        predictions,
        require_unique_row_id=require_unique_row_id,
    )

    if budget_fraction <= 0 or budget_fraction > 1:
        raise ValueError(
            "budget_fraction must be greater than 0 and at most 1. "
            f"Received: {budget_fraction}"
        )

    ranked = rank_predictions(
        predictions,
        require_unique_row_id=require_unique_row_id,
    )
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
        incremental_outcome_per_1k = (
            incremental_outcome / selected_count * 1000
        )

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
            require_unique_row_id=require_unique_row_id,
        ),
    }


def calculate_topk_policy_rows(
    policy_frames: dict[str, pd.DataFrame],
    budget_fractions: tuple[float, ...] = TOPK_BUDGET_FRACTIONS,
) -> list[dict[str, Any]]:
    """Calculate Top-K policy rows for each policy, split, and budget."""
    rows: list[dict[str, Any]] = []

    for policy_name, policy_frame in policy_frames.items():
        validate_standard_evaluation_splits(
            tuple(str(split) for split in policy_frame["split"].unique())
        )

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
    manifest_prediction_paths: dict[str, Path],
    outcome: str,
    random_seed: int,
    evaluation_splits: tuple[str, ...] = DEFAULT_EVALUATION_SPLITS,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path], list[str]]:
    """Load available policy frames and add deterministic random targeting."""
    policy_paths, warnings = resolve_expected_policy_paths(
        manifest_prediction_paths=manifest_prediction_paths,
        outcome=outcome,
    )
    policy_frames = load_policy_prediction_frames(
        policy_paths,
        evaluation_splits=evaluation_splits,
    )
    policy_frames = align_frames_by_row_id(
        policy_frames,
        label_columns=("treatment", "outcome", "split"),
        context="Top-K policy prediction frames",
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
    manifest_prediction_paths: dict[str, Path],
    metric_dir: Path,
    dataset_name: str,
    outcome: str,
    random_seed: int,
    evaluation_splits: tuple[str, ...] = DEFAULT_EVALUATION_SPLITS,
) -> tuple[Path, dict[str, Any]]:
    """Load predictions and save Top-K targeting policy JSON."""
    policy_frames, policy_paths, warnings = prepare_topk_policy_frames(
        manifest_prediction_paths=manifest_prediction_paths,
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
