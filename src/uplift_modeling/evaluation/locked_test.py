"""Locked test-set reporting for the selected champion policy."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from uplift_modeling.artifacts.json import save_json_artifact
from uplift_modeling.evaluation.bootstrap_config import DEFAULT_BASELINE_POLICY
from uplift_modeling.evaluation.topk_policy import (
    TOPK_BUDGET_FRACTIONS,
    calculate_topk_policy_metrics,
    parse_prediction_artifact_name,
)
from uplift_modeling.evaluation.uplift_metrics import PREDICTION_COLUMNS


LOGGER = logging.getLogger(__name__)
LOCKED_TEST_ARTIFACT_NAME = "locked_test_evaluation"
LOCKED_TEST_SPLIT = "test"


def load_selection_gate_payload(selection_artifact_path: Path) -> dict[str, Any]:
    """Load an existing model-selection gate artifact."""
    if not selection_artifact_path.exists():
        raise FileNotFoundError(
            f"Selection Gate artifact does not exist: {selection_artifact_path}"
        )

    with selection_artifact_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise ValueError(
            "Selection Gate artifact must contain a JSON object: "
            f"{selection_artifact_path}"
        )

    return payload


def get_champion_policy(selection_payload: Mapping[str, Any]) -> str:
    """Return the champion policy fixed by the Selection Gate artifact."""
    champion_policy = selection_payload.get("champion_policy")
    if not isinstance(champion_policy, str) or not champion_policy:
        raise ValueError(
            "Selection Gate artifact must contain a non-empty "
            "'champion_policy' string."
        )

    return champion_policy


def find_latest_policy_prediction_path(
    prediction_dir: Path,
    dataset_name: str,
    outcome: str,
    policy: str,
) -> Path | None:
    """Find the latest prediction artifact for one policy and outcome."""
    if not prediction_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory does not exist: {prediction_dir}"
        )

    latest_run: tuple[int, Path] | None = None
    for prediction_path in prediction_dir.iterdir():
        if not prediction_path.is_file():
            continue

        parsed_name = parse_prediction_artifact_name(
            prediction_path=prediction_path,
            db_name=dataset_name,
            outcome=outcome,
        )
        if parsed_name is None:
            continue

        artifact_policy, run_number = parsed_name
        if artifact_policy != policy:
            continue

        if latest_run is None or run_number > latest_run[0]:
            latest_run = (run_number, prediction_path)

    if latest_run is None:
        return None

    return latest_run[1]


def find_required_prediction_paths(
    prediction_dir: Path,
    dataset_name: str,
    outcome: str,
    champion_policy: str,
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
) -> dict[str, Path]:
    """Find prediction artifacts required for locked-test reporting."""
    policies = sorted({champion_policy, baseline_policy})
    policy_paths: dict[str, Path] = {}

    for policy in policies:
        prediction_path = find_latest_policy_prediction_path(
            prediction_dir=prediction_dir,
            dataset_name=dataset_name,
            outcome=outcome,
            policy=policy,
        )
        if prediction_path is None:
            role = (
                "Champion"
                if policy == champion_policy
                else "Baseline"
            )
            raise ValueError(
                f"{role} prediction artifact is missing for policy '{policy}', "
                f"outcome '{outcome}', and dataset '{dataset_name}' in "
                f"{prediction_dir}."
            )

        policy_paths[policy] = prediction_path

    return policy_paths


def load_locked_test_prediction_frames(
    policy_paths: Mapping[str, Path],
) -> dict[str, pd.DataFrame]:
    """Load required policy prediction artifacts and keep test rows only."""
    policy_frames: dict[str, pd.DataFrame] = {}

    for policy, prediction_path in policy_paths.items():
        LOGGER.info(
            "Loading locked-test predictions for %s from %s",
            policy,
            prediction_path,
        )
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

        test_frame = frame.loc[frame["split"] == LOCKED_TEST_SPLIT].copy()
        if test_frame.empty:
            raise ValueError(
                f"{policy} test predictions are missing from artifact "
                f"{prediction_path}."
            )

        test_frame["artifact_name"] = prediction_path.name
        test_frame["policy_name"] = policy
        policy_frames[policy] = test_frame

    return policy_frames


def calculate_locked_test_rows(
    policy_frames: Mapping[str, pd.DataFrame],
    outcome: str,
    budget_fractions: Iterable[float] = TOPK_BUDGET_FRACTIONS,
) -> list[dict[str, Any]]:
    """Calculate locked-test Top-K rows for required policies only."""
    rows: list[dict[str, Any]] = []

    for policy in sorted(policy_frames):
        policy_frame = policy_frames[policy]
        for budget_fraction in budget_fractions:
            metrics = calculate_topk_policy_metrics(
                policy_frame,
                budget_fraction=budget_fraction,
            )
            rows.append(
                {
                    "policy": policy,
                    "outcome": outcome,
                    "split": LOCKED_TEST_SPLIT,
                    "budget_fraction": metrics["budget_fraction"],
                    "budget_pct": metrics["budget_pct"],
                    "n_selected": metrics["n_selected"],
                    "policy_value": metrics["policy_value"],
                    "incremental_outcome": metrics["incremental_outcome"],
                }
            )

    return rows


def find_next_locked_test_run_number(
    metric_dir: Path,
    dataset_name: str,
    outcome: str,
) -> int:
    """Return next run number for locked-test evaluation artifacts."""
    prefix = f"{dataset_name}_{outcome}_{LOCKED_TEST_ARTIFACT_NAME}_run"
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


def save_locked_test_evaluation_artifact(
    metric_dir: Path,
    dataset_name: str,
    outcome: str,
    champion_policy: str,
    baseline_policy: str,
    selection_artifact_path: Path,
    prediction_paths: Mapping[str, Path],
    locked_test_rows: list[dict[str, Any]],
    budget_fractions: Iterable[float] = TOPK_BUDGET_FRACTIONS,
) -> tuple[Path, dict[str, Any]]:
    """Save the locked-test evaluation JSON artifact."""
    run_number = find_next_locked_test_run_number(
        metric_dir=metric_dir,
        dataset_name=dataset_name,
        outcome=outcome,
    )
    output_path = metric_dir / (
        f"{dataset_name}_{outcome}_{LOCKED_TEST_ARTIFACT_NAME}_"
        f"run{run_number:02d}.json"
    )
    payload = {
        "artifact_type": LOCKED_TEST_ARTIFACT_NAME,
        "dataset_name": dataset_name,
        "outcome": outcome,
        "split": LOCKED_TEST_SPLIT,
        "champion_policy": champion_policy,
        "baseline_policy": baseline_policy,
        "selection_artifact": selection_artifact_path.name,
        "prediction_artifacts": {
            policy: prediction_path.name
            for policy, prediction_path in sorted(prediction_paths.items())
        },
        "budget_fractions": [float(fraction) for fraction in budget_fractions],
        "locked_test_rows": locked_test_rows,
    }

    save_json_artifact(payload, output_path)
    LOGGER.info("Saved locked-test evaluation JSON to %s", output_path)
    return output_path, payload


def save_locked_test_evaluation(
    prediction_dir: Path,
    metric_dir: Path,
    dataset_name: str,
    outcome: str,
    selection_artifact_path: Path,
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
    budget_fractions: Iterable[float] = TOPK_BUDGET_FRACTIONS,
) -> tuple[Path, dict[str, Any]]:
    """Load fixed champion predictions and save locked-test reporting JSON."""
    selection_payload = load_selection_gate_payload(selection_artifact_path)
    champion_policy = get_champion_policy(selection_payload)
    fractions = tuple(float(fraction) for fraction in budget_fractions)
    prediction_paths = find_required_prediction_paths(
        prediction_dir=prediction_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        champion_policy=champion_policy,
        baseline_policy=baseline_policy,
    )
    policy_frames = load_locked_test_prediction_frames(prediction_paths)
    locked_test_rows = calculate_locked_test_rows(
        policy_frames=policy_frames,
        outcome=outcome,
        budget_fractions=fractions,
    )

    return save_locked_test_evaluation_artifact(
        metric_dir=metric_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        champion_policy=champion_policy,
        baseline_policy=baseline_policy,
        selection_artifact_path=selection_artifact_path,
        prediction_paths=prediction_paths,
        locked_test_rows=locked_test_rows,
        budget_fractions=fractions,
    )
