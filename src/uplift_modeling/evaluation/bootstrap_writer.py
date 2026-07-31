"""Artifact writing for Top-K bootstrap policy evaluation."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from uplift_modeling.artifacts.json import save_json_artifact
from uplift_modeling.evaluation.bootstrap_config import (
    DEFAULT_BASELINE_POLICY,
    DEFAULT_BOOTSTRAP_RANDOM_SEED,
    DEFAULT_N_BOOTSTRAP,
    validate_bootstrap_config,
)
from uplift_modeling.evaluation.bootstrap_summary import (
    calculate_bootstrap_policy_rows,
)
from uplift_modeling.evaluation.topk_policy import (
    TOPK_BUDGET_FRACTIONS,
)


LOGGER = logging.getLogger(__name__)
BOOTSTRAP_POLICY_ARTIFACT_NAME = "bootstrap_policy_metrics"
BOOTSTRAP_CONTRAST_ARTIFACT_NAME = "bootstrap_paired_contrasts"


def find_next_bootstrap_run_number(
    metric_dir: Path,
    db_name: str,
    outcome: str,
) -> int:
    """Return next run number for bootstrap policy artifacts."""
    artifact_names = (
        BOOTSTRAP_POLICY_ARTIFACT_NAME,
        BOOTSTRAP_CONTRAST_ARTIFACT_NAME,
    )
    patterns = [
        re.compile(
            rf"^{re.escape(db_name)}_{re.escape(outcome)}_"
            rf"{re.escape(artifact_name)}_run(\d+)\.json$"
        )
        for artifact_name in artifact_names
    ]
    run_numbers = []

    if metric_dir.exists():
        for artifact_path in metric_dir.iterdir():
            if not artifact_path.is_file():
                continue
            for pattern in patterns:
                match = pattern.match(artifact_path.name)
                if match:
                    run_numbers.append(int(match.group(1)))

    if not run_numbers:
        return 1

    return max(run_numbers) + 1


def _add_outcome_to_rows(
    rows: list[dict[str, Any]],
    outcome: str,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "outcome": outcome,
        }
        for row in rows
    ]


def save_bootstrap_policy_evaluation(
    policy_frames: dict[str, pd.DataFrame],
    metric_dir: Path,
    dataset_name: str,
    outcome: str,
    random_seed: int = DEFAULT_BOOTSTRAP_RANDOM_SEED,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    budget_fractions: Iterable[float] = TOPK_BUDGET_FRACTIONS,
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
    bootstrap_splits: Iterable[str] | None = None,
    prediction_artifacts: dict[str, Path] | None = None,
    warnings: Iterable[str] = (),
) -> tuple[Path, Path, dict[str, Any]]:
    """Save bootstrap policy metric and paired contrast JSON artifacts."""
    fractions = validate_bootstrap_config(
        budget_fractions=budget_fractions,
        n_bootstrap=n_bootstrap,
    )
    policy_rows, contrast_rows, bootstrap_warnings = (
        calculate_bootstrap_policy_rows(
            policy_frames=policy_frames,
            budget_fractions=fractions,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
            baseline_policy=baseline_policy,
            bootstrap_splits=bootstrap_splits,
        )
    )
    policy_rows = _add_outcome_to_rows(policy_rows, outcome=outcome)
    contrast_rows = _add_outcome_to_rows(contrast_rows, outcome=outcome)
    all_warnings = [*warnings, *bootstrap_warnings]

    evaluated_splits = sorted({str(row["split"]) for row in policy_rows})
    evaluated_policy_names = sorted({str(row["policy"]) for row in policy_rows})
    run_number = find_next_bootstrap_run_number(
        metric_dir=metric_dir,
        db_name=dataset_name,
        outcome=outcome,
    )

    metric_dir.mkdir(parents=True, exist_ok=True)
    policy_json_path = metric_dir / (
        f"{dataset_name}_{outcome}_{BOOTSTRAP_POLICY_ARTIFACT_NAME}_"
        f"run{run_number:02d}.json"
    )
    contrast_json_path = metric_dir / (
        f"{dataset_name}_{outcome}_{BOOTSTRAP_CONTRAST_ARTIFACT_NAME}_"
        f"run{run_number:02d}.json"
    )
    prediction_artifact_names = {
        policy_name: prediction_path.name
        for policy_name, prediction_path in sorted(
            (prediction_artifacts or {}).items()
        )
    }
    payload = {
        "dataset_name": dataset_name,
        "outcome": outcome,
        "evaluated_splits": evaluated_splits,
        "budget_fractions": list(fractions),
        "n_bootstrap": int(n_bootstrap),
        "random_seed": int(random_seed),
        "evaluated_policy_names": evaluated_policy_names,
        "baseline_policy": baseline_policy,
        "prediction_artifacts": prediction_artifact_names,
        "warnings": all_warnings,
        "regular_bootstrap_metric_rows": policy_rows,
        "paired_contrast_rows": contrast_rows,
    }

    save_json_artifact(
        {**payload, "artifact_type": "policy_metrics"},
        policy_json_path,
    )
    save_json_artifact(
        {**payload, "artifact_type": "paired_contrasts"},
        contrast_json_path,
    )
    LOGGER.info("Saved bootstrap policy metrics JSON to %s", policy_json_path)
    LOGGER.info("Saved bootstrap paired contrasts JSON to %s", contrast_json_path)

    return (
        policy_json_path,
        contrast_json_path,
        payload,
    )
