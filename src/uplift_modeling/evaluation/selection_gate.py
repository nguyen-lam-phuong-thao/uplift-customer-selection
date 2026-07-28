"""Deterministic model-selection gate for bootstrap policy contrasts."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from uplift_modeling.artifacts.json import save_json_artifact


LOGGER = logging.getLogger(__name__)
MODEL_SELECTION_ARTIFACT_NAME = "model_selection_gate"
SELECTION_METHOD = (
    "bootstrap_paired_contrast_ci_lower_positive_largest_mean_delta"
)
REQUIRED_BOOTSTRAP_FIELDS: tuple[str, ...] = (
    "policy",
    "baseline_policy",
    "split",
    "outcome",
    "budget_fraction",
    "metric",
    "mean_delta",
    "ci_lower",
)


@dataclass(frozen=True)
class SelectionGateSettings:
    """Primary settings used to choose a champion policy."""

    outcome: str
    split: str
    budget_fraction: float
    metric: str
    baseline_policy: str


def load_bootstrap_payload(bootstrap_json_path: Path) -> dict[str, Any]:
    """Load an existing bootstrap JSON artifact."""
    if not bootstrap_json_path.exists():
        raise FileNotFoundError(
            f"Bootstrap JSON artifact does not exist: {bootstrap_json_path}"
        )

    with bootstrap_json_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Bootstrap JSON artifact must contain an object: {bootstrap_json_path}"
        )
    return payload


def select_champion_from_bootstrap_payload(
    bootstrap_payload: Mapping[str, Any],
    settings: SelectionGateSettings,
) -> dict[str, Any]:
    """Select a champion policy from bootstrap paired-contrast rows."""
    contrast_rows = bootstrap_payload.get("paired_contrast_rows")
    if not isinstance(contrast_rows, list):
        raise ValueError(
            "Bootstrap payload must contain paired_contrast_rows as a list."
        )

    return select_champion_from_paired_contrasts(
        contrast_rows=contrast_rows,
        settings=settings,
    )


def select_champion_from_paired_contrasts(
    contrast_rows: Sequence[Mapping[str, Any]],
    settings: SelectionGateSettings,
) -> dict[str, Any]:
    """Apply the configured selection rule to paired-contrast rows."""
    _validate_required_fields(contrast_rows)
    _validate_baseline_present(contrast_rows, settings.baseline_policy)
    primary_rows = _filter_primary_rows(contrast_rows, settings)
    _validate_duplicate_primary_rows(primary_rows)
    explanation_rows = _build_explanation_rows(primary_rows, settings)

    passing_rows = [
        row for row in explanation_rows if bool(row["passed_selection_gate"])
    ]
    if passing_rows:
        champion_row = sorted(
            passing_rows,
            key=lambda row: (-float(row["mean_delta"]), str(row["policy"])),
        )[0]
        champion_policy = str(champion_row["policy"])
        selection_reason = (
            f"Selected '{champion_policy}' because it passed ci_lower > 0 "
            "and had the largest mean_delta among passing policies, with "
            "alphabetical policy name as the deterministic tie-break."
        )
    else:
        champion_policy = settings.baseline_policy
        selection_reason = (
            f"Selected fallback baseline '{settings.baseline_policy}' because "
            "no candidate policy passed ci_lower > 0."
        )

    for row in explanation_rows:
        row["is_champion"] = row["policy"] == champion_policy
        row["reason"] = _explain_policy_row(
            row=row,
            champion_policy=champion_policy,
            fallback_policy=settings.baseline_policy,
        )

    return {
        "champion_policy": champion_policy,
        "selection_method": SELECTION_METHOD,
        "selection_reason": selection_reason,
        "selection_settings": asdict(settings),
        "explanation_rows": explanation_rows,
    }


def find_next_model_selection_run_number(
    metric_dir: Path,
    dataset_name: str,
    outcome: str,
) -> int:
    """Return next run number for model-selection gate artifacts."""
    prefix = f"{dataset_name}_{outcome}_{MODEL_SELECTION_ARTIFACT_NAME}_run"
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


def save_model_selection_gate(
    metric_dir: Path,
    dataset_name: str,
    settings: SelectionGateSettings,
    bootstrap_payload: Mapping[str, Any] | None = None,
    bootstrap_json_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Save a model-selection gate JSON artifact."""
    if bootstrap_payload is None:
        if bootstrap_json_path is None:
            raise ValueError(
                "Either bootstrap_payload or bootstrap_json_path is required."
            )
        bootstrap_payload = load_bootstrap_payload(bootstrap_json_path)

    selection_result = select_champion_from_bootstrap_payload(
        bootstrap_payload=bootstrap_payload,
        settings=settings,
    )
    run_number = find_next_model_selection_run_number(
        metric_dir=metric_dir,
        dataset_name=dataset_name,
        outcome=settings.outcome,
    )
    output_path = metric_dir / (
        f"{dataset_name}_{settings.outcome}_{MODEL_SELECTION_ARTIFACT_NAME}_"
        f"run{run_number:02d}.json"
    )
    payload = {
        "artifact_type": MODEL_SELECTION_ARTIFACT_NAME,
        "dataset_name": dataset_name,
        "bootstrap_artifact": (
            bootstrap_json_path.name if bootstrap_json_path is not None else None
        ),
        **selection_result,
    }

    save_json_artifact(payload, output_path)
    LOGGER.info("Saved model-selection gate JSON to %s", output_path)
    return output_path, payload


def _validate_required_fields(
    contrast_rows: Sequence[Mapping[str, Any]],
) -> None:
    missing_by_index = {}

    for index, row in enumerate(contrast_rows):
        missing_fields = sorted(
            field for field in REQUIRED_BOOTSTRAP_FIELDS if field not in row
        )
        if missing_fields:
            missing_by_index[index] = missing_fields

    if missing_by_index:
        raise ValueError(
            "Bootstrap paired-contrast rows are missing required fields: "
            f"{missing_by_index}"
        )


def _validate_baseline_present(
    contrast_rows: Sequence[Mapping[str, Any]],
    baseline_policy: str,
) -> None:
    baseline_values = {str(row["baseline_policy"]) for row in contrast_rows}
    if baseline_policy not in baseline_values:
        raise ValueError(
            f"Baseline policy '{baseline_policy}' is missing from bootstrap "
            "paired-contrast rows."
        )


def _filter_primary_rows(
    contrast_rows: Sequence[Mapping[str, Any]],
    settings: SelectionGateSettings,
) -> list[Mapping[str, Any]]:
    string_matched_rows = [
        row
        for row in contrast_rows
        if str(row["baseline_policy"]) == settings.baseline_policy
        and str(row["outcome"]) == settings.outcome
        and str(row["split"]) == settings.split
        and str(row["metric"]) == settings.metric
    ]

    primary_rows = []
    for row in string_matched_rows:
        budget_fraction = _finite_number(
            row["budget_fraction"],
            "budget_fraction",
            row,
        )
        if math.isclose(
            budget_fraction,
            settings.budget_fraction,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            primary_rows.append(row)

    if not primary_rows:
        raise ValueError(
            "No bootstrap paired-contrast rows match the primary selection "
            f"setting: {asdict(settings)}."
        )

    return primary_rows


def _validate_duplicate_primary_rows(
    primary_rows: Sequence[Mapping[str, Any]],
) -> None:
    policy_counts = Counter(str(row["policy"]) for row in primary_rows)
    duplicates = sorted(
        policy for policy, count in policy_counts.items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "Duplicate primary bootstrap paired-contrast rows exist for "
            f"policies: {duplicates}."
        )


def _build_explanation_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    settings: SelectionGateSettings,
) -> list[dict[str, Any]]:
    explanation_rows = []

    for row in primary_rows:
        policy = str(row["policy"])
        if policy == settings.baseline_policy:
            continue

        mean_delta = _finite_number(row["mean_delta"], "mean_delta", row)
        ci_lower = _finite_number(row["ci_lower"], "ci_lower", row)
        explanation_rows.append(
            {
                "policy": policy,
                "baseline_policy": settings.baseline_policy,
                "outcome": settings.outcome,
                "split": settings.split,
                "budget_fraction": float(settings.budget_fraction),
                "metric": settings.metric,
                "mean_delta": mean_delta,
                "ci_lower": ci_lower,
                "ci_upper": row.get("ci_upper"),
                "passed_selection_gate": ci_lower > 0.0,
                "is_champion": False,
                "reason": None,
            }
        )

    return sorted(explanation_rows, key=lambda item: str(item["policy"]))


def _finite_number(
    value: Any,
    field_name: str,
    row: Mapping[str, Any],
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Bootstrap field '{field_name}' must be numeric for policy "
            f"'{row.get('policy')}'. Received: {value!r}."
        ) from error

    if not math.isfinite(number):
        raise ValueError(
            f"Bootstrap field '{field_name}' must be finite for policy "
            f"'{row.get('policy')}'. Received: {value!r}."
        )

    return number


def _explain_policy_row(
    row: Mapping[str, Any],
    champion_policy: str,
    fallback_policy: str,
) -> str:
    policy = str(row["policy"])
    if policy == champion_policy:
        return (
            "Selected because ci_lower > 0 and it had the largest mean_delta "
            "among passing policies, with alphabetical policy name as the "
            "deterministic tie-break."
        )
    if champion_policy == fallback_policy:
        return "Not selected because ci_lower <= 0."
    if bool(row["passed_selection_gate"]):
        return (
            "Passed ci_lower > 0 but was not selected because another passing "
            "policy had a larger mean_delta or won the alphabetical tie-break."
        )
    return "Not selected because ci_lower <= 0."
