"""Deterministic model-selection gate for bootstrap policy contrasts."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from uplift_modeling.artifacts.json import save_json_artifact
from uplift_modeling.models.scoring import (
    RESPONSE_MODEL_KIND,
    validate_model_artifact_identity,
)
LOGGER = logging.getLogger(__name__)
MODEL_SELECTION_ARTIFACT_NAME = "model_selection_gate"
UPLIFT_SELECTION_METHOD = "primary_topk_metric_largest_value"
REPLACEMENT_GATE_METHOD = "bootstrap_paired_contrast_ci_lower_positive"
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
    """Primary validation settings for uplift selection and replacement gate."""

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



def select_uplift_champion_from_topk_rows(
    topk_rows: Sequence[Mapping[str, Any]],
    uplift_candidate_policies: Sequence[str],
    settings: SelectionGateSettings,
) -> dict[str, Any]:
    """Select the best uplift candidate at the primary validation operating point."""
    candidate_policies = tuple(sorted(set(uplift_candidate_policies)))
    if not candidate_policies:
        raise ValueError("At least one uplift candidate policy is required.")

    if settings.baseline_policy in candidate_policies:
        raise ValueError("The response baseline must not participate in uplift selection.")

    primary_rows = []
    for row in topk_rows:
        policy = str(row.get("policy_name", ""))
        if policy not in candidate_policies or str(row.get("split")) != settings.split:
            continue

        budget_fraction = _finite_number(
            row.get("budget_fraction"),
            "budget_fraction",
            row,
        )
        if not math.isclose(
            budget_fraction,
            settings.budget_fraction,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            continue

        if settings.metric not in row:
            raise ValueError(
                f"Top-K row for policy '{policy}' is missing metric "
                f"'{settings.metric}'."
            )

        primary_rows.append(
            {
                "policy": policy,
                "metric_value": _finite_number(
                    row[settings.metric],
                    settings.metric,
                    row,
                ),
            }
        )

    policy_counts = Counter(row["policy"] for row in primary_rows)
    missing = sorted(set(candidate_policies).difference(policy_counts))
    duplicates = sorted(policy for policy, count in policy_counts.items() if count > 1)

    if missing:
        raise ValueError(f"Missing primary Top-K rows for uplift candidates: {missing}.")
    if duplicates:
        raise ValueError(f"Duplicate primary Top-K rows for uplift candidates: {duplicates}.")

    champion_row = min(
        primary_rows,
        key=lambda row: (-float(row["metric_value"]), str(row["policy"])),
    )

    return {
        "uplift_champion_policy": str(champion_row["policy"]),
        "uplift_selection_method": UPLIFT_SELECTION_METHOD,
        "uplift_candidate_rows": sorted(
            primary_rows,
            key=lambda row: str(row["policy"]),
        ),
    }


def evaluate_replacement_gate_from_paired_contrasts(
    contrast_rows: Sequence[Mapping[str, Any]],
    uplift_champion_policy: str,
    settings: SelectionGateSettings,
) -> dict[str, Any]:
    """Evaluate whether the uplift champion can replace the response baseline."""
    _validate_required_fields(contrast_rows)
    _validate_baseline_present(contrast_rows, settings.baseline_policy)

    primary_rows = _filter_primary_rows(contrast_rows, settings)
    _validate_duplicate_primary_rows(primary_rows)

    gate_rows = _build_replacement_gate_rows(primary_rows, settings)
    champion_rows = [
        row for row in gate_rows
        if row["policy"] == uplift_champion_policy
    ]

    if len(champion_rows) != 1:
        raise ValueError(
            "Expected exactly one primary bootstrap contrast for uplift "
            f"champion '{uplift_champion_policy}', received "
            f"{len(champion_rows)}."
        )

    contrast_row = champion_rows[0]
    gate_passed = bool(contrast_row["replacement_gate_passed"])

    return {
        "baseline_policy": settings.baseline_policy,
        "replacement_gate_method": REPLACEMENT_GATE_METHOD,
        "replacement_gate_passed": gate_passed,
        "recommended_deployment_policy": (
            uplift_champion_policy
            if gate_passed
            else settings.baseline_policy
        ),
        "baseline_contrast_row": contrast_row,
    }


def save_model_selection_gate(
    metric_dir: Path,
    dataset_name: str,
    experiment_id: str,
    settings: SelectionGateSettings,
    source_manifest_path: Path,
    topk_rows: Sequence[Mapping[str, Any]],
    topk_json_path: Path,
    model_artifacts: Mapping[str, Any] | None = None,
    bootstrap_payload: Mapping[str, Any] | None = None,
    bootstrap_json_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Save validation uplift-selection and replacement-gate results."""
    if bootstrap_payload is None:
        if bootstrap_json_path is None:
            raise ValueError(
                "Either bootstrap_payload or bootstrap_json_path is required."
            )
        bootstrap_payload = load_bootstrap_payload(bootstrap_json_path)

    contrast_rows = bootstrap_payload.get("paired_contrast_rows")
    if not isinstance(contrast_rows, list):
        raise ValueError(
            "Bootstrap payload must contain paired_contrast_rows as a list."
        )

    if not isinstance(model_artifacts, Mapping):
        raise ValueError(
            "Model artifacts are required to save a Selection Gate artifact."
        )

    model_kinds = {}
    for policy, model_artifact in model_artifacts.items():
        if not isinstance(model_artifact, Mapping):
            raise ValueError(
                f"Invalid model provenance for policy '{policy}'."
            )

        policy = str(policy)
        model_kinds[policy] = validate_model_artifact_identity(
            policy=policy,
            model_artifact=model_artifact,
        )

    baseline_model_kind = model_kinds.get(settings.baseline_policy)
    if baseline_model_kind != RESPONSE_MODEL_KIND:
        raise ValueError(
            f"Baseline policy '{settings.baseline_policy}' must be a response model."
        )

    uplift_candidate_policies = tuple(
        sorted(
            policy
            for policy, model_kind in model_kinds.items()
            if model_kind != RESPONSE_MODEL_KIND
        )
    )

    uplift_selection = select_uplift_champion_from_topk_rows(
        topk_rows=topk_rows,
        uplift_candidate_policies=uplift_candidate_policies,
        settings=settings,
    )
    uplift_champion_policy = str(
        uplift_selection["uplift_champion_policy"]
    )

    replacement_gate = evaluate_replacement_gate_from_paired_contrasts(
        contrast_rows=contrast_rows,
        uplift_champion_policy=uplift_champion_policy,
        settings=settings,
    )

    uplift_champion_model_artifact = model_artifacts[
        uplift_champion_policy
    ]

    output_path = metric_dir / (
        f"{dataset_name}_{settings.outcome}_{experiment_id}_"
        f"{MODEL_SELECTION_ARTIFACT_NAME}.json"
    )
    if output_path.exists():
        raise FileExistsError(
            "Selection Gate artifact already exists for experiment "
            f"'{experiment_id}' and cannot be overwritten: {output_path}"
        )

    payload = {
        "artifact_type": MODEL_SELECTION_ARTIFACT_NAME,
        "experiment_id": experiment_id,
        "dataset_name": dataset_name,
        "source_manifest_artifact": source_manifest_path.name,
        "topk_artifact": topk_json_path.name,
        "bootstrap_artifact": (
            bootstrap_json_path.name
            if bootstrap_json_path is not None
            else None
        ),
        "selection_settings": asdict(settings),
        "uplift_champion_model_artifact": dict(
            uplift_champion_model_artifact
        ),
        **uplift_selection,
        **replacement_gate,
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


def _build_replacement_gate_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    settings: SelectionGateSettings,
) -> list[dict[str, Any]]:
    gate_rows = []

    for row in primary_rows:
        policy = str(row["policy"])
        if policy == settings.baseline_policy:
            continue

        n_valid_bootstrap = row.get("n_valid_bootstrap")
        if (
            not isinstance(n_valid_bootstrap, int)
            or isinstance(n_valid_bootstrap, bool)
            or n_valid_bootstrap < 0
        ):
            raise ValueError(
                "Bootstrap field 'n_valid_bootstrap' must be an integer "
                f"greater than or equal to zero for policy '{policy}'. "
                f"Received: {n_valid_bootstrap!r}."
            )

        if "n_bootstrap" in row:
            n_bootstrap = row.get("n_bootstrap")
            if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool):
                raise ValueError(
                    "Bootstrap field 'n_bootstrap' must be an integer for "
                    f"policy '{policy}'. Received: {n_bootstrap!r}."
                )
            if n_valid_bootstrap > n_bootstrap:
                raise ValueError(
                    "'n_valid_bootstrap' must be less than or equal to "
                    f"'n_bootstrap' for policy '{policy}'."
                )

        if n_valid_bootstrap == 0:
            mean_delta = ci_lower = ci_upper = None
            gate_passed = False
        else:
            mean_delta = _finite_number(row["mean_delta"], "mean_delta", row)
            ci_lower = _finite_number(row["ci_lower"], "ci_lower", row)
            ci_upper = row.get("ci_upper")
            gate_passed = ci_lower > 0.0

        gate_rows.append(
            {
                "policy": policy,
                "baseline_policy": settings.baseline_policy,
                "outcome": settings.outcome,
                "split": settings.split,
                "budget_fraction": float(settings.budget_fraction),
                "metric": settings.metric,
                "mean_delta": mean_delta,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_bootstrap": row.get("n_bootstrap"),
                "n_valid_bootstrap": n_valid_bootstrap,
                "replacement_gate_passed": gate_passed,
            }
        )

    return sorted(gate_rows, key=lambda row: str(row["policy"]))


def _finite_number(
    value: Any,
    field_name: str,
    row: Mapping[str, Any],
) -> float:
    policy = row.get("policy", row.get("policy_name"))

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Field '{field_name}' must be numeric for policy "
            f"'{policy}'. Received: {value!r}."
        ) from error

    if not math.isfinite(number):
        raise ValueError(
            f"Field '{field_name}' must be finite for policy "
            f"'{policy}'. Received: {value!r}."
        )

    return number


