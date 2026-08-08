"""Locked test-set reporting for the selected champion policy."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from uplift_modeling.artifacts.json import save_json_artifact
from uplift_modeling.data.row_id import align_frames_by_row_id
from uplift_modeling.evaluation.bootstrap import (
    calculate_bootstrap_policy_metric_samples,
)
from uplift_modeling.evaluation.bootstrap_config import (
    DEFAULT_BOOTSTRAP_RANDOM_SEED,
    DEFAULT_N_BOOTSTRAP,
)
from uplift_modeling.evaluation.bootstrap_summary import (
    summarize_bootstrap_policy_metric_samples,
)
from uplift_modeling.evaluation.topk_policy import (
    SELECTION_SPLIT,
    TOPK_BUDGET_FRACTIONS,
    calculate_topk_policy_metrics,
)
from uplift_modeling.evaluation.uplift_metrics import (
    PREDICTION_COLUMNS,
    calculate_uplift_metrics,
    validate_prediction_frame,
)
from uplift_modeling.models.scoring import validate_model_artifact_identity


LOGGER = logging.getLogger(__name__)
LOCKED_TEST_ARTIFACT_NAME = "locked_test_evaluation"
LOCKED_TEST_SPLIT = "test"
SELECTION_ARTIFACT_TYPE = "model_selection_gate"


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
    artifact_type = payload.get("artifact_type")
    if artifact_type != SELECTION_ARTIFACT_TYPE:
        raise ValueError(
            "Selection Gate artifact_type must be "
            f"'{SELECTION_ARTIFACT_TYPE}'. Received: {artifact_type}"
        )
    source_manifest_path = payload.get("source_manifest_path")
    if not isinstance(source_manifest_path, str) or not source_manifest_path:
        raise ValueError(
            "Selection Gate artifact must contain a non-empty "
            "'source_manifest_path'."
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


def validate_selection_gate_payload(
    selection_payload: Mapping[str, Any],
    outcome: str,
) -> str:
    """Validate Selection Gate metadata and return its champion policy."""
    champion_policy = get_champion_policy(selection_payload)
    selection_settings = selection_payload.get("selection_settings")
    if not isinstance(selection_settings, Mapping):
        raise ValueError(
            "Selection Gate artifact must contain "
            "'selection_settings' as an object."
        )
    if selection_settings.get("split") != SELECTION_SPLIT:
        raise ValueError(
            "Selection Gate artifact must originate from validation. "
            f"Received split: {selection_settings.get('split')!r}."
        )
    if selection_settings.get("outcome") != outcome:
        raise ValueError(
            "Selection Gate artifact and runtime must have the same outcome. "
            f"Expected '{outcome}', received "
            f"{selection_settings.get('outcome')!r}."
        )

    champion_model_artifact = selection_payload.get("champion_model_artifact")
    if not isinstance(champion_model_artifact, Mapping):
        raise ValueError(
            "Selection Gate artifact must contain "
            "'champion_model_artifact' as an object."
        )
    validate_model_artifact_identity(
        policy=champion_policy,
        model_artifact=champion_model_artifact,
    )
    return champion_policy


def resolve_required_prediction_paths(
    manifest_prediction_paths: Mapping[str, Path],
    dataset_name: str,
    outcome: str,
    champion_policy: str,
) -> dict[str, Path]:
    """Resolve the champion locked-test prediction artifact."""
    prediction_path = manifest_prediction_paths.get(champion_policy)
    if prediction_path is None:
        raise ValueError(
            "Champion prediction artifact is missing for policy "
            f"'{champion_policy}', outcome '{outcome}', and dataset "
            f"'{dataset_name}' in the experiment manifest."
        )

    return {champion_policy: prediction_path}


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

        observed_splits = {
            str(split)
            for split in frame["split"].unique()
        }

        if observed_splits != {LOCKED_TEST_SPLIT}:
            raise ValueError(
                f"{policy} locked-test prediction artifact must contain "
                f"only '{LOCKED_TEST_SPLIT}' rows. "
                f"Received splits: {sorted(observed_splits)}."
            )

        test_frame = frame.copy()

        test_frame["artifact_name"] = prediction_path.name
        test_frame["policy_name"] = policy
        validate_prediction_frame(test_frame)
        policy_frames[policy] = test_frame

    return align_frames_by_row_id(
        policy_frames,
        label_columns=("treatment", "outcome", "split"),
        context="Locked-test prediction frames",
    )


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


def build_locked_test_evaluation_path(
    metric_dir: Path,
    dataset_name: str,
    outcome: str,
    experiment_id: str,
) -> Path:
    """Return the deterministic locked-test evaluation JSON path."""
    return metric_dir / (
        f"{dataset_name}_{outcome}_{experiment_id}_"
        f"{LOCKED_TEST_ARTIFACT_NAME}.json"
    )


def build_locked_test_prediction_path(
    prediction_dir: Path,
    dataset_name: str,
    outcome: str,
    champion_policy: str,
    experiment_id: str,
) -> Path:
    """Return the deterministic champion locked-test prediction path."""
    return prediction_dir / (
        f"{dataset_name}_{outcome}_{champion_policy}_{experiment_id}_"
        "locked_test_predictions.parquet"
    )


def load_locked_test_evaluation_payload(output_path: Path) -> dict[str, Any]:
    """Load an existing locked-test evaluation JSON artifact."""
    with output_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise ValueError(
            "Locked-test evaluation artifact must contain a JSON object: "
            f"{output_path}"
        )

    return payload


def validate_locked_test_evaluation_identity(
    payload: Mapping[str, Any],
    experiment_id: str,
    champion_policy: str,
    champion_model_artifact: Mapping[str, Any],
) -> None:
    """Validate that an existing final report belongs to the same champion."""
    artifact_type = payload.get("artifact_type")
    if artifact_type != LOCKED_TEST_ARTIFACT_NAME:
        raise ValueError(
            "Locked-test evaluation artifact_type must be "
            f"'{LOCKED_TEST_ARTIFACT_NAME}'. Received: {artifact_type}"
        )

    if payload.get("experiment_id") != experiment_id:
        raise ValueError(
            "Existing locked-test evaluation belongs to a different "
            "experiment_id."
        )
    if payload.get("champion_policy") != champion_policy:
        raise ValueError(
            "Existing locked-test evaluation belongs to a different "
            "champion_policy."
        )
    if payload.get("champion_model_artifact") != dict(champion_model_artifact):
        raise ValueError(
            "Existing locked-test evaluation belongs to a different "
            "champion_model_artifact."
        )


def load_existing_locked_test_evaluation(
    output_path: Path,
    experiment_id: str,
    champion_policy: str,
    champion_model_artifact: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Load and validate an existing deterministic final evaluation."""
    payload = load_locked_test_evaluation_payload(output_path)
    validate_locked_test_evaluation_identity(
        payload=payload,
        experiment_id=experiment_id,
        champion_policy=champion_policy,
        champion_model_artifact=champion_model_artifact,
    )
    return output_path, payload


def save_locked_test_evaluation_artifact(
    metric_dir: Path,
    dataset_name: str,
    outcome: str,
    experiment_id: str,
    champion_policy: str,
    champion_model_artifact: Mapping[str, Any],
    selection_artifact_path: Path,
    prediction_paths: Mapping[str, Path],
    uplift_metrics: Mapping[str, float | int],
    locked_test_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    n_bootstrap: int,
    random_seed: int,
    budget_fractions: Iterable[float] = TOPK_BUDGET_FRACTIONS,
) -> tuple[Path, dict[str, Any]]:
    """Save the locked-test evaluation JSON artifact."""
    output_path = build_locked_test_evaluation_path(
        metric_dir=metric_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        experiment_id=experiment_id,
    )
    if output_path.exists():
        raise FileExistsError(
            "Locked-test evaluation JSON already exists and will not be "
            f"overwritten: {output_path}"
        )

    payload = {
        "artifact_type": LOCKED_TEST_ARTIFACT_NAME,
        "experiment_id": experiment_id,
        "dataset_name": dataset_name,
        "outcome": outcome,
        "split": LOCKED_TEST_SPLIT,
        "champion_policy": champion_policy,
        "champion_model_artifact": dict(champion_model_artifact),
        "selection_artifact": selection_artifact_path.name,
        "prediction_artifacts": {
            policy: prediction_path.name
            for policy, prediction_path in sorted(prediction_paths.items())
        },
        "budget_fractions": [float(fraction) for fraction in budget_fractions],
        "uplift_metrics": dict(uplift_metrics),
        "locked_test_rows": locked_test_rows,
        "bootstrap": {
            "n_bootstrap": int(n_bootstrap),
            "random_seed": int(random_seed),
            "rows": bootstrap_rows,
        },
    }

    save_json_artifact(payload, output_path)
    LOGGER.info("Saved locked-test evaluation JSON to %s", output_path)
    return output_path, payload


def _save_locked_test_evaluation(
    manifest_prediction_paths: Mapping[str, Path],
    metric_dir: Path,
    dataset_name: str,
    outcome: str,
    selection_artifact_path: Path,
    budget_fractions: Iterable[float] = TOPK_BUDGET_FRACTIONS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = DEFAULT_BOOTSTRAP_RANDOM_SEED,
) -> tuple[Path, dict[str, Any]]:
    """Load fixed champion predictions and save locked-test reporting JSON."""
    selection_payload = load_selection_gate_payload(selection_artifact_path)
    champion_policy = validate_selection_gate_payload(
        selection_payload=selection_payload,
        outcome=outcome,
    )
    experiment_id = selection_payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError(
            "Selection Gate artifact must contain a non-empty "
            "'experiment_id'."
        )
    champion_model_artifact = selection_payload["champion_model_artifact"]
    output_path = build_locked_test_evaluation_path(
        metric_dir=metric_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        experiment_id=experiment_id,
    )
    if output_path.exists():
        return load_existing_locked_test_evaluation(
            output_path=output_path,
            experiment_id=experiment_id,
            champion_policy=champion_policy,
            champion_model_artifact=champion_model_artifact,
        )

    fractions = tuple(float(fraction) for fraction in budget_fractions)
    prediction_paths = resolve_required_prediction_paths(
        manifest_prediction_paths=manifest_prediction_paths,
        dataset_name=dataset_name,
        outcome=outcome,
        champion_policy=champion_policy,
    )
    policy_frames = load_locked_test_prediction_frames(prediction_paths)
    champion_frame = policy_frames[champion_policy]
    uplift_metrics = calculate_uplift_metrics(champion_frame)
    locked_test_rows = calculate_locked_test_rows(
        policy_frames=policy_frames,
        outcome=outcome,
        budget_fractions=fractions,
    )
    bootstrap_samples = calculate_bootstrap_policy_metric_samples(
        policy_frames={champion_policy: champion_frame},
        budget_fractions=fractions,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    bootstrap_rows = summarize_bootstrap_policy_metric_samples(
        samples=bootstrap_samples,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )

    return save_locked_test_evaluation_artifact(
        metric_dir=metric_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        experiment_id=experiment_id,
        champion_policy=champion_policy,
        champion_model_artifact=champion_model_artifact,
        selection_artifact_path=selection_artifact_path,
        prediction_paths=prediction_paths,
        uplift_metrics=uplift_metrics,
        locked_test_rows=locked_test_rows,
        bootstrap_rows=bootstrap_rows,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        budget_fractions=fractions,
    )
