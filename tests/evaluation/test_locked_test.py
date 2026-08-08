"""Focused tests for locked-test evaluation helpers."""

import json
from pathlib import Path

import pandas as pd
import pytest

from uplift_modeling.evaluation.locked_test import (
    get_champion_policy,
    load_locked_test_prediction_frames,
    load_selection_gate_payload,
    resolve_required_prediction_paths,
    validate_selection_gate_payload,
)


def _model_artifact(policy: str) -> dict:
    return {
        "artifact_type": "model_provenance",
        "dataset_name": "synthetic",
        "outcome": "visit",
        "policy_name": policy,
        "prediction_artifact": "prediction.parquet",
        "model_kind": "t_learner",
        "mlflow_run_id": "run-001",
        "treatment_model_uri": "runs:/run-001/treatment_model",
        "control_model_uri": "runs:/run-001/control_model",
    }


def _selection_payload(policy: str = "champion") -> dict:
    return {
        "artifact_type": "model_selection_gate",
        "experiment_id": "exp-001",
        "dataset_name": "synthetic",
        "source_manifest_path": "/tmp/manifest.json",
        "champion_policy": policy,
        "selection_settings": {
            "outcome": "visit",
            "split": "validation",
            "budget_fraction": 0.05,
            "metric": "policy_value",
            "baseline_policy": "baseline",
        },
        "champion_model_artifact": _model_artifact(policy),
    }


def _prediction_frame(policy: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "treatment": [1, 0, 1, 0],
            "outcome": [1, 0, 0, 1],
            "split": ["test"] * 4,
            "score": [0.9, 0.8, 0.7, 0.6],
            "model_name": [policy] * 4,
        }
    )


def test_selection_gate_must_come_from_validation() -> None:
    payload = _selection_payload()
    payload["selection_settings"]["split"] = "test"

    with pytest.raises(ValueError, match="originate from validation"):
        validate_selection_gate_payload(
            selection_payload=payload,
            outcome="visit",
        )


def test_locked_test_requires_only_selected_champion(tmp_path: Path) -> None:
    champion_path = tmp_path / "champion.parquet"
    other_path = tmp_path / "other.parquet"

    resolved = resolve_required_prediction_paths(
        manifest_prediction_paths={
            "champion": champion_path,
            "other": other_path,
        },
        dataset_name="synthetic",
        outcome="visit",
        champion_policy="champion",
    )

    assert resolved == {"champion": champion_path}


def test_locked_test_loads_test_rows_and_aligns_by_row_id(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"

    _prediction_frame("first").to_parquet(first_path, index=False)
    _prediction_frame("second").sort_values(
        "row_id",
        ascending=False,
    ).to_parquet(second_path, index=False)

    frames = load_locked_test_prediction_frames(
        {
            "first": first_path,
            "second": second_path,
        }
    )

    assert frames["first"]["row_id"].tolist() == [1, 2, 3, 4]
    assert frames["second"]["row_id"].tolist() == [1, 2, 3, 4]
    assert set(frames["first"]["split"]) == {"test"}


def test_locked_test_rejects_mixed_split_artifact(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "champion.parquet"

    frame = _prediction_frame("champion")
    frame.loc[0, "split"] = "validation"
    frame.to_parquet(prediction_path, index=False)

    with pytest.raises(
        ValueError,
        match="must contain only 'test' rows",
    ):
        load_locked_test_prediction_frames(
            {
                "champion": prediction_path,
            }
        )


def test_selection_gate_loader_rejects_wrong_artifact_type(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection.json"
    payload = _selection_payload()
    payload["artifact_type"] = "wrong_type"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_type"):
        load_selection_gate_payload(path)