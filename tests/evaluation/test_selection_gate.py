"""Tests for bootstrap-based model-selection gate logic."""

import json

import pytest

from uplift_modeling.evaluation.selection_gate import (
    SelectionGateSettings,
    save_model_selection_gate,
    select_champion_from_bootstrap_payload,
)


SETTINGS = SelectionGateSettings(
    outcome="visit",
    split="validation",
    budget_fraction=0.05,
    metric="policy_value",
    baseline_policy="treated_response_lgbm",
)


def _row(
    policy: str,
    mean_delta: float,
    ci_lower: float,
    budget_fraction: float = 0.05,
) -> dict[str, object]:
    """Build one primary paired-contrast row."""
    return {
        "policy": policy,
        "baseline_policy": "treated_response_lgbm",
        "split": "validation",
        "outcome": "visit",
        "budget_fraction": budget_fraction,
        "budget_pct": budget_fraction * 100,
        "metric": "policy_value",
        "mean_delta": mean_delta,
        "std_delta": 0.01,
        "ci_lower": ci_lower,
        "ci_upper": mean_delta + 0.02,
        "n_bootstrap": 1000,
        "random_seed": 42,
    }


def _payload(rows: list[dict[str, object]]) -> dict[str, object]:
    """Build a bootstrap payload with paired-contrast rows."""
    return {"paired_contrast_rows": rows}


def test_selection_gate_chooses_largest_mean_delta_passing_policy() -> None:
    """Among passing policies, largest mean_delta wins."""
    result = select_champion_from_bootstrap_payload(
        _payload(
            [
                _row("x_learner_lgbm", mean_delta=0.10, ci_lower=0.01),
                _row("t_learner_lgbm", mean_delta=0.20, ci_lower=0.02),
            ]
        ),
        SETTINGS,
    )

    assert result["champion_policy"] == "t_learner_lgbm"
    assert [
        row["policy"]
        for row in result["explanation_rows"]
        if row["is_champion"]
    ] == ["t_learner_lgbm"]
    assert all("selection_reason" in row for row in result["explanation_rows"])
    assert all("reason" not in row for row in result["explanation_rows"])


def test_selection_gate_ties_break_by_policy_name() -> None:
    """Equal passing mean_delta values use alphabetical policy order."""
    result = select_champion_from_bootstrap_payload(
        _payload(
            [
                _row("x_learner_lgbm", mean_delta=0.20, ci_lower=0.01),
                _row("t_learner_lgbm", mean_delta=0.20, ci_lower=0.02),
            ]
        ),
        SETTINGS,
    )

    assert result["champion_policy"] == "t_learner_lgbm"


def test_selection_gate_falls_back_to_baseline_when_no_policy_passes() -> None:
    """Baseline is selected only as fallback when no candidate passes."""
    result = select_champion_from_bootstrap_payload(
        _payload(
            [
                _row("x_learner_lgbm", mean_delta=0.10, ci_lower=0.00),
                _row("t_learner_lgbm", mean_delta=0.20, ci_lower=-0.01),
            ]
        ),
        SETTINGS,
    )

    assert result["champion_policy"] == "treated_response_lgbm"
    assert all(not row["is_champion"] for row in result["explanation_rows"])


def test_selection_gate_requires_bootstrap_fields() -> None:
    """Missing required paired-contrast fields raise a clear error."""
    broken_row = _row("x_learner_lgbm", mean_delta=0.10, ci_lower=0.01)
    del broken_row["ci_lower"]

    with pytest.raises(ValueError, match="missing required fields"):
        select_champion_from_bootstrap_payload(_payload([broken_row]), SETTINGS)


def test_selection_gate_requires_matching_baseline() -> None:
    """A missing configured baseline raises a clear error."""
    row = _row("x_learner_lgbm", mean_delta=0.10, ci_lower=0.01)
    row["baseline_policy"] = "pooled_response_lgbm"

    with pytest.raises(ValueError, match="Baseline policy"):
        select_champion_from_bootstrap_payload(_payload([row]), SETTINGS)


def test_selection_gate_requires_primary_rows() -> None:
    """No primary setting match raises a clear error."""
    with pytest.raises(ValueError, match="No bootstrap paired-contrast rows"):
        select_champion_from_bootstrap_payload(
            _payload(
                [
                    _row(
                        "x_learner_lgbm",
                        0.10,
                        0.01,
                        budget_fraction=0.10,
                    )
                ]
            ),
            SETTINGS,
        )


def test_selection_gate_rejects_duplicate_primary_rows() -> None:
    """Duplicate rows for the same primary policy are rejected."""
    with pytest.raises(ValueError, match="Duplicate primary"):
        select_champion_from_bootstrap_payload(
            _payload(
                [
                    _row("x_learner_lgbm", mean_delta=0.10, ci_lower=0.01),
                    _row("x_learner_lgbm", mean_delta=0.11, ci_lower=0.02),
                ]
            ),
            SETTINGS,
        )


def test_selection_gate_rejects_invalid_numeric_values() -> None:
    """Primary numeric values must be valid finite numbers."""
    row = _row("x_learner_lgbm", mean_delta=0.10, ci_lower=0.01)
    row["mean_delta"] = None

    with pytest.raises(ValueError, match="mean_delta"):
        select_champion_from_bootstrap_payload(_payload([row]), SETTINGS)


def test_save_model_selection_gate_writes_json_artifact(tmp_path) -> None:
    """Selection gate saves the selected champion and its exact provenance."""
    bootstrap_path = tmp_path / "bootstrap_paired_contrasts.json"
    bootstrap_path.write_text(
        json.dumps(_payload([_row("x_learner_lgbm", 0.10, 0.01)])),
        encoding="utf-8",
    )

    model_artifacts = {
        "x_learner_lgbm": {
            "artifact_type": "model_provenance",
            "dataset_name": "criteo",
            "outcome": "visit",
            "policy_name": "x_learner_lgbm",
            "prediction_artifact": (
                "criteo_visit_x_learner_lgbm_run01_predictions.parquet"
            ),
            "model_kind": "x_learner",
            "mlflow_run_id": "x-run-01",
            "tau0_model_uri": "runs:/x-run-01/tau0_model",
            "tau1_model_uri": "runs:/x-run-01/tau1_model",
            "constant_treatment_rate_weight": 0.5,
        }
    }

    output_path, payload = save_model_selection_gate(
        metric_dir=tmp_path,
        dataset_name="criteo",
        settings=SETTINGS,
        experiment_id="exp-001",
        model_artifacts=model_artifacts,
        bootstrap_json_path=bootstrap_path,
    )

    assert output_path.exists()
    assert output_path.name == "criteo_visit_exp-001_model_selection_gate.json"
    assert payload["champion_policy"] == "x_learner_lgbm"
    assert (
        payload["champion_model_artifact"]["mlflow_run_id"]
        == "x-run-01"
    )
    assert (
        payload["champion_model_artifact"]["tau0_model_uri"]
        == "runs:/x-run-01/tau0_model"
    )

def test_save_model_selection_gate_does_not_overwrite_experiment(
    tmp_path,
) -> None:
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text(
        json.dumps(
            _payload(
                [_row("x_learner_lgbm", 0.10, 0.01)]
            )
        ),
        encoding="utf-8",
    )

    model_artifacts = {
        "x_learner_lgbm": {
            "artifact_type": "model_provenance",
            "dataset_name": "criteo",
            "outcome": "visit",
            "policy_name": "x_learner_lgbm",
            "prediction_artifact": "x_predictions.parquet",
            "model_kind": "x_learner",
            "mlflow_run_id": "x-run-01",
            "tau0_model_uri": "runs:/x-run-01/tau0_model",
            "tau1_model_uri": "runs:/x-run-01/tau1_model",
            "constant_treatment_rate_weight": 0.5,
        }
    }

    save_model_selection_gate(
        metric_dir=tmp_path,
        dataset_name="criteo",
        experiment_id="exp-001",
        settings=SETTINGS,
        model_artifacts=model_artifacts,
        bootstrap_json_path=bootstrap_path,
    )

    with pytest.raises(
        FileExistsError,
        match="cannot be overwritten",
    ):
        save_model_selection_gate(
            metric_dir=tmp_path,
            dataset_name="criteo",
            experiment_id="exp-001",
            settings=SETTINGS,
            model_artifacts=model_artifacts,
            bootstrap_json_path=bootstrap_path,
        )    