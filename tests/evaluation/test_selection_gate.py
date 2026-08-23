"""Focused tests for validation uplift selection and replacement gating."""
import json
from pathlib import Path

from uplift_modeling.evaluation.selection_gate import (
    SelectionGateSettings,
    evaluate_replacement_gate_from_paired_contrasts,
    save_model_selection_gate,
    select_uplift_champion_from_topk_rows,
)

SETTINGS = SelectionGateSettings(
    outcome="visit",
    split="validation",
    budget_fraction=0.05,
    metric="policy_value",
    baseline_policy="baseline",
)


def _topk_row(policy: str, policy_value: float) -> dict:
    return {
        "policy_name": policy,
        "split": "validation",
        "budget_fraction": 0.05,
        "policy_value": policy_value,
    }


def _contrast_row(
    policy: str,
    *,
    ci_lower,
    mean_delta=0.02,
    n_valid_bootstrap: int = 100,
) -> dict:
    return {
        "policy": policy,
        "baseline_policy": "baseline",
        "outcome": "visit",
        "split": "validation",
        "budget_fraction": 0.05,
        "metric": "policy_value",
        "mean_delta": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": None if ci_lower is None else ci_lower + 0.01,
        "n_bootstrap": 100,
        "n_valid_bootstrap": n_valid_bootstrap,
    }


def test_selects_largest_validation_topk_metric() -> None:
    result = select_uplift_champion_from_topk_rows(
        topk_rows=[
            _topk_row("candidate_a", 0.10),
            _topk_row("candidate_b", 0.12),
            _topk_row("baseline", 0.20),
        ],
        uplift_candidate_policies=("candidate_a", "candidate_b"),
        settings=SETTINGS,
    )

    assert result["uplift_champion_policy"] == "candidate_b"


def test_exact_topk_tie_is_broken_by_policy_name() -> None:
    result = select_uplift_champion_from_topk_rows(
        topk_rows=[
            _topk_row("z_policy", 0.10),
            _topk_row("a_policy", 0.10),
        ],
        uplift_candidate_policies=("z_policy", "a_policy"),
        settings=SETTINGS,
    )

    assert result["uplift_champion_policy"] == "a_policy"


def test_replacement_gate_recommends_champion_when_ci_lower_is_positive() -> None:
    result = evaluate_replacement_gate_from_paired_contrasts(
        contrast_rows=[_contrast_row("candidate", ci_lower=0.01)],
        uplift_champion_policy="candidate",
        settings=SETTINGS,
    )

    assert result["replacement_gate_passed"] is True
    assert result["recommended_deployment_policy"] == "candidate"


def test_zero_valid_bootstrap_falls_back_to_baseline() -> None:
    result = evaluate_replacement_gate_from_paired_contrasts(
        contrast_rows=[
            _contrast_row(
                "candidate",
                ci_lower=None,
                mean_delta=None,
                n_valid_bootstrap=0,
            ),
        ],
        uplift_champion_policy="candidate",
        settings=SETTINGS,
    )

    assert result["replacement_gate_passed"] is False
    assert result["recommended_deployment_policy"] == "baseline"
    assert result["baseline_contrast_row"]["ci_lower"] is None

def test_saved_selection_gate_uses_portable_artifact_references(
    tmp_path: Path,
) -> None:
    manifest_path = (
        tmp_path
        / "synthetic_visit_exp001_experiment_manifest.json"
    )
    manifest_path.write_text("{}", encoding="utf-8")

    topk_path = (
        tmp_path
        / "synthetic_visit_topk_policy_evaluation_run01.json"
    )
    topk_path.write_text("{}", encoding="utf-8")

    bootstrap_path = (
        tmp_path
        / "synthetic_visit_bootstrap.json"
    )
    bootstrap_path.write_text("{}", encoding="utf-8")

    settings = SelectionGateSettings(
        outcome="visit",
        split="validation",
        budget_fraction=0.05,
        metric="policy_value",
        baseline_policy="baseline",
    )

    topk_rows = [
        _topk_row("candidate", 0.12),
        _topk_row("baseline", 0.10),
    ]

    bootstrap_payload = {
        "paired_contrast_rows": [
            _contrast_row(
                "candidate",
                ci_lower=0.01,
            )
        ]
    }

    model_artifacts = {
        "baseline": {
            "artifact_type": "model_provenance",
            "dataset_name": "synthetic",
            "outcome": "visit",
            "policy_name": "baseline",
            "prediction_artifact": "baseline.parquet",
            "model_kind": "response",
            "mlflow_run_id": "baseline-run",
            "model_uri": "runs:/baseline-run/model",
        },
        "candidate": {
            "artifact_type": "model_provenance",
            "dataset_name": "synthetic",
            "outcome": "visit",
            "policy_name": "candidate",
            "prediction_artifact": "candidate.parquet",
            "model_kind": "t_learner",
            "mlflow_run_id": "candidate-run",
            "treatment_model_uri":
                "runs:/candidate-run/treatment_model",
            "control_model_uri":
                "runs:/candidate-run/control_model",
        },
    }

    output_path, payload = save_model_selection_gate(
        metric_dir=tmp_path,
        dataset_name="synthetic",
        experiment_id="exp001",
        settings=settings,
        source_manifest_path=manifest_path,
        topk_rows=topk_rows,
        topk_json_path=topk_path,
        model_artifacts=model_artifacts,
        bootstrap_payload=bootstrap_payload,
        bootstrap_json_path=bootstrap_path,
    )

    assert output_path.exists()

    assert (
        payload["source_manifest_artifact"]
        == manifest_path.name
    )
    assert payload["topk_artifact"] == topk_path.name
    assert (
        payload["bootstrap_artifact"]
        == bootstrap_path.name
    )

    assert "/kaggle/" not in json.dumps(payload)