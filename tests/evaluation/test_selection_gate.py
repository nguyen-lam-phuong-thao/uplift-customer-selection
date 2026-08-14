"""Focused tests for validation uplift selection and replacement gating."""

from uplift_modeling.evaluation.selection_gate import (
    SelectionGateSettings,
    evaluate_replacement_gate_from_paired_contrasts,
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
