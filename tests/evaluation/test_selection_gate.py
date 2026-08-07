"""Focused tests for deterministic champion selection."""

from uplift_modeling.evaluation.selection_gate import (
    SelectionGateSettings,
    select_champion_from_paired_contrasts,
)


SETTINGS = SelectionGateSettings(
    outcome="visit",
    split="validation",
    budget_fraction=0.05,
    metric="policy_value",
    baseline_policy="baseline",
)


def _row(
    policy: str,
    *,
    mean_delta,
    ci_lower,
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


def test_selects_largest_supported_improvement() -> None:
    result = select_champion_from_paired_contrasts(
        [
            _row("candidate_a", mean_delta=0.02, ci_lower=0.01),
            _row("candidate_b", mean_delta=0.03, ci_lower=0.01),
        ],
        SETTINGS,
    )

    assert result["champion_policy"] == "candidate_b"


def test_falls_back_to_baseline_when_no_candidate_passes() -> None:
    result = select_champion_from_paired_contrasts(
        [
            _row("candidate_a", mean_delta=0.01, ci_lower=-0.01),
            _row("candidate_b", mean_delta=-0.01, ci_lower=-0.02),
        ],
        SETTINGS,
    )

    assert result["champion_policy"] == "baseline"


def test_zero_valid_bootstrap_does_not_crash_selection() -> None:
    result = select_champion_from_paired_contrasts(
        [
            _row(
                "candidate",
                mean_delta=None,
                ci_lower=None,
                n_valid_bootstrap=0,
            ),
        ],
        SETTINGS,
    )

    assert result["champion_policy"] == "baseline"
    explanation = result["explanation_rows"][0]
    assert explanation["mean_delta"] is None
    assert explanation["ci_lower"] is None
    assert explanation["passed_selection_gate"] is False


def test_exact_tie_is_broken_by_policy_name() -> None:
    result = select_champion_from_paired_contrasts(
        [
            _row("z_policy", mean_delta=0.02, ci_lower=0.01),
            _row("a_policy", mean_delta=0.02, ci_lower=0.01),
        ],
        SETTINGS,
    )

    assert result["champion_policy"] == "a_policy"