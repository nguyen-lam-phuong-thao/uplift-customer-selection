"""Shared configuration for Top-K bootstrap policy evaluation."""

from __future__ import annotations

from collections.abc import Iterable


DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_BOOTSTRAP_RANDOM_SEED = 42
DEFAULT_BASELINE_POLICY = "treated_response_lgbm"
BOOTSTRAP_METRICS: tuple[str, ...] = (
    "policy_value",
    "incremental_outcome",
)


def validate_bootstrap_config(
    budget_fractions: Iterable[float],
    n_bootstrap: int,
) -> tuple[float, ...]:
    """Validate bootstrap settings and return budget fractions as a tuple."""
    if n_bootstrap <= 0:
        raise ValueError(
            "n_bootstrap must be a positive integer. "
            f"Received: {n_bootstrap}"
        )

    fractions = tuple(float(fraction) for fraction in budget_fractions)
    if not fractions:
        raise ValueError("At least one budget fraction is required.")

    invalid_fractions = [
        fraction
        for fraction in fractions
        if fraction <= 0.0 or fraction > 1.0
    ]
    if invalid_fractions:
        raise ValueError(
            "Budget fractions must be greater than 0 and at most 1. "
            f"Received: {invalid_fractions}"
        )

    return fractions
