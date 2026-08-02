"""Summary statistics for Top-K bootstrap policy evaluation."""

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from uplift_modeling.evaluation.bootstrap import (
    calculate_bootstrap_policy_metric_samples,
)
from uplift_modeling.evaluation.bootstrap_config import (
    BOOTSTRAP_METRICS,
    DEFAULT_BASELINE_POLICY,
    DEFAULT_BOOTSTRAP_RANDOM_SEED,
    DEFAULT_N_BOOTSTRAP,
)
from uplift_modeling.evaluation.topk_policy import (
    DEFAULT_EVALUATION_SPLITS,
    TOPK_BUDGET_FRACTIONS,
    validate_standard_evaluation_splits,
)


LOGGER = logging.getLogger(__name__)


def _summary_value(value: float) -> float | None:
    if np.isnan(value):
        return None
    return float(value)


def _summarize_values(
    values: pd.Series,
    n_bootstrap: int,
    random_seed: int,
) -> dict[str, float | int | None]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite_values = array[np.isfinite(array)]
    n_valid_bootstrap = int(len(finite_values))

    if n_valid_bootstrap == 0:
        return {
            "mean": None,
            "std": None,
            "ci_lower": None,
            "ci_upper": None,
            "n_bootstrap": int(n_bootstrap),
            "n_valid_bootstrap": 0,
            "random_seed": int(random_seed),
        }

    if n_valid_bootstrap == 1:
        value = float(finite_values[0])
        return {
            "mean": value,
            "std": None,
            "ci_lower": value,
            "ci_upper": value,
            "n_bootstrap": int(n_bootstrap),
            "n_valid_bootstrap": 1,
            "random_seed": int(random_seed),
        }

    return {
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values, ddof=1)),
        "ci_lower": _summary_value(np.percentile(finite_values, 2.5)),
        "ci_upper": _summary_value(np.percentile(finite_values, 97.5)),
        "n_bootstrap": int(n_bootstrap),
        "n_valid_bootstrap": n_valid_bootstrap,
        "random_seed": int(random_seed),
    }


def summarize_bootstrap_policy_metric_samples(
    samples: pd.DataFrame,
    n_bootstrap: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Summarize bootstrap policy metrics into percentile CIs."""
    rows: list[dict[str, Any]] = []
    group_columns = ["policy", "split", "budget_fraction", "budget_pct"]

    for group_values, group in samples.groupby(group_columns, sort=True):
        policy, split, budget_fraction, budget_pct = group_values
        for metric in BOOTSTRAP_METRICS:
            rows.append(
                {
                    "policy": str(policy),
                    "split": str(split),
                    "budget_fraction": float(budget_fraction),
                    "budget_pct": float(budget_pct),
                    "metric": metric,
                    **_summarize_values(
                        group[metric],
                        n_bootstrap=n_bootstrap,
                        random_seed=random_seed,
                    ),
                }
            )

    return rows


def summarize_bootstrap_paired_contrasts(
    samples: pd.DataFrame,
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = DEFAULT_BOOTSTRAP_RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Summarize paired bootstrap contrasts against a baseline policy."""
    if baseline_policy not in set(samples["policy"]):
        warning = (
            "Paired bootstrap contrasts skipped because baseline policy "
            f"'{baseline_policy}' is missing."
        )
        LOGGER.warning(warning)
        return [], [warning]

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    index_columns = [
        "bootstrap_iteration",
        "split",
        "budget_fraction",
        "budget_pct",
    ]
    baseline = samples.loc[
        samples["policy"] == baseline_policy,
        [*index_columns, *BOOTSTRAP_METRICS],
    ]
    policy_names = sorted(
        policy
        for policy in samples["policy"].unique()
        if policy != baseline_policy
    )

    for policy_name in policy_names:
        policy_samples = samples.loc[
            samples["policy"] == policy_name,
            [*index_columns, *BOOTSTRAP_METRICS],
        ]
        paired = policy_samples.merge(
            baseline,
            on=index_columns,
            how="inner",
            suffixes=("_policy", "_baseline"),
        )

        expected_rows = len(policy_samples)
        if len(paired) != expected_rows:
            warning = (
                "Some paired bootstrap rows were unavailable for policy "
                f"'{policy_name}' against baseline '{baseline_policy}'. "
                f"Expected {expected_rows}, paired {len(paired)}."
            )
            LOGGER.warning(warning)
            warnings.append(warning)

        for group_values, group in paired.groupby(
            ["split", "budget_fraction", "budget_pct"],
            sort=True,
        ):
            split, budget_fraction, budget_pct = group_values
            for metric in BOOTSTRAP_METRICS:
                delta_values = (
                    group[f"{metric}_policy"]
                    - group[f"{metric}_baseline"]
                )
                summary = _summarize_values(
                    delta_values,
                    n_bootstrap=n_bootstrap,
                    random_seed=random_seed,
                )
                rows.append(
                    {
                        "policy": str(policy_name),
                        "baseline_policy": baseline_policy,
                        "split": str(split),
                        "outcome": None,
                        "budget_fraction": float(budget_fraction),
                        "budget_pct": float(budget_pct),
                        "metric": metric,
                        "mean_delta": summary["mean"],
                        "std_delta": summary["std"],
                        "ci_lower": summary["ci_lower"],
                        "ci_upper": summary["ci_upper"],
                        "n_bootstrap": int(n_bootstrap),
                        "n_valid_bootstrap": summary["n_valid_bootstrap"],
                        "random_seed": int(random_seed),
                    }
                )

    return rows, warnings


def _normalize_requested_splits(
    bootstrap_splits: Iterable[str] | None,
) -> tuple[str, ...] | None:
    if bootstrap_splits is None:
        return None

    requested_splits = tuple(str(split) for split in bootstrap_splits)
    if not requested_splits:
        raise ValueError("At least one bootstrap split is required.")

    return requested_splits


def _filter_policy_frames_to_splits(
    policy_frames: dict[str, pd.DataFrame],
    bootstrap_splits: Iterable[str] | None,
) -> dict[str, pd.DataFrame]:
    requested_splits = validate_standard_evaluation_splits(
        (
            DEFAULT_EVALUATION_SPLITS
            if bootstrap_splits is None
            else _normalize_requested_splits(bootstrap_splits)
        )
    )

    requested_split_set = set(requested_splits)
    available_splits = {
        str(split)
        for policy_frame in policy_frames.values()
        for split in policy_frame["split"].unique()
    }
    missing_splits = sorted(requested_split_set.difference(available_splits))
    if missing_splits:
        available_text = ", ".join(sorted(available_splits)) or "none"
        missing_text = ", ".join(missing_splits)
        raise ValueError(
            "Requested bootstrap split(s) are missing from policy frames: "
            f"{missing_text}. Available splits: {available_text}."
        )

    return {
        policy_name: policy_frame.loc[
            policy_frame["split"].isin(requested_split_set)
        ].copy()
        for policy_name, policy_frame in policy_frames.items()
    }


def calculate_bootstrap_policy_rows(
    policy_frames: dict[str, pd.DataFrame],
    budget_fractions: tuple[float, ...] = TOPK_BUDGET_FRACTIONS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = DEFAULT_BOOTSTRAP_RANDOM_SEED,
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
    bootstrap_splits: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Calculate regular and paired bootstrap summary rows."""
    filtered_policy_frames = _filter_policy_frames_to_splits(
        policy_frames=policy_frames,
        bootstrap_splits=bootstrap_splits,
    )
    samples = calculate_bootstrap_policy_metric_samples(
        policy_frames=filtered_policy_frames,
        budget_fractions=budget_fractions,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    policy_rows = summarize_bootstrap_policy_metric_samples(
        samples=samples,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    contrast_rows, warnings = summarize_bootstrap_paired_contrasts(
        samples=samples,
        baseline_policy=baseline_policy,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    return policy_rows, contrast_rows, warnings
