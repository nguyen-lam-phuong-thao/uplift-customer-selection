"""Bootstrap sample generation for Top-K policy evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from uplift_modeling.data.row_id import (
    align_frames_by_row_id,
)
from uplift_modeling.evaluation.bootstrap_config import (
    BOOTSTRAP_METRICS,
    DEFAULT_BASELINE_POLICY,
    DEFAULT_BOOTSTRAP_RANDOM_SEED,
    DEFAULT_N_BOOTSTRAP,
    validate_bootstrap_config,
)
from uplift_modeling.evaluation.topk_policy import (
    TOPK_BUDGET_FRACTIONS,
    calculate_topk_policy_metrics,
)


__all__ = [
    "BOOTSTRAP_METRICS",
    "DEFAULT_BASELINE_POLICY",
    "DEFAULT_BOOTSTRAP_RANDOM_SEED",
    "DEFAULT_N_BOOTSTRAP",
    "calculate_bootstrap_policy_metric_samples",
    "validate_bootstrap_config",
]


def _get_split_frames(
    policy_frames: dict[str, pd.DataFrame],
    split: str,
) -> dict[str, pd.DataFrame]:
    split_frames = {}

    for policy_name, policy_frame in policy_frames.items():
        split_frame = policy_frame.loc[
            policy_frame["split"] == split
        ].reset_index(drop=True)
        if not split_frame.empty:
            split_frames[policy_name] = split_frame

    return split_frames


def _validate_paired_split_frames(
    split: str,
    split_frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], int]:
    row_counts = {
        policy_name: len(split_frame)
        for policy_name, split_frame in split_frames.items()
    }
    unique_row_counts = set(row_counts.values())
    if len(unique_row_counts) != 1:
        raise ValueError(
            "Policy prediction frames must contain the same row count for "
            f"split '{split}' to reuse paired bootstrap samples. "
            f"Received: {row_counts}"
        )

    aligned_frames = align_frames_by_row_id(
        split_frames,
        label_columns=("treatment", "outcome", "split"),
        context=f"Policy prediction frames for paired bootstrap split '{split}'",
    )
    return aligned_frames, unique_row_counts.pop()


def _calculate_bootstrap_metrics_or_none(
    sampled_frame: pd.DataFrame,
    budget_fraction: float,
) -> dict[str, float | int | None]:
    """Return Top-K metrics, or null metrics for invalid bootstrap samples."""
    try:
        return calculate_topk_policy_metrics(
            sampled_frame,
            budget_fraction=budget_fraction,
            require_unique_row_id=False,
        )
    except ValueError as error:
        message = str(error)
        invalid_sample_messages = (
            "Prediction frame must contain both treatment and control rows.",
            "treatment rate must be strictly between 0 and 1.",
            "Selected rows must contain both treatment and control rows.",
        )
        if not any(text in message for text in invalid_sample_messages):
            raise
        return {
            "n_selected": None,
            "policy_value": None,
            "incremental_outcome": None,
        }


def calculate_bootstrap_policy_metric_samples(
    policy_frames: dict[str, pd.DataFrame],
    budget_fractions: Iterable[float] = TOPK_BUDGET_FRACTIONS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = DEFAULT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Calculate per-iteration Top-K bootstrap metrics.

    Each bootstrap iteration samples from a deterministic row_id ordering once
    per split and applies that same sampled row_id vector to every policy frame.
    """
    fractions = validate_bootstrap_config(
        budget_fractions=budget_fractions,
        n_bootstrap=n_bootstrap,
    )
    if not policy_frames:
        raise ValueError("At least one policy frame is required.")

    split_names = sorted(
        {
            str(split)
            for policy_frame in policy_frames.values()
            for split in policy_frame["split"].unique()
        }
    )
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []

    for split in split_names:
        split_frames = _get_split_frames(policy_frames, split=split)
        if not split_frames:
            continue

        aligned_split_frames, row_count = _validate_paired_split_frames(
            split=split,
            split_frames=split_frames,
        )
        for bootstrap_iteration in range(n_bootstrap):
            sample_positions = rng.integers(
                low=0,
                high=row_count,
                size=row_count,
            )

            for policy_name, split_frame in aligned_split_frames.items():
                sampled_frame = split_frame.iloc[sample_positions].reset_index(
                    drop=True
                )
                for budget_fraction in fractions:
                    metrics = _calculate_bootstrap_metrics_or_none(
                        sampled_frame,
                        budget_fraction=budget_fraction,
                    )
                    rows.append(
                        {
                            "bootstrap_iteration": int(bootstrap_iteration),
                            "policy": policy_name,
                            "split": split,
                            "budget_fraction": float(budget_fraction),
                            "budget_pct": float(budget_fraction * 100),
                            "n_selected": metrics["n_selected"],
                            "policy_value": metrics["policy_value"],
                            "incremental_outcome": metrics[
                                "incremental_outcome"
                            ],
                        }
                    )

    return pd.DataFrame(rows)
