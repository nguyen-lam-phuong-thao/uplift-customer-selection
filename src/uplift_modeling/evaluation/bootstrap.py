"""Bootstrap sample generation for Top-K policy evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

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
from uplift_modeling.evaluation.uplift_metrics import ROW_ID_COLUMN


_COMPATIBILITY_EXPORTS = {
    "calculate_bootstrap_policy_rows": (
        "uplift_modeling.evaluation.bootstrap_summary",
        "calculate_bootstrap_policy_rows",
    ),
    "summarize_bootstrap_paired_contrasts": (
        "uplift_modeling.evaluation.bootstrap_summary",
        "summarize_bootstrap_paired_contrasts",
    ),
    "summarize_bootstrap_policy_metric_samples": (
        "uplift_modeling.evaluation.bootstrap_summary",
        "summarize_bootstrap_policy_metric_samples",
    ),
    "find_next_bootstrap_run_number": (
        "uplift_modeling.evaluation.bootstrap_writer",
        "find_next_bootstrap_run_number",
    ),
    "save_bootstrap_policy_evaluation": (
        "uplift_modeling.evaluation.bootstrap_writer",
        "save_bootstrap_policy_evaluation",
    ),
}
__all__ = [
    "BOOTSTRAP_METRICS",
    "DEFAULT_BASELINE_POLICY",
    "DEFAULT_BOOTSTRAP_RANDOM_SEED",
    "DEFAULT_N_BOOTSTRAP",
    "calculate_bootstrap_policy_metric_samples",
    "validate_bootstrap_config",
    *_COMPATIBILITY_EXPORTS,
]


def __getattr__(name: str) -> Any:
    """Lazily resolve legacy bootstrap exports without import-time cycles."""
    if name not in _COMPATIBILITY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attribute_name = _COMPATIBILITY_EXPORTS[name]
    attribute = getattr(import_module(module_name), attribute_name)
    globals()[name] = attribute
    return attribute


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
) -> int:
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

    reference_policy = sorted(split_frames)[0]
    has_row_id = {
        policy_name: ROW_ID_COLUMN in split_frame.columns
        for policy_name, split_frame in split_frames.items()
    }
    if any(has_row_id.values()) and not all(has_row_id.values()):
        missing_row_id = sorted(
            policy_name
            for policy_name, contains_row_id in has_row_id.items()
            if not contains_row_id
        )
        raise ValueError(
            "Policy prediction frames must be row-aligned for paired "
            "bootstrap evaluation. Some policy frames contain row_id and "
            f"others do not for split '{split}'. Policies missing row_id: "
            f"{missing_row_id}."
        )

    alignment_columns = ["treatment", "outcome", "split"]
    if all(has_row_id.values()):
        alignment_columns.append(ROW_ID_COLUMN)

    reference_labels = split_frames[reference_policy][
        alignment_columns
    ].reset_index(drop=True)

    for policy_name, split_frame in split_frames.items():
        labels = split_frame[alignment_columns].reset_index(
            drop=True
        )
        if not labels.equals(reference_labels):
            raise ValueError(
                "Policy prediction frames must be row-aligned for paired "
                "bootstrap evaluation. Expected matching columns "
                f"{alignment_columns}. "
                f"Policy '{policy_name}' is not aligned to "
                f"'{reference_policy}' for split '{split}'."
            )

    return unique_row_counts.pop()


def calculate_bootstrap_policy_metric_samples(
    policy_frames: dict[str, pd.DataFrame],
    budget_fractions: Iterable[float] = TOPK_BUDGET_FRACTIONS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = DEFAULT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Calculate per-iteration Top-K bootstrap metrics.

    Each bootstrap iteration samples row positions once per split and applies
    that same sampled index vector to every available policy frame.
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

        row_count = _validate_paired_split_frames(
            split=split,
            split_frames=split_frames,
        )
        for bootstrap_iteration in range(n_bootstrap):
            sample_positions = rng.integers(
                low=0,
                high=row_count,
                size=row_count,
            )

            for policy_name, split_frame in split_frames.items():
                sampled_frame = split_frame.iloc[sample_positions].reset_index(
                    drop=True
                )
                for budget_fraction in fractions:
                    metrics = calculate_topk_policy_metrics(
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
                            "n_selected": int(metrics["n_selected"]),
                            "policy_value": metrics["policy_value"],
                            "incremental_outcome": metrics[
                                "incremental_outcome"
                            ],
                        }
                    )

    return pd.DataFrame(rows)
