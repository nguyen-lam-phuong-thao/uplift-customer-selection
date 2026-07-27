"""Reusable uplift and policy evaluation metrics for ranked predictions.

The shared prediction contract treats `score` as a ranking score. Response
Model scores are response probabilities, while later T-Learner and X-Learner
scores are expected to be uplift scores.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


PREDICTION_COLUMNS: tuple[str, ...] = (
    "treatment",
    "outcome",
    "split",
    "score",
    "model_name",
)


def validate_prediction_frame(
    predictions: pd.DataFrame,
    required_columns: Iterable[str] = PREDICTION_COLUMNS,
) -> None:
    """Validate the shared prediction-frame contract for ranked evaluation."""
    missing_columns = sorted(
        set(required_columns).difference(predictions.columns)
    )

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Prediction frame is missing columns: {missing_text}")

    if predictions.empty:
        raise ValueError("Prediction frame must contain at least one row.")

    binary_columns = ("treatment", "outcome")
    non_numeric_columns = [
        column
        for column in (*binary_columns, "score")
        if not is_numeric_dtype(predictions[column])
    ]

    if non_numeric_columns:
        non_numeric_text = ", ".join(non_numeric_columns)
        raise ValueError(
            "Prediction frame columns must be numeric: "
            f"{non_numeric_text}"
        )

    score_values = predictions["score"].to_numpy(dtype=float)
    if not np.isfinite(score_values).all():
        raise ValueError("Prediction frame score values must be finite.")

    for column in binary_columns:
        invalid_values = sorted(
            set(predictions[column].unique()).difference({0, 1})
        )

        if invalid_values:
            raise ValueError(
                f"Prediction frame {column} values must be 0 or 1. "
                f"Received: {invalid_values}"
            )

    treatment_values = set(predictions["treatment"].unique())
    if treatment_values != {0, 1}:
        raise ValueError(
            "Prediction frame must contain both treatment and control rows."
        )


def build_uplift_curve(
    predictions: pd.DataFrame,
    num_points: int | None = 100,
) -> pd.DataFrame:
    """Build a sampled cumulative curve ordered by descending ranking score."""
    validate_prediction_frame(predictions)

    if num_points is not None and num_points <= 0:
        raise ValueError(
            "num_points must be a positive integer or None. "
            f"Received: {num_points}"
        )

    ranked = predictions.sort_values(
        "score",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    treatment = ranked["treatment"].to_numpy(dtype=float)
    outcome = ranked["outcome"].to_numpy(dtype=float)
    treated = treatment == 1.0
    control = treatment == 0.0

    treated_count = np.cumsum(treated)
    control_count = np.cumsum(control)
    treated_outcome = np.cumsum(outcome * treated)
    control_outcome = np.cumsum(outcome * control)
    treated_rate = np.divide(
        treated_outcome,
        treated_count,
        out=np.zeros_like(treated_outcome, dtype=float),
        where=treated_count > 0,
    )
    control_rate = np.divide(
        control_outcome,
        control_count,
        out=np.zeros_like(control_outcome, dtype=float),
        where=control_count > 0,
    )
    expected_control_outcome = np.divide(
        control_outcome * treated_count,
        control_count,
        out=np.zeros_like(control_outcome, dtype=float),
        where=control_count > 0,
    )
    cumulative_incremental_outcome = (
        treated_outcome - expected_control_outcome
    )

    row_count = len(ranked)
    ranks = np.arange(1, row_count + 1)
    curve = pd.DataFrame(
        {
            "rank": ranks,
            "population_fraction": ranks / row_count,
            "treated_count": treated_count.astype(int),
            "control_count": control_count.astype(int),
            "treated_outcome_count": treated_outcome,
            "control_outcome_count": control_outcome,
            "treated_outcome_rate": treated_rate,
            "control_outcome_rate": control_rate,
            "uplift": treated_rate - control_rate,
            "cumulative_incremental_outcome": cumulative_incremental_outcome,
        }
    )

    if num_points is None or row_count <= num_points:
        return curve

    sampled_indices = np.unique(
        np.ceil(np.linspace(1, row_count, num_points)).astype(int) - 1
    )
    return curve.iloc[sampled_indices].reset_index(drop=True)


def _trapezoid_area(
    y_values: np.ndarray,
    x_values: np.ndarray,
) -> float:
    """Calculate trapezoid area across NumPy versions."""
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(y_values, x_values))


def calculate_auuc(curve: pd.DataFrame) -> float:
    """Calculate area under the cumulative incremental outcome curve."""
    population_fraction = np.insert(
        curve["population_fraction"].to_numpy(),
        0,
        0.0,
    )
    cumulative_incremental_outcome = np.insert(
        curve["cumulative_incremental_outcome"].to_numpy(),
        0,
        0.0,
    )

    return _trapezoid_area(
        y_values=cumulative_incremental_outcome,
        x_values=population_fraction,
    )


def calculate_qini(curve: pd.DataFrame) -> float:
    """Calculate Qini as model AUUC minus the random-ranking baseline."""
    final_incremental_outcome = float(
        curve["cumulative_incremental_outcome"].iloc[-1]
    )
    population_fraction = np.insert(
        curve["population_fraction"].to_numpy(),
        0,
        0.0,
    )
    random_baseline = population_fraction * final_incremental_outcome
    baseline_area = _trapezoid_area(
        y_values=random_baseline,
        x_values=population_fraction,
    )
    return calculate_auuc(curve) - baseline_area


def calculate_policy_value(
    predictions: pd.DataFrame,
    top_fraction: float = 0.3,
) -> float:
    """Estimate outcome-based policy value for treating top-ranked rows.

    This is not final net business value. It uses observed outcomes and the
    treatment rate, without value-per-outcome or contact-cost adjustments.
    """
    validate_prediction_frame(predictions)

    if top_fraction <= 0 or top_fraction > 1:
        raise ValueError(
            "top_fraction must be greater than 0 and at most 1. "
            f"Received: {top_fraction}"
        )

    ranked = predictions.sort_values(
        "score",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    selected_count = max(1, int(np.ceil(len(ranked) * top_fraction)))
    selected = np.zeros(len(ranked), dtype=bool)
    selected[:selected_count] = True

    treatment = ranked["treatment"].to_numpy(dtype=float)
    outcome = ranked["outcome"].to_numpy(dtype=float)
    treatment_rate = float(treatment.mean())

    treated_match = selected & (treatment == 1.0)
    control_match = ~selected & (treatment == 0.0)
    value = (
        (outcome * treated_match).sum() / treatment_rate
        + (outcome * control_match).sum() / (1.0 - treatment_rate)
    ) / len(ranked)

    return float(value)


def calculate_uplift_metrics(
    predictions: pd.DataFrame,
    top_fraction: float = 0.3,
) -> dict[str, float | int]:
    """Calculate uplift metrics from the full prediction ranking."""
    curve = build_uplift_curve(predictions, num_points=None)

    return {
        "qini": calculate_qini(curve),
        "auuc": calculate_auuc(curve),
        "cumulative_incremental_outcome": float(
            curve["cumulative_incremental_outcome"].iloc[-1]
        ),
        "policy_value": calculate_policy_value(
            predictions,
            top_fraction=top_fraction,
        ),
        "positive_rate": float(predictions["outcome"].mean()),
        "treatment_rate": float(predictions["treatment"].mean()),
        "row_count": int(len(predictions)),
    }
