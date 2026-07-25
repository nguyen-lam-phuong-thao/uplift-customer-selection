"""Dataset preparation decisions for Phase 1 Criteo modeling."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from uplift_modeling.data.criteo import FEATURE_COLUMNS


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> None:
    """Raise a clear error when required preparation columns are missing."""
    missing_columns = sorted(set(required_columns).difference(dataframe.columns))

    if missing_columns:
        missing_columns_text = ", ".join(missing_columns)
        raise ValueError(
            "Cannot build Criteo decision dataset because columns are missing: "
            f"{missing_columns_text}"
        )


def _validate_split_sizes(
    train_size: float,
    validation_size: float,
    test_size: float,
) -> None:
    """Validate train, validation, and test proportions."""
    split_sizes = {
        "train_size": train_size,
        "validation_size": validation_size,
        "test_size": test_size,
    }

    invalid_sizes = {
        name: value
        for name, value in split_sizes.items()
        if value <= 0 or value >= 1
    }

    if invalid_sizes:
        raise ValueError(
            "Split sizes must be between 0 and 1. "
            f"Received: {invalid_sizes}"
        )

    total_size = train_size + validation_size + test_size
    if abs(total_size - 1.0) > 1e-9:
        raise ValueError(
            "Split sizes must sum to 1. "
            f"Received total: {total_size}"
        )


def _joint_stratification_key(
    dataframe: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
) -> pd.Series:
    """Build the treatment-outcome stratification key."""
    required_columns = (treatment_column, outcome_column)
    _validate_columns(dataframe, required_columns)

    return (
        dataframe[treatment_column].astype(str)
        + "_"
        + dataframe[outcome_column].astype(str)
    )


def build_decision_frame(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    treatment_column: str = "treatment",
    outcome_columns: tuple[str, ...] = ("visit", "conversion"),
) -> pd.DataFrame:
    """Select columns approved by Layer 3 decisions.

    The returned frame keeps only anonymized pre-treatment features, treatment,
    and outcomes. It intentionally excludes `exposure` because it is observed
    after treatment and is leakage-prone.
    """
    selected_columns = (*feature_columns, treatment_column, *outcome_columns)
    _validate_columns(dataframe, selected_columns)

    return dataframe.loc[:, selected_columns].copy()


def assign_stratified_split(
    dataframe: pd.DataFrame,
    outcome_column: str,
    treatment_column: str = "treatment",
    train_size: float = 0.6,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    split_column: str = "split",
) -> pd.DataFrame:
    """Assign deterministic train, validation, and test labels.

    Splits are stratified by treatment and the selected outcome.
    """
    _validate_split_sizes(train_size, validation_size, test_size)
    _validate_columns(dataframe, (treatment_column, outcome_column))

    decision_dataset = dataframe.copy()
    row_indices = decision_dataset.index.to_series()
    stratification_key = _joint_stratification_key(
        decision_dataset,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
    )

    train_indices, holdout_indices = train_test_split(
        row_indices,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratification_key,
    )

    holdout_dataset = decision_dataset.loc[holdout_indices]
    holdout_stratification_key = _joint_stratification_key(
        holdout_dataset,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
    )
    validation_share_of_holdout = validation_size / (
        validation_size + test_size
    )

    validation_indices, test_indices = train_test_split(
        holdout_indices,
        train_size=validation_share_of_holdout,
        random_state=random_state,
        shuffle=True,
        stratify=holdout_stratification_key,
    )

    decision_dataset[split_column] = ""
    decision_dataset.loc[train_indices, split_column] = "train"
    decision_dataset.loc[validation_indices, split_column] = "validation"
    decision_dataset.loc[test_indices, split_column] = "test"

    return decision_dataset


def build_decision_dataset(
    dataframe: pd.DataFrame,
    outcome_column: str,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    treatment_column: str = "treatment",
    train_size: float = 0.6,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build one outcome-specific Criteo decision dataset."""
    selected_columns = (*feature_columns, treatment_column, outcome_column)
    _validate_columns(dataframe, selected_columns)

    outcome_dataset = dataframe.loc[:, selected_columns].copy()
    return assign_stratified_split(
        outcome_dataset,
        outcome_column=outcome_column,
        treatment_column=treatment_column,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
    )


def save_decision_dataset(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a decision dataset as parquet and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
    return path
