"""Split helpers for prepared uplift modeling datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from uplift_modeling.data.dataset_spec import (
    DatasetSpec,
    SplitConfig,
    VALID_SPLITS,
    validate_supported_outcome,
)


def ensure_split_column(
    dataframe: pd.DataFrame,
    outcome_column: str,
    dataset_spec: DatasetSpec,
    split_config: SplitConfig,
) -> pd.DataFrame:
    """Return a dataframe with a valid train/validation/test split column."""
    if dataset_spec.split_column in dataframe.columns:
        validate_split_column(dataframe, dataset_spec)
        return dataframe.copy()

    if not split_config.assign_if_missing:
        raise ValueError(
            f"Prepared dataset does not contain split column "
            f"'{dataset_spec.split_column}', and split.assign_if_missing is false."
        )

    return assign_stratified_split(
        dataframe=dataframe,
        outcome_column=outcome_column,
        dataset_spec=dataset_spec,
        split_config=split_config,
    )


def validate_split_column(
    dataframe: pd.DataFrame,
    dataset_spec: DatasetSpec,
) -> None:
    """Validate an existing split column."""
    split_column = dataset_spec.split_column

    if split_column not in dataframe.columns:
        raise ValueError(f"Missing split column: {split_column}")

    if dataframe[split_column].isna().any():
        raise ValueError(f"Split column '{split_column}' must not contain null values.")

    observed_splits = set(dataframe[split_column].astype(str).unique().tolist())
    invalid_splits = sorted(observed_splits.difference(VALID_SPLITS))
    if invalid_splits:
        raise ValueError(
            f"Split column '{split_column}' contains invalid values: "
            f"{invalid_splits}. Expected only: {list(VALID_SPLITS)}"
        )

    missing_splits = sorted(set(VALID_SPLITS).difference(observed_splits))
    if missing_splits:
        raise ValueError(
            f"Split column '{split_column}' must contain all required splits. "
            f"Missing: {missing_splits}"
        )


def assign_stratified_split(
    dataframe: pd.DataFrame,
    outcome_column: str,
    dataset_spec: DatasetSpec,
    split_config: SplitConfig,
) -> pd.DataFrame:
    """Assign deterministic train/validation/test labels.

    Splits are stratified by treatment and the selected outcome.
    """
    validate_supported_outcome(dataset_spec, outcome_column)

    required_columns = [
        dataset_spec.treatment_column,
        outcome_column,
    ]
    missing_columns = sorted(set(required_columns).difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "Cannot assign split because required columns are missing: "
            f"{missing_columns}"
        )

    output = dataframe.copy()
    row_positions = np.arange(len(output))

    stratification_key = _joint_stratification_key(
        output,
        treatment_column=dataset_spec.treatment_column,
        outcome_column=outcome_column,
    )

    try:
        train_positions, holdout_positions = train_test_split(
            row_positions,
            train_size=split_config.train_size,
            random_state=split_config.random_state,
            shuffle=True,
            stratify=stratification_key.to_numpy(),
        )

        holdout_frame = output.iloc[holdout_positions]
        holdout_key = _joint_stratification_key(
            holdout_frame,
            treatment_column=dataset_spec.treatment_column,
            outcome_column=outcome_column,
        )

        validation_share_of_holdout = split_config.validation_size / (
            split_config.validation_size + split_config.test_size
        )

        validation_positions, test_positions = train_test_split(
            holdout_positions,
            train_size=validation_share_of_holdout,
            random_state=split_config.random_state,
            shuffle=True,
            stratify=holdout_key.to_numpy(),
        )
    except ValueError as error:
        raise ValueError(
            "Cannot assign stratified split. Each treatment/outcome group must "
            "have enough rows for train, validation, and test. "
            "For very small datasets, create the split manually in the notebook "
            "and include the split column in the prepared table."
        ) from error

    split_values = np.empty(len(output), dtype=object)
    split_values[train_positions] = "train"
    split_values[validation_positions] = "validation"
    split_values[test_positions] = "test"

    output[dataset_spec.split_column] = split_values
    validate_split_column(output, dataset_spec)
    return output


def _joint_stratification_key(
    dataframe: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
) -> pd.Series:
    return (
        dataframe[treatment_column].astype(str)
        + "_"
        + dataframe[outcome_column].astype(str)
    )