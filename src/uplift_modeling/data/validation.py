"""Generic validation for prepared uplift modeling tables."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from uplift_modeling.data.dataset_spec import DatasetSpec
from uplift_modeling.data.row_id import (
    ROW_ID_COLUMN,
    validate_row_id_column,
)

def validate_prepared_dataset_contract(
    dataframe: pd.DataFrame,
    dataset_spec: DatasetSpec,
    *,
    require_row_id: bool = False,
) -> dict[str, Any]:
    """Validate a prepared modeling table against the dataset contract.

    This function assumes the user already finished EDA and feature engineering.
    It only checks framework-level requirements.
    """
    if dataframe.empty:
        raise ValueError("Prepared dataset must not be empty.")

    required_columns = [
        *dataset_spec.feature_columns,
        dataset_spec.treatment_column,
        *dataset_spec.outcome_columns,
    ]

    if require_row_id:
        required_columns.insert(0, ROW_ID_COLUMN)

    _validate_required_columns(dataframe, required_columns)

    if require_row_id:
        validate_row_id_column(
            dataframe,
            context="Standardized dataset",
        )

    _validate_binary_column(
        dataframe,
        dataset_spec.treatment_column,
        label="treatment column",
    )

    for outcome_column in dataset_spec.outcome_columns:
        _validate_binary_column(
            dataframe,
            outcome_column,
            label=f"outcome column '{outcome_column}'",
        )

    non_numeric_features = [
        column
        for column in dataset_spec.feature_columns
        if not is_numeric_dtype(dataframe[column])
    ]
    if non_numeric_features:
        raise ValueError(
            "Prepared feature columns must be numeric. "
            f"Non-numeric columns: {non_numeric_features}. "
            "Encode categorical features in the notebook before using the framework."
        )

    null_columns = [
        column
        for column in required_columns
        if dataframe[column].isna().any()
    ]
    if null_columns:
        raise ValueError(
            "Prepared dataset contains null values in required columns: "
            f"{null_columns}. Handle missing values before standardization."
        )

    return {
        "is_valid": True,
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "feature_columns": list(dataset_spec.feature_columns),
        "outcome_columns": list(dataset_spec.outcome_columns),
        "treatment_column": dataset_spec.treatment_column,
        "row_id_column": ROW_ID_COLUMN,
        "split_column": dataset_spec.split_column,
    }


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing_columns = sorted(set(required_columns).difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "Prepared dataset is missing required columns: "
            f"{missing_columns}"
        )


def _validate_binary_column(
    dataframe: pd.DataFrame,
    column: str,
    label: str,
) -> None:
    observed_values = set(dataframe[column].dropna().unique().tolist())
    invalid_values = sorted(observed_values.difference({0, 1}), key=str)

    if invalid_values:
        raise ValueError(
            f"Prepared {label} must contain only 0/1 values. "
            f"Invalid values: {invalid_values}"
        )