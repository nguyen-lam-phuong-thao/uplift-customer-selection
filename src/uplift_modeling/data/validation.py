from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from uplift_modeling.data.criteo import (
    BINARY_COLUMNS,
    CRITEO_COLUMNS,
    FEATURE_COLUMNS,
)


def _to_python_value(value: Any) -> Any:
    """Convert pandas and NumPy scalar values to plain Python values."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def _series_to_plain_dict(series: pd.Series) -> dict[str, Any]:
    """Convert a pandas Series with column-name indexes to a dictionary."""
    return {
        str(key): _to_python_value(value)
        for key, value in series.items()
    }


def _validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Raise an error when the confirmed Criteo schema is incomplete."""
    missing_columns = sorted(
        set(CRITEO_COLUMNS).difference(dataframe.columns)
    )

    if missing_columns:
        missing_columns_text = ", ".join(missing_columns)
        raise ValueError(
            "Cannot validate Criteo data because required columns "
            f"are missing: {missing_columns_text}"
        )


def _get_binary_unique_values(
    dataframe: pd.DataFrame,
) -> dict[str, list[Any]]:
    """Return observed non-null values for each binary column."""
    unique_values: dict[str, list[Any]] = {}

    for column in BINARY_COLUMNS:
        values = dataframe[column].dropna().unique().tolist()
        unique_values[column] = sorted(
            (_to_python_value(value) for value in values),
            key=str,
        )

    return unique_values


def _get_invalid_binary_values(
    dataframe: pd.DataFrame,
) -> dict[str, list[Any]]:
    """Return binary-column values outside the expected domain {0, 1}."""
    invalid_values: dict[str, list[Any]] = {}

    for column in BINARY_COLUMNS:
        observed_values = set(
            dataframe[column].dropna().unique().tolist()
        )
        unexpected_values = observed_values.difference({0, 1})

        if unexpected_values:
            invalid_values[column] = sorted(
                (
                    _to_python_value(value)
                    for value in unexpected_values
                ),
                key=str,
            )

    return invalid_values


def _get_non_numeric_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return feature columns that do not use a numeric data type."""
    return [
        column
        for column in FEATURE_COLUMNS
        if not is_numeric_dtype(dataframe[column])
    ]


def validate_criteo(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Create a JSON-serializable Criteo data-quality report.

    The validation covers:

    - required schema;
    - row and column counts;
    - data types;
    - null values;
    - numeric feature types;
    - binary-column domains.

    Duplicate detection, treatment analysis, outcome analysis and leakage
    interpretation belong to the EDA notebook and are not performed here.

    Parameters
    ----------
    dataframe:
        Criteo dataframe to validate.

    Returns
    -------
    dict[str, Any]
        JSON-serializable validation results.

    Raises
    ------
    ValueError
        If required Criteo columns are missing.
    """
    _validate_required_columns(dataframe)

    null_counts = dataframe.isna().sum()
    null_percentages = dataframe.isna().mean().mul(100)

    columns_with_nulls = [
        column
        for column, null_count in null_counts.items()
        if int(null_count) > 0
    ]

    binary_unique_values = _get_binary_unique_values(dataframe)
    invalid_binary_values = _get_invalid_binary_values(dataframe)
    non_numeric_features = _get_non_numeric_features(dataframe)

    is_valid = (
        not dataframe.empty
        and not columns_with_nulls
        and not invalid_binary_values
        and not non_numeric_features
    )

    return {
        "is_valid": is_valid,
        "is_empty": dataframe.empty,
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "column_names": list(dataframe.columns),
        "data_types": {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "null_counts": _series_to_plain_dict(null_counts),
        "null_percentages": _series_to_plain_dict(
            null_percentages
        ),
        "columns_with_nulls": columns_with_nulls,
        "non_numeric_feature_columns": non_numeric_features,
        "binary_unique_values": binary_unique_values,
        "invalid_binary_values": invalid_binary_values,
    }