"""Build standardized decision datasets from prepared modeling tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from uplift_modeling.data.dataset_spec import (
    DatasetSpec,
    validate_supported_outcome,
)
from uplift_modeling.data.row_id import validate_row_id_column


def build_decision_dataset(
    dataframe: pd.DataFrame,
    outcome_column: str,
    dataset_spec: DatasetSpec,
) -> pd.DataFrame:
    """Build one outcome-specific decision dataset.

    The input dataframe must already contain:
    - row_id
    - feature columns
    - treatment column
    - selected outcome column
    - split column
    """
    validate_supported_outcome(dataset_spec, outcome_column)

    selected_columns = [
        dataset_spec.row_id_column,
        *dataset_spec.feature_columns,
        dataset_spec.treatment_column,
        outcome_column,
        dataset_spec.split_column,
    ]

    missing_columns = sorted(set(selected_columns).difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "Cannot build decision dataset because columns are missing: "
            f"{missing_columns}"
        )

    validate_row_id_column(
        dataframe,
        row_id_column=dataset_spec.row_id_column,
        context="Decision dataset",
    )

    return dataframe.loc[:, selected_columns].copy()


def save_decision_dataset(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a decision dataset as parquet and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
    return path