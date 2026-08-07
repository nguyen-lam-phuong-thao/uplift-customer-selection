"""Standardize prepared modeling tables into decision datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from uplift_modeling.data.dataset_spec import DatasetConfig
from uplift_modeling.data.preparation import (
    build_decision_dataset,
    save_decision_dataset,
)
from uplift_modeling.data.row_id import add_row_id
from uplift_modeling.data.split import ensure_split_column
from uplift_modeling.data.validation import validate_prepared_dataset_contract


def standardize_prepared_dataset(
    dataset_config: DatasetConfig,
) -> dict[str, Path]:
    """Create one standardized decision dataset per configured outcome."""
    prepared_frame = load_prepared_table(dataset_config.prepared_path)
    dataset_spec = dataset_config.spec

    validate_prepared_dataset_contract(
        prepared_frame,
        dataset_spec,
        require_row_id=False,
    )

    prepared_frame = add_row_id(
        prepared_frame,
        row_id_column=dataset_spec.row_id_column,
    )

    validate_prepared_dataset_contract(
        prepared_frame,
        dataset_spec,
        require_row_id=True,
    )

    output_paths: dict[str, Path] = {}

    for outcome_column in dataset_spec.outcome_columns:
        frame_with_split = ensure_split_column(
            dataframe=prepared_frame,
            outcome_column=outcome_column,
            dataset_spec=dataset_spec,
            split_config=dataset_config.split,
        )

        decision_dataset = build_decision_dataset(
            dataframe=frame_with_split,
            outcome_column=outcome_column,
            dataset_spec=dataset_spec,
        )

        output_path = save_decision_dataset(
            dataframe=decision_dataset,
            output_path=dataset_config.processed_paths[outcome_column],
        )
        output_paths[outcome_column] = output_path

    return output_paths


def load_prepared_table(path: str | Path) -> pd.DataFrame:
    """Load a prepared modeling table.

    This is not raw data processing. The table must already be modeling-ready.
    """
    resolved_path = Path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Prepared dataset does not exist: {resolved_path}")

    suffix = resolved_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(resolved_path)

    if suffix == ".csv":
        return pd.read_csv(resolved_path)

    raise ValueError(
        "Prepared dataset must be a .parquet or .csv file. "
        f"Received: {resolved_path}"
    )