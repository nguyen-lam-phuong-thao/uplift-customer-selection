"""Prediction artifact builders and parquet writers."""

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from uplift_modeling.data.row_id import ROW_ID_COLUMN, validate_row_id_column


def build_prediction_frame(
    dataframe: pd.DataFrame,
    scores: np.ndarray,
    treatment_column: str,
    split_column: str,
    outcome_column: str,
    model_name: str,
) -> pd.DataFrame:
    """Build a prediction dataframe that follows the shared contract."""
    validate_row_id_column(
        dataframe,
        context="Prediction source dataframe",
    )

    required_columns = {
        treatment_column,
        split_column,
        outcome_column,
    }
    missing_columns = sorted(required_columns.difference(dataframe.columns))

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            f"Cannot build prediction frame because columns are missing: "
            f"{missing_text}"
        )

    if len(dataframe) != len(scores):
        raise ValueError(
            "dataframe and scores must have the same number of rows. "
            f"Received: {len(dataframe)} and {len(scores)}"
        )

    return pd.DataFrame(
        {
            ROW_ID_COLUMN: dataframe[ROW_ID_COLUMN].to_numpy(),
            "treatment": dataframe[treatment_column].to_numpy(),
            "outcome": dataframe[outcome_column].to_numpy(),
            "split": dataframe[split_column].to_numpy(),
            "score": scores,
            "model_name": model_name,
        }
    )


def save_prediction_parquet_in_batches(
    dataframes: Iterable[pd.DataFrame],
    output_path: Path,
    feature_columns: tuple[str, ...],
    treatment_column: str,
    split_column: str,
    outcome_column: str,
    model_name: str,
    batch_size: int,
    score_batch: Callable[[pd.DataFrame], np.ndarray],
) -> Path:
    """Score dataframes by batch and save predictions as parquet."""
    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be a positive integer. Received: {batch_size}"
        )

    required_columns = {
        *feature_columns,
        treatment_column,
        split_column,
        outcome_column,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    seen_row_ids: set[object] = set()

    try:
        for dataframe in dataframes:
            missing_columns = sorted(
                required_columns.difference(dataframe.columns)
            )

            if missing_columns:
                missing_text = ", ".join(missing_columns)
                raise ValueError(
                    "Cannot save predictions because columns are missing: "
                    f"{missing_text}"
                )

            for start_index in range(0, len(dataframe), batch_size):
                end_index = min(start_index + batch_size, len(dataframe))
                batch_frame = dataframe.iloc[start_index:end_index]
                scores = score_batch(batch_frame.loc[:, feature_columns])
                prediction_batch = build_prediction_frame(
                    dataframe=batch_frame,
                    scores=scores,
                    treatment_column=treatment_column,
                    split_column=split_column,
                    outcome_column=outcome_column,
                    model_name=model_name,
                )
                batch_row_ids = set(prediction_batch[ROW_ID_COLUMN].tolist())
                overlapping_ids = sorted(seen_row_ids.intersection(batch_row_ids))
                if overlapping_ids:
                    overlap_text = ", ".join(
                        repr(value) for value in overlapping_ids[:5]
                    )
                    if len(overlapping_ids) > 5:
                        overlap_text += ", ..."
                    raise ValueError(
                        "Prediction batches must contain disjoint row_id "
                        f"values. Overlap: {overlap_text}."
                    )
                seen_row_ids.update(batch_row_ids)
                table = pa.Table.from_pandas(
                    prediction_batch,
                    preserve_index=False,
                )

                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)

                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError("No prediction rows were available to save.")

    return output_path
