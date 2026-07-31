"""Shared row identifier contract for prepared data and predictions."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


ROW_ID_COLUMN = "row_id"


def _format_values(values: Sequence[Any], limit: int = 5) -> str:
    shown = [repr(value) for value in values[:limit]]
    suffix = "" if len(values) <= limit else ", ..."
    return ", ".join(shown) + suffix


def validate_row_id_column(
    dataframe: pd.DataFrame,
    row_id_column: str = ROW_ID_COLUMN,
    context: str = "Dataframe",
    *,
    require_unique: bool = True,
) -> None:
    """Validate the required non-null row identifier column."""
    if row_id_column not in dataframe.columns:
        raise ValueError(
            f"{context} must contain required row-ID column "
            f"'{row_id_column}'."
        )

    row_ids = dataframe[row_id_column]
    if row_ids.isna().any():
        null_count = int(row_ids.isna().sum())
        raise ValueError(
            f"{context} row-ID column '{row_id_column}' must be non-null. "
            f"Found {null_count} null value(s)."
        )

    if require_unique:
        duplicated = row_ids.loc[row_ids.duplicated(keep=False)]
        if not duplicated.empty:
            duplicate_values = duplicated.drop_duplicates().tolist()
            raise ValueError(
                f"{context} row-ID column '{row_id_column}' must be unique. "
                f"Duplicate value(s): {_format_values(duplicate_values)}."
            )
        

def add_row_id(
    dataframe: pd.DataFrame,
    row_id_column: str = ROW_ID_COLUMN,
) -> pd.DataFrame:
    """Return a copy with a deterministic row identifier column."""
    output = dataframe.copy()
    if row_id_column not in output.columns:
        output.insert(0, row_id_column, np.arange(len(output), dtype=np.int64))

    validate_row_id_column(
        output,
        row_id_column=row_id_column,
        context="Prepared dataset",
    )
    return output


def align_frames_by_row_id(
    frames: Mapping[str, pd.DataFrame],
    label_columns: Sequence[str],
    row_id_column: str = ROW_ID_COLUMN,
    context: str = "Prediction frames",
) -> dict[str, pd.DataFrame]:
    """Return frames sorted by row_id after validating matching IDs and labels."""
    if not frames:
        raise ValueError(f"{context} require at least one frame.")

    reference_name = sorted(frames)[0]
    reference_frame = frames[reference_name]
    validate_row_id_column(
        reference_frame,
        row_id_column=row_id_column,
        context=f"{context} reference '{reference_name}'",
    )

    reference_ids = set(reference_frame[row_id_column].tolist())
    for frame_name, frame in frames.items():
        validate_row_id_column(
            frame,
            row_id_column=row_id_column,
            context=f"{context} frame '{frame_name}'",
        )
        frame_ids = set(frame[row_id_column].tolist())
        missing_ids = sorted(reference_ids.difference(frame_ids))
        unexpected_ids = sorted(frame_ids.difference(reference_ids))
        if missing_ids or unexpected_ids:
            details = []
            if missing_ids:
                details.append(
                    "missing row_id value(s): "
                    f"{_format_values(missing_ids)}"
                )
            if unexpected_ids:
                details.append(
                    "unexpected row_id value(s): "
                    f"{_format_values(unexpected_ids)}"
                )
            detail_text = "; ".join(details)
            raise ValueError(
                f"{context} frame '{frame_name}' row_id values do not match "
                f"reference frame '{reference_name}': {detail_text}."
            )

    reference_sorted = reference_frame.sort_values(
        row_id_column,
        kind="mergesort",
    ).reset_index(drop=True)
    sorted_ids = reference_sorted[row_id_column].tolist()
    aligned_frames = {
        reference_name: reference_sorted,
    }

    for frame_name, frame in frames.items():
        if frame_name == reference_name:
            continue

        aligned_frame = (
            frame.set_index(row_id_column, verify_integrity=True)
            .loc[sorted_ids]
            .reset_index()
        )
        reference_labels = reference_sorted.loc[:, label_columns].reset_index(
            drop=True
        )
        aligned_labels = aligned_frame.loc[:, label_columns].reset_index(
            drop=True
        )
        if not aligned_labels.equals(reference_labels):
            label_text = ", ".join(label_columns)
            raise ValueError(
                f"{context} frame '{frame_name}' labels do not match "
                f"reference frame '{reference_name}' by row_id. "
                f"Checked label column(s): {label_text}."
            )

        aligned_frames[frame_name] = aligned_frame

    return {
        frame_name: aligned_frames[frame_name]
        for frame_name in frames
    }
