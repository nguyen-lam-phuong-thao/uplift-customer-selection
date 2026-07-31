"""Tests for decision dataset preparation."""

import pandas as pd

from uplift_modeling.data.criteo import FEATURE_COLUMNS
from uplift_modeling.data.preparation import (
    assign_stratified_split,
    build_decision_dataset,
    build_decision_frame,
)


def _raw_frame(row_count: int = 40) -> pd.DataFrame:
    rows = {
        feature: [float(index) for index in range(row_count)]
        for feature in FEATURE_COLUMNS
    }
    rows["treatment"] = [0, 1] * (row_count // 2)
    rows["visit"] = [0, 0, 1, 1] * (row_count // 4)
    rows["conversion"] = [1, 0, 0, 1] * (row_count // 4)
    rows["exposure"] = [0, 1] * (row_count // 2)
    return pd.DataFrame(rows)


def test_build_decision_frame_adds_valid_row_id() -> None:
    """Prepared decision data receives a deterministic unique row_id."""
    decision_frame = build_decision_frame(_raw_frame())

    assert decision_frame.columns[0] == "row_id"
    assert decision_frame["row_id"].isna().sum() == 0
    assert decision_frame["row_id"].is_unique
    assert decision_frame["row_id"].tolist() == list(range(len(decision_frame)))


def test_row_id_is_preserved_through_split_filter_and_reorder() -> None:
    """Supported decision-dataset transformations keep row_id unchanged."""
    decision_frame = build_decision_dataset(
        _raw_frame(),
        outcome_column="visit",
        random_state=7,
    )
    split_frame = assign_stratified_split(
        decision_frame.drop(columns="split"),
        outcome_column="visit",
        random_state=7,
    )
    filtered = split_frame.loc[split_frame["split"] == "validation"].copy()
    reordered = filtered.sort_values("f0", ascending=False)

    assert split_frame["row_id"].is_unique
    assert set(split_frame["row_id"]) == set(decision_frame["row_id"])
    assert set(reordered["row_id"]) == set(filtered["row_id"])


def test_build_decision_frame_preserves_existing_row_id() -> None:
    """Preparation preserves an existing valid row_id unchanged."""
    raw_frame = _raw_frame()
    expected_row_ids = list(range(1000, 1000 + len(raw_frame)))
    raw_frame.insert(0, "row_id", expected_row_ids)

    decision_frame = build_decision_frame(raw_frame)

    assert decision_frame["row_id"].tolist() == expected_row_ids


def test_split_does_not_use_dataframe_index_as_identity() -> None:
    """Split assignment remains correct when dataframe indexes are duplicated."""
    decision_frame = build_decision_frame(_raw_frame())
    expected_row_ids = decision_frame["row_id"].tolist()
    decision_frame.index = [0] * len(decision_frame)

    split_frame = assign_stratified_split(
        decision_frame,
        outcome_column="visit",
        random_state=7,
    )

    assert split_frame["row_id"].tolist() == expected_row_ids
    assert split_frame["row_id"].is_unique
    assert set(split_frame["split"]) == {"train", "validation", "test"}