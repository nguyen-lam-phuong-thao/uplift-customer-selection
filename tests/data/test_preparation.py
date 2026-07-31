"""Tests for decision dataset preparation."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from uplift_modeling.data.dataset_spec import (
    CRITEO_SPEC,
    DatasetSpec,
    get_dataset_spec,
)
from uplift_modeling.data.preparation import (
    assign_stratified_split,
    build_decision_dataset,
    build_decision_frame,
)


def _raw_frame(row_count: int = 40) -> pd.DataFrame:
    rows = {
        feature: [float(index) for index in range(row_count)]
        for feature in CRITEO_SPEC.feature_columns
    }
    rows["treatment"] = [0, 1] * (row_count // 2)
    rows["visit"] = [0, 0, 1, 1] * (row_count // 4)
    rows["conversion"] = [1, 0, 0, 1] * (row_count // 4)
    rows["exposure"] = [0, 1] * (row_count // 2)
    return pd.DataFrame(rows)


def _synthetic_spec() -> DatasetSpec:
    return DatasetSpec(
        name="synthetic",
        row_id_column="customer_id",
        treatment_column="mail_flag",
        split_column="partition",
        feature_columns=("age", "score"),
        outcome_columns=("purchase", "retention"),
    )


def _synthetic_frame(row_count: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": list(range(1000, 1000 + row_count)),
            "age": [30 + index for index in range(row_count)],
            "score": [float(index) / 10 for index in range(row_count)],
            "mail_flag": [0, 1] * (row_count // 2),
            "purchase": [0, 0, 1, 1] * (row_count // 4),
            "retention": [1, 0, 0, 1] * (row_count // 4),
            "criteo_like_noise": [1] * row_count,
        }
    )


def test_known_dataset_returns_expected_spec() -> None:
    """The Criteo dataset name resolves to the expected schema contract."""
    spec = get_dataset_spec("criteo")

    assert spec == CRITEO_SPEC
    assert spec.row_id_column == "row_id"
    assert spec.treatment_column == "treatment"
    assert spec.split_column == "split"
    assert spec.outcome_columns == ("visit", "conversion")


def test_dataset_spec_is_immutable() -> None:
    """DatasetSpec fields cannot be changed after construction."""
    with pytest.raises(FrozenInstanceError):
        CRITEO_SPEC.treatment_column = "other"


def test_criteo_feature_order_is_preserved() -> None:
    """Criteo feature order remains f0 through f11."""
    assert CRITEO_SPEC.feature_columns == tuple(
        f"f{index}" for index in range(12)
    )


def test_unknown_dataset_is_rejected() -> None:
    """Unknown dataset names raise a clear error."""
    with pytest.raises(ValueError, match="Unknown dataset name"):
        get_dataset_spec("retailhero")


def test_unsupported_outcome_is_rejected() -> None:
    """Preparation validates requested outcomes against the selected spec."""
    with pytest.raises(ValueError, match="supports outcomes"):
        build_decision_dataset(
            _raw_frame(),
            outcome_column="exposure",
            dataset_spec=CRITEO_SPEC,
        )


def test_build_decision_frame_requires_explicit_dataset_spec() -> None:
    """Generic preparation does not silently default to Criteo."""
    with pytest.raises(TypeError):
        build_decision_frame(_raw_frame())


def test_generic_preparation_works_with_non_criteo_spec() -> None:
    """Preparation uses the provided spec instead of Criteo column names."""
    spec = _synthetic_spec()

    decision_frame = build_decision_dataset(
        _synthetic_frame(),
        outcome_column="purchase",
        dataset_spec=spec,
        random_state=7,
    )

    assert decision_frame.columns.tolist() == [
        "customer_id",
        "age",
        "score",
        "mail_flag",
        "purchase",
        "partition",
    ]
    assert set(decision_frame["partition"]) == {"train", "validation", "test"}
    assert decision_frame["customer_id"].tolist() == list(range(1000, 1040))


def test_criteo_preparation_column_order_remains_unchanged() -> None:
    """Existing Criteo decision-frame output keeps the same column order."""
    decision_frame = build_decision_frame(
        _raw_frame(),
        dataset_spec=CRITEO_SPEC,
    )
    decision_dataset = build_decision_dataset(
        _raw_frame(),
        outcome_column="visit",
        dataset_spec=CRITEO_SPEC,
        random_state=7,
    )

    assert decision_frame.columns.tolist() == [
        "row_id",
        *CRITEO_SPEC.feature_columns,
        "treatment",
        "visit",
        "conversion",
    ]
    assert decision_dataset.columns.tolist() == [
        "row_id",
        *CRITEO_SPEC.feature_columns,
        "treatment",
        "visit",
        "split",
    ]


def test_build_decision_frame_adds_valid_row_id() -> None:
    """Prepared decision data receives a deterministic unique row_id."""
    decision_frame = build_decision_frame(
        _raw_frame(),
        dataset_spec=CRITEO_SPEC,
    )

    assert decision_frame.columns[0] == "row_id"
    assert decision_frame["row_id"].isna().sum() == 0
    assert decision_frame["row_id"].is_unique
    assert decision_frame["row_id"].tolist() == list(range(len(decision_frame)))


def test_row_id_is_preserved_through_split_filter_and_reorder() -> None:
    """Supported decision-dataset transformations keep row_id unchanged."""
    decision_frame = build_decision_dataset(
        _raw_frame(),
        outcome_column="visit",
        dataset_spec=CRITEO_SPEC,
        random_state=7,
    )
    split_frame = assign_stratified_split(
        decision_frame.drop(columns="split"),
        outcome_column="visit",
        dataset_spec=CRITEO_SPEC,
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

    decision_frame = build_decision_frame(
        raw_frame,
        dataset_spec=CRITEO_SPEC,
    )

    assert decision_frame["row_id"].tolist() == expected_row_ids


def test_split_does_not_use_dataframe_index_as_identity() -> None:
    """Split assignment remains correct when dataframe indexes are duplicated."""
    decision_frame = build_decision_frame(
        _raw_frame(),
        dataset_spec=CRITEO_SPEC,
    )
    expected_row_ids = decision_frame["row_id"].tolist()
    decision_frame.index = [0] * len(decision_frame)

    split_frame = assign_stratified_split(
        decision_frame,
        outcome_column="visit",
        dataset_spec=CRITEO_SPEC,
        random_state=7,
    )

    assert split_frame["row_id"].tolist() == expected_row_ids
    assert split_frame["row_id"].is_unique
    assert set(split_frame["split"]) == {"train", "validation", "test"}
