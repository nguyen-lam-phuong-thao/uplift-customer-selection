import pandas as pd
import pytest

from uplift_modeling.data.dataset_spec import DatasetSpec
from uplift_modeling.data.preparation import build_decision_dataset


@pytest.fixture()
def dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        name="unit_test",
        treatment_column="treatment",
        split_column="split",
        feature_columns=("feature_1", "feature_2"),
        outcome_columns=("visit", "conversion"),
    )


def test_build_decision_dataset_selects_expected_columns(dataset_spec: DatasetSpec) -> None:
    dataframe = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "feature_1": [0.1, 0.2, 0.3],
            "feature_2": [1.0, 2.0, 3.0],
            "treatment": [0, 1, 0],
            "visit": [0, 1, 0],
            "conversion": [0, 0, 1],
            "split": ["train", "validation", "test"],
            "unused_column": [10, 20, 30],
        }
    )

    result = build_decision_dataset(
        dataframe=dataframe,
        outcome_column="visit",
        dataset_spec=dataset_spec,
    )

    assert list(result.columns) == [
        "row_id",
        "feature_1",
        "feature_2",
        "treatment",
        "visit",
        "split",
    ]


def test_build_decision_dataset_rejects_unknown_outcome(
    dataset_spec: DatasetSpec,
) -> None:
    dataframe = pd.DataFrame(
        {
            "row_id": [0, 1],
            "feature_1": [0.1, 0.2],
            "feature_2": [1.0, 2.0],
            "treatment": [0, 1],
            "visit": [0, 1],
            "conversion": [0, 0],
            "split": ["train", "test"],
        }
    )

    with pytest.raises(ValueError, match="supports outcomes"):
        build_decision_dataset(
            dataframe=dataframe,
            outcome_column="revenue",
            dataset_spec=dataset_spec,
        )


def test_build_decision_dataset_requires_row_id(dataset_spec: DatasetSpec) -> None:
    dataframe = pd.DataFrame(
        {
            "feature_1": [0.1, 0.2],
            "feature_2": [1.0, 2.0],
            "treatment": [0, 1],
            "visit": [0, 1],
            "conversion": [0, 0],
            "split": ["train", "test"],
        }
    )

    with pytest.raises(ValueError, match="columns are missing"):
        build_decision_dataset(
            dataframe=dataframe,
            outcome_column="visit",
            dataset_spec=dataset_spec,
        )