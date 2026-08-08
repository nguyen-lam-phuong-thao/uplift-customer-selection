from pathlib import Path

import pandas as pd
import pytest

from uplift_modeling.data.dataset_spec import (
    DatasetConfig,
    DatasetSpec,
    SplitConfig,
)
from uplift_modeling.data.standardization import standardize_prepared_dataset


pytest.importorskip("pyarrow")


def make_prepared_frame(rows_per_group: int = 10) -> pd.DataFrame:
    rows = []

    for treatment in [0, 1]:
        for visit in [0, 1]:
            for index in range(rows_per_group):
                rows.append(
                    {
                        "feature_1": float(index),
                        "feature_2": float(index + 1),
                        "treatment": treatment,
                        "visit": visit,
                        "conversion": 0,
                    }
                )

    return pd.DataFrame(rows)


def test_standardize_prepared_dataset_writes_decision_datasets(
    tmp_path: Path,
) -> None:
    prepared_path = tmp_path / "prepared.parquet"
    visit_output_path = tmp_path / "decision_visit.parquet"
    conversion_output_path = tmp_path / "decision_conversion.parquet"

    make_prepared_frame().to_parquet(prepared_path, index=False)

    dataset_config = DatasetConfig(
        spec=DatasetSpec(
            name="unit_test",
            treatment_column="treatment",
            split_column="split",
            feature_columns=("feature_1", "feature_2"),
            outcome_columns=("visit", "conversion"),
        ),
        prepared_path=prepared_path,
        processed_paths={
            "visit": visit_output_path,
            "conversion": conversion_output_path,
        },
        split=SplitConfig(
            assign_if_missing=True,
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            random_state=42,
        ),
    )

    output_paths = standardize_prepared_dataset(dataset_config)

    assert output_paths["visit"] == visit_output_path
    assert output_paths["conversion"] == conversion_output_path
    assert visit_output_path.exists()
    assert conversion_output_path.exists()

    visit_decision = pd.read_parquet(visit_output_path)

    assert list(visit_decision.columns) == [
        "row_id",
        "feature_1",
        "feature_2",
        "treatment",
        "visit",
        "split",
    ]
    assert set(visit_decision["split"].unique()) == {
        "train",
        "validation",
        "test",
    }
