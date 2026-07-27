"""Tests for prediction artifact helpers."""

import numpy as np
import pandas as pd
import pytest

from uplift_modeling.artifacts.predictions import (
    build_prediction_frame,
    save_prediction_parquet_in_batches,
)


def test_build_prediction_frame_uses_shared_contract() -> None:
    """Prediction frames expose the columns used by evaluation."""
    source = pd.DataFrame(
        {
            "f0": [0.1, 0.2],
            "treatment_flag": [1, 0],
            "split_name": ["validation", "test"],
            "target": [1, 0],
        }
    )
    predictions = build_prediction_frame(
        dataframe=source,
        scores=np.array([0.8, 0.2]),
        treatment_column="treatment_flag",
        split_column="split_name",
        outcome_column="target",
        model_name="response_model_lgbm",
    )

    assert predictions.columns.tolist() == [
        "treatment",
        "outcome",
        "split",
        "score",
        "model_name",
    ]
    assert predictions["score"].tolist() == [0.8, 0.2]


def test_build_prediction_frame_rejects_score_length_mismatch() -> None:
    """Prediction scores must align to input rows."""
    with pytest.raises(ValueError, match="same number of rows"):
        build_prediction_frame(
            dataframe=pd.DataFrame({"t": [1], "s": ["test"], "y": [1]}),
            scores=np.array([0.1, 0.2]),
            treatment_column="t",
            split_column="s",
            outcome_column="y",
            model_name="model",
        )


def test_save_prediction_parquet_in_batches(tmp_path) -> None:
    """Batched writer scores and saves all rows."""
    source = pd.DataFrame(
        {
            "f0": [0.1, 0.2, 0.3],
            "treatment": [1, 0, 1],
            "split": ["train", "validation", "test"],
            "outcome": [1, 0, 1],
        }
    )
    output_path = tmp_path / "predictions.parquet"

    save_prediction_parquet_in_batches(
        dataframes=(source,),
        output_path=output_path,
        feature_columns=("f0",),
        treatment_column="treatment",
        split_column="split",
        outcome_column="outcome",
        model_name="response_model_lgbm",
        batch_size=2,
        score_batch=lambda X_batch: X_batch["f0"].to_numpy(),
    )
    predictions = pd.read_parquet(output_path)

    assert predictions.shape[0] == 3
    assert predictions["score"].tolist() == [0.1, 0.2, 0.3]
