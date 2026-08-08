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
            "row_id": [0, 1],
            "customer_id": [101, 102],
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
        "row_id",
        "treatment",
        "outcome",
        "split",
        "score",
        "model_name",
    ]
    assert predictions["row_id"].tolist() == [0, 1]
    assert predictions["score"].tolist() == [0.8, 0.2]


def test_build_prediction_frame_rejects_score_length_mismatch() -> None:
    """Prediction scores must align to input rows."""
    with pytest.raises(ValueError, match="same number of rows"):
        build_prediction_frame(
            dataframe=pd.DataFrame(
                {"row_id": [1], "t": [1], "s": ["test"], "y": [1]}
            ),
            scores=np.array([0.1, 0.2]),
            treatment_column="t",
            split_column="s",
            outcome_column="y",
            model_name="model",
        )


def test_build_prediction_frame_rejects_missing_row_id() -> None:
    """Prediction frames must not fall back to dataframe index values."""
    source = pd.DataFrame(
        {
            "treatment": [1, 0],
            "split": ["validation", "test"],
            "outcome": [1, 0],
        }
    )

    with pytest.raises(ValueError, match="row-ID column 'row_id'"):
        build_prediction_frame(
            dataframe=source,
            scores=np.array([0.8, 0.2]),
            treatment_column="treatment",
            split_column="split",
            outcome_column="outcome",
            model_name="response_model_lgbm",
        )


def test_build_prediction_frame_rejects_null_row_id() -> None:
    """Prediction row IDs must be non-null."""
    source = pd.DataFrame(
        {
            "row_id": [1, None],
            "treatment": [1, 0],
            "split": ["validation", "test"],
            "outcome": [1, 0],
        }
    )

    with pytest.raises(ValueError, match="non-null"):
        build_prediction_frame(
            dataframe=source,
            scores=np.array([0.8, 0.2]),
            treatment_column="treatment",
            split_column="split",
            outcome_column="outcome",
            model_name="response_model_lgbm",
        )


def test_build_prediction_frame_rejects_duplicate_row_id() -> None:
    """Prediction row IDs must be unique inside an artifact batch."""
    source = pd.DataFrame(
        {
            "row_id": [1, 1],
            "treatment": [1, 0],
            "split": ["validation", "test"],
            "outcome": [1, 0],
        }
    )

    with pytest.raises(ValueError, match="unique"):
        build_prediction_frame(
            dataframe=source,
            scores=np.array([0.8, 0.2]),
            treatment_column="treatment",
            split_column="split",
            outcome_column="outcome",
            model_name="response_model_lgbm",
        )


def test_save_prediction_parquet_in_batches(tmp_path) -> None:
    """Batched writer scores and saves all rows."""
    source = pd.DataFrame(
        {
            "row_id": [10, 11, 12],
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
    assert predictions["row_id"].tolist() == [10, 11, 12]
    assert predictions["score"].tolist() == [0.1, 0.2, 0.3]


def test_save_prediction_parquet_in_batches_rejects_duplicate_ids_across_batches(
    tmp_path,
) -> None:
    """Batched prediction writing rejects row_id overlap across batches."""
    first = pd.DataFrame(
        {
            "row_id": [10, 11],
            "f0": [0.1, 0.2],
            "treatment": [1, 0],
            "split": ["validation", "validation"],
            "outcome": [1, 0],
        }
    )
    second = pd.DataFrame(
        {
            "row_id": [11, 12],
            "f0": [0.3, 0.4],
            "treatment": [1, 0],
            "split": ["test", "test"],
            "outcome": [1, 0],
        }
    )

    with pytest.raises(ValueError, match="disjoint row_id"):
        save_prediction_parquet_in_batches(
            dataframes=(first, second),
            output_path=tmp_path / "predictions.parquet",
            feature_columns=("f0",),
            treatment_column="treatment",
            split_column="split",
            outcome_column="outcome",
            model_name="response_model_lgbm",
            batch_size=2,
            score_batch=lambda X_batch: X_batch["f0"].to_numpy(),
        )


def test_save_prediction_parquet_in_batches_accepts_disjoint_ids(
    tmp_path,
) -> None:
    """Batched prediction writing accepts disjoint row_id batches."""
    first = pd.DataFrame(
        {
            "row_id": [10, 11],
            "f0": [0.1, 0.2],
            "treatment": [1, 0],
            "split": ["validation", "validation"],
            "outcome": [1, 0],
        }
    )
    second = pd.DataFrame(
        {
            "row_id": [12, 13],
            "f0": [0.3, 0.4],
            "treatment": [1, 0],
            "split": ["test", "test"],
            "outcome": [1, 0],
        }
    )
    output_path = tmp_path / "predictions.parquet"

    save_prediction_parquet_in_batches(
        dataframes=(first, second),
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
    assert predictions["row_id"].tolist() == [10, 11, 12, 13]


def test_response_and_t_learner_scores_keep_same_row_id_contract(tmp_path) -> None:
    """Response and uplift scorers write predictions with identical row IDs."""
    source = pd.DataFrame(
        {
            "row_id": [31, 32, 33, 34],
            "f0": [0.1, 0.2, 0.3, 0.4],
            "f1": [0.4, 0.3, 0.2, 0.1],
            "treatment": [1, 0, 1, 0],
            "split": ["validation"] * 4,
            "outcome": [1, 0, 0, 1],
        }
    )
    response_path = tmp_path / "response.parquet"
    t_learner_path = tmp_path / "t_learner.parquet"

    save_prediction_parquet_in_batches(
        dataframes=(source,),
        output_path=response_path,
        feature_columns=("f0", "f1"),
        treatment_column="treatment",
        split_column="split",
        outcome_column="outcome",
        model_name="treated_response_lgbm",
        batch_size=2,
        score_batch=lambda X_batch: X_batch["f0"].to_numpy(),
    )
    save_prediction_parquet_in_batches(
        dataframes=(source.sort_values("row_id", ascending=False),),
        output_path=t_learner_path,
        feature_columns=("f0", "f1"),
        treatment_column="treatment",
        split_column="split",
        outcome_column="outcome",
        model_name="t_learner_lgbm",
        batch_size=2,
        score_batch=lambda X_batch: (
            X_batch["f0"].to_numpy() - X_batch["f1"].to_numpy()
        ),
    )

    response_predictions = pd.read_parquet(response_path)
    t_learner_predictions = pd.read_parquet(t_learner_path)

    assert set(response_predictions["row_id"]) == set(source["row_id"])
    assert set(t_learner_predictions["row_id"]) == set(source["row_id"])
    assert t_learner_predictions["row_id"].tolist() == [34, 33, 32, 31]
    actual_scores = dict(
        zip(
            t_learner_predictions["row_id"],
            t_learner_predictions["score"],
        )
    )
    expected_scores = {
        31: -0.3,
        32: -0.1,
        33: 0.1,
        34: 0.3,
    }

    assert actual_scores.keys() == expected_scores.keys()
    assert actual_scores == pytest.approx(expected_scores)