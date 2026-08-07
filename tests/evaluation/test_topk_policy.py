"""Focused tests for validation Top-K policy evaluation."""

from pathlib import Path

import pandas as pd
import pytest

from uplift_modeling.evaluation.topk_policy import (
    prepare_topk_policy_frames,
    resolve_expected_policy_paths,
    validate_standard_evaluation_splits,
)


def _prediction_frame(model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "treatment": [1, 0, 1, 0],
            "outcome": [1, 0, 0, 1],
            "split": ["validation"] * 4,
            "score": [0.9, 0.8, 0.7, 0.6],
            "model_name": [model_name] * 4,
        }
    )


def test_manifest_policy_resolution_is_generic(tmp_path: Path) -> None:
    first = tmp_path / "custom_a.parquet"
    second = tmp_path / "custom_b.parquet"

    paths, warnings = resolve_expected_policy_paths(
        {
            "my_custom_policy": first,
            "another_policy": second,
        },
        outcome="visit",
    )

    assert paths == {
        "another_policy": second,
        "my_custom_policy": first,
    }
    assert warnings == []


def test_random_targeting_must_not_be_stored_in_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="random_targeting"):
        resolve_expected_policy_paths(
            {"random_targeting": tmp_path / "random.parquet"},
            outcome="visit",
        )


def test_standard_evaluation_is_validation_only() -> None:
    assert validate_standard_evaluation_splits(
        ("validation",)
    ) == ("validation",)

    with pytest.raises(ValueError, match="test split is reserved"):
        validate_standard_evaluation_splits(("test",))


def test_prepare_topk_uses_manifest_artifacts_and_adds_random(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "custom.parquet"
    _prediction_frame("my_custom_policy").to_parquet(
        prediction_path,
        index=False,
    )

    frames, paths, warnings = prepare_topk_policy_frames(
        manifest_prediction_paths={
            "my_custom_policy": prediction_path,
        },
        outcome="visit",
        random_seed=42,
    )

    assert set(frames) == {
        "random_targeting",
        "my_custom_policy",
    }
    assert paths == {"my_custom_policy": prediction_path}
    assert warnings == []
    assert set(frames["my_custom_policy"]["split"]) == {"validation"}