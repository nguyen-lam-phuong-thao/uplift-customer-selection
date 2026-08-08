"""Tests for the response-model training schema boundary."""

import importlib
from pathlib import Path

import pandas as pd
import pytest

from uplift_modeling.data.dataset_spec import DatasetSpec, load_dataset_config
from uplift_modeling.pipelines.train_response_model import (
    get_training_splits,
    load_training_frame,
    validate_outcome,
)
from uplift_modeling.utils.config import get_config_section, load_yaml_config
from uplift_modeling.models.config import ModelCandidateConfig

pytest.importorskip("pyarrow")


OBSOLETE_SCHEMA_KEYS = {
    "row_id_column",
    "treatment_column",
    "split_column",
    "feature_columns",
    "outcome_columns",
    "primary_outcome_column",
    "secondary_outcome_column",
}

MODELING_CONFIG_PATH = "configs/modeling/uplift_lgbm.yaml"

def test_training_splits_default_to_train_and_validation() -> None:
    """Training uses the fixed pre-locked-test split boundary."""
    assert get_training_splits({}) == (
        "train",
        "validation",
    )


def test_training_splits_reject_test_as_validation() -> None:
    """Test cannot be used for early stopping."""
    with pytest.raises(
        ValueError,
        match="test split is reserved",
    ):
        get_training_splits(
            {
                "train_split": "train",
                "validation_split": "test",
            }
        )


def test_modeling_config_does_not_own_dataset_schema() -> None:
    """Modeling config contains training/model settings, not dataset schema."""
    project_root = Path(__file__).parents[2]

    config = load_yaml_config(
        project_root / MODELING_CONFIG_PATH
    )

    assert "data" not in config

    training_config = get_config_section(
        config,
        "training",
    )
    models_config = get_config_section(
        config,
        "models",
    )
    output_config = get_config_section(
        config,
        "outputs",
    )

    assert training_config["train_split"] == "train"
    assert training_config["validation_split"] == "validation"
    assert training_config["prediction_batch_size"] == 500000

    assert models_config["model_defaults"]["random_state"] == 42
    assert len(models_config["candidates"]) == 3

    assert output_config["prediction_dir"] == "artifacts/predictions"


def test_training_splits_reject_non_train_fitting_split() -> None:
    """Model fitting cannot use validation or test rows."""
    with pytest.raises(
        ValueError,
        match="training.train_split must be 'train'",
    ):
        get_training_splits(
            {
                "train_split": "validation",
                "validation_split": "validation",
            }
        )


def test_dataset_config_loads_criteo_schema_from_yaml() -> None:
    """Dataset config owns prepared paths and DatasetSpec schema."""
    project_root = Path(__file__).parents[2]
    dataset_config = load_dataset_config(
        project_root / "configs/datasets/criteo.yaml",
        project_root=project_root,
    )

    assert dataset_config.spec.name == "criteo"
    assert dataset_config.spec.treatment_column == "treatment"
    assert dataset_config.spec.split_column == "split"
    assert dataset_config.spec.feature_columns == tuple(
        f"f{index}" for index in range(12)
    )
    assert dataset_config.spec.outcome_columns == ("visit", "conversion")
    assert dataset_config.split.train_size == 0.6
    assert dataset_config.split.validation_size == 0.2
    assert dataset_config.split.test_size == 0.2


def test_training_frame_loads_columns_from_dataset_spec(tmp_path: Path) -> None:
    """Stable columns come from the resolved DatasetSpec, not config."""
    dataset_spec = DatasetSpec(
        name="synthetic",
        treatment_column="mail_flag",
        split_column="partition",
        feature_columns=("age", "score"),
        outcome_columns=("purchase",),
    )
    parquet_path = tmp_path / "training.parquet"
    pd.DataFrame(
        {
            "row_id": [0, 1],
            "customer_id": [1, 2],
            "age": [30, 40],
            "score": [0.1, 0.2],
            "mail_flag": [1, 0],
            "partition": ["train", "validation"],
            "purchase": [1, 0],
            "unused": ["drop", "drop"],
        }
    ).to_parquet(parquet_path, index=False)

    frame, target_column = load_training_frame(
        parquet_path=parquet_path,
        dataset_spec=dataset_spec,
        requested_outcome="purchase",
    )

    assert target_column == "purchase"
    assert frame.columns.tolist() == [
        "row_id",
        "age",
        "score",
        "mail_flag",
        "partition",
        "purchase",
    ]


def _write_debug_dataset_config(
    tmp_path: Path,
    data_path: Path,
) -> Path:
    config_path = tmp_path / "dataset.yaml"

    feature_lines = [
        f"    - f{index}"
        for index in range(12)
    ]

    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: synthetic",
                f"  prepared_path: {(tmp_path / 'unused.parquet').as_posix()}",
                "schema:",
                "  treatment_column: treatment",
                "  split_column: split",
                "  feature_columns:",
                *feature_lines,
                "  outcome_columns:",
                "    - visit",
                "split:",
                "  assign_if_missing: false",
                "  train_size: 0.6",
                "  validation_size: 0.2",
                "  test_size: 0.2",
                "  random_state: 42",
                "outputs:",
                "  processed_paths:",
                f"    visit: {data_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    return config_path


def _write_debug_modeling_config(
    tmp_path: Path,
) -> Path:
    config_path = tmp_path / "modeling.yaml"

    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  experiment_name: uplift-test",
                "training:",
                "  train_split: train",
                "  validation_split: validation",
                "  prediction_batch_size: 2",
                "  early_stopping_rounds: 1",
                "  log_evaluation_period: 1",
                "models:",
                "  model_defaults:",
                "    random_state: 42",
                "  candidates:",
                "    - name: treated_response_lgbm",
                "      kind: response",
                "      params: {}",
                "    - name: t_learner_lgbm",
                "      kind: t_learner",
                "      params: {}",
                "    - name: x_learner_lgbm",
                "      kind: x_learner",
                "      params: {}",
                "outputs:",
                f"  prediction_dir: {(tmp_path / 'predictions').as_posix()}",
                f"  metric_dir: {(tmp_path / 'metrics').as_posix()}",
                "tracking:",
                "  log_predictions: false",
                "debug:",
                "  sample_rows: 2",
                "  random_state: 42",
            ]
        ),
        encoding="utf-8",
    )

    return config_path


def _write_debug_sampling_data(tmp_path: Path) -> Path:
    """Write train, validation, and test rows for debug-sampling tests."""
    rows = []
    for row_id, split in enumerate(["train", "validation", "test"]):
        row = {
            "row_id": row_id,
            "treatment": row_id % 2,
            "split": split,
            "visit": row_id % 2,
        }
        row.update({f"f{index}": float(row_id + index) for index in range(12)})
        rows.append(row)

    data_path = tmp_path / "criteo_decision_visit.parquet"
    pd.DataFrame(rows).to_parquet(data_path, index=False)
    return data_path


@pytest.mark.parametrize(
    "module_name,function_name,model_name,model_kind",
    [
        (
            "uplift_modeling.pipelines.train_response_model",
            "train_response_pipeline",
            "treated_response_lgbm",
            "response",
        ),
        (
            "uplift_modeling.pipelines.train_t_learner",
            "train_t_learner_pipeline",
            "t_learner_lgbm",
            "t_learner",
        ),
        (
            "uplift_modeling.pipelines.train_x_learner",
            "train_x_learner_pipeline",
            "x_learner_lgbm",
            "x_learner",
        ),
    ],
)


def test_debug_sample_receives_train_and_validation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    function_name: str,
    model_name: str,
    model_kind: str,
) -> None:
    data_path = _write_debug_sampling_data(tmp_path)

    dataset_config_path = _write_debug_dataset_config(
        tmp_path,
        data_path=data_path,
    )
    modeling_config_path = _write_debug_modeling_config(
        tmp_path,
    )

    pipeline_module = importlib.import_module(module_name)
    observed_splits = set()

    def capture_debug_sample(
        dataframe,
        sample_rows,
        random_state,
    ):
        observed_splits.update(
            dataframe["split"].unique()
        )
        raise RuntimeError("stop after debug sample")

    monkeypatch.setattr(
        pipeline_module,
        "apply_debug_sample",
        capture_debug_sample,
    )

    model_candidate = ModelCandidateConfig(
        name=model_name,
        kind=model_kind,
        params={},
    )

    with pytest.raises(
        RuntimeError,
        match="stop after debug sample",
    ):
        getattr(
            pipeline_module,
            function_name,
        )(
            dataset_config_path=dataset_config_path,
            modeling_config_path=modeling_config_path,
            outcome="visit",
            model_candidate=model_candidate,
        )

    assert observed_splits == {
        "train",
        "validation",
    }