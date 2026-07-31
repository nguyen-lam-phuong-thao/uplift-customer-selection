"""Tests for the Criteo response-model training schema boundary."""

from pathlib import Path

import pandas as pd
import pytest

from uplift_modeling.data.dataset_spec import CRITEO_SPEC, DatasetSpec
from uplift_modeling.pipelines.train_criteo_response_model import (
    get_processed_data_path,
    load_training_frame,
    resolve_dataset_spec,
    validate_outcome,
)
from uplift_modeling.utils.config import get_config_section, load_yaml_config


OBSOLETE_SCHEMA_KEYS = {
    "row_id_column",
    "treatment_column",
    "split_column",
    "feature_columns",
    "outcome_columns",
    "primary_outcome_column",
    "secondary_outcome_column",
}

MODEL_CONFIG_PATHS = (
    "configs/modeling/criteo_response_lgbm.yaml",
    "configs/modeling/t_learner.yaml",
    "configs/modeling/x_learner.yaml",
)


def test_active_model_configs_load_without_obsolete_schema_keys() -> None:
    """Model configs keep runtime settings but no DatasetSpec-owned schema."""
    project_root = Path(__file__).parents[2]

    for relative_path in MODEL_CONFIG_PATHS:
        config = load_yaml_config(project_root / relative_path)
        data_config = get_config_section(config, "data")
        training_config = get_config_section(config, "training")
        model_config = get_config_section(config, "model")
        output_config = get_config_section(config, "outputs")
        selection_config = get_config_section(config, "selection")
        tracking_config = get_config_section(config, "tracking")

        assert OBSOLETE_SCHEMA_KEYS.isdisjoint(data_config)
        assert data_config["dataset_name"] == "criteo"
        assert data_config["processed_paths"] == {
            "visit": "data/processed/criteo/criteo_decision_visit.parquet",
            "conversion": (
                "data/processed/criteo/criteo_decision_conversion.parquet"
            ),
        }
        assert training_config["train_split"] == "train"
        assert training_config["validation_split"] == "validation"
        assert training_config["test_split"] == "test"
        assert training_config["prediction_batch_size"] == 500000
        assert model_config["params"]["random_state"] == 42
        assert output_config["prediction_splits"] == ["validation", "test"]
        assert selection_config["primary_outcome"] == "visit"
        assert selection_config["primary_split"] == "validation"
        assert tracking_config["log_predictions"] is False


def test_data_config_loads_without_dataset_spec_schema_keys() -> None:
    """Raw data config keeps paths and split settings only."""
    project_root = Path(__file__).parents[2]
    config = load_yaml_config(project_root / "configs/data.yaml")
    criteo_config = get_config_section(config, "criteo")

    assert OBSOLETE_SCHEMA_KEYS.isdisjoint(criteo_config)
    assert criteo_config["raw_path"].endswith("criteo-research-uplift-v2.1.csv.gz")
    assert criteo_config["processed_path"] == "data/processed/criteo"
    assert criteo_config["decision_visit_path"].endswith(
        "criteo_decision_visit.parquet"
    )
    assert criteo_config["decision_conversion_path"].endswith(
        "criteo_decision_conversion.parquet"
    )
    assert criteo_config["split"] == {
        "train_size": 0.6,
        "validation_size": 0.2,
        "test_size": 0.2,
        "random_state": 42,
    }


def test_conflicting_config_columns_cannot_override_dataset_spec() -> None:
    """Legacy schema entries are not authoritative after dataset resolution."""
    data_config = {
        "dataset_name": "criteo",
        "row_id_column": "wrong_id",
        "feature_columns": ["wrong_feature"],
        "treatment_column": "wrong_treatment",
        "split_column": "wrong_split",
        "outcome_columns": ["wrong_outcome"],
    }

    dataset_spec = resolve_dataset_spec(data_config)

    assert dataset_spec is CRITEO_SPEC
    assert dataset_spec.row_id_column == "row_id"
    assert dataset_spec.feature_columns == tuple(
        f"f{index}" for index in range(12)
    )
    assert dataset_spec.treatment_column == "treatment"
    assert dataset_spec.split_column == "split"
    assert dataset_spec.outcome_columns == ("visit", "conversion")


def test_obsolete_schema_keys_are_not_required_for_dataset_resolution() -> None:
    """Pipeline boundaries resolve schema from dataset_name alone."""
    assert resolve_dataset_spec({"dataset_name": "criteo"}) is CRITEO_SPEC


def test_unsupported_outcome_is_rejected_against_dataset_spec() -> None:
    """Outcome validation remains DatasetSpec-owned."""
    with pytest.raises(ValueError, match="supports outcomes"):
        validate_outcome("exposure", CRITEO_SPEC)


def test_processed_data_path_remains_outcome_configurable(tmp_path) -> None:
    """Configured input paths still vary by requested outcome."""
    data_config = {
        "dataset_name": "criteo",
        "processed_paths": {
            "visit": "visit.parquet",
            "conversion": "conversion.parquet",
        },
    }

    path = get_processed_data_path(data_config, "conversion", tmp_path)

    assert path == tmp_path / "conversion.parquet"


def test_training_frame_loads_columns_from_dataset_spec(tmp_path) -> None:
    """Stable columns come from the resolved DatasetSpec, not config."""
    dataset_spec = DatasetSpec(
        name="synthetic",
        row_id_column="customer_id",
        treatment_column="mail_flag",
        split_column="partition",
        feature_columns=("age", "score"),
        outcome_columns=("purchase",),
    )
    parquet_path = tmp_path / "training.parquet"
    pd.DataFrame(
        {
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
        "customer_id",
        "age",
        "score",
        "mail_flag",
        "partition",
        "purchase",
    ]
