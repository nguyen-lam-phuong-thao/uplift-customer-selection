"""Data contract and standardization helpers."""

from uplift_modeling.data.dataset_spec import (
    DatasetConfig,
    DatasetSpec,
    SplitConfig,
    get_dataset_spec,
    load_dataset_config,
    validate_supported_outcome,
)
from uplift_modeling.data.preparation import (
    build_decision_dataset,
    save_decision_dataset,
)
from uplift_modeling.data.row_id import (
    ROW_ID_COLUMN,
    add_row_id,
    align_frames_by_row_id,
    validate_row_id_column,
)
from uplift_modeling.data.split import (
    assign_stratified_split,
    ensure_split_column,
    validate_split_column,
)
from uplift_modeling.data.standardization import (
    load_prepared_table,
    standardize_prepared_dataset,
)
from uplift_modeling.data.validation import validate_prepared_dataset_contract

__all__ = [
    "DatasetConfig",
    "DatasetSpec",
    "SplitConfig",
    "get_dataset_spec",
    "load_dataset_config",
    "validate_supported_outcome",
    "build_decision_dataset",
    "save_decision_dataset",
    "ROW_ID_COLUMN",
    "add_row_id",
    "align_frames_by_row_id",
    "validate_row_id_column",
    "assign_stratified_split",
    "ensure_split_column",
    "validate_split_column",
    "load_prepared_table",
    "standardize_prepared_dataset",
    "validate_prepared_dataset_contract",
]