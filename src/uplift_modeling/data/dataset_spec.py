"""Typed dataset contracts loaded from dataset YAML configs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uplift_modeling.data.row_id import ROW_ID_COLUMN
from uplift_modeling.utils.config import load_yaml_config, resolve_project_path


VALID_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetSpec:
    """Column contract for a prepared modeling dataset.

    YAML is only an input format. After loading YAML, framework code should use
    this object instead of passing config dictionaries around.
    """

    name: str
    treatment_column: str
    split_column: str
    feature_columns: Sequence[str]
    outcome_columns: Sequence[str]
    entity_id_column: str | None = None

    def __post_init__(self) -> None:
        name = _as_non_empty_string(self.name, "dataset.name").strip().lower()

        treatment_column = _as_non_empty_string(
            self.treatment_column,
            "schema.treatment_column",
        )

        split_column = _as_non_empty_string(
            self.split_column,
            "schema.split_column",
        )

        feature_columns = _as_string_tuple(
            self.feature_columns,
            "schema.feature_columns",
        )

        outcome_columns = _as_string_tuple(
            self.outcome_columns,
            "schema.outcome_columns",
        )

        entity_id_column = (
            _as_non_empty_string(
                self.entity_id_column,
                "schema.entity_id_column",
            )
            if self.entity_id_column is not None
            else None
        )

        if (
            entity_id_column is not None
            and entity_id_column in feature_columns
        ):
            raise ValueError(
                "schema.entity_id_column must not be a model feature."
            )

        reserved_columns = {
            ROW_ID_COLUMN,
            treatment_column,
            split_column,
            *outcome_columns,
        }
        overlapping_features = sorted(set(feature_columns).intersection(reserved_columns))
        if overlapping_features:
            raise ValueError(
                "schema.feature_columns must not contain row_id, treatment, "
                f"split, or outcome columns. Invalid: {overlapping_features}"
            )

        duplicated_features = _find_duplicates(feature_columns)
        if duplicated_features:
            raise ValueError(
                f"schema.feature_columns contains duplicate columns: {duplicated_features}"
            )

        duplicated_outcomes = _find_duplicates(outcome_columns)
        if duplicated_outcomes:
            raise ValueError(
                f"schema.outcome_columns contains duplicate columns: {duplicated_outcomes}"
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "treatment_column", treatment_column)
        object.__setattr__(self, "split_column", split_column)
        object.__setattr__(self, "feature_columns", feature_columns)
        object.__setattr__(self, "outcome_columns", outcome_columns)
        object.__setattr__(
            self,
            "entity_id_column",
            entity_id_column,
        )


@dataclass(frozen=True)
class SplitConfig:
    """Train/validation/test split configuration."""

    assign_if_missing: bool
    train_size: float
    validation_size: float
    test_size: float
    random_state: int

    def __post_init__(self) -> None:
        train_size = float(self.train_size)
        validation_size = float(self.validation_size)
        test_size = float(self.test_size)
        random_state = int(self.random_state)

        split_sizes = {
            "split.train_size": train_size,
            "split.validation_size": validation_size,
            "split.test_size": test_size,
        }

        invalid_sizes = {
            name: value
            for name, value in split_sizes.items()
            if value <= 0 or value >= 1
        }
        if invalid_sizes:
            raise ValueError(
                f"Split sizes must be between 0 and 1. Invalid: {invalid_sizes}"
            )

        total = train_size + validation_size + test_size
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split sizes must sum to 1. Received: {total}")

        object.__setattr__(self, "train_size", train_size)
        object.__setattr__(self, "validation_size", validation_size)
        object.__setattr__(self, "test_size", test_size)
        object.__setattr__(self, "random_state", random_state)


@dataclass(frozen=True)
class DatasetConfig:
    """Full dataset config used by standardization."""

    spec: DatasetSpec
    prepared_path: Path
    processed_paths: Mapping[str, Path]
    split: SplitConfig

    def __post_init__(self) -> None:
        prepared_path = Path(self.prepared_path)

        processed_paths = {
            str(outcome): Path(path)
            for outcome, path in self.processed_paths.items()
        }

        missing_outputs = sorted(
            set(self.spec.outcome_columns).difference(processed_paths)
        )
        if missing_outputs:
            raise ValueError(
                "outputs.processed_paths must define a path for each outcome. "
                f"Missing: {missing_outputs}"
            )

        object.__setattr__(self, "prepared_path", prepared_path)
        object.__setattr__(self, "processed_paths", processed_paths)


def load_dataset_config(
    config_path: str | Path,
    project_root: Path | None = None,
) -> DatasetConfig:
    """Load a dataset YAML config into typed framework objects."""
    path = Path(config_path)
    config = load_yaml_config(path)

    if project_root is None:
        project_root = Path.cwd()

    dataset = config["dataset"]
    schema = config["schema"]
    split = config.get("split", {})
    outputs = config["outputs"]

    if "row_id_column" in schema:
        raise ValueError(
            "schema.row_id_column is not configurable."
            "The framework owns the internal 'row_id' column"
        )
    
    spec = DatasetSpec(
        name=dataset["name"],
        entity_id_column=schema.get("entity_id_column"),
        treatment_column=schema["treatment_column"],
        split_column=schema.get("split_column", "split"),
        feature_columns=schema["feature_columns"],
        outcome_columns=schema["outcome_columns"],
    )

    processed_paths = {
        outcome: resolve_project_path(path_value, project_root)
        for outcome, path_value in outputs["processed_paths"].items()
    }

    return DatasetConfig(
        spec=spec,
        prepared_path=resolve_project_path(dataset["prepared_path"], project_root),
        processed_paths=processed_paths,
        split=SplitConfig(
            assign_if_missing=bool(split.get("assign_if_missing", True)),
            train_size=split.get("train_size", 0.6),
            validation_size=split.get("validation_size", 0.2),
            test_size=split.get("test_size", 0.2),
            random_state=split.get("random_state", 42),
        ),
    )


def get_dataset_spec(
    dataset_name: str,
    config_dir: str | Path = "configs/datasets",
) -> DatasetSpec:
    """Load a dataset spec by dataset name.

    This keeps existing training/evaluation code reusable while removing
    hard-coded dataset specs from Python source.
    """
    normalized_name = dataset_name.strip().lower()
    config_path = Path(config_dir) / f"{normalized_name}.yaml"
    return load_dataset_config(config_path).spec


def validate_supported_outcome(
    dataset_spec: DatasetSpec,
    outcome_column: str,
) -> None:
    """Raise when an outcome is not supported by the selected dataset."""
    if outcome_column not in dataset_spec.outcome_columns:
        supported = ", ".join(dataset_spec.outcome_columns)
        raise ValueError(
            f"Dataset '{dataset_spec.name}' supports outcomes: {supported}. "
            f"Received: {outcome_column}"
        )


def _as_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _as_string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a YAML list of strings.")

    columns = tuple(_as_non_empty_string(item, label) for item in value)

    if not columns:
        raise ValueError(f"{label} must contain at least one column.")

    return columns


def _find_duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []

    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)

    return duplicates