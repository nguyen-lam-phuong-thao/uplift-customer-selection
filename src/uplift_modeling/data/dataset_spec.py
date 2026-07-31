"""Stable dataset schema contracts."""

from dataclasses import dataclass

from uplift_modeling.data.row_id import ROW_ID_COLUMN


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable column contract for one supported dataset."""

    name: str
    row_id_column: str
    treatment_column: str
    split_column: str
    feature_columns: tuple[str, ...]
    outcome_columns: tuple[str, ...]


CRITEO_SPEC = DatasetSpec(
    name="criteo",
    row_id_column=ROW_ID_COLUMN,
    treatment_column="treatment",
    split_column="split",
    feature_columns=tuple(f"f{index}" for index in range(12)),
    outcome_columns=("visit", "conversion"),
)

DATASET_SPECS: dict[str, DatasetSpec] = {
    CRITEO_SPEC.name: CRITEO_SPEC,
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    """Return the supported dataset spec for a configured dataset name."""
    normalized_name = dataset_name.strip().lower()

    try:
        return DATASET_SPECS[normalized_name]
    except KeyError as error:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(
            "Unknown dataset name. "
            f"Supported datasets: {supported}. Received: {dataset_name}"
        ) from error


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
