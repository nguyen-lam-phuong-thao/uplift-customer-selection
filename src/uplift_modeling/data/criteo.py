from collections.abc import Collection
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS: tuple[str, ...] = tuple(f"f{index}" for index in range(12))

BINARY_COLUMNS: tuple[str, ...] = (
    "treatment",
    "conversion",
    "visit",
    "exposure",
)

CRITEO_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + BINARY_COLUMNS


def _validate_file_path(file_path: Path) -> None:
    """Validate that the dataset path points to a supported CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Criteo dataset file does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Criteo dataset path is not a file: {file_path}"
        )

    suffixes = file_path.suffixes
    is_csv = file_path.suffix.lower() == ".csv"
    is_compressed_csv = (
        len(suffixes) >= 2
        and [suffix.lower() for suffix in suffixes[-2:]]
        == [".csv", ".gz"]
    )

    if not (is_csv or is_compressed_csv):
        raise ValueError(
            "Criteo dataset must be a .csv or .csv.gz file. "
            f"Received: {file_path}"
        )


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Collection[str],
) -> None:
    """Raise an error when required columns are missing."""
    missing_columns = sorted(
        set(required_columns).difference(dataframe.columns)
    )

    if missing_columns:
        missing_columns_text = ", ".join(missing_columns)
        raise ValueError(
            "Criteo dataset is missing required columns: "
            f"{missing_columns_text}"
        )


def load_criteo(
    file_path: str | Path,
    nrows: int | None = None,
    required_columns: Collection[str] | None = None,
) -> pd.DataFrame:
    """Load the Criteo uplift dataset from a CSV file.

    Parameters
    ----------
    file_path:
        Relative or absolute path to a `.csv` or `.csv.gz` file.
    nrows:
        Optional maximum number of rows to load. Use `None` to load the
        complete dataset.
    required_columns:
        Optional collection of columns that must exist. When omitted,
        the complete confirmed Criteo schema is required.

    Returns
    -------
    pandas.DataFrame
        The loaded Criteo dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If the path is invalid, the file type is unsupported, `nrows`
        is negative, or required columns are missing.
    """
    if nrows is not None and nrows < 0:
        raise ValueError(
            f"nrows must be non-negative or None. Received: {nrows}"
        )

    path = Path(file_path).expanduser()
    _validate_file_path(path)

    columns_to_validate = (
        CRITEO_COLUMNS
        if required_columns is None
        else tuple(required_columns)
    )

    dataframe = pd.read_csv(path, nrows=nrows)
    _validate_required_columns(dataframe, columns_to_validate)

    return dataframe