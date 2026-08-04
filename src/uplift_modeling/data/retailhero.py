from collections.abc import Collection
from pathlib import Path

import pandas as pd


RETAILHERO_FILES: dict[str, str] = {
    "clients": "clients.csv",
    "products": "products.csv",
    "purchases": "purchases.csv",
    "uplift_train": "uplift_train.csv",
    "uplift_test": "uplift_test.csv",
    "sample_submission": "uplift_sample_submission.csv",
}

RETAILHERO_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "clients": ("client_id",),
    "products": ("product_id",),
    "purchases": (
        "client_id",
        "transaction_id",
        "transaction_datetime",
        "product_id",
        "purchase_sum",
    ),
    "uplift_train": ("client_id", "treatment_flg", "target"),
    "uplift_test": ("client_id",),
    "sample_submission": ("client_id",),
}


def _validate_file_path(file_path: Path) -> None:
    """Validate that the raw file path points to a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"RetailHero raw file does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"RetailHero raw path is not a file: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            "RetailHero raw file must be a .csv file. "
            f"Received: {file_path}"
        )


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Collection[str],
    table_name: str,
) -> None:
    """Raise an error when required columns are missing."""
    missing_columns = sorted(
        set(required_columns).difference(dataframe.columns)
    )

    if missing_columns:
        missing_columns_text = ", ".join(missing_columns)
        raise ValueError(
            f"RetailHero {table_name} is missing required columns: "
            f"{missing_columns_text}"
        )


def get_retailhero_paths(raw_dir: str | Path) -> dict[str, Path]:
    """Return validated RetailHero raw file paths."""
    raw_path = Path(raw_dir).expanduser()

    if not raw_path.exists():
        raise FileNotFoundError(
            f"RetailHero raw directory does not exist: {raw_path}"
        )

    if not raw_path.is_dir():
        raise ValueError(
            f"RetailHero raw path is not a directory: {raw_path}"
        )

    data_paths = {
        table_name: raw_path / file_name
        for table_name, file_name in RETAILHERO_FILES.items()
    }

    for file_path in data_paths.values():
        _validate_file_path(file_path)

    return data_paths


def load_retailhero_table(
    raw_dir: str | Path,
    table_name: str,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load one RetailHero raw table with pandas."""
    if nrows is not None and nrows < 0:
        raise ValueError(
            f"nrows must be non-negative or None. Received: {nrows}"
        )

    data_paths = get_retailhero_paths(raw_dir)

    if table_name not in data_paths:
        valid_tables = ", ".join(sorted(data_paths))
        raise ValueError(
            f"Unsupported RetailHero table: {table_name}. "
            f"Expected one of: {valid_tables}"
        )

    dataframe = pd.read_csv(
        data_paths[table_name],
        nrows=nrows,
    )

    _validate_required_columns(
        dataframe=dataframe,
        required_columns=RETAILHERO_REQUIRED_COLUMNS[table_name],
        table_name=table_name,
    )

    return dataframe


def load_retailhero_raw(
    raw_dir: str | Path,
    nrows: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load small RetailHero raw tables.

    Purchases is excluded because it can be large.
    Use DuckDB for purchases in EDA or feature engineering.
    """
    return {
        "clients": load_retailhero_table(raw_dir, "clients", nrows=nrows),
        "products": load_retailhero_table(raw_dir, "products", nrows=nrows),
        "uplift_train": load_retailhero_table(
            raw_dir,
            "uplift_train",
            nrows=nrows,
        ),
        "uplift_test": load_retailhero_table(
            raw_dir,
            "uplift_test",
            nrows=nrows,
        ),
    }