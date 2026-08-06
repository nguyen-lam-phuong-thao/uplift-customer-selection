from pathlib import Path


RETAILHERO_FILES: dict[str, str] = {
    "clients": "clients.csv",
    "products": "products.csv",
    "purchases": "purchases.csv",
    "uplift_train": "uplift_train.csv",
    "uplift_test": "uplift_test.csv",
    "sample_submission": "uplift_sample_submission.csv",
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