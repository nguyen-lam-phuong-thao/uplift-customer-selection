"""MLflow and DagsHub tracking setup."""

import dagshub
import mlflow


def setup_mlflow(
    experiment_name: str = "criteo-uplift-modeling",
    repo_owner: str = "nguyen-lam-phuong-thao",
    repo_name: str = "uplift-customer-selection",
) -> None:
    """Initialize DagsHub MLflow tracking for the project."""
    dagshub.init(
        repo_owner=repo_owner,
        repo_name=repo_name,
        mlflow=True,
    )

    mlflow.set_experiment(experiment_name)
