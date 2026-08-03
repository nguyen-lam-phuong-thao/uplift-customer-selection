"""MLflow tracking setup."""

import mlflow


def setup_mlflow(experiment_name: str) -> None:
    """Set the MLflow experiment using the configured tracking environment."""
    mlflow.set_experiment(experiment_name)