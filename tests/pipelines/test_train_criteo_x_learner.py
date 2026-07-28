"""Tests for the Criteo X-Learner training pipeline wiring."""

from pathlib import Path


def test_x_learner_pipeline_logs_mlflow_metrics() -> None:
    """The X-Learner pipeline logs numeric diagnostics to MLflow."""
    project_root = Path(__file__).parents[2]
    pipeline_path = (
        project_root
        / "src"
        / "uplift_modeling"
        / "pipelines"
        / "train_criteo_x_learner.py"
    )
    source = pipeline_path.read_text(encoding="utf-8")

    assert "mlflow.log_metrics(mlflow_metrics)" in source
