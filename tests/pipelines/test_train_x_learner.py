"""Tests for the X-Learner training pipeline wiring."""

from pathlib import Path
import sys

import uplift_modeling.pipelines.train_x_learner as pipeline

def test_x_learner_pipeline_logs_mlflow_metrics() -> None:
    """The X-Learner pipeline logs numeric diagnostics to MLflow."""
    project_root = Path(__file__).parents[2]
    pipeline_path = (
        project_root
        / "src"
        / "uplift_modeling"
        / "pipelines"
        / "train_x_learner.py"
    )
    source = pipeline_path.read_text(encoding="utf-8")

    assert "mlflow.log_metrics(mlflow_metrics)" in source


def test_x_learner_cli_uses_shared_config_arguments(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_x_learner",
            "--dataset-config",
            "dataset.yaml",
            "--modeling-config",
            "modeling.yaml",
            "--outcome",
            "visit",
        ],
    )

    args = pipeline.parse_args()

    assert args.dataset_config == "dataset.yaml"
    assert args.modeling_config == "modeling.yaml"
    assert args.outcome == "visit"