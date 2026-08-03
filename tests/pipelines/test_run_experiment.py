"""Tests for the complete validation-stage experiment runner."""

from pathlib import Path

import pytest

import uplift_modeling.pipelines.run_experiment as pipeline


def test_run_experiment_passes_exact_training_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    """The runner passes exact newly created predictions into the manifest."""
    response_config = tmp_path / "response.yaml"
    t_config = tmp_path / "t.yaml"
    x_config = tmp_path / "x.yaml"

    response_prediction = tmp_path / "response.parquet"
    t_prediction = tmp_path / "t.parquet"
    x_prediction = tmp_path / "x.parquet"
    manifest_path = tmp_path / "manifest.json"
    selection_path = tmp_path / "selection.json"

    captured = {}

    monkeypatch.setattr(
        pipeline,
        "validate_shared_config_sections",
        lambda config_paths: None,
    )
    monkeypatch.setattr(
        pipeline,
        "train_response_pipeline",
        lambda **kwargs: (
            "treated_response_lgbm",
            response_prediction,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "train_t_learner_pipeline",
        lambda **kwargs: (
            "t_learner_lgbm",
            t_prediction,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "train_x_learner_pipeline",
        lambda **kwargs: (
            "x_learner_lgbm",
            x_prediction,
        ),
    )

    def fake_create_experiment_manifest(**kwargs):
        captured["manifest"] = kwargs
        return manifest_path

    def fake_evaluate_predictions(**kwargs):
        captured["evaluation"] = kwargs
        return selection_path

    monkeypatch.setattr(
        pipeline,
        "create_experiment_manifest",
        fake_create_experiment_manifest,
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_predictions",
        fake_evaluate_predictions,
    )

    result = pipeline.run_experiment(
        experiment_id="exp-001",
        outcome="visit",
        response_config_path=response_config,
        t_learner_config_path=t_config,
        x_learner_config_path=x_config,
        n_bootstrap=10,
    )

    assert result == (manifest_path, selection_path)
    assert captured["manifest"]["prediction_artifacts"] == {
        "treated_response_lgbm": response_prediction,
        "t_learner_lgbm": t_prediction,
        "x_learner_lgbm": x_prediction,
    }
    assert captured["evaluation"]["manifest_path"] == manifest_path
    assert captured["evaluation"]["skip_bootstrap"] is False


def test_shared_config_validation_rejects_mismatch(
    tmp_path,
) -> None:
    """Candidate configs must use the same dataset and experiment settings."""
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"

    first_path.write_text(
        "\n".join(
            [
                "project:",
                "  experiment_name: uplift-test",
                "data:",
                "  dataset_name: criteo",
                "training:",
                "  train_split: train",
                "outputs:",
                "  prediction_dir: predictions",
                "selection:",
                "  baseline_policy: baseline",
                "model:",
                "  name: first",
            ]
        ),
        encoding="utf-8",
    )
    second_path.write_text(
        "\n".join(
            [
                "project:",
                "  experiment_name: uplift-test",
                "data:",
                "  dataset_name: retailhero",
                "training:",
                "  train_split: train",
                "outputs:",
                "  prediction_dir: predictions",
                "selection:",
                "  baseline_policy: baseline",
                "model:",
                "  name: second",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mismatched config"):
        pipeline.validate_shared_config_sections(
            (first_path, second_path)
        )