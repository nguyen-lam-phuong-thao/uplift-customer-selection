"""Tests for the Criteo prediction evaluation pipeline wiring."""

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from uplift_modeling.pipelines import evaluate_criteo_predictions as pipeline


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal evaluation config for pipeline tests."""
    config_path = tmp_path / "config.yaml"
    prediction_dir = tmp_path / "predictions"
    metric_dir = tmp_path / "metrics"
    figure_dir = tmp_path / "figures"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_name: criteo",
                "outputs:",
                f"  prediction_dir: {prediction_dir}",
                f"  metric_dir: {metric_dir}",
                f"  figure_dir: {figure_dir}",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_manifest(tmp_path: Path, outcome: str = "visit") -> Path:
    """Write a minimal manifest with an existing prediction reference."""
    prediction_path = (
        tmp_path
        / f"criteo_{outcome}_t_learner_lgbm_run01_predictions.parquet"
    )
    _prediction_frame("t_learner_lgbm").to_parquet(
        prediction_path,
        index=False,
    )
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "experiment_manifest",
                "experiment_id": "exp-001",
                "dataset_name": "criteo",
                "outcome": outcome,
                "config_path": str(tmp_path / "config.yaml"),
                "prediction_artifacts": {
                    "t_learner_lgbm": str(prediction_path),
                },
                "model_artifacts": {
                    "t_learner_lgbm": {
                        "artifact_type": "model_provenance",
                        "dataset_name": "criteo",
                        "outcome": outcome,
                        "policy_name": "t_learner_lgbm",
                        "prediction_artifact": prediction_path.name,
                        "model_kind": "t_learner",
                        "mlflow_run_id": "t-run-01",
                        "treatment_model_uri": "runs:/t-run-01/treatment_model",
                        "control_model_uri": "runs:/t-run-01/control_model",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _prediction_frame(model_name: str, score: float = 0.8) -> pd.DataFrame:
    """Return a small prediction artifact frame."""
    return pd.DataFrame(
        {
            "row_id": list(range(4)),
            "treatment": [1, 0, 1, 0],
            "outcome": [1, 0, 0, 1],
            "split": ["validation"] * 4,
            "score": [score, 0.7, 0.6, 0.5],
            "model_name": [model_name] * 4,
        }
    )


def test_filter_evaluation_splits_keeps_validation_only() -> None:
    """Standard evaluation does not expose test prediction rows."""
    predictions = pd.DataFrame(
        {
            "split": ["validation", "test", "train"],
            "score": [0.3, 0.2, 0.1],
        }
    )

    filtered = pipeline.filter_evaluation_splits(predictions)

    assert filtered["split"].tolist() == ["validation"]


def test_get_bootstrap_splits_rejects_test() -> None:
    """Standard bootstrap split requests cannot include test."""
    with pytest.raises(ValueError, match="test split is reserved"):
        pipeline.get_bootstrap_splits(("test",))


def test_standard_metric_calculation_rejects_test_rows() -> None:
    """Standard evaluation must not calculate test metrics."""
    predictions = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "treatment": [1, 0, 1, 0],
            "outcome": [1, 0, 1, 0],
            "split": ["validation", "validation", "test", "test"],
            "score": [0.4, 0.3, 0.2, 0.1],
            "model_name": ["model"] * 4,
        }
    )

    with pytest.raises(ValueError, match="test split is reserved"):
        pipeline.calculate_split_metrics(predictions, top_fraction=0.5)


def test_standard_artifact_loading_rejects_test_rows(tmp_path) -> None:
    """Standard evaluation fails before accepting artifacts with test rows."""
    prediction_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "treatment": [1, 0, 1, 0],
            "outcome": [1, 0, 1, 0],
            "split": ["validation", "validation", "test", "test"],
            "score": [0.4, 0.3, 0.2, 0.1],
            "model_name": ["model"] * 4,
        }
    ).to_parquet(prediction_path, index=False)

    with pytest.raises(ValueError, match="test split is reserved"):
        pipeline.load_prediction_artifacts([prediction_path])


def test_skip_bootstrap_skips_bootstrap_and_selection_gate(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    """Skip-bootstrap still saves Top-K artifacts and skips later gates."""
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    calls = {
        "topk": False,
        "bootstrap": False,
        "selection_gate": False,
    }

    def fake_prepare_topk_policy_frames(**kwargs):
        return {"policy": object()}, {"policy": tmp_path / "pred.parquet"}, []

    def fake_save_topk_policy_evaluation_artifacts(**kwargs):
        calls["topk"] = True
        return tmp_path / "topk.json", {}

    def fake_save_bootstrap_policy_evaluation(**kwargs):
        calls["bootstrap"] = True
        raise AssertionError("bootstrap should be skipped")

    def fake_save_model_selection_gate(**kwargs):
        calls["selection_gate"] = True
        raise AssertionError("Selection Gate should be skipped")

    monkeypatch.setattr(
        pipeline,
        "prepare_topk_policy_frames",
        fake_prepare_topk_policy_frames,
    )
    monkeypatch.setattr(
        pipeline,
        "save_topk_policy_evaluation_artifacts",
        fake_save_topk_policy_evaluation_artifacts,
    )
    monkeypatch.setattr(
        pipeline,
        "save_bootstrap_policy_evaluation",
        fake_save_bootstrap_policy_evaluation,
    )
    monkeypatch.setattr(
        pipeline,
        "save_model_selection_gate",
        fake_save_model_selection_gate,
    )

    with caplog.at_level(logging.INFO):
        pipeline.evaluate_predictions(
            config_path=config_path,
            manifest_path=manifest_path,
            outcome="visit",
            top_fraction=0.3,
            curve_num_points=100,
            random_seed=42,
            n_bootstrap=3,
            topk_only=True,
            skip_bootstrap=True,
        )

    assert calls == {
        "topk": True,
        "bootstrap": False,
        "selection_gate": False,
    }
    assert "Skipping bootstrap evaluation and Selection Gate" in caplog.text


def test_evaluate_predictions_passes_bootstrap_filters(
    tmp_path,
    monkeypatch,
) -> None:
    """Bootstrap split and budget filters are passed to the writer."""
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    captured_bootstrap_kwargs = {}
    captured_selection_kwargs = {}  
    def fake_prepare_topk_policy_frames(**kwargs):
        return (
            {
                "random_targeting": object(),
                "t_learner_lgbm": object(),
                "x_learner_lgbm": object(),
            },
            {
                "t_learner_lgbm": tmp_path / "t_learner.parquet",
                "x_learner_lgbm": tmp_path / "x_learner.parquet",
            },
            [],
        )

    def fake_save_topk_policy_evaluation_artifacts(**kwargs):
        return tmp_path / "topk.json", {}

    def fake_save_bootstrap_policy_evaluation(**kwargs):
        captured_bootstrap_kwargs.update(kwargs)
        return (
            tmp_path / "bootstrap_policy.json",
            tmp_path / "bootstrap_contrast.json",
            {
                "paired_contrast_rows": [
                    {"policy": "random_targeting"},
                    {"policy": "t_learner_lgbm"},
                    {"policy": "x_learner_lgbm"},
                ]
            },
        )

    def fake_save_model_selection_gate(**kwargs):
        captured_selection_kwargs.update(kwargs)
        return tmp_path / "selection.json", {}

    monkeypatch.setattr(
        pipeline,
        "prepare_topk_policy_frames",
        fake_prepare_topk_policy_frames,
    )
    monkeypatch.setattr(
        pipeline,
        "save_topk_policy_evaluation_artifacts",
        fake_save_topk_policy_evaluation_artifacts,
    )
    monkeypatch.setattr(
        pipeline,
        "save_bootstrap_policy_evaluation",
        fake_save_bootstrap_policy_evaluation,
    )
    monkeypatch.setattr(
        pipeline,
        "save_model_selection_gate",
        fake_save_model_selection_gate,
    )

    pipeline.evaluate_predictions(
        config_path=config_path,
        manifest_path=manifest_path,
        outcome="visit",
        top_fraction=0.3,
        curve_num_points=100,
        random_seed=42,
        n_bootstrap=3,
        topk_only=True,
        bootstrap_splits=("validation",),
        bootstrap_budget_fractions=(0.05,),
    )

    assert captured_bootstrap_kwargs["bootstrap_splits"] == ("validation",)
    assert captured_bootstrap_kwargs["budget_fractions"] == (0.05,)
    assert [
        row["policy"]
        for row in captured_selection_kwargs[
            "bootstrap_payload"
        ]["paired_contrast_rows"]
    ] == [
        "t_learner_lgbm",
        "x_learner_lgbm",
    ]

def test_evaluate_predictions_defaults_model_comparison_to_validation(
    tmp_path,
    monkeypatch,
) -> None:
    """Top-K, bootstrap, and Selection Gate inputs are validation-only."""
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    captured_topk_kwargs = {}
    captured_bootstrap_kwargs = {}
    captured_selection_kwargs = {}
    bootstrap_payload = {
        "paired_contrast_rows": [
            {
                "policy": "t_learner_lgbm",
                "baseline_policy": "treated_response_lgbm",
                "split": "validation",
                "outcome": "visit",
                "budget_fraction": 0.05,
                "metric": "policy_value",
                "mean_delta": 0.1,
                "ci_lower": 0.01,
            }
        ]
    }

    def fake_prepare_topk_policy_frames(**kwargs):
        captured_topk_kwargs.update(kwargs)
        return (
            {"t_learner_lgbm": object()},
            {"t_learner_lgbm": tmp_path / "pred.parquet"},
            [],
        )

    def fake_save_topk_policy_evaluation_artifacts(**kwargs):
        return tmp_path / "topk.json", {}

    def fake_save_bootstrap_policy_evaluation(**kwargs):
        captured_bootstrap_kwargs.update(kwargs)
        return (
            tmp_path / "bootstrap_policy.json",
            tmp_path / "bootstrap_contrast.json",
            bootstrap_payload,
        )

    def fake_save_model_selection_gate(**kwargs):
        captured_selection_kwargs.update(kwargs)
        return tmp_path / "selection.json", {}

    monkeypatch.setattr(
        pipeline,
        "prepare_topk_policy_frames",
        fake_prepare_topk_policy_frames,
    )
    monkeypatch.setattr(
        pipeline,
        "save_topk_policy_evaluation_artifacts",
        fake_save_topk_policy_evaluation_artifacts,
    )
    monkeypatch.setattr(
        pipeline,
        "save_bootstrap_policy_evaluation",
        fake_save_bootstrap_policy_evaluation,
    )
    monkeypatch.setattr(
        pipeline,
        "save_model_selection_gate",
        fake_save_model_selection_gate,
    )

    pipeline.evaluate_predictions(
        config_path=config_path,
        manifest_path=manifest_path,
        outcome="visit",
        top_fraction=0.3,
        curve_num_points=100,
        random_seed=42,
        n_bootstrap=3,
        topk_only=True,
    )

    assert captured_topk_kwargs["evaluation_splits"] == ("validation",)
    assert captured_bootstrap_kwargs["bootstrap_splits"] == ("validation",)
    assert captured_selection_kwargs["settings"].split == "validation"
    assert {
        row["split"]
        for row in captured_selection_kwargs["bootstrap_payload"][
            "paired_contrast_rows"
        ]
    } == {"validation"}


def test_evaluate_predictions_runs_locked_test_after_selection(
    tmp_path,
    monkeypatch,
) -> None:
    """Full evaluation runs Locked Test with the selected champion."""
    config_path = _write_config(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    prediction_path = (
        tmp_path
        / "criteo_visit_t_learner_lgbm_run01_predictions.parquet"
    )

    _prediction_frame(
        "t_learner_lgbm",
        score=0.8,
    ).to_parquet(
        prediction_path,
        index=False,
    )

    captured_locked_test_kwargs = {}

    def fake_prepare_topk_policy_frames(**kwargs):
        return (
            {
                "t_learner_lgbm": object(),
            },
            {
                "t_learner_lgbm": prediction_path,
            },
            [],
        )

    def fake_save_topk_policy_evaluation_artifacts(**kwargs):
        return tmp_path / "topk.json", {}

    def fake_save_bootstrap_policy_evaluation(**kwargs):
        return (
            tmp_path / "bootstrap_policy.json",
            tmp_path / "bootstrap_contrast.json",
            {
                "paired_contrast_rows": [
                    {
                        "policy": "t_learner_lgbm",
                    }
                ]
            },
        )

    def fake_save_model_selection_gate(**kwargs):
        return tmp_path / "selection.json", {}

    def fake_evaluate_locked_test(**kwargs):
        captured_locked_test_kwargs.update(kwargs)
        return tmp_path / "locked_test.json"

    monkeypatch.setattr(
        pipeline,
        "prepare_topk_policy_frames",
        fake_prepare_topk_policy_frames,
    )
    monkeypatch.setattr(
        pipeline,
        "save_topk_policy_evaluation_artifacts",
        fake_save_topk_policy_evaluation_artifacts,
    )
    monkeypatch.setattr(
        pipeline,
        "save_bootstrap_policy_evaluation",
        fake_save_bootstrap_policy_evaluation,
    )
    monkeypatch.setattr(
        pipeline,
        "save_model_selection_gate",
        fake_save_model_selection_gate,
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_locked_test",
        fake_evaluate_locked_test,
    )

    pipeline.evaluate_predictions(
        config_path=config_path,
        manifest_path=manifest_path,
        outcome="visit",
        top_fraction=0.5,
        curve_num_points=10,
        random_seed=42,
        n_bootstrap=3,
        topk_only=False,
    )

    assert captured_locked_test_kwargs == {
        "config_path": config_path,
        "manifest_path": manifest_path,
        "selection_artifact_path": tmp_path / "selection.json",
        "outcome": "visit",
        "n_bootstrap": 3,
        "random_seed": 42,
    }

def test_selection_gate_uses_current_cli_outcome(
    tmp_path,
    monkeypatch,
) -> None:
    """Selection Gate uses the outcome being evaluated, not stale YAML outcome."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_name: criteo",
                "outputs:",
                f"  prediction_dir: {tmp_path / 'predictions'}",
                f"  metric_dir: {tmp_path / 'metrics'}",
                f"  figure_dir: {tmp_path / 'figures'}",
                "selection:",
                "  primary_outcome: visit",
                "  primary_split: validation",
                "  primary_budget_fraction: 0.05",
                "  primary_metric: policy_value",
                "  baseline_policy: treated_response_lgbm",
            ]
        ),
        encoding="utf-8",
    )
    captured_selection_kwargs = {}
    manifest_path = _write_manifest(tmp_path, outcome="conversion")

    def fake_prepare_topk_policy_frames(**kwargs):
        return {"policy": object()}, {"policy": tmp_path / "pred.parquet"}, []

    def fake_save_topk_policy_evaluation_artifacts(**kwargs):
        return tmp_path / "topk.json", {}

    def fake_save_bootstrap_policy_evaluation(**kwargs):
        return (
            tmp_path / "bootstrap_policy.json",
            tmp_path / "bootstrap_contrast.json",
            {"paired_contrast_rows": []},
        )

    def fake_save_model_selection_gate(**kwargs):
        captured_selection_kwargs.update(kwargs)
        return tmp_path / "selection.json", {}

    monkeypatch.setattr(
        pipeline,
        "prepare_topk_policy_frames",
        fake_prepare_topk_policy_frames,
    )
    monkeypatch.setattr(
        pipeline,
        "save_topk_policy_evaluation_artifacts",
        fake_save_topk_policy_evaluation_artifacts,
    )
    monkeypatch.setattr(
        pipeline,
        "save_bootstrap_policy_evaluation",
        fake_save_bootstrap_policy_evaluation,
    )
    monkeypatch.setattr(
        pipeline,
        "save_model_selection_gate",
        fake_save_model_selection_gate,
    )

    pipeline.evaluate_predictions(
        config_path=config_path,
        manifest_path=manifest_path,
        outcome="conversion",
        top_fraction=0.3,
        curve_num_points=100,
        random_seed=42,
        n_bootstrap=3,
        topk_only=True,
    )

    settings = captured_selection_kwargs["settings"]
    assert settings.outcome == "conversion"
    assert settings.split == "validation"
    assert settings.budget_fraction == 0.05


def test_selection_gate_rejects_test_primary_split(tmp_path) -> None:
    """Selection Gate config cannot select from test results."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_name: criteo",
                "selection:",
                "  primary_split: test",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="test split is reserved"):
        pipeline.get_selection_gate_settings(
            pipeline.load_yaml_config(config_path),
            outcome="visit",
        )


def test_standard_evaluation_uses_only_manifest_listed_artifacts(tmp_path) -> None:
    """A newer unlisted prediction file does not change standard evaluation."""
    config_path = _write_config(tmp_path)
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    manifest_prediction_path = (
        prediction_dir
        / "criteo_visit_t_learner_lgbm_run01_predictions.parquet"
    )
    newer_prediction_path = (
        prediction_dir
        / "criteo_visit_t_learner_lgbm_run99_predictions.parquet"
    )
    _prediction_frame("t_learner_lgbm", score=0.8).to_parquet(
        manifest_prediction_path,
        index=False,
    )
    _prediction_frame("t_learner_lgbm", score=0.0).to_parquet(
        newer_prediction_path,
        index=False,
    )
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "experiment_manifest",
                "experiment_id": "exp-001",
                "dataset_name": "criteo",
                "outcome": "visit",
                "config_path": str(config_path),
                "prediction_artifacts": {
                    "t_learner_lgbm": str(manifest_prediction_path),
                },
            }
        ),
        encoding="utf-8",
    )

    pipeline.evaluate_predictions(
        config_path=config_path,
        manifest_path=manifest_path,
        outcome="visit",
        top_fraction=0.3,
        curve_num_points=10,
        random_seed=42,
        n_bootstrap=3,
        topk_only=False,
        skip_bootstrap=True,
    )

    evaluation_path = (
        tmp_path
        / "metrics"
        / "criteo_visit_t_learner_lgbm_run01_evaluation.json"
    )
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))

    assert payload["prediction_artifacts"] == [manifest_prediction_path.name]

def test_standard_artifact_loading_checks_split_before_full_read(
    tmp_path,
    monkeypatch,
) -> None:
    """Standard evaluation checks split before loading full prediction rows."""
    prediction_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "treatment": [1, 0, 1, 0],
            "outcome": [1, 0, 1, 0],
            "split": ["validation", "validation", "test", "test"],
            "score": [0.4, 0.3, 0.2, 0.1],
            "model_name": ["model"] * 4,
        }
    ).to_parquet(prediction_path, index=False)

    original_read_parquet = pipeline.pd.read_parquet
    read_columns = []

    def tracking_read_parquet(*args, **kwargs):
        read_columns.append(kwargs.get("columns"))
        if kwargs.get("columns") is None:
            raise AssertionError("Full artifact should not be loaded before rejection.")
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pipeline.pd, "read_parquet", tracking_read_parquet)

    with pytest.raises(ValueError, match="test split is reserved"):
        pipeline.load_prediction_artifacts([prediction_path])

    assert read_columns == [["split"]]
