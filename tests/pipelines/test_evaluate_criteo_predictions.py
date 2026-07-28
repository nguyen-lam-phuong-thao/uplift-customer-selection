"""Tests for the Criteo prediction evaluation pipeline wiring."""

import logging
from pathlib import Path

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


def test_skip_bootstrap_skips_bootstrap_and_selection_gate(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    """Skip-bootstrap still saves Top-K artifacts and skips later gates."""
    config_path = _write_config(tmp_path)
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
    captured_bootstrap_kwargs = {}

    def fake_prepare_topk_policy_frames(**kwargs):
        return {"policy": object()}, {"policy": tmp_path / "pred.parquet"}, []

    def fake_save_topk_policy_evaluation_artifacts(**kwargs):
        return tmp_path / "topk.json", {}

    def fake_save_bootstrap_policy_evaluation(**kwargs):
        captured_bootstrap_kwargs.update(kwargs)
        return (
            tmp_path / "bootstrap_policy.json",
            tmp_path / "bootstrap_contrast.json",
            {"paired_contrast_rows": []},
        )

    def fake_save_model_selection_gate(**kwargs):
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
