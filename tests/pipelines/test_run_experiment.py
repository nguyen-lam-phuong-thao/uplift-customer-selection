"""Tests for the validation-stage experiment runner."""

from pathlib import Path

import uplift_modeling.pipelines.run_experiment as pipeline


def test_train_configured_candidates_dispatches_configured_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modeling_config = {
        "models": {
            "model_defaults": {
                "random_state": 42,
            },
            "candidates": [
                {
                    "name": "response_a",
                    "kind": "response",
                    "params": {},
                },
                {
                    "name": "response_b",
                    "kind": "response",
                    "params": {
                        "num_leaves": 63,
                    },
                },
                {
                    "name": "t_model",
                    "kind": "t_learner",
                    "params": {},
                },
                {
                    "name": "x_model",
                    "kind": "x_learner",
                    "params": {},
                },
            ],
        }
    }

    calls = []

    def make_pipeline(kind: str):
        def fake_pipeline(
            dataset_config_path,
            modeling_config_path,
            outcome,
            model_candidate,
        ):
            calls.append(
                (kind, model_candidate.name)
            )

            return (
                model_candidate.name,
                tmp_path / f"{model_candidate.name}.parquet",
            )

        return fake_pipeline
    
    monkeypatch.setitem(
        pipeline.TRAIN_PIPELINES,
        "response",
        make_pipeline("response"),
    )

    monkeypatch.setitem(
        pipeline.TRAIN_PIPELINES,
        "t_learner",
        make_pipeline("t_learner"),
    )

    monkeypatch.setitem(
        pipeline.TRAIN_PIPELINES,
        "x_learner",
        make_pipeline("x_learner"),
    )

    prediction_artifacts = (
        pipeline.train_configured_candidates(
            dataset_config_path=tmp_path / "dataset.yaml",
            modeling_config_path=tmp_path / "modeling.yaml",
            modeling_config=modeling_config,
            outcome="visit",
        )
    )

    assert calls == [
        ("response", "response_a"),
        ("response", "response_b"),
        ("t_learner", "t_model"),
        ("x_learner", "x_model"),
    ]

    assert prediction_artifacts == {
        "response_a": tmp_path / "response_a.parquet",
        "response_b": tmp_path / "response_b.parquet",
        "t_model": tmp_path / "t_model.parquet",
        "x_model": tmp_path / "x_model.parquet",
    }


def test_run_experiment_passes_exact_training_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_config_path = tmp_path / "dataset.yaml"
    modeling_config_path = tmp_path / "modeling.yaml"

    dataset_config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: synthetic",
                f"  prepared_path: {(tmp_path / 'prepared.parquet').as_posix()}",
                "schema:",
                "  treatment_column: treatment",
                "  split_column: split",
                "  feature_columns:",
                "    - f0",
                "  outcome_columns:",
                "    - visit",
                "split:",
                "  assign_if_missing: false",
                "  train_size: 0.6",
                "  validation_size: 0.2",
                "  test_size: 0.2",
                "  random_state: 42",
                "outputs:",
                "  processed_paths:",
                f"    visit: {(tmp_path / 'decision.parquet').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    modeling_config_path.write_text(
        "\n".join(
            [
                "models:",
                "  model_defaults: {}",
                "  candidates:",
                "    - name: response_model",
                "      kind: response",
                "      params: {}",
                "outputs:",
                f"  prediction_dir: {(tmp_path / 'predictions').as_posix()}",
                f"  metric_dir: {(tmp_path / 'metrics').as_posix()}",
                f"  figure_dir: {(tmp_path / 'figures').as_posix()}",
                "selection:",
                "  primary_split: validation",
                "  primary_budget_fraction: 0.05",
                "  primary_metric: policy_value",
            ]
        ),
        encoding="utf-8",
    )

    expected_predictions = {
        "response_model": tmp_path / "response.parquet",
    }

    manifest_path = tmp_path / "manifest.json"
    selection_path = tmp_path / "selection.json"

    observed_manifest_predictions = {}

    monkeypatch.setattr(
        pipeline,
        "standardize_prepared_dataset",
        lambda dataset_config: {
            "visit": tmp_path / "decision.parquet",
        },
    )

    monkeypatch.setattr(
        pipeline,
        "train_configured_candidates",
        lambda **kwargs: expected_predictions,
    )

    def fake_create_manifest(
        dataset_config_path,
        modeling_config_path,
        outcome,
        experiment_id,
        prediction_artifacts,
    ):
        observed_manifest_predictions.update(prediction_artifacts)
        return manifest_path

    monkeypatch.setattr(
        pipeline,
        "create_experiment_manifest",
        fake_create_manifest,
    )

    monkeypatch.setattr(
        pipeline,
        "evaluate_predictions",
        lambda **kwargs: selection_path,
    )

    result_manifest, result_selection = pipeline.run_experiment(
        dataset_config_path=dataset_config_path,
        modeling_config_path=modeling_config_path,
        experiment_id="synthetic-visit-001",
        outcome="visit",
    )

    assert observed_manifest_predictions == expected_predictions
    assert result_manifest == manifest_path
    assert result_selection == selection_path


def test_run_experiment_standardizes_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_config_path = tmp_path / "dataset.yaml"
    modeling_config_path = tmp_path / "modeling.yaml"

    dataset_config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: synthetic",
                f"  prepared_path: {(tmp_path / 'prepared.parquet').as_posix()}",
                "schema:",
                "  treatment_column: treatment",
                "  split_column: split",
                "  feature_columns:",
                "    - f0",
                "  outcome_columns:",
                "    - visit",
                "split:",
                "  assign_if_missing: false",
                "  train_size: 0.6",
                "  validation_size: 0.2",
                "  test_size: 0.2",
                "  random_state: 42",
                "outputs:",
                "  processed_paths:",
                f"    visit: {(tmp_path / 'decision.parquet').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    modeling_config_path.write_text(
        "\n".join(
            [
                "models:",
                "  model_defaults: {}",
                "  candidates:",
                "    - name: response_model",
                "      kind: response",
                "      params: {}",
                "outputs:",
                f"  prediction_dir: {(tmp_path / 'predictions').as_posix()}",
                f"  metric_dir: {(tmp_path / 'metrics').as_posix()}",
                f"  figure_dir: {(tmp_path / 'figures').as_posix()}",
                "selection:",
                "  primary_split: validation",
                "  primary_budget_fraction: 0.05",
                "  primary_metric: policy_value",
                "  baseline_policy: response_model",
            ]
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_standardize(dataset_config):
        calls.append("standardize")

        return {
            "visit": tmp_path / "decision.parquet",
        }

    def fake_train_configured_candidates(**kwargs):
        calls.append("train")

        return {
            "response_model": tmp_path / "response.parquet",
        }

    monkeypatch.setattr(
        pipeline,
        "standardize_prepared_dataset",
        fake_standardize,
    )

    monkeypatch.setattr(
        pipeline,
        "train_configured_candidates",
        fake_train_configured_candidates,
    )

    monkeypatch.setattr(
        pipeline,
        "create_experiment_manifest",
        lambda **kwargs: tmp_path / "manifest.json",
    )

    monkeypatch.setattr(
        pipeline,
        "evaluate_predictions",
        lambda **kwargs: tmp_path / "selection.json",
    )

    pipeline.run_experiment(
        dataset_config_path=dataset_config_path,
        modeling_config_path=modeling_config_path,
        experiment_id="synthetic-visit-001",
        outcome="visit",
    )

    assert calls == [
        "standardize",
        "train",
    ]


def test_train_configured_candidates_rejects_model_name_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modeling_config = {
        "models": {
            "model_defaults": {},
            "candidates": [
                {
                    "name": "expected_model",
                    "kind": "response",
                    "params": {},
                },
            ],
        }
    }

    def fake_pipeline(**kwargs):
        return (
            "wrong_model",
            tmp_path / "prediction.parquet",
        )

    monkeypatch.setitem(
        pipeline.TRAIN_PIPELINES,
        "response",
        fake_pipeline,
    )

    import pytest

    with pytest.raises(
        ValueError,
        match="returned model name",
    ):
        pipeline.train_configured_candidates(
            dataset_config_path=tmp_path / "dataset.yaml",
            modeling_config_path=tmp_path / "modeling.yaml",
            modeling_config=modeling_config,
            outcome="visit",
        )    