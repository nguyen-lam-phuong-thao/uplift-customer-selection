"""Focused tests for validation-stage prediction evaluation."""

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from uplift_modeling.pipelines import evaluate_predictions as pipeline


DATASET_NAME = "synthetic"
OUTCOME = "visit"
MODEL_NAME = "t_learner_lgbm"


def _write_dataset_config(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text(
        "\n".join(
            [
                "dataset:",
                f"  name: {DATASET_NAME}",
                f"  prepared_path: {(tmp_path / 'prepared.parquet').as_posix()}",
                "schema:",
                "  treatment_column: treatment",
                "  split_column: split",
                "  feature_columns:",
                "    - f0",
                "  outcome_columns:",
                "    - visit",
                "    - conversion",
                "split:",
                "  assign_if_missing: false",
                "  train_size: 0.6",
                "  validation_size: 0.2",
                "  test_size: 0.2",
                "  random_state: 42",
                "outputs:",
                "  processed_paths:",
                f"    visit: {(tmp_path / 'decision_visit.parquet').as_posix()}",
                f"    conversion: {(tmp_path / 'decision_conversion.parquet').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_modeling_config(
    tmp_path: Path,
    *,
    primary_split: str = "validation",
) -> Path:
    path = tmp_path / "modeling.yaml"
    path.write_text(
        "\n".join(
            [
                "outputs:",
                f"  prediction_dir: {(tmp_path / 'predictions').as_posix()}",
                f"  metric_dir: {(tmp_path / 'metrics').as_posix()}",
                f"  figure_dir: {(tmp_path / 'figures').as_posix()}",
                "selection:",
                f"  primary_split: {primary_split}",
                "  primary_budget_fraction: 0.05",
                "  primary_metric: policy_value",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _prediction_frame(
    model_name: str = MODEL_NAME,
    score: float = 0.8,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "treatment": [1, 0, 1, 0],
            "outcome": [1, 0, 0, 1],
            "split": ["validation"] * 4,
            "score": [score, 0.7, 0.6, 0.5],
            "model_name": [model_name] * 4,
        }
    )


def _write_manifest(
    tmp_path: Path,
    dataset_config_path: Path,
    modeling_config_path: Path,
    *,
    outcome: str = OUTCOME,
    prediction_path: Path | None = None,
) -> Path:
    if prediction_path is None:
        prediction_path = (
            tmp_path
            / f"{DATASET_NAME}_{outcome}_{MODEL_NAME}_run01_predictions.parquet"
        )
        _prediction_frame().to_parquet(prediction_path, index=False)

    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "experiment_manifest",
                "experiment_id": "exp-001",
                "dataset_name": DATASET_NAME,
                "outcome": outcome,
                "dataset_config_path": str(dataset_config_path.resolve()),
                "modeling_config_path": str(modeling_config_path.resolve()),
                "prediction_artifacts": {
                    MODEL_NAME: str(prediction_path),
                },
                "model_artifacts": {
                    MODEL_NAME: {
                        "artifact_type": "model_provenance",
                        "dataset_name": DATASET_NAME,
                        "outcome": outcome,
                        "policy_name": MODEL_NAME,
                        "prediction_artifact": prediction_path.name,
                        "model_kind": "t_learner",
                        "mlflow_run_id": "run-001",
                        "treatment_model_uri": "runs:/run-001/treatment_model",
                        "control_model_uri": "runs:/run-001/control_model",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _evaluation_files(
    tmp_path: Path,
    *,
    outcome: str = OUTCOME,
) -> tuple[Path, Path, Path]:
    dataset_config_path = _write_dataset_config(tmp_path)
    modeling_config_path = _write_modeling_config(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        dataset_config_path,
        modeling_config_path,
        outcome=outcome,
    )
    return dataset_config_path, modeling_config_path, manifest_path


def _topk_payload() -> dict:
    return {
        "table_rows": [
            {
                "policy_name": MODEL_NAME,
                "split": "validation",
                "budget_fraction": 0.05,
                "policy_value": 0.1,
            }
        ]
    }


def test_standard_evaluation_never_accepts_test_rows(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    frame = _prediction_frame()
    frame.loc[0, "split"] = "test"
    frame.to_parquet(prediction_path, index=False)

    with pytest.raises(ValueError, match="test split is reserved"):
        pipeline.load_prediction_artifacts([prediction_path])


def test_skip_bootstrap_skips_bootstrap_and_selection_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog,
) -> None:
    dataset_config_path, modeling_config_path, manifest_path = (
        _evaluation_files(tmp_path)
    )
    calls = {
        "topk": False,
        "bootstrap": False,
        "selection": False,
    }

    monkeypatch.setattr(
        pipeline,
        "prepare_topk_policy_frames",
        lambda **kwargs: (
            {MODEL_NAME: object()},
            {MODEL_NAME: tmp_path / "pred.parquet"},
            [],
        ),
    )

    def fake_topk(**kwargs):
        calls["topk"] = True
        return tmp_path / "topk.json", {}

    def fail_bootstrap(**kwargs):
        calls["bootstrap"] = True
        raise AssertionError("bootstrap should be skipped")

    def fail_selection(**kwargs):
        calls["selection"] = True
        raise AssertionError("selection gate should be skipped")

    monkeypatch.setattr(
        pipeline,
        "save_topk_policy_evaluation_artifacts",
        fake_topk,
    )
    monkeypatch.setattr(
        pipeline,
        "save_bootstrap_policy_evaluation",
        fail_bootstrap,
    )
    monkeypatch.setattr(
        pipeline,
        "save_model_selection_gate",
        fail_selection,
    )

    with caplog.at_level(logging.INFO):
        pipeline.evaluate_predictions(
            dataset_config_path=dataset_config_path,
            modeling_config_path=modeling_config_path,
            manifest_path=manifest_path,
            outcome=OUTCOME,
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
        "selection": False,
    }


def test_bootstrap_and_selection_use_validation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_config_path, modeling_config_path, manifest_path = (
        _evaluation_files(tmp_path)
    )
    captured_topk = {}
    captured_bootstrap = {}
    captured_selection = {}

    def fake_prepare(**kwargs):
        captured_topk.update(kwargs)
        return (
            {
                "random_targeting": object(),
                MODEL_NAME: object(),
            },
            {
                MODEL_NAME: tmp_path / "pred.parquet",
            },
            [],
        )

    monkeypatch.setattr(
        pipeline,
        "prepare_topk_policy_frames",
        fake_prepare,
    )
    monkeypatch.setattr(
        pipeline,
        "save_topk_policy_evaluation_artifacts",
        lambda **kwargs: (tmp_path / "topk.json", _topk_payload()),
    )

    def fake_bootstrap(**kwargs):
        captured_bootstrap.update(kwargs)
        return (
            tmp_path / "bootstrap.json",
            tmp_path / "contrast.json",
            {
                "paired_contrast_rows": [
                    {
                        "policy": "random_targeting",
                        "split": "validation",
                    },
                    {
                        "policy": MODEL_NAME,
                        "split": "validation",
                    },
                ]
            },
        )

    def fake_selection(**kwargs):
        captured_selection.update(kwargs)
        return tmp_path / "selection.json", {}

    monkeypatch.setattr(
        pipeline,
        "save_bootstrap_policy_evaluation",
        fake_bootstrap,
    )
    monkeypatch.setattr(
        pipeline,
        "save_model_selection_gate",
        fake_selection,
    )

    pipeline.evaluate_predictions(
        dataset_config_path=dataset_config_path,
        modeling_config_path=modeling_config_path,
        manifest_path=manifest_path,
        outcome=OUTCOME,
        top_fraction=0.3,
        curve_num_points=100,
        random_seed=42,
        n_bootstrap=3,
        topk_only=True,
        bootstrap_splits=("validation",),
        bootstrap_budget_fractions=(0.05,),
    )

    assert captured_topk["evaluation_splits"] == ("validation",)
    assert captured_bootstrap["bootstrap_splits"] == ("validation",)
    assert captured_bootstrap["budget_fractions"] == (0.05,)
    assert captured_selection["settings"].split == "validation"

    assert (
        captured_selection["topk_json_path"]
        == tmp_path / "topk.json"
    )

    assert [
        row["policy"]
        for row in captured_selection[
            "bootstrap_payload"
        ]["paired_contrast_rows"]
    ] == [MODEL_NAME]


def test_selection_gate_uses_current_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_config_path, modeling_config_path, manifest_path = (
        _evaluation_files(
            tmp_path,
            outcome="conversion",
        )
    )
    captured_selection = {}

    monkeypatch.setattr(
        pipeline,
        "prepare_topk_policy_frames",
        lambda **kwargs: (
            {MODEL_NAME: object()},
            {MODEL_NAME: tmp_path / "pred.parquet"},
            [],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "save_topk_policy_evaluation_artifacts",
        lambda **kwargs: (tmp_path / "topk.json", _topk_payload()),
    )
    monkeypatch.setattr(
        pipeline,
        "save_bootstrap_policy_evaluation",
        lambda **kwargs: (
            tmp_path / "bootstrap.json",
            tmp_path / "contrast.json",
            {"paired_contrast_rows": []},
        ),
    )

    def fake_selection(**kwargs):
        captured_selection.update(kwargs)
        return tmp_path / "selection.json", {}

    monkeypatch.setattr(
        pipeline,
        "save_model_selection_gate",
        fake_selection,
    )

    pipeline.evaluate_predictions(
        dataset_config_path=dataset_config_path,
        modeling_config_path=modeling_config_path,
        manifest_path=manifest_path,
        outcome="conversion",
        top_fraction=0.3,
        curve_num_points=100,
        random_seed=42,
        n_bootstrap=3,
        topk_only=True,
    )

    assert captured_selection["settings"].outcome == "conversion"


def test_evaluation_requires_same_configs_recorded_in_manifest(
    tmp_path: Path,
) -> None:
    dataset_config_path, modeling_config_path, manifest_path = (
        _evaluation_files(tmp_path)
    )

    other_modeling_config = tmp_path / "other_modeling.yaml"
    other_modeling_config.write_text(
        modeling_config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Modeling config does not match",
    ):
        pipeline.evaluate_predictions(
            dataset_config_path=dataset_config_path,
            modeling_config_path=other_modeling_config,
            manifest_path=manifest_path,
            outcome=OUTCOME,
            top_fraction=0.3,
            curve_num_points=100,
            random_seed=42,
            n_bootstrap=3,
            topk_only=True,
            skip_bootstrap=True,
        )


def test_standard_evaluation_uses_only_manifest_listed_artifact(
    tmp_path: Path,
) -> None:
    dataset_config_path = _write_dataset_config(tmp_path)
    modeling_config_path = _write_modeling_config(tmp_path)

    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()

    selected_path = (
        prediction_dir
        / f"{DATASET_NAME}_{OUTCOME}_{MODEL_NAME}_run01_predictions.parquet"
    )
    newer_path = (
        prediction_dir
        / f"{DATASET_NAME}_{OUTCOME}_{MODEL_NAME}_run99_predictions.parquet"
    )

    _prediction_frame(score=0.8).to_parquet(selected_path, index=False)
    _prediction_frame(score=0.0).to_parquet(newer_path, index=False)

    manifest_path = _write_manifest(
        tmp_path,
        dataset_config_path,
        modeling_config_path,
        prediction_path=selected_path,
    )

    pipeline.evaluate_predictions(
        dataset_config_path=dataset_config_path,
        modeling_config_path=modeling_config_path,
        manifest_path=manifest_path,
        outcome=OUTCOME,
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
        / f"{DATASET_NAME}_{OUTCOME}_{MODEL_NAME}_run01_evaluation.json"
    )
    payload = json.loads(
        evaluation_path.read_text(encoding="utf-8")
    )

    assert payload["prediction_artifacts"] == [selected_path.name]

def test_selection_gate_rejects_unsupported_primary_budget() -> None:
    with pytest.raises(
        ValueError,
        match="primary_budget_fraction must be one of",
    ):
        pipeline.get_selection_gate_settings(
            {
                "selection": {
                    "primary_split": "validation",
                    "primary_budget_fraction": 0.15,
                    "primary_metric": "policy_value",
                }
            },
            outcome="visit",
        )
