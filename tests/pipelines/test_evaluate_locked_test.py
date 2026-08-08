"""Focused tests for locked-test pipeline wiring."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift_modeling.pipelines import evaluate_locked_test as pipeline


DATASET_NAME = "synthetic"
OUTCOME = "visit"
EXPERIMENT_ID = "exp-001"
CHAMPION_POLICY = "t_learner_lgbm"


def _write_decision_dataset(tmp_path: Path) -> Path:
    rows = []
    splits = ["train"] * 2 + ["validation"] * 2 + ["test"] * 20
    for row_id, split in enumerate(splits):
        row = {
            "row_id": row_id,
            "treatment": row_id % 2,
            "split": split,
            OUTCOME: int(row_id % 5 == 0),
        }
        row.update({f"f{i}": float(row_id + i) for i in range(12)})
        rows.append(row)

    path = tmp_path / "synthetic_decision_visit.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_dataset_config(tmp_path: Path, data_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    features = [f"    - f{i}" for i in range(12)]
    path.write_text(
        "\n".join(
            [
                "dataset:",
                f"  name: {DATASET_NAME}",
                f"  prepared_path: {data_path.as_posix()}",
                "schema:",
                "  treatment_column: treatment",
                "  split_column: split",
                "  feature_columns:",
                *features,
                "  outcome_columns:",
                f"    - {OUTCOME}",
                "split:",
                "  assign_if_missing: false",
                "  train_size: 0.6",
                "  validation_size: 0.2",
                "  test_size: 0.2",
                "  random_state: 42",
                "outputs:",
                "  processed_paths:",
                f"    {OUTCOME}: {data_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_modeling_config(tmp_path: Path) -> Path:
    path = tmp_path / "modeling.yaml"
    path.write_text(
        "\n".join(
            [
                "project:",
                "  experiment_name: test-experiment",
                "training:",
                "  prediction_batch_size: 2",
                "outputs:",
                f"  prediction_dir: {(tmp_path / 'predictions').as_posix()}",
                f"  metric_dir: {(tmp_path / 'metrics').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _model_artifact(policy: str, run_id: str | None = None) -> dict:
    run_id = run_id or f"run-{policy}"
    base = {
        "artifact_type": "model_provenance",
        "dataset_name": DATASET_NAME,
        "outcome": OUTCOME,
        "policy_name": policy,
        "prediction_artifact": (
            f"{DATASET_NAME}_{OUTCOME}_{policy}_run01_predictions.parquet"
        ),
        "mlflow_run_id": run_id,
    }

    if policy == CHAMPION_POLICY:
        return {
            **base,
            "model_kind": "t_learner",
            "treatment_model_uri": f"runs:/{run_id}/treatment_model",
            "control_model_uri": f"runs:/{run_id}/control_model",
        }

    return {
        **base,
        "model_kind": "response",
        "model_uri": f"runs:/{run_id}/model",
    }


def _write_validation_prediction(path: Path, policy: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": [2, 3],
            "treatment": [0, 1],
            "outcome": [0, 1],
            "split": ["validation", "validation"],
            "score": [0.2, 0.8],
            "model_name": [policy, policy],
        }
    ).to_parquet(path, index=False)


def _write_manifest(
    tmp_path: Path,
    dataset_config_path: Path,
    modeling_config_path: Path,
    *,
    include_champion_prediction: bool = True,
) -> Path:
    validation_dir = tmp_path / "validation_predictions"
    policies = ("treated_response_lgbm", CHAMPION_POLICY)
    prediction_artifacts = {}
    model_artifacts = {}

    for policy in policies:
        artifact = _model_artifact(policy)
        prediction_path = validation_dir / artifact["prediction_artifact"]
        if policy != CHAMPION_POLICY or include_champion_prediction:
            _write_validation_prediction(prediction_path, policy)
            prediction_artifacts[policy] = str(prediction_path)
        model_artifacts[policy] = artifact

    path = tmp_path / "experiment_manifest.json"
    path.write_text(
        json.dumps(
            {
                "artifact_type": "experiment_manifest",
                "experiment_id": EXPERIMENT_ID,
                "dataset_name": DATASET_NAME,
                "outcome": OUTCOME,
                "dataset_config_path": str(dataset_config_path.resolve()),
                "modeling_config_path": str(modeling_config_path.resolve()),
                "prediction_artifacts": prediction_artifacts,
                "model_artifacts": model_artifacts,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_selection(
    tmp_path: Path,
    manifest_path: Path,
    *,
    source_manifest_path: Path | None = None,
    champion_artifact: dict | None = None,
) -> Path:
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "artifact_type": "model_selection_gate",
                "experiment_id": EXPERIMENT_ID,
                "dataset_name": DATASET_NAME,
                "outcome": OUTCOME,
                "source_manifest_path": str(
                    (source_manifest_path or manifest_path).resolve()
                ),
                "champion_policy": CHAMPION_POLICY,
                "champion_model_artifact": (
                    champion_artifact or _model_artifact(CHAMPION_POLICY)
                ),
                "selection_settings": {
                    "outcome": OUTCOME,
                    "split": "validation",
                    "budget_fraction": 0.1,
                    "metric": "policy_value",
                    "baseline_policy": "treated_response_lgbm",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    data_path = _write_decision_dataset(tmp_path)
    dataset_config_path = _write_dataset_config(tmp_path, data_path)
    modeling_config_path = _write_modeling_config(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        dataset_config_path,
        modeling_config_path,
    )
    selection_path = _write_selection(tmp_path, manifest_path)
    return manifest_path, selection_path


def _patch_runtime(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    loaded = []

    def build_score_batch(policy: str, model_artifact: dict):
        loaded.append((policy, model_artifact["mlflow_run_id"]))

        def score_batch(features: pd.DataFrame) -> np.ndarray:
            return np.arange(len(features), dtype=float)

        return score_batch

    def save_final(**kwargs):
        metric_dir = kwargs["metric_dir"]
        metric_dir.mkdir(parents=True, exist_ok=True)
        output_path = metric_dir / "locked_test_result.json"
        output_path.write_text("{}", encoding="utf-8")
        return output_path, {}

    monkeypatch.setattr(pipeline, "build_policy_score_batch", build_score_batch)
    monkeypatch.setattr(pipeline, "setup_mlflow", lambda _: None)
    monkeypatch.setattr(pipeline, "_save_locked_test_evaluation", save_final)
    return loaded


def test_locked_test_scores_only_exact_champion_on_test_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, selection_path = _setup(tmp_path)
    loaded = _patch_runtime(monkeypatch)

    output_path = pipeline.evaluate_locked_test(
        manifest_path=manifest_path,
        selection_artifact_path=selection_path,
        n_bootstrap=3,
        random_seed=42,
    )

    assert output_path.exists()
    assert loaded == [(CHAMPION_POLICY, f"run-{CHAMPION_POLICY}")]

    prediction_paths = list((tmp_path / "predictions").glob("*.parquet"))
    assert len(prediction_paths) == 1
    frame = pd.read_parquet(prediction_paths[0])
    assert frame["row_id"].tolist() == list(range(4, 24))
    assert set(frame["split"]) == {"test"}


def test_locked_test_source_frame_does_not_resplit(tmp_path: Path) -> None:
    data_path = _write_decision_dataset(tmp_path)
    dataset_config_path = _write_dataset_config(tmp_path, data_path)
    dataset_config = pipeline.load_dataset_config(
        dataset_config_path,
        project_root=tmp_path,
    )

    frame, target = pipeline.load_locked_test_source_frame(
        parquet_path=data_path,
        dataset_spec=dataset_config.spec,
        outcome=OUTCOME,
    )

    assert target == OUTCOME
    assert frame["row_id"].tolist() == list(range(4, 24))
    assert set(frame["split"]) == {"test"}


def test_locked_test_requires_champion_validation_prediction(tmp_path: Path) -> None:
    data_path = _write_decision_dataset(tmp_path)
    dataset_config_path = _write_dataset_config(tmp_path, data_path)
    modeling_config_path = _write_modeling_config(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        dataset_config_path,
        modeling_config_path,
        include_champion_prediction=False,
    )
    selection_path = _write_selection(tmp_path, manifest_path)

    with pytest.raises(ValueError, match=CHAMPION_POLICY):
        pipeline.evaluate_locked_test(
            manifest_path=manifest_path,
            selection_artifact_path=selection_path,
        )


def test_locked_test_rejects_selection_from_other_manifest(tmp_path: Path) -> None:
    data_path = _write_decision_dataset(tmp_path)
    dataset_config_path = _write_dataset_config(tmp_path, data_path)
    modeling_config_path = _write_modeling_config(tmp_path)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first_manifest = _write_manifest(
        first_dir,
        dataset_config_path,
        modeling_config_path,
    )
    second_manifest = _write_manifest(
        second_dir,
        dataset_config_path,
        modeling_config_path,
    )
    selection_path = _write_selection(
        tmp_path,
        second_manifest,
        source_manifest_path=first_manifest,
    )

    with pytest.raises(ValueError, match="does not reference"):
        pipeline.evaluate_locked_test(
            manifest_path=second_manifest,
            selection_artifact_path=selection_path,
        )


def test_locked_test_rejects_different_champion_model_run(tmp_path: Path) -> None:
    manifest_path, _ = _setup(tmp_path)
    selection_path = _write_selection(
        tmp_path,
        manifest_path,
        champion_artifact=_model_artifact(
            CHAMPION_POLICY,
            run_id="different-run",
        ),
    )

    with pytest.raises(ValueError, match="does not match the exact model"):
        pipeline.evaluate_locked_test(
            manifest_path=manifest_path,
            selection_artifact_path=selection_path,
        )