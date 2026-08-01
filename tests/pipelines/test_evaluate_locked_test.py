"""Tests for locked-test pipeline scoring wiring."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift_modeling.pipelines import evaluate_locked_test as pipeline


def _write_config(
    tmp_path: Path,
    data_path: Path,
    locked_test_split: str | None = None,
) -> Path:
    """Write a minimal locked-test config."""
    config_path = tmp_path / "config.yaml"

    config_lines = [
        "project:",
        "  experiment_name: test-experiment",
        "data:",
        "  dataset_name: criteo",
        "  processed_paths:",
        f"    visit: {data_path}",
        "training:",
        "  prediction_batch_size: 2",
        "outputs:",
        f"  prediction_dir: {tmp_path / 'predictions'}",
        f"  metric_dir: {tmp_path / 'metrics'}",
    ]

    if locked_test_split is not None:
        config_lines.extend(
            [
                "locked_test:",
                f"  split: {locked_test_split}",
            ]
        )

    config_path.write_text(
        "\n".join(config_lines),
        encoding="utf-8",
    )
    return config_path


def _write_prepared_dataset(tmp_path: Path) -> Path:
    """Write fixed train, validation, and test rows."""
    rows = []
    for row_id, split in enumerate(
        ["train"] * 2
        + ["validation"] * 2
        + ["test"] * 20
    ):
        row = {
            "row_id": row_id,
            "treatment": row_id % 2,
            "split": split,
            "visit": int(row_id in {2, 4}),
        }
        row.update({f"f{index}": float(row_id + index) for index in range(12)})
        rows.append(row)

    data_path = tmp_path / "criteo_decision_visit.parquet"
    pd.DataFrame(rows).to_parquet(data_path, index=False)
    return data_path


def _write_selection_artifact(
    tmp_path: Path,
    champion_policy: str,
) -> Path:
    """Write a minimal Selection Gate artifact."""
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "artifact_type": "model_selection_gate",
                "dataset_name": "criteo",
                "outcome": "visit",
                "champion_policy": champion_policy,
                "selection_settings": {
                    "outcome": "visit",
                    "split": "validation",
                    "budget_fraction": 0.1,
                    "metric": "policy_value",
                    "baseline_policy": "treated_response_lgbm",
                },
            }
        ),
        encoding="utf-8",
    )
    return selection_path


def _write_manifest(
    tmp_path: Path,
    config_path: Path,
    champion_policy: str = "t_learner_lgbm",
    include_champion_provenance: bool = True,
    include_ignored_candidates: bool = True,
    include_champion_prediction: bool = True,
) -> Path:
    """Write a validation manifest with model provenance."""
    prediction_dir = tmp_path / "validation_predictions"
    prediction_dir.mkdir()
    policies = []
    if include_ignored_candidates:
        policies.append("treated_response_lgbm")
    if include_champion_prediction:
        policies.append(champion_policy)
    if include_ignored_candidates:
        policies.append("x_learner_lgbm")

    prediction_artifacts = {}
    model_artifacts = {}
    for policy in policies:
        prediction_path = (
            prediction_dir / f"criteo_visit_{policy}_run01_predictions.parquet"
        )
        prediction_path.touch()
        prediction_artifacts[policy] = str(prediction_path)
        if policy == champion_policy and not include_champion_provenance:
            continue
        if policy != champion_policy:
            continue
        model_artifacts[policy] = {
            "artifact_type": "model_provenance",
            "dataset_name": "criteo",
            "outcome": "visit",
            "policy_name": policy,
            "prediction_artifact": prediction_path.name,
            "model_kind": "t_learner",
            "mlflow_run_id": f"run-{policy}",
            "model_uri": f"runs:/run-{policy}/model",
            "treatment_model_uri": f"runs:/run-{policy}/treatment_model",
            "control_model_uri": f"runs:/run-{policy}/control_model",
        }

    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "experiment_manifest",
                "experiment_id": "exp-001",
                "dataset_name": "criteo",
                "outcome": "visit",
                "config_path": str(config_path),
                "prediction_artifacts": prediction_artifacts,
                "model_artifacts": model_artifacts,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _patch_model_boundary(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Mock MLflow-backed model loading at the scoring boundary."""
    loaded: list[tuple[str, str]] = []

    def build_score_batch(policy: str, model_artifact: dict):
        loaded.append((policy, str(model_artifact["mlflow_run_id"])))

        def score_batch(features: pd.DataFrame) -> np.ndarray:
            offset = 1.0 if policy == "t_learner_lgbm" else 0.0
            return np.arange(len(features), dtype=float) + offset

        return score_batch

    monkeypatch.setattr(pipeline, "build_policy_score_batch", build_score_batch)
    monkeypatch.setattr(pipeline, "setup_mlflow", lambda experiment_name: None)
    return loaded


def test_locked_test_pipeline_scores_existing_test_rows_only(
    tmp_path,
    monkeypatch,
) -> None:
    """Locked test filters fixed test rows and scores only the champion."""
    data_path = _write_prepared_dataset(tmp_path)
    config_path = _write_config(tmp_path, data_path)
    manifest_path = _write_manifest(tmp_path, config_path)
    selection_path = _write_selection_artifact(tmp_path, "t_learner_lgbm")
    loaded = _patch_model_boundary(monkeypatch)

    output_path = pipeline.evaluate_locked_test(
        config_path=config_path,
        manifest_path=manifest_path,
        selection_artifact_path=selection_path,
        outcome="visit",
    )

    assert output_path.exists()
    assert loaded == [("t_learner_lgbm", "run-t_learner_lgbm")]

    prediction_paths = sorted((tmp_path / "predictions").glob("*.parquet"))
    assert len(prediction_paths) == 1
    assert prediction_paths[0].name.endswith(
        "t_learner_lgbm_run01_locked_test_predictions.parquet"
    )
    frame = pd.read_parquet(prediction_paths[0])
    assert frame["row_id"].tolist() == list(range(4, 24))
    assert set(frame["split"]) == {"test"}
    assert "row_id" in frame.columns

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "baseline_policy" not in payload
    assert payload["prediction_artifacts"] == {
        "t_learner_lgbm": prediction_paths[0].name,
    }
    assert {row["policy"] for row in payload["locked_test_rows"]} == {
        "t_learner_lgbm"
    }


def test_locked_test_pipeline_requires_existing_selection_artifact(
    tmp_path,
) -> None:
    """Locked test starts from an already-written Selection Gate artifact."""
    data_path = _write_prepared_dataset(tmp_path)
    config_path = _write_config(tmp_path, data_path)
    manifest_path = _write_manifest(tmp_path, config_path)

    with pytest.raises(FileNotFoundError, match="Selection Gate artifact"):
        pipeline.evaluate_locked_test(
            config_path=config_path,
            manifest_path=manifest_path,
            selection_artifact_path=tmp_path / "missing_selection.json",
            outcome="visit",
        )


def test_locked_test_pipeline_fails_without_champion_provenance(tmp_path) -> None:
    """A selected champion without model provenance fails clearly."""
    data_path = _write_prepared_dataset(tmp_path)
    config_path = _write_config(tmp_path, data_path)
    manifest_path = _write_manifest(
        tmp_path,
        config_path,
        include_champion_provenance=False,
    )
    selection_path = _write_selection_artifact(tmp_path, "t_learner_lgbm")

    with pytest.raises(ValueError, match="t_learner_lgbm"):
        pipeline.evaluate_locked_test(
            config_path=config_path,
            manifest_path=manifest_path,
            selection_artifact_path=selection_path,
            outcome="visit",
        )


def test_locked_test_pipeline_ignores_baseline_and_extra_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    """Baseline and extra candidates are not required or loaded on test."""
    data_path = _write_prepared_dataset(tmp_path)
    config_path = _write_config(tmp_path, data_path)
    manifest_path = _write_manifest(tmp_path, config_path)
    selection_path = _write_selection_artifact(tmp_path, "t_learner_lgbm")
    loaded = _patch_model_boundary(monkeypatch)

    output_path = pipeline.evaluate_locked_test(
        config_path=config_path,
        manifest_path=manifest_path,
        selection_artifact_path=selection_path,
        outcome="visit",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == [("t_learner_lgbm", "run-t_learner_lgbm")]
    assert set(payload["prediction_artifacts"]) == {"t_learner_lgbm"}


def test_locked_test_source_frame_does_not_resplit(tmp_path) -> None:
    """Locked test preserves row IDs from the configured test split."""
    data_path = _write_prepared_dataset(tmp_path)

    test_frame, target_column = pipeline.load_locked_test_source_frame(
        parquet_path=data_path,
        dataset_spec=pipeline.get_dataset_spec("criteo"),
        outcome="visit",
    )

    assert target_column == "visit"
    assert test_frame["row_id"].tolist() == list(range(4, 24))


def test_locked_test_pipeline_rejects_non_test_split(
    tmp_path,
) -> None:
    """Locked test cannot be configured to score validation rows."""
    data_path = _write_prepared_dataset(tmp_path)
    config_path = _write_config(
        tmp_path,
        data_path,
        locked_test_split="validation",
    )
    manifest_path = _write_manifest(tmp_path, config_path)
    selection_path = _write_selection_artifact(
        tmp_path,
        "t_learner_lgbm",
    )

    with pytest.raises(ValueError, match="test split only"):
        pipeline.evaluate_locked_test(
            config_path=config_path,
            manifest_path=manifest_path,
            selection_artifact_path=selection_path,
            outcome="visit",
        )


def test_locked_test_pipeline_fails_without_champion_prediction(tmp_path) -> None:
    """A selected champion without a prediction artifact fails clearly."""
    data_path = _write_prepared_dataset(tmp_path)
    config_path = _write_config(tmp_path, data_path)
    manifest_path = _write_manifest(
        tmp_path,
        config_path,
        include_champion_prediction=False,
    )
    selection_path = _write_selection_artifact(tmp_path, "t_learner_lgbm")

    with pytest.raises(ValueError, match="t_learner_lgbm"):
        pipeline.evaluate_locked_test(
            config_path=config_path,
            manifest_path=manifest_path,
            selection_artifact_path=selection_path,
            outcome="visit",
        )
