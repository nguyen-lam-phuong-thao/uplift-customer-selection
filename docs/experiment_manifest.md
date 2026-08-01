# Experiment Manifest

Standard validation evaluation and locked-test evaluation must receive one
experiment manifest. They resolve prediction artifacts only from
`prediction_artifacts`; they do not scan the prediction directory for the latest
run.

Create a manifest from the latest run-numbered prediction artifacts:

```bash
python -m uplift_modeling.pipelines.create_experiment_manifest \
  --config configs/modeling/criteo_response_lgbm.yaml \
  --outcome conversion
```

The default output path is:

```text
artifacts/metrics/criteo_conversion_experiment_manifest.json
```

Then pass that file to evaluation:

```bash
python -m uplift_modeling.pipelines.evaluate_criteo_predictions \
  --config configs/modeling/criteo_response_lgbm.yaml \
  --manifest artifacts/metrics/criteo_conversion_experiment_manifest.json \
  --outcome conversion
```

Use the same manifest for locked-test reporting:

```bash
python -m uplift_modeling.pipelines.evaluate_locked_test \
  --config configs/modeling/criteo_response_lgbm.yaml \
  --selection-artifact artifacts/metrics/criteo_conversion_model_selection_gate_run01.json \
  --manifest artifacts/metrics/criteo_conversion_experiment_manifest.json \
  --outcome conversion
```

Required JSON fields:

```json
{
  "artifact_type": "experiment_manifest",
  "experiment_id": "criteo-visit-2026-07-31",
  "dataset_name": "criteo",
  "outcome": "visit",
  "config_path": "configs/modeling/criteo_response_lgbm.yaml",
  "prediction_artifacts": {
    "treated_response_lgbm": "artifacts/predictions/criteo_visit_treated_response_lgbm_run01_predictions.parquet",
    "t_learner_lgbm": "artifacts/predictions/criteo_visit_t_learner_lgbm_run01_predictions.parquet",
    "x_learner_lgbm": "artifacts/predictions/criteo_visit_x_learner_lgbm_run01_predictions.parquet"
  }
}
```

`config_identity` may be used instead of `config_path`. Prediction artifact
entries must be explicit files. Directory references, glob patterns, duplicate
policy/model keys, duplicate artifact paths, missing files, dataset mismatches,
and outcome mismatches are rejected before evaluation.
