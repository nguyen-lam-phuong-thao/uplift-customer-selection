# Experiment Manifest

Standard validation evaluation and locked-test evaluation must receive one
experiment manifest. They resolve prediction artifacts only from
`prediction_artifacts`; they do not scan the prediction directory for the latest
run.

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
