# Phase 0 — Exploratory V1 Report

## 1. Purpose of This Report

This report records the current Week 2 exploratory results for the Criteo uplift-modeling workflow. These results are useful as an initial benchmark, but they are not final confirmatory model-selection results.

All current results in this report are labeled `exploratory_v1`. The label means the artifacts are treated as exploratory targeting-policy evidence only, not as locked final causal model-selection evidence.

## 2. What Has Been Completed

The following work has been completed in the current `exploratory_v1` artifact set:

- Criteo dataset validation.
- Train/validation/test split.
- Exposure removed because it is leakage-prone.
- Decision datasets created for `visit` and `conversion`.
- Response model trained.
- T-Learner trained.
- X-Learner trained.
- Prediction artifacts saved.
- Qini, AUUC, policy value, and figures generated.

## 3. Current Exploratory Results

The tables below summarize values found in the current artifacts under `artifacts/metrics/`, `artifacts/predictions/`, and `artifacts/figures/`. Existing artifacts were inspected only; no training, evaluation, prediction generation, or artifact regeneration was run in Phase 0.

### Metric Artifacts Inspected

| File | Size |
| --- | ---: |
| `criteo_validation.json` | 1647 bytes |
| `criteo_visit_response_lgbm_run02_metrics.json` | 527 bytes |
| `criteo_conversion_response_lgbm_run01_metrics.json` | 544 bytes |
| `criteo_visit_response_lgbm_run03_evaluation.json` | 917 bytes |
| `criteo_conversion_response_lgbm_run02_evaluation.json` | 935 bytes |
| `criteo_visit_response_lgbm_vs_t_learner_lgbm_run01_evaluation.json` | 1702 bytes |
| `criteo_conversion_response_lgbm_vs_t_learner_lgbm_run01_evaluation.json` | 1730 bytes |
| `criteo_visit_response_lgbm_vs_t_learner_lgbm_vs_x_learner_lgbm_run01_evaluation.json` | 2487 bytes |
| `criteo_conversion_response_lgbm_vs_t_learner_lgbm_vs_x_learner_lgbm_run01_evaluation.json` | 2522 bytes |

### Dataset Validation Summary

| Field | Value |
| --- | --- |
| `data_path` | `/kaggle/input/datasets/nguynlmphngtho/criteo-uplif/criteo-uplift-v2.1.csv` |
| `data_scope` | `full_dataset` |
| `is_valid` | `true` |
| `is_empty` | `false` |
| `row_count` | `13979592` |
| `column_count` | `16` |
| `columns_with_nulls` | `[]` |
| `non_numeric_feature_columns` | `[]` |
| `invalid_binary_values` | `{}` |

| Column | Data Type | Null Count | Null Percentage |
| --- | --- | ---: | ---: |
| `f0` | `float64` | 0 | 0.0 |
| `f1` | `float64` | 0 | 0.0 |
| `f2` | `float64` | 0 | 0.0 |
| `f3` | `float64` | 0 | 0.0 |
| `f4` | `float64` | 0 | 0.0 |
| `f5` | `float64` | 0 | 0.0 |
| `f6` | `float64` | 0 | 0.0 |
| `f7` | `float64` | 0 | 0.0 |
| `f8` | `float64` | 0 | 0.0 |
| `f9` | `float64` | 0 | 0.0 |
| `f10` | `float64` | 0 | 0.0 |
| `f11` | `float64` | 0 | 0.0 |
| `treatment` | `int64` | 0 | 0.0 |
| `conversion` | `int64` | 0 | 0.0 |
| `visit` | `int64` | 0 | 0.0 |
| `exposure` | `int64` | 0 | 0.0 |

| Binary Column | Unique Values |
| --- | --- |
| `treatment` | `[0, 1]` |
| `conversion` | `[0, 1]` |
| `visit` | `[0, 1]` |
| `exposure` | `[0, 1]` |

### Binary Response Model Metrics

| Outcome | Split | Average Precision | Log Loss | Positive Rate | ROC AUC | Row Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `visit` | `validation` | 0.5203856233412801 | 0.1028327909928364 | 0.04699207916684252 | 0.9467894619504809 | 2795918 |
| `visit` | `test` | 0.5179900301582123 | 0.1031247629514257 | 0.046992062359460345 | 0.9464557873732276 | 2795919 |
| `conversion` | `validation` | 0.2320263883960803 | 0.011667576963871284 | 0.002916394543759867 | 0.9602780226493312 | 2795918 |
| `conversion` | `test` | 0.23729900639533824 | 0.011676235180525503 | 0.0029167511648227292 | 0.9590294589765966 | 2795919 |

### All-Model Evaluation Results

The table below uses the combined response, T-Learner, and X-Learner evaluation artifacts:

- `criteo_visit_response_lgbm_vs_t_learner_lgbm_vs_x_learner_lgbm_run01_evaluation.json`
- `criteo_conversion_response_lgbm_vs_t_learner_lgbm_vs_x_learner_lgbm_run01_evaluation.json`

Both artifacts report `curve_num_points = 100` and `top_fraction = 0.3`.

| Outcome | Split | Model | AUUC | Qini | Policy Value | Row Count | Positive Rate | Treatment Rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `visit` | `test` | `response_model_lgbm` | 20683.646695360934 | 8393.960928243134 | 0.04814370133168464 | 2795919 | 0.046992062359460345 | 0.8499999463503771 |
| `visit` | `validation` | `response_model_lgbm` | 20820.67596588617 | 8531.09843487066 | 0.04818227623394379 | 2795918 | 0.04699207916684252 | 0.8500002503649964 |
| `visit` | `test` | `t_learner_lgbm` | 20783.410507860957 | 8493.724740743157 | 0.048006663446104526 | 2795919 | 0.046992062359460345 | 0.8499999463503771 |
| `visit` | `validation` | `t_learner_lgbm` | 20868.813248825958 | 8579.235717810448 | 0.04803866535715332 | 2795918 | 0.04699207916684252 | 0.8500002503649964 |
| `visit` | `test` | `x_learner_lgbm` | 20269.086822371864 | 7979.401055254064 | 0.048229397922831006 | 2795919 | 0.046992062359460345 | 0.8499999463503771 |
| `visit` | `validation` | `x_learner_lgbm` | 20615.579845381897 | 8326.002314366388 | 0.04832016428145979 | 2795918 | 0.04699207916684252 | 0.8500002503649964 |
| `conversion` | `test` | `response_model_lgbm` | 2479.8978020030017 | 1112.3968327334942 | 0.003055854637911946 | 2795919 | 0.0029167511648227292 | 0.8499999463503771 |
| `conversion` | `validation` | `response_model_lgbm` | 2459.5958965030513 | 1089.267080874527 | 0.003044072847825784 | 2795918 | 0.002916394543759867 | 0.8500002503649964 |
| `conversion` | `test` | `t_learner_lgbm` | 2197.2245817599314 | 829.7236124904239 | 0.0029289187833688567 | 2795919 | 0.0029167511648227292 | 0.8499999463503771 |
| `conversion` | `validation` | `t_learner_lgbm` | 2197.564609283904 | 827.2357936553794 | 0.0029193819910747986 | 2795918 | 0.002916394543759867 | 0.8500002503649964 |
| `conversion` | `test` | `x_learner_lgbm` | 2167.529060701153 | 800.0280914316454 | 0.003036218141283741 | 2795919 | 0.0029167511648227292 | 0.8499999463503771 |
| `conversion` | `validation` | `x_learner_lgbm` | 2099.079047371504 | 728.7502317429798 | 0.003010270278089808 | 2795918 | 0.002916394543759867 | 0.8500002503649964 |

### Prediction Artifact Inventory

Parquet contents were not read or modified in Phase 0. This inventory reports the existing file names and filesystem sizes.

| File | Size |
| --- | ---: |
| `criteo_conversion_response_lgbm_run01_predictions.parquet` | 27447676 bytes |
| `criteo_conversion_t_learner_lgbm_run01_predictions.parquet` | 29550555 bytes |
| `criteo_conversion_x_learner_lgbm_run01_predictions.parquet` | 15934302 bytes |
| `criteo_visit_response_lgbm_run02_predictions.parquet` | 30169843 bytes |
| `criteo_visit_t_learner_lgbm_run01_predictions.parquet` | 31221921 bytes |
| `criteo_visit_x_learner_lgbm_run01_predictions.parquet` | 23744952 bytes |

### Figure Artifact Inventory

| File | Size |
| --- | ---: |
| `criteo_conversion_response_lgbm_run02_qini_curve.png` | 91276 bytes |
| `criteo_conversion_response_lgbm_run02_uplift_curve.png` | 53989 bytes |
| `criteo_conversion_response_lgbm_vs_t_learner_lgbm_run01_qini_curve.png` | 126010 bytes |
| `criteo_conversion_response_lgbm_vs_t_learner_lgbm_run01_uplift_curve.png` | 62068 bytes |
| `criteo_conversion_response_lgbm_vs_t_learner_lgbm_vs_x_learner_lgbm_run01_qini_curve.png` | 182557 bytes |
| `criteo_conversion_response_lgbm_vs_t_learner_lgbm_vs_x_learner_lgbm_run01_uplift_curve.png` | 70544 bytes |
| `criteo_visit_response_lgbm_run03_qini_curve.png` | 92798 bytes |
| `criteo_visit_response_lgbm_run03_uplift_curve.png` | 52328 bytes |
| `criteo_visit_response_lgbm_vs_t_learner_lgbm_run01_qini_curve.png` | 126661 bytes |
| `criteo_visit_response_lgbm_vs_t_learner_lgbm_run01_uplift_curve.png` | 64632 bytes |
| `criteo_visit_response_lgbm_vs_t_learner_lgbm_vs_x_learner_lgbm_run01_qini_curve.png` | 171214 bytes |
| `criteo_visit_response_lgbm_vs_t_learner_lgbm_vs_x_learner_lgbm_run01_uplift_curve.png` | 74498 bytes |

## 4. Correct Interpretation

The response model is not an uplift model. It estimates response probability from features, not individual treatment effect.

Response targeting is a non-causal baseline policy. It can rank customers by predicted likelihood of outcome, but that ranking does not by itself estimate who is incrementally affected by treatment.

T-Learner and X-Learner are uplift learners. Their purpose is to estimate treatment-effect heterogeneity by using treatment and control information.

Qini and AUUC can be used to evaluate response targeting as a policy, but they should not be interpreted as proof that the response model estimates treatment effect.

The current `exploratory_v1` results should be interpreted as targeting-policy benchmark results, not final causal model-selection results.

## 5. Current Issues Found

1. Existing response model appears to be trained on the full training split, including treatment and control. This estimates `P(Y=1 | X)` rather than the campaign response baseline `P(Y=1 | X, treatment=1)`.
2. X-Learner needs implementation verification because its curves/results appear less stable than expected.
3. Current evaluation lacks explicit Top-K budget tables at 1%, 5%, 10%, 20%, and 30%.
4. Current evaluation lacks bootstrap confidence intervals.
5. Current evaluation lacks paired policy contrasts against treated response baseline.
6. Current model-selection rule is not yet formalized.
7. The current test results have already been inspected, so they should not be treated as a strict locked audit result.

## 6. Next Plan

1. Rename/label current results as `exploratory_v1`.
2. Update response modeling:
   - keep or rename the current full-data response model as `pooled_response_baseline`;
   - add or change the main campaign baseline to `treated_response_baseline`;
   - train treated response only on rows where `treatment == 1`;
   - use treated-only validation rows for early stopping;
   - still predict scores for the full validation/test population.
3. Audit/debug X-Learner:
   - verify treated/control outcome models;
   - verify pseudo-effect formulas;
   - verify final effect regressors;
   - verify propensity/weighting logic;
   - verify score sign, row alignment, NaN/inf values, score distribution, and stable sorting.
4. Retrain:
   - treated response baseline;
   - T-Learner;
   - X-Learner.
5. Add Top-K evaluation:
   - 1%, 5%, 10%, 20%, 30%.
6. Add bootstrap confidence intervals:
   - policy value;
   - incremental outcome;
   - paired contrast versus treated response baseline.
7. Add model-selection gate:
   - primary outcome: `visit`;
   - primary budget: `5%`;
   - baseline: `treated_response_baseline`;
   - select an uplift champion only if its lower 95% CI for paired contrast is above zero.
8. Run final test protocol:
   - select model on validation;
   - use locked test only once for final reporting.

## 7. Current Recommendation

The next implementation work should prioritize:

1. response baseline correction;
2. X-Learner verification;
3. Top-K evaluation;
4. bootstrap CI;
5. model-selection gate.

Do not add S-Learner or AIPW in Phase 0. They can be considered later as optional extensions after the core verification/evaluation pipeline is stable.
