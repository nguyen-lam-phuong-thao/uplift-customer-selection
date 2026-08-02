# Framework Workflow and Invariants

This document is the normative technical description of the current Customer Selection with Uplift Modeling framework.

It defines:

- The intended end-to-end experiment flow;
- The responsibility of each pipeline stage;
- The allowed use of train, validation, and test data;
- The deterministic statistical champion-selection rule;
- The artifact identities that must be preserved;
- The known implementation gaps that the next code changes must close.

The current source filenames still contain `criteo`. Renaming them is intentionally deferred until the framework behavior and contracts are stable.

## 1. Project Goal

The framework must produce customer-targeting decisions that are:

- Reliable;
- Reproducible;
- Trustworthy;
- Traceable;
- Auditable;
- Reusable across datasets and supported uplift models.

Criteo is the current development dataset. RetailHero will provide a separate preparation and feature-engineering layer while reusing the downstream framework contracts.

## 2. End-to-End Flow

```text
Raw dataset
    ↓
Dataset-specific validation and preparation
    ↓
Standard decision dataset
    ↓
Train deployable candidate models
    ↓
Log exact trained model runs to MLflow
    ↓
Write validation predictions and model provenance
    ↓
Create one experiment manifest
    ↓
Standardized validation evaluation
    ↓
Paired bootstrap uncertainty estimation
    ↓
Deterministic statistical Selection Gate
    ↓
Lock exact champion identity
    ↓
Reload exact champion from MLflow
    ↓
Generate champion-only locked-test predictions
    ↓
Final locked-test evaluation
```

The experiment lifecycle is one-way:

```text
TRAINED
    ↓
VALIDATION_EVALUATED
    ↓
CHAMPION_SELECTED
    ↓
CHAMPION_LOCKED
    ↓
FINAL_EVALUATED
```

The current implementation does not yet persist these lifecycle states explicitly. The next code changes must enforce the same one-way behavior through immutable artifacts and identity validation without introducing an unnecessary workflow service or database state machine.

## 3. Stage 1 — Dataset-Specific Preparation

### Responsibility

Convert raw dataset-specific inputs into the shared decision-dataset contract.

### Criteo output

```text
data/processed/criteo/criteo_decision_visit.parquet
data/processed/criteo/criteo_decision_conversion.parquet
```

### Required columns

```text
row_id
feature columns
treatment
one selected outcome
split
```

For Criteo:

```text
row_id
f0 ... f11
treatment
visit or conversion
split
```

### Invariants

- `row_id` is created once at the preparation boundary.
- `row_id` is non-null and unique within the prepared dataset.
- `row_id` does not depend on dataframe index after preparation.
- `row_id` must be preserved unchanged by every downstream stage.
- `exposure` is excluded because it is post-treatment.
- Split assignment is deterministic for the same input and configuration.
- Train, validation, and test rows are disjoint.

### Split policy

```text
train      = 60%
validation = 20%
test       = 20%
```

The split is stratified by treatment and the selected outcome.

## 4. Stage 2 — Candidate Model Training

### Current entry points

```text
src/uplift_modeling/pipelines/train_criteo_response_model.py
src/uplift_modeling/pipelines/train_criteo_t_learner.py
src/uplift_modeling/pipelines/train_criteo_x_learner.py
```

These filenames remain unchanged during the current refactor.

### Deployable candidates

```text
treated_response_lgbm
t_learner_lgbm
x_learner_lgbm
```

### Required behavior

Each training pipeline must:

1. Read one standardized decision dataset;
2. Fit using `split == train`;
3. Use `split == validation` for early stopping when required;
4. Never read test labels or test features for fitting, early stopping, tuning, or candidate comparison;
5. Log the exact trained model or model components to MLflow;
6. Write predictions for validation only;
7. Write model provenance linking the validation prediction artifact to the exact MLflow run.

### Model semantics

#### Treated-response baseline

Fit on treated training rows and rank all validation rows by predicted treated response.

```text
score(x) = P(Y = 1 | T = 1, X = x)
```

It is a deployable targeting baseline but not an individual-treatment-effect estimator.

#### T-Learner

```text
mu1(x) = P(Y = 1 | T = 1, X = x)
mu0(x) = P(Y = 1 | T = 0, X = x)
score(x) = mu1(x) - mu0(x)
```

#### X-Learner

Fit treatment and control outcome models, impute treatment effects, fit treatment-effect regressors, and combine their scores using the configured treatment-rate weight.

## 5. Stage 3 — Validation Prediction and Provenance Artifacts

### Prediction contract

Each deployable candidate writes one validation prediction artifact containing:

```text
row_id
treatment
outcome
split
score
model_name
```

### Required invariants

- `split` contains validation only.
- Each `row_id` occurs exactly once.
- The treatment and outcome labels match the prepared decision dataset for the same `row_id`.
- The score comes from one exact trained model instance.
- Artifact order is irrelevant; evaluation aligns by `row_id`.

### Model provenance

Each validation prediction artifact has a provenance sidecar containing the exact MLflow identity required to reproduce the score.

Examples:

```text
Response model:
- mlflow_run_id
- model_uri

T-Learner:
- mlflow_run_id
- treatment_model_uri
- control_model_uri

X-Learner:
- mlflow_run_id
- tau1_model_uri
- tau0_model_uri
- constant_treatment_rate_weight
```

The prediction artifact and provenance sidecar together identify one candidate model instance.

## 6. Stage 4 — Experiment Manifest

### Current entry point

```text
src/uplift_modeling/pipelines/create_experiment_manifest.py
```

### Responsibility

Group the exact validation prediction and model-provenance artifacts that belong to one candidate-comparison experiment.

### Required manifest identity

A manifest must identify:

```text
artifact_type
experiment_id
dataset_name
outcome
configuration identity
prediction artifacts by policy
model artifacts by policy
```

### Required invariants

- All entries belong to the same logical experiment.
- All entries use the same dataset, outcome, prepared dataset identity, and evaluation contract.
- Every deployable policy entry links validation predictions to matching model provenance.
- The manifest is immutable once used for validation evaluation or champion selection.
- A mutable `latest` name may exist only as a convenience pointer before selection; it must never be the identity stored in a champion or final-evaluation artifact.

### Current implementation gap

The current manifest creator discovers the highest run number for each model independently and defaults to a `latest`-style experiment ID and output filename. That discovery mechanism is part of the current pipeline, not a second evaluation pipeline, but it does not yet prove that all selected model artifacts belong to the same experiment batch.

The code-hardening step must replace or constrain this behavior so that a manifest cannot silently mix unrelated model versions and cannot change after selection.

## 7. Stage 5 — Standardized Validation Evaluation

### Current entry point

```text
src/uplift_modeling/pipelines/evaluate_criteo_predictions.py
```

### Responsibility

Evaluate every candidate under one protocol using validation data only.

### Data isolation

- Standard evaluation accepts validation prediction artifacts only.
- An artifact containing test rows is a hard error.
- Test is not a configurable standard-evaluation split.
- All candidate frames are aligned by `row_id` before metrics are calculated.

### Evaluation policies

The evaluation set may contain:

```text
random_targeting
treated_response_lgbm
t_learner_lgbm
x_learner_lgbm
```

`random_targeting` is generated deterministically from the validation labels and configured random seed. It is a benchmark only.

### Deployable candidate set

The Selection Gate may consider only policies backed by valid model provenance and loadable MLflow artifacts.

```text
treated_response_lgbm
t_learner_lgbm
x_learner_lgbm
```

Evaluation membership and champion eligibility are different concepts.

### Metrics

Validation evaluation may produce:

- Top-K policy value;
- Top-K uplift rate;
- incremental outcome;
- Qini;
- AUUC;
- Qini and uplift curves.

The configured primary selection metric is currently `policy_value` at a configured Top-K budget.

## 8. Stage 6 — Paired Bootstrap Evaluation

### Responsibility

Estimate uncertainty for policy metrics using validation-only paired resampling.

### Pairing rule

For each bootstrap iteration, the same sampled row positions are applied to every evaluated policy. This preserves paired comparisons between policies.

### Persisted outputs

For each configured policy, split, budget, and metric, persist a reproducible summary:

```text
mean
standard deviation
confidence interval
number of bootstrap resamples
random seed
```

For each candidate-versus-baseline comparison, persist:

```text
policy
baseline_policy
outcome
split
budget_fraction
metric
mean_delta
standard deviation of delta
confidence-interval lower bound
confidence-interval upper bound
number of bootstrap resamples
random seed
```

Raw bootstrap sample arrays are not required as persisted deliverables.

### Allowed use

Bootstrap confidence intervals are allowed to affect champion selection. Bootstrap on locked test is allowed only for final uncertainty reporting and must never change the champion.

## 9. Stage 7 — Deterministic Statistical Selection Gate

### Current implementation

```text
src/uplift_modeling/evaluation/selection_gate.py
```

### Responsibility

Consume precomputed validation paired-bootstrap contrasts and choose one champion deterministically.

The Selection Gate must not:

- read raw train data;
- read raw validation data;
- read test data;
- calculate model predictions;
- calculate uplift metrics;
- perform bootstrap resampling;
- evaluate the selected model on test.

### Configured selection dimensions

```text
primary outcome
primary split
primary budget fraction
primary metric
baseline policy
```

### Official selection rule

1. Filter to the configured primary outcome, split, budget, metric, and baseline.
2. Restrict candidates to deployable policies with valid provenance.
3. A candidate passes when `ci_lower > 0` for its paired metric delta versus the baseline.
4. If multiple candidates pass, select the largest `mean_delta`.
5. If mean deltas are exactly equal, use policy name as the deterministic tie-break.
6. If no candidate passes, select the baseline.

### Current implementation gap

The current Selection Gate correctly implements filtering by selection dimensions, `ci_lower > 0`, largest `mean_delta`, deterministic tie-breaking, and baseline fallback. It does not yet receive or enforce an explicit deployable-candidate set, while `random_targeting` is included in bootstrap policy frames.

The next code change must exclude non-deployable benchmarks before they can become champion candidates.

## 10. Stage 8 — Champion Lock

### Required output

Selection must produce an immutable champion artifact that identifies the exact model instance selected on validation.

Minimum identity:

```text
artifact_type
experiment_id
dataset_name
outcome
champion_policy
model_kind
mlflow_run_id
model_uri or component model URIs
source validation prediction artifact
source model-provenance artifact
source experiment manifest
source bootstrap paired-contrast artifact
selection settings
selection method
```

### Core invariant

```text
Exact model instance evaluated on validation
==
Exact model instance loaded for locked test
```

Matching only `champion_policy` is insufficient because multiple MLflow runs can share the same policy name.

### Current implementation gap

The current selection artifact locks `champion_policy` and selection evidence, while locked-test scoring obtains model identity from a separately supplied manifest. The next code change must make the champion self-contained or strictly bind the selection artifact to one immutable manifest entry.

## 11. Stage 9 — Locked-Test Scoring and Final Evaluation

### Current entry point

```text
src/uplift_modeling/pipelines/evaluate_locked_test.py
```

### Responsibility

Load the exact locked champion from MLflow, score the test split once, and create the final evaluation artifact.

### Required behavior

- Accept one locked champion only.
- Load the exact `run_id` and URI recorded at selection time.
- Read only `split == test` from the prepared decision dataset.
- Score only the champion.
- Generate one champion-only test prediction artifact.
- Calculate all final metrics from the same champion prediction artifact.
- Calculate final bootstrap uncertainty without changing selection.
- Write an immutable final-evaluation artifact.

### Final metrics

- Policy Value;
- Top-K incremental outcome;
- Qini;
- AUUC;
- bootstrap mean, standard deviation, and confidence interval;
- final evaluation metadata and provenance.

### Forbidden behavior

- Running the Selection Gate inside locked-test evaluation.
- Evaluating every candidate on test.
- Tuning after test results are visible.
- Replacing the champion because another model would have performed better on test.

## 12. One-Way Finalization

After a champion is selected for an experiment, the allowed transition is:

```text
champion artifact
    ↓
locked-test evaluation
```

The framework must reject attempts to:

- replace the champion for the same finalized experiment;
- combine a selection artifact with a different manifest;
- load a different MLflow run under the same policy name;
- overwrite final evaluation with different model, test dataset, or selection settings.

A technical rerun may be allowed only when all relevant identities are unchanged and the operation is reproducible and idempotent.

## 13. Configuration Contract

The current model YAML files define:

```text
project experiment name
dataset and processed paths
train and validation split names
model parameters
artifact output directories
validation prediction split
selection dimensions
MLflow prediction logging behavior
locked-test split
```

Selection configuration describes the statistical gate. It does not imply that the Selection Gate performs bootstrap itself.

The CLI `--outcome` currently determines the evaluated outcome. It must agree with `selection.primary_outcome`; the next code-hardening step should validate this agreement explicitly.

## 14. Current Code Alignment Summary

### Already aligned

- Validation-only training prediction outputs.
- Standard evaluator rejects artifacts containing test rows.
- `row_id` alignment across candidate predictions.
- Paired bootstrap comparisons.
- Deterministic `ci_lower > 0` gate.
- Largest-mean-delta selection and deterministic tie-break.
- Baseline fallback.
- Locked-test split fixed to `test`.
- Locked-test pipeline scores only the selected policy.
- Models are reloaded from MLflow for locked-test scoring.

### To be aligned in the next code phase

- Exclude non-deployable benchmark policies from selection candidates.
- Bind champion to exact MLflow run and URI.
- Make experiment manifests immutable and experiment-consistent.
- Reject mismatched selection, manifest, model-provenance, and dataset identities.
- Prevent backward selection after finalization.
- Reject training configurations that use test for training or early stopping.

## 15. Reuse Boundary for RetailHero

RetailHero may change:

- raw-data loading;
- schema validation;
- customer-level aggregation;
- feature engineering;
- treatment and outcome construction.

RetailHero must not require a separate implementation of:

- prediction artifact schema;
- experiment manifest schema;
- validation evaluation;
- paired bootstrap;
- Selection Gate;
- champion identity;
- locked-test evaluation;
- final evaluation reporting.

That boundary is the central test of framework reusability.
