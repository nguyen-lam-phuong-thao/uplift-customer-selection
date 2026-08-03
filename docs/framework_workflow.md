# Framework Workflow and Invariants

This document defines the current workflow and required invariants of the Customer Selection with Uplift Modeling framework.

The current source filenames still contain `criteo`. Renaming is deferred until the framework has been verified with another dataset.

## 1. Project Goal

The framework provides a consistent and traceable process for:

* Training uplift-model candidates.
* Comparing candidates on validation data.
* Selecting one champion.
* Locking the exact selected model.
* Evaluating the champion separately on test.
* Reusing the downstream workflow with another dataset.

Criteo is the current framework-validation dataset. RetailHero will use separate preparation and feature-engineering logic while reusing the downstream contracts.

---

## 2. End-to-End Workflow

### Validation experiment

```text
Dataset-specific preparation
        ↓
Standard decision dataset
        ↓
Train deployable candidates
        ↓
Log exact model runs to MLflow
        ↓
Write validation predictions and provenance
        ↓
Create exact experiment manifest
        ↓
Validation evaluation
        ↓
Paired bootstrap
        ↓
Selection Gate
        ↓
Champion-selection artifact
```

### Final evaluation

```text
Champion-selection artifact
        ↓
Validate manifest and champion identity
        ↓
Reload exact champion from MLflow
        ↓
Generate champion-only test predictions
        ↓
Final locked-test evaluation
```

Validation and locked test are separate pipeline stages.

---

## 3. Main Entry Points

### Full validation experiment

```text
src/uplift_modeling/pipelines/run_experiment.py
```

Example:

```bash
python -m uplift_modeling.pipelines.run_experiment \
  --experiment-id criteo-visit-001 \
  --outcome visit
```

The runner:

1. Validates shared candidate configuration sections.
2. Trains the current candidate models.
3. Receives the exact prediction path from each training pipeline.
4. Creates the experiment manifest from those exact paths.
5. Runs validation evaluation.
6. Runs paired bootstrap.
7. Runs the Selection Gate.
8. Stops before locked test.

### Locked-test evaluation

```text
src/uplift_modeling/pipelines/evaluate_locked_test.py
```

This stage is run explicitly after champion selection.

### Debug entry points

Individual training, manifest-creation, and validation-evaluation pipelines remain available for debugging.

---

## 4. Dataset Preparation

Each dataset-specific preparation pipeline must produce:

```text
row_id
feature columns
treatment
one selected outcome
split
```

### Required invariants

* `row_id` is created once during preparation.
* `row_id` is non-null and unique.
* `row_id` remains unchanged downstream.
* Train, validation, and test rows are disjoint.
* Split creation is deterministic.
* Features must not contain post-treatment information.
* Shared framework code must not depend on raw dataset structure.

### Criteo

Current Criteo decision datasets contain:

```text
row_id
f0 ... f11
treatment
visit or conversion
split
```

`exposure` is excluded because it occurs after treatment.

Current split:

```text
train      60%
validation 20%
test       20%
```

The split is stratified by treatment and the selected outcome.

---

## 5. Candidate Training

Current training entry points:

```text
train_criteo_response_model.py
train_criteo_t_learner.py
train_criteo_x_learner.py
```

Current deployable policies:

```text
treated_response_lgbm
t_learner_lgbm
x_learner_lgbm
```

Each training pipeline must:

1. Read one standard decision dataset.
2. Fit on `split == train`.
3. Use validation only where required for fitting or early stopping.
4. Never read test data during training or candidate comparison.
5. Log the trained model or model components to MLflow.
6. Write validation predictions only.
7. Write model provenance.
8. Return:

```text
(policy_name, prediction_path)
```

The returned path allows the experiment runner to pass the exact newly created artifact into the manifest.

---

## 6. Validation Prediction Contract

Each candidate writes one prediction artifact containing:

```text
row_id
treatment
outcome
split
score
model_name
```

Required behavior:

* `split` contains validation only.
* Every `row_id` occurs once.
* Candidate artifacts contain the same validation observations.
* Treatment and outcome values match by `row_id`.
* Artifact row order is irrelevant.
* `model_name` matches the manifest policy name.

Each prediction artifact has a provenance sidecar identifying:

* Dataset.
* Outcome.
* Policy.
* Prediction artifact.
* MLflow run ID.
* Model URI or component model URIs.
* Model-specific reload settings when required.

---

## 7. Experiment Manifest

Entry point:

```text
src/uplift_modeling/pipelines/create_experiment_manifest.py
```

The manifest groups the exact candidate artifacts used in one comparison.

It records:

```text
artifact_type
experiment_id
dataset_name
outcome
prediction artifacts by policy
model artifacts by policy
```

Required behavior:

* Prediction paths are supplied explicitly.
* The manifest does not scan an output directory.
* The manifest does not select artifacts by run number or recency.
* Every prediction artifact must exist.
* Every prediction artifact must contain the required columns.
* Every prediction artifact must contain validation rows only.
* `model_name` must match the declared policy.
* Every prediction artifact must have matching provenance.
* An existing manifest cannot be silently overwritten.

The experiment runner creates this mapping automatically from training outputs. Manual `POLICY=PATH` input is retained for debugging.

---

## 8. Validation Evaluation

Entry point:

```text
src/uplift_modeling/pipelines/evaluate_criteo_predictions.py
```

The evaluator:

* Loads prediction paths from one manifest.
* Uses all deployable policies declared in that manifest.
* Aligns candidate predictions by `row_id`.
* Rejects test prediction artifacts.
* Applies one evaluation configuration to all candidates.
* Produces validation metrics and curves.
* Returns the Selection Gate artifact path.

Possible outputs include:

* Top-K policy value.
* Incremental outcome.
* Qini.
* AUUC.
* Qini curve.
* Uplift curve.

The evaluator does not call locked-test evaluation.

---

## 9. Evaluation Benchmark

`random_targeting` is generated internally using the configured random seed.

It may be included in evaluation and bootstrap outputs, but:

* It is not stored as a deployable candidate in the manifest.
* It has no model provenance.
* It is not passed to the Selection Gate.
* It cannot become champion.

Evaluation membership and champion eligibility are separate concepts.

---

## 10. Paired Bootstrap

Bootstrap uses validation data only.

For each bootstrap iteration, the same sampled row positions are applied to every evaluated policy. This preserves paired candidate-versus-baseline comparisons.

Persisted summaries may include:

```text
policy
baseline_policy
outcome
split
budget_fraction
metric
mean_delta
standard deviation
confidence-interval bounds
number of resamples
random seed
```

Raw bootstrap arrays are not required as persisted artifacts.

Bootstrap on locked test may report final uncertainty, but it must not change the selected champion.

---

## 11. Selection Gate

Implementation:

```text
src/uplift_modeling/evaluation/selection_gate.py
```

The Selection Gate consumes validation paired-bootstrap contrasts.

It must not:

* Read train or test data.
* Generate predictions.
* Calculate model metrics.
* Perform bootstrap.
* Run locked-test evaluation.

### Selection rule

For the configured outcome, split, budget, metric, and baseline:

1. Restrict rows to deployable manifest policies.
2. A candidate passes when `ci_lower > 0`.
3. Select the passing candidate with the largest `mean_delta`.
4. Break an exact tie by policy name.
5. Select the baseline when no candidate passes.

The output records the selected policy, selection evidence, source manifest, and exact model provenance needed for final reload.

---

## 12. Champion Lock

The selected champion must identify one exact trained model instance.

Required identity includes:

```text
experiment_id
dataset_name
outcome
champion_policy
mlflow_run_id
model URI or component URIs
source prediction artifact
source model provenance
source experiment manifest
source bootstrap contrast
selection settings
```

Core invariant:

```text
Model evaluated on validation
==
Model reloaded for locked test
```

Matching only the policy name is insufficient because multiple MLflow runs may use the same policy name.

---

## 13. Locked-Test Evaluation

Entry point:

```text
src/uplift_modeling/pipelines/evaluate_locked_test.py
```

Required behavior:

* Accept one champion-selection artifact.
* Validate its relationship with the supplied manifest.
* Load the exact MLflow run and URI recorded for the champion.
* Read the standard decision dataset.
* Select `split == test`.
* Score the champion only.
* Write one champion-only test prediction artifact.
* Calculate final metrics from that artifact.
* Never rerun the Selection Gate.
* Never replace the champion based on test performance.

Locked-test results are final reporting outputs, not model-selection inputs.

---

## 14. Configuration

Candidate configs describe:

```text
project experiment name
dataset name and processed paths
training and validation split names
model parameters
artifact output directories
selection dimensions
locked-test split
```

The CLI `--outcome` determines the outcome for the current run.

Candidate configs used in one experiment must share the same:

```text
project
data
training
outputs
selection
```

Model-specific sections may differ.

MLflow tracking is supplied through the user environment. No personal tracking server is hard-coded in the framework.

---

## 15. RetailHero Reuse Boundary

RetailHero may change:

* Raw-data loading.
* Source schema validation.
* Customer-level aggregation.
* Feature engineering.
* Treatment construction.
* Outcome construction.
* Dataset configuration.

RetailHero must reuse:

* Standard decision-dataset contract.
* Prediction artifact contract.
* Model provenance contract.
* Experiment manifest.
* Validation evaluation.
* Paired bootstrap.
* Selection Gate.
* Champion identity.
* Locked-test evaluation.

If RetailHero requires a copied evaluation or selection pipeline, the framework boundary is not yet sufficiently reusable.
