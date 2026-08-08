# Framework Workflow and Invariants

This document describes the workflow, boundaries, and main invariants of the **Customer Selection with Uplift Modeling** framework.

The README provides a short project overview. This document explains the framework behavior in more detail.

---

## 1. Framework Goal

The framework is designed for customer-selection experiments using uplift modeling.

Its goal is to provide a consistent workflow for:

- Standardizing prepared datasets into one decision-dataset contract.
- Training supported candidate models.
- Generating validation predictions under one artifact contract.
- Comparing policies under the same conditions.
- Checking stability with paired bootstrap.
- Selecting a champion on validation data.
- Locking the exact selected model instance.
- Evaluating the champion separately on test data.
- Preserving enough provenance to trace results back to the correct experiment and model run.

The framework focuses on the **downstream modeling workflow**. It does not replace the data-science work required before modeling.

---

## 2. Framework Boundary

The framework starts from a **prepared dataset**.

### Outside the framework

The following steps are dataset-specific:

```text
raw data loading
data cleaning
EDA
feature engineering
encoding decisions
treatment/control definition
outcome definition
leakage decisions
```

These decisions depend on business context and the source dataset, so they are not part of the framework core.

### Inside the framework

```text
prepared dataset
        ↓
standardization
        ↓
internal row_id
        ↓
deterministic split
        ↓
candidate training
        ↓
validation predictions + provenance
        ↓
experiment manifest
        ↓
validation evaluation
        ↓
paired bootstrap
        ↓
selection gate
        ↓
champion artifact
        ↓
locked-test evaluation
```

The framework is dataset-generic: any dataset can be used if it can be mapped to the required contract.

The framework does not try to support every possible baseline, budget, or model family. These parts have a defined supported scope and can be extended in code when needed.

---

## 3. Prepared Dataset Contract

A prepared dataset must provide:

```text
feature columns
treatment column
outcome column or outcome columns
```

The dataset configuration describes:

```text
dataset name
prepared dataset path
feature columns
treatment column
outcome columns
split settings
standardized output path
```

The framework does not decide:

- which features should be used;
- what treatment means;
- which outcome is appropriate;
- which features contain leakage;
- which business transformations should be applied.

Those decisions must be completed before the dataset enters the framework.

---

## 4. Dataset Example

Example prepared dataset:

```text
age
tenure
past_purchase_count
average_order_value
treatment
conversion
```

Example Criteo dataset:

```text
f0
f1
...
f11
treatment
visit
conversion
```

One experiment selects one outcome to create a decision dataset.

For example, with `visit`:

```text
row_id
f0
f1
...
f11
treatment
visit -> outcome
split
```

The same prepared dataset can also be standardized for `conversion` as the selected outcome.

---

## 5. Standardization

Standardization is the first step inside the framework.

It is responsible for:

1. Loading the prepared dataset.
2. Validating columns declared in the dataset config.
3. Selecting the outcome for the experiment.
4. Creating an internal `row_id`.
5. Creating train/validation/test splits.
6. Writing the standardized decision dataset.

Output contract:

```text
row_id
feature columns
treatment
outcome
split
```

### `row_id`

`row_id` is an internal technical identifier created by the framework.

It keeps the same observation consistently identifiable across:

```text
dataset
model predictions
evaluation
bootstrap
manifest
locked test
```

The framework does not depend on a source customer ID for alignment.

After creation, `row_id` must be:

- non-null;
- unique;
- unchanged throughout the downstream workflow.

---

## 6. Split

The framework creates splits because train/validation/test separation is part of modeling and evaluation infrastructure.

Default split:

```text
train      60%
validation 20%
test       20%
```

Split creation must:

- be deterministic;
- use the configured random seed;
- create disjoint train, validation, and test groups;
- keep test independent from candidate selection.

When stratification is used, the framework stratifies by treatment and the selected outcome to preserve the main group distribution across splits.

### Split usage

```text
train
→ model fitting

validation
→ validation prediction
→ evaluation
→ paired bootstrap
→ champion selection

test
→ used only after champion lock
```

Test data must not influence champion selection.

---

## 7. Main Experiment Workflow

Main entry point:

```text
src/uplift_modeling/pipelines/run_experiment.py
```

Workflow:

```text
Load dataset config
        ↓
Load modeling config
        ↓
Standardize prepared dataset
        ↓
Resolve configured candidates
        ↓
Train each candidate
        ↓
Collect exact validation prediction paths
        ↓
Create experiment manifest
        ↓
Validation evaluation
        ↓
Paired bootstrap
        ↓
Selection Gate
        ↓
Champion-selection artifact
```

`run_experiment` stops after champion selection.

Locked-test evaluation is not run automatically so final test evaluation remains separate from model selection.

---

## 8. Candidate Configuration and Dispatch

The modeling config uses a candidate list.

Each candidate defines:

```text
name
kind
model parameters / overrides
```

The experiment runner resolves each `ModelCandidateConfig` and passes that candidate to the matching training function.

Conceptual dispatch:

```text
for candidate in candidates:

    treated_response
        → train response candidate

    t_learner
        → train T-Learner candidate

    x_learner
        → train X-Learner candidate
```

A training pipeline should not search again for “the only candidate of this model kind.”

The runner should know exactly which candidate is being trained and pass it directly to the corresponding training function.

Unsupported model kinds must fail explicitly instead of silently falling back.

---

## 9. Supported Models

The framework currently supports:

```text
treated_response_lgbm
t_learner_lgbm
x_learner_lgbm
```

The models share:

- the standardized dataset contract;
- train/validation/test splits;
- experiment-level settings where appropriate;
- the validation prediction contract;
- the evaluation workflow.

Algorithm-specific behavior remains inside each model implementation.

Additional model families can be added later through explicit implementation and candidate dispatch.

---

## 10. Validation Prediction Contract

Each deployable candidate creates one validation prediction artifact:

```text
row_id
treatment
outcome
split
score
model_name
```

Invariants:

- only validation rows are included;
- each `row_id` appears once;
- all candidates contain the same validation observations;
- treatment and outcome values match by `row_id`;
- row order is not used to align predictions;
- `model_name` matches the candidate/policy.

Each prediction artifact also has model provenance so the model that produced it can be traced.

Provenance may include:

```text
dataset
outcome
policy
prediction path
MLflow run ID
model URI / component URIs
reload information
```

---

## 11. Experiment Manifest

The experiment manifest stores the exact mapping between one experiment and the candidate artifacts used in that comparison.

It must identify:

```text
experiment_id
dataset_name
outcome
prediction artifact by policy
model provenance by policy
```

Invariants:

- prediction paths come directly from training outputs;
- the manifest does not scan folders to guess which file to use;
- artifacts are not selected by recency;
- prediction artifacts must exist and satisfy the contract;
- prediction artifacts must contain validation rows only;
- policy names must match model provenance;
- an existing manifest must not be silently overwritten.

The purpose is to prevent a case where validation evaluates predictions from one model while locked test reloads a different model.

---

## 12. Validation Evaluation

The validation evaluator:

1. Loads candidate prediction artifacts from the manifest.
2. Aligns observations by `row_id`.
3. Applies the same evaluation settings to every policy.
4. Evaluates ranking under the supported top-k budgets.
5. Generates the required evaluation artifacts.
6. Produces the data needed for paired bootstrap.

Possible outputs include:

```text
Top-K policy value
Incremental outcome
Qini
AUUC
Qini curve
Uplift curve
```

Classification metrics, when calculated, are used for diagnostics or reporting and do not replace the uplift-policy selection rule.

---

## 13. Evaluation Budgets

Default supported top-k budgets:

```text
1%
5%
10%
20%
30%
```

The framework does not automatically support arbitrary budgets outside the configured list.

If a new budget is required, evaluation, bootstrap, and selection settings must be updated consistently.

---

## 14. Evaluation Benchmark

`random_targeting` is generated internally as a benchmark.

It:

- may appear in evaluation output;
- may appear in bootstrap reporting;
- is not a trained model;
- does not have deployable model provenance;
- cannot become champion.

A benchmark and a deployable candidate are different concepts.

---

## 15. Paired Bootstrap

Paired bootstrap uses validation data only.

For each bootstrap iteration:

1. Sample validation row positions with replacement.
2. Apply the same sampled positions to every policy.
3. Calculate the policy metric on the same bootstrap sample.
4. Calculate the delta between each candidate and the baseline.

Using the same sample for every policy makes the comparison paired rather than independent.

Persisted summaries may include:

```text
policy
baseline_policy
budget_fraction
metric
mean_delta
ci_lower
ci_upper
standard deviation
number of resamples
random seed
```

Locked-test data is not used by bootstrap to select the champion.

---

## 16. Baseline Policy

The default workflow currently uses one baseline:

```text
treated_response_lgbm
```

Bootstrap contrasts and the Selection Gate use this baseline.

The framework does not try to support arbitrary baselines in the default implementation.

If another baseline is required, bootstrap, selection, and model-eligibility logic must be extended together rather than changing only one config field.

---

## 17. Selection Gate

The Selection Gate consumes validation paired-bootstrap results only.

It does not:

- train models;
- generate predictions;
- calculate bootstrap samples;
- read the test split;
- run locked-test evaluation.

Current selection rule:

1. Compare deployable candidates with `treated_response_lgbm`.
2. A candidate passes when `ci_lower > 0`.
3. If multiple candidates pass, select the one with the largest `mean_delta`.
4. If no candidate passes, keep the baseline.
5. Exact ties may be broken deterministically by policy name.

The Selection Gate output must preserve enough evidence to explain why the champion was selected.

---

## 18. Champion Lock

The champion artifact must identify **one exact trained model instance**, not only a model name.

Champion identity should be traceable through:

```text
experiment_id
dataset_name
outcome
champion_policy
MLflow run ID
model URI / component URIs
source prediction artifact
source model provenance
source manifest
selection evidence
```

Core invariant:

```text
model evaluated on validation
        ==
model reloaded for locked test
```

Policy name alone is not enough because multiple training runs can use the same policy name.

---

## 19. Locked-Test Evaluation

Locked-test entry point:

```text
src/uplift_modeling/pipelines/evaluate_locked_test.py
```

The locked-test pipeline:

1. Loads the champion-selection artifact.
2. Validates the champion against the experiment manifest.
3. Reloads the exact selected model from the recorded MLflow run/model URI.
4. Loads the standardized decision dataset.
5. Selects `split == test`.
6. Scores the champion only.
7. Writes a champion-only test prediction artifact.
8. Calculates final test metrics.

Locked test does not:

- rerun the Selection Gate;
- compare candidates again to replace the champion;
- use test results as model-selection input.

Locked test is the final evaluation and reporting stage.

---

## 20. Reuse with Another Dataset

A new dataset may have a completely different:

```text
raw source
schema
cleaning process
EDA
feature engineering
treatment definition
outcome definition
prepared-data format
```

As long as the prepared dataset can be described by the dataset config and mapped to:

```text
features
treatment
outcome
```

the downstream workflow can reuse:

```text
standardization
row_id
split
training interface
prediction contract
manifest
evaluation
paired bootstrap
selection
champion lock
locked test
```

If a new dataset requires a copied evaluation or selection pipeline only because its source schema is different, the framework boundary is not generic enough.

---

## 21. Supported Scope

The framework intentionally keeps a defined scope.

It does not automatically provide:

- raw data processing;
- EDA;
- feature engineering;
- leakage detection;
- treatment/outcome construction;
- every possible model family;
- every possible baseline policy;
- every possible top-k budget;
- automatic hyperparameter search.

These areas can be extended, but they are not part of the default framework behavior.

---

## 22. Artifact Hygiene

Generated runtime artifacts should not be committed with the source code:

```text
artifacts/predictions/
artifacts/metrics/
artifacts/figures/
mlruns/
cache/
temporary files
```

If example artifacts are needed for testing or documentation, they should be stored in a dedicated fixture/example location instead of active output directories.
