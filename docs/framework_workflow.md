# Framework Workflow and Invariants

This document describes the current workflow, boundaries, and main invariants of the **Customer Selection with Uplift Modeling** framework.

The README provides a short project overview. This document explains the framework behavior in more detail.

---

## 1. Framework Goal

The framework provides a consistent downstream workflow for customer-selection experiments using uplift modeling.

It is responsible for:

- standardizing prepared datasets into one decision-dataset contract;
- training supported candidate models;
- generating aligned validation predictions;
- evaluating policies under the same conditions;
- selecting the best uplift candidate on validation;
- checking whether that uplift candidate can reliably replace the Response baseline;
- preserving experiment and model provenance;
- evaluating the frozen uplift champion and Response baseline on locked test data.

The framework starts **after dataset-specific Data Science preparation has been completed**.

---

## 2. Framework Boundary

### Outside the framework

The following work is dataset-specific and must be completed before the dataset enters the framework:

```text
raw data loading
data cleaning
EDA / hypothesis checking
feature engineering
categorical encoding
treatment/control definition
outcome definition
leakage decisions
```

### Inside the framework

```text
prepared dataset
        ↓
validate dataset contract
        ↓
standardize to outcome-specific decision dataset
        ↓
add / preserve row_id
        ↓
create / validate train-validation-test split
        ↓
train Response + uplift candidates
        ↓
validation predictions + model provenance
        ↓
experiment manifest
        ↓
validation evaluation
        ↓
Top-K uplift champion selection
        ↓
paired-bootstrap replacement gate vs Response baseline
        ↓
selection artifact
        ↓
locked-test evaluation of uplift champion + Response baseline
```

The framework is dataset-generic. A new dataset does not need a new training or evaluation pipeline as long as its prepared data can be described by the dataset contract.

---

## 3. Input Dataset

To use a new dataset, the user provides:

```text
1. one prepared Parquet file
2. one dataset YAML config
```

The prepared table must already be modeling-ready.

At minimum it contains:

```text
feature columns
treatment column
outcome column(s)
```

Example:

| age | tenure | past_purchase_count | treatment | conversion |
|---:|---:|---:|---:|---:|
| 32 | 12 | 5 | 1 | 1 |
| 45 | 30 | 10 | 0 | 0 |
| 27 | 4 | 2 | 1 | 0 |

Here:

- `age`, `tenure`, `past_purchase_count` are features;
- `treatment = 1` is treatment and `treatment = 0` is control;
- `conversion` is the observed outcome.

The user is responsible for deciding which columns are valid pre-treatment features and for removing leakage before creating this file.

### Dataset config

The YAML config tells the framework how to interpret the prepared file.

```yaml
dataset:
  name: my_campaign
  prepared_path: data/interim/my_campaign/prepared.parquet

schema:
  treatment_column: treatment
  split_column: split

  feature_columns:
    - age
    - tenure
    - past_purchase_count

  outcome_columns:
    - conversion

split:
  assign_if_missing: true
  train_size: 0.6
  validation_size: 0.2
  test_size: 0.2
  random_state: 42

outputs:
  processed_paths:
    conversion: data/processed/my_campaign/decision_conversion.parquet
```

`row_id` and `split` do not need to exist in the prepared dataset when the framework is configured to create them.

---

## 4. Dataset Examples

### Criteo

The prepared Criteo table contains:

```text
f0 ... f11
treatment
visit
conversion
```

The framework creates separate decision datasets for the configured outcomes.

For `visit`:

```text
row_id
f0 ... f11
treatment
visit
split
```

For `conversion`:

```text
row_id
f0 ... f11
treatment
conversion
split
```

`exposure` is excluded during external data preparation because it is a post-treatment variable.

### Hillstrom

The original Hillstrom experiment contains:

```text
No E-Mail
Mens E-Mail
Womens E-Mail
```

Binary treatment definition is handled outside the framework by creating two prepared experiments:

```text
hillstrom_mens
No E-Mail   = 0
Mens E-Mail = 1
```

```text
hillstrom_womens
No E-Mail     = 0
Womens E-Mail = 1
```

After categorical encoding and feature preparation, both datasets enter the same framework contract:

```text
customer features
treatment
visit
conversion
```

No Hillstrom-specific training or evaluation pipeline is required.

---

## 5. Standardization

Standardization is the first modeling step inside the framework.

It:

1. loads the prepared Parquet;
2. validates the columns declared in the dataset config;
3. keeps only the configured features, treatment, and outcome;
4. adds or preserves the configured `row_id`;
5. creates or validates the split;
6. writes an outcome-specific decision dataset.

Decision-dataset contract:

```text
row_id
feature columns
treatment
outcome
split
```

### `row_id`

`row_id` is used for technical alignment, not as a model feature.

It must remain:

- non-null;
- unique inside the decision dataset;
- unchanged across prediction and evaluation artifacts.

All model comparisons are aligned by `row_id`, not by row order.

---

## 6. Split

Default split:

```text
train       60%
validation  20%
test        20%
```

When the framework creates the split, it is deterministic and uses the configured random seed.

For each outcome, stratification uses:

```text
(treatment, outcome)
```

to keep treatment/control and outcome distributions reasonably stable.

Split roles:

```text
train
→ model fitting

validation
→ early stopping
→ prediction
→ policy evaluation
→ uplift champion selection
→ paired-bootstrap replacement gate

test
→ used only after validation decisions are frozen
```

The test split must not influence uplift selection or the replacement gate.

---

## 7. Main Experiment Workflow

Main entry point:

```text
src/uplift_modeling/pipelines/run_experiment.py
```

Conceptual flow:

```text
load dataset config
        ↓
load modeling config
        ↓
standardize prepared dataset
        ↓
resolve configured candidates
        ↓
train each candidate
        ↓
collect validation prediction artifacts
        ↓
create experiment manifest
        ↓
validation evaluation
        ↓
Top-K uplift champion selection
        ↓
paired bootstrap
        ↓
replacement gate vs Response baseline
        ↓
selection artifact
```

`run_experiment` stops after the validation decision has been saved.

Locked-test evaluation is run separately.

---

## 8. Candidate Configuration and Supported Models

The modeling config defines candidate models through:

```text
name
kind
parameter overrides
```

Current supported policies:

| Policy | Role |
|---|---|
| `treated_response_lgbm` | Response baseline |
| `t_learner_lgbm` | Uplift candidate |
| `x_learner_lgbm` | Uplift candidate |

The experiment runner resolves each configured candidate and dispatches it to the correct training implementation.

Unsupported model kinds must fail explicitly.

`treated_response_lgbm` is the baseline and does **not** participate in uplift-champion selection.

---

## 9. Validation Prediction Contract

Each trained candidate produces one validation prediction artifact:

```text
row_id
treatment
outcome
split
score
model_name
```

Required invariants:

- prediction artifacts contain validation rows only;
- each `row_id` appears once per policy;
- all policies cover the same validation observations;
- treatment, outcome, and split match for the same `row_id`;
- row order is not used for alignment;
- `model_name` matches the policy.

Each prediction artifact is linked to model provenance so the exact trained model can be traced and reloaded later.

---

## 10. Experiment Manifest

The experiment manifest freezes the exact candidate artifacts used in one validation experiment.

It identifies:

```text
experiment_id
dataset_name
outcome
dataset config
modeling config
prediction artifact by policy
model provenance by policy
```

The manifest must not discover artifacts by scanning for the newest file.

Its purpose is to make sure that:

```text
model evaluated on validation
        ==
model referenced by downstream selection / locked test
```

---

## 11. Validation Evaluation

Validation evaluation loads the candidates from the manifest and compares them on the same rows.

Main outputs include:

```text
Top-K policy value
incremental outcome
Qini
AUUC
Qini curve
uplift curve
```

Classification metrics may be used as diagnostics, but they do not decide which uplift policy should replace the Response baseline.

---

## 12. Top-K Budgets

Default evaluated budgets are:

```text
1%
5%
10%
20%
30%
```

One configured **primary budget** is used for the validation decision.

For example, different experiments may use different primary budgets while still reporting the same default Top-K range.

The primary metric is currently `policy_value`.

---

## 13. Evaluation Benchmark

`random_targeting` is generated internally as a benchmark.

It may appear in evaluation output, but it:

- is not a trained model;
- has no deployable model provenance;
- does not participate in uplift selection;
- cannot become the recommended deployment policy.

---

## 14. Uplift Champion Selection

Uplift selection uses validation Top-K results at the configured primary operating point.

Only uplift candidates participate.

Current rule:

1. keep the configured uplift candidates;
2. read their primary validation Top-K metric;
3. select the candidate with the largest metric value;
4. break an exact tie deterministically by policy name;
5. save the result as `uplift_champion_policy`.

Conceptually:

```text
T-Learner ─┐
           ├─ compare primary Top-K metric ─→ uplift champion
X-Learner ─┘
```

The Response baseline is excluded from this step.

---

## 15. Paired Bootstrap and Replacement Gate

After the uplift champion has been fixed, paired bootstrap is used to compare that uplift champion with the Response baseline.

For each bootstrap iteration:

1. sample validation row positions with replacement;
2. apply the same sampled positions to every compared policy;
3. calculate the configured policy metric;
4. calculate the uplift-champion-minus-baseline delta.

The replacement rule is:

```text
ci_lower > 0
→ replacement_gate_passed = true

ci_lower <= 0
→ replacement_gate_passed = false
```

The framework therefore keeps three separate decisions:

```text
uplift_champion_policy
replacement_gate_passed
recommended_deployment_policy
```

Recommendation rule:

```text
if replacement_gate_passed:
    recommended_deployment_policy = uplift_champion_policy
else:
    recommended_deployment_policy = treated_response_lgbm
```

This means an uplift model can be the best uplift candidate without being good enough to replace the Response baseline.

---

## 16. Selection Artifact

The Selection Gate artifact records the frozen validation decision.

Important fields include:

```text
experiment_id
dataset_name
uplift_champion_policy
baseline_policy
uplift_selection_method
uplift candidate Top-K rows
replacement_gate_method
replacement_gate_passed
baseline contrast row
recommended_deployment_policy
uplift champion model provenance
selection settings
source manifest
bootstrap artifact
```

This artifact is the boundary between validation model selection and final locked-test reporting.

---

## 17. Model Provenance and Lock

The selected uplift champion must identify one exact trained model instance, not only a policy name.

Relevant identity includes:

```text
dataset
outcome
policy
MLflow run ID
model URI / component URIs
source prediction artifact
source experiment manifest
```

This prevents a later run with the same policy name from silently replacing the model that was evaluated during validation.

---

## 18. Locked-Test Evaluation

Locked-test entry point:

```text
src/uplift_modeling/pipelines/evaluate_locked_test.py
```

Locked test evaluates the two policies frozen by validation:

```text
uplift_champion_policy
+
treated_response_lgbm baseline
```

The pipeline:

1. loads the Selection Gate artifact;
2. validates it against the experiment manifest;
3. reloads the required model artifacts;
4. reads only `split == test`;
5. scores the uplift champion and Response baseline;
6. aligns both policies by `row_id`;
7. reports test Top-K and uplift metrics;
8. reports bootstrap uncertainty and paired contrast on test;
9. saves the locked-test evaluation artifact.

Locked test does **not**:

- select a new uplift champion;
- rerun the replacement gate as a deployment decision;
- replace `recommended_deployment_policy`;
- feed test results back into validation selection.

Its role is final confirmation and reporting on unseen data.

---

## 19. Reuse with Another Dataset

A new dataset may have a completely different raw schema and preparation process.

The reusable boundary is:

```text
dataset-specific work
raw → EDA → features → treatment/outcome → prepared Parquet

                         ↓

shared framework
prepared Parquet
→ standardization
→ row_id / split
→ candidate training
→ prediction contract
→ manifest
→ validation evaluation
→ uplift selection
→ replacement gate
→ locked test
```

If a new dataset requires copied training, evaluation, or selection code only because its source columns are different, the framework boundary is not generic enough.

---

## 20. Supported Scope

The framework intentionally does not own:

- raw data processing;
- EDA;
- feature engineering;
- encoding decisions;
- automatic leakage detection;
- treatment/outcome construction;
- multi-treatment modeling;
- arbitrary model families;
- arbitrary baseline policies;
- automatic hyperparameter search;
- production deployment or automatic retraining.

These can be extended later, but they are outside the current framework scope.

---

## 21. Artifact Hygiene

Generated runtime artifacts should not be committed with source code:

```text
artifacts/predictions/
artifacts/metrics/
artifacts/figures/
mlruns/
cache/
temporary files
```

If example artifacts are needed for documentation or tests, they should live in dedicated fixture/example locations instead of active runtime output folders.
