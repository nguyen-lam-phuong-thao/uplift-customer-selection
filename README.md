# Customer Selection with Uplift Modeling

This repository provides a reusable framework for customer selection using uplift modeling. The framework is designed to support reliable model development, evaluation, and deployment while remaining reusable across datasets, uplift models, and business scenarios.

The primary objective is to produce **reliable, reproducible, and trustworthy customer-selection decisions**. Every experiment must follow the same standardized workflow for data preparation, model training, validation evaluation, statistical champion selection, and locked-test evaluation.

Core principles:

- Reuse the framework across different datasets.
- Integrate different uplift models through a common artifact and evaluation contract.
- Evaluate every candidate with the same standardized protocol.
- Prevent data leakage, inconsistent evaluation, and test-set reuse.
- Produce artifacts that can be traced, audited, and reproduced.
- Keep the current implementation simple; do not add full MLOps infrastructure before it is required.

The framework currently supports the Criteo Uplift Prediction Dataset. It is designed to be reused for RetailHero by replacing dataset-specific preparation and feature engineering while keeping the downstream training, evaluation, selection, and locked-test contracts unchanged.

## Current Scope

The current implementation provides the core workflow for reliable uplift-model evaluation and customer selection.

Implemented:

- Load and validate Criteo data.
- Prepare reproducible decision datasets with a canonical `row_id`.
- Create deterministic train, validation, and test splits.
- Train a treated-response baseline, T-Learner, and X-Learner.
- Log trained models and model components to MLflow.
- Save validation-only prediction artifacts and model-provenance sidecars.
- Align model predictions by `row_id` before comparison.
- Evaluate targeting policies with Top-K metrics, policy value, Qini, and AUUC.
- Estimate uncertainty with paired bootstrap resampling.
- Select a champion through a deterministic statistical gate.
- Reload the selected policy from MLflow and evaluate only that policy on the locked test split.

Framework hardening still to be completed:

- Restrict the Selection Gate to deployable model candidates only.
- Lock the champion to the exact MLflow `run_id` and model URI or component URIs used to create its validation predictions.
- Make experiment manifests immutable and prevent models from unrelated experiment runs from being mixed.
- Enforce a one-way transition from completed selection to locked-test evaluation for the same experiment.
- Validate all artifact identities and provenance before final evaluation.

These hardening items are part of the current framework work. Automated retraining, model registry workflows, online deployment, monitoring, and champion/challenger operations remain future MLOps extensions.

## Standardized Workflow

```text
Dataset-specific preparation
        ↓
Standard decision dataset
        ↓
Train deployable candidate models
        ↓
Log exact model runs to MLflow
        ↓
Write validation predictions and provenance
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
Evaluate champion once on locked test
        ↓
Final evaluation artifact
```

The flow is one-way:

```text
Train → Validation → Selection Gate → Champion → Locked Test
```

The test split must not influence training, early stopping, validation evaluation, bootstrap selection, or champion selection. After the champion is locked for an experiment, locked-test results must not be used to select a different champion for that same experiment.

The detailed normative workflow is documented in [`docs/framework_workflow.md`](docs/framework_workflow.md). Artifact contracts are documented in [`contracts/README.md`](contracts/README.md).

## Champion-Selection Rule

Champion selection uses a deterministic statistical gate on **validation-only paired bootstrap contrasts**.

For the configured primary outcome, split, budget, metric, and baseline:

1. Consider deployable model candidates only.
2. Compare every candidate with the configured baseline using paired bootstrap samples.
3. A candidate passes only when its confidence-interval lower bound for the metric delta is greater than zero.
4. Among passing candidates, select the candidate with the largest mean delta.
5. Break an exact mean-delta tie deterministically by policy name.
6. If no candidate passes, keep the baseline as champion.

Bootstrap is therefore allowed to affect champion selection. The Selection Gate does not perform resampling itself; it consumes the standardized paired-contrast artifact produced by validation evaluation.

`random_targeting` is an evaluation benchmark, not a deployable champion candidate.

## Data Contract

Prepared decision datasets must contain:

```text
row_id
feature columns
treatment
one selected outcome
split
```

For Criteo, the decision datasets are:

```text
data/processed/criteo/criteo_decision_visit.parquet
data/processed/criteo/criteo_decision_conversion.parquet
```

They contain:

```text
row_id
f0 ... f11
treatment
visit or conversion
split
```

`exposure` is excluded because it is post-treatment and would create leakage risk.

Current split policy:

```text
Train / validation / test = 60% / 20% / 20%
```

The split is stratified by treatment and the selected outcome. `row_id` is created once at the preparation boundary and must remain unchanged through training, prediction writing, evaluation, bootstrap, selection, and locked-test reporting.

## Models

The current deployable candidate models are:

- `treated_response_lgbm`: a treated-group response-ranking baseline.
- `t_learner_lgbm`: separate treatment and control outcome models.
- `x_learner_lgbm`: outcome models, imputed treatment effects, and treatment-effect regressors.

Each training pipeline:

- Reads the configured processed decision dataset;
- Fits on `train`;
- Uses `validation` for early stopping;
- Writes predictions for `validation` only;
- Logs the trained model or model components to MLflow;
- Writes a provenance sidecar linking the prediction artifact to the MLflow run.

The existing source filenames still include `criteo` and will be renamed only after the framework contracts and implementation are stable. Until then, current module names remain the supported entry points.

## Current Pipeline Entry Points

Train the candidate models:

```bash
python -m uplift_modeling.pipelines.train_criteo_response_model \
  --config configs/modeling/criteo_response_lgbm.yaml \
  --outcome visit

python -m uplift_modeling.pipelines.train_criteo_t_learner \
  --config configs/modeling/t_learner.yaml \
  --outcome visit

python -m uplift_modeling.pipelines.train_criteo_x_learner \
  --config configs/modeling/x_learner.yaml \
  --outcome visit
```

Create the current experiment manifest:

```bash
python -m uplift_modeling.pipelines.create_experiment_manifest \
  --config configs/modeling/criteo_response_lgbm.yaml \
  --outcome visit \
  --experiment-id <experiment-id> \
  --output artifacts/metrics/<experiment-id>_manifest.json
```

Run validation evaluation, bootstrap, and the Selection Gate:

```bash
python -m uplift_modeling.pipelines.evaluate_criteo_predictions \
  --config configs/modeling/criteo_response_lgbm.yaml \
  --manifest artifacts/metrics/<experiment-id>_manifest.json \
  --outcome visit \
  --n-bootstrap 100
```

Run locked-test scoring and final evaluation:

```bash
python -m uplift_modeling.pipelines.evaluate_locked_test \
  --config configs/modeling/criteo_response_lgbm.yaml \
  --manifest artifacts/metrics/<experiment-id>_manifest.json \
  --selection-artifact artifacts/metrics/<selection-artifact>.json \
  --outcome visit \
  --n-bootstrap 100
```

The explicit `--experiment-id` and `--output` arguments are recommended because the current manifest creator otherwise uses a mutable `latest`-style default. The code-hardening stage will replace that temporary behavior with an immutable experiment contract.

## Evaluation Outputs

Validation evaluation produces standardized artifacts for:

- Top-K targeting metrics at configured budget fractions;
- Policy value;
- Incremental outcome;
- Qini and AUUC;
- Paired bootstrap mean, standard deviation, and confidence intervals;
- Paired contrasts against the configured baseline;
- Deterministic Selection Gate output.

Raw bootstrap samples are not required as persisted deliverables. Reproducibility depends on preserving the source artifacts, bootstrap count, random seed, evaluation settings, and exact code/config identity.

Locked-test evaluation:

- Loads only the locked champion;
- Generates one test prediction artifact for that champion;
- Computes final uplift and Top-K metrics from the same champion predictions;
- Computes final bootstrap uncertainty without changing the champion;
- Writes one final evaluation artifact.

## Artifact Trust Model

The intended artifact chain is:

```text
Decision dataset
    ↓
Validation prediction + model provenance
    ↓
Experiment manifest
    ↓
Validation evaluation + paired bootstrap contrasts
    ↓
Champion artifact
    ↓
Locked-test prediction
    ↓
Final evaluation artifact
```

Every downstream artifact must identify its upstream source artifacts. The champion must ultimately be locked by exact model identity, not by policy name alone.

## Reuse for RetailHero

RetailHero must provide a dataset-specific adapter that creates the same standardized decision-dataset contract. It requires separate feature engineering and separate model training; a model trained on Criteo must never be used for RetailHero inference.

Expected reuse boundary:

```text
Criteo preparation ─────┐
                       ├─→ Standard decision dataset
RetailHero preparation ┘              ↓
                              Shared training contract
                                      ↓
                              Shared evaluation contract
                                      ↓
                              Shared selection contract
                                      ↓
                              Shared locked-test contract
```

If RetailHero requires copying or rewriting the downstream evaluation and selection pipeline, the framework is not yet reusable enough.

## Repository Structure

```text
.
├── app/
├── artifacts/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── configs/
│   ├── data.yaml
│   └── modeling/
├── contracts/
├── data/
│   ├── processed/
│   └── raw/
├── docs/
├── notebooks/
├── sql/
├── src/uplift_modeling/
│   ├── artifacts/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   ├── tracking/
│   └── utils/
├── tests/
├── requirements.txt
└── README.md
```

Generated datasets, prediction artifacts, metrics, figures, MLflow databases, local environments, and temporary files must not be committed.

## Documentation Authority

For framework behavior, use the following order of authority:

1. [`docs/framework_workflow.md`](docs/framework_workflow.md) — normative workflow and stage invariants.
2. [`contracts/README.md`](contracts/README.md) — artifact and identity contracts.
3. `configs/` — supported configuration values and defaults.
4. This README — project overview and operating guide.
5. Historical weekly notes — context only, not current implementation requirements.

Coding rules are documented in [`docs/code_rules.md`](docs/code_rules.md).

## Long-Term Vision

Future work may extend the framework into a complete MLOps lifecycle with automated retraining, model registry workflows, deployment, monitoring, champion/challenger management, and continuous improvement. Those capabilities are outside the current core scope and must not be introduced prematurely at the cost of a simple, auditable evaluation pipeline.
