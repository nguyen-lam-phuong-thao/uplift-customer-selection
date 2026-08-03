# Customer Selection with Uplift Modeling

## Project Overview

This project provides a reusable framework for customer selection with uplift modeling.

The framework standardizes:

* Candidate-model training.
* Validation prediction generation.
* Consistent model evaluation.
* Paired-bootstrap comparison.
* Deterministic champion selection.
* Champion locking.
* Separate locked-test evaluation.
* Experiment artifact tracking.

New datasets require their own data preparation and feature engineering. New models require a small training integration, but the shared evaluation, bootstrap, selection, and locked-test logic should remain unchanged.

The framework is reusable; fitted models are not. A model trained on Criteo must not be used directly on RetailHero or another unrelated dataset.

---

## Purpose

Uplift-modeling experiments often use separate training scripts and manually selected prediction files. This can lead to:

* Predictions aligned with the wrong observations.
* Models evaluated under different settings.
* Validation and test data used inconsistently.
* Artifacts from unrelated runs being mixed.
* A selected model that cannot be traced to its exact training run.

This framework provides one controlled workflow for training, comparing, selecting, and evaluating uplift models.

---

## Main Workflow

```text
Dataset-specific preparation
        ↓
Standard decision dataset
        ↓
Train candidate models
        ↓
Write validation predictions and model provenance
        ↓
Create an exact experiment manifest
        ↓
Validation evaluation
        ↓
Paired bootstrap comparison
        ↓
Selection Gate
        ↓
Lock champion
```

Locked-test evaluation is a separate final step:

```text
Locked champion
        ↓
Reload exact model from MLflow
        ↓
Score test split
        ↓
Final evaluation artifact
```

The test split must not influence training, validation evaluation, bootstrap comparison, or champion selection.

---

## Run a Validation Experiment

The main entry point runs the three currently supported candidate pipelines, creates the manifest from the exact prediction artifacts just produced, evaluates them, and selects a champion:

```bash
python -m uplift_modeling.pipelines.run_experiment \
  --experiment-id criteo-visit-001 \
  --outcome visit
```

The runner performs:

```text
Train treated-response baseline
        ↓
Train T-Learner
        ↓
Train X-Learner
        ↓
Collect exact prediction paths
        ↓
Create immutable experiment manifest
        ↓
Run validation evaluation
        ↓
Run paired bootstrap
        ↓
Run Selection Gate
```

The runner stops after champion selection. It does not run locked test automatically.

Individual training, manifest, and evaluation commands remain available for debugging.

---

## Run Locked-Test Evaluation

After reviewing the selected champion, run the final evaluation separately:

```bash
python -m uplift_modeling.pipelines.evaluate_locked_test \
  --config configs/modeling/criteo_response_lgbm.yaml \
  --manifest artifacts/metrics/<manifest>.json \
  --selection-artifact artifacts/metrics/<selection>.json \
  --outcome visit
```

The locked-test pipeline:

* Validates the selection and manifest identities.
* Loads the exact selected model from MLflow.
* Scores test rows for the champion only.
* Produces final metrics without changing the champion.

---

## Integration Contracts

### Dataset contract

Each dataset-preparation pipeline must produce:

```text
row_id
feature columns
treatment
outcome
split
```

The dataset-specific layer is responsible for:

* Reading and validating raw data.
* Feature engineering.
* Defining treatment and outcome.
* Preventing post-treatment leakage.
* Creating deterministic train, validation, and test splits.

### Model contract

Each candidate training pipeline must:

* Train on the training split.
* Use validation only where required for fitting.
* Never use test data during model development.
* Produce one validation uplift score per `row_id`.
* Log the trained model or model components to MLflow.
* Write model provenance.
* Return its policy name and prediction artifact path.

### Artifact contract

The framework uses the following artifact chain:

```text
Decision dataset
        ↓
Validation prediction + model provenance
        ↓
Experiment manifest
        ↓
Validation metrics + bootstrap contrasts
        ↓
Champion-selection artifact
        ↓
Locked-test prediction
        ↓
Final evaluation artifact
```

The manifest records exact artifact paths. It does not search for the newest files in an output directory.

---

## Current Models

The Criteo implementation currently includes:

* `treated_response_lgbm`
* `t_learner_lgbm`
* `x_learner_lgbm`

`random_targeting` is generated internally as an evaluation benchmark. It is not a deployable model and is not passed to the Selection Gate.

---

## Champion-Selection Rule

Champion selection uses validation-only paired-bootstrap contrasts.

For the configured outcome, split, budget, metric, and baseline:

1. Consider deployable policies listed in the experiment manifest.
2. Compare each candidate with the baseline.
3. A candidate passes when `ci_lower > 0`.
4. Select the passing candidate with the largest `mean_delta`.
5. Break an exact tie by policy name.
6. Keep the baseline when no candidate passes.

The Selection Gate consumes existing bootstrap results. It does not perform resampling itself.

---

## Deliverables

The project produces:

1. Standard decision datasets with stable `row_id`.
2. Candidate-model training pipelines.
3. Validation prediction and model-provenance artifacts.
4. Exact experiment manifests.
5. Standard validation metrics.
6. Paired-bootstrap comparison artifacts.
7. Champion-selection artifacts.
8. Champion-only locked-test results.
9. Automated tests and technical documentation.
10. RetailHero customer features, model results, and customer-selection analysis after dataset integration.
11. A dashboard for business users that presents campaign recommendations and experiment results without requiring access to the underlying machine-learning pipeline.


---

## Current Status

The framework currently supports the full Criteo workflow:

* Data preparation and validation.
* Stable `row_id`.
* Deterministic data splitting.
* Three candidate-model pipelines.
* MLflow model logging.
* Validation-only prediction artifacts.
* Exact manifest creation.
* Prediction alignment by `row_id`.
* Standardized uplift evaluation.
* Paired bootstrap.
* Deterministic Selection Gate.
* Separate locked-test evaluation.
* One-command validation experiment runner.
* Automated tests for the main contracts and pipeline stages.

MLflow tracking uses the environment configured by the user. The repository does not hard-code a personal tracking server.

Current source filenames still contain `criteo`. Renaming is deferred until the framework has been tested with the second dataset.

---

## Reuse for RetailHero

RetailHero requires a separate preparation layer:

```text
RetailHero raw data
        ↓
RetailHero validation
        ↓
Customer-level feature engineering
        ↓
Treatment and outcome construction
        ↓
Standard decision dataset
```

After that boundary, RetailHero should reuse:

* Prediction artifact contracts.
* Model provenance contracts.
* Experiment manifest creation.
* Validation evaluation.
* Paired bootstrap.
* Selection Gate.
* Champion locking.
* Locked-test evaluation.

The main reuse requirement is:

> Integrating RetailHero must not require copying or rewriting the shared evaluation and selection pipeline.

---

## Repository Structure

```text
.
├── artifacts/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── configs/
│   └── modeling/
├── contracts/
├── data/
│   ├── processed/
│   └── raw/
├── docs/
├── notebooks/
├── src/uplift_modeling/
│   ├── artifacts/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── pipelines/
│   ├── tracking/
│   └── utils/
├── tests/
├── requirements.txt
└── README.md
```

Generated datasets, predictions, metrics, figures, local MLflow files, environments, and temporary files must not be committed.

---

## Next Step

The next development stage is to integrate RetailHero:

1. Understand and validate the raw tables.
2. Define the observation unit, treatment, and outcome.
3. Define feature and outcome time windows.
4. Build customer-level features without post-treatment leakage.
5. Create the standard decision dataset.
6. Train the supported candidates on RetailHero.
7. Verify that the shared downstream workflow runs without dataset-specific evaluation code.

Model expansion, tuning, deployment, monitoring, and automated retraining are outside the current core scope.
