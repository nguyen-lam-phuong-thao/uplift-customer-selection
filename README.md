Customer Selection with Uplift Modeling

Project Overview

This project provides a reusable framework for customer selection with uplift modeling.

The framework standardizes the workflow for:

Training candidate models.

Generating validation predictions.

Evaluating all policies under the same conditions.

Comparing candidates with paired bootstrap.

Selecting and locking a champion model.

Evaluating the selected champion on a separate locked test.

Preserving the artifacts required to reproduce each experiment.

The framework is reusable across datasets, but fitted models are not. Each new dataset still requires its own data preparation, feature engineering, and model training.

Purpose

Uplift-modeling experiments often rely on separate training scripts, manually selected prediction files, and inconsistent evaluation steps. This creates risks such as:

Predictions being aligned with the wrong observations.

Models being compared under different settings.

Validation and test data being used inconsistently.

Artifacts from unrelated experiments being mixed.

A selected model not being traceable to its exact training run.

This framework provides one controlled workflow for training, comparing, selecting, and evaluating uplift models.

Main Workflow

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

Locked-test evaluation is performed separately:

Locked champion
        ↓
Reload exact model from MLflow
        ↓
Score test split
        ↓
Final evaluation artifact

The test split must not influence training, validation evaluation, bootstrap comparison, or champion selection.

Installation

python -m pip install -r requirements.txt
python -m pip install -e .

Run Validation Experiment

The main experiment runner trains the supported candidates, creates a manifest from the exact prediction artifacts produced in that run, evaluates the candidates on validation, runs paired bootstrap, and selects a champion.

python -m uplift_modeling.pipelines.run_experiment \
  --experiment-id criteo-visit-001 \
  --outcome visit

The runner performs:

Train treated-response baseline
        ↓
Train T-Learner
        ↓
Train X-Learner
        ↓
Collect exact validation prediction paths
        ↓
Create experiment manifest
        ↓
Run validation evaluation
        ↓
Run paired bootstrap
        ↓
Run Selection Gate

The runner stops after champion selection. It does not run the locked test automatically.

Run Locked Test

After the champion has been selected and reviewed, run the final evaluation separately:

python -m uplift_modeling.pipelines.evaluate_locked_test \
  --config configs/modeling/criteo_response_lgbm.yaml \
  --manifest artifacts/metrics/<manifest>.json \
  --selection-artifact artifacts/metrics/<selection>.json \
  --outcome visit

The locked-test pipeline:

Validates the selection artifact against the experiment manifest.

Loads the exact selected model from MLflow.

Scores only the test split for the champion policy.

Produces final evaluation metrics without changing the champion.

Current Models

The Criteo implementation currently includes:

treated_response_lgbm

t_learner_lgbm

x_learner_lgbm

random_targeting is generated internally as an evaluation benchmark. It is not a deployable model and cannot become the champion.

Champion Selection

Champion selection uses validation-only paired-bootstrap comparisons.

For the configured outcome, validation split, budget, metric, and baseline:

Compare each deployable candidate with the baseline.

A candidate passes when ci_lower > 0.

If multiple candidates pass, select the one with the largest mean_delta.

If no candidate passes, keep the baseline.

The Selection Gate only consumes previously calculated bootstrap results. It does not perform prediction, metric calculation, or resampling itself.

Repository Structure

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

Generated datasets, predictions, metrics, figures, local MLflow files, environments, and temporary files must not be committed.

Detailed Documentation

docs/framework_workflow.md: detailed end-to-end workflow, evaluation stages, bootstrap comparison, champion selection, and locked-test lifecycle.

contracts/README.md: dataset, prediction, provenance, manifest, selection, and artifact contracts.