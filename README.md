# Customer Selection with Uplift Modeling

## Project Overview

This project provides a reusable framework for customer selection with uplift modeling.

The framework standardizes the entire workflow, including:

- Training uplift-model candidates.
- Evaluating candidates on validation data.
- Comparing models under identical evaluation settings.
- Selecting a champion with a deterministic statistical rule.
- Locking the selected champion before final test evaluation.
- Preserving artifacts required to reproduce each experiment.

In short, the framework provides a consistent pipeline for training, evaluating, selecting, and validating uplift models. New datasets and models can be integrated through small, explicit contracts instead of modifying the core workflow.

The framework is reusable, but trained models are not. Every dataset still requires its own data preparation, feature engineering, and model training.

---

# Purpose

Many uplift-modeling projects evolve into collections of independent training scripts, temporary prediction files, and manually selected experiment results. As the project grows, this makes experiments increasingly difficult to trust and reproduce.

Common problems include:

- predictions matched to the wrong observations;
- models evaluated under different settings;
- inconsistent use of validation and test data;
- stale artifacts mixed into new experiments;
- champions that cannot be traced back to the exact model run;
- results that cannot be reproduced reliably.

This framework separates dataset-specific work from the shared evaluation pipeline. New datasets and models can be added without rewriting evaluation, bootstrap comparison, or champion selection.

---

# Expected Users

The framework is intended for data scientists who already have:

- a treatment variable;
- an outcome variable;
- customer-level features;
- train, validation, and test splits;
- one or more uplift-model candidates.

A typical workflow is:

1. Prepare a dataset using the standard decision-dataset format.
2. Choose an existing model or integrate a new one through the model contract.
3. Configure the experiment.
4. Run the shared training, evaluation, selection, and locked-test pipeline.

The core evaluation logic should remain unchanged regardless of the dataset being used.

---

# Workflow

```text
Dataset preparation
        ↓
Standard decision dataset
        ↓
Train candidate models
        ↓
Generate validation predictions
        ↓
Create experiment manifest
        ↓
Validation evaluation
        ↓
Paired bootstrap comparison
        ↓
Selection Gate
        ↓
Lock champion
        ↓
Reload champion
        ↓
Locked-test evaluation
        ↓
Final evaluation artifact
```

The workflow is strictly one-way:

```text
Train
  ↓
Validation evaluation
  ↓
Champion selection
  ↓
Champion locking
  ↓
Locked-test evaluation
```

The locked test is used only once, after the champion has been selected from validation.

---

# Integration Contracts

The framework is organized around small, explicit integration contracts.

## Dataset Contract

Each dataset-preparation pipeline must produce a decision dataset containing:

```text
row_id
feature columns
treatment
outcome
split
```

The `row_id` must remain stable throughout the entire workflow, including prediction generation, evaluation, bootstrap comparison, champion selection, and locked-test reporting.

Dataset-specific responsibilities include:

- reading raw data;
- cleaning and validation;
- feature engineering;
- defining treatment and outcome columns;
- creating train, validation, and test splits.

---

## Model Contract

Each candidate model must:

- train using the training split;
- use validation only when required (for example, early stopping);
- produce one uplift score per requested row;
- preserve the corresponding `row_id`;
- record the information required to reload the trained model.

Existing models can be selected through configuration. Adding a new model should only require a small integration layer and must not require changes to evaluation or champion selection.

---

## Artifact Contract

Every stage records enough metadata to trace both its inputs and outputs.

The artifact chain is:

```text
Decision dataset
        ↓
Validation predictions
        ↓
Experiment manifest
        ↓
Validation metrics
        ↓
Bootstrap contrasts
        ↓
Champion-selection artifact
        ↓
Locked-test predictions
        ↓
Final evaluation artifact
```

Each artifact must be reproducible and linked to the experiment that created it.

---

# Current Model Candidates

The current Criteo implementation includes:

- `treated_response_lgbm`
- `t_learner_lgbm`
- `x_learner_lgbm`

These models are used to validate the framework itself.

`random_targeting` may be evaluated as a statistical benchmark, but it is **not** a deployable policy and must not be considered by the Selection Gate.

---

# Champion Selection

Champion selection is performed **only on the validation split**.

For the configured outcome, budget, metric, and baseline:

1. Evaluate deployable candidate models only.
2. Compare each candidate with the baseline using paired bootstrap contrasts.
3. A candidate passes when the lower confidence bound of its metric delta is greater than zero.
4. Among all passing candidates, choose the one with the largest mean delta.
5. Break exact ties deterministically using the policy name.
6. Keep the baseline if no candidate passes.

The Selection Gate consumes the paired-bootstrap artifact. It does not perform bootstrap resampling itself.

# Deliverables

The framework is expected to produce the following outputs.

## 1. Standard Decision Dataset

Prepared datasets containing:

- stable `row_id`;
- model features;
- treatment indicator;
- selected outcome;
- deterministic train, validation, and test splits.

---

## 2. Model Training Pipelines

Training pipelines for the supported uplift models, including model logging and validation prediction generation.

---

## 3. Validation Prediction Artifacts

Prediction files containing:

- `row_id`;
- evaluation split;
- treatment and outcome values required for evaluation;
- policy or model name;
- predicted uplift score;
- model provenance.

---

## 4. Validation Evaluation

Standardized validation outputs, including:

- Top-K targeting metrics;
- policy value;
- incremental outcome;
- Qini;
- AUUC;
- paired-bootstrap confidence intervals;
- paired contrasts against the configured baseline.

---

## 5. Champion-Selection Artifact

An artifact recording:

- the selected policy;
- selection settings;
- validation evidence;
- the exact model identity required for reload;
- the source experiment and upstream artifacts.

---

## 6. Locked-Test Evaluation

A final evaluation produced **only** from the locked champion.

The locked test reports final performance and must never be used to select or replace the champion.

---

## 7. Documentation and Tests

Documentation and automated checks covering:

- dataset contracts;
- prediction and artifact schemas;
- experiment identity;
- validation/test isolation;
- champion selection;
- locked-test evaluation;
- reproducible use of `row_id`.

---

# Current Status

The first implementation targets the **Criteo Uplift Prediction Dataset**.

The current workflow includes:

- Criteo data preparation and validation;
- stable `row_id` generation;
- deterministic train, validation, and test splits;
- treated-response, T-Learner, and X-Learner training pipelines;
- MLflow model logging;
- validation prediction artifacts;
- prediction alignment by `row_id`;
- standardized uplift evaluation;
- paired bootstrap comparison;
- a deterministic Selection Gate;
- a separate locked-test evaluation pipeline.

The project is currently in the framework-hardening stage.

Current priorities are:

- ensuring each experiment uses one consistent set of artifacts;
- validating prediction and evaluation schemas;
- linking the champion to the exact model run that produced its validation predictions;
- preventing anything other than the locked champion from entering final evaluation;
- keeping the documentation aligned with the implementation.

Current filenames still contain dataset-specific names such as `criteo`. Renaming modules will be handled after the framework contracts are finalized to avoid unnecessary changes during hardening.

---

# Reusing the Framework

Each dataset requires its own preparation and feature-engineering pipeline.

For example:

```text
Criteo preparation ──────┐
                         ├──→ Standard decision dataset
RetailHero preparation ──┘               ↓
                                  Shared model contract
                                           ↓
                                Shared evaluation
                                           ↓
                                 Shared selection
                                           ↓
                               Locked-test evaluation
```

The reusable part is the workflow—not the trained model.

A model trained on Criteo must not be used directly on RetailHero or another unrelated dataset.

The framework is considered reusable when a new dataset can be integrated without rewriting the downstream evaluation, bootstrap, selection, or locked-test logic.

---

# Temporary Repository Structure

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

This structure is temporary and may be simplified once the framework contracts and entry points are finalized.

Generated datasets, model artifacts, predictions, metrics, figures, local MLflow databases, environments, and temporary files should not be committed to the repository.

---

# Next Development Steps

The next stage focuses on:

1. Completing the remaining framework-hardening tasks.
2. Aligning the documentation with the implemented behavior.
3. Removing obsolete and duplicated code once the workflow is stable.
4. Finalizing dataset, model, artifact, and experiment contracts.
5. Improving configuration-driven execution so supported models can be added without changing the core framework.
6. Integrating RetailHero through a separate dataset adapter.
7. Verifying that downstream evaluation can be reused without copying Criteo-specific code.

Future extensions may include automated retraining, online serving, production monitoring, model registry operations, and champion/challenger automation. These are outside the scope of the current project.

---

# References

The framework design is informed by established work in uplift modeling, statistical model evaluation, and reproducible machine learning workflows.

1. **Gutierrez, P., & Gérardy, J.-Y. (2017).** *Causal Inference and Uplift Modelling: A Review of the Literature.* Proceedings of the International Conference on Predictive Applications and APIs (PAPIs).
   - Provides a comprehensive review of uplift-modeling methods, evaluation metrics, and practical applications.

2. **Zhao, Y., Fang, X., & Simchi-Levi, D. (2017).** *Uplift Modeling with Multiple Treatments and General Response Types.* Proceedings of the SIAM International Conference on Data Mining (SDM).
   - Introduces a general framework for uplift modeling and discusses policy-oriented evaluation.

3. **Raschka, S. (2018).** *Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning.* arXiv:1811.12808.
   - Reviews best practices for train/validation/test separation, statistical comparison, and reproducible model selection.

4. **Savvides, C., et al. (2023).** *Model Selection with Bootstrap Validation.* Statistical Analysis and Data Mining.
   - Describes bootstrap-based approaches for estimating uncertainty during model selection.

These references provide the statistical and methodological background for the framework. The implementation itself is an engineering framework that combines these established principles into a reproducible workflow for training, evaluating, selecting, and validating uplift models.