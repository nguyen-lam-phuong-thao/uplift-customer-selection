# Customer Selection with Uplift Modeling

> A reusable framework for customer selection with uplift modeling, from a prepared dataset to model selection and locked-test evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Purpose](#purpose)
- [Framework Boundary](#framework-boundary)
- [Input Dataset](#input-dataset)
- [Current Supported Models](#current-supported-models)
- [Champion Selection](#champion-selection)
- [Training Example](#training-example)
- [Kaggle Training Notebook](#kaggle-training-notebook)
- [Repository Structure](#repository-structure)
- [Documentation](#documentation)

---

## Project Overview

This project provides a reusable framework for **customer selection with uplift modeling**.

In a business setting, the framework can be used as a reference workflow when a team wants to evaluate and propose which customers should be prioritized for a marketing campaign, promotion, or other treatment.

Instead of only predicting who is likely to produce an outcome, uplift modeling focuses on identifying customers whose behavior is more likely to change **because of the treatment**.

The framework standardizes the workflow from a prepared dataset to model selection:

```text
Prepared dataset
        ↓
Standardize dataset
        ↓
Train candidate models
        ↓
Validation evaluation
        ↓
Paired bootstrap
        ↓
Champion selection
        ↓
Locked test
```

The framework can be reused across datasets, but each dataset still requires its own preparation before entering the framework.

---

## Purpose

The goal of this project is to provide a reproducible workflow for uplift-modeling experiments, especially for customer selection.

The framework helps:

| Capability | Description |
|---|---|
| Candidate training | Train candidate models on the same dataset and splits. |
| Prediction alignment | Keep predictions aligned with the correct observations. |
| Policy evaluation | Evaluate policies under the same conditions. |
| Statistical comparison | Compare candidates with paired bootstrap. |
| Champion selection | Select and lock a champion using validation data. |
| Test isolation | Keep the test split independent for final evaluation. |
| Experiment traceability | Preserve artifacts and model provenance so experiment results can be traced. |

The final outputs can be used to evaluate models and support a ranked customer recommendation based on the selected policy.

---

## Framework Boundary

The framework starts from a **prepared dataset**.

### Outside the framework

The following steps are outside the framework:

- Raw data processing.
- EDA.
- Feature engineering.
- Treatment/control definition.
- Outcome definition.
- Leakage decisions.
- Dataset-specific encoding or transformation.

### Inside the framework

```text
Prepared dataset
        ↓
Standardize to a common decision dataset
        ↓
Add internal `row_id`
        ↓
Create train / validation / test split
        ↓
Train supported candidates
        ↓
Validation predictions
        ↓
Evaluation + paired bootstrap
        ↓
Selection Gate
        ↓
Locked champion
        ↓
Locked-test evaluation
```

`row_id` is an internal technical identifier created by the framework to keep predictions and evaluation rows aligned across models.

Splits are created deterministically. Validation is used for model comparison and champion selection; test is used only after the champion has been locked.

---

## Input Dataset

A prepared dataset should contain:

| Required component | Description |
|---|---|
| Feature columns | Input features used by the uplift models. |
| Treatment column | Indicates treatment/control assignment. |
| Outcome column(s) | Outcome or outcomes used for the experiment. |

A simple example:

```text
age
tenure
past_purchase_count
average_order_value
treatment
conversion
```

For the Criteo dataset used in this project:

```text
f0
f1
...
f11
treatment
visit
conversion
```

After standardization, one decision dataset for a selected outcome follows this structure:

```text
row_id
feature columns
treatment
outcome
split
```

The input dataset does not need to provide `row_id`; the framework creates it for internal alignment.

---

## Current Supported Models

The framework currently supports:

| Model | Role |
|---|---|
| `treated_response_lgbm` | Default baseline in the current workflow |
| `t_learner_lgbm` | Candidate uplift model |
| `x_learner_lgbm` | Candidate uplift model |

`treated_response_lgbm` is the default baseline in the current workflow.

`random_targeting` is used only as an evaluation benchmark. It is not a deployable model and cannot become the champion.

The framework is designed so that additional model families can be added later through explicit implementation and candidate configuration.

---

## Champion Selection

Champion selection uses **validation data only**.

### Selection rules

| Step | Rule |
|---:|---|
| 1 | Compare each candidate with `treated_response_lgbm`. |
| 2 | A candidate passes when `ci_lower > 0`. |
| 3 | If multiple candidates pass, select the candidate with the largest `mean_delta`. |
| 4 | If no candidate passes, keep the baseline. |

### Default top-k budgets

| Budget |
|---:|
| 1% |
| 5% |
| 10% |
| 20% |
| 30% |

> **Test isolation:** The test split is not used during model selection.

---

## Training Example

Run the full validation experiment with:

```bash
python -m uplift_modeling.pipelines.run_experiment   --dataset-config <dataset-config>   --modeling-config <modeling-config>   --experiment-id <experiment-id>   --outcome <outcome>
```

The runner standardizes the prepared dataset, trains configured candidates, creates validation predictions, evaluates the candidates, runs paired bootstrap, and executes the Selection Gate.

Locked-test evaluation is run separately after the champion has been selected.

---

## Kaggle Training Notebook

A Uplift-training notebook is available here:

[**Criteo uplift-training notebook**](https://www.kaggle.com/code/nguynlmphngtho/criteo-uplift-training)

---

## Repository Structure

```text
.
├── configs/
├── contracts/
├── data/
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

---

## Documentation

For the full workflow, dataset contracts, prediction artifacts, bootstrap, Selection Gate, and locked-test rules, see:

| Document | Description |
|---|---|
| [`docs/framework_workflow.md`](docs/framework_workflow.md) | Full workflow, dataset contracts, prediction artifacts, bootstrap, Selection Gate, and locked-test rules. |

