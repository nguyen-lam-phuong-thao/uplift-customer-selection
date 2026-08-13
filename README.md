# Customer Selection with Uplift Modeling

> A reusable framework for customer selection with uplift modeling, from a prepared dataset to model selection and locked-test evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Purpose](#purpose)
- [Framework Boundary](#framework-boundary)
- [Input Dataset](#input-dataset)
- [Current Supported Models](#current-supported-models)
- [Uplift Selection and Replacement Gate](#uplift-selection-and-replacement-gate)
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
Train Response + uplift candidates
        ↓
Validation evaluation
        ↓
Select uplift champion
        ↓
Replacement gate vs Response baseline
        ↓
Locked-test evaluation
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
| Uplift selection | Select the best uplift model using validation Top-K performance. |
| Replacement gate | Test whether the selected uplift model reliably outperforms the Response baseline. |
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
Train Response + uplift candidates
        ↓
Validation evaluation
        ↓
Select uplift champion
        ↓
Replacement gate vs Response baseline
        ↓
Locked-test evaluation of uplift champion + baseline
```

`row_id` is an internal technical identifier created by the framework to keep predictions and evaluation rows aligned across models.

Splits are created deterministically. Validation is used for uplift selection and the replacement gate, test is used only after those validation decisions have been frozen.

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

## Uplift Selection and Replacement Gate

Both decisions use **validation data only**.

### Uplift selection

Only uplift models participate in uplift-champion selection.

| Step | Rule |
|---:|---|
| 1 | Evaluate the configured uplift candidates at the primary validation Top-K operating point. |
| 2 | Select the candidate with the largest configured primary metric. |
| 3 | Break exact ties deterministically by policy name. |
| 4 | Save the result as `uplift_champion_policy`. |

`treated_response_lgbm` is not allowed to become the uplift champion.

### Replacement gate

After the uplift champion is fixed, compare it with `treated_response_lgbm` using the existing paired bootstrap contrast at the same primary operating point.

| Result | Meaning |
|---|---|
| `ci_lower > 0` | `replacement_gate_passed = true`: there is stable evidence that the uplift champion outperforms the Response baseline. |
| `ci_lower <= 0` | `replacement_gate_passed = false`: the uplift champion remains unchanged, but the Response baseline remains the recommended deployment policy. |

The framework keeps three separate concepts:

- `uplift_champion_policy`: the best uplift model.
- `replacement_gate_passed`: whether the uplift champion has enough evidence to replace the Response baseline.
- `recommended_deployment_policy`: the uplift champion when the gate passes, otherwise the Response baseline.

### Default top-k budgets

| Budget |
|---:|
| 1% |
| 5% |
| 10% |
| 20% |
| 30% |

> **Test isolation:** The locked test does not select models or change the replacement-gate decision. It evaluates the validation-selected uplift champion and the Response baseline as frozen policies.

---

## Training Example

Run the full validation experiment with:

```bash
python -m uplift_modeling.pipelines.run_experiment   --dataset-config <dataset-config>   --modeling-config <modeling-config>   --experiment-id <experiment-id>   --outcome <outcome>
```

The runner standardizes the prepared dataset, trains the configured models, creates validation predictions, evaluates them, selects the uplift champion from validation Top-K performance, and runs the paired-bootstrap replacement gate against the Response baseline.

Locked-test evaluation is run separately after validation decisions are frozen. It scores both the selected uplift champion and `treated_response_lgbm`, reports their test performance and paired contrast, and does not re-select a model.

---

## Kaggle Training Notebook

A Uplift-training notebook is available here:

[**Criteo uplift-training notebook**](https://www.kaggle.com/code/nguynlmphngtho/criteo-uplift-training)

[**Hillstrom_womens Uplift-training notebook**](https://www.kaggle.com/code/nguynlmphngtho/hillstrom-womens-uplift-training)

[**Hillstrom_mens Uplift-training notebook**](https://www.kaggle.com/code/nguynlmphngtho/hillstrom-men-uplift-training)

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


