# Customer Selection with Uplift Modeling

> A reusable framework for customer selection with uplift modeling, from a prepared dataset to model selection and locked-test evaluation.

---

## Training & Evaluation Notebooks

These notebooks show the Data Science workflow and the shared framework training/evaluation on the datasets used in this project.

| Dataset | Outcome | Notebook |
|---|---|---|
| Hillstrom | Visit | [**Hillstrom Uplift — Visit**](https://www.kaggle.com/code/nguynlmphngtho/hillstrom-uplift-visit) |
| Hillstrom | Conversion | [**Hillstrom Uplift — Conversion**](https://www.kaggle.com/code/nguynlmphngtho/hillstrom-uplift-conversion) |
| Criteo | Visit | [**Criteo Uplift Training**](https://www.kaggle.com/code/nguynlmphngtho/criteo-uplift-training) |
| RetailHero | Purchase | [**RetailHero Data Understanding & Cleaning**](notebooks/phase2_retailhero/01_retailhero_data_understanding_and_cleaning.ipynb) |
| RetailHero | Purchase | [**RetailHero EDA**](notebooks/phase2_retailhero/02_retailhero_eda.ipynb) |
| RetailHero | Purchase | [**RetailHero Feature Engineering & Training**](notebooks/phase2_retailhero/03_retailhero_feature_engineering.ipynb) |

---

## Experiment Reports

- [**Criteo Training Results**](docs/week_3/criteo_train_result.md)
- [**Hillstrom Training Results**](docs/week_4/hillstrom_train_result.md)


---

## Table of Contents

- [Training & Evaluation Notebooks](#training--evaluation-notebooks)
- [Project Overview](#project-overview)
- [Purpose](#purpose)
- [Framework Boundary](#framework-boundary)
- [Input Dataset](#input-dataset)
- [Current Supported Models](#current-supported-models)
- [Uplift Selection and Replacement Gate](#uplift-selection-and-replacement-gate)
- [Training Example](#training-example)
- [Repository Structure](#repository-structure)
- [Documentation](#documentation)

---

## Project Overview

This project provides a reusable framework for **customer selection with uplift modeling**.

In a business setting, the framework can be used as a reference workflow when a team wants to evaluate and propose which customers should be prioritized for a marketing campaign, promotion, or other treatment.

Instead of only predicting who is likely to produce an outcome, uplift modeling focuses on identifying customers whose behavior is more likely to change **because of the treatment**.

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
Add internal row_id
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

Splits are created deterministically. Validation is used for uplift selection and the replacement gate; test is used only after those validation decisions have been frozen.

---

## Input Dataset

To use the framework with a new dataset, the user needs to provide:

1. A **prepared Parquet dataset**.
2. A **dataset config** describing which columns are features, treatment, and outcomes.

The prepared dataset should already be modeling-ready:

```text
feature columns
treatment
outcome column(s)
```

For example:

| age | tenure | past_purchase_count | treatment | conversion |
|---:|---:|---:|---:|---:|
| 32 | 12 | 5 | 1 | 1 |
| 45 | 30 | 10 | 0 | 0 |
| 27 | 4 | 2 | 1 | 0 |

Here:

- `age`, `tenure`, `past_purchase_count` are model features.
- `treatment = 1` means treated and `treatment = 0` means control.
- `conversion` is the observed outcome.

The framework does not perform feature engineering or decide which columns are safe to use. Those decisions must already be completed when the prepared dataset is created.

### Dataset config

The dataset config tells the framework how to interpret the prepared file.

Example:

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

The input dataset does not need to contain `row_id` or `split` when the framework is configured to create them.

### Example: Criteo

The prepared Criteo dataset contains:

```text
f0
f1
...
f11
treatment
visit
conversion
```

From this single prepared dataset, the framework creates one decision dataset for each outcome:

```text
visit:
row_id
f0 ... f11
treatment
visit
split
```

```text
conversion:
row_id
f0 ... f11
treatment
conversion
split
```

### Example: Hillstrom

Hillstrom originally contains three campaign groups:

```text
No E-Mail
Mens E-Mail
Womens E-Mail
```

Because the framework uses binary treatment, preparation creates two separate experiments:

```text
hillstrom_mens:
No E-Mail   = 0
Mens E-Mail = 1
```

```text
hillstrom_womens:
No E-Mail     = 0
Womens E-Mail = 1
```

Categorical variables are encoded before entering the framework.

A prepared Hillstrom dataset therefore has the same logical structure as Criteo:

```text
customer features
treatment
visit
conversion
```

The framework then standardizes it into separate `visit` and `conversion` decision datasets.

In short:

```text
User prepares:

features + treatment + outcome(s)
            ↓
        dataset config
            ↓
Framework creates:

row_id + features + treatment + one outcome + split
```

---

## Current Supported Models

The framework currently supports:

| Model | Role |
|---|---|
| `treated_response_lgbm` | Traditional Response baseline |
| `t_learner_lgbm` | Uplift candidate |
| `x_learner_lgbm` | Uplift candidate |

`treated_response_lgbm` ranks customers by predicted outcome response and is used as the baseline for the replacement gate.

`T-Learner` and `X-Learner` estimate treatment effect and participate in uplift-champion selection.

`random_targeting` is used only as an evaluation benchmark and cannot become champion.

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

After the uplift champion is fixed, compare it with `treated_response_lgbm` using the paired bootstrap contrast at the same primary operating point.

| Result | Meaning |
|---|---|
| `ci_lower > 0` | `replacement_gate_passed = true`: there is stable evidence that the uplift champion outperforms the Response baseline. |
| `ci_lower <= 0` | `replacement_gate_passed = false`: the Response baseline remains the recommended deployment policy. |

The framework keeps three separate concepts:

- `uplift_champion_policy`: the best uplift model.
- `replacement_gate_passed`: whether the uplift champion has enough evidence to replace the Response baseline.
- `recommended_deployment_policy`: the uplift champion when the gate passes, otherwise the Response baseline.

### Default Top-K budgets

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
python -m uplift_modeling.pipelines.run_experiment \
  --dataset-config <dataset-config> \
  --modeling-config <modeling-config> \
  --experiment-id <experiment-id> \
  --outcome <outcome>
```

Example:

```bash
python -m uplift_modeling.pipelines.run_experiment \
  --dataset-config configs/datasets/criteo.yaml \
  --modeling-config configs/modeling/uplift_lgbm.yaml \
  --experiment-id criteo-visit-001 \
  --outcome visit
```

Locked-test evaluation is run separately after validation decisions are frozen.

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
| [`docs/week_3/criteo_train_result.md`](docs/week_3/criteo_train_result.md) | Criteo experiment results, model evaluation, comparison, and findings. |
| [`docs/week_4/hillstrom_train_result.md`](docs/week_4/hillstrom_train_result.md) | Hillstrom experiment results, model evaluation, comparison, and treatment-effect analysis. |
