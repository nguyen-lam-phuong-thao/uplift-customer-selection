# Customer Selection with Uplift Modeling

This repository is organized as a two-phase uplift-modeling project for marketing customer selection.

Core principle: **Reuse the framework, not the Criteo model.**

Phase 1 uses the Criteo Uplift Prediction Dataset to build the shared project foundation. Phase 2 will reuse the same framework for RetailHero, but RetailHero models must be trained separately on RetailHero data.

## Current Scope

Only the Phase 1 Criteo data-loading, validation, EDA, and dataset-decision workflow has working implementation now.

Implemented:

- Load Criteo data from local or Kaggle-provided `.csv` / `.csv.gz` paths.
- Validate required Criteo columns.
- Produce a JSON-serializable validation report.
- Run the Criteo EDA notebook with shared code from `src/uplift_modeling`.
- Define Layer 3 data decisions and export decision datasets for the later preparation/modeling stages.

Planned for later:

- Phase 1 Criteo response model, T-Learner, X-Learner, Qini, AUUC, and policy value.
- Phase 2 RetailHero feature engineering, training, model registry, batch scoring, Top-K targeting, monitoring, and retraining.
- Application, SQL workflows, contracts, model artifacts, and production tracking.

Future modules currently contain only short responsibility docstrings. They do not contain placeholder classes, empty functions, or fake implementations.

## Repository Structure

```text
.
├── app/
├── artifacts/
│   ├── figures/
│   └── metrics/
├── configs/
│   └── data.yaml
├── contracts/
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
│       ├── criteo/
│       └── retailhero/
├── docs/
│   ├── code_rules.md
│   └── week_1/
├── notebooks/
│   ├── phase1_criteo/
│   │   └── 01_criteo_eda.ipynb
│   └── phase2_retailhero/
├── sql/
├── src/
│   └── uplift_modeling/
│       ├── artifacts/
│       ├── data/
│       │   ├── criteo.py
│       │   ├── preparation.py
│       │   └── validation.py
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       ├── pipelines/
│       ├── tracking/
│       └── utils/
├── tests/
├── requirements.txt
└── README.md
```

Raw datasets, processed datasets, local environments, Python bytecode, and temporary files should not be committed.

## Phase 1 — Criteo

Criteo is the current working phase.

Expected Criteo columns:

```text
f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, treatment, conversion, visit, exposure
```

Default local data path:

```text
data/raw/criteo/criteo-research-uplift-v2.1.csv.gz
```

The loader validates by column name, not by column position.

Run the EDA notebook:

```text
notebooks/phase1_criteo/01_criteo_eda.ipynb
```

The notebook reads `configs/data.yaml`, imports shared logic from `src/uplift_modeling`, loads the configured Criteo file, and writes the validation report to:

```text
artifacts/metrics/criteo_validation.json
```

Layer 3 decisions in the notebook create decision datasets for the later modeling stage:

```text
data/processed/criteo/criteo_decision_visit.parquet
data/processed/criteo/criteo_decision_conversion.parquet
```

These datasets keep `f0`-`f11`, `treatment`, the selected outcome, and a deterministic `split` column. They exclude `exposure` because it is a post-treatment variable and would create leakage risk.
They also include a canonical `row_id` column created at the preparation
boundary. `row_id` is deterministic for the prepared dataset, non-null, unique
within that dataset, and must be preserved unchanged through training,
prediction artifacts, evaluation, bootstrap, selection, and locked-test
reporting.

Current Phase 1 split policy:

```text
Train / validation / test = 60% / 20% / 20%
```

The split is stratified by `treatment` and the selected outcome.

For Kaggle, keep the same source code and override the notebook `DATA_PATH` value when the dataset is mounted outside `data/raw/criteo/`.

## Phase 2 — RetailHero

RetailHero is a future extension phase. It should reuse shared concepts, package structure, evaluation methods, and pipeline patterns from Phase 1.

It must not reuse a model trained on Criteo for RetailHero inference. RetailHero requires separate feature engineering and separate model training because the schema, customer identifiers, business context, and historical purchase signals differ from Criteo.

Planned RetailHero direction:

- Build customer-level features from purchases and products.
- Join treatment and outcome labels.
- Train RetailHero-specific uplift models.
- Score customers and select Top-K targets.
- Monitor model and targeting performance.
- Retrain when data or performance changes require it.

## Coding Standard

Repository coding rules are documented in:

```text
docs/code_rules.md
```

Reusable logic belongs in `src/uplift_modeling`. Notebooks should mainly orchestrate, inspect, and visualize.
