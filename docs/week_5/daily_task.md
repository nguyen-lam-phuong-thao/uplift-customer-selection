# Week 5 Work Plan

| Field         | Value                                              |
| ------------- | -------------------------------------------------- |
| Period        | August 17–23, 2026                                |
| Author        | Nguyễn Lâm Phương Thảo                            |
| Project       | Customer Selection with Uplift Modeling           |
| Team size     | 1                                                  |
| Current phase | RetailHero integration and preliminary dashboard  |

## Week 5 Objectives

* [x] Perform EDA and data understanding for the RetailHero dataset.
* [x] Define the RetailHero treatment, outcome, feature, and observation unit.
* [x] Define the feature and outcome observation windows for RetailHero.
* [x] Perform initial feature engineering and construct the RetailHero decision dataset.
* [x] Validate that the RetailHero decision dataset follows the standard decision-data contract.
* [x] Integrate RetailHero with the existing framework without changing the shared training and evaluation logic.
* [x] Run the first RetailHero training and validation experiment.
* [x] Analyze the initial RetailHero model and uplift evaluation results.
* [x] Define the preliminary purpose, users, metrics, and information structure of the dashboard.
* [x] Create an initial dashboard layout/mockup for presenting uplift modeling results.

---

## Daily Tasks

| Date               | Main Work                                                                                                                                                                                            | Required Result                                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Mon, Aug 17        | Perform RetailHero data understanding and EDA. Inspect tables, schema, observation unit, treatment candidates, outcome candidates, missing values, duplicates, data quality, and potential leakage. | Completed RetailHero data-understanding and EDA notes with a clear understanding of the available data.     |
| Tue, Aug 18        | Define the RetailHero treatment, control, outcome, observation unit, feature window, and outcome window. Identify which variables are valid pre-treatment features.                                  | Clear RetailHero experiment definition and feature/outcome observation windows.                             |
| Wed, Aug 19        | Perform initial feature engineering and aggregate RetailHero transactional data into a customer-level feature table. Validate feature quality and remove inappropriate or leakage-prone variables. | A validated customer-level feature table ready for decision-dataset construction.                           |
| Thu, Aug 20        | Construct the standard RetailHero decision dataset, create the RetailHero configuration, and integrate it with the existing framework.                                                              | RetailHero decision dataset and configuration compatible with the existing framework.                       |
| Fri, Aug 21        | Run the first RetailHero training and validation experiment. Verify training, prediction generation, evaluation artifacts, and model comparison.                                                    | RetailHero successfully completes the existing training and validation evaluation workflow.                 |
| Sat, Aug 22        | Analyze RetailHero validation results and initial treatment-effect patterns. Identify important findings, limitations, and issues that require follow-up.                                           | Initial RetailHero experiment analysis and documented findings.                                             |
| Sun, Aug 23        | Define the preliminary dashboard scope and design. Identify dashboard users, key questions, required metrics, charts, filters, and create an initial dashboard layout/mockup.                       | Preliminary dashboard specification and initial layout/mockup for communicating uplift modeling results.    |

---

## RetailHero Deliverables

By the end of the RetailHero integration phase, the following should be available:

* [x] RetailHero data-understanding note.
* [x] Observation unit definition.
* [x] Treatment and control definition.
* [x] Outcome definition.
* [x] Feature window definition.
* [x] Outcome window definition.
* [x] EDA covering data quality and feature distributions.
* [x] Missing-value and duplicate analysis.
* [x] Potential leakage analysis.
* [x] Initial feature-engineering pipeline.
* [x] Customer-level feature table.
* [x] Standard RetailHero decision dataset.
* [x] RetailHero configuration.
* [x] At least one completed model run.
* [x] Validation predictions.
* [x] Model evaluation results.
* [x] Initial model comparison.
* [x] Initial RetailHero experiment analysis.

---

## Preliminary Dashboard Deliverables

The dashboard work this week is limited to **initial analysis and design**, not full implementation.

By the end of the week, the following should be available:

* [X] Dashboard purpose and target-user definition.
* [X] Main business and analytical questions the dashboard should answer.
* [X] List of metrics to present.
* [X] List of required charts and visualizations.
* [X] Initial dashboard sections and information hierarchy.
* [X] Initial filter and interaction requirements.
* [X] Initial dashboard layout or wireframe/mockup.
* [X] Mapping between framework output artifacts and dashboard visualizations.

Possible dashboard information includes:

* Experiment and dataset overview.
* Treatment and control population statistics.
* Outcome statistics.
* Model comparison.
* Top-k policy performance.
* Incremental outcome.
* Uplift curve.
* Qini curve.
* AUUC and Qini metrics.
* Treatment-effect patterns across customer subgroups.
* Champion model and baseline comparison.

---

## Analysis Questions

### RetailHero

The RetailHero analysis should answer:

1. **What is the unit of observation?**
2. **What represents treatment and control?**
3. **What is the target outcome?**
4. **What are the feature and outcome observation windows?**
5. **Which features are available before treatment?**
6. **Are there missing values, duplicates, or inconsistent records?**
7. **Are there potential leakage variables?**
8. **How are treatment and outcome distributed?**
9. **Which features require transformation or aggregation?**
10. **Can the resulting data be represented using the existing decision-data contract?**
11. **Can RetailHero run through the existing framework without dataset-specific changes to the shared training and evaluation logic?**
12. **What do the initial validation results indicate about treatment effectiveness and customer selection?**

### Dashboard

The preliminary dashboard analysis should answer:

1. **Who is the dashboard intended for?**
2. **What decisions should the dashboard support?**
3. **Which framework outputs are important enough to visualize?**
4. **Which metrics should be visible at a glance?**
5. **How should model comparison be presented?**
6. **How should top-k policy performance and incremental outcome be communicated?**
7. **How should treatment heterogeneity across customer groups be presented?**
8. **Which filters or interactions would make the results easier to explore?**
9. **How should the dashboard distinguish validation results from locked-test results?**
10. **What information is required before dashboard implementation begins?**

---

## Week 5 Priority

```text
Monday:
RetailHero EDA and data understanding

Tuesday:
Define treatment, outcome, observation unit, and observation windows

Wednesday:
Feature engineering and customer-level feature construction

Thursday:
Build the RetailHero decision dataset and integrate it with the framework

Friday:
Run the first RetailHero training and validation experiment

Saturday:
Analyze RetailHero results and document findings

Sunday:
Define preliminary dashboard requirements and create an initial layout/mockup
