# Week 3 Work Plan

| Field         | Value                                          |
| ------------- | ---------------------------------------------- |
| Period        | August 3–7, 2026                               |
| Author        | Nguyễn Lâm Phương Thảo                         |
| Project       | Customer Selection with Uplift Modeling        |
| Team size     | 1                                              |
| Current phase | Framework completion and Hillstrom integration |

## Week 3 Objectives

* [x] Complete the remaining framework tasks on Monday and Tuesday.
* [x] Successfully run the full workflow on Criteo.
* [x] Synchronize the code, tests, configuration, and documentation.
* [x] Start Hillstrom integration from Wednesday.
* [x] Complete initial data understanding and basic feature engineering.
* [x] Create the Hillstrom decision dataset.
* [x] Run the framework on Hillstrom before the end of the week.

---

## Daily Tasks

| Date       | Main Work                                                                                                                                           | Required Result                                                                                                      |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Mon, Aug 3 | Complete the remaining framework-hardening tasks: manifest, artifact validation, `row_id`, dataset contract, champion identity, and test isolation. | No major logic issues identified during the audit remain unresolved.                                                 |
| Tue, Aug 4 | Run tests, fix integration bugs, remove redundant code, update documentation, and run the full Criteo experiment.                                   | The complete workflow runs successfully from training through locked-test evaluation.                                |
| Wed, Aug 5 | Review and inspect the Hillstrom dataset; define the observation unit, treatment, outcome, feature window, and outcome window.                      | Completed data-understanding notes and a plan for creating the decision dataset.                                     |
| Thu, Aug 6 | Build customer-level features using data available before treatment.                                                                                | A feature table with missing values, duplicates, and potential leakage checked.                                      |
| Fri, Aug 7 | Create `row_id`, train/validation/test splits, and the Hillstrom decision dataset; run the first model.                                             | Hillstrom successfully runs through training and validation evaluation without rewriting the shared evaluation code. |

---

## Framework Completion Criteria

The framework is considered complete when:

* Predictions are joined using `row_id` and do not depend on row order.
* Intermediate evaluation uses validation data only.
* The manifest contains artifacts from only one experiment.
* The Selection Gate accepts only deployable candidates.
* `random_targeting` is not passed into the Selection Gate.
* The champion is linked to the correct MLflow run and model artifact.
* The locked-test pipeline only loads the selected champion.
* The full Criteo workflow and automated tests run successfully.
* The documentation is consistent with the current codebase.

---

## Hillstrom Minimum Deliverables

By the end of the week:

* [x] Data-understanding notes.
* [x] Treatment and outcome definitions.
* [x] Initial feature-engineering pipeline.
* [x] Customer-level feature table.
* [x] Standard decision dataset.
* [x] Hillstrom configuration.
* [x] At least one model run.
* [x] Validation predictions and evaluation output.

---

## Week 3 Priority

```text
Monday–Tuesday: Complete and verify the framework
Wednesday: Understand the Hillstrom data
Thursday: Build customer-level features
Friday: Create the decision dataset and run the first model
```

### Not Prioritized This Week

* Adding new models.
* Hyperparameter tuning.
* Renaming the entire codebase.
* Dashboard, deployment, or monitoring.
