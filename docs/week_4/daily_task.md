# Week 4 Work Plan

| Field         | Value                                         |
| ------------- | --------------------------------------------- |
| Period        | August 10–16, 2026                            |
| Author        | Nguyễn Lâm Phương Thảo                        |
| Project       | Customer Selection with Uplift Modeling       |
| Team size     | 1                                             |
| Current phase | Hillstrom training and RetailHero integration |

## Week 4 Objectives

* [X] Train the completed uplift modeling framework on the Hillstrom dataset.
* [X] Evaluate and analyze Hillstrom model results using the existing evaluation workflow.
* [X] Investigate treatment effects across different customer features and subgroups.
* [ ] Perform EDA and data understanding for the RetailHero dataset.
* [ ] Define the RetailHero treatment, outcome, feature, and observation unit.
* [ ] Perform initial feature engineering and construct the RetailHero decision dataset.
* [ ] Integrate RetailHero with the existing framework without changing the shared evaluation logic.
* [ ] Run the first RetailHero training and validation experiment.

---

## Daily Tasks

| Date               | Main Work                                                                                                                                                                                      | Required Result                                                                                     |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Mon, Aug 10        | Prepare Hillstrom configuration and run the completed framework on Hillstrom. Verify data loading, training, prediction, and validation evaluation.                                            | Hillstrom runs successfully through the existing training and evaluation workflow.                  |
| Tue, Aug 11        | Analyze Hillstrom model performance and treatment effects. Compare models using the existing evaluation metrics and investigate uplift across customer subgroups.                              | A clear analysis of Hillstrom results, including model performance and treatment-effect patterns.   |
| Wed, Aug 12        | Start RetailHero data understanding and EDA. Inspect schema, observation unit, treatment, outcome, feature distributions, missing values, duplicates, and potential leakage.                   | Completed RetailHero data-understanding and EDA notes with defined treatment and outcome variables. |
| Thu, Aug 13        | Perform initial RetailHero feature engineering and construct the customer-level decision dataset. Validate feature quality and ensure the dataset follows the standard decision-data contract. | A validated RetailHero decision dataset ready for framework training.                               |
| Fri, Aug 14        | Create RetailHero configuration and run the first training and validation experiment using the existing framework.                                                                             | RetailHero completes training and validation evaluation without modifying shared framework logic.   |
| Sat–Sun, Aug 15–16 | Review and analyze RetailHero results, document findings, and identify issues or follow-up work for the next phase.                                                                            | Initial RetailHero experiment report and a clear list of next steps.                                |

---

## Hillstrom Deliverables

By the end of the Hillstrom phase, the following should be available:

* [X] Hillstrom configuration.
* [X] Hillstrom decision dataset.
* [X] At least one completed model run.
* [X] Validation predictions.
* [X] Model evaluation results.
* [X] Model comparison.
* [X] Treatment-effect analysis by customer features.
* [X] Initial interpretation of which customer groups respond differently to treatment.
* [X] Short analysis note documenting the main findings.

---

## RetailHero Minimum Deliverables

By the end of the week, the following should be available:

* [ ] RetailHero data-understanding note.
* [ ] Observation unit definition.
* [ ] Treatment definition.
* [ ] Outcome definition.
* [ ] Feature and outcome window definition.
* [ ] EDA covering data quality and feature distributions.
* [ ] Initial feature-engineering pipeline.
* [ ] Customer-level feature table.
* [ ] Standard RetailHero decision dataset.
* [ ] RetailHero configuration.
* [ ] At least one model run.
* [ ] Validation predictions.
* [ ] Initial validation evaluation output.

---

## Analysis Questions

### Hillstrom

The analysis should answer:

1. **Does the existing uplift framework work correctly on Hillstrom?**
2. **Which model performs best under the existing evaluation criteria?**
3. **Does treatment effectiveness differ across customer subgroups?**
4. **Which customer features are associated with stronger or weaker treatment effects?**
5. **Are there customer segments where the campaign appears particularly effective or ineffective?**

The goal is not only to report model metrics, but to understand **who benefits from the treatment and whether the treatment effect is heterogeneous across customers**.

### RetailHero

The initial RetailHero analysis should answer:

1. **What is the unit of observation?**
2. **What represents treatment and control?**
3. **What is the target outcome?**
4. **Which features are available before treatment?**
5. **Are there missing values, duplicates, or inconsistent records?**
6. **Are there potential leakage variables?**
7. **How are treatment and outcome distributed?**
8. **Which features require transformation or aggregation?**
9. **Can the resulting data be represented using the existing decision-data contract?**

---

## Week 4 Priority

```text
Monday:
Run Hillstrom through the completed framework

Tuesday:
Analyze Hillstrom model results and treatment heterogeneity

Wednesday:
EDA and data understanding for RetailHero

Thursday:
Feature engineering and decision dataset construction for RetailHero

Friday:
Train the framework on RetailHero and evaluate on validation

Weekend:
Analyze results and document findings
```

## Not Prioritized This Week

* Adding new uplift models.
* Hyperparameter tuning.
* Redesigning the framework architecture.
* Rewriting shared evaluation logic for individual datasets.
* Dashboard development.
* Deployment or monitoring.
* Large-scale refactoring unrelated to Hillstrom or RetailHero integration.

## Week 4 Success Criteria

The week is considered successful when:

1. **Hillstrom has been successfully trained and evaluated using the completed framework.**
2. **Hillstrom results have been analyzed beyond aggregate model metrics, including treatment heterogeneity across customer features.**
3. **RetailHero has completed its initial EDA and data-understanding phase.**
4. **RetailHero has a validated customer-level decision dataset compatible with the existing framework.**
5. **The first RetailHero model has been trained and evaluated on validation data.**
6. **The main findings and follow-up issues from both datasets are documented.**
