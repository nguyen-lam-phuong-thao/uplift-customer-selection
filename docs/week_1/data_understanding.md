# Data Understanding

## Dataset Overview

The Criteo Uplift dataset was successfully loaded and validated.

| Item              |      Value |
| ----------------- | ---------: |
| Number of rows    | 13,979,592 |
| Number of columns |         16 |
| Memory usage      |    ~1.7 GB |

The dataset is large enough for uplift modeling experiments and can be processed in a high-memory environment.

---

## Data Validation

The dataset passed the basic integrity checks.

* Expected 16-column schema is present.
* No missing columns.
* No missing values.
* All anonymized features (`f0`–`f11`) are numeric.
* Binary variables (`treatment`, `exposure`, `visit`, `conversion`) contain valid values only.

No missing-value imputation is required before modeling.

---

## Treatment Allocation

Treatment assignment is intentionally imbalanced.

| Group     |       Rows | Percentage |
| --------- | ---------: | ---------: |
| Control   |  2,096,937 |      15.0% |
| Treatment | 11,882,655 |      85.0% |

This imbalance is part of the dataset design and should be preserved in later analysis.

---

## Outcome Distribution

The dataset contains two binary outcomes:

* `visit`
* `conversion`

| Outcome    |    Rate |
| ---------- | ------: |
| Visit      | 4.6992% |
| Conversion | 0.2917% |

`conversion` is much rarer than `visit`, so modeling conversion is expected to be more difficult.

---

## Aggregate Treatment-Control Comparison

### Visit

| Group     |    Rate |
| --------- | ------: |
| Control   | 3.8201% |
| Treatment | 4.8543% |

Absolute difference:

```text
+1.0342 percentage points
```

Relative increase:

```text
+27.07%
```

### Conversion

| Group     |    Rate |
| --------- | ------: |
| Control   | 0.1938% |
| Treatment | 0.3089% |

Absolute difference:

```text
+0.1152 percentage points
```

Relative increase:

```text
+59.45%
```

These are aggregate treatment-control differences only. They should not be interpreted as individual treatment effects or used directly for customer targeting.

---

## Exposure Variable Analysis

The `exposure` variable behaves differently from the other variables.

Observed patterns:

* Every control sample has `exposure = 0`.
* Only a subset of treated samples has `exposure = 1`.
* Approximately 3.60% of treated samples were actually exposed.
* Exposed samples have much higher visit and conversion rates than non-exposed treated samples.

However, `exposure` occurs after treatment assignment.

```text
treatment → exposure → outcome
```

Therefore, `exposure` is a post-treatment variable. Using it as an input feature may introduce post-treatment leakage.

---

## Feature Inspection

The input features `f0`–`f11` are anonymized numerical variables.

Because their meanings are hidden:

* business interpretation is not possible;
* domain-driven feature engineering is not appropriate;
* feature importance should not be interpreted from a business perspective.

The features are numerically valid and suitable for modeling.

---

## EDA Summary

| Variable     | Usage                                  |
| ------------ | -------------------------------------- |
| `f0`–`f11`   | Pre-treatment input features           |
| `treatment`  | Treatment indicator                    |
| `visit`      | Primary outcome                        |
| `conversion` | Secondary outcome                      |
| `exposure`   | Excluded due to post-treatment leakage |

Overall, the dataset is clean, internally consistent, and suitable for uplift modeling experiments.
