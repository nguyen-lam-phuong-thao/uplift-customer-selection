# Criteo Visit vs Conversion Results

## 1. Objective

The purpose of this report is to compare how the same uplift-modeling framework behaves across the two Criteo outcomes:

- `visit`
- `conversion`

These outcomes represent different customer behaviors and have very different positive rates. The goal is therefore **not to decide which outcome is better**, but to understand how the modeling evidence, model selection, and final targeting decision differ between the two experiments.

Both experiments use the same raw Criteo dataset, the same prepared feature set, the same model family, the same 60/20/20 split structure, the same **Top-5% primary targeting budget**, and the same evaluation workflow:

**Prepared data → train models → validation evaluation → Top-5% uplift-candidate selection → uplift champion → bootstrap replacement gate → locked-test evaluation**

The raw dataset contains **13,979,592 rows** with:

- `f0` to `f11` as anonymized numeric features;
- `treatment` as the treatment indicator;
- `visit` and `conversion` as outcomes;
- `exposure` as a post-treatment variable.

The modeling table keeps only:

**`f0`–`f11` + `treatment` + `visit` + `conversion`**

`exposure` is removed before modeling. The preparation step then creates separate decision datasets for `visit` and `conversion`, adding `row_id` and the train/validation/test split.

For each outcome, the framework trains:

| Model | Role |
|---|---|
| `treated_response_lgbm` | Response baseline |
| `t_learner_lgbm` | Uplift candidate |
| `x_learner_lgbm` | Uplift candidate |

The model-selection process has two distinct stages.

First, only **T-Learner and X-Learner** are compared at the primary 5% budget using `policy_value`. The stronger of those two becomes the **uplift champion**.

Second, that uplift champion is compared with the Response baseline using paired bootstrap. The Response baseline is replaced only when:

```text
ci_lower > 0
```

A narrower 95% confidence interval means the estimated difference is more precise. However, the replacement decision depends on the **location** of the interval, not only its width: the entire interval must lie above zero.

The resulting deployment policy is fixed on validation before the locked test is opened.

---

## 2. Validation Data Sufficiency

The prepared dataset is split as follows:

| Split | Rows | Fraction | Treatment rate |
|---|---:|---:|---:|
| Train | 8,387,755 | 60% | 85% |
| Validation | 2,795,918 | 20% | 85% |
| Locked test | 2,795,919 | 20% | 85% |

Treatment remains strongly imbalanced in every split because the original Criteo population is approximately **85% Treatment and 15% Control**.

The clearest difference between the two outcomes is the number of positive observations available for learning and evaluation.

| Outcome | Validation rows | Positive rate | Positive observations |
|---|---:|---:|---:|
| `visit` | 2,795,918 | 4.6992% | 131,386 |
| `conversion` | 2,795,918 | 0.2916% | 8,154 |

`visit` therefore has more than sixteen times as many positive validation observations as `conversion`.

This matters because `conversion` provides much less raw positive-outcome information than `visit`. However, the number of positive observations alone does not determine whether a policy comparison is precise. Precision is assessed directly from the bootstrap confidence interval: a narrower interval indicates a more precise estimate, while the replacement gate still requires the entire interval to lie above zero.

---

# 3. Visit Results

## 3.1 Validation Whole-Curve Results

The validation whole-curve metrics are:

| Model | AUUC | Qini |
|---|---:|---:|
| `t_learner_lgbm` | **20,868.813** | **8,579.236** |
| `treated_response_lgbm` | 20,815.119 | 8,525.542 |
| `x_learner_lgbm` | 20,615.580 | 8,326.002 |

T-Learner has the highest AUUC and Qini for `visit`.

This means T-Learner produces the strongest cumulative ranking when performance is summarized across the full targeting curve.

However, the framework does **not** select the deployment policy from AUUC or Qini. The actual model-selection rule is based on `policy_value` at the configured **5% budget**, followed by the bootstrap replacement gate.

![Visit Qini Curve](<../../artifacts/figures/criteo_visit_t_learner_lgbm_vs_treated_response_lgbm_vs_x_learner_lgbm_run01_qini_curve.png>)

![Visit Uplift Curve](<../../artifacts/figures/criteo_visit_t_learner_lgbm_vs_treated_response_lgbm_vs_x_learner_lgbm_run01_uplift_curve.png>)

---

## 3.2 Visit Top-5% Policy Results

At the primary 5% budget:

| Model | Policy value | Incremental visits |
|---|---:|---:|
| `t_learner_lgbm` | 0.045709 | 12,250.205 |
| `treated_response_lgbm` | 0.043857 | 7,516.300 |
| `x_learner_lgbm` | **0.045962** | **12,675.340** |

The uplift-candidate step compares only T-Learner and X-Learner.

X-Learner has the higher Top-5% `policy_value`:

- T-Learner: **0.045709**
- X-Learner: **0.045962**

Therefore:

**Visit uplift champion: `x_learner_lgbm`**

This result is different from the whole-curve ranking. T-Learner has the strongest AUUC and Qini, while X-Learner is stronger at the actual 5% operating point used for model selection.

X-Learner also produces about **5,159 more estimated incremental visits** than the Response baseline at the same validation budget.

---

## 3.3 Visit Replacement Gate

Selecting X-Learner as the uplift champion does not automatically replace the Response baseline.

The framework next compares:

`X-Learner policy value − Response policy value`

using paired bootstrap at the same 5% budget.

| Uplift champion | Baseline | Mean Δ policy value | 95% CI | Gate |
|---|---|---:|---:|---|
| `x_learner_lgbm` | `treated_response_lgbm` | **+0.002111** | **[0.001744, 0.002444]** | **Passed** |

The 95% CI is narrow and remains fully above zero, so X-Learner shows a clear and stable advantage over the Response Model. The evidence therefore supports replacing the Response baseline.

**Recommended Visit policy: `x_learner_lgbm`**

Only after this decision is fixed is the locked test opened.

---

## 3.4 Visit Locked-Test Results

The locked test evaluates the already-selected X-Learner policy on unseen data.

At the primary 5% budget:

| Policy | Policy value | Incremental visits |
|---|---:|---:|
| `x_learner_lgbm` | **0.045762** | **12,244.809** |
| `treated_response_lgbm` | 0.043826 | 7,783.349 |

The selected X-Learner remains ahead at the same operating point.

Its Top-5% policy value changes only slightly:

- Validation: **0.045962**
- Locked test: **0.045762**

The whole-curve metrics are somewhat lower on test:

| Metric | Validation | Locked test | Difference |
|---|---:|---:|---:|
| AUUC | 20,615.580 | 20,269.087 | -346.493 |
| Qini | 8,326.002 | 7,979.401 | -346.601 |

The main budget-based decision nevertheless remains stable.

Bootstrap for the selected X-Learner policy at 5% gives:

- Mean incremental visits: **12,334.584**
- 95% CI: **[11,321.628, 13,539.809]**

The interval is completely above zero.

The paired locked-test comparison with the Response baseline also remains positive:

- Mean Δ policy value: **+0.001933**
- 95% CI: **[0.001553, 0.002310]**

The locked test therefore supports the validation decision to deploy X-Learner for `visit`.

---

# 4. Conversion Results

## 4.1 Validation Whole-Curve Results

The validation whole-curve metrics are:

| Model | AUUC | Qini |
|---|---:|---:|
| `treated_response_lgbm` | **2,458.966** | **1,088.637** |
| `t_learner_lgbm` | 2,197.565 | 827.236 |
| `x_learner_lgbm` | 2,099.079 | 728.750 |

For `conversion`, the Response Model has the strongest AUUC and Qini.

This gives a different whole-curve pattern from `visit`. The conventional Response ranking is strongest across the full conversion targeting curve.

As before, however, whole-curve metrics do not directly determine the final deployment decision. The framework still first selects the stronger uplift candidate at the 5% budget and then tests whether that candidate is strong enough to replace the Response baseline.

![Conversion Qini Curve](<../../artifacts/figures/criteo_conversion_t_learner_lgbm_vs_treated_response_lgbm_vs_x_learner_lgbm_run01_qini_curve.png>)

![Conversion Uplift Curve](<../../artifacts/figures/criteo_conversion_t_learner_lgbm_vs_treated_response_lgbm_vs_x_learner_lgbm_run01_uplift_curve.png>)

---

## 4.2 Conversion Top-5% Policy Results

At the primary 5% budget:

| Model | Policy value | Incremental conversions |
|---|---:|---:|
| `t_learner_lgbm` | 0.002731 | 1,619.349 |
| `treated_response_lgbm` | **0.002814** | 1,733.937 |
| `x_learner_lgbm` | 0.002803 | **1,734.434** |

This table shows why the framework must keep its selection metric explicit.

X-Learner has the highest estimated incremental conversions by only about **0.497 conversions** compared with the Response Model. However, the framework's selection and replacement metric is `policy_value`, and the Response Model has the higher Top-5% policy value:

- Response Model: **0.002814**
- X-Learner: **0.002803**

The uplift-candidate step still compares only T-Learner and X-Learner:

- T-Learner policy value: **0.002731**
- X-Learner policy value: **0.002803**

Therefore:

**Conversion uplift champion: `x_learner_lgbm`**

This means X-Learner is the stronger **uplift candidate**. It does not mean that it has already beaten the Response baseline.

---

## 4.3 Conversion Replacement Gate

The selected X-Learner is then compared with the Response baseline using paired bootstrap at 5%.

| Uplift champion | Baseline | Mean Δ policy value | 95% CI | Gate |
|---|---|---:|---:|---|
| `x_learner_lgbm` | `treated_response_lgbm` | -0.000010 | [-0.000049, 0.000031] | **Failed** |

The mean difference is almost zero and the confidence interval is quite narrow, but it still crosses zero.

The 95% CI is narrow but crosses zero, indicating that X-Learner and the Response Model have very similar policy values at the 5% budget. Therefore, there is not enough evidence to replace the Response Model.

**Recommended Conversion policy: `treated_response_lgbm`**

The distinction is important:

- **Uplift champion:** `x_learner_lgbm`
- **Recommended deployment policy:** `treated_response_lgbm`

---

## 4.4 Conversion Locked-Test Results

The locked test evaluates the selected Response policy on unseen data.

At the primary 5% budget:

| Policy | Policy value | Incremental conversions |
|---|---:|---:|
| `treated_response_lgbm` | **0.002885** | **1,900.676** |
| `x_learner_lgbm` | 0.002873 | 1,862.896 |

The Response baseline remains slightly ahead on the metric used by the framework.

Its Top-5% policy value is also stable relative to validation:

- Validation: **0.002814**
- Locked test: **0.002885**

The whole-curve metrics improve slightly:

| Metric | Validation | Locked test | Difference |
|---|---:|---:|---:|
| AUUC | 2,458.966 | 2,479.210 | +20.243 |
| Qini | 1,088.637 | 1,111.709 | +23.071 |

Bootstrap for the selected Response policy at 5% gives:

- Mean incremental conversions: **1,877.574**
- 95% CI: **[1,408.160, 2,253.077]**

The interval remains fully above zero, showing that the selected Response policy produces positive incremental conversions on unseen data.

The paired locked-test comparison between X-Learner and the Response baseline gives:

- Mean Δ policy value: **-0.000013**
- 95% CI: **[-0.000054, 0.000034]**

This interval crosses zero, so the locked test does not show a reliable X-Learner advantage over Response.

---

# 5. Visit vs Conversion

The two outcomes use the same data preparation and the same model-selection workflow, but the evidence leads to different deployment decisions.

| Comparison | Visit | Conversion |
|---|---|---|
| Validation rows | 2,795,918 | 2,795,918 |
| Positive rate | 4.6992% | 0.2916% |
| Positive observations | 131,386 | 8,154 |
| Whole-curve leader | T-Learner | Response Model |
| Uplift champion at Top 5% | X-Learner | X-Learner |
| Uplift champion beats Response on validation policy value? | Yes | No |
| Replacement gate | **Passed** | **Failed** |
| Recommended policy | **X-Learner** | **Response Model** |
| Locked-test selected-policy incremental CI | Above zero | Above zero |
| Locked-test paired contrast vs Response | Positive | Crosses zero |

The most important difference is not simply the name of the uplift champion.

For `visit`, X-Learner has a clear Top-5% policy-value advantage over the Response baseline, and that advantage remains fully above zero under paired bootstrap. The framework therefore replaces the baseline.

For `conversion`, X-Learner is still the stronger uplift candidate when compared with T-Learner, but its Top-5% policy value is slightly below the Response Model and the paired-bootstrap interval crosses zero. The framework therefore keeps the baseline.

`conversion` does have far fewer positive observations than `visit`, but the bootstrap result shows that the replacement failure should not be described simply as a precision problem. The validation interval is narrow and centered around zero, suggesting that X-Learner and Response have genuinely very similar policy values at the configured decision point.


---

# 6. Conclusion

The Criteo experiment answers the main question of whether uplift modeling can improve customer targeting over a conventional Response Model under the same budget. The results show that the answer depends on the outcome: X-Learner provides a clear improvement for `visit`, while no uplift model shows enough advantage to replace the Response baseline for `conversion`.

The next step is to test the framework on additional datasets with different scales and data characteristics. Comparing the results across datasets can help identify which data conditions are more suitable for uplift modeling and when it provides a meaningful advantage over conventional Response targeting.



