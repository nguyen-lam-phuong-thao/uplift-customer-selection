# Hillstrom Visit vs Conversion Results

## 1. Objective

This report compares the results of the two Hillstrom outcomes:

- `visit`
- `conversion`

Both outcomes use the same prepared Hillstrom data structure, the same model set, the same Top-20% targeting rule, and the same evaluation workflow:

**Prepared data → train models → validation evaluation → Top-20 selection → uplift champion → bootstrap replacement gate → locked-test evaluation**

For each campaign, the framework trains:

| Model | Role |
|---|---|
| `treated_response_lgbm` | Response baseline |
| `t_learner_lgbm` | Uplift candidate |
| `x_learner_lgbm` | Uplift candidate |

The uplift champion is selected only between T-Learner and X-Learner. The selected uplift model is then compared with the Response baseline using paired bootstrap.

The baseline is replaced only when:

```text
ci_lower > 0
```

The purpose of this comparison is not to decide whether `visit` or `conversion` is a "better" outcome. They measure different customer behaviors. The goal is to understand how the modeling and evaluation results differ between a relatively common outcome and a much rarer one.

---

## 2. Validation Data Sufficiency

The clearest difference between the two outcomes appears before model comparison.

| Experiment | Outcome | Validation rows | Positive rate | Positive observations |
|---|---|---:|---:|---:|
| Mens | `visit` | 8,523 | 14.443% | 1,231 |
| Womens | `visit` | 8,539 | 12.880% | 1,100 |
| Mens | `conversion` | 8,523 | 0.915% | 78 |
| Womens | `conversion` | 8,539 | 0.726% | 62 |

`visit` has more than one thousand positive observations in each validation set, while `conversion` has fewer than one hundred.

This matters because the framework evaluates only the selected Top-20% group at the main business decision point. For `conversion`, that means the policy estimate is based on a much smaller number of positive outcomes.

As a result, conversion estimates are expected to be less stable, especially near the beginning of the uplift curve where the selected population is still small.

---

# 3. Visit Results

## 3.1 Response Model Diagnostics

The Response Model shows:

| Experiment | ROC-AUC | Average Precision |
|---|---:|---:|
| Mens Visit | 0.628 | 0.2093 |
| Womens Visit | 0.621 | 0.1825 |

These metrics are used only as model diagnostics. They do not decide the final targeting policy.

---

## 3.2 Validation Whole-Curve Results

### Mens Visit

| Model | AUUC | Qini |
|---|---:|---:|
| `t_learner_lgbm` | 159.916 | -3.531 |
| `treated_response_lgbm` | **175.328** | **11.881** |
| `x_learner_lgbm` | 161.625 | -1.822 |

For Mens Visit, the Response Model has the strongest whole-curve result.

Both uplift models have negative Qini values, which means their full-population ranking is weaker than the Response ranking in this validation sample.

![Mens Visit Qini Curve](../../artifacts/figures/hillstrom_mens_visit_visit_t_learner_lgbm_vs_visit_treated_response_lgbm_vs_visit_x_learner_lgbm_run04_qini_curve.png)

![Mens Visit Uplift Curve](../../artifacts/figures/hillstrom_mens_visit_visit_t_learner_lgbm_vs_visit_treated_response_lgbm_vs_visit_x_learner_lgbm_run04_uplift_curve.png)

### Womens Visit

| Model | AUUC | Qini |
|---|---:|---:|
| `t_learner_lgbm` | 132.998 | 35.900 |
| `treated_response_lgbm` | 118.839 | 21.741 |
| `x_learner_lgbm` | **134.315** | **37.217** |

For Womens Visit, X-Learner has the strongest AUUC and Qini.

This is different from Mens Visit, where the Response Model leads across the full curve. Therefore, the two email campaigns do not produce the same ranking pattern even when the outcome is the same.

![Womens Visit Qini Curve](../../artifacts/figures/hillstrom_womens_visit_visit_t_learner_lgbm_vs_visit_treated_response_lgbm_vs_visit_x_learner_lgbm_run04_qini_curve.png)

![Womens Visit Uplift Curve](../../artifacts/figures/hillstrom_womens_visit_visit_t_learner_lgbm_vs_visit_treated_response_lgbm_vs_visit_x_learner_lgbm_run04_uplift_curve.png)

---

## 3.3 Visit Top-20% Policy Results

| Experiment | Model | Incremental visits | Policy value |
|---|---|---:|---:|
| Mens | `t_learner_lgbm` | 157.279 | 0.120149 |
| Mens | `treated_response_lgbm` | **177.874** | **0.123902** |
| Mens | `x_learner_lgbm` | 161.878 | 0.123434 |
| Womens | `t_learner_lgbm` | 142.914 | 0.122834 |
| Womens | `treated_response_lgbm` | **166.871** | 0.123486 |
| Womens | `x_learner_lgbm` | 136.673 | **0.123773** |

For Mens Visit, the Response Model has the highest Top-20% policy value.

For Womens Visit, X-Learner has the highest policy value, but the difference from the Response Model is very small.

The uplift-candidate step compares only T-Learner and X-Learner. X-Learner therefore becomes the uplift champion for both Visit experiments.

---

## 3.4 Visit Replacement Gate

| Experiment | Uplift champion | Mean Δ policy value vs Response | 95% CI | Gate | Recommended policy |
|---|---|---:|---:|---|---|
| Mens Visit | `x_learner_lgbm` | -0.000929 | [-0.008207, 0.005315] | Failed | `treated_response_lgbm` |
| Womens Visit | `x_learner_lgbm` | +0.000305 | [-0.007136, 0.008921] | Failed | `treated_response_lgbm` |

Neither confidence interval lies completely above zero.

For Mens Visit, the mean difference is slightly negative. For Womens Visit, the mean difference is slightly positive, but the interval still crosses zero.

The validation evidence is therefore not strong enough to replace the Response baseline in either Visit experiment.

---

## 3.5 Visit Locked-Test Results

The selected Response policies are then evaluated on the locked test.

| Experiment | Test Qini | Top-20% incremental visits | Bootstrap mean | 95% CI |
|---|---:|---:|---:|---:|
| Mens Visit | 5.816 | 164.616 | 161.490 | **[104.522, 233.457]** |
| Womens Visit | 13.304 | 130.000 | 135.916 | **[84.528, 192.438]** |

Both confidence intervals are completely above zero.

This gives a clear result for `visit`: the selected Response policies produce positive incremental visits on unseen data for both email campaigns.

---

# 4. Conversion Results

## 4.1 Response Model Diagnostics

The Response Model shows:

| Experiment | ROC-AUC | Average Precision |
|---|---:|---:|
| Mens Conversion | 0.580 | 0.0145 |
| Womens Conversion | 0.592 | 0.0099 |

Average Precision is much lower than for `visit`, which is expected because conversion is much rarer.

These metrics are still only diagnostics and do not decide the final policy.

---

## 4.2 Validation Whole-Curve Results

### Mens Conversion

| Model | AUUC | Qini |
|---|---:|---:|
| `t_learner_lgbm` | **19.446** | **4.449** |
| `treated_response_lgbm` | 17.786 | 2.788 |
| `x_learner_lgbm` | 18.657 | 3.660 |

For Mens Conversion, T-Learner has the strongest whole-curve result.

![Mens Conversion Qini Curve](../../artifacts/figures/hillstrom_mens_conversion_conversion_t_learner_lgbm_vs_conversion_treated_response_lgbm_vs_conversion_x_learner_lgbm_run04_qini_curve.png)

![Mens Conversion Uplift Curve](../../artifacts/figures/hillstrom_mens_conversion_conversion_t_learner_lgbm_vs_conversion_treated_response_lgbm_vs_conversion_x_learner_lgbm_run04_uplift_curve.png)

### Womens Conversion

| Model | AUUC | Qini |
|---|---:|---:|
| `t_learner_lgbm` | **9.506** | **2.554** |
| `treated_response_lgbm` | 8.139 | 1.187 |
| `x_learner_lgbm` | 9.463 | 2.511 |

T-Learner also leads Womens Conversion, although X-Learner is very close.

So, unlike Visit, both Conversion experiments give the same whole-curve winner.

![Womens Conversion Qini Curve](../../artifacts/figures/hillstrom_womens_conversion_conversion_t_learner_lgbm_vs_conversion_treated_response_lgbm_vs_conversion_x_learner_lgbm_run04_qini_curve.png)

![Womens Conversion Uplift Curve](../../artifacts/figures/hillstrom_womens_conversion_conversion_t_learner_lgbm_vs_conversion_treated_response_lgbm_vs_conversion_x_learner_lgbm_run04_uplift_curve.png)

---

## 4.3 Conversion Top-20% Policy Results

| Experiment | Model | Incremental conversions | Policy value |
|---|---|---:|---:|
| Mens | `t_learner_lgbm` | **22.607** | **0.008213** |
| Mens | `treated_response_lgbm` | 20.965 | 0.007978 |
| Mens | `x_learner_lgbm` | 21.361 | 0.007979 |
| Womens | `t_learner_lgbm` | **1.909** | **0.005861** |
| Womens | `treated_response_lgbm` | -3.775 | 0.005157 |
| Womens | `x_learner_lgbm` | 1.483 | 0.005859 |

T-Learner has the highest Top-20% policy value in both Conversion experiments.

It is therefore selected as the uplift champion for both campaigns.

However, the differences in policy value are small, especially for Womens Conversion where T-Learner and X-Learner are almost identical.

---

## 4.4 Conversion Replacement Gate

| Experiment | Uplift champion | Mean Δ policy value vs Response | 95% CI | Gate | Recommended policy |
|---|---|---:|---:|---|---|
| Mens Conversion | `t_learner_lgbm` | +0.000331 | [-0.000836, 0.001532] | Failed | `treated_response_lgbm` |
| Womens Conversion | `t_learner_lgbm` | +0.000677 | [-0.000355, 0.001647] | Failed | `treated_response_lgbm` |

Both point estimates are positive, but both confidence intervals cross zero.

Therefore, T-Learner does not show a stable enough improvement over the Response baseline to justify replacement.

As with Visit, the final validation decision is to keep the Response Model.

---

## 4.5 Conversion Locked-Test Results

| Experiment | Test Qini | Top-20% incremental conversions | Bootstrap mean | 95% CI |
|---|---:|---:|---:|---:|
| Mens Conversion | 0.733 | 11.642 | 12.171 | [-1.274, 27.925] |
| Womens Conversion | 3.018 | 13.117 | 12.906 | [-2.705, 29.989] |

Unlike Visit, both conversion confidence intervals cross zero.

The point estimates are positive, but the locked-test evidence is not strong enough to conclude that the incremental conversion outcome is clearly above zero.

This is consistent with the small number of conversion events observed in validation.

---

# 5. Visit vs Conversion

The two outcomes follow the same modeling and evaluation workflow, but they produce different levels of statistical stability.

| Comparison | Visit | Conversion |
|---|---|---|
| Validation positive observations | 1,100–1,231 | 62–78 |
| Uplift champion | X-Learner | T-Learner |
| Champion passes replacement gate? | No | No |
| Final recommended policy | Response Model | Response Model |
| Locked-test incremental outcome CI | Above zero for both campaigns | Crosses zero for both campaigns |
| Overall stability | Stronger | Weaker |

The most important difference is not the name of the uplift champion.

For `visit`, there are many more positive observations. The locked-test Response policies have confidence intervals fully above zero for both Mens and Womens campaigns.

For `conversion`, the outcome is much rarer. Even when T-Learner has the strongest validation ranking and Top-20% result, its advantage over the Response baseline is not stable under bootstrap. On the locked test, the incremental-conversion confidence intervals also cross zero.

The absolute values of `visit` and `conversion` metrics should not be directly compared because they represent different outcomes with very different base rates. The useful comparison is the strength and stability of the evidence.

---

# 6. Main Findings

1. **Visit provides stronger evaluation evidence than conversion.**  
   Visit has many more positive observations, so the models have more outcome signal to learn from and the Top-20% and locked-test estimates are more stable.

2. **The best uplift candidate is different for the two outcomes.**  
   X-Learner is selected for `visit`, while T-Learner is selected for `conversion`.

3. **Neither uplift champion is strong enough to replace the Response baseline.**  
   All four replacement-gate confidence intervals include zero.

4. **The final recommended policy is the Response Model for both outcomes and both campaigns.**  
   The framework does not replace the baseline based only on a better validation point estimate.

5. **The locked test shows a clearer result for visit.**  
   `visit` has positive incremental outcomes with confidence intervals fully above zero, while `conversion` remains uncertain because its confidence intervals still include zero.

---

# 7. Conclusion

Hillstrom shows that the same uplift-modeling workflow can produce different results depending on the outcome.

For `visit`, the higher positive rate provides more outcome signal for model learning and evaluation. X-Learner is the stronger uplift candidate, but it does not show a stable improvement over the Response baseline. The selected Response policies then show clearly positive incremental visits on the locked test.

For `conversion`, the positive outcome is very rare compared with the full dataset. This gives the models much less signal to learn from and makes treatment-effect and Top-20% estimates less stable. T-Learner is the stronger uplift candidate, but its improvement over the Response baseline is uncertain, and the locked-test confidence intervals also include zero.

The final decision is therefore the same for both outcomes:

**Keep the Response Model as the recommended policy.**

The main difference is the strength of the evidence. `visit` gives a clearer and more stable result, while `conversion` is harder to learn and evaluate because positive outcomes are too sparse.

This shows that model quality depends not only on the modeling method, but also on how much useful outcome signal is available in the dataset.

