# RetailHero Uplift Modeling Report

## 1. Dataset and Business Problem

RetailHero is a customer-level uplift modeling problem built from a randomized promotional campaign.

The raw data is stored in four related tables:

| Table | What it contains |
|---|---|
| `clients` | Customer information such as age, gender, loyalty-card issue date, and first point-redemption date |
| `products` | Product categories, brand, vendor, segment, and product attributes |
| `purchases` | Customer purchase history before the campaign |
| `uplift_train` | The labeled experiment population, including treatment assignment and the post-campaign purchase outcome |

The labeled experiment contains **200,039 customers**.

For this project:

- `treatment_flg = 1` means the customer received the promotional communication;
- `treatment_flg = 0` means the customer was in the Control group;
- `target = 1` means the customer made a purchase after the campaign.

The purchase history available for feature construction covers **21 November 2018 to 18 March 2019** and is treated as the pre-treatment observation window.

The business problem is not simply to predict who will purchase. Some customers may already be likely to buy even without receiving the campaign, so targeting them does not necessarily create additional value.

The actual question is:

> **Which customers actually change their purchase behavior because of the campaign?**

To answer this, we compare the traditional `Response Model` with two uplift models, `T-Learner and X-Learner`, to see which approach selects customers who generate more incremental purchases under the same targeting budget.

The Response Model targets customers who are most likely to purchase, while the uplift models target customers whose purchase probability is most likely to increase because of the campaign.

---

## 2. Data Understanding and Cleaning

### 2.1 Define the modeling population first

The project only models customers available in `uplift_train`. All other tables are restricted to this same population before detailed analysis.

This gives the working scope:

- **200,039 customers**
- **22,882,690 purchase rows**
- **4,024,949 transactions**
- **40,716 products** appearing in those customers' histories

The initial structural checks found no duplicate customer/product primary identifiers where uniqueness was expected, no invalid treatment or target values, and no broken customer or product links.

As a result, the majority of the study concentrated on values that potentially skew future customer features.

### 2.2 Customer data issues

The clearest issue is `age`.

The raw values range from **-7,491 to 1,852**, which cannot be interpreted as real customer ages. Out of 200,039 customers, **807** fall outside the practical analytical range of **13–100**.

Because the true ages cannot be reconstructed from the available data, we do not guess replacements. These invalid values are converted to missing values and handled later during feature engineering.

Customer dates also require checking. There are **245 customers** where `first_redeem_date` appears earlier than `first_issue_date`.

Most of these cases are only a few seconds apart and happen on the same day, so they are treated as recording errors and corrected by setting `first_redeem_date` equal to `first_issue_date`. One case has a much larger gap of about 43 days, so its redemption date is set to missing because the correct date cannot be determined.

No customer is removed.

### 2.3 Product data issues

### 2.3 Product Data Issues

Most product fields are consistent, but two areas require attention: `netto` and `product_quantity`.

`netto` is highly right-skewed, with a median of about 0.30 and a maximum of 1,150. However, its exact meaning and unit are not documented, so we cannot reliably determine whether the largest values are valid or incorrect. We therefore keep the observed values unchanged and do not use `netto` for customer feature construction. The 3 missing values are filled with the median `netto` of the relevant product population as a simple cleaning step.

The `product` hierarchy itself is consistent, with:

* 42 Level-2 categories,
* 201 Level-3 categories,
* 783 Level-4 categories,
* 4,183 brands,
* 116 segments,
* 3,136 vendors.

Missing category or identifier values are represented as `UNKNOWN` instead of assigning an unsupported category.

`product_quantity` also contains many zero values and a small number of very large values. However, the checks do not show clear inconsistencies: there are no negative or fractional quantities, zero quantities appear across many products, and the largest quantities are found in high-value transactions. We therefore keep `product_quantity` unchanged.

Since `product_quantity` has a clear meaning, it is then aggregated at transaction level into basket quantity. This is later used to describe customer shopping behavior through features such as average and median basket size.

### 2.4 Transaction Structure and Aggregation

When inspecting the purchase history, the same `transaction_id` appears across multiple rows with different products. This suggests that one row represents a **product line within a transaction**, rather than a complete transaction.

We then check the transaction identifiers and find that `transaction_id` is reused in **8 cases**. Therefore, it cannot be used alone as a unique transaction key. A transaction is instead identified by:

> **client_id + transaction_id + transaction_datetime + store_id**

Next, we check how the remaining fields behave within each transaction. `purchase_sum` and loyalty-point values stay the same across all product rows of the same transaction, while `product_id`, `product_quantity`, `trn_sum_from_iss`, and `trn_sum_from_red` vary by product line.

Based on this structure:

* `purchase_sum` and loyalty-point values are counted once per transaction
* `product_quantity` and other product-line information are aggregated within each transaction
* `trn_sum_from_iss` is kept separate from `purchase_sum` because they represent different values

`trn_sum_from_red` is dropped because **93.39%** of its values are missing and its available records largely overlap with point-redemption information already captured by the loyalty-point fields.

After cleaning, no purchase history is removed. The dataset retains **200,039 customers**, **40,716 relevant products**, and **22,882,690 purchase rows**, which are later aggregated correctly for customer-level analysis.

---

## 3. Exploratory Data Analysis

### 3.1 First check whether the experiment behaves as expected

Treatment and Control are almost perfectly balanced:

- **Control:** 50.02%
- **Treatment:** 49.98%

The post-campaign purchase rate is:

- **Control:** 60.33%
- **Treatment:** 63.65%

This gives an observed overall Treatment-Control gap of about **+3.32 percentage points**.

The campaign therefore appears to create a positive average effect. However, an average effect does not tell us which customers are actually worth targeting.

![Treatment population](<../week_6/treatment population.png>)


![positive outcome](<../week_6/positive outcome.png>)

### 3.2 Convert purchase history into customer behavior

The purchase history is then summarized for each customer to describe their shopping behavior before the campaign.

The EDA is organized around six behavior areas:

1. **Engagement**: How often and how recently the customer shops
2. **Customer Value**: How much the customer spends
3. **Current Momentum**: Whether the customer's recent activity is keeping pace with their normal behavior
4. **Shopping Preference**: How broad or repetitive their product choices are
5. **Shopping Footprint**: How many stores they use and how concentrated they are in a favorite store
6. **Loyalty Behavior**: How customers earn and redeem loyalty points.

These are not manually assigned customer labels for the final model. They are used to understand which parts of historical behavior may contain useful treatment-response signal.

#### 3.2.1 Customer Shopping Activity Before the Campaign

Shopping frequency varies widely across the customer base, but most customers remain active close to the end of the pre-treatment window on 18 March 2019. The number of active shopping days is also close to the number of transactions, which suggests that customers usually make one transaction when they visit rather than splitting purchases into several orders on the same day.

![Transaction frequency](<../week_6/transaction frequency.png>)

Total customer value depends both on how often a customer returns and on how much they typically spend when they shop. The analysis also shows that high historical spend is generally built through repeated purchases over time, while customers with similar basket sizes can still have very different transaction values depending on the products they choose.

![Customer total spend](<../week_6/customer total spend.png>)


![Typical Basket Size vs Transaction Value](<../week_6/basket size.png>)

Historical engagement and value can still miss an important point: whether the customer is still shopping at their usual pace. We therefore examine `Current Momentum` by comparing recent activity with each customer's own normal shopping rhythm.

For the most recent 30-day period, from `17 February` to `18 March 2019`, most customers remain on or ahead of their usual pace, while about 24.4% are shopping more slowly than normal. This separates customers who have always been less active from customers whose behavior has recently started to decline.

![Customer Shopping Momentum](<../week_6/momentum.png>)

The customer base remains active before the campaign, but shopping behavior varies substantially across customers. A single “active vs. inactive” view is therefore not enough to describe the population. These differences provide the baseline for testing whether different customer profiles respond differently to the campaign.

#### 3.2.2 Shopping Patterns Across Products and Stores

A typical customer buys `62` different products, but this variety still comes with strong repeat behavior. The most frequently purchased product appears in about `42%` of an average customer's transactions. Own-brand products are also widely adopted, while alcohol purchases are much more selective.

A similar pattern appears in store usage. About `75%` of customers shop at more than one store, but a typical customer still makes around `79%` of purchases at one favorite location.

This shows that customers explore different products and stores while still maintaining clear preferences for familiar items and a primary store.


![Single-Store vs Multi-Store Customers](<../week_6/store.png>)

![Favorite-Store Concentration Among Multi-Store Customers](<../week_6/favorite store.png>)

#### 3.2.3 Loyalty behavior

Beyond shopping patterns, customers also differ in how they interact with the loyalty program. Almost all customers receive regular points, but only `60.4%` have ever redeemed them, showing a clear gap between earning and actually using rewards.

![Point Receiving vs Redemption](<../week_6/point receiving.png>)


Among customers, redemption activity also varies in recency: `39.0%` have never redeemed points, `32.3%` redeemed in the past but not within the latest 30 days, and `28.7%` redeemed recently.

![Customer Redemption Recency](<../week_6/redemption recency.png>)


This difference also appears in the type of shopping trip where points are used. When customers redeem points, their transaction value and basket size are typically around `87%` of their usual level, suggesting that redemption tends to happen on somewhat smaller purchases rather than on their largest shopping trips.


### 3.3 Which customer behaviors actually change campaign response?

The overall campaign effect is `+3.32 percentage points`, but this is only an average across all customers. To see whether the campaign works differently for different types of customers, we group customers by their pre-campaign behavior and compare the post-campaign purchase rate between Treatment and Control within each group.

For each group:

> **Treatment gap = Treatment purchase rate − Control purchase rate**

A larger gap means that receiving the campaign is associated with a larger increase in purchase probability for that customer group.

The strongest differences appear in shopping frequency, historical customer value, and store footprint.

| Behavior           | Comparison                   |    Observed treatment gap | What it shows                                              |
| ------------------ | ---------------------------- | ------------------------: | ---------------------------------------------------------- |
| Shopping frequency | Low vs. high frequency       | **+4.41 pp vs. +1.39 pp** | Less frequent shoppers show a larger campaign response     |
| Historical value   | Low vs. high value           | **+5.23 pp vs. +1.04 pp** | Lower-value customers show a much larger campaign response |
| Store footprint    | Single-store vs. multi-store | **+4.50 pp vs. +2.94 pp** | Single-store customers respond more strongly               |


![Treatment Gap by Shopping Frequency](<../week_6/treatment shopping frequency.png>)


![Treatment Gap by Historical Customer Value](<../week_6/treatment gap historical.png>)


![Treatment Gap by Store Footprint](<../week_6/store footprint.png>)

The frequency pattern is especially important. High-frequency customers already have a `87.38%` Control purchase rate, compared with only `32.43%` for low-frequency customers. However, the campaign adds much less for the high-frequency group. In other words, customers who are already very likely to purchase are not necessarily the customers whose behavior changes most because of the campaign.

The same pattern appears even more clearly for customer value: the Treatment-Control gap falls from `+5.23 pp` among low-value customers to only `+1.04 pp` among high-value customers.

Statistical heterogeneity tests support treatment-effect differences for Engagement, Customer Value, and Shopping Footprint, while Current Momentum, Shopping Preference, and Loyalty Behavior do not show clear evidence of heterogeneous response.

The main finding is therefore that high purchase likelihood and high campaign responsiveness are not the same thing. Less frequent, lower-value, and single-store customers show stronger evidence of being influenced by the campaign, which directly motivates the use of uplift modeling instead of targeting customers only by their predicted purchase probability.

---

## 4. Feature Engineering

By the end of EDA, the original tables have already been combined into `retailhero_customer_base`, where each customer is represented by one row of pre-campaign behavior. The table contains customer information together with shopping activity, spending, product preference, store usage, and loyalty behavior.

For example, a customer may already have:

| Feature                         |     Example | Meaning                                               |
| ------------------------------- | ----------: | ----------------------------------------------------- |
| `age`                           |          30 | Customer age                                          |
| `first_redeem_date`             | 20 Jan 2019 | First recorded point use                              |
| `recency_days`                  |           8 | Days since the latest purchase                        |
| `transaction_count`             |           3 | Number of transactions                                |
| `total_spend`                   |         600 | Total historical spending                             |
| `unique_products`               |           2 | Number of different products purchased                |
| `favorite_product_share`        |        0.67 | Share of transactions containing the favorite product |
| `loyalty_tenure_days`           |          76 | Days since joining the loyalty program                |
| `total_express_points_redeemed` |          20 | Total express points used                             |

Most of these behavioral features can already be used directly. Feature engineering therefore focuses only on fields that still need transformation before modeling.

### 4.1 Convert raw dates into pre-campaign behavior

Raw dates are not passed directly to the model. `first_issue_date` has already been summarized as loyalty_tenure_days, so the raw date is removed.

For `first_redeem_date`, the important information is how long ago the customer first used points before the modeling cutoff. With the cutoff at `18 March 2019`, a customer whose first redemption was on `20 January 2019` receives:

> days_since_first_points_use = 57

If no point use is observed before the cutoff, the feature is set to `-1`. This keeps the timing information while preventing post-cutoff activity from entering the model.

### 4.2 Prepare customer demographics

Invalid or missing ages were already converted to missing values during cleaning. Since valid ages are between `13 and 100`, FE replaces missing age with `0`, which acts as a separate out-of-range value for unavailable age.

Gender is converted from text into: gender_F, gender_M

For example, gender = F becomes:

> gender_F = 1, gender_M = 0

If both values are 0, the original gender was Unknown.

### 4.3 Add spending intensity

Total spend alone does not distinguish between customers who generated the same amount over very different periods.

FE therefore adds:

> avg_spend_per_history_day = total_spend / observed shopping-history span

This measures how quickly a customer generates spending during the period in which they were actively shopping.

### 4.4 Add express-point usage intensity

Total redeemed points can also be affected by how often a customer shops. FE therefore adds:

> express_points_redeemed_per_transaction = total_express_points_redeemed / transaction_count

For example, if a customer redeemed 20 express points across 3 transactions:

> 20 / 3 ≈ 6.67

This measures point usage relative to shopping frequency rather than using only the total amount redeemed.

### 4.5 Handle features that cannot be calculated

Some behavioral measures are undefined for particular customers. For example, a customer with only one shopping day has no meaningful average shopping gap, while a customer who never redeemed points has no redemption interval.

These structural missing values are therefore set to `-1`, meaning that the behavior could not be calculated from the available history rather than representing a normal numeric value.

After these transformations, the modeling table keeps all `200,039 customers` and contains `58 numeric features`, together with `treatment_flg` and `target`.

---

## 5. Modeling and Evaluation Setup

After feature engineering, the dataset is split into three parts:

- **Train:** 120,023 customers (**60%**) for model training
- **Validation:** 40,008 customers (**20%**) for model comparison and selection
- **Test:** 40,008 customers (**20%**) kept untouched for the final evaluation

The split is stratified by `treatment_flg × target` so that the Treatment/Control and purchase/non-purchase proportions remain similar across all three sets.

Three customer-selection approaches are then trained using the same 58 features:

| Model | How customers are ranked |
|---|---|
| `treated_response_lgbm` | Customers most likely to purchase when treated |
| `t_learner_lgbm` | Customers with the largest estimated increase in purchase probability caused by treatment |
| `x_learner_lgbm` | Customers with the largest estimated treatment effect using the X-Learner approach |

The main difference is that the **Response Model searches for likely buyers**, while the **T-Learner and X-Learner search for customers whose purchase behavior is most likely to change because of the campaign**.

The main business scenario assumes that only **5% of customers can be contacted**. On the validation set of 40,008 customers, each model therefore ranks all customers and selects its top **2,001 customers**.

The main evaluation question is:

> **If only 5% of customers can be targeted, which model selects the group that generates the most incremental purchases?**

The **5% budget** is used as the main comparison point, while **1%, 10%, 20%, and 30%** are also evaluated to check whether the ranking remains useful at different targeting sizes.

All model selection is performed on the **validation set**. Once the best policy is chosen, that decision is fixed and the **test set** is used only for the final evaluation.

---

## 6. Validation Results

### 6.1 Top-5% targeting result

At the primary 5% budget:

| Policy | Policy value | Estimated incremental purchases |
|---|---:|---:|
| `t_learner_lgbm` | **0.610984** | **330.34** |
| `x_learner_lgbm` | 0.608533 | 282.07 |
| Random targeting | 0.605989 | 82.23 |
| `treated_response_lgbm` | 0.603352 | 19.62 |

The result is clear at the configured decision point.

Both uplift models outperform the conventional Response Model, and **T-Learner produces the strongest result**.

The difference also has an important business interpretation. The Response Model tends to rank customers who are already very likely to purchase. T-Learner instead finds a group with a much larger observed Treatment-Control difference, which translates into substantially more estimated incremental purchases under the same 5% budget.

### 6.2 Whole-curve result

The broader ranking metrics show the same ordering:

| Model | AUUC | Qini |
|---|---:|---:|
| `t_learner_lgbm` | **452.80** | **120.57** |
| `x_learner_lgbm` | 449.76 | 117.54 |
| `treated_response_lgbm` | 259.82 | -72.41 |

Higher AUUC and Qini indicate that useful incremental-response customers are ranked earlier across the targeting curve.

T-Learner therefore leads both at the main 5% budget and across the wider validation ranking.

![retailhero_target_t_learner_lgbm_vs_treated_response_lgbm_vs_x_learner_lgbm_run01_qini_curve](<../week_6/retailhero_target_t_learner_lgbm_vs_treated_response_lgbm_vs_x_learner_lgbm_run01_qini_curve.png>)


![retailhero_target_t_learner_lgbm_vs_treated_response_lgbm_vs_x_learner_lgbm_run01_uplift_curve](<../week_6/retailhero_target_t_learner_lgbm_vs_treated_response_lgbm_vs_x_learner_lgbm_run01_uplift_curve.png>)

---

## 7. Model Selection and Replacement Gate

The first selection step compares only the two uplift candidates.

At the 5% validation budget:

- T-Learner policy value: **0.610984**
- X-Learner policy value: **0.608533**

T-Learner is therefore selected as the **uplift champion**.

The next question is stricter: is T-Learner's advantage over the Response Model stable enough to justify changing the targeting policy?

The project uses paired bootstrap for this comparison. The replacement gate passes only when the lower bound of the 95% confidence interval for:

`T-Learner policy value − Response policy value`

is greater than zero.

The validation result is:

- Mean difference: **+0.007818**
- 95% CI: **[0.002821, 0.011853]**

The complete interval is above zero.

**Replacement gate: PASS**

The deployment decision is therefore frozen as:

**Recommended policy: `t_learner_lgbm`**

Only after this decision is fixed do we open the test set.

---

## 8. Locked-Test Results

The locked test checks whether the validation decision carries over to unseen customers.

At the same 5% targeting budget:

| Policy | Policy value | Estimated incremental purchases |
|---|---:|---:|
| `t_learner_lgbm` | **0.609257** | **239.02** |
| `treated_response_lgbm` | 0.605576 | 37.57 |

T-Learner remains ahead in the point estimates.

The whole-curve metrics also remain stronger:

| Model | AUUC | Qini |
|---|---:|---:|
| `t_learner_lgbm` | **402.33** | **70.00** |
| `treated_response_lgbm` | 294.79 | -37.53 |

Performance does decline from validation:

- AUUC: **452.80 → 402.33**
- Qini: **120.57 → 70.00**
- 5% policy value: **0.610984 → 0.609257**

The Top-5% policy-value point estimate changes only slightly from validation to test, so there is no large performance drop at the primary operating point.

However, the locked-test paired-bootstrap result is less certain:

- Mean policy-value difference: about **+0.00363**
- 95% CI: **[-0.000333, 0.008091]**

The interval crosses zero. The locked test does not provide strong statistical evidence of a positive T-Learner policy-value advantage over the Response Model.

---

## 9. Conclusion

The RetailHero experiment shows that uplift-based targeting can outperform conventional Response targeting at the selected Top-5% budget. T-Learner achieved the highest validation policy value and passed the replacement gate against the Response Model, supporting its selection as the recommended policy.

On the locked test, T-Learner maintained a slightly higher policy-value point estimate than the Response Model. However, the paired confidence interval crossed zero, meaning that the locked test does not provide strong statistical evidence of a positive T-Learner advantage on unseen data. Therefore, the result supports the validation-based policy selection while showing that the magnitude of the advantage remains uncertain.

Overall, the RetailHero results reinforce the main finding of this project: uplift modeling can provide additional targeting value over conventional Response modeling, but the advantage is not consistent across all datasets and outcomes. Across the evaluated benchmarks, the effectiveness of uplift-based targeting depends on the specific data setting, and model selection should therefore be based on empirical policy-level evaluation rather than assuming that an uplift model will always outperform a Response Model.


