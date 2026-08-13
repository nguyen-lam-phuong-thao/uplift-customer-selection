## Experiments — Hillstrom

### Dataset và tính phù hợp với framework

Dataset được sử dụng là **Hillstrom Email Marketing Dataset**, gồm **64,000 khách hàng** trong một randomized marketing experiment. Mỗi khách hàng được phân ngẫu nhiên vào một trong ba nhóm:

- `No E-Mail`: không nhận email, dùng làm control;
- `Mens E-Mail`: nhận email quảng bá sản phẩm nam;
- `Womens E-Mail`: nhận email quảng bá sản phẩm nữ.

Dữ liệu phù hợp với bài toán uplift vì có đầy đủ ba thành phần: **pre-treatment customer features, treatment assignment và post-treatment outcome**. 

- Ba experimental group gần như được chia đều;

    | Segment         | Customers | Percentage |
    |-----------------|-----------|------------|
    | Womens E-Mail   | 21,387    | 33.4172%   |
    | Mens E-Mail     | 21,307    | 33.2922%   |
    | No E-Mail       | 21,306    | 33.2906%   |

- Sample-ratio test không cho thấy allocation bất thường (`χ² = 0.203`, `p = 0.904`);
- Covariate balance tốt, maximum absolute SMD chỉ khoảng **0.014**,
- Mọi observed level của các feature được kiểm tra đều có observation ở cả ba treatment arm,

    | Feature     | Zero Cells | Minimum Group Size |
    |-------------|------------|--------------------|
    | recency     | 0          | 767                |
    | history_segment | 0       | 416                |
    | mens        | 0          | 9,519              |
    | womens      | 0          | 9,558              |
    | newbie      | 0          | 10,611             |
    | zip_code    | 0          | 3,139              |
    | channel     | 0          | 2,577              |

- Customer characteristics được ghi nhận trước campaign, còn `visit` và `conversion` là outcome sau campaign.


Do framework hiện xử lý binary treatment, Hillstrom được tách thành hai experiment độc lập:

| Experiment | Control | Treatment | Prepared rows |
|---|---|---|---:|
| `hillstrom_mens` | `No E-Mail` | `Mens E-Mail` | 42,613 |
| `hillstrom_womens` | `No E-Mail` | `Womens E-Mail` | 42,693 |

Sau feature engineering, cả hai experiment dùng cùng **12 features** và treatment/control gần như 50/50. Hai outcome được train riêng là `visit` và `conversion`.

Mục tiêu của experiment là:

> Nếu chỉ target một phần khách hàng, policy nào chọn được nhóm tạo ra incremental outcome tốt và ổn định nhất?

---

### Data Understanding

Outcome trung bình khác nhau rõ giữa ba experimental group:


| Group | Visit rate | Conversion rate |
|---|---:|---:|
| `No E-Mail` | 10.62% | 0.57% |
| `Mens E-Mail` | 18.28% | 1.25% |
| `Womens E-Mail` | 15.14% | 0.88% |


![Visit Rate by Experimental Group](<../week_3/visit rate by group.png>)

![Conversion Rate by Experimental Group](<../week_3/conversion rate by group.png>)


Kết quả ban đầu cho thấy cả `Mens E-Mail` và `Womens E-Mail` đều có positive average uplift so với `No E-Mail`. Nhìn chung, Mens E-Mail có raw uplift cao hơn Womens E-Mail.

Tuy nhiên, kết quả trung bình chưa cho biết treatment effect có giống nhau giữa các nhóm khách hàng hay không. Vì vậy, bước tiếp theo của EDA kiểm tra raw uplift theo `recency`, `purchase history`, `historical spending`, `geographic area` và `channel`.

---

Theo `recency`, một số mốc như 6, 9 và 12 tháng có uplift cao hơn và khoảng cách giữa hai campaign cũng thay đổi theo từng thời điểm. Nhưng các pattern này chưa đủ rõ để kết luận recency tạo treatment-effect heterogeneity ổn định.

![Customer Recency](<../week_3/recency.png>)

Pattern rõ nhất xuất hiện ở **purchase history**. Khách hàng chỉ từng mua Mens products có raw uplift cao hơn với Mens E-Mail; nhóm từng mua Womens products có raw uplift khá tương đồng giữa hai campaign; còn nhóm từng mua cả hai loại sản phẩm có raw uplift đặc biệt cao với Mens E-Mail.

![Combine Product](<../week_3/combine product.png>)

Raw uplift cũng thay đổi giữa các nhóm `historical spending`, trong đó nhóm `$500–750` nổi bật ở cả hai campaign. Khi xem thêm `recency` trong từng spending group, subgroup `$500–750 × recency 6` có raw uplift tương đối cao cho cả Mens và Womens E-Mail.

Tuy nhiên, các high-spending subgroup có ít observation hơn và uplift dao động khá mạnh, nên các extreme value ở vùng này là pattern ổn định.

![History Spending](<../week_3/history spending.png>)


---

Các pattern quan sát được kiểm tra bằng hypothesis testing:

| Hypothesis | Kết quả | Kết luận |
|---|---|---|
| **H1 — Purchase History** | `p < 0.001` | Reject H₀. Có evidence treatment effect thay đổi theo purchase history |
| **H2 — Recency + Historical Spending** | `p = 0.2549` | Fail to reject H₀. Chưa đủ evidence về systematic treatment-effect variation |
| **H3 — `$500–750 × recency 6`** | Mens: `+7.63 pp`, `p = 0.1248`; Womens: `+10.62 pp`, `p = 0.0260` | Chỉ Womens contrast đạt ngưỡng `α = 0.05` |
| **H4 — Geographic Area** | `p = 0.3077` | Fail to reject H₀. Chưa đủ evidence về systematic geographic heterogeneity |

### Conclusion

EDA cho thấy raw uplift thay đổi theo một số đặc điểm khách hàng, nhưng hypothesis testing không hỗ trợ tất cả các pattern quan sát được.

`Purchase history` là yếu tố có evidence rõ nhất về treatment-effect heterogeneity (`p < 0.001`). Ngược lại, các test cho `recency`, `historical spending` và `geographic area` chưa cung cấp đủ evidence về systematic treatment-effect variation.

Subgroup `$500–750 × recency 6` cũng đáng chú ý trong exploratory analysis. So với các khách hàng còn lại, treatment-effect difference ước lượng là `+7.63 pp` cho Mens E-Mail (`p = 0.1248`) và `+10.62 pp` cho Womens E-Mail (`p = 0.0260`). Chỉ Womens contrast đạt ngưỡng `α = 0.05`.

Nhìn chung, `purchase history` là customer characteristic đáng chú ý nhất để tiếp tục theo dõi trong modeling. Các pattern còn lại được xem là exploratory thay vì dùng trực tiếp để đưa ra quyết định targeting.

Một hạn chế quan trọng là `conversion` rất hiếm, điều này có thể làm các treatment-effect estimate cho outcome này kém ổn định hơn khi chuyển sang modeling.

---

### Modeling Config

Hillstrom có quy mô nhỏ hơn nhiều so với dataset lớn như Criteo, nên experiment sử dụng modeling config riêng:

```yaml
training:
  prediction_batch_size: 10000
  early_stopping_rounds: 100

models:
  model_defaults:
    objective: binary
    boosting_type: gbdt
    n_estimators: 5000
    learning_rate: 0.03
    num_leaves: 31
    max_depth: 6
    min_child_samples: 200
    subsample: 0.8
    subsample_freq: 1
    colsample_bytree: 0.8
    reg_alpha: 0.1
    reg_lambda: 1.0
    random_state: 42

selection:
  primary_split: validation
  primary_budget_fraction: 0.20
  primary_metric: policy_value
```

Tree complexity được giữ ở mức vừa phải với `num_leaves = 31`, `max_depth = 6` và `min_child_samples = 200`. Primary selection budget được đặt ở **20%**, tương ứng khoảng **1.7k khách hàng được chọn trên validation** cho mỗi experiment.

Ba candidate được train cho từng outcome:

| Model | Vai trò |
|---|---|
| `treated_response_lgbm` | Response baseline |
| `t_learner_lgbm` | Uplift candidate |
| `x_learner_lgbm` | Uplift candidate |

---

### Model Training

Validation set của hai experiment có quy mô gần như giống nhau nhưng số positive giữa hai outcome rất khác:

| Experiment | Outcome | Validation rows | Positive rate | Positive cases |
|---|---|---:|---:|---:|
| Mens | `conversion` | 8,523 | 0.915% | 78 |
| Mens | `visit` | 8,523 | 14.44% | 1,231 |
| Womens | `conversion` | 8,539 | 0.726% | 62 |
| Womens | `visit` | 8,539 | 12.88% | 1,100 |

Response Model training diagnostics:

| Experiment | Outcome | ROC-AUC | Average Precision |
|---|---|---:|---:|
| Mens | `conversion` | 0.580 | 0.0145 |
| Womens | `conversion` | 0.592 | 0.0099 |
| Mens | `visit` | 0.628 | 0.2093 |
| Womens | `visit` | 0.621 | 0.1825 |


Điểm đáng chú ý là `conversion` chỉ có **78 positive cases ở Mens** và **62 ở Womens**, trong khi `visit` có hơn một nghìn positive cases. Vì vậy, việc ước lượng treatment effect và so sánh policy cho `conversion` dựa trên ít observed outcomes hơn đáng kể.

---

### Validation Evaluation: Qini và AUUC

#### Mens — Conversion

| Model | AUUC | Qini |
|---|---:|---:|
| `t_learner_lgbm` | **19.446** | **4.449** |
| `treated_response_lgbm` | 17.786 | 2.788 |
| `x_learner_lgbm` | 18.657 | 3.660 |

![Mens Conversion Qini Curve](../../artifacts/figures/hillstrom_mens_conversion_conversion_t_learner_lgbm_vs_conversion_treated_response_lgbm_vs_conversion_x_learner_lgbm_run01_qini_curve.png)


![Mens Conversion Uplift Curve](../../artifacts/figures/hillstrom_mens_conversion_conversion_t_learner_lgbm_vs_conversion_treated_response_lgbm_vs_conversion_x_learner_lgbm_run01_uplift_curve.png)

T-Learner có AUUC và Qini cao nhất. Xét tổng thể trên toàn ranking curve, T-Learner có kết quả tốt hơn Response Model và X-Learner theo hai metric này.

---

#### Womens — Conversion

| Model | AUUC | Qini |
|---|---:|---:|
| `t_learner_lgbm` | **9.506** | **2.554** |
| `treated_response_lgbm` | 8.139 | 1.187 |
| `x_learner_lgbm` | 9.463 | 2.511 |


![Womens Conversion Qini Curve](../../artifacts/figures/hillstrom_womens_conversion_conversion_t_learner_lgbm_vs_conversion_treated_response_lgbm_vs_conversion_x_learner_lgbm_run01_qini_curve.png)


![Womens Conversion Uplift Curve](../../artifacts/figures/hillstrom_womens_conversion_conversion_t_learner_lgbm_vs_conversion_treated_response_lgbm_vs_conversion_x_learner_lgbm_run01_uplift_curve.png)


Các curve gợi ý một số khác biệt trong ranking giữa các model, nhưng kết quả ở vùng Top-K nhỏ vẫn dao động khá mạnh.

Với chỉ **62 positive conversions trên validation**, nên chưa xem chênh lệch Qini/AUUC là đủ để kết luận uplift learner tốt hơn baseline.

---

#### Mens — Visit

| Model | AUUC | Qini |
|---|---:|---:|
| `t_learner_lgbm` | 159.916 | -3.531 |
| `treated_response_lgbm` | **175.328** | **11.881** |
| `x_learner_lgbm` | 161.625 | -1.822 |


![Mens Visit Qini Curve](../../artifacts/figures/hillstrom_mens_visit_visit_t_learner_lgbm_vs_visit_treated_response_lgbm_vs_visit_x_learner_lgbm_run01_qini_curve.png)


![Mens Visit Uplift Curve](../../artifacts/figures/hillstrom_mens_visit_visit_t_learner_lgbm_vs_visit_treated_response_lgbm_vs_visit_x_learner_lgbm_run01_uplift_curve.png)

Xét tổng thể trên validation curve, ranking của Response Model tạo cumulative incremental outcome tốt hơn ranking của T-Learner và X-Learner.

Uplift Curve của cả ba model vẫn dương sau vùng Top-K đầu, tức campaign có tác động trên các nhóm được chọn. Tuy nhiên, T/X chưa sắp xếp được những khách hàng có treatment response cao lên phía trước tốt hơn Response Model.

---

#### Womens — Visit

| Model | AUUC | Qini |
|---|---:|---:|
| `t_learner_lgbm` | 132.998 | 35.900 |
| `treated_response_lgbm` | 118.839 | 21.741 |
| `x_learner_lgbm` | **134.315** | **37.217** |

![Womens Visit Qini Curve](../../artifacts/figures/hillstrom_womens_visit_visit_t_learner_lgbm_vs_visit_treated_response_lgbm_vs_visit_x_learner_lgbm_run01_qini_curve.png)


![Womens Visit Uplift Curve](../../artifacts/figures/hillstrom_womens_visit_visit_t_learner_lgbm_vs_visit_treated_response_lgbm_vs_visit_x_learner_lgbm_run01_uplift_curve.png)

Womens visit lại cho pattern ngược với Mens. T-Learner và X-Learner đều có AUUC/Qini cao hơn Response Model, trong đó X-Learner đứng đầu. Trên Qini Curve, hai uplift learner tách lên khá rõ ở phần giữa population, cho thấy chúng sắp xếp được một số nhóm khách hàng có incremental response cao lên trước baseline.

Uplift Curve cũng ổn định hơn `conversion`: sau vùng đầu dao động, uplift duy trì dương trên phần lớn population. `visit` cũng có khoảng **1,100 positive cases** trên validation, nhiều hơn đáng kể so với `conversion`, nên các treatment-effect estimate dựa trên nhiều observed outcomes hơn.

Trong bốn experiment–outcome combinations, `Womens visit` cho thấy lợi thế rõ nhất của uplift learners khi xét Qini/AUUC trên toàn curve. Tuy nhiên, primary selection của experiment diễn ra tại **Top 20%**, nên kết quả toàn curve chưa quyết định champion.

---

### Top-20% Policy Evaluation

| Experiment | Outcome | T-Learner | Response Model | X-Learner | Best point estimate |
|---|---|---:|---:|---:|---|
| Mens | `conversion` | **22.607** | 20.965 | 21.361 | T-Learner |
| Womens | `conversion` | **1.909** | -3.775 | 1.483 | T-Learner |
| Mens | `visit` | 157.279 | **177.874** | 161.878 | Response Model |
| Womens | `visit` | 142.914 | **166.871** | 136.673 | Response Model |

Các giá trị trong bảng là **estimated incremental outcome** trên validation tại Top 20%.

Với `conversion`, T-Learner có point estimate tốt nhất ở cả Mens và Womens. Tuy nhiên số conversion quan sát được quá ít nên chênh lệch point estimate chưa đủ để chọn model.

Với `visit`, Response Model tạo incremental outcome cao nhất tại Top 20% ở cả hai experiment. Đáng chú ý nhất là Womens: X-Learner có Qini tốt nhất trên toàn curve nhưng không thắng tại đúng operating point 20%. Điều này cho thấy **model tốt trên toàn ranking chưa chắc là model tốt nhất tại budget deployment cụ thể**.

---

### Bootstrap và Selection Gate

Selection Gate không chọn champion trực tiếp từ point estimate. Framework dùng paired bootstrap trên validation và so candidate với `treated_response_lgbm`.

Điều kiện pass:

```text
ci_lower > 0
```

Kết quả tại primary budget 20%:

| Experiment | Outcome | Candidate | Mean Δ policy value | 95% CI | Gate |
|---|---|---|---:|---:|---|
| Mens | `conversion` | T-Learner | +0.000331 | [-0.000836, 0.001532] | Failed |
| Mens | `conversion` | X-Learner | +0.000066 | [-0.001069, 0.001408] | Failed |
| Mens | `visit` | T-Learner | -0.003730 | [-0.011974, 0.005253] | Failed |
| Mens | `visit` | X-Learner | -0.000929 | [-0.008207, 0.005315] | Failed |
| Womens | `conversion` | T-Learner | +0.000677 | [-0.000355, 0.001647] | Failed |
| Womens | `conversion` | X-Learner | +0.000558 | [-0.000466, 0.001760] | Failed |
| Womens | `visit` | T-Learner | -0.000033 | [-0.008631, 0.007723] | Failed |
| Womens | `visit` | X-Learner | +0.000305 | [-0.007136, 0.008921] | Failed |

Không candidate nào có 95% confidence interval hoàn toàn lớn hơn 0, nên `treated_response_lgbm` được giữ làm champion cho cả bốn experiment.

Với `visit`, Response Model đã có point estimate cạnh tranh hoặc cao nhất tại Top 20%. Với `conversion`, T-Learner và X-Learner có một số point estimate cao hơn baseline, nhưng bootstrap interval vẫn chứa 0 nên lợi thế đó chưa đủ ổn định để thay champion.

---

### Locked Test Evaluation

Sau khi champion được chọn trên validation, locked test chỉ đánh giá exact selected Response Model. Test không được dùng để chọn lại model.

| Experiment | Outcome | Test Qini | Top-20% incremental outcome | Bootstrap mean | 95% CI |
|---|---|---:|---:|---:|---:|
| Mens | `conversion` | 0.733 | 11.642 | 12.171 | [-1.274, 27.925] |
| Womens | `conversion` | 3.018 | 13.117 | 12.906 | [-2.705, 29.989] |
| Mens | `visit` | 5.816 | 164.616 | 161.490 | **[104.522, 233.457]** |
| Womens | `visit` | 13.304 | 130.000 | 135.916 | **[84.528, 192.438]** |

Với `visit`, selected Response Model tiếp tục có positive estimated incremental outcome trên dữ liệu chưa được dùng để selection. Ở Top 20%, Mens đạt khoảng **164.6 incremental visits** và Womens khoảng **130.0**; 95% bootstrap interval của cả hai đều hoàn toàn lớn hơn 0. Kết quả này cho thấy selected champion vẫn tạo positive incremental outcome trên locked test.

Với `conversion`, selected champion cũng có positive point estimate trên test, nhưng 95% confidence interval vẫn chứa 0 ở cả Mens và Womens. Với dữ liệu hiện tại, chưa có đủ evidence rằng incremental conversion của policy ổn định khác 0.

---

### Results

Với Hillstrom, cả bốn experiment đều chọn `treated_response_lgbm` làm champion tại budget 20%.

Với `conversion`, một hạn chế rõ ràng là **số positive rất ít**: validation chỉ có 62–78 conversions. Các uplift candidate có lúc cao hơn baseline về Qini hoặc point estimate, nhưng bootstrap vẫn cho confidence interval chứa 0. Vì vậy, experiment hiện chưa cung cấp đủ evidence để thay Response Model bằng T-Learner hoặc X-Learner.

Với `visit`, outcome rarity ít nghiêm trọng hơn `conversion` vì validation có hơn một nghìn positive cases. EDA cho thấy prior product affinity là nguồn treatment-effect variation rõ nhất, trong khi recency, spending và geography chưa có evidence thống kê rõ ràng.

Kết quả modeling cũng không đồng nhất giữa Mens và Womens. Với Mens, Response Model tốt hơn các uplift learners cả trên whole-curve metrics và tại Top 20%. Với Womens, T/X-Learner tốt hơn Response Model về Qini/AUUC trên toàn curve, nhưng lợi thế đó không xuất hiện tại primary budget 20%. Bootstrap sau đó cũng không cho thấy candidate nào cải thiện ổn định so với baseline.

So với Criteo, Hillstrom cho một kết quả selection khác dù sử dụng cùng logic đánh giá. Ở Criteo, `visit` chọn X-Learner làm champion; trong khi ở Hillstrom, cả bốn experiment đều giữ Response Model.

Điểm quan trọng ở đây không phải framework luôn phải chọn uplift model, mà là candidate chỉ thay baseline khi chứng minh được improvement đủ ổn định theo cùng selection rule.

Qua Criteo và Hillstrom, tôi đã chạy cùng workflow:

`train → validation evaluation → bootstrap → selection → locked test`

trên hai dataset có quy mô và đặc điểm khác nhau mà không thay đổi logic đánh giá cốt lõi.

Kết quả này cho thấy framework hiện có thể được reuse cho một prepared uplift dataset mới và vẫn áp dụng cùng nguyên tắc model selection, thay vì gắn logic đánh giá với riêng Criteo hoặc một loại model cụ thể.

Bước tiếp theo là thử framework trên một dataset có nhiều feature và cấu trúc treatment effect phức tạp hơn để tiếp tục kiểm tra khả năng reuse và độ ổn định của workflow.

---

### Experiment 004 - Metrics Summary

#### Model Training

| Experiment | Outcome | Validation rows | Positive rate | Positive cases |
|---|---|---:|---:|---:|
| Mens | `conversion` | 6,392 | 0.923% | 59 |
| Mens | `visit` | 6,392 | 14.456% | 924 |
| Womens | `conversion` | 6,404 | 0.750% | 48 |
| Womens | `visit` | 6,404 | 12.883% | 825 |

Response Model training diagnostics:

| Experiment | Outcome | ROC-AUC | Average Precision |
|---|---|---:|---:|
| Mens | `conversion` | 0.631 | 0.0145 |
| Womens | `conversion` | 0.600 | 0.0107 |
| Mens | `visit` | 0.632 | 0.2190 |
| Womens | `visit` | 0.620 | 0.1801 |

#### Validation Evaluation: Qini và AUUC

| Experiment | Outcome | Model | AUUC | Qini |
|---|---|---|---:|---:|
| Mens | `conversion` | `t_learner_lgbm` | 10.568 | 0.068 |
| Mens | `conversion` | `treated_response_lgbm` | 11.773 | 1.273 |
| Mens | `conversion` | `x_learner_lgbm` | 11.993 | 1.493 |
| Womens | `conversion` | `t_learner_lgbm` | 9.049 | 4.085 |
| Womens | `conversion` | `treated_response_lgbm` | 8.896 | 3.932 |
| Womens | `conversion` | `x_learner_lgbm` | 9.355 | 4.391 |
| Mens | `visit` | `t_learner_lgbm` | 128.038 | 6.038 |
| Mens | `visit` | `treated_response_lgbm` | 125.490 | 3.490 |
| Mens | `visit` | `x_learner_lgbm` | 122.097 | 0.097 |
| Womens | `visit` | `t_learner_lgbm` | 85.608 | 13.640 |
| Womens | `visit` | `treated_response_lgbm` | 80.685 | 8.717 |
| Womens | `visit` | `x_learner_lgbm` | 95.291 | 23.323 |

#### Top-20% Policy Evaluation

| Experiment | Outcome | T-Learner | Response Model | X-Learner | Best point estimate |
|---|---|---:|---:|---:|---|
| Mens | `conversion` | 7.088 | 8.659 | 13.360 | X-Learner |
| Womens | `conversion` | 10.023 | 10.102 | 8.169 | Response Model |
| Mens | `visit` | 119.185 | 108.031 | 104.368 | T-Learner |
| Womens | `visit` | 110.792 | 76.835 | 128.777 | X-Learner |

#### Bootstrap và Selection Gate

| Experiment | Outcome | Candidate | Mean Δ policy value | 95% CI | Gate |
|---|---|---|---:|---:|---|
| Mens | `conversion` | T-Learner | -0.000509 | [-0.002670, 0.001860] | Failed |
| Mens | `conversion` | X-Learner | +0.000142 | [-0.001419, 0.001559] | Failed |
| Mens | `visit` | T-Learner | +0.002677 | [-0.005861, 0.010290] | Failed |
| Mens | `visit` | X-Learner | +0.000169 | [-0.006399, 0.006612] | Failed |
| Womens | `conversion` | T-Learner | -0.000013 | [-0.000633, 0.000926] | Failed |
| Womens | `conversion` | X-Learner | -0.000540 | [-0.002172, 0.000926] | Failed |
| Womens | `visit` | T-Learner | +0.004762 | [-0.004441, 0.015202] | Failed |
| Womens | `visit` | X-Learner | +0.009522 | [-0.000201, 0.019971] | Failed |

| Experiment | Outcome | Champion |
|---|---|---|
| Mens | `conversion` | `treated_response_lgbm` |
| Womens | `conversion` | `treated_response_lgbm` |
| Mens | `visit` | `treated_response_lgbm` |
| Womens | `visit` | `treated_response_lgbm` |

#### Locked Test Evaluation

| Experiment | Outcome | Test Qini | Top-20% incremental outcome | Bootstrap mean | 95% CI |
|---|---|---:|---:|---:|---:|
| Mens | `conversion` | 3.555 | 18.102 | 16.938 | [5.095, 29.629] |
| Womens | `conversion` | 1.092 | 1.366 | 1.109 | [-10.708, 14.990] |
| Mens | `visit` | 7.163 | 139.413 | 135.773 | [78.872, 184.539] |
| Womens | `visit` | 17.821 | 131.083 | 127.189 | [77.056, 189.841] |

