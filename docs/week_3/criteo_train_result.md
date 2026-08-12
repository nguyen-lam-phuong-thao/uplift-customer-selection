## Experiments

### Dataset và mục tiêu thí nghiệm

Dataset được sử dụng là **Criteo Uplift Modeling dataset**. Mục tiêu của thí nghiệm là xây dựng một workflow Uplift Modeling hoàn chỉnh để chọn nhóm khách hàng nên được target trong chiến dịch marketing.

Dataset có các thành phần chính:

| Thành phần | Ý nghĩa |
|---|---|
| `f0` → `f11` | Các đặc trưng khách hàng |
| `treatment` | Khách hàng có nhận tác động marketing hay không |
| `visit` | Outcome: khách hàng có truy cập hay không |
| `conversion` | Outcome: khách hàng có chuyển đổi hay không |

Bài toán không chỉ là dự đoán ai có khả năng `visit` hoặc `conversion` cao, mà là:

> Nếu chỉ có ngân sách target một phần khách hàng, model nào chọn được nhóm tạo ra nhiều incremental outcome nhất?

---

### Data Understanding

#### Quy mô Treatment và Control

![Treatment and control population sizes](<../week_1/treatment and control poplation.png>)

Biểu đồ cho thấy dữ liệu bị lệch mạnh giữa hai nhóm. Nhóm **Treatment (1)** có khoảng **11.9 triệu dòng**, trong khi nhóm **Control (0)** có khoảng **2.1 triệu dòng**.

| Nhóm | Quy mô xấp xỉ | Nhận xét |
|---|---:|---|
| Control (0) | ~2.1M | Nhóm không nhận ads, dùng làm đối chứng |
| Treatment (1) | ~11.9M | Nhóm nhận ads |
| Treatment rate | ~85% | Treatment chiếm phần lớn dữ liệu |

Treatment chiếm khoảng 85% dữ liệu, trong khi Control chiếm khoảng 15%. Đây là phân phối treatment quan sát được trong dataset. Control cung cấp nhóm đối chứng để ước lượng phần outcome tăng thêm liên quan đến treatment.

---

#### Outcome rate theo Treatment và Control

![Outcome rates by treatment group](<../week_1/outcome rates by treatment group.png>)

Biểu đồ cho thấy cả hai outcome đều cao hơn ở nhóm Treatment so với Control.

| Outcome | Control rate | Treatment rate | Nhận xét |
|---|---:|---:|---|
| `visit` | ~3.8% | ~4.8% | Treatment có visit rate cao hơn Control |
| `conversion` | ~0.18% | ~0.30% | Treatment có conversion rate cao hơn Control, nhưng conversion rất hiếm |

`conversion` hiếm hơn `visit` rất nhiều. Vì số positive conversion ít, các ước lượng uplift cho outcome này có thể dao động nhiều hơn.

Tuy nhiên, outcome rate trung bình mới chỉ cho thấy sự khác biệt giữa Treatment và Control trên toàn bộ tập khách hàng. Nó chưa trả lời được câu hỏi quan trọng hơn:

> Nên target khách hàng nào trước?

Đó là câu hỏi mà phần Uplift Modeling phía sau cần giải quyết.

---

#### Khác biệt feature giữa Treatment và Control

![Feature mean differences](<../week_1/feature mean differences.png>)

Biểu đồ đo chênh lệch trung bình feature giữa hai nhóm: 

    `mean(feature | treatment = 1) - mean(feature | treatment = 0)`.

| Feature | Xu hướng | Nhận xét |
|---|---:|---|
| `f9` | Lệch dương mạnh | Treatment group có mean cao hơn Control rõ rệt |
| `f6` | Lệch âm mạnh | Control group có mean cao hơn Treatment rõ rệt |
| `f3`, `f0` | Lệch âm vừa | Có khác biệt đáng chú ý |
| Các feature còn lại | Gần 0 | Hai nhóm tương đối gần nhau |

Điều này cho thấy Treatment và Control không hoàn toàn giống nhau theo mọi feature. Vì vậy, nếu chỉ so outcome trung bình giữa hai nhóm thì có thể bị nhiễu bởi khác biệt sẵn có trong dữ liệu. Model cần học ranking dựa trên feature để tìm nhóm khách hàng có incremental effect cao nhất.

---

### Train/Validation/Test split

Dữ liệu sau preprocessing được chia thành train, validation và locked test.

| Split | Vai trò |
|---|---|
| Train | Dùng để train model |
| Validation | Dùng để evaluate, bootstrap và chọn champion |
| Locked test | Chỉ dùng sau khi champion đã được chọn |

Thống kê validation và locked test:

| Outcome | Validation rows | Validation positive rate | Validation treatment rate | Locked test rows | Locked test positive rate | Locked test treatment rate |
|---|---:|---:|---:|---:|---:|---:|
| `conversion` | 2,795,918 | 0.002916 | 0.850000 | 2,795,919 | 0.002917 | 0.850000 |
| `visit` | 2,795,918 | 0.046992 | 0.850000 | 2,795,919 | 0.046992 | 0.850000 |

Validation và locked test có phân phối gần như giống nhau. Treatment rate đều giữ ở mức khoảng **85%**, đúng với phân phối gốc. Điều này giúp kết quả validation và locked test có thể so sánh được.

---

### Model Training

Framework train ba policy chính cho từng outcome.

| Model | Loại model | Vai trò |
|---|---|---|
| `treated_response_lgbm` | Response model | Baseline truyền thống, rank khách hàng theo xác suất outcome |
| `t_learner_lgbm` | Uplift model | Ước lượng uplift bằng chênh lệch giữa treatment model và control model |
| `x_learner_lgbm` | Uplift model | Ước lượng treatment effect bằng cơ chế imputed treatment effect |

Response model có thêm classification metrics sau training:

| Outcome      | Model                   |  ROC-AUC | Average Precision | Log Loss | Positive rate |
| ------------ | ----------------------- | -------: | ----------------: | -------: | ------------: |
| `conversion` | `treated_response_lgbm` | 0.960246 |          0.230235 | 0.011696 |      0.002916 |
| `visit`      | `treated_response_lgbm` | 0.946741 |          0.520291 | 0.102875 |      0.046992 |

Response Model có khả năng xếp hạng khá tốt những khách hàng dễ xảy ra outcome, với ROC-AUC khoảng **0.96 cho conversion** và **0.95 cho visit**.

Conversion là outcome rất hiếm, chỉ khoảng **0.29%**, nhưng Average Precision đạt **0.23**, cho thấy model vẫn tìm được nhóm khách hàng có khả năng conversion cao hơn đáng kể so với mức trung bình. Với visit, positive rate khoảng **4.7%** và Average Precision đạt **0.52**, nên model nhận diện nhóm dễ visit tốt hơn.


Champion selection phía sau dựa trên:

| Mục đích | Metric dùng |
|---|---|
| So policy theo budget | `policy_value`, `incremental_outcome` |
| Kiểm tra độ ổn định | Paired bootstrap confidence interval |
| Chọn champion | Selection Gate trên validation |
| Báo cáo cuối | Locked-test policy value, incremental outcome, Qini/AUUC |

--- 

### Validation Evaluation: Qini và AUUC

#### <b>Conversion</b>

![Conversion Qini Curve](<../../artifacts/figures/criteo_conversion_t_learner_lgbm_vs_treated_response_lgbm_vs_x_learner_lgbm_run01_qini_curve.png>)


| Model | AUUC | Qini | Rank theo Qini |
|---|---:|---:|---:|
| `treated_response_lgbm` | 2,458.966 | 1,088.637 | 1 |
| `t_learner_lgbm` | 2,197.565 | 827.236 | 2 |
| `x_learner_lgbm` | 2,099.079 | 728.750 | 3 |


![Conversion Uplift Curve](<../../artifacts/figures/criteo_conversion_t_learner_lgbm_vs_treated_response_lgbm_vs_x_learner_lgbm_run01_uplift_curve.png>)

Với `conversion`, Response Model có AUUC và Qini cao nhất. Điều này cho thấy khi mở rộng dần danh sách khách hàng theo thứ tự score, Response Model tạo ra incremental outcome tích lũy tốt hơn T-Learner và X-Learner trên toàn bộ đường cong.

Tuy nhiên, Uplift Curve tăng mạnh ở phần đầu rồi giảm nhanh. Do conversion rất hiếm, kết quả tại các Top-k nhỏ có thể dao động lớn vì chỉ dựa trên số lượng positive rất ít. Vì vậy, ngoài kết quả tổng thể từ Qini và AUUC, cần xem thêm khoảng tin cậy bootstrap tại các mức Top-k quan trọng để kiểm tra lợi thế của Response Model có ổn định hay không.



---

#### <b>Visit</b>

![Visit Qini Curve](<../../artifacts/figures/criteo_visit_t_learner_lgbm_vs_treated_response_lgbm_vs_x_learner_lgbm_run01_qini_curve.png>)

| Model | AUUC | Qini | Rank theo Qini |
|---|---:|---:|---:|
| `t_learner_lgbm` | 20,868.813 | 8,579.236 | 1 |
| `treated_response_lgbm` | 20,815.119 | 8,525.542 | 2 |
| `x_learner_lgbm` | 20,615.580 | 8,326.002 | 3 |


![Visit Uplift Curve](<../../artifacts/figures/criteo_visit_t_learner_lgbm_vs_treated_response_lgbm_vs_x_learner_lgbm_run01_uplift_curve.png>)

Với `visit`, T-Learner có AUUC và Qini cao nhất. Điều này cho thấy khi mở rộng dần danh sách khách hàng theo thứ tự score, T-Learner tạo ra incremental outcome tích lũy tốt hơn hai model còn lại.

Uplift Curve cũng cho thấy nhóm khách hàng ở đầu bảng xếp hạng có uplift cao nhất. Khi mở rộng danh sách, uplift rate giảm dần vì model phải lấy thêm những khách hàng có mức phản ứng thấp hơn.

Tuy nhiên, chênh lệch AUUC và Qini giữa ba model không lớn. Vì vậy, cần xem thêm bootstrap tại các mức Top-k quan trọng để kiểm tra lợi thế của T-Learner có đủ ổn định hay không.

---

### Top-K Policy Evaluation

Top-K evaluation mô phỏng bài toán business: nếu chỉ có ngân sách để target top 1%, 5%, 10%, 20% hoặc 30% khách hàng, policy nào tạo nhiều incremental outcome hơn?

#### Conversion

| Budget | Random | T-Learner | Response model | X-Learner | Best policy |
|---:|---:|---:|---:|---:|---|
| 1% | 14.992 | 764.132 | 887.518 | 860.220 | Response model |
| 5% | 44.808 | 1,619.349 | 1,733.937 | 1,734.434 | X-Learner |
| 10% | 288.049 | 2,017.899 | 2,221.671 | 2,134.577 | Response model |
| 20% | 595.940 | 2,430.029 | 2,688.386 | 2,607.969 | Response model |
| 30% | 829.311 | 2,589.204 | 2,912.551 | 2,812.711 | Response model |

Với `conversion`, các model học được đều tốt hơn random targeting. Tuy nhiên, response model thắng ở 4/5 budget. Tại budget 5%, X-Learner nhỉnh hơn response model nhưng chỉ hơn khoảng **0.497 incremental conversions**, tức là rất nhỏ.

Do đó, chưa thể kết luận X-Learner tốt hơn nếu chỉ nhìn point estimate tại 5%. Cần kiểm tra bootstrap.

---

#### Visit

| Budget | Random | T-Learner | Response model | X-Learner | Best policy |
|---:|---:|---:|---:|---:|---|
| 1% | 181.444 | 4,808.173 | 1,527.175 | 5,230.658 | X-Learner |
| 5% | 1,425.756 | 12,250.205 | 7,516.300 | 12,675.340 | X-Learner |
| 10% | 2,828.405 | 16,791.890 | 14,129.341 | 17,915.712 | X-Learner |
| 20% | 5,783.854 | 22,383.057 | 21,739.922 | 22,867.744 | X-Learner |
| 30% | 8,906.365 | 24,737.099 | 24,845.066 | 25,544.834 | X-Learner |

Với `visit`, X-Learner là policy tốt nhất ở toàn bộ budget từ 1% đến 30%.

Tại budget 5%, X-Learner tạo ra **12,675.340 incremental visits**, trong khi response model chỉ đạt **7,516.300**. Chênh lệch khoảng **5,159 incremental visits** trên cùng số lượng khách hàng được chọn.

Đây là bằng chứng mạnh cho thấy X-Learner phù hợp hơn response model khi mục tiêu là tăng visit.

---

### Bootstrap và Selection Gate

Selection Gate không chọn model bằng point estimate trực tiếp. Framework dùng paired bootstrap để kiểm tra chênh lệch giữa candidate và baseline có ổn định hay không.

Tiêu chí chọn: `ci_lower > 0`.

Tức là lower bound của confidence interval phải lớn hơn 0.

| Outcome | Candidate | Baseline | Budget | Mean Δ policy value | 95% CI | Gate result |
|---|---|---|---:|---:|---:|---|
| `conversion` | `t_learner_lgbm` | `treated_response_lgbm` | 5% | -0.000081 | [-0.000137, -0.000014] | Failed |
| `conversion` | `x_learner_lgbm` | `treated_response_lgbm` | 5% | -0.000010 | [-0.000049, 0.000031] | Failed |
| `visit` | `t_learner_lgbm` | `treated_response_lgbm` | 5% | 0.001862 | [0.001561, 0.002193] | Passed |
| `visit` | `x_learner_lgbm` | `treated_response_lgbm` | 5% | 0.002111 | [0.001744, 0.002444] | Passed |

Với `conversion`, cả hai uplift learners đều không pass gate. X-Learner dù nhỉnh hơn response model ở Top-K 5% point estimate, nhưng confidence interval vẫn cắt qua 0. Vì vậy, framework không đủ bằng chứng để thay baseline.

Với `visit`, cả T-Learner và X-Learner đều pass gate. X-Learner có mean delta lớn hơn, nên được chọn làm champion.

---

### Champion Selection

| Outcome | Selected champion | Lý do |
|---|---|---|
| `conversion` | `treated_response_lgbm` | Không uplift candidate nào pass `ci_lower > 0`, nên giữ baseline |
| `visit` | `x_learner_lgbm` | Pass bootstrap gate và có mean delta lớn nhất tại budget 5% |

Kết quả này cho thấy framework không tự động chọn uplift model chỉ vì đó là uplift model. Nếu uplift model không chứng minh được lợi ích ổn định, hệ thống giữ lại response model.

---

### Locked Test Evaluation

Sau khi chọn champion trên validation, locked test chỉ dùng để đánh giá final champion. Locked test không được dùng để chọn lại model.

#### Conversion champion: `treated_response_lgbm`

| Budget | Selected rows | Policy value | Incremental outcome |
|---:|---:|---:|---:|
| 1% | 27,960 | 0.002516 | 1,093.690 |
| 5% | 139,796 | 0.002885 | 1,900.676 |
| 10% | 279,592 | 0.002961 | 2,267.996 |
| 20% | 559,184 | 0.003021 | 2,681.238 |
| 30% | 838,776 | 0.003058 | 2,934.199 |

Trên locked test, conversion champion đạt **1,900.676 incremental conversions** ở budget 5% và **2,934.199 incremental conversions** ở budget 30%.

| Metric | Validation | Locked test | Difference |
|---|---:|---:|---:|
| AUUC | 2,458.966 | 2,479.210 | +20.243 |
| Qini | 1,088.637 | 1,111.709 | +23.071 |
| Policy value | 0.003042 | 0.003058 | +0.000016 |

Kết quả test không cho thấy degradation. Policy value gần như giữ nguyên, còn AUUC và Qini tăng nhẹ.

---

#### Visit champion: `x_learner_lgbm`

| Budget | Selected rows | Policy value | Incremental outcome |
|---:|---:|---:|---:|
| 1% | 27,960 | 0.041429 | 4,847.578 |
| 5% | 139,796 | 0.045762 | 12,244.809 |
| 10% | 279,592 | 0.047270 | 17,113.218 |
| 20% | 559,184 | 0.048011 | 21,895.526 |
| 30% | 838,776 | 0.048229 | 24,612.025 |

Trên locked test, visit champion đạt **12,244.809 incremental visits** ở budget 5% và **24,612.025 incremental visits** ở budget 30%.

| Metric | Validation | Locked test | Difference |
|---|---:|---:|---:|
| AUUC | 20,615.580 | 20,269.087 | -346.493 |
| Qini | 8,326.002 | 7,979.401 | -346.601 |
| Policy value | 0.048320 | 0.048229 | -0.000091 |

AUUC và Qini giảm nhẹ trên test, nhưng policy value gần như không đổi. Với mục tiêu deployment theo budget, đây là kết quả ổn định.

---

### Kết luận

Kết quả cuối cùng của experiment:

| Outcome | Champion | Kết luận |
|---|---|---|
| `conversion` | `treated_response_lgbm` | Conversion quá hiếm, uplift learners không vượt baseline ổn định, nên giữ response model |
| `visit` | `x_learner_lgbm` | X-Learner thắng Top-K ở mọi budget, pass bootstrap gate, và giữ policy value ổn định trên locked test |

Kết quả cho thấy thiết kế framework ảnh hưởng trực tiếp đến cách chọn model. Model không được chọn chỉ vì có ROC-AUC, Qini hay AUUC cao nhất, mà phải tạo ra incremental outcome tốt tại các mức budget thực tế và giữ được lợi thế qua bootstrap.

Nhờ đó, framework giúp tránh chọn model theo một metric đơn lẻ và đưa ra quyết định phù hợp hơn với mục tiêu triển khai. Champion cuối cùng là model tạo ra giá trị tăng thêm tốt và ổn định nhất, sau đó mới được xác nhận trên locked test.