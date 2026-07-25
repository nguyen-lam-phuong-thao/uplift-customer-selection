# Causal Inference Foundations for Uplift Modeling

## 1. Project objective

Trong marketing, Response Model thường được sử dụng để tìm những khách hàng có khả năng phản hồi cao.

Tuy nhiên, khách hàng có khả năng phản hồi cao chưa chắc phản hồi **do chiến dịch**. Họ có thể vẫn mua hàng hoặc truy cập website ngay cả khi không nhận quảng cáo.

Uplift Modeling tập trung trả lời câu hỏi:

> Khách hàng nào sẽ thay đổi hành vi do tác động của chiến dịch?

Mục tiêu của project là:

- Ước lượng treatment effect của khách hàng.
- Xếp hạng khách hàng theo uplift score.
- So sánh targeting bằng response probability và uplift score.
- Lựa chọn khách hàng mang lại incremental impact cao.
- Hỗ trợ phân bổ ngân sách marketing.

---

## 2. Potential outcomes và counterfactual

Với mỗi khách hàng $i$, tồn tại hai potential outcomes:

$$
Y_i(1)
$$

là outcome nếu khách hàng nhận treatment, và:

$$
Y_i(0)
$$

là outcome nếu khách hàng không nhận treatment.

Treatment effect của khách hàng được định nghĩa là:

$$
\tau_i = Y_i(1) - Y_i(0)
$$

Trong thực tế, mỗi khách hàng chỉ thuộc treatment hoặc control nên chỉ quan sát được một trong hai potential outcomes.

Outcome còn lại không được quan sát được gọi là **counterfactual**.

Do không thể đồng thời quan sát $Y_i(1)$ và $Y_i(0)$ của cùng một khách hàng, treatment effect thật ở cấp cá nhân không thể được xác định trực tiếp.

Model chỉ có thể ước lượng treatment effect dựa trên những khách hàng có đặc điểm tương tự nhau.

---

## 3. Treatment effect, ATE và CATE

### Individual Treatment Effect

Individual Treatment Effect của khách hàng $i$ là:

$$
\tau_i = Y_i(1) - Y_i(0)
$$

Do chỉ quan sát được một trong hai potential outcomes, ITE thật không thể được kiểm tra trực tiếp ở cấp từng khách hàng.

### Average Treatment Effect

Average Treatment Effect đo tác động trung bình của treatment trên toàn bộ population:

$$
ATE = \mathbb{E}[Y(1)-Y(0)]
$$

ATE trả lời câu hỏi:

> Trung bình, chiến dịch có làm outcome tăng hay giảm không?

### Conditional Average Treatment Effect

Conditional Average Treatment Effect đo treatment effect của những khách hàng có đặc điểm $X=x$:

$$
\tau(x)
=
\mathbb{E}[Y(1)-Y(0)\mid X=x]
$$

CATE trả lời câu hỏi:

> Với một nhóm khách hàng có đặc điểm cụ thể, treatment làm outcome thay đổi trung bình bao nhiêu?

Project tập trung vào CATE vì mục tiêu là xếp hạng và lựa chọn khách hàng.

Predicted CATE được sử dụng làm **uplift score**.

---

## 4. Response prediction và uplift prediction

### Response prediction

Response Model dự đoán:

$$
P(Y=1\mid T=1,X=x)
$$

Nó trả lời:

> Nếu nhận treatment, khách hàng nào có khả năng phản hồi cao?

Response probability không cho biết khách hàng có phản hồi do treatment hay vẫn phản hồi khi không có treatment.

### Uplift prediction

Uplift Model ước lượng:

$$
\tau(x)
=
P(Y=1\mid T=1,X=x)
-
P(Y=1\mid T=0,X=x)
$$

Nó trả lời:

> Treatment làm xác suất phản hồi của khách hàng tăng hoặc giảm bao nhiêu?

Ví dụ:

| Customer | Treatment response | Control response | Uplift |
|---|---:|---:|---:|
| A | 0.80 | 0.75 | 0.05 |
| B | 0.55 | 0.20 | 0.35 |

Response Model ưu tiên khách hàng A vì response probability cao hơn.

Uplift Model ưu tiên khách hàng B vì treatment tạo ra nhiều incremental impact hơn.

Kết luận:

> Khách hàng có response probability cao chưa chắc có treatment effect cao.

---

## 5. Models used in the project

### 5.1 Response Model

Response Model được train trên treatment group để dự đoán:

$$
\hat{\mu}_1(x)
=
\widehat{P}(Y=1\mid T=1,X=x)
$$

Model được sử dụng làm baseline cho chiến lược targeting truyền thống.

Quy trình:

```text
Treatment group
      ↓
Train response classifier
      ↓
Predict response probability
      ↓
Rank customers by response score
```

Hạn chế chính của Response Model là không ước lượng kết quả nếu khách hàng không nhận treatment.

Vì vậy, model không phân biệt được:

- Khách hàng phản hồi do treatment.
- Khách hàng vẫn phản hồi dù không có treatment.

---

### 5.2 T-Learner

T-Learner sử dụng hai model độc lập.

Treatment model:

$$
\hat{\mu}_1(x)
=
\widehat{\mathbb{E}}[Y\mid T=1,X=x]
$$

Control model:

$$
\hat{\mu}_0(x)
=
\widehat{\mathbb{E}}[Y\mid T=0,X=x]
$$

Với cùng một khách hàng, hai model lần lượt dự đoán outcome khi nhận và không nhận treatment.

Uplift score được tính bằng:

$$
\hat{\tau}_T(x)
=
\hat{\mu}_1(x)-\hat{\mu}_0(x)
$$

Quy trình:

```text
Treatment data → Train treatment model μ₁
Control data   → Train control model μ₀

Customer features X
        ↓
Predict Ŷ(1) và Ŷ(0)
        ↓
Uplift = Ŷ(1) - Ŷ(0)
```

T-Learner dễ triển khai nhưng hai model học độc lập.

Nếu treatment và control không cân bằng, model của nhóm nhỏ có thể học không tốt.

---

### 5.3 X-Learner

X-Learner mở rộng T-Learner bằng cách tạo pseudo-treatment effects và train thêm các treatment-effect models.

#### Bước 1: Train outcome models

Treatment outcome model:

$$
\hat{\mu}_1(x)
=
\widehat{\mathbb{E}}[Y\mid T=1,X=x]
$$

Control outcome model:

$$
\hat{\mu}_0(x)
=
\widehat{\mathbb{E}}[Y\mid T=0,X=x]
$$

#### Bước 2: Tạo pseudo-treatment effects

Với treatment group, observed outcome là outcome khi nhận treatment. Counterfactual được dự đoán bằng control model:

$$
D_i^1
=
Y_i-\hat{\mu}_0(X_i)
$$

Với control group, observed outcome là outcome khi không nhận treatment. Counterfactual được dự đoán bằng treatment model:

$$
D_i^0
=
\hat{\mu}_1(X_i)-Y_i
$$

#### Bước 3: Train treatment-effect models

Train effect model trên treatment group:

$$
\hat{\tau}_1(x)
=
\widehat{\mathbb{E}}[D^1\mid X=x,T=1]
$$

Train effect model trên control group:

$$
\hat{\tau}_0(x)
=
\widehat{\mathbb{E}}[D^0\mid X=x,T=0]
$$

#### Bước 4: Kết hợp kết quả

Hai treatment-effect estimates được kết hợp bằng một trọng số $g(x)$:

$$
\hat{\tau}_X(x)
=
g(x)\hat{\tau}_0(x)
+
[1-g(x)]\hat{\tau}_1(x)
$$

Trọng số $g(x)$ thường được xác định dựa trên propensity score:

$$
e(x)=P(T=1\mid X=x)
$$

Quy trình tổng quát:

```text
Train μ₁ và μ₀
      ↓
Dự đoán counterfactual
      ↓
Tạo pseudo-effects D¹ và D⁰
      ↓
Train τ₁ và τ₀
      ↓
Kết hợp thành final uplift score
```

X-Learner phức tạp hơn T-Learner nhưng có thể tận dụng dữ liệu tốt hơn khi treatment và control không cân bằng.

---

## 6. Model comparison

| Tiêu chí | Response Model | T-Learner | X-Learner |
|---|---|---|---|
| Mục tiêu | Dự đoán response | Ước lượng CATE | Ước lượng CATE |
| Output | Response probability | Uplift score | Uplift score |
| Treatment model | Có | Có | Có |
| Control model | Không | Có | Có |
| Pseudo-effect | Không | Không | Có |
| Độ phức tạp | Thấp | Trung bình | Cao |
| Vai trò | Business baseline | Uplift baseline | Advanced uplift model |
| Hạn chế chính | Không đo incremental effect | Hai model học độc lập | Phụ thuộc first-stage models |

---

## 7. Project assumptions

Assumptions là những điều kiện cần để uplift score có thể được diễn giải là treatment effect.


| Assumption | Ý nghĩa | Cách xử lý trong project |
|---|---|---|
| Exchangeability | Treatment và control có thể được so sánh công bằng | Dựa vào random assignment và kiểm tra feature balance |
| Positivity | Các nhóm khách hàng cần có khả năng xuất hiện ở cả treatment và control | Kiểm tra overlap và treatment rate theo feature |
| Consistency | Treatment và outcome phải được định nghĩa nhất quán | Đối chiếu dataset documentation |
| SUTVA / No interference | Treatment của một khách hàng không ảnh hưởng outcome của khách hàng khác | Ghi nhận như assumption và limitation |
| Pre-treatment features | Feature đầu vào phải tồn tại trước treatment | Loại bỏ treatment, outcome và post-treatment variables khỏi $X$ |
| Representativeness | Dữ liệu phải phù hợp với population mà kết quả được áp dụng | Giới hạn phạm vi kết luận của project |

### Exchangeability

Exchangeability yêu cầu treatment và control có thể được so sánh công bằng.

Trong Criteo, assumption này được hỗ trợ bởi random assignment. Tuy nhiên, project vẫn cần kiểm tra:

- Tỷ lệ treatment và control.
- Phân phối feature giữa hai nhóm.
- Missing rate giữa hai nhóm.
- Standardized Mean Difference của các feature.

### Positivity

Positivity yêu cầu mỗi loại khách hàng phải có khả năng xuất hiện ở cả treatment và control:

$$
0<P(T=1\mid X=x)<1
$$

Nếu một nhóm khách hàng chỉ có treatment mà không có control, model sẽ thiếu cơ sở để ước lượng counterfactual.

Project cần kiểm tra:

- Treatment rate theo feature.
- Số lượng treatment và control trong từng phân khúc.
- Các vùng dữ liệu không có overlap.
- Các nhóm quá hiếm.

### Pre-treatment features

Các feature dùng cho inference phải tồn tại trước thời điểm treatment được phân bổ.

Không sử dụng:

- Outcome sau campaign.
- Click hoặc conversion sau treatment.
- Feature được tính từ outcome.
- Post-treatment variables.
- Biến gây data leakage.

### Consistency

Treatment và outcome phải được định nghĩa giống nhau trong toàn bộ project.

Ví dụ:

- `treatment = 1` phải luôn đại diện cho cùng một trạng thái treatment.
- `treatment = 0` phải luôn đại diện cho control.
- Mỗi experiment chỉ sử dụng một outcome được xác định rõ ràng.

### SUTVA / No interference

SUTVA giả định treatment của một khách hàng không làm thay đổi outcome của khách hàng khác.

Assumption này không thể được chứng minh hoàn toàn bằng EDA và sẽ được ghi nhận như một limitation của project.

### Representativeness

Dataset phải đủ đại diện cho population mà model được áp dụng.

Do các feature của Criteo đã được ẩn danh, kết quả project phù hợp để:

- So sánh các targeting strategies.
- Đánh giá Uplift Modeling.
- Mô phỏng customer selection.
- Mô phỏng budget và business value.

Project không sử dụng các feature ẩn danh để đưa ra diễn giải business cụ thể về tuổi, khu vực hoặc hành vi khách hàng.

---

## 8. Variable definition for Criteo

Trong Criteo Dataset, các biến được sử dụng như sau:

```text
X = f0, f1, ..., f11
T = treatment
Y = visit hoặc conversion
```

Project sẽ chọn một outcome cho mỗi experiment, không sử dụng đồng thời `visit` và `conversion` làm cùng một target.

Các biến sau không được đưa vào feature matrix:

```text
treatment
visit
conversion
exposure
```

`treatment` là biến intervention.

`visit` và `conversion` là outcome variables.

`exposure` xảy ra sau treatment assignment nên không được xem là pre-treatment customer feature.

---

## 9. EDA checklist derived from assumptions

Trong giai đoạn EDA cần kiểm tra:

- Số dòng, số cột và data types.
- Missing values và duplicate observations.
- Số lượng và tỷ lệ treatment/control.
- Outcome rate của treatment và control.
- Class imbalance của `visit` và `conversion`.
- Feature distribution giữa treatment và control.
- Standardized Mean Difference của các feature.
- Treatment overlap trong các vùng feature.
- Các nhóm khách hàng quá hiếm.
- Data leakage và post-treatment variables.

EDA không thể chứng minh hoàn toàn tất cả causal assumptions.

Các assumptions như SUTVA và consistency chủ yếu được đánh giá dựa trên cách dataset và experiment được thiết kế.

---

## 10. Project direction

```text
Criteo Dataset
      ↓
Data validation and EDA
      ↓
Train / validation / test split
      ↓
Response Model baseline
      ↓
T-Learner
      ↓
X-Learner
      ↓
Predict uplift scores
      ↓
Rank customers
      ↓
Evaluate Qini, AUUC and incremental conversions
      ↓
Select Top-K customers under budget constraints
      ↓
Estimate campaign cost, policy value and net value
```


