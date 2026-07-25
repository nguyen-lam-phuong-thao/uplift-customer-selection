# Week 1 Work Log

| Field | Value |
|---|---|
| Period | July 20–24, 2026 |
| Author | Nguyễn Lâm Phương Thảo |
| Project | Customer Selection with Uplift Modeling |
| Team size | 1 |
| Current phase | Research foundations and data preparation |

## Week 1 objectives

- [x] Xác định đề tài và vấn đề kinh doanh của project.
- [x] Phân biệt response prediction và uplift prediction.
- [x] Nghiên cứu potential outcomes, counterfactual, ATE và CATE.
- [x] Nghiên cứu Response Model, T-Learner và X-Learner.
- [x] Tổng hợp causal assumptions và EDA checklist.
- [x] Hoàn thành research note của Task 1.
- [x] Xác định hướng mở rộng project bằng RetailHero Dataset.
- [x] Hoàn thành xử lý và EDA cho Criteo Dataset.

---

## Work log

| Period | Work completed | Result or evidence | Next step |
|---|---|---|---|
| Mon–Tue, Jul 20–21 | Đọc tài liệu về Causal Inference và Uplift Modeling; nghiên cứu potential outcomes, counterfactual, treatment effect, ATE, CATE, Response Model, T-Learner và X-Learner. | [Causal Inference Foundations](causal_inference_foundations.md) | Tổng hợp nội dung nghiên cứu và xác định assumptions của project. |
| Wed–Thu, Jul 22–23 | Hoàn thiện phạm vi project; phân biệt hướng applied Data Science với hướng cải tiến model; tìm kiếm hướng mở rộng sau customer selection và so sánh các dataset marketing, CRM, Hillstrom và RetailHero. | [Project README](../../README.md) | Chọn dataset mở rộng và xác định vai trò của từng giai đoạn. |
| Fri, Jul 24 | Chọn X5 RetailHero làm dataset mở rộng cho Phase 2; xây dựng Criteo loader, validation, data understanding, EDA và Layer 3 data decisions. | [Data Understanding](data_understanding.md), [Phase 1 notebook](../../notebooks/phase1_criteo/01_criteo_eda.ipynb) | Bắt đầu Response Model baseline. |

---

## Week 1 summary

### Completed

- Xác định bài toán lựa chọn khách hàng bằng Uplift Modeling.
- Hoàn thành nghiên cứu nền tảng về Causal Inference.
- Phân biệt response probability và treatment effect.
- Hiểu cách hoạt động của Response Model, T-Learner và X-Learner.
- Xác định causal assumptions và các kiểm tra cần thực hiện trong EDA.
- Hoàn thành `docs/week_1/causal_inference_foundations.md`.
- Xác định hai giai đoạn của project:
  - Phase 1: Criteo uplift decision pipeline.
  - Phase 2: RetailHero customer analytics extension.
- Làm rõ project tập trung vào business decisioning, không tập trung cải tiến thuật toán.

### Current status

Task nghiên cứu nền tảng đã hoàn thành.

Project đã hoàn thành các bước Phase 1 nền tảng:

```text
Criteo data preparation
      ↓
Data validation
      ↓
Exploratory Data Analysis
      ↓
Layer 3 data decisions
```

Notebook hiện tại chốt dữ liệu đưa sang bước modeling:

- Features: `f0`-`f11`.
- Treatment indicator: `treatment`.
- Primary outcome: `visit`.
- Secondary outcome: `conversion`.
- Excluded column: `exposure`.
- Split: 60% train, 20% validation, 20% test.

### Main decision

Criteo được giữ làm dataset chính để xây dựng và đánh giá pipeline.

RetailHero được chọn làm dataset mở rộng để:

- Tạo customer features từ lịch sử giao dịch.
- Retrain pipeline trên retail data.
- Phân tích nhóm khách hàng uplift cao.
- Đưa ra customer insights và campaign recommendations.

Model train trên Criteo sẽ không được sử dụng trực tiếp để inference trên RetailHero.

---

## Next tasks

- [x] Tạo Criteo data loader.
- [x] Đọc thành công file `.csv.gz`.
- [x] Kiểm tra schema và data types.
- [x] Kiểm tra missing values và duplicate observations.
- [x] Kiểm tra tỷ lệ treatment/control.
- [x] Kiểm tra outcome rate của `visit` và `conversion`.
- [x] So sánh feature distributions giữa treatment và control.
- [x] Chọn primary outcome cho experiment đầu tiên.
- [x] Tạo train, validation và test split.
- [ ] Bắt đầu Response Model baseline.

---

## References reviewed

- [Causal Inference in Python](https://www.oreilly.com/library/view/causal-inference-in/9781098140243/)
- [Chapter 7: Metalearners](https://www.oreilly.com/library/view/causal-inference-in/9781098140243/ch07.html)
- [Meta-learners for Estimating Heterogeneous Treatment Effects Using Machine Learning](https://arxiv.org/abs/1706.03461)
- [A Large Scale Benchmark for Uplift Modeling](https://www.adkdd.org/papers/a-large-scale-benchmark-for-uplift-modeling/2018)
- [Criteo Uplift Prediction Dataset](https://huggingface.co/datasets/criteo/criteo-uplift)
- [X5 RetailHero Uplift Modeling Dataset](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data)
