# Week 2 Work Log

| Field         | Value                                                           |
| ------------- | --------------------------------------------------------------- |
| Period        | July 27–31, 2026                                                |
| Author        | Nguyễn Lâm Phương Thảo                                          |
| Project       | Customer Selection with Uplift Modeling                         |
| Team size     | 1                                                               |
| Current phase | Model development, evaluation workflow, and framework hardening |

## Week 2 objectives

* [x] Hoàn thiện Response Model baseline.
* [x] Xây dựng T-Learner và X-Learner.
* [x] Ghi model và model components vào MLflow.
* [x] Tạo validation prediction artifacts cho các candidate model.
* [x] Chuẩn hóa việc căn chỉnh prediction bằng `row_id`.
* [x] Xây dựng các uplift và customer-selection evaluation metrics.
* [x] Thêm paired bootstrap để ước lượng uncertainty và so sánh candidate với baseline.
* [x] Xây dựng Selection Gate theo quy tắc deterministic.
* [x] Tách locked-test evaluation khỏi intermediate validation evaluation.
* [x] Xây dựng bước reload selected policy để tạo locked-test predictions.
* [x] Audit lại workflow, artifact flow và các điểm có nguy cơ làm sai experiment identity.
* [x] Xác định các framework-hardening task cần hoàn thiện trước khi mở rộng dataset.
* [ ] Hoàn thiện toàn bộ manifest, artifact schema, champion-locking và final-evaluation contracts.

---

## Work log

| Period      | Work completed                                                                                                                                                                                                                                      | Result or evidence                                                                                               | Next step                                                                                      |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Mon, Jul 27 | Hoàn thiện treated-response baseline và chuẩn hóa bước đọc processed decision dataset, chọn feature, train trên train split và sử dụng validation cho model fitting.                                                                                | [Training pipelines](../../src/uplift_modeling/pipelines/), [Model configurations](../../configs/modeling/)      | Xây dựng các uplift-model candidate còn lại theo cùng data contract.                           |
| Tue, Jul 28 | Xây dựng T-Learner và X-Learner; ghi model hoặc model components vào MLflow; tạo validation predictions và model-provenance information.                                                                                                            | [Model modules](../../src/uplift_modeling/models/), [Tracking modules](../../src/uplift_modeling/tracking/)      | Chuẩn hóa prediction artifacts để các model được đánh giá trên cùng observation.               |
| Wed, Jul 29 | Hoàn thiện evaluation workflow cho Top-K targeting metrics, policy value, incremental outcome, Qini và AUUC; căn chỉnh predictions bằng `row_id`; thêm paired bootstrap và paired contrasts với baseline.                                           | [Evaluation modules](../../src/uplift_modeling/evaluation/)                                                      | Xây dựng quy tắc chọn champion từ validation evidence.                                         |
| Thu, Jul 30 | Xây dựng deterministic Selection Gate; tạo selection output; tách locked-test evaluation thành pipeline riêng; reload selected policy từ MLflow để tạo test predictions và final evaluation.                                                        | [Framework workflow](../framework_workflow.md), [Artifact contracts](../../contracts/README.md)                  | Kiểm tra lại experiment identity, provenance và test isolation trên toàn workflow.             |
| Fri, Jul 31 | Audit lại code và documentation; xác định các task hardening liên quan đến `row_id`, test isolation, experiment manifest, artifact schema version, `DatasetSpec` và config cleanup; bắt đầu sửa các điểm ảnh hưởng trực tiếp đến evaluation safety. | [Coding rules](../code_rules.md), [Project README](../../README.md), automated tests in [`tests/`](../../tests/) | Hoàn thiện từng hardening task riêng, giữ thay đổi nhỏ và không mở rộng logic không cần thiết. |

---

## Week 2 summary

### Completed

* Hoàn thiện ba model pipeline ban đầu:

  * Treated-response baseline.
  * T-Learner.
  * X-Learner.
* Chuẩn hóa việc train model từ processed decision dataset.
* Ghi trained models và model components vào MLflow.
* Tạo validation-only prediction artifacts.
* Duy trì `row_id` để liên kết prediction với đúng observation.
* Xây dựng uplift và targeting evaluation metrics.
* Thêm paired bootstrap để:

  * Ước lượng uncertainty.
  * So sánh từng candidate với baseline trên cùng bootstrap samples.
* Xây dựng Selection Gate với quy tắc deterministic.
* Tạo pipeline riêng cho locked-test evaluation.
* Reload selected policy để tạo champion-only test predictions.
* Audit lại toàn bộ workflow thay vì chỉ kiểm tra từng model riêng lẻ.
* Xác định các contract còn cần hardening trước khi framework được xem là ổn định.

### Current status

Project đã có workflow chính:

```text
Prepared decision dataset
        ↓
Candidate model training
        ↓
Validation predictions
        ↓
Prediction alignment by row_id
        ↓
Validation evaluation
        ↓
Paired bootstrap comparison
        ↓
Selection Gate
        ↓
Selected champion
        ↓
Locked-test prediction
        ↓
Final evaluation
```

Các thành phần chính đã tồn tại, nhưng project vẫn đang trong giai đoạn hardening.

Những phần cần tiếp tục kiểm tra không chủ yếu liên quan đến việc thêm model mới. Trọng tâm hiện tại là bảo đảm rằng các stage đang có được liên kết đúng với nhau và không thể sử dụng nhầm dataset, prediction artifact, model run hoặc experiment result.

### Framework-hardening tasks identified

Các nhóm vấn đề được xác định trong quá trình audit gồm:

1. **`row_id` integrity**

   * `row_id` phải được tạo một lần tại preparation boundary.
   * Prediction và dataset phải được ghép bằng `row_id`, không phụ thuộc vào row order.
   * Không được có missing, duplicated hoặc unmatched `row_id`.

2. **Test isolation**

   * Intermediate evaluation chỉ được sử dụng validation data.
   * Test split chỉ được truy cập thông qua locked-test pipeline.
   * Test results không được ảnh hưởng đến champion selection.

3. **Experiment manifest**

   * Một manifest phải mô tả đúng một experiment.
   * Không được tự động trộn prediction artifacts từ các lần chạy khác nhau.
   * Manifest cần liên kết rõ dataset, config, predictions và model provenance.

4. **Artifact schema version**

   * Các artifact cần có schema rõ ràng và có thể validate.
   * Downstream stages không nên chấp nhận artifact thiếu identity hoặc provenance bắt buộc.

5. **`DatasetSpec`**

   * Dataset-dependent column names và feature definitions cần được truyền qua một contract rõ ràng.
   * Shared evaluation code không nên phụ thuộc trực tiếp vào tên cột đặc thù của Criteo.

6. **Configuration cleanup**

   * Loại bỏ config dư, trùng hoặc không còn được code sử dụng.
   * Giữ config đơn giản và phản ánh đúng behavior hiện tại.

7. **Champion identity and final evaluation**

   * Champion cần được liên kết với exact MLflow run và model URI hoặc component URIs.
   * Locked-test evaluation chỉ được reload model đã được chọn.
   * Final evaluation không được làm thay đổi champion của cùng experiment.

---

## Main decisions

### Validation and test usage

Validation được sử dụng cho:

* Model fitting decisions được cấu hình trước, ví dụ early stopping.
* Intermediate evaluation.
* Paired bootstrap comparison.
* Champion selection.

Test chỉ được sử dụng sau khi champion đã được chọn và khóa.

```text
Train → Validation → Selection Gate → Champion → Locked Test
```

### `random_targeting`

`random_targeting` được giữ làm evaluation benchmark.

Nó không phải deployable model candidate và không được truyền vào Selection Gate. Không cần thêm một cơ chế eligibility phức tạp khi có thể giới hạn trực tiếp danh sách candidate đầu vào của Selection Gate.

### File and module names

Các filename và module name hiện tại vẫn còn chứa `criteo`.

Việc đổi tên được hoãn lại cho đến khi:

* Workflow ổn định.
* Contracts được hoàn thiện.
* Code và documentation đã đồng nhất.

Quyết định này tránh việc phải sửa import, declaration và entry point trong khi logic vẫn đang được harden.

### Framework scope

Dataset và model mới sẽ cần phần tích hợp riêng, nhưng không nên yêu cầu viết lại:

* Evaluation metrics.
* Paired bootstrap.
* Selection Gate.
* Champion artifact logic.
* Locked-test evaluation.

Model train trên Criteo sẽ không được sử dụng trực tiếp cho một dataset khác.

---

## Current implementation boundary

### Dataset-specific components

```text
Raw-data loading
Data validation
Feature engineering
Treatment and outcome definition
Train / validation / test creation
```

### Shared framework components

```text
Candidate training contract
Prediction artifact contract
row_id alignment
Validation evaluation
Paired bootstrap
Selection Gate
Champion locking
Locked-test evaluation
```

Mục tiêu của bước hardening là làm rõ ranh giới này mà không thêm abstraction hoặc infrastructure không cần thiết.

---

## Next tasks

* [x] Tạo stable `row_id` tại data-preparation boundary.
* [x] Duy trì `row_id` trong prediction artifacts.
* [x] Căn chỉnh dataset và predictions bằng `row_id`.
* [x] Giới hạn intermediate evaluation vào validation split.
* [x] Giữ locked-test evaluation ở pipeline riêng.
* [x] Xác định `random_targeting` chỉ là benchmark.
* [x] Hoãn đổi tên file cho đến khi workflow ổn định.
* [ ] Hoàn thiện immutable experiment manifest.
* [ ] Validate identity của toàn bộ manifest inputs.
* [ ] Thêm và enforce artifact schema version.
* [ ] Hoàn thiện `DatasetSpec` boundary.
* [ ] Dọn config không còn được sử dụng.
* [ ] Khóa champion bằng exact MLflow run và model URI.
* [ ] Enforce one-way transition từ selection sang final evaluation.
* [ ] Kiểm tra final-evaluation idempotence và experiment sealing.
* [ ] Cập nhật toàn bộ documentation theo code hiện tại.
* [ ] Chạy lại automated tests và full-repository audit sau khi hoàn thành các task.

---

## Main project files reviewed

* [`README.md`](../../README.md)
* [`docs/framework_workflow.md`](../framework_workflow.md)
* [`docs/code_rules.md`](../code_rules.md)
* [`contracts/README.md`](../../contracts/README.md)
* [`src/uplift_modeling/data/`](../../src/uplift_modeling/data/)
* [`src/uplift_modeling/models/`](../../src/uplift_modeling/models/)
* [`src/uplift_modeling/evaluation/`](../../src/uplift_modeling/evaluation/)
* [`src/uplift_modeling/pipelines/`](../../src/uplift_modeling/pipelines/)
* [`tests/`](../../tests/)
