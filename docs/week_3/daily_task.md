# Week 3 Work Plan

| Field         | Value                                           |
| ------------- | ----------------------------------------------- |
| Period        | August 3–7, 2026                                |
| Author        | Nguyễn Lâm Phương Thảo                          |
| Project       | Customer Selection with Uplift Modeling         |
| Team size     | 1                                               |
| Current phase | Framework completion and RetailHero integration |

## Week 3 objectives

* [ ] Hoàn thiện framework trong thứ Hai và thứ Ba.
* [ ] Chạy thành công toàn bộ workflow trên Criteo.
* [ ] Đồng bộ code, tests, config và documentation.
* [ ] Bắt đầu tích hợp RetailHero từ thứ Tư.
* [ ] Hoàn thành data understanding và feature engineering cơ bản.
* [ ] Tạo RetailHero decision dataset.
* [ ] Chạy thử framework với RetailHero trước cuối tuần.

---

## Daily tasks

| Date       | Main work                                                                                                                                        | Required result                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Mon, Aug 3 | Hoàn thiện các framework-hardening task còn lại: manifest, artifact validation, `row_id`, dataset contract, champion identity và test isolation. | Không còn lỗi logic chính đã được xác định trong audit.                                              |
| Tue, Aug 4 | Chạy tests, sửa integration bugs, xóa code dư, cập nhật docs và chạy full Criteo experiment.                                                     | Workflow chạy hoàn chỉnh từ training đến locked-test evaluation.                                     |
| Wed, Aug 5 | Đọc và kiểm tra RetailHero; xác định observation unit, treatment, outcome, feature window và outcome window.                                     | Hoàn thành data-understanding note và kế hoạch tạo decision dataset.                                 |
| Thu, Aug 6 | Xây dựng customer-level features từ dữ liệu trước treatment.                                                                                     | Có feature table đã kiểm tra missing values, duplicates và leakage.                                  |
| Fri, Aug 7 | Tạo `row_id`, train/validation/test split và RetailHero decision dataset; chạy thử model đầu tiên.                                               | RetailHero chạy được qua training và validation evaluation mà không viết lại shared evaluation code. |

---

## Framework completion criteria

Framework được xem là hoàn thành khi:

* Prediction được ghép bằng `row_id`, không phụ thuộc row order.
* Intermediate evaluation chỉ sử dụng validation.
* Manifest chỉ chứa artifacts của một experiment.
* Selection Gate chỉ nhận deployable candidates.
* `random_targeting` không được truyền vào Selection Gate.
* Champion được liên kết với đúng MLflow run và model artifact.
* Locked-test pipeline chỉ load champion đã được chọn.
* Full Criteo workflow và automated tests chạy thành công.
* Documentation khớp với code hiện tại.

---

## RetailHero minimum deliverables

Đến cuối tuần cần có:

* [ ] Data-understanding note.
* [ ] Treatment và outcome definitions.
* [ ] Feature-engineering pipeline phiên bản đầu tiên.
* [ ] Customer-level feature table.
* [ ] Standard decision dataset.
* [ ] RetailHero configuration.
* [ ] Ít nhất một model run.
* [ ] Validation prediction và evaluation output.

---

## Week 3 priority

```text
Monday–Tuesday: Complete and verify framework
Wednesday: Understand RetailHero data
Thursday: Build customer features
Friday: Create decision dataset and run first model
```

Không ưu tiên trong tuần này:

* Thêm model mới.
* Hyperparameter tuning.
* Đổi tên toàn bộ file.
* Dashboard, deployment hoặc monitoring.
