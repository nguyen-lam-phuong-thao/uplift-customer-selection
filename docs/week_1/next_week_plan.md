# Week 2 Plan — Modeling & Evaluation

## Objective

Train and evaluate baseline uplift models on the prepared Criteo dataset.

Primary outcome: `visit`

Models:

- Response Model
- X-Learner
- T-Learner

Metrics:

- AUUC
- Qini curve
- Uplift curve

---

## Input

```text
data/processed/criteo/criteo_decision_visit.parquet
```

Use columns:

```text
Features: f0-f11
Treatment: treatment
Outcome: visit
Split: train / validation / test
```

---

## Tasks

### 1. Train Response Model

Train a baseline response prediction model:

```text
X → visit
```

Output score:

```text
P(visit = 1 | X)
```

Purpose:

```text
Non-causal baseline for comparison.
```

---

### 2. Train X-Learner

Train one model using features and treatment:

```text
[X, treatment] → visit
```

Generate uplift score:

```text
score = P(visit | X, treatment = 1) - P(visit | X, treatment = 0)
```

---

### 3. Train T-Learner

Train two separate models:

```text
Treatment model: X → visit where treatment = 1
Control model:   X → visit where treatment = 0
```

Generate uplift score:

```text
score = μ1(X) - μ0(X)
```

---

### 4. Save Prediction Outputs

Save predictions to:

```text
artifacts/predictions/
    visit_response_model_predictions.parquet
    visit_s_learner_predictions.parquet
    visit_t_learner_predictions.parquet
```

Each file should contain:

```text
treatment
outcome
split
score
model_name
```

---

### 5. Evaluate Models

Evaluate all models on validation and test sets.

Metrics:

```text
AUUC
Qini curve
Uplift curve
```

Save outputs to:

```text
artifacts/metrics/
    visit_model_evaluation.json

artifacts/figures/
    visit_qini_curve.png
    visit_uplift_curve.png
```

---

## Expected Outputs

```text
src/uplift_modeling/models/
    response_model.py
    s_learner.py
    t_learner.py

src/uplift_modeling/evaluation/
    uplift_metrics.py
    uplift_curves.py

artifacts/predictions/
    visit_response_model_predictions.parquet
    visit_s_learner_predictions.parquet
    visit_t_learner_predictions.parquet

artifacts/metrics/
    visit_model_evaluation.json

artifacts/figures/
    visit_qini_curve.png
    visit_uplift_curve.png
```

---

## Definition of Done

```text
✓ Response Model trained
✓ X-Learner trained
✓ T-Learner trained
✓ Prediction scores exported
✓ AUUC calculated
✓ Qini curve generated
✓ Uplift curve generated
✓ Model comparison table created
✓ Best baseline model selected
```

