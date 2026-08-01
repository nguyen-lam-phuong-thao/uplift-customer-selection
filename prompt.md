Fix **Task 2: Evaluation/Test Isolation** across the active repository.

Use the smallest possible deletion-style patch.

Do **not** create new helper functions, new utility modules, new classes, new abstraction layers, compatibility wrappers, or split-validation frameworks.

The goal is to remove invalid non-locked `test` usage, not to redesign the project.

## Core rule

Only the explicit locked-test pipeline may use the `test` split.

Every other code path must operate only on:

- train
- validation

Outside the locked-test pipeline, code must not:

- load `test`
- materialize `test`
- build `test_frame`
- predict on `test`
- calculate test metrics
- log test metrics
- save test predictions
- save non-locked test evaluation artifacts
- include `"test"` in default prediction splits
- allow normal configs to request `"test"`
- allow standard evaluation, Top-K, bootstrap, selection, or model comparison to evaluate `test`

## Implementation principles

Prefer deleting obsolete code over adding new code.

If a code path exists only to support normal `test` evaluation, remove it.

Do not preserve backward compatibility for the previous unsafe behavior.

Do not redesign the architecture.

The expected result of this task is a **smaller and simpler codebase**, not a more abstract one.

## Important constraints

Do not:

- create new helper functions
- create new validation helpers
- create new artifact abstractions
- redesign manifest logic
- redesign locked-test logic
- refactor unrelated code
- change Task 1 / row_id behavior
- add compatibility flags such as `allow_test=True`
- modify the adjacent legacy `Uplift-Modeling` project
- regenerate generated artifacts
- perform style-only edits

Inspect every file identified by the audit.

Only modify a file if it contains an actual remaining violation of the Task 2 contract.

Do not modify generic helper code merely for defensive programming unless that helper is still reachable from an active non-locked code path.

## Files to inspect

Pay particular attention to:

- `src/uplift_modeling/pipelines/train_criteo_response_model.py`
- `src/uplift_modeling/pipelines/train_criteo_t_learner.py`
- `src/uplift_modeling/pipelines/train_criteo_x_learner.py`
- `src/uplift_modeling/pipelines/evaluate_criteo_predictions.py`
- `src/uplift_modeling/evaluation/topk_policy.py`
- `src/uplift_modeling/evaluation/bootstrap.py`
- `src/uplift_modeling/evaluation/bootstrap_summary.py`
- `src/uplift_modeling/evaluation/bootstrap_writer.py`
- `src/uplift_modeling/evaluation/selection_gate.py`
- `src/uplift_modeling/artifacts/manifest.py`
- `src/uplift_modeling/pipelines/create_experiment_manifest.py`
- `configs/modeling/`
- `tests/`

## Required changes

### 1. Normal training pipelines

Normal training must do only:

- load prepared data
- create train frame
- create validation frame
- fit on train
- early stop on validation
- save validation predictions
- save validation metrics
- log validation metrics

Remove all remaining normal `test` usage.

This includes removing or stopping the use of:

- `test_split`
- `test_frame`
- `test_scores`
- `metrics["test"]`
- `test_roc_auc`
- `test_average_precision`
- `test_log_loss`
- `test_rows`
- prediction artifacts containing test rows

If code currently contains

```python
DEFAULT_PREDICTION_SPLITS = ("validation", "test")
```

change it to

```python
DEFAULT_PREDICTION_SPLITS = ("validation",)
```

Remove `"test"` from normal allowed prediction splits.

Use the existing config validation to reject configs requesting `"test"`.

Do not introduce a new validation helper.

### 2. Normal modeling configs

Normal configs should produce validation predictions only.

Change

```yaml
prediction_splits:
  - validation
  - test
```

to

```yaml
prediction_splits:
  - validation
```

If `test_split` is only used by normal training, remove it.

Do not introduce replacement config options.

### 3. Standard evaluation

Standard evaluation must not process prediction artifacts containing `test`.

Keep the implementation simple.

If a standard evaluator loads an artifact containing `split == "test"`:

- fail immediately with `ValueError`
- do not continue
- do not silently filter after treating the artifact as valid

Do not redesign artifact loading.

### 4. Top-K, bootstrap and selection

Only modify these components if they still allow an active non-locked pipeline to evaluate `test`.

Do not harden generic helper APIs merely for defensive programming.

The goal is to remove active violations, not to add defensive infrastructure.

If a normal code path reaches one of these functions with `test`, reject it locally.

If a function is used only by locked-test, leave it unchanged.

### 5. Manifest and latest-artifact behavior

Do not redesign manifest logic.

After training becomes validation-only, normal artifact creation and evaluation must no longer accept prediction artifacts containing `test`.

Keep changes minimal.

Do not introduce new manifest abstractions.

### 6. Locked-test pipeline

Keep locked-test behavior unchanged.

Locked-test remains the only place allowed to use `test`.

Do not move locked-test logic into normal training or evaluation.

Only modify locked-test if cleanup causes a direct break, and keep that change minimal.

## Tests

Update existing focused tests that encode the previous unsafe behavior.

Prefer modifying existing tests over creating new ones.

Only add a new test if there is no nearby test covering that component.

Do not run any tests yourself.

Do not execute pytest or any project commands.

I will run the test suite after your changes.

## Deliverable

After editing, provide:

1. Files changed.
2. Every removed or changed non-locked `test` code path.
3. Tests updated.
4. Any remaining suspicious path intentionally left unchanged, and why.

Keep the patch small.

If a line exists only to support non-locked `test` behavior, delete it.

If you feel tempted to introduce a new helper, abstraction, or compatibility layer, do not.

Simplify or remove the invalid code instead.