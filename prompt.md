Implement only a simple Final Evaluation reuse rule.

Read `docs/code_rules.md` before editing.

Keep the implementation minimal. Do not add a lifecycle framework, state machine, registry, database, lock service, hashing, or new architecture.

The required behavior is:

```text
one experiment
= one selected champion
= one official Test result
```

## Exact behavior

### First run

When the experiment has no Final Evaluation result yet:

1. Use all existing Manifest, Selection, config, outcome, champion, and MLflow provenance validation.
2. Load the exact selected champion.
3. Score that champion on the Test split.
4. Save the locked-test prediction.
5. Save the final evaluation JSON.

### Later runs

When the Final Evaluation JSON for that experiment already exists:

1. Load the existing JSON.
2. Verify that it belongs to the same:

   * `experiment_id`;
   * `champion_policy`;
   * `champion_model_artifact`.
3. Return and print the existing result.
4. Do not score Test again.
5. Do not load the MLflow model again.
6. Do not write another prediction.
7. Do not recalculate metrics or bootstrap.
8. Do not overwrite the existing JSON.

In simple terms:

```text
Final result exists
→ read it
→ print it again
→ stop
```

If the existing result belongs to a different champion, raise `ValueError`.

## Deterministic filenames

Replace incrementing Locked Test names such as:

```text
run01
run02
```

with one deterministic result per experiment.

Use:

```text
{dataset_name}_{outcome}_{experiment_id}_locked_test_evaluation.json
```

and:

```text
{dataset_name}_{outcome}_{champion_policy}_{experiment_id}_locked_test_predictions.parquet
```

There must not be a second official Locked Test result such as `run02`.

## Important execution order

In `evaluate_locked_test()`:

1. Load and validate the Selection and Manifest using existing logic.
2. Determine the deterministic final-report path.
3. If the final report exists:

   * load it;
   * check `experiment_id`, `champion_policy`, and `champion_model_artifact`;
   * print or return the existing result immediately.
4. Only if the report does not exist:

   * load Test data;
   * set up MLflow;
   * load the champion;
   * score Test;
   * save the prediction and final report.

The existing-result check must happen before Test data loading and model scoring.

## Existing incomplete artifacts

If the final JSON does not exist but the deterministic prediction file already exists, raise a clear error.

Do not overwrite the prediction and do not add recovery logic.

Example:

```text
Incomplete Final Evaluation state: prediction exists without final report.
```

## Final report

Preserve the current metrics and bootstrap output.

Ensure the report records:

```text
artifact_type
experiment_id
dataset_name
outcome
split
champion_policy
champion_model_artifact
prediction_artifacts
budget_fractions
uplift_metrics
locked_test_rows
bootstrap
```

`champion_model_artifact` must come directly from the validated Selection artifact and must retain the exact MLflow run and model URI information.

Do not change any metric or bootstrap calculation.

## Scope

Keep changes mainly in:

```text
src/uplift_modeling/pipelines/evaluate_locked_test.py
src/uplift_modeling/evaluation/locked_test.py
tests/pipelines/test_evaluate_locked_test.py
tests/evaluation/test_locked_test.py
```

Change another file only if directly necessary.

Do not modify:

* training;
* validation evaluation;
* champion selection;
* Selection Gate logic;
* Manifest creation;
* bootstrap logic;
* metric formulas;
* row alignment;
* model scoring formulas;
* previous correctness fixes.

## Minimal tests

Only add or update basic tests.

### Test 1

First run creates:

* one deterministic champion Test prediction;
* one deterministic Final Evaluation JSON.

### Test 2

Run the same experiment again.

Verify:

* the same final result is returned;
* Test scoring is not called again;
* no new prediction or JSON is created.

### Test 3

If the existing final result records a different champion, verify that `ValueError` is raised.

Update existing tests that expect `run01` filenames.

Do not build a large test matrix.

## Verification

Run:

```text
pytest tests/pipelines/test_evaluate_locked_test.py -q
pytest tests/evaluation/test_locked_test.py -q
python -m compileall -q src tests
pytest -q
```

## Definition of done

The task is complete when:

```text
first run
→ evaluates the selected champion on Test
→ saves one result

later run
→ reads and prints the saved result
→ does not evaluate Test again
```

Also confirm:

* no `run02` official Locked Test result can be created;
* existing results are never overwritten;
* a different champion cannot reuse the result;
* all existing tests pass;
* no unrelated logic was changed.

When finished, report only:

1. files changed;
2. how the existing result is reused;
3. test results;
4. confirmation that no unrelated logic was changed.
