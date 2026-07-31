Implement **Task 6 — Config Cleanup only**.

Task 5 DatasetSpec is already complete and all tests currently pass.

Make a small, focused patch that removes obsolete configuration duplication without changing runtime behavior.

You may inspect files, search references, review diffs, run syntax checks, lint, formatting, and other non-test commands.

Do not run:

```text
pytest
unittest
tox
nox
coverage
test scripts
individual test files
the full test suite
```

I will run all tests locally.

## Goal

Configuration should contain only values that can legitimately vary between runs, such as:

```text
dataset selection
requested outcome
input/output paths
model parameters
training parameters
evaluation parameters
debug and tracking settings
```

Stable dataset structure is now owned by `DatasetSpec` and must not remain as a second authoritative source in YAML or config-loading code.

Remove obsolete configuration fields for stable dataset schema, including fields equivalent to:

```text
row_id_column
treatment_column
split_column
feature_columns
outcome_columns
```

Remove only fields that are now supplied by `DatasetSpec`.

Do not remove a similarly named field if inspection proves it has a different runtime purpose.

## Required work

1. Find every remaining YAML/config definition and Python access for stable dataset schema fields now owned by `DatasetSpec`.

2. Remove those obsolete fields from active configuration files.

3. Remove Python code that reads, validates, forwards, or falls back to those obsolete config fields.

4. Pipeline boundaries should:

```text
read dataset_name from config
resolve DatasetSpec once
read the requested outcome from config or CLI
validate that outcome against DatasetSpec
pass the resolved DatasetSpec to shared code
```

5. Do not keep compatibility fallback logic such as:

```python
treatment_column = config.get(
    "treatment_column",
    dataset_spec.treatment_column,
)
```

Use:

```python
treatment_column = dataset_spec.treatment_column
```

6. Do not silently accept conflicting obsolete schema keys.

The preferred result is that those obsolete keys no longer exist in active configs and are no longer read by source code.

7. Remove dead helper parameters, local variables, imports, validation branches, and comments that existed only for the removed config keys.

8. Preserve all current model parameters, paths, training settings, evaluation settings, debug settings, and tracking settings.

9. Preserve all behavior completed in:

```text
Task 1 — row_id
Task 2 — Test Isolation
Task 3 — Experiment Manifest
Task 5 — DatasetSpec
```

## Config inheritance

Inspect the current YAML inheritance or merge mechanism.

Clean up only inheritance made obsolete by the removed dataset-schema fields.

Do not redesign the config system.

Do not introduce:

```text
a new configuration library
new base-config architecture
new merge semantics
new environment-variable framework
new validation framework
compatibility adapters
automatic config migrations
```

Keep existing inheritance and merge behavior unless a branch exists solely to support removed schema keys.

Do not flatten every YAML file unless that is strictly necessary.

## Scope

Modify only directly related files, primarily:

```text
configs/**
src/uplift_modeling/utils/config.py
shared config-loading or validation helpers
pipeline boundaries still reading obsolete schema keys
directly related tests
directly related documentation
```

Modify training or evaluation pipeline files only where they still read or forward obsolete config schema values.

Do not modify:

```text
DatasetSpec fields or registry design
data preparation logic
row_id semantics
split assignment
model algorithms
prediction scoring
metric formulas
Top-K or bootstrap behavior
validation/test isolation
experiment manifest behavior
artifact schema versioning
artifact naming
run-number logic
unrelated documentation or tests
```

If completing the cleanup appears to require changes outside this scope, do not expand the patch. Report the exact dependency.

## Important limits

Do not rename unrelated configuration keys.

Do not reformat every YAML file.

Do not reorder large configuration sections unless required by the deletion.

Do not replace explicit configuration with hard-coded runtime values.

Do not remove keys merely because they appear unused until you confirm there are no source or test references.

Do not add deprecated aliases for removed keys.

Do not preserve two sources of truth.

## Tests to write or update

Update only focused test code necessary to reflect the cleaned contract.

Cover:

1. Active configs load without the removed dataset-schema keys.
2. Pipeline boundaries resolve stable columns from `DatasetSpec`.
3. Dataset name and outcome remain configurable.
4. Unsupported outcomes remain rejected.
5. Model, path, training, evaluation, debug, and tracking settings remain available.
6. Obsolete schema keys are not required.
7. Obsolete schema keys cannot override `DatasetSpec`.
8. Existing config inheritance still behaves the same for legitimate runtime settings.
9. Existing Task 1–3 and Task 5 behavior remains unchanged.

Do not weaken unrelated tests.

Write or update tests, but do not execute them.

## Allowed verification

You may run non-test checks such as:

```text
repository search for removed keys
git diff
git diff --check
syntax compilation
configured formatter
configured linter
```

At completion, search the active source and configs for each removed schema key and verify that no authoritative reads remain.

References in historical documentation or tests may remain only when they intentionally describe rejected legacy behavior.

## Completion response

Report only:

1. Obsolete config fields removed.
2. Config readers and pipeline consumers simplified.
3. Config inheritance cleanup performed, if any.
4. Files changed and why.
5. Tests written or updated.
6. Non-test checks performed.
7. Any unresolved dependency.
