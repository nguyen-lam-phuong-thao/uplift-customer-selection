# Contracts

## Row Identifier

Prepared decision datasets and prediction artifacts use one canonical
observation identifier: `row_id`.

`row_id` is created once when a decision dataset is prepared. It is
deterministic for the same prepared dataset, non-null, and unique within that
dataset. Pipeline steps must preserve it unchanged through filtering, splitting,
reordering, batching, training, prediction writing, evaluation, bootstrap,
selection, and locked-test reporting.

Prediction artifacts must expose the canonical `row_id` field. Evaluation code
must align model and policy artifacts by `row_id`; dataframe indexes, row
positions, sort order, and artifact order are not valid substitutes. Missing,
null, duplicated, unexpected, or mismatched `row_id` values are hard errors.
