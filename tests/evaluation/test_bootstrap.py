"""Tests for Top-K bootstrap policy evaluation."""

import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift_modeling.evaluation.bootstrap import (
    calculate_bootstrap_policy_metric_samples,
)

from uplift_modeling.evaluation.bootstrap_summary import (
    calculate_bootstrap_policy_rows,
    summarize_bootstrap_paired_contrasts,
    summarize_bootstrap_policy_metric_samples,
)

from uplift_modeling.evaluation.bootstrap_writer import (
    save_bootstrap_policy_evaluation,
)
from uplift_modeling.evaluation.topk_policy import TOPK_BUDGET_FRACTIONS
from uplift_modeling.evaluation.uplift_metrics import (
    calculate_policy_value,
    calculate_selected_incremental_outcome,
)


def _base_frame(model_name: str, scores: list[float]) -> pd.DataFrame:
    """Return a row-aligned synthetic prediction frame."""
    return pd.DataFrame(
        {
            "row_id": list(range(20)),
            "treatment": [1, 0] * 10,
            "outcome": [1, 0, 0, 1, 1, 0, 0, 0, 1, 0] * 2,
            "split": ["validation"] * 10 + ["test"] * 10,
            "score": scores,
            "model_name": [model_name] * 20,
        }
    )


def _policy_frames() -> dict[str, pd.DataFrame]:
    """Return aligned policy frames with different rankings."""
    baseline_scores = [
        0.90,
        0.89,
        0.78,
        0.77,
        0.66,
        0.65,
        0.54,
        0.53,
        0.42,
        0.41,
        0.90,
        0.89,
        0.78,
        0.77,
        0.66,
        0.65,
        0.54,
        0.53,
        0.42,
        0.41,
    ]
    policy_scores = [
        0.91,
        0.50,
        0.80,
        0.49,
        0.70,
        0.48,
        0.60,
        0.47,
        0.59,
        0.46,
        0.91,
        0.50,
        0.80,
        0.49,
        0.70,
        0.48,
        0.60,
        0.47,
        0.59,
        0.46,
    ]
    return {
        "treated_response_lgbm": _base_frame(
            "treated_response_lgbm",
            baseline_scores,
        ),
        "t_learner_lgbm": _base_frame("t_learner_lgbm", policy_scores),
    }


def _policy_frames_with_row_id() -> dict[str, pd.DataFrame]:
    """Return aligned policy frames with stable row identifiers."""
    return _policy_frames()


def _run_fresh_python_import_check(code: str) -> None:
    """Run an import check in a fresh interpreter with local src on path."""
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(project_root / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=env,
        check=True,
    )


@pytest.mark.parametrize(
    "module_order",
    list(
        itertools.permutations(
            (
                "uplift_modeling.evaluation.bootstrap_writer",
                "uplift_modeling.evaluation.bootstrap_summary",
                "uplift_modeling.evaluation.bootstrap",
            )
        )
    ),
)
def test_bootstrap_modules_import_in_any_order(
    module_order: tuple[str, ...],
) -> None:
    """Bootstrap modules import independently without circular imports."""
    code = "\n".join(
        [
            "import importlib",
            *[
                f"importlib.import_module('{module}')"
                for module in module_order
            ],
        ]
    )

    _run_fresh_python_import_check(code)


def test_bootstrap_output_has_expected_columns_and_valid_ci() -> None:
    """Bootstrap summaries expose the required metric columns."""

    policy_rows, contrast_rows, warnings = calculate_bootstrap_policy_rows(
        _policy_frames(),
        budget_fractions=(0.5, 1.0),
        n_bootstrap=8,
        random_seed=42,
    )

    required_policy_columns = {
        "policy",
        "split",
        "budget_fraction",
        "budget_pct",
        "metric",
        "mean",
        "std",
        "ci_lower",
        "ci_upper",
        "n_bootstrap",
        "n_valid_bootstrap",
        "random_seed",
    }
    required_contrast_columns = {
        "policy",
        "baseline_policy",
        "split",
        "budget_fraction",
        "budget_pct",
        "metric",
        "mean_delta",
        "std_delta",
        "ci_lower",
        "ci_upper",
        "n_bootstrap",
        "n_valid_bootstrap",
        "random_seed",
    }

    assert warnings == []
    assert {row["split"] for row in policy_rows} == {"validation"}
    assert {row["split"] for row in contrast_rows} == {"validation"}
    assert required_policy_columns.issubset(policy_rows[0])
    assert required_contrast_columns.issubset(contrast_rows[0])
    assert all(
        0 <= row["n_valid_bootstrap"] <= row["n_bootstrap"]
        for row in [*policy_rows, *contrast_rows]
    )
    assert all(
        row["ci_lower"] <= row["ci_upper"]
        for row in [*policy_rows, *contrast_rows]
        if row["ci_lower"] is not None and row["ci_upper"] is not None
    )


def test_bootstrap_results_are_deterministic_with_fixed_seed() -> None:
    """The same seed produces identical bootstrap summary rows."""
    first = calculate_bootstrap_policy_rows(
        _policy_frames(),
        budget_fractions=(1.0,),
        n_bootstrap=6,
        random_seed=7,
    )
    second = calculate_bootstrap_policy_rows(
        _policy_frames(),
        budget_fractions=(1.0,),
        n_bootstrap=6,
        random_seed=7,
    )

    assert first == second


def test_bootstrap_summary_counts_only_valid_samples() -> None:
    """Finite bootstrap samples define n_valid_bootstrap."""
    samples = pd.DataFrame(
        {
            "policy": ["policy"] * 3,
            "split": ["validation"] * 3,
            "budget_fraction": [1.0] * 3,
            "budget_pct": [100.0] * 3,
            "policy_value": [1.0, np.nan, 3.0],
            "incremental_outcome": [2.0, None, 4.0],
        }
    )

    rows = summarize_bootstrap_policy_metric_samples(
        samples=samples,
        n_bootstrap=3,
        random_seed=42,
    )

    assert {row["n_valid_bootstrap"] for row in rows} == {2}
    assert {row["n_bootstrap"] for row in rows} == {3}


def test_bootstrap_summary_uses_single_value_for_one_valid_sample() -> None:
    """One valid bootstrap sample keeps mean and uses it for both CI bounds."""
    samples = pd.DataFrame(
        {
            "policy": ["policy"] * 3,
            "split": ["validation"] * 3,
            "budget_fraction": [1.0] * 3,
            "budget_pct": [100.0] * 3,
            "policy_value": [None, 4.0, np.nan],
            "incremental_outcome": [None, 2.0, np.nan],
        }
    )

    rows = summarize_bootstrap_policy_metric_samples(
        samples=samples,
        n_bootstrap=3,
        random_seed=42,
    )

    assert {row["n_valid_bootstrap"] for row in rows} == {1}
    assert all(row["std"] is None for row in rows)
    assert all(row["mean"] == row["ci_lower"] == row["ci_upper"] for row in rows)


def test_bootstrap_summary_returns_nulls_when_no_valid_samples_remain() -> None:
    """A metric with zero finite bootstrap samples returns a null summary."""
    samples = pd.DataFrame(
        {
            "policy": ["policy"] * 2,
            "split": ["validation"] * 2,
            "budget_fraction": [1.0] * 2,
            "budget_pct": [100.0] * 2,
            "policy_value": [np.nan, None],
            "incremental_outcome": [np.nan, None],
        }
    )

    rows = summarize_bootstrap_policy_metric_samples(
        samples=samples,
        n_bootstrap=2,
        random_seed=42,
    )

    assert rows
    for row in rows:
        assert row["n_bootstrap"] == 2
        assert row["n_valid_bootstrap"] == 0
        assert row["mean"] is None
        assert row["std"] is None
        assert row["ci_lower"] is None
        assert row["ci_upper"] is None


def test_paired_contrast_summary_returns_nulls_when_no_valid_samples_remain() -> None:
    """Paired contrast metrics with zero finite deltas return null summaries."""
    samples = pd.DataFrame(
        {
            "bootstrap_iteration": [0, 0, 1, 1],
            "policy": [
                "treated_response_lgbm",
                "t_learner_lgbm",
                "treated_response_lgbm",
                "t_learner_lgbm",
            ],
            "split": ["validation"] * 4,
            "budget_fraction": [1.0] * 4,
            "budget_pct": [100.0] * 4,
            "policy_value": [np.nan, np.nan, None, None],
            "incremental_outcome": [np.nan, np.nan, None, None],
        }
    )

    rows, warnings = summarize_bootstrap_paired_contrasts(
        samples=samples,
        baseline_policy="treated_response_lgbm",
        n_bootstrap=2,
        random_seed=42,
    )

    assert warnings == []
    assert rows
    for row in rows:
        assert row["n_bootstrap"] == 2
        assert row["n_valid_bootstrap"] == 0
        assert row["mean_delta"] is None
        assert row["std_delta"] is None
        assert row["ci_lower"] is None
        assert row["ci_upper"] is None


def test_bootstrap_calculation_can_be_restricted_to_validation_split() -> None:
    """Bootstrap summaries can evaluate only the requested split."""
    policy_rows, contrast_rows, warnings = calculate_bootstrap_policy_rows(
        _policy_frames(),
        budget_fractions=(0.5,),
        n_bootstrap=4,
        random_seed=42,
        bootstrap_splits=("validation",),
    )

    assert warnings == []
    assert {row["split"] for row in policy_rows} == {"validation"}
    assert {row["split"] for row in contrast_rows} == {"validation"}


def test_bootstrap_calculation_can_be_restricted_to_one_budget() -> None:
    """Bootstrap summaries can evaluate only the requested budget."""
    policy_rows, contrast_rows, warnings = calculate_bootstrap_policy_rows(
        _policy_frames(),
        budget_fractions=(0.5,),
        n_bootstrap=4,
        random_seed=42,
    )

    assert warnings == []
    assert {row["budget_fraction"] for row in policy_rows} == {0.5}
    assert {row["budget_fraction"] for row in contrast_rows} == {0.5}


def test_restricted_bootstrap_artifact_records_evaluated_splits_and_budgets(
    tmp_path,
) -> None:
    """Restricted bootstrap artifacts contain only requested split and budget."""
    _, _, payload = save_bootstrap_policy_evaluation(
        policy_frames=_policy_frames(),
        metric_dir=tmp_path,
        dataset_name="criteo",
        outcome="visit",
        random_seed=42,
        n_bootstrap=3,
        budget_fractions=(0.5,),
        bootstrap_splits=("validation",),
    )

    policy_rows = payload["regular_bootstrap_metric_rows"]
    contrast_rows = payload["paired_contrast_rows"]

    assert payload["evaluated_splits"] == ["validation"]
    assert payload["budget_fractions"] == [0.5]
    assert payload["random_seed"] == 42
    assert payload["n_bootstrap"] == 3
    assert {row["split"] for row in policy_rows} == {"validation"}
    assert {row["budget_fraction"] for row in policy_rows} == {0.5}
    assert {row["split"] for row in contrast_rows} == {"validation"}
    assert {row["budget_fraction"] for row in contrast_rows} == {0.5}


def test_bootstrap_artifact_defaults_to_validation_only(tmp_path) -> None:
    """Model-comparison bootstrap artifacts do not expose test results."""
    _, _, payload = save_bootstrap_policy_evaluation(
        policy_frames=_policy_frames(),
        metric_dir=tmp_path,
        dataset_name="criteo",
        outcome="visit",
        random_seed=42,
        n_bootstrap=3,
        budget_fractions=(1.0,),
    )

    assert payload["evaluated_splits"] == ["validation"]
    assert {
        row["split"] for row in payload["regular_bootstrap_metric_rows"]
    } == {"validation"}
    assert {row["split"] for row in payload["paired_contrast_rows"]} == {
        "validation"
    }


def test_bootstrap_artifact_rejects_test_split_request(tmp_path) -> None:
    """Standard bootstrap model comparison cannot evaluate test rows."""
    with pytest.raises(ValueError, match="test split is reserved"):
        save_bootstrap_policy_evaluation(
            policy_frames=_policy_frames(),
            metric_dir=tmp_path,
            dataset_name="criteo",
            outcome="visit",
            random_seed=42,
            n_bootstrap=3,
            budget_fractions=(1.0,),
            bootstrap_splits=("test",),
        )


def test_omitted_bootstrap_filters_default_to_validation() -> None:
    """Omitting split filters keeps bootstrap output validation-only."""
    frames = _policy_frames()

    default_rows = calculate_bootstrap_policy_rows(
        frames,
        n_bootstrap=3,
        random_seed=42,
    )
    explicit_validation_rows = calculate_bootstrap_policy_rows(
        frames,
        budget_fractions=TOPK_BUDGET_FRACTIONS,
        n_bootstrap=3,
        random_seed=42,
        bootstrap_splits=("validation",),
    )

    assert default_rows == explicit_validation_rows


def test_invalid_bootstrap_budget_fraction_still_raises() -> None:
    """Budget validation still rejects invalid bootstrap fractions."""
    with pytest.raises(ValueError, match="Budget fractions"):
        calculate_bootstrap_policy_rows(
            _policy_frames(),
            budget_fractions=(0.0,),
            n_bootstrap=2,
            random_seed=42,
        )


def test_requested_missing_bootstrap_split_raises_clear_error() -> None:
    """A requested split must exist in at least one policy frame."""
    with pytest.raises(ValueError, match="holdout"):
        calculate_bootstrap_policy_rows(
            _policy_frames(),
            budget_fractions=(0.5,),
            n_bootstrap=2,
            random_seed=42,
            bootstrap_splits=("holdout",),
        )


def test_bootstrap_samples_keep_topk_selected_count() -> None:
    """Each bootstrap sample selects ceil(budget * sampled row count)."""
    samples = calculate_bootstrap_policy_metric_samples(
        _policy_frames(),
        budget_fractions=(0.5,),
        n_bootstrap=4,
        random_seed=42,
    )

    assert samples["n_selected"].unique().tolist() == [5]


def test_bootstrap_succeeds_when_all_row_ids_match() -> None:
    """Paired bootstrap accepts row-aligned frames when row_id is available."""
    samples = calculate_bootstrap_policy_metric_samples(
        _policy_frames_with_row_id(),
        budget_fractions=(1.0,),
        n_bootstrap=2,
        random_seed=42,
    )

    assert not samples.empty


def test_bootstrap_raises_when_row_ids_are_misaligned() -> None:
    """Paired bootstrap rejects frames with mismatched row_id labels."""
    frames = _policy_frames_with_row_id()
    misaligned_frame = frames["t_learner_lgbm"].copy()

    for split in ("validation", "test"):
        split_mask = misaligned_frame["split"].eq(split)
        misaligned_frame.loc[split_mask, "row_id"] = (
            misaligned_frame.loc[split_mask, "row_id"].to_numpy()[::-1]
        )

    frames["t_learner_lgbm"] = misaligned_frame

    with pytest.raises(ValueError, match="labels do not match.*row_id"):
        calculate_bootstrap_policy_metric_samples(
            frames,
            budget_fractions=(1.0,),
            n_bootstrap=2,
            random_seed=42,
        )


def test_bootstrap_raises_when_only_some_frames_have_row_id() -> None:
    """Missing row_id availability is rejected because pairing is ambiguous."""
    frames = _policy_frames()
    frames["t_learner_lgbm"] = frames["t_learner_lgbm"].drop(columns="row_id")

    with pytest.raises(ValueError, match="row-ID column 'row_id'"):
        calculate_bootstrap_policy_metric_samples(
            frames,
            budget_fractions=(1.0,),
            n_bootstrap=2,
            random_seed=42,
        )


def test_bootstrap_validates_labels_by_row_id() -> None:
    """Paired bootstrap rejects treatment/outcome mismatches by row_id."""
    frames = _policy_frames()
    frames["t_learner_lgbm"] = frames["t_learner_lgbm"].assign(
        outcome=1 - frames["t_learner_lgbm"]["outcome"]
    )

    with pytest.raises(ValueError, match="labels do not match"):
        calculate_bootstrap_policy_metric_samples(
            frames,
            budget_fractions=(1.0,),
            n_bootstrap=2,
            random_seed=42,
        )


def test_bootstrap_rejects_missing_prediction_ids() -> None:
    """Paired bootstrap rejects artifacts missing expected row_id values."""
    frames = _policy_frames()
    frames["t_learner_lgbm"] = frames["t_learner_lgbm"].assign(
        row_id=list(range(19)) + [99]
    )

    with pytest.raises(ValueError, match="missing row_id"):
        calculate_bootstrap_policy_metric_samples(
            frames,
            budget_fractions=(1.0,),
            n_bootstrap=2,
            random_seed=42,
        )


def test_bootstrap_rejects_unexpected_extra_prediction_ids() -> None:
    """Paired bootstrap rejects artifacts with unexpected row_id values."""
    frames = _policy_frames()
    frames["treated_response_lgbm"] = frames["treated_response_lgbm"].assign(
        row_id=list(range(19)) + [99]
    )

    with pytest.raises(ValueError, match="unexpected row_id"):
        calculate_bootstrap_policy_metric_samples(
            frames,
            budget_fractions=(1.0,),
            n_bootstrap=2,
            random_seed=42,
        )


def test_bootstrap_accepts_reordered_policy_artifacts() -> None:
    """Paired bootstrap aligns reordered policy rows by row_id."""
    frames = _policy_frames()
    frames["t_learner_lgbm"] = frames["t_learner_lgbm"].sort_values(
        "row_id",
        ascending=False,
    )

    reordered_samples = calculate_bootstrap_policy_metric_samples(
        frames,
        budget_fractions=(1.0,),
        n_bootstrap=2,
        random_seed=42,
    )
    baseline_samples = calculate_bootstrap_policy_metric_samples(
        _policy_frames(),
        budget_fractions=(1.0,),
        n_bootstrap=2,
        random_seed=42,
    )

    pd.testing.assert_frame_equal(reordered_samples, baseline_samples)


def test_bootstrap_metrics_use_existing_metric_logic() -> None:
    """Per-iteration values match the shared policy metric helpers."""
    frames = {
        policy_name: frame.loc[
            frame["split"] == "validation"
        ].reset_index(drop=True)
        for policy_name, frame in _policy_frames().items()
    }
    samples = calculate_bootstrap_policy_metric_samples(
        frames,
        budget_fractions=(1.0,),
        n_bootstrap=1,
        random_seed=123,
    )
    row = samples.loc[
        (samples["policy"] == "t_learner_lgbm")
        & (samples["split"] == "validation")
    ].iloc[0]
    rng = np.random.default_rng(123)
    sample_positions = rng.integers(low=0, high=10, size=10)
    expected_frame = frames["t_learner_lgbm"].iloc[
        sample_positions
    ].reset_index(drop=True)

    assert row["policy_value"] == pytest.approx(
        calculate_policy_value(expected_frame, top_fraction=1.0, require_unique_row_id=False)
    )
    assert row["incremental_outcome"] == pytest.approx(
        calculate_selected_incremental_outcome(
            expected_frame,
            top_fraction=1.0,
            require_unique_row_id=False
        )
    )


def test_policy_value_is_computed_at_each_budget_fraction() -> None:
    """Every policy/split/budget receives a policy value summary."""
    policy_rows, _, _ = calculate_bootstrap_policy_rows(
        _policy_frames(),
        budget_fractions=(0.5, 1.0),
        n_bootstrap=5,
        random_seed=42,
    )
    policy_value_rows = [
        row for row in policy_rows if row["metric"] == "policy_value"
    ]

    assert {
        row["budget_fraction"] for row in policy_value_rows
    } == {0.5, 1.0}
    assert len(policy_value_rows) == 4


def test_paired_contrast_uses_same_bootstrap_sample() -> None:
    """An identical non-baseline policy has an exact zero paired contrast."""
    baseline = _policy_frames()["treated_response_lgbm"]
    frames = {
        "treated_response_lgbm": baseline,
        "t_learner_lgbm": baseline.assign(model_name="t_learner_lgbm"),
    }

    _, contrast_rows, warnings = calculate_bootstrap_policy_rows(
        frames,
        budget_fractions=(1.0,),
        n_bootstrap=5,
        random_seed=42,
    )

    assert warnings == []
    assert contrast_rows
    assert all(row["policy"] != "treated_response_lgbm" for row in contrast_rows)
    assert all(row["mean_delta"] == pytest.approx(0.0) for row in contrast_rows)


def test_missing_baseline_skips_paired_contrast_with_warning() -> None:
    """Regular bootstrap metrics still run without the treated baseline."""
    frames = {"t_learner_lgbm": _policy_frames()["t_learner_lgbm"]}

    policy_rows, contrast_rows, warnings = calculate_bootstrap_policy_rows(
        frames,
        budget_fractions=(1.0,),
        n_bootstrap=3,
        random_seed=42,
    )

    assert policy_rows
    assert contrast_rows == []
    assert "treated_response_lgbm" in warnings[0]


def test_save_bootstrap_policy_evaluation_handles_missing_baseline(tmp_path) -> None:
    """Saving bootstrap artifacts does not require optional baseline artifacts."""
    frames = {"t_learner_lgbm": _policy_frames()["t_learner_lgbm"]}

    policy_json, contrast_json, payload = (
        save_bootstrap_policy_evaluation(
            policy_frames=frames,
            metric_dir=tmp_path,
            dataset_name="criteo",
            outcome="visit",
            random_seed=42,
            n_bootstrap=3,
            budget_fractions=(1.0,),
        )
    )

    assert policy_json.exists()
    assert contrast_json.exists()
    assert policy_json.name == "criteo_visit_bootstrap_policy_metrics_run01.json"
    assert (
        contrast_json.name
        == "criteo_visit_bootstrap_paired_contrasts_run01.json"
    )
    assert not list(tmp_path.glob("*.csv"))
    assert payload["regular_bootstrap_metric_rows"]
    assert payload["paired_contrast_rows"] == []
    assert payload["warnings"]


def test_bootstrap_does_not_mutate_input_frames() -> None:
    """Bootstrap sampling leaves input prediction frames unchanged."""
    frames = _policy_frames()
    before = {
        policy_name: frame.copy(deep=True)
        for policy_name, frame in frames.items()
    }

    calculate_bootstrap_policy_rows(
        frames,
        budget_fractions=(1.0,),
        n_bootstrap=3,
        random_seed=42,
    )

    for policy_name, frame in frames.items():
        pd.testing.assert_frame_equal(frame, before[policy_name])
