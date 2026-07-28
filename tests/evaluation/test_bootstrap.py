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
    calculate_bootstrap_policy_rows,
    save_bootstrap_policy_evaluation,
)
from uplift_modeling.evaluation.bootstrap_summary import (
    calculate_bootstrap_policy_rows as calculate_bootstrap_policy_rows_new,
)
from uplift_modeling.evaluation.bootstrap_writer import (
    save_bootstrap_policy_evaluation as save_bootstrap_policy_evaluation_new,
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
    return {
        policy_name: frame.assign(row_id=np.arange(len(frame)))
        for policy_name, frame in _policy_frames().items()
    }


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
def test_bootstrap_modules_import_in_any_order(module_order: tuple[str, ...]) -> None:
    """Bootstrap modules import independently without circular import failures."""
    module_list = ", ".join(repr(module_name) for module_name in module_order)
    _run_fresh_python_import_check(
        "import importlib\n"
        f"for module_name in ({module_list},):\n"
        "    importlib.import_module(module_name)\n"
    )


def test_bootstrap_legacy_symbols_import_directly() -> None:
    """Legacy symbols remain importable from the bootstrap module."""
    _run_fresh_python_import_check(
        "from uplift_modeling.evaluation.bootstrap import (\n"
        "    calculate_bootstrap_policy_rows,\n"
        "    find_next_bootstrap_run_number,\n"
        "    save_bootstrap_policy_evaluation,\n"
        "    summarize_bootstrap_paired_contrasts,\n"
        ")\n"
    )


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
        "random_seed",
    }

    assert warnings == []
    assert required_policy_columns.issubset(policy_rows[0])
    assert required_contrast_columns.issubset(contrast_rows[0])
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
        budget_fractions=(0.05,),
        n_bootstrap=4,
        random_seed=42,
    )

    assert warnings == []
    assert {row["budget_fraction"] for row in policy_rows} == {0.05}
    assert {row["budget_fraction"] for row in contrast_rows} == {0.05}


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
        budget_fractions=(0.05,),
        bootstrap_splits=("validation",),
    )

    policy_rows = payload["regular_bootstrap_metric_rows"]
    contrast_rows = payload["paired_contrast_rows"]

    assert payload["evaluated_splits"] == ["validation"]
    assert payload["budget_fractions"] == [0.05]
    assert payload["random_seed"] == 42
    assert payload["n_bootstrap"] == 3
    assert {row["split"] for row in policy_rows} == {"validation"}
    assert {row["budget_fraction"] for row in policy_rows} == {0.05}
    assert {row["split"] for row in contrast_rows} == {"validation"}
    assert {row["budget_fraction"] for row in contrast_rows} == {0.05}


def test_omitted_bootstrap_filters_preserve_existing_behavior() -> None:
    """Omitting split and budget filters keeps the existing bootstrap output."""
    frames = _policy_frames()

    default_rows = calculate_bootstrap_policy_rows(
        frames,
        n_bootstrap=3,
        random_seed=42,
    )
    explicit_unfiltered_rows = calculate_bootstrap_policy_rows(
        frames,
        budget_fractions=TOPK_BUDGET_FRACTIONS,
        n_bootstrap=3,
        random_seed=42,
        bootstrap_splits=None,
    )

    assert default_rows == explicit_unfiltered_rows


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


def test_bootstrap_compatibility_imports_match_new_modules() -> None:
    """Legacy bootstrap imports resolve to the refactored implementations."""
    frames = _policy_frames()

    legacy_rows = calculate_bootstrap_policy_rows(
        frames,
        budget_fractions=(0.5, 1.0),
        n_bootstrap=4,
        random_seed=42,
    )
    new_rows = calculate_bootstrap_policy_rows_new(
        frames,
        budget_fractions=(0.5, 1.0),
        n_bootstrap=4,
        random_seed=42,
    )

    assert legacy_rows == new_rows
    assert save_bootstrap_policy_evaluation is save_bootstrap_policy_evaluation_new


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
    """Paired bootstrap rejects frames with mismatched row_id order."""
    frames = _policy_frames_with_row_id()
    frames["t_learner_lgbm"] = frames["t_learner_lgbm"].assign(
        row_id=np.arange(len(frames["t_learner_lgbm"]))[::-1]
    )

    with pytest.raises(ValueError, match="row-aligned.*row_id"):
        calculate_bootstrap_policy_metric_samples(
            frames,
            budget_fractions=(1.0,),
            n_bootstrap=2,
            random_seed=42,
        )


def test_bootstrap_raises_when_only_some_frames_have_row_id() -> None:
    """Mixed row_id availability is rejected because pairing is ambiguous."""
    frames = _policy_frames()
    frames["treated_response_lgbm"] = frames["treated_response_lgbm"].assign(
        row_id=np.arange(len(frames["treated_response_lgbm"]))
    )

    with pytest.raises(ValueError, match="Some policy frames contain row_id"):
        calculate_bootstrap_policy_metric_samples(
            frames,
            budget_fractions=(1.0,),
            n_bootstrap=2,
            random_seed=42,
        )


def test_bootstrap_without_row_id_still_validates_labels() -> None:
    """Without row_id, paired bootstrap still validates treatment/outcome/split."""
    frames = _policy_frames()
    frames["t_learner_lgbm"] = frames["t_learner_lgbm"].assign(
        outcome=1 - frames["t_learner_lgbm"]["outcome"]
    )

    with pytest.raises(ValueError, match="row-aligned"):
        calculate_bootstrap_policy_metric_samples(
            frames,
            budget_fractions=(1.0,),
            n_bootstrap=2,
            random_seed=42,
        )


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
        calculate_policy_value(expected_frame, top_fraction=1.0)
    )
    assert row["incremental_outcome"] == pytest.approx(
        calculate_selected_incremental_outcome(
            expected_frame,
            top_fraction=1.0,
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
    assert len(policy_value_rows) == 8


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
