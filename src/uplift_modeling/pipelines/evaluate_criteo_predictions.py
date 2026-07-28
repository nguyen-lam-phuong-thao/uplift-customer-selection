"""Evaluate Criteo model prediction artifacts with uplift metrics."""

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from uplift_modeling.artifacts.json import save_json_artifact
from uplift_modeling.artifacts.naming import (
    build_artifact_filename,
    build_model_comparison_name,
    find_latest_prediction_paths,
    find_next_run_number,
)
from uplift_modeling.evaluation.bootstrap import (
    BOOTSTRAP_METRICS,
    DEFAULT_BASELINE_POLICY,
    DEFAULT_N_BOOTSTRAP,
)
from uplift_modeling.evaluation.bootstrap_writer import (
    save_bootstrap_policy_evaluation,
)
from uplift_modeling.evaluation.selection_gate import (
    SelectionGateSettings,
    save_model_selection_gate,
)
from uplift_modeling.evaluation.topk_policy import (
    TOPK_BUDGET_FRACTIONS,
    prepare_topk_policy_frames,
    save_topk_policy_evaluation_artifacts,
)
from uplift_modeling.evaluation.uplift_metrics import (
    PREDICTION_COLUMNS,
    build_uplift_curve,
    calculate_uplift_metrics,
    validate_prediction_frame,
)
from uplift_modeling.utils.config import (
    get_config_section,
    get_project_root,
    load_yaml_config,
    resolve_project_path,
)


LOGGER = logging.getLogger(__name__)
VALID_OUTCOMES = ("visit", "conversion")
EVALUATION_SPLITS = ("validation", "test")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Criteo prediction evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Criteo prediction artifacts."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the response-model YAML config.",
    )
    parser.add_argument(
        "--outcome",
        default="visit",
        choices=VALID_OUTCOMES,
        help="Outcome to evaluate. Defaults to visit.",
    )
    parser.add_argument(
        "--top-fraction",
        default=0.3,
        type=float,
        help="Top ranked share used for policy value.",
    )
    parser.add_argument(
        "--curve-num-points",
        default=100,
        type=int,
        help="Maximum number of points to plot in each curve.",
    )
    parser.add_argument(
        "--random-seed",
        default=42,
        type=int,
        help="Random seed for deterministic random targeting.",
    )
    parser.add_argument(
        "--n-bootstrap",
        default=DEFAULT_N_BOOTSTRAP,
        type=int,
        help="Number of bootstrap resamples for Top-K confidence intervals.",
    )
    parser.add_argument(
        "--topk-only",
        action="store_true",
        help="Only save Top-K policy and bootstrap evaluation artifacts.",
    )
    return parser.parse_args()


def get_prediction_paths(
    prediction_dir: Path,
    db_name: str,
    outcome: str,
) -> list[Path]:
    """Return the latest matching prediction artifacts for one outcome."""
    return find_latest_prediction_paths(
        prediction_dir=prediction_dir,
        db_name=db_name,
        outcome=outcome,
    )


def load_prediction_artifacts(prediction_paths: list[Path]) -> pd.DataFrame:
    """Load and validate prediction artifacts."""
    frames = []

    for prediction_path in prediction_paths:
        LOGGER.info("Loading predictions from %s", prediction_path)
        frame = pd.read_parquet(prediction_path)
        missing_columns = sorted(
            set(PREDICTION_COLUMNS).difference(frame.columns)
        )

        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(
                f"Prediction artifact {prediction_path} is missing columns: "
                f"{missing_text}"
            )

        frame = frame.copy()
        frame["artifact_name"] = prediction_path.name
        frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True)
    return predictions


def filter_evaluation_splits(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep only validation and test prediction rows for evaluation."""
    evaluation_predictions = predictions.loc[
        predictions["split"].isin(EVALUATION_SPLITS)
    ].copy()

    if evaluation_predictions.empty:
        split_text = ", ".join(EVALUATION_SPLITS)
        raise ValueError(
            f"No prediction rows found for evaluation splits: {split_text}"
        )

    return evaluation_predictions


def calculate_split_metrics(
    predictions: pd.DataFrame,
    top_fraction: float,
) -> list[dict[str, Any]]:
    """Calculate uplift metrics for each model and evaluation split."""
    rows: list[dict[str, Any]] = []

    for (model_name, split), group in predictions.groupby(
        ["model_name", "split"],
        sort=True,
    ):
        validate_prediction_frame(group)
        metrics = calculate_uplift_metrics(
            group,
            top_fraction=top_fraction,
        )
        rows.append(
            {
                "model_name": str(model_name),
                "split": str(split),
                **metrics,
            }
        )

    if not rows:
        split_text = ", ".join(EVALUATION_SPLITS)
        raise ValueError(
            f"No prediction rows found for evaluation splits: {split_text}"
        )

    return rows


def get_selection_gate_settings(
    config: dict[str, Any],
    outcome: str,
) -> SelectionGateSettings:
    """Return configured primary model-selection settings."""
    selection_config = config.get("selection", {})
    if selection_config is None:
        selection_config = {}
    if not isinstance(selection_config, dict):
        raise ValueError(
            "Config section 'selection' must be a mapping when present."
        )

    return SelectionGateSettings(
        outcome=str(selection_config.get("primary_outcome", outcome)),
        split=str(selection_config.get("primary_split", EVALUATION_SPLITS[0])),
        budget_fraction=float(
            selection_config.get(
                "primary_budget_fraction",
                TOPK_BUDGET_FRACTIONS[1],
            )
        ),
        metric=str(selection_config.get("primary_metric", BOOTSTRAP_METRICS[0])),
        baseline_policy=str(
            selection_config.get("baseline_policy", DEFAULT_BASELINE_POLICY)
        ),
    )


def save_qini_curve(
    predictions: pd.DataFrame,
    output_path: Path,
    curve_num_points: int,
) -> Path:
    """Save a Qini curve figure for validation and test predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))

    for (model_name, split), group in predictions.groupby(
        ["model_name", "split"],
        sort=True,
    ):
        curve = build_uplift_curve(group, num_points=curve_num_points)
        final_incremental_outcome = curve[
            "cumulative_incremental_outcome"
        ].iloc[-1]
        random_baseline = (
            curve["population_fraction"] * final_incremental_outcome
        )
        label = f"{model_name} {split}"
        axis.plot(
            curve["population_fraction"],
            curve["cumulative_incremental_outcome"],
            label=label,
        )
        axis.plot(
            curve["population_fraction"],
            random_baseline,
            linestyle="--",
            alpha=0.35,
            label=f"{label} random",
        )

    axis.set_title("Qini Curve")
    axis.set_xlabel("Population fraction")
    axis.set_ylabel("Cumulative incremental outcome")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_uplift_curve(
    predictions: pd.DataFrame,
    output_path: Path,
    curve_num_points: int,
) -> Path:
    """Save an uplift curve figure for validation and test predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))

    for (model_name, split), group in predictions.groupby(
        ["model_name", "split"],
        sort=True,
    ):
        curve = build_uplift_curve(group, num_points=curve_num_points)
        axis.plot(
            curve["population_fraction"],
            curve["uplift"],
            label=f"{model_name} {split}",
        )

    axis.set_title("Uplift Curve")
    axis.set_xlabel("Population fraction")
    axis.set_ylabel("Observed outcome-rate difference")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def evaluate_predictions(
    config_path: Path,
    outcome: str,
    top_fraction: float,
    curve_num_points: int,
    random_seed: int,
    n_bootstrap: int,
    topk_only: bool,
) -> None:
    """Evaluate local Criteo prediction artifacts."""
    project_root = get_project_root(Path(__file__))
    config = load_yaml_config(config_path)
    data_config = get_config_section(config, "data")
    output_config = get_config_section(config, "outputs")
    dataset_name = str(data_config["dataset_name"])

    prediction_dir = resolve_project_path(
        output_config["prediction_dir"],
        project_root,
    )
    metric_dir = resolve_project_path(
        output_config["metric_dir"],
        project_root,
    )
    figure_dir = resolve_project_path(
        output_config["figure_dir"],
        project_root,
    )

    policy_frames, policy_paths, topk_warnings = prepare_topk_policy_frames(
        prediction_dir=prediction_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        random_seed=random_seed,
    )
    save_topk_policy_evaluation_artifacts(
        policy_frames=policy_frames,
        policy_paths=policy_paths,
        warnings=topk_warnings,
        metric_dir=metric_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        random_seed=random_seed,
    )
    _, bootstrap_contrast_path, bootstrap_payload = save_bootstrap_policy_evaluation(
        policy_frames=policy_frames,
        metric_dir=metric_dir,
        dataset_name=dataset_name,
        outcome=outcome,
        random_seed=random_seed,
        n_bootstrap=n_bootstrap,
        prediction_artifacts=policy_paths,
        warnings=topk_warnings,
    )
    save_model_selection_gate(
        metric_dir=metric_dir,
        dataset_name=dataset_name,
        settings=get_selection_gate_settings(config, outcome=outcome),
        bootstrap_payload=bootstrap_payload,
        bootstrap_json_path=bootstrap_contrast_path,
    )

    if topk_only:
        return

    prediction_paths = get_prediction_paths(
        prediction_dir=prediction_dir,
        db_name=dataset_name,
        outcome=outcome,
    )
    model_comparison_name = build_model_comparison_name(prediction_paths)
    run_number = find_next_run_number(
        artifact_dirs=(metric_dir, figure_dir),
        db_name=dataset_name,
        outcome=outcome,
        model_name=model_comparison_name,
    )
    metrics_path = resolve_project_path(
        metric_dir
        / build_artifact_filename(
            db_name=dataset_name,
            outcome=outcome,
            model_name=model_comparison_name,
            run_number=run_number,
            artifact_name="evaluation",
            extension="json",
        ),
        project_root,
    )
    qini_curve_path = resolve_project_path(
        figure_dir
        / build_artifact_filename(
            db_name=dataset_name,
            outcome=outcome,
            model_name=model_comparison_name,
            run_number=run_number,
            artifact_name="qini_curve",
            extension="png",
        ),
        project_root,
    )
    uplift_curve_path = resolve_project_path(
        figure_dir
        / build_artifact_filename(
            db_name=dataset_name,
            outcome=outcome,
            model_name=model_comparison_name,
            run_number=run_number,
            artifact_name="uplift_curve",
            extension="png",
        ),
        project_root,
    )

    predictions = filter_evaluation_splits(
        load_prediction_artifacts(prediction_paths)
    )
    split_metrics = calculate_split_metrics(
        predictions,
        top_fraction=top_fraction,
    )
    payload = {
        "outcome": outcome,
        "top_fraction": top_fraction,
        "curve_num_points": curve_num_points,
        "prediction_artifacts": [
            prediction_path.name for prediction_path in prediction_paths
        ],
        "metrics": split_metrics,
    }

    save_json_artifact(payload, metrics_path)
    save_qini_curve(
        predictions,
        qini_curve_path,
        curve_num_points=curve_num_points,
    )
    save_uplift_curve(
        predictions,
        uplift_curve_path,
        curve_num_points=curve_num_points,
    )

    LOGGER.info("Saved evaluation metrics to %s", metrics_path)
    LOGGER.info("Saved Qini curve to %s", qini_curve_path)
    LOGGER.info("Saved uplift curve to %s", uplift_curve_path)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    project_root = get_project_root(Path(__file__))
    config_path = resolve_project_path(args.config, project_root)
    evaluate_predictions(
        config_path=config_path,
        outcome=args.outcome,
        top_fraction=args.top_fraction,
        curve_num_points=args.curve_num_points,
        random_seed=args.random_seed,
        n_bootstrap=args.n_bootstrap,
        topk_only=args.topk_only,
    )


if __name__ == "__main__":
    main()
