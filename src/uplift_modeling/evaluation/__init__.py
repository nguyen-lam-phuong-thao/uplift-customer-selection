"""Evaluation utilities for binary and uplift-modeling metrics."""

from uplift_modeling.evaluation.binary_metrics import calculate_binary_metrics
from uplift_modeling.evaluation.bootstrap import DEFAULT_N_BOOTSTRAP
from uplift_modeling.evaluation.bootstrap_summary import (
    calculate_bootstrap_policy_rows,
)
from uplift_modeling.evaluation.selection_gate import (
    SelectionGateSettings,
    select_champion_from_bootstrap_payload,
    save_model_selection_gate,
)
from uplift_modeling.evaluation.topk_policy import (
    TOPK_BUDGET_FRACTIONS,
    calculate_topk_policy_metrics,
)
from uplift_modeling.evaluation.uplift_metrics import (
    build_uplift_curve,
    calculate_auuc,
    calculate_policy_value,
    calculate_qini,
    calculate_selected_incremental_outcome,
    calculate_uplift_metrics,
    validate_prediction_frame,
)

__all__ = [
    "DEFAULT_N_BOOTSTRAP",
    "TOPK_BUDGET_FRACTIONS",
    "build_uplift_curve",
    "calculate_auuc",
    "calculate_binary_metrics",
    "calculate_bootstrap_policy_rows",
    "calculate_policy_value",
    "calculate_qini",
    "calculate_selected_incremental_outcome",
    "calculate_topk_policy_metrics",
    "calculate_uplift_metrics",
    "save_model_selection_gate",
    "select_champion_from_bootstrap_payload",
    "SelectionGateSettings",
    "validate_prediction_frame",
]
