"""Transform framework artifacts into business-facing dashboard values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from data_loader import DashboardData


POLICY_LABELS = {
    "t_learner_lgbm": "T-Learner",
    "x_learner_lgbm": "X-Learner",
    "treated_response_lgbm": "Response Targeting",
    "random_targeting": "Random Targeting",
}


@dataclass(frozen=True)
class ProfileMetric:
    """One selected-vs-overall customer profile comparison."""

    title: str
    description: str
    targeted_value: float
    overall_value: float
    insight: str


@dataclass(frozen=True)
class DashboardView:
    """Business-facing values rendered by the dashboard."""

    available_budgets: tuple[float, ...]
    budget_fraction: float
    recommended_policy: str
    recommended_policy_label: str
    validated: bool
    n_selected: int
    incremental_outcome: float
    chart_data: pd.DataFrame
    customer_table: pd.DataFrame
    profile_metrics: tuple[ProfileMetric, ...]
    score_label: str


def policy_label(policy_name: str) -> str:
    """Return a human-readable policy label."""

    return POLICY_LABELS.get(
        policy_name,
        policy_name.replace("_", " ").title(),
    )


def _get_policy_column(rows: pd.DataFrame) -> str:
    for column in ("policy", "policy_name", "model_name"):
        if column in rows.columns:
            return column

    raise ValueError(
        "Locked-test rows must contain policy, policy_name, or model_name."
    )


def _get_budget_row(
    rows: pd.DataFrame,
    policy_column: str,
    policy_name: str,
    budget_fraction: float,
) -> pd.Series:
    matches = rows[
        (rows[policy_column] == policy_name)
        & (rows["budget_fraction"] - budget_fraction).abs().lt(1e-9)
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one locked-test row for "
            f"policy={policy_name!r}, budget={budget_fraction}, "
            f"found {len(matches)}."
        )

    return matches.iloc[0]


def _profile_specs(
    dashboard_config: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_specs = dashboard_config.get("profile_metrics", [])

    if raw_specs is None:
        return []

    if not isinstance(raw_specs, list):
        raise ValueError(
            "dashboard profile_metrics must be a list."
        )

    specs: list[dict[str, Any]] = []

    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError(
                "Each dashboard profile metric must be a JSON object."
            )

        if "column" not in raw_spec or "title" not in raw_spec:
            raise ValueError(
                "Each dashboard profile metric requires column and title."
            )

        specs.append(raw_spec)

    return specs


def _comparison_insight(
    spec: dict[str, Any],
    targeted_value: float,
    overall_value: float,
) -> str:
    if targeted_value < overall_value:
        return str(
            spec.get(
                "when_lower",
                f"Targeted customers have lower {str(spec['title']).lower()}.",
            )
        )

    if targeted_value > overall_value:
        return str(
            spec.get(
                "when_higher",
                f"Targeted customers have higher {str(spec['title']).lower()}.",
            )
        )

    return str(
        spec.get(
            "when_equal",
            "Targeted and overall customers are similar.",
        )
    )


def _build_profile_metrics(
    selected_customers: pd.DataFrame,
    overall_population: pd.DataFrame,
    dashboard_config: dict[str, Any],
) -> tuple[ProfileMetric, ...]:
    metrics: list[ProfileMetric] = []

    for spec in _profile_specs(dashboard_config):
        column_name = str(spec["column"])

        if column_name not in overall_population.columns:
            raise ValueError(
                f"Configured profile column {column_name!r} "
                "does not exist in the decision dataset."
            )

        targeted_value = float(
            selected_customers[column_name].median()
        )
        overall_value = float(
            overall_population[column_name].median()
        )

        metrics.append(
            ProfileMetric(
                title=str(spec["title"]),
                description=str(spec.get("description", "")),
                targeted_value=targeted_value,
                overall_value=overall_value,
                insight=_comparison_insight(
                    spec,
                    targeted_value,
                    overall_value,
                ),
            )
        )

    return tuple(metrics)


def build_dashboard_view(
    data: DashboardData,
    budget_fraction: float,
) -> DashboardView:
    """Build all dashboard values for one targeting-budget scenario."""

    locked_rows = pd.DataFrame(
        data.locked_test["locked_test_rows"]
    ).copy()

    required_columns = {
        "budget_fraction",
        "n_selected",
        "incremental_outcome",
    }

    missing_columns = required_columns.difference(
        locked_rows.columns
    )

    if missing_columns:
        raise ValueError(
            "Locked-test rows are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    policy_column = _get_policy_column(locked_rows)

    available_budgets = tuple(
        sorted(
            float(value)
            for value in locked_rows["budget_fraction"].unique()
        )
    )

    if not any(
        abs(budget_fraction - value) < 1e-9
        for value in available_budgets
    ):
        raise ValueError(
            f"Unsupported targeting budget: {budget_fraction}"
        )

    recommended_policy = str(
        data.selection["recommended_deployment_policy"]
    )
    baseline_policy = str(
        data.selection["baseline_policy"]
    )

    recommended_row = _get_budget_row(
        locked_rows,
        policy_column,
        recommended_policy,
        budget_fraction,
    )

    n_selected = int(recommended_row["n_selected"])
    incremental_outcome = float(
        recommended_row["incremental_outcome"]
    )

    comparison_policies = [recommended_policy]

    if baseline_policy != recommended_policy:
        comparison_policies.append(baseline_policy)
    else:
        uplift_champion = data.selection.get(
            "uplift_champion_policy"
        )

        if (
            uplift_champion
            and uplift_champion != recommended_policy
        ):
            comparison_policies.append(str(uplift_champion))

    chart_data = locked_rows[
        locked_rows[policy_column].isin(comparison_policies)
    ][
        [
            policy_column,
            "budget_fraction",
            "incremental_outcome",
        ]
    ].copy()

    chart_data["budget_pct"] = (
        chart_data["budget_fraction"] * 100
    ).round().astype(int)

    chart_data["policy_label"] = chart_data[
        policy_column
    ].map(policy_label)

    prediction_data = data.predictions[
        recommended_policy
    ].copy()

    if "split" in prediction_data.columns:
        prediction_data = prediction_data[
            prediction_data["split"] == "test"
        ].copy()

    decision_test = data.decision_data[
        data.decision_data["split"] == "test"
    ].copy()

    profile_columns = [
        str(spec["column"])
        for spec in _profile_specs(data.dashboard_config)
    ]

    decision_columns = list(
        dict.fromkeys(
            [
                "row_id",
                data.entity_id_column,
                *profile_columns,
            ]
        )
    )

    population = prediction_data[
        ["row_id", "score"]
    ].merge(
        decision_test[decision_columns],
        on="row_id",
        how="left",
        validate="one_to_one",
    )

    if population[data.entity_id_column].isna().any():
        raise ValueError(
            "Some locked-test predictions could not be joined "
            "to the decision dataset."
        )

    population = population.sort_values(
        "score",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    selected_customers = population.head(
        n_selected
    ).copy()

    selected_customers.insert(
        0,
        "Rank",
        range(1, len(selected_customers) + 1),
    )

    score_label = (
        "Response Score"
        if recommended_policy == "treated_response_lgbm"
        else "Uplift Score"
    )

    customer_table = selected_customers[
        [
            "Rank",
            data.entity_id_column,
            "score",
        ]
    ].rename(
        columns={
            data.entity_id_column: "Customer ID",
            "score": score_label,
        }
    )

    profile_metrics = _build_profile_metrics(
        selected_customers=selected_customers,
        overall_population=population,
        dashboard_config=data.dashboard_config,
    )

    return DashboardView(
        available_budgets=available_budgets,
        budget_fraction=budget_fraction,
        recommended_policy=recommended_policy,
        recommended_policy_label=policy_label(
            recommended_policy
        ),
        validated=bool(
            data.selection["replacement_gate_passed"]
        ),
        n_selected=n_selected,
        incremental_outcome=incremental_outcome,
        chart_data=chart_data,
        customer_table=customer_table,
        profile_metrics=profile_metrics,
        score_label=score_label,
    )
