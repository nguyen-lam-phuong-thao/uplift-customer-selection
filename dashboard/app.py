"""Streamlit customer-targeting decision dashboard."""

from __future__ import annotations

from html import escape
from math import ceil

import plotly.graph_objects as go
import streamlit as st

from data_loader import DashboardData, discover_datasets, load_dashboard_data
from view_model import DashboardView, build_dashboard_view


st.set_page_config(
    page_title="Customer Targeting Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1220px;
            padding-top: 1.15rem;
            padding-bottom: 1.35rem;
        }

        h1 {
            font-size: 2.65rem !important;
            line-height: 1.08 !important;
            letter-spacing: -0.045em;
            margin-bottom: 0.18rem !important;
        }

        h2, h3 {
            letter-spacing: -0.025em;
        }

        .section-note {
            color: #475467;
            font-size: 0.88rem;
            line-height: 1.45;
            margin-top: 0.05rem;
            margin-bottom: 0.65rem;
        }

        .sub-note {
            color: #667085;
            font-size: 0.80rem;
            line-height: 1.35;
            margin-top: -0.25rem;
            margin-bottom: 0.65rem;
        }

        .kpi-card {
            min-height: 142px;
            border: 1px solid #e4e7ec;
            border-radius: 14px;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            padding: 1rem 1.1rem;
        }

        .kpi-title {
            color: #475467;
            font-size: 0.80rem;
            font-weight: 650;
            margin-bottom: 0.60rem;
        }

        .kpi-value {
            color: #101828;
            font-size: 2.00rem;
            font-weight: 760;
            line-height: 1.05;
            letter-spacing: -0.04em;
            margin-bottom: 0.50rem;
        }

        .kpi-purple {
            color: #6938ef;
        }

        .kpi-green {
            color: #039855;
        }

        .kpi-caption {
            color: #667085;
            font-size: 0.78rem;
            line-height: 1.35;
        }

        .validated-badge {
            display: inline-block;
            background: #ecfdf3;
            color: #067647;
            border-radius: 999px;
            padding: 0.16rem 0.52rem;
            font-size: 0.70rem;
            font-weight: 650;
            margin-bottom: 0.45rem;
        }

        .profile-card {
            border: 1px solid #e4e7ec;
            border-radius: 12px;
            background: #ffffff;
            padding: 0.95rem 1rem;
            min-height: 214px;
        }

        .profile-title {
            color: #101828;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .profile-desc {
            color: #667085;
            font-size: 0.76rem;
            line-height: 1.3;
            margin-bottom: 1rem;
        }

        .profile-compare-row {
            display: grid;
            grid-template-columns: 62px 1fr 58px;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 0.55rem;
        }

        .profile-label {
            color: #475467;
            font-size: 0.73rem;
            font-weight: 600;
        }

        .profile-track {
            height: 7px;
            background: #eef0f4;
            border-radius: 999px;
            overflow: hidden;
        }

        .profile-bar-selected {
            height: 100%;
            background: #6938ef;
            border-radius: 999px;
        }

        .profile-bar-overall {
            height: 100%;
            background: #98a2b3;
            border-radius: 999px;
        }

        .profile-value {
            color: #101828;
            font-size: 0.80rem;
            font-weight: 700;
            text-align: right;
            white-space: nowrap;
        }

        .profile-legend {
            display: flex;
            gap: 0.9rem;
            margin-bottom: 0.85rem;
            color: #667085;
            font-size: 0.70rem;
        }

        .profile-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 999px;
            margin-right: 0.3rem;
            vertical-align: middle;
        }

        .profile-dot-selected {
            background: #6938ef;
        }

        .profile-dot-overall {
            background: #98a2b3;
        }

        .profile-insight {
            background: #f5f3ff;
            color: #5925dc;
            border-radius: 9px;
            padding: 0.55rem 0.65rem;
            font-size: 0.76rem;
            line-height: 1.35;
            margin-top: 0.75rem;
        }

        .pager-note {
            height: 2.35rem;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #344054;
            font-size: 0.80rem;
            font-weight: 600;
            line-height: 1;
        }

        .fine-note {
            color: #667085;
            font-size: 0.78rem;
            line-height: 1.35;
            margin-top: 0.4rem;
        }

        .customer-table-wrap {
            border: 1px solid #d0d5dd;
            border-radius: 10px;
            overflow: hidden;
            background: #ffffff;
        }

        .customer-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }

        .customer-table th {
            background: #f8fafc;
            color: #344054;
            font-size: 0.82rem;
            font-weight: 700;
            text-align: left;
            padding: 0.68rem 0.75rem;
            border-bottom: 1px solid #d0d5dd;
        }

        .customer-table td {
            color: #101828;
            font-size: 0.86rem;
            font-weight: 550;
            padding: 0.68rem 0.75rem;
            border-bottom: 1px solid #eaecf0;
            vertical-align: middle;
        }

        .customer-table tr:last-child td {
            border-bottom: none;
        }

        .customer-table th:nth-child(1),
        .customer-table td:nth-child(1) {
            width: 12%;
            text-align: right;
        }

        .customer-table th:nth-child(2),
        .customer-table td:nth-child(2) {
            width: 46%;
        }

        .customer-table th:nth-child(3),
        .customer-table td:nth-child(3) {
            width: 42%;
            text-align: right;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e4e7ec;
            border-radius: 10px;
            overflow: hidden;
        }

        div[data-testid="stDownloadButton"] button,
        div[data-testid="stButton"] button {
            border-radius: 8px;
            min-height: 2.35rem;
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stTextInput"] label,
        div[data-testid="stRadio"] label {
            color: #475467;
            font-size: 0.80rem;
            font-weight: 650;
        }

        hr {
            margin-top: 0.95rem !important;
            margin-bottom: 0.95rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_dashboard_data(dataset_key: str) -> DashboardData:
    """Load one prepared dashboard dataset."""

    return load_dashboard_data(dataset_key)


def dataset_label(dataset_key: str) -> str:
    """Format dataset key for display."""

    return dataset_key.replace("_", " ").title()


def format_number(value: float) -> str:
    """Format numbers compactly."""

    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def render_note(text: str, css_class: str = "section-note") -> None:
    """Render a small descriptive note."""

    st.markdown(
        f'<div class="{css_class}">{text}</div>',
        unsafe_allow_html=True,
    )


def render_kpis(view: DashboardView) -> None:
    """Render the headline KPI cards."""

    col1, col2, col3 = st.columns(3)

    validated = (
        '<div class="validated-badge">Validated</div>'
        if view.validated
        else ""
    )

    with col1:
        st.markdown(
            (
                '<div class="kpi-card">'
                '<div class="kpi-title">Recommended Strategy</div>'
                f'<div class="kpi-value kpi-purple">{view.recommended_policy_label}</div>'
                f'{validated}'
                '<div class="kpi-caption">Chosen targeting strategy for this campaign</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            (
                '<div class="kpi-card">'
                '<div class="kpi-title">Customers to Target</div>'
                f'<div class="kpi-value">{view.n_selected:,}</div>'
                f'<div class="kpi-caption">Top {view.budget_fraction * 100:.0f}% of the evaluation population</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            (
                '<div class="kpi-card">'
                '<div class="kpi-title">Expected Incremental Outcome</div>'
                f'<div class="kpi-value kpi-green">{view.incremental_outcome:+,.0f}</div>'
                '<div class="kpi-caption">Estimated incremental outcome vs. no campaign</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


def render_budget_selector(available_budgets: list[int]) -> int:
    """Render the simple inline Top-K selector like the approved mockup."""

    default_budget = 5 if 5 in available_budgets else available_budgets[0]

    st.subheader("Targeting Budget (Top-K)")
    render_note(
        "Choose the percentage of customers to target in this campaign.",
        "section-note",
    )

    selected_budget = st.radio(
        "Targeting Budget",
        options=available_budgets,
        index=available_budgets.index(default_budget),
        format_func=lambda value: f"{value}%",
        horizontal=True,
        label_visibility="collapsed",
    )

    return int(selected_budget)


def render_impact_chart(view: DashboardView) -> None:
    """Render expected incremental outcome across targeting budgets."""

    with st.container(border=True):
        st.subheader("Expected Incremental Outcome by Targeting Budget")
        render_note(
            "Compare the recommended strategy with the baseline across the available targeting budgets.",
            "sub-note",
        )

        figure = go.Figure()

        for policy_label, rows in view.chart_data.groupby(
            "policy_label",
            sort=False,
        ):
            rows = rows.sort_values("budget_pct")
            is_recommended = policy_label == view.recommended_policy_label

            figure.add_trace(
                go.Scatter(
                    x=rows["budget_pct"],
                    y=rows["incremental_outcome"],
                    mode="lines+markers+text",
                    text=[
                        f"{value:,.0f}"
                        for value in rows["incremental_outcome"]
                    ],
                    textposition="top center",
                    cliponaxis=False,
                    name=(
                        f"{policy_label} (Recommended)"
                        if is_recommended
                        else f"{policy_label} (Baseline)"
                    ),
                    line={
                        "width": 3 if is_recommended else 2,
                        "color": "#6938ef" if is_recommended else "#98a2b3",
                    },
                    marker={
                        "size": 8 if is_recommended else 7,
                        "color": "#6938ef" if is_recommended else "#98a2b3",
                    },
                    textfont={
                        "size": 11,
                        "color": "#6938ef" if is_recommended else "#344054",
                    },
                )
            )

        figure.add_vline(
            x=view.budget_fraction * 100,
            line_width=1,
            line_dash="dash",
            line_color="#6938ef",
            opacity=0.42,
        )

        figure.update_layout(
            height=310,
            margin={"l": 10, "r": 10, "t": 18, "b": 5},
            template="plotly_white",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
                "font": {"size": 10},
            },
            xaxis={
                "title": "Targeting Budget (Top-K)",
                "ticksuffix": "%",
                "tickmode": "array",
                "tickvals": sorted(
                    view.chart_data["budget_pct"].unique().tolist()
                ),
                "gridcolor": "#f2f4f7",
                "zeroline": False,
            },
            yaxis={
                "title": "Incremental Outcome",
                "rangemode": "tozero",
                "gridcolor": "#f2f4f7",
                "zeroline": False,
            },
            hovermode="x unified",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            config={"displayModeBar": False},
        )


def render_customer_table(view: DashboardView) -> None:
    """Render five selected customers per page with simple styling."""

    budget_pct = int(round(view.budget_fraction * 100))

    with st.container(border=True):
        title_col, download_col = st.columns([4.7, 1.1])

        with title_col:
            st.subheader(f"Recommended Customers (Top {budget_pct}%)")
            render_note(
                f"Ranked by {view.score_label.lower()} from {view.recommended_policy_label}.",
                "sub-note",
            )

        with download_col:
            st.download_button(
                "Download CSV",
                data=view.customer_table.to_csv(index=False).encode("utf-8"),
                file_name=f"recommended_customers_top_{budget_pct}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        search_value = st.text_input(
            "Search customer ID",
            placeholder="Search customer ID...",
            label_visibility="collapsed",
            key=f"customer_search_{view.recommended_policy}_{budget_pct}",
        )

        filtered_table = view.customer_table

        if search_value.strip():
            filtered_table = filtered_table[
                filtered_table["Customer ID"]
                .astype(str)
                .str.contains(search_value.strip(), case=False, na=False)
            ]

        rows_per_page = 5
        total_rows = len(filtered_table)
        total_pages = max(1, ceil(total_rows / rows_per_page))

        page_key = f"customer_page_{view.recommended_policy}_{budget_pct}"
        current_page = int(st.session_state.get(page_key, 1))
        current_page = min(max(current_page, 1), total_pages)
        st.session_state[page_key] = current_page

        start = (current_page - 1) * rows_per_page
        end = start + rows_per_page
        page_table = filtered_table.iloc[start:end].copy()

        rows_html = "".join(
            (
                "<tr>"
                f"<td>{int(row['Rank'])}</td>"
                f"<td>{escape(str(row['Customer ID']))}</td>"
                f"<td>{float(row[view.score_label]):.4f}</td>"
                "</tr>"
            )
            for _, row in page_table.iterrows()
        )

        table_html = (
            '<div class="customer-table-wrap">'
            '<table class="customer-table">'
            "<thead><tr>"
            "<th>Rank</th>"
            "<th>Customer ID</th>"
            f"<th>{escape(view.score_label)}</th>"
            "</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
            "</div>"
        )

        st.markdown(
            table_html,
            unsafe_allow_html=True,
        )

        prev_col, page_col, next_col, spacer = st.columns([1, 1.35, 1, 4.3])

        with prev_col:
            if st.button(
                "← Previous",
                disabled=current_page <= 1,
                use_container_width=True,
                key=f"previous_{page_key}",
            ):
                st.session_state[page_key] = current_page - 1
                st.rerun()

        with page_col:
            st.markdown(
                f'<div class="pager-note">Page {current_page} of {total_pages}</div>',
                unsafe_allow_html=True,
            )

        with next_col:
            if st.button(
                "Next →",
                disabled=current_page >= total_pages,
                use_container_width=True,
                key=f"next_{page_key}",
            ):
                st.session_state[page_key] = current_page + 1
                st.rerun()

        shown_start = 0 if total_rows == 0 else start + 1
        shown_end = min(end, total_rows)

        st.markdown(
            (
                f'<div class="fine-note">Showing {shown_start:,}–{shown_end:,} '
                f'of {total_rows:,} selected customers. This is evaluation evidence, '
                'not a live production campaign audience.</div>'
            ),
            unsafe_allow_html=True,
        )


def render_profile(view: DashboardView) -> None:
    """Render scannable selected-vs-overall customer profile cards."""

    if not view.profile_metrics:
        return

    budget_pct = int(round(view.budget_fraction * 100))

    with st.container(border=True):
        st.subheader("Who are we targeting?")
        render_note(
            f"Selected Top {budget_pct}% compared with the overall evaluation population.",
            "sub-note",
        )

        columns = st.columns(len(view.profile_metrics))

        for column, metric in zip(columns, view.profile_metrics):
            selected_value = float(metric.targeted_value)
            overall_value = float(metric.overall_value)
            scale = max(abs(selected_value), abs(overall_value), 1e-9)

            selected_width = max(
                5.0,
                min(100.0, abs(selected_value) / scale * 100),
            )
            overall_width = max(
                5.0,
                min(100.0, abs(overall_value) / scale * 100),
            )

            card_html = (
                '<div class="profile-card">'
                f'<div class="profile-title">{metric.title}</div>'
                f'<div class="profile-desc">{metric.description}</div>'
                '<div class="profile-legend">'
                '<span><span class="profile-dot profile-dot-selected"></span>Selected</span>'
                '<span><span class="profile-dot profile-dot-overall"></span>Overall</span>'
                '</div>'
                '<div class="profile-compare-row">'
                '<div class="profile-label">Selected</div>'
                '<div class="profile-track">'
                f'<div class="profile-bar-selected" style="width:{selected_width:.1f}%"></div>'
                '</div>'
                f'<div class="profile-value">{format_number(selected_value)}</div>'
                '</div>'
                '<div class="profile-compare-row">'
                '<div class="profile-label">Overall</div>'
                '<div class="profile-track">'
                f'<div class="profile-bar-overall" style="width:{overall_width:.1f}%"></div>'
                '</div>'
                f'<div class="profile-value">{format_number(overall_value)}</div>'
                '</div>'
                f'<div class="profile-insight">{metric.insight}</div>'
                '</div>'
            )

            with column:
                st.markdown(card_html, unsafe_allow_html=True)


dataset_keys = discover_datasets()

if not dataset_keys:
    st.error("No dashboard dataset was found in dashboard_data/.")
    st.stop()


title_col, selector_col = st.columns([4.5, 1.2])

with title_col:
    st.title("Customer Targeting Dashboard")
    render_note(
        "Identify customers most likely to generate incremental outcome from a campaign.",
        "section-note",
    )

with selector_col:
    selected_dataset = st.selectbox(
        "Dataset",
        options=dataset_keys,
        format_func=dataset_label,
    )


dashboard_data = get_dashboard_data(selected_dataset)

available_budgets = sorted(
    {
        int(round(float(row["budget_fraction"]) * 100))
        for row in dashboard_data.locked_test["locked_test_rows"]
    }
)

selected_budget_pct = render_budget_selector(available_budgets)

view = build_dashboard_view(
    dashboard_data,
    selected_budget_pct / 100,
)


render_kpis(view)

st.divider()

render_impact_chart(view)

st.divider()

render_customer_table(view)

st.divider()

render_profile(view)
