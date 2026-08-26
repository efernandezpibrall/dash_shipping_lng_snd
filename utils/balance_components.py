"""Small shared components for balance-style pages."""

from __future__ import annotations

from dash import dcc, html


def create_balance_empty_state(message: str) -> html.Div:
    return html.Div(message, className="balance-empty-state")


def build_balance_section_summary(
    raw_df,
    matrix_df,
    destination_aggregation: str,
    other_countries_mode: str,
    metadata_lines: list[str] | None = None,
    time_view: str = "monthly",
    *,
    time_view_period_labels: dict[str, str],
    destination_aggregation_labels: dict[str, str],
) -> html.Div:
    summary_children = []

    if raw_df.empty:
        summary_children.append(
            html.Div("No source data returned.", className="balance-summary-row")
        )
    else:
        month_start = (
            matrix_df["Month"].iloc[0]
            if not matrix_df.empty
            else raw_df["month"].min().strftime("%Y-%m")
        )
        month_end = (
            matrix_df["Month"].iloc[-1]
            if not matrix_df.empty
            else raw_df["month"].max().strftime("%Y-%m")
        )
        visible_country_count = max(len(matrix_df.columns) - 2, 0)
        standardized_country_count = raw_df["country_name"].nunique()
        period_label = time_view_period_labels.get(time_view, "month")
        count_label = period_label if len(matrix_df) == 1 else f"{period_label}s"
        aggregation_label = destination_aggregation_labels.get(
            destination_aggregation,
            "Country",
        )
        visible_column_label = (
            "visible country columns"
            if destination_aggregation == "country"
            else f"visible {aggregation_label.lower()} columns"
        )
        visibility_note = (
            "Other countries grouped into Rest of the World."
            if destination_aggregation == "country"
            and other_countries_mode == "rest_of_world"
            else (
                "Only selected countries are included in the totals."
                if destination_aggregation == "country"
                else (
                    f"Destination aggregation: {aggregation_label}. "
                    f"All groups shown as explicit columns."
                )
            )
        )

        summary_children.append(
            html.Div(
                [
                    html.Span(f"{len(matrix_df):,} {count_label}"),
                    html.Span(f"{month_start} to {month_end}"),
                    html.Span(f"{visible_country_count} {visible_column_label}"),
                    html.Span(f"{standardized_country_count} standardized source countries"),
                    html.Span(visibility_note),
                ],
                className="balance-summary-row",
            )
        )

    if metadata_lines:
        summary_children.append(
            html.Div(
                [html.Span(line) for line in metadata_lines],
                className="balance-metadata-row",
            )
        )

    return html.Div(summary_children)


EXPORT_BUTTON_STYLE = {
    "marginLeft": "20px",
    "padding": "5px 15px",
    "backgroundColor": "#28a745",
    "color": "white",
    "border": "none",
    "borderRadius": "4px",
    "cursor": "pointer",
    "fontWeight": "bold",
    "fontSize": "12px",
}


def create_balance_comparison_section(
    title: str,
    export_button_id: str,
    baseline_summary_id: str,
    default_comparison_source: str,
    comparison_source_dropdown_id: str,
    comparison_st_dropdown_id: str,
    comparison_lt_dropdown_id: str,
    comparison_ea_upload_dropdown_id: str,
    comparison_woodmac_controls_id: str,
    comparison_ea_controls_id: str,
    baseline_table_container_id: str,
    comparison_summary_id: str,
    comparison_table_container_id: str,
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(title, className="balance-section-title"),
                            html.Button(
                                "Export to Excel",
                                id=export_button_id,
                                n_clicks=0,
                                style=EXPORT_BUTTON_STYLE,
                            ),
                        ],
                        className="inline-section-header",
                        style={"display": "flex", "alignItems": "center"},
                    ),
                ],
                className="balance-section-header",
            ),
            html.Div(
                [
                    html.Div(
                        "Baseline Table",
                        className="balance-panel-title balance-panel-title-left",
                    ),
                    html.Div(
                        "Delta vs Selected Snapshot",
                        className="balance-panel-title balance-panel-title-right",
                        title="Delta formula: left baseline table - selected snapshot",
                        style={
                            "textDecoration": "underline dotted",
                            "textUnderlineOffset": "3px",
                            "cursor": "help",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                id=baseline_summary_id,
                                className="balance-pane-summary",
                            )
                        ],
                        className="balance-pane-top-area balance-pane-top-area-left",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Source:",
                                        htmlFor=comparison_source_dropdown_id,
                                        className="filter-label",
                                    ),
                                    dcc.Dropdown(
                                        id=comparison_source_dropdown_id,
                                        options=[
                                            {
                                                "label": "WoodMac",
                                                "value": "woodmac",
                                            },
                                            {
                                                "label": "Energy Aspects",
                                                "value": "ea",
                                            },
                                        ],
                                        value=default_comparison_source,
                                        clearable=False,
                                        className="filter-dropdown",
                                        style={"minWidth": "180px"},
                                    ),
                                ],
                                className="filter-group",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "ST publication:",
                                                htmlFor=comparison_st_dropdown_id,
                                                className="filter-label",
                                            ),
                                            dcc.Dropdown(
                                                id=comparison_st_dropdown_id,
                                                options=[],
                                                value=None,
                                                clearable=False,
                                                className="filter-dropdown",
                                                style={"minWidth": "260px"},
                                            ),
                                        ],
                                        className="filter-group",
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "LT publication:",
                                                htmlFor=comparison_lt_dropdown_id,
                                                className="filter-label",
                                            ),
                                            dcc.Dropdown(
                                                id=comparison_lt_dropdown_id,
                                                options=[],
                                                value=None,
                                                clearable=False,
                                                className="filter-dropdown",
                                                style={"minWidth": "260px"},
                                            ),
                                        ],
                                        className="filter-group",
                                    ),
                                ],
                                id=comparison_woodmac_controls_id,
                                className="balance-comparison-control-row",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "upload_timestamp_utc:",
                                                htmlFor=comparison_ea_upload_dropdown_id,
                                                className="filter-label",
                                            ),
                                            dcc.Dropdown(
                                                id=comparison_ea_upload_dropdown_id,
                                                options=[],
                                                value=None,
                                                clearable=False,
                                                className="filter-dropdown",
                                                style={"minWidth": "280px"},
                                            ),
                                        ],
                                        className="filter-group",
                                    ),
                                ],
                                id=comparison_ea_controls_id,
                                className="balance-comparison-control-row",
                            ),
                        ],
                        className="balance-comparison-controls balance-pane-top-area balance-pane-top-area-right",
                    ),
                    html.Div(
                        id=baseline_table_container_id,
                        className="balance-table-container balance-table-container-left",
                    ),
                    html.Div(
                        [
                            html.Div(id=comparison_summary_id),
                            html.Div(
                                id=comparison_table_container_id,
                                className="balance-table-container",
                            ),
                        ],
                        className="balance-table-shell balance-table-shell-right",
                    ),
                ],
                className="balance-comparison-grid",
            ),
        ],
        className="balance-section-card",
    )
