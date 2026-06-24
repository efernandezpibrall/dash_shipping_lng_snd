"""Small shared components for balance-style pages."""

from __future__ import annotations

from dash import html


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
