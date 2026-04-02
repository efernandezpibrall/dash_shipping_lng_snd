from io import BytesIO, StringIO
import datetime as dt
import json

import pandas as pd
from dash import dcc, html, dash_table, callback, Input, Output, State
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate

from utils.export_flow_data import (
    build_export_flow_matrix,
    default_selected_countries,
    fetch_ea_export_flow_raw_data,
    fetch_ea_export_flow_raw_data_for_upload,
    fetch_ea_export_flow_metadata,
    fetch_ea_upload_options,
    fetch_woodmac_export_flow_raw_data_for_publications,
    fetch_woodmac_export_flow_raw_data,
    fetch_woodmac_export_flow_metadata,
    fetch_woodmac_publication_options,
    get_available_countries,
)
from utils.table_styles import StandardTableStyleManager, TABLE_COLORS


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


def _serialize_dataframe(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None

    return df.to_json(date_format="iso", orient="split")


def _deserialize_dataframe(data: str | None) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()

    return pd.read_json(StringIO(data), orient="split")


def _normalize_month_date(value) -> pd.Timestamp | None:
    if not value:
        return None

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None

    return timestamp.to_period("M").to_timestamp()


def _get_date_bounds(dataframes: list[pd.DataFrame]) -> tuple[str | None, str | None]:
    non_empty_frames = [df for df in dataframes if df is not None and not df.empty]
    if not non_empty_frames:
        return None, None

    combined_df = pd.concat(non_empty_frames, ignore_index=True)
    min_month = pd.to_datetime(combined_df["month"]).min()
    max_month = pd.to_datetime(combined_df["month"]).max()

    return min_month.strftime("%Y-%m-%d"), max_month.strftime("%Y-%m-%d")


def _get_default_interval_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    current_year = pd.Timestamp.now().year
    default_start = pd.Timestamp(year=current_year, month=1, day=1)
    default_end = pd.Timestamp(year=current_year + 5, month=12, day=1)
    return default_start, default_end


def _filter_by_date_range(
    raw_df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    filtered_df = raw_df.copy()
    filtered_df["month"] = pd.to_datetime(filtered_df["month"]).dt.to_period("M").dt.to_timestamp()

    start_month = _normalize_month_date(start_date)
    end_month = _normalize_month_date(end_date)

    if start_month is not None:
        filtered_df = filtered_df[filtered_df["month"] >= start_month]
    if end_month is not None:
        filtered_df = filtered_df[filtered_df["month"] <= end_month]

    return filtered_df


def _create_empty_state(message: str) -> html.Div:
    return html.Div(message, className="balance-empty-state")


def _serialize_snapshot_value(payload: dict[str, str | None]) -> str:
    return json.dumps(payload, sort_keys=True)


def _deserialize_snapshot_value(value: str | dict | None) -> dict[str, str | None]:
    if not value:
        return {}

    if isinstance(value, dict):
        return value

    try:
        parsed_value = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}

    return parsed_value if isinstance(parsed_value, dict) else {}


def _default_previous_option_value(options: list[dict]) -> str | None:
    if len(options) > 1:
        return options[1]["value"]
    if options:
        return options[0]["value"]
    return None


def _build_woodmac_snapshot_dropdown_options(
    publication_options: list[dict[str, str | None]],
) -> list[dict[str, str]]:
    dropdown_options = []
    for option in publication_options:
        publication_label = option.get("market_outlook", "Unknown publication")
        publication_timestamp = _format_metadata_timestamp(
            option.get("publication_timestamp")
        )
        label = publication_label
        if publication_timestamp:
            label = f"{publication_label} | {publication_timestamp}"

        dropdown_options.append(
            {
                "label": label,
                "value": _serialize_snapshot_value(option),
            }
        )

    return dropdown_options


def _build_ea_upload_dropdown_options(
    upload_timestamps: list[str],
) -> list[dict[str, str]]:
    dropdown_options = []
    for upload_timestamp in upload_timestamps:
        formatted_timestamp = _format_metadata_timestamp(upload_timestamp) or upload_timestamp
        dropdown_options.append(
            {
                "label": formatted_timestamp,
                "value": upload_timestamp,
            }
        )

    return dropdown_options


def _build_delta_matrix(
    baseline_matrix: pd.DataFrame,
    comparison_matrix: pd.DataFrame,
) -> pd.DataFrame:
    if baseline_matrix.empty:
        return pd.DataFrame(columns=["Month", "Total MMTPA"])

    numeric_columns = [column for column in baseline_matrix.columns if column != "Month"]
    delta_df = baseline_matrix.copy()
    delta_df[numeric_columns] = delta_df[numeric_columns].astype(float)

    comparison_aligned = comparison_matrix.copy()
    for column in numeric_columns:
        if column not in comparison_aligned.columns:
            comparison_aligned[column] = 0.0

    comparison_aligned = comparison_aligned[["Month"] + numeric_columns]
    comparison_aligned[numeric_columns] = comparison_aligned[numeric_columns].astype(float)
    comparison_aligned = comparison_aligned.set_index("Month").reindex(
        delta_df["Month"],
        fill_value=0.0,
    )

    delta_index = delta_df.set_index("Month")
    delta_index[numeric_columns] = (
        delta_index[numeric_columns] - comparison_aligned[numeric_columns]
    ).round(2)

    return delta_index.reset_index()


def _format_table_cell_value(value) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"

    return str(value)


def _build_responsive_column_styles(df: pd.DataFrame) -> list[dict]:
    column_styles = []
    column_weights = {}
    column_min_widths = {}

    for column_name in df.columns:
        header_length = len(str(column_name))
        value_lengths = df[column_name].map(_format_table_cell_value).map(len)
        max_length = max([header_length] + value_lengths.tolist()) if not df.empty else header_length

        if column_name == "Month":
            column_weights[column_name] = max(8, min(max_length, 12))
            column_min_widths[column_name] = 92
        elif column_name == "Total MMTPA":
            column_weights[column_name] = max(8, min(max_length, 14))
            column_min_widths[column_name] = 96
        else:
            column_weights[column_name] = max(6, min(max_length, 18))
            column_min_widths[column_name] = 72

    total_weight = sum(column_weights.values()) or 1

    for column_name in df.columns:
        width_pct = column_weights[column_name] / total_weight * 100
        style_entry = {
            "if": {"column_id": column_name},
            "minWidth": f"{column_min_widths[column_name]}px",
            "width": f"{width_pct:.2f}%",
        }

        if column_name == "Month":
            style_entry["textAlign"] = "left"

        column_styles.append(style_entry)

    return column_styles


def _format_metadata_timestamp(value) -> str | None:
    if not value:
        return None

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return str(value)

    return timestamp.strftime("%Y-%m-%d %H:%M")


def _build_woodmac_metadata_lines(metadata: dict | None) -> list[str]:
    if not metadata:
        return []

    lines = []
    short_term_line = metadata.get("short_term_market_outlook")
    short_term_timestamp = _format_metadata_timestamp(
        metadata.get("short_term_publication_timestamp")
    )
    if short_term_line:
        if short_term_timestamp:
            lines.append(
                f"ST publication: {short_term_line} | publication_date: {short_term_timestamp}"
            )
        else:
            lines.append(f"ST publication: {short_term_line}")

    long_term_line = metadata.get("long_term_market_outlook")
    long_term_timestamp = _format_metadata_timestamp(
        metadata.get("long_term_publication_timestamp")
    )
    if long_term_line:
        if long_term_timestamp:
            lines.append(
                f"LT publication: {long_term_line} | publication_date: {long_term_timestamp}"
            )
        else:
            lines.append(f"LT publication: {long_term_line}")

    return lines


def _build_ea_metadata_lines(metadata: dict | None) -> list[str]:
    if not metadata:
        return []

    upload_timestamp = _format_metadata_timestamp(metadata.get("upload_timestamp_utc"))
    if not upload_timestamp:
        return []

    return [f"upload_timestamp_utc: {upload_timestamp}"]
def _create_balance_table(
    table_id: str,
    df: pd.DataFrame,
    table_mode: str = "absolute",
) -> dash_table.DataTable | html.Div:
    if df.empty:
        return _create_empty_state("No data available for the current selection.")

    base_config = StandardTableStyleManager.get_base_datatable_config()
    numeric_columns = [column for column in df.columns if column != "Month"]

    columns = [{"name": "Month", "id": "Month"}]
    columns.extend(
        {
            "name": column_name,
            "id": column_name,
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        }
        for column_name in numeric_columns
    )

    style_data_conditional = list(base_config["style_data_conditional"])
    style_data_conditional.append(
        {
            "if": {"column_id": "Month"},
            "backgroundColor": "#f8fafc",
            "fontWeight": "600",
            "color": TABLE_COLORS["text_primary"],
        }
    )

    if table_mode == "delta":
        for column_name in numeric_columns:
            style_data_conditional.extend(
                [
                    {
                        "if": {
                            "column_id": column_name,
                            "filter_query": f"{{{column_name}}} > 0",
                        },
                        "backgroundColor": "#ecfdf5",
                        "color": "#166534",
                        "fontWeight": "600",
                    },
                    {
                        "if": {
                            "column_id": column_name,
                            "filter_query": f"{{{column_name}}} < 0",
                        },
                        "backgroundColor": "#fef2f2",
                        "color": "#991b1b",
                        "fontWeight": "600",
                    },
                    {
                        "if": {
                            "column_id": column_name,
                            "filter_query": f"{{{column_name}}} = 0",
                        },
                        "backgroundColor": "#f8fafc",
                        "color": "#64748b",
                    },
                ]
            )
    else:
        style_data_conditional.extend(
            [
                {
                    "if": {"column_id": "Total MMTPA"},
                    "backgroundColor": "#edf6fd",
                    "fontWeight": "700",
                    "color": TABLE_COLORS["primary_dark"],
                },
                {
                    "if": {"column_id": "Rest of the World"},
                    "backgroundColor": "#f8f9fa",
                    "color": TABLE_COLORS["text_secondary"],
                },
            ]
        )

    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=df.to_dict("records"),
        sort_action="native",
        page_action="none",
        fill_width=True,
        fixed_rows={"headers": True},
        fixed_columns={"headers": True, "data": 1},
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": "560px",
            "width": "100%",
            "minWidth": "100%",
        },
        style_header=base_config["style_header"],
        style_cell={
            **base_config["style_cell"],
            "minWidth": "72px",
            "width": "72px",
            "maxWidth": "none",
            "border": f"1px solid {TABLE_COLORS['border_light']}",
            "padding": "6px 8px",
        },
        style_cell_conditional=_build_responsive_column_styles(df),
        style_data_conditional=style_data_conditional,
    )


def _build_section_summary(
    raw_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    other_countries_mode: str,
    metadata_lines: list[str] | None = None,
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
        visibility_note = (
            "Other countries grouped into Rest of the World."
            if other_countries_mode == "rest_of_world"
            else "Only selected countries are included in the totals."
        )

        summary_children.append(
            html.Div(
                [
                    html.Span(f"{len(matrix_df):,} months"),
                    html.Span(f"{month_start} to {month_end}"),
                    html.Span(f"{visible_country_count} visible country columns"),
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

    return html.Div(
        summary_children
    )


def _build_comparison_metadata_lines(
    comparison_source: str,
    short_term_value: str | None,
    long_term_value: str | None,
    ea_upload_value: str | None,
) -> list[str]:
    if comparison_source == "woodmac":
        short_term_snapshot = _deserialize_snapshot_value(short_term_value)
        long_term_snapshot = _deserialize_snapshot_value(long_term_value)
        lines = ["Delta formula: left baseline table - selected snapshot"]

        short_term_line = short_term_snapshot.get("market_outlook")
        short_term_timestamp = _format_metadata_timestamp(
            short_term_snapshot.get("publication_timestamp")
        )
        if short_term_line:
            if short_term_timestamp:
                lines.append(
                    f"Comparison source: WoodMac | ST publication: {short_term_line} | publication_date: {short_term_timestamp}"
                )
            else:
                lines.append(
                    f"Comparison source: WoodMac | ST publication: {short_term_line}"
                )

        long_term_line = long_term_snapshot.get("market_outlook")
        long_term_timestamp = _format_metadata_timestamp(
            long_term_snapshot.get("publication_timestamp")
        )
        if long_term_line:
            if long_term_timestamp:
                lines.append(
                    f"LT publication: {long_term_line} | publication_date: {long_term_timestamp}"
                )
            else:
                lines.append(f"LT publication: {long_term_line}")

        return lines

    ea_upload_label = _format_metadata_timestamp(ea_upload_value) or ea_upload_value
    if ea_upload_label:
        return [
            "Delta formula: left baseline table - selected snapshot",
            f"Comparison source: Energy Aspects | upload_timestamp_utc: {ea_upload_label}",
        ]

    return ["Delta formula: left baseline table - selected snapshot"]


def _build_comparison_summary(
    delta_df: pd.DataFrame,
    metadata_lines: list[str],
) -> html.Div:
    return html.Div()


def _create_source_section(
    title: str,
    subtitle: str,
    summary_id: str,
    table_container_id: str,
    export_button_id: str,
) -> html.Div:
    header_children = [
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
        )
    ]

    if subtitle:
        header_children.append(
            html.P(subtitle, className="balance-section-subtitle")
        )

    header_children.append(html.Div(id=summary_id))

    return html.Div(
        [
            html.Div(
                header_children,
                className="balance-section-header",
            ),
            html.Div(id=table_container_id, className="balance-table-container"),
        ],
        className="balance-section-card",
    )


def _create_woodmac_comparison_section() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("WoodMac Exports Flow", className="balance-section-title"),
                            html.Button(
                                "Export to Excel",
                                id="balance-export-woodmac-button",
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
                                id="balance-woodmac-summary",
                                className="balance-pane-summary",
                            )
                        ],
                        className="balance-pane-top-area balance-pane-top-area-left",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Source:", className="filter-label"),
                                    dcc.Dropdown(
                                        id="balance-comparison-source-dropdown",
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
                                        value="woodmac",
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
                                                className="filter-label",
                                            ),
                                            dcc.Dropdown(
                                                id="balance-comparison-st-dropdown",
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
                                                className="filter-label",
                                            ),
                                            dcc.Dropdown(
                                                id="balance-comparison-lt-dropdown",
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
                                id="balance-comparison-woodmac-controls",
                                className="balance-comparison-control-row",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "upload_timestamp_utc:",
                                                className="filter-label",
                                            ),
                                            dcc.Dropdown(
                                                id="balance-comparison-ea-upload-dropdown",
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
                                id="balance-comparison-ea-controls",
                                className="balance-comparison-control-row",
                            ),
                        ],
                        className="balance-comparison-controls balance-pane-top-area balance-pane-top-area-right",
                    ),
                    html.Div(
                        id="balance-woodmac-table-container",
                        className="balance-table-container balance-table-container-left",
                    ),
                    html.Div(
                        [
                            html.Div(id="balance-comparison-summary"),
                            html.Div(
                                id="balance-comparison-table-container",
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


layout = html.Div(
    [
        dcc.Store(id="balance-woodmac-data-store", storage_type="memory"),
        dcc.Store(id="balance-ea-data-store", storage_type="memory"),
        dcc.Store(id="balance-country-options-store", storage_type="memory"),
        dcc.Store(id="balance-refresh-timestamp-store", storage_type="memory"),
        dcc.Store(id="balance-load-error-store", storage_type="memory"),
        dcc.Store(id="balance-woodmac-metadata-store", storage_type="memory"),
        dcc.Store(id="balance-ea-metadata-store", storage_type="memory"),
        dcc.Store(id="balance-comparison-options-store", storage_type="memory"),
        dcc.Download(id="balance-download-woodmac-excel"),
        dcc.Download(id="balance-download-ea-excel"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Date Range", className="filter-group-header"),
                                html.Label("Month interval:", className="filter-label"),
                                html.Div(
                                    [
                                        dcc.DatePickerRange(
                                            id="balance-date-range",
                                            start_date=None,
                                            end_date=None,
                                            min_date_allowed=None,
                                            max_date_allowed=None,
                                            minimum_nights=0,
                                            display_format="YYYY-MM",
                                            month_format="YYYY-MM",
                                            start_date_placeholder_text="Start month",
                                            end_date_placeholder_text="End month",
                                            clearable=False,
                                            number_of_months_shown=2,
                                        )
                                    ],
                                    className="professional-date-picker",
                                ),
                            ],
                            className="filter-section filter-section-destination",
                        ),
                        html.Div(
                            [
                                html.Div("Country Columns", className="filter-group-header"),
                                html.Label("Countries:", className="filter-label"),
                                dcc.Dropdown(
                                    id="balance-country-dropdown",
                                    options=[],
                                    value=None,
                                    multi=True,
                                    placeholder="Select countries to keep as separate columns",
                                    className="filter-dropdown",
                                    style={"minWidth": "380px"},
                                ),
                            ],
                            className="filter-section filter-section-origin-exp",
                        ),
                        html.Div(
                            [
                                html.Div("Other Countries", className="filter-group-header"),
                                html.Label("Handling:", className="filter-label"),
                                dcc.RadioItems(
                                    id="balance-other-country-mode",
                                    options=[
                                        {
                                            "label": "Include as Rest of the World",
                                            "value": "rest_of_world",
                                        },
                                        {
                                            "label": "Exclude from the table",
                                            "value": "exclude",
                                        },
                                    ],
                                    value="rest_of_world",
                                    className="balance-radio-group",
                                    labelStyle={"display": "inline-flex", "alignItems": "center"},
                                    inputStyle={"marginRight": "6px"},
                                ),
                            ],
                            className="filter-section filter-section-volume",
                        ),
                        html.Div(
                            [
                                html.Div("Status", className="filter-group-header"),
                                html.Div(
                                    id="balance-refresh-indicator",
                                    className="text-tertiary",
                                    style={"fontSize": "12px", "whiteSpace": "nowrap"},
                                ),
                                html.Div(
                                    id="balance-meta-indicator",
                                    className="text-tertiary",
                                    style={"fontSize": "12px", "maxWidth": "260px"},
                                ),
                            ],
                            className="filter-section filter-section-analysis",
                        ),
                    ],
                    className="filter-bar-grouped",
                )
            ],
            className="professional-section-header",
        ),
        html.Div(
            [
                html.Div(id="balance-load-error-banner"),
                dcc.Loading(
                    children=[
                        html.Div(
                            [
                                _create_woodmac_comparison_section(),
                                _create_source_section(
                                    "Energy Aspects Exports Flow",
                                    "",
                                    "balance-ea-summary",
                                    "balance-ea-table-container",
                                    "balance-export-ea-button",
                                ),
                            ],
                            className="balance-results-stack",
                        )
                    ],
                    type="default",
                ),
            ],
            className="balance-page-shell",
        ),
    ]
)


@callback(
    Output("balance-woodmac-data-store", "data"),
    Output("balance-ea-data-store", "data"),
    Output("balance-country-options-store", "data"),
    Output("balance-refresh-timestamp-store", "data"),
    Output("balance-load-error-store", "data"),
    Output("balance-woodmac-metadata-store", "data"),
    Output("balance-ea-metadata-store", "data"),
    Output("balance-comparison-options-store", "data"),
    Input("global-refresh-button", "n_clicks"),
)
def load_balance_source_data(_):
    woodmac_df = pd.DataFrame()
    ea_df = pd.DataFrame()
    woodmac_metadata = {}
    ea_metadata = {}
    comparison_options = {
        "woodmac": {"short_term": [], "long_term": []},
        "ea_uploads": [],
    }
    errors = []

    try:
        woodmac_df = fetch_woodmac_export_flow_raw_data()
    except Exception as exc:
        errors.append(f"WoodMac load failed: {exc}")

    try:
        woodmac_metadata = fetch_woodmac_export_flow_metadata()
    except Exception as exc:
        errors.append(f"WoodMac metadata load failed: {exc}")

    try:
        ea_df = fetch_ea_export_flow_raw_data()
    except Exception as exc:
        errors.append(f"Energy Aspects load failed: {exc}")

    try:
        ea_metadata = fetch_ea_export_flow_metadata()
    except Exception as exc:
        errors.append(f"Energy Aspects metadata load failed: {exc}")

    try:
        comparison_options["woodmac"] = fetch_woodmac_publication_options()
    except Exception as exc:
        errors.append(f"WoodMac comparison options load failed: {exc}")

    try:
        comparison_options["ea_uploads"] = fetch_ea_upload_options()
    except Exception as exc:
        errors.append(f"Energy Aspects comparison options load failed: {exc}")

    available_countries = get_available_countries([woodmac_df, ea_df])
    refresh_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = " | ".join(errors) if errors else None

    return (
        _serialize_dataframe(woodmac_df),
        _serialize_dataframe(ea_df),
        available_countries,
        refresh_timestamp,
        error_message,
        woodmac_metadata,
        ea_metadata,
        comparison_options,
    )


@callback(
    Output("balance-country-dropdown", "options"),
    Output("balance-country-dropdown", "value"),
    Input("balance-country-options-store", "data"),
    State("balance-country-dropdown", "value"),
)
def update_balance_country_options(available_countries, current_selection):
    available_countries = available_countries or []
    options = [{"label": country, "value": country} for country in available_countries]

    if current_selection is None:
        selected_values = default_selected_countries(available_countries)
    else:
        selected_values = [
            country for country in current_selection if country in available_countries
        ]
        if current_selection and not selected_values:
            selected_values = default_selected_countries(available_countries)

    return options, selected_values


@callback(
    Output("balance-comparison-st-dropdown", "options"),
    Output("balance-comparison-st-dropdown", "value"),
    Output("balance-comparison-lt-dropdown", "options"),
    Output("balance-comparison-lt-dropdown", "value"),
    Output("balance-comparison-ea-upload-dropdown", "options"),
    Output("balance-comparison-ea-upload-dropdown", "value"),
    Output("balance-comparison-woodmac-controls", "style"),
    Output("balance-comparison-ea-controls", "style"),
    Input("balance-comparison-source-dropdown", "value"),
    Input("balance-comparison-options-store", "data"),
    State("balance-comparison-st-dropdown", "value"),
    State("balance-comparison-lt-dropdown", "value"),
    State("balance-comparison-ea-upload-dropdown", "value"),
)
def update_comparison_snapshot_controls(
    comparison_source,
    comparison_options,
    current_st_value,
    current_lt_value,
    current_ea_upload_value,
):
    comparison_options = comparison_options or {}
    woodmac_options = comparison_options.get("woodmac", {})
    short_term_options = _build_woodmac_snapshot_dropdown_options(
        woodmac_options.get("short_term", [])
    )
    long_term_options = _build_woodmac_snapshot_dropdown_options(
        woodmac_options.get("long_term", [])
    )
    ea_upload_options = _build_ea_upload_dropdown_options(
        comparison_options.get("ea_uploads", [])
    )

    short_term_values = {option["value"] for option in short_term_options}
    long_term_values = {option["value"] for option in long_term_options}
    ea_upload_values = {option["value"] for option in ea_upload_options}

    short_term_value = (
        current_st_value
        if current_st_value in short_term_values
        else _default_previous_option_value(short_term_options)
    )
    long_term_value = (
        current_lt_value
        if current_lt_value in long_term_values
        else _default_previous_option_value(long_term_options)
    )
    ea_upload_value = (
        current_ea_upload_value
        if current_ea_upload_value in ea_upload_values
        else _default_previous_option_value(ea_upload_options)
    )

    if comparison_source == "ea":
        return (
            short_term_options,
            short_term_value,
            long_term_options,
            long_term_value,
            ea_upload_options,
            ea_upload_value,
            {"display": "none"},
            {"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "flex-end"},
        )

    return (
        short_term_options,
        short_term_value,
        long_term_options,
        long_term_value,
        ea_upload_options,
        ea_upload_value,
        {"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "flex-end"},
        {"display": "none"},
    )


@callback(
    Output("balance-date-range", "min_date_allowed"),
    Output("balance-date-range", "max_date_allowed"),
    Output("balance-date-range", "start_date"),
    Output("balance-date-range", "end_date"),
    Input("balance-woodmac-data-store", "data"),
    Input("balance-ea-data-store", "data"),
    State("balance-date-range", "start_date"),
    State("balance-date-range", "end_date"),
)
def update_balance_date_range(woodmac_data, ea_data, current_start_date, current_end_date):
    woodmac_raw_df = _deserialize_dataframe(woodmac_data)
    ea_raw_df = _deserialize_dataframe(ea_data)

    min_date, max_date = _get_date_bounds([woodmac_raw_df, ea_raw_df])
    if min_date is None or max_date is None:
        return None, None, None, None

    normalized_min = _normalize_month_date(min_date)
    normalized_max = _normalize_month_date(max_date)
    default_start, default_end = _get_default_interval_window()

    normalized_start = _normalize_month_date(current_start_date) or default_start
    normalized_end = _normalize_month_date(current_end_date) or default_end

    if normalized_start < normalized_min:
        normalized_start = normalized_min
    if normalized_end > normalized_max:
        normalized_end = normalized_max
    if normalized_start > normalized_end:
        normalized_start = normalized_min
        normalized_end = normalized_max

    return (
        normalized_min.strftime("%Y-%m-%d"),
        normalized_max.strftime("%Y-%m-%d"),
        normalized_start.strftime("%Y-%m-%d"),
        normalized_end.strftime("%Y-%m-%d"),
    )


@callback(
    Output("balance-refresh-indicator", "children"),
    Output("balance-meta-indicator", "children"),
    Input("balance-refresh-timestamp-store", "data"),
    Input("balance-country-options-store", "data"),
    Input("balance-date-range", "start_date"),
    Input("balance-date-range", "end_date"),
)
def update_balance_status(refresh_timestamp, available_countries, start_date, end_date):
    refresh_text = (
        f"Last refreshed: {refresh_timestamp}"
        if refresh_timestamp
        else "Last refreshed: waiting for data"
    )
    if start_date and end_date:
        start_label = _normalize_month_date(start_date).strftime("%Y-%m")
        end_label = _normalize_month_date(end_date).strftime("%Y-%m")
        range_text = f"Showing {start_label} to {end_label}."
    else:
        range_text = "Country names are standardized with at_lng.mappings_country."

    meta_text = (
        f"{len(available_countries or []):,} countries available after standardization. {range_text}"
        if available_countries
        else range_text
    )
    return refresh_text, meta_text


@callback(
    Output("balance-load-error-banner", "children"),
    Input("balance-load-error-store", "data"),
)
def update_balance_error_banner(error_message):
    if not error_message:
        return html.Div()

    return html.Div(error_message, className="balance-error-banner")


@callback(
    Output("balance-woodmac-summary", "children"),
    Output("balance-woodmac-table-container", "children"),
    Output("balance-ea-summary", "children"),
    Output("balance-ea-table-container", "children"),
    Input("balance-woodmac-data-store", "data"),
    Input("balance-ea-data-store", "data"),
    Input("balance-woodmac-metadata-store", "data"),
    Input("balance-ea-metadata-store", "data"),
    Input("balance-country-dropdown", "value"),
    Input("balance-other-country-mode", "value"),
    Input("balance-date-range", "start_date"),
    Input("balance-date-range", "end_date"),
)
def render_balance_tables(
    woodmac_data,
    ea_data,
    woodmac_metadata,
    ea_metadata,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
):
    woodmac_raw_df = _filter_by_date_range(
        _deserialize_dataframe(woodmac_data),
        start_date,
        end_date,
    )
    ea_raw_df = _filter_by_date_range(
        _deserialize_dataframe(ea_data),
        start_date,
        end_date,
    )

    available_countries = get_available_countries([woodmac_raw_df, ea_raw_df])
    if selected_countries is None:
        resolved_countries = default_selected_countries(available_countries)
    else:
        resolved_countries = [
            country for country in selected_countries if country in available_countries
        ]

    woodmac_matrix = build_export_flow_matrix(
        woodmac_raw_df,
        resolved_countries,
        other_countries_mode,
    )
    ea_matrix = build_export_flow_matrix(
        ea_raw_df,
        resolved_countries,
        other_countries_mode,
    )

    woodmac_summary = _build_section_summary(
        woodmac_raw_df,
        woodmac_matrix,
        other_countries_mode,
        _build_woodmac_metadata_lines(woodmac_metadata),
    )
    ea_summary = _build_section_summary(
        ea_raw_df,
        ea_matrix,
        other_countries_mode,
        _build_ea_metadata_lines(ea_metadata),
    )

    if resolved_countries == [] and other_countries_mode == "exclude":
        empty_message = _create_empty_state(
            "Select at least one country or switch to Rest of the World mode."
        )
        return woodmac_summary, empty_message, ea_summary, empty_message

    woodmac_table = _create_balance_table("balance-woodmac-table", woodmac_matrix)
    ea_table = _create_balance_table("balance-ea-table", ea_matrix)

    return woodmac_summary, woodmac_table, ea_summary, ea_table


@callback(
    Output("balance-comparison-summary", "children"),
    Output("balance-comparison-table-container", "children"),
    Input("balance-woodmac-data-store", "data"),
    Input("balance-ea-data-store", "data"),
    Input("balance-country-dropdown", "value"),
    Input("balance-other-country-mode", "value"),
    Input("balance-date-range", "start_date"),
    Input("balance-date-range", "end_date"),
    Input("balance-comparison-source-dropdown", "value"),
    Input("balance-comparison-st-dropdown", "value"),
    Input("balance-comparison-lt-dropdown", "value"),
    Input("balance-comparison-ea-upload-dropdown", "value"),
)
def render_comparison_delta_table(
    woodmac_data,
    ea_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    comparison_source,
    short_term_value,
    long_term_value,
    ea_upload_value,
):
    baseline_raw_df = _filter_by_date_range(
        _deserialize_dataframe(woodmac_data),
        start_date,
        end_date,
    )
    ea_filtered_df = _filter_by_date_range(
        _deserialize_dataframe(ea_data),
        start_date,
        end_date,
    )

    available_countries = get_available_countries([baseline_raw_df, ea_filtered_df])
    if selected_countries is None:
        resolved_countries = default_selected_countries(available_countries)
    else:
        resolved_countries = [
            country for country in selected_countries if country in available_countries
        ]

    baseline_matrix = build_export_flow_matrix(
        baseline_raw_df,
        resolved_countries,
        other_countries_mode,
    )

    metadata_lines = _build_comparison_metadata_lines(
        comparison_source,
        short_term_value,
        long_term_value,
        ea_upload_value,
    )

    if baseline_matrix.empty:
        return (
            _build_comparison_summary(pd.DataFrame(columns=["Month", "Total MMTPA"]), metadata_lines),
            _create_empty_state("No baseline WoodMac data available for the current selection."),
        )

    if resolved_countries == [] and other_countries_mode == "exclude":
        return (
            _build_comparison_summary(
                pd.DataFrame(columns=baseline_matrix.columns),
                metadata_lines,
            ),
            _create_empty_state(
                "Select at least one country or switch to Rest of the World mode."
            ),
        )

    try:
        if comparison_source == "ea":
            if not ea_upload_value:
                return (
                    _build_comparison_summary(
                        pd.DataFrame(columns=baseline_matrix.columns),
                        metadata_lines,
                    ),
                    _create_empty_state("No Energy Aspects upload_timestamp_utc available."),
                )
            comparison_raw_df = fetch_ea_export_flow_raw_data_for_upload(ea_upload_value)
        else:
            short_term_snapshot = _deserialize_snapshot_value(short_term_value)
            long_term_snapshot = _deserialize_snapshot_value(long_term_value)
            if not short_term_snapshot or not long_term_snapshot:
                return (
                    _build_comparison_summary(
                        pd.DataFrame(columns=baseline_matrix.columns),
                        metadata_lines,
                    ),
                    _create_empty_state("No WoodMac comparison publications available."),
                )

            comparison_raw_df = fetch_woodmac_export_flow_raw_data_for_publications(
                short_term_snapshot.get("market_outlook"),
                short_term_snapshot.get("publication_timestamp"),
                long_term_snapshot.get("market_outlook"),
                long_term_snapshot.get("publication_timestamp"),
            )
    except Exception as exc:
        error_summary = _build_comparison_summary(
            pd.DataFrame(columns=baseline_matrix.columns),
            metadata_lines + [f"Comparison load failed: {exc}"],
        )
        return error_summary, _create_empty_state("Unable to load comparison snapshot.")

    comparison_filtered_df = _filter_by_date_range(
        comparison_raw_df,
        start_date,
        end_date,
    )
    comparison_matrix = build_export_flow_matrix(
        comparison_filtered_df,
        resolved_countries,
        other_countries_mode,
    )
    delta_matrix = _build_delta_matrix(baseline_matrix, comparison_matrix)

    comparison_summary = _build_comparison_summary(delta_matrix, metadata_lines)
    comparison_table = _create_balance_table(
        "balance-comparison-delta-table",
        delta_matrix,
        table_mode="delta",
    )

    return comparison_summary, comparison_table


def _export_matrix_to_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        worksheet = writer.sheets[sheet_name]
        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter
            for cell in column_cells:
                cell_value = "" if cell.value is None else str(cell.value)
                if len(cell_value) > max_length:
                    max_length = len(cell_value)
            worksheet.column_dimensions[column_letter].width = min(max_length + 2, 24)

    output.seek(0)
    return output.getvalue()


def _build_filtered_matrix_for_export(
    source_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
) -> pd.DataFrame:
    raw_df = _filter_by_date_range(
        _deserialize_dataframe(source_data),
        start_date,
        end_date,
    )
    return build_export_flow_matrix(
        raw_df,
        selected_countries,
        other_countries_mode,
    )


@callback(
    Output("balance-download-woodmac-excel", "data"),
    Input("balance-export-woodmac-button", "n_clicks"),
    State("balance-woodmac-data-store", "data"),
    State("balance-country-dropdown", "value"),
    State("balance-other-country-mode", "value"),
    State("balance-date-range", "start_date"),
    State("balance-date-range", "end_date"),
    prevent_initial_call=True,
)
def export_woodmac_balance_excel(
    n_clicks,
    woodmac_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
):
    if not n_clicks:
        raise PreventUpdate

    export_df = _build_filtered_matrix_for_export(
        woodmac_data,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
    )
    if export_df.empty:
        raise PreventUpdate

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"WoodMac_Exports_Flow_{timestamp}.xlsx"
    return dcc.send_bytes(
        _export_matrix_to_excel_bytes(export_df, "Exports Flow"),
        filename,
    )


@callback(
    Output("balance-download-ea-excel", "data"),
    Input("balance-export-ea-button", "n_clicks"),
    State("balance-ea-data-store", "data"),
    State("balance-country-dropdown", "value"),
    State("balance-other-country-mode", "value"),
    State("balance-date-range", "start_date"),
    State("balance-date-range", "end_date"),
    prevent_initial_call=True,
)
def export_ea_balance_excel(
    n_clicks,
    ea_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
):
    if not n_clicks:
        raise PreventUpdate

    export_df = _build_filtered_matrix_for_export(
        ea_data,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
    )
    if export_df.empty:
        raise PreventUpdate

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"EA_Exports_Flow_{timestamp}.xlsx"
    return dcc.send_bytes(
        _export_matrix_to_excel_bytes(export_df, "Exports Flow"),
        filename,
    )
