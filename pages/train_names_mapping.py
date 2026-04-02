from dash import html, dcc, dash_table, callback, Output, Input, State, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import configparser
import math
import os
from sqlalchemy import create_engine, text

from utils.table_styles import StandardTableStyleManager


try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    CONFIG_FILE_PATH = os.path.join(config_dir, "config.ini")
except Exception:
    CONFIG_FILE_PATH = "config.ini"

config_reader = configparser.ConfigParser(interpolation=None)
config_reader.read(CONFIG_FILE_PATH)

DB_CONNECTION_STRING = config_reader.get("DATABASE", "CONNECTION_STRING", fallback=None)
DB_SCHEMA = config_reader.get("DATABASE", "SCHEMA", fallback="at_lng")
TABLE_NAME = "mapping_plant_train_name"

engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)

DISPLAY_COLUMNS = [
    "country_name",
    "plant_name",
    "provider",
    "parent_source_field",
    "parent_source_name",
    "source_field",
    "source_name",
    "scope_hint",
    "component_hint",
    "train",
    "allocation_share",
    "notes",
]

EDITABLE_COLUMNS = {"scope_hint", "component_hint", "train", "allocation_share", "notes"}
RAW_SOURCE_KEY_COLUMNS = [
    "country_name",
    "plant_name",
    "provider",
    "parent_source_field",
    "parent_source_name",
    "source_field",
    "source_name",
]


def _clean_text_value(value):
    if pd.isna(value):
        return ""

    text_value = str(value).strip()
    if not text_value:
        return ""

    return " ".join(text_value.split())


def _clean_train_value(value):
    if pd.isna(value) or str(value).strip() == "":
        return ""
    numeric_value = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(numeric_value):
        return ""
    integer_value = int(numeric_value)
    if integer_value <= 0:
        return ""
    return integer_value


def _clean_allocation_share(value):
    if pd.isna(value) or str(value).strip() == "":
        return ""
    numeric_value = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(numeric_value):
        return ""
    if float(numeric_value) <= 0:
        return ""
    return round(float(numeric_value), 8)


def fetch_train_name_mappings_data(db_engine, schema=DB_SCHEMA):
    try:
        with db_engine.connect() as conn:
            query = text(
                f"""
                SELECT
                    country_name,
                    plant_name,
                    provider,
                    parent_source_field,
                    parent_source_name,
                    source_field,
                    source_name,
                    scope_hint,
                    component_hint,
                    train,
                    allocation_share,
                    notes
                FROM {schema}.{TABLE_NAME}
                ORDER BY country_name, plant_name, provider, train, parent_source_name, source_name
                """
            )
            df = pd.read_sql(query, conn)
    except Exception as exc:
        print(f"Error fetching train name mappings data: {exc}")
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    for column_name in DISPLAY_COLUMNS:
        if column_name == "train":
            df[column_name] = df[column_name].map(_clean_train_value)
        elif column_name == "allocation_share":
            df[column_name] = df[column_name].map(_clean_allocation_share)
        else:
            df[column_name] = df[column_name].map(_clean_text_value)

    df = df.drop_duplicates(
        subset=RAW_SOURCE_KEY_COLUMNS + ["train"],
        keep="last",
    ).reset_index(drop=True)

    return df


def create_summary_cards(df):
    if df.empty:
        return html.Div("No train name mapping data available")

    total_alias_rows = len(df)
    canonical_trains = (
        df[["country_name", "plant_name", "train"]]
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .shape[0]
    )
    plants = df["plant_name"].replace("", pd.NA).dropna().nunique()
    providers = df["provider"].replace("", pd.NA).dropna().nunique()

    card_specs = [
        ("Alias Rows", f"{total_alias_rows:,}"),
        ("Canonical Trains", f"{canonical_trains:,}"),
        ("Plants", f"{plants:,}"),
        ("Providers", f"{providers:,}"),
    ]

    cards = []
    for label, value in card_specs:
        cards.append(
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H6(
                                        label,
                                        className="text-secondary",
                                        style={"marginBottom": "8px"},
                                    ),
                                    html.H3(value, className="text-primary font-bold"),
                                ]
                            )
                        ],
                        className="shadow-sm h-100",
                    )
                ],
                width=3,
            )
        )

    return dbc.Row(cards, className="mb-4")


def _build_empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": "#64748b"},
    )
    fig.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _build_bar_figure(series: pd.Series, title: str, color: str) -> go.Figure:
    if series.empty:
        return _build_empty_figure("No data available for the current selection.")

    plot_df = series.reset_index()
    plot_df.columns = ["label", "count"]

    fig = px.bar(
        plot_df,
        x="count",
        y="label",
        orientation="h",
        text="count",
    )
    fig.update_traces(marker_color=color, textposition="outside", cliponaxis=False)
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=max(320, 60 + (len(plot_df) * 28)),
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        xaxis_title="Rows",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_yaxes(categoryorder="total ascending")
    return fig


def _filter_mapping_df(
    df: pd.DataFrame,
    countries,
    plants,
    providers,
    parent_source_fields,
    source_fields,
    scope_hints,
    search_text,
) -> pd.DataFrame:
    filtered_df = df.copy()

    if countries:
        filtered_df = filtered_df[filtered_df["country_name"].isin(countries)]
    if plants:
        filtered_df = filtered_df[filtered_df["plant_name"].isin(plants)]
    if providers:
        filtered_df = filtered_df[filtered_df["provider"].isin(providers)]
    if parent_source_fields:
        filtered_df = filtered_df[filtered_df["parent_source_field"].isin(parent_source_fields)]
    if source_fields:
        filtered_df = filtered_df[filtered_df["source_field"].isin(source_fields)]
    if scope_hints:
        filtered_df = filtered_df[filtered_df["scope_hint"].isin(scope_hints)]

    search_value = _clean_text_value(search_text).lower()
    if search_value:
        search_columns = [
            "country_name",
            "plant_name",
            "parent_source_name",
            "source_name",
            "component_hint",
            "notes",
        ]
        combined_search = filtered_df[search_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        filtered_df = filtered_df[combined_search.str.contains(search_value, regex=False)]

    return filtered_df.reset_index(drop=True)


def _create_mapping_table():
    table_config = StandardTableStyleManager.get_base_datatable_config()

    columns = []
    for column_name in DISPLAY_COLUMNS:
        column_config = {
            "name": column_name.replace("_", " ").title(),
            "id": column_name,
            "editable": column_name in EDITABLE_COLUMNS,
        }
        if column_name in {"train", "allocation_share"}:
            column_config["type"] = "numeric"
        columns.append(column_config)

    return dash_table.DataTable(
        id="train-name-mappings-table",
        columns=columns,
        data=[],
        editable=True,
        sort_action="native",
        filter_action="native",
        page_action="native",
        page_size=50,
        export_format="xlsx",
        export_headers="display",
        style_table=table_config["style_table"],
        style_header=table_config["style_header"],
        style_cell=table_config["style_cell"],
        style_data_conditional=table_config["style_data_conditional"] + [
            {
                "if": {"column_id": "source_name"},
                "backgroundColor": "#f8fafc",
                "fontWeight": "600",
                "color": "#1e3a5f",
            },
            {
                "if": {"column_id": "scope_hint"},
                "backgroundColor": "rgba(59, 130, 246, 0.06)",
            },
            {
                "if": {"column_id": "component_hint"},
                "backgroundColor": "rgba(59, 130, 246, 0.06)",
            },
            {
                "if": {"column_id": "train"},
                "backgroundColor": "rgba(34, 197, 94, 0.08)",
                "fontWeight": "700",
            },
            {
                "if": {"column_id": "allocation_share"},
                "backgroundColor": "rgba(34, 197, 94, 0.08)",
            },
        ],
        style_cell_conditional=[
            {"if": {"column_id": "country_name"}, "minWidth": "140px", "textAlign": "left"},
            {"if": {"column_id": "plant_name"}, "minWidth": "220px", "maxWidth": "280px", "textAlign": "left"},
            {"if": {"column_id": "provider"}, "minWidth": "130px", "textAlign": "left"},
            {"if": {"column_id": "parent_source_field"}, "minWidth": "130px", "textAlign": "left"},
            {"if": {"column_id": "parent_source_name"}, "minWidth": "240px", "maxWidth": "320px", "textAlign": "left"},
            {"if": {"column_id": "source_field"}, "minWidth": "150px", "textAlign": "left"},
            {"if": {"column_id": "source_name"}, "minWidth": "220px", "maxWidth": "280px", "textAlign": "left"},
            {"if": {"column_id": "scope_hint"}, "minWidth": "110px", "textAlign": "left"},
            {"if": {"column_id": "component_hint"}, "minWidth": "150px", "maxWidth": "220px", "textAlign": "left"},
            {"if": {"column_id": "train"}, "minWidth": "80px", "textAlign": "center"},
            {"if": {"column_id": "allocation_share"}, "minWidth": "120px", "textAlign": "center"},
            {"if": {"column_id": "notes"}, "minWidth": "220px", "maxWidth": "320px", "textAlign": "left"},
        ],
    )


layout = html.Div(
    [
        dcc.Store(id="train-name-mappings-data-store", storage_type="memory"),
        dcc.Interval(
            id="train-name-mappings-load-trigger",
            interval=1000 * 60 * 60 * 24,
            n_intervals=0,
            max_intervals=1,
        ),
        html.Div(
            [
                html.H2("Train Names Mapping", className="page-title"),
                html.P(
                    "Canonical train mapping layer used to reconcile Woodmac lng_train_name_short and Energy Aspects train_name into plant-scoped numeric trains.",
                    className="text-secondary",
                    style={"marginBottom": "20px"},
                ),
            ],
            style={"marginBottom": "30px", "padding": "20px 20px 0px 20px"},
        ),
        html.Div(id="train-name-mappings-summary-cards-container", style={"padding": "0px 20px"}),
        html.Div(
            [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.H5(
                                            "Filters",
                                            className="text-primary font-bold",
                                            style={"marginBottom": "15px"},
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Country", className="text-secondary font-semibold", style={"fontSize": "13px", "marginBottom": "5px"}),
                                                        dcc.Dropdown(id="train-name-country-filter", multi=True, placeholder="Select country(s)...", style={"fontSize": "13px"}),
                                                    ],
                                                    width=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Plant", className="text-secondary font-semibold", style={"fontSize": "13px", "marginBottom": "5px"}),
                                                        dcc.Dropdown(id="train-name-plant-filter", multi=True, placeholder="Select plant(s)...", style={"fontSize": "13px"}),
                                                    ],
                                                    width=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Provider", className="text-secondary font-semibold", style={"fontSize": "13px", "marginBottom": "5px"}),
                                                        dcc.Dropdown(id="train-name-provider-filter", multi=True, placeholder="Select provider(s)...", style={"fontSize": "13px"}),
                                                    ],
                                                    width=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Parent Source Field", className="text-secondary font-semibold", style={"fontSize": "13px", "marginBottom": "5px"}),
                                                        dcc.Dropdown(id="train-name-parent-source-field-filter", multi=True, placeholder="Select parent field(s)...", style={"fontSize": "13px"}),
                                                    ],
                                                    width=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Source Field", className="text-secondary font-semibold", style={"fontSize": "13px", "marginBottom": "5px"}),
                                                        dcc.Dropdown(id="train-name-source-field-filter", multi=True, placeholder="Select source field(s)...", style={"fontSize": "13px"}),
                                                    ],
                                                    width=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Scope Hint", className="text-secondary font-semibold", style={"fontSize": "13px", "marginBottom": "5px"}),
                                                        dcc.Dropdown(id="train-name-scope-filter", multi=True, placeholder="Select scope hint(s)...", style={"fontSize": "13px"}),
                                                    ],
                                                    width=2,
                                                ),
                                            ],
                                            className="g-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Search", className="text-secondary font-semibold", style={"fontSize": "13px", "marginBottom": "5px"}),
                                                        dcc.Input(
                                                            id="train-name-search-input",
                                                            type="text",
                                                            placeholder="Search source name, plant, parent source, component, notes...",
                                                            style={
                                                                "width": "100%",
                                                                "fontSize": "13px",
                                                                "padding": "8px 10px",
                                                                "border": "1px solid #ced4da",
                                                                "borderRadius": "4px",
                                                            },
                                                        ),
                                                    ],
                                                    width=9,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(" ", style={"display": "block", "marginBottom": "5px"}),
                                                        dbc.Button(
                                                            "Clear Filters",
                                                            id="train-name-clear-filters-btn",
                                                            color="secondary",
                                                            outline=True,
                                                            size="sm",
                                                            style={"width": "100%", "marginTop": "20px"},
                                                        ),
                                                    ],
                                                    width=3,
                                                ),
                                            ],
                                            className="g-3",
                                            style={"marginTop": "4px"},
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    className="shadow-sm mb-4",
                )
            ],
            style={"padding": "0px 20px"},
        ),
        html.Div(
            [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.H5(
                                            "Train Mapping Data",
                                            className="text-primary font-bold",
                                            style={"marginBottom": "10px"},
                                        ),
                                        html.P(
                                            "This table stores train-mapping exceptions only. Simple raw Train N labels are inferred automatically in the Capacity page. Edit scope_hint, component_hint, train, allocation_share, and notes here for the cases that still need overrides, splits, or manual alignment.",
                                            className="text-secondary",
                                            style={"marginBottom": "12px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    id="train-name-mappings-table-summary",
                                                    className="text-secondary",
                                                    style={"fontSize": "13px"},
                                                ),
                                                dbc.Button(
                                                    "Save Train Mappings",
                                                    id="train-name-save-btn",
                                                    color="primary",
                                                    size="sm",
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "justifyContent": "space-between",
                                                "alignItems": "center",
                                                "marginBottom": "12px",
                                                "gap": "12px",
                                            },
                                        ),
                                        html.Div(
                                            id="train-name-save-message",
                                            style={"marginBottom": "10px"},
                                        ),
                                        _create_mapping_table(),
                                    ]
                                )
                            ]
                        )
                    ],
                    className="shadow-sm mb-4",
                )
            ],
            style={"padding": "0px 20px"},
        ),
        html.Div(
            [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.H5(
                                            "Mapping Overview",
                                            className="text-primary font-bold",
                                            style={"marginBottom": "15px"},
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(dcc.Graph(id="train-name-provider-chart"), width=4),
                                                dbc.Col(dcc.Graph(id="train-name-plant-chart"), width=4),
                                                dbc.Col(dcc.Graph(id="train-name-scope-chart"), width=4),
                                            ],
                                            className="g-3",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    className="shadow-sm mb-4",
                )
            ],
            style={"padding": "0px 20px", "marginBottom": "30px"},
        ),
    ]
)


@callback(
    Output("train-name-mappings-data-store", "data"),
    Input("train-name-mappings-load-trigger", "n_intervals"),
    Input("global-refresh-button", "n_clicks"),
)
def load_train_name_mapping_data(_, __):
    df = fetch_train_name_mappings_data(engine)
    return df.to_dict("records") if not df.empty else []


@callback(
    Output("train-name-country-filter", "options"),
    Output("train-name-plant-filter", "options"),
    Output("train-name-provider-filter", "options"),
    Output("train-name-parent-source-field-filter", "options"),
    Output("train-name-source-field-filter", "options"),
    Output("train-name-scope-filter", "options"),
    Output("train-name-mappings-summary-cards-container", "children"),
    Input("train-name-mappings-data-store", "data"),
)
def update_filter_options_and_summary(data):
    if not data:
        return [], [], [], [], [], [], html.Div("Loading...")

    df = pd.DataFrame(data)
    country_options = [{"label": value, "value": value} for value in sorted(df["country_name"].replace("", pd.NA).dropna().unique())]
    plant_options = [{"label": value, "value": value} for value in sorted(df["plant_name"].replace("", pd.NA).dropna().unique())]
    provider_options = [{"label": value, "value": value} for value in sorted(df["provider"].replace("", pd.NA).dropna().unique())]
    parent_source_field_options = [{"label": value, "value": value} for value in sorted(df["parent_source_field"].replace("", pd.NA).dropna().unique())]
    source_field_options = [{"label": value, "value": value} for value in sorted(df["source_field"].replace("", pd.NA).dropna().unique())]
    scope_options = [{"label": value, "value": value} for value in sorted(df["scope_hint"].replace("", pd.NA).dropna().unique())]

    return (
        country_options,
        plant_options,
        provider_options,
        parent_source_field_options,
        source_field_options,
        scope_options,
        create_summary_cards(df),
    )


@callback(
    Output("train-name-mappings-table-summary", "children"),
    Output("train-name-mappings-table", "data"),
    Output("train-name-provider-chart", "figure"),
    Output("train-name-plant-chart", "figure"),
    Output("train-name-scope-chart", "figure"),
    Input("train-name-mappings-data-store", "data"),
    Input("train-name-country-filter", "value"),
    Input("train-name-plant-filter", "value"),
    Input("train-name-provider-filter", "value"),
    Input("train-name-parent-source-field-filter", "value"),
    Input("train-name-source-field-filter", "value"),
    Input("train-name-scope-filter", "value"),
    Input("train-name-search-input", "value"),
)
def update_mapping_views(data, countries, plants, providers, parent_source_fields, source_fields, scope_hints, search_text):
    if not data:
        empty_fig = _build_empty_figure("No train name mapping data available.")
        return "No rows loaded.", [], empty_fig, empty_fig, empty_fig

    df = pd.DataFrame(data)
    filtered_df = _filter_mapping_df(
        df,
        countries,
        plants,
        providers,
        parent_source_fields,
        source_fields,
        scope_hints,
        search_text,
    )

    canonical_trains = (
        filtered_df[["country_name", "plant_name", "train"]]
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .shape[0]
    )
    summary_text = (
        f"{len(filtered_df):,} rows shown | "
        f"{canonical_trains:,} canonical trains | "
        f"{filtered_df['plant_name'].replace('', pd.NA).dropna().nunique():,} plants"
    )

    provider_series = filtered_df["provider"].replace("", "Unspecified").value_counts().sort_values()
    plant_series = filtered_df["plant_name"].replace("", "Unspecified").value_counts().head(15).sort_values()
    scope_series = filtered_df["scope_hint"].replace("", "Unspecified").value_counts().sort_values()

    provider_fig = _build_bar_figure(provider_series, "Mappings By Provider", "#1d4ed8")
    plant_fig = _build_bar_figure(plant_series, "Top Plants By Mapping Rows", "#0f766e")
    scope_fig = _build_bar_figure(scope_series, "Mappings By Scope Hint", "#7c3aed")

    return summary_text, filtered_df.to_dict("records"), provider_fig, plant_fig, scope_fig


@callback(
    Output("train-name-country-filter", "value"),
    Output("train-name-plant-filter", "value"),
    Output("train-name-provider-filter", "value"),
    Output("train-name-parent-source-field-filter", "value"),
    Output("train-name-source-field-filter", "value"),
    Output("train-name-scope-filter", "value"),
    Output("train-name-search-input", "value"),
    Input("train-name-clear-filters-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_filters(_):
    return None, None, None, None, None, None, ""


@callback(
    Output("train-name-save-message", "children"),
    Output("train-name-mappings-data-store", "data", allow_duplicate=True),
    Input("train-name-save-btn", "n_clicks"),
    State("train-name-mappings-table", "data"),
    prevent_initial_call=True,
)
def save_train_name_mappings(n_clicks, table_data):
    if not n_clicks:
        raise PreventUpdate

    mapping_df = pd.DataFrame(table_data or [])
    if mapping_df.empty:
        return (
            html.Div(
                "No rows are currently loaded in the table.",
                style={"color": "#9a3412", "fontSize": "12px", "fontWeight": "600"},
            ),
            no_update,
        )

    for column_name in DISPLAY_COLUMNS:
        if column_name not in mapping_df.columns:
            mapping_df[column_name] = ""

    for column_name in DISPLAY_COLUMNS:
        if column_name == "train":
            mapping_df[column_name] = mapping_df[column_name].map(_clean_train_value)
        elif column_name == "allocation_share":
            mapping_df[column_name] = mapping_df[column_name].map(_clean_allocation_share)
        else:
            mapping_df[column_name] = mapping_df[column_name].map(_clean_text_value)

    invalid_train_mask = mapping_df["train"].eq("")
    if invalid_train_mask.any():
        return (
            html.Div(
                "Every saved row must have a positive numeric train value.",
                style={"color": "#991b1b", "fontSize": "12px", "fontWeight": "600"},
            ),
            no_update,
        )

    invalid_share_mask = mapping_df["allocation_share"].eq("")
    if invalid_share_mask.any():
        return (
            html.Div(
                "Every saved row must have a positive allocation_share value.",
                style={"color": "#991b1b", "fontSize": "12px", "fontWeight": "600"},
            ),
            no_update,
        )

    save_df = mapping_df[DISPLAY_COLUMNS].copy()
    for column_name in DISPLAY_COLUMNS:
        if column_name not in {"train", "allocation_share"}:
            save_df[column_name] = save_df[column_name].replace("", pd.NA)

    save_df = save_df.dropna(
        subset=RAW_SOURCE_KEY_COLUMNS + ["train", "allocation_share"]
    ).copy()
    save_df["train"] = save_df["train"].astype(int)
    save_df["allocation_share"] = save_df["allocation_share"].astype(float)
    save_df = save_df.drop_duplicates(
        subset=RAW_SOURCE_KEY_COLUMNS + ["train"],
        keep="last",
    )

    if save_df.empty:
        return (
            html.Div(
                "Fill the required source fields, train, and allocation_share in the currently loaded rows before saving.",
                style={"color": "#9a3412", "fontSize": "12px", "fontWeight": "600"},
            ),
            no_update,
        )

    share_check = (
        save_df.groupby(RAW_SOURCE_KEY_COLUMNS, as_index=False)["allocation_share"]
        .sum()
    )
    invalid_share_df = share_check[
        ~share_check["allocation_share"].map(
            lambda value: math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-9)
        )
    ]
    if not invalid_share_df.empty:
        return (
            html.Div(
                "allocation_share must sum to 1.0 for each raw source row before saving.",
                style={"color": "#991b1b", "fontSize": "12px", "fontWeight": "600"},
            ),
            no_update,
        )

    table_ref = f'"{DB_SCHEMA}"."{TABLE_NAME}"'
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_ref} (
                        country_name TEXT NOT NULL,
                        plant_name TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        parent_source_field TEXT NOT NULL,
                        parent_source_name TEXT NOT NULL,
                        source_field TEXT NOT NULL,
                        source_name TEXT NOT NULL,
                        scope_hint TEXT,
                        component_hint TEXT,
                        train INTEGER NOT NULL,
                        allocation_share DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                        notes TEXT
                    )
                    """
                )
            )

            key_df = save_df[RAW_SOURCE_KEY_COLUMNS].drop_duplicates().reset_index(drop=True)
            for row in key_df.where(pd.notna(key_df), None).to_dict("records"):
                connection.execute(
                    text(
                        f"""
                        DELETE FROM {table_ref}
                        WHERE country_name = :country_name
                          AND plant_name = :plant_name
                          AND provider = :provider
                          AND parent_source_field = :parent_source_field
                          AND parent_source_name = :parent_source_name
                          AND source_field = :source_field
                          AND source_name = :source_name
                        """
                    ),
                    row,
                )

            for row in save_df.where(pd.notna(save_df), None).to_dict("records"):
                connection.execute(
                    text(
                        f"""
                        INSERT INTO {table_ref} (
                            country_name,
                            plant_name,
                            provider,
                            parent_source_field,
                            parent_source_name,
                            source_field,
                            source_name,
                            scope_hint,
                            component_hint,
                            train,
                            allocation_share,
                            notes
                        ) VALUES (
                            :country_name,
                            :plant_name,
                            :provider,
                            :parent_source_field,
                            :parent_source_name,
                            :source_field,
                            :source_name,
                            :scope_hint,
                            :component_hint,
                            :train,
                            :allocation_share,
                            :notes
                        )
                        """
                    ),
                    row,
                )

        refreshed_df = fetch_train_name_mappings_data(engine)
        return (
            html.Div(
                f"Saved {len(save_df):,} train mapping rows to at_lng.mapping_plant_train_name.",
                style={"color": "#166534", "fontSize": "12px", "fontWeight": "600"},
            ),
            refreshed_df.to_dict("records"),
        )
    except Exception as exc:
        return (
            html.Div(
                f"Train mapping save failed: {exc}",
                style={"color": "#991b1b", "fontSize": "12px", "fontWeight": "600"},
            ),
            no_update,
        )
