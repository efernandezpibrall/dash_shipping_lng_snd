from dash import html, dcc, dash_table, callback, Output, Input, State, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import configparser
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
TABLE_NAME = "mapping_plant_name"

engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)

DISPLAY_COLUMNS = [
    "country_name",
    "provider",
    "source_field",
    "source_name",
    "scope_hint",
    "component_hint",
    "plant_name",
]

EDITABLE_COLUMNS = {"scope_hint", "component_hint", "plant_name"}


def _clean_text_value(value):
    if pd.isna(value):
        return ""

    text_value = str(value).strip()
    if not text_value:
        return ""

    return " ".join(text_value.split())


def fetch_plant_name_mappings_data(db_engine, schema=DB_SCHEMA):
    try:
        with db_engine.connect() as conn:
            query = text(
                f"""
                SELECT
                    country_name,
                    provider,
                    source_field,
                    source_name,
                    scope_hint,
                    component_hint,
                    plant_name
                FROM {schema}.{TABLE_NAME}
                ORDER BY country_name, provider, source_field, source_name
                """
            )
            df = pd.read_sql(query, conn)
    except Exception as exc:
        print(f"Error fetching plant name mappings data: {exc}")
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    for column_name in DISPLAY_COLUMNS:
        df[column_name] = df[column_name].map(_clean_text_value)

    df = df.drop_duplicates(
        subset=["country_name", "provider", "source_field", "source_name"],
        keep="last",
    ).reset_index(drop=True)

    return df


def create_summary_cards(df):
    if df.empty:
        return html.Div("No plant name mapping data available")

    total_aliases = len(df)
    standardized_plants = df["plant_name"].replace("", pd.NA).dropna().nunique()
    countries = df["country_name"].replace("", pd.NA).dropna().nunique()
    providers = df["provider"].replace("", pd.NA).dropna().nunique()

    card_specs = [
        ("Alias Rows", f"{total_aliases:,}"),
        ("Standardized Plants", f"{standardized_plants:,}"),
        ("Countries", f"{countries:,}"),
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
    providers,
    source_fields,
    scope_hints,
    search_text,
) -> pd.DataFrame:
    filtered_df = df.copy()

    if countries:
        filtered_df = filtered_df[filtered_df["country_name"].isin(countries)]
    if providers:
        filtered_df = filtered_df[filtered_df["provider"].isin(providers)]
    if source_fields:
        filtered_df = filtered_df[filtered_df["source_field"].isin(source_fields)]
    if scope_hints:
        filtered_df = filtered_df[filtered_df["scope_hint"].isin(scope_hints)]

    search_value = _clean_text_value(search_text).lower()
    if search_value:
        search_columns = ["country_name", "source_name", "component_hint", "plant_name"]
        combined_search = filtered_df[search_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        filtered_df = filtered_df[combined_search.str.contains(search_value, regex=False)]

    return filtered_df.reset_index(drop=True)


def _create_mapping_table():
    table_config = StandardTableStyleManager.get_base_datatable_config()

    columns = []
    for column_name in DISPLAY_COLUMNS:
        columns.append(
            {
                "name": column_name.replace("_", " ").title(),
                "id": column_name,
                "editable": column_name in EDITABLE_COLUMNS,
            }
        )

    return dash_table.DataTable(
        id="plant-name-mappings-table",
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
                "if": {"column_id": "plant_name"},
                "backgroundColor": "rgba(34, 197, 94, 0.08)",
                "fontWeight": "600",
            },
        ],
        style_cell_conditional=[
            {"if": {"column_id": "country_name"}, "minWidth": "150px", "textAlign": "left"},
            {"if": {"column_id": "provider"}, "minWidth": "130px", "textAlign": "left"},
            {"if": {"column_id": "source_field"}, "minWidth": "130px", "textAlign": "left"},
            {"if": {"column_id": "source_name"}, "minWidth": "260px", "maxWidth": "340px", "textAlign": "left"},
            {"if": {"column_id": "scope_hint"}, "minWidth": "120px", "textAlign": "left"},
            {"if": {"column_id": "component_hint"}, "minWidth": "180px", "maxWidth": "240px", "textAlign": "left"},
            {"if": {"column_id": "plant_name"}, "minWidth": "220px", "maxWidth": "300px", "textAlign": "left"},
        ],
    )


layout = html.Div(
    [
        dcc.Store(id="plant-name-mappings-data-store", storage_type="memory"),
        dcc.Interval(
            id="plant-name-mappings-load-trigger",
            interval=1000 * 60 * 60 * 24,
            n_intervals=0,
            max_intervals=1,
        ),
        html.Div(
            [
                html.H2("Plant Names Mapping", className="page-title"),
                html.P(
                    "Canonical mapping layer used to reconcile Woodmac plant_name and Energy Aspects project_name into a shared plant name.",
                    className="text-secondary",
                    style={"marginBottom": "20px"},
                ),
            ],
            style={"marginBottom": "30px", "padding": "20px 20px 0px 20px"},
        ),
        html.Div(id="plant-name-mappings-summary-cards-container", style={"padding": "0px 20px"}),
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
                                                        html.Label(
                                                            "Country",
                                                            className="text-secondary font-semibold",
                                                            style={"fontSize": "13px", "marginBottom": "5px"},
                                                        ),
                                                        dcc.Dropdown(
                                                            id="plant-name-country-filter",
                                                            multi=True,
                                                            placeholder="Select country(s)...",
                                                            style={"fontSize": "13px"},
                                                        ),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Provider",
                                                            className="text-secondary font-semibold",
                                                            style={"fontSize": "13px", "marginBottom": "5px"},
                                                        ),
                                                        dcc.Dropdown(
                                                            id="plant-name-provider-filter",
                                                            multi=True,
                                                            placeholder="Select provider(s)...",
                                                            style={"fontSize": "13px"},
                                                        ),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Source Field",
                                                            className="text-secondary font-semibold",
                                                            style={"fontSize": "13px", "marginBottom": "5px"},
                                                        ),
                                                        dcc.Dropdown(
                                                            id="plant-name-source-field-filter",
                                                            multi=True,
                                                            placeholder="Select source field(s)...",
                                                            style={"fontSize": "13px"},
                                                        ),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Scope Hint",
                                                            className="text-secondary font-semibold",
                                                            style={"fontSize": "13px", "marginBottom": "5px"},
                                                        ),
                                                        dcc.Dropdown(
                                                            id="plant-name-scope-filter",
                                                            multi=True,
                                                            placeholder="Select scope hint(s)...",
                                                            style={"fontSize": "13px"},
                                                        ),
                                                    ],
                                                    width=3,
                                                ),
                                            ],
                                            className="g-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Search",
                                                            className="text-secondary font-semibold",
                                                            style={"fontSize": "13px", "marginBottom": "5px"},
                                                        ),
                                                        dcc.Input(
                                                            id="plant-name-search-input",
                                                            type="text",
                                                            placeholder="Search source name, plant name, country, component...",
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
                                                            id="plant-name-clear-filters-btn",
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
                                            "Plant Mapping Data",
                                            className="text-primary font-bold",
                                            style={"marginBottom": "10px"},
                                        ),
                                        html.P(
                                            "Edit scope_hint, component_hint, and plant_name directly in the table, then save the current rows back to at_lng.mapping_plant_name.",
                                            className="text-secondary",
                                            style={"marginBottom": "12px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    id="plant-name-mappings-table-summary",
                                                    className="text-secondary",
                                                    style={"fontSize": "13px"},
                                                ),
                                                dbc.Button(
                                                    "Save Plant Mappings",
                                                    id="plant-name-save-btn",
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
                                            id="plant-name-save-message",
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
                                                dbc.Col(dcc.Graph(id="plant-name-provider-chart"), width=4),
                                                dbc.Col(dcc.Graph(id="plant-name-country-chart"), width=4),
                                                dbc.Col(dcc.Graph(id="plant-name-scope-chart"), width=4),
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
    Output("plant-name-mappings-data-store", "data"),
    Input("plant-name-mappings-load-trigger", "n_intervals"),
    Input("global-refresh-button", "n_clicks"),
)
def load_plant_name_mapping_data(_, __):
    df = fetch_plant_name_mappings_data(engine)
    return df.to_dict("records") if not df.empty else []


@callback(
    Output("plant-name-country-filter", "options"),
    Output("plant-name-provider-filter", "options"),
    Output("plant-name-source-field-filter", "options"),
    Output("plant-name-scope-filter", "options"),
    Output("plant-name-mappings-summary-cards-container", "children"),
    Input("plant-name-mappings-data-store", "data"),
)
def update_filter_options_and_summary(data):
    if not data:
        return [], [], [], [], html.Div("Loading...")

    df = pd.DataFrame(data)
    country_options = [
        {"label": value, "value": value}
        for value in sorted(df["country_name"].replace("", pd.NA).dropna().unique())
    ]
    provider_options = [
        {"label": value, "value": value}
        for value in sorted(df["provider"].replace("", pd.NA).dropna().unique())
    ]
    source_field_options = [
        {"label": value, "value": value}
        for value in sorted(df["source_field"].replace("", pd.NA).dropna().unique())
    ]
    scope_options = [
        {"label": value, "value": value}
        for value in sorted(df["scope_hint"].replace("", pd.NA).dropna().unique())
    ]

    return (
        country_options,
        provider_options,
        source_field_options,
        scope_options,
        create_summary_cards(df),
    )


@callback(
    Output("plant-name-mappings-table-summary", "children"),
    Output("plant-name-mappings-table", "data"),
    Output("plant-name-provider-chart", "figure"),
    Output("plant-name-country-chart", "figure"),
    Output("plant-name-scope-chart", "figure"),
    Input("plant-name-mappings-data-store", "data"),
    Input("plant-name-country-filter", "value"),
    Input("plant-name-provider-filter", "value"),
    Input("plant-name-source-field-filter", "value"),
    Input("plant-name-scope-filter", "value"),
    Input("plant-name-search-input", "value"),
)
def update_mapping_views(data, countries, providers, source_fields, scope_hints, search_text):
    if not data:
        empty_fig = _build_empty_figure("No plant name mapping data available.")
        return "No rows loaded.", [], empty_fig, empty_fig, empty_fig

    df = pd.DataFrame(data)
    filtered_df = _filter_mapping_df(
        df,
        countries,
        providers,
        source_fields,
        scope_hints,
        search_text,
    )

    summary_text = (
        f"{len(filtered_df):,} rows shown | "
        f"{filtered_df['plant_name'].replace('', pd.NA).dropna().nunique():,} standardized plants | "
        f"{filtered_df['country_name'].replace('', pd.NA).dropna().nunique():,} countries"
    )

    provider_series = filtered_df["provider"].replace("", "Unspecified").value_counts().sort_values()
    country_series = filtered_df["country_name"].replace("", "Unspecified").value_counts().head(15).sort_values()
    scope_series = filtered_df["scope_hint"].replace("", "Unspecified").value_counts().sort_values()

    provider_fig = _build_bar_figure(provider_series, "Aliases By Provider", "#1d4ed8")
    country_fig = _build_bar_figure(country_series, "Top Countries By Alias Rows", "#0f766e")
    scope_fig = _build_bar_figure(scope_series, "Aliases By Scope Hint", "#7c3aed")

    return summary_text, filtered_df.to_dict("records"), provider_fig, country_fig, scope_fig


@callback(
    Output("plant-name-country-filter", "value"),
    Output("plant-name-provider-filter", "value"),
    Output("plant-name-source-field-filter", "value"),
    Output("plant-name-scope-filter", "value"),
    Output("plant-name-search-input", "value"),
    Input("plant-name-clear-filters-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_filters(_):
    return None, None, None, None, ""


@callback(
    Output("plant-name-save-message", "children"),
    Output("plant-name-mappings-data-store", "data", allow_duplicate=True),
    Input("plant-name-save-btn", "n_clicks"),
    State("plant-name-mappings-table", "data"),
    prevent_initial_call=True,
)
def save_plant_name_mappings(n_clicks, table_data):
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
        mapping_df[column_name] = mapping_df[column_name].map(_clean_text_value)

    save_df = mapping_df[DISPLAY_COLUMNS].copy()
    for column_name in DISPLAY_COLUMNS:
        save_df[column_name] = save_df[column_name].replace("", pd.NA)

    save_df = save_df.dropna(
        subset=["country_name", "provider", "source_field", "source_name", "plant_name"]
    ).copy()
    save_df = save_df.drop_duplicates(
        subset=["country_name", "provider", "source_field", "source_name"],
        keep="last",
    )

    if save_df.empty:
        return (
            html.Div(
                "Fill at least one plant_name in the currently loaded rows before saving.",
                style={"color": "#9a3412", "fontSize": "12px", "fontWeight": "600"},
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
                        provider TEXT NOT NULL,
                        source_field TEXT NOT NULL,
                        source_name TEXT NOT NULL,
                        scope_hint TEXT,
                        component_hint TEXT,
                        plant_name TEXT NOT NULL
                    )
                    """
                )
            )

            for row in save_df.where(pd.notna(save_df), None).to_dict("records"):
                connection.execute(
                    text(
                        f"""
                        DELETE FROM {table_ref}
                        WHERE country_name = :country_name
                          AND provider = :provider
                          AND source_field = :source_field
                          AND source_name = :source_name
                        """
                    ),
                    row,
                )
                connection.execute(
                    text(
                        f"""
                        INSERT INTO {table_ref} (
                            country_name,
                            provider,
                            source_field,
                            source_name,
                            scope_hint,
                            component_hint,
                            plant_name
                        ) VALUES (
                            :country_name,
                            :provider,
                            :source_field,
                            :source_name,
                            :scope_hint,
                            :component_hint,
                            :plant_name
                        )
                        """
                    ),
                    row,
                )

        refreshed_df = fetch_plant_name_mappings_data(engine)
        return (
            html.Div(
                f"Saved {len(save_df):,} plant mappings to at_lng.mapping_plant_name.",
                style={"color": "#166534", "fontSize": "12px", "fontWeight": "600"},
            ),
            refreshed_df.to_dict("records"),
        )
    except Exception as exc:
        return (
            html.Div(
                f"Plant mapping save failed: {exc}",
                style={"color": "#991b1b", "fontSize": "12px", "fontWeight": "600"},
            ),
            no_update,
        )
