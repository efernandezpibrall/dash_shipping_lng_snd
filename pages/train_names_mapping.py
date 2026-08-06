import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update
from dash.exceptions import PreventUpdate

from fundamentals.lng.terminals.terminal_registry_utils import (
    find_terminal_train_candidates,
    replace_provider_source_allocations,
)
from utils.database import DB_SCHEMA, engine
from utils.mappings_section import create_mappings_section_header

SOURCE_COLUMNS = ["provider_name", "provider_plant_id", "provider_train_id"]
DISPLAY_COLUMNS = SOURCE_COLUMNS + [
    "country_name",
    "terminal_name",
    "train_label",
    "allocation_share",
]


def fetch_provider_allocations(db_engine, schema=DB_SCHEMA):
    query = f"""
        SELECT links.provider_name,
               links.provider_plant_id,
               links.provider_train_id,
               terminals.country_name,
               terminals.terminal_name,
               trains.train_label,
               links.allocation_share::double precision AS allocation_share
        FROM {schema}.fundamentals_terminal_train_provider_links links
        JOIN {schema}.fundamentals_terminal_train_registry trains USING (train_key)
        JOIN {schema}.fundamentals_terminal_registry terminals USING (terminal_key)
        ORDER BY links.provider_name, links.provider_plant_id,
                 links.provider_train_id, terminals.country_name,
                 terminals.terminal_name, trains.train_label
    """
    return pd.read_sql_query(query, db_engine)


def _clean_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _provider_allocation_grid():
    column_defs = [
        {
            "field": "provider_name",
            "headerName": "Provider",
            "editable": True,
            "minWidth": 145,
        },
        {
            "field": "provider_plant_id",
            "headerName": "Provider Plant ID",
            "editable": True,
            "minWidth": 190,
        },
        {
            "field": "provider_train_id",
            "headerName": "Provider Train ID",
            "editable": True,
            "minWidth": 190,
        },
        {
            "field": "country_name",
            "headerName": "Canonical Country",
            "editable": True,
            "minWidth": 160,
        },
        {
            "field": "terminal_name",
            "headerName": "Canonical Terminal",
            "editable": True,
            "minWidth": 230,
        },
        {
            "field": "train_label",
            "headerName": "Canonical Train",
            "editable": True,
            "minWidth": 180,
        },
        {
            "field": "allocation_share",
            "headerName": "Allocation",
            "editable": True,
            "type": "numericColumn",
            "valueFormatter": {"function": "params.value == null ? '' : Number(params.value).toFixed(8)"},
            "minWidth": 130,
        },
    ]
    return dag.AgGrid(
        id="train-name-mappings-table",
        columnDefs=column_defs,
        rowData=[],
        defaultColDef={
            "sortable": True,
            "filter": True,
            "resizable": True,
        },
        dashGridOptions={
            "rowSelection": {"mode": "multiRow"},
            "animateRows": False,
            "pagination": True,
            "paginationPageSize": 50,
        },
        style={"height": "620px", "width": "100%"},
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
        create_mappings_section_header(
            title="Provider Train Allocations",
            description=(
                "Authoritative provider-source allocations to Capacity business trains. "
                "Several sources may aggregate into one train, and one source may be split "
                "across several trains when the allocation totals exactly 1.0."
            ),
            active_href="/train_names_mapping",
        ),
        html.Div(
            [
                dbc.Card(
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="train-name-provider-filter",
                                            multi=True,
                                            placeholder="Filter provider",
                                        ),
                                        width=3,
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="train-name-country-filter",
                                            multi=True,
                                            placeholder="Filter canonical country",
                                        ),
                                        width=3,
                                    ),
                                    dbc.Col(
                                        dcc.Input(
                                            id="train-name-search-input",
                                            type="text",
                                            placeholder="Search source or canonical names",
                                            style={"width": "100%", "padding": "7px 10px"},
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Clear",
                                            id="train-name-clear-filters-btn",
                                            color="secondary",
                                            outline=True,
                                            size="sm",
                                            style={"width": "100%"},
                                        ),
                                        width=2,
                                    ),
                                ],
                                className="g-3",
                            )
                        ]
                    ),
                    className="shadow-sm mb-4",
                ),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        id="train-name-mappings-table-summary",
                                        className="text-secondary",
                                    ),
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Add allocation",
                                                id="train-name-add-row-btn",
                                                color="secondary",
                                                outline=True,
                                                size="sm",
                                            ),
                                            dbc.Button(
                                                "Remove selected",
                                                id="train-name-remove-row-btn",
                                                color="danger",
                                                outline=True,
                                                size="sm",
                                            ),
                                            dbc.Button(
                                                "Save allocations",
                                                id="train-name-save-btn",
                                                color="primary",
                                                size="sm",
                                            ),
                                        ],
                                        style={"display": "flex", "gap": "8px"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "marginBottom": "12px",
                                },
                            ),
                            html.Div(id="train-name-save-message", style={"marginBottom": "10px"}),
                            _provider_allocation_grid(),
                        ]
                    ),
                    className="shadow-sm mb-4",
                ),
            ],
            style={"padding": "0 20px 30px"},
        ),
    ]
)


@callback(
    Output("train-name-mappings-data-store", "data"),
    Input("train-name-mappings-load-trigger", "n_intervals"),
    Input("global-refresh-button", "n_clicks"),
)
def load_provider_allocations(_, __):
    frame = fetch_provider_allocations(engine)
    return frame.to_dict("records")


@callback(
    Output("train-name-provider-filter", "options"),
    Output("train-name-country-filter", "options"),
    Input("train-name-mappings-data-store", "data"),
)
def update_filter_options(data):
    frame = pd.DataFrame(data or [])
    if frame.empty:
        return [], []
    return (
        sorted(frame["provider_name"].dropna().unique()),
        sorted(frame["country_name"].dropna().unique()),
    )


@callback(
    Output("train-name-mappings-table-summary", "children"),
    Output("train-name-mappings-table", "rowData"),
    Input("train-name-mappings-data-store", "data"),
    Input("train-name-provider-filter", "value"),
    Input("train-name-country-filter", "value"),
    Input("train-name-search-input", "value"),
)
def update_allocation_view(data, providers, countries, search_text):
    frame = pd.DataFrame(data or [], columns=DISPLAY_COLUMNS)
    if providers:
        frame = frame[frame["provider_name"].isin(providers)]
    if countries:
        frame = frame[frame["country_name"].isin(countries)]
    if search_text:
        needle = str(search_text).strip().casefold()
        haystack = frame[DISPLAY_COLUMNS[:-1]].fillna("").astype(str).agg(" ".join, axis=1)
        frame = frame[haystack.str.casefold().str.contains(needle, regex=False)]
    source_count = frame[SOURCE_COLUMNS].drop_duplicates().shape[0] if not frame.empty else 0
    split_count = (
        frame.groupby(SOURCE_COLUMNS).size().gt(1).sum()
        if not frame.empty
        else 0
    )
    summary = (
        f"{len(frame):,} allocation rows | {source_count:,} provider sources | "
        f"{split_count:,} split sources"
    )
    return summary, frame.to_dict("records")


@callback(
    Output("train-name-provider-filter", "value"),
    Output("train-name-country-filter", "value"),
    Output("train-name-search-input", "value"),
    Input("train-name-clear-filters-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_filters(_):
    return None, None, ""


@callback(
    Output("train-name-mappings-table", "rowData", allow_duplicate=True),
    Input("train-name-add-row-btn", "n_clicks"),
    Input("train-name-remove-row-btn", "n_clicks"),
    State("train-name-mappings-table", "rowData"),
    State("train-name-mappings-table", "selectedRows"),
    prevent_initial_call=True,
)
def edit_allocation_rows(add_clicks, remove_clicks, rows, selected_rows):
    from dash import ctx

    rows = list(rows or [])
    if ctx.triggered_id == "train-name-add-row-btn":
        return rows + [{column: "" for column in DISPLAY_COLUMNS}]
    if ctx.triggered_id == "train-name-remove-row-btn":
        selected = {tuple(str(row.get(column, "")) for column in DISPLAY_COLUMNS) for row in selected_rows or []}
        return [
            row
            for row in rows
            if tuple(str(row.get(column, "")) for column in DISPLAY_COLUMNS) not in selected
        ]
    raise PreventUpdate


@callback(
    Output("train-name-save-message", "children"),
    Output("train-name-mappings-data-store", "data", allow_duplicate=True),
    Input("train-name-save-btn", "n_clicks"),
    State("train-name-mappings-table", "rowData"),
    prevent_initial_call=True,
)
def save_provider_allocations(n_clicks, table_data):
    if not n_clicks:
        raise PreventUpdate
    frame = pd.DataFrame(table_data or [])
    if frame.empty:
        return html.Div("No allocation rows to save.", style={"color": "#9a3412"}), no_update
    for column in DISPLAY_COLUMNS:
        if column not in frame:
            frame[column] = ""
    for column in DISPLAY_COLUMNS[:-1]:
        frame[column] = frame[column].map(_clean_text)
    frame["allocation_share"] = pd.to_numeric(frame["allocation_share"], errors="coerce")
    incomplete = frame[
        frame[DISPLAY_COLUMNS[:-1]].eq("").any(axis=1)
        | frame["allocation_share"].isna()
    ]
    if not incomplete.empty:
        return (
            html.Div(
                "Every row requires provider IDs, canonical country, terminal, train, and allocation.",
                style={"color": "#991b1b"},
            ),
            no_update,
        )

    resolved = []
    ambiguous = []
    for row in frame.to_dict("records"):
        matches = find_terminal_train_candidates(
            engine,
            row["country_name"],
            row["terminal_name"],
            row["train_label"],
        )
        if len(matches) != 1:
            ambiguous.append(
                f"{row['country_name']} / {row['terminal_name']} / {row['train_label']}"
            )
        else:
            resolved.append({**row, "train_key": str(matches[0]["train_key"])})
    if ambiguous:
        return (
            html.Div(
                "Canonical identities must resolve exactly once: " + "; ".join(ambiguous[:5]),
                style={"color": "#991b1b"},
            ),
            no_update,
        )

    resolved_frame = pd.DataFrame(resolved)
    try:
        changed = 0
        for source_key, rows in resolved_frame.groupby(SOURCE_COLUMNS, sort=True):
            result = replace_provider_source_allocations(
                engine,
                provider_name=source_key[0],
                provider_plant_id=source_key[1],
                provider_train_id=source_key[2],
                allocations=rows[["train_key", "allocation_share"]].to_dict("records"),
            )
            changed += int(result["changed"])
        refreshed = fetch_provider_allocations(engine)
        return (
            html.Div(
                f"Validated {resolved_frame[SOURCE_COLUMNS].drop_duplicates().shape[0]:,} source groups; "
                f"{changed:,} allocation set(s) changed.",
                style={"color": "#166534", "fontWeight": "600"},
            ),
            refreshed.to_dict("records"),
        )
    except Exception as exc:
        return (
            html.Div(f"Provider allocation save failed: {exc}", style={"color": "#991b1b"}),
            no_update,
        )
