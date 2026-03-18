"""
Terminal Output Adjustments Editor

This page allows analysts to:
1. Select and manage scenarios (base_view, best_view, test_1, test_2, etc.)
2. View Woodmac baseline data with adjustments overlaid
3. Create/edit/delete adjustments for monthly terminal production
4. Duplicate scenarios to test different assumptions
5. View adjustment history with full audit trail

Uses append-only design - never updates or deletes, only inserts with timestamps.
"""

import datetime
import base64
import io
import pandas as pd
import numpy as np
import dash
from dash import html, dash_table, dcc, callback, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
from sqlalchemy import create_engine, text
import configparser
import os
import sys

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

from fundamentals.terminals.scenario_utils import (
    get_available_scenarios,
    duplicate_scenario,
    get_latest_adjustments,
    save_adjustments_bulk,
    get_scenario_summary,
    delete_scenario
)

###############################################################################
# Database Connection
###############################################################################

# Load config
try:
    config_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    CONFIG_FILE_PATH = os.path.join(config_dir, 'config.ini')
except:
    CONFIG_FILE_PATH = 'config.ini'

config_reader = configparser.ConfigParser(interpolation=None)
config_reader.read(CONFIG_FILE_PATH)

DB_CONNECTION_STRING = config_reader.get('DATABASE', 'CONNECTION_STRING', fallback=None)
DB_SCHEMA = config_reader.get('DATABASE', 'SCHEMA', fallback='at_lng')

if not DB_CONNECTION_STRING:
    raise ValueError(f"Missing DATABASE CONNECTION_STRING in {CONFIG_FILE_PATH}")

engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)

###############################################################################
# Data Fetching Functions
###############################################################################

def fetch_woodmac_baseline():
    """
    Fetch Woodmac baseline data for all plants/trains.
    Returns monthly production data.
    """
    query = f"""
        SELECT
            id_plant,
            id_lng_train,
            plant_name,
            lng_train_name_short,
            country_name,
            year,
            month,
            metric_value as baseline_output
        FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_output_mta
        WHERE metric_value IS NOT NULL
        ORDER BY plant_name, lng_train_name_short, year, month
    """

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df


def fetch_adjustments_with_baseline(scenario_name):
    """
    Fetch Woodmac baseline with adjustments overlaid for a specific scenario.

    Returns DataFrame with both baseline and adjusted values.
    """
    if scenario_name == 'base_view':
        # Base view = Woodmac only, no adjustments
        df = fetch_woodmac_baseline()
        df['adjusted_output'] = None
        df['scenario_name'] = 'base_view'
        df['data_source'] = 'woodmac'
        df['comments'] = None
        return df

    query = f"""
        WITH woodmac_baseline AS (
            SELECT
                id_plant,
                id_lng_train,
                plant_name,
                lng_train_name_short,
                country_name,
                year,
                month,
                metric_value as baseline_output
            FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_output_mta
            WHERE metric_value IS NOT NULL
        ),
        latest_adjustments AS (
            SELECT DISTINCT ON (id_plant, id_lng_train, year, month)
                id_plant,
                id_lng_train,
                year,
                month,
                adjusted_output,
                comments,
                upload_timestamp_utc
            FROM {DB_SCHEMA}.fundamentals_terminals_output_adjustments
            WHERE scenario_name = %(scenario_name)s
            ORDER BY id_plant, id_lng_train, year, month, upload_timestamp_utc DESC
        )
        SELECT
            wb.id_plant,
            wb.id_lng_train,
            wb.plant_name,
            wb.lng_train_name_short,
            wb.country_name,
            wb.year,
            wb.month,
            wb.baseline_output,
            la.adjusted_output,
            COALESCE(la.adjusted_output, wb.baseline_output) as final_output,
            'adjusted' as data_source,
            la.comments,
            %(scenario_name)s as scenario_name
        FROM woodmac_baseline wb
        INNER JOIN latest_adjustments la
            ON wb.id_plant = la.id_plant
            AND wb.id_lng_train = la.id_lng_train
            AND wb.year = la.year
            AND wb.month = la.month
        ORDER BY wb.plant_name, wb.lng_train_name_short, wb.year, wb.month
    """

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={'scenario_name': scenario_name})

    return df


def get_plants_list():
    """Get unique list of plants for filter dropdown."""
    query = f"""
        SELECT DISTINCT plant_name, country_name
        FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_output_mta
        ORDER BY country_name, plant_name
    """

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df


def get_trains_list():
    """Get unique list of trains for filter dropdown."""
    query = f"""
        SELECT DISTINCT
            plant_name,
            lng_train_name_short,
            country_name
        FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_output_mta
        ORDER BY country_name, plant_name, lng_train_name_short
    """

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df


###############################################################################
# Layout
###############################################################################

# Fetch initial data
scenarios = get_available_scenarios(engine)
plants_df = get_plants_list()
trains_df = get_trains_list()

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Terminal Output Adjustments Editor", className="mb-4"),
            html.P([
                "Manage scenario-based adjustments for LNG terminal production forecasts. ",
                "Adjustments override Woodmac baseline data for selected trains and time periods."
            ], className="text-muted mb-4"),
        ])
    ]),

    # Return to terminals dashboard link
    dbc.Row([
        dbc.Col([
            dcc.Link("← Return to Terminals Dashboard", href="/terminals",
                    className="btn btn-link mb-3")
        ])
    ]),

    # Scenario management section
    dbc.Card([
        dbc.CardHeader(html.H5("Scenario Management")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Scenario:", className="fw-bold"),
                    dcc.Dropdown(
                        id='scenario-selector',
                        options=[{'label': s, 'value': s} for s in scenarios],
                        value='best_view',
                        clearable=False,
                        className="mb-2"
                    ),
                    html.Small("base_view = Woodmac only (read-only), other scenarios = with adjustments",
                              className="text-muted")
                ], width=4),

                dbc.Col([
                    html.Label("Scenario Actions:", className="fw-bold"),
                    html.Div([
                        dbc.Button("Duplicate Scenario", id="duplicate-scenario-btn",
                                  color="primary", size="sm", className="me-2"),
                        dbc.Button("Delete Scenario", id="delete-scenario-btn",
                                  color="danger", size="sm", className="me-2"),
                        dbc.Button("Refresh Scenarios", id="refresh-scenarios-btn",
                                  color="secondary", size="sm"),
                    ])
                ], width=5),

                dbc.Col([
                    html.Label("Scenario Info:", className="fw-bold"),
                    html.Div(id="scenario-info", className="text-muted")
                ], width=3),
            ])
        ])
    ], className="mb-4"),

    # Copy trains from base_view section
    dbc.Card([
        dbc.CardHeader(html.H5("Copy Trains from Base View")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Trains to Copy:", className="fw-bold"),
                    dcc.Dropdown(
                        id='trains-to-copy',
                        options=[{'label': f"{row['plant_name']} - {row['lng_train_name_short']}",
                                 'value': f"{row['plant_name']}|{row['lng_train_name_short']}"}
                                for _, row in trains_df.iterrows()],
                        placeholder="Select trains from base_view",
                        multi=True
                    ),
                    html.Small("Select one or more trains to copy their baseline data as adjustments",
                              className="text-muted")
                ], width=6),

                dbc.Col([
                    html.Label("Destination Scenario:", className="fw-bold"),
                    dbc.Input(
                        id='copy-destination-scenario',
                        type='text',
                        placeholder="Type scenario name or select from list",
                        value='best_view',
                        list='scenario-list',
                        className="form-control"
                    ),
                    html.Datalist(
                        id='scenario-list',
                        children=[html.Option(value=s) for s in scenarios if s != 'base_view']
                    ),
                    html.Small("Type new scenario name or choose from existing",
                              className="text-muted")
                ], width=4),

                dbc.Col([
                    html.Label("Action:", className="fw-bold"),
                    html.Div([
                        dbc.Button("Copy Trains", id="copy-trains-btn",
                                  color="primary", size="md", className="w-100"),
                    ])
                ], width=2),
            ]),
            html.Div(id="copy-message", className="alert alert-info d-none mt-3", role="alert"),
        ])
    ], className="mb-4"),

    # Filters section
    dbc.Card([
        dbc.CardHeader(html.H5("Data Filters")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Plant:", className="fw-bold"),
                    dcc.Dropdown(
                        id='plant-filter',
                        options=[{'label': f"{row['plant_name']} ({row['country_name']})",
                                 'value': row['plant_name']}
                                for _, row in plants_df.iterrows()],
                        placeholder="All plants",
                        multi=True
                    )
                ], width=3),

                dbc.Col([
                    html.Label("Train:", className="fw-bold"),
                    dcc.Dropdown(
                        id='train-filter',
                        options=[{'label': f"{row['plant_name']} - {row['lng_train_name_short']}",
                                 'value': f"{row['plant_name']}|{row['lng_train_name_short']}"}
                                for _, row in trains_df.iterrows()],
                        placeholder="All trains",
                        multi=True
                    )
                ], width=3),

                dbc.Col([
                    html.Label("Year Range:", className="fw-bold"),
                    dcc.RangeSlider(
                        id='year-range-slider',
                        min=2020,
                        max=2035,
                        step=1,
                        value=[2024, 2030],
                        marks={year: str(year) for year in range(2020, 2036, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=6),
            ])
        ])
    ], className="mb-4"),

    # Data table section
    dbc.Card([
        dbc.CardHeader([
            html.H5("Adjustments Data", className="d-inline"),
            html.Span(id="row-count-badge", className="badge bg-secondary ms-2")
        ]),
        dbc.CardBody([
            # Action buttons
            html.Div([
                dbc.Button("Add Row", id="add-row-btn", color="success", size="sm", className="me-2"),
                dbc.Button("Save Changes", id="save-changes-btn", color="primary", size="sm", className="me-2"),
                dbc.Button("Download CSV", id="download-csv-btn", color="info", size="sm", className="me-2"),
                dcc.Upload(
                    id='upload-data',
                    children=dbc.Button("Upload CSV/Excel", color="warning", size="sm"),
                    multiple=False
                ),
            ], className="mb-3"),

            # Message area
            html.Div(id="message", className="alert alert-info d-none", role="alert"),

            # Data table
            dash_table.DataTable(
                id='adjustments-table',
                columns=[
                    {"name": "Plant", "id": "plant_name", "editable": False},
                    {"name": "Train", "id": "lng_train_name_short", "editable": False},
                    {"name": "Country", "id": "country_name", "editable": False},
                    {"name": "Year", "id": "year", "editable": True, "type": "numeric"},
                    {"name": "Month", "id": "month", "editable": True, "type": "numeric"},
                    {"name": "Baseline (MTPA)", "id": "baseline_output", "editable": False, "type": "numeric",
                     "format": {"specifier": ".3f"}},
                    {"name": "Adjusted (MTPA)", "id": "adjusted_output", "editable": True, "type": "numeric",
                     "format": {"specifier": ".3f"}},
                    {"name": "Final (MTPA)", "id": "final_output", "editable": False, "type": "numeric",
                     "format": {"specifier": ".3f"}},
                    {"name": "Source", "id": "data_source", "editable": False},
                    {"name": "Comments", "id": "comments", "editable": True},
                ],
                data=[],
                editable=True,
                row_deletable=False,
                row_selectable='multi',
                selected_rows=[],
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=50,
                style_table={"overflowX": "auto"},
                style_cell={
                    "minWidth": "100px",
                    "width": "120px",
                    "maxWidth": "200px",
                    "textAlign": "left",
                    "padding": "8px"
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                        'if': {
                            'filter_query': '{data_source} = "adjusted"',
                            'column_id': 'adjusted_output'
                        },
                        'backgroundColor': '#ffffcc',
                        'fontWeight': 'bold'
                    },
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            ),

            # Download component
            dcc.Download(id="download-dataframe-csv"),
        ])
    ], className="mb-4"),

    # Modals
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Duplicate Scenario")),
        dbc.ModalBody([
            html.Label("New Scenario Name:", className="fw-bold"),
            dbc.Input(id="new-scenario-name", type="text", placeholder="e.g., test_1"),
            html.Small("Current scenario will be copied to new name", className="text-muted mt-2")
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="duplicate-cancel-btn", className="me-2"),
            dbc.Button("Create", id="duplicate-confirm-btn", color="primary"),
        ]),
    ], id="duplicate-modal", is_open=False),

    # Store components
    dcc.Store(id='table-data-store'),
    dcc.Store(id='scenario-list-store', data=scenarios),

], fluid=True, className="p-4")


###############################################################################
# Callbacks
###############################################################################

@callback(
    [Output('adjustments-table', 'data'),
     Output('row-count-badge', 'children'),
     Output('table-data-store', 'data')],
    [Input('scenario-selector', 'value'),
     Input('plant-filter', 'value'),
     Input('train-filter', 'value'),
     Input('year-range-slider', 'value')],
    prevent_initial_call=False
)
def update_table_data(scenario, plants, trains, year_range):
    """Load and filter data based on scenario and filters."""
    if not scenario:
        return [], "0 rows", []

    # Fetch data
    df = fetch_adjustments_with_baseline(scenario)

    if df.empty:
        return [], "0 rows", []

    # Apply filters
    if plants:
        df = df[df['plant_name'].isin(plants)]

    if trains:
        # Parse train filter values (format: "plant_name|train_name")
        train_filters = []
        for train_val in trains:
            plant_name, train_name = train_val.split('|')
            train_filters.append((plant_name, train_name))

        # Filter by matching plant_name AND lng_train_name_short
        mask = pd.Series([False] * len(df))
        for plant_name, train_name in train_filters:
            mask |= (df['plant_name'] == plant_name) & (df['lng_train_name_short'] == train_name)
        df = df[mask]

    if year_range:
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

    # Round numeric columns (only if they are actually numeric)
    numeric_cols = ['baseline_output', 'adjusted_output', 'final_output']
    for col in numeric_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(3)

    data = df.to_dict('records')
    row_count = f"{len(data)} rows"

    return data, row_count, data


@callback(
    Output('duplicate-modal', 'is_open'),
    [Input('duplicate-scenario-btn', 'n_clicks'),
     Input('duplicate-cancel-btn', 'n_clicks'),
     Input('duplicate-confirm-btn', 'n_clicks')],
    [State('duplicate-modal', 'is_open')],
    prevent_initial_call=True
)
def toggle_duplicate_modal(open_click, cancel_click, confirm_click, is_open):
    """Toggle the duplicate scenario modal."""
    return not is_open


@callback(
    [Output('scenario-selector', 'options'),
     Output('scenario-list-store', 'data'),
     Output('message', 'children'),
     Output('message', 'className')],
    [Input('duplicate-confirm-btn', 'n_clicks'),
     Input('delete-scenario-btn', 'n_clicks'),
     Input('refresh-scenarios-btn', 'n_clicks')],
    [State('scenario-selector', 'value'),
     State('new-scenario-name', 'value')],
    prevent_initial_call=True
)
def manage_scenarios(duplicate_click, delete_click, refresh_click, current_scenario, new_name):
    """Handle scenario duplication, deletion, and refresh."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, "", "alert alert-info d-none"

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        if trigger_id == 'duplicate-confirm-btn':
            if not new_name or not new_name.strip():
                return dash.no_update, dash.no_update, "Please enter a scenario name", "alert alert-warning"

            result = duplicate_scenario(current_scenario, new_name.strip(), engine)
            scenarios = get_available_scenarios(engine)
            options = [{'label': s, 'value': s} for s in scenarios]
            message = f"Successfully created '{new_name}' with {result['records_copied']} adjustments from '{current_scenario}'"
            return options, scenarios, message, "alert alert-success"

        elif trigger_id == 'delete-scenario-btn':
            if current_scenario in ['base_view', 'best_view']:
                return dash.no_update, dash.no_update, f"Cannot delete reserved scenario '{current_scenario}'", "alert alert-danger"

            result = delete_scenario(current_scenario, engine)
            scenarios = get_available_scenarios(engine)
            options = [{'label': s, 'value': s} for s in scenarios]
            message = f"Deleted scenario '{current_scenario}' ({result['records_deleted']} records removed)"
            return options, scenarios, message, "alert alert-warning"

        elif trigger_id == 'refresh-scenarios-btn':
            scenarios = get_available_scenarios(engine)
            options = [{'label': s, 'value': s} for s in scenarios]
            return options, scenarios, "Scenarios refreshed", "alert alert-info"

    except Exception as e:
        return dash.no_update, dash.no_update, f"Error: {str(e)}", "alert alert-danger"

    return dash.no_update, dash.no_update, "", "alert alert-info d-none"


@callback(
    Output('scenario-info', 'children'),
    [Input('scenario-selector', 'value')],
    prevent_initial_call=False
)
def update_scenario_info(scenario):
    """Display information about selected scenario."""
    if not scenario:
        return "No scenario selected"

    if scenario == 'base_view':
        return html.Div([
            html.Strong("Base View"),
            html.Br(),
            html.Small("Woodmac baseline only (read-only)")
        ])

    try:
        summary_df = get_scenario_summary(engine)
        row = summary_df[summary_df['scenario_name'] == scenario]

        if row.empty:
            return html.Div([
                html.Strong(scenario),
                html.Br(),
                html.Small("No adjustments yet")
            ])

        row = row.iloc[0]
        return html.Div([
            html.Strong(scenario),
            html.Br(),
            html.Small(f"{row['adjustment_count']} adjustments"),
            html.Br(),
            html.Small(f"Years: {row['earliest_year']}-{row['latest_year']}")
        ])
    except:
        return html.Div([html.Strong(scenario)])


@callback(
    [Output('adjustments-table', 'data', allow_duplicate=True),
     Output('message', 'children', allow_duplicate=True),
     Output('message', 'className', allow_duplicate=True)],
    [Input('add-row-btn', 'n_clicks'),
     Input('save-changes-btn', 'n_clicks'),
     Input('upload-data', 'contents')],
    [State('adjustments-table', 'data'),
     State('scenario-selector', 'value'),
     State('upload-data', 'filename')],
    prevent_initial_call=True
)
def handle_table_actions(add_clicks, save_clicks, upload_contents, current_data, scenario, filename):
    """Handle add row, save, and upload actions."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, "", "alert alert-info d-none"

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        if trigger_id == 'add-row-btn':
            if scenario == 'base_view':
                return dash.no_update, "Cannot add adjustments to base_view", "alert alert-warning"

            new_row = {
                'plant_name': '',
                'lng_train_name_short': '',
                'country_name': '',
                'year': 2025,
                'month': 1,
                'baseline_output': None,
                'adjusted_output': None,
                'final_output': None,
                'data_source': 'adjusted',
                'comments': ''
            }
            updated_data = current_data + [new_row]
            return updated_data, "New row added. Fill in details and save.", "alert alert-info"

        elif trigger_id == 'save-changes-btn':
            if scenario == 'base_view':
                return dash.no_update, "Cannot save adjustments to base_view", "alert alert-warning"

            if not current_data:
                return dash.no_update, "No data to save", "alert alert-warning"

            # Filter and prepare data for saving
            df = pd.DataFrame(current_data)

            # Only save rows that have adjusted_output
            df_to_save = df[df['adjusted_output'].notna()].copy()

            if df_to_save.empty:
                return dash.no_update, "No adjustments to save", "alert alert-warning"

            # Ensure required columns exist
            required_cols = ['id_plant', 'id_lng_train', 'plant_name', 'lng_train_name_short',
                           'year', 'month', 'adjusted_output']

            # Only keep columns that exist in the database table
            # The table has: id_plant, id_lng_train, plant_name, lng_train_name_short,
            #                year, month, adjusted_output, scenario_name, source_name,
            #                comments, upload_timestamp_utc, created_by
            table_columns = ['id_plant', 'id_lng_train', 'plant_name', 'lng_train_name_short',
                           'year', 'month', 'adjusted_output', 'comments']

            # Filter to only columns that exist in both the DataFrame and the table
            cols_to_save = [col for col in table_columns if col in df_to_save.columns]
            df_to_save = df_to_save[cols_to_save].copy()

            # Save to database
            result = save_adjustments_bulk(df_to_save, scenario, engine)

            # Reload data
            df_reloaded = fetch_adjustments_with_baseline(scenario)
            data = df_reloaded.to_dict('records')

            message = f"Saved {result['records_inserted']} adjustments to '{scenario}'"
            return data, message, "alert alert-success"

        elif trigger_id == 'upload-data' and upload_contents:
            if scenario == 'base_view':
                return dash.no_update, "Cannot upload to base_view", "alert alert-warning"

            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)

            if filename.lower().endswith('.csv'):
                df_upload = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df_upload = pd.read_excel(io.BytesIO(decoded))
            else:
                return dash.no_update, "Unsupported file format", "alert alert-danger"

            # Append to current data
            updated_data = current_data + df_upload.to_dict('records')
            message = f"Uploaded {len(df_upload)} rows. Review and save."
            return updated_data, message, "alert alert-info"

    except Exception as e:
        return dash.no_update, f"Error: {str(e)}", "alert alert-danger"

    return dash.no_update, "", "alert alert-info d-none"


@callback(
    Output("download-dataframe-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State('adjustments-table', 'data'),
    State('scenario-selector', 'value'),
    prevent_initial_call=True
)
def download_csv(n_clicks, data, scenario):
    """Download current table data as CSV."""
    if not data:
        return dash.no_update

    df = pd.DataFrame(data)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"terminal_adjustments_{scenario}_{timestamp}.csv"

    return dcc.send_data_frame(df.to_csv, filename, index=False)


@callback(
    [Output('copy-message', 'children'),
     Output('copy-message', 'className')],
    Input('copy-trains-btn', 'n_clicks'),
    [State('trains-to-copy', 'value'),
     State('copy-destination-scenario', 'value')],
    prevent_initial_call=True
)
def copy_trains_to_scenario(n_clicks, trains_to_copy, destination_scenario):
    """Copy selected trains from base_view to a scenario."""
    if not trains_to_copy or not destination_scenario:
        return "Please select trains and destination scenario", "alert alert-warning"

    try:
        # Parse train selections (format: "plant_name|train_name")
        train_filters = []
        for train_val in trains_to_copy:
            plant_name, train_name = train_val.split('|')
            train_filters.append((plant_name, train_name))

        # Fetch baseline data for selected trains
        # Get latest baseline data from Woodmac
        query = f"""
            WITH latest_output AS (
                SELECT DISTINCT ON (id_plant, id_lng_train, year, month)
                    id_plant,
                    id_lng_train,
                    plant_name,
                    lng_train_name_short,
                    year,
                    month,
                    metric_value as baseline_output,
                    upload_timestamp_utc
                FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_output_mta
                WHERE metric_value IS NOT NULL
                ORDER BY id_plant, id_lng_train, year, month, upload_timestamp_utc DESC
            )
            SELECT
                id_plant,
                id_lng_train,
                plant_name,
                lng_train_name_short,
                year,
                month,
                baseline_output
            FROM latest_output
            WHERE 1=1
        """

        # Add train filters to query
        train_conditions = []
        for plant_name, train_name in train_filters:
            train_conditions.append(f"(plant_name = '{plant_name}' AND lng_train_name_short = '{train_name}')")

        if train_conditions:
            query += " AND (" + " OR ".join(train_conditions) + ")"

        query += " ORDER BY plant_name, lng_train_name_short, year, month"

        # Fetch data
        with engine.connect() as conn:
            df_baseline = pd.read_sql(query, conn)

        if df_baseline.empty:
            return f"No baseline data found for selected trains", "alert alert-warning"

        # Prepare data for insertion
        # Only keep columns that exist in the database table
        df_to_insert = df_baseline[['id_plant', 'id_lng_train', 'plant_name',
                                     'lng_train_name_short', 'year', 'month']].copy()
        df_to_insert['adjusted_output'] = df_baseline['baseline_output']
        df_to_insert['comments'] = 'Copied from base_view'

        # Save to destination scenario
        result = save_adjustments_bulk(df_to_insert, destination_scenario, engine)

        message = f"Successfully copied {result['records_inserted']} records for {len(train_filters)} train(s) to '{destination_scenario}'"
        return message, "alert alert-success"

    except Exception as e:
        return f"Error copying trains: {str(e)}", "alert alert-danger"
