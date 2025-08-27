from dash import html, dcc, dash_table, callback, Output, Input, State, Dash, ALL, callback_context
from dash.dash_table.Format import Format, Group, Scheme
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from io import StringIO
from dash.exceptions import PreventUpdate
import configparser
import os
from sqlalchemy import create_engine, text
import calendar

from utils.table_styles import StandardTableStyleManager, TABLE_COLORS

############################################ postgres sql connection ###################################################
#------ code to be able to access config.ini, even having the path in the .virtualenvs is not working without it ------#
try:
    # Get the directory where your script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the directory containing config.ini
    # Adjust the number of '..' as needed to reach the correct directory
    config_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Go up two levels
    CONFIG_FILE_PATH = os.path.join(config_dir, 'config.ini')
except:
    CONFIG_FILE_PATH = 'config.ini'  # Assumes it's in the same directory or the path it is detected


# --- Load Configuration from INI File ---
config_reader = configparser.ConfigParser(interpolation=None)
config_reader.read(CONFIG_FILE_PATH)

# Read values from the ini file sections
DB_CONNECTION_STRING = config_reader.get('DATABASE', 'CONNECTION_STRING', fallback=None)
DB_SCHEMA = config_reader.get('DATABASE', 'SCHEMA', fallback=None)

# create engine
engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)


def setup_database_connection():
    """Setup database connection using existing configuration"""
    return engine, DB_SCHEMA


def fetch_global_supply_data(engine, schema):
    """Fetch daily global LNG supply data for seasonal chart"""
    
    try:
        with engine.connect() as conn:
            # Get daily aggregated data from 2024-01-01 onwards - NO EXCLUSIONS
            query = text(f"""
            WITH latest_data AS (
                SELECT MAX(upload_timestamp_utc) as max_timestamp
                FROM {schema}.kpler_trades
            )
            SELECT 
                kt.start::date as date,
                SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as mcmd
            FROM {schema}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.start >= '2023-11-01'
                AND kt.start <= CURRENT_DATE
            GROUP BY kt.start::date
            ORDER BY kt.start::date
            """)
            
            df = pd.read_sql(query, conn)
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Calculate 30-day rolling average
            df = df.set_index('date').sort_index()
            df['rolling_avg'] = df['mcmd'].rolling(window=30, min_periods=1).mean().round(1)
            
            # Extract year and day of year for seasonal comparison
            df['year'] = df.index.year
            df['day_of_year'] = df.index.dayofyear
            df['month_day'] = df.index.strftime('%b %d')
            
            return df.reset_index()
            
    except Exception as e:
        print(f"Error fetching global supply data: {e}")
        return pd.DataFrame()


def fetch_country_supply_data(engine, schema, country_name):
    """Fetch daily LNG supply data for a specific country"""
    
    try:
        with engine.connect() as conn:
            # Get daily aggregated data for specific country
            query = text(f"""
            WITH latest_data AS (
                SELECT MAX(upload_timestamp_utc) as max_timestamp
                FROM {schema}.kpler_trades
            )
            SELECT 
                kt.start::date as date,
                SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as mcmd
            FROM {schema}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name = :country
                AND kt.start >= '2023-11-01'
                AND kt.start <= CURRENT_DATE
            GROUP BY kt.start::date
            ORDER BY kt.start::date
            """)
            
            df = pd.read_sql(query, conn, params={"country": country_name})
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Calculate 30-day rolling average
            df = df.set_index('date').sort_index()
            df['rolling_avg'] = df['mcmd'].rolling(window=30, min_periods=1).mean().round(1)
            
            # Extract year and day of year for seasonal comparison
            df['year'] = df.index.year
            df['day_of_year'] = df.index.dayofyear
            df['month_day'] = df.index.strftime('%b %d')
            
            return df.reset_index()
            
    except Exception as e:
        print(f"Error fetching {country_name} supply data: {e}")
        return pd.DataFrame()


def fetch_installation_data(engine, schema):
    """Fetch installation data and process for quarters, months, and weeks"""
    
    EXCLUDED_COUNTRIES = [
        "Japan", "Belgium", "Brazil", "Chile", "China", "Equatorial Guinea",
        "France", "Greece", "India", "Netherlands", "Peru", "Singapore Republic",
        "South Korea", "Spain", "Portugal","United States Virgin Islands", "Germany", 
        "Finland", "Jamaica","Lithuania","Sweden","Turkey"
    ]
    
    current_date = datetime.now()
    
    try:
        with engine.connect() as conn:
            # Get all data in one query
            base_query = text(f"""
            WITH latest_data AS (
                SELECT MAX(upload_timestamp_utc) as max_timestamp
                FROM {schema}.kpler_trades
            )
            SELECT 
                kt.origin_country_name,
                kt.installation_origin_name,
                kt.start::date as flow_date,
                EXTRACT(YEAR FROM kt.start) as year,
                EXTRACT(QUARTER FROM kt.start) as quarter,
                EXTRACT(MONTH FROM kt.start) as month,
                EXTRACT(WEEK FROM kt.start) as week,
                kt.cargo_origin_cubic_meters as volume
            FROM {schema}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name NOT IN :excluded
                AND kt.installation_origin_name IS NOT NULL
                AND kt.start IS NOT NULL
                AND kt.start >= '2022-01-01'
                AND kt.start <= CURRENT_DATE
            """)
            
            df = pd.read_sql(base_query, conn, params={"excluded": tuple(EXCLUDED_COUNTRIES)})
    
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Common data preparation - do once for all processing functions
    df['origin_country_name'] = df['origin_country_name'].str.strip()
    df['installation_origin_name'] = df['installation_origin_name'].str.strip()
    df['mcmd'] = df['volume'] * 0.6 / 1000
    df['flow_date'] = pd.to_datetime(df['flow_date'])
    
    # Process data for three separate tables
    quarters_df = process_quarters_data(df, current_date)
    months_df = process_months_data(df, current_date)
    weeks_df = process_weeks_data(df, current_date)
    
    return quarters_df, months_df, weeks_df


def process_quarters_data(df, current_date):
    """Process data for last 8 quarters using vectorized operations"""
    df = df.copy()
    
    # Create quarter period
    df['period'] = df['flow_date'].dt.to_period('Q')
    
    # Filter to last 8 quarters
    current_quarter = pd.Period(current_date, freq='Q')
    start_quarter = current_quarter - 7
    df_filtered = df[(df['period'] >= start_quarter) & (df['period'] <= current_quarter)]
    
    # Create pivot table with all quarters
    pivot = df_filtered.pivot_table(
        index=['origin_country_name', 'installation_origin_name'],
        columns='period',
        values='mcmd',
        aggfunc='sum',
        fill_value=0
    )
    
    # Calculate daily averages - handle incomplete periods correctly
    for col in pivot.columns:
        # For current period, use actual days elapsed
        if col == current_quarter:
            # Days from start of quarter to current date
            days = (current_date.date() - col.start_time.date()).days + 1
        else:
            # For complete quarters, use full quarter days
            days = (col.end_time.date() - col.start_time.date()).days + 1
        pivot[col] = (pivot[col] / days).round(1)
    
    # Rename columns to display format
    pivot.columns = [f"Q{p.quarter}'{str(p.year)[2:]}" for p in pivot.columns]
    
    # Reset index to get back to regular dataframe
    result = pivot.reset_index()
    
    # Ensure we have all quarters even if empty
    all_quarters = [current_quarter - i for i in range(7, -1, -1)]
    expected_cols = [f"Q{q.quarter}'{str(q.year)[2:]}" for q in all_quarters]
    for col in expected_cols:
        if col not in result.columns:
            result[col] = 0
    
    # Reorder columns
    col_order = ['origin_country_name', 'installation_origin_name'] + expected_cols
    result = result[col_order]
    
    # Add country totals
    numeric_cols = expected_cols
    country_totals = result.groupby('origin_country_name')[numeric_cols].sum().round(1).reset_index()
    country_totals['installation_origin_name'] = 'Total'
    
    # Combine and sort
    final_df = pd.concat([result, country_totals], ignore_index=True)
    final_df['sort_key'] = final_df['installation_origin_name'].apply(lambda x: (1, '') if x == 'Total' else (0, x))
    final_df = final_df.sort_values(['origin_country_name', 'sort_key']).drop('sort_key', axis=1).reset_index(drop=True)
    
    # Mark current quarter column for highlighting
    current_quarter_label = f"Q{current_quarter.quarter}'{str(current_quarter.year)[2:]}"
    final_df.attrs['current_period'] = current_quarter_label
    
    return final_df


def process_months_data(df, current_date):
    """Process data for last 12 months using vectorized operations"""
    df = df.copy()
    
    # Create month period
    df['period'] = df['flow_date'].dt.to_period('M')
    
    # Filter to last 12 months
    current_month = pd.Period(current_date, freq='M')
    start_month = current_month - 11
    df_filtered = df[(df['period'] >= start_month) & (df['period'] <= current_month)]
    
    # Create pivot table
    pivot = df_filtered.pivot_table(
        index=['origin_country_name', 'installation_origin_name'],
        columns='period',
        values='mcmd',
        aggfunc='sum',
        fill_value=0
    )
    
    # Calculate daily averages - handle incomplete periods correctly
    for col in pivot.columns:
        # For current period, use actual days elapsed
        if col == current_month:
            # Days from start of month to current date
            days = current_date.day  # Day of month gives us days elapsed
        else:
            # For complete months, use full month days
            days = col.days_in_month
        pivot[col] = (pivot[col] / days).round(1)
    
    # Rename columns to display format
    pivot.columns = [f"{calendar.month_abbr[p.month]}'{str(p.year)[2:]}" for p in pivot.columns]
    
    # Reset index
    result = pivot.reset_index()
    
    # Ensure we have all months even if empty
    all_months = [current_month - i for i in range(11, -1, -1)]
    expected_cols = [f"{calendar.month_abbr[m.month]}'{str(m.year)[2:]}" for m in all_months]
    for col in expected_cols:
        if col not in result.columns:
            result[col] = 0
    
    # Reorder columns
    col_order = ['origin_country_name', 'installation_origin_name'] + expected_cols
    result = result[col_order]
    
    # Add country totals
    numeric_cols = expected_cols
    country_totals = result.groupby('origin_country_name')[numeric_cols].sum().round(1).reset_index()
    country_totals['installation_origin_name'] = 'Total'
    
    # Combine and sort
    final_df = pd.concat([result, country_totals], ignore_index=True)
    final_df['sort_key'] = final_df['installation_origin_name'].apply(lambda x: (1, '') if x == 'Total' else (0, x))
    final_df = final_df.sort_values(['origin_country_name', 'sort_key']).drop('sort_key', axis=1).reset_index(drop=True)
    
    # Mark current month column for highlighting
    current_month_label = f"{calendar.month_abbr[current_month.month]}'{str(current_month.year)[2:]}"
    final_df.attrs['current_period'] = current_month_label
    
    return final_df


def process_weeks_data(df, current_date):
    """Process data for last 12 weeks using vectorized operations"""
    df = df.copy()
    
    # Create week period
    df['period'] = df['flow_date'].dt.to_period('W')
    
    # Filter to last 12 weeks
    current_week = pd.Period(current_date, freq='W')
    start_week = current_week - 11
    df_filtered = df[(df['period'] >= start_week) & (df['period'] <= current_week)]
    
    # Create pivot table
    pivot = df_filtered.pivot_table(
        index=['origin_country_name', 'installation_origin_name'],
        columns='period',
        values='mcmd',
        aggfunc='sum',
        fill_value=0
    )
    
    # Calculate daily averages - handle incomplete periods correctly
    for col in pivot.columns:
        # For current week, use actual days elapsed
        if col == current_week:
            # Days from start of week to current date
            week_start = col.start_time.date()
            days = min(7, (current_date.date() - week_start).days + 1)
        else:
            # For complete weeks, always 7 days
            days = 7
        pivot[col] = (pivot[col] / days).round(1)
    
    # Rename columns to display format
    pivot.columns = [f"W{p.start_time.isocalendar()[1]}'{str(p.year)[2:]}" for p in pivot.columns]
    
    # Reset index
    result = pivot.reset_index()
    
    # Ensure we have all weeks even if empty
    all_weeks = [current_week - i for i in range(11, -1, -1)]
    expected_cols = [f"W{w.start_time.isocalendar()[1]}'{str(w.year)[2:]}" for w in all_weeks]
    for col in expected_cols:
        if col not in result.columns:
            result[col] = 0
    
    # Reorder columns
    col_order = ['origin_country_name', 'installation_origin_name'] + expected_cols
    result = result[col_order]
    
    # Add country totals
    numeric_cols = expected_cols
    country_totals = result.groupby('origin_country_name')[numeric_cols].sum().round(1).reset_index()
    country_totals['installation_origin_name'] = 'Total'
    
    # Combine and sort
    final_df = pd.concat([result, country_totals], ignore_index=True)
    final_df['sort_key'] = final_df['installation_origin_name'].apply(lambda x: (1, '') if x == 'Total' else (0, x))
    final_df = final_df.sort_values(['origin_country_name', 'sort_key']).drop('sort_key', axis=1).reset_index(drop=True)
    
    # Mark current week column for highlighting
    current_week_label = f"W{current_week.start_time.isocalendar()[1]}'{str(current_week.year)[2:]}"
    final_df.attrs['current_period'] = current_week_label
    
    return final_df


def prepare_table_for_display(df, table_type, expanded_countries=None):
    """Prepare data for display in DataTable with expandable rows"""
    if df.empty:
        return pd.DataFrame(), []
    
    expanded_countries = expanded_countries or []
    
    # Filter data based on expanded state
    filtered_rows = []
    country_totals_for_grand = []  # Store country totals for grand total calculation
    
    for country in df['origin_country_name'].unique():
        country_data = df[df['origin_country_name'] == country]
        
        # Always show country total
        total_row = country_data[country_data['installation_origin_name'] == 'Total']
        if not total_row.empty:
            # Store for grand total calculation
            country_totals_for_grand.append(total_row.copy())
            
            # Add expand/collapse indicator to country name
            total_row = total_row.copy()
            if country in expanded_countries:
                total_row.loc[:, 'origin_country_name'] = f"▼ {country}"
            else:
                total_row.loc[:, 'origin_country_name'] = f"▶ {country}"
            filtered_rows.append(total_row)
        
        # Only show installations if country is expanded
        if country in expanded_countries:
            installations = country_data[country_data['installation_origin_name'] != 'Total'].copy()
            if not installations.empty:
                # Indent installation names for visual hierarchy
                installations.loc[:, 'installation_origin_name'] = "    " + installations['installation_origin_name']
                installations.loc[:, 'origin_country_name'] = ""  # Clear country name for installations
                filtered_rows.append(installations)
    
    # Add Grand Total row
    if country_totals_for_grand:
        grand_total_df = pd.concat(country_totals_for_grand, ignore_index=True)
        numeric_cols = [col for col in grand_total_df.columns if col not in ['origin_country_name', 'installation_origin_name']]
        
        grand_total_row = pd.DataFrame([{
            'origin_country_name': 'GRAND TOTAL',
            'installation_origin_name': '',
            **{col: grand_total_df[col].sum() for col in numeric_cols}
        }])
        
        filtered_rows.append(grand_total_row)
    
    # Combine all rows
    if filtered_rows:
        display_df = pd.concat(filtered_rows, ignore_index=True)
    else:
        display_df = pd.DataFrame()
    
    # Rename columns for display
    display_df.columns = [
        'Country' if col == 'origin_country_name' 
        else 'Installation' if col == 'installation_origin_name'
        else col for col in display_df.columns
    ]
    
    # Create column definitions for DataTable
    columns = []
    for col in display_df.columns:
        if col in ['Country', 'Installation']:
            columns.append({
                'name': col,
                'id': col,
                'type': 'text'
            })
        else:
            columns.append({
                'name': col,
                'id': col,
                'type': 'numeric',
                'format': Format(precision=1, scheme=Scheme.fixed)
            })
    
    return display_df, columns


def get_table_conditional_styles():
    """Get conditional styling for tables"""
    styles = []
    
    # Alternating row colors (lowest priority)
    styles.append({
        'if': {'row_index': 'odd'}, 
        'backgroundColor': '#f8f9fa'
    })
    
    # Country total rows styling (medium priority)
    styles.append({
        'if': {'filter_query': '{Installation} = "Total"'},
        'backgroundColor': TABLE_COLORS['bg_lighter'],
        'fontWeight': 'bold',
        'color': TABLE_COLORS['text_primary']
    })
    
    # Grand Total row styling (highest priority - must be last)
    styles.append({
        'if': {'filter_query': '{Country} = "GRAND TOTAL"'},
        'backgroundColor': '#2E86C1',  # McKinsey blue
        'fontWeight': 'bold',
        'color': 'white'
    })
    
    return styles


# Dashboard layout
layout = html.Div([
    # Store components for caching data
    dcc.Store(id='global-supply-data-store', storage_type='local'),
    dcc.Store(id='us-supply-data-store', storage_type='local'),
    dcc.Store(id='australia-supply-data-store', storage_type='local'),
    dcc.Store(id='qatar-supply-data-store', storage_type='local'),
    dcc.Store(id='quarters-data-store', storage_type='local'),
    dcc.Store(id='months-data-store', storage_type='local'),
    dcc.Store(id='weeks-data-store', storage_type='local'),
    
    # Store for expanded state of countries
    dcc.Store(id='quarters-expanded-countries', data=[]),
    dcc.Store(id='months-expanded-countries', data=[]),
    dcc.Store(id='weeks-expanded-countries', data=[]),

    # Supply Charts Row - All 4 charts in one row
    html.Div([
        # Header matching the style of Installation Trends
        html.Div([
            html.H3('LNG Supply - 30-Day Rolling Average', className="section-title-inline"),
        ], className="inline-section-header"),
        html.Div([
            # Global Supply Chart
            html.Div([
                html.H5('Global', style={'textAlign': 'center', 'marginBottom': '10px'}),
                dcc.Loading(
                    id="global-supply-loading",
                    children=[
                        dcc.Graph(id='global-supply-chart', style={'height': '350px'})
                    ],
                    type="default",
                )
            ], style={'width': '25%', 'display': 'inline-block', 'padding': '0 10px'}),
            
            # United States Supply Chart
            html.Div([
                html.H5('United States', style={'textAlign': 'center', 'marginBottom': '10px'}),
                dcc.Loading(
                    id="us-supply-loading",
                    children=[
                        dcc.Graph(id='us-supply-chart', style={'height': '350px'})
                    ],
                    type="default",
                )
            ], style={'width': '25%', 'display': 'inline-block', 'padding': '0 10px'}),
            
            # Australia Supply Chart
            html.Div([
                html.H5('Australia', style={'textAlign': 'center', 'marginBottom': '10px'}),
                dcc.Loading(
                    id="australia-supply-loading",
                    children=[
                        dcc.Graph(id='australia-supply-chart', style={'height': '350px'})
                    ],
                    type="default",
                )
            ], style={'width': '25%', 'display': 'inline-block', 'padding': '0 10px'}),
            
            # Qatar Supply Chart
            html.Div([
                html.H5('Qatar', style={'textAlign': 'center', 'marginBottom': '10px'}),
                dcc.Loading(
                    id="qatar-supply-loading",
                    children=[
                        dcc.Graph(id='qatar-supply-chart', style={'height': '350px'})
                    ],
                    type="default",
                )
            ], style={'width': '25%', 'display': 'inline-block', 'padding': '0 10px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'})
    ], className="main-section-container", style={'marginBottom': '30px'}),

    # Exporters Section
    html.Div([
        # Header
        html.Div([
            html.H3('Exporters', className="section-title-inline"),
        ], className="inline-section-header"),
        
        # Instructions only
        html.Div([
            html.Span("Click on ▶ country row to expand and see installations", 
                     style={'fontSize': '14px', 'color': '#666', 'fontStyle': 'italic'})
        ], style={'marginTop': '15px', 'marginBottom': '20px'}),
        
        # Hidden components to maintain callback compatibility
        html.Div([
            dcc.Dropdown(id='country-filter-dropdown', value=None, style={'display': 'none'}),
            dcc.Checklist(id='show-country-totals-checkbox', value=['show_totals'])
        ], style={'display': 'none'}),
        
        # Loading wrapper for the tables
        dcc.Loading(
            id="installation-trends-loading",
            children=[
                # Quarterly Table
                html.Div([
                    html.Div([
                        html.H4('Quarterly Trends (Last 8 Quarters)', className="subheader-title-inline"),
                    ], className="inline-subheader"),
                    html.Div(id='quarters-table-container')
                ]),
                
                # Monthly Table
                html.Div([
                    html.Div([
                        html.H4('Monthly Trends (Last 12 Months)', className="subheader-title-inline"),
                    ], className="inline-subheader", style={'marginTop': '20px'}),
                    html.Div(id='months-table-container')
                ]),
                
                # Weekly Table
                html.Div([
                    html.Div([
                        html.H4('Weekly Trends (Last 12 Weeks)', className="subheader-title-inline"),
                    ], className="inline-subheader", style={'marginTop': '20px'}),
                    html.Div(id='weeks-table-container')
                ])
            ],
            type="default",
        )
    ], className="main-section-container")
])


# Callbacks
@callback(
    [Output('global-supply-data-store', 'data'),
     Output('us-supply-data-store', 'data'),
     Output('australia-supply-data-store', 'data'),
     Output('qatar-supply-data-store', 'data'),
     Output('quarters-data-store', 'data'),
     Output('months-data-store', 'data'),
     Output('weeks-data-store', 'data')],
    [Input('global-refresh-button', 'n_clicks')],
    prevent_initial_call=False
)
def refresh_all_data(n_clicks):
    """Load all data from database"""
    try:
        # Fetch data
        engine_inst, schema = setup_database_connection()
        
        # Fetch global supply data
        global_supply_df = fetch_global_supply_data(engine_inst, schema)
        
        # Fetch country-specific supply data
        us_supply_df = fetch_country_supply_data(engine_inst, schema, "United States")
        australia_supply_df = fetch_country_supply_data(engine_inst, schema, "Australia")
        qatar_supply_df = fetch_country_supply_data(engine_inst, schema, "Qatar")
        
        # Fetch installation data
        quarters_df, months_df, weeks_df = fetch_installation_data(engine_inst, schema)
        
        return (global_supply_df.to_dict('records') if not global_supply_df.empty else [],
                us_supply_df.to_dict('records') if not us_supply_df.empty else [],
                australia_supply_df.to_dict('records') if not australia_supply_df.empty else [],
                qatar_supply_df.to_dict('records') if not qatar_supply_df.empty else [],
                quarters_df.to_dict('records'), 
                months_df.to_dict('records'), 
                weeks_df.to_dict('records'))
        
    except Exception as e:
        return [], [], [], [], [], [], []


def create_supply_chart(data, title_prefix="", show_legend=True):
    """Create seasonal comparison chart for LNG supply with professional styling"""
    
    if not data:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available. Please refresh to load data.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color='#6b7280')
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=350,
            paper_bgcolor='white'
        )
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for display (only show from 2024-01-01 onwards)
    # The rolling average is already calculated on the full dataset
    df = df[df['date'] >= '2024-01-01']
    
    # Create figure
    fig = go.Figure()
    
    # Get unique years and assign colors - McKinsey blue palette
    years = sorted(df['year'].unique())
    colors = ['#2E86C1', '#1B4F72', '#5DADE2', '#3498DB', '#76D7C4']  # McKinsey blue palette
    
    # Plot each year's data
    for i, year in enumerate(years):
        year_data = df[df['year'] == year].copy()
        
        # For seasonal comparison, use day of year as x-axis
        # Create a dummy date with same year for proper month display
        year_data['plot_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(year_data['day_of_year'] - 1, unit='d')
        
        fig.add_trace(go.Scatter(
            x=year_data['plot_date'],
            y=year_data['rolling_avg'],
            mode='lines',
            name=str(year),
            line=dict(
                color=colors[i % len(colors)],
                width=3 if year == years[-1] else 2  # Highlight current year
            ),
            hovertemplate=f'<b>{year}</b><br>' +
                         '%{text}<br>' +
                         'Supply: %{y:.1f} Mcm/d<br>' +
                         '<extra></extra>',
            text=year_data['month_day']
        ))
    
    # Update layout with professional styling standards
    fig.update_layout(
        # X-Axis Professional Styling
        xaxis=dict(
            title=dict(text='', font=dict(size=13, color='#4A4A4A')),
            tickformat='%b',
            dtick='M1',
            tickangle=0,
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            gridwidth=0.5,
            linecolor='#CCCCCC',
            linewidth=1,
            tickfont=dict(size=11, color='#666666')
        ),
        
        # Y-Axis Professional Styling
        yaxis=dict(
            title=dict(text='mcm/d', font=dict(size=13, color='#4A4A4A')),
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            gridwidth=0.5,
            linecolor='#CCCCCC',
            linewidth=1,
            tickfont=dict(size=11, color='#666666'),
            zeroline=True,
            zerolinecolor='rgba(150, 150, 150, 0.4)',
            zerolinewidth=1
        ),
        
        # Show or hide legend based on parameter
        showlegend=show_legend,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.08,
            xanchor='left',  # Position at left for the first chart
            x=0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            borderwidth=0,
            font=dict(size=12, color='#4A4A4A'),
            itemsizing='constant',
            itemwidth=30
        ) if show_legend else None,
        
        # Professional Background and Margins
        plot_bgcolor='rgba(248, 249, 250, 0.5)',
        paper_bgcolor='white',
        margin=dict(l=50, r=20, t=20, b=60),
        
        # Enhanced Interactivity
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(200, 200, 200, 0.8)',
            font=dict(size=11, color='#2C3E50'),
            align='left'
        ),
        
        # Height
        height=350,
        
        # Smooth Animations
        transition=dict(duration=300, easing='cubic-in-out')
    )
    
    return fig


@callback(
    Output('global-supply-chart', 'figure'),
    [Input('global-supply-data-store', 'data')],
    prevent_initial_call=False
)
def update_global_supply_chart(global_data):
    """Create seasonal comparison chart of global LNG supply"""
    return create_supply_chart(global_data, "Global", show_legend=True)  # Show legend on first chart


@callback(
    Output('us-supply-chart', 'figure'),
    [Input('us-supply-data-store', 'data')],
    prevent_initial_call=False
)
def update_us_supply_chart(us_data):
    """Create seasonal comparison chart of US LNG supply"""
    return create_supply_chart(us_data, "United States", show_legend=False)  # Hide legend


@callback(
    Output('australia-supply-chart', 'figure'),
    [Input('australia-supply-data-store', 'data')],
    prevent_initial_call=False
)
def update_australia_supply_chart(australia_data):
    """Create seasonal comparison chart of Australia LNG supply"""
    return create_supply_chart(australia_data, "Australia", show_legend=False)  # Hide legend


@callback(
    Output('qatar-supply-chart', 'figure'),
    [Input('qatar-supply-data-store', 'data')],
    prevent_initial_call=False
)
def update_qatar_supply_chart(qatar_data):
    """Create seasonal comparison chart of Qatar LNG supply"""
    return create_supply_chart(qatar_data, "Qatar", show_legend=False)  # Hide legend


@callback(
    [Output('quarters-table-container', 'children'),
     Output('months-table-container', 'children'),
     Output('weeks-table-container', 'children')],
    [Input('quarters-data-store', 'data'),
     Input('months-data-store', 'data'),
     Input('weeks-data-store', 'data'),
     Input('country-filter-dropdown', 'value'),
     Input('show-country-totals-checkbox', 'value'),
     Input('quarters-expanded-countries', 'data'),
     Input('months-expanded-countries', 'data'),
     Input('weeks-expanded-countries', 'data')],
    prevent_initial_call=False
)
def update_installation_tables(quarters_data, months_data, weeks_data, selected_countries, show_totals_option,
                              quarters_expanded, months_expanded, weeks_expanded):
    """Update all three installation trends tables with expandable rows"""
    
    # Initialize expanded states if None
    quarters_expanded = quarters_expanded or []
    months_expanded = months_expanded or []
    weeks_expanded = weeks_expanded or []
    
    def create_table(stored_data, table_type, expanded_countries):
        """Helper function to create a table with expandable rows"""
        if not stored_data:
            return html.Div("No data available. Please refresh to load data.", 
                          style={'textAlign': 'center', 'padding': '20px'})
        
        # Convert stored data back to DataFrame
        df = pd.DataFrame(stored_data)
        
        # Determine current period column based on table type
        from datetime import datetime
        current_date = datetime.now()
        if table_type == 'quarters':
            current_period = pd.Period(current_date, freq='Q')
            current_col = f"Q{current_period.quarter}'{str(current_period.year)[2:]}"
        elif table_type == 'months':
            current_period = pd.Period(current_date, freq='M')
            current_col = f"{calendar.month_abbr[current_period.month]}'{str(current_period.year)[2:]}"
        else:  # weeks
            current_period = pd.Period(current_date, freq='W')
            current_col = f"W{current_period.start_time.isocalendar()[1]}'{str(current_period.year)[2:]}"
        
        # No country filter needed anymore
        
        # Prepare data for table display with expansion
        display_df, columns = prepare_table_for_display(df, table_type, expanded_countries)
        
        if display_df.empty:
            return html.Div("No data matches the selected filters.", 
                          style={'textAlign': 'center', 'padding': '20px'})
        
        # Get conditional styles (includes alternating rows, country totals, grand total)
        conditional_styles = get_table_conditional_styles()
        
        # Add style for indented installations (before Grand Total to maintain priority)
        conditional_styles.insert(1, {
            'if': {'filter_query': '{Country} = ""'},
            'backgroundColor': '#f9f9f9',
            'fontSize': '13px'
        })
        
        # Add left alignment for Country and Installation columns
        conditional_styles.append({
            'if': {'column_id': 'Country'},
            'textAlign': 'left'
        })
        conditional_styles.append({
            'if': {'column_id': 'Installation'},
            'textAlign': 'left'
        })
        
        # Add right alignment for all numeric columns (quarterly data)
        # This ensures better readability for numbers per dash_style.md guidelines
        for col in display_df.columns:
            if col not in ['Country', 'Installation']:
                conditional_styles.append({
                    'if': {'column_id': col},
                    'textAlign': 'right',
                    'paddingRight': '12px'  # Add padding for better readability
                })
        
        # Re-add Grand Total style at the end to ensure highest priority
        conditional_styles.append({
            'if': {'filter_query': '{Country} = "GRAND TOTAL"'},
            'backgroundColor': '#2E86C1',  # McKinsey blue
            'fontWeight': 'bold',
            'color': 'white'
        })
        
        # Get base table configuration
        table_config = StandardTableStyleManager.get_base_datatable_config()
        table_config['style_data_conditional'] = conditional_styles
        
        # Create the DataTable with pattern matching ID
        table = dash_table.DataTable(
            id={'type': 'expandable-table', 'index': table_type},
            data=display_df.to_dict('records'),
            columns=columns,
            page_size=100,  # Increased to show expanded rows
            sort_mode='multi',
            fill_width=False,
            **table_config
        )
        
        return table
    
    try:
        quarters_table = create_table(quarters_data, 'quarters', quarters_expanded)
        months_table = create_table(months_data, 'months', months_expanded)
        weeks_table = create_table(weeks_data, 'weeks', weeks_expanded)
        
        return quarters_table, months_table, weeks_table
        
    except Exception as e:
        error_div = html.Div(f"Error updating tables: {str(e)}", 
                           style={'textAlign': 'center', 'padding': '20px', 'color': 'red'})
        return error_div, error_div, error_div


# Callback to handle expanding/collapsing rows
@callback(
    [Output('quarters-expanded-countries', 'data', allow_duplicate=True),
     Output('months-expanded-countries', 'data', allow_duplicate=True),
     Output('weeks-expanded-countries', 'data', allow_duplicate=True)],
    [Input({'type': 'expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'expandable-table', 'index': ALL}, 'data'),
     State('quarters-expanded-countries', 'data'),
     State('months-expanded-countries', 'data'),
     State('weeks-expanded-countries', 'data'),
     State('quarters-data-store', 'data'),
     State('months-data-store', 'data'),
     State('weeks-data-store', 'data')],
    prevent_initial_call=True
)
def handle_row_expansion(active_cells, table_data_list, quarters_expanded, months_expanded, weeks_expanded,
                        quarters_data, months_data, weeks_data):
    """Handle clicking on rows to expand/collapse"""
    
    # Initialize if None
    quarters_expanded = quarters_expanded or []
    months_expanded = months_expanded or []
    weeks_expanded = weeks_expanded or []
    
    # Check which table was clicked
    ctx = callback_context
    if not ctx.triggered:
        return quarters_expanded, months_expanded, weeks_expanded
    
    # Get the triggered input
    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    
    # Parse which table and what was clicked
    if 'expandable-table' in prop_id and '.active_cell' in prop_id:
        try:
            import json
            # Extract the ID part
            id_str = prop_id.split('.active_cell')[0]
            id_dict = json.loads(id_str)
            table_type = id_dict.get('index')
            
            # Get the active cell value
            active_cell = triggered['value']
            if not active_cell:
                return quarters_expanded, months_expanded, weeks_expanded
            
            # Get the corresponding table data
            if table_type == 'quarters' and quarters_data:
                df = pd.DataFrame(quarters_data)
                display_df, _ = prepare_table_for_display(df, 'quarters', quarters_expanded)
                if active_cell['row'] < len(display_df):
                    clicked_row = display_df.iloc[active_cell['row']]
                    country_col = clicked_row.get('Country', '')
                    if country_col and (country_col.startswith('▶') or country_col.startswith('▼')):
                        country = country_col[2:].strip()
                        if country in quarters_expanded:
                            quarters_expanded.remove(country)
                        else:
                            quarters_expanded.append(country)
            
            elif table_type == 'months' and months_data:
                df = pd.DataFrame(months_data)
                display_df, _ = prepare_table_for_display(df, 'months', months_expanded)
                if active_cell['row'] < len(display_df):
                    clicked_row = display_df.iloc[active_cell['row']]
                    country_col = clicked_row.get('Country', '')
                    if country_col and (country_col.startswith('▶') or country_col.startswith('▼')):
                        country = country_col[2:].strip()
                        if country in months_expanded:
                            months_expanded.remove(country)
                        else:
                            months_expanded.append(country)
            
            elif table_type == 'weeks' and weeks_data:
                df = pd.DataFrame(weeks_data)
                display_df, _ = prepare_table_for_display(df, 'weeks', weeks_expanded)
                if active_cell['row'] < len(display_df):
                    clicked_row = display_df.iloc[active_cell['row']]
                    country_col = clicked_row.get('Country', '')
                    if country_col and (country_col.startswith('▶') or country_col.startswith('▼')):
                        country = country_col[2:].strip()
                        if country in weeks_expanded:
                            weeks_expanded.remove(country)
                        else:
                            weeks_expanded.append(country)
                            
        except Exception as e:
            pass
    
    return quarters_expanded, months_expanded, weeks_expanded