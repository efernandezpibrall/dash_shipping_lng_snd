from dash import html, dcc, dash_table, callback, Output, Input, State, Dash, ALL
from dash.dash_table.Format import Format, Group, Scheme
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime as dt
import json
import copy
import plotly.io as pio
from io import StringIO, BytesIO
from dash.exceptions import PreventUpdate
import configparser
import os
from sqlalchemy import create_engine

from fundamentals.kpler_fundamentals import *
from fundamentals.shipping_balance_calculator import global_shipping_balance as calc_global_shipping_balance, kpler_analysis

############################################ postgres sql connection ###################################################
#------ code to be able to access config.ini, even having the path in the .virtualenvs is not working without it ------#
try:
    # Get the directory where your script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the directory containing config.ini
    # Adjust the number of '..' as needed to reach the correct directory
    config_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Go up one level
    CONFIG_FILE_PATH = os.path.join(config_dir, 'config.ini')
except:
    CONFIG_FILE_PATH = 'config.ini'  # Assumes it's in the same directory or the path it is detected


# --- Load Configuration from INI File ---
config_reader = configparser.ConfigParser(interpolation=None)
config_reader.read(CONFIG_FILE_PATH)

# Read values from the ini file sections
DB_CONNECTION_STRING = config_reader.get('DATABASE', 'CONNECTION_STRING', fallback=None)
DB_SCHEMA = config_reader.get('DATABASE', 'SCHEMA', fallback=None)



# --- Essential Variable Checks ---
if not DB_CONNECTION_STRING:
    raise ValueError(f"Missing DATABASE CONNECTION_STRING in {CONFIG_FILE_PATH}")

# create engine
engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)





def prepare_table_data(df, metric, selected_regions=None, selected_year=None, selected_statuses=None,
                       is_intracountry=False):
    """
    Prepare data for tables showing the metric values by region pair/country and vessel type.
    Args:
        df: DataFrame with trade data
        metric: The column to aggregate ('count_trades' or 'sum_ton_miles')
        selected_regions: List of origin shipping regions to filter by (only for non-intracountry)
        selected_year: Year to filter by (defaults to latest)
        selected_statuses: List of status values to filter by
        is_intracountry: Whether this is for intracountry data
    Returns:
        DataFrame formatted for display in a DataTable
    """
    # Filter the data for years 2019 and later
    filtered_df = df[df['year'] >= 2019]

    # Get the latest year if not specified
    if not selected_year or selected_year == "All Years":
        selected_year = filtered_df['year'].max()
    else:
        selected_year = int(selected_year)

    # Filter by year
    filtered_df = filtered_df[filtered_df['year'] == selected_year]

    # Apply filters specific to trade region or intracountry
    if is_intracountry:
        # Check if origin_country_name exists
        if 'origin_country_name' not in filtered_df.columns:
            # Return empty DataFrame if column doesn't exist
            return pd.DataFrame({'Error': ['origin_country_name column not found in data']})
        index_field = 'origin_country_name'
    else:
        # Check if required columns exist
        if 'origin_shipping_region' not in filtered_df.columns or 'destination_shipping_region' not in filtered_df.columns:
            # Return empty DataFrame if columns don't exist
            return pd.DataFrame({'Error': ['Required shipping region columns not found in data']})
        # Create a new column combining origin and destination regions
        filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
            'destination_shipping_region']
        index_field = 'region_pair'

        # Apply region filter if selected (only for non-intracountry)
        if selected_regions and 'All Regions' not in selected_regions:
            filtered_df = filtered_df[filtered_df['origin_shipping_region'].isin(selected_regions)]

    # Apply status filter if selected
    if selected_statuses and 'All Statuses' not in selected_statuses:
        filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]

    # Check if data is empty after filtering
    if filtered_df.empty:
        return pd.DataFrame({index_field: ['No data'], 'Total': [0]})

    # Determine aggregation method based on metric type
    if metric.startswith('median_'):
        agg_method = 'median'
    elif metric.startswith('mean_'):
        agg_method = 'mean'
    elif metric in ['count_trades', 'sum_ton_miles']:
        agg_method = 'sum'
    else:
        # Default to mean for other metrics that might be averages
        agg_method = 'mean'

    # Aggregate the data (vessel_type removed)
    agg_data = filtered_df.groupby([index_field])[metric].agg(agg_method).reset_index()

    # Create simplified table without vessel type pivoting
    pivot_table = agg_data.copy()

    # Rename metric column to Total for consistency
    pivot_table = pivot_table.rename(columns={metric: 'Total'})

    # Format numeric values based on metric type
    if metric == 'count_trades':
        # Integer formatting for count data
        pivot_table['Total'] = pivot_table['Total'].astype(int)
    elif agg_method in ['median', 'mean']:
        # Round to 2 decimal places for median/mean metrics
        pivot_table['Total'] = pivot_table['Total'].round(2)

    # Add a Total row
    total_row = {index_field: 'Total'}
    if agg_method in ['median', 'mean']:
        # For median/mean metrics, calculate overall median/mean across regions
        value = pivot_table[pivot_table[index_field] != 'Total']['Total'].median() if agg_method == 'median' else pivot_table[pivot_table[index_field] != 'Total']['Total'].mean()
        total_row['Total'] = round(value, 2) if not pd.isna(value) else 0
    else:
        # For sum metrics, sum across regions
        total_row['Total'] = int(pivot_table['Total'].sum()) if metric == 'count_trades' else pivot_table['Total'].sum()
    pivot_table = pd.concat([pivot_table, pd.DataFrame([total_row])], ignore_index=True)

    # Sort by Total value in descending order (except Total row)
    non_total_rows = pivot_table[pivot_table[index_field] != 'Total'].copy()
    total_row = pivot_table[pivot_table[index_field] == 'Total'].copy()
    non_total_rows = non_total_rows.sort_values('Total', ascending=False)

    # Recombine sorted rows with total row at bottom
    pivot_table = pd.concat([non_total_rows, total_row], ignore_index=True)

    return pivot_table


def create_stacked_bar_chart(df, metric, title_suffix, selected_statuses=None, is_intracountry=False):
    """
    Create a Plotly visualization showing data by year and shipping regions/countries.
    Args:
        df: DataFrame with trade data
        metric: The column name to sum and visualize
        title_suffix: Text to use in the title describing the metric
        selected_statuses: List of status values to filter by
        is_intracountry: Whether this is for intracountry data
    Returns:
        A Plotly figure object
    """
    # Filter the data for years 2019 and later
    filtered_df = df[df['year'] >= 2019].copy()

    # Apply status filter if selected
    if selected_statuses and 'All Statuses' not in selected_statuses:
        filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]

    # Set up color palette
    distinct_colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff',
        '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
        '#000075', '#a9a9a9', '#008080', '#e6beff', '#9a6324', '#fffac8',
        '#800000', '#aaffc3', '#808000', '#ffd8b1', '#e6194b', '#3cb44b'
    ]

    # Set grouping field based on data type
    if is_intracountry:
        # Check if origin_country_name exists
        if 'origin_country_name' not in filtered_df.columns:
            # Return empty figure if column doesn't exist
            fig = go.Figure()
            fig.add_annotation(
                text="No intracountry data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        group_field = 'origin_country_name'
        chart_title = f'Intracountry {title_suffix} by Year and Origin Country (2019+)'
        legend_title = 'Origin Country'
    else:
        # Check if required columns exist
        if 'origin_shipping_region' not in filtered_df.columns or 'destination_shipping_region' not in filtered_df.columns:
            # Return empty figure if columns don't exist
            fig = go.Figure()
            fig.add_annotation(
                text="Required shipping region data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        # Create a new column combining origin and destination regions
        filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
            'destination_shipping_region']
        group_field = 'region_pair'
        chart_title = f'{title_suffix} by Year and Shipping Regions (2019+)'
        legend_title = 'Shipping Regions (Origin → Destination)'

    # Aggregate the filtered data (vessel_type removed)
    stacked_data = filtered_df.groupby(['year', group_field])[metric].sum().reset_index()

    # Get unique values
    years = sorted(stacked_data['year'].unique())
    group_values = sorted(stacked_data[group_field].unique())

    # Create figure
    fig = go.Figure()

    # Create a dictionary to keep track of the legend items added
    legend_items = set()

    # Calculate position for each year
    bar_width = 0.8
    year_spacing = 0.2

    # Create a dictionary to store the cumulative stack values
    stack_values = {year: 0 for year in years}

    # Create the stacked bars
    for i, group_value in enumerate(group_values):
        color = distinct_colors[i % len(distinct_colors)]

        for y_idx, year in enumerate(years):
            # Filter data for this combination
            data = stacked_data[(stacked_data['year'] == year) &
                                (stacked_data[group_field] == group_value)]

            if not data.empty:
                value = data[metric].values[0]
                x_pos = y_idx * (1 + year_spacing)

                # Add a trace for this segment of the stacked bar
                showlegend = group_value not in legend_items
                if showlegend:
                    legend_items.add(group_value)

                fig.add_trace(go.Bar(
                    x=[x_pos],
                    y=[value],
                    name=group_value,
                    marker_color=color,
                    showlegend=showlegend,
                    base=stack_values[year],
                    width=bar_width,
                    hovertemplate='%{y:,.0f}<extra></extra>'  # Show only y-value in hover
                ))

                # Update the stack value for the next segment
                stack_values[year] += value

    # Create custom x-axis ticks and labels
    x_ticks = []
    x_labels = []

    # Add year labels
    for y_idx, year in enumerate(years):
        x_pos = y_idx * (1 + year_spacing)
        x_ticks.append(x_pos)
        x_labels.append(str(year))

    # Update layout with professional styling from dash_style.md
    fig.update_layout(
        # Professional Title Styling (following dash_style.md standards)
        title=dict(
            text=chart_title,
            font=dict(size=22, color='#2C3E50', family='Segoe UI, -apple-system, BlinkMacSystemFont, sans-serif'),
            x=0.5,  # Centered
            y=0.98,  # Top positioning
            xanchor='center',
            pad=dict(b=20)
        ),

        # X-Axis Professional Styling
        xaxis=dict(
            title=dict(text='Year', font=dict(size=13, color='#4A4A4A')),
            tickvals=x_ticks,
            ticktext=x_labels,
            tickangle=0,
            tickmode='array',  # Use array mode to ensure custom ticks are used
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',  # Subtle grid
            gridwidth=0.5,
            linecolor='#CCCCCC',
            linewidth=1,
            tickfont=dict(size=11, color='#666666')
        ),

        # Y-Axis Professional Styling
        yaxis=dict(
            title=dict(text=title_suffix, font=dict(size=13, color='#4A4A4A')),
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

        barmode='stack',

        # Professional Legend Positioning (horizontal below chart as per dash_style.md)
        legend=dict(
            orientation='h',  # Horizontal layout
            yanchor='top',
            y=-0.15,  # Below chart
            xanchor='center',
            x=0.5,  # Centered
            title=dict(text=legend_title, font=dict(size=12, color='#4A4A4A')),
            bgcolor='rgba(255, 255, 255, 0)',  # Transparent
            bordercolor='rgba(255, 255, 255, 0)',
            borderwidth=0,
            font=dict(size=12, color='#4A4A4A'),
            itemsizing='constant',
            itemwidth=30
        ),

        # Professional Background and Margins (following dash_style.md standards)
        plot_bgcolor='rgba(248, 249, 250, 0.5)',  # Subtle background
        paper_bgcolor='white',
        height=600,  # Standard height as per dash_style.md
        margin=dict(l=70, r=70, t=80, b=120),  # Professional margins with extra bottom for legend

        # Enhanced Interactivity
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(200, 200, 200, 0.8)',
            font=dict(size=11, color='#2C3E50'),
            align='left'
        ),

        # Smooth Animations
        transition=dict(duration=300, easing='cubic-in-out'),
        autosize=True
    )

    return fig


def global_shipping_balance(aggregation_level='monthly', life_expectancy=20, lng_view='demand', utilization_rate=0.85, window_end_date=None):
    """
    Wrapper function that calls the refactored global_shipping_balance from fundamentals.
    Maintains backward compatibility for existing code.
    """
    return calc_global_shipping_balance(
        engine=engine,
        aggregation_level=aggregation_level,
        life_expectancy=life_expectancy,
        lng_view=lng_view,
        utilization_rate=utilization_rate,
        window_end_date=window_end_date
    )


def create_datatable(data, index_field):
    """Create a formatted DataTable from the provided data."""
    columns = []
    for col in data.columns:
        if col == index_field:
            display_name = "Origin Country" if index_field == "origin_country_name" else col
            columns.append({"name": display_name, "id": col})
        else:
            # Format numeric columns with thousand separators
            columns.append({
                "name": str(col),
                "id": str(col),
                "type": "numeric",
                "format": Format(
                    group=Group.yes,
                    scheme=Scheme.fixed,
                    precision=0,
                    group_delimiter=',',
                    decimal_delimiter='.'
                )
            })

    return dash_table.DataTable(
        columns=columns,
        data=data.to_dict('records'),
        style_table={
            'overflowX': 'auto',
            'width': '100%'
        },
        style_cell={
            'textAlign': 'center',
            'padding': '8px',
            'fontSize': '12px',
            'minWidth': '80px',
            'fontFamily': 'Segoe UI, -apple-system, BlinkMacSystemFont, sans-serif'
        },
        style_header={
            'backgroundColor': '#2E86C1',  # McKinsey blue
            'color': 'white',
            'fontWeight': 'bold',
            'fontSize': '13px',
            'padding': '10px',
            'border': '1px solid #1B4F72',
            'whiteSpace': 'pre-wrap',
            'lineHeight': '1.2',
            'height': 'auto'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {'row_index': len(data) - 1},  # Total row
                'fontWeight': 'bold',
                'backgroundColor': '#f0f8ff',  # Light blue for total row
                'borderTop': '2px solid #2E86C1'
            }
        ],
        style_data={
            'border': '1px solid #e3e6f0',
            'color': '#1f2937'
        },
        page_size=30,
        sort_action='native',
        filter_action='native',
        fill_width=False,
        export_format='xlsx',
        export_headers='display',
        export_columns='visible'
    )


# Dashboard layout
layout = html.Div([
    # Store components for caching data (using session storage to avoid stale data issues)
    dcc.Store(id='trades-shipping-data-store', storage_type='session'),
    dcc.Store(id='shipping-balance-data-store', storage_type='session'),
    dcc.Store(id='shipping-balance-supply-data-store', storage_type='session'),
    dcc.Store(id='shipping-balance-regional-data-store', storage_type='session'),
    dcc.Store(id='shipping-balance-supply-regional-data-store', storage_type='session'),
    dcc.Store(id='vessel-type-options-store', storage_type='session'),
    dcc.Store(id='dropdown-options-store', storage_type='session'),
    dcc.Store(id='refresh-timestamp-store', storage_type='session'),
    dcc.Store(id='intracountry-data-store', storage_type='session'),
    dcc.Download(id='download-demand-metrics-excel'),
    dcc.Download(id='download-supply-metrics-excel'),
    dcc.Download(id='download-intracountry-count-excel'),
    dcc.Download(id='download-intracountry-tonmiles-excel'),
    dcc.Download(id='download-fleet-stats-excel'),

    # Global Shipping Balance Overview Section - Sticky Professional Header
    html.Div([
        html.Div([

            # --- Group 1: Title ---
            html.Div([
                html.Div("Overview", className='filter-group-header'),
                html.Div('Global Shipping Balance', style={'fontWeight': '600', 'fontSize': '15px', 'color': '#1e3a5f'}),
            ], className='filter-section filter-section-destination'),

            # --- Group 2: Scenario Settings ---
            html.Div([
                html.Div("Scenario Settings", className='filter-group-header'),
                html.Div([
                    html.Div([
                        html.Label("Aggregation:", className='filter-label'),
                        dcc.Dropdown(
                            id='aggregation-dropdown',
                            options=[
                                {'label': 'Year+Quarter', 'value': 'quarterly'},
                                {'label': 'Year+Month', 'value': 'monthly'},
                                {'label': 'Year+Season', 'value': 'seasonal'},
                                {'label': 'Year', 'value': 'yearly'}
                            ],
                            value='quarterly',
                            clearable=False,
                            className='filter-dropdown',
                            style={'min-width': '160px'}
                        ),
                    ], className='filter-group'),
                    html.Div([
                        html.Label("Scenario Window End:", className='filter-label'),
                        dcc.Input(
                            id='window-end-date-input',
                            type='text',
                            placeholder='YYYY-MM-DD',
                            value=(pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                            className='filter-input',
                            style={'width': '120px', 'height': '36px', 'fontSize': '13px', 'padding': '6px 8px'}
                        ),
                    ], className='filter-group'),
                ], style={'display': 'flex', 'gap': '12px', 'alignItems': 'flex-end'}),
            ], className='filter-section filter-section-analysis'),

            # --- Group 3: Vessel Parameters ---
            html.Div([
                html.Div("Vessel Parameters", className='filter-group-header'),
                html.Div([
                    html.Div([
                        html.Label("Age (years):", className='filter-label'),
                        dcc.Input(
                            id='vessel-age-input',
                            type='number',
                            value=20,
                            min=1,
                            max=50,
                            step=1,
                            className='filter-input',
                            style={'width': '80px', 'height': '36px', 'fontSize': '13px', 'padding': '6px 8px'}
                        ),
                    ], className='filter-group'),
                    html.Div([
                        html.Label("Utilization (%):", className='filter-label'),
                        dcc.Input(
                            id='utilization-rate-input',
                            type='number',
                            value=85,
                            min=0,
                            max=100,
                            step=1,
                            className='filter-input',
                            style={'width': '80px', 'height': '36px', 'fontSize': '13px', 'padding': '6px 8px'}
                        ),
                    ], className='filter-group'),
                    html.Div([
                        dcc.Checklist(
                            id='use-kpler-historical-checkbox',
                            options=[{'label': ' Kpler Historical', 'value': 'kpler'}],
                            value=[],
                            labelStyle={'fontSize': '13px'}
                        ),
                    ], className='filter-group', style={'alignSelf': 'flex-end', 'paddingBottom': '4px'}),
                ], style={'display': 'flex', 'gap': '12px', 'alignItems': 'flex-end'}),
            ], className='filter-section filter-section-origin'),

            # --- Group 4: Status Info ---
            html.Div([
                html.Div("Status", className='filter-group-header'),
                html.Div([
                    html.Div(id='last-refresh-indicator', className='text-tertiary', style={'fontSize': '12px', 'whiteSpace': 'nowrap'}),
                    html.Div(id='window-info-text', className='text-tertiary', style={'fontSize': '12px', 'whiteSpace': 'nowrap'}),
                ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '4px'}),
            ], className='filter-section filter-section-analysis'),

        ], className='filter-bar-grouped')
    ], className='professional-section-header'),

    # Charts Container with Professional Layout
    html.Div([
        html.Div([
            # Left column - Demand View
            html.Div([
                html.Div([
                    html.H4('Demand View', className="subheader-title-inline"),
                    html.Button(
                        'Export to Excel',
                        id='export-demand-metrics-button',
                        n_clicks=0,
                        style={
                            'marginLeft': '20px',
                            'padding': '5px 15px',
                            'backgroundColor': '#28a745',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'fontWeight': 'bold',
                            'fontSize': '12px'
                        }
                    ),
                ], className="inline-subheader", style={'marginBottom': '12px'}),
                dcc.Graph(id='global-shipping-balance', style={'height': '400px'}),
                # Regional Breakdown Section
                html.Div([
                    html.H5('Regional Breakdown', style={'marginTop': '24px', 'marginBottom': '12px', 'color': '#2C3E50'}),
                    html.Div([
                        html.Label("Aggregation:", style={'marginRight': '8px', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='demand-regional-aggregation-dropdown',
                            options=[
                                {'label': 'Year+Quarter', 'value': 'quarterly'},
                                {'label': 'Year+Month', 'value': 'monthly'},
                                {'label': 'Year+Season', 'value': 'seasonal'},
                                {'label': 'Year', 'value': 'yearly'}
                            ],
                            value='quarterly',
                            clearable=False,
                            style={'width': '150px', 'display': 'inline-block', 'marginRight': '20px'}
                        ),
                        html.Label("Year:", style={'marginRight': '8px', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='demand-regional-year-dropdown',
                            clearable=False,
                            style={'width': '120px', 'display': 'inline-block', 'marginRight': '20px'}
                        ),
                        html.Label("Period:", style={'marginRight': '8px', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='demand-regional-period-dropdown',
                            clearable=False,
                            style={'width': '120px', 'display': 'inline-block', 'marginRight': '20px'}
                        ),
                        html.Label("Vessel Type:", style={'marginRight': '8px', 'fontSize': '13px', 'display': 'none'}),
                        dcc.Dropdown(
                            id='demand-regional-vessel-type-dropdown',
                            multi=True,
                            placeholder='All Vessel Types',
                            style={'width': '200px', 'display': 'none'}
                        )
                    ], style={'marginBottom': '12px'}),
                    html.Div(id='demand-regional-table-container', style={'overflow-x': 'auto'}),

                    # Laden Days Chart
                    html.Div([
                        html.H6('Laden Days Trends by Route', style={'marginTop': '20px', 'marginBottom': '12px', 'color': '#2C3E50'}),
                        html.Div([
                            html.Label("Origin Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='demand-regional-laden-origin-filter',
                                multi=True,
                                placeholder='All Origins',
                                style={'width': '250px', 'display': 'inline-block', 'marginRight': '20px'}
                            ),
                            html.Label("Destination Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='demand-regional-laden-dest-filter',
                                multi=True,
                                placeholder='All Destinations',
                                style={'width': '250px', 'display': 'inline-block'}
                            ),
                        ], style={'marginBottom': '12px'}),
                        dcc.Graph(id='demand-regional-laden-chart', style={'height': '450px'})
                    ]),

                    # Ballast Days Chart
                    html.Div([
                        html.H6('Ballast Days Trends by Route', style={'marginTop': '20px', 'marginBottom': '12px', 'color': '#2C3E50'}),
                        html.Div([
                            html.Label("Origin Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='demand-regional-ballast-origin-filter',
                                multi=True,
                                placeholder='All Origins',
                                style={'width': '250px', 'display': 'inline-block', 'marginRight': '20px'}
                            ),
                            html.Label("Destination Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='demand-regional-ballast-dest-filter',
                                multi=True,
                                placeholder='All Destinations',
                                style={'width': '250px', 'display': 'inline-block'}
                            ),
                        ], style={'marginBottom': '12px'}),
                        dcc.Graph(id='demand-regional-ballast-chart', style={'height': '450px'})
                    ])
                ])
            ], style={'flex': '1', 'paddingRight': '12px'}),

            # Right column - Supply View
            html.Div([
                html.Div([
                    html.H4('Supply View', className="subheader-title-inline"),
                    html.Button(
                        'Export to Excel',
                        id='export-supply-metrics-button',
                        n_clicks=0,
                        style={
                            'marginLeft': '20px',
                            'padding': '5px 15px',
                            'backgroundColor': '#28a745',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'fontWeight': 'bold',
                            'fontSize': '12px'
                        }
                    ),
                ], className="inline-subheader", style={'marginBottom': '12px'}),
                dcc.Graph(id='global-shipping-balance-supply', style={'height': '400px'}),
                # Regional Breakdown Section
                html.Div([
                    html.H5('Regional Breakdown', style={'marginTop': '24px', 'marginBottom': '12px', 'color': '#2C3E50'}),
                    html.Div([
                        html.Label("Aggregation:", style={'marginRight': '8px', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='supply-regional-aggregation-dropdown',
                            options=[
                                {'label': 'Year+Quarter', 'value': 'quarterly'},
                                {'label': 'Year+Month', 'value': 'monthly'},
                                {'label': 'Year+Season', 'value': 'seasonal'},
                                {'label': 'Year', 'value': 'yearly'}
                            ],
                            value='quarterly',
                            clearable=False,
                            style={'width': '150px', 'display': 'inline-block', 'marginRight': '20px'}
                        ),
                        html.Label("Year:", style={'marginRight': '8px', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='supply-regional-year-dropdown',
                            clearable=False,
                            style={'width': '120px', 'display': 'inline-block', 'marginRight': '20px'}
                        ),
                        html.Label("Period:", style={'marginRight': '8px', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='supply-regional-period-dropdown',
                            clearable=False,
                            style={'width': '120px', 'display': 'inline-block', 'marginRight': '20px'}
                        ),
                        html.Label("Vessel Type:", style={'marginRight': '8px', 'fontSize': '13px', 'display': 'none'}),
                        dcc.Dropdown(
                            id='supply-regional-vessel-type-dropdown',
                            multi=True,
                            placeholder='All Vessel Types',
                            style={'width': '200px', 'display': 'none'}
                        )
                    ], style={'marginBottom': '12px'}),
                    html.Div(id='supply-regional-table-container', style={'overflow-x': 'auto'}),

                    # Laden Days Chart
                    html.Div([
                        html.H6('Laden Days Trends by Route', style={'marginTop': '20px', 'marginBottom': '12px', 'color': '#2C3E50'}),
                        html.Div([
                            html.Label("Origin Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='supply-regional-laden-origin-filter',
                                multi=True,
                                placeholder='All Origins',
                                style={'width': '250px', 'display': 'inline-block', 'marginRight': '20px'}
                            ),
                            html.Label("Destination Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='supply-regional-laden-dest-filter',
                                multi=True,
                                placeholder='All Destinations',
                                style={'width': '250px', 'display': 'inline-block'}
                            ),
                        ], style={'marginBottom': '12px'}),
                        dcc.Graph(id='supply-regional-laden-chart', style={'height': '450px'})
                    ]),

                    # Ballast Days Chart
                    html.Div([
                        html.H6('Ballast Days Trends by Route', style={'marginTop': '20px', 'marginBottom': '12px', 'color': '#2C3E50'}),
                        html.Div([
                            html.Label("Origin Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='supply-regional-ballast-origin-filter',
                                multi=True,
                                placeholder='All Origins',
                                style={'width': '250px', 'display': 'inline-block', 'marginRight': '20px'}
                            ),
                            html.Label("Destination Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='supply-regional-ballast-dest-filter',
                                multi=True,
                                placeholder='All Destinations',
                                style={'width': '250px', 'display': 'inline-block'}
                            ),
                        ], style={'marginBottom': '12px'}),
                        dcc.Graph(id='supply-regional-ballast-chart', style={'height': '450px'})
                    ])
                ])
            ], style={'flex': '1', 'paddingLeft': '12px'}),
        ], style={'display': 'flex', 'gap': '24px'})
    ], className='section-container'),

    # Fleet Statistics Section
    html.Div([
        html.Div([
            html.H3('Fleet Statistics', className="section-title-inline"),
            html.Button(
                'Export to Excel',
                id='export-fleet-stats-button',
                n_clicks=0,
                style={
                    'marginLeft': '20px',
                    'padding': '5px 15px',
                    'backgroundColor': '#28a745',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                    'fontSize': '12px'
                }
            ),
        ], className="inline-section-header"),
        dcc.Graph(id='fleet-stats-chart', style={'height': '400px', 'marginTop': '16px'})
    ], className='section-container'),


    # Intracountry Trade Analysis Section
    html.Div([
        html.Div([
            html.H2("Intracountry Trade Analysis", className='section-title-inline'),
            html.P("Analysis of domestic shipping patterns by origin country", className='section-subtitle')
        ], className='header-content'),

        html.Div([
            html.Div([
                html.Label("Filter by Year:", className='filter-label'),
                dcc.Dropdown(
                    id='intracountry-year-dropdown',
                    options=[{'label': 'All Years', 'value': 'All Years'}],
                    value=None,
                    clearable=False,
                    className='inline-dropdown'
                )
            ], className='filter-group'),

            html.Div([
                html.Label("Filter by Status:", className='filter-label'),
                dcc.Dropdown(
                    id='intracountry-status-dropdown',
                    options=[{'label': 'All Statuses', 'value': 'All Statuses'}],
                    value=['All Statuses'],
                    multi=True,
                    clearable=False,
                    className='inline-dropdown'
                )
            ], className='filter-group')
        ], className='filter-bar')
    ], className='inline-section-header'),

    # Intracountry Trade Visualizations Section - Enterprise Standard
    html.Div([
        # Chart Container with Professional Layout
        html.Div([
            # Left column - Trade Count
            html.Div([
                html.Div([
                    html.H4('Count of Intracountry Trades', className="subheader-title-inline"),
                    html.Button(
                        'Export to Excel',
                        id='export-intracountry-count-button',
                        n_clicks=0,
                        style={
                            'marginLeft': '20px',
                            'padding': '5px 15px',
                            'backgroundColor': '#28a745',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'fontWeight': 'bold',
                            'fontSize': '12px'
                        }
                    ),
                ], className="inline-subheader"),
                dcc.Graph(id='intracountry-count-visualization', style={'height': '600px'})
            ], style={'flex': '1', 'paddingRight': '12px'}),

            # Right column - Ton Miles
            html.Div([
                html.Div([
                    html.H4('Intracountry Ton Miles', className="subheader-title-inline"),
                    html.Button(
                        'Export to Excel',
                        id='export-intracountry-tonmiles-button',
                        n_clicks=0,
                        style={
                            'marginLeft': '20px',
                            'padding': '5px 15px',
                            'backgroundColor': '#28a745',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'fontWeight': 'bold',
                            'fontSize': '12px'
                        }
                    ),
                ], className="inline-subheader"),
                dcc.Graph(id='intracountry-tonmiles-visualization', style={'height': '600px'})
            ], style={'flex': '1', 'paddingLeft': '12px'})
        ], style={'display': 'flex', 'gap': '24px', 'marginTop': '16px'})
    ], className='section-container'),

])


# Callbacks
# Update the refresh_data callback to include the aggregation dropdown and window date picker
@callback(
    Output('trades-shipping-data-store', 'data'),
    Output('shipping-balance-data-store', 'data'),
    Output('shipping-balance-supply-data-store', 'data'),
    Output('shipping-balance-regional-data-store', 'data'),
    Output('shipping-balance-supply-regional-data-store', 'data'),
    Output('vessel-type-options-store', 'data'),
    Output('dropdown-options-store', 'data'),
    Output('refresh-timestamp-store', 'data'),
    Output('intracountry-data-store', 'data'),
    Output('window-info-text', 'children'),
    Output('window-end-date-input', 'placeholder'),
    Output('window-end-date-input', 'min'),
    Output('window-end-date-input', 'value'),
    Input('global-refresh-button', 'n_clicks'),
    Input('aggregation-dropdown', 'value'),
    Input('vessel-age-input', 'value'),
    Input('utilization-rate-input', 'value'),
    Input('window-end-date-input', 'value'),
    Input('use-kpler-historical-checkbox', 'value'),
    prevent_initial_call=False
)
def refresh_data(n_clicks, aggregation_level='monthly', vessel_age=20, utilization_rate=85, window_end_date=None, use_kpler_checked=None):
    """Fetch and prepare all data needed for the dashboard."""
    # Parse window_end_date if it's provided as a string
    if window_end_date and isinstance(window_end_date, str) and window_end_date.strip():
        try:
            window_end_date = pd.to_datetime(window_end_date)
        except:
            window_end_date = None
    elif not window_end_date or (isinstance(window_end_date, str) and not window_end_date.strip()):
        window_end_date = None

    # Get the date range from Kpler trades for date picker limits
    query_date_range = '''
        SELECT MIN("end") as min_date, MAX("end") as max_date
        FROM at_lng.kpler_trades
        WHERE status = 'Delivered' AND "end" IS NOT NULL
    '''
    date_range = pd.read_sql(query_date_range, engine)
    min_date = date_range['min_date'].iloc[0]
    max_date = date_range['max_date'].iloc[0]

    # Set default window_end_date to yesterday if not provided
    if window_end_date is None:
        yesterday = pd.Timestamp.now() - pd.Timedelta(days=1)
        # Use yesterday or max_date, whichever is earlier
        window_end_date = min(yesterday, pd.to_datetime(max_date)) if max_date else yesterday

    # Create window info text
    if window_end_date:
        window_end_dt = pd.to_datetime(window_end_date)
        window_start_dt = window_end_dt - pd.Timedelta(days=365)
        window_info = f"📊 Using patterns from {window_start_dt.strftime('%Y-%m-%d')} to {window_end_dt.strftime('%Y-%m-%d')} for future projections"
    else:
        window_info = "📊 Using default recent patterns (2024+) for future projections"

    # Convert utilization rate from percentage to decimal
    utilization_rate_decimal = utilization_rate / 100.0

    # Determine if Kpler historical data should be used
    use_kpler_historical = 'kpler' in (use_kpler_checked or [])

    # Fetch shipping balance data with selected parameters including window - with regional data
    df_regional_demand, df_global_shipping_balance = calc_global_shipping_balance(
        engine=engine,
        aggregation_level=aggregation_level,
        life_expectancy=vessel_age,
        lng_view='demand',
        utilization_rate=utilization_rate_decimal,
        window_end_date=window_end_date,
        return_regional=True,
        use_kpler_historical=use_kpler_historical
    )

    # Fetch shipping balance supply data with window - with regional data
    df_regional_supply, df_global_shipping_balance_supply = calc_global_shipping_balance(
        engine=engine,
        aggregation_level=aggregation_level,
        life_expectancy=vessel_age,
        lng_view='supply',
        utilization_rate=utilization_rate_decimal,
        window_end_date=window_end_date,
        return_regional=True,
        use_kpler_historical=use_kpler_historical
    )

    # Rest of the function remains the same
    # Get the maximum historical date to limit kpler_analysis to historical data only
    query_max_hist_date = '''
        SELECT MAX("end") as hist_date_max
        FROM at_lng.kpler_trades
        WHERE status='Delivered'
        AND upload_timestamp_utc = (SELECT MAX(upload_timestamp_utc) FROM at_lng.kpler_trades)
    '''
    hist_date_max_df = pd.read_sql(query_max_hist_date, engine)
    # Return the last day of the month containing the max date
    max_date = pd.to_datetime(hist_date_max_df.hist_date_max.dt.date[0])
    hist_date_max = max_date + pd.offsets.MonthEnd(0)

    # Fetch trade shipping data - limit to historical data only
    df_intracountry_trades, df_trades_shipping_region = kpler_analysis(engine, end_date=hist_date_max, aggregation_level=aggregation_level)

    # Extract unique values for dropdown options
    # Filter for relevant years (2019+)
    df_filtered = df_trades_shipping_region[df_trades_shipping_region['year'] >= 2019]

    # Get origin regions for dropdown
    origin_regions = sorted(df_filtered['origin_shipping_region'].unique())
    region_options = [{'label': 'All Regions', 'value': 'All Regions'}] + [
        {'label': region, 'value': region} for region in origin_regions
    ]

    # Get years for dropdown
    years = sorted(df_filtered['year'].unique())
    latest_year = max(years)
    year_options = [{'label': 'All Years', 'value': 'All Years'}] + [
        {'label': str(year), 'value': str(year)} for year in years
    ]

    # Get status values for dropdowns
    region_statuses = sorted(df_trades_shipping_region['status'].unique())
    intracountry_statuses = sorted(df_intracountry_trades['status'].unique())

    status_options_region = [{'label': 'All Statuses', 'value': 'All Statuses'}] + [
        {'label': status.capitalize(), 'value': status} for status in region_statuses
    ]

    status_options_intracountry = [{'label': 'All Statuses', 'value': 'All Statuses'}] + [
        {'label': status.capitalize(), 'value': status} for status in intracountry_statuses
    ]

    # Create single-select version of status options for metrics dropdown
    status_options_single = [{'label': status['label'], 'value': status['value']}
                             for status in status_options_intracountry]

    # Vessel type options removed as vessel_type column no longer exists
    vessel_type_options = []

    # Store options data
    options_data = {
        'region_options': region_options,
        'year_options': year_options,
        'latest_year': str(latest_year),
        'status_options_region': status_options_region,
        'status_options_intracountry': status_options_intracountry,
        'status_options_single': status_options_single,
        'aggregation_level': aggregation_level,  # Store the current aggregation level
        'vessel_age': vessel_age,  # Store the current vessel age
        'utilization_rate': utilization_rate,  # Store the current utilization rate
        'hist_date_max': hist_date_max.isoformat()  # Store the historical max date for current period marker
    }

    # Convert DataFrames to JSON for storage
    shipping_data = df_trades_shipping_region.to_json(date_format='iso', orient='split')
    shipping_balance = df_global_shipping_balance.to_json(date_format='iso', orient='split')
    shipping_balance_supply = df_global_shipping_balance_supply.to_json(date_format='iso', orient='split')
    shipping_balance_regional = df_regional_demand.to_json(date_format='iso', orient='split')
    shipping_balance_supply_regional = df_regional_supply.to_json(date_format='iso', orient='split')
    intracountry_data = df_intracountry_trades.to_json(date_format='iso', orient='split')

    # Store timestamp
    refresh_timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Format dates for the text input
    placeholder_text = f"YYYY-MM-DD (Max: {max_date.strftime('%Y-%m-%d')})" if max_date else "YYYY-MM-DD"
    min_date_str = min_date.strftime('%Y-%m-%d') if min_date else ''

    # Format window_end_date consistently
    if window_end_date:
        if hasattr(window_end_date, 'strftime'):
            window_end_date_str = window_end_date.strftime('%Y-%m-%d')
        else:
            window_end_date_str = str(window_end_date)[:10]  # Take first 10 chars for YYYY-MM-DD
    else:
        # Default to yesterday if nothing set
        window_end_date_str = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    return (shipping_data, shipping_balance, shipping_balance_supply, shipping_balance_regional,
            shipping_balance_supply_regional, vessel_type_options, options_data, refresh_timestamp,
            intracountry_data, window_info, placeholder_text, min_date_str, window_end_date_str)


@callback(
    Output('last-refresh-indicator', 'children'),
    Input('refresh-timestamp-store', 'data'),
    prevent_initial_call=False
)
def update_refresh_time(timestamp):
    """Update the refresh time indicator."""
    if timestamp is None:
        return "No data loaded yet. Click 'Refresh data' to load data."
    return f"Last refreshed: {timestamp}"


# Update the update_visualizations callback to handle aggregation levels in chart formatting
@callback(
    Output('global-shipping-balance', 'figure'),
    Output('global-shipping-balance-supply', 'figure'),
    Input('trades-shipping-data-store', 'data'),
    Input('shipping-balance-data-store', 'data'),
    Input('shipping-balance-supply-data-store', 'data'),
    Input('dropdown-options-store', 'data'),
)
def update_visualizations(shipping_data, shipping_balance, shipping_balance_supply, dropdown_options):
    """Update visualizations and tables based on selected filters."""
    # Check if data is available
    if shipping_data is None or shipping_balance is None or shipping_balance_supply is None or dropdown_options is None:
        raise PreventUpdate

    # Convert stored JSON back to DataFrames
    df_trades_shipping_region = pd.read_json(StringIO(shipping_data), orient='split')
    df_global_shipping_balance = pd.read_json(StringIO(shipping_balance), orient='split')
    df_global_shipping_balance_supply = pd.read_json(StringIO(shipping_balance_supply), orient='split')

    # Ensure date columns are datetime type (JSON conversion loses dtype)
    if 'date' in df_global_shipping_balance.columns:
        df_global_shipping_balance['date'] = pd.to_datetime(df_global_shipping_balance['date'])
    if 'date' in df_global_shipping_balance_supply.columns:
        df_global_shipping_balance_supply['date'] = pd.to_datetime(df_global_shipping_balance_supply['date'])

    # Get the aggregation level if available
    aggregation_level = dropdown_options.get('aggregation_level', 'monthly')



    # Create global shipping balance chart with professional formatting
    fig_global_shipping = make_subplots(specs=[[{"secondary_y": True}]])

    # Add main traces with professional colors
    fig_global_shipping.add_trace(
        go.Scatter(
            x=df_global_shipping_balance['date'],
            y=df_global_shipping_balance['total_active_ships'],
            name='Total Active Ships',
            mode='lines+markers',
            line=dict(color='#2E86C1', width=2),  # McKinsey blue
            marker=dict(size=6, color='#2E86C1'),
            opacity=0.9,
        ),
        secondary_y=False,
    )

    fig_global_shipping.add_trace(
        go.Scatter(
            x=df_global_shipping_balance['date'],
            y=df_global_shipping_balance['ships_demand'],
            name='Ships Demand Total',
            mode='lines+markers',
            line=dict(color='#22c55e', width=2),  # Professional green
            marker=dict(size=6, color='#22c55e'),
            opacity=0.9,
        ),
        secondary_y=False,
    )

    fig_global_shipping.add_trace(
        go.Bar(
            x=df_global_shipping_balance['date'],
            y=df_global_shipping_balance['net'],
            name='Net New',
            marker_color='rgba(239, 68, 68, 0.5)',  # Professional red with transparency
            marker_line_color='rgba(239, 68, 68, 0.8)',
            marker_line_width=1,
        ),
        secondary_y=True,
    )

    # Set the chart title to include the aggregation level
    aggregation_title = {
        'monthly': 'Monthly',
        'quarterly': 'Quarterly',
        'seasonal': 'Seasonal',
        'yearly': 'Yearly'
    }.get(aggregation_level, 'Monthly')

    # Professional chart layout following dash_style.md standards
    fig_global_shipping.update_layout(
        # No title - using external Demand View label
        title=None,

        # Professional Legend Positioning
        legend=dict(
            orientation='h',  # Horizontal layout
            yanchor='top',
            y=-0.08,  # Below chart
            xanchor='center',
            x=0.5,  # Centered
            bgcolor='rgba(255, 255, 255, 0)',  # Transparent
            bordercolor='rgba(255, 255, 255, 0)',
            borderwidth=0,
            font=dict(size=12, color='#4A4A4A'),
            itemsizing='constant',
            itemwidth=30
        ),

        # Professional Background and Margins
        plot_bgcolor='rgba(248, 249, 250, 0.5)',  # Subtle background
        paper_bgcolor='white',
        margin=dict(l=70, r=70, t=70, b=90),  # Reduced vertical spacing

        # Enhanced Interactivity
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(200, 200, 200, 0.8)',
            font=dict(size=11, color='#2C3E50'),
            align='left'
        ),

        # Bar gap for better visualization
        bargap=0.2,

        # Height to fit container
        height=400
    )

    # Professional Y-Axis Styling - Primary
    fig_global_shipping.update_yaxes(
        title=dict(text='Number of Ships', font=dict(size=13, color='#4A4A4A')),
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)',
        gridwidth=0.5,
        linecolor='#CCCCCC',
        linewidth=1,
        tickfont=dict(size=11, color='#666666'),
        zeroline=True,
        zerolinecolor='rgba(150, 150, 150, 0.4)',
        zerolinewidth=1,
        secondary_y=False
    )

    # Professional Y-Axis Styling - Secondary
    fig_global_shipping.update_yaxes(
        title=dict(text='Net New Ships', font=dict(size=13, color='#4A4A4A')),
        showgrid=False,  # No grid for secondary axis
        linecolor='#CCCCCC',
        linewidth=1,
        tickfont=dict(size=11, color='#666666'),
        secondary_y=True
    )

    # Set appropriate x-axis formatting based on aggregation level
    if aggregation_level == 'monthly':
        tick_format = '%b %Y'
        dtick = "M3"  # Every 3 months
    elif aggregation_level == 'quarterly':
        tick_format = '%Y Q%q'
        dtick = "M3"  # Every 3 months
    elif aggregation_level == 'seasonal':
        tick_format = '%Y %b'  # Show month abbreviation (Jan/Jul)
        dtick = "M6"  # Every 6 months
    else:  # yearly
        tick_format = '%Y'
        dtick = "M12"  # Every 12 months

    # Professional X-Axis Styling
    fig_global_shipping.update_xaxes(
        title=None,  # Remove x-axis title
        tickformat=tick_format,
        tickangle=0,  # Angled for better readability with dates
        dtick=dtick,
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)',
        gridwidth=0.5,
        linecolor='#CCCCCC',
        linewidth=1,
        tickfont=dict(size=11, color='#666666'),
        tickmode='auto',  # Changed from 'linear' to 'auto' to prevent tick overflow
        nticks=15  # Limit the number of ticks to prevent overcrowding
    )

    # Customize hover template based on aggregation level
    hover_templates = {
        'monthly': '%{x|%b %Y}<br><b>%{y:,.0f}</b><extra></extra>',
        'quarterly': '%{x|%Y Q%q}<br><b>%{y:,.0f}</b><extra></extra>',
        'seasonal': '%{x|%Y %b}<br><b>%{y:,.0f}</b><extra></extra>',
        'yearly': '%{x|%Y}<br><b>%{y:,.0f}</b><extra></extra>'
    }

    for trace in fig_global_shipping.data:
        trace.hovertemplate = hover_templates.get(aggregation_level, '%{x}<br><b>%{y:,.0f}</b><extra></extra>')

    # Add vertical line for current period using historical max from Kpler
    hist_date_max_str = dropdown_options.get('hist_date_max')
    if hist_date_max_str:
        hist_date_max = pd.to_datetime(hist_date_max_str)
        fig_global_shipping.add_shape(
            type="line",
            x0=hist_date_max,
            x1=hist_date_max,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig_global_shipping.add_annotation(
            x=hist_date_max,
            y=1,
            yref="paper",
            text="Current Period",
            showarrow=False,
            yanchor="bottom",
            font=dict(color="red", size=10)
        )

    # Create global shipping balance supply chart with professional formatting
    fig_global_shipping_supply = make_subplots(specs=[[{"secondary_y": True}]])

    # Add main traces for supply view with professional colors
    fig_global_shipping_supply.add_trace(
        go.Scatter(
            x=df_global_shipping_balance_supply['date'],
            y=df_global_shipping_balance_supply['total_active_ships'],
            name='Total Active Ships',
            mode='lines+markers',
            line=dict(color='#2E86C1', width=2),  # McKinsey blue
            marker=dict(size=6, color='#2E86C1'),
            opacity=0.9,
        ),
        secondary_y=False,
    )

    fig_global_shipping_supply.add_trace(
        go.Scatter(
            x=df_global_shipping_balance_supply['date'],
            y=df_global_shipping_balance_supply['ships_demand'],
            name='Ships Supply Total',
            mode='lines+markers',
            line=dict(color='#F7DC6F', width=2),  # Professional yellow/orange
            marker=dict(size=6, color='#F7DC6F'),
            opacity=0.9,
        ),
        secondary_y=False,
    )

    fig_global_shipping_supply.add_trace(
        go.Bar(
            x=df_global_shipping_balance_supply['date'],
            y=df_global_shipping_balance_supply['net'],
            name='Net New',
            marker_color='rgba(247, 220, 111, 0.5)',  # Professional yellow with transparency
            marker_line_color='rgba(247, 220, 111, 0.8)',
            marker_line_width=1,
        ),
        secondary_y=True,
    )

    # Professional chart layout following dash_style.md standards
    fig_global_shipping_supply.update_layout(
        # No title - using external Supply View label
        title=None,

        # Professional Legend Positioning
        legend=dict(
            orientation='h',  # Horizontal layout
            yanchor='top',
            y=-0.08,  # Below chart
            xanchor='center',
            x=0.5,  # Centered
            bgcolor='rgba(255, 255, 255, 0)',  # Transparent
            bordercolor='rgba(255, 255, 255, 0)',
            borderwidth=0,
            font=dict(size=12, color='#4A4A4A'),
            itemsizing='constant',
            itemwidth=30
        ),

        # Professional Background and Margins
        plot_bgcolor='rgba(248, 249, 250, 0.5)',  # Subtle background
        paper_bgcolor='white',
        margin=dict(l=70, r=70, t=70, b=90),  # Reduced vertical spacing

        # Enhanced Interactivity
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(200, 200, 200, 0.8)',
            font=dict(size=11, color='#2C3E50'),
            align='left'
        ),

        # Bar gap for better visualization
        bargap=0.2,

        # Height to fit container
        height=400
    )

    # Professional Y-Axis Styling - Primary
    fig_global_shipping_supply.update_yaxes(
        title=dict(text='Number of Ships', font=dict(size=13, color='#4A4A4A')),
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)',
        gridwidth=0.5,
        linecolor='#CCCCCC',
        linewidth=1,
        tickfont=dict(size=11, color='#666666'),
        zeroline=True,
        zerolinecolor='rgba(150, 150, 150, 0.4)',
        zerolinewidth=1,
        secondary_y=False
    )

    # Professional Y-Axis Styling - Secondary
    fig_global_shipping_supply.update_yaxes(
        title=dict(text='Net New Ships', font=dict(size=13, color='#4A4A4A')),
        showgrid=False,  # No grid for secondary axis
        linecolor='#CCCCCC',
        linewidth=1,
        tickfont=dict(size=11, color='#666666'),
        secondary_y=True
    )

    # Professional X-Axis Styling
    fig_global_shipping_supply.update_xaxes(
        title=None,  # Remove x-axis title
        tickformat=tick_format,
        tickangle=0,  # Angled for better readability with dates
        dtick=dtick,
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)',
        gridwidth=0.5,
        linecolor='#CCCCCC',
        linewidth=1,
        tickfont=dict(size=11, color='#666666'),
        tickmode='auto',  # Changed from 'linear' to 'auto' to prevent tick overflow
        nticks=15  # Limit the number of ticks to prevent overcrowding
    )

    for trace in fig_global_shipping_supply.data:
        trace.hovertemplate = hover_templates.get(aggregation_level, '%{x}<br><b>%{y:,.0f}</b><extra></extra>')

    # Add vertical line for current period using historical max from Kpler
    if hist_date_max_str:
        hist_date_max = pd.to_datetime(hist_date_max_str)
        fig_global_shipping_supply.add_shape(
            type="line",
            x0=hist_date_max,
            x1=hist_date_max,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig_global_shipping_supply.add_annotation(
            x=hist_date_max,
            y=1,
            yref="paper",
            text="Current Period",
            showarrow=False,
            yanchor="bottom",
            font=dict(color="red", size=10)
        )

    return (
        fig_global_shipping,
        fig_global_shipping_supply,
    )


@callback(
    Output('download-demand-metrics-excel', 'data'),
    Input('export-demand-metrics-button', 'n_clicks'),
    State('shipping-balance-data-store', 'data'),
    State('dropdown-options-store', 'data'),
    prevent_initial_call=True
)
def export_demand_metrics_to_excel(n_clicks, shipping_balance, dropdown_options):
    """Export Demand View metrics to Excel."""
    if not n_clicks or shipping_balance is None or dropdown_options is None:
        raise PreventUpdate

    df = pd.read_json(StringIO(shipping_balance), orient='split')
    df['date'] = pd.to_datetime(df['date'])

    aggregation_level = dropdown_options.get('aggregation_level', 'monthly')

    table_data = df[['date', 'total_active_ships', 'ships_demand', 'net', 'utilization_ratio', 'value']].copy()
    table_data = table_data.rename(columns={
        'date': 'Date',
        'total_active_ships': 'Total Active Ships',
        'ships_demand': 'Ships Demand Total',
        'net': 'Net Balance',
        'utilization_ratio': 'Utilization (%)',
        'value': 'Volume (M m³)'
    })
    table_data['Date'] = pd.to_datetime(table_data['Date'])
    table_data['Volume (M m³)'] = (table_data['Volume (M m³)'] / 1000000).round(2)
    if aggregation_level == 'quarterly':
        table_data['Date'] = table_data['Date'].dt.to_period('Q').astype(str)
    elif aggregation_level == 'seasonal':
        table_data['Date'] = table_data['Date'].apply(lambda x: f"{x.year}-{'W' if x.month == 1 else 'S'}")
    elif aggregation_level == 'yearly':
        table_data['Date'] = table_data['Date'].dt.year
    else:
        table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m')
    for col in ['Total Active Ships', 'Ships Demand Total', 'Net Balance']:
        table_data[col] = table_data[col].round(1)
    table_data['Utilization (%)'] = table_data['Utilization (%)'].round(2)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        table_data.to_excel(writer, sheet_name='Demand Metrics', index=False)
        for worksheet in writer.sheets.values():
            for column_cells in worksheet.columns:
                max_length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 50)
    output.seek(0)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(output.getvalue(), f'Demand_Metrics_{aggregation_level}_{timestamp}.xlsx')


@callback(
    Output('download-supply-metrics-excel', 'data'),
    Input('export-supply-metrics-button', 'n_clicks'),
    State('shipping-balance-supply-data-store', 'data'),
    State('dropdown-options-store', 'data'),
    prevent_initial_call=True
)
def export_supply_metrics_to_excel(n_clicks, shipping_balance_supply, dropdown_options):
    """Export Supply View metrics to Excel."""
    if not n_clicks or shipping_balance_supply is None or dropdown_options is None:
        raise PreventUpdate

    df = pd.read_json(StringIO(shipping_balance_supply), orient='split')
    df['date'] = pd.to_datetime(df['date'])

    aggregation_level = dropdown_options.get('aggregation_level', 'monthly')

    table_data = df[['date', 'total_active_ships', 'ships_demand', 'net', 'utilization_ratio', 'value']].copy()
    table_data = table_data.rename(columns={
        'date': 'Date',
        'total_active_ships': 'Total Active Ships',
        'ships_demand': 'Ships Supply Total',
        'net': 'Net Balance',
        'utilization_ratio': 'Utilization (%)',
        'value': 'Volume (M m³)'
    })
    table_data['Date'] = pd.to_datetime(table_data['Date'])
    table_data['Volume (M m³)'] = (table_data['Volume (M m³)'] / 1000000).round(2)
    if aggregation_level == 'quarterly':
        table_data['Date'] = table_data['Date'].dt.to_period('Q').astype(str)
    elif aggregation_level == 'seasonal':
        table_data['Date'] = table_data['Date'].apply(lambda x: f"{x.year}-{'W' if x.month == 1 else 'S'}")
    elif aggregation_level == 'yearly':
        table_data['Date'] = table_data['Date'].dt.year
    else:
        table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m')
    for col in ['Total Active Ships', 'Ships Supply Total', 'Net Balance']:
        table_data[col] = table_data[col].round(1)
    table_data['Utilization (%)'] = table_data['Utilization (%)'].round(2)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        table_data.to_excel(writer, sheet_name='Supply Metrics', index=False)
        for worksheet in writer.sheets.values():
            for column_cells in worksheet.columns:
                max_length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 50)
    output.seek(0)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(output.getvalue(), f'Supply_Metrics_{aggregation_level}_{timestamp}.xlsx')


# Fleet Statistics Export Callback
@callback(
    Output('download-fleet-stats-excel', 'data'),
    Input('export-fleet-stats-button', 'n_clicks'),
    State('shipping-balance-data-store', 'data'),
    State('dropdown-options-store', 'data'),
    prevent_initial_call=True
)
def export_fleet_stats_to_excel(n_clicks, shipping_balance, dropdown_options):
    """Export Fleet Statistics data to Excel."""
    if not n_clicks or shipping_balance is None or dropdown_options is None:
        raise PreventUpdate

    df_global = pd.read_json(StringIO(shipping_balance), orient='split')
    df_global['date'] = pd.to_datetime(df_global['date'])

    aggregation_level = dropdown_options.get('aggregation_level', 'monthly')

    fleet_cols = ['date', 'total_active_ships', 'average_size_cubic_meters', 'ships_added', 'ships_removed']
    display_df = df_global[fleet_cols].copy()
    display_df['net_change'] = display_df['ships_added'] - display_df['ships_removed']

    if aggregation_level == 'monthly':
        display_df['date_formatted'] = display_df['date'].dt.strftime('%Y-%m')
    elif aggregation_level == 'quarterly':
        display_df['date_formatted'] = display_df['date'].dt.year.astype(str) + '-Q' + display_df['date'].dt.quarter.astype(str)
    elif aggregation_level == 'seasonal':
        display_df['date_formatted'] = display_df['date'].dt.year.astype(str) + '-' + display_df['date'].dt.month.map({1: 'Winter', 7: 'Summer'})
    elif aggregation_level == 'yearly':
        display_df['date_formatted'] = display_df['date'].dt.year.astype(str)
    else:
        display_df['date_formatted'] = display_df['date'].dt.strftime('%Y-%m')

    display_df = display_df.rename(columns={
        'date_formatted': 'Period',
        'total_active_ships': 'Total Active Ships',
        'average_size_cubic_meters': 'Avg Capacity (m³)',
        'ships_added': 'Ships Added',
        'ships_removed': 'Ships Removed',
        'net_change': 'Net Change'
    })
    display_df = display_df[['Period', 'Total Active Ships', 'Avg Capacity (m³)', 'Ships Added', 'Ships Removed', 'Net Change']]
    display_df = display_df.sort_index(ascending=False)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        display_df.to_excel(writer, sheet_name='Fleet Statistics', index=False)
        for worksheet in writer.sheets.values():
            for column_cells in worksheet.columns:
                max_length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 50)
    output.seek(0)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(output.getvalue(), f'Fleet_Statistics_{aggregation_level}_{timestamp}.xlsx')


# Fleet Statistics Chart Callback
@callback(
    Output('fleet-stats-chart', 'figure'),
    Input('shipping-balance-data-store', 'data'),
    Input('dropdown-options-store', 'data'),
    prevent_initial_call=False
)
def update_fleet_statistics_chart(shipping_balance, dropdown_options):
    """Update fleet statistics chart with dual axes."""
    if shipping_balance is None or dropdown_options is None:
        return {}

    try:
        # Convert stored JSON back to DataFrame
        df_global = pd.read_json(StringIO(shipping_balance), orient='split')

        # Ensure date column is datetime
        if 'date' in df_global.columns:
            df_global['date'] = pd.to_datetime(df_global['date'])

        # Get aggregation level
        aggregation_level = dropdown_options.get('aggregation_level', 'monthly')

        # Select relevant columns
        fleet_cols = ['date', 'total_active_ships', 'average_size_cubic_meters']
        display_df = df_global[fleet_cols].copy()

        # Sort by date ascending for chart
        display_df = display_df.sort_values('date')

        # Create dual-axis figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bar trace for Total Active Ships (primary y-axis)
        fig.add_trace(
            go.Bar(
                x=display_df['date'],
                y=display_df['total_active_ships'],
                name='Total Active Ships',
                marker_color='#2E86C1',
                yaxis='y',
                hovertemplate='%{x}<br>Total Ships: %{y:,.0f}<extra></extra>'
            ),
            secondary_y=False
        )

        # Add line trace for Average Capacity (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=display_df['date'],
                y=display_df['average_size_cubic_meters'],
                name='Avg Capacity (m³)',
                mode='lines+markers',
                line=dict(color='#E67E22', width=2),
                marker=dict(size=6),
                yaxis='y2',
                hovertemplate='%{x}<br>Avg Capacity: %{y:,.0f} m³<extra></extra>'
            ),
            secondary_y=True
        )

        # Update layout
        fig.update_layout(
            title=None,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.08,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)',
                borderwidth=0,
                font=dict(size=12, color='#4A4A4A'),
                itemsizing='constant',
                itemwidth=30
            ),
            plot_bgcolor='rgba(248, 249, 250, 0.5)',
            paper_bgcolor='white',
            margin=dict(l=70, r=70, t=70, b=90),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='rgba(200, 200, 200, 0.8)',
                font=dict(size=11, color='#2C3E50'),
                align='left'
            ),
            bargap=0.2,
            height=400
        )

        # Primary y-axis
        fig.update_yaxes(
            title=dict(text='Total Active Ships', font=dict(size=13, color='#4A4A4A')),
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            gridwidth=0.5,
            linecolor='#CCCCCC',
            linewidth=1,
            tickfont=dict(size=11, color='#666666'),
            zeroline=True,
            zerolinecolor='rgba(150, 150, 150, 0.4)',
            zerolinewidth=1,
            secondary_y=False
        )

        # Secondary y-axis
        fig.update_yaxes(
            title=dict(text='Average Capacity (m³)', font=dict(size=13, color='#4A4A4A')),
            showgrid=False,
            linecolor='#CCCCCC',
            linewidth=1,
            tickfont=dict(size=11, color='#666666'),
            secondary_y=True
        )

        # Aggregation-aware x-axis formatting
        if aggregation_level == 'monthly':
            tick_format = '%b %Y'
            dtick = "M3"
        elif aggregation_level == 'quarterly':
            tick_format = '%Y Q%q'
            dtick = "M3"
        elif aggregation_level == 'seasonal':
            tick_format = '%Y %b'
            dtick = "M6"
        else:  # yearly
            tick_format = '%Y'
            dtick = "M12"

        fig.update_xaxes(
            title=None,
            tickformat=tick_format,
            tickangle=0,
            dtick=dtick,
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            gridwidth=0.5,
            linecolor='#CCCCCC',
            linewidth=1,
            tickfont=dict(size=11, color='#666666'),
            tickmode='auto',
            nticks=15
        )

        # Aggregation-aware hover templates
        hover_templates = {
            'monthly': '%{x|%b %Y}<br><b>%{y:,.0f}</b><extra></extra>',
            'quarterly': '%{x|%Y Q%q}<br><b>%{y:,.0f}</b><extra></extra>',
            'seasonal': '%{x|%Y %b}<br><b>%{y:,.0f}</b><extra></extra>',
            'yearly': '%{x|%Y}<br><b>%{y:,.0f}</b><extra></extra>'
        }
        for trace in fig.data:
            trace.hovertemplate = hover_templates.get(aggregation_level, '%{x}<br><b>%{y:,.0f}</b><extra></extra>')

        # Add vertical line for current period using historical max from Kpler
        hist_date_max_str = dropdown_options.get('hist_date_max')
        if hist_date_max_str:
            hist_date_max = pd.to_datetime(hist_date_max_str)
            fig.add_shape(
                type="line",
                x0=hist_date_max,
                x1=hist_date_max,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="red", width=2, dash="dash")
            )
            fig.add_annotation(
                x=hist_date_max,
                y=1,
                yref="paper",
                text="Current Period",
                showarrow=False,
                yanchor="bottom",
                font=dict(color="red", size=10)
            )

        return fig

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'data': [],
            'layout': {
                'title': f'Error loading chart: {str(e)}',
                'xaxis': {'title': 'Period'},
                'yaxis': {'title': 'Value'}
            }
        }




@callback(
    Output('intracountry-count-visualization', 'figure'),
    Output('intracountry-tonmiles-visualization', 'figure'),
    Output('intracountry-year-dropdown', 'options'),
    Output('intracountry-year-dropdown', 'value'),
    Output('intracountry-status-dropdown', 'options'),
    Output('intracountry-status-dropdown', 'value'),
    Input('intracountry-data-store', 'data'),
    Input('dropdown-options-store', 'data'),
    Input('intracountry-year-dropdown', 'value'),
    Input('intracountry-status-dropdown', 'value'),
    prevent_initial_call=False
)
def update_intracountry_visualizations(intracountry_data, dropdown_options, selected_year, selected_statuses):
    """Update intracountry visualizations and tables based on selected filters."""
    # Check if data is available
    if intracountry_data is None or dropdown_options is None:
        raise PreventUpdate

    # Convert stored JSON back to DataFrame
    df_intracountry_trades = pd.read_json(StringIO(intracountry_data), orient='split')

    # Extract dropdown options
    year_options = dropdown_options['year_options']
    status_options = dropdown_options['status_options_intracountry']
    latest_year = dropdown_options['latest_year']

    # Set default values if needed
    if selected_year is None:
        selected_year = latest_year
    if selected_statuses is None:
        selected_statuses = ['All Statuses']

    # Create visualizations
    fig_intracountry_count = create_stacked_bar_chart(
        df_intracountry_trades,
        metric='count_trades',
        title_suffix='Count of Trades',
        selected_statuses=selected_statuses,
        is_intracountry=True
    )

    fig_intracountry_tonmiles = create_stacked_bar_chart(
        df_intracountry_trades,
        metric='sum_ton_miles',
        title_suffix='Ton Miles',
        selected_statuses=selected_statuses,
        is_intracountry=True
    )

    return (
        fig_intracountry_count,
        fig_intracountry_tonmiles,
        year_options,
        selected_year,
        status_options,
        selected_statuses,
    )


@callback(
    Output('download-intracountry-count-excel', 'data'),
    Input('export-intracountry-count-button', 'n_clicks'),
    State('intracountry-data-store', 'data'),
    State('intracountry-year-dropdown', 'value'),
    State('intracountry-status-dropdown', 'value'),
    prevent_initial_call=True
)
def export_intracountry_count_to_excel(n_clicks, intracountry_data, selected_year, selected_statuses):
    """Export Count of Intracountry Trades data to Excel."""
    if not n_clicks or intracountry_data is None:
        raise PreventUpdate

    df = pd.read_json(StringIO(intracountry_data), orient='split')
    table_data = prepare_table_data(
        df, 'count_trades',
        selected_year=selected_year,
        selected_statuses=selected_statuses,
        is_intracountry=True
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        table_data.to_excel(writer, sheet_name='Intracountry Count', index=False)
        for worksheet in writer.sheets.values():
            for column_cells in worksheet.columns:
                max_length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 50)
    output.seek(0)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(output.getvalue(), f'Intracountry_Count_{timestamp}.xlsx')


@callback(
    Output('download-intracountry-tonmiles-excel', 'data'),
    Input('export-intracountry-tonmiles-button', 'n_clicks'),
    State('intracountry-data-store', 'data'),
    State('intracountry-year-dropdown', 'value'),
    State('intracountry-status-dropdown', 'value'),
    prevent_initial_call=True
)
def export_intracountry_tonmiles_to_excel(n_clicks, intracountry_data, selected_year, selected_statuses):
    """Export Intracountry Ton Miles data to Excel."""
    if not n_clicks or intracountry_data is None:
        raise PreventUpdate

    df = pd.read_json(StringIO(intracountry_data), orient='split')
    table_data = prepare_table_data(
        df, 'sum_ton_miles',
        selected_year=selected_year,
        selected_statuses=selected_statuses,
        is_intracountry=True
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        table_data.to_excel(writer, sheet_name='Intracountry Ton Miles', index=False)
        for worksheet in writer.sheets.values():
            for column_cells in worksheet.columns:
                max_length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 50)
    output.seek(0)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(output.getvalue(), f'Intracountry_TonMiles_{timestamp}.xlsx')


def prepare_custom_metrics_data(df, metric, selected_year=None, selected_status=None, region_direction=None,
                                is_intracountry=False):
    """
    Prepare data for custom metrics table with vessel types as columns.

    Args:
        df: DataFrame with trade data
        metric: The column to display (e.g., 'median_delivery_days')
        selected_year: Year to filter by (defaults to latest)
        selected_status: Status value to filter by (defaults to 'laden')
        region_direction: How to handle regions for region metrics
        is_intracountry: Flag indicating if using intracountry data (which has origin_country_name)

    Returns:
        DataFrame formatted for display in a DataTable
    """
    # Filter data for years 2019 and later
    filtered_df = df[df['year'] >= 2019].copy()

    # Get the latest year if not specified
    if not selected_year or selected_year == "All Years":
        selected_year = filtered_df['year'].max()
    else:
        selected_year = int(selected_year)

    # Filter by year
    filtered_df = filtered_df[filtered_df['year'] == selected_year]

    # Filter by status if specified
    if selected_status and selected_status != "All Statuses":
        filtered_df = filtered_df[filtered_df['status'] == selected_status]

    # Check the type of analysis we're doing
    if region_direction:
        # For region metrics using trades_shipping_region data
        if region_direction == 'origin_to_destination':
            # Create a combined field for origin-destination pairs
            filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
                'destination_shipping_region']
            index_field = 'region_pair'
        elif region_direction == 'origin':
            index_field = 'origin_shipping_region'
        elif region_direction == 'destination':
            index_field = 'destination_shipping_region'
        else:
            # Default to region_pair if invalid value
            filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
                'destination_shipping_region']
            index_field = 'region_pair'
        # Select only the required columns
        keep_cols = ['year', index_field, 'status', metric]
    elif is_intracountry:
        # For intracountry data with origin_country_name
        index_field = 'origin_country_name'
        # Select only required columns
        keep_cols = ['year', index_field, 'status', metric]
    else:
        # For vessel metrics: analyze by year and status
        index_field = 'year'
        # Select only required columns
        keep_cols = ['year', 'status', metric]
    # Keep only the required columns that exist in the DataFrame
    existing_cols = [col for col in keep_cols if col in filtered_df.columns]
    filtered_df = filtered_df[existing_cols]

    # For vessel metrics, handle differently
    if not region_direction and not is_intracountry:
        # For vessel metrics - aggregate the data by year
        agg_data = filtered_df.groupby(['year']).agg({
            metric: 'mean'
        }).reset_index()
        # Sort values by year
        pivot_table = agg_data.sort_values('year')
    else:
        # For region or country metrics - aggregate without vessel type
        try:
            pivot_table = filtered_df.groupby([index_field, 'year', 'status'])[metric].mean().reset_index()
            # Sort the data by the index field
            pivot_table = pivot_table.sort_values(index_field)
        except KeyError as e:
            # Handle case where expected columns aren't found
            print(f"KeyError in groupby: {e}")
            # Create a simple DataFrame to show the error
            pivot_table = pd.DataFrame({
                'Error': [f"Missing column: {e}"]
            })

    return pivot_table


@callback(
    Output('custom-metrics-table-title', 'children'),
    Output('custom-metrics-table-container', 'children'),
    Input('intracountry-data-store', 'data'),
    Input('dropdown-options-store', 'data'),
    Input('metrics-year-dropdown', 'value'),
    Input('metrics-status-dropdown', 'value'),
    Input('metric-dropdown', 'value'),
    prevent_initial_call=False
)
def update_custom_metrics_table(intracountry_data, dropdown_options, selected_year, selected_status, selected_metric):
    """Update custom metrics table based on selected filters."""
    # Check if data is available
    if intracountry_data is None or dropdown_options is None:
        raise PreventUpdate

    # Convert stored JSON back to DataFrame
    df_intracountry_trades = pd.read_json(StringIO(intracountry_data), orient='split')

    # Extract dropdown options
    year_options = dropdown_options['year_options']
    status_options = dropdown_options['status_options_intracountry']
    latest_year = dropdown_options['latest_year']

    # Set default values if needed
    if selected_year is None:
        selected_year = latest_year

    # Set default status to 'laden' if not specified
    if selected_status is None:
        # Find 'laden' in status options
        laden_option = next((option['value'] for option in status_options
                             if option['value'].lower() == 'laden'), None)
        if laden_option:
            selected_status = laden_option
        else:
            # If 'laden' is not available, use the first status
            selected_status = status_options[0]['value'] if status_options else None

    # Get human-readable metric name
    metric_name = next((option['label'] for option in [
        {'label': 'Median Delivery Days', 'value': 'median_delivery_days'},
        # {'label': 'Mean Delivery Days', 'value': 'mean_delivery_days'},
        # {'label': 'Std Delivery Days', 'value': 'std_delivery_days'},
        # {'label': 'Mean Mileage (Nautical Miles)', 'value': 'mean_mileage_nautical_miles'},
        {'label': 'Median Mileage (Nautical Miles)', 'value': 'median_mileage_nautical_miles'},
        # {'label': 'Std Mileage (Nautical Miles)', 'value': 'std_mileage_nautical_miles'},
        {'label': 'Median Ton Miles', 'value': 'median_ton_miles'},
        # {'label': 'Std Ton Miles', 'value': 'std_ton_miles'},
        # {'label': 'Mean Utilization Rate', 'value': 'mean_utilization_rate'},
        {'label': 'Median Utilization Rate', 'value': 'median_utilization_rate'},
        # {'label': 'Mean Cargo Volume (m³)', 'value': 'mean_cargo_destination_cubic_meters'},
        {'label': 'Median Cargo Volume (m³)', 'value': 'median_cargo_destination_cubic_meters'},
        # {'label': 'Std Cargo Volume (m³)', 'value': 'std_cargo_destination_cubic_meters'},
        # {'label': 'Mean Vessel Capacity (m³)', 'value': 'mean_vessel_capacity_cubic_meters'},
        {'label': 'Median Vessel Capacity (m³)', 'value': 'median_vessel_capacity_cubic_meters'},
        {'label': 'Count of Trades', 'value': 'count_trades'}
    ] if option['value'] == selected_metric), selected_metric)

    # Prepare table title
    status_display = selected_status if selected_status != "All Statuses" else "All Statuses"
    table_title = f"{metric_name} by Vessel Type ({selected_year}, {status_display})"

    try:
        # Prepare data for table - note we specify is_intracountry=True
        custom_metrics_data = prepare_custom_metrics_data(
            df_intracountry_trades,
            selected_metric,
            selected_year,
            selected_status,
            region_direction=None,
            is_intracountry=True  # Specify that we're using intracountry data
        )
        # Check if we got data
        if custom_metrics_data.empty:
            return table_title, html.Div("No data available for the selected filters.")
        # Handle error case
        if 'Error' in custom_metrics_data.columns:
            return table_title, html.Div(f"Error: {custom_metrics_data['Error'].values[0]}")
        # Create columns for the table
        columns = []
        for col in custom_metrics_data.columns:
            if col in ['origin_country_name', 'year', 'status']:
                display_name = {
                    'origin_country_name': 'Origin Country',
                    'year': 'Year',
                    'status': 'Status'
                }.get(col, col.capitalize())
                columns.append({"name": display_name, "id": col})
            else:
                # Other columns (the metric or vessel types)
                columns.append({
                    "name": col,
                    "id": col,
                    "type": "numeric",
                    "format": Format(
                        precision=2,
                        scheme=Scheme.fixed,
                        group=Group.yes,
                        group_delimiter=',',
                        decimal_delimiter='.'
                    )
                })

        # Create the table
        custom_metrics_table = dash_table.DataTable(
            id='custom-metrics-table',
            columns=columns,
            data=custom_metrics_data.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'minWidth': '100px',
                'maxWidth': '300px',
                'whiteSpace': 'normal',
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            page_size=25,
            sort_action='native',
            filter_action='native',
            fill_width=False,
            export_format='xlsx',
            export_headers='display',
            export_columns='visible'
        )
        return table_title, custom_metrics_table
    except Exception as e:
        print(f"Error in update_custom_metrics_table: {str(e)}")
        return table_title, html.Div(f"Error: {str(e)}")


# First callback - just handles the dropdowns
@callback(
    Output('region-metrics-year-dropdown', 'options'),
    Output('region-metrics-year-dropdown', 'value'),
    Output('region-metrics-status-dropdown', 'options'),
    Output('region-metrics-status-dropdown', 'value'),
    Input('dropdown-options-store', 'data'),
    prevent_initial_call=False
)
def update_region_metrics_dropdowns(dropdown_options):
    """Update region metrics dropdown options."""
    # Check if data is available
    if dropdown_options is None:
        raise PreventUpdate

    # Extract dropdown options
    year_options = dropdown_options['year_options']
    status_options = dropdown_options['status_options_region']
    latest_year = dropdown_options['latest_year']

    # Set default values
    selected_year = latest_year
    # Set default status to 'laden' if available
    laden_option = next((option['value'] for option in status_options
                         if option['value'].lower() == 'laden'), None)
    if laden_option:
        selected_status = laden_option
    else:
        # If 'laden' is not available, use the first status
        selected_status = status_options[0]['value'] if status_options else None

    return year_options, selected_year, status_options, selected_status


# Second callback - handles the table and title
@callback(
    Output('region-custom-metrics-table-title', 'children'),
    Output('region-custom-metrics-table-container', 'children'),
    Input('trades-shipping-data-store', 'data'),
    Input('region-metrics-year-dropdown', 'value'),
    Input('region-metrics-status-dropdown', 'value'),
    Input('region-metric-dropdown', 'value'),
    # Input('region-direction', 'value'),
    prevent_initial_call=False
)
def update_region_metrics_table(shipping_data, selected_year, selected_status,
                                selected_metric#, region_direction
                                ):
    """Update region custom metrics table based on selected filters."""
    # Check if data is available
    if shipping_data is None:
        raise PreventUpdate

    try:
        # Convert stored JSON back to DataFrame
        df_trades_shipping_region = pd.read_json(StringIO(shipping_data), orient='split')
        # Get human-readable metric name
        metric_options = [
            {'label': 'Median Delivery Days', 'value': 'median_delivery_days'},
            # {'label': 'Mean Delivery Days', 'value': 'mean_delivery_days'},
            # {'label': 'Mean Mileage (Nautical Miles)', 'value': 'mean_mileage_nautical_miles'},
            {'label': 'Median Mileage (Nautical Miles)', 'value': 'median_mileage_nautical_miles'},
            {'label': 'Median Ton Miles', 'value': 'median_ton_miles'},
            # {'label': 'Mean Utilization Rate', 'value': 'mean_utilization_rate'},
            {'label': 'Median Utilization Rate', 'value': 'median_utilization_rate'},
            # {'label': 'Mean Cargo Volume (m³)', 'value': 'mean_cargo_destination_cubic_meters'},
            {'label': 'Median Cargo Volume (m³)', 'value': 'median_cargo_destination_cubic_meters'},
            # {'label': 'Mean Vessel Capacity (m³)', 'value': 'mean_vessel_capacity_cubic_meters'},
            {'label': 'Median Vessel Capacity (m³)', 'value': 'median_vessel_capacity_cubic_meters'},
            {'label': 'Count of Trades', 'value': 'count_trades'},
        ]

        metric_name = next((option['label'] for option in metric_options
                            if option['value'] == selected_metric), selected_metric)

        # Get region type description
        region_type_desc = 'Shipping Regions'
        #     {
        #     'origin_to_destination': 'Shipping Region Pairs',
        #     'origin': 'Origin Shipping Regions',
        #     'destination': 'Destination Shipping Regions'
        # }.get(region_direction, 'Shipping Regions')
        region_direction= 'origin_to_destination'

        # Prepare table title
        status_display = selected_status if selected_status != "All Statuses" else "All Statuses"
        table_title = f"{metric_name} by {region_type_desc} and Vessel Type ({selected_year}, {status_display})"

        # Filter data for years 2019 and later
        filtered_df = df_trades_shipping_region[df_trades_shipping_region['year'] >= 2019].copy()

        # Get the latest year if not specified
        if not selected_year or selected_year == "All Years":
            selected_year = filtered_df['year'].max()
        else:
            selected_year = int(selected_year)

        # Filter by year
        filtered_df = filtered_df[filtered_df['year'] == selected_year]

        # Filter by status if specified
        if selected_status and selected_status != "All Statuses":
            filtered_df = filtered_df[filtered_df['status'] == selected_status]

        # Handle region direction
        if region_direction == 'origin_to_destination':
            # Create a combined field for origin-destination pairs
            filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
                'destination_shipping_region']
            index_field = 'region_pair'
        elif region_direction == 'origin':
            index_field = 'origin_shipping_region'
        elif region_direction == 'destination':
            index_field = 'destination_shipping_region'
        else:
            # Default to region_pair if invalid value
            filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
                'destination_shipping_region']
            index_field = 'region_pair'

        # Directly aggregate the data without using the helper function (vessel_type removed)
        pivot_table = filtered_df.groupby([index_field, 'year', 'season', 'quarter', 'status'])[
            selected_metric].mean().reset_index()
        # Create columns for the table
        columns = []
        # Add all columns
        for col in pivot_table.columns:
            if col in [index_field, 'year', 'season', 'quarter', 'status']:
                display_name = {
                    'region_pair': 'Region Pair',
                    'origin_shipping_region': 'Origin Region',
                    'destination_shipping_region': 'Destination Region',
                    'year': 'Year',
                    'season': 'Season',
                    'quarter': 'Quarter',
                    'status': 'Status'
                }.get(col, col.capitalize())
                columns.append({"name": display_name, "id": col})
            else:
                # Metric columns
                columns.append({
                    "name": col,
                    "id": col,
                    "type": "numeric",
                    "format": Format(
                        precision=2,
                        scheme=Scheme.fixed,
                        group=Group.yes,
                        group_delimiter=',',
                        decimal_delimiter='.'
                    )
                })

        # Create the table
        custom_metrics_table = dash_table.DataTable(
            id='region-custom-metrics-table',
            columns=columns,
            data=pivot_table.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'minWidth': '100px',
                'maxWidth': '300px',
                'whiteSpace': 'normal',
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            page_size=25,
            sort_action='native',
            filter_action='native',
            fill_width=False,
            export_format='xlsx',
            export_headers='display',
            export_columns='visible'
        )

        return table_title, custom_metrics_table
    except Exception as e:
        # Print for debugging
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error in update_region_metrics_table: {str(e)}")
        print(traceback_str)
        # Return error message to the user
        return "Error in Custom Metrics Analysis", html.Div([
            html.P(f"An error occurred: {str(e)}"),
            html.Pre(traceback_str)
        ])


# Regional Breakdown Callbacks
@callback(
    Output('demand-regional-year-dropdown', 'options'),
    Output('demand-regional-year-dropdown', 'value'),
    Output('demand-regional-period-dropdown', 'options'),
    Output('demand-regional-period-dropdown', 'value'),
    Input('shipping-balance-regional-data-store', 'data'),
    Input('demand-regional-aggregation-dropdown', 'value'),
    prevent_initial_call=False
)
def update_demand_regional_dropdowns(regional_data, aggregation_level):
    """Update year and period dropdown options for demand regional table."""
    if regional_data is None:
        raise PreventUpdate

    # Convert stored JSON back to DataFrame
    df_regional = pd.read_json(StringIO(regional_data), orient='split')

    # Extract unique years from the date column
    df_regional['year'] = pd.to_datetime(df_regional['date']).dt.year
    df_regional['month'] = pd.to_datetime(df_regional['date']).dt.month
    df_regional['quarter'] = pd.to_datetime(df_regional['date']).dt.quarter

    years = sorted(df_regional['year'].unique())
    year_options = [{'label': str(year), 'value': year} for year in years]
    default_year = years[-1] if years else None

    # Create period options based on aggregation level
    if aggregation_level == 'monthly':
        period_options = [{'label': f'Month {i}', 'value': i} for i in range(1, 13)]
        default_period = df_regional[df_regional['year'] == default_year]['month'].max() if default_year else 1
    elif aggregation_level == 'quarterly':
        period_options = [{'label': f'Q{i}', 'value': i} for i in range(1, 5)]
        default_period = df_regional[df_regional['year'] == default_year]['quarter'].max() if default_year else 1
    elif aggregation_level == 'seasonal':
        period_options = [{'label': 'Winter', 'value': 1}, {'label': 'Summer', 'value': 7}]
        default_period = df_regional[df_regional['year'] == default_year]['month'].max() if default_year else 1
        default_period = 1 if default_period in [1, 2, 3, 4, 5, 6] else 7
    else:  # yearly
        period_options = [{'label': 'Full Year', 'value': 1}]
        default_period = 1

    return year_options, default_year, period_options, default_period


@callback(
    Output('supply-regional-year-dropdown', 'options'),
    Output('supply-regional-year-dropdown', 'value'),
    Output('supply-regional-period-dropdown', 'options'),
    Output('supply-regional-period-dropdown', 'value'),
    Input('shipping-balance-supply-regional-data-store', 'data'),
    Input('supply-regional-aggregation-dropdown', 'value'),
    prevent_initial_call=False
)
def update_supply_regional_dropdowns(regional_data, aggregation_level):
    """Update year and period dropdown options for supply regional table."""
    if regional_data is None:
        raise PreventUpdate

    # Convert stored JSON back to DataFrame
    df_regional = pd.read_json(StringIO(regional_data), orient='split')

    # Extract unique years from the date column
    df_regional['year'] = pd.to_datetime(df_regional['date']).dt.year
    df_regional['month'] = pd.to_datetime(df_regional['date']).dt.month
    df_regional['quarter'] = pd.to_datetime(df_regional['date']).dt.quarter

    years = sorted(df_regional['year'].unique())
    year_options = [{'label': str(year), 'value': year} for year in years]
    default_year = years[-1] if years else None

    # Create period options based on aggregation level
    if aggregation_level == 'monthly':
        period_options = [{'label': f'Month {i}', 'value': i} for i in range(1, 13)]
        default_period = df_regional[df_regional['year'] == default_year]['month'].max() if default_year else 1
    elif aggregation_level == 'quarterly':
        period_options = [{'label': f'Q{i}', 'value': i} for i in range(1, 5)]
        default_period = df_regional[df_regional['year'] == default_year]['quarter'].max() if default_year else 1
    elif aggregation_level == 'seasonal':
        period_options = [{'label': 'Winter', 'value': 1}, {'label': 'Summer', 'value': 7}]
        default_period = df_regional[df_regional['year'] == default_year]['month'].max() if default_year else 1
        default_period = 1 if default_period in [1, 2, 3, 4, 5, 6] else 7
    else:  # yearly
        period_options = [{'label': 'Full Year', 'value': 1}]
        default_period = 1

    return year_options, default_year, period_options, default_period


# Vessel Type Dropdown Callbacks
@callback(
    Output('demand-regional-vessel-type-dropdown', 'options'),
    Input('vessel-type-options-store', 'data'),
    prevent_initial_call=False
)
def update_demand_vessel_type_options(vessel_type_options):
    """Update vessel type dropdown options for demand regional table."""
    if vessel_type_options is None:
        return []
    return vessel_type_options


@callback(
    Output('supply-regional-vessel-type-dropdown', 'options'),
    Input('vessel-type-options-store', 'data'),
    prevent_initial_call=False
)
def update_supply_vessel_type_options(vessel_type_options):
    """Update vessel type dropdown options for supply regional table."""
    if vessel_type_options is None:
        return []
    return vessel_type_options


@callback(
    Output('demand-regional-table-container', 'children'),
    Input('shipping-balance-regional-data-store', 'data'),
    Input('demand-regional-aggregation-dropdown', 'value'),
    Input('demand-regional-year-dropdown', 'value'),
    Input('demand-regional-period-dropdown', 'value'),
    prevent_initial_call=False
)
def update_demand_regional_table(regional_data, aggregation_level, selected_year, selected_period):
    """Update demand regional breakdown table."""
    if regional_data is None or selected_year is None or selected_period is None:
        raise PreventUpdate

    try:
        # Convert stored JSON back to DataFrame
        df_regional = pd.read_json(StringIO(regional_data), orient='split')

        # Add time columns
        df_regional['year'] = pd.to_datetime(df_regional['date']).dt.year
        df_regional['month'] = pd.to_datetime(df_regional['date']).dt.month
        df_regional['quarter'] = pd.to_datetime(df_regional['date']).dt.quarter

        # Filter by year and period based on aggregation level
        if aggregation_level == 'monthly':
            df_filtered = df_regional[(df_regional['year'] == selected_year) & (df_regional['month'] == selected_period)]
        elif aggregation_level == 'quarterly':
            df_filtered = df_regional[(df_regional['year'] == selected_year) & (df_regional['quarter'] == selected_period)]
        elif aggregation_level == 'seasonal':
            # selected_period is 1 (Winter/Jan) or 7 (Summer/Jul)
            df_filtered = df_regional[(df_regional['year'] == selected_year) & (df_regional['month'] == selected_period)]
        else:  # yearly
            df_filtered = df_regional[df_regional['year'] == selected_year]

        if df_filtered.empty:
            return html.Div("No data available for the selected period.", style={'padding': '10px', 'color': '#666'})

        # Select and rename columns for display
        # MODIFIED: Now including origin and destination shipping regions
        # Use utilization_ratio (already in %) instead of aggregate_utilization_rate (0-1 range)
        # Include sum_cargo and sum_vessel_capacity columns
        cols_to_select = ['origin_shipping_region', 'destination_shipping_region', 'value', 'ships_demand',
                         'mean_vessel_capacity', 'trade_vessel_capacity',
                         'mean_cargo_cubic_meters', 'count_laden_trades', 'count_nonladen_trades',
                         'sum_ton_miles', 'median_vessel_speed_laden', 'median_vessel_speed_nonladen',
                         'utilization_ratio', 'median_laden_days', 'median_nonladen_days']

        # Add optional columns if they exist
        if 'sum_cargo' in df_filtered.columns:
            cols_to_select.append('sum_cargo')
        if 'sum_vessel_capacity' in df_filtered.columns:
            cols_to_select.append('sum_vessel_capacity')

        display_df = df_filtered[cols_to_select].copy()

        # Convert value from m³ to M m³ (divide by 1,000,000)
        display_df['value'] = (display_df['value'] / 1000000).round(2)

        # Calculate volume percentage
        total_volume = display_df['value'].sum()
        display_df['volume_pct'] = (display_df['value'] / total_volume * 100).round(2) if total_volume > 0 else 0

        # Convert sum_cargo and sum_vessel_capacity from m³ to M m³ if they exist
        if 'sum_cargo' in display_df.columns:
            display_df['sum_cargo'] = (display_df['sum_cargo'] / 1000000).round(2)
        if 'sum_vessel_capacity' in display_df.columns:
            display_df['sum_vessel_capacity'] = (display_df['sum_vessel_capacity'] / 1000000).round(2)

        # Build rename dictionary
        rename_dict = {
            'origin_shipping_region': 'Origin Region',
            'destination_shipping_region': 'Destination Region',
            'value': 'LNG Volume (M m³)',
            'volume_pct': 'Volume Share (%)',
            'ships_demand': 'Ships Demand',
            'mean_vessel_capacity': 'Fleet Avg Size (m³)',
            'trade_vessel_capacity': 'Trade Avg Size (m³)',
            'mean_cargo_cubic_meters': 'Avg Cargo Volume (m³)',
            'count_laden_trades': 'Laden Trades',
            'count_nonladen_trades': 'Ballast Trades',
            'sum_ton_miles': 'Total Ton-Miles',
            'median_vessel_speed_laden': 'Speed Laden (kts)',
            'median_vessel_speed_nonladen': 'Speed Ballast (kts)',
            'utilization_ratio': 'Utilization Rate (%)',
            'median_laden_days': 'Laden Days',
            'median_nonladen_days': 'Ballast Days'
        }

        # Add optional columns to rename dict if they exist
        if 'sum_cargo' in display_df.columns:
            rename_dict['sum_cargo'] = 'Sum Cargo (M m³)'
        if 'sum_vessel_capacity' in display_df.columns:
            rename_dict['sum_vessel_capacity'] = 'Sum Vessel Capacity (M m³)'

        display_df = display_df.rename(columns=rename_dict)

        # Sort by Ships Demand descending
        display_df = display_df.sort_values('Ships Demand', ascending=False)

        # Create DataTable
        columns = [
            {"name": "Origin Region", "id": "Origin Region"},
            {"name": "Destination Region", "id": "Destination Region"},
            {"name": "LNG Volume (M m³)", "id": "LNG Volume (M m³)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=2)},
            {"name": "Volume Share (%)", "id": "Volume Share (%)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=2)},
            {"name": "Ships Demand", "id": "Ships Demand", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=1)},
            {"name": "Fleet Avg Size (m³)", "id": "Fleet Avg Size (m³)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=0)},
            {"name": "Trade Avg Size (m³)", "id": "Trade Avg Size (m³)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=0)},
            {"name": "Avg Cargo Volume (m³)", "id": "Avg Cargo Volume (m³)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=0)},
            {"name": "Laden Trades", "id": "Laden Trades", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=0)},
            {"name": "Ballast Trades", "id": "Ballast Trades", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=0)},
            {"name": "Total Ton-Miles", "id": "Total Ton-Miles", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=0)},
            {"name": "Speed Laden (kts)", "id": "Speed Laden (kts)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=1)},
            {"name": "Speed Ballast (kts)", "id": "Speed Ballast (kts)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=1)},
            {"name": "Utilization Rate (%)", "id": "Utilization Rate (%)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=2)},
            {"name": "Laden Days", "id": "Laden Days", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=1)},
            {"name": "Ballast Days", "id": "Ballast Days", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=1)}
        ]

        # Add optional columns to DataTable if they exist in display_df
        if 'Sum Cargo (M m³)' in display_df.columns:
            columns.insert(-2, {"name": "Sum Cargo (M m³)", "id": "Sum Cargo (M m³)", "type": "numeric",
                               "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=2)})
        if 'Sum Vessel Capacity (M m³)' in display_df.columns:
            columns.insert(-2, {"name": "Sum Vessel Capacity (M m³)", "id": "Sum Vessel Capacity (M m³)", "type": "numeric",
                               "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=2)})

        # Calculate total LNG volume for validation against WoodMac
        total_lng_volume = display_df['LNG Volume (M m³)'].sum()

        # Fetch WoodMac data for comparison
        try:
            # Determine grouping and date filter based on aggregation level
            if aggregation_level == 'monthly':
                group_by_clause = "country_name, start_date::DATE, publication_date, source"
                date_column = "start_date::DATE"
                date_filter = f"EXTRACT(YEAR FROM start_date) = {selected_year} AND EXTRACT(MONTH FROM start_date) = {selected_period}"
            elif aggregation_level == 'quarterly':
                group_by_clause = "country_name, DATE_TRUNC('quarter', start_date::DATE), publication_date, source"
                date_column = "DATE_TRUNC('quarter', start_date::DATE)"
                # Filter on the truncated quarter start date
                quarter_start_month = (selected_period - 1) * 3 + 1
                date_filter = f"start_date = DATE '{selected_year}-{quarter_start_month:02d}-01'"
            elif aggregation_level == 'seasonal':
                group_by_clause = "country_name, CASE WHEN EXTRACT(MONTH FROM start_date::DATE) BETWEEN 1 AND 6 THEN DATE_TRUNC('year', start_date::DATE) ELSE DATE_TRUNC('year', start_date::DATE) + INTERVAL '6 months' END, publication_date, source"
                date_column = "CASE WHEN EXTRACT(MONTH FROM start_date::DATE) BETWEEN 1 AND 6 THEN DATE_TRUNC('year', start_date::DATE) ELSE DATE_TRUNC('year', start_date::DATE) + INTERVAL '6 months' END"
                date_filter = f"start_date = DATE '{selected_year}-{selected_period:02d}-01'"
            else:  # yearly
                group_by_clause = "country_name, DATE_TRUNC('year', start_date::DATE), publication_date, source"
                date_column = "DATE_TRUNC('year', start_date::DATE)"
                date_filter = f"start_date = DATE '{selected_year}-01-01'"

            wm_query = f'''
            WITH latest_short_term AS (
                SELECT
                    country_name,
                    {date_column}::DATE as start_date,
                    SUM(metric_value) / 12 * 2222*1000 AS value,
                    publication_date,
                    'Short Term' as source
                FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
                WHERE market_outlook = (
                    SELECT market_outlook
                    FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
                    WHERE release_type = 'Short Term Outlook'
                    GROUP BY market_outlook
                    ORDER BY TO_DATE(
                        (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{{4}})'))[1]
                        || ' ' ||
                        (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{{4}})'))[2],
                        'Month YYYY'
                    ) DESC, MAX(publication_date) DESC
                    LIMIT 1
                )
                AND release_type = 'Short Term Outlook'
                AND direction = 'Import'
                AND measured_at = 'Entry'
                AND metric_name = 'Flow'
                AND start_date::DATE < '2036-01-01'
                GROUP BY {group_by_clause}
                HAVING SUM(metric_value) > 0
            ),
            short_term_max_date AS (
                SELECT MAX(start_date::DATE) as max_date
                FROM latest_short_term
            ),
            latest_long_term_raw AS (
                SELECT
                    country_name,
                    {date_column}::DATE as start_date,
                    SUM(metric_value) / 12 * 2222*1000 AS value,
                    publication_date,
                    'Long Term' as source
                FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
                WHERE market_outlook = (
                    SELECT market_outlook
                    FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
                    WHERE release_type = 'Long Term Outlook'
                    GROUP BY market_outlook
                    ORDER BY TO_DATE(
                        (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{{4}})'))[1]
                        || ' ' ||
                        (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{{4}})'))[2],
                        'Month YYYY'
                    ) DESC, MAX(publication_date) DESC
                    LIMIT 1
                )
                AND release_type = 'Long Term Outlook'
                AND direction = 'Import'
                AND measured_at = 'Entry'
                AND metric_name = 'Flow'
                AND start_date::DATE < '2036-01-01'
                GROUP BY {group_by_clause}
                HAVING SUM(metric_value) > 0
            ),
            latest_long_term AS (
                SELECT *
                FROM latest_long_term_raw
                WHERE start_date > (SELECT max_date FROM short_term_max_date)
            ),
            combined AS (
                SELECT * FROM latest_short_term
                UNION ALL
                SELECT * FROM latest_long_term
            ),
            filtered AS (
                SELECT * FROM combined WHERE {date_filter}
            )
            SELECT COALESCE(SUM(value) / 1000000, 0) as woodmac_total,
                   STRING_AGG(DISTINCT source, ' + ') as sources
            FROM filtered
            '''

            wm_result = pd.read_sql(wm_query, engine)
            woodmac_total = wm_result['woodmac_total'].iloc[0]
            woodmac_sources = wm_result['sources'].iloc[0] if wm_result['sources'].iloc[0] else 'N/A'

            difference = total_lng_volume - woodmac_total
            pct_diff = (difference / woodmac_total * 100) if woodmac_total > 0 else 0

            # Color coding based on percentage difference
            if abs(pct_diff) < 5:
                diff_color = '#27AE60'  # Green
            elif abs(pct_diff) < 10:
                diff_color = '#F39C12'  # Orange
            else:
                diff_color = '#E74C3C'  # Red

        except Exception as e:
            woodmac_total = 0
            woodmac_sources = 'Error'
            difference = 0
            pct_diff = 0
            diff_color = '#999'

        # Create validation summary with WoodMac comparison
        validation_summary = html.Div([
            html.Div([
                html.Strong("Total LNG Volume (Demand): ", style={'fontSize': '14px'}),
                html.Span(f"Regional: {total_lng_volume:,.2f} M m³", style={'fontSize': '14px', 'fontWeight': 'bold', 'color': '#2E86C1', 'marginRight': '15px'}),
                html.Span(" | ", style={'fontSize': '14px', 'color': '#999'}),
                html.Span(f"WoodMac ({woodmac_sources}): {woodmac_total:,.2f} M m³", style={'fontSize': '14px', 'fontWeight': 'bold', 'color': '#2E86C1', 'marginLeft': '15px', 'marginRight': '15px'}),
                html.Span(" | ", style={'fontSize': '14px', 'color': '#999'}),
                html.Span(f"Diff: {difference:+,.2f} M m³ ({pct_diff:+.1f}%)", style={'fontSize': '14px', 'fontWeight': 'bold', 'color': diff_color, 'marginLeft': '15px'})
            ], style={'padding': '10px', 'marginBottom': '10px', 'backgroundColor': '#E8F4F8', 'borderLeft': '4px solid #2E86C1', 'borderRadius': '4px'})
        ])

        table = dash_table.DataTable(
            columns=columns,
            data=display_df.to_dict('records'),
            style_table={'overflowX': 'auto', 'width': '100%'},
            style_cell={
                'textAlign': 'center',
                'padding': '8px',
                'fontSize': '12px',
                'fontFamily': 'Segoe UI, -apple-system, BlinkMacSystemFont, sans-serif'
            },
            style_header={
                'backgroundColor': '#2E86C1',
                'color': 'white',
                'fontWeight': 'bold',
                'fontSize': '12px',
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            page_size=15,
            export_format='xlsx'
        )

        return html.Div([validation_summary, table])

    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div(f"Error loading regional table: {str(e)}", style={'padding': '10px', 'color': 'red'})


@callback(
    Output('supply-regional-table-container', 'children'),
    Input('shipping-balance-supply-regional-data-store', 'data'),
    Input('supply-regional-aggregation-dropdown', 'value'),
    Input('supply-regional-year-dropdown', 'value'),
    Input('supply-regional-period-dropdown', 'value'),
    prevent_initial_call=False
)
def update_supply_regional_table(regional_data, aggregation_level, selected_year, selected_period):
    """Update supply regional breakdown table."""
    if regional_data is None or selected_year is None or selected_period is None:
        raise PreventUpdate

    try:
        # Convert stored JSON back to DataFrame
        df_regional = pd.read_json(StringIO(regional_data), orient='split')

        # Add time columns
        df_regional['year'] = pd.to_datetime(df_regional['date']).dt.year
        df_regional['month'] = pd.to_datetime(df_regional['date']).dt.month
        df_regional['quarter'] = pd.to_datetime(df_regional['date']).dt.quarter

        # Filter by year and period based on aggregation level
        if aggregation_level == 'monthly':
            df_filtered = df_regional[(df_regional['year'] == selected_year) & (df_regional['month'] == selected_period)]
        elif aggregation_level == 'quarterly':
            df_filtered = df_regional[(df_regional['year'] == selected_year) & (df_regional['quarter'] == selected_period)]
        elif aggregation_level == 'seasonal':
            # selected_period is 1 (Winter/Jan) or 7 (Summer/Jul)
            df_filtered = df_regional[(df_regional['year'] == selected_year) & (df_regional['month'] == selected_period)]
        else:  # yearly
            df_filtered = df_regional[df_regional['year'] == selected_year]

        if df_filtered.empty:
            return html.Div("No data available for the selected period.", style={'padding': '10px', 'color': '#666'})

        # Select and rename columns for display
        # MODIFIED: Now including origin and destination shipping regions
        # Use utilization_ratio (already in %) instead of aggregate_utilization_rate (0-1 range)
        # Include sum_cargo and sum_vessel_capacity columns
        cols_to_select = ['origin_shipping_region', 'destination_shipping_region', 'value', 'ships_demand',
                         'mean_vessel_capacity', 'trade_vessel_capacity',
                         'mean_cargo_cubic_meters', 'count_laden_trades', 'count_nonladen_trades',
                         'sum_ton_miles', 'median_vessel_speed_laden', 'median_vessel_speed_nonladen',
                         'utilization_ratio', 'median_laden_days', 'median_nonladen_days']

        # Add optional columns if they exist
        if 'sum_cargo' in df_filtered.columns:
            cols_to_select.append('sum_cargo')
        if 'sum_vessel_capacity' in df_filtered.columns:
            cols_to_select.append('sum_vessel_capacity')

        display_df = df_filtered[cols_to_select].copy()

        # Convert value from m³ to M m³ (divide by 1,000,000)
        display_df['value'] = (display_df['value'] / 1000000).round(2)

        # Calculate volume percentage
        total_volume = display_df['value'].sum()
        display_df['volume_pct'] = (display_df['value'] / total_volume * 100).round(2) if total_volume > 0 else 0

        # Convert sum_cargo and sum_vessel_capacity from m³ to M m³ if they exist
        if 'sum_cargo' in display_df.columns:
            display_df['sum_cargo'] = (display_df['sum_cargo'] / 1000000).round(2)
        if 'sum_vessel_capacity' in display_df.columns:
            display_df['sum_vessel_capacity'] = (display_df['sum_vessel_capacity'] / 1000000).round(2)

        # Build rename dictionary
        rename_dict = {
            'origin_shipping_region': 'Origin Region',
            'destination_shipping_region': 'Destination Region',
            'value': 'LNG Volume (M m³)',
            'volume_pct': 'Volume Share (%)',
            'ships_demand': 'Ships Demand',
            'mean_vessel_capacity': 'Fleet Avg Size (m³)',
            'trade_vessel_capacity': 'Trade Avg Size (m³)',
            'mean_cargo_cubic_meters': 'Avg Cargo Volume (m³)',
            'count_laden_trades': 'Laden Trades',
            'count_nonladen_trades': 'Ballast Trades',
            'sum_ton_miles': 'Total Ton-Miles',
            'median_vessel_speed_laden': 'Speed Laden (kts)',
            'median_vessel_speed_nonladen': 'Speed Ballast (kts)',
            'utilization_ratio': 'Utilization Rate (%)',
            'median_laden_days': 'Laden Days',
            'median_nonladen_days': 'Ballast Days'
        }

        # Add optional columns to rename dict if they exist
        if 'sum_cargo' in display_df.columns:
            rename_dict['sum_cargo'] = 'Sum Cargo (M m³)'
        if 'sum_vessel_capacity' in display_df.columns:
            rename_dict['sum_vessel_capacity'] = 'Sum Vessel Capacity (M m³)'

        display_df = display_df.rename(columns=rename_dict)

        # Sort by Ships Demand descending
        display_df = display_df.sort_values('Ships Demand', ascending=False)

        # Create DataTable
        columns = [
            {"name": "Origin Region", "id": "Origin Region"},
            {"name": "Destination Region", "id": "Destination Region"},
            {"name": "LNG Volume (M m³)", "id": "LNG Volume (M m³)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=2)},
            {"name": "Volume Share (%)", "id": "Volume Share (%)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=2)},
            {"name": "Ships Demand", "id": "Ships Demand", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=1)},
            {"name": "Fleet Avg Size (m³)", "id": "Fleet Avg Size (m³)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=0)},
            {"name": "Trade Avg Size (m³)", "id": "Trade Avg Size (m³)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=0)},
            {"name": "Avg Cargo Volume (m³)", "id": "Avg Cargo Volume (m³)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=0)},
            {"name": "Laden Trades", "id": "Laden Trades", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=0)},
            {"name": "Ballast Trades", "id": "Ballast Trades", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=0)},
            {"name": "Total Ton-Miles", "id": "Total Ton-Miles", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=0)},
            {"name": "Speed Laden (kts)", "id": "Speed Laden (kts)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=1)},
            {"name": "Speed Ballast (kts)", "id": "Speed Ballast (kts)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=1)},
            {"name": "Utilization Rate (%)", "id": "Utilization Rate (%)", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=2)},
            {"name": "Laden Days", "id": "Laden Days", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=1)},
            {"name": "Ballast Days", "id": "Ballast Days", "type": "numeric",
             "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=1)}
        ]

        # Add optional columns to DataTable if they exist in display_df
        if 'Sum Cargo (M m³)' in display_df.columns:
            columns.insert(-2, {"name": "Sum Cargo (M m³)", "id": "Sum Cargo (M m³)", "type": "numeric",
                               "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=2)})
        if 'Sum Vessel Capacity (M m³)' in display_df.columns:
            columns.insert(-2, {"name": "Sum Vessel Capacity (M m³)", "id": "Sum Vessel Capacity (M m³)", "type": "numeric",
                               "format": Format(group=Group.yes, scheme=Scheme.fixed, precision=2)})

        # Calculate total LNG volume for validation against WoodMac
        total_lng_volume = display_df['LNG Volume (M m³)'].sum()

        # Fetch WoodMac data for comparison
        try:
            # Determine grouping and date filter based on aggregation level
            if aggregation_level == 'monthly':
                group_by_clause = "country_name, start_date::DATE, publication_date, source"
                date_column = "start_date::DATE"
                date_filter = f"EXTRACT(YEAR FROM start_date) = {selected_year} AND EXTRACT(MONTH FROM start_date) = {selected_period}"
            elif aggregation_level == 'quarterly':
                group_by_clause = "country_name, DATE_TRUNC('quarter', start_date::DATE), publication_date, source"
                date_column = "DATE_TRUNC('quarter', start_date::DATE)"
                # Filter on the truncated quarter start date
                quarter_start_month = (selected_period - 1) * 3 + 1
                date_filter = f"start_date = DATE '{selected_year}-{quarter_start_month:02d}-01'"
            elif aggregation_level == 'seasonal':
                group_by_clause = "country_name, CASE WHEN EXTRACT(MONTH FROM start_date::DATE) BETWEEN 1 AND 6 THEN DATE_TRUNC('year', start_date::DATE) ELSE DATE_TRUNC('year', start_date::DATE) + INTERVAL '6 months' END, publication_date, source"
                date_column = "CASE WHEN EXTRACT(MONTH FROM start_date::DATE) BETWEEN 1 AND 6 THEN DATE_TRUNC('year', start_date::DATE) ELSE DATE_TRUNC('year', start_date::DATE) + INTERVAL '6 months' END"
                date_filter = f"start_date = DATE '{selected_year}-{selected_period:02d}-01'"
            else:  # yearly
                group_by_clause = "country_name, DATE_TRUNC('year', start_date::DATE), publication_date, source"
                date_column = "DATE_TRUNC('year', start_date::DATE)"
                date_filter = f"start_date = DATE '{selected_year}-01-01'"

            wm_query = f'''
            WITH latest_short_term AS (
                SELECT
                    country_name,
                    {date_column}::DATE as start_date,
                    SUM(metric_value) / 12 * 2222*1000 AS value,
                    publication_date,
                    'Short Term' as source
                FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
                WHERE market_outlook = (
                    SELECT market_outlook
                    FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
                    WHERE release_type = 'Short Term Outlook'
                    GROUP BY market_outlook
                    ORDER BY TO_DATE(
                        (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{{4}})'))[1]
                        || ' ' ||
                        (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{{4}})'))[2],
                        'Month YYYY'
                    ) DESC, MAX(publication_date) DESC
                    LIMIT 1
                )
                AND release_type = 'Short Term Outlook'
                AND direction = 'Export'
                AND measured_at = 'Exit'
                AND metric_name = 'Flow'
                AND start_date::DATE < '2036-01-01'
                GROUP BY {group_by_clause}
                HAVING SUM(metric_value) > 0
            ),
            short_term_max_date AS (
                SELECT MAX(start_date::DATE) as max_date
                FROM latest_short_term
            ),
            latest_long_term_raw AS (
                SELECT
                    country_name,
                    {date_column}::DATE as start_date,
                    SUM(metric_value) / 12 * 2222*1000 AS value,
                    publication_date,
                    'Long Term' as source
                FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
                WHERE market_outlook = (
                    SELECT market_outlook
                    FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
                    WHERE release_type = 'Long Term Outlook'
                    GROUP BY market_outlook
                    ORDER BY TO_DATE(
                        (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{{4}})'))[1]
                        || ' ' ||
                        (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{{4}})'))[2],
                        'Month YYYY'
                    ) DESC, MAX(publication_date) DESC
                    LIMIT 1
                )
                AND release_type = 'Long Term Outlook'
                AND direction = 'Export'
                AND measured_at = 'Exit'
                AND metric_name = 'Flow'
                AND start_date::DATE < '2036-01-01'
                GROUP BY {group_by_clause}
                HAVING SUM(metric_value) > 0
            ),
            latest_long_term AS (
                SELECT *
                FROM latest_long_term_raw
                WHERE start_date > (SELECT max_date FROM short_term_max_date)
            ),
            combined AS (
                SELECT * FROM latest_short_term
                UNION ALL
                SELECT * FROM latest_long_term
            ),
            filtered AS (
                SELECT * FROM combined WHERE {date_filter}
            )
            SELECT COALESCE(SUM(value) / 1000000, 0) as woodmac_total,
                   STRING_AGG(DISTINCT source, ' + ') as sources
            FROM filtered
            '''

            wm_result = pd.read_sql(wm_query, engine)
            woodmac_total = wm_result['woodmac_total'].iloc[0]
            woodmac_sources = wm_result['sources'].iloc[0] if wm_result['sources'].iloc[0] else 'N/A'

            difference = total_lng_volume - woodmac_total
            pct_diff = (difference / woodmac_total * 100) if woodmac_total > 0 else 0

            # Color coding based on percentage difference
            if abs(pct_diff) < 5:
                diff_color = '#27AE60'  # Green
            elif abs(pct_diff) < 10:
                diff_color = '#F39C12'  # Orange
            else:
                diff_color = '#E74C3C'  # Red

        except Exception as e:
            woodmac_total = 0
            woodmac_sources = 'Error'
            difference = 0
            pct_diff = 0
            diff_color = '#999'

        # Create validation summary with WoodMac comparison
        validation_summary = html.Div([
            html.Div([
                html.Strong("Total LNG Volume (Supply): ", style={'fontSize': '14px'}),
                html.Span(f"Regional: {total_lng_volume:,.2f} M m³", style={'fontSize': '14px', 'fontWeight': 'bold', 'color': '#D4AF37', 'marginRight': '15px'}),
                html.Span(" | ", style={'fontSize': '14px', 'color': '#999'}),
                html.Span(f"WoodMac ({woodmac_sources}): {woodmac_total:,.2f} M m³", style={'fontSize': '14px', 'fontWeight': 'bold', 'color': '#D4AF37', 'marginLeft': '15px', 'marginRight': '15px'}),
                html.Span(" | ", style={'fontSize': '14px', 'color': '#999'}),
                html.Span(f"Diff: {difference:+,.2f} M m³ ({pct_diff:+.1f}%)", style={'fontSize': '14px', 'fontWeight': 'bold', 'color': diff_color, 'marginLeft': '15px'})
            ], style={'padding': '10px', 'marginBottom': '10px', 'backgroundColor': '#FEF9E7', 'borderLeft': '4px solid #F7DC6F', 'borderRadius': '4px'})
        ])

        table = dash_table.DataTable(
            columns=columns,
            data=display_df.to_dict('records'),
            style_table={'overflowX': 'auto', 'width': '100%'},
            style_cell={
                'textAlign': 'center',
                'padding': '8px',
                'fontSize': '12px',
                'fontFamily': 'Segoe UI, -apple-system, BlinkMacSystemFont, sans-serif'
            },
            style_header={
                'backgroundColor': '#F7DC6F',
                'color': 'black',
                'fontWeight': 'bold',
                'fontSize': '12px',
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            page_size=15,
            export_format='xlsx'
        )

        return html.Div([validation_summary, table])

    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div(f"Error loading regional table: {str(e)}", style={'padding': '10px', 'color': 'red'})

############################################ Regional Trends Chart Callbacks ###################################################

# Demand Regional Laden Filter Options
@callback(
    Output('demand-regional-laden-origin-filter', 'options'),
    Output('demand-regional-laden-dest-filter', 'options'),
    Input('shipping-balance-regional-data-store', 'data'),
    prevent_initial_call=False
)
def update_demand_regional_laden_filters(regional_data):
    if regional_data is None:
        return [], []
    try:
        df_regional = pd.read_json(StringIO(regional_data), orient='split')
        origins = sorted(df_regional['origin_shipping_region'].unique())
        destinations = sorted(df_regional['destination_shipping_region'].unique())
        return [{'label': r, 'value': r} for r in origins], [{'label': r, 'value': r} for r in destinations]
    except:
        return [], []


# Demand Regional Ballast Filter Options
@callback(
    Output('demand-regional-ballast-origin-filter', 'options'),
    Output('demand-regional-ballast-dest-filter', 'options'),
    Input('shipping-balance-regional-data-store', 'data'),
    prevent_initial_call=False
)
def update_demand_regional_ballast_filters(regional_data):
    if regional_data is None:
        return [], []
    try:
        df_regional = pd.read_json(StringIO(regional_data), orient='split')
        origins = sorted(df_regional['origin_shipping_region'].unique())
        destinations = sorted(df_regional['destination_shipping_region'].unique())
        return [{'label': r, 'value': r} for r in origins], [{'label': r, 'value': r} for r in destinations]
    except:
        return [], []


# Supply Regional Laden Filter Options
@callback(
    Output('supply-regional-laden-origin-filter', 'options'),
    Output('supply-regional-laden-dest-filter', 'options'),
    Input('shipping-balance-supply-regional-data-store', 'data'),
    prevent_initial_call=False
)
def update_supply_regional_laden_filters(regional_data):
    if regional_data is None:
        return [], []
    try:
        df_regional = pd.read_json(StringIO(regional_data), orient='split')
        origins = sorted(df_regional['origin_shipping_region'].unique())
        destinations = sorted(df_regional['destination_shipping_region'].unique())
        return [{'label': r, 'value': r} for r in origins], [{'label': r, 'value': r} for r in destinations]
    except:
        return [], []


# Supply Regional Ballast Filter Options
@callback(
    Output('supply-regional-ballast-origin-filter', 'options'),
    Output('supply-regional-ballast-dest-filter', 'options'),
    Input('shipping-balance-supply-regional-data-store', 'data'),
    prevent_initial_call=False
)
def update_supply_regional_ballast_filters(regional_data):
    if regional_data is None:
        return [], []
    try:
        df_regional = pd.read_json(StringIO(regional_data), orient='split')
        origins = sorted(df_regional['origin_shipping_region'].unique())
        destinations = sorted(df_regional['destination_shipping_region'].unique())
        return [{'label': r, 'value': r} for r in origins], [{'label': r, 'value': r} for r in destinations]
    except:
        return [], []


# Demand Regional Laden Chart
@callback(
    Output('demand-regional-laden-chart', 'figure'),
    Input('shipping-balance-regional-data-store', 'data'),
    Input('demand-regional-laden-origin-filter', 'value'),
    Input('demand-regional-laden-dest-filter', 'value'),
    Input('dropdown-options-store', 'data'),
    prevent_initial_call=False
)
def update_demand_regional_laden_chart(regional_data, origin_filter, dest_filter, dropdown_options):
    if regional_data is None:
        return {}
    try:
        df = pd.read_json(StringIO(regional_data), orient='split')

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        if origin_filter:
            df = df[df['origin_shipping_region'].isin(origin_filter)]
        if dest_filter:
            df = df[df['destination_shipping_region'].isin(dest_filter)]
        if df.empty:
            return {'data': [], 'layout': {'title': 'No data'}}

        df['route'] = df['origin_shipping_region'] + ' → ' + df['destination_shipping_region']
        route_totals = df.groupby('route')['value'].sum().sort_values(ascending=False)
        df = df[df['route'].isin(route_totals.head(15).index)]
        df = df.sort_values('date')

        metric_col = 'median_laden_days'
        title = 'Laden Days by Route Over Time'

        fig = px.line(df, x='date', y=metric_col, color='route',
                     labels={'date': 'Period', metric_col: 'Laden Days', 'route': 'Route'},
                     title=f'{title} (Top 15 Routes)',
                     template=copy.deepcopy(pio.templates["plotly_white"]))
        fig.update_layout(hovermode='x unified',
                         legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                         margin=dict(r=200))
        fig.update_traces(hovertemplate='%{fullData.name}: %{y:.1f} days<extra></extra>')

        # Add vertical line for current period
        if dropdown_options:
            hist_date_max_str = dropdown_options.get('hist_date_max')
            if hist_date_max_str:
                hist_date_max = pd.to_datetime(hist_date_max_str)
                fig.add_shape(
                    type="line",
                    x0=hist_date_max,
                    x1=hist_date_max,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="red", width=2, dash="dash")
                )
                fig.add_annotation(
                    x=hist_date_max,
                    y=1,
                    yref="paper",
                    text="Current Period",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(color="red", size=10)
                )

        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'data': [], 'layout': {'title': f'Error: {e}'}}


# Demand Regional Ballast Chart
@callback(
    Output('demand-regional-ballast-chart', 'figure'),
    Input('shipping-balance-regional-data-store', 'data'),
    Input('demand-regional-ballast-origin-filter', 'value'),
    Input('demand-regional-ballast-dest-filter', 'value'),
    Input('dropdown-options-store', 'data'),
    prevent_initial_call=False
)
def update_demand_regional_ballast_chart(regional_data, origin_filter, dest_filter, dropdown_options):
    if regional_data is None:
        return {}
    try:
        df = pd.read_json(StringIO(regional_data), orient='split')
        if origin_filter:
            df = df[df['origin_shipping_region'].isin(origin_filter)]
        if dest_filter:
            df = df[df['destination_shipping_region'].isin(dest_filter)]
        if df.empty:
            return {'data': [], 'layout': {'title': 'No data'}}

        df['route'] = df['origin_shipping_region'] + ' → ' + df['destination_shipping_region']
        route_totals = df.groupby('route')['value'].sum().sort_values(ascending=False)
        df = df[df['route'].isin(route_totals.head(15).index)]
        df = df.sort_values('date')

        metric_col = 'median_nonladen_days'
        title = 'Ballast Days by Route Over Time'

        fig = px.line(df, x='date', y=metric_col, color='route',
                     labels={'date': 'Period', metric_col: 'Ballast Days', 'route': 'Route'},
                     title=f'{title} (Top 15 Routes)',
                     template=copy.deepcopy(pio.templates["plotly_white"]))
        fig.update_layout(hovermode='x unified',
                         legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                         margin=dict(r=200))
        fig.update_traces(hovertemplate='%{fullData.name}: %{y:.1f} days<extra></extra>')

        # Add vertical line for current period
        if dropdown_options:
            hist_date_max_str = dropdown_options.get('hist_date_max')
            if hist_date_max_str:
                hist_date_max = pd.to_datetime(hist_date_max_str)
                fig.add_shape(
                    type="line",
                    x0=hist_date_max,
                    x1=hist_date_max,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="red", width=2, dash="dash")
                )
                fig.add_annotation(
                    x=hist_date_max,
                    y=1,
                    yref="paper",
                    text="Current Period",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(color="red", size=10)
                )

        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'data': [], 'layout': {'title': f'Error: {e}'}}


# Supply Regional Laden Chart
@callback(
    Output('supply-regional-laden-chart', 'figure'),
    Input('shipping-balance-supply-regional-data-store', 'data'),
    Input('supply-regional-laden-origin-filter', 'value'),
    Input('supply-regional-laden-dest-filter', 'value'),
    Input('dropdown-options-store', 'data'),
    prevent_initial_call=False
)
def update_supply_regional_laden_chart(regional_data, origin_filter, dest_filter, dropdown_options):
    if regional_data is None:
        return {}
    try:
        df = pd.read_json(StringIO(regional_data), orient='split')

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        if origin_filter:
            df = df[df['origin_shipping_region'].isin(origin_filter)]
        if dest_filter:
            df = df[df['destination_shipping_region'].isin(dest_filter)]
        if df.empty:
            return {'data': [], 'layout': {'title': 'No data'}}

        df['route'] = df['origin_shipping_region'] + ' → ' + df['destination_shipping_region']
        route_totals = df.groupby('route')['value'].sum().sort_values(ascending=False)
        df = df[df['route'].isin(route_totals.head(15).index)]
        df = df.sort_values('date')

        metric_col = 'median_laden_days'
        title = 'Laden Days by Route Over Time'

        fig = px.line(df, x='date', y=metric_col, color='route',
                     labels={'date': 'Period', metric_col: 'Laden Days', 'route': 'Route'},
                     title=f'{title} (Top 15 Routes)',
                     template=copy.deepcopy(pio.templates["plotly_white"]))
        fig.update_layout(hovermode='x unified',
                         legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                         margin=dict(r=200))
        fig.update_traces(hovertemplate='%{fullData.name}: %{y:.1f} days<extra></extra>')

        # Add vertical line for current period
        if dropdown_options:
            hist_date_max_str = dropdown_options.get('hist_date_max')
            if hist_date_max_str:
                hist_date_max = pd.to_datetime(hist_date_max_str)
                fig.add_shape(
                    type="line",
                    x0=hist_date_max,
                    x1=hist_date_max,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="red", width=2, dash="dash")
                )
                fig.add_annotation(
                    x=hist_date_max,
                    y=1,
                    yref="paper",
                    text="Current Period",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(color="red", size=10)
                )

        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'data': [], 'layout': {'title': f'Error: {e}'}}


# Supply Regional Ballast Chart
@callback(
    Output('supply-regional-ballast-chart', 'figure'),
    Input('shipping-balance-supply-regional-data-store', 'data'),
    Input('supply-regional-ballast-origin-filter', 'value'),
    Input('supply-regional-ballast-dest-filter', 'value'),
    Input('dropdown-options-store', 'data'),
    prevent_initial_call=False
)
def update_supply_regional_ballast_chart(regional_data, origin_filter, dest_filter, dropdown_options):
    if regional_data is None:
        return {}
    try:
        df = pd.read_json(StringIO(regional_data), orient='split')

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        if origin_filter:
            df = df[df['origin_shipping_region'].isin(origin_filter)]
        if dest_filter:
            df = df[df['destination_shipping_region'].isin(dest_filter)]
        if df.empty:
            return {'data': [], 'layout': {'title': 'No data'}}

        df['route'] = df['origin_shipping_region'] + ' → ' + df['destination_shipping_region']
        route_totals = df.groupby('route')['value'].sum().sort_values(ascending=False)
        df = df[df['route'].isin(route_totals.head(15).index)]
        df = df.sort_values('date')

        metric_col = 'median_nonladen_days'
        title = 'Ballast Days by Route Over Time'

        fig = px.line(df, x='date', y=metric_col, color='route',
                     labels={'date': 'Period', metric_col: 'Ballast Days', 'route': 'Route'},
                     title=f'{title} (Top 15 Routes)',
                     template=copy.deepcopy(pio.templates["plotly_white"]))
        fig.update_layout(hovermode='x unified',
                         legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                         margin=dict(r=200))
        fig.update_traces(hovertemplate='%{fullData.name}: %{y:.1f} days<extra></extra>')

        # Add vertical line for current period
        if dropdown_options:
            hist_date_max_str = dropdown_options.get('hist_date_max')
            if hist_date_max_str:
                hist_date_max = pd.to_datetime(hist_date_max_str)
                fig.add_shape(
                    type="line",
                    x0=hist_date_max,
                    x1=hist_date_max,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="red", width=2, dash="dash")
                )
                fig.add_annotation(
                    x=hist_date_max,
                    y=1,
                    yref="paper",
                    text="Current Period",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(color="red", size=10)
                )

        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'data': [], 'layout': {'title': f'Error: {e}'}}
