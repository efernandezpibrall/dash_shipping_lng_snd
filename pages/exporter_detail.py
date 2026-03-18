from dash import html, dcc, dash_table, callback, Output, Input, State, ALL, ctx, no_update
from dash.dash_table.Format import Format, Group, Scheme
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime as dt
import calendar
import json
from io import BytesIO, StringIO
from dash.exceptions import PreventUpdate
import traceback

import configparser
import os
from sqlalchemy import create_engine

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

# create engine
engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)


# Desired vessel order (can be defined globally or inside the function)
DESIRED_VESSEL_ORDER = ['XS (Pressure Gas)',
                        'S (Small Scale)',
                        'M (Med Max)',
                        'L (Lower Conventional)',
                        'XL (Upper Conventional)',
                        'Q-Flex',
                        'Q-Max']

MCM_PER_CUBIC_METER = 0.6 / 1000
WOODMAC_IMPORT_EXPORTS_TABLE = 'at_lng.woodmac_gas_imports_exports_monthly__mmtpa'
WOODMAC_LNG_CUBIC_METERS_PER_MMTPA_MONTH = 2222 * 1000 / 12
WOODMAC_FORECAST_YEARS_AHEAD = 2
SUPPLY_ALLOCATION_RUNS_TABLE = f'{DB_SCHEMA}.fundamentals_supply_allocation_runs'
SUPPLY_ALLOCATION_COUNTRY_FLOWS_TABLE = f'{DB_SCHEMA}.fundamentals_supply_allocation_country_flows'
SUPPLY_ALLOCATION_DEMAND_DETAIL_TABLE = f'{DB_SCHEMA}.fundamentals_supply_allocation_demand_detail'
SUPPLY_ALLOCATION_DEMAND_SUMMARY_TABLE = f'{DB_SCHEMA}.fundamentals_supply_allocation_demand_summary'

# ========================================
# PROFESSIONAL CHART STYLING CONFIGURATION
# ========================================

# McKinsey Professional Color Palette
PROFESSIONAL_COLORS = {
    'primary': '#2E86C1',           # McKinsey blue - primary brand color
    'primary_dark': '#1B4F72',      # Darker McKinsey blue
    'primary_light': '#5DADE2',     # Lighter McKinsey blue
    'secondary': '#E8F4FD',         # Very light blue background
    'text_primary': '#1f2937',      # Dark gray for text
    'text_secondary': '#374151',    # Medium gray for secondary text
    'text_tertiary': '#6b7280',     # Light gray for tertiary text
    'bg_white': '#ffffff',          # Pure white background
    'bg_light': '#f8f9fa',          # Light background
    'grid_color': '#e5e7eb',        # Light grid color
    'success': '#22c55e',           # Success green
    'warning': '#f59e0b',           # Warning orange
    'danger': '#ef4444',            # Danger red
}

# Professional qualitative color palette for multiple series
PROFESSIONAL_CHART_COLORS = [
    '#2E86C1',  # McKinsey blue
    '#22c55e',  # Success green
    '#f59e0b',  # Warning orange
    '#ef4444',  # Danger red
    '#8b5cf6',  # Purple
    '#06b6d4',  # Cyan
    '#84cc16',  # Lime
    '#f97316',  # Orange
    '#ec4899',  # Pink
    '#6366f1',  # Indigo
    '#10b981',  # Emerald
    '#f43f5e',  # Rose
]

def apply_professional_chart_styling(fig, title="", height=600, show_legend=True, legend_title=""):
    """
    Apply consistent professional styling to Plotly charts following McKinsey design standards.
    
    Args:
        fig: Plotly figure object
        title: Chart title
        height: Chart height in pixels
        show_legend: Whether to show legend
        legend_title: Legend title text
    
    Returns:
        Updated figure with professional styling
    """
    
    fig.update_layout(
        # Typography and title styling
        title=dict(
            text=title,
            font=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                size=18,
                color=PROFESSIONAL_COLORS['text_primary']
            ),
            x=0.02,  # Left-align title
            xanchor='left',
            pad=dict(t=20, b=20)
        ),
        
        # Background and layout
        paper_bgcolor=PROFESSIONAL_COLORS['bg_white'],
        plot_bgcolor=PROFESSIONAL_COLORS['bg_white'],
        
        # Font styling
        font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=12,
            color=PROFESSIONAL_COLORS['text_secondary']
        ),
        
        # Legend styling
        legend=dict(
            title=dict(
                text=legend_title,
                font=dict(
                    family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                    size=13,
                    color=PROFESSIONAL_COLORS['text_primary']
                )
            ),
            font=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                size=11,
                color=PROFESSIONAL_COLORS['text_secondary']
            ),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=PROFESSIONAL_COLORS['grid_color'],
            borderwidth=1,
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        
        # Height and margins
        height=height,
        margin=dict(l=60, r=200, t=80, b=60),
        
        # Hover styling
        hoverlabel=dict(
            bgcolor=PROFESSIONAL_COLORS['bg_white'],
            bordercolor=PROFESSIONAL_COLORS['primary'],
            font=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                size=11,
                color=PROFESSIONAL_COLORS['text_primary']
            )
        )
    )
    
    # Update x-axis styling
    fig.update_xaxes(
        title_font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=13,
            color=PROFESSIONAL_COLORS['text_primary']
        ),
        tickfont=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=11,
            color=PROFESSIONAL_COLORS['text_secondary']
        ),
        gridcolor=PROFESSIONAL_COLORS['grid_color'],
        gridwidth=0.5,
        linecolor=PROFESSIONAL_COLORS['grid_color'],
        linewidth=1,
        showgrid=True,
        zeroline=False
    )
    
    # Update y-axis styling
    fig.update_yaxes(
        title_font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=13,
            color=PROFESSIONAL_COLORS['text_primary']
        ),
        tickfont=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=11,
            color=PROFESSIONAL_COLORS['text_secondary']
        ),
        gridcolor=PROFESSIONAL_COLORS['grid_color'],
        gridwidth=0.5,
        linecolor=PROFESSIONAL_COLORS['grid_color'],
        linewidth=1,
        showgrid=True,
        zeroline=False
    )
    
    # Hide legend if requested
    if not show_legend:
        fig.update_layout(showlegend=False)
    
    return fig

def get_professional_colors(n_colors):
    """
    Get n professional colors, cycling through the palette if needed.
    
    Args:
        n_colors: Number of colors needed
        
    Returns:
        List of professional colors
    """
    colors = []
    for i in range(n_colors):
        colors.append(PROFESSIONAL_CHART_COLORS[i % len(PROFESSIONAL_CHART_COLORS)])
    return colors


def normalize_rolling_window_days(window_days, default=30):
    """Ensure the rolling window input is always a positive integer."""
    try:
        normalized_window_days = int(window_days)
        return normalized_window_days if normalized_window_days > 0 else default
    except (TypeError, ValueError):
        return default


def calculate_period_days(row, aggregation_level):
    """Return the number of calendar days represented by the selected aggregation bucket."""
    year = row.get('year')
    if pd.isna(year):
        return np.nan

    year = int(year)

    if aggregation_level == 'Year':
        return 366 if calendar.isleap(year) else 365

    if aggregation_level == 'Year+Quarter':
        quarter = str(row.get('quarter', ''))
        try:
            quarter_num = int(quarter.replace('Q', ''))
        except ValueError:
            return np.nan
        quarter_months = {
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, 8, 9],
            4: [10, 11, 12],
        }.get(quarter_num)
        if not quarter_months:
            return np.nan
        return sum(calendar.monthrange(year, month)[1] for month in quarter_months)

    if aggregation_level == 'Month':
        month = row.get('month')
        if pd.isna(month):
            return np.nan
        return calendar.monthrange(year, int(month))[1]

    if aggregation_level == 'Week':
        return 7

    if aggregation_level == 'Year+Season':
        season = row.get('season')
        if season == 'W':
            season_months = [1, 2, 3, 10, 11, 12]
        elif season == 'S':
            season_months = [4, 5, 6, 7, 8, 9]
        else:
            return np.nan
        return sum(calendar.monthrange(year, month)[1] for month in season_months)

    return np.nan


def convert_trade_analysis_volume_metric(df, value_column, aggregation_level, metric_name):
    """Convert LNG cargo totals into the display metric used by trade analysis charts/tables."""
    if df.empty or value_column not in df.columns or metric_name not in {'mcm_d', 'mtpa'}:
        return df

    converted_df = df.copy()
    period_days = converted_df.apply(
        lambda row: calculate_period_days(row, aggregation_level),
        axis=1
    ).astype(float)

    valid_days = period_days > 0
    converted_df[value_column] = converted_df[value_column].astype(float)

    if metric_name == 'mcm_d':
        converted_df.loc[valid_days, value_column] = (
            converted_df.loc[valid_days, value_column] * 0.6 / 1000 / period_days[valid_days]
        )
    elif metric_name == 'mtpa':
        converted_df.loc[valid_days, value_column] = (
            converted_df.loc[valid_days, value_column] * 0.45 / 1_000_000 * 365.25 / period_days[valid_days]
        )

    converted_df.loc[~valid_days, value_column] = np.nan
    return converted_df


def format_rolling_window_label(window_days):
    return f"{normalize_rolling_window_days(window_days)}D"


def format_rolling_window_title(window_days):
    normalized_window_days = normalize_rolling_window_days(window_days)
    return f"{normalized_window_days}-Day Rolling Average"


def process_trade_and_distance_data(engine,origin_country_name):
    """
    Loads trade and distance data from a database, joins them, calculates mileage ratios,
    determines the most likely route based on ratios, and flags deviations.

    Args:
        neon_db_url (str): The database connection URL string.
        trades_table_name (str): The name of the database table containing trade data.
        distance_table_name (str): The name of the database table containing distance data.

    Returns:
        pandas.DataFrame: A DataFrame containing the joined and processed data,
                          including calculated ratios, selected routes, and flags.
                          Returns None if a critical error occurs during data loading.
    """
    trades_table_name = "kpler_trades"
    distance_table_name = "kpler_distance_matrix"

    final_df = None  # Initialize final_df to None

    try:
        # 1. Database Connection Created

        # 2. Load Initial Trades Data
        try:
            trades_df = pd.read_sql(f'''SELECT voyage_id, vessel_name, start, "end", origin_location_name,origin_country_name, zone_origin_name,
                                            destination_location_name, zone_destination_name, destination_country_name, mileage_nautical_miles,
                                            origin_reload_sts_partial, destination_reload_sts_partial
                                        FROM {DB_SCHEMA}.{trades_table_name}
                                        WHERE status='Delivered'
                                        and origin_country_name='{origin_country_name}'
                                        and zone_origin_name<>zone_destination_name
                                        and upload_timestamp_utc = (select max(upload_timestamp_utc) from {DB_SCHEMA}.{trades_table_name})
                                        ''', engine)
        except Exception as e:
            return None  # Cannot proceed without trades data

        # 3. Load Initial Distance Data
        try:
            distance_df = pd.read_sql(f'''SELECT "originLocationName", "destinationLocationName", "distanceDirect", "distanceViaSuez", "distanceViaPanama"
                                          FROM {DB_SCHEMA}.{distance_table_name}''', engine)
            # Standardize column names from DB
            distance_df.columns = ['originLocationName', 'destinationLocationName', 'distanceDirect', 'distanceViaSuez',
                                   'distanceViaPanama']
        except Exception as e:
            # Create an empty DataFrame with expected columns if table doesn't exist or is empty
            distance_df = pd.DataFrame(
                columns=['originLocationName', 'destinationLocationName', 'distanceDirect', 'distanceViaSuez',
                         'distanceViaPanama'])

        # --- Sections for identifying missing origins, fetching, and reloading are removed ---

        # 4. Join DataFrames
        # Perform the merge using specified columns
        final_df = pd.merge(
            trades_df,
            distance_df,
            how='left',
            left_on=['zone_origin_name', 'zone_destination_name'],
            right_on=['originLocationName', 'destinationLocationName']
        )

        # Extract date components
        final_df['year'] = final_df['end'].dt.year
        final_df['month'] = final_df['end'].dt.month

        # Determine season and quarter based on end month
        final_df['season'] = np.where(final_df['month'].isin([10, 11, 12, 1, 2, 3]), 'W', 'S')
        final_df['quarter'] = 'Q' + final_df['end'].dt.quarter.astype(str)


        # 5. Calculate Ratios

        # Ensure columns are numeric, converting non-numeric values to NaN
        final_df['mileage_nautical_miles'] = pd.to_numeric(final_df['mileage_nautical_miles'], errors='coerce')
        final_df['distanceDirect'] = pd.to_numeric(final_df['distanceDirect'], errors='coerce')
        final_df['distanceViaSuez'] = pd.to_numeric(final_df['distanceViaSuez'], errors='coerce')
        final_df['distanceViaPanama'] = pd.to_numeric(final_df['distanceViaPanama'], errors='coerce')

        # Calculate ratios, division by zero or NaN will result in inf or NaN
        final_df['ratio_miles_distancedirect'] = final_df['mileage_nautical_miles'] / final_df['distanceDirect']
        final_df['ratio_miles_distanceviasuez'] = final_df['mileage_nautical_miles'] / final_df['distanceViaSuez']
        final_df['ratio_miles_distanceviapanama'] = final_df['mileage_nautical_miles'] / final_df['distanceViaPanama']

        # Replace inf/-inf with NaN if desired
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 6. Calculate Absolute Difference from 1
        final_df['diff_direct'] = (final_df['ratio_miles_distancedirect'] - 1).abs()
        final_df['diff_suez'] = (final_df['ratio_miles_distanceviasuez'] - 1).abs()
        final_df['diff_panama'] = (final_df['ratio_miles_distanceviapanama'] - 1).abs()

        # 7. Identify Minimum Difference and Determine Selected Route
        diff_cols = ['diff_direct', 'diff_suez', 'diff_panama']
        # Only call idxmin on rows with at least one non-NaN diff to avoid FutureWarning
        has_valid = final_df[diff_cols].notna().any(axis=1)
        final_df['closest_route_col'] = pd.NA
        final_df.loc[has_valid, 'closest_route_col'] = (
            final_df.loc[has_valid, diff_cols].idxmin(axis=1, skipna=True)
        )

        # Map the column name back to a simpler route name
        route_map = {
            'diff_direct': 'Direct',
            'diff_suez': 'ViaSuez',
            'diff_panama': 'ViaPanama'
        }
        final_df['selected_route'] = final_df['closest_route_col'].map(route_map)
        # Where idxmin returns NaN (all diffs were NaN), selected_route will be NaN

        # 8. Check Closeness and Create Indicator Flag
        closeness_tolerance = 0.2  # Example tolerance (20%)
        lower_bound = 1 - closeness_tolerance
        upper_bound = 1 + closeness_tolerance

        # Check if each ratio is within the bounds (result is False if ratio is NaN)
        is_direct_close = final_df['ratio_miles_distancedirect'].between(lower_bound, upper_bound, inclusive='both')
        is_suez_close = final_df['ratio_miles_distanceviasuez'].between(lower_bound, upper_bound, inclusive='both')
        is_panama_close = final_df['ratio_miles_distanceviapanama'].between(lower_bound, upper_bound, inclusive='both')

        # Combine: True if *any* ratio is close
        any_ratio_is_close = is_direct_close | is_suez_close | is_panama_close

        # Indicator: True if *none* of the ratios were close
        final_df['no_ratio_close_to_1'] = ~any_ratio_is_close



        missing_distance_count = final_df['distanceDirect'].isnull().sum()



    except Exception as e:
        # Optionally re-raise the exception if the caller should handle it
        # raise e
        return None  # Return None to indicate failure

    finally:
        # Clean up database connection if engine was created
        if 'engine' in locals() and engine:
            engine.dispose()

    return final_df



def kpler_analysis(engine,
                   destination_level='destination_shipping_region',
                   origin_country='United States'):
    """
    Fetches Kpler trade data, calculates non-laden voyages, adds region mappings,
    and aggregates metrics by region pair, vessel type, year, season, quarter, and status.

    Args:
        engine: SQLAlchemy engine object for database connection.
        destination_level (str): Level of destination aggregation: 'destination_shipping_region' or 'destination_country_name'.

    Returns:
        pd.DataFrame: Aggregated DataFrame with trade metrics.
    """
    # Fetch laden trades and map vessel type based on capacity
    query_trades = f'''
        SELECT
            a.vessel_name,
            b.vessel_type,
            a."start",
            a.origin_country_name,
            a.zone_origin_name,
            a.origin_location_name,
            a."end",
            a.destination_country_name,
            a.continent_destination_name,
            a.destination_location_name,
            a.zone_destination_name,
            a.status,
            a.vessel_capacity_cubic_meters,
            a.cargo_origin_cubic_meters,
            a.cargo_destination_cubic_meters,
            a.mileage_nautical_miles,
            c."distanceDirect",
            c."distanceViaSuez",
            c."distanceViaPanama",
            a.ton_miles,
            a.origin_reload_sts_partial,
            a.destination_reload_sts_partial,
            a.upload_timestamp_utc
        FROM {DB_SCHEMA}.kpler_trades a
        LEFT JOIN {DB_SCHEMA}.mapping_vessel_type_capacity b
            ON a.vessel_capacity_cubic_meters >= b.capacity_cubic_meters_min
          AND a.vessel_capacity_cubic_meters < b.capacity_cubic_meters_max
        LEFT JOIN {DB_SCHEMA}.kpler_distance_matrix c
            ON a.zone_origin_name = c."originLocationName"
            AND a.zone_destination_name = c."destinationLocationName"
        WHERE a.upload_timestamp_utc = (SELECT MAX(upload_timestamp_utc) FROM {DB_SCHEMA}.kpler_trades)
          AND a.destination_country_name IS NOT NULL
          AND a.origin_country_name='{origin_country}'
          AND a.destination_country_name != '{origin_country}'
          AND a."end" IS NOT NULL
          AND a."start" IS NOT NULL
          AND status = 'Delivered'
    '''
    df_trades = pd.read_sql(query_trades, engine)

    # --- Infer Non-Laden Voyages ---
    # Sort to easily find previous voyages for each vessel
    df_trades = df_trades.sort_values(['vessel_name', 'start', 'end']).reset_index(drop=True)

    # Get previous voyage details using shift within each vessel group
    df_trades['prev_end'] = df_trades.groupby('vessel_name')['end'].shift(1)
    df_trades['prev_dest_country'] = df_trades.groupby('vessel_name')['destination_country_name'].shift(1)
    df_trades['prev_dest_location'] = df_trades.groupby('vessel_name')['destination_location_name'].shift(1)
    df_trades['prev_vessel_capacity'] = df_trades.groupby('vessel_name')['vessel_capacity_cubic_meters'].shift(1)

    # Create non-laden rows where a previous voyage exists
    mask = df_trades['prev_end'].notna()
    new_rows = pd.DataFrame({
        'vessel_name': df_trades.loc[mask, 'vessel_name'],
        'vessel_type': df_trades.loc[mask, 'vessel_type'],  # Assume type doesn't change between legs
        'start': df_trades.loc[mask, 'prev_end'],
        'origin_country_name': df_trades.loc[mask, 'prev_dest_country'],
        'origin_location_name': df_trades.loc[mask, 'prev_dest_location'],
        'end': df_trades.loc[mask, 'start'],
        'destination_country_name': df_trades.loc[mask, 'origin_country_name'],
        'destination_location_name': df_trades.loc[mask, 'origin_location_name'],
        'status': 'non_laden',
        'vessel_capacity_cubic_meters': df_trades.loc[mask, 'prev_vessel_capacity'],
        'cargo_origin_cubic_meters': 0,
        'cargo_destination_cubic_meters': 0,
        'upload_timestamp_utc': df_trades.loc[mask, 'upload_timestamp_utc']
        # mileage/ton_miles for non-laden are typically NaN or estimated separately if needed
    })

    # Combine laden and inferred non-laden voyages
    df_trades = pd.concat([df_trades, new_rows], ignore_index=True)

    # Clean up temporary columns used for non-laden inference
    df_trades = df_trades.drop(columns=[
        'prev_end', 'prev_dest_country', 'prev_dest_location', 'prev_vessel_capacity'
    ])

    # Re-sort for chronological order per vessel
    df_trades = df_trades.sort_values(['vessel_name', 'start']).reset_index(drop=True)

    # --- Feature Engineering ---
    df_trades['delivery_days'] = (df_trades['end'] - df_trades['start']).dt.days
    # Filter out voyages with non-positive duration (likely data issues or same-day)
    # Use .copy() to avoid potential SettingWithCopyWarning later
    df_trades = df_trades[df_trades['delivery_days'] > 0].copy()

    # Calculate speed safely, avoiding division by zero
    df_trades['speed'] = np.nan  # Initialize column
    valid_speed_mask = (df_trades['mileage_nautical_miles'].notna()) & (df_trades['delivery_days'] > 0)
    df_trades.loc[valid_speed_mask, 'speed'] = (
            df_trades.loc[valid_speed_mask, 'mileage_nautical_miles'] /
            df_trades.loc[valid_speed_mask, 'delivery_days'] / 24
        # Convert days to hours for speed (e.g., NM/hour) - adjust if unit is NM/day
    )

    # Extract date components
    df_trades['year'] = df_trades['end'].dt.year
    df_trades['month'] = df_trades['end'].dt.month
    df_trades['week'] = df_trades['end'].dt.isocalendar().week

    # Standardize status: ensure it's either 'laden' or 'non_laden'
    df_trades['status'] = np.where(df_trades['status'] != 'non_laden', 'laden', 'non_laden')

    # Determine season and quarter based on end month
    df_trades['season'] = np.where(df_trades['month'].isin([10, 11, 12, 1, 2, 3]), 'W', 'S')
    df_trades['quarter'] = 'Q' + df_trades['end'].dt.quarter.astype(str)

    # Calculate utilization rate safely, handling division by zero or zero capacity
    df_trades['utilization_rate'] = np.nan  # Initialize
    valid_util_mask = (df_trades['vessel_capacity_cubic_meters'].notna()) & (
            df_trades['vessel_capacity_cubic_meters'] > 0)
    df_trades.loc[valid_util_mask, 'utilization_rate'] = (
            df_trades.loc[valid_util_mask, 'cargo_destination_cubic_meters'] /
            df_trades.loc[valid_util_mask, 'vessel_capacity_cubic_meters']
    )

    # --- Add Shipping Region Classification ---
    query_regions = f'''
        SELECT DISTINCT country, shipping_region, basin, subcontinent, country_classification_level1, country_classification
        FROM {DB_SCHEMA}.mappings_country
    '''
    df_mapping_country = pd.read_sql(query_regions, engine)

    # Merge origin regions
    df_trades = pd.merge(
        df_trades,
        df_mapping_country.rename(
            columns={'country': 'origin_country_name', 'shipping_region': 'origin_shipping_region'}),
        how='left',
        on='origin_country_name'
    )
    # Merge destination regions, basin and classification columns
    df_trades = pd.merge(
        df_trades,
        df_mapping_country.rename(columns={
            'country': 'destination_country_name',
            'shipping_region': 'destination_shipping_region',
            'basin': 'destination_basin',
            'subcontinent': 'destination_subcontinent',
            'country_classification_level1': 'destination_classification_level1',
            'country_classification': 'destination_classification',
        }),
        how='left',
        on='destination_country_name'
    )

    # --- Final Aggregation ---
    # Determine grouping columns based on destination_level parameter
    group_columns = [
        'vessel_type', 'status',
        'year', 'season', 'quarter', 'month', 'week',
        'origin_country_name'
    ]

    # Add appropriate destination column based on parameter
    if destination_level == 'destination_country_name':
        group_columns.append('destination_country_name')
    elif destination_level == 'destination_basin':
        group_columns.append('destination_basin')
    elif destination_level == 'continent_destination_name':
        group_columns.append('continent_destination_name')
    elif destination_level == 'destination_subcontinent':
        group_columns.append('destination_subcontinent')
    elif destination_level == 'destination_classification_level1':
        group_columns.append('destination_classification_level1')
    elif destination_level == 'destination_classification':
        group_columns.append('destination_classification')
    else:
        # Default to shipping region grouping
        group_columns.append('destination_shipping_region')

    # Group by desired dimensions and calculate metrics
    # Use observed=False to keep all potential category combinations
    # Use dropna=False to avoid dropping groups with NaN keys (e.g., missing regions)
    df_trades_shipping_region = df_trades.groupby(
        group_columns, observed=False, dropna=False
    ).agg(
        median_delivery_days=('delivery_days', 'median'),
        median_mileage_nautical_miles=('mileage_nautical_miles', 'median'),
        median_ton_miles=('ton_miles', 'median'),
        median_speed=('speed', 'median'),
        median_utilization_rate=('utilization_rate', 'median'),
        median_cargo_destination_cubic_meters=('cargo_destination_cubic_meters', 'median'),
        median_vessel_capacity_cubic_meters=('vessel_capacity_cubic_meters', 'median'),
        sum_ton_miles=('ton_miles', 'sum'),
        sum_cargo_destination_cubic_meters=('cargo_destination_cubic_meters', 'sum'),
        # Count non-null vessel_name as a reliable way to count trades/legs
        count_trades=('vessel_name', 'count')
    ).reset_index()


    return df_trades_shipping_region


def prepare_pivot_table(df, values_col, filters, aggregation_level='Year', add_total_column=False, aggfunc='sum',
                        destination_level='destination_shipping_region'):
    """
    Prepare data pivoted by destination region/country for the tables, with flexible time aggregation.

    Args:
        df (pd.DataFrame): Input DataFrame (pre-aggregated by year, season, quarter, status, vessel, destination).
        values_col (str): The column with values for the table cells.
        filters (dict): Dictionary of filters {'status': ..., 'vessel_type': ...}.
        aggregation_level (str): How to aggregate time ('Year', 'Year+Season', 'Year+Quarter').
        add_total_column (bool): Whether to add a 'Total' column summing regions.
        aggfunc (str): Aggregation function for pivoting (usually 'sum' or 'mean').
        destination_level (str): Column to use for destination ('destination_shipping_region' or 'destination_country_name')

    Returns:
        pd.DataFrame: Pivoted data ready for DataTable, or empty DataFrame on error/no data.
    """
    empty_df = pd.DataFrame()
    if df is None or df.empty:
        return empty_df

    # --- Determine Index Columns based on Aggregation Level ---
    if aggregation_level == 'Year':
        index_cols = ['year']
    elif aggregation_level == 'Year+Season':
        index_cols = ['year', 'season']
    elif aggregation_level == 'Year+Quarter':
        index_cols = ['year', 'quarter']
    elif aggregation_level == 'Month':
        index_cols = ['year', 'month']
    elif aggregation_level == 'Week':
        index_cols = ['year', 'week']
    else:
        index_cols = ['year']

    # --- Essential Column Checks ---
    required_input_cols = [destination_level, values_col] + list(filters.keys()) + index_cols
    missing_input = [col for col in required_input_cols if col not in df.columns]
    if missing_input:
        return empty_df

    filtered_df = df.copy()

    # --- Apply Filters (Status, Vessel Type) ---
    for col, value in filters.items():
        if value is not None:
            if col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col] == value]
            else:
                return empty_df

    if filtered_df.empty:
        return empty_df

    # --- Group before Pivoting (Ensure unique index/column combinations) ---
    # This step aggregates metrics if there are multiple rows for the same
    # filter criteria and desired pivot index/columns (e.g., if original data wasn't fully unique)
    grouping_cols = index_cols + [destination_level]  # Use the specified destination level column
    if not all(col in filtered_df.columns for col in grouping_cols):
        return empty_df

    # Perform the aggregation using the specified aggfunc for the values_col
    # Keep only necessary columns for pivoting
    agg_spec = {values_col: aggfunc}
    try:
        grouped_df = filtered_df.groupby(grouping_cols, observed=False, dropna=False).agg(agg_spec).reset_index()
    except Exception as e:
        return empty_df

    if grouped_df.empty:
        return empty_df

    # --- Pivot Data ---
    try:
        pivot_df = grouped_df.pivot_table(
            index=index_cols,
            columns=destination_level,  # Use the specified destination level
            values=values_col,
            aggfunc='first',  # Use 'first' as data is already aggregated by the groupby above
            fill_value=np.nan  # Fill missing region/time combinations with NaN
        )

    except Exception as e:
        return empty_df

    # --- Sort Index (Year, Season/Quarter) ---
    if not pivot_df.empty:
        pivot_df = pivot_df.sort_index()

    # --- Add Total Column (summing across regions for each time period) ---
    if add_total_column and not pivot_df.empty:
        region_cols = pivot_df.columns.tolist()
        if region_cols:  # Ensure there are region columns to sum
            try:
                pivot_df['Total'] = pivot_df[region_cols].sum(axis=1, skipna=True)
            except Exception as e:
                # Proceed without total column if calculation fails
                pass

    # --- Reset Index to make Year/Season/Quarter regular columns ---
    if not pivot_df.empty:
        pivot_df = pivot_df.reset_index()

    return pivot_df


def create_stacked_bar_chart(df, metric, title_suffix, selected_status=None, selected_vessel_type=None,
                             aggregation_level='Year', is_intracountry=False,
                             destination_level='destination_shipping_region'):
    """
    Create a Plotly visualization showing data by selected aggregation level and shipping regions/countries.
    Now supports configurable destination level (region or country).
    Args:
        df: DataFrame with trade data
        metric: The column name to sum and visualize
        title_suffix: Text to use in the title describing the metric
        selected_status: String status value ('laden' or 'non_laden')
        selected_vessel_type: String vessel type or 'All' to include all types
        aggregation_level: How to aggregate time ('Year', 'Year+Season', 'Year+Quarter')
        is_intracountry: Whether this is for intracountry data
        destination_level: Column to use for destination grouping ('destination_shipping_region' or 'destination_country_name')
    Returns:
        A Plotly figure object
    """

    # Filter the data
    filtered_df = df.copy()

    # Apply status filter directly
    if selected_status and 'status' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['status'] == selected_status]
    elif 'status' not in filtered_df.columns:
        pass
    # Apply vessel type filter
    if selected_vessel_type and selected_vessel_type != 'All' and 'vessel_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['vessel_type'] == selected_vessel_type]
    elif 'vessel_type' not in filtered_df.columns:
        pass

    if filtered_df.empty:
        # Return an empty figure with a message if no data after filtering
        fig = go.Figure()
        fig.update_layout(
            title=f"No data available for {title_suffix} with selected filters",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig

    # Determine grouping columns based on aggregation level
    if aggregation_level == 'Year':
        groupby_time_cols = ['year']
        x_axis_title = 'Year'
    elif aggregation_level == 'Year+Season':
        groupby_time_cols = ['year', 'season']
        x_axis_title = 'Year-Season'
    elif aggregation_level == 'Year+Quarter':
        groupby_time_cols = ['year', 'quarter']
        x_axis_title = 'Year-Quarter'
    else:
        groupby_time_cols = ['year']
        x_axis_title = 'Year'
    # Set grouping field based on data type
    if is_intracountry:
        # Ensure the necessary column exists
        if 'origin_country_name' not in filtered_df.columns:
            fig = go.Figure()
            fig.update_layout(
                title="Error: Missing 'origin_country_name' column",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            return fig
        group_field = 'origin_country_name'
        # Update chart title to include vessel type and aggregation level
        vessel_type_text = f", {selected_vessel_type}" if selected_vessel_type and selected_vessel_type != 'All' else ""
        chart_title = f'Intracountry {title_suffix} by {x_axis_title}{vessel_type_text} and Origin Country'
        legend_title = 'Origin Country'
    else:
        # Ensure the necessary columns exist before creating the combined field
        if destination_level not in filtered_df.columns:
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: Missing {destination_level} column",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            return fig

        group_field = destination_level
        # Determine label for the destination level in chart title
        destination_labels = {
            'destination_country_name': 'Countries',
            'destination_shipping_region': 'Shipping Regions',
            'destination_basin': 'Basins',
            'continent_destination_name': 'Continents',
            'destination_subcontinent': 'Subcontinents',
            'destination_classification_level1': 'Classifications Level 1',
            'destination_classification': 'Classifications',
        }
        legend_labels = {
            'destination_country_name': 'Country',
            'destination_shipping_region': 'Shipping Region',
            'destination_basin': 'Basin',
            'continent_destination_name': 'Continent',
            'destination_subcontinent': 'Subcontinent',
            'destination_classification_level1': 'Classification Level 1',
            'destination_classification': 'Classification',
        }
        destination_label = destination_labels.get(destination_level, 'Regions')
        vessel_type_text = f", {selected_vessel_type}" if selected_vessel_type and selected_vessel_type != 'All' else ""
        chart_title = f'{title_suffix} by {x_axis_title}{vessel_type_text} and Destination {destination_label}'
        legend_title = 'Destination ' + legend_labels.get(destination_level, 'Region')

    # Check required columns
    all_groupby_cols = groupby_time_cols + [group_field]
    missing_cols = [col for col in all_groupby_cols if col not in filtered_df.columns]
    if missing_cols:
        fig = go.Figure()
        fig.update_layout(
            title=f"Error: Missing columns: {', '.join(missing_cols)}",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    # Ensure metric column exists
    if metric not in filtered_df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=f"Error: Metric '{metric}' not found",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    try:
        # Aggregate data by time and destination region/country
        stacked_data = filtered_df.groupby(all_groupby_cols, observed=False)[metric].sum().reset_index()
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            title=f"Error during data aggregation: {str(e)}",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig

    if stacked_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No aggregated data for {title_suffix}",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig

    # Create x-axis labels based on aggregation level
    if len(groupby_time_cols) > 1:
        # Create combined time labels for multi-level time aggregation
        if 'year' in stacked_data.columns:
            if 'season' in stacked_data.columns:
                stacked_data['time_label'] = stacked_data['year'].astype(str) + '-' + stacked_data['season'].astype(str)
            elif 'quarter' in stacked_data.columns:
                stacked_data['time_label'] = stacked_data['year'].astype(str) + '-' + stacked_data['quarter'].astype(
                    str)
            else:
                stacked_data['time_label'] = stacked_data['year'].astype(str)
        else:
            # Fallback if expected columns aren't present
            stacked_data['time_label'] = 'Unknown'
    else:
        # For single level (just year), use year directly
        stacked_data['time_label'] = stacked_data['year'].astype(str)

    # Get unique values
    time_labels = sorted(stacked_data['time_label'].unique())
    group_values = sorted(stacked_data[group_field].unique())

    # Use professional color palette
    distinct_colors = get_professional_colors(len(group_values))

    # Create figure
    fig = go.Figure()

    # Create the stacked bars by region/country
    for i, group_value in enumerate(group_values):
        # Get color for this region/country
        color = distinct_colors[i % len(distinct_colors)]
        # Filter data for this region/country
        filtered_group_data = stacked_data[stacked_data[group_field] == group_value]
        # Add a trace for this region/country
        fig.add_trace(go.Bar(
            x=filtered_group_data['time_label'],
            y=filtered_group_data[metric],
            name=group_value,
            marker_color=color,
            showlegend=True,
            hoverinfo='y+name+x'
        ))

    # Apply professional styling
    fig.update_layout(barmode='stack')
    fig.update_xaxes(
        title=x_axis_title,
        type='category',
        categoryorder='category ascending'
    )
    fig.update_yaxes(title=title_suffix)
    
    # Apply professional chart styling
    fig = apply_professional_chart_styling(
        fig, 
        title=chart_title,
        height=700,
        show_legend=True,
        legend_title=legend_title
    )

    return fig


def create_datatable(data, metric_for_format=None, aggregation_level='Year'):
    """
    Create a formatted DataTable from the provided pivoted data.
    Handles different aggregation levels and applies formatting.
    If metric_for_format is 'sum_ton_miles', values are divided by 1M.

    Args:
        data (pd.DataFrame): Pivoted data (time periods as rows/index cols, regions as data cols).
        metric_for_format (str, optional): The original metric name used for specific formatting rules. Defaults to None.
        aggregation_level (str): The aggregation level used ('Year', 'Year+Season', 'Year+Quarter'). Helps identify time columns.

    Returns:
        dash_table.DataTable: The configured DataTable component.
    """
    columns = []
    if data is None or data.empty:
        return dash_table.DataTable(
            columns=[{'name': 'Status', 'id': 'status_col'}],
            data=[{'status_col': 'No data available for the selected filters.'}],
            style_cell={'textAlign': 'center'}
        )

    # --- Identify Time Columns vs Data Columns ---
    time_cols = []
    if aggregation_level == 'Year':
        time_cols = ['year']
    elif aggregation_level == 'Year+Season':
        time_cols = ['year', 'season']
    elif aggregation_level == 'Year+Quarter':
        time_cols = ['year', 'quarter']
    elif aggregation_level == 'Month':
        time_cols = ['year', 'month']
    elif aggregation_level == 'Week':
        time_cols = ['year', 'week']

    # Ensure time columns actually exist in the dataframe
    time_cols = [col for col in time_cols if col in data.columns]
    data_cols = [col for col in data.columns if col not in time_cols]

    # --- Sort data by Year (and then Quarter if applicable) in descending order ---
    data_display = data.copy()
    # Create sorting order based on time columns
    sort_columns = []
    sort_ascending = []
    if 'year' in time_cols:
        sort_columns.append('year')
        sort_ascending.append(False)  # Descending order for year
    if 'quarter' in time_cols:
        # First extract the quarter number for sorting
        if 'quarter' in data_display.columns:
            # Extract numeric part from quarter (e.g., 'Q1' -> 1)
            data_display['quarter_num'] = data_display['quarter'].str.extract(r'(\d+)').astype(int)
            sort_columns.append('quarter_num')
            sort_ascending.append(False)  # Descending order for quarter
    elif 'season' in time_cols:
        # For season, we can use alphabetical order (S comes before W)
        # Since we want descending, summer (S) should come before winter (W)
        sort_columns.append('season')
        sort_ascending.append(False)
    elif 'month' in time_cols:
        # Sort by month number in descending order
        if 'month' in data_display.columns:
            sort_columns.append('month')
            sort_ascending.append(False)
    elif 'week' in time_cols:
        # Sort by week number in descending order
        if 'week' in data_display.columns:
            sort_columns.append('week')
            sort_ascending.append(False)
    # Apply sorting if we have any sort columns
    if sort_columns:
        data_display = data_display.sort_values(by=sort_columns, ascending=sort_ascending)
        # If we added quarter_num, drop it after sorting
        if 'quarter_num' in data_display.columns:
            data_display = data_display.drop(columns=['quarter_num'])

    # --- Data Transformation for Ton Miles (Millions) ---
    is_ton_miles = (metric_for_format == 'sum_ton_miles')

    if is_ton_miles:
        cols_to_divide = [col for col in data_cols if pd.api.types.is_numeric_dtype(data_display[col])]
        if cols_to_divide:
            data_display[cols_to_divide] = data_display[cols_to_divide] / 1_000_000
        else:
            pass

    # --- Column Definitions ---
    for col in data_display.columns:
        col_name = str(col).replace('_', ' ').title()
        col_id = str(col)

        if col in time_cols:
            # Determine type for time columns (treat season/quarter as text for alignment)
            col_type = "text" if col in ['season', 'quarter'] else "numeric"
            columns.append({
                "name": col_name,
                "id": col_id,
                "type": col_type
            })
        elif col in data_cols:
            precision = 0
            if metric_for_format == 'median_speed':
                precision = 2
            elif metric_for_format == 'median_utilization_rate':
                precision = 2
            elif metric_for_format in {'mcm_d', 'mtpa'}:
                precision = 1

            col_header = col_name
            columns.append({
                "name": col_header,
                "id": col_id,
                "type": "numeric",
                "format": Format(
                    group=Group.yes,
                    scheme=Scheme.fixed,
                    precision=precision,
                    group_delimiter=',',
                    decimal_delimiter='.'
                )
            })
        else:
            columns.append({"name": col_name, "id": col_id})

    # --- Conditional Styles ---
    conditional_styles = []

    # Style the 'Total' column if it exists
    if 'Total' in data_display.columns:
        conditional_styles.append({
            'if': {'column_id': 'Total'},  # Correct: column_id is a string
            'fontWeight': 'bold',
            'backgroundColor': 'rgb(240, 240, 240)'
        })

    # Right-align numeric DATA columns
    numeric_data_cols = [
        col for col in data_cols  # Iterate through data columns only
        if pd.api.types.is_numeric_dtype(data_display[col])  # Check if the column is numeric
    ]

    for col_id in numeric_data_cols:
        conditional_styles.append({
            'if': {'column_id': col_id},  # Correct: Provide the actual string column ID
            'textAlign': 'right'
        })

    # --- DataTable Creation ---
    return dash_table.DataTable(
        columns=columns,
        data=data_display.to_dict('records'),
        style_table={'overflowX': 'auto', 'width': '100%'},  # Ensure table tries to use available width
        style_cell={
            'textAlign': 'left',  # Default alignment is left
            'padding': '5px',
            'minWidth': '80px',
            'width': 'auto',  # Let table determine width based on content/headers
            'maxWidth': '180px',  # Set a max width
            'whiteSpace': 'normal',
            'font_size': '12px',
            'border': '1px solid grey'  # Add borders for clarity
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center',  # Center headers
            'border': '1px solid grey'
        },
        style_data_conditional=conditional_styles,  # Use the generated list
        merge_duplicate_headers=True,
        page_size=20,
        sort_action='native',
        # fill_width=False, # Usually set fill_width=False when using overflowX
        export_format='xlsx',
        export_headers='display',
        export_columns='visible',
        fill_width=False
    )


def create_route_analysis_table(df, aggregation_level, route_scenario_title, include_route_column=True,
                                destination_level='destination_country_name'):
    """
    Creates a table showing aggregated trade counts by destination column (columns) and time periods (rows)
    for a specific route analysis scenario. Tables are ordered in descending order by time period.
    Now supports multi-level headers where the first level contains the route values.

    Args:
        df (pd.DataFrame): The filtered DataFrame for a specific route scenario
        aggregation_level (str): How to aggregate time ('Year', 'Year+Season', 'Year+Quarter')
        route_scenario_title (str): Title of the route scenario for reference
        include_route_column (bool): Whether to include the selected_route column in the table
        destination_level (str): Column to use for destination ('destination_shipping_region' or 'destination_country_name')

    Returns:
        dash_table.DataTable: The configured DataTable component.
    """
    # Default for empty data
    if df is None or df.empty:
        return dash_table.DataTable(
            columns=[{'name': 'Status', 'id': 'status_col'}],
            data=[{'status_col': f'No data available for {route_scenario_title}'}],
            style_cell={'textAlign': 'center'}
        )

    # Determine index columns based on aggregation level
    if aggregation_level == 'Year':
        index_cols = ['year']
    elif aggregation_level == 'Year+Season':
        index_cols = ['year', 'season']
    elif aggregation_level == 'Year+Quarter':
        index_cols = ['year', 'quarter']
    else:
        index_cols = ['year']

    # Check for required columns
    required_cols = index_cols + [destination_level, 'voyage_id']
    if include_route_column:
        required_cols.append('selected_route')

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return dash_table.DataTable(
            columns=[{'name': 'Error', 'id': 'error_col'}],
            data=[{'error_col': f'Missing columns: {", ".join(missing_cols)}'}],
            style_cell={'textAlign': 'center'}
        )

    try:
        # Ensure selected_route exists before we try to group by it
        if include_route_column and 'selected_route' not in df.columns:
            df['selected_route'] = 'Unknown'

        # Get unique routes if include_route_column is True
        unique_routes = ['All Routes']
        if include_route_column:
            unique_routes = df['selected_route'].unique().tolist()

        # Create pivot tables for each route
        result_tables = {}

        if include_route_column:
            # Process each route separately
            for route in unique_routes:
                route_df = df[df['selected_route'] == route].copy()

                # Group by time period and destination
                groupby_cols = index_cols + [destination_level]
                grouped_df = route_df.groupby(groupby_cols, observed=True)['voyage_id'].count().reset_index()

                # Create pivot table with destination as columns
                pivot_df = pd.pivot_table(
                    grouped_df,
                    index=index_cols,
                    columns=destination_level,
                    values='voyage_id',
                    aggfunc='sum',
                    fill_value=0
                ).reset_index()

                result_tables[route] = pivot_df
        else:
            # No route column, just one pivot table
            # Group by time period and destination
            groupby_cols = index_cols + [destination_level]
            grouped_df = df.groupby(groupby_cols, observed=True)['voyage_id'].count().reset_index()

            # Create pivot table with destination as columns
            pivot_df = pd.pivot_table(
                grouped_df,
                index=index_cols,
                columns=destination_level,
                values='voyage_id',
                aggfunc='sum',
                fill_value=0
            ).reset_index()

            result_tables['All Routes'] = pivot_df

        # Combine all pivot tables into one with multi-level columns
        all_pivots = []
        for route, pivot_df in result_tables.items():
            # Create a time_period column if needed (for Year+Season or Year+Quarter)
            if len(index_cols) > 1:
                if 'season' in index_cols and 'year' in index_cols and 'season' in pivot_df.columns:
                    pivot_df['time_period'] = pivot_df['year'].astype(str) + '-' + pivot_df['season'].astype(str)
                elif 'quarter' in index_cols and 'year' in index_cols and 'quarter' in pivot_df.columns:
                    pivot_df['time_period'] = pivot_df['year'].astype(str) + '-' + pivot_df['quarter'].astype(str)

            # Add route as a prefix to all data columns (but not to time/index columns)
            # First, identify which columns are time/index columns
            time_cols = index_cols + ['time_period']
            time_cols = [col for col in time_cols if col in pivot_df.columns]

            # Get destination columns (all non-time columns)
            dest_cols = [col for col in pivot_df.columns if col not in time_cols]

            # Copy the pivot table to avoid modifying the original
            pivot_df_copy = pivot_df.copy()

            # Rename destination columns to include route as prefix
            col_map = {col: f"{route}||{col}" for col in dest_cols}
            pivot_df_copy = pivot_df_copy.rename(columns=col_map)

            all_pivots.append(pivot_df_copy)

        # Merge all pivot tables on time columns
        final_df = all_pivots[0]
        for pivot_df in all_pivots[1:]:
            final_df = pd.merge(
                final_df,
                pivot_df,
                on=time_cols,
                how='outer'
            )

        # Sort by year and (if applicable) quarter/season in DESCENDING order
        # Create sorting keys
        if 'year' in final_df.columns:
            if 'quarter' in final_df.columns:
                # Extract quarter number for proper sorting
                final_df['quarter_num'] = final_df['quarter'].str.extract(r'(\d+)').astype(int)
                final_df['sort_key'] = final_df['year'] * 10 + final_df['quarter_num']
                final_df = final_df.sort_values('sort_key', ascending=False)
                final_df = final_df.drop(columns=['quarter_num', 'sort_key'])
            elif 'season' in final_df.columns:
                # Create a numeric value for season (S=1, W=2)
                final_df['season_num'] = final_df['season'].apply(lambda x: 1 if x == 'S' else 2)
                final_df['sort_key'] = final_df['year'] * 10 + final_df['season_num']
                final_df = final_df.sort_values('sort_key', ascending=False)
                final_df = final_df.drop(columns=['season_num', 'sort_key'])
            else:
                # Just sort by year
                final_df = final_df.sort_values('year', ascending=False)
        elif 'time_period' in final_df.columns:
            # If there's already a time_period column, try to sort it appropriately
            # First create a sort key from the time_period
            def create_sort_key(time_str):
                parts = str(time_str).split('-')
                try:
                    year = int(parts[0])
                    if len(parts) > 1:
                        if parts[1].startswith('Q'):
                            quarter = int(parts[1][1:])
                            return year * 10 + quarter
                        elif parts[1] in ['S', 'W']:
                            season_num = 1 if parts[1] == 'S' else 2
                            return year * 10 + season_num
                    return year * 100
                except (ValueError, IndexError):
                    return 0

            final_df['sort_key'] = final_df['time_period'].apply(create_sort_key)
            final_df = final_df.sort_values('sort_key', ascending=False)
            final_df = final_df.drop(columns=['sort_key'])

        # Determine which columns to display
        display_cols = []

        # Add time columns
        if 'time_period' in final_df.columns:
            display_cols.append('time_period')
        else:
            # If there's no time_period column, add all index columns
            display_cols.extend(index_cols)

        # Add all data columns
        data_cols = [col for col in final_df.columns if col not in display_cols and '||' in col]
        display_cols.extend(data_cols)

        # Make sure all columns we want to display actually exist in the dataframe
        display_cols = [col for col in display_cols if col in final_df.columns]

        # Reorder columns for display
        final_df = final_df[display_cols]

        # Create column definitions for the DataTable with multi-level headers
        table_columns = []

        # Add time columns (Year or Time Period)
        if 'time_period' in final_df.columns:
            table_columns.append({
                "name": ["Time Period", ""],
                "id": "time_period",
                "type": "text"
            })
        elif 'year' in final_df.columns:
            table_columns.append({
                "name": ["Year", ""],
                "id": "year",
                "type": "numeric"
            })
            # Add season or quarter if present
            if 'season' in final_df.columns:
                table_columns.append({
                    "name": ["Season", ""],
                    "id": "season",
                    "type": "text"
                })
            elif 'quarter' in final_df.columns:
                table_columns.append({
                    "name": ["Quarter", ""],
                    "id": "quarter",
                    "type": "text"
                })

        # Add data columns with multi-level headers
        for col in data_cols:
            # Split into route and destination parts
            route, destination = col.split('||', 1)

            table_columns.append({
                "name": [route, destination],  # Multi-level header
                "id": col,
                "type": "numeric",
                "format": Format(
                    group=Group.yes,
                    scheme=Scheme.fixed,
                    precision=0,
                    group_delimiter=',',
                    decimal_delimiter='.'
                )
            })

        # Conditional styles for the DataTable
        conditional_styles = []

        # Right-align numeric columns
        for col_id in data_cols:
            conditional_styles.append({
                'if': {'column_id': str(col_id)},
                'textAlign': 'right'
            })

        # Create the DataTable with multi-level headers
        return dash_table.DataTable(
            columns=table_columns,
            data=final_df.to_dict('records'),
            style_table={'overflowX': 'auto', 'width': '100%'},
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'minWidth': '80px',
                'width': 'auto',
                'maxWidth': '180px',
                'whiteSpace': 'normal',
                'font_size': '12px',
                'border': '1px solid grey'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'border': '1px solid grey'
            },
            style_data_conditional=conditional_styles,
            merge_duplicate_headers=True,  # Important for multi-level headers
            page_size=25,  # Smaller page size for these tables
            sort_action='native',
            export_format='xlsx',
            export_headers='display',
            export_columns='visible',
            fill_width=False
        )

    except Exception as e:
        return dash_table.DataTable(
            columns=[{'name': 'Error', 'id': 'error_col'}],
            data=[{'error_col': f'Error: {str(e)}'}],
            style_cell={'textAlign': 'center'}
        )


# Dashboard layout
layout = html.Div([
    # Store components for US data
    dcc.Store(id='us-region-data-store', storage_type='local'),
    dcc.Store(id='us-dropdown-options-store', storage_type='local'),
    dcc.Store(id='us-refresh-timestamp-store', storage_type='local'),
    dcc.Store(id='diversion-processed-data', storage_type='local'),
    dcc.Store(id='destination-expanded-continents', data=[]),  # Store for expanded state of continents
    dcc.Store(id='origin-plant-expanded-zones', data=[]),  # Store for expanded state of origin zones
    dcc.Store(id='maintenance-expanded-plants', data=[]),  # Store for expanded state of plants
    dcc.Store(id='exp-destination-forecast-expanded-continents', data=[]),  # Store for WoodMac forecast table expansion
    dcc.Download(id='download-exporter-detail-supply-excel'),
    dcc.Download(id='download-trade-analysis-excel'),
    dcc.Download(id='download-route-analysis-excel'),
    dcc.Download(id='download-diversion-summary-excel'),

    # Professional Section Header - Exporter Analysis Configuration
    html.Div([
        html.Div([

            # --- Group 1: Origin ---
            html.Div([
                html.Div("Origin", className='filter-group-header'),
                html.Div([
                    html.Div([
                        html.Label("Origin Country:", className='filter-label'),
                        dcc.Dropdown(
                            id='origin-country-dropdown',
                            options=[],
                            value='United States',
                            multi=False,
                            clearable=False,
                            className='filter-dropdown',
                            style={'min-width': '180px'}
                        ),
                    ], className='filter-group'),
                ], style={'display': 'flex', 'gap': '8px', 'alignItems': 'flex-end'}),
            ], className='filter-section filter-section-origin-exp'),

            # --- Group 2: Destination ---
            html.Div([
                html.Div("Destination", className='filter-group-header'),
                html.Div([
                    html.Div([
                        html.Label("Destination Level:", className='filter-label'),
                        dcc.Dropdown(
                            id='destination-level-dropdown',
                            options=[
                                {'label': 'Shipping Region',        'value': 'destination_shipping_region'},
                                {'label': 'Country',                'value': 'destination_country_name'},
                                {'label': 'Basin',                  'value': 'destination_basin'},
                                {'label': 'Continent',              'value': 'continent_destination_name'},
                                {'label': 'Subcontinent',           'value': 'destination_subcontinent'},
                                {'label': 'Classification Level 1', 'value': 'destination_classification_level1'},
                                {'label': 'Classification',         'value': 'destination_classification'},
                            ],
                            value='destination_shipping_region',
                            multi=False,
                            clearable=False,
                            className='filter-dropdown',
                            style={'min-width': '180px'}
                        ),
                    ], className='filter-group'),
                ], style={'display': 'flex', 'gap': '8px', 'alignItems': 'flex-end'}),
            ], className='filter-section filter-section-destination'),

            # --- Group 3: Analysis Settings ---
            html.Div([
                html.Div("Analysis Settings", className='filter-group-header'),
                html.Div([
                    html.Div([
                        html.Label("Aggregation:", className='filter-label'),
                        dcc.Dropdown(
                            id='aggregation-dropdown',
                            options=[
                                {'label': 'Year', 'value': 'Year'},
                                {'label': 'Year + Season', 'value': 'Year+Season'},
                                {'label': 'Year + Quarter', 'value': 'Year+Quarter'},
                                {'label': 'Month', 'value': 'Month'},
                                {'label': 'Week', 'value': 'Week'},
                            ],
                            value='Year+Quarter',
                            multi=False,
                            clearable=False,
                            className='filter-dropdown',
                            style={'min-width': '144px'}
                        ),
                    ], className='filter-group'),
                    html.Div([
                        html.Label("Status:", className='filter-label'),
                        dcc.Dropdown(
                            id='us-region-status-dropdown',
                            options=[
                                {'label': 'Laden', 'value': 'laden'},
                                {'label': 'Non-Laden', 'value': 'non_laden'}
                            ],
                            value='laden',
                            multi=False,
                            clearable=False,
                            className='filter-dropdown',
                            style={'min-width': '100px'}
                        ),
                    ], className='filter-group'),
                    html.Div([
                        html.Label("Vessel Size:", className='filter-label'),
                        dcc.Dropdown(
                            id='us-vessel-type-dropdown',
                            options=[],
                            value='All',
                            clearable=False,
                            className='filter-dropdown',
                            style={'min-width': '160px'}
                        ),
                    ], className='filter-group'),
                    html.Div([
                        html.Label("Metric:", className='filter-label'),
                        dcc.Dropdown(
                            id='chart-metric-dropdown',
                            options=[
                                {'label': 'Count of Trades', 'value': 'count_trades'},
                                {'label': 'MTPA', 'value': 'mtpa'},
                                {'label': 'mcm/d', 'value': 'mcm_d'},
                                {'label': 'm³', 'value': 'm3'},
                                {'label': 'Median Delivery Days', 'value': 'median_delivery_days'},
                                {'label': 'Median Speed', 'value': 'median_speed'},
                                {'label': 'Median Mileage (Nautical Miles)', 'value': 'median_mileage_nautical_miles'},
                                {'label': 'Median Ton Miles', 'value': 'median_ton_miles'},
                                {'label': 'Median Utilization Rate', 'value': 'median_utilization_rate'},
                                {'label': 'Median Cargo Volume (m³)', 'value': 'median_cargo_destination_cubic_meters'},
                                {'label': 'Median Vessel Capacity (m³)', 'value': 'median_vessel_capacity_cubic_meters'}
                            ],
                            value='mcm_d',
                            clearable=False,
                            className='filter-dropdown',
                            style={'min-width': '220px'}
                        ),
                    ], className='filter-group'),
                ], style={'display': 'flex', 'gap': '12px', 'alignItems': 'flex-end'}),
            ], className='filter-section filter-section-analysis'),

        ], className='filter-bar-grouped')
    ], className='professional-section-header'),

    # Country Supply Charts Section - Three charts side by side
    html.Div([
        # Section Header
        html.Div([
            html.H3(
                'LNG Supply Analysis - 30-Day Rolling Average',
                id='supply-analysis-title',
                className="section-title-inline"
            ),
            html.Label("Window (days):", className="inline-filter-label", style={'marginLeft': '20px'}),
            dcc.Input(
                id='supply-rolling-window-input',
                type='number',
                value=30,
                min=1,
                step=1,
                debounce=True,
                className='filter-input',
                style={'width': '80px', 'height': '34px', 'fontSize': '14px', 'padding': '6px 8px'}
            ),
            html.Button(
                'Export to Excel',
                id='export-supply-analysis-button',
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
        ], className="inline-section-header", style={'display': 'flex', 'alignItems': 'center'}),
        
        # Charts Container - Three charts side by side
        html.Div([
            # Left Chart - Country Supply
            html.Div([
                html.H4(id='country-supply-header', children='Total Supply', 
                       style={'fontSize': '14px', 'marginBottom': '10px', 'color': '#4A4A4A'}),
                dcc.Loading(
                    id="country-supply-loading",
                    children=[
                        dcc.Graph(id='country-supply-chart', style={'height': '400px'})
                    ],
                    type="default",
                )
            ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            # Spacer
            html.Div(style={'width': '2%', 'display': 'inline-block'}),
            
            # Middle Chart - Continent Destinations (Absolute)
            html.Div([
                html.H4(id='continent-destination-header', children='By Destination Continent (mcm/d)',
                       style={'fontSize': '14px', 'marginBottom': '10px', 'color': '#4A4A4A'}),
                dcc.Loading(
                    id="continent-destination-loading",
                    children=[
                        dcc.Graph(id='continent-destination-chart', style={'height': '400px'})
                    ],
                    type="default",
                )
            ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            # Spacer
            html.Div(style={'width': '2%', 'display': 'inline-block'}),
            
            # Right Chart - Continent Destinations (Percentage)
            html.Div([
                html.H4(id='continent-percentage-header', children='By Destination Continent (%)',
                       style={'fontSize': '14px', 'marginBottom': '10px', 'color': '#4A4A4A'}),
                dcc.Loading(
                    id="continent-percentage-loading",
                    children=[
                        dcc.Graph(id='continent-percentage-chart', style={'height': '400px'})
                    ],
                    type="default",
                )
            ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'padding': '20px'})
    ], className="main-section-container", style={'marginBottom': '30px'}),

    # Origin Plant Summary + Train Maintenance Schedule (side by side)
    html.Div([
        # Left: Origin Plant Summary
        html.Div([
            html.Div([
                html.H3('Origin Plant Summary (mcm/d)', className="section-title-inline"),
            ], className="inline-section-header"),
            html.Div([
                dcc.Loading(
                    id="origin-plant-summary-loading",
                    children=[html.Div(id='origin-plant-summary-table-container')],
                    type="default"
                )
            ], style={'marginTop': '20px'})
        ], className='section-container', style={'flex': '1', 'minWidth': '0', 'margin': '0'}),

        # Right: Train Maintenance Schedule
        html.Div([
            html.Div([
                html.H3('Train Maintenance Schedule (MCM/D Impact)', className="section-title-inline"),
            ], className="inline-section-header"),
            html.Div([
                dcc.Loading(
                    id="maintenance-summary-loading",
                    children=[
                        html.Div(id='maintenance-summary-container')
                    ],
                    type="default"
                )
            ], style={'marginTop': '20px'})
        ], className='section-container', style={'flex': '1', 'minWidth': '0', 'margin': '0'}),
    ], style={'display': 'flex', 'gap': '24px', 'alignItems': 'flex-start', 'marginBottom': '32px'}),

    # Destination Analysis Summary + Trade Analysis Chart (side by side)
    html.Div([
        # Left: Destination Analysis Summary
        html.Div([
            html.Div([
                html.H3('Destination Analysis Summary (mcm/d)', className="section-title-inline"),
            ], className="inline-section-header"),
            html.Div([
                dcc.Loading(
                    id="destination-summary-loading",
                    children=[
                        html.Div(id='destination-summary-table-container')
                    ],
                    type="default"
                )
            ], style={'marginTop': '20px'})
        ], className='section-container', style={'flex': '1', 'minWidth': '0', 'margin': '0'}),

        # Right: Trade Analysis Chart
        html.Div([
            html.Div([
                html.H3(id='trade-analysis-header', className='section-title-inline'),
                html.Button('Export to Excel', id='export-trade-analysis-button', n_clicks=0,
                    style={'marginLeft': '20px', 'padding': '5px 15px', 'backgroundColor': '#28a745',
                           'color': 'white', 'border': 'none', 'borderRadius': '4px',
                           'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '12px'}),
            ], className='inline-section-header', style={'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                dcc.Graph(id='us-trade-count-visualization', style={'height': '600px'})
            ], style={'marginTop': '20px'}),
        ], className='section-container', style={'flex': '1', 'minWidth': '0', 'margin': '0'}),
    ], style={'display': 'flex', 'gap': '24px', 'alignItems': 'flex-start', 'marginBottom': '32px'}),

    # Destination Forecast Allocation Summary (WoodMac)
    html.Div([
        html.Div([
            html.H3('Destination Forecast Allocation Summary (WoodMac, mcm/d)', className="section-title-inline"),
        ], className="inline-section-header"),
        html.Div(
            id='exp-destination-forecast-summary-subtitle',
            style={
                'marginTop': '8px',
                'fontSize': '12px',
                'color': '#6b7280',
                'fontStyle': 'italic'
            }
        ),
        html.Div([
            dcc.Loading(
                id="exp-destination-forecast-summary-loading",
                children=[
                    html.Div(id='exp-destination-forecast-summary-table-container')
                ],
                type="default"
            )
        ], style={'marginTop': '20px'})
    ], className='section-container', style={'margin-bottom': '32px'}),

    # Route Analysis Section
    html.Div([
        html.Div([
            html.H3("Route Analysis", className='section-title-inline'),
            html.Button('Export to Excel', id='export-route-analysis-button', n_clicks=0,
                style={'marginLeft': '20px', 'padding': '5px 15px', 'backgroundColor': '#28a745',
                       'color': 'white', 'border': 'none', 'borderRadius': '4px',
                       'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '12px'}),
        ], className='inline-section-header', style={'display': 'flex', 'alignItems': 'center'}),

        html.Div([
            dcc.Graph(id='graph-route-suez-only', style={'height': '600px'})
        ], style={'margin-bottom': '24px'}),
    ], className='section-container', style={'margin-bottom': '0'}),

    # Diversions Analysis Section
    html.Div([
        html.Div([
            html.H2("Diversions Analysis", className='section-title-inline'),
            html.P("Analyze route deviations and alternative shipping patterns by destination level", className='section-subtitle')
        ], className='header-content'),
        
        html.Div([
            html.Div([
                html.Label("Select Destination Level:", className='filter-label'),
                dcc.RadioItems(
                    id='destination-level-radio',
                    options=[
                        {'label': 'Basin', 'value': 'basin_combo'},
                        {'label': 'Region', 'value': 'region_combo'},
                        {'label': 'Country', 'value': 'country_combo'}
                    ],
                    value='basin_combo',
                    inline=True,
                    style={'display': 'flex', 'gap': '16px'}
                )
            ], className='filter-group'),
            
        ], className='filter-bar')
    ], className='inline-section-header', style={'marginTop': '0'}),
    # Diversions Analysis Chart + Summary Table
    html.Div([
        html.Div([
            dcc.Graph(id='diversion-count-chart', style={'height': '600px'})
        ], style={'margin-bottom': '24px'}),

        html.Div([
            html.H2("Diversions Summary Table", className='mckinsey-header', style={'margin': '0'}),
            html.Button('Export to Excel', id='export-diversion-summary-button', n_clicks=0,
                style={'marginLeft': '20px', 'padding': '5px 15px', 'backgroundColor': '#28a745',
                       'color': 'white', 'border': 'none', 'borderRadius': '4px',
                       'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '12px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
        dash_table.DataTable(
            id='diversion-table',
            data=[],
            columns=[],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
                       'padding': '8px', 'fontSize': '12px'},
            style_header={'backgroundColor': '#2E86C1', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '12px'},
            sort_action="native",
            page_action="native",
            page_size=20,
            fill_width=False
        )
    ], className='section-container', style={'paddingTop': '0'})

])

# ========================================
# DESTINATION SUMMARY TABLE FUNCTIONS
# ========================================

def fetch_destination_summary_data(engine, origin_country, status, vessel_type, rolling_window_days=30, destination_level='destination_country_name'):
    """
    Fetch destination summary data with expandable continent/country hierarchy.
    destination_level controls how destinations are grouped.
    """

    try:
        # Always fetch at country + continent level from kpler_trades
        dest_country_col = "COALESCE(NULLIF(destination_country_name, ''), 'Unknown')"
        dest_continent_col = "COALESCE(NULLIF(continent_destination_name, ''), 'Unknown')"

        # Build WHERE clause based on filters
        where_conditions = [
            f"origin_country_name = '{origin_country}'",
            f"destination_country_name != '{origin_country}'"
        ]

        where_clause = " AND ".join(where_conditions)

        with engine.connect() as conn:
            quarters_df = fetch_destination_periods_data_hierarchical(conn, dest_continent_col, dest_country_col, where_clause, 'quarter')
            months_df = fetch_destination_periods_data_hierarchical(conn, dest_continent_col, dest_country_col, where_clause, 'month')
            weeks_df = fetch_destination_periods_data_hierarchical(conn, dest_continent_col, dest_country_col, where_clause, 'week')
            rolling_df = fetch_destination_rolling_windows_hierarchical(
                conn, dest_continent_col, dest_country_col, where_clause, rolling_window_days
            )
            result = combine_destination_summary_data_hierarchical(
                quarters_df, months_df, weeks_df, rolling_df, rolling_window_days
            )

        # If a non-country level is selected, replace the 'continent' grouping column
        # with the selected level while keeping country rows intact for expand/collapse
        if destination_level != 'destination_country_name' and not result.empty:
            level_col_map = {
                'continent_destination_name':          'continent',
                'destination_shipping_region':         'shipping_region',
                'destination_basin':                   'basin',
                'destination_subcontinent':            'subcontinent',
                'destination_classification_level1':   'country_classification_level1',
                'destination_classification':          'country_classification',
            }
            mapping_col = level_col_map.get(destination_level)
            if mapping_col and mapping_col != 'continent':
                mapping_df = pd.read_sql(
                    f"SELECT DISTINCT country_name AS country, {mapping_col} FROM {DB_SCHEMA}.mappings_country WHERE country_name IS NOT NULL",
                    engine
                )
                result = pd.merge(result, mapping_df, on='country', how='left')
                result[mapping_col] = result[mapping_col].fillna('Unknown')
                # Replace continent with the selected level grouping; keep country intact
                result['continent'] = result[mapping_col]
                result = result.drop(columns=[mapping_col])
            # For continent_destination_name: continent column is already correct

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def fetch_origin_plant_summary_data(engine, origin_country, rolling_window_days=30):
    """Fetch origin plant summary data with expandable zone/plant hierarchy."""
    try:
        zone_col = "COALESCE(NULLIF(zone_origin_name, ''), 'Unknown')"
        plant_col = "COALESCE(NULLIF(origin_location_name, ''), 'Unknown')"
        where_clause = f"origin_country_name = '{origin_country}' AND destination_country_name != '{origin_country}'"

        with engine.connect() as conn:
            quarters_df = fetch_destination_periods_data_hierarchical(conn, zone_col, plant_col, where_clause, 'quarter')
            months_df   = fetch_destination_periods_data_hierarchical(conn, zone_col, plant_col, where_clause, 'month')
            weeks_df    = fetch_destination_periods_data_hierarchical(conn, zone_col, plant_col, where_clause, 'week')
            rolling_df  = fetch_destination_rolling_windows_hierarchical(conn, zone_col, plant_col, where_clause, rolling_window_days)
            return combine_destination_summary_data_hierarchical(quarters_df, months_df, weeks_df, rolling_df, rolling_window_days)
    except Exception:
        return pd.DataFrame()


def fetch_destination_periods_data(conn, dest_col, where_clause, period_type):
    """Fetch data for specific period type (quarter, month, week)"""
    from sqlalchemy import text
    
    
    try:
        # Determine the grouping and formatting based on period type
        if period_type == 'quarter':
            period_expr = """
                'Q' || EXTRACT(QUARTER FROM "start")::text || '''' || 
                RIGHT(EXTRACT(YEAR FROM "start")::text, 2) as period
            """
            order_expr = "EXTRACT(YEAR FROM \"start\"), EXTRACT(QUARTER FROM \"start\")"
        elif period_type == 'month':
            period_expr = """
                TO_CHAR("start", 'Mon') || '''' || 
                RIGHT(EXTRACT(YEAR FROM "start")::text, 2) as period
            """
            order_expr = "EXTRACT(YEAR FROM \"start\"), EXTRACT(MONTH FROM \"start\")"
        else:  # week
            period_expr = """
                'W' || EXTRACT(WEEK FROM "start")::text || '''' || 
                RIGHT(EXTRACT(YEAR FROM "start")::text, 2) as period
            """
            order_expr = "EXTRACT(YEAR FROM \"start\"), EXTRACT(WEEK FROM \"start\")"
        
        # First check if we have any data
        check_query = text(f"""
            SELECT COUNT(*) as count
            FROM {DB_SCHEMA}.kpler_trades
            WHERE origin_country_name = :origin_country
            AND "start" IS NOT NULL
        """)

        # Extract origin_country from where_clause
        import re
        origin_match = re.search(r"origin_country_name = '([^']*)'", where_clause)
        origin_country = origin_match.group(1) if origin_match else None
        
        if origin_country:
            check_result = conn.execute(check_query, {"origin_country": origin_country})
            count = check_result.fetchone()[0]
        
        # Build query with proper daily average calculation using total calendar days
        if period_type == 'quarter':
            # Quarters have ~90-92 days, we'll use a fixed 91.25 days (365.25/4)
            query_str = f"""
                WITH latest_timestamp AS (
                    SELECT MAX(upload_timestamp_utc) as max_ts
                    FROM {DB_SCHEMA}.kpler_trades
                ),
                period_data AS (
                    SELECT 
                        {dest_col} as destination,
                        {period_expr},
                        SUM(cargo_origin_cubic_meters * 0.6 / 1000) / 91.25 as mcm_d
                    FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                    WHERE upload_timestamp_utc = max_ts
                        AND {where_clause}
                        AND "start" >= CURRENT_DATE - INTERVAL '2 years'
                        AND "start" < CURRENT_DATE
                    GROUP BY {dest_col}, period, {order_expr}
                    ORDER BY {dest_col}, {order_expr}
                )
                SELECT destination, period, mcm_d
                FROM period_data
            """
        elif period_type == 'month':
            # For months, we need to calculate days per each month/year combination
            query_str = f"""
                WITH latest_timestamp AS (
                    SELECT MAX(upload_timestamp_utc) as max_ts
                    FROM {DB_SCHEMA}.kpler_trades
                ),
                period_data AS (
                    SELECT 
                        {dest_col} as destination,
                        {period_expr},
                        EXTRACT(YEAR FROM "start") as year,
                        EXTRACT(MONTH FROM "start") as month,
                        SUM(cargo_origin_cubic_meters * 0.6 / 1000) as total_mcm
                    FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                    WHERE upload_timestamp_utc = max_ts
                        AND {where_clause}
                        AND "start" >= CURRENT_DATE - INTERVAL '2 years'
                        AND "start" < CURRENT_DATE
                    GROUP BY {dest_col}, period, EXTRACT(YEAR FROM "start"), EXTRACT(MONTH FROM "start")
                ),
                with_days AS (
                    SELECT 
                        destination,
                        period,
                        total_mcm / EXTRACT(DAY FROM 
                            (DATE_TRUNC('month', MAKE_DATE(year::int, month::int, 1)) + 
                             INTERVAL '1 month' - INTERVAL '1 day')
                        ) as mcm_d
                    FROM period_data
                )
                SELECT destination, period, mcm_d
                FROM with_days
                ORDER BY destination, period
            """
        else:  # week
            # Weeks always have 7 days
            query_str = f"""
                WITH latest_timestamp AS (
                    SELECT MAX(upload_timestamp_utc) as max_ts
                    FROM {DB_SCHEMA}.kpler_trades
                ),
                period_data AS (
                    SELECT 
                        {dest_col} as destination,
                        {period_expr},
                        SUM(cargo_origin_cubic_meters * 0.6 / 1000) / 7.0 as mcm_d
                    FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                    WHERE upload_timestamp_utc = max_ts
                        AND {where_clause}
                        AND "start" >= CURRENT_DATE - INTERVAL '2 years'
                        AND "start" < CURRENT_DATE
                    GROUP BY {dest_col}, period, {order_expr}
                    ORDER BY {dest_col}, {order_expr}
                )
                SELECT destination, period, mcm_d
                FROM period_data
            """
        
        
        query = text(query_str)
        df = pd.read_sql(query, conn)
        
        if df.empty:
            return pd.DataFrame()
        
        # Pivot to get periods as columns
        pivot_df = df.pivot_table(
            index='destination',
            columns='period',
            values='mcm_d',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        return pivot_df
        
    except Exception as e:
        return pd.DataFrame()

def fetch_destination_rolling_windows(conn, dest_col, where_clause):
    """Fetch 7D and 30D rolling window data with deltas"""
    from sqlalchemy import text
    from datetime import datetime, timedelta
    
    try:
        current_date = datetime.now().date()
        date_7d_ago = current_date - timedelta(days=7)
        date_30d_ago = current_date - timedelta(days=30)
        date_30d_y1_start = current_date - timedelta(days=365) - timedelta(days=30)
        date_30d_y1_end = current_date - timedelta(days=365)
        
        query = text(f"""
            WITH latest_timestamp AS (
                SELECT MAX(upload_timestamp_utc) as max_ts
                FROM {DB_SCHEMA}.kpler_trades
            ),
            window_7d AS (
                SELECT 
                    {dest_col} as destination,
                    SUM(cargo_origin_cubic_meters * 0.6 / 1000) / 7.0 as avg_7d
                FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                WHERE upload_timestamp_utc = max_ts
                    AND {where_clause}
                    AND "start" > '{date_7d_ago}'
                    AND "start" <= '{current_date}'
                GROUP BY {dest_col}
            ),
            window_30d AS (
                SELECT 
                    {dest_col} as destination,
                    SUM(cargo_origin_cubic_meters * 0.6 / 1000) / 30.0 as avg_30d
                FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                WHERE upload_timestamp_utc = max_ts
                    AND {where_clause}
                    AND "start" > '{date_30d_ago}'
                    AND "start" <= '{current_date}'
                GROUP BY {dest_col}
            ),
            window_30d_y1 AS (
                SELECT 
                    {dest_col} as destination,
                    SUM(cargo_origin_cubic_meters * 0.6 / 1000) / 30.0 as avg_30d_y1
                FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                WHERE upload_timestamp_utc = max_ts
                    AND {where_clause}
                    AND "start" > '{date_30d_y1_start}'
                    AND "start" <= '{date_30d_y1_end}'
                GROUP BY {dest_col}
            )
            SELECT 
                COALESCE(w7.destination, w30.destination, w30y1.destination) as destination,
                COALESCE(w7.avg_7d, 0) as "7D",
                COALESCE(w30.avg_30d, 0) as "30D",
                COALESCE(w30y1.avg_30d_y1, 0) as "30D_Y1",
                COALESCE(w7.avg_7d, 0) - COALESCE(w30.avg_30d, 0) as "Δ 7D-30D",
                COALESCE(w30.avg_30d, 0) - COALESCE(w30y1.avg_30d_y1, 0) as "Δ 30D Y/Y"
            FROM window_7d w7
            FULL OUTER JOIN window_30d w30 ON w7.destination = w30.destination
            FULL OUTER JOIN window_30d_y1 w30y1 ON COALESCE(w7.destination, w30.destination) = w30y1.destination
        """)
        df = pd.read_sql(query, conn)
        return df
        
    except Exception as e:
        return pd.DataFrame()

def fetch_destination_periods_data_hierarchical(conn, continent_col, country_col, where_clause, period_type):
    """Fetch data for specific period type with continent/country hierarchy"""
    from sqlalchemy import text
    
    
    try:
        # Determine the grouping and formatting based on period type
        if period_type == 'quarter':
            period_expr = """
                'Q' || EXTRACT(QUARTER FROM "start")::text || '''' || 
                RIGHT(EXTRACT(YEAR FROM "start")::text, 2) as period
            """
            order_expr = "EXTRACT(YEAR FROM \"start\"), EXTRACT(QUARTER FROM \"start\")"
            days_divisor = "91.25"
        elif period_type == 'month':
            period_expr = """
                TO_CHAR("start", 'Mon') || '''' || 
                RIGHT(EXTRACT(YEAR FROM "start")::text, 2) as period
            """
            order_expr = "EXTRACT(YEAR FROM \"start\"), EXTRACT(MONTH FROM \"start\")"
            # For months, we need a subquery approach as before
        else:  # week
            period_expr = """
                'W' || EXTRACT(WEEK FROM "start")::text || '''' || 
                RIGHT(EXTRACT(YEAR FROM "start")::text, 2) as period
            """
            order_expr = "EXTRACT(YEAR FROM \"start\"), EXTRACT(WEEK FROM \"start\")"
            days_divisor = "7.0"
        
        if period_type == 'month':
            # Special handling for months to get actual days
            query_str = f"""
                WITH latest_timestamp AS (
                    SELECT MAX(upload_timestamp_utc) as max_ts
                    FROM {DB_SCHEMA}.kpler_trades
                ),
                period_data AS (
                    SELECT 
                        {continent_col} as continent,
                        {country_col} as country,
                        {period_expr},
                        EXTRACT(YEAR FROM "start") as year,
                        EXTRACT(MONTH FROM "start") as month,
                        SUM(cargo_origin_cubic_meters * 0.6 / 1000) as total_mcm
                    FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                    WHERE upload_timestamp_utc = max_ts
                        AND {where_clause}
                        AND "start" >= CURRENT_DATE - INTERVAL '2 years'
                        AND "start" < CURRENT_DATE
                    GROUP BY {continent_col}, {country_col}, period, EXTRACT(YEAR FROM "start"), EXTRACT(MONTH FROM "start")
                ),
                with_days AS (
                    SELECT 
                        continent,
                        country,
                        period,
                        total_mcm / EXTRACT(DAY FROM 
                            (DATE_TRUNC('month', MAKE_DATE(year::int, month::int, 1)) + 
                             INTERVAL '1 month' - INTERVAL '1 day')
                        ) as mcm_d
                    FROM period_data
                )
                SELECT continent, country, period, mcm_d
                FROM with_days
            """
        else:
            # Quarters and weeks
            query_str = f"""
                WITH latest_timestamp AS (
                    SELECT MAX(upload_timestamp_utc) as max_ts
                    FROM {DB_SCHEMA}.kpler_trades
                ),
                period_data AS (
                    SELECT 
                        {continent_col} as continent,
                        {country_col} as country,
                        {period_expr},
                        SUM(cargo_origin_cubic_meters * 0.6 / 1000) / {days_divisor} as mcm_d
                    FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                    WHERE upload_timestamp_utc = max_ts
                        AND {where_clause}
                        AND "start" >= CURRENT_DATE - INTERVAL '2 years'
                        AND "start" < CURRENT_DATE
                    GROUP BY {continent_col}, {country_col}, period, {order_expr}
                )
                SELECT continent, country, period, mcm_d
                FROM period_data
            """
        
        query = text(query_str)
        df = pd.read_sql(query, conn)
        
        if df.empty:
            return pd.DataFrame()
        
        # Pivot to get periods as columns, keeping continent and country
        pivot_df = df.pivot_table(
            index=['continent', 'country'],
            columns='period',
            values='mcm_d',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        return pivot_df
        
    except Exception as e:
        return pd.DataFrame()

def fetch_destination_rolling_windows_hierarchical(conn, continent_col, country_col, where_clause, rolling_window_days=30):
    """Fetch 7D and configurable rolling window data with continent/country hierarchy."""
    from sqlalchemy import text
    from datetime import datetime, timedelta
    
    try:
        normalized_window_days = normalize_rolling_window_days(rolling_window_days)
        rolling_window_label = format_rolling_window_label(normalized_window_days)
        current_date = datetime.now().date()
        date_7d_ago = current_date - timedelta(days=7)
        date_window_ago = current_date - timedelta(days=normalized_window_days)
        date_window_y1_start = current_date - timedelta(days=365) - timedelta(days=normalized_window_days)
        date_window_y1_end = current_date - timedelta(days=365)
        
        query = text(f"""
            WITH latest_timestamp AS (
                SELECT MAX(upload_timestamp_utc) as max_ts
                FROM {DB_SCHEMA}.kpler_trades
            ),
            -- Get all unique continent/country combinations
            all_destinations AS (
                SELECT DISTINCT
                    {continent_col} as continent,
                    {country_col} as country
                FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                WHERE upload_timestamp_utc = max_ts
                    AND {where_clause}
                    AND "start" >= '{date_window_y1_start}'
                    AND "start" <= '{current_date}'
            ),
            -- Generate date series for each window
            dates_7d AS (
                SELECT generate_series(
                    '{date_7d_ago}'::date + INTERVAL '1 day',
                    '{current_date}'::date,
                    '1 day'::interval
                )::date as date
            ),
            dates_window AS (
                SELECT generate_series(
                    '{date_window_ago}'::date + INTERVAL '1 day',
                    '{current_date}'::date,
                    '1 day'::interval
                )::date as date
            ),
            dates_window_y1 AS (
                SELECT generate_series(
                    '{date_window_y1_start}'::date + INTERVAL '1 day',
                    '{date_window_y1_end}'::date,
                    '1 day'::interval
                )::date as date
            ),
            -- Create matrices for each period
            matrix_7d AS (
                SELECT d.date, a.continent, a.country
                FROM dates_7d d
                CROSS JOIN all_destinations a
            ),
            matrix_window AS (
                SELECT d.date, a.continent, a.country
                FROM dates_window d
                CROSS JOIN all_destinations a
            ),
            matrix_window_y1 AS (
                SELECT d.date, a.continent, a.country
                FROM dates_window_y1 d
                CROSS JOIN all_destinations a
            ),
            -- Get actual daily data
            daily_data AS (
                SELECT 
                    "start"::date as date,
                    {continent_col} as continent,
                    {country_col} as country,
                    SUM(cargo_origin_cubic_meters * 0.6 / 1000) as daily_mcmd
                FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                WHERE upload_timestamp_utc = max_ts
                    AND {where_clause}
                    AND "start" >= '{date_window_y1_start}'
                    AND "start" <= '{current_date}'
                GROUP BY "start"::date, {continent_col}, {country_col}
            ),
            -- Calculate averages for 7-day window with zeros for missing days
            window_7d AS (
                SELECT 
                    m.continent,
                    m.country,
                    AVG(COALESCE(d.daily_mcmd, 0)) as avg_7d
                FROM matrix_7d m
                LEFT JOIN daily_data d 
                    ON m.date = d.date 
                    AND m.continent = d.continent 
                    AND m.country = d.country
                GROUP BY m.continent, m.country
            ),
            -- Calculate averages for configurable window with zeros for missing days
            window_current AS (
                SELECT 
                    m.continent,
                    m.country,
                    AVG(COALESCE(d.daily_mcmd, 0)) as avg_window
                FROM matrix_window m
                LEFT JOIN daily_data d 
                    ON m.date = d.date 
                    AND m.continent = d.continent 
                    AND m.country = d.country
                GROUP BY m.continent, m.country
            ),
            -- Calculate averages for configurable window year ago with zeros for missing days
            window_y1 AS (
                SELECT 
                    m.continent,
                    m.country,
                    AVG(COALESCE(d.daily_mcmd, 0)) as avg_window_y1
                FROM matrix_window_y1 m
                LEFT JOIN daily_data d 
                    ON m.date = d.date 
                    AND m.continent = d.continent 
                    AND m.country = d.country
                GROUP BY m.continent, m.country
            )
            SELECT 
                COALESCE(w7.continent, wc.continent, wy1.continent) as continent,
                COALESCE(w7.country, wc.country, wy1.country) as country,
                COALESCE(w7.avg_7d, 0) as "7D",
                COALESCE(wc.avg_window, 0) as "{rolling_window_label}",
                COALESCE(wy1.avg_window_y1, 0) as "{rolling_window_label}_Y1",
                COALESCE(w7.avg_7d, 0) - COALESCE(wc.avg_window, 0) as "Δ 7D-{rolling_window_label}",
                COALESCE(wc.avg_window, 0) - COALESCE(wy1.avg_window_y1, 0) as "Δ {rolling_window_label} Y/Y"
            FROM window_7d w7
            FULL OUTER JOIN window_current wc ON w7.continent = wc.continent AND w7.country = wc.country
            FULL OUTER JOIN window_y1 wy1 ON COALESCE(w7.continent, wc.continent) = wy1.continent 
                AND COALESCE(w7.country, wc.country) = wy1.country
        """)
        df = pd.read_sql(query, conn)
        return df
        
    except Exception as e:
        return pd.DataFrame()

def prepare_destination_table_for_display(df, expanded_continents=None):
    """Prepare destination data for display with expandable continent/country rows"""
    if df.empty:
        return pd.DataFrame()
    
    expanded_continents = expanded_continents or []
    
    # Filter data based on expanded state
    filtered_rows = []
    continent_totals_for_grand = []  # Store continent totals for grand total calculation
    
    # Group by continent first
    for continent in df['continent'].unique():
        continent_data = df[df['continent'] == continent]
        
        # Calculate continent total
        numeric_cols = [col for col in df.columns if col not in ['continent', 'country']]
        continent_total = pd.DataFrame([{
            'Continent': f"▼ {continent}" if continent in expanded_continents else f"▶ {continent}",
            'Country': 'Total',
            **{col: continent_data[col].sum() for col in numeric_cols}
        }])
        
        filtered_rows.append(continent_total)
        continent_totals_for_grand.append(pd.DataFrame([{
            'continent': continent,
            **{col: continent_data[col].sum() for col in numeric_cols}
        }]))
        
        # Only show countries if continent is expanded
        if continent in expanded_continents:
            countries = continent_data.copy()
            # Indent country names for visual hierarchy
            countries.loc[:, 'country'] = "    " + countries['country']
            countries.loc[:, 'continent'] = ""  # Clear continent name for countries
            # Rename columns for display
            countries = countries.rename(columns={'continent': 'Continent', 'country': 'Country'})
            filtered_rows.append(countries)
    
    # Add GRAND TOTAL row
    if continent_totals_for_grand:
        grand_total_df = pd.concat(continent_totals_for_grand, ignore_index=True)
        
        grand_total_row = pd.DataFrame([{
            'Continent': 'GRAND TOTAL',
            'Country': '',
            **{col: grand_total_df[col].sum() for col in numeric_cols}
        }])
        
        filtered_rows.append(grand_total_row)
    
    # Combine all rows
    if filtered_rows:
        display_df = pd.concat(filtered_rows, ignore_index=True)
    else:
        display_df = pd.DataFrame()
    
    return display_df

def combine_destination_summary_data_hierarchical(quarters_df, months_df, weeks_df, rolling_df, rolling_window_days=30):
    """Combine all period data with continent/country hierarchy into final summary table."""
    from datetime import datetime
    
    try:
        rolling_window_label = format_rolling_window_label(rolling_window_days)
        # Get all unique continent/country combinations
        all_combinations = set()
        
        for df in [quarters_df, months_df, weeks_df, rolling_df]:
            if not df.empty and 'continent' in df.columns and 'country' in df.columns:
                all_combinations.update(df[['continent', 'country']].apply(tuple, axis=1))
        
        if not all_combinations:
            return pd.DataFrame()
        
        # Create base DataFrame
        result = pd.DataFrame(list(all_combinations), columns=['continent', 'country'])
        
        # Get current date for filtering completed periods
        current_date = datetime.now()
        current_quarter = (current_date.month - 1) // 3 + 1
        current_year = current_date.year
        
        # Process quarters - get last 5 completed quarters
        if not quarters_df.empty:
            quarter_cols = [col for col in quarters_df.columns if col not in ['continent', 'country']]
            
            # Filter completed quarters
            completed_quarters = []
            for col in quarter_cols:
                if "Q" in col and "'" in col:
                    q_num = int(col.split("Q")[1].split("'")[0])
                    year = int("20" + col.split("'")[1])
                    if year < current_year or (year == current_year and q_num < current_quarter):
                        completed_quarters.append(col)
            
            # Sort and take last 5
            completed_quarters = sorted(completed_quarters, 
                                      key=lambda x: (x.split("'")[1], x.split("Q")[1].split("'")[0]))[-5:]
            
            if completed_quarters:
                quarters_subset = quarters_df[['continent', 'country'] + completed_quarters]
                result = result.merge(quarters_subset, on=['continent', 'country'], how='left')
        
        # Process months - get last 3 completed months
        if not months_df.empty:
            month_cols = [col for col in months_df.columns if col not in ['continent', 'country']]
            month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                          'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            
            # Filter completed months
            completed_months = []
            for col in month_cols:
                if "'" in col:
                    month_abbr = col.split("'")[0]
                    year = int("20" + col.split("'")[1])
                    month_num = month_order.get(month_abbr, 0)
                    if year < current_year or (year == current_year and month_num < current_date.month):
                        completed_months.append(col)
            
            # Sort and take last 3
            completed_months = sorted(completed_months,
                                    key=lambda x: (x.split("'")[1], month_order.get(x.split("'")[0], 0)))[-3:]
            
            if completed_months:
                months_subset = months_df[['continent', 'country'] + completed_months]
                result = result.merge(months_subset, on=['continent', 'country'], how='left')
        
        # Add configurable rolling window column from rolling data
        if not rolling_df.empty and rolling_window_label in rolling_df.columns:
            result = result.merge(
                rolling_df[['continent', 'country', rolling_window_label]],
                on=['continent', 'country'],
                how='left'
            )
        
        # Process weeks - get last 3 completed weeks  
        if not weeks_df.empty:
            week_cols = [col for col in weeks_df.columns if col not in ['continent', 'country']]
            current_week = current_date.isocalendar()[1]
            
            # Filter completed weeks
            completed_weeks = []
            for col in week_cols:
                if "W" in col and "'" in col:
                    week_num = int(col.split("W")[1].split("'")[0])
                    year = int("20" + col.split("'")[1])
                    if year < current_year or (year == current_year and week_num < current_week):
                        completed_weeks.append(col)
            
            # Sort and take last 3
            completed_weeks = sorted(completed_weeks,
                                   key=lambda x: (x.split("'")[1], x.split("W")[1].split("'")[0].zfill(2)))[-3:]
            
            if completed_weeks:
                weeks_subset = weeks_df[['continent', 'country'] + completed_weeks]
                result = result.merge(weeks_subset, on=['continent', 'country'], how='left')
        
        # Add remaining rolling columns
        if not rolling_df.empty:
            remaining_cols = ['7D', f'Δ 7D-{rolling_window_label}', f'Δ {rolling_window_label} Y/Y']
            for col in remaining_cols:
                if col in rolling_df.columns:
                    result = result.merge(rolling_df[['continent', 'country', col]], on=['continent', 'country'], how='left')
        
        # Fill NaN values
        result = result.fillna(0)
        
        # Round numeric columns
        numeric_cols = [col for col in result.columns if col not in ['continent', 'country']]
        for col in numeric_cols:
            result[col] = result[col].round(1)
        
        return result
        
    except Exception as e:
        return pd.DataFrame()

def combine_destination_summary_data(quarters_df, months_df, weeks_df, rolling_df):
    """Combine all period data into final summary table"""
    from datetime import datetime
    
    try:
        # Start with destinations from rolling data (most complete)
        if not rolling_df.empty:
            result = rolling_df[['destination']].copy()
        else:
            # Get unique destinations from all dataframes
            all_destinations = set()
            if not quarters_df.empty:
                all_destinations.update(quarters_df['destination'].unique())
            if not months_df.empty:
                all_destinations.update(months_df['destination'].unique())
            if not weeks_df.empty:
                all_destinations.update(weeks_df['destination'].unique())
            
            if not all_destinations:
                return pd.DataFrame()
            
            result = pd.DataFrame({'destination': sorted(list(all_destinations))})
        
        # Get current date for filtering completed periods
        current_date = datetime.now()
        current_quarter = (current_date.month - 1) // 3 + 1
        current_year = current_date.year
        
        # Process quarters - get last 5 completed quarters
        if not quarters_df.empty:
            quarter_cols = [col for col in quarters_df.columns if col != 'destination']
            
            # Filter completed quarters
            completed_quarters = []
            for col in quarter_cols:
                if "Q" in col and "'" in col:
                    q_num = int(col.split("Q")[1].split("'")[0])
                    year = int("20" + col.split("'")[1])
                    if year < current_year or (year == current_year and q_num < current_quarter):
                        completed_quarters.append(col)
            
            # Sort and take last 5
            completed_quarters = sorted(completed_quarters, 
                                      key=lambda x: (x.split("'")[1], x.split("Q")[1].split("'")[0]))[-5:]
            
            # Merge selected quarters
            if completed_quarters:
                quarters_subset = quarters_df[['destination'] + completed_quarters]
                result = result.merge(quarters_subset, on='destination', how='left')
        
        # Process months - get last 3 completed months
        if not months_df.empty:
            month_cols = [col for col in months_df.columns if col != 'destination']
            month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                          'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            
            # Filter completed months
            completed_months = []
            for col in month_cols:
                if "'" in col:
                    month_abbr = col.split("'")[0]
                    year = int("20" + col.split("'")[1])
                    month_num = month_order.get(month_abbr, 0)
                    if year < current_year or (year == current_year and month_num < current_date.month):
                        completed_months.append(col)
            
            # Sort and take last 3
            completed_months = sorted(completed_months,
                                    key=lambda x: (x.split("'")[1], month_order.get(x.split("'")[0], 0)))[-3:]
            
            # Merge selected months
            if completed_months:
                months_subset = months_df[['destination'] + completed_months]
                result = result.merge(months_subset, on='destination', how='left')
        
        # Add 30D column from rolling data
        if not rolling_df.empty and '30D' in rolling_df.columns:
            result = result.merge(rolling_df[['destination', '30D']], on='destination', how='left')
        
        # Process weeks - get last 3 completed weeks
        if not weeks_df.empty:
            week_cols = [col for col in weeks_df.columns if col != 'destination']
            current_week = current_date.isocalendar()[1]
            
            # Filter completed weeks
            completed_weeks = []
            for col in week_cols:
                if "W" in col and "'" in col:
                    week_num = int(col.split("W")[1].split("'")[0])
                    year = int("20" + col.split("'")[1])
                    if year < current_year or (year == current_year and week_num < current_week):
                        completed_weeks.append(col)
            
            # Sort and take last 3
            completed_weeks = sorted(completed_weeks,
                                   key=lambda x: (x.split("'")[1], x.split("W")[1].split("'")[0].zfill(2)))[-3:]
            
            # Merge selected weeks
            if completed_weeks:
                weeks_subset = weeks_df[['destination'] + completed_weeks]
                result = result.merge(weeks_subset, on='destination', how='left')
        
        # Add remaining rolling columns (7D and deltas)
        if not rolling_df.empty:
            remaining_cols = ['7D', 'Δ 7D-30D', 'Δ 30D Y/Y']
            for col in remaining_cols:
                if col in rolling_df.columns:
                    result = result.merge(rolling_df[['destination', col]], on='destination', how='left')
        
        # Add TOTAL row
        numeric_cols = [col for col in result.columns if col != 'destination']
        total_row = pd.DataFrame([{
            'destination': 'TOTAL',
            **{col: result[col].sum() for col in numeric_cols}
        }])
        result = pd.concat([result, total_row], ignore_index=True)
        
        # Fill NaN values and round
        result = result.fillna(0)
        for col in numeric_cols:
            result[col] = result[col].round(1)
        
        return result
        
    except Exception as e:
        return pd.DataFrame()

def fetch_country_supply_chart_data(country_name, rolling_window_days=30):
    """Fetch seasonal comparison data for selected country's LNG supply."""
    normalized_window_days = normalize_rolling_window_days(rolling_window_days)
    preceding_days = normalized_window_days - 1

    query = f"""
        WITH latest_data AS (
            SELECT MAX(upload_timestamp_utc) as max_timestamp
            FROM {DB_SCHEMA}.kpler_trades
        ),
        -- Get all unique continents for this country
        all_continents AS (
            SELECT DISTINCT
                COALESCE(NULLIF(continent_destination_name, ''), 'Unknown') as continent_destination
            FROM {DB_SCHEMA}.kpler_trades kt
            , latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name = %(country_name)s
                AND kt.start >= '2023-11-01'
        ),
        -- Get all dates in our range
        all_dates AS (
            SELECT generate_series(
                '2023-11-01'::date,
                (CURRENT_DATE + INTERVAL '14 days')::date,
                '1 day'::interval
            )::date as date
        ),
        -- Create complete date/continent matrix
        date_continent_matrix AS (
            SELECT
                d.date,
                c.continent_destination
            FROM all_dates d
            CROSS JOIN all_continents c
        ),
        -- Get actual daily exports
        daily_exports_raw AS (
            SELECT
                kt.start::date as date,
                COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination,
                SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
            FROM {DB_SCHEMA}.kpler_trades kt
            , latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name = %(country_name)s
                AND kt.start >= '2023-11-01'
                AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
            GROUP BY kt.start::date, COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown')
        ),
        -- Join to get complete dataset with zeros for missing data
        daily_exports_complete AS (
            SELECT
                dcm.date,
                dcm.continent_destination,
                COALESCE(der.daily_export_mcmd, 0) as daily_export_mcmd
            FROM date_continent_matrix dcm
            LEFT JOIN daily_exports_raw der
                ON dcm.date = der.date
                AND dcm.continent_destination = der.continent_destination
        ),
        -- Sum across all continents for total daily supply
        daily_supply AS (
            SELECT
                date,
                SUM(daily_export_mcmd) as mcmd
            FROM daily_exports_complete
            GROUP BY date
        ),
        rolling_supply AS (
            SELECT
                date,
                mcmd,
                AVG(mcmd) OVER (
                    ORDER BY date
                    ROWS BETWEEN {preceding_days} PRECEDING AND CURRENT ROW
                ) as rolling_avg,
                CASE
                    WHEN date > CURRENT_DATE THEN true
                    ELSE false
                END as is_forecast
            FROM daily_supply
        )
        SELECT
            date,
            EXTRACT(YEAR FROM date) as year,
            EXTRACT(DOY FROM date) as day_of_year,
            TO_CHAR(date, 'Mon DD') as month_day,
            rolling_avg,
            is_forecast
        FROM rolling_supply
        WHERE date >= '2024-01-01'
        ORDER BY date
        """

    return pd.read_sql(query, engine, params={'country_name': country_name})


def fetch_continent_destination_chart_data(country_name, rolling_window_days=30):
    """Fetch continent destination seasonal comparison data."""
    normalized_window_days = normalize_rolling_window_days(rolling_window_days)
    preceding_days = normalized_window_days - 1

    query = f"""
        WITH latest_data AS (
            SELECT MAX(upload_timestamp_utc) as max_timestamp
            FROM {DB_SCHEMA}.kpler_trades
        ),
        -- Get all unique continents that have ever had exports from this country
        all_continents AS (
            SELECT DISTINCT
                COALESCE(NULLIF(continent_destination_name, ''), 'Unknown') as continent_destination
            FROM {DB_SCHEMA}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name = %(country_name)s
                AND kt.start >= '2023-11-01'
        ),
        -- Get all dates in our range
        all_dates AS (
            SELECT generate_series(
                '2023-11-01'::date,
                (CURRENT_DATE + INTERVAL '14 days')::date,
                '1 day'::interval
            )::date as date
        ),
        -- Create complete date/continent matrix
        date_continent_matrix AS (
            SELECT
                d.date,
                c.continent_destination
            FROM all_dates d
            CROSS JOIN all_continents c
        ),
        -- Get actual daily exports
        daily_exports_raw AS (
            SELECT
                kt.start::date as date,
                COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination,
                SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
            FROM {DB_SCHEMA}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name = %(country_name)s
                AND kt.start >= '2023-11-01'
                AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
            GROUP BY kt.start::date, kt.continent_destination_name
        ),
        -- Join to get complete dataset with zeros for missing data
        daily_exports_complete AS (
            SELECT
                dcm.date,
                dcm.continent_destination,
                COALESCE(der.daily_export_mcmd, 0) as daily_export_mcmd
            FROM date_continent_matrix dcm
            LEFT JOIN daily_exports_raw der
                ON dcm.date = der.date
                AND dcm.continent_destination = der.continent_destination
        ),
        -- Calculate rolling averages on complete dataset
        rolling_exports AS (
            SELECT
                date,
                continent_destination,
                daily_export_mcmd,
                AVG(daily_export_mcmd) OVER (
                    PARTITION BY continent_destination
                    ORDER BY date
                    ROWS BETWEEN {preceding_days} PRECEDING AND CURRENT ROW
                ) as rolling_avg,
                CASE
                    WHEN date > CURRENT_DATE THEN true
                    ELSE false
                END as is_forecast
            FROM daily_exports_complete
        )
        SELECT
            date,
            continent_destination,
            EXTRACT(YEAR FROM date) as year,
            EXTRACT(DOY FROM date) as day_of_year,
            TO_CHAR(date, 'Mon DD') as month_day,
            rolling_avg,
            is_forecast
        FROM rolling_exports
        WHERE date >= '2024-01-01'
        ORDER BY continent_destination, date
        """

    return pd.read_sql(query, engine, params={'country_name': country_name})


def fetch_continent_percentage_chart_data(country_name, rolling_window_days=30):
    """Fetch continent market share seasonal comparison data."""
    normalized_window_days = normalize_rolling_window_days(rolling_window_days)
    preceding_days = normalized_window_days - 1

    query = f"""
        WITH latest_data AS (
            SELECT MAX(upload_timestamp_utc) as max_timestamp
            FROM {DB_SCHEMA}.kpler_trades
        ),
        -- Get all unique continents that have ever had exports from this country
        all_continents AS (
            SELECT DISTINCT
                COALESCE(NULLIF(continent_destination_name, ''), 'Unknown') as continent_destination
            FROM {DB_SCHEMA}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name = %(country_name)s
                AND kt.start >= '2023-11-01'
        ),
        -- Get all dates in our range
        all_dates AS (
            SELECT generate_series(
                '2023-11-01'::date,
                (CURRENT_DATE + INTERVAL '14 days')::date,
                '1 day'::interval
            )::date as date
        ),
        -- Create complete date/continent matrix
        date_continent_matrix AS (
            SELECT
                d.date,
                c.continent_destination
            FROM all_dates d
            CROSS JOIN all_continents c
        ),
        -- Get actual daily exports
        daily_exports_raw AS (
            SELECT
                kt.start::date as date,
                COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination,
                SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
            FROM {DB_SCHEMA}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name = %(country_name)s
                AND kt.start >= '2023-11-01'
                AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
            GROUP BY kt.start::date, kt.continent_destination_name
        ),
        -- Join to get complete dataset with zeros for missing data
        daily_exports_complete AS (
            SELECT
                dcm.date,
                dcm.continent_destination,
                COALESCE(der.daily_export_mcmd, 0) as daily_export_mcmd
            FROM date_continent_matrix dcm
            LEFT JOIN daily_exports_raw der
                ON dcm.date = der.date
                AND dcm.continent_destination = der.continent_destination
        ),
        -- Calculate rolling averages on complete dataset
        rolling_continents AS (
            SELECT
                date,
                continent_destination,
                daily_export_mcmd,
                AVG(daily_export_mcmd) OVER (
                    PARTITION BY continent_destination
                    ORDER BY date
                    ROWS BETWEEN {preceding_days} PRECEDING AND CURRENT ROW
                ) as rolling_avg,
                CASE
                    WHEN date > CURRENT_DATE THEN true
                    ELSE false
                END as is_forecast
            FROM daily_exports_complete
        ),
        -- Sum rolling averages for total
        rolling_totals AS (
            SELECT
                date,
                SUM(rolling_avg) as total_rolling_avg
            FROM rolling_continents
            GROUP BY date
        )
        SELECT
            rc.date,
            rc.continent_destination,
            EXTRACT(YEAR FROM rc.date) as year,
            EXTRACT(DOY FROM rc.date) as day_of_year,
            TO_CHAR(rc.date, 'Mon DD') as month_day,
            rc.rolling_avg,
            CASE
                WHEN rt.total_rolling_avg > 0
                THEN (rc.rolling_avg / rt.total_rolling_avg) * 100
                ELSE 0
            END as percentage,
            rc.is_forecast
        FROM rolling_continents rc
        JOIN rolling_totals rt ON rc.date = rt.date
        WHERE rc.date >= '2024-01-01'
        ORDER BY rc.continent_destination, rc.date
        """

    return pd.read_sql(query, engine, params={'country_name': country_name})


def deduplicate_woodmac_monthly_forecast_data(monthly_df):
    """Keep one monthly WoodMac forecast row per month, preferring short-term data over long-term."""
    expected_columns = ['start_date', 'metric_value', 'source']
    if monthly_df is None or monthly_df.empty:
        return pd.DataFrame(columns=expected_columns)

    deduped_df = monthly_df.copy()
    if 'source' not in deduped_df.columns:
        deduped_df['source'] = 'WoodMac'

    deduped_df['start_date'] = pd.to_datetime(deduped_df['start_date'], errors='coerce').dt.normalize()
    deduped_df['metric_value'] = pd.to_numeric(deduped_df['metric_value'], errors='coerce')
    deduped_df = deduped_df[
        deduped_df['start_date'].notna() & deduped_df['metric_value'].notna()
    ][['start_date', 'metric_value', 'source']].copy()
    if deduped_df.empty:
        return pd.DataFrame(columns=expected_columns)

    deduped_df['source'] = deduped_df['source'].fillna('WoodMac').astype(str)
    deduped_df = deduped_df.groupby(['start_date', 'source'], as_index=False)['metric_value'].sum()
    source_priority = {'Short Term': 0, 'Long Term': 1}
    deduped_df['source_priority'] = deduped_df['source'].map(source_priority).fillna(99)
    deduped_df = deduped_df.sort_values(['start_date', 'source_priority', 'source'])
    deduped_df = deduped_df.drop_duplicates(subset=['start_date'], keep='first')
    deduped_df = deduped_df.drop(columns=['source_priority']).reset_index(drop=True)
    return deduped_df


def expand_woodmac_monthly_forecast_to_daily(monthly_df):
    """Expand monthly WoodMac MMTPA values into flat daily mcm/d rows for the full month."""
    expected_columns = ['date', 'year', 'day_of_year', 'month_day', 'mcmd', 'is_forecast', 'source']
    deduped_df = deduplicate_woodmac_monthly_forecast_data(monthly_df)
    if deduped_df.empty:
        return pd.DataFrame(columns=expected_columns)

    daily_frames = []
    for row in deduped_df.itertuples(index=False):
        start_date = pd.Timestamp(row.start_date).normalize()
        month_end = start_date + pd.offsets.MonthEnd(0)
        daily_dates = pd.date_range(start_date, month_end, freq='D')
        days_in_month = len(daily_dates)
        daily_mcmd = (
            row.metric_value
            * WOODMAC_LNG_CUBIC_METERS_PER_MMTPA_MONTH
            * MCM_PER_CUBIC_METER
            / days_in_month
        )
        daily_frames.append(pd.DataFrame({
            'date': daily_dates,
            'year': daily_dates.year.astype(int),
            'day_of_year': daily_dates.dayofyear.astype(int),
            'month_day': daily_dates.strftime('%b %d'),
            'mcmd': daily_mcmd,
            'is_forecast': True,
            'source': row.source
        }))

    return pd.concat(daily_frames, ignore_index=True)


def filter_woodmac_forecast_horizon(forecast_df, current_date=None):
    """Limit WoodMac forecast rows to the current year plus the next two calendar years."""
    expected_columns = ['date', 'year', 'day_of_year', 'month_day', 'mcmd', 'is_forecast', 'source']
    if forecast_df is None or forecast_df.empty:
        return pd.DataFrame(columns=expected_columns)

    reference_date = pd.Timestamp(current_date or dt.date.today()).normalize()
    max_year = reference_date.year + WOODMAC_FORECAST_YEARS_AHEAD
    filtered_df = forecast_df.copy()
    filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
    filtered_df = filtered_df[filtered_df['date'].notna()].copy()
    if filtered_df.empty:
        return pd.DataFrame(columns=expected_columns)

    filtered_df = filtered_df[
        (filtered_df['date'] >= pd.Timestamp(reference_date.year, reference_date.month, 1)) &
        (filtered_df['date'].dt.year <= max_year)
    ].copy()
    if filtered_df.empty:
        return pd.DataFrame(columns=expected_columns)

    filtered_df['year'] = filtered_df['date'].dt.year.astype(int)
    filtered_df['day_of_year'] = filtered_df['date'].dt.dayofyear.astype(int)
    filtered_df['month_day'] = filtered_df['date'].dt.strftime('%b %d')
    if 'is_forecast' not in filtered_df.columns:
        filtered_df['is_forecast'] = True
    if 'source' not in filtered_df.columns:
        filtered_df['source'] = 'WoodMac'
    return filtered_df[expected_columns].reset_index(drop=True)


def _hex_to_rgba(color, alpha):
    if not isinstance(color, str) or not color.startswith('#') or len(color) != 7:
        return color
    return f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {alpha})"


def _apply_time_series_chart_layout(fig, yaxis_title):
    fig.update_layout(
        xaxis=dict(
            title='',
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
        yaxis=dict(
            title=yaxis_title,
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            gridwidth=0.5,
            linecolor='#CCCCCC',
            linewidth=1,
            tickfont=dict(size=11, color='#666666'),
            zeroline=False
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.12,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0)',
            bordercolor='rgba(255,255,255,0)',
            borderwidth=0,
            font=dict(size=10, color='#4A4A4A'),
            itemsizing='constant'
        ),
        height=400,
        margin=dict(l=55, r=40, t=30, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='x unified',
        title=None
    )
    return fig


def _empty_timeseries_chart(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref='paper',
        yref='paper',
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color='#6b7280')
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        height=400,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig


def fetch_woodmac_country_export_forecast_data(origin_country):
    """Fetch WoodMac monthly export forecasts for a single origin country and expand to daily mcm/d values."""
    if not origin_country:
        return pd.DataFrame(columns=['date', 'year', 'day_of_year', 'month_day', 'mcmd', 'is_forecast', 'source'])

    # Resolve WoodMac country name alias (e.g. "Russian Federation" → "Russia")
    woodmac_country_name = origin_country
    try:
        from sqlalchemy import text as sa_text
        alias_q = sa_text("""
            SELECT country_name
            FROM at_lng.mappings_country
            WHERE country = :country
              AND country_name IS NOT NULL
            LIMIT 1
        """)
        with engine.connect() as conn:
            result = conn.execute(alias_q, {'country': origin_country}).fetchone()
        if result and result[0]:
            woodmac_country_name = result[0]
    except Exception:
        pass

    market_outlook_order_expr = """
        TO_DATE(
            (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{4})'))[1]
            || ' ' ||
            (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{4})'))[2],
            'Month YYYY'
        ) DESC,
        MAX(publication_date) DESC
    """
    from sqlalchemy import text as sa_text, bindparam
    query = sa_text(f"""
        WITH latest_short_term AS (
            SELECT
                start_date::date AS start_date,
                SUM(metric_value) AS metric_value,
                'Short Term' AS source
            FROM {WOODMAC_IMPORT_EXPORTS_TABLE}
            WHERE market_outlook = (
                SELECT market_outlook
                FROM {WOODMAC_IMPORT_EXPORTS_TABLE}
                WHERE release_type = 'Short Term Outlook'
                GROUP BY market_outlook
                ORDER BY {market_outlook_order_expr}
                LIMIT 1
            )
                AND release_type = 'Short Term Outlook'
                AND direction = 'Export'
                AND measured_at = 'Exit'
                AND metric_name = 'Flow'
                AND country_name = :woodmac_country_name
                AND start_date::date >= DATE_TRUNC('month', CURRENT_DATE)::date
                AND start_date::date < (DATE_TRUNC('year', CURRENT_DATE) + INTERVAL '3 years')::date
            GROUP BY start_date::date
            HAVING SUM(metric_value) > 0
        ),
        short_term_max_date AS (
            SELECT MAX(start_date) AS max_date
            FROM latest_short_term
        ),
        latest_long_term_raw AS (
            SELECT
                start_date::date AS start_date,
                SUM(metric_value) AS metric_value,
                'Long Term' AS source
            FROM {WOODMAC_IMPORT_EXPORTS_TABLE}
            WHERE market_outlook = (
                SELECT market_outlook
                FROM {WOODMAC_IMPORT_EXPORTS_TABLE}
                WHERE release_type = 'Long Term Outlook'
                GROUP BY market_outlook
                ORDER BY {market_outlook_order_expr}
                LIMIT 1
            )
                AND release_type = 'Long Term Outlook'
                AND direction = 'Export'
                AND measured_at = 'Exit'
                AND metric_name = 'Flow'
                AND country_name = :woodmac_country_name
                AND start_date::date >= DATE_TRUNC('month', CURRENT_DATE)::date
                AND start_date::date < (DATE_TRUNC('year', CURRENT_DATE) + INTERVAL '3 years')::date
            GROUP BY start_date::date
            HAVING SUM(metric_value) > 0
        ),
        latest_long_term AS (
            SELECT *
            FROM latest_long_term_raw
            WHERE (SELECT max_date FROM short_term_max_date) IS NULL
                OR start_date > (SELECT max_date FROM short_term_max_date)
        ),
        combined AS (
            SELECT * FROM latest_short_term
            UNION ALL
            SELECT * FROM latest_long_term
        )
        SELECT
            start_date,
            metric_value,
            source
        FROM combined
        ORDER BY start_date
    """)
    monthly_df = pd.read_sql(query, engine, params={'woodmac_country_name': woodmac_country_name})
    forecast_df = expand_woodmac_monthly_forecast_to_daily(monthly_df)
    return filter_woodmac_forecast_horizon(forecast_df)


def _create_total_export_chart_with_woodmac_forecast(historical_df, forecast_df):
    forecast_df = filter_woodmac_forecast_horizon(forecast_df)
    if historical_df.empty and forecast_df.empty:
        return _empty_timeseries_chart("No data available")

    fig = go.Figure()
    chart_colors = ['#2E86C1', '#1B4F72', '#5DADE2', '#3498DB', '#76D7C4']
    all_years = sorted(set(historical_df.get('year', pd.Series(dtype=int)).dropna().astype(int).tolist()) |
                       set(forecast_df.get('year', pd.Series(dtype=int)).dropna().astype(int).tolist()))
    color_map = {
        year: chart_colors[idx % len(chart_colors)]
        for idx, year in enumerate(all_years)
    }
    latest_historical_year = (
        int(historical_df['year'].dropna().max())
        if not historical_df.empty and historical_df['year'].notna().any()
        else None
    )

    for year in sorted(historical_df['year'].dropna().unique()):
        year = int(year)
        year_data = historical_df[historical_df['year'] == year].copy().sort_values('date')
        if year_data.empty:
            continue
        year_data['plot_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(
            year_data['day_of_year'] - 1,
            unit='d'
        )
        base_color = color_map.get(year, chart_colors[0])
        line_width = 3 if year == latest_historical_year else 2

        actual_data = year_data[~year_data['is_forecast']] if 'is_forecast' in year_data.columns else year_data
        kpler_fc_data = year_data[year_data['is_forecast']] if 'is_forecast' in year_data.columns else pd.DataFrame()

        fig.add_trace(go.Scatter(
            x=actual_data['plot_date'],
            y=actual_data['rolling_avg'],
            mode='lines',
            name=str(year),
            line=dict(color=base_color, width=line_width),
            text=actual_data['month_day'],
            hovertemplate=(
                f'<b>{year} (Historical)</b><br>%{{text}}'
                '<br>Exports: %{y:.1f} mcm/d<extra></extra>'
            )
        ))

        if not kpler_fc_data.empty:
            connect_data = pd.concat([actual_data.tail(1), kpler_fc_data])
            fig.add_trace(go.Scatter(
                x=connect_data['plot_date'],
                y=connect_data['rolling_avg'],
                mode='lines',
                name=f'{year} Kpler Forecast',
                line=dict(color=_hex_to_rgba(base_color, 0.5), width=line_width, dash='dot'),
                opacity=0.8,
                text=connect_data['month_day'],
                hovertemplate=(
                    f'<b>{year} (Kpler Forecast)</b><br>%{{text}}'
                    '<br>Exports: %{y:.1f} mcm/d<extra></extra>'
                ),
                showlegend=False
            ))

    forecast_years = sorted(forecast_df.get('year', pd.Series(dtype=int)).dropna().astype(int).unique().tolist())
    current_year = dt.date.today().year
    default_visible_forecast_year = (
        current_year if current_year in forecast_years else (forecast_years[0] if forecast_years else None)
    )
    for year in forecast_years:
        year_data = forecast_df[forecast_df['year'] == year].copy().sort_values('date')
        if year_data.empty:
            continue
        year_data['plot_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(
            year_data['day_of_year'] - 1,
            unit='d'
        )
        base_color = color_map.get(year, chart_colors[0])
        fig.add_trace(go.Scatter(
            x=year_data['plot_date'],
            y=year_data['mcmd'],
            mode='lines',
            name=f'{year} WoodMac Forecast',
            line=dict(
                color=_hex_to_rgba(base_color, 0.5),
                width=3 if year == default_visible_forecast_year else 2,
                dash='dash'
            ),
            opacity=0.85,
            text=year_data['month_day'],
            customdata=year_data['source'],
            hovertemplate=(
                f'<b>{year} WoodMac Forecast</b><br>%{{text}}'
                '<br>Exports: %{y:.1f} mcm/d'
                '<br>Source: %{customdata}<extra></extra>'
            ),
            visible=True if year == default_visible_forecast_year else 'legendonly'
        ))

    return _apply_time_series_chart_layout(fig, 'mcm/d')


# ─── Supply Allocation helpers (Destination Forecast table) ──────────────────

def get_origin_forecast_period_config(current_date=None):
    """Return monthly and annual period definitions for the WoodMac destination forecast table."""
    current_ts = pd.Timestamp(current_date or dt.datetime.now()).normalize()
    current_month_start = current_ts.replace(day=1)
    current_year = current_month_start.year
    month_starts = pd.date_range(
        start=current_month_start,
        end=pd.Timestamp(year=current_year, month=12, day=1),
        freq='MS'
    )
    annual_years = [current_year + offset for offset in range(1, WOODMAC_FORECAST_YEARS_AHEAD + 1)]
    ordered_labels = [month.strftime("%b'%y") for month in month_starts]
    ordered_labels.extend([f"{year} Avg" for year in annual_years])
    return {
        'current_date': current_ts,
        'current_month_start': current_month_start,
        'current_year': current_year,
        'month_starts': month_starts,
        'annual_years': annual_years,
        'ordered_labels': ordered_labels,
        'horizon_end': pd.Timestamp(year=current_year + WOODMAC_FORECAST_YEARS_AHEAD, month=12, day=31),
    }


def build_supply_allocation_country_alias_lookup(mapping_df):
    """Create alias rows so WoodMac/Kpler country naming variants map to one display country and continent."""
    extra_cols = [c for c in ['basin', 'shipping_region', 'subcontinent', 'country_classification_level1', 'country_classification']
                  if c in (mapping_df.columns if mapping_df is not None else [])]
    base_cols = ['alias', 'country_display', 'continent']
    all_cols = base_cols + extra_cols

    if mapping_df is None or mapping_df.empty:
        return pd.DataFrame(columns=all_cols)

    alias_frames = []
    for alias_col in ['country', 'country_name']:
        if alias_col not in mapping_df.columns:
            continue

        cols_to_select = [alias_col, 'continent'] + extra_cols
        if alias_col == 'country':
            cols_to_select = [alias_col, 'country_name', 'continent'] + extra_cols
            alias_df = mapping_df[cols_to_select].copy()
            alias_df = alias_df.rename(columns={'country': 'alias', 'country_name': 'country_display'})
        else:
            alias_df = mapping_df[cols_to_select].copy()
            alias_df = alias_df.rename(columns={'country_name': 'alias'})
            alias_df['country_display'] = alias_df['alias']

        alias_df = alias_df[alias_df['alias'].notna()].copy()
        alias_df['alias'] = alias_df['alias'].astype(str).str.strip()
        alias_df = alias_df[alias_df['alias'] != '']
        alias_frames.append(alias_df)

    if not alias_frames:
        return pd.DataFrame(columns=all_cols)

    alias_lookup = pd.concat(alias_frames, ignore_index=True)
    alias_lookup['country_display'] = alias_lookup['country_display'].replace('', np.nan)
    alias_lookup['country_display'] = alias_lookup['country_display'].fillna(alias_lookup['alias'])
    alias_lookup['continent'] = alias_lookup['continent'].replace('', np.nan).fillna('Unknown')
    for col in extra_cols:
        alias_lookup[col] = alias_lookup[col].replace('', np.nan).fillna('Unknown')
    alias_lookup = alias_lookup.drop_duplicates(subset=['alias'], keep='first')
    return alias_lookup[all_cols]


def fetch_latest_supply_allocation_run_metadata(engine):
    """Return the latest compatible monthly country-level base-view split-by-contract allocation run."""
    from sqlalchemy import text as sa_text
    query = sa_text(f"""
        SELECT
            run_id,
            analysis_date,
            forecast_start,
            forecast_end,
            supply_scenario,
            split_by_contract,
            woodmac_short_term_outlook,
            woodmac_long_term_outlook
        FROM {SUPPLY_ALLOCATION_RUNS_TABLE}
        WHERE aggregation_level = 'monthly'
            AND origin_aggregation = 'country_name'
            AND destination_aggregation = 'country_name'
            AND split_by_contract = TRUE
            AND supply_scenario = 'base_view'
        ORDER BY analysis_date DESC, id DESC
        LIMIT 1
    """)
    run_df = pd.read_sql(query, engine)
    if run_df.empty:
        return None

    return run_df.iloc[0].to_dict()


def format_supply_allocation_run_subtitle(run_metadata):
    """Build the subtitle shown above the SQL-backed WoodMac destination forecast table."""
    if not run_metadata:
        return "No compatible WoodMac supply-allocation SQL run is currently available."

    analysis_date = pd.to_datetime(run_metadata.get('analysis_date'), errors='coerce')
    forecast_start = pd.to_datetime(run_metadata.get('forecast_start'), errors='coerce')
    forecast_end = pd.to_datetime(run_metadata.get('forecast_end'), errors='coerce')
    parts = ["Modeled destination allocation from SQL outputs"]

    if pd.notna(analysis_date):
        parts.append(f"Run: {analysis_date.strftime('%Y-%m-%d %H:%M UTC')}")
    if run_metadata.get('supply_scenario'):
        parts.append(f"Scenario: {run_metadata['supply_scenario']}")
    if pd.notna(forecast_start) and pd.notna(forecast_end):
        parts.append(
            f"Forecast Range: {forecast_start.strftime('%b %Y')} - {forecast_end.strftime('%b %Y')}"
        )
    if run_metadata.get('woodmac_short_term_outlook'):
        parts.append(f"ST: {run_metadata['woodmac_short_term_outlook']}")
    if run_metadata.get('woodmac_long_term_outlook'):
        parts.append(f"LT: {run_metadata['woodmac_long_term_outlook']}")

    return " | ".join(parts)


def build_destination_forecast_period_table(df, value_col, group_cols, current_date=None):
    """Convert monthly BCM data into current-year monthly and next-two-years annual-average mcm/d columns."""
    period_config = get_origin_forecast_period_config(current_date)
    ordered_labels = period_config['ordered_labels']

    if df is None or df.empty:
        return pd.DataFrame(columns=group_cols + ordered_labels)

    working_df = df.copy()
    working_df['date'] = pd.to_datetime(working_df['date'], errors='coerce')
    working_df = working_df[working_df['date'].notna()].copy()
    if working_df.empty:
        return pd.DataFrame(columns=group_cols + ordered_labels)

    working_df = working_df[
        (working_df['date'] >= period_config['current_month_start']) &
        (working_df['date'] <= period_config['horizon_end'])
    ].copy()
    if working_df.empty:
        return pd.DataFrame(columns=group_cols + ordered_labels)

    period_frames = []

    monthly_df = working_df[working_df['date'].dt.year == period_config['current_year']].copy()
    if not monthly_df.empty:
        monthly_df['period_label'] = monthly_df['date'].dt.strftime("%b'%y")
        monthly_df['period_value'] = (
            monthly_df[value_col].astype(float) * 1000 / monthly_df['date'].dt.days_in_month
        )
        monthly_summary = monthly_df.groupby(group_cols + ['period_label'], as_index=False)['period_value'].sum()
        period_frames.append(monthly_summary)

    annual_df = working_df[working_df['date'].dt.year.isin(period_config['annual_years'])].copy()
    if not annual_df.empty:
        annual_df['forecast_year'] = annual_df['date'].dt.year.astype(int)
        annual_summary = annual_df.groupby(group_cols + ['forecast_year'], as_index=False)[value_col].sum()
        annual_summary['period_label'] = annual_summary['forecast_year'].map(lambda year: f"{year} Avg")
        annual_summary['period_value'] = annual_summary.apply(
            lambda row: (
                float(row[value_col]) * 1000 /
                (366 if calendar.isleap(int(row['forecast_year'])) else 365)
            ),
            axis=1
        )
        period_frames.append(annual_summary[group_cols + ['period_label', 'period_value']])

    if not period_frames:
        return pd.DataFrame(columns=group_cols + ordered_labels)

    period_values_df = pd.concat(period_frames, ignore_index=True)
    pivot_df = period_values_df.pivot_table(
        index=group_cols,
        columns='period_label',
        values='period_value',
        aggfunc='sum'
    ).reset_index()

    for column in ordered_labels:
        if column not in pivot_df.columns:
            pivot_df[column] = np.nan

    return pivot_df[group_cols + ordered_labels]


def build_destination_forecast_total_values(df, value_col, current_date=None):
    """Return period-value totals for one monthly BCM series."""
    period_config = get_origin_forecast_period_config(current_date)
    total_table = build_destination_forecast_period_table(
        pd.DataFrame(df).assign(_metric='Total'),
        value_col,
        ['_metric'],
        current_date=current_date
    )
    if total_table.empty:
        return {label: None for label in period_config['ordered_labels']}

    row = total_table.iloc[0]
    totals = {}
    for label in period_config['ordered_labels']:
        value = row.get(label)
        totals[label] = None if pd.isna(value) else round(float(value), 1)
    return totals


def prepare_destination_forecast_table_for_display(df, expanded_continents=None, footer_rows=None):
    """Prepare the WoodMac destination forecast table with expandable continents and footer totals."""
    footer_rows = footer_rows or []
    if df.empty and not footer_rows:
        return pd.DataFrame()

    expanded_continents = expanded_continents or []
    numeric_cols = []
    if not df.empty:
        numeric_cols.extend([col for col in df.columns if col not in ['continent', 'country']])
    if footer_rows:
        footer_numeric_cols = [
            col for col in pd.DataFrame(footer_rows).columns
            if col not in ['Continent', 'Country']
        ]
        for col in footer_numeric_cols:
            if col not in numeric_cols:
                numeric_cols.append(col)

    filtered_rows = []
    continent_totals_for_grand = []

    if not df.empty:
        for continent in df['continent'].dropna().unique():
            continent_data = df[df['continent'] == continent].copy()
            continent_total = {'Continent': f"▼ {continent}" if continent in expanded_continents else f"▶ {continent}",
                               'Country': 'Total'}
            for col in numeric_cols:
                continent_total[col] = continent_data[col].sum(min_count=1) if col in continent_data.columns else np.nan
            filtered_rows.append(pd.DataFrame([continent_total]))

            grand_total_row = {'continent': continent}
            for col in numeric_cols:
                grand_total_row[col] = continent_data[col].sum(min_count=1) if col in continent_data.columns else np.nan
            continent_totals_for_grand.append(pd.DataFrame([grand_total_row]))

            if continent in expanded_continents:
                countries = continent_data.copy()
                countries.loc[:, 'country'] = "    " + countries['country']
                countries.loc[:, 'continent'] = ""
                filtered_rows.append(countries.rename(columns={'continent': 'Continent', 'country': 'Country'}))

    if continent_totals_for_grand:
        grand_total_df = pd.concat(continent_totals_for_grand, ignore_index=True)
        filtered_rows.append(pd.DataFrame([{
            'Continent': 'GRAND TOTAL',
            'Country': '',
            **{col: grand_total_df[col].sum(min_count=1) for col in numeric_cols}
        }]))

    if footer_rows:
        footer_df = pd.DataFrame(footer_rows)
        for col in numeric_cols:
            if col not in footer_df.columns:
                footer_df[col] = np.nan
        filtered_rows.append(footer_df[['Continent', 'Country'] + numeric_cols])

    if not filtered_rows:
        return pd.DataFrame(columns=['Continent', 'Country'] + numeric_cols)

    display_df = pd.concat(filtered_rows, ignore_index=True)
    for col in numeric_cols:
        numeric_series = pd.to_numeric(display_df[col], errors='coerce').round(1)
        display_df[col] = numeric_series.where(pd.notnull(numeric_series), None)

    return display_df


def fetch_destination_forecast_summary_data(engine, origin_country, current_date=None, destination_level='destination_country_name'):
    """Fetch WoodMac supply allocation data for the selected exporter origin country, grouped by destination."""
    if not origin_country:
        return pd.DataFrame(), [], None

    from sqlalchemy import text as sa_text, bindparam

    run_metadata = fetch_latest_supply_allocation_run_metadata(engine)
    if not run_metadata:
        return pd.DataFrame(), [], None

    period_config = get_origin_forecast_period_config(current_date)
    mappings_query = sa_text(f"""
        SELECT DISTINCT
            country,
            country_name,
            continent,
            basin,
            shipping_region,
            subcontinent,
            country_classification_level1,
            country_classification
        FROM {DB_SCHEMA}.mappings_country
        WHERE country IS NOT NULL
    """)
    mapping_df = pd.read_sql(mappings_query, engine)

    # Resolve origin country aliases
    origin_aliases = set([origin_country])
    if not mapping_df.empty and 'country' in mapping_df.columns:
        matching_rows = mapping_df[mapping_df['country'] == origin_country].copy()
        if 'country_name' in matching_rows.columns:
            origin_aliases.update(
                value.strip()
                for value in matching_rows['country_name'].dropna().astype(str).tolist()
                if value.strip()
            )
        matching_rows2 = mapping_df[mapping_df['country_name'] == origin_country].copy()
        if 'country' in matching_rows2.columns:
            origin_aliases.update(
                value.strip()
                for value in matching_rows2['country'].dropna().astype(str).tolist()
                if value.strip()
            )
    origin_aliases = tuple(sorted(origin_aliases))

    allocation_query = sa_text(f"""
        SELECT
            date,
            destination AS destination_country,
            origin,
            COALESCE(new_total_allocated_bcm, total_allocated_bcm) AS allocated_volume_bcm
        FROM {SUPPLY_ALLOCATION_DEMAND_DETAIL_TABLE}
        WHERE run_id = :run_id
            AND origin IN :origin_aliases
            AND COALESCE(new_total_allocated_bcm, total_allocated_bcm) IS NOT NULL
            AND date >= :current_month_start
            AND date <= :horizon_end
    """)
    allocation_df = pd.read_sql(
        allocation_query,
        engine,
        params={
            'run_id': run_metadata['run_id'],
            'origin_aliases': origin_aliases,
            'current_month_start': period_config['current_month_start'].date(),
            'horizon_end': period_config['horizon_end'].date(),
        }
    )

    if allocation_df.empty:
        return pd.DataFrame(), [], run_metadata

    alias_lookup = build_supply_allocation_country_alias_lookup(mapping_df)
    allocation_df['date'] = pd.to_datetime(allocation_df['date'], errors='coerce')
    allocation_df = allocation_df[allocation_df['date'].notna()].copy()
    allocation_df = allocation_df[~allocation_df['destination_country'].isin(origin_aliases)].copy()
    allocation_df = allocation_df.groupby(['date', 'destination_country'], as_index=False)['allocated_volume_bcm'].sum()
    allocation_df = pd.merge(
        allocation_df,
        alias_lookup,
        how='left',
        left_on='destination_country',
        right_on='alias'
    )
    allocation_df['continent'] = allocation_df['continent'].replace('', np.nan).fillna('Unknown')
    allocation_df['country'] = allocation_df['country_display'].replace('', np.nan)
    allocation_df['country'] = allocation_df['country'].fillna(allocation_df['destination_country'])

    # Map destination_level to alias_lookup column name
    level_col_map = {
        'destination_country_name':          None,  # use continent + country hierarchy
        'continent_destination_name':        'continent',
        'destination_shipping_region':       'shipping_region',
        'destination_basin':                 'basin',
        'destination_subcontinent':          'subcontinent',
        'destination_classification_level1': 'country_classification_level1',
        'destination_classification':        'country_classification',
    }
    flat_col = level_col_map.get(destination_level)

    if flat_col is None:
        # Default: continent → country hierarchy
        group_cols = ['continent', 'country']
        df_for_table = allocation_df[['date', 'continent', 'country', 'allocated_volume_bcm']]
    else:
        # Flat grouping by the selected level
        if flat_col in allocation_df.columns:
            allocation_df[flat_col] = allocation_df[flat_col].replace('', np.nan).fillna('Unknown')
        else:
            allocation_df[flat_col] = 'Unknown'
        group_cols = ['continent', flat_col]
        allocation_df['continent'] = allocation_df[flat_col]  # reuse continent slot for display
        df_for_table = allocation_df[['date', 'continent', flat_col, 'allocated_volume_bcm']].rename(
            columns={flat_col: 'country'}
        )
        group_cols = ['continent', 'country']

    summary_df = build_destination_forecast_period_table(
        df_for_table,
        'allocated_volume_bcm',
        group_cols,
        current_date=current_date
    )
    if not summary_df.empty:
        sort_cols = [c for c in group_cols if c in summary_df.columns]
        summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)
        for col in period_config['ordered_labels']:
            summary_df[col] = summary_df[col].round(1)

    allocated_values = build_destination_forecast_total_values(
        allocation_df[['date', 'allocated_volume_bcm']],
        'allocated_volume_bcm',
        current_date=current_date
    )

    footer_rows = [
        {'Continent': 'ALLOCATED SUPPLY TOTAL', 'Country': '', **allocated_values},
    ]

    return summary_df, footer_rows, run_metadata


# ─────────────────────────────────────────────────────────────────────────────

def create_country_supply_chart(country_name, rolling_window_days=30):
    """Create seasonal comparison chart for selected country's LNG supply with WoodMac forecast overlay."""
    try:
        df = fetch_country_supply_chart_data(country_name, rolling_window_days)
        historical_df = df.copy() if not df.empty else pd.DataFrame()

        forecast_df = fetch_woodmac_country_export_forecast_data(country_name)

        if historical_df.empty and forecast_df.empty:
            return _empty_timeseries_chart(f"No supply data available for {country_name}")

        return _create_total_export_chart_with_woodmac_forecast(historical_df, forecast_df)

    except Exception as e:
        return _empty_timeseries_chart(f"Error loading supply data for {country_name}")

def create_continent_destination_chart(country_name, rolling_window_days=30):
    """Create seasonal comparison chart by continent destination for selected country's LNG exports."""

    try:
        df = fetch_continent_destination_chart_data(country_name, rolling_window_days)

        if not df.empty:
            pass

        if df.empty:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No export data available for {country_name}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color='#6b7280')
            )
            fig.update_layout(
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Get unique years and continents
        years = sorted(df['year'].unique())
        continents = sorted(df['continent_destination'].unique())
        
        # Define color palette for continents - distinct colors for actual data
        continent_colors = {
            'Africa': '#8E24AA',        # Purple
            'Americas': '#43A047',      # Green
            'Asia': '#FF4444',          # Bright Red
            'Europe': '#1E88E5',        # Strong Blue
            'Unknown': '#757575',       # Gray
            # Add fallback colors just in case
            'Oceania': '#FB8C00',       # Orange
            'Middle East': '#00ACC1',   # Cyan
            'North America': '#D81B60', # Pink
            'South America': '#FFC107'  # Amber
        }
        
        # Get current year for line width highlighting
        current_year = max(years)
        
        # Process each continent and year combination
        continent_legend_shown = {}  # Track which continents have shown legend
        
        for continent in continents:
            continent_data = df[df['continent_destination'] == continent]
            
            # Get color for this continent
            color = continent_colors.get(continent, '#808080')
            
            for year in years:
                year_continent_data = continent_data[continent_data['year'] == year].copy()
                
                if year_continent_data.empty:
                    continue
                
                # For seasonal comparison, use day of year as x-axis
                year_continent_data['plot_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(year_continent_data['day_of_year'] - 1, unit='d')
                
                # Split data into historical and forecast
                historical_data = year_continent_data[~year_continent_data['is_forecast']]
                forecast_data = year_continent_data[year_continent_data['is_forecast']]
                
                # Determine line width based on year
                line_width = 3 if year == current_year else 1.5
                
                # Show legend only for the first occurrence of this continent
                show_legend = bool(continent not in continent_legend_shown)
                if show_legend:
                    continent_legend_shown[continent] = True
                
                # Plot historical data
                if not historical_data.empty:
                    fig.add_trace(go.Scatter(
                        x=historical_data['plot_date'],
                        y=historical_data['rolling_avg'],
                        mode='lines',
                        name=continent if show_legend else None,
                        legendgroup=continent,
                        line=dict(
                            color=color,
                            width=line_width,
                            dash='solid'
                        ),
                        hovertemplate=f'<b>{continent} - {int(year)}</b><br>' +
                                     '%{text}<br>' +
                                     'Export: %{y:.1f} mcm/d<br>' +
                                     '<extra></extra>',
                        text=historical_data['month_day'],
                        showlegend=show_legend
                    ))
                
                # Plot forecast data with transparency
                if not forecast_data.empty:
                    # Connect forecast line to historical
                    if not historical_data.empty:
                        connect_data = pd.concat([historical_data.tail(1), forecast_data])
                    else:
                        connect_data = forecast_data
                    
                    # Create transparent version of color for forecast
                    import re
                    if color.startswith('#'):
                        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                        forecast_color = f"rgba({r}, {g}, {b}, 0.4)"
                    else:
                        forecast_color = color
                    
                    fig.add_trace(go.Scatter(
                        x=connect_data['plot_date'],
                        y=connect_data['rolling_avg'],
                        mode='lines',
                        name=None,
                        legendgroup=continent,
                        line=dict(
                            color=forecast_color,
                            width=line_width,
                            dash='solid'
                        ),
                        opacity=0.6,
                        hovertemplate=f'<b>{continent} - {int(year)} (Forecast)</b><br>' +
                                     '%{text}<br>' +
                                     'Export: %{y:.1f} mcm/d<br>' +
                                     '<extra></extra>',
                        text=connect_data['month_day'],
                        showlegend=False
                    ))
        
        # Update layout
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
                zeroline=False
            ),
            
            # Legend positioning - compact for three-chart layout
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.15,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)',
                borderwidth=0,
                font=dict(size=10, color='#4A4A4A'),
                itemsizing='constant'
            ),
            
            # General Layout
            height=400,
            margin=dict(l=50, r=120, t=30, b=50),  # Adjusted margins for three-chart layout
            paper_bgcolor='white',
            plot_bgcolor='white',
            hovermode='x unified',
            
            # Title - removed since we have headers
            title=None
        )
        
        return fig
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading export data for {country_name}: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='#ef4444')
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=400,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig

def create_continent_percentage_chart(country_name, rolling_window_days=30):
    """Create percentage distribution chart by continent destination for selected country's LNG exports."""

    try:
        df = fetch_continent_percentage_chart_data(country_name, rolling_window_days)

        if not df.empty:
            pass

        if df.empty:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No export data available for {country_name}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color='#6b7280')
            )
            fig.update_layout(
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Get unique years and continents
        years = sorted(df['year'].unique())
        continents = sorted(df['continent_destination'].unique())
        
        # Define color palette for continents - same as absolute chart
        continent_colors = {
            'Africa': '#8E24AA',        # Purple
            'Americas': '#43A047',      # Green
            'Asia': '#FF4444',          # Bright Red
            'Europe': '#1E88E5',        # Strong Blue
            'Unknown': '#757575',       # Gray
            # Add fallback colors just in case
            'Oceania': '#FB8C00',       # Orange
            'Middle East': '#00ACC1',   # Cyan
            'North America': '#D81B60', # Pink
            'South America': '#FFC107'  # Amber
        }
        
        # Get current year for line width highlighting
        current_year = max(years)
        
        # Process each continent and year combination
        continent_legend_shown = {}  # Track which continents have shown legend
        
        for continent in continents:
            continent_data = df[df['continent_destination'] == continent]
            
            # Get color for this continent
            color = continent_colors.get(continent, '#808080')
            
            for year in years:
                year_continent_data = continent_data[continent_data['year'] == year].copy()
                
                if year_continent_data.empty:
                    continue
                
                # For seasonal comparison, use day of year as x-axis
                year_continent_data['plot_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(year_continent_data['day_of_year'] - 1, unit='d')
                
                # Split data into historical and forecast
                historical_data = year_continent_data[~year_continent_data['is_forecast']]
                forecast_data = year_continent_data[year_continent_data['is_forecast']]
                
                # Determine line width based on year
                line_width = 3 if year == current_year else 1.5
                
                # Show legend only for the first occurrence of this continent
                show_legend = bool(continent not in continent_legend_shown)
                if show_legend:
                    continent_legend_shown[continent] = True
                
                # Plot historical data
                if not historical_data.empty:
                    fig.add_trace(go.Scatter(
                        x=historical_data['plot_date'],
                        y=historical_data['percentage'],
                        mode='lines',
                        name=continent if show_legend else None,
                        legendgroup=continent,
                        line=dict(
                            color=color,
                            width=line_width,
                            dash='solid'
                        ),
                        hovertemplate=f'<b>{continent} - {int(year)}</b><br>' +
                                     '%{text}<br>' +
                                     'Share: %{y:.1f}%<br>' +
                                     '<extra></extra>',
                        text=historical_data['month_day'],
                        showlegend=show_legend
                    ))
                
                # Plot forecast data with transparency
                if not forecast_data.empty:
                    # Connect forecast line to historical
                    if not historical_data.empty:
                        connect_data = pd.concat([historical_data.tail(1), forecast_data])
                    else:
                        connect_data = forecast_data
                    
                    # Create transparent version of color for forecast
                    import re
                    if color.startswith('#'):
                        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                        forecast_color = f"rgba({r}, {g}, {b}, 0.4)"
                    else:
                        forecast_color = color
                    
                    fig.add_trace(go.Scatter(
                        x=connect_data['plot_date'],
                        y=connect_data['percentage'],
                        mode='lines',
                        name=None,
                        legendgroup=continent,
                        line=dict(
                            color=forecast_color,
                            width=line_width,
                            dash='solid'
                        ),
                        opacity=0.6,
                        hovertemplate=f'<b>{continent} - {int(year)} (Forecast)</b><br>' +
                                     '%{text}<br>' +
                                     'Share: %{y:.1f}%<br>' +
                                     '<extra></extra>',
                        text=connect_data['month_day'],
                        showlegend=False
                    ))
        
        # Update layout
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
            
            # Y-Axis Professional Styling - Percentage scale
            yaxis=dict(
                title=dict(text='Share (%)', font=dict(size=13, color='#4A4A4A')),
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.3)',
                gridwidth=0.5,
                linecolor='#CCCCCC',
                linewidth=1,
                tickfont=dict(size=11, color='#666666'),
                zeroline=False,
                range=[0, 100],  # Fixed range for percentage
                ticksuffix='%',
                dtick=10,  # Show tick marks every 10%
                tick0=0    # Start ticks at 0
            ),
            
            # Legend positioning - compact for three-chart layout
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.15,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)',
                borderwidth=0,
                font=dict(size=10, color='#4A4A4A'),
                itemsizing='constant'
            ),
            
            # General Layout
            height=400,
            margin=dict(l=50, r=120, t=30, b=50),  # Adjusted margins for three-chart layout
            paper_bgcolor='white',
            plot_bgcolor='white',
            hovermode='x unified',
            
            # Title - removed since we have headers
            title=None
        )
        
        return fig
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading percentage data for {country_name}: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='#ef4444')
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=400,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig

def create_destination_forecast_summary_table(display_df):
    """Create the WoodMac destination forecast summary table for the exporter detail page."""
    footer_row_labels = [
        'ALLOCATED SUPPLY TOTAL',
    ]
    col_display_names = {'Continent': 'Destination Level', 'Country': 'Country'}
    columns = []
    for col in display_df.columns:
        if col in ['Continent', 'Country']:
            columns.append({'name': col_display_names.get(col, col), 'id': col, 'type': 'text'})
        else:
            columns.append({
                'name': col,
                'id': col,
                'type': 'numeric',
            })

    conditional_styles = [
        {'if': {'filter_query': '{Country} = "Total"'}, 'backgroundColor': '#e3f2fd', 'fontWeight': 'bold'},
        {'if': {'filter_query': '{Continent} = ""'}, 'backgroundColor': '#f9f9f9', 'fontSize': '13px'},
        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f5f5f5'},
        {'if': {'column_id': 'Continent'}, 'textAlign': 'left'},
        {'if': {'column_id': 'Country'}, 'textAlign': 'left'},
    ]
    for col in display_df.columns:
        if col not in ['Continent', 'Country']:
            conditional_styles.append({
                'if': {'column_id': col},
                'textAlign': 'right',
                'paddingRight': '12px'
            })

    month_columns = [
        col for col in display_df.columns
        if "'" in col and not col.startswith('Q') and not col.startswith('W') and col not in ['Continent', 'Country']
    ]
    annual_avg_columns = [col for col in display_df.columns if col.endswith(' Avg')]

    for col in display_df.columns:
        if col in month_columns:
            if month_columns and col == month_columns[0]:
                conditional_styles.append({'if': {'column_id': col}, 'borderLeft': '3px solid white'})
        elif col in annual_avg_columns:
            conditional_styles.append({'if': {'column_id': col}, 'backgroundColor': '#eef2ff', 'fontWeight': '500'})
            if annual_avg_columns and col == annual_avg_columns[0]:
                conditional_styles.append({'if': {'column_id': col}, 'borderLeft': '3px solid white'})

    conditional_styles.append({
        'if': {'filter_query': '{Continent} = "GRAND TOTAL"'},
        'backgroundColor': '#2E86C1',
        'color': 'white',
        'fontWeight': 'bold'
    })
    footer_row_colors = {
        'ALLOCATED SUPPLY TOTAL': {'backgroundColor': '#e8f4fd', 'fontWeight': 'bold', 'color': '#1B4F72'},
    }
    for row_label in footer_row_labels:
        conditional_styles.append({
            'if': {'filter_query': f'{{Continent}} = "{row_label}"'},
            **footer_row_colors[row_label]
        })

    header_styles = []
    for col in month_columns:
        header_styles.append({'if': {'column_id': col}, 'backgroundColor': '#f3e5f5'})
    for col in annual_avg_columns:
        header_styles.append({'if': {'column_id': col}, 'backgroundColor': '#eef2ff'})
    if month_columns:
        header_styles.append({'if': {'column_id': month_columns[0]}, 'borderLeft': '3px solid white'})
    if annual_avg_columns:
        header_styles.append({'if': {'column_id': annual_avg_columns[0]}, 'borderLeft': '3px solid white'})

    return dash_table.DataTable(
        id={'type': 'exp-destination-forecast-expandable-table', 'index': 'summary'},
        data=display_df.to_dict('records'),
        columns=columns,
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': '#2E86C1',
            'color': 'white',
            'fontWeight': 'bold',
            'fontSize': '12px',
            'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'textAlign': 'center'
        },
        style_header_conditional=header_styles,
        style_cell={
            'textAlign': 'center',
            'fontSize': '12px',
            'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'padding': '8px',
            'minWidth': '80px'
        },
        style_data_conditional=conditional_styles,
        sort_action='native',
        page_size=50,
        fill_width=False
    )


@callback(
    Output('exp-destination-forecast-summary-subtitle', 'children'),
    Output('exp-destination-forecast-summary-table-container', 'children'),
    Input('origin-country-dropdown', 'value'),
    Input('us-region-status-dropdown', 'value'),
    Input('exp-destination-forecast-expanded-continents', 'data'),
    Input('destination-level-dropdown', 'value'),
    prevent_initial_call=False
)
def update_destination_forecast_summary_table(origin_country, status, expanded_continents, destination_level):
    if not origin_country:
        return (
            "Modeled destination allocation from SQL outputs.",
            html.Div("Please select an origin country.", style={'textAlign': 'center', 'padding': '20px'})
        )
    if status == 'non_laden':
        return (
            "Modeled destination allocation from SQL outputs.",
            html.Div(
                "WoodMac destination forecast allocation is not shown for non-laden selections.",
                style={'textAlign': 'center', 'padding': '20px'}
            )
        )

    try:
        expanded_continents = expanded_continents or []
        destination_level = destination_level or 'destination_country_name'
        summary_df, footer_rows, run_metadata = fetch_destination_forecast_summary_data(
            engine,
            origin_country,
            destination_level=destination_level
        )
        subtitle = format_supply_allocation_run_subtitle(run_metadata)
        if run_metadata is None:
            return (
                subtitle,
                html.Div(
                    "No compatible WoodMac supply-allocation SQL run is currently available.",
                    style={'textAlign': 'center', 'padding': '20px'}
                )
            )

        display_df = prepare_destination_forecast_table_for_display(
            summary_df,
            expanded_continents=expanded_continents,
            footer_rows=footer_rows
        )
        if display_df.empty:
            return (
                subtitle,
                html.Div(
                    f"No WoodMac destination forecast allocation data is available for {origin_country}.",
                    style={'textAlign': 'center', 'padding': '20px'}
                )
            )

        return subtitle, create_destination_forecast_summary_table(display_df)
    except Exception as e:
        return (
            "Modeled destination allocation from SQL outputs.",
            html.Div(
                f"Error loading data: {str(e)}",
                style={'textAlign': 'center', 'padding': '20px', 'color': 'red'}
            )
        )


@callback(
    Output('exp-destination-forecast-expanded-continents', 'data', allow_duplicate=True),
    [Input({'type': 'exp-destination-forecast-expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'exp-destination-forecast-expandable-table', 'index': ALL}, 'data'),
     State('exp-destination-forecast-expanded-continents', 'data')],
    prevent_initial_call=True
)
def toggle_destination_forecast_continent_expansion(active_cells, table_data_list, expanded_continents):
    if not any(active_cells):
        return expanded_continents or []

    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    if 'exp-destination-forecast-expandable-table' in prop_id and '.active_cell' in prop_id:
        active_cell = active_cells[0]
        if not active_cell:
            return expanded_continents or []
        table_data = table_data_list[0]
        if not table_data or active_cell['column_id'] != 'Continent':
            return expanded_continents or []
        clicked_row = table_data[active_cell['row']]
        continent_value = clicked_row.get('Continent', '')
        if continent_value.startswith('▶') or continent_value.startswith('▼'):
            continent_name = continent_value[2:].strip()
            expanded_continents = expanded_continents or []
            if continent_name in expanded_continents:
                expanded_continents.remove(continent_name)
            else:
                expanded_continents.append(continent_name)
            return expanded_continents

    return expanded_continents or []


@callback(
    Output('supply-analysis-title', 'children'),
    Input('supply-rolling-window-input', 'value')
)
def update_supply_analysis_title(rolling_window_days):
    return f"LNG Supply Analysis - {format_rolling_window_title(rolling_window_days)} + WoodMac Forecast"


@callback(
    [Output('country-supply-chart', 'figure'),
     Output('country-supply-header', 'children')],
    Input('origin-country-dropdown', 'value'),
    Input('supply-rolling-window-input', 'value')
)
def update_country_supply_chart(selected_country, rolling_window_days):
    """Update the supply chart based on selected country."""
    if not selected_country:
        # Return empty chart if no country selected
        fig = go.Figure()
        fig.update_layout(height=400)
        return fig, "Total Supply"
    
    # Create the chart
    fig = create_country_supply_chart(selected_country, rolling_window_days)
    
    # Update header with country name
    header_text = f"{selected_country} - Total Supply"
    
    return fig, header_text

@callback(
    [Output('continent-destination-chart', 'figure'),
     Output('continent-destination-header', 'children')],
    Input('origin-country-dropdown', 'value'),
    Input('supply-rolling-window-input', 'value')
)
def update_continent_destination_chart(selected_country, rolling_window_days):
    """Update the continent destination chart based on selected country."""
    if not selected_country:
        # Return empty chart if no country selected
        fig = go.Figure()
        fig.update_layout(height=400)
        return fig, "By Destination Continent"
    
    # Create the chart
    fig = create_continent_destination_chart(selected_country, rolling_window_days)
    
    # Update header
    header_text = f"{selected_country} - By Destination Continent"
    
    return fig, header_text

@callback(
    [Output('continent-percentage-chart', 'figure'),
     Output('continent-percentage-header', 'children')],
    Input('origin-country-dropdown', 'value'),
    Input('supply-rolling-window-input', 'value')
)
def update_continent_percentage_chart(selected_country, rolling_window_days):
    """Update the continent percentage chart based on selected country."""
    if not selected_country:
        # Return empty chart if no country selected
        fig = go.Figure()
        fig.update_layout(height=400)
        return fig, "By Destination Continent (%)"
    
    # Create the chart
    fig = create_continent_percentage_chart(selected_country, rolling_window_days)
    
    # Update header
    header_text = f"{selected_country} - Market Share (%)"
    
    return fig, header_text


@callback(
    Output('download-exporter-detail-supply-excel', 'data'),
    Input('export-supply-analysis-button', 'n_clicks'),
    State('origin-country-dropdown', 'value'),
    State('supply-rolling-window-input', 'value'),
    State('us-region-status-dropdown', 'value'),
    State('us-vessel-type-dropdown', 'value'),
    prevent_initial_call=True
)
def export_supply_analysis_to_excel(n_clicks, selected_country, rolling_window_days, status, vessel_type):
    """Export LNG Supply Analysis data for the selected country to Excel."""
    if not n_clicks or not selected_country:
        raise PreventUpdate

    normalized_window_days = normalize_rolling_window_days(rolling_window_days)
    rolling_window_label = format_rolling_window_label(normalized_window_days)

    supply_df = fetch_country_supply_chart_data(selected_country, normalized_window_days)
    continent_df = fetch_continent_destination_chart_data(selected_country, normalized_window_days)
    percentage_df = fetch_continent_percentage_chart_data(selected_country, normalized_window_days)
    summary_df = fetch_destination_summary_data(
        engine,
        selected_country,
        status,
        vessel_type,
        normalized_window_days
    )

    if supply_df.empty and continent_df.empty and percentage_df.empty and summary_df.empty:
        raise PreventUpdate

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not supply_df.empty:
            supply_export_df = supply_df.copy()
            supply_export_df.to_excel(writer, sheet_name='Total Supply', index=False)

        if not continent_df.empty:
            continent_export_df = continent_df.copy()
            continent_export_df.to_excel(writer, sheet_name='Continent mcmd', index=False)

        if not percentage_df.empty:
            percentage_export_df = percentage_df.copy()
            percentage_export_df.to_excel(writer, sheet_name='Continent Share', index=False)

        if not summary_df.empty:
            summary_export_df = summary_df.copy()
            summary_export_df.to_excel(writer, sheet_name='Destination Summary', index=False)

        for worksheet in writer.sheets.values():
            for column_cells in worksheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter
                for cell in column_cells:
                    cell_value = "" if cell.value is None else str(cell.value)
                    if len(cell_value) > max_length:
                        max_length = len(cell_value)
                worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)

    output.seek(0)
    safe_country = "".join(char if char.isalnum() else "_" for char in selected_country).strip("_") or "country"
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{safe_country}_LNG_Supply_Analysis_{rolling_window_label}_{timestamp}.xlsx"

    return dcc.send_bytes(output.getvalue(), filename)

@callback(
    Output('trade-analysis-header', 'children'),
    Input('origin-country-dropdown', 'value')
)
def update_trade_analysis_header(selected_country):
    return f'Trade Analysis: {selected_country} → Destination'

@callback(
    Output('origin-country-dropdown', 'options'),
    Output('origin-country-dropdown', 'value'),
    Input('global-refresh-button', 'n_clicks'),
    prevent_initial_call=False  # This will run on page load
)
def initialize_country_dropdown(n_clicks):
    """Initialize the country dropdown with all available origin countries."""
    try:
        # Query to get all unique origin countries from the database
        query = f"""
        SELECT DISTINCT origin_country_name
        FROM {DB_SCHEMA}.kpler_trades
        WHERE origin_country_name IS NOT NULL
        AND upload_timestamp_utc = (SELECT MAX(upload_timestamp_utc) FROM {DB_SCHEMA}.kpler_trades)
        ORDER BY origin_country_name
        """
        countries_df = pd.read_sql(query, engine)
        if countries_df.empty:
            # If no countries found, provide a default
            return [{'label': 'United States', 'value': 'United States'}], 'United States'
        # Convert countries to dropdown options format
        country_options = [{'label': country, 'value': country}
                          for country in countries_df['origin_country_name'].tolist()]
        # Set United States as default if available, otherwise first country
        default_country = 'United States'
        if default_country not in countries_df['origin_country_name'].values:
            default_country = countries_df['origin_country_name'].iloc[0]
        return country_options, default_country
    except Exception as e:
        # Return a default option if there's an error
        return [{'label': 'United States', 'value': 'United States'}], 'United States'

# ========================================
# TRAIN MAINTENANCE DATA FUNCTIONS
# ========================================

def fetch_train_maintenance_data(engine, country_name=None):
    """
    Fetch and process maintenance data from both planned and unplanned tables.
    Returns raw maintenance data for the specified country or all countries.
    """
    try:
        # Resolve WoodMac country name alias (e.g. "Russian Federation" → "Russia")
        woodmac_country_name = country_name
        if country_name:
            try:
                from sqlalchemy import text as sa_text
                alias_q = sa_text("""
                    SELECT country_name FROM at_lng.mappings_country
                    WHERE country = :country AND country_name IS NOT NULL LIMIT 1
                """)
                with engine.connect() as conn:
                    result = conn.execute(alias_q, {'country': country_name}).fetchone()
                if result and result[0]:
                    woodmac_country_name = result[0]
            except Exception:
                pass

        # Build the query with optional country filter
        country_filter = ""
        if woodmac_country_name:
            country_filter = f"AND country_name = '{woodmac_country_name}'"
        
        query = f"""
        WITH combined_maintenance AS (
            SELECT 
                plant_name,
                country_name,
                lng_train_name_short,
                year,
                month,
                year_actual_forecast,
                SUM(metric_value) as total_mtpa,
                STRING_AGG(metric_comment, '; ') as metric_comment
            FROM (
                SELECT plant_name, country_name, lng_train_name_short, 
                       year, month, year_actual_forecast, metric_value, metric_comment
                FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_unplanned_downtime_mta
                WHERE metric_value > 0
                UNION ALL
                SELECT plant_name, country_name, lng_train_name_short, 
                       year, month, year_actual_forecast, metric_value, metric_comment
                FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_planned_maintenance_mta
                WHERE metric_value > 0
            ) maintenance_data
            WHERE 1=1
            {country_filter}
            GROUP BY plant_name, country_name, lng_train_name_short, 
                     year, month, year_actual_forecast
        )
        SELECT * FROM combined_maintenance
        ORDER BY plant_name, lng_train_name_short, year, month
        """
        
        df = pd.read_sql(query, engine)
        
        # Create date column for easier processing
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        
        return df
        
    except Exception as e:
        return pd.DataFrame()


def process_maintenance_periods_hierarchical(df, expanded_plants=None):
    """
    Process maintenance data into hierarchical structure with plant totals and train details.
    Returns data suitable for expandable table display.
    
    Conversion: 1 MTPA = 1.372 MCM/D (approximately)
    """
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Get current date
        current_date = pd.Timestamp.now()
        current_quarter = current_date.quarter
        current_year = current_date.year
        
        # Define MTPA to MCM/D conversion factor
        MTPA_TO_MCM_D = 1.372
        
        # Initialize expanded plants list
        expanded_plants = expanded_plants or []
        
        # Define period columns
        period_cols = ([f'Q-{i}' for i in range(5, 0, -1)] + 
                      [f'M-{i}' for i in range(3, 0, -1)] + 
                      [f'M+{i}' for i in range(1, 4)] + 
                      [f'Q+{i}' for i in range(1, 5)])
        
        # Calculate period boundaries
        last_month_end = pd.Timestamp(year=current_date.year, month=current_date.month, day=1) - pd.DateOffset(days=1)
        next_3m_start = pd.Timestamp(year=current_date.year, month=current_date.month, day=1)
        
        # Process each plant-train combination
        train_data = []
        plant_totals = {}
        plant_trains = {}  # Store trains for each plant
        comments_data = {}  # Store comments for tooltips
        
        for (plant, train), group_df in df.groupby(['plant_name', 'lng_train_name_short']):
            row = {
                'Plant': '',  # Will be filled later for expanded rows
                'Train': train,
                'Type': 'train',
                '_plant': plant
            }
            
            # Initialize plant total if not exists
            if plant not in plant_totals:
                plant_totals[plant] = {col: 0 for col in period_cols}
                plant_trains[plant] = []
                comments_data[plant] = {}
            
            plant_trains[plant].append(train)
            comments_data[plant][train] = {}
            
            # Process last 5 quarters
            for q_offset in range(5, 0, -1):
                # Calculate target quarter and year
                target_q = current_quarter - q_offset
                target_year = current_year
                while target_q <= 0:
                    target_q += 4
                    target_year -= 1
                
                # Calculate quarter boundaries (Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec)
                q_start_month = (target_q - 1) * 3 + 1  # Q1->1, Q2->4, Q3->7, Q4->10
                q_start = pd.Timestamp(year=target_year, month=q_start_month, day=1)
                q_end = q_start + pd.DateOffset(months=3) - pd.DateOffset(days=1)
                
                q_data = group_df[(group_df['date'] >= q_start) & (group_df['date'] <= q_end)]
                days_in_quarter = (q_end - q_start).days + 1
                total_mtpa = q_data['total_mtpa'].sum()
                avg_mcm_d = (total_mtpa * MTPA_TO_MCM_D * 365) / days_in_quarter if days_in_quarter > 0 else 0
                
                quarter_label = f"Q-{q_offset}"
                value = round(avg_mcm_d, 1)
                row[quarter_label] = value if value > 0 else None
                plant_totals[plant][quarter_label] += value
                
                # Store comments for this period
                if not q_data.empty and 'metric_comment' in q_data.columns:
                    comments = q_data['metric_comment'].dropna().unique()
                    if len(comments) > 0:
                        comments_data[plant][train][quarter_label] = '; '.join(comments)
            
            # Process last 3 months
            for m_offset in range(3, 0, -1):
                m_date = last_month_end - pd.DateOffset(months=m_offset-1)
                m_start = pd.Timestamp(year=m_date.year, month=m_date.month, day=1)
                m_end = m_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                
                m_data = group_df[(group_df['date'] >= m_start) & (group_df['date'] <= m_end)]
                days_in_month = m_end.day
                total_mtpa = m_data['total_mtpa'].sum()
                avg_mcm_d = (total_mtpa * MTPA_TO_MCM_D * 365) / days_in_month if days_in_month > 0 else 0
                
                month_label = f"M-{m_offset}"
                value = round(avg_mcm_d, 1)
                row[month_label] = value if value > 0 else None
                plant_totals[plant][month_label] += value
                
                # Store comments for this period
                if not m_data.empty and 'metric_comment' in m_data.columns:
                    comments = m_data['metric_comment'].dropna().unique()
                    if len(comments) > 0:
                        comments_data[plant][train][month_label] = '; '.join(comments)
            
            # Process next 3 months (future)
            for m_offset in range(1, 4):
                m_date = next_3m_start + pd.DateOffset(months=m_offset-1)
                m_start = pd.Timestamp(year=m_date.year, month=m_date.month, day=1)
                m_end = m_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                
                m_data = group_df[(group_df['date'] >= m_start) & (group_df['date'] <= m_end)]
                days_in_month = m_end.day
                total_mtpa = m_data['total_mtpa'].sum()
                avg_mcm_d = (total_mtpa * MTPA_TO_MCM_D * 365) / days_in_month if days_in_month > 0 else 0
                
                month_label = f"M+{m_offset}"
                value = round(avg_mcm_d, 1)
                row[month_label] = value if value > 0 else None
                plant_totals[plant][month_label] += value
                
                # Store comments for this period
                if not m_data.empty and 'metric_comment' in m_data.columns:
                    comments = m_data['metric_comment'].dropna().unique()
                    if len(comments) > 0:
                        comments_data[plant][train][month_label] = '; '.join(comments)
            
            # Process next 4 quarters (future)
            for q_offset in range(1, 5):
                # Calculate target quarter and year
                target_q = current_quarter + q_offset
                target_year = current_year
                while target_q > 4:
                    target_q -= 4
                    target_year += 1

                # Calculate quarter boundaries (Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec)
                q_start_month = (target_q - 1) * 3 + 1  # Q1->1, Q2->4, Q3->7, Q4->10
                q_start = pd.Timestamp(year=target_year, month=q_start_month, day=1)
                q_end = q_start + pd.DateOffset(months=3) - pd.DateOffset(days=1)
                
                q_data = group_df[(group_df['date'] >= q_start) & (group_df['date'] <= q_end)]
                days_in_quarter = (q_end - q_start).days + 1
                total_mtpa = q_data['total_mtpa'].sum()
                avg_mcm_d = (total_mtpa * MTPA_TO_MCM_D * 365) / days_in_quarter if days_in_quarter > 0 else 0
                
                quarter_label = f"Q+{q_offset}"
                value = round(avg_mcm_d, 1)
                row[quarter_label] = value if value > 0 else None
                plant_totals[plant][quarter_label] += value
                
                # Store comments for this period
                if not q_data.empty and 'metric_comment' in q_data.columns:
                    comments = q_data['metric_comment'].dropna().unique()
                    if len(comments) > 0:
                        comments_data[plant][train][quarter_label] = '; '.join(comments)
            
            train_data.append(row)
        
        # Build hierarchical data with plant totals
        final_data = []
        grand_total = {col: 0 for col in period_cols}
        
        for plant in sorted(plant_totals.keys()):
            # Add arrow indicator for expandable plant
            is_expanded = plant in expanded_plants
            arrow = '− ' if is_expanded else '+ '

            # Add plant total row
            plant_row = {
                'Plant': arrow + plant,
                'Train': '',
                'Type': 'plant'
            }
            
            # Add period values
            for col in period_cols:
                value = round(plant_totals[plant][col], 1)
                plant_row[col] = value if value > 0 else None
                grand_total[col] += plant_totals[plant][col]
            
            final_data.append(plant_row)
            
            # Add train rows if plant is expanded
            if is_expanded:
                plant_train_rows = [r for r in train_data if r.get('_plant') == plant]
                for row in plant_train_rows:
                    # Remove the hidden _plant column and keep Plant column empty for train rows
                    row.pop('_plant', None)
                    row['Plant'] = ''  # Empty for train detail rows
                final_data.extend(plant_train_rows)
        
        # Add grand total row
        grand_total_row = {
            'Plant': 'GRAND TOTAL',
            'Train': '',
            'Type': 'total'
        }
        for col in period_cols:
            value = round(grand_total[col], 1)
            grand_total_row[col] = value if value > 0 else None
        
        final_data.append(grand_total_row)
        
        return pd.DataFrame(final_data), comments_data
        
    except Exception as e:
        import traceback
        return pd.DataFrame(), {}


def create_maintenance_summary_table(df, comments_data=None):
    """
    Create expandable Dash DataTable with maintenance summary showing outage impact in MCM/D.
    McKinsey board style: historical vs current vs forecast zones, magnitude heat-map, no decimals.
    """
    if df.empty:
        return html.Div("No maintenance data available", className="no-data-message")

    try:
        current_date = pd.Timestamp.now()
        current_year = current_date.year
        current_quarter = current_date.quarter

        # Column IDs by period category (names unchanged)
        historical_col_ids = [f'Q-{i}' for i in range(5, 0, -1)] + [f'M-{i}' for i in range(3, 0, -1)]
        current_col_ids = ['M+1']
        nearterm_col_ids = ['M+2', 'M+3', 'Q+1']
        outlook_col_ids = ['Q+2', 'Q+3', 'Q+4']

        columns = [
            {'name': 'Plant', 'id': 'Plant', 'type': 'text'},
            {'name': 'Train', 'id': 'Train', 'type': 'text'},
        ]

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for i in range(5, 0, -1):
            q_num = current_quarter - i
            q_year = current_year
            while q_num <= 0:
                q_num += 4
                q_year -= 1
            columns.append({
                'name': f"Q{q_num}'{str(q_year)[2:]}",
                'id': f'Q-{i}',
                'type': 'numeric',
                'format': {'specifier': '.0f'},
            })

        for i in range(3, 0, -1):
            month_date = current_date - pd.DateOffset(months=i)
            columns.append({
                'name': f"{month_names[month_date.month - 1]}'{str(month_date.year)[2:]}",
                'id': f'M-{i}',
                'type': 'numeric',
                'format': {'specifier': '.0f'},
            })

        for i in range(1, 4):
            month_date = current_date + pd.DateOffset(months=i - 1)
            columns.append({
                'name': f"{month_names[month_date.month - 1]}'{str(month_date.year)[2:]}",
                'id': f'M+{i}',
                'type': 'numeric',
                'format': {'specifier': '.0f'},
            })

        for i in range(1, 5):
            q_num = current_quarter + i
            q_year = current_year
            if q_num > 4:
                q_num -= 4
                q_year += 1
            columns.append({
                'name': f"Q{q_num}'{str(q_year)[2:]}",
                'id': f'Q+{i}',
                'type': 'numeric',
                'format': {'specifier': '.0f'},
            })

        columns.append({'name': 'Type', 'id': 'Type', 'type': 'text'})

        data = df.to_dict('records')

        # ── Row-level styles ──────────────────────────────────────────────────
        style_data_conditional = [
            # Plant: off-white, bold navy, left accent border — clickable
            {'if': {'filter_query': '{Type} = "plant"'},
             'backgroundColor': '#f0f4f8', 'fontWeight': '700',
             'color': '#1e3a5f', 'borderLeft': '4px solid #1e3a5f'},
            # Train: white, normal weight, muted
            {'if': {'filter_query': '{Type} = "train"'},
             'backgroundColor': '#ffffff', 'fontWeight': '400',
             'color': '#475569', 'fontSize': '11px'},
            # Grand Total: solid navy — border matches bg so empty cells are seamless
            {'if': {'filter_query': '{Plant} = "GRAND TOTAL"'},
             'backgroundColor': '#1e3a5f', 'color': 'white',
             'fontWeight': '700', 'fontSize': '12px',
             'border': '1px solid #1e3a5f', 'borderTop': '2px solid #93c5fd'},
            # Text alignment
            {'if': {'column_id': 'Plant'}, 'textAlign': 'left', 'cursor': 'pointer'},
            {'if': {'column_id': 'Train'}, 'textAlign': 'left'},
        ]

        # ── Historical cells: blue tint ───────────────────────────────────────
        for col_id in historical_col_ids:
            style_data_conditional.append({
                'if': {'column_id': col_id, 'filter_query': f'{{{col_id}}} > 0'},
                'backgroundColor': 'rgba(59, 130, 246, 0.10)',
                'color': '#1e40af',
            })

        # ── Current month: stronger blue highlight ────────────────────────────
        for col_id in current_col_ids:
            style_data_conditional.append({
                'if': {'column_id': col_id, 'filter_query': f'{{{col_id}}} > 0'},
                'backgroundColor': 'rgba(29, 78, 216, 0.15)',
                'color': '#1d4ed8', 'fontWeight': '600',
            })

        # ── Near-term forecast: amber tiers by magnitude ─────────────────────
        for col_id in nearterm_col_ids:
            style_data_conditional.append({
                'if': {'column_id': col_id, 'filter_query': f'{{{col_id}}} > 0'},
                'backgroundColor': 'rgba(251, 191, 36, 0.20)', 'color': '#92400e',
            })
            style_data_conditional.append({
                'if': {'column_id': col_id, 'filter_query': f'{{{col_id}}} >= 1'},
                'backgroundColor': 'rgba(245, 158, 11, 0.35)',
                'color': '#78350f', 'fontWeight': '600',
            })
            style_data_conditional.append({
                'if': {'column_id': col_id, 'filter_query': f'{{{col_id}}} >= 5'},
                'backgroundColor': 'rgba(220, 38, 38, 0.15)',
                'color': '#991b1b', 'fontWeight': '700',
            })

        # ── Outlook: muted amber tiers ────────────────────────────────────────
        for col_id in outlook_col_ids:
            style_data_conditional.append({
                'if': {'column_id': col_id, 'filter_query': f'{{{col_id}}} > 0'},
                'backgroundColor': 'rgba(251, 191, 36, 0.12)', 'color': '#92400e',
            })
            style_data_conditional.append({
                'if': {'column_id': col_id, 'filter_query': f'{{{col_id}}} >= 1'},
                'backgroundColor': 'rgba(245, 158, 11, 0.22)',
                'color': '#78350f', 'fontWeight': '600',
            })
            style_data_conditional.append({
                'if': {'column_id': col_id, 'filter_query': f'{{{col_id}}} >= 5'},
                'backgroundColor': 'rgba(220, 38, 38, 0.10)',
                'color': '#991b1b', 'fontWeight': '600',
            })

        # ── Column separators ─────────────────────────────────────────────────
        style_data_conditional.append({'if': {'column_id': 'M+1'}, 'borderLeft': '3px solid #94a3b8'})
        style_data_conditional.append({'if': {'column_id': 'Q+2'}, 'borderLeft': '2px solid #cbd5e1'})
        style_data_conditional.append({'if': {'column_id': 'M-3'}, 'borderLeft': '2px solid #e2e8f0'})

        # ── Header conditional styles: zone colouring + bottom border accent ──
        header_styles = []
        for col_id in historical_col_ids:
            header_styles.append({'if': {'column_id': col_id},
                                   'backgroundColor': '#64748b', 'color': '#e2e8f0',
                                   'fontStyle': 'italic', 'fontWeight': '400',
                                   'borderBottom': '3px solid #94a3b8'})
        for col_id in current_col_ids:
            header_styles.append({'if': {'column_id': col_id},
                                   'backgroundColor': '#1d4ed8', 'color': 'white',
                                   'borderLeft': '3px solid #93c5fd',
                                   'borderBottom': '3px solid #93c5fd'})
        for col_id in nearterm_col_ids:
            header_styles.append({'if': {'column_id': col_id},
                                   'backgroundColor': '#92400e', 'color': 'white',
                                   'borderBottom': '3px solid #d97706'})
        for col_id in outlook_col_ids:
            header_styles.append({'if': {'column_id': col_id},
                                   'backgroundColor': '#374151', 'color': '#f9fafb',
                                   'borderLeft': '2px solid #6b7280',
                                   'borderBottom': '3px solid #4b5563'})
        header_styles.append({'if': {'column_id': 'M+1'}, 'borderLeft': '3px solid #93c5fd'})
        header_styles.append({'if': {'column_id': 'Q+2'}, 'borderLeft': '2px solid #6b7280'})
        header_styles.append({'if': {'column_id': 'M-3'}, 'borderLeft': '2px solid #6b7280'})

        # ── Tooltips ──────────────────────────────────────────────────────────
        tooltip_data = []
        if comments_data:
            current_plant = None
            for row in data:
                tooltip_row = {}
                if row.get('Type') == 'plant':
                    current_plant = row.get('Plant', '').replace('▼ ', '').replace('▶ ', '')
                elif row.get('Type') == 'train' and current_plant:
                    train = row.get('Train', '')
                    if current_plant in comments_data and train in comments_data[current_plant]:
                        train_comments = comments_data[current_plant][train]
                        for col in columns:
                            if col['id'] not in ['Plant', 'Train', 'Type']:
                                if col['id'] in train_comments and row.get(col['id']):
                                    tooltip_row[col['id']] = {'value': train_comments[col['id']], 'type': 'text'}
                tooltip_data.append(tooltip_row)

        legend = html.Div([
            html.Span('■ Realized', style={
                'color': '#1e40af', 'marginRight': '20px', 'fontSize': '11px', 'fontWeight': '500'}),
            html.Span('■ Current month', style={
                'color': '#1d4ed8', 'marginRight': '20px', 'fontSize': '11px', 'fontWeight': '500'}),
            html.Span('■ Near-term (0–3M)', style={
                'color': '#92400e', 'marginRight': '20px', 'fontSize': '11px', 'fontWeight': '500'}),
            html.Span('■ Outlook (Q+2–Q+4)', style={
                'color': '#374151', 'marginRight': '20px', 'fontSize': '11px', 'fontWeight': '500'}),
            html.Span('|', style={'color': '#d1d5db', 'marginRight': '16px', 'fontSize': '11px'}),
            html.Span('Red = ≥5 MCM/D impact', style={
                'color': '#991b1b', 'marginRight': '20px', 'fontSize': '11px', 'fontWeight': '500'}),
            html.Span('Click plant row to expand trains', style={
                'color': '#6b7280', 'fontSize': '11px', 'fontStyle': 'italic'}),
        ], style={
            'padding': '4px 0 12px 2px',
            'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
        })

        table = dash_table.DataTable(
            id={'type': 'maintenance-expandable-table', 'index': 0},
            columns=columns,
            data=data,
            tooltip_data=tooltip_data if tooltip_data else None,
            tooltip_delay=0,
            tooltip_duration=None,
            style_table={'overflowX': 'auto', 'borderRadius': '4px', 'border': '1px solid #e2e8f0'},
            style_header={
                'backgroundColor': '#1e293b',
                'color': 'white',
                'fontWeight': '700',
                'fontSize': '11px',
                'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                'textAlign': 'center',
                'textTransform': 'uppercase',
                'letterSpacing': '0.05em',
                'padding': '10px 8px',
            },
            style_header_conditional=header_styles,
            style_cell={
                'textAlign': 'center',
                'fontSize': '12px',
                'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                'padding': '7px 10px',
                'minWidth': '72px',
                'maxWidth': '120px',
                'border': '1px solid #f1f5f9',
            },
            style_cell_conditional=[
                {'if': {'column_id': 'Plant'}, 'minWidth': '200px', 'maxWidth': '280px'},
                {'if': {'column_id': 'Train'}, 'minWidth': '80px', 'maxWidth': '110px'},
            ],
            style_data_conditional=style_data_conditional,
            hidden_columns=['Type'],
            sort_action='native',
            page_size=50,
            fill_width=False,
        )
        return html.Div([legend, table])

    except Exception as e:
        return html.Div(f"Error creating table: {str(e)}", className="error-message")


# --- Callbacks for US Data ---
@callback(
    Output('us-region-data-store', 'data'),
    Output('us-dropdown-options-store', 'data'),
    Output('us-refresh-timestamp-store', 'data'),
    Output('us-vessel-type-dropdown', 'options'),
    Output('us-vessel-type-dropdown', 'value'),

    Input('global-refresh-button', 'n_clicks'),
    Input('destination-level-dropdown', 'value'),
    Input('origin-country-dropdown', 'value'),  # Add this input
    prevent_initial_call=False
)
def refresh_us_data(n_clicks, selected_destination_level, selected_origin_country):
    """Fetch all data, filter for selected country, and prepare dropdown options."""
    # --- Default values in case of error ---
    default_vessel_options = []
    default_vessel_value = None
    default_country_options = []

    try:
        # Pass the selected destination level to kpler_analysis
        df_trades_shipping_region_all = kpler_analysis(engine, destination_level=selected_destination_level, origin_country=selected_origin_country)
        if df_trades_shipping_region_all is None or df_trades_shipping_region_all.empty:
            raise ValueError("kpler_analysis returned empty or None data.")
    except Exception as e:
        error_msg = f"Error loading base data: {e}. Ensure 'engine' is configured."
        us_options_data_error = {
            'vessel_type_options': default_vessel_options,
            'default_vessel_type': default_vessel_value
        }
        # Return defaults and error message
        return None, us_options_data_error, error_msg, default_vessel_options, default_vessel_value

    # --- Get unique countries for dropdown ---
    if 'origin_country_name' not in df_trades_shipping_region_all.columns:
        error_msg = "Error: Missing 'origin_country_name' column."
        us_options_data_error = {
            'vessel_type_options': default_vessel_options,
            'default_vessel_type': default_vessel_value
        }
        return None, us_options_data_error, error_msg, default_vessel_options, default_vessel_value



    # --- Filter for Selected Country ---
    df_trades_shipping_region_filtered = df_trades_shipping_region_all.copy()

    # --- Prepare Dropdown Options ---
    us_options_data = {
        'vessel_type_options': default_vessel_options,
        'default_vessel_type': default_vessel_value
    }
    us_vessel_options = default_vessel_options
    us_default_vessel_value = default_vessel_value

    # Get unique years from filtered data
    if not df_trades_shipping_region_filtered.empty and 'year' in df_trades_shipping_region_filtered.columns:
        us_years = sorted(df_trades_shipping_region_filtered['year'].unique(), reverse=True)
        us_latest_year = us_years[0] if us_years else None
        us_year_options = [{'label': 'All Years', 'value': 'All Years'}] + [
            {'label': str(year), 'value': str(year)} for year in us_years
        ]
        us_options_data['year_options'] = us_year_options
        us_options_data['latest_year'] = str(us_latest_year) if us_latest_year else None

    # Get unique vessel types from filtered data
    if not df_trades_shipping_region_filtered.empty and 'vessel_type' in df_trades_shipping_region_filtered.columns:
        available_vessel_types = df_trades_shipping_region_filtered['vessel_type'].unique()
        available_vessel_types_set = set(available_vessel_types)

        # Order them based on DESIRED_VESSEL_ORDER
        ordered_part = [v for v in DESIRED_VESSEL_ORDER if v in available_vessel_types_set]
        unexpected_types = available_vessel_types_set - set(DESIRED_VESSEL_ORDER)
        sorted_unexpected_part = sorted(list(unexpected_types))
        final_vessel_order = ordered_part + sorted_unexpected_part

        # Add 'All' option at the beginning of the list
        us_vessel_options = [{'label': 'All', 'value': 'All'}] + [{'label': v_type, 'value': v_type} for v_type in
                                                                  final_vessel_order]
        us_options_data['vessel_type_options'] = us_vessel_options

        # Set default vessel type
        # Default to 'All' for vessel size selection
        us_default_vessel_value = "All"
        us_options_data['default_vessel_type'] = us_default_vessel_value

    # Save the selected destination level and origin country in the options data
    us_options_data['destination_level'] = selected_destination_level
    us_options_data['origin_country'] = selected_origin_country

    # Convert Filtered DataFrames to JSON for storage
    us_shipping_data_json = df_trades_shipping_region_filtered.to_json(date_format='iso', orient='split')

    # Store timestamp
    us_refresh_timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return (
        us_shipping_data_json,
        us_options_data,
        us_refresh_timestamp,
        us_vessel_options,
        us_default_vessel_value,
    )



@callback(
    # Outputs for US Region Trades section
    Output('us-trade-count-visualization', 'figure'),

    # Inputs
    Input('us-region-data-store', 'data'),
    Input('us-dropdown-options-store', 'data'),
    Input('aggregation-dropdown', 'value'),
    Input('us-region-status-dropdown', 'value'),
    Input('us-vessel-type-dropdown', 'value'),
    Input('destination-level-dropdown', 'value'),
    Input('origin-country-dropdown', 'value'),
    Input('chart-metric-dropdown', 'value'),
    prevent_initial_call=True
)
def update_us_region_visualizations(us_shipping_data, us_dropdown_options,
                                    selected_aggregation,
                                    selected_status, selected_vessel_type,
                                    selected_destination_level,
                                    origin_country, selected_chart_metric):

    # Default empty figures and messages
    no_data_msg = "No data available for the selected filters."

    # Create an empty figure for the default case
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title=f"Loading {origin_country} Trade Data...",
        height=600
    )

    # --- Input Data Checks ---
    if us_shipping_data is None or us_dropdown_options is None:
        error_msg = f"{origin_country} data not loaded."
        empty_fig.update_layout(title_text=error_msg)
        return empty_fig

    if not all([selected_aggregation, selected_status, selected_destination_level]):
        empty_fig.update_layout(title_text="Please select Aggregation, Status, and Destination Level")
        return empty_fig

    try:
        df_trades_shipping_region_us = pd.read_json(StringIO(us_shipping_data), orient='split')
        if df_trades_shipping_region_us.empty:
            raise ValueError(f"Loaded {origin_country} shipping data is empty.")

        # --- Check if the selected destination level column exists ---
        if selected_destination_level not in df_trades_shipping_region_us.columns:
            error_msg = f"Selected destination level column '{selected_destination_level}' not found in data."
            empty_fig.update_layout(title_text=error_msg)
            return empty_fig

        # --- Filter data for charts ---
        df_for_charts = df_trades_shipping_region_us[df_trades_shipping_region_us['status'] == selected_status].copy()

        # Apply vessel type filter if needed
        if selected_vessel_type and selected_vessel_type != 'All':
            df_for_charts = df_for_charts[df_for_charts['vessel_type'] == selected_vessel_type]

        if df_for_charts.empty:
            empty_fig.update_layout(title_text="No data available after filtering")
            return empty_fig

        # Use professional color palette
        unique_destinations = df_for_charts[selected_destination_level].unique()
        distinct_colors = get_professional_colors(len(unique_destinations))

        # Determine metric column and title based on selection
        metric_mapping = {
            'count_trades': {
                'column': 'count_trades',
                'title': 'Count of Trades',
                'unit': 'trades'
            },
            'mtpa': {
                'column': 'sum_cargo_destination_cubic_meters',
                'title': 'MTPA',
                'unit': 'MTPA'
            },
            'mcm_d': {
                'column': 'sum_cargo_destination_cubic_meters',
                'title': 'mcm/d',
                'unit': 'mcm/d'
            },
            'm3': {
                'column': 'sum_cargo_destination_cubic_meters',
                'title': 'm³',
                'unit': 'm³'
            },
            'median_delivery_days': {
                'column': 'median_delivery_days',
                'title': 'Median Delivery Days',
                'unit': 'days'
            },
            'median_speed': {
                'column': 'median_speed',
                'title': 'Median Speed',
                'unit': 'knots'
            },
            'median_mileage_nautical_miles': {
                'column': 'median_mileage_nautical_miles',
                'title': 'Median Mileage',
                'unit': 'nautical miles'
            },
            'median_ton_miles': {
                'column': 'median_ton_miles',
                'title': 'Median Ton Miles',
                'unit': 'ton miles'
            },
            'median_utilization_rate': {
                'column': 'median_utilization_rate',
                'title': 'Median Utilization Rate',
                'unit': '%'
            },
            'median_cargo_destination_cubic_meters': {
                'column': 'median_cargo_destination_cubic_meters',
                'title': 'Median Cargo Volume',
                'unit': 'm³'
            },
            'median_vessel_capacity_cubic_meters': {
                'column': 'median_vessel_capacity_cubic_meters',
                'title': 'Median Vessel Capacity',
                'unit': 'm³'
            }
        }
        
        selected_metric_info = metric_mapping.get(selected_chart_metric, metric_mapping['mcm_d'])
        metric_column = selected_metric_info['column']
        metric_title = selected_metric_info['title']
        metric_unit = selected_metric_info['unit']

        # Create single figure for selected metric
        fig = go.Figure()

        # Determine grouping columns based on aggregation level
        if selected_aggregation == 'Year':
            groupby_time_cols = ['year']
            x_axis_title = 'Year'
        elif selected_aggregation == 'Year+Season':
            groupby_time_cols = ['year', 'season']
            x_axis_title = 'Year-Season'
        elif selected_aggregation == 'Year+Quarter':
            groupby_time_cols = ['year', 'quarter']
            x_axis_title = 'Year-Quarter'
        elif selected_aggregation == 'Month':
            groupby_time_cols = ['year', 'month']
            x_axis_title = 'Month'
        elif selected_aggregation == 'Week':
            groupby_time_cols = ['year', 'week']
            x_axis_title = 'Week'
        else:
            groupby_time_cols = ['year']
            x_axis_title = 'Year'

        # Get unique destination values for consistent coloring
        unique_destinations = df_for_charts[selected_destination_level].unique()
        color_mapping = {dest: distinct_colors[i % len(distinct_colors)] for i, dest in
                         enumerate(sorted(unique_destinations))}

        # --- Prepare and add traces for selected metric subplot ---
        try:
            # Prepare data
            all_groupby_cols = groupby_time_cols + [selected_destination_level]
            
            # Use appropriate aggregation based on metric type
            if 'median' in selected_chart_metric:
                # For median metrics, use median aggregation
                selected_metric_data = df_for_charts.groupby(all_groupby_cols, observed=False)[
                    metric_column].median().reset_index()
            else:
                # For count and sum metrics, use sum aggregation
                selected_metric_data = df_for_charts.groupby(all_groupby_cols, observed=False)[
                    metric_column].sum().reset_index()

            selected_metric_data = convert_trade_analysis_volume_metric(
                selected_metric_data,
                metric_column,
                selected_aggregation,
                selected_chart_metric
            )

            # Create time labels and sorting fields
            if 'year' in selected_metric_data.columns:
                if 'quarter' in selected_metric_data.columns and 'quarter' in groupby_time_cols:
                    # Extract quarter number for sorting
                    selected_metric_data['quarter_num'] = selected_metric_data['quarter'].str.extract(r'(\d+)').astype(int)
                    selected_metric_data['time_label'] = selected_metric_data['year'].astype(str) + '-' + selected_metric_data['quarter'].astype(str)
                    # Create a sort key (higher values appear first in the chart)
                    selected_metric_data['sort_key'] = selected_metric_data['year'] * 10 + selected_metric_data['quarter_num']
                elif 'season' in selected_metric_data.columns and 'season' in groupby_time_cols:
                    selected_metric_data['time_label'] = selected_metric_data['year'].astype(str) + '-' + selected_metric_data['season'].astype(str)
                    # Create a sort key with a value of 1 for Summer (S) and 2 for Winter (W)
                    selected_metric_data['season_num'] = selected_metric_data['season'].apply(lambda x: 1 if x == 'S' else 2)
                    selected_metric_data['sort_key'] = selected_metric_data['year'] * 10 + selected_metric_data['season_num']
                elif 'month' in selected_metric_data.columns and 'month' in groupby_time_cols:
                    # Handle month aggregation
                    selected_metric_data['time_label'] = selected_metric_data['year'].astype(str) + '-' + selected_metric_data['month'].astype(str)
                    selected_metric_data['sort_key'] = selected_metric_data['year'] * 100 + selected_metric_data['month']
                elif 'week' in selected_metric_data.columns and 'week' in groupby_time_cols:
                    # Handle week aggregation
                    selected_metric_data['time_label'] = selected_metric_data['year'].astype(str) + '-W' + selected_metric_data['week'].astype(str)
                    selected_metric_data['sort_key'] = selected_metric_data['year'] * 100 + selected_metric_data['week']
                else:
                    selected_metric_data['time_label'] = selected_metric_data['year'].astype(str)
                    selected_metric_data['sort_key'] = selected_metric_data['year']
            else:
                selected_metric_data['time_label'] = 'Unknown'
                selected_metric_data['sort_key'] = 0

            # Sort data by sort_key in ascending order
            selected_metric_data = selected_metric_data.sort_values('sort_key', ascending=True)

            # Get the unique time labels in the sorted order
            sorted_time_labels = selected_metric_data['time_label'].unique()

            # Add traces for each destination
            for dest in sorted(unique_destinations):
                dest_data = selected_metric_data[selected_metric_data[selected_destination_level] == dest]
                if not dest_data.empty:
                    fig.add_trace(
                        go.Bar(
                            x=dest_data['time_label'],
                            y=dest_data[metric_column],
                            name=dest,
                            marker_color=color_mapping[dest],
                            legendgroup=dest,
                        )
                    )

            # Update x-axis to enforce the correct order
            fig.update_xaxes(
                categoryorder='array',
                categoryarray=sorted_time_labels
            )

        except Exception as e:
            pass

        # Apply professional styling
        destination_display = "Countries" if selected_destination_level == "destination_country_name" else "Shipping Regions"
        
        fig.update_layout(
            title=None,
            barmode='stack',
            height=600,
            paper_bgcolor=PROFESSIONAL_COLORS['bg_white'],
            plot_bgcolor=PROFESSIONAL_COLORS['bg_white'],
            font=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                size=12,
                color=PROFESSIONAL_COLORS['text_secondary']
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.15,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)',
                borderwidth=0,
                font=dict(size=10, color='#4A4A4A'),
                itemsizing='constant'
            ),
            margin=dict(l=60, r=60, t=80, b=80),
        )

        # Update axes with professional styling
        fig.update_xaxes(
            title_text=x_axis_title,
            title_font=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                size=13,
                color=PROFESSIONAL_COLORS['text_primary']
            ),
            tickfont=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                size=11,
                color=PROFESSIONAL_COLORS['text_secondary']
            ),
            gridcolor=PROFESSIONAL_COLORS['grid_color'],
            gridwidth=0.5,
            linecolor=PROFESSIONAL_COLORS['grid_color'],
            linewidth=1,
            showgrid=True,
            zeroline=False
        )
        
        # Y-axis title with professional styling
        fig.update_yaxes(
            title_text=metric_unit,
            title_font=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                size=13,
                color=PROFESSIONAL_COLORS['text_primary']
            ),
            tickfont=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                size=11,
                color=PROFESSIONAL_COLORS['text_secondary']
            ),
            gridcolor=PROFESSIONAL_COLORS['grid_color'],
            gridwidth=0.5,
            linecolor=PROFESSIONAL_COLORS['grid_color'],
            linewidth=1,
            showgrid=True,
            zeroline=False
        )

        # --- Prepare Data for Tables 1 & 2 ---
        # Apply vessel type filter appropriately
        if selected_vessel_type and selected_vessel_type != 'All':
            table_filters = {'status': selected_status, 'vessel_type': selected_vessel_type}
        else:
            table_filters = {'status': selected_status}  # No vessel filter if 'All'

        # Use the same metric as selected in the chart dropdown
        # Get the metric column from the chart metric mapping
        chart_metric_mapping = {
            'count_trades': 'count_trades',
            'mtpa': 'sum_cargo_destination_cubic_meters', 
            'mcm_d': 'sum_cargo_destination_cubic_meters',
            'm3': 'sum_cargo_destination_cubic_meters',
            'median_delivery_days': 'median_delivery_days',
            'median_speed': 'median_speed',
            'median_mileage_nautical_miles': 'median_mileage_nautical_miles',
            'median_ton_miles': 'median_ton_miles',
            'median_utilization_rate': 'median_utilization_rate',
            'median_cargo_destination_cubic_meters': 'median_cargo_destination_cubic_meters',
            'median_vessel_capacity_cubic_meters': 'median_vessel_capacity_cubic_meters'
        }
        
        selected_metric_column = chart_metric_mapping.get(selected_chart_metric, 'count_trades')
        
        # Determine aggregation function based on metric type
        if 'median' in selected_chart_metric:
            agg_func = 'median'
        else:
            agg_func = 'sum'
        
        # Use the existing prepare_pivot_table function with selected metric
        table_source_df = convert_trade_analysis_volume_metric(
            df_trades_shipping_region_us,
            selected_metric_column,
            selected_aggregation,
            selected_chart_metric
        )

        count_table_data = prepare_pivot_table(
            df=table_source_df,
            values_col=selected_metric_column,
            filters=table_filters,
            aggregation_level=selected_aggregation,
            add_total_column=True,
            aggfunc=agg_func,
            destination_level=selected_destination_level
        )

        return fig

    except Exception as e:
        error_message = f"Error updating US region visuals/tables: {e}"
        empty_fig.update_layout(title_text=error_message)
        return empty_fig


@callback(
    Output('download-trade-analysis-excel', 'data'),
    Input('export-trade-analysis-button', 'n_clicks'),
    State('us-region-data-store', 'data'),
    State('us-dropdown-options-store', 'data'),
    State('aggregation-dropdown', 'value'),
    State('us-region-status-dropdown', 'value'),
    State('us-vessel-type-dropdown', 'value'),
    State('destination-level-dropdown', 'value'),
    State('origin-country-dropdown', 'value'),
    State('chart-metric-dropdown', 'value'),
    prevent_initial_call=True
)
def export_trade_analysis_to_excel(n_clicks, us_shipping_data, us_dropdown_options,
                                   selected_aggregation, selected_status, selected_vessel_type,
                                   selected_destination_level, origin_country, selected_chart_metric):
    if not n_clicks:
        raise PreventUpdate
    if us_shipping_data is None or not all([selected_aggregation, selected_status, selected_destination_level]):
        raise PreventUpdate

    try:
        df = pd.read_json(StringIO(us_shipping_data), orient='split')
        if df.empty:
            raise PreventUpdate

        chart_metric_mapping = {
            'count_trades': 'count_trades',
            'mtpa': 'sum_cargo_destination_cubic_meters',
            'mcm_d': 'sum_cargo_destination_cubic_meters',
            'm3': 'sum_cargo_destination_cubic_meters',
            'median_delivery_days': 'median_delivery_days',
            'median_speed': 'median_speed',
            'median_mileage_nautical_miles': 'median_mileage_nautical_miles',
            'median_ton_miles': 'median_ton_miles',
            'median_utilization_rate': 'median_utilization_rate',
            'median_cargo_destination_cubic_meters': 'median_cargo_destination_cubic_meters',
            'median_vessel_capacity_cubic_meters': 'median_vessel_capacity_cubic_meters'
        }
        selected_metric_column = chart_metric_mapping.get(selected_chart_metric, 'count_trades')
        agg_func = 'median' if 'median' in selected_chart_metric else 'sum'

        if selected_vessel_type and selected_vessel_type != 'All':
            table_filters = {'status': selected_status, 'vessel_type': selected_vessel_type}
        else:
            table_filters = {'status': selected_status}

        table_source_df = convert_trade_analysis_volume_metric(
            df,
            selected_metric_column,
            selected_aggregation,
            selected_chart_metric
        )

        pivot_df = prepare_pivot_table(
            df=table_source_df,
            values_col=selected_metric_column,
            filters=table_filters,
            aggregation_level=selected_aggregation,
            add_total_column=True,
            aggfunc=agg_func,
            destination_level=selected_destination_level
        )

        if pivot_df.empty:
            raise PreventUpdate

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pivot_df.to_excel(writer, sheet_name='Trade Analysis', index=False)
            for worksheet in writer.sheets.values():
                for column_cells in worksheet.columns:
                    max_length = 0
                    column_letter = column_cells[0].column_letter
                    for cell in column_cells:
                        cell_value = "" if cell.value is None else str(cell.value)
                        if len(cell_value) > max_length:
                            max_length = len(cell_value)
                    worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)

        output.seek(0)
        safe_country = "".join(c if c.isalnum() else "_" for c in (origin_country or "country")).strip("_") or "country"
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_country}_Trade_Analysis_{selected_aggregation}_{timestamp}.xlsx"
        return dcc.send_bytes(output.getvalue(), filename)

    except PreventUpdate:
        raise
    except Exception as e:
        raise PreventUpdate


@callback(
    Output('destination-summary-table-container', 'children'),
    [Input('origin-country-dropdown', 'value'),
     Input('supply-rolling-window-input', 'value'),
     Input('us-region-status-dropdown', 'value'),
     Input('us-vessel-type-dropdown', 'value'),
     Input('destination-expanded-continents', 'data'),
     Input('destination-level-dropdown', 'value')],
    prevent_initial_call=False
)
def update_destination_summary_table(origin_country, rolling_window_days, status, vessel_type, expanded_continents, destination_level):
    """Update the destination summary table with expandable continent/country hierarchy."""
    
    
    if not origin_country:
        return html.Div("Please select an origin country.", 
                       style={'textAlign': 'center', 'padding': '20px'})
    
    try:
        # Initialize expanded continents if None
        expanded_continents = expanded_continents or []
        
        destination_level = destination_level or 'destination_country_name'

        # Fetch the data
        df = fetch_destination_summary_data(
            engine,
            origin_country,
            status,
            vessel_type,
            rolling_window_days,
            destination_level=destination_level
        )

        if df.empty:
            return html.Div("No data available for the selected filters.",
                           style={'textAlign': 'center', 'padding': '20px'})

        # Sort groups by 30D descending before display
        rolling_col = next((c for c in df.columns if c.endswith('D') and c[:-1].isdigit() and c != '7D'), None)
        if rolling_col and rolling_col in df.columns:
            continent_order = df.groupby('continent')[rolling_col].sum().sort_values(ascending=False).index.tolist()
            df = pd.concat([df[df['continent'] == c] for c in continent_order], ignore_index=True)

        # expand/collapse always works: continent = selected group, country = destination country
        display_df = prepare_destination_table_for_display(df, expanded_continents)
        
        # Create column definitions
        columns = []
        col_display_names = {'Continent': 'Destination Level', 'Country': 'Country'}
        for col in display_df.columns:
            if col in ['Continent', 'Country']:
                columns.append({
                    'name': col_display_names.get(col, col),
                    'id': col,
                    'type': 'text'
                })
            else:
                columns.append({
                    'name': col,
                    'id': col,
                    'type': 'numeric',
                    'format': {'specifier': '.0f'},
                })

        # Create conditional styles
        conditional_styles = []

        # Style for continent total rows
        conditional_styles.append({
            'if': {'filter_query': '{Country} = "Total"'},
            'backgroundColor': '#e3f2fd',
            'fontWeight': 'bold'
        })
        
        # Style for indented countries
        conditional_styles.append({
            'if': {'filter_query': '{Continent} = ""'},
            'backgroundColor': '#f9f9f9',
            'fontSize': '13px'
        })
        
        # Alternating row colors
        conditional_styles.append({
            'if': {'row_index': 'odd'},
            'backgroundColor': '#f5f5f5'
        })
        
        # Left align text columns
        conditional_styles.append({
            'if': {'column_id': 'Continent'},
            'textAlign': 'left'
        })
        conditional_styles.append({
            'if': {'column_id': 'Country'},
            'textAlign': 'left'
        })
        
        # Right align numeric columns
        for col in display_df.columns:
            if col not in ['Continent', 'Country']:
                conditional_styles.append({
                    'if': {'column_id': col},
                    'textAlign': 'right',
                    'paddingRight': '12px'
                })
        
        # Get column groups for adding white separators
        quarter_columns = [col for col in display_df.columns if col.startswith('Q') and "'" in col]
        month_columns = [col for col in display_df.columns if "'" in col and not col.startswith('Q') and not col.startswith('W') and col not in ['Continent', 'Country']]
        week_columns = [col for col in display_df.columns if col.startswith('W') and "'" in col]
        rolling_window_columns = [
            col for col in display_df.columns
            if col == '7D' or (col.endswith('D') and col[:-1].isdigit())
        ]
        delta_vs_window_column = next((col for col in display_df.columns if col.startswith('Δ 7D-')), None)
        delta_yoy_column = next((col for col in display_df.columns if col.startswith('Δ ') and col.endswith(' Y/Y')), None)
        
        # Color coding for different time periods and add white separators
        for col in display_df.columns:
            if col.startswith('Q') and "'" in col:  # Quarter columns
                # Add left border to first quarter column for visual separation
                if quarter_columns and col == quarter_columns[0]:
                    conditional_styles.append({
                        'if': {'column_id': col},
                        'borderLeft': '3px solid white'
                    })
            elif col.startswith('W') and "'" in col:  # Week columns
                # Add left border to first week column for visual separation from the rolling window
                if week_columns and col == week_columns[0]:
                    conditional_styles.append({
                        'if': {'column_id': col},
                        'borderLeft': '3px solid white'
                    })
            elif "'" in col and not col.startswith('Q') and not col.startswith('W'):  # Month columns
                # Add left border to first month column for visual separation
                if month_columns and col == month_columns[0]:
                    conditional_styles.append({
                        'if': {'column_id': col},
                        'borderLeft': '3px solid white'
                    })
            elif col in rolling_window_columns:  # Rolling windows
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#fff3e0',  # Light orange
                    'fontWeight': '500'
                })
            elif col == delta_vs_window_column:  # Delta column
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#f5f5f5',
                    'fontWeight': '600',
                    'borderLeft': '3px solid white'  # Add separator before first delta
                })
                # Green for positive
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': f'{{{col}}} > 0'
                    },
                    'color': '#2e7d32'
                })
                # Red for negative
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': f'{{{col}}} < 0'
                    },
                    'color': '#c62828'
                })
            elif col == delta_yoy_column:  # Year-over-year delta
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#e8f5e9',
                    'fontWeight': '600',
                    'borderLeft': '3px solid white'  # Add separator between the two deltas
                })
                # Dark green for positive
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': f'{{{col}}} > 0'
                    },
                    'color': '#1b5e20'
                })
                # Dark red for negative
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': f'{{{col}}} < 0'
                    },
                    'color': '#b71c1c'
                })
        
        # Style for GRAND TOTAL row - added last for highest priority
        conditional_styles.append({
            'if': {'filter_query': '{Continent} = "GRAND TOTAL"'},
            'backgroundColor': '#2E86C1',
            'color': 'white',
            'fontWeight': 'bold'
        })
        
        # Add header styles for the white column separators
        header_styles = []

        # Background colors per column type
        for col in quarter_columns:
            header_styles.append({'if': {'column_id': col}, 'backgroundColor': '#e3f2fd'})
        for col in month_columns:
            header_styles.append({'if': {'column_id': col}, 'backgroundColor': '#f3e5f5'})
        for col in week_columns:
            header_styles.append({'if': {'column_id': col}, 'backgroundColor': '#e8f5e9'})
        for col in rolling_window_columns:
            header_styles.append({'if': {'column_id': col}, 'backgroundColor': '#fff3e0'})
        if delta_vs_window_column:
            header_styles.append({'if': {'column_id': delta_vs_window_column}, 'backgroundColor': '#f5f5f5'})
        if delta_yoy_column:
            header_styles.append({'if': {'column_id': delta_yoy_column}, 'backgroundColor': '#e8f5e9'})

        # Add separator before first quarter column
        if quarter_columns:
            header_styles.append({
                'if': {'column_id': quarter_columns[0]},
                'borderLeft': '3px solid white'
            })
        
        # Add separator before first month column
        if month_columns:
            header_styles.append({
                'if': {'column_id': month_columns[0]},
                'borderLeft': '3px solid white'
            })
        
        # Add separator before first week column
        if week_columns:
            header_styles.append({
                'if': {'column_id': week_columns[0]},
                'borderLeft': '3px solid white'
            })
        
        # Add separator before first delta column
        if delta_vs_window_column:
            header_styles.append({
                'if': {'column_id': delta_vs_window_column},
                'borderLeft': '3px solid white'
            })
        
        # Add separator before second delta column
        if delta_yoy_column:
            header_styles.append({
                'if': {'column_id': delta_yoy_column},
                'borderLeft': '3px solid white'
            })
        
        # Create the DataTable with expandable ID for click handling
        table = dash_table.DataTable(
            id={'type': 'destination-expandable-table', 'index': 'summary'},
            data=display_df.to_dict('records'),
            columns=columns,
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#2E86C1',
                'color': 'white',
                'fontWeight': 'bold',
                'fontSize': '12px',
                'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                'textAlign': 'center'
            },
            style_header_conditional=header_styles,  # Apply the header separators
            style_cell={
                'textAlign': 'center',
                'fontSize': '12px',
                'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                'padding': '8px',
                'minWidth': '80px'
            },
            style_data_conditional=conditional_styles,
            sort_action='native',
            page_size=50,
            fill_width=False
        )
        
        return table
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div(f"Error loading data: {str(e)}", 
                       style={'textAlign': 'center', 'padding': '20px', 'color': 'red'})


@callback(
    Output('origin-plant-summary-table-container', 'children'),
    [Input('origin-country-dropdown', 'value'),
     Input('supply-rolling-window-input', 'value'),
     Input('us-region-status-dropdown', 'value'),
     Input('us-vessel-type-dropdown', 'value'),
     Input('origin-plant-expanded-zones', 'data')],
    prevent_initial_call=False
)
def update_origin_plant_summary_table(origin_country, rolling_window_days, status, vessel_type, expanded_zones):
    """Update the origin plant summary table with expandable zone/plant hierarchy."""
    if not origin_country:
        return html.Div("Please select an origin country.", style={'textAlign': 'center', 'padding': '20px'})

    try:
        expanded_zones = expanded_zones or []

        df = fetch_origin_plant_summary_data(engine, origin_country, rolling_window_days)

        if df.empty:
            return html.Div("No data available for the selected filters.",
                            style={'textAlign': 'center', 'padding': '20px'})

        # Sort zones by 30D descending before display
        rolling_col = next((c for c in df.columns if c.endswith('D') and c[:-1].isdigit() and c != '7D'), None)
        if rolling_col and rolling_col in df.columns:
            zone_order = df.groupby('continent')[rolling_col].sum().sort_values(ascending=False).index.tolist()
            df = pd.concat([df[df['continent'] == z] for z in zone_order], ignore_index=True)

        # Build flat zone-only table: sum each zone, no expand/collapse needed
        numeric_cols = [c for c in df.columns if c not in ['continent', 'country']]
        zone_rows = []
        for zone in df['continent'].unique():
            zone_data = df[df['continent'] == zone]
            zone_rows.append({'Zone': zone, **{c: zone_data[c].sum() for c in numeric_cols}})
        # Add GRAND TOTAL
        zone_rows.append({'Zone': 'GRAND TOTAL', **{c: df[c].sum() for c in numeric_cols}})
        display_df = pd.DataFrame(zone_rows)

        # Column definitions
        columns = []
        for col in display_df.columns:
            if col == 'Zone':
                columns.append({'name': col, 'id': col, 'type': 'text'})
            else:
                columns.append({'name': col, 'id': col, 'type': 'numeric', 'format': {'specifier': '.0f'}})

        # Conditional styles
        conditional_styles = []
        # Alternating blue/grey rows like continent total rows
        conditional_styles.append({'if': {'filter_query': '{Zone} != "GRAND TOTAL"'}, 'backgroundColor': '#e3f2fd', 'fontWeight': 'bold'})
        conditional_styles.append({'if': {'row_index': 'odd', 'filter_query': '{Zone} != "GRAND TOTAL"'}, 'backgroundColor': '#f5f5f5', 'fontWeight': 'bold'})
        conditional_styles.append({'if': {'column_id': 'Zone'}, 'textAlign': 'left'})

        for col in display_df.columns:
            if col != 'Zone':
                conditional_styles.append({'if': {'column_id': col}, 'textAlign': 'right', 'paddingRight': '12px'})

        quarter_columns = [col for col in display_df.columns if col.startswith('Q') and "'" in col]
        month_columns = [col for col in display_df.columns if "'" in col and not col.startswith('Q') and not col.startswith('W') and col != 'Zone']
        week_columns = [col for col in display_df.columns if col.startswith('W') and "'" in col]
        rolling_window_columns = [col for col in display_df.columns if col == '7D' or (col.endswith('D') and col[:-1].isdigit())]
        delta_vs_window_column = next((col for col in display_df.columns if col.startswith('Δ 7D-')), None)
        delta_yoy_column = next((col for col in display_df.columns if col.startswith('Δ ') and col.endswith(' Y/Y')), None)

        for col in display_df.columns:
            if col.startswith('Q') and "'" in col:
                if quarter_columns and col == quarter_columns[0]:
                    conditional_styles.append({'if': {'column_id': col}, 'borderLeft': '3px solid white'})
            elif col.startswith('W') and "'" in col:
                if week_columns and col == week_columns[0]:
                    conditional_styles.append({'if': {'column_id': col}, 'borderLeft': '3px solid white'})
            elif "'" in col and not col.startswith('Q') and not col.startswith('W'):
                if month_columns and col == month_columns[0]:
                    conditional_styles.append({'if': {'column_id': col}, 'borderLeft': '3px solid white'})
            elif col in rolling_window_columns:
                conditional_styles.append({'if': {'column_id': col}, 'backgroundColor': '#fff3e0', 'fontWeight': '500'})
            elif col == delta_vs_window_column:
                conditional_styles.append({'if': {'column_id': col}, 'backgroundColor': '#f5f5f5', 'fontWeight': '600', 'borderLeft': '3px solid white'})
                conditional_styles.append({'if': {'column_id': col, 'filter_query': f'{{{col}}} > 0'}, 'color': '#2e7d32'})
                conditional_styles.append({'if': {'column_id': col, 'filter_query': f'{{{col}}} < 0'}, 'color': '#c62828'})
            elif col == delta_yoy_column:
                conditional_styles.append({'if': {'column_id': col}, 'backgroundColor': '#e8f5e9', 'fontWeight': '600', 'borderLeft': '3px solid white'})
                conditional_styles.append({'if': {'column_id': col, 'filter_query': f'{{{col}}} > 0'}, 'color': '#1b5e20'})
                conditional_styles.append({'if': {'column_id': col, 'filter_query': f'{{{col}}} < 0'}, 'color': '#b71c1c'})

        conditional_styles.append({'if': {'filter_query': '{Zone} = "GRAND TOTAL"'}, 'backgroundColor': '#2E86C1', 'color': 'white', 'fontWeight': 'bold'})

        header_styles = []
        for col in quarter_columns:
            header_styles.append({'if': {'column_id': col}, 'backgroundColor': '#e3f2fd'})
        for col in month_columns:
            header_styles.append({'if': {'column_id': col}, 'backgroundColor': '#f3e5f5'})
        for col in week_columns:
            header_styles.append({'if': {'column_id': col}, 'backgroundColor': '#e8f5e9'})
        for col in rolling_window_columns:
            header_styles.append({'if': {'column_id': col}, 'backgroundColor': '#fff3e0'})
        if delta_vs_window_column:
            header_styles.append({'if': {'column_id': delta_vs_window_column}, 'backgroundColor': '#f5f5f5'})
        if delta_yoy_column:
            header_styles.append({'if': {'column_id': delta_yoy_column}, 'backgroundColor': '#e8f5e9'})
        if quarter_columns:
            header_styles.append({'if': {'column_id': quarter_columns[0]}, 'borderLeft': '3px solid white'})
        if month_columns:
            header_styles.append({'if': {'column_id': month_columns[0]}, 'borderLeft': '3px solid white'})
        if week_columns:
            header_styles.append({'if': {'column_id': week_columns[0]}, 'borderLeft': '3px solid white'})
        if delta_vs_window_column:
            header_styles.append({'if': {'column_id': delta_vs_window_column}, 'borderLeft': '3px solid white'})
        if delta_yoy_column:
            header_styles.append({'if': {'column_id': delta_yoy_column}, 'borderLeft': '3px solid white'})

        table = dash_table.DataTable(
            id={'type': 'origin-plant-expandable-table', 'index': 'summary'},
            data=display_df.to_dict('records'),
            columns=columns,
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#2E86C1',
                'color': 'white',
                'fontWeight': 'bold',
                'fontSize': '12px',
                'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                'textAlign': 'center'
            },
            style_header_conditional=header_styles,
            style_cell={
                'textAlign': 'center',
                'fontSize': '12px',
                'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                'padding': '8px',
                'minWidth': '80px'
            },
            style_data_conditional=conditional_styles,
            sort_action='native',
            page_size=50,
            fill_width=False
        )
        return table

    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div(f"Error loading data: {str(e)}", style={'textAlign': 'center', 'padding': '20px', 'color': 'red'})


@callback(
    Output('graph-route-suez-only', 'figure'),
    [Input('aggregation-dropdown', 'value'),
     Input('destination-level-dropdown', 'value'),
     Input('origin-country-dropdown', 'value')]
)
def update_route_analysis_charts_and_tables(agg_level, destination_level, origin_country_name):
    """
    This callback updates the route analysis charts and tables based on the selected aggregation level
    and destination level. Uses a unified subplot figure with shared legend.
    """
    # --- Process the data ---
    try:
        processed_df = process_trade_and_distance_data(engine, origin_country_name=origin_country_name)
        # Verify destination_level column exists in the data
        if destination_level not in processed_df.columns:
            # If destination_country_name is requested but not in data, attempt to add it
            if destination_level == 'destination_country_name' and 'destination_shipping_region' in processed_df.columns:
                try:
                    # Pull the mapping from regions to countries
                    query_regions = f'''
                        SELECT DISTINCT country, shipping_region
                        FROM {DB_SCHEMA}.mappings_country
                    '''
                    region_map_df = pd.read_sql(query_regions, engine)
                    region_map_df.columns = ['destination_country_name', 'destination_shipping_region']
                    # Merge the mapping to add destination_country_name
                    processed_df = pd.merge(
                        processed_df,
                        region_map_df,
                        on='destination_shipping_region',
                        how='left'
                    )
                except Exception as mapping_err:
                    # Continue with destination_shipping_region as fallback
                    destination_level = 'destination_shipping_region'
            elif destination_level == 'destination_shipping_region' and 'destination_country_name' in processed_df.columns:
                try:
                    # Pull the mapping from countries to regions
                    query_regions = f'''
                        SELECT DISTINCT country, shipping_region
                        FROM {DB_SCHEMA}.mappings_country
                    '''
                    region_map_df = pd.read_sql(query_regions, engine)
                    region_map_df.columns = ['destination_country_name', 'destination_shipping_region']
                    # Merge the mapping to add destination_shipping_region
                    processed_df = pd.merge(
                        processed_df,
                        region_map_df,
                        on='destination_country_name',
                        how='left'
                    )
                except Exception as mapping_err:
                    # Continue with destination_country_name as fallback
                    destination_level = 'destination_country_name'
            elif destination_level not in processed_df.columns:
                # For basin, continent, subcontinent, classification levels
                level_col_map = {
                    'continent_destination_name': 'continent',
                    'destination_basin': 'basin',
                    'destination_subcontinent': 'subcontinent',
                    'destination_classification_level1': 'country_classification_level1',
                    'destination_classification': 'country_classification',
                }
                mapping_col = level_col_map.get(destination_level)
                if mapping_col and 'destination_country_name' in processed_df.columns:
                    try:
                        mapping_df = pd.read_sql(
                            f"SELECT DISTINCT country_name AS destination_country_name, {mapping_col} AS \"{destination_level}\" "
                            f"FROM {DB_SCHEMA}.mappings_country WHERE country_name IS NOT NULL",
                            engine
                        )
                        processed_df = pd.merge(processed_df, mapping_df, on='destination_country_name', how='left')
                    except Exception:
                        pass  # fall through to the column-not-found error below
    except Exception as e:
        # Create error figure and table
        error_fig = go.Figure().update_layout(
            title="Error Processing Data",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return error_fig

    # Check again if the destination level column exists after attempted mapping
    if destination_level not in processed_df.columns:
        error_msg = f"Selected destination level column '{destination_level}' not found in processed data."
        error_fig = go.Figure().update_layout(
            title=error_msg,
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return error_fig

    # --- Data Filtering for the four scenarios ---
    try:
        # Ensure the destination column is available for all dataframes
        required_cols = ['destination_country_name', 'destination_shipping_region']
        for col in required_cols:
            if col not in processed_df.columns:
                # Create the column if it doesn't exist
                processed_df[col] = None

        # 1. Suez available, Panama not available
        df_suez_only = processed_df[
            processed_df['distanceDirect'].notna() &
            processed_df['distanceViaSuez'].notna() &
            processed_df['distanceViaPanama'].isna()
            ].copy()

        # 2. Panama available, Suez not available
        df_panama_only = processed_df[
            processed_df['distanceDirect'].notna() &
            processed_df['distanceViaPanama'].notna() &
            processed_df['distanceViaSuez'].isna()
            ].copy()

        # 3. Both Suez and Panama available
        df_both = processed_df[
            processed_df['distanceDirect'].notna() &
            processed_df['distanceViaPanama'].notna() &
            processed_df['distanceViaSuez'].notna()
            ].copy()

        # 4. Direct only (no Suez, no Panama)
        df_direct_only = processed_df[
            processed_df['distanceDirect'].notna() &
            processed_df['distanceViaSuez'].isna() &
            processed_df['distanceViaPanama'].isna()
            ].copy()

        # Add a selected_route column to df_direct_only if it doesn't already have one
        if 'selected_route' not in df_direct_only.columns:
            df_direct_only['selected_route'] = 'Direct'
    except KeyError as e:
        error_fig = go.Figure().update_layout(
            title=f"Error: Missing Column {e}",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return error_fig
    except Exception as e:
        error_fig = go.Figure().update_layout(
            title="Error Filtering Data",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return error_fig

    # --- Create a unified subplot figure ---
    try:
        from plotly.subplots import make_subplots
        # Create a 1x3 subplot with secondary y-axes for each subplot
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Route Usage: Suez Available (Not Panama)",
                "Route Usage: Panama Available (Not Suez)",
                "Route Usage: Both Suez & Panama Available"
            ],
            horizontal_spacing=0.12,  # Increase spacing between subplots
            specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]]
        )

        # Define professional colors for routes
        route_colors = {
            'Direct': PROFESSIONAL_COLORS['primary'],      # McKinsey blue
            'ViaSuez': PROFESSIONAL_COLORS['warning'],     # Professional orange  
            'ViaPanama': PROFESSIONAL_COLORS['success']    # Professional green
        }

        # Track which items have already been shown in the legend
        shown_in_legend = set()

        # Define figure update function to avoid repetitive code
        def add_stacked_bar_line_to_subplot(df, df_name, row, col):
            if df is None or df.empty:
                return False  # Return False if no data

            # Determine index columns based on aggregation level
            if agg_level == 'Year':
                index_cols = ['year']
            elif agg_level == 'Year+Season':
                index_cols = ['year', 'season']
            elif agg_level == 'Year+Quarter':
                index_cols = ['year', 'quarter']
            elif agg_level == 'Month':
                index_cols = ['year', 'month']
            elif agg_level == 'Week':
                index_cols = ['year', 'week']
            else:
                return False

            # Check for required columns
            required_cols = index_cols + ['selected_route', 'voyage_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return False

            # Group by aggregation level and route to get counts
            grouped_cols = index_cols + ['selected_route']
            try:
                grouped = df.groupby(grouped_cols, observed=True)['voyage_id'].count().unstack(fill_value=0)
            except Exception as e:
                return False

            if grouped.empty:
                return False

            # Calculate total trades
            total_trades = grouped.sum(axis=1)
            percentage_df = grouped.divide(total_trades, axis=0).fillna(0) * 100

           # Create x-axis labels from index
            if isinstance(percentage_df.index, pd.MultiIndex):
                if agg_level == 'Year+Quarter':
                    x_labels = [f"{idx[0]}-Q{idx[1][1:]}" for idx in percentage_df.index]
                elif agg_level == 'Year+Season':
                    x_labels = [f"{idx[0]}-{idx[1]}" for idx in percentage_df.index]
                elif agg_level == 'Month':
                    x_labels = [f"{idx[0]}-{idx[1]:02d}" for idx in percentage_df.index]
                elif agg_level == 'Week':
                    x_labels = [f"{idx[0]}-W{idx[1]:02d}" for idx in percentage_df.index]
                else:
                    x_labels = [' '.join(map(str, idx)) for idx in percentage_df.index]
            else:
                x_labels = percentage_df.index.tolist()

            # Order x-axis chronologically (oldest to newest)
            # Create a sort key based on year and quarter/season
            sort_keys = []
            for label in x_labels:
                parts = str(label).split('-')
                try:
                    year = int(parts[0])
                    if len(parts) > 1:
                        if parts[1].startswith('Q'):
                            quarter = int(parts[1][1:])
                            sort_key = year * 10 + quarter
                        elif parts[1] in ['S', 'W']:
                            season_num = 1 if parts[1] == 'S' else 2
                            sort_key = year * 10 + season_num
                        else:
                            sort_key = year * 100
                    else:
                        sort_key = year * 100
                except ValueError:
                    sort_key = 0  # Default sort key for non-numeric values
                sort_keys.append(sort_key)

            # Create sorted indices
            sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
            sorted_x_labels = [x_labels[i] for i in sorted_indices]

            # Add bar traces (percentage)
            available_routes = percentage_df.columns
            for route in available_routes:
                if route in route_colors:
                    color = route_colors[route]
                else:
                    # Generate a consistent color for this route
                    import hashlib
                    h = hashlib.md5(route.encode()).hexdigest()
                    r = int(h[:2], 16) % 200 + 30  # Values between 30-230
                    g = int(h[2:4], 16) % 200 + 30
                    b = int(h[4:6], 16) % 200 + 30
                    color = f"rgba({r}, {g}, {b}, 0.8)"
                    route_colors[route] = color  # Add to colors dict for consistency

                y_values = [percentage_df.iloc[idx][route] for idx in sorted_indices]

                # Important: Only show legend for the first instance of each route
                if f"route_{route}" not in shown_in_legend:
                    show_legend = True
                    shown_in_legend.add(f"route_{route}")
                else:
                    show_legend = False

                fig.add_trace(
                    go.Bar(
                        x=sorted_x_labels,
                        y=y_values,
                        name=f"{route}",
                        marker_color=color,
                        legendgroup=route,  # Group by route for consistent selection
                        showlegend=show_legend
                    ),
                    row=row, col=col,
                    secondary_y=False  # Explicitly set primary y-axis
                )

            # Add line trace (Total Count)
            y_values = [total_trades.iloc[idx] for idx in sorted_indices]

            # Show Total only once in the legend
            if "total" not in shown_in_legend:
                show_total_legend = True
                shown_in_legend.add("total")
            else:
                show_total_legend = False

            fig.add_trace(
                go.Scatter(
                    x=sorted_x_labels,
                    y=y_values,
                    name="Total Trades Count",
                    mode='lines+markers',
                    line=dict(color='black', width=3),
                    marker=dict(color='black', size=8),
                    legendgroup="Total",  # Group for consistent selection
                    showlegend=show_total_legend
                ),
                row=row, col=col,
                secondary_y=True  # Explicitly set secondary y-axis
            )

            # Update axis labels
            fig.update_xaxes(
                title_text=agg_level.replace('+', '-'),
                row=row, col=col,
                tickangle=45 if len(sorted_x_labels) > 3 else 0  # Rotate labels if many
            )

            return True  # Return True if data was added

        # Add the three charts to the subplot and track which have data
        has_data_1 = add_stacked_bar_line_to_subplot(df_suez_only, "df_suez_only", 1, 1)
        has_data_2 = add_stacked_bar_line_to_subplot(df_panama_only, "df_panama_only", 1, 2)
        has_data_3 = add_stacked_bar_line_to_subplot(df_both, "df_both", 1, 3)

        # Configure all y-axes AFTER adding all data
        # Primary y-axes (percentage)
        for col in range(1, 4):
            if (col == 1 and has_data_1) or (col == 2 and has_data_2) or (col == 3 and has_data_3):
                fig.update_yaxes(
                    title_text="Percentage of Trades (%)",
                    range=[0, 100],
                    ticksuffix="%",
                    showgrid=True,
                    showticklabels=True,
                    row=1, col=col,
                    secondary_y=False
                )

        # Secondary y-axes (total count)
        for col in range(1, 4):
            if (col == 1 and has_data_1) or (col == 2 and has_data_2) or (col == 3 and has_data_3):
                fig.update_yaxes(
                    title_text="Total Trade Count",
                    showgrid=False,
                    showticklabels=True,
                    row=1, col=col,
                    secondary_y=True
                )

        # Update the overall layout
        fig.update_layout(
            barmode='stack',
            height=500,
            legend=dict(
                orientation="v",  # Vertical orientation
                x=1.05,  # Position to the right of the last chart
                y=0.5,  # Center vertically
                xanchor='left',  # Anchor to the left side of the legend box
                yanchor='middle',  # Anchor to the middle of the legend box
                title_text='Route'
            ),
            margin=dict(l=60, r=150, t=70, b=120),  # Increase right margin for legend
            hovermode="x unified"
        )

    except Exception as e:
        fig = go.Figure().update_layout(title=f"Error generating charts: {str(e)}")

    # --- Ensure both destination columns exist in all dataframes ---
    for df in [df_suez_only, df_panama_only, df_both, df_direct_only]:
        if df is not None and not df.empty:
            # Make sure required destination columns exist
            for col in ['destination_country_name', 'destination_shipping_region']:
                if col not in df.columns:
                    df[col] = None

    return fig


@callback(
    Output('download-route-analysis-excel', 'data'),
    Input('export-route-analysis-button', 'n_clicks'),
    State('aggregation-dropdown', 'value'),
    State('destination-level-dropdown', 'value'),
    State('origin-country-dropdown', 'value'),
    prevent_initial_call=True
)
def export_route_analysis_to_excel(n_clicks, agg_level, destination_level, origin_country_name):
    if not n_clicks:
        raise PreventUpdate

    try:
        processed_df = process_trade_and_distance_data(engine, origin_country_name=origin_country_name)
        if processed_df is None or processed_df.empty:
            raise PreventUpdate
    except PreventUpdate:
        raise
    except Exception as e:
        raise PreventUpdate

    # Ensure destination columns exist
    for col in ['destination_country_name', 'destination_shipping_region']:
        if col not in processed_df.columns:
            processed_df[col] = None

    # Apply the 4 route filters
    df_suez_only = processed_df[
        processed_df['distanceDirect'].notna() &
        processed_df['distanceViaSuez'].notna() &
        processed_df['distanceViaPanama'].isna()
    ].copy()
    df_panama_only = processed_df[
        processed_df['distanceDirect'].notna() &
        processed_df['distanceViaPanama'].notna() &
        processed_df['distanceViaSuez'].isna()
    ].copy()
    df_both = processed_df[
        processed_df['distanceDirect'].notna() &
        processed_df['distanceViaPanama'].notna() &
        processed_df['distanceViaSuez'].notna()
    ].copy()
    df_direct_only = processed_df[
        processed_df['distanceDirect'].notna() &
        processed_df['distanceViaSuez'].isna() &
        processed_df['distanceViaPanama'].isna()
    ].copy()

    # Determine index columns from aggregation level
    if agg_level == 'Year':
        index_cols = ['year']
    elif agg_level == 'Year+Season':
        index_cols = ['year', 'season']
    elif agg_level == 'Year+Quarter':
        index_cols = ['year', 'quarter']
    elif agg_level == 'Month':
        index_cols = ['year', 'month']
    elif agg_level == 'Week':
        index_cols = ['year', 'week']
    else:
        index_cols = ['year']

    def build_route_pivot(df, col_level):
        if df is None or df.empty:
            return pd.DataFrame()
        required = index_cols + [col_level, 'voyage_id']
        if not all(c in df.columns for c in required):
            return pd.DataFrame()
        grouped = df.groupby(index_cols + [col_level], observed=True)['voyage_id'].count().reset_index()
        try:
            pivot = grouped.pivot_table(index=index_cols, columns=col_level, values='voyage_id', aggfunc='sum', fill_value=0)
            pivot.columns = [str(c) for c in pivot.columns]
            return pivot.reset_index()
        except Exception:
            return grouped

    sheets = {
        'Suez Only': build_route_pivot(df_suez_only, destination_level),
        'Panama Only': build_route_pivot(df_panama_only, destination_level),
        'Both Routes': build_route_pivot(df_both, destination_level),
        'Direct Only': build_route_pivot(df_direct_only, destination_level),
    }

    if all(df.empty for df in sheets.values()):
        raise PreventUpdate

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                for column_cells in worksheet.columns:
                    max_length = max((len(str(cell.value or "")) for cell in column_cells), default=0)
                    worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 50)

    output.seek(0)
    safe_country = "".join(c if c.isalnum() else "_" for c in (origin_country_name or "country")).strip("_") or "country"
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{safe_country}_Route_Analysis_{agg_level}_{timestamp}.xlsx"
    return dcc.send_bytes(output.getvalue(), filename)


@callback(
    Output('diversion-processed-data', 'data'),
    Input('global-refresh-button', 'n_clicks'),
    Input('origin-country-dropdown', 'value'),
    # prevent_initial_call=True
)
def process_diversion_data(n_clicks,origin_country_name):
    # -------------------------------- Table with the trades ----------------------------------------------------------#
    query = f'''
            select 
                diversion_date as "Diversion date",
                vessel_name as "Vessel",
                vessel_state as "State",
                charterer_name as "Charterer",
                cargo_origin_cubic_meters as "Cubic Meters",
                --cargo_origin_tons as "MT",
                origin_diversion_location_name as "Origin location",   
                --origin_diversion_continent_name as "Origin continent",   
                origin_diversion_country_name as "Origin country", 
                origin_diversion_date  as "Origin date",
                diverted_from_location_name as "Diverted from location",
                --diverted_from_continent_name as "Diverted from continent",
                diverted_from_country_name as "Diverted from country" ,
                diverted_from_date as "Diverted from date",
                new_destination_location_name as "New destination location",
                --new_destination_continent_name  as "New destination continent",
                new_destination_country_name as "New destination country",
                new_destination_date  as "New destination date"
                from {DB_SCHEMA}.kpler_lng_diversions
            where upload_timestamp_utc = (select max(upload_timestamp_utc) from {DB_SCHEMA}.kpler_lng_diversions)
            and origin_diversion_country_name = '{origin_country_name}'
        '''
    df_kpler_diversions = pd.read_sql(query, engine)
    df_kpler_diversions['Added shipping days'] = (df_kpler_diversions['New destination date'] - df_kpler_diversions['Diverted from date']).dt.days
    # Convert date columns to strings for JSON serialization
    date_columns = ['Diversion date', 'Origin date', 'Diverted from date', 'New destination date']
    for col in date_columns:
        df_kpler_diversions[col] = df_kpler_diversions[col].dt.date.astype(str)
    # Data frame to data for dash (filter for recent dates)
    # Convert to datetime for comparison, then back to string
    filter_date = dt.date(2024, 1, 1)
    df_kpler_diversions_filtered = df_kpler_diversions[pd.to_datetime(df_kpler_diversions['Diversion date']).dt.date >= filter_date]
    data_kpler_diversions = df_kpler_diversions_filtered.to_dict("records")
    # -------------------------------- charts ----------------------------------------------------------#
    df_kpler_charts = df_kpler_diversions.copy()
    df_kpler_charts = df_kpler_charts[df_kpler_charts.State=='Loaded']
    # country mapping
    query_country_mapping = f'''
                select  *
                    from {DB_SCHEMA}.mappings_country
            '''
    df_mapping_country = pd.read_sql(query_country_mapping, engine)
    # destination mapping
    query_location_mapping = f'''
                    select  *
                        from {DB_SCHEMA}.mapping_destination_location_name
                '''
    df_mapping_location = pd.read_sql(query_location_mapping, engine)
    ######## Mapings from Diverted from
    df_kpler_charts = pd.merge(df_kpler_charts,
             df_mapping_country[['country','basin','shipping_region']].rename(columns={'country':'Diverted from country',
                                                                                      'basin':'Diverted from basin 1',
                                                                                      'shipping_region':'Diverted from shipping region 1'}),
             on='Diverted from country',
             how='left')
    df_kpler_charts = pd.merge(df_kpler_charts,
                               df_mapping_location[['destination_location_name', 'basin', 'shipping_region']].rename(
                                   columns={'destination_location_name': 'Diverted from location',
                                            'basin': 'Diverted from basin 2',
                                            'shipping_region': 'Diverted from shipping region 2'}),
                               on='Diverted from location',
                               how='left')
    ######## Mapings for New destination
    df_kpler_charts = pd.merge(df_kpler_charts,
                               df_mapping_country[['country', 'basin', 'shipping_region']].rename(
                                   columns={'country': 'New destination country',
                                            'basin': 'New destination basin 1',
                                            'shipping_region': 'New destination shipping region 1'}),
                               on='New destination country',
                               how='left')
    df_kpler_charts = pd.merge(df_kpler_charts,
                               df_mapping_location[['destination_location_name', 'basin', 'shipping_region']].rename(
                                   columns={'destination_location_name': 'New destination location',
                                            'basin': 'New destination basin 2',
                                            'shipping_region': 'New destination shipping region 2'}),
                               on='New destination location',
                               how='left')
    ## Diverted from
    df_kpler_charts['Diverted from basin'] = np.where(df_kpler_charts['Diverted from basin 1'].isnull(),
                                                        df_kpler_charts['Diverted from basin 2'],
                                                        df_kpler_charts['Diverted from basin 1'])
    df_kpler_charts['Diverted from shipping region'] = np.where(
        df_kpler_charts['Diverted from shipping region 1'].isnull(),
        df_kpler_charts['Diverted from shipping region 2'],
        df_kpler_charts['Diverted from shipping region 1'])
    ## New destination
    df_kpler_charts['New destination basin'] = np.where(df_kpler_charts['New destination basin 1'].isnull(),
                                                        df_kpler_charts['New destination basin 2'],
                                                        df_kpler_charts['New destination basin 1'])
    df_kpler_charts['New destination shipping region'] = np.where(df_kpler_charts['New destination shipping region 1'].isnull(),
                                                        df_kpler_charts['New destination shipping region 2'],
                                                        df_kpler_charts['New destination shipping region 1'])
    # -----------------------------------------------------
    # Create new columns
    # -----------------------------------------------------
    # Convert "Diversion date" to year-month for grouping
    df_kpler_charts["Diversion_month"] = pd.to_datetime(df_kpler_charts["Diversion date"])
    # Extract year-month and convert to first day of month
    df_kpler_charts["Diversion_month"] = df_kpler_charts["Diversion_month"].dt.to_period("M").dt.to_timestamp()
    # Convert to string for JSON serialization
    df_kpler_charts["Diversion_month"] = df_kpler_charts["Diversion_month"].astype(str)
    # basin combo
    df_kpler_charts["basin_combo"] = (df_kpler_charts["Diverted from basin"].fillna('Unknown') + " -> " +
                                     df_kpler_charts["New destination basin"].fillna('Unknown'))
    # region combo
    df_kpler_charts["region_combo"] = (
                df_kpler_charts["Diverted from shipping region"].fillna('Unknown') + " -> " +
                df_kpler_charts["New destination shipping region"].fillna('Unknown'))
    # country combo
    df_kpler_charts["country_combo"] = (
                df_kpler_charts["Diverted from country"].fillna('Unknown') + " -> " +
                df_kpler_charts["New destination country"].fillna('Unknown'))
    # Replace any NaN values with None for JSON serialization
    df_kpler_charts = df_kpler_charts.where(pd.notnull(df_kpler_charts), None)
    # Convert to dict format to store in dcc.Store
    # Only keep necessary columns to reduce data size
    needed_columns = [
        'Diversion_month', 'basin_combo', 'region_combo', 'country_combo',
        'Added shipping days', 'Cubic Meters'#, 'MT'
    ]
    charts_data = df_kpler_charts[needed_columns].to_dict('records')
    # Store only the necessary data for the main table and charts
    return {
        'main_data': data_kpler_diversions,
        'charts_data': charts_data
    }


# Second callback to update all UI components using stored data
@callback(
    Output('diversion-table', 'data'),
    Output('diversion-table', 'columns'),
    Output('diversion-count-chart', 'figure'),
    Input('diversion-processed-data', 'data'),
    Input('destination-level-radio', 'value'),
)
def update_diversion_ui(stored_data, destination_level):
    if not stored_data:
        # Return empty/placeholder values if no data is available yet
        empty_fig = go.Figure().update_layout(title="No data available")
        empty_columns = [{"name": "No Data", "id": "no_data"}]
        return [], empty_columns, empty_fig

    # Get data from the store
    data_kpler_diversions = stored_data['main_data']

    # Create columns definition for the main diversion table
    if data_kpler_diversions and len(data_kpler_diversions) > 0:
        diversion_columns = [{"name": col, "id": col} for col in data_kpler_diversions[0].keys()]
    else:
        diversion_columns = [{"name": "No Data", "id": "no_data"}]

    # Convert charts data back to DataFrame
    df_kpler_charts = pd.DataFrame(stored_data['charts_data'])

    # Convert string back to datetime for Diversion_month
    df_kpler_charts["Diversion_month"] = pd.to_datetime(df_kpler_charts["Diversion_month"])

    # Format the Diversion_month column to a date format (YYYY-MM-DD)
    df_kpler_charts["Month_Display"] = df_kpler_charts["Diversion_month"].dt.strftime('%Y-%m-%d')

    # Determine which destination level to use based on user selection
    combo_field = destination_level  # 'basin_combo', 'region_combo', or 'country_combo'

    # -----------------------------------------------------------------
    # Get a common color mapping for all charts to ensure legend consistency
    # -----------------------------------------------------------------
    # First, get all unique values for the selected combo field
    all_combo_values = df_kpler_charts[combo_field].unique()

    # Create a professional color mapping dictionary
    n_combos = len(all_combo_values)
    professional_colors = get_professional_colors(n_combos)
    color_mapping = {combo: professional_colors[i] for i, combo in enumerate(sorted(all_combo_values))}

    # -----------------------------------------------------------------
    # PREPARE DATA FOR CHARTS
    # -----------------------------------------------------------------
    # Data for Count of trades
    df_count = df_kpler_charts.groupby(["Diversion_month", combo_field]).size().reset_index(name='Count')

    # Data for Added shipping days
    df_days = df_kpler_charts.groupby(["Diversion_month", combo_field], as_index=False)["Added shipping days"].sum()

    # Data for Cargo volumes
    df_volumes = df_kpler_charts.groupby(["Diversion_month", combo_field], as_index=False)["Cubic Meters"].sum()

    # -----------------------------------------------------------------
    # CREATE UNIFIED CHART WITH SUBPLOTS
    # -----------------------------------------------------------------


    # Create subplot with 1 row and 3 columns
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Count of Trades", "Added Shipping Days", "Cargo Volumes"),
        shared_xaxes=True,
        horizontal_spacing=0.08
    )

    # Add traces for Count chart (first subplot)
    for combo in sorted(all_combo_values):
        combo_data = df_count[df_count[combo_field] == combo]
        if not combo_data.empty:
            fig.add_trace(
                go.Bar(
                    x=combo_data["Diversion_month"],
                    y=combo_data["Count"],
                    name=combo,
                    marker_color=color_mapping[combo],
                    legendgroup=combo,  # Use legendgroup to link items across subplots
                ),
                row=1, col=1
            )

    # Add traces for Added Days chart (second subplot)
    for combo in sorted(all_combo_values):
        combo_data = df_days[df_days[combo_field] == combo]
        if not combo_data.empty:
            fig.add_trace(
                go.Bar(
                    x=combo_data["Diversion_month"],
                    y=combo_data["Added shipping days"],
                    name=combo,
                    marker_color=color_mapping[combo],
                    legendgroup=combo,  # Use legendgroup to link items across subplots
                    showlegend=False,  # Hide duplicate legends
                ),
                row=1, col=2
            )

    # Add traces for Cargo Volumes chart (third subplot)
    for combo in sorted(all_combo_values):
        combo_data = df_volumes[df_volumes[combo_field] == combo]
        if not combo_data.empty:
            fig.add_trace(
                go.Bar(
                    x=combo_data["Diversion_month"],
                    y=combo_data["Cubic Meters"],
                    name=combo,
                    marker_color=color_mapping[combo],
                    legendgroup=combo,  # Use legendgroup to link items across subplots
                    showlegend=False,  # Hide duplicate legends
                ),
                row=1, col=3
            )

    # Get origin country from data if available, otherwise use generic title
    try:
        # Try to extract origin country from the data
        origin_country = stored_data.get('origin_country', 'LNG')
        if not origin_country and df_kpler_charts is not None and not df_kpler_charts.empty:
            # Try to get from first row of data if available
            if 'Origin country' in df_kpler_charts.columns:
                origin_country = df_kpler_charts['Origin country'].iloc[0]
            else:
                origin_country = 'LNG'
    except:
        origin_country = 'LNG'
    
    # Apply professional styling for diversion charts
    fig.update_layout(
        barmode='stack',
        height=500,
        paper_bgcolor=PROFESSIONAL_COLORS['bg_white'],
        plot_bgcolor=PROFESSIONAL_COLORS['bg_white'],
        font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=12,
            color=PROFESSIONAL_COLORS['text_secondary']
        ),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.20,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            borderwidth=0,
            font=dict(size=10, color='#4A4A4A'),
            itemsizing='constant'
        ),
        margin=dict(l=60, r=60, t=80, b=130),
    )

    # Update axes with professional styling
    fig.update_xaxes(
        title_text="Month",
        title_font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=13,
            color=PROFESSIONAL_COLORS['text_primary']
        ),
        tickfont=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=11,
            color=PROFESSIONAL_COLORS['text_secondary']
        ),
        gridcolor=PROFESSIONAL_COLORS['grid_color'],
        gridwidth=0.5,
        linecolor=PROFESSIONAL_COLORS['grid_color'],
        linewidth=1,
        showgrid=True,
        zeroline=False
    )
    
    # Individual y-axis titles with professional styling
    fig.update_yaxes(
        title_text="Count",
        title_font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=13,
            color=PROFESSIONAL_COLORS['text_primary']
        ),
        tickfont=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=11,
            color=PROFESSIONAL_COLORS['text_secondary']
        ),
        gridcolor=PROFESSIONAL_COLORS['grid_color'],
        gridwidth=0.5,
        linecolor=PROFESSIONAL_COLORS['grid_color'],
        linewidth=1,
        showgrid=True,
        zeroline=False,
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Added Days",
        title_font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=13,
            color=PROFESSIONAL_COLORS['text_primary']
        ),
        tickfont=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=11,
            color=PROFESSIONAL_COLORS['text_secondary']
        ),
        gridcolor=PROFESSIONAL_COLORS['grid_color'],
        gridwidth=0.5,
        linecolor=PROFESSIONAL_COLORS['grid_color'],
        linewidth=1,
        showgrid=True,
        zeroline=False,
        row=1, col=2
    )
    
    fig.update_yaxes(
        title_text="Cubic Meters",
        title_font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=13,
            color=PROFESSIONAL_COLORS['text_primary']
        ),
        tickfont=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=11,
            color=PROFESSIONAL_COLORS['text_secondary']
        ),
        gridcolor=PROFESSIONAL_COLORS['grid_color'],
        gridwidth=0.5,
        linecolor=PROFESSIONAL_COLORS['grid_color'],
        linewidth=1,
        showgrid=True,
        zeroline=False,
        row=1, col=3
    )

    # -----------------------------------------------------
    # PIVOTED TABLES - Same as before
    # -----------------------------------------------------

    # Helper function to create multi-level column headers
    def create_split_header_columns(df):
        columns = []

        # First, add the Month column
        columns.append({
            "name": ["Month", ""],  # Multi-level header
            "id": "Month_Display"
        })

        # Then add all the combo columns with split headers
        for col in df.columns:
            if col != "Month_Display":
                # Split the column name by " -> "
                from_to_parts = str(col).split(" -> ")

                if len(from_to_parts) == 2:
                    from_part, to_part = from_to_parts
                    columns.append({
                        "name": [from_part, to_part],  # Multi-level header
                        "id": col
                    })
                else:
                    # Fallback if not in expected format
                    columns.append({
                        "name": [str(col), ""],
                        "id": col
                    })

        return columns

    return data_kpler_diversions, diversion_columns, fig


@callback(
    Output('download-diversion-summary-excel', 'data'),
    Input('export-diversion-summary-button', 'n_clicks'),
    State('diversion-table', 'data'),
    State('diversion-table', 'columns'),
    prevent_initial_call=True
)
def export_diversion_summary_to_excel(n_clicks, table_data, table_columns):
    if not n_clicks or not table_data:
        raise PreventUpdate
    df = pd.DataFrame(table_data)
    if table_columns:
        col_ids = [c['id'] for c in table_columns if c['id'] in df.columns]
        if col_ids:
            df = df[col_ids]
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Diversions Summary', index=False)
        worksheet = writer.sheets['Diversions Summary']
        for column_cells in worksheet.columns:
            max_length = max((len(str(cell.value or "")) for cell in column_cells), default=0)
            worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 50)
    output.seek(0)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(output.getvalue(), f'Diversions_Summary_{timestamp}.xlsx')


# Callback to handle expanding/collapsing rows for destination summary table
@callback(
    Output('destination-expanded-continents', 'data', allow_duplicate=True),
    [Input({'type': 'destination-expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'destination-expandable-table', 'index': ALL}, 'data'),
     State('destination-expanded-continents', 'data')],
    prevent_initial_call=True
)
def toggle_destination_continent_expansion(active_cells, table_data_list, expanded_continents):
    """Handle expanding/collapsing of continent rows in destination summary table"""
    
    if not any(active_cells):
        return expanded_continents or []
    
    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    
    # Parse which table and what was clicked
    if 'destination-expandable-table' in prop_id and '.active_cell' in prop_id:
        try:
            # Get the active cell
            active_cell = active_cells[0]
            if not active_cell:
                return expanded_continents or []
            
            # Get the data from the table
            table_data = table_data_list[0]
            if not table_data:
                return expanded_continents or []
            
            row_index = active_cell['row']
            col_id = active_cell['column_id']
            
            # Only respond to clicks on the Continent column
            if col_id != 'Continent':
                return expanded_continents or []
            
            # Get the clicked row
            clicked_row = table_data[row_index]
            continent_value = clicked_row.get('Continent', '')
            
            # Check if this is a continent total row (has arrow indicator)
            if continent_value.startswith('▶') or continent_value.startswith('▼'):
                # Extract continent name (remove arrow and spaces)
                continent_name = continent_value[2:].strip()
                
                # Initialize expanded list if None
                expanded_continents = expanded_continents or []
                
                # Toggle expansion state
                if continent_name in expanded_continents:
                    expanded_continents.remove(continent_name)
                else:
                    expanded_continents.append(continent_name)
                
                return expanded_continents
            
        except Exception as e:
            pass

    return expanded_continents or []


# Callback to handle expanding/collapsing rows for origin plant summary table
@callback(
    Output('origin-plant-expanded-zones', 'data', allow_duplicate=True),
    [Input({'type': 'origin-plant-expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'origin-plant-expandable-table', 'index': ALL}, 'data'),
     State('origin-plant-expanded-zones', 'data')],
    prevent_initial_call=True
)
def toggle_origin_plant_zone_expansion(active_cells, table_data_list, expanded_zones):
    """Handle expanding/collapsing of zone rows in origin plant summary table."""
    if not any(active_cells):
        return expanded_zones or []

    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']

    if 'origin-plant-expandable-table' in prop_id and '.active_cell' in prop_id:
        try:
            active_cell = active_cells[0]
            if not active_cell:
                return expanded_zones or []

            table_data = table_data_list[0]
            if not table_data:
                return expanded_zones or []

            row_index = active_cell['row']
            col_id = active_cell['column_id']

            if col_id != 'Zone':
                return expanded_zones or []

            clicked_row = table_data[row_index]
            zone_value = clicked_row.get('Zone', '')

            if zone_value.startswith('▶') or zone_value.startswith('▼'):
                zone_name = zone_value[2:].strip()
                expanded_zones = expanded_zones or []
                if zone_name in expanded_zones:
                    expanded_zones.remove(zone_name)
                else:
                    expanded_zones.append(zone_name)
                return expanded_zones

        except Exception:
            pass

    return expanded_zones or []


# Callback for Train Maintenance Schedule
@callback(
    Output('maintenance-summary-container', 'children'),
    Input('origin-country-dropdown', 'value'),
    State('maintenance-expanded-plants', 'data')
)
def update_maintenance_table(selected_country, expanded_plants):
    """
    Update maintenance table based on selected country.
    Shows planned and unplanned maintenance outages converted to MCM/D impact.
    """
    if not selected_country:
        return html.Div("Please select an origin country.", 
                       style={'textAlign': 'center', 'padding': '20px'})
    
    try:
        # Fetch maintenance data for selected country
        raw_data = fetch_train_maintenance_data(engine, selected_country)
        
        if raw_data.empty:
            return html.Div(f"No maintenance data available for {selected_country}.", 
                          style={'textAlign': 'center', 'padding': '20px'})
        
        # Process data into hierarchical format with expanded plants
        processed_data, comments_data = process_maintenance_periods_hierarchical(raw_data, expanded_plants)
        
        if processed_data.empty:
            return html.Div("No maintenance data to display.", 
                          style={'textAlign': 'center', 'padding': '20px'})
        
        # Create and return the maintenance table
        return create_maintenance_summary_table(processed_data, comments_data)
        
    except Exception as e:
        import traceback
        return html.Div(f"Error loading maintenance data: {str(e)}", 
                       style={'textAlign': 'center', 'padding': '20px', 'color': 'red'})


# Callback for handling plant expansion/collapse in maintenance table
@callback(
    Output('maintenance-expanded-plants', 'data', allow_duplicate=True),
    Output('maintenance-summary-container', 'children', allow_duplicate=True),
    [Input({'type': 'maintenance-expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'maintenance-expandable-table', 'index': ALL}, 'data'),
     State('maintenance-expanded-plants', 'data'),
     State('origin-country-dropdown', 'value')],
    prevent_initial_call=True
)
def toggle_maintenance_plant_expansion(active_cells, table_data_list, expanded_plants, selected_country):
    """Handle expanding/collapsing of plant rows in maintenance summary table"""
    
    if not any(active_cells):
        raise PreventUpdate
    
    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    
    # Parse which table and what was clicked
    if 'maintenance-expandable-table' in prop_id and '.active_cell' in prop_id:
        try:
            # Get the active cell
            active_cell = active_cells[0]
            if not active_cell:
                raise PreventUpdate
            
            # Get the data from the table
            table_data = table_data_list[0]
            if not table_data:
                raise PreventUpdate
            
            row_index = active_cell['row']
            col_id = active_cell['column_id']
            
            # Only respond to clicks on the Plant column
            if col_id != 'Plant':
                raise PreventUpdate
            
            # Get the clicked row
            clicked_row = table_data[row_index]
            plant_value = clicked_row.get('Plant', '')
            
            # Skip if it's GRAND TOTAL row or empty
            if plant_value == 'GRAND TOTAL' or not plant_value or plant_value.strip() == '':
                raise PreventUpdate
            
            # Check if this is a plant total row (has +/− indicator)
            if plant_value.startswith('+ ') or plant_value.startswith('− '):
                # Extract plant name (remove indicator and spaces)
                plant_name = plant_value[2:].strip()
                
                # Initialize expanded list if None
                expanded_plants = expanded_plants or []
                
                # Toggle expansion state
                if plant_name in expanded_plants:
                    expanded_plants.remove(plant_name)
                else:
                    expanded_plants.append(plant_name)
                
                # Re-fetch and regenerate the table with new expanded state
                try:
                    raw_data = fetch_train_maintenance_data(engine, selected_country)
                    if not raw_data.empty:
                        processed_data, comments_data = process_maintenance_periods_hierarchical(raw_data, expanded_plants)
                        if not processed_data.empty:
                            updated_table = create_maintenance_summary_table(processed_data, comments_data)
                            return expanded_plants, updated_table
                except Exception as e:
                    pass

                return expanded_plants, no_update

        except Exception as e:
            pass


    raise PreventUpdate
 
