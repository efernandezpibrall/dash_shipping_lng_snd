from dash import html, dcc, dash_table, callback, Output, Input, State, Dash, ALL, ctx
from dash.dash_table.Format import Format, Group, Scheme
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime as dt
import json
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
        print(f"Loading data from '{trades_table_name}'...")
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
            print(f"Loaded {len(trades_df)} trades.")
        except Exception as e:
            print(f"ERROR: Failed to load trades data from {trades_table_name}: {e}")
            return None  # Cannot proceed without trades data

        # 3. Load Initial Distance Data
        print(f"Loading data from '{distance_table_name}'...")
        try:
            distance_df = pd.read_sql(f'''SELECT "originLocationName", "destinationLocationName", "distanceDirect", "distanceViaSuez", "distanceViaPanama"
                                          FROM {DB_SCHEMA}.{distance_table_name}''', engine)
            # Standardize column names from DB
            distance_df.columns = ['originLocationName', 'destinationLocationName', 'distanceDirect', 'distanceViaSuez',
                                   'distanceViaPanama']
            print(f"Loaded {len(distance_df)} existing distance pairs.")
        except Exception as e:
            print(f"Warning: Could not load data from {distance_table_name}. Assuming empty or error. Error: {e}")
            # Create an empty DataFrame with expected columns if table doesn't exist or is empty
            distance_df = pd.DataFrame(
                columns=['originLocationName', 'destinationLocationName', 'distanceDirect', 'distanceViaSuez',
                         'distanceViaPanama'])
            print(f"Proceeding with an empty distance matrix.")

        # --- Sections for identifying missing origins, fetching, and reloading are removed ---

        # 4. Join DataFrames
        print("Joining trades data with distance data...")
        # Perform the merge using specified columns
        final_df = pd.merge(
            trades_df,
            distance_df,
            how='left',
            left_on=['zone_origin_name', 'zone_destination_name'],
            right_on=['originLocationName', 'destinationLocationName']
        )
        print(f"Join completed. Resulting table has {len(final_df)} rows.")

        # Extract date components
        final_df['year'] = final_df['end'].dt.year
        final_df['month'] = final_df['end'].dt.month

        # Determine season and quarter based on end month
        final_df['season'] = np.where(final_df['month'].isin([10, 11, 12, 1, 2, 3]), 'W', 'S')
        final_df['quarter'] = 'Q' + final_df['end'].dt.quarter.astype(str)


        # 5. Calculate Ratios
        print("Calculating mileage ratios...")

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
        print("Calculating differences from ideal ratio (1)...")
        final_df['diff_direct'] = (final_df['ratio_miles_distancedirect'] - 1).abs()
        final_df['diff_suez'] = (final_df['ratio_miles_distanceviasuez'] - 1).abs()
        final_df['diff_panama'] = (final_df['ratio_miles_distanceviapanama'] - 1).abs()

        # 7. Identify Minimum Difference and Determine Selected Route
        print("Selecting route based on closest ratio to 1...")
        diff_cols = ['diff_direct', 'diff_suez', 'diff_panama']
        # idxmin skips NaN values, finding the column name with the minimum difference
        final_df['closest_route_col'] = final_df[diff_cols].idxmin(axis=1, skipna=True)

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
        print(
            f"\nNumber of rows with missing distanceDirect after join: {missing_distance_count} out of {len(final_df)}")

        print(f"\nNumber of rows where no ratio was close to 1: {final_df['no_ratio_close_to_1'].sum()}")


    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        # Optionally re-raise the exception if the caller should handle it
        # raise e
        return None  # Return None to indicate failure

    finally:
        # Clean up database connection if engine was created
        if 'engine' in locals() and engine:
            engine.dispose()
            print("Database connection closed.")

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
        SELECT DISTINCT country, shipping_region
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
    # Merge destination regions
    df_trades = pd.merge(
        df_trades,
        df_mapping_country.rename(
            columns={'country': 'destination_country_name', 'shipping_region': 'destination_shipping_region'}),
        how='left',
        on='destination_country_name'
    )

    # --- Final Aggregation ---
    # Determine grouping columns based on destination_level parameter
    group_columns = [
        'vessel_type', 'status',
        'year', 'season', 'quarter',
        'origin_country_name'
    ]

    # Add appropriate destination column based on parameter
    if destination_level == 'destination_country_name':
        group_columns.append('destination_country_name')
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
        print("prepare_pivot_table: Input DataFrame is empty.")
        return empty_df

    # --- Determine Index Columns based on Aggregation Level ---
    if aggregation_level == 'Year':
        index_cols = ['year']
    elif aggregation_level == 'Year+Season':
        index_cols = ['year', 'season']
    elif aggregation_level == 'Year+Quarter':
        index_cols = ['year', 'quarter']
    else:
        print(f"prepare_pivot_table: Invalid aggregation_level: {aggregation_level}. Defaulting to 'Year'.")
        index_cols = ['year']

    # --- Essential Column Checks ---
    required_input_cols = [destination_level, values_col] + list(filters.keys()) + index_cols
    missing_input = [col for col in required_input_cols if col not in df.columns]
    if missing_input:
        print(f"prepare_pivot_table: Missing required columns in input df: {missing_input}")
        return empty_df

    filtered_df = df.copy()

    # --- Apply Filters (Status, Vessel Type) ---
    print(f"prepare_pivot_table: Applying filters: {filters}")
    for col, value in filters.items():
        if value is not None:
            if col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col] == value]
            else:
                print(f"prepare_pivot_table: Warning - Filter column '{col}' not found.")
                return empty_df

    print(f"prepare_pivot_table: Shape after filtering: {filtered_df.shape}")
    if filtered_df.empty:
        print(f"prepare_pivot_table: DataFrame empty after applying filters.")
        return empty_df

    # --- Group before Pivoting (Ensure unique index/column combinations) ---
    # This step aggregates metrics if there are multiple rows for the same
    # filter criteria and desired pivot index/columns (e.g., if original data wasn't fully unique)
    grouping_cols = index_cols + [destination_level]  # Use the specified destination level column
    if not all(col in filtered_df.columns for col in grouping_cols):
        print(f"prepare_pivot_table: Missing grouping columns for aggregation before pivot: {grouping_cols}")
        return empty_df

    # Perform the aggregation using the specified aggfunc for the values_col
    # Keep only necessary columns for pivoting
    agg_spec = {values_col: aggfunc}
    try:
        grouped_df = filtered_df.groupby(grouping_cols, observed=False, dropna=False).agg(agg_spec).reset_index()
    except Exception as e:
        print(f"prepare_pivot_table: Error during pre-pivot grouping: {e}")
        print(traceback.format_exc())
        return empty_df

    print(f"prepare_pivot_table: Shape after grouping: {grouped_df.shape}")
    if grouped_df.empty:
        print(f"prepare_pivot_table: DataFrame empty after grouping.")
        return empty_df

    # --- Pivot Data ---
    try:
        print(
            f"prepare_pivot_table: Pivoting on index={index_cols}, columns='{destination_level}', values='{values_col}'")
        pivot_df = grouped_df.pivot_table(
            index=index_cols,
            columns=destination_level,  # Use the specified destination level
            values=values_col,
            aggfunc='first',  # Use 'first' as data is already aggregated by the groupby above
            fill_value=np.nan  # Fill missing region/time combinations with NaN
        )
        print(f"prepare_pivot_table: Pivot successful. Shape: {pivot_df.shape}")

    except Exception as e:
        print(f"prepare_pivot_table: Error during pivot: {e}")
        print(traceback.format_exc())
        return empty_df

    # --- Sort Index (Year, Season/Quarter) ---
    if not pivot_df.empty:
        pivot_df = pivot_df.sort_index()
        print(f"prepare_pivot_table: Index sorted.")

    # --- Add Total Column (summing across regions for each time period) ---
    if add_total_column and not pivot_df.empty:
        region_cols = pivot_df.columns.tolist()
        if region_cols:  # Ensure there are region columns to sum
            try:
                pivot_df['Total'] = pivot_df[region_cols].sum(axis=1, skipna=True)
                print(f"prepare_pivot_table: Added 'Total' column.")
            except Exception as e:
                print(f"prepare_pivot_table: Error calculating 'Total' column: {e}")
                # Proceed without total column if calculation fails

    # --- Reset Index to make Year/Season/Quarter regular columns ---
    if not pivot_df.empty:
        pivot_df = pivot_df.reset_index()
        print(f"prepare_pivot_table: Index reset. Final columns: {pivot_df.columns.tolist()}")

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
        print(f"Warning (create_stacked_bar_chart): 'status' column missing, cannot filter.")
    # Apply vessel type filter
    if selected_vessel_type and selected_vessel_type != 'All' and 'vessel_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['vessel_type'] == selected_vessel_type]
    elif 'vessel_type' not in filtered_df.columns:
        print(f"Warning (create_stacked_bar_chart): 'vessel_type' column missing, cannot filter.")

    if filtered_df.empty:
        # Return an empty figure with a message if no data after filtering
        fig = go.Figure()
        fig.update_layout(
            title=f"No data available for {title_suffix} with selected filters",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig

    # Use professional color palette
    n_groups = len(unique_groups)
    distinct_colors = get_professional_colors(n_groups)

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
        print(f"Warning: Invalid aggregation_level '{aggregation_level}'. Defaulting to 'Year'.")
        groupby_time_cols = ['year']
        x_axis_title = 'Year'
    # Set grouping field based on data type
    if is_intracountry:
        # Ensure the necessary column exists
        if 'origin_country_name' not in filtered_df.columns:
            print("Error (create_stacked_bar_chart): 'origin_country_name' missing for intracountry chart.")
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
            print(f"Error (create_stacked_bar_chart): Missing {destination_level} column for region/country chart.")
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: Missing {destination_level} column",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            return fig

        group_field = destination_level
        # Determine label for the destination level in chart title
        destination_label = "Countries" if destination_level == "destination_country_name" else "Shipping Regions"
        # Update chart title to include vessel type and aggregation level
        vessel_type_text = f", {selected_vessel_type}" if selected_vessel_type and selected_vessel_type != 'All' else ""
        chart_title = f'{title_suffix} by {x_axis_title}{vessel_type_text} and Destination {destination_label}'
        legend_title = 'Destination ' + (
            'Country' if destination_level == "destination_country_name" else 'Shipping Region')

    # Check required columns
    all_groupby_cols = groupby_time_cols + [group_field]
    missing_cols = [col for col in all_groupby_cols if col not in filtered_df.columns]
    if missing_cols:
        print(f"Error (create_stacked_bar_chart): Missing required columns: {missing_cols}")
        fig = go.Figure()
        fig.update_layout(
            title=f"Error: Missing columns: {', '.join(missing_cols)}",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    # Ensure metric column exists
    if metric not in filtered_df.columns:
        print(f"Error (create_stacked_bar_chart): Metric '{metric}' not found.")
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
        print(f"Error during aggregation: {e}")
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
    # Apply sorting if we have any sort columns
    if sort_columns:
        data_display = data_display.sort_values(by=sort_columns, ascending=sort_ascending)
        # If we added quarter_num, drop it after sorting
        if 'quarter_num' in data_display.columns:
            data_display = data_display.drop(columns=['quarter_num'])

    # --- Data Transformation for Ton Miles (Millions) ---
    is_ton_miles = (metric_for_format == 'sum_ton_miles')

    if is_ton_miles:
        print(f"create_datatable: Formatting '{metric_for_format}' in Millions.")
        cols_to_divide = [col for col in data_cols if pd.api.types.is_numeric_dtype(data_display[col])]
        if cols_to_divide:
            data_display[cols_to_divide] = data_display[cols_to_divide] / 1_000_000
            print(f"create_datatable: Divided data columns {cols_to_divide} by 1M for display.")
        else:
            print(f"create_datatable: No numeric data columns found to divide for '{metric_for_format}'.")

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
        print(f"Invalid aggregation_level: {aggregation_level}. Defaulting to 'Year'.")
        index_cols = ['year']

    # Check for required columns
    required_cols = index_cols + [destination_level, 'voyage_id']
    if include_route_column:
        required_cols.append('selected_route')

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns for {route_scenario_title} table: {missing_cols}")
        return dash_table.DataTable(
            columns=[{'name': 'Error', 'id': 'error_col'}],
            data=[{'error_col': f'Missing columns: {", ".join(missing_cols)}'}],
            style_cell={'textAlign': 'center'}
        )

    try:
        # Ensure selected_route exists before we try to group by it
        if include_route_column and 'selected_route' not in df.columns:
            df['selected_route'] = 'Unknown'
            print(f"Added missing 'selected_route' column with default value 'Unknown'")

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
        print(f"Error creating route analysis table for {route_scenario_title}: {e}")
        print(traceback.format_exc())
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

    # Professional Section Header - Exporter Analysis Configuration
    html.Div([
        html.Div([
            html.Label("Origin Country:", className='filter-label', style={'margin-right': '2px', 'align-self': 'center', 'padding-right': '0px'}),
            dcc.Dropdown(
                id='origin-country-dropdown',
                options=[],  # Will be populated by callback
                value='United States',  # Default to United States
                multi=False,
                clearable=False,
                className='filter-dropdown',
                style={'min-width': '200px'}
            ),

            html.Label("Destination Level:", className='filter-label', style={'margin-right': '2px', 'margin-left': '12px', 'align-self': 'center', 'padding-right': '0px'}),
            dcc.Dropdown(
                id='destination-level-dropdown',
                options=[
                    {'label': 'Shipping Region', 'value': 'destination_shipping_region'},
                    {'label': 'Country', 'value': 'destination_country_name'}
                ],
                value='destination_shipping_region',  # Default to shipping region
                multi=False,
                clearable=False,
                className='filter-dropdown',
                style={'min-width': '200px'}
            ),

            html.Label("Aggregation:", className='filter-label', style={'margin-right': '2px', 'margin-left': '12px', 'align-self': 'center', 'padding-right': '0px'}),
            dcc.Dropdown(
                id='aggregation-dropdown',
                options=[
                    {'label': 'Year', 'value': 'Year'},
                    {'label': 'Year + Season', 'value': 'Year+Season'},
                    {'label': 'Year + Quarter', 'value': 'Year+Quarter'},
                    {'label': 'Month', 'value': 'Month'},
                    {'label': 'Week', 'value': 'Week'},
                ],
                value='Year+Quarter',  # Default aggregation
                multi=False,
                clearable=False,
                className='filter-dropdown',
                style={'min-width': '144px'}
            ),

            html.Label("Status:", className='filter-label', style={'margin-right': '2px', 'margin-left': '12px', 'align-self': 'center', 'padding-right': '0px'}),
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
                style={'width': '100px'}
            ),

            html.Label("Vessel Size:", className='filter-label', style={'margin-right': '2px', 'margin-left': '12px', 'align-self': 'center', 'padding-right': '0px'}),
            dcc.Dropdown(
                id='us-vessel-type-dropdown',
                options=[],  # Options populated by callback
                value='All',  # Default value
                clearable=False,
                className='filter-dropdown',
                style={'min-width': '200px'}
            ),

            html.Label("Metric:", className='filter-label', style={'margin-right': '2px', 'margin-left': '12px', 'align-self': 'center', 'padding-right': '0px'}),
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
                value='mcm_d',  # Default to mcm/d
                clearable=False,
                className='filter-dropdown',
                style={'min-width': '250px'}
            )
        ], className='filter-bar')
    ], className='professional-section-header'),

    # Country Supply Charts Section - Three charts side by side
    html.Div([
        # Section Header
        html.Div([
            html.H3('LNG Supply Analysis - 30-Day Rolling Average', className="section-title-inline"),
        ], className="inline-section-header"),
        
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

    # Destination Analysis Summary Section
    html.Div([
        # Header
        html.Div([
            html.H3('Destination Analysis Summary (mcm/d)', className="section-title-inline"),
        ], className="inline-section-header"),
        
        # Summary Table
        html.Div([
            dcc.Loading(
                id="destination-summary-loading",
                children=[
                    html.Div(id='destination-summary-table-container')
                ],
                type="default"
            )
        ], style={'marginTop': '20px'})
    ], className='section-container', style={'margin-bottom': '32px'}),

    # Trade Analysis by Destination Section (Chart and Table combined)
    html.Div([
        # Enterprise Standard Inline Section Header
        html.Div([
            html.H3(id='trade-analysis-header', className='section-title-inline'),
        ], className='inline-section-header'),
        
        # Chart
        html.Div([
            dcc.Graph(id='us-trade-count-visualization', style={'height': '600px'})
        ]),
        
        # Data Table (no separator)
        html.Div([
            html.Div(id='us-trade-count-table-container', style={'overflow-x': 'auto', 'margin-top': '20px'})
        ])
    ], className='section-container', style={'margin-bottom': '32px'}),

    # Route Analysis Section
    html.Div([
        html.H2("Route Analysis", className='mckinsey-header'),
        html.P("Comparison of shipping routes by canal availability and geographic constraints", className='section-subtitle'),
        
        # Route Visualization Chart
        html.Div([
            dcc.Graph(id='graph-route-suez-only', style={'height': '600px'})
        ], style={'margin-bottom': '24px'}),
        
        # Route Analysis Tables - First Row
        html.Div([
            html.Div([
                html.H4("Suez Available (Not Panama)", className='professional-header'),
                html.Div(id='table-route-suez-only')
            ], className='section-container', style={'width': '33%', 'margin-right': '8px'}),
            
            html.Div([
                html.H4("Panama Available (Not Suez)", className='professional-header'),
                html.Div(id='table-route-panama-only')
            ], className='section-container', style={'width': '33%', 'margin': '0 4px'}),
            
            html.Div([
                html.H4("Both Suez & Panama Available", className='professional-header'),
                html.Div(id='table-route-both')
            ], className='section-container', style={'width': '33%', 'margin-left': '8px'})
        ], style={'display': 'flex', 'margin-bottom': '24px'}),
        
        # Direct Route Table - Second Row
        html.Div([
            html.H4("Direct Available Only (No Suez, No Panama)", className='professional-header'),
            html.Div(id='table-route-direct-only')
        ], className='section-container')
    ], className='section-container'),

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
    ], className='inline-section-header'),
    # Diversions Analysis Charts and Tables
    html.Div([
        html.H2("Diversions Analysis Visualization", className='mckinsey-header'),
        
        # Diversion Chart
        html.Div([
            dcc.Graph(id='diversion-count-chart', style={'height': '600px'})
        ], style={'margin-bottom': '24px'}),
        
        # Three Analysis Tables
        html.Div([
            html.Div([
                html.H4("Count of Trades Data", className='professional-header'),
                dash_table.DataTable(
                    id='count-trades-table',
                    style_table={'overflowX': 'auto', 'height': '400px', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '12px'},
                    style_header={'backgroundColor': '#2E86C1', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '12px'},
                    page_size=20,
                    fill_width=False,
                    sort_action="native",
                    merge_duplicate_headers=True
                )
            ], className='section-container', style={'width': '50%', 'margin-right': '8px'}),
            
            html.Div([
                html.H4("Added Shipping Days Data", className='professional-header'),
                dash_table.DataTable(
                    id='added-days-table',
                    style_table={'overflowX': 'auto', 'height': '400px', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '12px'},
                    style_header={'backgroundColor': '#2E86C1', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '12px'},
                    page_size=20,
                    fill_width=False,
                    sort_action="native",
                    merge_duplicate_headers=True
                )
            ], className='section-container', style={'width': '50%', 'margin': '0 4px'})
        ], style={'display': 'flex', 'margin-bottom': '24px'})
    ], className='section-container'),
    
    # Main Diversion Summary Table
    html.Div([
        html.H2("Diversions Summary Table", className='mckinsey-header'),
        dash_table.DataTable(
            id='diversion-table',
            data=[],
            columns=[],
            style_table={'height': '600px', 'overflowY': 'auto', 'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'minWidth': '100px', 'width': '150px', 'maxWidth': '200px', 
                       'padding': '8px', 'fontSize': '12px'},
            style_header={'backgroundColor': '#2E86C1', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '12px'},
            sort_action="native",
            page_size=20,
            fill_width=False
        )
    ], className='section-container')

])

# ========================================
# DESTINATION SUMMARY TABLE FUNCTIONS
# ========================================

def fetch_destination_summary_data(engine, origin_country, status, vessel_type):
    """
    Fetch destination summary data with expandable continent/country hierarchy
    """
    print(f"DEBUG fetch_destination_summary_data called with:")
    print(f"  - origin_country: {origin_country}")
    print(f"  - status: {status}")
    print(f"  - vessel_type: {vessel_type}")
    
    try:
        # We'll fetch data at country level with continent information
        dest_country_col = "COALESCE(NULLIF(destination_country_name, ''), 'Unknown')"
        dest_continent_col = "COALESCE(NULLIF(continent_destination_name, ''), 'Unknown')"
        print(f"DEBUG: Using country and continent columns")
        
        # Build WHERE clause based on filters
        where_conditions = [f"origin_country_name = '{origin_country}'"]
        
        where_clause = " AND ".join(where_conditions)
        print(f"DEBUG: WHERE clause: {where_clause}")
        
        with engine.connect() as conn:
            # Get quarters, months, weeks data with continent/country hierarchy
            quarters_df = fetch_destination_periods_data_hierarchical(conn, dest_continent_col, dest_country_col, where_clause, 'quarter')
            print(f"DEBUG: quarters_df shape: {quarters_df.shape if not quarters_df.empty else 'EMPTY'}")
            
            months_df = fetch_destination_periods_data_hierarchical(conn, dest_continent_col, dest_country_col, where_clause, 'month')
            print(f"DEBUG: months_df shape: {months_df.shape if not months_df.empty else 'EMPTY'}")
            
            weeks_df = fetch_destination_periods_data_hierarchical(conn, dest_continent_col, dest_country_col, where_clause, 'week')
            print(f"DEBUG: weeks_df shape: {weeks_df.shape if not weeks_df.empty else 'EMPTY'}")
            
            # Get rolling windows data with hierarchy
            rolling_df = fetch_destination_rolling_windows_hierarchical(conn, dest_continent_col, dest_country_col, where_clause)
            print(f"DEBUG: rolling_df shape: {rolling_df.shape if not rolling_df.empty else 'EMPTY'}")
            
            # Combine all data with hierarchy
            result = combine_destination_summary_data_hierarchical(quarters_df, months_df, weeks_df, rolling_df)
            print(f"DEBUG: Combined result shape: {result.shape if not result.empty else 'EMPTY'}")
            if not result.empty:
                print(f"DEBUG: Result columns: {result.columns.tolist()}")
                print(f"DEBUG: First few rows:\n{result.head()}")
            
            return result
            
    except Exception as e:
        print(f"Error fetching destination summary data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def fetch_destination_periods_data(conn, dest_col, where_clause, period_type):
    """Fetch data for specific period type (quarter, month, week)"""
    from sqlalchemy import text
    
    print(f"DEBUG fetch_destination_periods_data: period_type={period_type}, dest_col={dest_col}")
    
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
            print(f"DEBUG: Total rows for {origin_country}: {count}")
        
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
        
        print(f"DEBUG SQL Query for {period_type}:")
        print(query_str[:500])  # Print first 500 chars
        
        query = text(query_str)
        df = pd.read_sql(query, conn)
        print(f"DEBUG: Query returned {len(df)} rows")
        
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
        print(f"Error fetching {period_type} data: {e}")
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
        print(f"Error fetching rolling windows data: {e}")
        return pd.DataFrame()

def fetch_destination_periods_data_hierarchical(conn, continent_col, country_col, where_clause, period_type):
    """Fetch data for specific period type with continent/country hierarchy"""
    from sqlalchemy import text
    
    print(f"DEBUG fetch_destination_periods_data_hierarchical: period_type={period_type}")
    
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
        print(f"Error fetching hierarchical {period_type} data: {e}")
        return pd.DataFrame()

def fetch_destination_rolling_windows_hierarchical(conn, continent_col, country_col, where_clause):
    """Fetch 7D and 30D rolling window data with continent/country hierarchy"""
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
                    {continent_col} as continent,
                    {country_col} as country,
                    SUM(cargo_origin_cubic_meters * 0.6 / 1000) / 7.0 as avg_7d
                FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                WHERE upload_timestamp_utc = max_ts
                    AND {where_clause}
                    AND "start" > '{date_7d_ago}'
                    AND "start" <= '{current_date}'
                GROUP BY {continent_col}, {country_col}
            ),
            window_30d AS (
                SELECT 
                    {continent_col} as continent,
                    {country_col} as country,
                    SUM(cargo_origin_cubic_meters * 0.6 / 1000) / 30.0 as avg_30d
                FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                WHERE upload_timestamp_utc = max_ts
                    AND {where_clause}
                    AND "start" > '{date_30d_ago}'
                    AND "start" <= '{current_date}'
                GROUP BY {continent_col}, {country_col}
            ),
            window_30d_y1 AS (
                SELECT 
                    {continent_col} as continent,
                    {country_col} as country,
                    SUM(cargo_origin_cubic_meters * 0.6 / 1000) / 30.0 as avg_30d_y1
                FROM {DB_SCHEMA}.kpler_trades, latest_timestamp
                WHERE upload_timestamp_utc = max_ts
                    AND {where_clause}
                    AND "start" > '{date_30d_y1_start}'
                    AND "start" <= '{date_30d_y1_end}'
                GROUP BY {continent_col}, {country_col}
            )
            SELECT 
                COALESCE(w7.continent, w30.continent, w30y1.continent) as continent,
                COALESCE(w7.country, w30.country, w30y1.country) as country,
                COALESCE(w7.avg_7d, 0) as "7D",
                COALESCE(w30.avg_30d, 0) as "30D",
                COALESCE(w30y1.avg_30d_y1, 0) as "30D_Y1",
                COALESCE(w7.avg_7d, 0) - COALESCE(w30.avg_30d, 0) as "Δ 7D-30D",
                COALESCE(w30.avg_30d, 0) - COALESCE(w30y1.avg_30d_y1, 0) as "Δ 30D Y/Y"
            FROM window_7d w7
            FULL OUTER JOIN window_30d w30 ON w7.continent = w30.continent AND w7.country = w30.country
            FULL OUTER JOIN window_30d_y1 w30y1 ON COALESCE(w7.continent, w30.continent) = w30y1.continent 
                AND COALESCE(w7.country, w30.country) = w30y1.country
        """)
        
        df = pd.read_sql(query, conn)
        return df
        
    except Exception as e:
        print(f"Error fetching hierarchical rolling windows data: {e}")
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

def combine_destination_summary_data_hierarchical(quarters_df, months_df, weeks_df, rolling_df):
    """Combine all period data with continent/country hierarchy into final summary table"""
    from datetime import datetime
    
    try:
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
        
        # Add 30D column from rolling data
        if not rolling_df.empty and '30D' in rolling_df.columns:
            result = result.merge(rolling_df[['continent', 'country', '30D']], on=['continent', 'country'], how='left')
        
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
            remaining_cols = ['7D', 'Δ 7D-30D', 'Δ 30D Y/Y']
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
        print(f"Error combining hierarchical destination summary data: {e}")
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
        print(f"Error combining destination summary data: {e}")
        return pd.DataFrame()

def create_country_supply_chart(country_name):
    """Create seasonal comparison chart for selected country's LNG supply"""
    
    try:
        # Query to get supply data for the selected country
        query = f"""
        WITH latest_data AS (
            SELECT MAX(upload_timestamp_utc) as max_timestamp
            FROM {DB_SCHEMA}.kpler_trades
        ),
        daily_supply AS (
            SELECT 
                kt.start::date as date,
                kt.origin_country_name,
                SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_supply_mcmd
            FROM {DB_SCHEMA}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name = %(country_name)s
                AND kt.start >= '2023-11-01'
                AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
            GROUP BY kt.start::date, kt.origin_country_name
        ),
        rolling_supply AS (
            SELECT 
                date,
                origin_country_name,
                daily_supply_mcmd,
                AVG(daily_supply_mcmd) OVER (
                    PARTITION BY origin_country_name 
                    ORDER BY date 
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as rolling_avg_30d,
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
            rolling_avg_30d as rolling_avg,
            is_forecast
        FROM rolling_supply
        WHERE date >= '2024-01-01'
        ORDER BY date
        """
        
        df = pd.read_sql(query, engine, params={'country_name': country_name})
        
        if df.empty:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No supply data available for {country_name}",
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
        
        # Get unique years and assign colors - McKinsey blue palette (same as exporters.py)
        years = sorted(df['year'].unique())
        colors = ['#2E86C1', '#1B4F72', '#5DADE2', '#3498DB', '#76D7C4']  # McKinsey blue palette
        
        # Plot each year's data
        for i, year in enumerate(years):
            year_data = df[df['year'] == year].copy()
            
            # For seasonal comparison, use day of year as x-axis
            year_data['plot_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(year_data['day_of_year'] - 1, unit='d')
            
            # Split data into historical and forecast
            historical_data = year_data[~year_data['is_forecast']]
            forecast_data = year_data[year_data['is_forecast']]
            
            # Plot historical data with solid line
            if not historical_data.empty:
                fig.add_trace(go.Scatter(
                    x=historical_data['plot_date'],
                    y=historical_data['rolling_avg'],
                    mode='lines',
                    name=str(int(year)),  # Convert to int to remove decimal
                    line=dict(
                        color=colors[i % len(colors)],
                        width=3 if year == years[-1] else 2,  # Highlight current year
                        dash='solid'
                    ),
                    hovertemplate=f'<b>{int(year)} (Historical)</b><br>' +  # Convert to int here too
                                 '%{text}<br>' +
                                 'Supply: %{y:.1f} mcm/d<br>' +
                                 '<extra></extra>',
                    text=historical_data['month_day'],
                    showlegend=True
                ))
            
            # Plot forecast data with lighter/transparent color
            if not forecast_data.empty:
                # Connect forecast line to historical by including last historical point
                if not historical_data.empty:
                    # Get last historical point to connect the lines
                    connect_data = pd.concat([historical_data.tail(1), forecast_data])
                else:
                    connect_data = forecast_data
                
                # Create a lighter version of the color for forecast
                base_color = colors[i % len(colors)]
                # Convert hex to rgba with transparency
                forecast_color = f"rgba({int(base_color[1:3], 16)}, {int(base_color[3:5], 16)}, {int(base_color[5:7], 16)}, 0.5)"
                    
                fig.add_trace(go.Scatter(
                    x=connect_data['plot_date'],
                    y=connect_data['rolling_avg'],
                    mode='lines',
                    name=f"{int(year)} (Forecast)",
                    line=dict(
                        color=forecast_color,  # Lighter/transparent version
                        width=3 if year == years[-1] else 2,  # Same width as historical
                        dash='solid'  # Solid line instead of dashed
                    ),
                    opacity=0.7,  # Additional opacity for visual distinction
                    hovertemplate=f'<b>{int(year)} (Forecast)</b><br>' +
                                 '%{text}<br>' +
                                 'Supply: %{y:.1f} mcm/d<br>' +
                                 '<extra></extra>',
                    text=connect_data['month_day'],
                    showlegend=False  # Don't show separate legend entry for forecast
                ))
        
        # Update layout with professional styling standards (matching exporters.py)
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
            
            # Legend positioning (matching exporters.py)
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.08,
                xanchor='left',
                x=0,
                bgcolor='rgba(255, 255, 255, 0)',
                bordercolor='rgba(255, 255, 255, 0)',
                borderwidth=0,
                font=dict(size=12, color='#4A4A4A'),
                itemsizing='constant'
            ),
            
            # General Layout
            height=400,
            margin=dict(l=60, r=40, t=40, b=60),
            paper_bgcolor='white',
            plot_bgcolor='white',
            hovermode='x unified',
            
            # Title - removed since we have headers
            title=None
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating supply chart for {country_name}: {e}")
        # Return empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading supply data for {country_name}",
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

def create_continent_destination_chart(country_name):
    """Create seasonal comparison chart by continent destination for selected country's LNG exports"""
    
    try:
        print(f"DEBUG: Creating continent destination chart for: '{country_name}'")
        # Query to get export data by continent destination for the selected country
        query = f"""
        WITH latest_data AS (
            SELECT MAX(upload_timestamp_utc) as max_timestamp
            FROM {DB_SCHEMA}.kpler_trades
        ),
        daily_exports AS (
            SELECT 
                kt.start::date as date,
                kt.origin_country_name,
                COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination,
                SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
            FROM {DB_SCHEMA}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name = %(country_name)s
                AND kt.start >= '2023-11-01'
                AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
            GROUP BY kt.start::date, kt.origin_country_name, kt.continent_destination_name
        ),
        rolling_exports AS (
            SELECT 
                date,
                origin_country_name,
                continent_destination,
                daily_export_mcmd,
                AVG(daily_export_mcmd) OVER (
                    PARTITION BY origin_country_name, continent_destination
                    ORDER BY date 
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as rolling_avg_30d,
                CASE 
                    WHEN date > CURRENT_DATE THEN true
                    ELSE false
                END as is_forecast
            FROM daily_exports
        )
        SELECT 
            date,
            continent_destination,
            EXTRACT(YEAR FROM date) as year,
            EXTRACT(DOY FROM date) as day_of_year,
            TO_CHAR(date, 'Mon DD') as month_day,
            rolling_avg_30d as rolling_avg,
            is_forecast
        FROM rolling_exports
        WHERE date >= '2024-01-01'
        ORDER BY continent_destination, date
        """
        
        df = pd.read_sql(query, engine, params={'country_name': country_name})
        
        print(f"DEBUG: Query returned {len(df)} rows")
        if not df.empty:
            print(f"DEBUG: Continents found: {df['continent_destination'].unique()}")
            print(f"DEBUG: Years found: {sorted(df['year'].unique())}")
            print(f"DEBUG: Date range: {df['date'].min()} to {df['date'].max()}")
        
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
        print(f"ERROR creating continent destination chart for {country_name}: {e}")
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

def create_continent_percentage_chart(country_name):
    """Create percentage distribution chart by continent destination for selected country's LNG exports"""
    
    try:
        print(f"DEBUG: Creating continent percentage chart for: '{country_name}'")
        # Query to get export data by continent destination for percentage calculation
        query = f"""
        WITH latest_data AS (
            SELECT MAX(upload_timestamp_utc) as max_timestamp
            FROM {DB_SCHEMA}.kpler_trades
        ),
        daily_exports AS (
            SELECT 
                kt.start::date as date,
                kt.origin_country_name,
                COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination,
                SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
            FROM {DB_SCHEMA}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name = %(country_name)s
                AND kt.start >= '2023-11-01'
                AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
            GROUP BY kt.start::date, kt.origin_country_name, kt.continent_destination_name
        ),
        daily_totals AS (
            SELECT 
                date,
                SUM(daily_export_mcmd) as total_daily_export
            FROM daily_exports
            GROUP BY date
        ),
        rolling_data AS (
            SELECT 
                de.date,
                de.origin_country_name,
                de.continent_destination,
                de.daily_export_mcmd,
                -- Rolling average for each continent
                AVG(de.daily_export_mcmd) OVER (
                    PARTITION BY de.continent_destination
                    ORDER BY de.date 
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as rolling_avg_30d,
                -- Rolling average for total
                AVG(dt.total_daily_export) OVER (
                    ORDER BY dt.date 
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as total_rolling_avg_30d,
                CASE 
                    WHEN de.date > CURRENT_DATE THEN true
                    ELSE false
                END as is_forecast
            FROM daily_exports de
            JOIN daily_totals dt ON de.date = dt.date
        )
        SELECT 
            date,
            continent_destination,
            EXTRACT(YEAR FROM date) as year,
            EXTRACT(DOY FROM date) as day_of_year,
            TO_CHAR(date, 'Mon DD') as month_day,
            rolling_avg_30d as rolling_avg,
            CASE 
                WHEN total_rolling_avg_30d > 0 
                THEN (rolling_avg_30d / total_rolling_avg_30d) * 100
                ELSE 0
            END as percentage,
            is_forecast
        FROM rolling_data
        WHERE date >= '2024-01-01'
        ORDER BY continent_destination, date
        """
        
        df = pd.read_sql(query, engine, params={'country_name': country_name})
        
        print(f"DEBUG: Percentage query returned {len(df)} rows")
        if not df.empty:
            print(f"DEBUG: Continents found: {df['continent_destination'].unique()}")
            print(f"DEBUG: Percentage range: {df['percentage'].min():.1f}% to {df['percentage'].max():.1f}%")
        
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
        print(f"ERROR creating continent percentage chart for {country_name}: {e}")
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

@callback(
    [Output('country-supply-chart', 'figure'),
     Output('country-supply-header', 'children')],
    Input('origin-country-dropdown', 'value')
)
def update_country_supply_chart(selected_country):
    """Update the supply chart based on selected country"""
    if not selected_country:
        # Return empty chart if no country selected
        fig = go.Figure()
        fig.update_layout(height=400)
        return fig, "Total Supply"
    
    # Create the chart
    fig = create_country_supply_chart(selected_country)
    
    # Update header with country name
    header_text = f"{selected_country} - Total Supply"
    
    return fig, header_text

@callback(
    [Output('continent-destination-chart', 'figure'),
     Output('continent-destination-header', 'children')],
    Input('origin-country-dropdown', 'value')
)
def update_continent_destination_chart(selected_country):
    """Update the continent destination chart based on selected country"""
    if not selected_country:
        # Return empty chart if no country selected
        fig = go.Figure()
        fig.update_layout(height=400)
        return fig, "By Destination Continent"
    
    # Create the chart
    fig = create_continent_destination_chart(selected_country)
    
    # Update header
    header_text = f"{selected_country} - By Destination Continent"
    
    return fig, header_text

@callback(
    [Output('continent-percentage-chart', 'figure'),
     Output('continent-percentage-header', 'children')],
    Input('origin-country-dropdown', 'value')
)
def update_continent_percentage_chart(selected_country):
    """Update the continent percentage chart based on selected country"""
    if not selected_country:
        # Return empty chart if no country selected
        fig = go.Figure()
        fig.update_layout(height=400)
        return fig, "By Destination Continent (%)"
    
    # Create the chart
    fig = create_continent_percentage_chart(selected_country)
    
    # Update header
    header_text = f"{selected_country} - Market Share (%)"
    
    return fig, header_text

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
        print(f"Error initializing country dropdown: {e}")
        # Return a default option if there's an error
        return [{'label': 'United States', 'value': 'United States'}], 'United States'



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
    print(f"Refreshing data for {selected_origin_country} with destination level: {selected_destination_level}...")
    # --- Default values in case of error ---
    default_vessel_options = []
    default_vessel_value = None
    default_country_options = []

    try:
        # Pass the selected destination level to kpler_analysis
        df_trades_shipping_region_all = kpler_analysis(engine, destination_level=selected_destination_level, origin_country=selected_origin_country)
        if df_trades_shipping_region_all is None or df_trades_shipping_region_all.empty:
            raise ValueError("kpler_analysis returned empty or None data.")
        print(f"Loaded all data. Region trades shape: {df_trades_shipping_region_all.shape}")
    except Exception as e:
        print(f"Error loading base data: {e}")
        error_msg = f"Error loading base data: {e}. Ensure 'engine' is configured."
        us_options_data_error = {
            'vessel_type_options': default_vessel_options,
            'default_vessel_type': default_vessel_value
        }
        # Return defaults and error message
        return None, us_options_data_error, error_msg, default_vessel_options, default_vessel_value, default_country_options

    # --- Get unique countries for dropdown ---
    if 'origin_country_name' not in df_trades_shipping_region_all.columns:
        print("Error: 'origin_country_name' column not found in loaded data.")
        error_msg = "Error: Missing 'origin_country_name' column."
        us_options_data_error = {
            'vessel_type_options': default_vessel_options,
            'default_vessel_type': default_vessel_value
        }
        return None, us_options_data_error, error_msg, default_vessel_options, default_vessel_value, default_country_options



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
    print("Data refresh complete.")

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
    Output('us-trade-count-table-container', 'children'),

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
        return empty_fig, html.Div(error_msg)

    if not all([selected_aggregation, selected_status, selected_destination_level]):
        empty_fig.update_layout(title_text="Please select Aggregation, Status, and Destination Level")
        return empty_fig, html.Div("Please select all filter values.")

    try:
        df_trades_shipping_region_us = pd.read_json(us_shipping_data, orient='split')
        if df_trades_shipping_region_us.empty:
            raise ValueError(f"Loaded {origin_country} shipping data is empty.")

        # --- Check if the selected destination level column exists ---
        if selected_destination_level not in df_trades_shipping_region_us.columns:
            error_msg = f"Selected destination level column '{selected_destination_level}' not found in data."
            print(f"Update US Region Viz: {error_msg}")
            empty_fig.update_layout(title_text=error_msg)
            return empty_fig, html.Div(error_msg)

        # --- Filter data for charts ---
        df_for_charts = df_trades_shipping_region_us[df_trades_shipping_region_us['status'] == selected_status].copy()

        # Apply vessel type filter if needed
        if selected_vessel_type and selected_vessel_type != 'All':
            df_for_charts = df_for_charts[df_for_charts['vessel_type'] == selected_vessel_type]

        if df_for_charts.empty:
            empty_fig.update_layout(title_text="No data available after filtering")
            return empty_fig, html.Div(no_data_msg)

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
            
            # Convert to appropriate units for display
            if selected_chart_metric == 'mtpa':
                # Convert cubic meters to MTPA (assuming LNG density of ~0.45 tonnes/m³)
                selected_metric_data[metric_column] = selected_metric_data[metric_column] * 0.45 / 1_000_000 * 365.25
            elif selected_chart_metric == 'mcm_d':
                # Convert cubic meters to million cubic meters per day
                selected_metric_data[metric_column] = selected_metric_data[metric_column] / 1_000_000
            elif selected_chart_metric == 'm3':
                # Keep in cubic meters (no conversion needed)
                pass

            # Create time labels and sorting fields
            if 'year' in selected_metric_data.columns:
                if 'quarter' in selected_metric_data.columns and 'quarter' in groupby_time_cols:
                    # Extract quarter number for sorting
                    selected_metric_data['quarter_num'] = selected_metric_data['quarter'].str.extract(r'(\d+)').astype(int)
                    selected_metric_data['time_label'] = selected_metric_data['year'].astype(str) + '-' + selected_metric_data[
                        'quarter'].astype(str)
                    # Create a sort key (higher values appear first in the chart)
                    selected_metric_data['sort_key'] = selected_metric_data['year'] * 10 + selected_metric_data['quarter_num']
                elif 'season' in selected_metric_data.columns and 'season' in groupby_time_cols:
                    selected_metric_data['time_label'] = selected_metric_data['year'].astype(str) + '-' + selected_metric_data[
                        'season'].astype(str)
                    # Create a sort key with a value of 1 for Summer (S) and 2 for Winter (W)
                    selected_metric_data['season_num'] = selected_metric_data['season'].apply(lambda x: 1 if x == 'S' else 2)
                    selected_metric_data['sort_key'] = selected_metric_data['year'] * 10 + selected_metric_data['season_num']
                elif 'month' in selected_metric_data.columns and 'month' in groupby_time_cols:
                    # Handle month aggregation
                    selected_metric_data['time_label'] = selected_metric_data['year'].astype(str) + '-' + selected_metric_data[
                        'month'].astype(str).str.zfill(2)
                    selected_metric_data['sort_key'] = selected_metric_data['year'] * 100 + selected_metric_data['month']
                elif 'week' in selected_metric_data.columns and 'week' in groupby_time_cols:
                    # Handle week aggregation  
                    selected_metric_data['time_label'] = selected_metric_data['year'].astype(str) + '-W' + selected_metric_data[
                        'week'].astype(str).str.zfill(2)
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
            print(f"Error creating trade count chart: {e}")
            print(traceback.format_exc())



        # Apply professional styling
        destination_display = "Countries" if selected_destination_level == "destination_country_name" else "Shipping Regions"
        
        fig.update_layout(
            title=dict(
                text=f'{metric_title} ({origin_country} Origin, Status: {selected_status})',
                font=dict(
                    family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                    size=18,
                    color=PROFESSIONAL_COLORS['text_primary']
                ),
                x=0.5,
                xanchor='center'
            ),
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
                title=dict(
                    text=f"Destination {destination_display}",
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
                x=1.02,
                y=0.5,
                xanchor="left",
                yanchor="middle",
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=PROFESSIONAL_COLORS['grid_color'],
                borderwidth=1
            ),
            margin=dict(l=60, r=200, t=80, b=60),
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
        count_table_data = prepare_pivot_table(
            df=df_trades_shipping_region_us,
            values_col=selected_metric_column,
            filters=table_filters,
            aggregation_level=selected_aggregation,
            add_total_column=True,
            aggfunc=agg_func,
            destination_level=selected_destination_level
        )


        # Apply unit conversions to table data (same as chart)
        if not count_table_data.empty and selected_chart_metric in ['mtpa', 'mcm_d']:
            # Create a copy to avoid modifying the original data
            count_table_data_converted = count_table_data.copy()
            
            # Apply conversions to numeric columns (skip time-related columns)
            numeric_cols = count_table_data_converted.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if selected_chart_metric == 'mtpa':
                    # Convert cubic meters to MTPA 
                    count_table_data_converted[col] = count_table_data_converted[col] * 0.45 / 1_000_000 * 365.25
                elif selected_chart_metric == 'mcm_d':
                    # Convert cubic meters to million cubic meters per day
                    count_table_data_converted[col] = count_table_data_converted[col] / 1_000_000
        else:
            count_table_data_converted = count_table_data.copy()
        
        # --- Create DataTables ---
        if not count_table_data_converted.empty:
            table1_content = create_datatable(
                count_table_data_converted,
                metric_for_format=selected_chart_metric,
                aggregation_level=selected_aggregation
            )
        else:
            table1_content = html.Div(f"{no_data_msg}")

        return fig, table1_content

    except Exception as e:
        error_message = f"Error updating US region visuals/tables: {e}"
        empty_fig.update_layout(title_text=error_message)
        return empty_fig, html.Div(error_message)



@callback(
    Output('destination-summary-table-container', 'children'),
    [Input('origin-country-dropdown', 'value'),
     Input('us-region-status-dropdown', 'value'),
     Input('us-vessel-type-dropdown', 'value'),
     Input('destination-expanded-continents', 'data')],
    prevent_initial_call=False
)
def update_destination_summary_table(origin_country, status, vessel_type, expanded_continents):
    """Update the destination summary table with expandable continent/country hierarchy"""
    
    print(f"DEBUG: update_destination_summary_table called with:")
    print(f"  - origin_country: {origin_country}")
    print(f"  - status: {status}")
    print(f"  - vessel_type: {vessel_type}")
    print(f"  - expanded_continents: {expanded_continents}")
    
    if not origin_country:
        print("DEBUG: No origin country selected")
        return html.Div("Please select an origin country.", 
                       style={'textAlign': 'center', 'padding': '20px'})
    
    try:
        # Initialize expanded continents if None
        expanded_continents = expanded_continents or []
        
        # Fetch the data
        print("DEBUG: Calling fetch_destination_summary_data...")
        df = fetch_destination_summary_data(engine, origin_country, status, vessel_type)
        
        print(f"DEBUG: Returned df shape: {df.shape if not df.empty else 'EMPTY'}")
        if not df.empty:
            print(f"DEBUG: df columns: {df.columns.tolist()}")
            print(f"DEBUG: df head:\n{df.head()}")
        
        if df.empty:
            print("DEBUG: DataFrame is empty, returning no data message")
            return html.Div("No data available for the selected filters.", 
                           style={'textAlign': 'center', 'padding': '20px'})
        
        # Prepare data for display with expandable rows
        display_df = prepare_destination_table_for_display(df, expanded_continents)
        
        # Create column definitions
        columns = []
        for col in display_df.columns:
            if col in ['Continent', 'Country']:
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
        
        # Color coding for different time periods and add white separators
        for col in display_df.columns:
            if col.startswith('Q') and "'" in col:  # Quarter columns
                conditional_styles.append({
                    'if': {'column_id': col, 'row_index': 0},
                    'backgroundColor': '#e3f2fd'  # Light blue
                })
                # Add left border to first quarter column for visual separation
                if quarter_columns and col == quarter_columns[0]:
                    conditional_styles.append({
                        'if': {'column_id': col},
                        'borderLeft': '3px solid white'
                    })
            elif col.startswith('W') and "'" in col:  # Week columns
                conditional_styles.append({
                    'if': {'column_id': col, 'row_index': 0},
                    'backgroundColor': '#e8f5e9'  # Light green
                })
                # Add left border to first week column for visual separation from 30D
                if week_columns and col == week_columns[0]:
                    conditional_styles.append({
                        'if': {'column_id': col},
                        'borderLeft': '3px solid white'
                    })
            elif "'" in col and not col.startswith('Q') and not col.startswith('W'):  # Month columns
                conditional_styles.append({
                    'if': {'column_id': col, 'row_index': 0},
                    'backgroundColor': '#f3e5f5'  # Light purple
                })
                # Add left border to first month column for visual separation
                if month_columns and col == month_columns[0]:
                    conditional_styles.append({
                        'if': {'column_id': col},
                        'borderLeft': '3px solid white'
                    })
            elif col in ['30D', '7D']:  # Rolling windows
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#fff3e0',  # Light orange
                    'fontWeight': '500'
                })
                # No separator between months and 30D - removed the borderLeft for 30D
            elif col == 'Δ 7D-30D':  # Delta column
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
            elif col == 'Δ 30D Y/Y':  # Year-over-year delta
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
                        'filter_query': '{Δ 30D Y/Y} > 0'
                    },
                    'color': '#1b5e20'
                })
                # Dark red for negative
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': '{Δ 30D Y/Y} < 0'
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
        
        # No separator before 30D column - removed
        
        # Add separator before first week column
        if week_columns:
            header_styles.append({
                'if': {'column_id': week_columns[0]},
                'borderLeft': '3px solid white'
            })
        
        # Add separator before first delta column
        if 'Δ 7D-30D' in df.columns:
            header_styles.append({
                'if': {'column_id': 'Δ 7D-30D'},
                'borderLeft': '3px solid white'
            })
        
        # Add separator before second delta column
        if 'Δ 30D Y/Y' in df.columns:
            header_styles.append({
                'if': {'column_id': 'Δ 30D Y/Y'},
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
        print(f"Error updating destination summary table: {e}")
        import traceback
        traceback.print_exc()
        return html.Div(f"Error loading data: {str(e)}", 
                       style={'textAlign': 'center', 'padding': '20px', 'color': 'red'})


@callback(
    [Output('graph-route-suez-only', 'figure'),
     Output('table-route-suez-only', 'children'),
     Output('table-route-panama-only', 'children'),
     Output('table-route-both', 'children'),
     Output('table-route-direct-only', 'children')],
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
                    print(f"Added {destination_level} column through mapping")
                except Exception as mapping_err:
                    print(f"Error adding destination country mapping: {mapping_err}")
                    # Continue with destination_shipping_region as fallback
                    destination_level = 'destination_shipping_region'
                    print(f"Falling back to destination_shipping_region")
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
                    print(f"Added {destination_level} column through mapping")
                except Exception as mapping_err:
                    print(f"Error adding destination region mapping: {mapping_err}")
                    # Continue with destination_country_name as fallback
                    destination_level = 'destination_country_name'
                    print(f"Falling back to destination_country_name")
    except Exception as e:
        print(f"Error calling process_trade_and_distance_data: {e}")
        # Create error figure and table
        error_fig = go.Figure().update_layout(
            title="Error Processing Data",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        error_table = html.Div(f"Error loading data: {str(e)}")
        # Return errors for all outputs
        return error_fig, error_table, error_table, error_table, error_table

    # Check again if the destination level column exists after attempted mapping
    if destination_level not in processed_df.columns:
        error_msg = f"Selected destination level column '{destination_level}' not found in processed data."
        print(error_msg)
        error_fig = go.Figure().update_layout(
            title=error_msg,
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        error_table = html.Div(error_msg)
        return error_fig, error_table, error_table, error_table, error_table

    # --- Data Filtering for the four scenarios ---
    try:
        # Ensure the destination column is available for all dataframes
        required_cols = ['destination_country_name', 'destination_shipping_region']
        for col in required_cols:
            if col not in processed_df.columns:
                # Create the column if it doesn't exist
                processed_df[col] = None
                print(f"Created empty column {col} in processed_df")

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
        print(f"Error filtering data: Missing column {e}")
        error_fig = go.Figure().update_layout(
            title=f"Error: Missing Column {e}",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        error_table = html.Div(f"Error: Missing Column {e}")
        return error_fig, error_table, error_table, error_table, error_table
    except Exception as e:
        print(f"Unexpected error during filtering: {e}")
        error_fig = go.Figure().update_layout(
            title="Error Filtering Data",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        error_table = html.Div(f"Error filtering data: {str(e)}")
        return error_fig, error_table, error_table, error_table, error_table

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
            else:
                print(f"Invalid aggregation_level: {agg_level}. Cannot proceed.")
                return False

            # Check for required columns
            required_cols = index_cols + ['selected_route', 'voyage_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Missing required columns in {df_name}: {missing_cols}")
                return False

            # Group by aggregation level and route to get counts
            grouped_cols = index_cols + ['selected_route']
            try:
                grouped = df.groupby(grouped_cols, observed=True)['voyage_id'].count().unstack(fill_value=0)
            except Exception as e:
                print(f"Error during grouping in {df_name}: {e}")
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
        print(f"Error creating unified route analysis chart: {e}")
        print(traceback.format_exc())
        fig = go.Figure().update_layout(title=f"Error generating charts: {str(e)}")

    # --- Ensure both destination columns exist in all dataframes ---
    for df in [df_suez_only, df_panama_only, df_both, df_direct_only]:
        if df is not None and not df.empty:
            # Make sure required destination columns exist
            for col in ['destination_country_name', 'destination_shipping_region']:
                if col not in df.columns:
                    df[col] = None
                    print(f"Added missing column {col} to dataframe")

    # --- Generate the tables ---
    try:
        table_suez_only = create_route_analysis_table(
            df_suez_only,
            agg_level,
            "Route Usage: Suez Available (Not Panama)",
            include_route_column=True,
            destination_level=destination_level
        )
    except Exception as e:
        print(f"Error creating Suez-only table: {e}")
        print(traceback.format_exc())
        table_suez_only = html.Div(f"Error creating table: {str(e)}")

    try:
        table_panama_only = create_route_analysis_table(
            df_panama_only,
            agg_level,
            "Route Usage: Panama Available (Not Suez)",
            include_route_column=True,
            destination_level=destination_level
        )
    except Exception as e:
        print(f"Error creating Panama-only table: {e}")
        print(traceback.format_exc())
        table_panama_only = html.Div(f"Error creating table: {str(e)}")

    try:
        table_both = create_route_analysis_table(
            df_both,
            agg_level,
            "Route Usage: Both Suez & Panama Available",
            include_route_column=True,
            destination_level=destination_level
        )
    except Exception as e:
        print(f"Error creating Both-routes table: {e}")
        print(traceback.format_exc())
        table_both = html.Div(f"Error creating table: {str(e)}")

    try:
        table_direct_only = create_route_analysis_table(
            df_direct_only,
            agg_level,
            "Route Usage: Direct Only",
            include_route_column=False,
            destination_level=destination_level
        )
    except Exception as e:
        print(f"Error creating Direct-only table: {e}")
        print(traceback.format_exc())
        table_direct_only = html.Div(f"Error creating Direct-only table: {str(e)}")

    # --- Return the figure and tables ---
    return fig, table_suez_only, table_panama_only, table_both, table_direct_only


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
    # Outputs for the main diversion table
    Output('diversion-table', 'data'),
    Output('diversion-table', 'columns'),
    # Single chart output (not three)
    Output('diversion-count-chart', 'figure'),
    # Rest of the outputs
    Output('count-trades-table', 'data'),
    Output('count-trades-table', 'columns'),
    Output('added-days-table', 'data'),
    Output('added-days-table', 'columns'),
    # Inputs
    Input('diversion-processed-data', 'data'),
    Input('destination-level-radio', 'value'),
)
def update_diversion_ui(stored_data, destination_level):
    if not stored_data:
        # Return empty/placeholder values if no data is available yet
        empty_fig = go.Figure().update_layout(title="No data available")
        empty_columns = [{"name": "No Data", "id": "no_data"}]
        empty_data = [{"no_data": "No data available"}]
        return ([], empty_columns, empty_fig,  # One figure instead of three
                empty_data, empty_columns,
                empty_data, empty_columns,
                empty_data, empty_columns)

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
        shared_xaxes=True,  # Share x-axes for better alignment
        horizontal_spacing=0.02  # Reduce spacing between subplots
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
        title=dict(
            text=f'{origin_country} Diversions Analysis by {combo_field.replace("_", " ").title()}',
            font=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                size=18,
                color=PROFESSIONAL_COLORS['text_primary']
            ),
            x=0.02,
            xanchor='left'
        ),
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
            title=dict(
                text=combo_field.replace('_', ' ').title(),
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
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=PROFESSIONAL_COLORS['grid_color'],
            borderwidth=1
        ),
        margin=dict(l=60, r=250, t=80, b=60),
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

    # 1. Pivoted Count Table
    # Add Month_Display to df_count for better readability
    df_count = pd.merge(
        df_count,
        df_kpler_charts[["Diversion_month", "Month_Display"]].drop_duplicates(),
        on="Diversion_month",
        how="left"
    )

    # Create pivot table with Month_Display as index and combo_field as columns
    pivot_count = df_count.pivot_table(
        index="Month_Display",
        columns=combo_field,
        values="Count",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    # Sort in descending order (newest first)
    pivot_count = pivot_count.sort_values("Month_Display", ascending=False)

    # Convert to records for DataTable
    count_table_data = pivot_count.to_dict('records')
    # Create columns with split headers
    count_table_columns = create_split_header_columns(pivot_count)

    # 2. Pivoted Added Shipping Days Table
    df_days = pd.merge(
        df_days,
        df_kpler_charts[["Diversion_month", "Month_Display"]].drop_duplicates(),
        on="Diversion_month",
        how="left"
    )

    pivot_days = df_days.pivot_table(
        index="Month_Display",
        columns=combo_field,
        values="Added shipping days",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    # Sort in descending order (newest first)
    pivot_days = pivot_days.sort_values("Month_Display", ascending=False)

    # Round values to integers
    for col in pivot_days.columns:
        if col != "Month_Display":
            pivot_days[col] = pivot_days[col].round(0).astype(int)

    days_table_data = pivot_days.to_dict('records')
    # Create columns with split headers
    days_table_columns = create_split_header_columns(pivot_days)

    # 3. Pivoted Cargo Volumes Table
    df_volumes = pd.merge(
        df_volumes,
        df_kpler_charts[["Diversion_month", "Month_Display"]].drop_duplicates(),
        on="Diversion_month",
        how="left"
    )

    pivot_volumes = df_volumes.pivot_table(
        index="Month_Display",
        columns=combo_field,
        values="Cubic Meters",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    # Sort in descending order (newest first)
    pivot_volumes = pivot_volumes.sort_values("Month_Display", ascending=False)

    # Round values to integers
    for col in pivot_volumes.columns:
        if col != "Month_Display":
            pivot_volumes[col] = pivot_volumes[col].round(0).astype(int)

    volumes_table_data = pivot_volumes.to_dict('records')
    # Create columns with split headers
    volumes_table_columns = create_split_header_columns(pivot_volumes)

    # Return all outputs - note we're returning ONE figure instead of three now
    return (
        data_kpler_diversions, diversion_columns,  # Main diversion table data & columns
        fig,  # Single combined figure with subplots
        count_table_data, count_table_columns,  # Count table data and columns
        days_table_data, days_table_columns  # Added days table data and columns
    )

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
            print(f"Error in toggle_destination_continent_expansion: {e}")
    
    return expanded_continents or []
 