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


def fetch_rolling_windows_data(engine, schema, classification_mode='Country'):
    """
    Fetch 7-day and 30-day rolling window data for each country/classification and installation,
    including previous year data for seasonal comparison
    
    Args:
        engine: Database engine
        schema: Database schema
        classification_mode: 'Country' or 'Classification Level 1'
    """
    
    try:
        with engine.connect() as conn:
            # Get data for current period and same period last year
            if classification_mode == 'Classification Level 1':
                query = text(f"""
                WITH latest_data AS (
                    SELECT MAX(upload_timestamp_utc) as max_timestamp
                    FROM {schema}.kpler_trades
                )
                SELECT 
                    COALESCE(mc.country_classification_level1, 'Unknown') as origin_country_name,
                    kt.origin_country_name as actual_country_name,
                    kt.installation_origin_name,
                    kt.start::date as flow_date,
                    kt.cargo_origin_cubic_meters * 0.6 / 1000 as mcmd
                FROM {schema}.kpler_trades kt
                LEFT JOIN {schema}.mappings_country mc ON kt.origin_country_name = mc.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND (
                        -- Current 30-day window
                        (kt.start >= CURRENT_DATE - INTERVAL '30 days' AND kt.start <= CURRENT_DATE)
                        OR
                        -- Same 30-day window last year
                        (kt.start >= CURRENT_DATE - INTERVAL '1 year' - INTERVAL '30 days' 
                         AND kt.start <= CURRENT_DATE - INTERVAL '1 year')
                    )
                ORDER BY kt.start::date
                """)
            else:
                query = text(f"""
                WITH latest_data AS (
                    SELECT MAX(upload_timestamp_utc) as max_timestamp
                    FROM {schema}.kpler_trades
                )
                SELECT 
                    kt.origin_country_name,
                    kt.installation_origin_name,
                    kt.start::date as flow_date,
                    kt.cargo_origin_cubic_meters * 0.6 / 1000 as mcmd
                FROM {schema}.kpler_trades kt, latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND (
                        -- Current 30-day window
                        (kt.start >= CURRENT_DATE - INTERVAL '30 days' AND kt.start <= CURRENT_DATE)
                        OR
                        -- Same 30-day window last year
                        (kt.start >= CURRENT_DATE - INTERVAL '1 year' - INTERVAL '30 days' 
                         AND kt.start <= CURRENT_DATE - INTERVAL '1 year')
                    )
                ORDER BY kt.start::date
                """)
            
            df = pd.read_sql(query, conn)
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert to datetime
            df['flow_date'] = pd.to_datetime(df['flow_date'])
            
            # Calculate 7D and 30D averages for each country-installation combination
            current_date = datetime.now().date()
            date_7d_ago = current_date - timedelta(days=7)
            date_30d_ago = current_date - timedelta(days=30)
            date_30d_y1_start = current_date - timedelta(days=365) - timedelta(days=30)
            date_30d_y1_end = current_date - timedelta(days=365)
            
            # Check if we have actual_country_name column for Classification Level 1 mode
            if 'actual_country_name' in df.columns and classification_mode == 'Classification Level 1':
                group_cols = ['origin_country_name', 'actual_country_name', 'installation_origin_name']
                result_cols_7d = ['origin_country_name', 'actual_country_name', 'installation_origin_name', '7D']
                result_cols_30d = ['origin_country_name', 'actual_country_name', 'installation_origin_name', '30D']
                result_cols_30d_y1 = ['origin_country_name', 'actual_country_name', 'installation_origin_name', '30D_Y1']
            else:
                group_cols = ['origin_country_name', 'installation_origin_name']
                result_cols_7d = ['origin_country_name', 'installation_origin_name', '7D']
                result_cols_30d = ['origin_country_name', 'installation_origin_name', '30D']
                result_cols_30d_y1 = ['origin_country_name', 'installation_origin_name', '30D_Y1']
            
            # Filter for 7D window (current)
            df_7d = df[(df['flow_date'].dt.date > date_7d_ago) & 
                      (df['flow_date'].dt.date <= current_date)].copy()
            # Group and calculate daily average
            rolling_7d = df_7d.groupby(group_cols)['mcmd'].sum() / 7
            rolling_7d = rolling_7d.round(1).reset_index()
            rolling_7d.columns = result_cols_7d
            
            # Filter for 30D window (current)
            df_30d = df[(df['flow_date'].dt.date > date_30d_ago) & 
                       (df['flow_date'].dt.date <= current_date)].copy()
            # Group and calculate daily average
            rolling_30d = df_30d.groupby(group_cols)['mcmd'].sum() / 30
            rolling_30d = rolling_30d.round(1).reset_index()
            rolling_30d.columns = result_cols_30d
            
            # Filter for 30D window (previous year)
            df_30d_y1 = df[(df['flow_date'].dt.date > date_30d_y1_start) & 
                          (df['flow_date'].dt.date <= date_30d_y1_end)].copy()
            # Group and calculate daily average for previous year
            rolling_30d_y1 = df_30d_y1.groupby(group_cols)['mcmd'].sum() / 30
            rolling_30d_y1 = rolling_30d_y1.round(1).reset_index()
            rolling_30d_y1.columns = result_cols_30d_y1
            
            # Merge all data
            if 'actual_country_name' in rolling_7d.columns:
                merge_cols = ['origin_country_name', 'actual_country_name', 'installation_origin_name']
            else:
                merge_cols = ['origin_country_name', 'installation_origin_name']
                
            result = rolling_7d.merge(
                rolling_30d,
                on=merge_cols,
                how='outer'
            )
            result = result.merge(
                rolling_30d_y1,
                on=merge_cols,
                how='outer'
            )
            
            # Add classification/country totals
            country_totals_7d = result.groupby('origin_country_name')['7D'].sum().round(1)
            country_totals_30d = result.groupby('origin_country_name')['30D'].sum().round(1)
            country_totals_30d_y1 = result.groupby('origin_country_name')['30D_Y1'].sum().round(1)
            
            if 'actual_country_name' in result.columns:
                country_totals = pd.DataFrame({
                    'origin_country_name': country_totals_7d.index,
                    'actual_country_name': '',
                    'installation_origin_name': 'Total',
                    '30D': country_totals_30d.values,
                    '7D': country_totals_7d.values,
                    '30D_Y1': country_totals_30d_y1.values
                })
            else:
                country_totals = pd.DataFrame({
                    'origin_country_name': country_totals_7d.index,
                    'installation_origin_name': 'Total',
                    '30D': country_totals_30d.values,
                    '7D': country_totals_7d.values,
                    '30D_Y1': country_totals_30d_y1.values
                })
            
            # Combine installation-level and country-level data
            final_result = pd.concat([result, country_totals], ignore_index=True)
            final_result = final_result.fillna(0)
            
            # Calculate deltas
            final_result['Δ 7D-30D'] = (final_result['7D'] - final_result['30D']).round(1)
            final_result['Δ 30D Y/Y'] = (final_result['30D'] - final_result['30D_Y1']).round(1)
            
            # Keep the 30D_Y1 column for percentage calculations
            # It will be hidden in the display
            
            return final_result
            
    except Exception as e:
        print(f"Error fetching rolling windows data: {e}")
        return pd.DataFrame()


def get_all_classification_groups(engine, schema):
    """
    Get all distinct Classification Level 1 groups that have LNG export data
    Returns list of classification names ordered by total volume
    """
    try:
        with engine.connect() as conn:
            query = text(f"""
            WITH latest_data AS (
                SELECT MAX(upload_timestamp_utc) as max_timestamp
                FROM {schema}.kpler_trades
            ),
            classification_volumes AS (
                SELECT 
                    mc.country_classification_level1 as classification,
                    SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as total_mcmd
                FROM {schema}.kpler_trades kt
                INNER JOIN {schema}.mappings_country mc ON kt.origin_country_name = mc.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start >= CURRENT_DATE - INTERVAL '30 days'
                    AND kt.start <= CURRENT_DATE
                    AND kt.installation_origin_name IS NOT NULL
                    AND mc.country_classification_level1 IS NOT NULL
                GROUP BY mc.country_classification_level1
                HAVING SUM(kt.cargo_origin_cubic_meters) > 0
            )
            SELECT classification
            FROM classification_volumes
            WHERE classification IS NOT NULL
            ORDER BY total_mcmd DESC
            """)
            
            result = pd.read_sql(query, conn)
            
            if result.empty:
                return []
            
            return result['classification'].tolist()
            
    except Exception as e:
        print(f"Error fetching classification groups: {e}")
        return []


def get_completed_periods(current_date=None):
    """
    Calculate the last 5 completed quarters, last 3 completed months, 
    and last 3 completed weeks based on current date.
    """
    if current_date is None:
        current_date = datetime.now()
    
    # Initialize lists
    completed_quarters = []
    completed_months = []
    completed_weeks = []
    
    # Calculate completed quarters (last 5)
    current_quarter = (current_date.month - 1) // 3 + 1
    current_year = current_date.year
    
    # Start from previous quarter to ensure it's completed
    quarter = current_quarter - 1
    year = current_year
    if quarter < 1:
        quarter = 4
        year -= 1
    
    for _ in range(5):
        completed_quarters.append(f"{year}-Q{quarter}")
        quarter -= 1
        if quarter < 1:
            quarter = 4
            year -= 1
    
    completed_quarters.reverse()
    
    # Calculate completed months (last 3)
    # Previous month is the last completed month
    month = current_date.month - 1
    year = current_date.year
    if month < 1:
        month = 12
        year -= 1
    
    for _ in range(3):
        month_name = calendar.month_abbr[month]
        completed_months.append(f"{month_name}-{year}")
        month -= 1
        if month < 1:
            month = 12
            year -= 1
    
    completed_months.reverse()
    
    # Calculate completed weeks (last 3 full weeks Mon-Sun)
    # Find the most recent Sunday
    days_since_sunday = (current_date.weekday() + 1) % 7
    last_sunday = current_date - timedelta(days=days_since_sunday)
    
    # If today is Sunday and it's not complete, go to previous Sunday
    if days_since_sunday == 0:
        last_sunday = last_sunday - timedelta(days=7)
    
    # Get last 3 complete weeks
    for i in range(3):
        week_end = last_sunday - timedelta(days=7*i)
        week_start = week_end - timedelta(days=6)
        week_num = week_end.isocalendar()[1]
        year = week_end.year
        # Handle year transition for week numbering
        if week_num == 1 and week_end.month == 12:
            year += 1
        completed_weeks.append(f"W{week_num:02d}-{year}")
    
    completed_weeks.reverse()
    
    return {
        'quarters': completed_quarters,
        'months': completed_months,
        'weeks': completed_weeks
    }


def fetch_supply_dest_rolling_windows(engine, schema, classification_mode='Country'):
    """
    Fetch 7-day and 30-day rolling window data for supply-destination pairs,
    including previous year data for seasonal comparison
    
    Args:
        engine: Database engine
        schema: Database schema
        classification_mode: 'Country' or 'Classification Level 1'
    """
    
    try:
        with engine.connect() as conn:
            # Get data for current period and same period last year
            if classification_mode == 'Classification Level 1':
                query = text(f"""
                WITH latest_data AS (
                    SELECT MAX(upload_timestamp_utc) as max_timestamp
                    FROM {schema}.kpler_trades
                )
                SELECT 
                    COALESCE(mc_origin.country_classification_level1, 'Unknown') as supply_classification,
                    kt.origin_country_name as supply_country,
                    COALESCE(mc_dest.country_classification_level1, 'Unknown') as demand_classification,
                    COALESCE(kt.destination_country_name, 'Unknown') as demand_country,
                    kt.start::date as flow_date,
                    kt.cargo_origin_cubic_meters * 0.6 / 1000 as mcmd
                FROM {schema}.kpler_trades kt
                LEFT JOIN {schema}.mappings_country mc_origin ON kt.origin_country_name = mc_origin.country
                LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND (
                        -- Current 30-day window
                        (kt.start >= CURRENT_DATE - INTERVAL '30 days' AND kt.start <= CURRENT_DATE)
                        OR
                        -- Same 30-day window last year
                        (kt.start >= CURRENT_DATE - INTERVAL '1 year' - INTERVAL '30 days' 
                         AND kt.start <= CURRENT_DATE - INTERVAL '1 year')
                    )
                ORDER BY kt.start::date
                """)
            else:
                query = text(f"""
                WITH latest_data AS (
                    SELECT MAX(upload_timestamp_utc) as max_timestamp
                    FROM {schema}.kpler_trades
                )
                SELECT 
                    kt.origin_country_name as supply_country,
                    COALESCE(kt.destination_country_name, 'Unknown') as demand_country,
                    kt.start::date as flow_date,
                    kt.cargo_origin_cubic_meters * 0.6 / 1000 as mcmd
                FROM {schema}.kpler_trades kt, latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND (
                        -- Current 30-day window
                        (kt.start >= CURRENT_DATE - INTERVAL '30 days' AND kt.start <= CURRENT_DATE)
                        OR
                        -- Same 30-day window last year
                        (kt.start >= CURRENT_DATE - INTERVAL '1 year' - INTERVAL '30 days' 
                         AND kt.start <= CURRENT_DATE - INTERVAL '1 year')
                    )
                ORDER BY kt.start::date
                """)
            
            df = pd.read_sql(query, conn)
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert to datetime
            df['flow_date'] = pd.to_datetime(df['flow_date'])
            
            # Get current date and calculate date ranges
            from datetime import datetime, timedelta
            current_date = datetime.now().date()
            date_7d_ago = current_date - timedelta(days=7)
            date_30d_ago = current_date - timedelta(days=30)
            date_30d_y1_start = current_date - timedelta(days=365) - timedelta(days=30)
            date_30d_y1_end = current_date - timedelta(days=365)
            
            # Determine group columns based on mode
            if classification_mode == 'Classification Level 1':
                group_cols = ['supply_classification', 'demand_classification', 'demand_country', 'supply_country']
            else:
                group_cols = ['supply_country', 'demand_country']
            
            # Filter for 7D window (current)
            df_7d = df[(df['flow_date'].dt.date > date_7d_ago) & 
                      (df['flow_date'].dt.date <= current_date)].copy()
            # Group and calculate daily average
            rolling_7d = df_7d.groupby(group_cols)['mcmd'].sum() / 7
            rolling_7d = rolling_7d.round(1).reset_index()
            rolling_7d.columns = group_cols + ['7D']
            
            # Filter for 30D window (current)
            df_30d = df[(df['flow_date'].dt.date > date_30d_ago) & 
                       (df['flow_date'].dt.date <= current_date)].copy()
            # Group and calculate daily average
            rolling_30d = df_30d.groupby(group_cols)['mcmd'].sum() / 30
            rolling_30d = rolling_30d.round(1).reset_index()
            rolling_30d.columns = group_cols + ['30D']
            
            # Filter for 30D window (previous year)
            df_30d_y1 = df[(df['flow_date'].dt.date > date_30d_y1_start) & 
                          (df['flow_date'].dt.date <= date_30d_y1_end)].copy()
            # Group and calculate daily average for previous year
            rolling_30d_y1 = df_30d_y1.groupby(group_cols)['mcmd'].sum() / 30
            rolling_30d_y1 = rolling_30d_y1.round(1).reset_index()
            rolling_30d_y1.columns = group_cols + ['30D_Y1']
            
            # Merge all data
            result = rolling_7d.merge(rolling_30d, on=group_cols, how='outer')
            result = result.merge(rolling_30d_y1, on=group_cols, how='outer')
            
            # Add totals based on mode
            if classification_mode == 'Classification Level 1':
                # Classification to Classification totals
                class_totals_7d = result.groupby(['supply_classification', 'demand_classification'])['7D'].sum().round(1)
                class_totals_30d = result.groupby(['supply_classification', 'demand_classification'])['30D'].sum().round(1)
                class_totals_30d_y1 = result.groupby(['supply_classification', 'demand_classification'])['30D_Y1'].sum().round(1)
                
                class_totals = pd.DataFrame({
                    'supply_classification': class_totals_7d.index.get_level_values(0),
                    'demand_classification': class_totals_7d.index.get_level_values(1),
                    'demand_country': 'Total',
                    'supply_country': 'Total',
                    '7D': class_totals_7d.values,
                    '30D': class_totals_30d.values,
                    '30D_Y1': class_totals_30d_y1.values
                })
                
                # Country totals
                country_totals_7d = result.groupby(['supply_classification', 'demand_classification', 'demand_country'])['7D'].sum().round(1)
                country_totals_30d = result.groupby(['supply_classification', 'demand_classification', 'demand_country'])['30D'].sum().round(1)
                country_totals_30d_y1 = result.groupby(['supply_classification', 'demand_classification', 'demand_country'])['30D_Y1'].sum().round(1)
                
                country_totals = pd.DataFrame({
                    'supply_classification': country_totals_7d.index.get_level_values(0),
                    'demand_classification': country_totals_7d.index.get_level_values(1),
                    'demand_country': country_totals_7d.index.get_level_values(2),
                    'supply_country': 'Total',
                    '7D': country_totals_7d.values,
                    '30D': country_totals_30d.values,
                    '30D_Y1': country_totals_30d_y1.values
                })
                
                # Combine all
                final_result = pd.concat([result, country_totals, class_totals], ignore_index=True)
            else:
                final_result = result
            
            final_result = final_result.fillna(0)
            
            # Calculate deltas
            final_result['Δ 7D-30D'] = (final_result['7D'] - final_result['30D']).round(1)
            final_result['Δ 30D Y/Y'] = (final_result['30D'] - final_result['30D_Y1']).round(1)
            
            # Keep the 30D_Y1 column for percentage calculations
            # It will be hidden in the display
            
            return final_result
            
    except Exception as e:
        print(f"Error fetching supply-destination rolling windows data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_supply_dest_summary_data(engine, schema, classification_mode, quarters_df, months_df, weeks_df):
    """Combine supply-destination quarters, months, and weeks data into summary format
    
    Similar to fetch_summary_table_data but for supply-destination bilateral flows
    """
    try:
        if quarters_df.empty and months_df.empty and weeks_df.empty:
            return pd.DataFrame()
        
        # Get current date to determine what's complete
        current_date = datetime.now()
        current_quarter = (current_date.month - 1) // 3 + 1
        current_year = current_date.year
        
        # Fetch rolling windows data for supply-destination pairs
        rolling_data = fetch_supply_dest_rolling_windows(engine, schema, classification_mode)
        print(f"Supply-dest rolling data shape: {rolling_data.shape if not rolling_data.empty else 'Empty'}")
        
        # Also fetch the global rolling totals (same as LNG loadings uses) for GRAND TOTAL
        global_rolling_data = fetch_rolling_windows_data(engine, schema, classification_mode)
        print(f"Global rolling data shape: {global_rolling_data.shape if not global_rolling_data.empty else 'Empty'}")
        
        # Identify columns based on classification mode
        if classification_mode == 'Classification Level 1':
            id_cols = ['supply_classification', 'demand_classification', 'demand_country', 'supply_country']
        else:
            id_cols = ['supply_country', 'demand_country']
        
        # Get period columns
        quarter_cols = [col for col in quarters_df.columns if col not in id_cols]
        month_cols = [col for col in months_df.columns if col not in id_cols]
        week_cols = [col for col in weeks_df.columns if col not in id_cols]
        
        # Filter out current/incomplete periods - same logic as LNG loadings table
        # For quarters: exclude current quarter
        quarter_cols_filtered = []
        for col in quarter_cols:
            if "Q" in col and "'" in col:
                q_num = int(col.split("Q")[1].split("'")[0])
                year = int("20" + col.split("'")[1])
                # Exclude if it's the current quarter or future
                if year < current_year or (year == current_year and q_num < current_quarter):
                    quarter_cols_filtered.append(col)
        
        # Sort and take last 5 completed quarters
        quarter_cols_sorted = sorted(quarter_cols_filtered, 
                                    key=lambda x: (x.split("'")[1], x.split("Q")[1].split("'")[0]))
        selected_quarter_cols = quarter_cols_sorted[-5:] if len(quarter_cols_sorted) >= 5 else quarter_cols_sorted
        
        # For months: exclude current month
        month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        month_cols_filtered = []
        for col in month_cols:
            if "'" in col and not col.startswith("Q") and not col.startswith("W"):
                month_abbr = col.split("'")[0]
                year = int("20" + col.split("'")[1])
                month_num = month_order.get(month_abbr, 0)
                # Exclude if it's the current month or future
                if year < current_year or (year == current_year and month_num < current_date.month):
                    month_cols_filtered.append(col)
        
        # Sort and take last 3 completed months
        month_cols_sorted = sorted(month_cols_filtered,
                                  key=lambda x: (x.split("'")[1], month_order.get(x.split("'")[0], 0)))
        selected_month_cols = month_cols_sorted[-3:] if len(month_cols_sorted) >= 3 else month_cols_sorted
        
        # For weeks: exclude current week
        # Calculate current week number
        current_week_num = current_date.isocalendar()[1]
        week_cols_filtered = []
        for col in week_cols:
            if "W" in col and "'" in col:
                week_num = int(col.split("W")[1].split("'")[0])
                year = int("20" + col.split("'")[1])
                # Exclude if it's the current week or future
                if year < current_year or (year == current_year and week_num < current_week_num):
                    week_cols_filtered.append(col)
        
        # Sort and take last 3 completed weeks
        week_cols_sorted = sorted(week_cols_filtered,
                                key=lambda x: (x.split("'")[1], x.split("W")[1].split("'")[0].zfill(2)))
        selected_week_cols = week_cols_sorted[-3:] if len(week_cols_sorted) >= 3 else week_cols_sorted
        
        # Create subsets with selected columns
        quarters_subset = quarters_df[id_cols + selected_quarter_cols].copy()
        months_subset = months_df[id_cols + selected_month_cols].copy()
        weeks_subset = weeks_df[id_cols + selected_week_cols].copy()
        
        # Merge quarters and months first
        result = quarters_subset.copy()
        result = result.merge(months_subset, on=id_cols, how='outer')
        
        # Add 30D column right after months (before weeks)
        if not rolling_data.empty:
            rolling_30d_cols = id_cols + ['30D']
            result = result.merge(
                rolling_data[rolling_30d_cols],
                on=id_cols,
                how='left'
            )
        else:
            result['30D'] = 0
        
        # Then merge with weeks
        result = result.merge(weeks_subset, on=id_cols, how='outer')
        
        # Finally add the remaining rolling window columns
        if not rolling_data.empty:
            final_rolling_cols = id_cols + ['7D', 'Δ 7D-30D', 'Δ 30D Y/Y']
            result = result.merge(
                rolling_data[final_rolling_cols],
                on=id_cols,
                how='left'
            )
        else:
            result['7D'] = 0
            result['Δ 7D-30D'] = 0
            result['Δ 30D Y/Y'] = 0
        
        # Fill NaN values with 0
        result = result.fillna(0)
        
        # Ensure numeric columns are float
        numeric_cols = selected_quarter_cols + selected_month_cols + ['30D'] + selected_week_cols + ['7D', 'Δ 7D-30D', 'Δ 30D Y/Y']
        for col in numeric_cols:
            if col in result.columns:
                result[col] = result[col].astype(float)
        
        # Sort by classification/country pairs
        if classification_mode == 'Classification Level 1':
            result = result.sort_values(['supply_classification', 'demand_classification', 'demand_country', 'supply_country'])
        else:
            result = result.sort_values(['supply_country', 'demand_country'])
        
        result = result.reset_index(drop=True)
        
        # Add a GRAND TOTAL row using global rolling data for 7D, 30D columns
        # This ensures consistency with LNG loadings table
        if not global_rolling_data.empty:
            # Get the global totals (sum of all country/classification totals)
            global_totals = global_rolling_data[global_rolling_data['installation_origin_name'] == 'Total']
            
            if not global_totals.empty:
                # Calculate sums for rolling columns
                global_7d = global_totals['7D'].sum() if '7D' in global_totals.columns else 0
                global_30d = global_totals['30D'].sum() if '30D' in global_totals.columns else 0
                global_delta_7d_30d = global_totals['Δ 7D-30D'].sum() if 'Δ 7D-30D' in global_totals.columns else 0
                global_delta_30d_yy = global_totals['Δ 30D Y/Y'].sum() if 'Δ 30D Y/Y' in global_totals.columns else 0
                
                # Calculate sums for other columns from the result dataframe
                other_cols = {col: result[col].sum() for col in selected_quarter_cols + selected_month_cols + selected_week_cols}
                
                # Create GRAND TOTAL row
                if classification_mode == 'Classification Level 1':
                    grand_total_row = pd.DataFrame([{
                        'supply_classification': 'GRAND TOTAL',
                        'demand_classification': '',
                        'demand_country': '',
                        'supply_country': '',
                        **other_cols,
                        '30D': global_30d,
                        '7D': global_7d,
                        'Δ 7D-30D': global_delta_7d_30d,
                        'Δ 30D Y/Y': global_delta_30d_yy
                    }])
                else:
                    grand_total_row = pd.DataFrame([{
                        'supply_country': 'GRAND TOTAL',
                        'demand_country': '',
                        **other_cols,
                        '30D': global_30d,
                        '7D': global_7d,
                        'Δ 7D-30D': global_delta_7d_30d,
                        'Δ 30D Y/Y': global_delta_30d_yy
                    }])
                
                # Append GRAND TOTAL row
                result = pd.concat([result, grand_total_row], ignore_index=True)
        
        return result
            
    except Exception as e:
        print(f"Error fetching supply-destination summary data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_summary_table_data(engine, schema, classification_mode='Country'):
    """
    Fetch data for summary table showing last 5 completed quarters, 3 completed months, 3 completed weeks,
    and 7D/30D rolling windows
    Returns data in the same format as quarters/months/weeks tables for consistency
    
    Args:
        engine: Database engine
        schema: Database schema
        classification_mode: 'Country' or 'Classification Level 1'
    """
    try:
        # Fetch the existing data using the same function
        quarters_df, months_df, weeks_df = fetch_installation_data(engine, schema, classification_mode)
        
        if quarters_df.empty and months_df.empty and weeks_df.empty:
            return pd.DataFrame()
        
        # Also fetch 7D and 30D rolling data directly from database
        rolling_data = fetch_rolling_windows_data(engine, schema, classification_mode)
        
        if rolling_data.empty:
            # If no rolling data, create empty columns
            rolling_data = pd.DataFrame()
        
        # Get current date to determine what's complete
        current_date = datetime.now()
        current_quarter = (current_date.month - 1) // 3 + 1
        current_year = current_date.year
        
        # Get period columns - identify by pattern (exclude id columns)
        quarter_cols = [col for col in quarters_df.columns 
                       if col not in ['origin_country_name', 'actual_country_name', 'installation_origin_name']]
        month_cols = [col for col in months_df.columns 
                     if col not in ['origin_country_name', 'actual_country_name', 'installation_origin_name']]
        week_cols = [col for col in weeks_df.columns 
                    if col not in ['origin_country_name', 'actual_country_name', 'installation_origin_name']]
        
        # Filter out current/incomplete periods
        # For quarters: exclude current quarter
        quarter_cols_filtered = []
        for col in quarter_cols:
            if "Q" in col and "'" in col:
                q_num = int(col.split("Q")[1].split("'")[0])
                year = int("20" + col.split("'")[1])
                # Exclude if it's the current quarter or future
                if year < current_year or (year == current_year and q_num < current_quarter):
                    quarter_cols_filtered.append(col)
        
        # Sort and take last 5 completed quarters
        quarter_cols_sorted = sorted(quarter_cols_filtered, 
                                    key=lambda x: (x.split("'")[1], x.split("Q")[1].split("'")[0]))
        selected_quarter_cols = quarter_cols_sorted[-5:] if len(quarter_cols_sorted) >= 5 else quarter_cols_sorted
        
        # For months: exclude current month
        month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        month_cols_filtered = []
        for col in month_cols:
            if "'" in col and not col.startswith("Q") and not col.startswith("W"):
                month_abbr = col.split("'")[0]
                year = int("20" + col.split("'")[1])
                month_num = month_order.get(month_abbr, 0)
                # Exclude if it's the current month or future
                if year < current_year or (year == current_year and month_num < current_date.month):
                    month_cols_filtered.append(col)
        
        # Sort and take last 3 completed months
        month_cols_sorted = sorted(month_cols_filtered,
                                  key=lambda x: (x.split("'")[1], month_order.get(x.split("'")[0], 0)))
        selected_month_cols = month_cols_sorted[-3:] if len(month_cols_sorted) >= 3 else month_cols_sorted
        
        # For weeks: exclude current week
        # Calculate current week number
        current_week_num = current_date.isocalendar()[1]
        week_cols_filtered = []
        for col in week_cols:
            if "W" in col and "'" in col:
                week_num = int(col.split("W")[1].split("'")[0])
                year = int("20" + col.split("'")[1])
                # Exclude if it's the current week or future
                if year < current_year or (year == current_year and week_num < current_week_num):
                    week_cols_filtered.append(col)
        
        # Sort and take last 3 completed weeks
        week_cols_sorted = sorted(week_cols_filtered,
                                 key=lambda x: (x.split("'")[1], x.split("W")[1].split("'")[0].zfill(2)))
        selected_week_cols = week_cols_sorted[-3:] if len(week_cols_sorted) >= 3 else week_cols_sorted
        
        # Create subset dataframes with selected columns - preserve actual_country_name if it exists
        if 'actual_country_name' in quarters_df.columns:
            quarters_subset = quarters_df[['origin_country_name', 'actual_country_name', 'installation_origin_name'] + selected_quarter_cols].copy()
        else:
            quarters_subset = quarters_df[['origin_country_name', 'installation_origin_name'] + selected_quarter_cols].copy()
            
        if 'actual_country_name' in months_df.columns:
            months_subset = months_df[['origin_country_name', 'actual_country_name', 'installation_origin_name'] + selected_month_cols].copy()
        else:
            months_subset = months_df[['origin_country_name', 'installation_origin_name'] + selected_month_cols].copy()
            
        if 'actual_country_name' in weeks_df.columns:
            weeks_subset = weeks_df[['origin_country_name', 'actual_country_name', 'installation_origin_name'] + selected_week_cols].copy()
        else:
            weeks_subset = weeks_df[['origin_country_name', 'installation_origin_name'] + selected_week_cols].copy()
        
        # Merge the dataframes in desired order
        # Start with quarters
        result = quarters_subset.copy()
        
        # Determine merge keys based on whether actual_country_name exists
        if 'actual_country_name' in result.columns:
            merge_keys = ['origin_country_name', 'actual_country_name', 'installation_origin_name']
        else:
            merge_keys = ['origin_country_name', 'installation_origin_name']
        
        # Merge with months
        result = result.merge(
            months_subset,
            on=merge_keys,
            how='outer'
        )
        
        # Now add 30D column right after months (before weeks)
        if not rolling_data.empty:
            # Check if actual_country_name exists in rolling_data
            if 'actual_country_name' in rolling_data.columns and 'actual_country_name' in result.columns:
                rolling_merge_cols = ['origin_country_name', 'actual_country_name', 'installation_origin_name', '30D']
            else:
                rolling_merge_cols = ['origin_country_name', 'installation_origin_name', '30D']
            
            # First merge just the 30D column
            result = result.merge(
                rolling_data[rolling_merge_cols],
                on=merge_keys,
                how='left'
            )
        else:
            result['30D'] = 0
        
        # Then merge with weeks
        result = result.merge(
            weeks_subset,
            on=merge_keys,
            how='outer'
        )
        
        # Finally add the remaining rolling window columns
        if not rolling_data.empty:
            # Check columns for the final merge
            if 'actual_country_name' in rolling_data.columns and 'actual_country_name' in result.columns:
                final_rolling_cols = ['origin_country_name', 'actual_country_name', 'installation_origin_name', '7D', 'Δ 7D-30D', 'Δ 30D Y/Y']
            else:
                final_rolling_cols = ['origin_country_name', 'installation_origin_name', '7D', 'Δ 7D-30D', 'Δ 30D Y/Y']
            
            result = result.merge(
                rolling_data[final_rolling_cols],
                on=merge_keys,
                how='left'
            )
        else:
            result['7D'] = 0
            result['Δ 7D-30D'] = 0
            result['Δ 30D Y/Y'] = 0
        
        # Fill NaN values with 0
        result = result.fillna(0)
        
        # Ensure numeric columns are float (in the order they appear in the dataframe)
        numeric_cols = selected_quarter_cols + selected_month_cols + ['30D'] + selected_week_cols + ['7D', 'Δ 7D-30D', 'Δ 30D Y/Y']
        for col in numeric_cols:
            if col in result.columns:
                result[col] = result[col].astype(float)
        
        # Sort by country and installation (Total rows should be at bottom of each country)
        result['is_total'] = result['installation_origin_name'] == 'Total'
        result = result.sort_values(['origin_country_name', 'is_total', 'installation_origin_name'])
        result = result.drop('is_total', axis=1)
        
        # Round all numeric values
        for col in numeric_cols:
            if col in result.columns:
                result[col] = result[col].round(1)
        
        return result
            
    except Exception as e:
        print(f"Error fetching summary table data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_global_supply_data(engine, schema, classification_mode='Country'):
    """Fetch daily global LNG supply data for seasonal chart
    
    Args:
        engine: Database engine
        schema: Database schema
        classification_mode: 'Country' or 'Classification Level 1' (not used for global data)
    """
    
    try:
        with engine.connect() as conn:
            # Get daily aggregated data with rolling average calculated in SQL
            query = text(f"""
            WITH latest_data AS (
                SELECT MAX(upload_timestamp_utc) as max_timestamp
                FROM {schema}.kpler_trades
            ),
            -- Get all unique continents globally
            all_continents AS (
                SELECT DISTINCT 
                    COALESCE(NULLIF(continent_destination_name, ''), 'Unknown') as continent_destination
                FROM {schema}.kpler_trades kt
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
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
                FROM {schema}.kpler_trades kt
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
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
                        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
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
            """)
            
            df = pd.read_sql(query, conn)
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
    except Exception as e:
        print(f"Error fetching global supply data: {e}")
        return pd.DataFrame()


def fetch_country_supply_data(engine, schema, country_name, classification_mode='Country'):
    """Fetch daily LNG supply data for a specific country or classification
    
    Args:
        engine: Database engine
        schema: Database schema
        country_name: Country name or classification level name
        classification_mode: 'Country' or 'Classification Level 1'
    """
    
    try:
        with engine.connect() as conn:
            # Get daily aggregated data for specific country/classification with rolling average calculated in SQL
            if classification_mode == 'Classification Level 1':
                query = text(f"""
                WITH latest_data AS (
                    SELECT MAX(upload_timestamp_utc) as max_timestamp
                    FROM {schema}.kpler_trades
                ),
                -- Get all unique continents for this classification
                all_continents AS (
                    SELECT DISTINCT 
                        COALESCE(NULLIF(continent_destination_name, ''), 'Unknown') as continent_destination
                    FROM {schema}.kpler_trades kt
                    INNER JOIN {schema}.mappings_country mc ON kt.origin_country_name = mc.country
                    , latest_data ld
                    WHERE kt.upload_timestamp_utc = ld.max_timestamp
                        AND mc.country_classification_level1 = :country
                        AND mc.country_classification_level1 IS NOT NULL
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
                    FROM {schema}.kpler_trades kt
                    INNER JOIN {schema}.mappings_country mc ON kt.origin_country_name = mc.country
                    , latest_data ld
                    WHERE kt.upload_timestamp_utc = ld.max_timestamp
                        AND mc.country_classification_level1 = :country
                        AND mc.country_classification_level1 IS NOT NULL
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
                            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
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
                """)
            else:
                query = text(f"""
                WITH latest_data AS (
                    SELECT MAX(upload_timestamp_utc) as max_timestamp
                    FROM {schema}.kpler_trades
                ),
                -- Get all unique continents for this country
                all_continents AS (
                    SELECT DISTINCT 
                        COALESCE(NULLIF(continent_destination_name, ''), 'Unknown') as continent_destination
                    FROM {schema}.kpler_trades kt
                    , latest_data ld
                    WHERE kt.upload_timestamp_utc = ld.max_timestamp
                        AND kt.origin_country_name = :country
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
                    FROM {schema}.kpler_trades kt
                    , latest_data ld
                    WHERE kt.upload_timestamp_utc = ld.max_timestamp
                        AND kt.origin_country_name = :country
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
                            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
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
                """)
            
            df = pd.read_sql(query, conn, params={"country": country_name})
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
    except Exception as e:
        print(f"Error fetching {country_name} supply data: {e}")
        return pd.DataFrame()


def fetch_installation_data(engine, schema, classification_mode='Country'):
    """Fetch installation data and process for quarters, months, and weeks
    
    Args:
        engine: Database engine
        schema: Database schema
        classification_mode: 'Country' or 'Classification Level 1'
    """
    
    current_date = datetime.now()
    
    try:
        with engine.connect() as conn:
            # Get all data in one query
            if classification_mode == 'Classification Level 1':
                base_query = text(f"""
                WITH latest_data AS (
                    SELECT MAX(upload_timestamp_utc) as max_timestamp
                    FROM {schema}.kpler_trades
                )
                SELECT 
                    COALESCE(mc.country_classification_level1, 'Unknown') as classification_name,
                    kt.origin_country_name as country_name,
                    kt.installation_origin_name,
                    kt.start::date as flow_date,
                    EXTRACT(YEAR FROM kt.start) as year,
                    EXTRACT(QUARTER FROM kt.start) as quarter,
                    EXTRACT(MONTH FROM kt.start) as month,
                    EXTRACT(WEEK FROM kt.start) as week,
                    kt.cargo_origin_cubic_meters as volume
                FROM {schema}.kpler_trades kt
                LEFT JOIN {schema}.mappings_country mc ON kt.origin_country_name = mc.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start IS NOT NULL
                    AND kt.start >= '2022-01-01'
                    AND kt.start <= CURRENT_DATE
                """)
            else:
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
                    AND kt.start IS NOT NULL
                    AND kt.start >= '2022-01-01'
                    AND kt.start <= CURRENT_DATE
                """)
            
            df = pd.read_sql(base_query, conn)
    
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Common data preparation - do once for all processing functions
    if classification_mode == 'Classification Level 1':
        # In classification mode, we have both classification_name and country_name
        df['classification_name'] = df['classification_name'].str.strip()
        df['country_name'] = df['country_name'].str.strip()
        df['origin_country_name'] = df['classification_name']  # Use classification as primary grouping
        df['actual_country_name'] = df['country_name']  # Keep actual country for sub-grouping
    else:
        df['origin_country_name'] = df['origin_country_name'].str.strip()
        df['actual_country_name'] = df['origin_country_name']  # In country mode, they're the same
    
    # Handle NULL installations - replace with 'Unknown' or keep as NaN
    df['installation_origin_name'] = df['installation_origin_name'].fillna('Unknown')
    df['installation_origin_name'] = df['installation_origin_name'].str.strip()
    df['mcmd'] = df['volume'] * 0.6 / 1000
    df['flow_date'] = pd.to_datetime(df['flow_date'])
    
    # Process data for three separate tables
    quarters_df = process_quarters_data(df, current_date, classification_mode)
    months_df = process_months_data(df, current_date, classification_mode)
    weeks_df = process_weeks_data(df, current_date, classification_mode)
    
    return quarters_df, months_df, weeks_df


def fetch_supply_destination_data(engine, schema, classification_mode='Country'):
    """Fetch bilateral trade flow data for supply-destination table
    
    Args:
        engine: Database engine
        schema: Database schema
        classification_mode: 'Country' or 'Classification Level 1'
    """
    
    current_date = datetime.now()
    
    try:
        with engine.connect() as conn:
            # Get bilateral trade flow data
            if classification_mode == 'Classification Level 1':
                base_query = text(f"""
                WITH latest_data AS (
                    SELECT MAX(upload_timestamp_utc) as max_timestamp
                    FROM {schema}.kpler_trades
                )
                SELECT 
                    COALESCE(mc_origin.country_classification_level1, 'Unknown') as supply_classification,
                    kt.origin_country_name as supply_country,
                    COALESCE(mc_dest.country_classification_level1, 'Unknown') as demand_classification,
                    COALESCE(kt.destination_country_name, 'Unknown') as demand_country,
                    kt.start::date as flow_date,
                    EXTRACT(YEAR FROM kt.start) as year,
                    EXTRACT(QUARTER FROM kt.start) as quarter,
                    EXTRACT(MONTH FROM kt.start) as month,
                    EXTRACT(WEEK FROM kt.start) as week,
                    kt.cargo_origin_cubic_meters as volume
                FROM {schema}.kpler_trades kt
                LEFT JOIN {schema}.mappings_country mc_origin ON kt.origin_country_name = mc_origin.country
                LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start IS NOT NULL
                    AND kt.start >= '2022-01-01'
                    AND kt.start <= CURRENT_DATE
                """)
            else:
                base_query = text(f"""
                WITH latest_data AS (
                    SELECT MAX(upload_timestamp_utc) as max_timestamp
                    FROM {schema}.kpler_trades
                )
                SELECT 
                    kt.origin_country_name as supply_country,
                    COALESCE(kt.destination_country_name, 'Unknown') as demand_country,
                    kt.start::date as flow_date,
                    EXTRACT(YEAR FROM kt.start) as year,
                    EXTRACT(QUARTER FROM kt.start) as quarter,
                    EXTRACT(MONTH FROM kt.start) as month,
                    EXTRACT(WEEK FROM kt.start) as week,
                    kt.cargo_origin_cubic_meters as volume
                FROM {schema}.kpler_trades kt, latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start IS NOT NULL
                    AND kt.start >= '2022-01-01'
                    AND kt.start <= CURRENT_DATE
                """)
            
            df = pd.read_sql(base_query, conn)
    
    except Exception as e:
        print(f"Error fetching supply-destination data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Debug: Check for Unknown values
    if classification_mode == 'Classification Level 1':
        print(f"Supply classifications: {df['supply_classification'].unique()[:5]}")
        print(f"Demand classifications: {df['demand_classification'].unique()[:5]}")
        unknown_supply = df[df['supply_classification'] == 'Unknown'].shape[0]
        unknown_demand = df[df['demand_classification'] == 'Unknown'].shape[0]
        print(f"Records with Unknown supply classification: {unknown_supply}")
        print(f"Records with Unknown demand classification: {unknown_demand}")
    
    # Common data preparation
    if classification_mode == 'Classification Level 1':
        # In classification mode, we have classification and country names for both supply and demand
        # Handle NULL values and strip strings
        df['supply_classification'] = df['supply_classification'].fillna('Unknown')
        df['demand_classification'] = df['demand_classification'].fillna('Unknown')
        df['supply_classification'] = df['supply_classification'].astype(str).str.strip()
        df['supply_country'] = df['supply_country'].fillna('Unknown').astype(str).str.strip()
        df['demand_classification'] = df['demand_classification'].astype(str).str.strip()
        df['demand_country'] = df['demand_country'].fillna('Unknown').astype(str).str.strip()
    else:
        df['supply_country'] = df['supply_country'].fillna('Unknown').astype(str).str.strip()
        df['demand_country'] = df['demand_country'].fillna('Unknown').astype(str).str.strip()
    
    df['mcmd'] = df['volume'] * 0.6 / 1000
    df['flow_date'] = pd.to_datetime(df['flow_date'])
    
    # Process data for three separate tables
    quarters_df = process_supply_dest_quarters(df, current_date, classification_mode)
    months_df = process_supply_dest_months(df, current_date, classification_mode)
    weeks_df = process_supply_dest_weeks(df, current_date, classification_mode)
    
    return quarters_df, months_df, weeks_df


def process_supply_dest_quarters(df, current_date, classification_mode='Country'):
    """Process supply-destination data for quarters"""
    df = df.copy()
    
    # Create quarter period
    df['period'] = df['flow_date'].dt.to_period('Q')
    
    # Filter to last 8 quarters
    current_quarter = pd.Period(current_date, freq='Q')
    start_quarter = current_quarter - 7
    df_filtered = df[(df['period'] >= start_quarter) & (df['period'] <= current_quarter)]
    
    # Create pivot table based on mode
    if classification_mode == 'Classification Level 1':
        # Group by classifications and countries
        pivot = df_filtered.pivot_table(
            index=['supply_classification', 'demand_classification', 'demand_country', 'supply_country'],
            columns='period',
            values='mcmd',
            aggfunc='sum',
            fill_value=0
        )
    else:
        # Group by countries only
        pivot = df_filtered.pivot_table(
            index=['supply_country', 'demand_country'],
            columns='period',
            values='mcmd',
            aggfunc='sum',
            fill_value=0
        )
    
    # Calculate daily averages
    for col in pivot.columns:
        if col == current_quarter:
            days = (current_date.date() - col.start_time.date()).days + 1
        else:
            days = (col.end_time.date() - col.start_time.date()).days + 1
        pivot[col] = (pivot[col] / days).round(1)
    
    # Rename columns
    pivot.columns = [f"Q{p.quarter}'{str(p.year)[2:]}" for p in pivot.columns]
    
    # Reset index
    result = pivot.reset_index()
    
    # Add aggregation totals
    if classification_mode == 'Classification Level 1':
        # Add multiple levels of totals
        numeric_cols = [col for col in result.columns if col.startswith('Q')]
        
        # Classification to Classification totals
        class_totals = result.groupby(['supply_classification', 'demand_classification'])[numeric_cols].sum().round(1).reset_index()
        class_totals['demand_country'] = 'Total'
        class_totals['supply_country'] = 'Total'
        
        # Classification to Country totals
        country_totals = result.groupby(['supply_classification', 'demand_classification', 'demand_country'])[numeric_cols].sum().round(1).reset_index()
        country_totals['supply_country'] = 'Total'
        
        # Combine all
        final_df = pd.concat([result, country_totals, class_totals], ignore_index=True)
        final_df = final_df.sort_values(['supply_classification', 'demand_classification', 'demand_country', 'supply_country']).reset_index(drop=True)
    else:
        # Add country pair totals
        numeric_cols = [col for col in result.columns if col.startswith('Q')]
        final_df = result
    
    return final_df


def process_supply_dest_months(df, current_date, classification_mode='Country'):
    """Process supply-destination data for months"""
    df = df.copy()
    
    # Create month period
    df['period'] = df['flow_date'].dt.to_period('M')
    
    # Filter to last 12 months
    current_month = pd.Period(current_date, freq='M')
    start_month = current_month - 11
    df_filtered = df[(df['period'] >= start_month) & (df['period'] <= current_month)]
    
    # Create pivot table based on mode
    if classification_mode == 'Classification Level 1':
        pivot = df_filtered.pivot_table(
            index=['supply_classification', 'demand_classification', 'demand_country', 'supply_country'],
            columns='period',
            values='mcmd',
            aggfunc='sum',
            fill_value=0
        )
    else:
        pivot = df_filtered.pivot_table(
            index=['supply_country', 'demand_country'],
            columns='period',
            values='mcmd',
            aggfunc='sum',
            fill_value=0
        )
    
    # Calculate daily averages
    for col in pivot.columns:
        if col == current_month:
            days = current_date.day
        else:
            days = col.days_in_month
        pivot[col] = (pivot[col] / days).round(1)
    
    # Rename columns
    pivot.columns = [f"{calendar.month_abbr[p.month]}'{str(p.year)[2:]}" for p in pivot.columns]
    
    # Reset index and add totals (similar to quarters)
    result = pivot.reset_index()
    
    if classification_mode == 'Classification Level 1':
        numeric_cols = [col for col in result.columns if "'" in col and not col.startswith('Q')]
        
        class_totals = result.groupby(['supply_classification', 'demand_classification'])[numeric_cols].sum().round(1).reset_index()
        class_totals['demand_country'] = 'Total'
        class_totals['supply_country'] = 'Total'
        
        country_totals = result.groupby(['supply_classification', 'demand_classification', 'demand_country'])[numeric_cols].sum().round(1).reset_index()
        country_totals['supply_country'] = 'Total'
        
        final_df = pd.concat([result, country_totals, class_totals], ignore_index=True)
        final_df = final_df.sort_values(['supply_classification', 'demand_classification', 'demand_country', 'supply_country']).reset_index(drop=True)
    else:
        final_df = result
    
    return final_df


def process_supply_dest_weeks(df, current_date, classification_mode='Country'):
    """Process supply-destination data for weeks"""
    df = df.copy()
    
    # Create week period
    df['period'] = df['flow_date'].dt.to_period('W')
    
    # Filter to last 12 weeks
    current_week = pd.Period(current_date, freq='W')
    start_week = current_week - 11
    df_filtered = df[(df['period'] >= start_week) & (df['period'] <= current_week)]
    
    # Create pivot table based on mode
    if classification_mode == 'Classification Level 1':
        pivot = df_filtered.pivot_table(
            index=['supply_classification', 'demand_classification', 'demand_country', 'supply_country'],
            columns='period',
            values='mcmd',
            aggfunc='sum',
            fill_value=0
        )
    else:
        pivot = df_filtered.pivot_table(
            index=['supply_country', 'demand_country'],
            columns='period',
            values='mcmd',
            aggfunc='sum',
            fill_value=0
        )
    
    # Calculate daily averages
    for col in pivot.columns:
        if col == current_week:
            days = (current_date.date() - col.start_time.date()).days + 1
        else:
            days = 7
        pivot[col] = (pivot[col] / days).round(1)
    
    # Rename columns
    pivot.columns = [f"W{w.start_time.isocalendar()[1]}'{str(w.year)[2:]}" for w in pivot.columns]
    
    # Reset index and add totals
    result = pivot.reset_index()
    
    if classification_mode == 'Classification Level 1':
        numeric_cols = [col for col in result.columns if col.startswith('W')]
        
        class_totals = result.groupby(['supply_classification', 'demand_classification'])[numeric_cols].sum().round(1).reset_index()
        class_totals['demand_country'] = 'Total'
        class_totals['supply_country'] = 'Total'
        
        country_totals = result.groupby(['supply_classification', 'demand_classification', 'demand_country'])[numeric_cols].sum().round(1).reset_index()
        country_totals['supply_country'] = 'Total'
        
        final_df = pd.concat([result, country_totals, class_totals], ignore_index=True)
        final_df = final_df.sort_values(['supply_classification', 'demand_classification', 'demand_country', 'supply_country']).reset_index(drop=True)
    else:
        final_df = result
    
    return final_df


def process_quarters_data(df, current_date, classification_mode='Country'):
    """Process data for last 8 quarters using vectorized operations"""
    df = df.copy()
    
    # Create quarter period
    df['period'] = df['flow_date'].dt.to_period('Q')
    
    # Filter to last 8 quarters
    current_quarter = pd.Period(current_date, freq='Q')
    start_quarter = current_quarter - 7
    df_filtered = df[(df['period'] >= start_quarter) & (df['period'] <= current_quarter)]
    
    # Create pivot table with all quarters
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in df_filtered.columns:
        # Include actual_country_name in the index for three-level hierarchy
        pivot = df_filtered.pivot_table(
            index=['origin_country_name', 'actual_country_name', 'installation_origin_name'],
            columns='period',
            values='mcmd',
            aggfunc='sum',
            fill_value=0
        )
    else:
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
    
    # Reorder columns based on mode
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in result.columns:
        col_order = ['origin_country_name', 'actual_country_name', 'installation_origin_name'] + expected_cols
    else:
        col_order = ['origin_country_name', 'installation_origin_name'] + expected_cols
    result = result[col_order]
    
    # Add classification/country totals
    numeric_cols = expected_cols
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in result.columns:
        # For Classification Level 1, just add classification totals (countries will be computed in display)
        classification_totals = result.groupby('origin_country_name')[numeric_cols].sum().round(1).reset_index()
        classification_totals['installation_origin_name'] = 'Total'
        classification_totals['actual_country_name'] = ''
    else:
        # For Country mode, add country totals
        country_totals = result.groupby('origin_country_name')[numeric_cols].sum().round(1).reset_index()
        country_totals['installation_origin_name'] = 'Total'
        classification_totals = country_totals
    
    # Combine and sort
    final_df = pd.concat([result, classification_totals], ignore_index=True)
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in final_df.columns:
        final_df['sort_key'] = final_df['installation_origin_name'].apply(lambda x: (1, '') if x == 'Total' else (0, x))
        final_df = final_df.sort_values(['origin_country_name', 'actual_country_name', 'sort_key']).drop('sort_key', axis=1).reset_index(drop=True)
    else:
        final_df['sort_key'] = final_df['installation_origin_name'].apply(lambda x: (1, '') if x == 'Total' else (0, x))
        final_df = final_df.sort_values(['origin_country_name', 'sort_key']).drop('sort_key', axis=1).reset_index(drop=True)
    
    # Mark current quarter column for highlighting
    current_quarter_label = f"Q{current_quarter.quarter}'{str(current_quarter.year)[2:]}"
    final_df.attrs['current_period'] = current_quarter_label
    
    return final_df


def process_months_data(df, current_date, classification_mode='Country'):
    """Process data for last 12 months using vectorized operations"""
    df = df.copy()
    
    # Create month period
    df['period'] = df['flow_date'].dt.to_period('M')
    
    # Filter to last 12 months
    current_month = pd.Period(current_date, freq='M')
    start_month = current_month - 11
    df_filtered = df[(df['period'] >= start_month) & (df['period'] <= current_month)]
    
    # Create pivot table
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in df_filtered.columns:
        # Include actual_country_name in the index for three-level hierarchy
        pivot = df_filtered.pivot_table(
            index=['origin_country_name', 'actual_country_name', 'installation_origin_name'],
            columns='period',
            values='mcmd',
            aggfunc='sum',
            fill_value=0
        )
    else:
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
    
    # Reorder columns based on mode
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in result.columns:
        col_order = ['origin_country_name', 'actual_country_name', 'installation_origin_name'] + expected_cols
    else:
        col_order = ['origin_country_name', 'installation_origin_name'] + expected_cols
    result = result[col_order]
    
    # Add classification/country totals
    numeric_cols = expected_cols
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in result.columns:
        # For Classification Level 1, just add classification totals
        classification_totals = result.groupby('origin_country_name')[numeric_cols].sum().round(1).reset_index()
        classification_totals['installation_origin_name'] = 'Total'
        classification_totals['actual_country_name'] = ''
        country_totals = classification_totals
    else:
        country_totals = result.groupby('origin_country_name')[numeric_cols].sum().round(1).reset_index()
        country_totals['installation_origin_name'] = 'Total'
    
    # Combine and sort
    final_df = pd.concat([result, country_totals], ignore_index=True)
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in final_df.columns:
        final_df['sort_key'] = final_df['installation_origin_name'].apply(lambda x: (1, '') if x == 'Total' else (0, x))
        final_df = final_df.sort_values(['origin_country_name', 'actual_country_name', 'sort_key']).drop('sort_key', axis=1).reset_index(drop=True)
    else:
        final_df['sort_key'] = final_df['installation_origin_name'].apply(lambda x: (1, '') if x == 'Total' else (0, x))
        final_df = final_df.sort_values(['origin_country_name', 'sort_key']).drop('sort_key', axis=1).reset_index(drop=True)
    
    # Mark current month column for highlighting
    current_month_label = f"{calendar.month_abbr[current_month.month]}'{str(current_month.year)[2:]}"
    final_df.attrs['current_period'] = current_month_label
    
    return final_df


def process_weeks_data(df, current_date, classification_mode='Country'):
    """Process data for last 12 weeks using vectorized operations"""
    df = df.copy()
    
    # Create week period
    df['period'] = df['flow_date'].dt.to_period('W')
    
    # Filter to last 12 weeks
    current_week = pd.Period(current_date, freq='W')
    start_week = current_week - 11
    df_filtered = df[(df['period'] >= start_week) & (df['period'] <= current_week)]
    
    # Create pivot table
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in df_filtered.columns:
        # Include actual_country_name in the index for three-level hierarchy
        pivot = df_filtered.pivot_table(
            index=['origin_country_name', 'actual_country_name', 'installation_origin_name'],
            columns='period',
            values='mcmd',
            aggfunc='sum',
            fill_value=0
        )
    else:
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
    
    # Reorder columns based on mode
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in result.columns:
        col_order = ['origin_country_name', 'actual_country_name', 'installation_origin_name'] + expected_cols
    else:
        col_order = ['origin_country_name', 'installation_origin_name'] + expected_cols
    result = result[col_order]
    
    # Add classification/country totals
    numeric_cols = expected_cols
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in result.columns:
        # For Classification Level 1, just add classification totals
        classification_totals = result.groupby('origin_country_name')[numeric_cols].sum().round(1).reset_index()
        classification_totals['installation_origin_name'] = 'Total'
        classification_totals['actual_country_name'] = ''
        country_totals = classification_totals
    else:
        country_totals = result.groupby('origin_country_name')[numeric_cols].sum().round(1).reset_index()
        country_totals['installation_origin_name'] = 'Total'
    
    # Combine and sort
    final_df = pd.concat([result, country_totals], ignore_index=True)
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in final_df.columns:
        final_df['sort_key'] = final_df['installation_origin_name'].apply(lambda x: (1, '') if x == 'Total' else (0, x))
        final_df = final_df.sort_values(['origin_country_name', 'actual_country_name', 'sort_key']).drop('sort_key', axis=1).reset_index(drop=True)
    else:
        final_df['sort_key'] = final_df['installation_origin_name'].apply(lambda x: (1, '') if x == 'Total' else (0, x))
        final_df = final_df.sort_values(['origin_country_name', 'sort_key']).drop('sort_key', axis=1).reset_index(drop=True)
    
    # Mark current week column for highlighting
    current_week_label = f"W{current_week.start_time.isocalendar()[1]}'{str(current_week.year)[2:]}"
    final_df.attrs['current_period'] = current_week_label
    
    return final_df


def create_continent_destination_chart(entity_name, engine, db_schema, classification_mode='Country'):
    """Create seasonal comparison chart by continent destination for selected entity's LNG exports
    
    Args:
        entity_name: Country name or classification group name
        engine: Database engine
        db_schema: Database schema
        classification_mode: 'Country' or 'Classification Level 1'
    """
    
    try:
        # Handle Global case
        if entity_name == "Global":
            country_filter = ""
        elif classification_mode == 'Classification Level 1':
            country_filter = "AND mc.country_classification_level1 = %(entity_name)s"
        else:
            country_filter = "AND kt.origin_country_name = %(entity_name)s"
        
        # Query to get export data by continent destination
        # Add join for classification mode
        join_clause = ""
        if classification_mode == 'Classification Level 1' and entity_name != "Global":
            join_clause = f"INNER JOIN {db_schema}.mappings_country mc ON kt.origin_country_name = mc.country"
        
        query = f"""
        WITH latest_data AS (
            SELECT MAX(upload_timestamp_utc) as max_timestamp
            FROM {db_schema}.kpler_trades
        ),
        -- Get all unique continents
        all_continents AS (
            SELECT DISTINCT 
                COALESCE(NULLIF(continent_destination_name, ''), 'Unknown') as continent_destination
            FROM {db_schema}.kpler_trades kt
            {join_clause}
            , latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                {country_filter}
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
            FROM {db_schema}.kpler_trades kt
            {join_clause}
            , latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                {country_filter}
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
        -- Calculate rolling averages on complete dataset
        rolling_exports AS (
            SELECT 
                date,
                continent_destination,
                daily_export_mcmd,
                AVG(daily_export_mcmd) OVER (
                    PARTITION BY continent_destination
                    ORDER BY date 
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as rolling_avg_30d,
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
            rolling_avg_30d as rolling_avg,
            is_forecast
        FROM rolling_exports
        WHERE date >= '2024-01-01'
        ORDER BY continent_destination, date
        """
        
        params = {} if entity_name == "Global" else {'entity_name': entity_name}
        df = pd.read_sql(query, engine, params=params)
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No export data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color='#6b7280')
            )
            fig.update_layout(
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                height=300,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Get unique years and continents
        years = sorted(df['year'].unique())
        continents = sorted(df['continent_destination'].unique())
        
        # Define color palette for continents
        continent_colors = {
            'Africa': '#8E24AA',
            'Americas': '#43A047',
            'Asia': '#FF4444',
            'Europe': '#1E88E5',
            'Unknown': '#757575',
            'Oceania': '#FB8C00',
            'Middle East': '#00ACC1',
            'North America': '#D81B60',
            'South America': '#FFC107'
        }
        
        current_year = max(years)
        continent_legend_shown = {}
        
        for continent in continents:
            continent_data = df[df['continent_destination'] == continent]
            color = continent_colors.get(continent, '#808080')
            
            for year in years:
                year_continent_data = continent_data[continent_data['year'] == year].copy()
                
                if year_continent_data.empty:
                    continue
                
                year_continent_data['plot_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(year_continent_data['day_of_year'] - 1, unit='d')
                
                historical_data = year_continent_data[~year_continent_data['is_forecast']]
                forecast_data = year_continent_data[year_continent_data['is_forecast']]
                
                line_width = 2.5 if year == current_year else 1.5
                show_legend = bool(continent not in continent_legend_shown)
                if show_legend:
                    continent_legend_shown[continent] = True
                
                if not historical_data.empty:
                    fig.add_trace(go.Scatter(
                        x=historical_data['plot_date'],
                        y=historical_data['rolling_avg'],
                        mode='lines',
                        name=continent if show_legend else None,
                        legendgroup=continent,
                        line=dict(color=color, width=line_width, dash='solid'),
                        hovertemplate=f'<b>{continent} - {int(year)}</b><br>' +
                                     '%{text}<br>' +
                                     'Volume: %{y:.1f} mcm/d<br>' +
                                     '<extra></extra>',
                        text=historical_data['month_day'],
                        showlegend=show_legend
                    ))
                
                if not forecast_data.empty:
                    if not historical_data.empty:
                        connect_data = pd.concat([historical_data.tail(1), forecast_data])
                    else:
                        connect_data = forecast_data
                    
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
                        line=dict(color=forecast_color, width=line_width, dash='solid'),
                        opacity=0.6,
                        hovertemplate=f'<b>{continent} - {int(year)} (Forecast)</b><br>' +
                                     '%{text}<br>' +
                                     'Volume: %{y:.1f} mcm/d<br>' +
                                     '<extra></extra>',
                        text=connect_data['month_day'],
                        showlegend=False
                    ))
        
        # Update layout
        fig.update_layout(
            xaxis=dict(
                tickformat='%b',
                dtick='M1',
                tickangle=0,
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.3)',
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title=dict(text='mcm/d', font=dict(size=11)),
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.3)',
                tickfont=dict(size=10)
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.3,
                xanchor='center',
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(l=50, r=20, t=20, b=80),
            height=300,
            hovermode='x',
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig
        
    except Exception as e:
        print(f"Error creating continent destination chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text="Error loading data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300, paper_bgcolor='white', plot_bgcolor='white')
        return fig


def create_continent_percentage_chart(entity_name, engine, db_schema, classification_mode='Country'):
    """Create percentage distribution chart by continent destination
    
    Args:
        entity_name: Country name or classification group name
        engine: Database engine
        db_schema: Database schema
        classification_mode: 'Country' or 'Classification Level 1'
    """
    
    try:
        # Handle Global case
        if entity_name == "Global":
            country_filter = ""
        elif classification_mode == 'Classification Level 1':
            country_filter = "AND mc.country_classification_level1 = %(entity_name)s"
        else:
            country_filter = "AND kt.origin_country_name = %(entity_name)s"
        
        # Query to get export data by continent destination for percentage calculation
        # Add join for classification mode
        join_clause = ""
        if classification_mode == 'Classification Level 1' and entity_name != "Global":
            join_clause = f"INNER JOIN {db_schema}.mappings_country mc ON kt.origin_country_name = mc.country"
        
        query = f"""
        WITH latest_data AS (
            SELECT MAX(upload_timestamp_utc) as max_timestamp
            FROM {db_schema}.kpler_trades
        ),
        -- Get all unique continents
        all_continents AS (
            SELECT DISTINCT 
                COALESCE(NULLIF(continent_destination_name, ''), 'Unknown') as continent_destination
            FROM {db_schema}.kpler_trades kt
            {join_clause}
            , latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                {country_filter}
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
            FROM {db_schema}.kpler_trades kt
            {join_clause}
            , latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                {country_filter}
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
        -- Calculate rolling averages on complete dataset
        rolling_continents AS (
            SELECT 
                date,
                continent_destination,
                daily_export_mcmd,
                AVG(daily_export_mcmd) OVER (
                    PARTITION BY continent_destination
                    ORDER BY date 
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as rolling_avg_30d,
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
                SUM(rolling_avg_30d) as total_rolling_avg_30d
            FROM rolling_continents
            GROUP BY date
        )
        SELECT 
            rc.date,
            rc.continent_destination,
            EXTRACT(YEAR FROM rc.date) as year,
            EXTRACT(DOY FROM rc.date) as day_of_year,
            TO_CHAR(rc.date, 'Mon DD') as month_day,
            rc.rolling_avg_30d as rolling_avg,
            CASE 
                WHEN rt.total_rolling_avg_30d > 0 
                THEN (rc.rolling_avg_30d / rt.total_rolling_avg_30d) * 100
                ELSE 0
            END as percentage,
            rc.is_forecast
        FROM rolling_continents rc
        JOIN rolling_totals rt ON rc.date = rt.date
        WHERE rc.date >= '2024-01-01'
        ORDER BY rc.continent_destination, rc.date
        """
        
        params = {} if entity_name == "Global" else {'entity_name': entity_name}
        df = pd.read_sql(query, engine, params=params)
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No export data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color='#6b7280')
            )
            fig.update_layout(
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                height=300,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Get unique years and continents
        years = sorted(df['year'].unique())
        continents = sorted(df['continent_destination'].unique())
        
        # Define color palette - same as absolute chart
        continent_colors = {
            'Africa': '#8E24AA',
            'Americas': '#43A047',
            'Asia': '#FF4444',
            'Europe': '#1E88E5',
            'Unknown': '#757575',
            'Oceania': '#FB8C00',
            'Middle East': '#00ACC1',
            'North America': '#D81B60',
            'South America': '#FFC107'
        }
        
        current_year = max(years)
        continent_legend_shown = {}
        
        for continent in continents:
            continent_data = df[df['continent_destination'] == continent]
            color = continent_colors.get(continent, '#808080')
            
            for year in years:
                year_continent_data = continent_data[continent_data['year'] == year].copy()
                
                if year_continent_data.empty:
                    continue
                
                year_continent_data['plot_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(year_continent_data['day_of_year'] - 1, unit='d')
                
                historical_data = year_continent_data[~year_continent_data['is_forecast']]
                forecast_data = year_continent_data[year_continent_data['is_forecast']]
                
                line_width = 2.5 if year == current_year else 1.5
                show_legend = bool(continent not in continent_legend_shown)
                if show_legend:
                    continent_legend_shown[continent] = True
                
                if not historical_data.empty:
                    fig.add_trace(go.Scatter(
                        x=historical_data['plot_date'],
                        y=historical_data['percentage'],
                        mode='lines',
                        name=continent if show_legend else None,
                        legendgroup=continent,
                        line=dict(color=color, width=line_width, dash='solid'),
                        hovertemplate=f'<b>{continent} - {int(year)}</b><br>' +
                                     '%{text}<br>' +
                                     'Share: %{y:.1f}%<br>' +
                                     '<extra></extra>',
                        text=historical_data['month_day'],
                        showlegend=show_legend
                    ))
                
                if not forecast_data.empty:
                    if not historical_data.empty:
                        connect_data = pd.concat([historical_data.tail(1), forecast_data])
                    else:
                        connect_data = forecast_data
                    
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
                        line=dict(color=forecast_color, width=line_width, dash='solid'),
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
            xaxis=dict(
                tickformat='%b',
                dtick='M1',
                tickangle=0,
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.3)',
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                title=dict(text='%', font=dict(size=11)),
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.3)',
                tickfont=dict(size=10),
                range=[0, 100]
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.3,
                xanchor='center',
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(l=50, r=20, t=20, b=80),
            height=300,
            hovermode='x',
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig
        
    except Exception as e:
        print(f"Error creating continent percentage chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text="Error loading data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300, paper_bgcolor='white', plot_bgcolor='white')
        return fig


def prepare_table_for_display(df, table_type, expanded_countries=None, classification_mode='Country', expanded_classifications=None):
    """Prepare data for display in DataTable with expandable rows
    
    Args:
        df: DataFrame with the data
        table_type: Type of table (quarters, months, weeks, summary)
        expanded_countries: List of expanded country names
        classification_mode: 'Country' or 'Classification Level 1'
        expanded_classifications: List of expanded classification names (for three-level hierarchy)
    """
    if df.empty:
        return pd.DataFrame(), []
    
    expanded_countries = expanded_countries or []
    expanded_classifications = expanded_classifications or []
    
    # Helper function to check if a row has all zeros
    def has_non_zero_values(row_df):
        """Check if a DataFrame row has any non-zero numeric values"""
        numeric_cols = [col for col in row_df.columns if col not in 
                       ['origin_country_name', 'installation_origin_name', 'actual_country_name']]
        if not numeric_cols:
            return True
        return (row_df[numeric_cols] != 0).any().any()
    
    # Filter data based on expanded state
    filtered_rows = []
    entity_totals_for_grand = []  # Store entity totals for grand total calculation
    
    if classification_mode == 'Classification Level 1':
        # Three-level hierarchy: Classification -> Country -> Installation
        # The origin_country_name column contains classification names in this mode
        # The actual_country_name column contains the real country names
        
        # Group by classification (which is in origin_country_name)
        for classification in df['origin_country_name'].unique():
            classification_data = df[df['origin_country_name'] == classification]
            
            # Get classification total
            classification_total = classification_data[classification_data['installation_origin_name'] == 'Total']
            if not classification_total.empty and has_non_zero_values(classification_total):
                # Store for grand total calculation
                entity_totals_for_grand.append(classification_total.copy())
                
                # Add expand/collapse indicator and ensure actual_country_name is empty for classification totals
                classification_total = classification_total.copy()
                if classification in expanded_classifications:
                    classification_total.loc[:, 'origin_country_name'] = f"▼ {classification}"
                else:
                    classification_total.loc[:, 'origin_country_name'] = f"▶ {classification}"
                # Ensure actual_country_name column exists and is empty for classification totals
                if 'actual_country_name' not in classification_total.columns:
                    classification_total['actual_country_name'] = ''
                else:
                    classification_total.loc[:, 'actual_country_name'] = ''
                filtered_rows.append(classification_total)
            
            # If classification is expanded, show countries
            if classification in expanded_classifications:
                # Get non-total rows for this classification
                classification_installations = classification_data[classification_data['installation_origin_name'] != 'Total'].copy()
                
                # Debug: Check if actual_country_name column exists
                if 'actual_country_name' not in classification_installations.columns:
                    print(f"Warning: actual_country_name column not found. Columns: {classification_installations.columns.tolist()}")
                    # Try to display installations directly if no country grouping available
                    if not classification_installations.empty:
                        classification_installations.loc[:, 'installation_origin_name'] = "    " + classification_installations['installation_origin_name']
                        classification_installations.loc[:, 'origin_country_name'] = ""
                        filtered_rows.append(classification_installations)
                elif not classification_installations.empty:
                    # Group by actual country
                    for country in classification_installations['actual_country_name'].unique():
                        if pd.isna(country) or country == '':
                            continue
                            
                        country_data = classification_installations[classification_installations['actual_country_name'] == country]
                        
                        # Calculate country total
                        numeric_cols = [col for col in country_data.columns if col not in 
                                      ['origin_country_name', 'installation_origin_name', 'actual_country_name']]
                        
                        country_total = pd.DataFrame([{
                            'origin_country_name': '',
                            'actual_country_name': f"  ▶ {country}",  # Country with expandable indicator in actual_country_name column
                            'installation_origin_name': 'Total',  # Show 'Total' for country aggregation
                            **{col: country_data[col].sum() for col in numeric_cols}
                        }])
                        
                        # Only add country total if it has non-zero values
                        if has_non_zero_values(country_total):
                            # Check if this country is expanded
                            country_key = f"{classification}_{country}"
                            if country_key in expanded_countries:
                                # Change indicator to expanded
                                country_total.loc[:, 'actual_country_name'] = f"  ▼ {country}"
                                filtered_rows.append(country_total)
                                
                                # Show installations for this country (filter out zeros)
                                country_installations = country_data.copy()
                                # Filter out installations with all zeros
                                non_zero_mask = country_installations.apply(lambda row: has_non_zero_values(row.to_frame().T), axis=1)
                                country_installations = country_installations[non_zero_mask]
                                
                                if not country_installations.empty:
                                    country_installations = country_installations.copy()
                                    country_installations.loc[:, 'installation_origin_name'] = "      " + country_installations['installation_origin_name']
                                    country_installations.loc[:, 'origin_country_name'] = ""
                                    country_installations.loc[:, 'actual_country_name'] = ""  # Clear country name for installations
                                    filtered_rows.append(country_installations)
                            else:
                                filtered_rows.append(country_total)
    else:
        # Two-level hierarchy: Country -> Installation (original logic)
        for country in df['origin_country_name'].unique():
            country_data = df[df['origin_country_name'] == country]
            
            # Only show country total if it has non-zero values
            total_row = country_data[country_data['installation_origin_name'] == 'Total']
            if not total_row.empty and has_non_zero_values(total_row):
                # Store for grand total calculation
                entity_totals_for_grand.append(total_row.copy())
                
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
                        # Filter out installations with all zeros
                        non_zero_mask = installations.apply(lambda row: has_non_zero_values(row.to_frame().T), axis=1)
                        installations = installations[non_zero_mask]
                        
                        if not installations.empty:
                            installations = installations.copy()
                            # Indent installation names for visual hierarchy
                            installations.loc[:, 'installation_origin_name'] = "    " + installations['installation_origin_name']
                            installations.loc[:, 'origin_country_name'] = ""  # Clear country name for installations
                            filtered_rows.append(installations)
    
    # Add Grand Total row
    if entity_totals_for_grand:
        grand_total_df = pd.concat(entity_totals_for_grand, ignore_index=True)
        numeric_cols = [col for col in grand_total_df.columns if col not in ['origin_country_name', 'installation_origin_name', 'actual_country_name']]
        
        if classification_mode == 'Classification Level 1' and 'actual_country_name' in df.columns:
            grand_total_row = pd.DataFrame([{
                'origin_country_name': 'GRAND TOTAL',
                'actual_country_name': '',
                'installation_origin_name': '',
                **{col: grand_total_df[col].sum() for col in numeric_cols}
            }])
        else:
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
    
    # Rename columns for display based on classification mode
    if classification_mode == 'Classification Level 1' and 'actual_country_name' in display_df.columns:
        # For Classification Level 1, rename columns differently
        new_columns = []
        for col in display_df.columns:
            if col == 'origin_country_name':
                new_columns.append('Aggregation')
            elif col == 'actual_country_name':
                new_columns.append('Country')
            elif col == 'installation_origin_name':
                new_columns.append('Installation')
            else:
                new_columns.append(col)
        display_df.columns = new_columns
    else:
        # Original naming for Country mode
        display_df.columns = [
            'Country' if col == 'origin_country_name' 
            else 'Installation' if col == 'installation_origin_name'
            else col for col in display_df.columns
        ]
    
    # Create column definitions for DataTable
    columns = []
    for col in display_df.columns:
        if col in ['Country', 'Installation', 'Aggregation']:  # Include all text columns
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


def prepare_supply_dest_table_for_display(df, table_type, classification_mode='Country', 
                                          expanded_classifications=None, expanded_countries=None, 
                                          expanded_supply_countries=None, view_type='absolute'):
    """Prepare supply-destination data for display in DataTable with expandable rows
    
    Args:
        df: DataFrame with the supply-destination data
        table_type: Type of table (quarters, months, weeks, summary)
        classification_mode: 'Country' or 'Classification Level 1'
        expanded_classifications: List of expanded classification pairs
        expanded_countries: List of expanded demand countries
        expanded_supply_countries: List of expanded supply countries
        view_type: 'absolute' for mcm/d values, 'percentage' for market share
    """
    if df.empty:
        return pd.DataFrame(), []
    
    expanded_classifications = expanded_classifications or []
    expanded_countries = expanded_countries or []
    expanded_supply_countries = expanded_supply_countries or []
    
    # Helper function to check if a row has all zeros
    def has_non_zero_values(row_df):
        """Check if a DataFrame row has any non-zero numeric values"""
        numeric_cols = [col for col in row_df.columns if col not in 
                       ['supply_classification', 'demand_classification', 'supply_country', 'demand_country']]
        if not numeric_cols:
            return True
        return (row_df[numeric_cols] != 0).any().any()
    
    # Filter data based on expanded state
    filtered_rows = []
    entity_totals_for_grand = []
    supply_subtotals = {}  # Store subtotals by supply classification
    
    if classification_mode == 'Classification Level 1':
        # Four-level hierarchy: Supply Class → Demand Class → Demand Country → Supply Country
        
        # First, calculate subtotals for each supply classification
        for supply_class in df['supply_classification'].unique():
            supply_data = df[df['supply_classification'] == supply_class]
            supply_totals = supply_data[(supply_data['demand_country'] == 'Total') & 
                                       (supply_data['supply_country'] == 'Total')]
            if not supply_totals.empty:
                numeric_cols = [col for col in supply_totals.columns if col not in 
                              ['supply_classification', 'demand_classification', 'supply_country', 'demand_country']]
                supply_subtotals[supply_class] = {col: supply_totals[col].sum() for col in numeric_cols}
        
        # Process each supply classification group
        for supply_class in sorted(df['supply_classification'].unique()):
            supply_class_rows = []
            
            # Get all classification pairs for this supply
            supply_pairs = df[df['supply_classification'] == supply_class].groupby(['supply_classification', 'demand_classification']).size().reset_index()[['supply_classification', 'demand_classification']]
            
            for _, pair in supply_pairs.iterrows():
                demand_class = pair['demand_classification']
                pair_key = f"{supply_class}→{demand_class}"
                
                # Get data for this classification pair
                pair_data = df[(df['supply_classification'] == supply_class) & 
                              (df['demand_classification'] == demand_class)]
                
                # Get classification pair total
                class_total = pair_data[(pair_data['demand_country'] == 'Total') & 
                                       (pair_data['supply_country'] == 'Total')]
                
                if not class_total.empty and has_non_zero_values(class_total):
                    # Store for grand total
                    entity_totals_for_grand.append(class_total.copy())
                    
                    # Add expand/collapse indicator
                    class_total = class_total.copy()
                    if pair_key in expanded_classifications:
                        class_total.loc[:, 'supply_classification'] = f"▼ {supply_class}"
                        class_total.loc[:, 'demand_classification'] = f"{demand_class}"
                    else:
                        class_total.loc[:, 'supply_classification'] = f"▶ {supply_class}"
                        class_total.loc[:, 'demand_classification'] = f"{demand_class}"
                    class_total['demand_country'] = ''
                    class_total['supply_country'] = ''
                    supply_class_rows.append(class_total)
                
                # If classification pair is expanded, show demand countries
                if pair_key in expanded_classifications:
                    # Get unique demand countries for this pair
                    demand_countries = pair_data[pair_data['demand_country'] != 'Total']['demand_country'].unique()
                    
                    for demand_country in demand_countries:
                        country_key = f"{pair_key}→{demand_country}"
                        country_data = pair_data[pair_data['demand_country'] == demand_country]
                        
                        # Get country total
                        country_total = country_data[country_data['supply_country'] == 'Total']
                        
                        if not country_total.empty and has_non_zero_values(country_total):
                            country_total = country_total.copy()
                            country_total['supply_classification'] = ''
                            country_total['demand_classification'] = ''
                            if country_key in expanded_countries:
                                country_total.loc[:, 'demand_country'] = f"  ▼ {demand_country}"
                            else:
                                country_total.loc[:, 'demand_country'] = f"  ▶ {demand_country}"
                            country_total['supply_country'] = 'Total'
                            supply_class_rows.append(country_total)
                        
                        # If demand country is expanded, show supply countries
                        if country_key in expanded_countries:
                            supply_countries = country_data[country_data['supply_country'] != 'Total']
                            if not supply_countries.empty:
                                # Filter out supply countries with all zeros
                                non_zero_mask = supply_countries.apply(lambda row: has_non_zero_values(row.to_frame().T), axis=1)
                                supply_countries = supply_countries[non_zero_mask]
                                
                                if not supply_countries.empty:
                                    supply_countries = supply_countries.copy()
                                    supply_countries['supply_classification'] = ''
                                    supply_countries['demand_classification'] = ''
                                    supply_countries['demand_country'] = ''
                                    supply_countries.loc[:, 'supply_country'] = "    " + supply_countries['supply_country']
                                    supply_class_rows.append(supply_countries)
            
            # Add all rows for this supply classification
            filtered_rows.extend(supply_class_rows)
            
            # Add subtotal row for this supply classification
            # Only show when no classification pairs are expanded for this supply
            supply_pairs_expanded = any(pair_key.startswith(f"{supply_class}→") for pair_key in expanded_classifications)
            
            if supply_class in supply_subtotals and supply_class_rows and not supply_pairs_expanded:
                subtotal_row = pd.DataFrame([{
                    'supply_classification': supply_class,
                    'demand_classification': 'Total',
                    'demand_country': '',
                    'supply_country': '',
                    **supply_subtotals[supply_class]
                }])
                filtered_rows.append(subtotal_row)
    
    else:
        # Two-level hierarchy: Supply Country → Demand Country
        # First, calculate subtotals for each supply country
        for supply_country in sorted(df['supply_country'].unique()):
            supply_country_rows = []
            supply_country_total = None
            
            for demand_country in df['demand_country'].unique():
                pair_data = df[(df['supply_country'] == supply_country) & 
                              (df['demand_country'] == demand_country)]
                
                if not pair_data.empty and has_non_zero_values(pair_data):
                    # Store for grand total
                    entity_totals_for_grand.append(pair_data.copy())
                    
                    # Add to supply country rows
                    supply_country_rows.append(pair_data)
            
            # Add all rows for this supply country
            filtered_rows.extend(supply_country_rows)
            
            # Add subtotal row for this supply country
            if supply_country_rows:
                # Calculate subtotal for this supply country
                supply_total_df = pd.concat(supply_country_rows, ignore_index=True)
                numeric_cols = [col for col in supply_total_df.columns if col not in 
                               ['supply_country', 'demand_country']]
                
                subtotal_row = pd.DataFrame([{
                    'supply_country': supply_country,
                    'demand_country': 'Total',
                    **{col: supply_total_df[col].sum() for col in numeric_cols}
                }])
                filtered_rows.append(subtotal_row)
    
    # Add Grand Total row (only if not in percentage mode)
    if entity_totals_for_grand and view_type != 'percentage':
        grand_total_df = pd.concat(entity_totals_for_grand, ignore_index=True)
        numeric_cols = [col for col in grand_total_df.columns if col not in 
                       ['supply_classification', 'demand_classification', 'supply_country', 'demand_country']]
        
        if classification_mode == 'Classification Level 1':
            grand_total_row = pd.DataFrame([{
                'supply_classification': 'GRAND TOTAL',
                'demand_classification': '',
                'demand_country': '',
                'supply_country': '',
                **{col: grand_total_df[col].sum() for col in numeric_cols}
            }])
        else:
            grand_total_row = pd.DataFrame([{
                'supply_country': 'GRAND TOTAL',
                'demand_country': '',
                **{col: grand_total_df[col].sum() for col in numeric_cols}
            }])
        
        filtered_rows.append(grand_total_row)
    
    # Combine all rows
    if filtered_rows:
        display_df = pd.concat(filtered_rows, ignore_index=True)
    else:
        display_df = pd.DataFrame()
    
    # Rename columns for display based on classification mode
    if classification_mode == 'Classification Level 1':
        new_columns = []
        for col in display_df.columns:
            if col == 'supply_classification':
                new_columns.append('Aggregation Supply')
            elif col == 'demand_classification':
                new_columns.append('Aggregation Demand')
            elif col == 'demand_country':
                new_columns.append('Country Demand')
            elif col == 'supply_country':
                new_columns.append('Supply Country')
            else:
                new_columns.append(col)
        display_df.columns = new_columns
    else:
        new_columns = []
        for col in display_df.columns:
            if col == 'supply_country':
                new_columns.append('Supply Country')
            elif col == 'demand_country':
                new_columns.append('Demand Country')
            else:
                new_columns.append(col)
        display_df.columns = new_columns
    
    # Create column definitions for DataTable
    columns = []
    text_cols = ['Aggregation Supply', 'Aggregation Demand', 'Country Demand', 'Supply Country', 'Demand Country']
    delta_cols = ['Δ 7D-30D', 'Δ 30D Y/Y']
    hidden_cols = ['30D_Y1']  # Hide the Y-1 data column
    
    for col in display_df.columns:
        if col in hidden_cols:
            continue  # Skip hidden columns
        elif col in text_cols:
            columns.append({
                'name': col,
                'id': col,
                'type': 'text'
            })
        elif col in delta_cols:
            # Delta columns should always show as fixed numbers (percentage points when in percentage mode)
            columns.append({
                'name': col,
                'id': col,
                'type': 'numeric',
                'format': Format(precision=2, scheme=Scheme.fixed)
            })
        else:
            # Use percentage format if in percentage mode
            if view_type == 'percentage':
                columns.append({
                    'name': col,
                    'id': col,
                    'type': 'numeric',
                    'format': Format(precision=1, scheme=Scheme.percentage)
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
    # Interval component to trigger initial data load (runs once on page load)
    dcc.Interval(id='initial-load-trigger', interval=1000*60*60*24, n_intervals=0, max_intervals=1),
    
    # Store components for caching data
    dcc.Store(id='supply-charts-data', storage_type='local'),  # Single store for all supply chart data
    dcc.Store(id='continent-charts-data', storage_type='local'),  # Store for continent charts data
    dcc.Store(id='summary-data-store', storage_type='local'),
    dcc.Store(id='supply-dest-data-store', storage_type='local'),  # Store for supply-destination data
    
    # Store for expanded state of countries in summary table
    dcc.Store(id='summary-expanded-countries', data=[]),
    # Store for expanded state of classifications in summary table (for three-level hierarchy)
    dcc.Store(id='summary-expanded-classifications', data=[]),
    
    # Store for expanded states in supply-destination table
    dcc.Store(id='supply-dest-expanded-classifications', data=[]),  # For classification pairs
    dcc.Store(id='supply-dest-expanded-countries', data=[]),  # For demand countries
    dcc.Store(id='supply-dest-expanded-supply-countries', data=[]),  # For supply countries
    
    # Store for country classification mode
    dcc.Store(id='country-classification-mode', data='Country'),

    # Country Classification Dropdown - Top left
    html.Div([
        html.Div([
            html.Label('Country classification:', 
                      style={'display': 'inline-block', 'marginRight': '10px', 
                             'fontWeight': 'bold', 'fontSize': '14px'}),
            dcc.Dropdown(
                id='country-classification-dropdown',
                options=[
                    {'label': 'Country', 'value': 'Country'},
                    {'label': 'Classification Level 1', 'value': 'Classification Level 1'}
                ],
                value='Country',
                style={'width': '250px', 'display': 'inline-block', 'verticalAlign': 'middle'},
                clearable=False
            )
        ], style={'padding': '10px 20px', 'marginBottom': '10px'})
    ]),

    # Supply Charts Section - Dynamic container
    html.Div([
        # Header matching the style of Installation Trends
        html.Div([
            html.H3('LNG Supply - 30-Day Rolling Average', className="section-title-inline"),
        ], className="inline-section-header"),
        # Dynamic charts container - will be populated by callback
        dcc.Loading(
            id="supply-charts-loading",
            children=[
                html.Div(id='supply-charts-container')
            ],
            type="default",
        )
    ], className="main-section-container", style={'marginBottom': '30px'}),

    # Continent Destination Charts Section - 30-Day Rolling Average
    html.Div([
        # Header with dropdown
        html.Div([
            html.H3('LNG Supply by Destination Continent - 30-Day Rolling Average', className="section-title-inline"),
            dcc.Dropdown(
                id='continent-chart-type',
                options=[
                    {'label': 'By Continent (mcm/d)', 'value': 'absolute'},
                    {'label': 'Market Share (%)', 'value': 'percentage'}
                ],
                value='absolute',
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'marginLeft': '20px', 'verticalAlign': 'middle'}
            )
        ], className="inline-section-header", style={'display': 'flex', 'alignItems': 'center'}),
        
        # Dynamic continent charts container - will be populated by callback
        dcc.Loading(
            id="continent-charts-loading",
            children=[
                html.Div(id='continent-charts-container')
            ],
            type="default",
        )
    ], className="main-section-container", style={'marginBottom': '30px'}),

    # LNG Supply by Destination Section
    html.Div([
        # Header with dropdown
        html.Div([
            html.H3('LNG Supply by Destination', className="section-title-inline"),
            dcc.Dropdown(
                id='supply-dest-view-type',
                options=[
                    {'label': 'By Destination (mcm/d)', 'value': 'absolute'},
                    {'label': 'Market Share (%)', 'value': 'percentage'}
                ],
                value='absolute',
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'marginLeft': '20px', 'verticalAlign': 'middle'}
            )
        ], className="inline-section-header", style={'display': 'flex', 'alignItems': 'center'}),
        
        # Supply-Destination Table
        html.Div([
            # Instructions at the top (will be updated dynamically)
            html.Div(id='supply-dest-table-instructions', style={'marginTop': '20px', 'marginBottom': '15px'}),
            
            dcc.Loading(
                id="supply-dest-table-loading",
                children=[
                    html.Div(id='supply-dest-table-container')
                ],
                type="default"
            )
        ], style={'marginBottom': '30px'})
    ], className="main-section-container", style={'marginBottom': '30px'}),

    # LNG Loadings Section
    html.Div([
        # Header
        html.Div([
            html.H3('LNG loadings on export countries and installations (mcm/d)', className="section-title-inline"),
        ], className="inline-section-header"),
        
        # Summary Table
        html.Div([
            # Instructions at the top (will be updated dynamically)
            html.Div(id='summary-table-instructions', style={'marginTop': '20px', 'marginBottom': '15px'}),
            
            dcc.Loading(
                id="summary-table-loading",
                children=[
                    html.Div(id='summary-table-container')
                ],
                type="default"
            )
        ], style={'marginBottom': '30px'})
    ], className="main-section-container")
])


# Callbacks

# Callback to update the classification mode store
@callback(
    Output('country-classification-mode', 'data'),
    Input('country-classification-dropdown', 'value'),
    prevent_initial_call=False
)
def update_classification_mode(value):
    """Store the selected classification mode"""
    return value

@callback(
    [Output('supply-charts-data', 'data'),
     Output('continent-charts-data', 'data'),
     Output('summary-data-store', 'data'),
     Output('supply-dest-data-store', 'data')],
    [Input('initial-load-trigger', 'n_intervals'),
     Input('country-classification-mode', 'data')],
    prevent_initial_call=False
)
def refresh_all_data(n_intervals, classification_mode):
    """Load all data from database"""
    try:
        # Fetch data
        engine_inst, schema = setup_database_connection()
        
        # Default to 'Country' if classification_mode is None
        if classification_mode is None:
            classification_mode = 'Country'
        
        # Dictionary to store all chart data
        charts_data = {}
        
        # Always fetch global supply data
        global_supply_df = fetch_global_supply_data(engine_inst, schema, classification_mode)
        charts_data['Global'] = global_supply_df.to_dict('records') if not global_supply_df.empty else []
        
        if classification_mode == 'Classification Level 1':
            # Get all classification groups
            classification_groups = get_all_classification_groups(engine_inst, schema)
            
            # Fetch data for each classification group
            for group in classification_groups:
                group_df = fetch_country_supply_data(engine_inst, schema, group, classification_mode)
                charts_data[group] = group_df.to_dict('records') if not group_df.empty else []
        else:
            # Fetch data for predefined countries
            countries = [
                "United States", "Australia", "Qatar", "Russian Federation",
                "Nigeria", "Angola", "Malaysia"
            ]
            
            for country in countries:
                country_df = fetch_country_supply_data(engine_inst, schema, country, classification_mode)
                charts_data[country] = country_df.to_dict('records') if not country_df.empty else []
        
        # Fetch summary table data (which internally uses fetch_installation_data)
        summary_df = fetch_summary_table_data(engine_inst, schema, classification_mode)
        
        # Fetch supply-destination data
        supply_dest_quarters, supply_dest_months, supply_dest_weeks = fetch_supply_destination_data(engine_inst, schema, classification_mode)
        
        # Combine supply-destination data similar to summary table
        supply_dest_df = fetch_supply_dest_summary_data(engine_inst, schema, classification_mode, 
                                                        supply_dest_quarters, supply_dest_months, supply_dest_weeks)
        
        # For continent charts, we use the same entities as the supply charts
        # The actual continent data will be fetched when the charts are created
        continent_charts_entities = list(charts_data.keys())
        
        return (charts_data,
                continent_charts_entities,  # Just pass the entity names for continent charts
                summary_df.to_dict('records') if not summary_df.empty else [],
                supply_dest_df.to_dict('records') if not supply_dest_df.empty else [])
                
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return {}, [], []


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
        
        # Check if we have is_forecast column
        if 'is_forecast' in year_data.columns:
            # Split data into historical and forecast
            historical_data = year_data[~year_data['is_forecast']]
            forecast_data = year_data[year_data['is_forecast']]
            
            # Plot historical data with solid line
            if not historical_data.empty:
                fig.add_trace(go.Scatter(
                    x=historical_data['plot_date'],
                    y=historical_data['rolling_avg'],
                    mode='lines',
                    name=str(int(year)) if pd.api.types.is_numeric_dtype(type(year)) else str(year),
                    line=dict(
                        color=colors[i % len(colors)],
                        width=3 if year == years[-1] else 2,  # Highlight current year
                        dash='solid'
                    ),
                    hovertemplate=f'<b>{int(year) if pd.api.types.is_numeric_dtype(type(year)) else year} (Historical)</b><br>' +
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
                    name=f"{int(year) if pd.api.types.is_numeric_dtype(type(year)) else year} (Forecast)",
                    line=dict(
                        color=forecast_color,  # Lighter/transparent version
                        width=3 if year == years[-1] else 2,  # Same width as historical
                        dash='solid'  # Solid line instead of dashed
                    ),
                    opacity=0.7,  # Additional opacity for visual distinction
                    hovertemplate=f'<b>{int(year) if pd.api.types.is_numeric_dtype(type(year)) else year} (Forecast)</b><br>' +
                                 '%{text}<br>' +
                                 'Supply: %{y:.1f} mcm/d<br>' +
                                 '<extra></extra>',
                    text=connect_data['month_day'],
                    showlegend=False  # Don't show separate legend entry for forecast
                ))
        else:
            # Old behavior if no forecast column (backward compatibility)
            fig.add_trace(go.Scatter(
                x=year_data['plot_date'],
                y=year_data['rolling_avg'],
                mode='lines',
                name=str(int(year)) if pd.api.types.is_numeric_dtype(type(year)) else str(year),
                line=dict(
                    color=colors[i % len(colors)],
                    width=3 if year == years[-1] else 2  # Highlight current year
                ),
                hovertemplate=f'<b>{int(year) if pd.api.types.is_numeric_dtype(type(year)) else year}</b><br>' +
                             '%{text}<br>' +
                             'Supply: %{y:.1f} mcm/d<br>' +
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
    return create_supply_chart(global_data, "Global", show_legend=False)  # Hide legend


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
    Output('russia-supply-chart', 'figure'),
    [Input('russia-supply-data-store', 'data')],
    prevent_initial_call=False
)
def update_russia_supply_chart(russia_data):
    """Create seasonal comparison chart of Russian Federation LNG supply"""
    return create_supply_chart(russia_data, "Russian Federation", show_legend=True)  # Show legend on this chart


@callback(
    Output('nigeria-supply-chart', 'figure'),
    [Input('nigeria-supply-data-store', 'data')],
    prevent_initial_call=False
)
def update_nigeria_supply_chart(nigeria_data):
    """Create seasonal comparison chart of Nigeria LNG supply"""
    return create_supply_chart(nigeria_data, "Nigeria", show_legend=False)  # Hide legend


@callback(
    Output('angola-supply-chart', 'figure'),
    [Input('angola-supply-data-store', 'data')],
    prevent_initial_call=False
)
def update_angola_supply_chart(angola_data):
    """Create seasonal comparison chart of Angola LNG supply"""
    return create_supply_chart(angola_data, "Angola", show_legend=False)  # Hide legend


@callback(
    Output('malaysia-supply-chart', 'figure'),
    [Input('malaysia-supply-data-store', 'data')],
    prevent_initial_call=False
)
def update_malaysia_supply_chart(malaysia_data):
    """Create seasonal comparison chart of Malaysia LNG supply"""
    return create_supply_chart(malaysia_data, "Malaysia", show_legend=False)  # Hide legend


@callback(
    [Output('summary-table-container', 'children'),
     Output('summary-table-instructions', 'children')],
    [Input('summary-data-store', 'data'),
     Input('summary-expanded-countries', 'data'),
     Input('summary-expanded-classifications', 'data'),
     Input('country-classification-mode', 'data')],
    prevent_initial_call=False
)
def update_summary_table(summary_data, expanded_countries, expanded_classifications, classification_mode):
    """Update the summary table with 5 quarters, 3 months, 3 weeks data with expandable rows"""
    
    # Set instruction text based on mode
    if classification_mode == 'Classification Level 1':
        instruction_text = "Click on ▶ classification row to expand and see countries and installations"
    else:
        instruction_text = "Click on ▶ country row to expand and see installations"
    
    instruction_element = html.Span(instruction_text, 
                                   style={'fontSize': '14px', 'color': '#666', 'fontStyle': 'italic'})
    
    if not summary_data:
        return (html.Div("No data available. Please refresh to load data.", 
                        style={'textAlign': 'center', 'padding': '20px'}),
                instruction_element)
    
    try:
        # Convert stored data back to DataFrame (same pattern as working tables)
        df = pd.DataFrame(summary_data)
        
        if df.empty:
            return (html.Div("No data available. Please refresh to load data.", 
                           style={'textAlign': 'center', 'padding': '20px'}),
                    instruction_element)
        
        # Initialize expanded states if None
        expanded_countries = expanded_countries or []
        expanded_classifications = expanded_classifications or []
        
        # Prepare data for display with expandable rows
        display_df, columns = prepare_table_for_display(df, 'summary', expanded_countries, 
                                                       classification_mode, expanded_classifications)
        
        if display_df.empty:
            return (html.Div("No data to display.", 
                           style={'textAlign': 'center', 'padding': '20px'}),
                    instruction_element)
        
        # Get conditional styles (includes alternating rows, country totals, grand total)
        conditional_styles = get_table_conditional_styles()
        
        # Add style for indented installations
        conditional_styles.insert(1, {
            'if': {'filter_query': '{Country} = ""'},
            'backgroundColor': '#f9f9f9',
            'fontSize': '13px'
        })
        
        # Add alignment styles
        conditional_styles.append({
            'if': {'column_id': 'Country'},
            'textAlign': 'left'
        })
        conditional_styles.append({
            'if': {'column_id': 'Installation'},
            'textAlign': 'left'
        })
        
        # Add right alignment for all numeric columns
        for col in display_df.columns:
            if col not in ['Country', 'Installation']:
                conditional_styles.append({
                    'if': {'column_id': col},
                    'textAlign': 'right',
                    'paddingRight': '12px'
                })
        
        # Find the last quarter column to add separator after it
        quarter_columns = [col for col in display_df.columns if col.startswith('Q') and "'" in col]
        last_quarter_col = quarter_columns[-1] if quarter_columns else None
        
        # Different background colors for different period types
        for col in display_df.columns:
            if col.startswith('Q') and "'" in col:  # Quarter columns
                conditional_styles.append({
                    'if': {'column_id': col, 'row_index': 0},
                    'backgroundColor': '#e3f2fd'  # Light blue for quarters
                })
                # Add left border to first quarter column for visual separation from Installation
                quarter_columns = [c for c in display_df.columns if c.startswith('Q') and "'" in c]
                if quarter_columns and col == quarter_columns[0]:
                    conditional_styles.append({
                        'if': {'column_id': col},
                        'borderLeft': '3px solid white'
                    })
            elif col.startswith('W') and "'" in col:  # Week columns
                conditional_styles.append({
                    'if': {'column_id': col, 'row_index': 0},
                    'backgroundColor': '#e8f5e9'  # Light green for weeks
                })
                # Add left border to first week column for visual separation from 30D
                week_columns = [c for c in display_df.columns if c.startswith('W') and "'" in c]
                if week_columns and col == week_columns[0]:
                    conditional_styles.append({
                        'if': {'column_id': col},
                        'borderLeft': '3px solid white'
                    })
            elif "'" in col and not col.startswith('Q') and not col.startswith('W') and col not in ['Country', 'Installation']:  # Month columns
                conditional_styles.append({
                    'if': {'column_id': col, 'row_index': 0},
                    'backgroundColor': '#f3e5f5'  # Light purple for months
                })
                # Add left border to first month column for visual separation
                month_columns = [c for c in display_df.columns if "'" in c and not c.startswith('Q') and not c.startswith('W') and c not in ['Country', 'Installation']]
                if month_columns and col == month_columns[0]:
                    conditional_styles.append({
                        'if': {'column_id': col},
                        'borderLeft': '3px solid white'
                    })
            elif col in ['30D', '7D']:  # Rolling window columns
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#fff3e0',  # Light orange for rolling windows
                    'fontWeight': '500'
                })
            elif col == 'Δ 7D-30D':  # Delta column
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#f5f5f5',  # Light gray background
                    'fontWeight': '600',
                    'borderLeft': '3px solid white'  # Add separator before first delta
                })
                # Add conditional formatting for positive values (green)
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': f'{{{col}}} > 0'
                    },
                    'color': '#2e7d32',  # Green for positive
                    'fontWeight': '600'
                })
                # Add conditional formatting for negative values (red)
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': f'{{{col}}} < 0'
                    },
                    'color': '#c62828',  # Red for negative
                    'fontWeight': '600'
                })
            elif col == 'Δ 30D Y/Y':  # Seasonal delta column
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#e8f5e9',  # Light green background for seasonal
                    'fontWeight': '600',
                    'borderLeft': '3px solid white'  # Add separator between the two deltas
                })
                # Add conditional formatting for positive values (dark green)
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': '{Δ 30D Y/Y} > 0'
                    },
                    'color': '#1b5e20',  # Dark green for positive Y/Y growth
                    'fontWeight': '600'
                })
                # Add conditional formatting for negative values (dark red)
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': '{Δ 30D Y/Y} < 0'
                    },
                    'color': '#b71c1c',  # Dark red for negative Y/Y growth
                    'fontWeight': '600'
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
        
        # Add header styles for the separators
        header_styles = []
        
        # Find first quarter column for header separator (after Installation)
        quarter_columns_for_header = [c for c in display_df.columns if c.startswith('Q') and "'" in c]
        if quarter_columns_for_header:
            first_quarter_col = quarter_columns_for_header[0]
            header_styles.append({
                'if': {'column_id': first_quarter_col},
                'borderLeft': '3px solid white'
            })
        
        # Find first month column for header separator
        month_columns_for_header = [c for c in display_df.columns if "'" in c and not c.startswith('Q') and not c.startswith('W') and c not in ['Country', 'Installation']]
        if month_columns_for_header:
            first_month_col = month_columns_for_header[0]
            header_styles.append({
                'if': {'column_id': first_month_col},
                'borderLeft': '3px solid white'
            })
        
        # Find first week column for header separator (after 30D)
        week_columns_for_header = [c for c in display_df.columns if c.startswith('W') and "'" in c]
        if week_columns_for_header:
            first_week_col = week_columns_for_header[0]
            header_styles.append({
                'if': {'column_id': first_week_col},
                'borderLeft': '3px solid white'
            })
        
        # Add separator before first delta column (Δ 7D-30D)
        if 'Δ 7D-30D' in display_df.columns:
            header_styles.append({
                'if': {'column_id': 'Δ 7D-30D'},
                'borderLeft': '3px solid white'
            })
        
        # Add separator before second delta column (Δ 30D Y/Y)
        if 'Δ 30D Y/Y' in display_df.columns:
            header_styles.append({
                'if': {'column_id': 'Δ 30D Y/Y'},
                'borderLeft': '3px solid white'
            })
        
        # Apply header styles
        if header_styles:
            table_config['style_header_conditional'] = header_styles
        
        # Create the DataTable with pattern matching ID for click handling
        table = dash_table.DataTable(
            id={'type': 'expandable-table', 'index': 'summary'},
            data=display_df.to_dict('records'),
            columns=columns,
            page_size=100,  # Increased to show expanded rows
            sort_mode='multi',
            fill_width=False,
            **table_config
        )
        
        # Calculate date ranges for footnote
        from datetime import datetime, timedelta
        today = datetime.now().date()
        date_7d_start = (today - timedelta(days=6)).strftime('%b %d, %Y')
        date_30d_start = (today - timedelta(days=29)).strftime('%b %d, %Y')
        date_today = today.strftime('%b %d, %Y')
        
        # Previous year 30D window dates
        date_30d_y1_start = (today - timedelta(days=365) - timedelta(days=29)).strftime('%b %d, %Y')
        date_30d_y1_end = (today - timedelta(days=365)).strftime('%b %d, %Y')
        
        # Create footnote
        footnote = html.Div([
            html.P([
                html.Span('Note: ', style={'fontWeight': 'bold'}),
                html.Span(f'30D: {date_30d_start} to {date_today} | ', style={'color': '#666'}),
                html.Span(f'7D: {date_7d_start} to {date_today} | ', style={'color': '#666'}),
                html.Span(f'30D Y-1: {date_30d_y1_start} to {date_30d_y1_end}', style={'color': '#666'})
            ], style={'fontSize': '12px', 'fontStyle': 'italic', 'marginTop': '10px', 'color': '#666'})
        ])
        
        # Return table with footnote and instruction
        return (html.Div([table, footnote]), instruction_element)
        
    except Exception as e:
        return (html.Div(f"Error creating summary table: {str(e)}", 
                        style={'textAlign': 'center', 'padding': '20px', 'color': 'red'}),
                instruction_element)


# Callback to update supply-destination table
@callback(
    [Output('supply-dest-table-container', 'children'),
     Output('supply-dest-table-instructions', 'children')],
    [Input('supply-dest-data-store', 'data'),
     Input('supply-dest-expanded-classifications', 'data'),
     Input('supply-dest-expanded-countries', 'data'),
     Input('supply-dest-expanded-supply-countries', 'data'),
     Input('country-classification-mode', 'data'),
     Input('supply-dest-view-type', 'value')],
    prevent_initial_call=False
)
def update_supply_dest_table(supply_dest_data, expanded_classifications, expanded_countries, 
                             expanded_supply_countries, classification_mode, view_type):
    """Update the supply-destination table with expandable rows"""
    
    # Set instruction text based on mode
    if classification_mode == 'Classification Level 1':
        instruction_text = "Click on ▶ to expand classification pairs, then demand countries, then supply countries"
    else:
        instruction_text = "Click on supply-demand pairs to see details"
    
    instruction_element = html.Span(instruction_text, 
                                   style={'fontSize': '14px', 'color': '#666', 'fontStyle': 'italic'})
    
    if not supply_dest_data:
        return (html.Div("No data available. Please refresh to load data.", 
                        style={'textAlign': 'center', 'padding': '20px'}),
                instruction_element)
    
    try:
        # Convert stored data back to DataFrame
        df = pd.DataFrame(supply_dest_data)
        
        if df.empty:
            return (html.Div("No data available. Please refresh to load data.", 
                           style={'textAlign': 'center', 'padding': '20px'}),
                    instruction_element)
        
        # Initialize expanded states if None
        expanded_classifications = expanded_classifications or []
        expanded_countries = expanded_countries or []
        expanded_supply_countries = expanded_supply_countries or []
        
        # Prepare data for display with expandable rows
        display_df, columns = prepare_supply_dest_table_for_display(
            df, 'summary', classification_mode,
            expanded_classifications, expanded_countries, expanded_supply_countries, view_type
        )
        
        if display_df.empty:
            return (html.Div("No data available", 
                           style={'textAlign': 'center', 'padding': '20px'}),
                    instruction_element)
        
        # Convert to percentages if requested
        if view_type == 'percentage' and classification_mode == 'Classification Level 1':
            # Identify numeric columns (exclude text columns and delta columns)
            text_cols = ['Aggregation Supply', 'Aggregation Demand', 'Country Demand', 'Supply Country']
            delta_cols = ['Δ 7D-30D', 'Δ 30D Y/Y']
            # Include 30D_Y1 in numeric columns for subtotal calculation
            numeric_cols = [col for col in display_df.columns if col not in text_cols and col not in delta_cols]
            # Also include 30D_Y1 if present for subtotal storage
            numeric_cols_with_y1 = numeric_cols + (['30D_Y1'] if '30D_Y1' in display_df.columns else [])
            
            # Store original values for delta calculation
            original_7d_values = display_df['7D'].copy() if '7D' in display_df.columns else None
            original_30d_values = display_df['30D'].copy() if '30D' in display_df.columns else None
            original_30d_y1_values = None
            
            # Calculate percentages relative to supply subtotals
            current_supply_class = None
            subtotal_values = {}
            
            # First pass: find all subtotal rows and store their values
            for idx, row in display_df.iterrows():
                # Subtotal rows have supply class name and 'Total' in Aggregation Demand
                if str(row.get('Aggregation Demand', '')) == 'Total' and row.get('Aggregation Supply', ''):
                    supply_name = str(row['Aggregation Supply'])
                    subtotal_values[supply_name] = {col: row[col] for col in numeric_cols_with_y1 if col in row}
            
            # Second pass: convert values to percentages (excluding delta columns)
            for idx, row in display_df.iterrows():
                agg_supply = str(row.get('Aggregation Supply', ''))
                agg_demand = str(row.get('Aggregation Demand', ''))
                
                # Determine which supply class this row belongs to
                if agg_supply:
                    # Extract supply class (remove expand/collapse indicators)
                    if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                        current_supply_class = agg_supply[2:].strip()
                    elif agg_demand == 'Total':
                        # This is a subtotal row
                        current_supply_class = agg_supply
                
                # Apply percentage calculation (excluding delta columns and 30D_Y1)
                if current_supply_class and current_supply_class in subtotal_values:
                    for col in numeric_cols:
                        if col in display_df.columns and col != '30D_Y1' and col in subtotal_values[current_supply_class]:
                            subtotal_val = subtotal_values[current_supply_class][col]
                            if subtotal_val != 0:
                                if agg_demand == 'Total' and agg_supply == current_supply_class:
                                    # Subtotal rows should show 100%
                                    display_df.at[idx, col] = 1.0
                                else:
                                    # Convert to percentage relative to subtotal
                                    display_df.at[idx, col] = (row[col] / subtotal_val)
            
            # Recalculate delta columns as percentage point differences
            if 'Δ 7D-30D' in display_df.columns and '7D' in display_df.columns and '30D' in display_df.columns:
                display_df['Δ 7D-30D'] = (display_df['7D'] - display_df['30D'])
            
            # For Y/Y comparison, calculate percentage point difference
            if 'Δ 30D Y/Y' in display_df.columns and '30D_Y1' in display_df.columns:
                # Convert 30D_Y1 to percentage using the same logic
                for idx, row in display_df.iterrows():
                    agg_supply = str(row.get('Aggregation Supply', ''))
                    agg_demand = str(row.get('Aggregation Demand', ''))
                    
                    # Determine supply class for this row
                    supply_class_for_row = None
                    if agg_supply:
                        if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                            supply_class_for_row = agg_supply[2:].strip()
                        elif agg_demand == 'Total':
                            supply_class_for_row = agg_supply
                    
                    # Convert 30D_Y1 to percentage and calculate delta
                    # Check if this is a subtotal row (has 'Total' in Aggregation Demand)
                    if agg_demand == 'Total':
                        # Subtotal rows should have Y/Y delta = 0 (100% - 100% = 0)
                        display_df.at[idx, 'Δ 30D Y/Y'] = 0
                    elif supply_class_for_row and supply_class_for_row in subtotal_values:
                        if '30D_Y1' in subtotal_values[supply_class_for_row]:
                            subtotal_y1 = subtotal_values[supply_class_for_row]['30D_Y1']
                            if subtotal_y1 != 0:
                                # Convert Y-1 value to percentage
                                y1_pct = row['30D_Y1'] / subtotal_y1
                                # Calculate percentage point difference
                                display_df.at[idx, 'Δ 30D Y/Y'] = display_df.at[idx, '30D'] - y1_pct
                            else:
                                display_df.at[idx, 'Δ 30D Y/Y'] = 0
                        else:
                            display_df.at[idx, 'Δ 30D Y/Y'] = 0
            
            # Drop 30D_Y1 column after using it for calculations
            if '30D_Y1' in display_df.columns:
                display_df = display_df.drop('30D_Y1', axis=1)
        
        elif view_type == 'percentage' and classification_mode != 'Classification Level 1':
            # For Country mode, calculate percentages relative to supply country subtotals
            text_cols = ['Supply Country', 'Demand Country']
            delta_cols = ['Δ 7D-30D', 'Δ 30D Y/Y']
            # Exclude delta columns from percentage conversion but include 30D_Y1 for subtotal storage
            numeric_cols = [col for col in display_df.columns if col not in text_cols and col not in delta_cols]
            numeric_cols_with_y1 = numeric_cols + (['30D_Y1'] if '30D_Y1' in display_df.columns else [])
            
            # Calculate percentages relative to supply country subtotals
            current_supply_country = None
            supply_subtotals = {}
            
            # First pass: find all subtotal rows and store their values
            for idx, row in display_df.iterrows():
                # Subtotal rows have 'Total' in Demand Country
                if str(row.get('Demand Country', '')) == 'Total' and row.get('Supply Country', ''):
                    supply_name = str(row['Supply Country'])
                    supply_subtotals[supply_name] = {col: row[col] for col in numeric_cols_with_y1 if col in row}
            
            # Second pass: convert values to percentages
            for idx, row in display_df.iterrows():
                supply_country = str(row.get('Supply Country', ''))
                demand_country = str(row.get('Demand Country', ''))
                
                # Use the supply country as the grouping key
                if supply_country:
                    current_supply_country = supply_country
                
                # Apply percentage calculation (excluding 30D_Y1)
                if current_supply_country and current_supply_country in supply_subtotals:
                    for col in numeric_cols:
                        if col in display_df.columns and col != '30D_Y1' and col in supply_subtotals[current_supply_country]:
                            subtotal_val = supply_subtotals[current_supply_country][col]
                            if subtotal_val != 0:
                                if demand_country == 'Total' and supply_country == current_supply_country:
                                    # Subtotal rows should show 100%
                                    display_df.at[idx, col] = 1.0
                                else:
                                    # Convert to percentage relative to subtotal
                                    display_df.at[idx, col] = (row[col] / subtotal_val)
            
            # Recalculate delta columns as percentage point differences
            if 'Δ 7D-30D' in display_df.columns and '7D' in display_df.columns and '30D' in display_df.columns:
                display_df['Δ 7D-30D'] = (display_df['7D'] - display_df['30D'])
            
            # For Y/Y comparison, calculate percentage point difference
            if 'Δ 30D Y/Y' in display_df.columns and '30D_Y1' in display_df.columns:
                # Convert 30D_Y1 to percentage using the same logic
                for idx, row in display_df.iterrows():
                    supply_country = str(row.get('Supply Country', ''))
                    demand_country = str(row.get('Demand Country', ''))
                    
                    # Convert 30D_Y1 to percentage and calculate delta
                    # Check if this is a subtotal row (has 'Total' in Demand Country)
                    if demand_country == 'Total':
                        # Subtotal rows should have Y/Y delta = 0 (100% - 100% = 0)
                        display_df.at[idx, 'Δ 30D Y/Y'] = 0
                    elif supply_country and supply_country in supply_subtotals:
                        if '30D_Y1' in supply_subtotals[supply_country]:
                            subtotal_y1 = supply_subtotals[supply_country]['30D_Y1']
                            if subtotal_y1 != 0:
                                # Convert Y-1 value to percentage
                                y1_pct = row['30D_Y1'] / subtotal_y1
                                # Calculate percentage point difference
                                display_df.at[idx, 'Δ 30D Y/Y'] = display_df.at[idx, '30D'] - y1_pct
                            else:
                                display_df.at[idx, 'Δ 30D Y/Y'] = 0
                        else:
                            display_df.at[idx, 'Δ 30D Y/Y'] = 0
            
            # Drop 30D_Y1 column after using it for calculations
            if '30D_Y1' in display_df.columns:
                display_df = display_df.drop('30D_Y1', axis=1)
        
        # Get conditional styles (includes alternating rows, country totals, grand total)
        conditional_styles = get_table_conditional_styles()
        
        # Add style for indented rows
        conditional_styles.insert(1, {
            'if': {'filter_query': '{Aggregation Supply} = ""'},
            'backgroundColor': '#f9f9f9',
            'fontSize': '13px'
        })
        
        # Add style for subtotal rows (bold format like Installation Total)
        # Works for both Classification Level 1 mode (Aggregation Demand = Total) and Country mode (Demand Country = Total)
        conditional_styles.append({
            'if': {'filter_query': '{Aggregation Demand} = "Total"'},
            'backgroundColor': TABLE_COLORS['bg_lighter'],
            'fontWeight': 'bold',
            'color': TABLE_COLORS['text_primary']
        })
        
        conditional_styles.append({
            'if': {'filter_query': '{Demand Country} = "Total"'},
            'backgroundColor': TABLE_COLORS['bg_lighter'],
            'fontWeight': 'bold',
            'color': TABLE_COLORS['text_primary']
        })
        
        # Add alignment styles for text columns
        text_columns = ['Aggregation Supply', 'Aggregation Demand', 'Country Demand', 'Supply Country', 'Demand Country']
        for col in text_columns:
            if col in display_df.columns:
                conditional_styles.append({
                    'if': {'column_id': col},
                    'textAlign': 'left'
                })
        
        # Add right alignment for all numeric columns
        for col in display_df.columns:
            if col not in text_columns:
                conditional_styles.append({
                    'if': {'column_id': col},
                    'textAlign': 'right',
                    'paddingRight': '12px'
                })
        
        # Find the last quarter column to add separator after it
        quarter_columns = [col for col in display_df.columns if col.startswith('Q') and "'" in col]
        last_quarter_col = quarter_columns[-1] if quarter_columns else None
        
        # Different background colors for different period types
        for col in display_df.columns:
            if col.startswith('Q') and "'" in col:  # Quarter columns
                conditional_styles.append({
                    'if': {'column_id': col, 'row_index': 0},
                    'backgroundColor': '#e3f2fd'  # Light blue for quarters
                })
                # Add left border to first quarter column for visual separation
                quarter_columns = [c for c in display_df.columns if c.startswith('Q') and "'" in c]
                if quarter_columns and col == quarter_columns[0]:
                    conditional_styles.append({
                        'if': {'column_id': col},
                        'borderLeft': '3px solid white'
                    })
            elif col.startswith('W') and "'" in col:  # Week columns
                conditional_styles.append({
                    'if': {'column_id': col, 'row_index': 0},
                    'backgroundColor': '#e8f5e9'  # Light green for weeks
                })
                # Add left border to first week column for visual separation
                week_columns = [c for c in display_df.columns if c.startswith('W') and "'" in c]
                if week_columns and col == week_columns[0]:
                    conditional_styles.append({
                        'if': {'column_id': col},
                        'borderLeft': '3px solid white'
                    })
            elif "'" in col and not col.startswith('Q') and not col.startswith('W') and col not in text_columns:  # Month columns
                conditional_styles.append({
                    'if': {'column_id': col, 'row_index': 0},
                    'backgroundColor': '#f3e5f5'  # Light purple for months
                })
                # Add left border to first month column for visual separation
                month_columns = [c for c in display_df.columns if "'" in c and not c.startswith('Q') and not c.startswith('W') and c not in text_columns]
                if month_columns and col == month_columns[0]:
                    conditional_styles.append({
                        'if': {'column_id': col},
                        'borderLeft': '3px solid white'
                    })
            elif col in ['30D', '7D']:  # Rolling window columns
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#fff3e0',  # Light orange for rolling windows
                    'fontWeight': '500'
                })
            elif col == 'Δ 7D-30D':  # Delta column
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#f5f5f5',  # Light gray background
                    'fontWeight': '600',
                    'borderLeft': '3px solid white'  # Add separator before first delta
                })
                # Add conditional formatting for positive values (green)
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': f'{{{col}}} > 0'
                    },
                    'color': '#2e7d32',  # Green for positive
                    'fontWeight': '600'
                })
                # Add conditional formatting for negative values (red)
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': f'{{{col}}} < 0'
                    },
                    'color': '#c62828',  # Red for negative
                    'fontWeight': '600'
                })
            elif col == 'Δ 30D Y/Y':  # Seasonal delta column
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#e8f5e9',  # Light green background for seasonal
                    'fontWeight': '600',
                    'borderLeft': '3px solid white'  # Add separator between the two deltas
                })
                # Add conditional formatting for positive values (dark green)
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': '{Δ 30D Y/Y} > 0'
                    },
                    'color': '#1b5e20',  # Dark green for positive Y/Y growth
                    'fontWeight': '600'
                })
                # Add conditional formatting for negative values (dark red)
                conditional_styles.append({
                    'if': {
                        'column_id': col,
                        'filter_query': '{Δ 30D Y/Y} < 0'
                    },
                    'color': '#b71c1c',  # Dark red for negative Y/Y growth
                    'fontWeight': '600'
                })
        
        # Re-add Grand Total style at the end to ensure highest priority
        conditional_styles.append({
            'if': {'filter_query': '{Aggregation Supply} = "GRAND TOTAL" or {Supply Country} = "GRAND TOTAL"'},
            'backgroundColor': '#2E86C1',  # McKinsey blue
            'fontWeight': 'bold',
            'color': 'white'
        })
        
        # Get base table configuration
        table_config = StandardTableStyleManager.get_base_datatable_config()
        table_config['style_data_conditional'] = conditional_styles
        
        # Add header styles for the separators
        header_styles = []
        
        # Find first quarter column for header separator
        quarter_columns_for_header = [c for c in display_df.columns if c.startswith('Q') and "'" in c]
        if quarter_columns_for_header:
            first_quarter_col = quarter_columns_for_header[0]
            header_styles.append({
                'if': {'column_id': first_quarter_col},
                'borderLeft': '3px solid white'
            })
        
        # Find first month column for header separator
        month_columns_for_header = [c for c in display_df.columns if "'" in c and not c.startswith('Q') and not c.startswith('W') and c not in text_columns]
        if month_columns_for_header:
            first_month_col = month_columns_for_header[0]
            header_styles.append({
                'if': {'column_id': first_month_col},
                'borderLeft': '3px solid white'
            })
        
        # Find first week column for header separator
        week_columns_for_header = [c for c in display_df.columns if c.startswith('W') and "'" in c]
        if week_columns_for_header:
            first_week_col = week_columns_for_header[0]
            header_styles.append({
                'if': {'column_id': first_week_col},
                'borderLeft': '3px solid white'
            })
        
        # Add separator before first delta column (Δ 7D-30D)
        if 'Δ 7D-30D' in display_df.columns:
            header_styles.append({
                'if': {'column_id': 'Δ 7D-30D'},
                'borderLeft': '3px solid white'
            })
        
        # Add separator before second delta column (Δ 30D Y/Y)
        if 'Δ 30D Y/Y' in display_df.columns:
            header_styles.append({
                'if': {'column_id': 'Δ 30D Y/Y'},
                'borderLeft': '3px solid white'
            })
        
        # Apply header styles
        if header_styles:
            table_config['style_header_conditional'] = header_styles
        
        # Create the DataTable with pattern matching ID for click handling
        table = dash_table.DataTable(
            id={'type': 'supply-dest-expandable-table', 'index': 'summary'},
            data=display_df.to_dict('records'),
            columns=columns,
            page_size=100,
            sort_mode='multi',
            fill_width=False,
            export_format='xlsx',
            export_headers='display',
            **table_config
        )
        
        # Calculate date ranges for footnote
        from datetime import datetime, timedelta
        today = datetime.now().date()
        date_7d_start = (today - timedelta(days=6)).strftime('%b %d, %Y')
        date_30d_start = (today - timedelta(days=29)).strftime('%b %d, %Y')
        date_today = today.strftime('%b %d, %Y')
        
        # Previous year 30D window dates
        date_30d_y1_start = (today - timedelta(days=365) - timedelta(days=29)).strftime('%b %d, %Y')
        date_30d_y1_end = (today - timedelta(days=365)).strftime('%b %d, %Y')
        
        # Create footnote
        footnote = html.Div([
            html.P([
                html.Span('Note: ', style={'fontWeight': 'bold'}),
                html.Span(f'30D: {date_30d_start} to {date_today} | ', style={'color': '#666'}),
                html.Span(f'7D: {date_7d_start} to {date_today} | ', style={'color': '#666'}),
                html.Span(f'30D Y-1: {date_30d_y1_start} to {date_30d_y1_end} | ', style={'color': '#666'}),
                html.Span(f'Values shown are bilateral trade flows in mcm/d', style={'color': '#666'})
            ], style={'fontSize': '12px', 'fontStyle': 'italic', 'marginTop': '10px', 'color': '#666'})
        ])
        
        # Return table with footnote and instruction
        return (html.Div([table, footnote]), instruction_element)
        
    except Exception as e:
        return (html.Div(f"Error creating supply-destination table: {str(e)}", 
                        style={'textAlign': 'center', 'padding': '20px', 'color': 'red'}),
                instruction_element)


# Callback to handle expanding/collapsing rows for summary table
@callback(
    [Output('summary-expanded-countries', 'data', allow_duplicate=True),
     Output('summary-expanded-classifications', 'data', allow_duplicate=True)],
    [Input({'type': 'expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'expandable-table', 'index': ALL}, 'data'),
     State('summary-expanded-countries', 'data'),
     State('summary-expanded-classifications', 'data'),
     State('summary-data-store', 'data'),
     State('country-classification-mode', 'data')],
    prevent_initial_call=True
)
def handle_row_expansion(active_cells, table_data_list, summary_expanded, summary_expanded_classifications, 
                        summary_data, classification_mode):
    """Handle clicking on rows to expand/collapse in summary table"""
    
    # Initialize if None
    summary_expanded = summary_expanded or []
    summary_expanded_classifications = summary_expanded_classifications or []
    
    # Check which table was clicked
    ctx = callback_context
    if not ctx.triggered:
        return summary_expanded, summary_expanded_classifications
    
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
                return summary_expanded, summary_expanded_classifications
            
            # Get the corresponding table data
            if table_type == 'summary' and summary_data:
                df = pd.DataFrame(summary_data)
                display_df, _ = prepare_table_for_display(df, 'summary', summary_expanded, 
                                                         classification_mode, summary_expanded_classifications)
                if active_cell['row'] < len(display_df):
                    clicked_row = display_df.iloc[active_cell['row']]
                    
                    if classification_mode == 'Classification Level 1':
                        # Check if it's a classification (in Aggregation column)
                        aggregation_col = clicked_row.get('Aggregation', '')
                        if aggregation_col and (aggregation_col.startswith('▶') or aggregation_col.startswith('▼')):
                            classification = aggregation_col[2:].strip()
                            if classification in summary_expanded_classifications:
                                summary_expanded_classifications.remove(classification)
                                # Also collapse all countries within this classification
                                summary_expanded = [c for c in summary_expanded if not c.startswith(f"{classification}_")]
                            else:
                                summary_expanded_classifications.append(classification)
                        
                        # Check if it's a country (in Country column with indentation)
                        country_col = clicked_row.get('Country', '')
                        if country_col and country_col.strip() and (
                            country_col.strip().startswith('▶') or country_col.strip().startswith('▼')):
                            # Extract country name and parent classification
                            country = country_col.strip()[2:].strip()
                            # Find parent classification by looking at previous rows
                            for i in range(active_cell['row'] - 1, -1, -1):
                                prev_row = display_df.iloc[i]
                                prev_aggregation = prev_row.get('Aggregation', '')
                                if prev_aggregation and (prev_aggregation.startswith('▶') or prev_aggregation.startswith('▼')):
                                    parent_classification = prev_aggregation[2:].strip()
                                    country_key = f"{parent_classification}_{country}"
                                    if country_key in summary_expanded:
                                        summary_expanded.remove(country_key)
                                    else:
                                        summary_expanded.append(country_key)
                                    break
                    else:
                        # Original two-level logic
                        country_col = clicked_row.get('Country', '')
                        if country_col and (country_col.startswith('▶') or country_col.startswith('▼')):
                            country = country_col[2:].strip()
                            if country in summary_expanded:
                                summary_expanded.remove(country)
                            else:
                                summary_expanded.append(country)
                            
        except Exception as e:
            pass
    
    return summary_expanded, summary_expanded_classifications


# Callback to handle expanding/collapsing rows for supply-destination table
@callback(
    [Output('supply-dest-expanded-classifications', 'data', allow_duplicate=True),
     Output('supply-dest-expanded-countries', 'data', allow_duplicate=True),
     Output('supply-dest-expanded-supply-countries', 'data', allow_duplicate=True)],
    [Input({'type': 'supply-dest-expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'supply-dest-expandable-table', 'index': ALL}, 'data'),
     State('supply-dest-expanded-classifications', 'data'),
     State('supply-dest-expanded-countries', 'data'),
     State('supply-dest-expanded-supply-countries', 'data'),
     State('supply-dest-data-store', 'data'),
     State('country-classification-mode', 'data')],
    prevent_initial_call=True
)
def handle_supply_dest_row_expansion(active_cells, table_data_list, expanded_classifications,
                                     expanded_countries, expanded_supply_countries,
                                     supply_dest_data, classification_mode):
    """Handle clicking on rows to expand/collapse in supply-destination table"""
    
    # Initialize if None
    expanded_classifications = expanded_classifications or []
    expanded_countries = expanded_countries or []
    expanded_supply_countries = expanded_supply_countries or []
    
    # Check which table was clicked
    ctx = callback_context
    if not ctx.triggered:
        return expanded_classifications, expanded_countries, expanded_supply_countries
    
    # Get the triggered input
    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    
    # Parse which table and what was clicked
    if 'supply-dest-expandable-table' in prop_id and '.active_cell' in prop_id:
        try:
            import json
            # Extract the ID part
            id_str = prop_id.split('.active_cell')[0]
            id_dict = json.loads(id_str)
            
            # Get the active cell value
            active_cell = triggered['value']
            if not active_cell:
                return expanded_classifications, expanded_countries, expanded_supply_countries
            
            # Get the corresponding table data
            if supply_dest_data:
                df = pd.DataFrame(supply_dest_data)
                display_df, _ = prepare_supply_dest_table_for_display(
                    df, 'summary', classification_mode,
                    expanded_classifications, expanded_countries, expanded_supply_countries, 'absolute'
                )
                
                if active_cell['row'] < len(display_df):
                    clicked_row = display_df.iloc[active_cell['row']]
                    
                    if classification_mode == 'Classification Level 1':
                        # Check if it's a classification pair (Aggregation Supply column)
                        supply_agg = clicked_row.get('Aggregation Supply', '')
                        demand_agg = clicked_row.get('Aggregation Demand', '')
                        
                        if supply_agg and (supply_agg.startswith('▶') or supply_agg.startswith('▼')):
                            # Extract classification names
                            supply_class = supply_agg[2:].strip()
                            pair_key = f"{supply_class}→{demand_agg}"
                            
                            if pair_key in expanded_classifications:
                                expanded_classifications.remove(pair_key)
                                # Also collapse all children
                                expanded_countries = [c for c in expanded_countries if not c.startswith(pair_key)]
                                expanded_supply_countries = [c for c in expanded_supply_countries if not c.startswith(pair_key)]
                            else:
                                expanded_classifications.append(pair_key)
                        
                        # Check if it's a demand country (Country Demand column)
                        country_demand = clicked_row.get('Country Demand', '')
                        if country_demand and country_demand.strip() and (
                            country_demand.strip().startswith('▶') or country_demand.strip().startswith('▼')):
                            # Extract country name
                            demand_country = country_demand.strip()[2:].strip()
                            
                            # Find parent classification pair
                            for i in range(active_cell['row'] - 1, -1, -1):
                                prev_row = display_df.iloc[i]
                                prev_supply = prev_row.get('Aggregation Supply', '')
                                prev_demand = prev_row.get('Aggregation Demand', '')
                                if prev_supply and (prev_supply.startswith('▶') or prev_supply.startswith('▼')):
                                    supply_class = prev_supply[2:].strip()
                                    pair_key = f"{supply_class}→{prev_demand}"
                                    country_key = f"{pair_key}→{demand_country}"
                                    
                                    if country_key in expanded_countries:
                                        expanded_countries.remove(country_key)
                                        # Also collapse supply countries under this
                                        expanded_supply_countries = [c for c in expanded_supply_countries if not c.startswith(country_key)]
                                    else:
                                        expanded_countries.append(country_key)
                                    break
                            
        except Exception as e:
            print(f"Error in supply-dest row expansion: {e}")
    
    return expanded_classifications, expanded_countries, expanded_supply_countries


# Callback to dynamically generate continent charts based on classification mode
@callback(
    Output('continent-charts-container', 'children'),
    [Input('continent-charts-data', 'data'),
     Input('continent-chart-type', 'value'),
     Input('country-classification-mode', 'data')],
    prevent_initial_call=False
)
def update_continent_charts(entities_list, chart_type, classification_mode):
    """Dynamically generate continent charts based on classification mode"""
    if not entities_list:
        return html.Div("No data available", style={'textAlign': 'center', 'padding': '20px'})
    
    # Setup database connection
    engine_inst, schema = setup_database_connection()
    
    charts = []
    
    # Determine the number of charts and calculate width
    num_charts = len(entities_list)
    if num_charts <= 4:
        chart_width = '25%'
    elif num_charts <= 8:
        chart_width = '25%'
    elif num_charts <= 12:
        chart_width = '20%'
    else:
        chart_width = '16.66%'  # 6 columns for many charts
    
    # Create chart for each entity (country or classification group)
    for entity_name in entities_list:
        # Create the figure based on chart type
        if chart_type == 'percentage':
            fig = create_continent_percentage_chart(entity_name, engine_inst, schema, classification_mode)
        else:
            fig = create_continent_destination_chart(entity_name, engine_inst, schema, classification_mode)
        
        # Create chart container
        chart_div = html.Div([
            html.H5(entity_name, style={'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '14px'}),
            dcc.Graph(
                id=f'continent-chart-{entity_name.replace(" ", "-").lower()}',
                figure=fig,
                style={'height': '350px'}
            )
        ], style={'width': chart_width, 'display': 'inline-block', 'padding': '0 10px', 'marginBottom': '20px'})
        
        charts.append(chart_div)
    
    # Wrap charts in a flex container
    return html.Div(
        charts,
        style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'flex-start'}
    )


# Callback to dynamically generate supply charts based on classification mode
@callback(
    Output('supply-charts-container', 'children'),
    [Input('supply-charts-data', 'data'),
     Input('country-classification-mode', 'data')],
    prevent_initial_call=False
)
def update_supply_charts(charts_data, classification_mode):
    """Dynamically generate supply charts based on classification mode"""
    if not charts_data:
        return html.Div("No data available", style={'textAlign': 'center', 'padding': '20px'})
    
    charts = []
    
    # Determine the number of charts and calculate width
    num_charts = len(charts_data)
    if num_charts <= 4:
        chart_width = '25%'
    elif num_charts <= 8:
        chart_width = '25%'
    elif num_charts <= 12:
        chart_width = '20%'
    else:
        chart_width = '16.66%'  # 6 columns for many charts
    
    # Create chart for each entity (country or classification group)
    for idx, (entity_name, entity_data) in enumerate(charts_data.items()):
        # Create the figure
        fig = create_supply_chart(entity_data, entity_name, show_legend=False)
        
        # Create chart container
        chart_div = html.Div([
            html.H5(entity_name, style={'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '14px'}),
            dcc.Graph(
                id=f'supply-chart-{entity_name.replace(" ", "-").lower()}',
                figure=fig,
                style={'height': '350px'}
            )
        ], style={'width': chart_width, 'display': 'inline-block', 'padding': '0 10px', 'marginBottom': '20px'})
        
        charts.append(chart_div)
    
    # Wrap charts in a flex container
    return html.Div(
        charts,
        style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'flex-start'}
    )