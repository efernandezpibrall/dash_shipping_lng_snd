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


def fetch_rolling_windows_data(engine, schema):
    """
    Fetch 7-day and 30-day rolling window data for each country and installation,
    including previous year data for seasonal comparison
    """
    EXCLUDED_COUNTRIES = [
        "Japan", "Belgium", "Brazil", "Chile", "China", "Equatorial Guinea",
        "France", "Greece", "India", "Netherlands", "Peru", "Singapore Republic",
        "South Korea", "Spain", "Portugal","United States Virgin Islands", "Germany", 
        "Finland", "Jamaica","Lithuania","Sweden","Turkey"
    ]
    
    try:
        with engine.connect() as conn:
            # Get data for current period and same period last year
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
                AND kt.origin_country_name NOT IN :excluded
                AND kt.installation_origin_name IS NOT NULL
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
            
            df = pd.read_sql(query, conn, params={"excluded": tuple(EXCLUDED_COUNTRIES)})
            
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
            
            # Filter for 7D window (current)
            df_7d = df[(df['flow_date'].dt.date > date_7d_ago) & 
                      (df['flow_date'].dt.date <= current_date)].copy()
            # Group and calculate daily average
            rolling_7d = df_7d.groupby(['origin_country_name', 'installation_origin_name'])['mcmd'].sum() / 7
            rolling_7d = rolling_7d.round(1).reset_index()
            rolling_7d.columns = ['origin_country_name', 'installation_origin_name', '7D']
            
            # Filter for 30D window (current)
            df_30d = df[(df['flow_date'].dt.date > date_30d_ago) & 
                       (df['flow_date'].dt.date <= current_date)].copy()
            # Group and calculate daily average
            rolling_30d = df_30d.groupby(['origin_country_name', 'installation_origin_name'])['mcmd'].sum() / 30
            rolling_30d = rolling_30d.round(1).reset_index()
            rolling_30d.columns = ['origin_country_name', 'installation_origin_name', '30D']
            
            # Filter for 30D window (previous year)
            df_30d_y1 = df[(df['flow_date'].dt.date > date_30d_y1_start) & 
                          (df['flow_date'].dt.date <= date_30d_y1_end)].copy()
            # Group and calculate daily average for previous year
            rolling_30d_y1 = df_30d_y1.groupby(['origin_country_name', 'installation_origin_name'])['mcmd'].sum() / 30
            rolling_30d_y1 = rolling_30d_y1.round(1).reset_index()
            rolling_30d_y1.columns = ['origin_country_name', 'installation_origin_name', '30D_Y1']
            
            # Merge all data
            result = rolling_7d.merge(
                rolling_30d,
                on=['origin_country_name', 'installation_origin_name'],
                how='outer'
            )
            result = result.merge(
                rolling_30d_y1,
                on=['origin_country_name', 'installation_origin_name'],
                how='outer'
            )
            
            # Add country totals
            country_totals_7d = result.groupby('origin_country_name')['7D'].sum().round(1)
            country_totals_30d = result.groupby('origin_country_name')['30D'].sum().round(1)
            country_totals_30d_y1 = result.groupby('origin_country_name')['30D_Y1'].sum().round(1)
            
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
            
            # Drop the 30D_Y1 column as we only need it for calculation
            final_result = final_result.drop('30D_Y1', axis=1)
            
            return final_result
            
    except Exception as e:
        print(f"Error fetching rolling windows data: {e}")
        return pd.DataFrame()


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


def fetch_summary_table_data(engine, schema):
    """
    Fetch data for summary table showing last 5 completed quarters, 3 completed months, 3 completed weeks,
    and 7D/30D rolling windows
    Returns data in the same format as quarters/months/weeks tables for consistency
    """
    try:
        # Fetch the existing data using the same function
        quarters_df, months_df, weeks_df = fetch_installation_data(engine, schema)
        
        if quarters_df.empty and months_df.empty and weeks_df.empty:
            return pd.DataFrame()
        
        # Also fetch 7D and 30D rolling data directly from database
        rolling_data = fetch_rolling_windows_data(engine, schema)
        
        if rolling_data.empty:
            # If no rolling data, create empty columns
            rolling_data = pd.DataFrame()
        
        # Get current date to determine what's complete
        current_date = datetime.now()
        current_quarter = (current_date.month - 1) // 3 + 1
        current_year = current_date.year
        
        # Get period columns - identify by pattern
        quarter_cols = [col for col in quarters_df.columns 
                       if col not in ['origin_country_name', 'installation_origin_name']]
        month_cols = [col for col in months_df.columns 
                     if col not in ['origin_country_name', 'installation_origin_name']]
        week_cols = [col for col in weeks_df.columns 
                    if col not in ['origin_country_name', 'installation_origin_name']]
        
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
        
        # Create subset dataframes with selected columns
        quarters_subset = quarters_df[['origin_country_name', 'installation_origin_name'] + selected_quarter_cols].copy()
        months_subset = months_df[['origin_country_name', 'installation_origin_name'] + selected_month_cols].copy()
        weeks_subset = weeks_df[['origin_country_name', 'installation_origin_name'] + selected_week_cols].copy()
        
        # Merge the dataframes in desired order
        # Start with quarters
        result = quarters_subset.copy()
        
        # Merge with months
        result = result.merge(
            months_subset,
            on=['origin_country_name', 'installation_origin_name'],
            how='outer'
        )
        
        # Now add 30D column right after months (before weeks)
        if not rolling_data.empty:
            # First merge just the 30D column
            result = result.merge(
                rolling_data[['origin_country_name', 'installation_origin_name', '30D']],
                on=['origin_country_name', 'installation_origin_name'],
                how='left'
            )
        else:
            result['30D'] = 0
        
        # Then merge with weeks
        result = result.merge(
            weeks_subset,
            on=['origin_country_name', 'installation_origin_name'],
            how='outer'
        )
        
        # Finally add the remaining rolling window columns
        if not rolling_data.empty:
            result = result.merge(
                rolling_data[['origin_country_name', 'installation_origin_name', '7D', 'Δ 7D-30D', 'Δ 30D Y/Y']],
                on=['origin_country_name', 'installation_origin_name'],
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


def fetch_global_supply_data(engine, schema):
    """Fetch daily global LNG supply data for seasonal chart"""
    
    try:
        with engine.connect() as conn:
            # Get daily aggregated data with rolling average calculated in SQL
            query = text(f"""
            WITH latest_data AS (
                SELECT MAX(upload_timestamp_utc) as max_timestamp
                FROM {schema}.kpler_trades
            ),
            daily_supply AS (
                SELECT 
                    kt.start::date as date,
                    SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as mcmd
                FROM {schema}.kpler_trades kt, latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start >= '2023-11-01'
                    AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                GROUP BY kt.start::date
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


def fetch_country_supply_data(engine, schema, country_name):
    """Fetch daily LNG supply data for a specific country"""
    
    try:
        with engine.connect() as conn:
            # Get daily aggregated data for specific country with rolling average calculated in SQL
            query = text(f"""
            WITH latest_data AS (
                SELECT MAX(upload_timestamp_utc) as max_timestamp
                FROM {schema}.kpler_trades
            ),
            daily_supply AS (
                SELECT 
                    kt.start::date as date,
                    SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as mcmd
                FROM {schema}.kpler_trades kt, latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.origin_country_name = :country
                    AND kt.start >= '2023-11-01'
                    AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                GROUP BY kt.start::date
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
    # Interval component to trigger initial data load (runs once on page load)
    dcc.Interval(id='initial-load-trigger', interval=1000*60*60*24, n_intervals=0, max_intervals=1),
    
    # Store components for caching data
    dcc.Store(id='global-supply-data-store', storage_type='local'),
    dcc.Store(id='us-supply-data-store', storage_type='local'),
    dcc.Store(id='australia-supply-data-store', storage_type='local'),
    dcc.Store(id='qatar-supply-data-store', storage_type='local'),
    dcc.Store(id='summary-data-store', storage_type='local'),
    
    # Store for expanded state of countries in summary table
    dcc.Store(id='summary-expanded-countries', data=[]),

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

    # LNG Loadings Section
    html.Div([
        # Header
        html.Div([
            html.H3('LNG loadings on export countries and installations (mcm/d)', className="section-title-inline"),
        ], className="inline-section-header"),
        
        # Summary Table
        html.Div([
            # Instructions at the top
            html.Div([
                html.Span("Click on ▶ country row to expand and see installations", 
                         style={'fontSize': '14px', 'color': '#666', 'fontStyle': 'italic'})
            ], style={'marginTop': '20px', 'marginBottom': '15px'}),
            
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
@callback(
    [Output('global-supply-data-store', 'data'),
     Output('us-supply-data-store', 'data'),
     Output('australia-supply-data-store', 'data'),
     Output('qatar-supply-data-store', 'data'),
     Output('summary-data-store', 'data')],
    [Input('initial-load-trigger', 'n_intervals')],
    prevent_initial_call=False
)
def refresh_all_data(n_intervals):
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
        
        # Fetch summary table data (which internally uses fetch_installation_data)
        summary_df = fetch_summary_table_data(engine_inst, schema)
        
        return (global_supply_df.to_dict('records') if not global_supply_df.empty else [],
                us_supply_df.to_dict('records') if not us_supply_df.empty else [],
                australia_supply_df.to_dict('records') if not australia_supply_df.empty else [],
                qatar_supply_df.to_dict('records') if not qatar_supply_df.empty else [],
                summary_df.to_dict('records') if not summary_df.empty else [])
        
    except Exception as e:
        return [], [], [], [], []


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
    Output('summary-table-container', 'children'),
    [Input('summary-data-store', 'data'),
     Input('summary-expanded-countries', 'data')],
    prevent_initial_call=False
)
def update_summary_table(summary_data, expanded_countries):
    """Update the summary table with 5 quarters, 3 months, 3 weeks data with expandable rows"""
    
    if not summary_data:
        return html.Div("No data available. Please refresh to load data.", 
                       style={'textAlign': 'center', 'padding': '20px'})
    
    try:
        # Convert stored data back to DataFrame (same pattern as working tables)
        df = pd.DataFrame(summary_data)
        
        if df.empty:
            return html.Div("No data available. Please refresh to load data.", 
                          style={'textAlign': 'center', 'padding': '20px'})
        
        # Initialize expanded countries if None
        expanded_countries = expanded_countries or []
        
        # Prepare data for display with expandable rows
        display_df, columns = prepare_table_for_display(df, 'summary', expanded_countries)
        
        if display_df.empty:
            return html.Div("No data to display.", 
                          style={'textAlign': 'center', 'padding': '20px'})
        
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
        
        # Return table with footnote
        return html.Div([table, footnote])
        
    except Exception as e:
        return html.Div(f"Error creating summary table: {str(e)}", 
                       style={'textAlign': 'center', 'padding': '20px', 'color': 'red'})


# Callback to handle expanding/collapsing rows for summary table
@callback(
    Output('summary-expanded-countries', 'data', allow_duplicate=True),
    [Input({'type': 'expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'expandable-table', 'index': ALL}, 'data'),
     State('summary-expanded-countries', 'data'),
     State('summary-data-store', 'data')],
    prevent_initial_call=True
)
def handle_row_expansion(active_cells, table_data_list, summary_expanded, summary_data):
    """Handle clicking on rows to expand/collapse in summary table"""
    
    # Initialize if None
    summary_expanded = summary_expanded or []
    
    # Check which table was clicked
    ctx = callback_context
    if not ctx.triggered:
        return summary_expanded
    
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
                return summary_expanded
            
            # Get the corresponding table data
            if table_type == 'summary' and summary_data:
                df = pd.DataFrame(summary_data)
                display_df, _ = prepare_table_for_display(df, 'summary', summary_expanded)
                if active_cell['row'] < len(display_df):
                    clicked_row = display_df.iloc[active_cell['row']]
                    country_col = clicked_row.get('Country', '')
                    if country_col and (country_col.startswith('▶') or country_col.startswith('▼')):
                        country = country_col[2:].strip()
                        if country in summary_expanded:
                            summary_expanded.remove(country)
                        else:
                            summary_expanded.append(country)
                            
        except Exception as e:
            pass
    
    return summary_expanded