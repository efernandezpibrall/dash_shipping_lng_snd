from dash import html, dcc, dash_table, callback, Output, Input, State, ALL, callback_context, no_update
from dash.dash_table.Format import Format, Scheme
import plotly.graph_objects as go
import pandas as pd
from copy import deepcopy
from datetime import datetime, timedelta
from io import BytesIO
import configparser
import os
from sqlalchemy import create_engine, text
import calendar

from utils.table_styles import StandardTableStyleManager, TABLE_COLORS

# Month order constant for sorting (used in multiple functions)
MONTH_ORDER = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
               'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

SUPPLY_DEST_PERIOD_VIEW_OPTIONS = [
    {'label': 'Monthly', 'value': 'monthly'},
    {'label': 'Quarterly', 'value': 'quarterly'},
    {'label': 'Seasonally', 'value': 'seasonally'},
    {'label': 'Yearly', 'value': 'yearly'}
]

SUPPLY_DEST_PERIOD_COUNT_OPTIONS = {
    'monthly': [12, 24, 36],
    'quarterly': [4, 8],
    'seasonally': [4, 6],
    'yearly': [3, 5]
}

SUPPLY_DEST_COMPARISON_BASIS_OPTIONS = [
    {'label': 'Levels', 'value': 'levels'},
    {'label': 'vs Previous Period', 'value': 'previous_period'},
    {'label': 'vs Same Period Last Year', 'value': 'same_period_last_year'}
]

SUPPLY_DEST_COUNTRY_GROUPING_OPTIONS = [
    {'label': 'Show all countries', 'value': 'show_all'},
    {'label': 'Group small countries', 'value': 'group_small_countries'}
]

SUPPLY_DEST_TEXT_COLUMNS = [
    'Aggregation Supply',
    'Aggregation Demand',
    'Country Demand',
    'Demand Country',
    'Supply Country'
]


def _empty_supply_dest_period_analysis_payload():
    """Return the default empty payload for the supply-destination period-analysis store."""
    return {
        key: {
            'records': [],
            'current_period_label': None,
            'current_period_previous_label': None,
            'current_period_previous_records': [],
            'current_period_prior_year_label': None,
            'current_period_prior_year_records': []
        }
        for key in ['monthly', 'quarterly', 'seasonally', 'yearly']
    }


def _empty_supply_dest_summary_store_payload():
    """Return the default store payload for grouped and ungrouped overview tables."""
    return {
        'show_all': [],
        'group_small_countries': []
    }


def _empty_supply_dest_period_store_payload():
    """Return the default store payload for grouped and ungrouped period-analysis views."""
    return {
        'show_all': _empty_supply_dest_period_analysis_payload(),
        'group_small_countries': _empty_supply_dest_period_analysis_payload()
    }


def _build_lng_season_periods(dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Match the seasonal period definition used on the capacity page."""
    normalized_dates = pd.to_datetime(dates, errors='coerce').dt.to_period('M').dt.to_timestamp()
    is_summer = normalized_dates.dt.month.between(4, 9)
    season_year = (
        normalized_dates.dt.year - normalized_dates.dt.month.isin([1, 2, 3]).astype(int)
    ).astype('Int64')

    season_start_month = pd.Series(10, index=normalized_dates.index, dtype='int64')
    season_start_month.loc[is_summer] = 4

    season_code = pd.Series('W', index=normalized_dates.index, dtype='object')
    season_code.loc[is_summer] = 'S'

    season_start = pd.to_datetime(
        {
            'year': season_year,
            'month': season_start_month,
            'day': 1
        },
        errors='coerce'
    )
    season_label = season_year.astype(str)
    season_label = season_label.where(normalized_dates.notna(), '')
    season_label = season_label + '-' + season_code.where(normalized_dates.notna(), '')
    return season_start, season_label


def _normalize_supply_dest_period_view(period_view):
    """Normalize the selected period view for supply-destination analysis."""
    if period_view in {'monthly', 'quarterly', 'seasonally', 'yearly'}:
        return period_view
    return None


def _normalize_supply_dest_comparison_basis(comparison_basis):
    """Normalize comparison basis while keeping the current table as the default state."""
    if comparison_basis in {'levels', 'previous_period', 'same_period_last_year'}:
        return comparison_basis
    return 'levels'


def _normalize_supply_dest_country_grouping(grouping_mode):
    """Normalize the small-country grouping mode for the period-analysis table."""
    if grouping_mode == 'group_small_countries':
        return 'group_small_countries'
    return 'show_all'


def _resolve_supply_dest_period_payload(period_payload, grouping_mode='show_all'):
    """Resolve the grouped or ungrouped period payload from the store data."""
    grouping_mode = _normalize_supply_dest_country_grouping(grouping_mode)
    if not isinstance(period_payload, dict):
        return _empty_supply_dest_period_analysis_payload()

    period_keys = {'monthly', 'quarterly', 'seasonally', 'yearly'}
    if period_keys.issubset(set(period_payload.keys())):
        return period_payload

    selected_payload = period_payload.get(grouping_mode)
    if isinstance(selected_payload, dict):
        return selected_payload

    fallback_payload = period_payload.get('show_all')
    if isinstance(fallback_payload, dict):
        return fallback_payload

    return _empty_supply_dest_period_analysis_payload()


def _resolve_supply_dest_summary_payload(summary_payload, grouping_mode='show_all'):
    """Resolve the grouped or ungrouped overview payload from the store data."""
    grouping_mode = _normalize_supply_dest_country_grouping(grouping_mode)
    if isinstance(summary_payload, list):
        return summary_payload
    if not isinstance(summary_payload, dict):
        return []

    selected_payload = summary_payload.get(grouping_mode)
    if isinstance(selected_payload, list):
        return selected_payload

    fallback_payload = summary_payload.get('show_all')
    if isinstance(fallback_payload, list):
        return fallback_payload

    return []


def _format_supply_dest_period_label(period_start, period_view):
    """Format the period label shown in historical supply-destination views."""
    if pd.isna(period_start):
        return ''

    period_start = pd.Timestamp(period_start)
    if period_view == 'monthly':
        return f"{calendar.month_abbr[period_start.month]}'{str(period_start.year)[2:]}"
    if period_view == 'quarterly':
        return f"Q{period_start.quarter}'{str(period_start.year)[2:]}"
    if period_view == 'yearly':
        return str(period_start.year)
    return ''


def _get_supply_dest_period_sort_key(label, period_view):
    """Return a sortable tuple for visible period labels."""
    if period_view == 'monthly':
        month_abbr, year_suffix = label.split("'")
        return (int(f"20{year_suffix}"), MONTH_ORDER.get(month_abbr, 0))
    if period_view == 'quarterly':
        quarter_part, year_suffix = label.split("'")
        return (int(f"20{year_suffix}"), int(quarter_part.replace('Q', '')))
    if period_view == 'seasonally':
        year_text, season_code = label.split('-')
        return (int(year_text), 0 if season_code == 'S' else 1)
    if period_view == 'yearly':
        return (int(label), 0)
    return (0, 0)


def _get_supply_dest_previous_period_label(label, period_view):
    """Return the immediately preceding completed period label."""
    if period_view == 'monthly':
        month_abbr, year_suffix = label.split("'")
        year = int(f"20{year_suffix}")
        month = MONTH_ORDER[month_abbr]
        previous = pd.Timestamp(year=year, month=month, day=1) - pd.offsets.MonthBegin(1)
        return _format_supply_dest_period_label(previous, 'monthly')
    if period_view == 'quarterly':
        quarter_part, year_suffix = label.split("'")
        year = int(f"20{year_suffix}")
        quarter = int(quarter_part.replace('Q', ''))
        if quarter == 1:
            return f"Q4'{str(year - 1)[2:]}"
        return f"Q{quarter - 1}'{str(year)[2:]}"
    if period_view == 'seasonally':
        year_text, season_code = label.split('-')
        year = int(year_text)
        if season_code == 'S':
            return f"{year - 1}-W"
        return f"{year}-S"
    if period_view == 'yearly':
        return str(int(label) - 1)
    return None


def _get_supply_dest_prior_year_label(label, period_view):
    """Return the same period one year earlier."""
    if period_view == 'monthly':
        month_abbr, year_suffix = label.split("'")
        return f"{month_abbr}'{str(int(f'20{year_suffix}') - 1)[2:]}"
    if period_view == 'quarterly':
        quarter_part, year_suffix = label.split("'")
        return f"{quarter_part}'{str(int(f'20{year_suffix}') - 1)[2:]}"
    if period_view == 'seasonally':
        year_text, season_code = label.split('-')
        return f"{int(year_text) - 1}-{season_code}"
    if period_view == 'yearly':
        return str(int(label) - 1)
    return None


def _get_supply_dest_period_options(period_view):
    """Return period count options for the selected time view."""
    values = SUPPLY_DEST_PERIOD_COUNT_OPTIONS.get(period_view, [])
    return [{'label': str(value), 'value': value} for value in values]


def safe_concat(dataframes, **kwargs):
    """Concatenate DataFrames, filtering out empty ones to avoid FutureWarning."""
    non_empty_dfs = [df for df in dataframes if not df.empty]
    if not non_empty_dfs:
        return pd.DataFrame()
    return pd.concat(non_empty_dfs, **kwargs)


def normalize_demand_aggregation_mode(demand_aggregation_mode):
    """Normalize demand aggregation mode so the dropdown can safely default to None."""
    if demand_aggregation_mode in (None, '', 'None'):
        return 'None'
    return demand_aggregation_mode


def use_demand_classification_mode(classification_mode, demand_aggregation_mode):
    """Demand aggregation is only applied when supply is already in Classification Level 1 mode."""
    return (
        classification_mode == 'Classification Level 1'
        and normalize_demand_aggregation_mode(demand_aggregation_mode) == 'Classification Level 1'
    )


def use_demand_country_mode(demand_aggregation_mode):
    """Country granularity is shown whenever demand aggregation is not None."""
    return normalize_demand_aggregation_mode(demand_aggregation_mode) != 'None'


def get_supply_dest_id_cols(classification_mode='Country', demand_aggregation_mode='None'):
    """Return the identifier columns used by the supply-destination table for the active mode."""
    if classification_mode == 'Classification Level 1':
        if use_demand_classification_mode(classification_mode, demand_aggregation_mode):
            return ['supply_classification', 'demand_classification', 'demand_country', 'supply_country']
        if use_demand_country_mode(demand_aggregation_mode):
            return ['supply_classification', 'demand_country', 'supply_country']
        return ['supply_classification', 'supply_country']
    if use_demand_country_mode(demand_aggregation_mode):
        return ['supply_country', 'demand_country']
    return ['supply_country']


def get_supply_dest_small_country_grouping_config(classification_mode='Country', demand_aggregation_mode='None'):
    """Return the visible country axis and its parent hierarchy for small-country grouping."""
    if classification_mode == 'Classification Level 1':
        if use_demand_classification_mode(classification_mode, demand_aggregation_mode):
            return 'demand_country', ['supply_classification', 'demand_classification']
        if use_demand_country_mode(demand_aggregation_mode):
            return 'demand_country', ['supply_classification']
        return 'supply_country', ['supply_classification']
    if use_demand_country_mode(demand_aggregation_mode):
        return 'demand_country', ['supply_country']
    return 'supply_country', []


def exclude_internal_destination_flows(df, classification_mode='Country',
                                       origin_country_col='supply_country',
                                       destination_country_col='demand_country',
                                       origin_classification_col='supply_classification',
                                       destination_classification_col='demand_classification'):
    """Exclude internal destination flows for the active page mode."""
    if df is None or df.empty:
        return df

    if classification_mode == 'Classification Level 1':
        if origin_classification_col not in df.columns or destination_classification_col not in df.columns:
            return df
        origin_values = df[origin_classification_col].fillna('Unknown').astype(str).str.strip()
        destination_values = df[destination_classification_col].fillna('Unknown').astype(str).str.strip()
        return df[origin_values != destination_values].copy()

    if origin_country_col not in df.columns or destination_country_col not in df.columns:
        return df

    origin_values = df[origin_country_col].fillna('Unknown').astype(str).str.strip()
    destination_values = df[destination_country_col].fillna('Unknown').astype(str).str.strip()
    return df[origin_values != destination_values].copy()


############################################ postgres sql connection ###################################################
#------ code to be able to access config.ini, even having the path in the .virtualenvs is not working without it ------#
try:
    # Get the directory where your script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the directory containing config.ini
    # Adjust the number of '..' as needed to reach the correct directory
    config_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Go up two levels
    CONFIG_FILE_PATH = os.path.join(config_dir, 'config.ini')
except Exception:
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


def get_latest_upload_timestamp(engine, schema):
    """Fetch the latest upload_timestamp_utc from kpler_trades table"""
    try:
        query = text(f"""
            SELECT MAX(upload_timestamp_utc) as max_timestamp
            FROM {schema}.kpler_trades
        """)

        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            if result and result[0]:
                return result[0]
            return None
    except Exception as e:
        return None


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
                LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                        IS DISTINCT FROM COALESCE(mc.country_classification_level1, 'Unknown')
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
                    AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                        IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
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
            final_result = safe_concat([result, country_totals], ignore_index=True)
            final_result = final_result.fillna(0)
            
            # Calculate deltas
            final_result['Δ 7D-30D'] = (final_result['7D'] - final_result['30D']).round(1)
            final_result['Δ 30D Y/Y'] = (final_result['30D'] - final_result['30D_Y1']).round(1)
            
            # Keep the 30D_Y1 column for percentage calculations
            # It will be hidden in the display
            
            return final_result
            
    except Exception as e:
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
                LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start >= CURRENT_DATE - INTERVAL '30 days'
                    AND kt.start <= CURRENT_DATE
                    AND kt.installation_origin_name IS NOT NULL
                    AND mc.country_classification_level1 IS NOT NULL
                    AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                        IS DISTINCT FROM COALESCE(mc.country_classification_level1, 'Unknown')
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


def _build_supply_dest_rolling_windows_from_df(df, classification_mode='Country', demand_aggregation_mode='None'):
    """Build supply-destination rolling windows from a bilateral flow dataframe."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = exclude_internal_destination_flows(
        df.copy(),
        classification_mode,
        origin_country_col='supply_country',
        destination_country_col='demand_country',
        origin_classification_col='supply_classification',
        destination_classification_col='demand_classification'
    )
    if df.empty:
        return pd.DataFrame()

    df['flow_date'] = pd.to_datetime(df['flow_date'])

    current_date = datetime.now().date()
    date_7d_ago = current_date - timedelta(days=7)
    date_30d_ago = current_date - timedelta(days=30)
    date_30d_y1_start = current_date - timedelta(days=365) - timedelta(days=30)
    date_30d_y1_end = current_date - timedelta(days=365)

    group_cols = get_supply_dest_id_cols(classification_mode, demand_aggregation_mode)

    df_7d = df[
        (df['flow_date'].dt.date > date_7d_ago)
        & (df['flow_date'].dt.date <= current_date)
    ].copy()
    rolling_7d = (df_7d.groupby(group_cols)['mcmd'].sum() / 7).round(1).reset_index()
    rolling_7d.columns = group_cols + ['7D']

    df_30d = df[
        (df['flow_date'].dt.date > date_30d_ago)
        & (df['flow_date'].dt.date <= current_date)
    ].copy()
    rolling_30d = (df_30d.groupby(group_cols)['mcmd'].sum() / 30).round(1).reset_index()
    rolling_30d.columns = group_cols + ['30D']

    df_30d_y1 = df[
        (df['flow_date'].dt.date > date_30d_y1_start)
        & (df['flow_date'].dt.date <= date_30d_y1_end)
    ].copy()
    rolling_30d_y1 = (df_30d_y1.groupby(group_cols)['mcmd'].sum() / 30).round(1).reset_index()
    rolling_30d_y1.columns = group_cols + ['30D_Y1']

    result = rolling_7d.merge(rolling_30d, on=group_cols, how='outer')
    result = result.merge(rolling_30d_y1, on=group_cols, how='outer')

    if classification_mode == 'Classification Level 1':
        if use_demand_classification_mode(classification_mode, demand_aggregation_mode):
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

            country_totals_7d = result.groupby(
                ['supply_classification', 'demand_classification', 'demand_country']
            )['7D'].sum().round(1)
            country_totals_30d = result.groupby(
                ['supply_classification', 'demand_classification', 'demand_country']
            )['30D'].sum().round(1)
            country_totals_30d_y1 = result.groupby(
                ['supply_classification', 'demand_classification', 'demand_country']
            )['30D_Y1'].sum().round(1)

            country_totals = pd.DataFrame({
                'supply_classification': country_totals_7d.index.get_level_values(0),
                'demand_classification': country_totals_7d.index.get_level_values(1),
                'demand_country': country_totals_7d.index.get_level_values(2),
                'supply_country': 'Total',
                '7D': country_totals_7d.values,
                '30D': country_totals_30d.values,
                '30D_Y1': country_totals_30d_y1.values
            })

            final_result = safe_concat([result, country_totals, class_totals], ignore_index=True)
        elif use_demand_country_mode(demand_aggregation_mode):
            class_totals_7d = result.groupby(['supply_classification'])['7D'].sum().round(1)
            class_totals_30d = result.groupby(['supply_classification'])['30D'].sum().round(1)
            class_totals_30d_y1 = result.groupby(['supply_classification'])['30D_Y1'].sum().round(1)

            class_totals = pd.DataFrame({
                'supply_classification': class_totals_7d.index,
                'demand_country': 'Total',
                'supply_country': 'Total',
                '7D': class_totals_7d.values,
                '30D': class_totals_30d.values,
                '30D_Y1': class_totals_30d_y1.values
            })

            country_totals_7d = result.groupby(['supply_classification', 'demand_country'])['7D'].sum().round(1)
            country_totals_30d = result.groupby(['supply_classification', 'demand_country'])['30D'].sum().round(1)
            country_totals_30d_y1 = result.groupby(['supply_classification', 'demand_country'])['30D_Y1'].sum().round(1)

            country_totals = pd.DataFrame({
                'supply_classification': country_totals_7d.index.get_level_values(0),
                'demand_country': country_totals_7d.index.get_level_values(1),
                'supply_country': 'Total',
                '7D': country_totals_7d.values,
                '30D': country_totals_30d.values,
                '30D_Y1': country_totals_30d_y1.values
            })

            final_result = safe_concat([result, country_totals, class_totals], ignore_index=True)
        else:
            class_totals_7d = result.groupby(['supply_classification'])['7D'].sum().round(1)
            class_totals_30d = result.groupby(['supply_classification'])['30D'].sum().round(1)
            class_totals_30d_y1 = result.groupby(['supply_classification'])['30D_Y1'].sum().round(1)

            class_totals = pd.DataFrame({
                'supply_classification': class_totals_7d.index,
                'demand_country': 'Total',
                'supply_country': 'Total',
                '7D': class_totals_7d.values,
                '30D': class_totals_30d.values,
                '30D_Y1': class_totals_30d_y1.values
            })
            final_result = safe_concat([result, class_totals], ignore_index=True)
    else:
        final_result = result

    final_result = final_result.fillna(0)
    final_result['Δ 7D-30D'] = (final_result['7D'] - final_result['30D']).round(1)
    final_result['Δ 30D Y/Y'] = (final_result['30D'] - final_result['30D_Y1']).round(1)
    return final_result


def fetch_supply_dest_rolling_windows(engine, schema, classification_mode='Country', demand_aggregation_mode='None'):
    """
    Fetch 7-day and 30-day rolling window data for supply-destination pairs,
    including previous year data for seasonal comparison
    
    Args:
        engine: Database engine
        schema: Database schema
        classification_mode: 'Country' or 'Classification Level 1'
        demand_aggregation_mode: 'None', 'Country', or 'Classification Level 1'
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

            return _build_supply_dest_rolling_windows_from_df(
                df,
                classification_mode,
                demand_aggregation_mode
            )
            
    except Exception as e:
        return pd.DataFrame()


def fetch_supply_dest_summary_data(engine, schema, classification_mode, demand_aggregation_mode,
                                   quarters_df, months_df, weeks_df,
                                   rolling_data=None, global_rolling_data=None):
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
        
        if rolling_data is None:
            rolling_data = fetch_supply_dest_rolling_windows(
                engine,
                schema,
                classification_mode,
                demand_aggregation_mode
            )

        if global_rolling_data is None:
            global_rolling_data = fetch_rolling_windows_data(engine, schema, classification_mode)
        
        id_cols = get_supply_dest_id_cols(classification_mode, demand_aggregation_mode)
        
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
        month_order = MONTH_ORDER
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
        result = result.sort_values(id_cols)
        
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
                    grand_total_payload = {
                        'supply_classification': 'GRAND TOTAL',
                        'supply_country': '',
                        **other_cols,
                        '30D': global_30d,
                        '7D': global_7d,
                        'Δ 7D-30D': global_delta_7d_30d,
                        'Δ 30D Y/Y': global_delta_30d_yy
                    }
                    if use_demand_classification_mode(classification_mode, demand_aggregation_mode):
                        grand_total_payload['demand_classification'] = ''
                        grand_total_payload['demand_country'] = ''
                    elif use_demand_country_mode(demand_aggregation_mode):
                        grand_total_payload['demand_country'] = ''
                    grand_total_row = pd.DataFrame([grand_total_payload])
                else:
                    grand_total_payload = {
                        'supply_country': 'GRAND TOTAL',
                        **other_cols,
                        '30D': global_30d,
                        '7D': global_7d,
                        'Δ 7D-30D': global_delta_7d_30d,
                        'Δ 30D Y/Y': global_delta_30d_yy
                    }
                    if use_demand_country_mode(demand_aggregation_mode):
                        grand_total_payload['demand_country'] = ''
                    grand_total_row = pd.DataFrame([grand_total_payload])
                
                # Append GRAND TOTAL row
                result = safe_concat([result, grand_total_row], ignore_index=True)
        
        return result
            
    except Exception as e:
        return pd.DataFrame()


def build_supply_dest_summary_store_payload(engine, schema, base_df, classification_mode='Country',
                                            demand_aggregation_mode='None', global_rolling_data=None):
    """Build grouped and ungrouped overview payloads for the supply-destination summary table."""
    if base_df is None or base_df.empty:
        return _empty_supply_dest_summary_store_payload()

    filtered_base_df = exclude_internal_destination_flows(
        base_df.copy(),
        classification_mode,
        origin_country_col='supply_country',
        destination_country_col='demand_country',
        origin_classification_col='supply_classification',
        destination_classification_col='demand_classification'
    )
    if filtered_base_df.empty:
        return _empty_supply_dest_summary_store_payload()

    if global_rolling_data is None:
        global_rolling_data = fetch_rolling_windows_data(engine, schema, classification_mode)

    def _build_payload_records(summary_base_df):
        quarters_df, months_df, weeks_df = fetch_supply_destination_data(
            engine,
            schema,
            classification_mode,
            demand_aggregation_mode,
            summary_base_df
        )
        rolling_df = _build_supply_dest_rolling_windows_from_df(
            summary_base_df,
            classification_mode,
            demand_aggregation_mode
        )
        summary_df = fetch_supply_dest_summary_data(
            engine,
            schema,
            classification_mode,
            demand_aggregation_mode,
            quarters_df,
            months_df,
            weeks_df,
            rolling_data=rolling_df,
            global_rolling_data=global_rolling_data
        )
        return summary_df.to_dict('records') if not summary_df.empty else []

    grouped_base_df = group_small_supply_dest_countries(
        filtered_base_df,
        classification_mode,
        demand_aggregation_mode
    )

    return {
        'show_all': _build_payload_records(filtered_base_df),
        'group_small_countries': _build_payload_records(grouped_base_df)
    }


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
        month_order = MONTH_ORDER
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
        return pd.DataFrame()


def fetch_global_supply_data(engine, schema, classification_mode='Country'):
    """Fetch daily global LNG supply data for seasonal chart
    
    Args:
        engine: Database engine
        schema: Database schema
        classification_mode: 'Country' or 'Classification Level 1'
    """
    
    try:
        with engine.connect() as conn:
            classification_join_clause = ""
            internal_flow_filter = """
                    AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                        IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
            """
            if classification_mode == 'Classification Level 1':
                classification_join_clause = f"""
                LEFT JOIN {schema}.mappings_country mc_origin ON kt.origin_country_name = mc_origin.country
                LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                """
                internal_flow_filter = """
                    AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                        IS DISTINCT FROM COALESCE(mc_origin.country_classification_level1, 'Unknown')
                """

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
                {classification_join_clause}
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start >= '2023-11-01'
                    {internal_flow_filter}
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
                {classification_join_clause}
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start >= '2023-11-01'
                    AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                    {internal_flow_filter}
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
                LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND mc.country_classification_level1 = :country
                    AND mc.country_classification_level1 IS NOT NULL
                    AND kt.start >= '2023-11-01'
                    AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                        IS DISTINCT FROM COALESCE(mc.country_classification_level1, 'Unknown')
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
                LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND mc.country_classification_level1 = :country
                    AND mc.country_classification_level1 IS NOT NULL
                    AND kt.start >= '2023-11-01'
                    AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                    AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                        IS DISTINCT FROM COALESCE(mc.country_classification_level1, 'Unknown')
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
                        AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                            IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
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
                        AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                            IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
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
                LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start IS NOT NULL
                    AND kt.start >= '2022-01-01'
                    AND kt.start <= CURRENT_DATE
                    AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                        IS DISTINCT FROM COALESCE(mc.country_classification_level1, 'Unknown')
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
                    AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                        IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
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


def fetch_supply_destination_base_data(engine, schema):
    """Fetch and normalize bilateral trade flow data used across supply-destination views."""
    try:
        with engine.connect() as conn:
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
            
            df = pd.read_sql(base_query, conn)
    
    except Exception as e:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Common data preparation
    df['supply_classification'] = df['supply_classification'].fillna('Unknown').astype(str).str.strip()
    df['supply_country'] = df['supply_country'].fillna('Unknown').astype(str).str.strip()
    df['demand_classification'] = df['demand_classification'].fillna('Unknown').astype(str).str.strip()
    df['demand_country'] = df['demand_country'].fillna('Unknown').astype(str).str.strip()
    
    df['mcmd'] = df['volume'] * 0.6 / 1000
    df['flow_date'] = pd.to_datetime(df['flow_date'])
    return df


def group_small_supply_dest_countries(df, classification_mode='Country',
                                      demand_aggregation_mode='None',
                                      threshold_mcmd=10, lookback_months=24):
    """Group small countries on the visible country axis into Rest of countries."""
    if df.empty:
        return df

    country_col, parent_cols = get_supply_dest_small_country_grouping_config(
        classification_mode,
        demand_aggregation_mode
    )
    if country_col not in df.columns:
        return df

    grouped_df = df.copy()
    current_timestamp = pd.Timestamp(datetime.now()).normalize()
    current_month = current_timestamp.to_period('M')
    start_month = current_month - (lookback_months - 1)

    lookback_df = grouped_df[
        grouped_df['flow_date'].dt.to_period('M') >= start_month
    ].copy()
    if lookback_df.empty:
        return grouped_df

    lookback_df['__month_period'] = lookback_df['flow_date'].dt.to_period('M')
    monthly_totals = (
        lookback_df.groupby(parent_cols + [country_col, '__month_period'], dropna=False)['mcmd']
        .sum()
        .reset_index()
    )
    if monthly_totals.empty:
        return grouped_df

    monthly_totals['__days'] = monthly_totals['__month_period'].apply(
        lambda month_period: (
            current_timestamp.day if month_period == current_month else month_period.days_in_month
        )
    )
    monthly_totals['__monthly_mcmd'] = (
        monthly_totals['mcmd'] / monthly_totals['__days']
    ).fillna(0)

    pair_cols = parent_cols + [country_col]
    max_monthly_by_pair = (
        monthly_totals.groupby(pair_cols, dropna=False)['__monthly_mcmd']
        .max()
        .reset_index()
    )
    all_pairs = grouped_df[pair_cols].drop_duplicates()
    pair_threshold_df = all_pairs.merge(max_monthly_by_pair, on=pair_cols, how='left')
    pair_threshold_df['__monthly_mcmd'] = pair_threshold_df['__monthly_mcmd'].fillna(0)
    small_pairs = pair_threshold_df[pair_threshold_df['__monthly_mcmd'] <= threshold_mcmd][pair_cols].copy()

    if small_pairs.empty:
        return grouped_df

    small_pairs['__group_small_country'] = True
    grouped_df = grouped_df.merge(small_pairs, on=pair_cols, how='left')
    grouped_df['__group_small_country'] = grouped_df['__group_small_country'].eq(True)
    grouped_df.loc[
        grouped_df['__group_small_country'],
        country_col
    ] = 'Rest of countries'
    grouped_df = grouped_df.drop(columns='__group_small_country')
    return grouped_df


def _apply_supply_dest_period_totals(result, classification_mode='Country', demand_aggregation_mode='None'):
    """Add hierarchy subtotal rows required by the expandable supply-destination table."""
    if result.empty:
        return result

    numeric_cols = [
        col for col in result.columns
        if col not in ['supply_classification', 'demand_classification', 'supply_country', 'demand_country']
    ]

    if classification_mode == 'Classification Level 1':
        if use_demand_classification_mode(classification_mode, demand_aggregation_mode):
            class_totals = result.groupby(
                ['supply_classification', 'demand_classification']
            )[numeric_cols].sum().round(1).reset_index()
            class_totals['demand_country'] = 'Total'
            class_totals['supply_country'] = 'Total'

            country_totals = result.groupby(
                ['supply_classification', 'demand_classification', 'demand_country']
            )[numeric_cols].sum().round(1).reset_index()
            country_totals['supply_country'] = 'Total'

            final_df = safe_concat([result, country_totals, class_totals], ignore_index=True)
            return final_df.sort_values(
                ['supply_classification', 'demand_classification', 'demand_country', 'supply_country']
            ).reset_index(drop=True)

        if use_demand_country_mode(demand_aggregation_mode):
            class_totals = result.groupby(['supply_classification'])[numeric_cols].sum().round(1).reset_index()
            class_totals['demand_country'] = 'Total'
            class_totals['supply_country'] = 'Total'

            country_totals = result.groupby(
                ['supply_classification', 'demand_country']
            )[numeric_cols].sum().round(1).reset_index()
            country_totals['supply_country'] = 'Total'

            final_df = safe_concat([result, country_totals, class_totals], ignore_index=True)
            return final_df.sort_values(
                ['supply_classification', 'demand_country', 'supply_country']
            ).reset_index(drop=True)

        class_totals = result.groupby(['supply_classification'])[numeric_cols].sum().round(1).reset_index()
        class_totals['supply_country'] = 'Total'
        final_df = safe_concat([result, class_totals], ignore_index=True)
        return final_df.sort_values(['supply_classification', 'supply_country']).reset_index(drop=True)

    return result.reset_index(drop=True)


def _get_supply_dest_period_days(period_start, period_view):
    """Return the number of days in a completed reporting period."""
    period_start = pd.Timestamp(period_start)

    if period_view == 'monthly':
        return period_start.days_in_month
    if period_view == 'quarterly':
        period_end = period_start.to_period('Q').end_time.normalize()
        return (period_end - period_start).days + 1
    if period_view == 'seasonally':
        if period_start.month == 4:
            period_end = pd.Timestamp(year=period_start.year, month=9, day=30)
        else:
            period_end = pd.Timestamp(year=period_start.year + 1, month=3, day=31)
        return (period_end - period_start).days + 1
    if period_view == 'yearly':
        period_end = period_start.to_period('Y').end_time.normalize()
        return (period_end - period_start).days + 1
    return 1


def _get_supply_dest_elapsed_period_days(period_start, current_date):
    """Return elapsed days from the start of the current reporting period to today."""
    period_start = pd.Timestamp(period_start).normalize()
    current_timestamp = pd.Timestamp(current_date).normalize()
    if current_timestamp < period_start:
        return 0
    return (current_timestamp - period_start).days + 1


def _get_supply_dest_current_period_details(current_date, period_view):
    """Return the current timestamp, period start, and display label for the active period."""
    current_timestamp = pd.Timestamp(current_date).normalize()

    if period_view == 'monthly':
        current_period_start = current_timestamp.to_period('M').start_time
        current_period_label = _format_supply_dest_period_label(current_period_start, period_view)
    elif period_view == 'quarterly':
        current_period_start = current_timestamp.to_period('Q').start_time
        current_period_label = _format_supply_dest_period_label(current_period_start, period_view)
    elif period_view == 'seasonally':
        current_period_start, current_period_label = _build_lng_season_periods(pd.Series([current_timestamp]))
        current_period_start = current_period_start.iloc[0]
        current_period_label = current_period_label.iloc[0]
    elif period_view == 'yearly':
        current_period_start = current_timestamp.to_period('Y').start_time
        current_period_label = _format_supply_dest_period_label(current_period_start, period_view)
    else:
        return current_timestamp, None, None

    return current_timestamp, current_period_start, current_period_label


def _get_supply_dest_previous_period_start(period_start, period_view):
    """Return the start timestamp for the immediately previous reporting period."""
    period_start = pd.Timestamp(period_start).normalize()

    if period_view == 'monthly':
        return (period_start.to_period('M') - 1).start_time
    if period_view == 'quarterly':
        return (period_start.to_period('Q') - 1).start_time
    if period_view == 'seasonally':
        if period_start.month == 4:
            return pd.Timestamp(year=period_start.year - 1, month=10, day=1)
        return pd.Timestamp(year=period_start.year, month=4, day=1)
    if period_view == 'yearly':
        return (period_start.to_period('Y') - 1).start_time
    return None


def _build_supply_dest_current_period_previous_reference(df, current_date, period_view,
                                                         classification_mode='Country',
                                                         demand_aggregation_mode='None'):
    """Build the previous-period-to-date reference for the current active period."""
    if df.empty:
        return pd.DataFrame(), None

    current_timestamp, current_period_start, current_period_label = _get_supply_dest_current_period_details(
        current_date,
        period_view
    )
    if current_period_start is None or not current_period_label:
        return pd.DataFrame(), None

    previous_period_label = _get_supply_dest_previous_period_label(current_period_label, period_view)
    previous_period_start = _get_supply_dest_previous_period_start(current_period_start, period_view)
    if not previous_period_label or previous_period_start is None:
        return pd.DataFrame(), None

    elapsed_days = _get_supply_dest_elapsed_period_days(current_period_start, current_timestamp)
    if elapsed_days <= 0:
        return pd.DataFrame(), None

    previous_period_end = previous_period_start + pd.Timedelta(days=elapsed_days - 1)
    reference_label = f"{previous_period_label} PTD"
    id_cols = get_supply_dest_id_cols(classification_mode, demand_aggregation_mode)

    reference_df = df[
        (df['flow_date'] >= previous_period_start)
        & (df['flow_date'] <= previous_period_end)
    ].copy()

    if reference_df.empty:
        return pd.DataFrame(columns=id_cols + [reference_label]), reference_label

    grouped_df = (
        reference_df.groupby(id_cols, dropna=False)['mcmd']
        .sum()
        .reset_index()
    )
    grouped_df[reference_label] = (grouped_df['mcmd'] / elapsed_days).round(1)
    grouped_df = grouped_df.drop(columns=['mcmd'])
    grouped_df = _apply_supply_dest_period_totals(
        grouped_df,
        classification_mode,
        demand_aggregation_mode
    )
    return grouped_df, reference_label


def _build_supply_dest_current_period_prior_year_reference(df, current_date, period_view,
                                                           classification_mode='Country',
                                                           demand_aggregation_mode='None'):
    """Build the same-period-last-year-to-date reference for the current active period."""
    if df.empty:
        return pd.DataFrame(), None

    current_timestamp, current_period_start, current_period_label = _get_supply_dest_current_period_details(
        current_date,
        period_view
    )
    if current_period_start is None or not current_period_label:
        return pd.DataFrame(), None

    prior_year_label = _get_supply_dest_prior_year_label(current_period_label, period_view)
    if not prior_year_label:
        return pd.DataFrame(), None

    elapsed_days = _get_supply_dest_elapsed_period_days(current_period_start, current_timestamp)
    if elapsed_days <= 0:
        return pd.DataFrame(), None

    prior_year_start = pd.Timestamp(current_period_start) - pd.DateOffset(years=1)
    prior_year_end = prior_year_start + pd.Timedelta(days=elapsed_days - 1)
    reference_label = f"{prior_year_label} PTD"
    id_cols = get_supply_dest_id_cols(classification_mode, demand_aggregation_mode)

    reference_df = df[
        (df['flow_date'] >= prior_year_start)
        & (df['flow_date'] <= prior_year_end)
    ].copy()

    if reference_df.empty:
        return pd.DataFrame(columns=id_cols + [reference_label]), reference_label

    grouped_df = (
        reference_df.groupby(id_cols, dropna=False)['mcmd']
        .sum()
        .reset_index()
    )
    grouped_df[reference_label] = (grouped_df['mcmd'] / elapsed_days).round(1)
    grouped_df = grouped_df.drop(columns=['mcmd'])
    grouped_df = _apply_supply_dest_period_totals(
        grouped_df,
        classification_mode,
        demand_aggregation_mode
    )
    return grouped_df, reference_label


def _build_supply_dest_period_matrix(df, current_date, period_view='monthly',
                                     classification_mode='Country', demand_aggregation_mode='None',
                                     include_current_period=False):
    """Build a period matrix for monthly, quarterly, seasonal, or yearly views."""
    if df.empty:
        return pd.DataFrame()

    period_df = df.copy()
    current_timestamp, current_period_start, current_period_label = _get_supply_dest_current_period_details(
        current_date,
        period_view
    )

    if current_period_start is None:
        return pd.DataFrame()

    if period_view == 'monthly':
        period_df['__period_start'] = period_df['flow_date'].dt.to_period('M').dt.start_time
    elif period_view == 'quarterly':
        period_df['__period_start'] = period_df['flow_date'].dt.to_period('Q').dt.start_time
    elif period_view == 'seasonally':
        period_df['__period_start'], period_df['__period_label'] = _build_lng_season_periods(period_df['flow_date'])
    elif period_view == 'yearly':
        period_df['__period_start'] = period_df['flow_date'].dt.to_period('Y').dt.start_time

    period_df = period_df[
        period_df['__period_start'].notna()
        & (
            period_df['__period_start'] <= current_period_start
            if include_current_period
            else period_df['__period_start'] < current_period_start
        )
    ].copy()

    if period_df.empty:
        return pd.DataFrame()

    pivot = period_df.pivot_table(
        index=get_supply_dest_id_cols(classification_mode, demand_aggregation_mode),
        columns='__period_start',
        values='mcmd',
        aggfunc='sum',
        fill_value=0
    )

    if include_current_period and current_period_start not in pivot.columns:
        pivot[current_period_start] = 0

    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    for col in pivot.columns:
        if include_current_period and pd.Timestamp(col) == current_period_start:
            days = _get_supply_dest_elapsed_period_days(col, current_timestamp)
        else:
            days = _get_supply_dest_period_days(col, period_view)
        pivot[col] = (pivot[col] / days).round(1) if days else 0

    if period_view == 'seasonally':
        label_map = (
            period_df[['__period_start', '__period_label']]
            .drop_duplicates(subset=['__period_start'])
            .set_index('__period_start')['__period_label']
            .to_dict()
        )
        if include_current_period and current_period_label:
            label_map[pd.Timestamp(current_period_start)] = current_period_label
        pivot.columns = [label_map.get(pd.Timestamp(col), '') for col in pivot.columns]
    else:
        pivot.columns = [_format_supply_dest_period_label(col, period_view) for col in pivot.columns]

    result = pivot.reset_index()
    result = _apply_supply_dest_period_totals(result, classification_mode, demand_aggregation_mode)
    result.attrs['current_period_label'] = current_period_label if include_current_period else None
    return result


def build_supply_dest_period_analysis_payload(df, classification_mode='Country', demand_aggregation_mode='None'):
    """Precompute period datasets for the historical analysis controls."""
    if df.empty:
        return _empty_supply_dest_period_analysis_payload()

    current_date = datetime.now()
    payload = {}
    for period_view in ['monthly', 'quarterly', 'seasonally', 'yearly']:
        period_df = _build_supply_dest_period_matrix(
            df,
            current_date,
            period_view,
            classification_mode,
            demand_aggregation_mode,
            include_current_period=True
        )
        current_previous_df, current_previous_label = _build_supply_dest_current_period_previous_reference(
            df,
            current_date,
            period_view,
            classification_mode,
            demand_aggregation_mode
        )
        current_prior_year_df, current_prior_year_label = _build_supply_dest_current_period_prior_year_reference(
            df,
            current_date,
            period_view,
            classification_mode,
            demand_aggregation_mode
        )
        payload[period_view] = {
            'records': period_df.to_dict('records') if not period_df.empty else [],
            'current_period_label': period_df.attrs.get('current_period_label'),
            'current_period_previous_label': current_previous_label,
            'current_period_previous_records': (
                current_previous_df.to_dict('records') if not current_previous_df.empty else []
            ),
            'current_period_prior_year_label': current_prior_year_label,
            'current_period_prior_year_records': (
                current_prior_year_df.to_dict('records') if not current_prior_year_df.empty else []
            )
        }
    return payload


def build_supply_dest_period_store_payload(df, classification_mode='Country', demand_aggregation_mode='None'):
    """Build both grouped and ungrouped payloads for the period-analysis table."""
    df = exclude_internal_destination_flows(
        df,
        classification_mode,
        origin_country_col='supply_country',
        destination_country_col='demand_country',
        origin_classification_col='supply_classification',
        destination_classification_col='demand_classification'
    )
    if df.empty:
        return _empty_supply_dest_period_store_payload()

    grouped_df = group_small_supply_dest_countries(
        df,
        classification_mode,
        demand_aggregation_mode
    )
    return {
        'show_all': build_supply_dest_period_analysis_payload(
            df,
            classification_mode,
            demand_aggregation_mode
        ),
        'group_small_countries': build_supply_dest_period_analysis_payload(
            grouped_df,
            classification_mode,
            demand_aggregation_mode
        )
    }


def fetch_supply_destination_data(engine, schema, classification_mode='Country',
                                  demand_aggregation_mode='None', base_df=None):
    """Fetch bilateral trade flow data for the default supply-destination table."""
    current_date = datetime.now()
    df = base_df.copy() if base_df is not None else fetch_supply_destination_base_data(engine, schema)
    df = exclude_internal_destination_flows(
        df,
        classification_mode,
        origin_country_col='supply_country',
        destination_country_col='demand_country',
        origin_classification_col='supply_classification',
        destination_classification_col='demand_classification'
    )

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    quarters_df = process_supply_dest_quarters(df, current_date, classification_mode, demand_aggregation_mode)
    months_df = process_supply_dest_months(df, current_date, classification_mode, demand_aggregation_mode)
    weeks_df = process_supply_dest_weeks(df, current_date, classification_mode, demand_aggregation_mode)
    
    return quarters_df, months_df, weeks_df


def process_supply_dest_quarters(df, current_date, classification_mode='Country', demand_aggregation_mode='None'):
    """Process supply-destination data for quarters"""
    return _build_supply_dest_period_matrix(
        df,
        current_date,
        'quarterly',
        classification_mode,
        demand_aggregation_mode
    )


def process_supply_dest_months(df, current_date, classification_mode='Country', demand_aggregation_mode='None'):
    """Process supply-destination data for months"""
    return _build_supply_dest_period_matrix(
        df,
        current_date,
        'monthly',
        classification_mode,
        demand_aggregation_mode
    )


def process_supply_dest_weeks(df, current_date, classification_mode='Country', demand_aggregation_mode='None'):
    """Process supply-destination data for weeks"""
    df = df.copy()
    
    # Create week period
    df['period'] = df['flow_date'].dt.to_period('W')
    
    # Filter to last 12 weeks
    current_week = pd.Period(current_date, freq='W')
    start_week = current_week - 11
    df_filtered = df[(df['period'] >= start_week) & (df['period'] <= current_week)]
    
    pivot = df_filtered.pivot_table(
        index=get_supply_dest_id_cols(classification_mode, demand_aggregation_mode),
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
        
        if use_demand_classification_mode(classification_mode, demand_aggregation_mode):
            class_totals = result.groupby(['supply_classification', 'demand_classification'])[numeric_cols].sum().round(1).reset_index()
            class_totals['demand_country'] = 'Total'
            class_totals['supply_country'] = 'Total'
            
            country_totals = result.groupby(['supply_classification', 'demand_classification', 'demand_country'])[numeric_cols].sum().round(1).reset_index()
            country_totals['supply_country'] = 'Total'
            
            final_df = safe_concat([result, country_totals, class_totals], ignore_index=True)
            final_df = final_df.sort_values(['supply_classification', 'demand_classification', 'demand_country', 'supply_country']).reset_index(drop=True)
        elif use_demand_country_mode(demand_aggregation_mode):
            class_totals = result.groupby(['supply_classification'])[numeric_cols].sum().round(1).reset_index()
            class_totals['demand_country'] = 'Total'
            class_totals['supply_country'] = 'Total'
            
            country_totals = result.groupby(['supply_classification', 'demand_country'])[numeric_cols].sum().round(1).reset_index()
            country_totals['supply_country'] = 'Total'
            
            final_df = safe_concat([result, country_totals, class_totals], ignore_index=True)
            final_df = final_df.sort_values(['supply_classification', 'demand_country', 'supply_country']).reset_index(drop=True)
        else:
            class_totals = result.groupby(['supply_classification'])[numeric_cols].sum().round(1).reset_index()
            class_totals['supply_country'] = 'Total'
            
            final_df = safe_concat([result, class_totals], ignore_index=True)
            final_df = final_df.sort_values(['supply_classification', 'supply_country']).reset_index(drop=True)
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
    final_df = safe_concat([result, classification_totals], ignore_index=True)
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
    final_df = safe_concat([result, country_totals], ignore_index=True)
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
    final_df = safe_concat([result, country_totals], ignore_index=True)
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
        internal_flow_filter = """
                AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                    IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
        """
        if classification_mode == 'Classification Level 1':
            if entity_name != "Global":
                join_clause = f"""
                INNER JOIN {db_schema}.mappings_country mc ON kt.origin_country_name = mc.country
                LEFT JOIN {db_schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                """
            else:
                join_clause = f"""
                LEFT JOIN {db_schema}.mappings_country mc ON kt.origin_country_name = mc.country
                LEFT JOIN {db_schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                """
            internal_flow_filter = """
                AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                    IS DISTINCT FROM COALESCE(mc.country_classification_level1, 'Unknown')
            """
        
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
                {internal_flow_filter}
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
                {internal_flow_filter}
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
        internal_flow_filter = """
                AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                    IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
        """
        if classification_mode == 'Classification Level 1':
            if entity_name != "Global":
                join_clause = f"""
                INNER JOIN {db_schema}.mappings_country mc ON kt.origin_country_name = mc.country
                LEFT JOIN {db_schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                """
            else:
                join_clause = f"""
                LEFT JOIN {db_schema}.mappings_country mc ON kt.origin_country_name = mc.country
                LEFT JOIN {db_schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                """
            internal_flow_filter = """
                AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                    IS DISTINCT FROM COALESCE(mc.country_classification_level1, 'Unknown')
            """
        
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
                {internal_flow_filter}
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
                {internal_flow_filter}
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
                
                # Check if actual_country_name column exists
                if 'actual_country_name' not in classification_installations.columns:
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
        grand_total_df = safe_concat(entity_totals_for_grand, ignore_index=True)
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
        display_df = safe_concat(filtered_rows, ignore_index=True)
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


def _build_supply_dest_columns(display_df, view_type='absolute', hidden_cols=None, delta_like_cols=None):
    """Build DataTable column definitions for supply-destination tables."""
    hidden_cols = set(hidden_cols or ['30D_Y1'])
    delta_cols = set(col for col in display_df.columns if str(col).startswith('Δ '))
    delta_cols.update(delta_like_cols or [])

    columns = []
    for col in display_df.columns:
        if col in hidden_cols:
            continue
        if col in SUPPLY_DEST_TEXT_COLUMNS:
            columns.append({'name': col, 'id': col, 'type': 'text'})
        elif col in delta_cols:
            columns.append({
                'name': col,
                'id': col,
                'type': 'numeric',
                'format': Format(precision=0, scheme=Scheme.fixed)
            })
        elif view_type == 'percentage':
            columns.append({
                'name': col,
                'id': col,
                'type': 'numeric',
                'format': Format(precision=0, scheme=Scheme.percentage)
            })
        else:
            columns.append({
                'name': col,
                'id': col,
                'type': 'numeric',
                'format': Format(precision=0, scheme=Scheme.fixed)
            })
    return columns


def _drop_supply_dest_grand_total_rows(display_df):
    """Remove grand total rows when market-share mode should stay focused on components."""
    filtered_df = display_df.copy()
    if 'Aggregation Supply' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Aggregation Supply'] != 'GRAND TOTAL']
    if 'Supply Country' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Supply Country'] != 'GRAND TOTAL']
    return filtered_df.reset_index(drop=True)


def _convert_supply_dest_period_display_to_percentage(display_df, classification_mode='Country',
                                                      demand_aggregation_mode='None', value_cols=None):
    """Convert historical period values into shares using the same hierarchy logic as the default table."""
    value_cols = [col for col in (value_cols or []) if col in display_df.columns]
    if not value_cols or display_df.empty:
        return display_df

    percentage_df = display_df.copy()
    show_demand_aggregation = use_demand_classification_mode(classification_mode, demand_aggregation_mode)
    show_demand_country = use_demand_country_mode(demand_aggregation_mode)

    if classification_mode == 'Classification Level 1' and show_demand_aggregation:
        current_supply_class = None
        subtotal_values = {}

        for _, row in percentage_df.iterrows():
            agg_supply = str(row.get('Aggregation Supply', ''))
            agg_demand = str(row.get('Aggregation Demand', ''))
            if agg_demand == 'Total' and agg_supply:
                subtotal_values[agg_supply] = {col: row[col] for col in value_cols}

        for idx, row in percentage_df.iterrows():
            agg_supply = str(row.get('Aggregation Supply', ''))
            agg_demand = str(row.get('Aggregation Demand', ''))

            if agg_supply:
                if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                    current_supply_class = agg_supply[2:].strip()
                elif agg_demand == 'Total':
                    current_supply_class = agg_supply

            if current_supply_class and current_supply_class in subtotal_values:
                for col in value_cols:
                    subtotal_val = subtotal_values[current_supply_class].get(col, 0)
                    percentage_df.at[idx, col] = (
                        1.0 if agg_demand == 'Total' and agg_supply == current_supply_class
                        else (row[col] / subtotal_val if subtotal_val else 0)
                    )

    elif classification_mode == 'Classification Level 1' and show_demand_country:
        current_supply_class = None
        subtotal_values = {}

        for _, row in percentage_df.iterrows():
            agg_supply = str(row.get('Aggregation Supply', ''))
            demand_country = str(row.get('Demand Country', ''))
            if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                subtotal_values[agg_supply[2:].strip()] = {col: row[col] for col in value_cols}
            elif demand_country == 'Total' and agg_supply:
                subtotal_values[agg_supply] = {col: row[col] for col in value_cols}

        for idx, row in percentage_df.iterrows():
            agg_supply = str(row.get('Aggregation Supply', ''))
            demand_country = str(row.get('Demand Country', ''))

            if agg_supply:
                if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                    current_supply_class = agg_supply[2:].strip()
                elif demand_country == 'Total':
                    current_supply_class = agg_supply

            if current_supply_class and current_supply_class in subtotal_values:
                for col in value_cols:
                    subtotal_val = subtotal_values[current_supply_class].get(col, 0)
                    if subtotal_val:
                        if agg_supply.startswith('▶') or agg_supply.startswith('▼') or (
                            demand_country == 'Total' and agg_supply == current_supply_class
                        ):
                            percentage_df.at[idx, col] = 1.0
                        else:
                            percentage_df.at[idx, col] = row[col] / subtotal_val
                    else:
                        percentage_df.at[idx, col] = 0

    elif classification_mode == 'Classification Level 1':
        current_supply_class = None
        subtotal_values = {}

        for _, row in percentage_df.iterrows():
            agg_supply = str(row.get('Aggregation Supply', ''))
            if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                subtotal_values[agg_supply[2:].strip()] = {col: row[col] for col in value_cols}

        for idx, row in percentage_df.iterrows():
            agg_supply = str(row.get('Aggregation Supply', ''))
            if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                current_supply_class = agg_supply[2:].strip()

            if current_supply_class and current_supply_class in subtotal_values:
                for col in value_cols:
                    subtotal_val = subtotal_values[current_supply_class].get(col, 0)
                    if subtotal_val:
                        percentage_df.at[idx, col] = (
                            1.0 if agg_supply.startswith('▶') or agg_supply.startswith('▼')
                            else row[col] / subtotal_val
                        )
                    else:
                        percentage_df.at[idx, col] = 0

    elif show_demand_country:
        current_supply_country = None
        supply_subtotals = {}

        for _, row in percentage_df.iterrows():
            supply_country = str(row.get('Supply Country', ''))
            demand_country = str(row.get('Demand Country', ''))
            if demand_country == 'Total' and supply_country:
                supply_subtotals[supply_country] = {col: row[col] for col in value_cols}

        for idx, row in percentage_df.iterrows():
            supply_country = str(row.get('Supply Country', ''))
            demand_country = str(row.get('Demand Country', ''))
            if supply_country:
                current_supply_country = supply_country

            if current_supply_country and current_supply_country in supply_subtotals:
                for col in value_cols:
                    subtotal_val = supply_subtotals[current_supply_country].get(col, 0)
                    if subtotal_val:
                        percentage_df.at[idx, col] = (
                            1.0 if demand_country == 'Total' and supply_country == current_supply_country
                            else row[col] / subtotal_val
                        )
                    else:
                        percentage_df.at[idx, col] = 0

    else:
        total_values = {col: percentage_df[col].sum() for col in value_cols}
        for idx, row in percentage_df.iterrows():
            for col in value_cols:
                total_val = total_values.get(col, 0)
                percentage_df.at[idx, col] = row[col] / total_val if total_val else 0

    return percentage_df


def prepare_supply_dest_table_for_display(df, table_type, classification_mode='Country', 
                                          expanded_classifications=None, expanded_countries=None, 
                                          expanded_supply_countries=None, view_type='absolute',
                                          demand_aggregation_mode='None'):
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
    
    df = df.copy()
    if classification_mode == 'Classification Level 1' and 'supply_classification' in df.columns:
        df = df[df['supply_classification'] != 'GRAND TOTAL']
    elif 'supply_country' in df.columns:
        df = df[df['supply_country'] != 'GRAND TOTAL']
    
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
    
    show_demand_aggregation = use_demand_classification_mode(classification_mode, demand_aggregation_mode)
    show_demand_country = use_demand_country_mode(demand_aggregation_mode)

    # Filter data based on expanded state
    filtered_rows = []
    entity_totals_for_grand = []
    
    if classification_mode == 'Classification Level 1' and show_demand_aggregation:
        # Four-level hierarchy: Supply Class → Demand Class → Demand Country → Supply Country
        supply_subtotals = {}
        for supply_class in df['supply_classification'].unique():
            supply_data = df[df['supply_classification'] == supply_class]
            supply_totals = supply_data[
                (supply_data['demand_country'] == 'Total')
                & (supply_data['supply_country'] == 'Total')
            ]
            if not supply_totals.empty:
                numeric_cols = [
                    col for col in supply_totals.columns
                    if col not in ['supply_classification', 'demand_classification', 'supply_country', 'demand_country']
                ]
                supply_subtotals[supply_class] = {
                    col: supply_totals[col].sum() for col in numeric_cols
                }

        for supply_class in sorted(df['supply_classification'].unique()):
            supply_class_rows = []
            
            supply_pairs = df[df['supply_classification'] == supply_class].groupby(
                ['supply_classification', 'demand_classification']
            ).size().reset_index()[['supply_classification', 'demand_classification']]
            
            for _, pair in supply_pairs.iterrows():
                demand_class = pair['demand_classification']
                pair_key = f"{supply_class}→{demand_class}"
                
                pair_data = df[
                    (df['supply_classification'] == supply_class)
                    & (df['demand_classification'] == demand_class)
                ]
                
                class_total = pair_data[
                    (pair_data['demand_country'] == 'Total')
                    & (pair_data['supply_country'] == 'Total')
                ]
                
                if not class_total.empty and has_non_zero_values(class_total):
                    entity_totals_for_grand.append(class_total.copy())
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
                
                if pair_key in expanded_classifications:
                    demand_countries = pair_data[pair_data['demand_country'] != 'Total']['demand_country'].unique()
                    
                    for demand_country in demand_countries:
                        country_key = f"{pair_key}→{demand_country}"
                        country_data = pair_data[pair_data['demand_country'] == demand_country]
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
                        
                        if country_key in expanded_countries:
                            supply_countries = country_data[country_data['supply_country'] != 'Total']
                            if not supply_countries.empty:
                                non_zero_mask = supply_countries.apply(
                                    lambda row: has_non_zero_values(row.to_frame().T),
                                    axis=1
                                )
                                supply_countries = supply_countries[non_zero_mask]
                                
                                if not supply_countries.empty:
                                    supply_countries = supply_countries.copy()
                                    supply_countries['supply_classification'] = ''
                                    supply_countries['demand_classification'] = ''
                                    supply_countries['demand_country'] = ''
                                    supply_countries.loc[:, 'supply_country'] = "    " + supply_countries['supply_country']
                                    supply_class_rows.append(supply_countries)
            
            filtered_rows.extend(supply_class_rows)

            supply_pairs_expanded = any(
                pair_key.startswith(f"{supply_class}→")
                for pair_key in expanded_classifications
            )
            if supply_class in supply_subtotals and supply_class_rows and not supply_pairs_expanded:
                subtotal_row = pd.DataFrame([{
                    'supply_classification': supply_class,
                    'demand_classification': 'Total',
                    'demand_country': '',
                    'supply_country': '',
                    **supply_subtotals[supply_class]
                }])
                filtered_rows.append(subtotal_row)
    
    elif classification_mode == 'Classification Level 1' and show_demand_country:
        # Three-level hierarchy: Supply Class → Demand Country → Supply Country
        for supply_class in sorted(df['supply_classification'].unique()):
            supply_data = df[df['supply_classification'] == supply_class]
            supply_total = supply_data[
                (supply_data['demand_country'] == 'Total')
                & (supply_data['supply_country'] == 'Total')
            ]
            
            if not supply_total.empty and has_non_zero_values(supply_total):
                entity_totals_for_grand.append(supply_total.copy())
                supply_total = supply_total.copy()
                if supply_class in expanded_classifications:
                    supply_total.loc[:, 'supply_classification'] = f"▼ {supply_class}"
                else:
                    supply_total.loc[:, 'supply_classification'] = f"▶ {supply_class}"
                supply_total['demand_country'] = ''
                supply_total['supply_country'] = ''
                filtered_rows.append(supply_total)
            
            if supply_class in expanded_classifications:
                demand_countries = supply_data[
                    (supply_data['demand_country'] != 'Total')
                    & (supply_data['supply_country'] == 'Total')
                ]['demand_country'].unique()
                
                for demand_country in demand_countries:
                    country_key = f"{supply_class}→{demand_country}"
                    country_data = supply_data[supply_data['demand_country'] == demand_country]
                    country_total = country_data[country_data['supply_country'] == 'Total']
                    
                    if not country_total.empty and has_non_zero_values(country_total):
                        country_total = country_total.copy()
                        country_total['supply_classification'] = ''
                        if country_key in expanded_countries:
                            country_total.loc[:, 'demand_country'] = f"  ▼ {demand_country}"
                        else:
                            country_total.loc[:, 'demand_country'] = f"  ▶ {demand_country}"
                        country_total['supply_country'] = 'Total'
                        filtered_rows.append(country_total)
                    
                    if country_key in expanded_countries:
                        supply_countries = country_data[country_data['supply_country'] != 'Total']
                        if not supply_countries.empty:
                            non_zero_mask = supply_countries.apply(
                                lambda row: has_non_zero_values(row.to_frame().T),
                                axis=1
                            )
                            supply_countries = supply_countries[non_zero_mask]
                            
                            if not supply_countries.empty:
                                supply_countries = supply_countries.copy()
                                supply_countries['supply_classification'] = ''
                                supply_countries['demand_country'] = ''
                                supply_countries.loc[:, 'supply_country'] = "    " + supply_countries['supply_country']
                                filtered_rows.append(supply_countries)
    
    elif classification_mode == 'Classification Level 1':
        # Two-level hierarchy: Supply Class → Supply Country
        for supply_class in sorted(df['supply_classification'].unique()):
            supply_data = df[df['supply_classification'] == supply_class]
            supply_total = supply_data[supply_data['supply_country'] == 'Total']
            
            if not supply_total.empty and has_non_zero_values(supply_total):
                entity_totals_for_grand.append(supply_total.copy())
                supply_total = supply_total.copy()
                if supply_class in expanded_classifications:
                    supply_total.loc[:, 'supply_classification'] = f"▼ {supply_class}"
                else:
                    supply_total.loc[:, 'supply_classification'] = f"▶ {supply_class}"
                supply_total['supply_country'] = ''
                filtered_rows.append(supply_total)
            
            if supply_class in expanded_classifications:
                supply_countries = supply_data[supply_data['supply_country'] != 'Total']
                if not supply_countries.empty:
                    non_zero_mask = supply_countries.apply(
                        lambda row: has_non_zero_values(row.to_frame().T),
                        axis=1
                    )
                    supply_countries = supply_countries[non_zero_mask]
                    
                    if not supply_countries.empty:
                        supply_countries = supply_countries.copy()
                        supply_countries['supply_classification'] = ''
                        supply_countries.loc[:, 'supply_country'] = "  " + supply_countries['supply_country']
                        filtered_rows.append(supply_countries)
    
    elif show_demand_country:
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
                supply_total_df = safe_concat(supply_country_rows, ignore_index=True)
                numeric_cols = [col for col in supply_total_df.columns if col not in 
                               ['supply_country', 'demand_country']]
                
                subtotal_row = pd.DataFrame([{
                    'supply_country': supply_country,
                    'demand_country': 'Total',
                    **{col: supply_total_df[col].sum() for col in numeric_cols}
                }])
                filtered_rows.append(subtotal_row)
    
    else:
        for supply_country in sorted(df['supply_country'].unique()):
            supply_country_data = df[df['supply_country'] == supply_country]
            if not supply_country_data.empty and has_non_zero_values(supply_country_data):
                entity_totals_for_grand.append(supply_country_data.copy())
                filtered_rows.append(supply_country_data.copy())
    
    # Add Grand Total row (only if not in percentage mode)
    if entity_totals_for_grand and view_type != 'percentage':
        grand_total_df = safe_concat(entity_totals_for_grand, ignore_index=True)
        numeric_cols = [col for col in grand_total_df.columns if col not in 
                       ['supply_classification', 'demand_classification', 'supply_country', 'demand_country']]
        
        if classification_mode == 'Classification Level 1' and show_demand_aggregation:
            grand_total_row = pd.DataFrame([{
                'supply_classification': 'GRAND TOTAL',
                'demand_classification': '',
                'demand_country': '',
                'supply_country': '',
                **{col: grand_total_df[col].sum() for col in numeric_cols}
            }])
        elif classification_mode == 'Classification Level 1' and show_demand_country:
            grand_total_row = pd.DataFrame([{
                'supply_classification': 'GRAND TOTAL',
                'demand_country': '',
                'supply_country': '',
                **{col: grand_total_df[col].sum() for col in numeric_cols}
            }])
        elif classification_mode == 'Classification Level 1':
            grand_total_row = pd.DataFrame([{
                'supply_classification': 'GRAND TOTAL',
                'supply_country': '',
                **{col: grand_total_df[col].sum() for col in numeric_cols}
            }])
        elif show_demand_country:
            grand_total_row = pd.DataFrame([{
                'supply_country': 'GRAND TOTAL',
                'demand_country': '',
                **{col: grand_total_df[col].sum() for col in numeric_cols}
            }])
        else:
            grand_total_row = pd.DataFrame([{
                'supply_country': 'GRAND TOTAL',
                **{col: grand_total_df[col].sum() for col in numeric_cols}
            }])
        
        filtered_rows.append(grand_total_row)
    
    # Combine all rows
    if filtered_rows:
        display_df = safe_concat(filtered_rows, ignore_index=True)
    else:
        display_df = pd.DataFrame()
    
    # Rename columns for display based on classification mode
    if classification_mode == 'Classification Level 1' and show_demand_aggregation:
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
    elif classification_mode == 'Classification Level 1' and show_demand_country:
        new_columns = []
        for col in display_df.columns:
            if col == 'supply_classification':
                new_columns.append('Aggregation Supply')
            elif col == 'demand_country':
                new_columns.append('Demand Country')
            elif col == 'supply_country':
                new_columns.append('Supply Country')
            else:
                new_columns.append(col)
        display_df.columns = new_columns
    elif classification_mode == 'Classification Level 1':
        new_columns = []
        for col in display_df.columns:
            if col == 'supply_classification':
                new_columns.append('Aggregation Supply')
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
            elif col == 'demand_country' and show_demand_country:
                new_columns.append('Demand Country')
            else:
                new_columns.append(col)
        display_df.columns = new_columns
    
    return display_df, _build_supply_dest_columns(display_df, view_type, hidden_cols=['30D_Y1'])


# Pre-computed conditional styles (computed once at module load)
TABLE_CONDITIONAL_STYLES = [
    # Alternating row colors (lowest priority)
    {
        'if': {'row_index': 'odd'},
        'backgroundColor': '#f8f9fa'
    },
    # Country total rows styling (medium priority)
    {
        'if': {'filter_query': '{Installation} = "Total"'},
        'backgroundColor': TABLE_COLORS['bg_lighter'],
        'fontWeight': 'bold',
        'color': TABLE_COLORS['text_primary']
    },
    # Grand Total row styling (highest priority - must be last)
    {
        'if': {'filter_query': '{Country} = "GRAND TOTAL"'},
        'backgroundColor': '#2E86C1',  # McKinsey blue
        'fontWeight': 'bold',
        'color': 'white'
    }
]


def get_table_conditional_styles():
    """Get conditional styling for tables"""
    return deepcopy(TABLE_CONDITIONAL_STYLES)


def _build_supply_dest_period_view_table(period_payload, period_view, period_count, comparison_basis,
                                         classification_mode, demand_aggregation_mode,
                                         expanded_classifications=None, expanded_countries=None,
                                         expanded_supply_countries=None, view_type='absolute',
                                         country_grouping_mode='show_all'):
    """Build the historical period-analysis table while keeping the default overview untouched."""
    period_view = _normalize_supply_dest_period_view(period_view)
    comparison_basis = _normalize_supply_dest_comparison_basis(comparison_basis)
    country_grouping_mode = _normalize_supply_dest_country_grouping(country_grouping_mode)
    expanded_classifications = expanded_classifications or []
    expanded_countries = expanded_countries or []
    expanded_supply_countries = expanded_supply_countries or []

    if not period_view:
        return pd.DataFrame(), [], {}

    period_payload = _resolve_supply_dest_period_payload(period_payload, country_grouping_mode)
    period_entry = period_payload.get(period_view, [])
    if isinstance(period_entry, dict):
        records = period_entry.get('records', [])
        current_period_label = period_entry.get('current_period_label')
        current_period_previous_label = period_entry.get('current_period_previous_label')
        current_period_previous_records = period_entry.get('current_period_previous_records', [])
        current_period_prior_year_label = period_entry.get('current_period_prior_year_label')
        current_period_prior_year_records = period_entry.get('current_period_prior_year_records', [])
    else:
        records = period_entry
        current_period_label = None
        current_period_previous_label = None
        current_period_previous_records = []
        current_period_prior_year_label = None
        current_period_prior_year_records = []

    source_df = pd.DataFrame(records)
    if source_df.empty:
        return pd.DataFrame(), [], {}

    id_cols = get_supply_dest_id_cols(classification_mode, demand_aggregation_mode)
    if current_period_previous_label:
        current_previous_df = pd.DataFrame(current_period_previous_records)
        if not current_previous_df.empty:
            source_df = source_df.merge(current_previous_df, on=id_cols, how='left')
            source_df[current_period_previous_label] = source_df[current_period_previous_label].fillna(0)
        else:
            source_df[current_period_previous_label] = 0
    if current_period_prior_year_label:
        current_prior_year_df = pd.DataFrame(current_period_prior_year_records)
        if not current_prior_year_df.empty:
            source_df = source_df.merge(current_prior_year_df, on=id_cols, how='left')
            source_df[current_period_prior_year_label] = source_df[current_period_prior_year_label].fillna(0)
        else:
            source_df[current_period_prior_year_label] = 0

    period_cols = [
        col for col in source_df.columns
        if col not in id_cols
        and col != current_period_previous_label
        and col != current_period_prior_year_label
    ]
    if not period_cols:
        return pd.DataFrame(), [], {}

    period_cols = sorted(
        period_cols,
        key=lambda col: _get_supply_dest_period_sort_key(col, period_view)
    )

    if current_period_label not in period_cols:
        current_period_label = None

    completed_period_cols = [
        col for col in period_cols
        if col != current_period_label
    ]

    available_counts = SUPPLY_DEST_PERIOD_COUNT_OPTIONS.get(period_view, [])
    default_count = available_counts[0] if available_counts else len(completed_period_cols)
    resolved_count = period_count if period_count in available_counts else default_count

    if completed_period_cols:
        resolved_count = max(1, min(int(resolved_count), len(completed_period_cols)))
        visible_period_cols = completed_period_cols[-resolved_count:]
    else:
        resolved_count = 0
        visible_period_cols = []

    if current_period_label:
        visible_period_cols = visible_period_cols + [current_period_label]

    latest_label = current_period_label or (visible_period_cols[-1] if visible_period_cols else None)
    if not latest_label:
        return pd.DataFrame(), [], {}

    comparison_reference_map = {}
    reference_label = None
    comparison_label = None
    if comparison_basis == 'previous_period':
        for visible_col in visible_period_cols:
            if visible_col == current_period_label and current_period_previous_label:
                reference_col = current_period_previous_label
            else:
                reference_col = _get_supply_dest_previous_period_label(visible_col, period_view)
            if reference_col:
                comparison_reference_map[visible_col] = reference_col
    elif comparison_basis == 'same_period_last_year':
        for visible_col in visible_period_cols:
            if visible_col == current_period_label and current_period_prior_year_label:
                reference_col = current_period_prior_year_label
            else:
                reference_col = _get_supply_dest_prior_year_label(visible_col, period_view)
            if reference_col:
                comparison_reference_map[visible_col] = reference_col

    required_period_cols = list(visible_period_cols)
    if reference_label and reference_label in period_cols and reference_label not in required_period_cols:
        required_period_cols.append(reference_label)
    if comparison_basis in {'previous_period', 'same_period_last_year'}:
        for reference_col in comparison_reference_map.values():
            if reference_col in source_df.columns and reference_col not in required_period_cols:
                required_period_cols.append(reference_col)

    working_df = source_df[id_cols + required_period_cols].copy()
    display_df, _ = prepare_supply_dest_table_for_display(
        working_df,
        'summary',
        classification_mode,
        expanded_classifications,
        expanded_countries,
        expanded_supply_countries,
        'absolute',
        demand_aggregation_mode
    )

    if display_df.empty:
        return pd.DataFrame(), [], {}

    if view_type == 'percentage':
        display_df = _drop_supply_dest_grand_total_rows(display_df)
        display_df = _convert_supply_dest_period_display_to_percentage(
            display_df,
            classification_mode,
            demand_aggregation_mode,
            value_cols=required_period_cols
        )

    comparison_source_df = display_df.copy()

    if comparison_basis in {'previous_period', 'same_period_last_year'}:
        for visible_col in visible_period_cols:
            reference_col = comparison_reference_map.get(visible_col)
            if visible_col not in comparison_source_df.columns:
                continue
            if reference_col in comparison_source_df.columns:
                display_df[visible_col] = comparison_source_df[visible_col] - comparison_source_df[reference_col]
                if view_type == 'percentage':
                    display_df[visible_col] = display_df[visible_col] * 100
            else:
                display_df[visible_col] = pd.NA

    if comparison_label and reference_label and reference_label in display_df.columns and latest_label in display_df.columns:
        display_df[comparison_label] = display_df[latest_label] - display_df[reference_label]
        if view_type == 'percentage':
            display_df[comparison_label] = display_df[comparison_label] * 100
    else:
        if comparison_basis != 'same_period_last_year':
            comparison_label = None
            reference_label = None

    hidden_cols = []
    if reference_label and reference_label not in visible_period_cols:
        hidden_cols.append(reference_label)
        display_df = display_df.drop(columns=[reference_label], errors='ignore')
    if comparison_basis in {'previous_period', 'same_period_last_year'}:
        comparison_reference_cols = [
            reference_col
            for reference_col in comparison_reference_map.values()
            if reference_col not in visible_period_cols
        ]
        if comparison_reference_cols:
            hidden_cols.extend(comparison_reference_cols)
            display_df = display_df.drop(columns=comparison_reference_cols, errors='ignore')

    delta_like_cols = visible_period_cols if comparison_basis in {'previous_period', 'same_period_last_year'} else None
    columns = _build_supply_dest_columns(
        display_df,
        view_type,
        hidden_cols=hidden_cols,
        delta_like_cols=delta_like_cols
    )
    return display_df, columns, {
        'period_view': period_view,
        'period_count': resolved_count,
        'visible_period_cols': visible_period_cols,
        'latest_label': latest_label,
        'reference_label': reference_label,
        'comparison_label': comparison_label,
        'comparison_basis': comparison_basis,
        'current_period_label': current_period_label,
        'comparison_reference_map': comparison_reference_map,
        'current_period_previous_label': current_period_previous_label,
        'current_period_prior_year_label': current_period_prior_year_label
    }


def _is_supply_dest_grand_total_row(display_df):
    """Identify GRAND TOTAL rows in supply-destination tables after display labels are applied."""
    if display_df is None or display_df.empty:
        return pd.Series(dtype=bool)

    grand_total_mask = pd.Series(False, index=display_df.index)
    for col in ['Aggregation Supply', 'Supply Country']:
        if col in display_df.columns:
            grand_total_mask = grand_total_mask | display_df[col].eq('GRAND TOTAL')
    return grand_total_mask


def _sort_supply_dest_period_display_df(display_df, sort_by=None):
    """Sort the period-analysis display while keeping GRAND TOTAL pinned to the bottom."""
    if display_df is None or display_df.empty or not sort_by:
        return display_df

    valid_sort_rules = [
        sort_rule for sort_rule in sort_by
        if sort_rule.get('column_id') in display_df.columns
    ]
    if not valid_sort_rules:
        return display_df

    grand_total_mask = _is_supply_dest_grand_total_row(display_df)
    sortable_df = display_df.loc[~grand_total_mask].copy()
    grand_total_df = display_df.loc[grand_total_mask].copy()

    if sortable_df.empty:
        return display_df

    sort_key_cols = []
    ascending = []
    for idx, sort_rule in enumerate(valid_sort_rules):
        column_id = sort_rule['column_id']
        sort_key = f'__sort_key_{idx}'
        column_series = sortable_df[column_id]

        if pd.api.types.is_numeric_dtype(column_series):
            sortable_df[sort_key] = pd.to_numeric(column_series, errors='coerce')
        else:
            sortable_df[sort_key] = (
                column_series.fillna('')
                .astype(str)
                .str.replace(r'^[\s▶▼]+', '', regex=True)
                .str.lower()
            )

        sort_key_cols.append(sort_key)
        ascending.append(sort_rule.get('direction', 'asc') != 'desc')

    sortable_df = sortable_df.sort_values(
        by=sort_key_cols,
        ascending=ascending,
        na_position='last',
        kind='mergesort'
    ).drop(columns=sort_key_cols, errors='ignore')

    return safe_concat([sortable_df, grand_total_df], ignore_index=True)

def _build_supply_dest_comparison_heatmap_styles(display_df, visible_period_cols):
    """Create diverging red/green heatmap styles using one shared scale across visible periods."""
    if display_df is None or display_df.empty or not visible_period_cols:
        return []

    comparison_df = display_df.copy()
    grand_total_mask = _is_supply_dest_grand_total_row(comparison_df)
    if grand_total_mask.any():
        comparison_df = comparison_df.loc[~grand_total_mask].copy()

    positive_palette = ['#ecfdf5', '#d1fae5', '#a7f3d0', '#6ee7b7', '#34d399']
    negative_palette = ['#fef2f2', '#fee2e2', '#fecaca', '#fca5a5', '#f87171']

    available_cols = [col for col in visible_period_cols if col in comparison_df.columns]
    if not available_cols:
        return []

    numeric_df = comparison_df[available_cols].apply(pd.to_numeric, errors='coerce')
    flattened_values = numeric_df.stack(dropna=True)
    if flattened_values.empty:
        return []

    global_max_abs = flattened_values.abs().max()
    if pd.isna(global_max_abs) or global_max_abs <= 0:
        return []

    heatmap_styles = []
    bucket_thresholds = [0.2, 0.4, 0.6, 0.8, 1.0]
    for visible_col in available_cols:
        numeric_series = numeric_df[visible_col].dropna()
        if numeric_series.empty:
            continue

        heatmap_styles.append({
            'if': {'column_id': visible_col},
            'backgroundColor': '#f8fafc',
            'fontWeight': '600',
            'color': TABLE_COLORS['text_primary']
        })
        for row_idx, value in numeric_series.items():
            if pd.isna(value) or value == 0:
                continue

            value_ratio = min(abs(float(value)) / float(global_max_abs), 1.0)
            palette_index = next(
                (
                    idx for idx, threshold in enumerate(bucket_thresholds)
                    if value_ratio <= threshold
                ),
                len(bucket_thresholds) - 1
            )
            is_positive = float(value) > 0
            heatmap_styles.append({
                'if': {'column_id': visible_col, 'row_index': int(row_idx)},
                'backgroundColor': (
                    positive_palette[palette_index]
                    if is_positive else negative_palette[palette_index]
                ),
                'color': '#14532d' if is_positive else '#7f1d1d',
                'fontWeight': '600'
            })

    return heatmap_styles


def _get_supply_dest_hierarchy_instruction(classification_mode, demand_aggregation_mode):
    """Return the hierarchy help text for supply-destination tables."""
    show_demand_aggregation = use_demand_classification_mode(classification_mode, demand_aggregation_mode)
    show_demand_country = use_demand_country_mode(demand_aggregation_mode)

    if classification_mode == 'Classification Level 1' and show_demand_aggregation:
        return "Click on ▶ to expand classification pairs, then demand countries, then supply countries"
    if classification_mode == 'Classification Level 1' and show_demand_country:
        return "Click on ▶ to expand supply classifications, then demand countries, then supply countries"
    if classification_mode == 'Classification Level 1':
        return "Click on ▶ to expand supply classifications and see supply countries"
    if show_demand_country:
        return "Click on supply-demand pairs to see details"
    return "Showing total supply by supply country"


def _render_supply_dest_period_analysis_table(period_payload, classification_mode, demand_aggregation_mode,
                                              view_type, period_view, period_count, comparison_basis,
                                              expanded_classifications=None, expanded_countries=None,
                                              expanded_supply_countries=None, period_sort_by=None,
                                              country_grouping_mode='show_all'):
    """Render the separate historical period-analysis table section."""
    selected_period_view = _normalize_supply_dest_period_view(period_view)
    country_grouping_mode = _normalize_supply_dest_country_grouping(country_grouping_mode)
    hierarchy_instruction_text = _get_supply_dest_hierarchy_instruction(
        classification_mode,
        demand_aggregation_mode
    )

    if not selected_period_view:
        return (
            html.Div(
                "Select a time view to display a separate historical destination table.",
                style={'textAlign': 'center', 'padding': '28px 20px', 'color': '#64748b'}
            ),
            html.Span(
                "Choose Time View, Periods, and Comparison Basis above to populate this section.",
                style={'fontSize': '13px', 'color': '#64748b', 'fontStyle': 'italic'}
            )
        )

    try:
        display_df, columns, period_meta = _build_supply_dest_period_view_table(
            period_payload,
            selected_period_view,
            period_count,
            comparison_basis,
            classification_mode,
            demand_aggregation_mode,
            expanded_classifications,
            expanded_countries,
            expanded_supply_countries,
            view_type,
            country_grouping_mode
        )

        if display_df.empty:
            return (
                html.Div(
                    "No data available for the selected period analysis.",
                    style={'textAlign': 'center', 'padding': '20px'}
                ),
                html.Span(
                    hierarchy_instruction_text,
                    style={'fontSize': '13px', 'color': '#666', 'fontStyle': 'italic'}
                )
            )

        visible_period_cols = period_meta.get('visible_period_cols', [])
        latest_label = period_meta.get('latest_label')
        reference_label = period_meta.get('reference_label')
        comparison_label = period_meta.get('comparison_label')
        comparison_basis_mode = period_meta.get('comparison_basis', 'levels')
        period_count_label = period_meta.get('period_count', len(visible_period_cols))
        current_period_label = period_meta.get('current_period_label')
        current_period_previous_label = period_meta.get('current_period_previous_label')
        current_period_prior_year_label = period_meta.get('current_period_prior_year_label')
        current_period_visible = bool(current_period_label and current_period_label in display_df.columns)
        per_column_previous_mode = comparison_basis_mode == 'previous_period'
        per_column_yoy_mode = comparison_basis_mode == 'same_period_last_year'
        per_column_comparison_mode = per_column_previous_mode or per_column_yoy_mode
        period_view_title = {
            'monthly': 'Monthly',
            'quarterly': 'Quarterly',
            'seasonally': 'Seasonally',
            'yearly': 'Yearly'
        }.get(selected_period_view, 'Historical')

        instruction_children = []
        if per_column_yoy_mode:
            instruction_children.extend([
                html.Span(
                    "Each visible period column shows the YoY level change versus the same period last year.",
                    style={'fontSize': '13px', 'color': '#64748b'}
                )
            ])
            if current_period_visible:
                instruction_children.extend([
                    html.Br(),
                    html.Span(
                        (
                            f"{current_period_label} uses {current_period_prior_year_label} "
                            "for the same period-to-date comparison."
                        ),
                        style={'fontSize': '13px', 'color': '#64748b'}
                    )
                ])
        elif per_column_previous_mode:
            instruction_children.extend([
                html.Span(
                    "Each visible period column shows the change versus the immediately previous period.",
                    style={'fontSize': '13px', 'color': '#64748b'}
                )
            ])
            if current_period_visible:
                instruction_children.extend([
                    html.Br(),
                    html.Span(
                        (
                            f"{current_period_label} uses {current_period_previous_label} "
                            "for the previous-period-to-date comparison."
                        ),
                        style={'fontSize': '13px', 'color': '#64748b'}
                    )
                ])
        elif comparison_label and latest_label and reference_label:
            instruction_children.extend([
                html.Br(),
                html.Span(
                    f"{comparison_label} compares {latest_label} against {reference_label}.",
                    style={'fontSize': '13px', 'color': '#64748b'}
                )
            ])
        if per_column_comparison_mode:
            instruction_children.extend([
                html.Br(),
                html.Span(
                    "Heatmap shading is applied across all visible period columns on one shared scale: green for increases, red for decreases, and darker shades for larger moves.",
                    style={'fontSize': '13px', 'color': '#64748b'}
                )
            ])
        instruction_element = html.Div(instruction_children, style={'lineHeight': '1.5'})

        default_sort_by = []
        if comparison_label and comparison_label in display_df.columns:
            default_sort_by = [{'column_id': comparison_label, 'direction': 'desc'}]
        elif latest_label and latest_label in display_df.columns:
            default_sort_by = [{'column_id': latest_label, 'direction': 'desc'}]

        current_sort_by = period_sort_by or default_sort_by
        current_sort_by = [
            sort_rule for sort_rule in current_sort_by
            if sort_rule.get('column_id') in display_df.columns
        ]
        if not current_sort_by:
            current_sort_by = default_sort_by
        display_df = _sort_supply_dest_period_display_df(display_df, current_sort_by)

        conditional_styles = get_table_conditional_styles()

        if 'Aggregation Supply' in display_df.columns:
            conditional_styles.insert(1, {
                'if': {'filter_query': '{Aggregation Supply} = ""'},
                'backgroundColor': '#f9f9f9',
                'fontSize': '13px'
            })

        if 'Aggregation Demand' in display_df.columns:
            conditional_styles.append({
                'if': {'filter_query': '{Aggregation Demand} = "Total"'},
                'backgroundColor': TABLE_COLORS['bg_lighter'],
                'fontWeight': 'bold',
                'color': TABLE_COLORS['text_primary']
            })

        if 'Demand Country' in display_df.columns:
            conditional_styles.append({
                'if': {'filter_query': '{Demand Country} = "Total"'},
                'backgroundColor': TABLE_COLORS['bg_lighter'],
                'fontWeight': 'bold',
                'color': TABLE_COLORS['text_primary']
            })

        for col in SUPPLY_DEST_TEXT_COLUMNS:
            if col in display_df.columns:
                conditional_styles.append({
                    'if': {'column_id': col},
                    'textAlign': 'left'
                })

        delta_columns = [col for col in display_df.columns if str(col).startswith('Δ ')]
        numeric_columns = [
            col for col in display_df.columns
            if col not in SUPPLY_DEST_TEXT_COLUMNS
        ]

        for col in numeric_columns:
            conditional_styles.append({
                'if': {'column_id': col},
                'textAlign': 'right',
                'paddingRight': '12px'
            })

        period_color_map = {
            'monthly': '#f3e5f5',
            'quarterly': '#e3f2fd',
            'seasonally': '#e8f5e9',
            'yearly': '#fff3e0'
        }
        header_color = period_color_map.get(selected_period_view, '#f8fafc')

        if per_column_comparison_mode:
            conditional_styles.extend(
                _build_supply_dest_comparison_heatmap_styles(display_df, visible_period_cols)
            )

        if current_period_visible:
            if per_column_comparison_mode:
                conditional_styles.append({
                    'if': {'column_id': current_period_label},
                    'fontWeight': '700',
                    'borderLeft': '2px solid #f59e0b',
                    'borderRight': '2px solid #f59e0b'
                })
            else:
                conditional_styles.append({
                    'if': {'column_id': current_period_label},
                    'backgroundColor': '#fffbeb',
                    'fontWeight': '600'
                })

        if visible_period_cols:
            conditional_styles.append({
                'if': {'column_id': visible_period_cols[0]},
                'borderLeft': '3px solid white'
            })

        for delta_col in delta_columns:
            conditional_styles.append({
                'if': {'column_id': delta_col},
                'backgroundColor': '#f5f5f5',
                'fontWeight': '600',
                'borderLeft': '3px solid white'
            })
            conditional_styles.append({
                'if': {'column_id': delta_col, 'filter_query': f'{{{delta_col}}} > 0'},
                'color': '#2e7d32',
                'fontWeight': '600'
            })
            conditional_styles.append({
                'if': {'column_id': delta_col, 'filter_query': f'{{{delta_col}}} < 0'},
                'color': '#c62828',
                'fontWeight': '600'
            })

        conditional_styles.append({
            'if': {'filter_query': '{Aggregation Supply} = "GRAND TOTAL" or {Supply Country} = "GRAND TOTAL"'},
            'backgroundColor': '#2E86C1',
            'fontWeight': 'bold',
            'color': 'white'
        })

        table_config = StandardTableStyleManager.get_base_datatable_config()
        table_config['style_data_conditional'] = conditional_styles

        header_styles = []
        if visible_period_cols:
            for col in visible_period_cols:
                if col in display_df.columns:
                    header_styles.append({
                        'if': {'column_id': col},
                        'backgroundColor': header_color
                    })
            header_styles.append({
                'if': {'column_id': visible_period_cols[0]},
                'borderLeft': '3px solid white'
            })
        if current_period_visible:
            header_styles.append({
                'if': {'column_id': current_period_label},
                'backgroundColor': '#fde68a',
                'color': '#78350f',
                'borderBottom': '2px solid #f59e0b'
            })
        if comparison_label and comparison_label in display_df.columns:
            header_styles.append({
                'if': {'column_id': comparison_label},
                'borderLeft': '3px solid white',
                'backgroundColor': '#f1f5f9'
            })
        if header_styles:
            table_config['style_header_conditional'] = header_styles

        table = dash_table.DataTable(
            id={'type': 'supply-dest-period-expandable-table', 'index': 'period'},
            data=display_df.to_dict('records'),
            columns=columns,
            page_size=100,
            sort_action='custom',
            sort_mode='multi',
            sort_by=current_sort_by,
            fill_width=False,
            export_format='none',
            **table_config
        )

        if per_column_yoy_mode:
            value_label = (
                'YoY change in market share, shown in percentage points'
                if view_type == 'percentage'
                else 'YoY change in bilateral trade flows in mcm/d'
            )
        elif per_column_previous_mode:
            value_label = (
                'change versus the previous period in market share, shown in percentage points'
                if view_type == 'percentage'
                else 'change versus the previous period in bilateral trade flows in mcm/d'
            )
        else:
            value_label = 'market share' if view_type == 'percentage' else 'bilateral trade flows in mcm/d'
        footnote_parts = [
            html.Span('Note: ', style={'fontWeight': 'bold'}),
            html.Span(
                (
                    f"{period_view_title} view shows {period_count_label} completed periods"
                    + (
                        f" plus the current in-progress period ({current_period_label}), highlighted in amber. "
                        if current_period_visible else ". "
                    )
                ),
                style={'color': '#666'}
            ),
            html.Span(
                f"Values shown are {value_label}. ",
                style={'color': '#666'}
            )
        ]
        if country_grouping_mode == 'group_small_countries':
            footnote_parts.append(
                html.Span(
                    "Small countries on the visible country axis are grouped into Rest of countries when their monthly average never exceeds 10 mcm/d across the last 24 months. When demand countries are shown, the threshold is evaluated within each supply-side parent. ",
                    style={'color': '#666'}
                )
            )
        if per_column_previous_mode:
            footnote_parts.append(
                html.Span(
                    "Every visible period column is compared against the immediately previous period. Heatmap shading uses one shared scale across the visible periods: green for increases, red for decreases, and darker shades for larger moves. ",
                    style={'color': '#666'}
                )
            )
        if per_column_yoy_mode:
            footnote_parts.append(
                html.Span(
                    "Every visible period column is compared against the same period last year. Heatmap shading uses one shared scale across the visible periods: green for increases, red for decreases, and darker shades for larger moves. ",
                    style={'color': '#666'}
                )
            )
        if current_period_visible:
            current_period_note = (
                f"{current_period_label} is normalized using days elapsed so far in the current period"
            )
            if per_column_previous_mode and current_period_previous_label:
                current_period_note += (
                    f", and its previous-period baseline uses {current_period_previous_label}"
                )
            if per_column_yoy_mode and current_period_prior_year_label:
                current_period_note += (
                    f", and its YoY baseline uses {current_period_prior_year_label}"
                )
            current_period_note += "."
            footnote_parts.append(
                html.Span(
                    current_period_note,
                    style={'color': '#666'}
                )
            )
        if comparison_label and latest_label and reference_label:
            footnote_parts.append(
                html.Span(
                    (
                        f"{comparison_label} uses {latest_label} versus {reference_label}"
                        + (" and is shown in percentage points." if view_type == 'percentage' else '.')
                    ),
                    style={'color': '#666'}
                )
            )

        footnote = html.Div([
            html.P(
                footnote_parts,
                style={'fontSize': '12px', 'fontStyle': 'italic', 'marginTop': '10px', 'color': '#666'}
            )
        ])

        return html.Div([table, footnote]), instruction_element

    except Exception as e:
        return (
            html.Div(
                f"Error creating historical supply-destination table: {str(e)}",
                style={'textAlign': 'center', 'padding': '20px', 'color': 'red'}
            ),
            html.Span(
                hierarchy_instruction_text,
                style={'fontSize': '13px', 'color': '#666', 'fontStyle': 'italic'}
            )
        )


def _create_top_exporters_selector_region():
    """Render the sticky exporter controls using the shared header format."""
    radio_label_style = {
        'display': 'inline-flex',
        'alignItems': 'center',
        'marginRight': '10px',
        'fontSize': '12px',
        'fontWeight': '600',
        'color': '#334155'
    }
    control_shell_style = {
        'display': 'inline-flex',
        'alignItems': 'center',
        'gap': '10px',
        'padding': '6px 8px 6px 12px',
        'backgroundColor': '#ffffff',
        'border': '1px solid #dbe4ee',
        'borderRadius': '999px',
        'boxShadow': '0 1px 2px rgba(15, 23, 42, 0.05)'
    }

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div('Country Classification', className='filter-group-header'),
                            html.Div(
                                [
                                    dcc.RadioItems(
                                        id='country-classification-dropdown',
                                        options=[
                                            {'label': 'Country', 'value': 'Country'},
                                            {'label': 'Classification Level 1', 'value': 'Classification Level 1'}
                                        ],
                                        value='Country',
                                        inline=True,
                                        labelStyle=radio_label_style,
                                        inputStyle={'marginRight': '6px'},
                                        style={
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'flexWrap': 'wrap'
                                        }
                                    )
                                ],
                                style=control_shell_style
                            )
                        ],
                        className='filter-section filter-section-destination'
                    ),
                    html.Div(
                        [
                            html.Div('Country Grouping', className='filter-group-header'),
                            dcc.Dropdown(
                                id='supply-dest-country-grouping-dropdown',
                                options=SUPPLY_DEST_COUNTRY_GROUPING_OPTIONS,
                                value='group_small_countries',
                                clearable=False,
                                className='filter-dropdown',
                                style={'minWidth': '220px', 'width': '100%'}
                            )
                        ],
                        className='filter-section filter-section-analysis'
                    ),
                    html.Div(
                        [
                            html.Div('Latest Upload', className='filter-group-header'),
                            html.Div(
                                id='data-timestamp-display',
                                className='text-tertiary',
                                style={'fontSize': '11px', 'maxWidth': '420px', 'width': '100%'}
                            )
                        ],
                        className='filter-section filter-section-analysis'
                    )
                ],
                className='filter-bar-grouped'
            )
        ]
    )


# Dashboard layout
layout = html.Div([
    # Interval component to trigger initial data load (runs once on page load)
    dcc.Interval(id='initial-load-trigger', interval=1000*60*60*24, n_intervals=0, max_intervals=1),
    
    # Store components for caching data (memory is faster than local storage)
    dcc.Store(id='supply-charts-data', storage_type='memory'),  # Single store for all supply chart data
    dcc.Store(id='continent-charts-data', storage_type='memory'),  # Store for continent charts data
    dcc.Store(id='summary-data-store', storage_type='memory'),
    dcc.Store(id='supply-dest-data-store', storage_type='memory'),  # Store for supply-destination data
    dcc.Store(id='supply-dest-period-data-store', storage_type='memory'),
    dcc.Store(id='supply-dest-sort-by', data=[{'column_id': '30D', 'direction': 'desc'}]),
    dcc.Store(id='supply-dest-period-sort-by', data=[]),
    
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

    # Store for latest upload timestamp
    dcc.Store(id='latest-upload-timestamp', data=None),

    # Download components for Excel exports
    dcc.Download(id='download-supply-charts-excel'),
    dcc.Download(id='download-continent-charts-excel'),
    dcc.Download(id='download-supply-dest-excel'),
    dcc.Download(id='download-supply-dest-period-excel'),

    html.Div(
        [_create_top_exporters_selector_region()],
        className='professional-section-header',
        style={'margin': '0'}
    ),

    # Supply Charts Section - Dynamic container
    html.Div([
        # Header matching the style of Installation Trends
        html.Div([
            html.H3('LNG Supply - 30-Day Rolling Average', className="section-title-inline"),
            html.Button(
                'Export to Excel',
                id='export-supply-charts-button',
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
            html.Div([
                html.H3('LNG Supply by Destination Continent - 30-Day Rolling Average', className="section-title-inline"),
                html.Button(
                    'Export to Excel',
                    id='export-continent-charts-button',
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
            ], style={'display': 'flex', 'alignItems': 'center'}),
            dcc.Dropdown(
                id='continent-chart-type',
                options=[
                    {'label': 'By Continent (mcm/d)', 'value': 'absolute'},
                    {'label': 'Market Share (%)', 'value': 'percentage'}
                ],
                value='absolute',
                clearable=False,
                style={'width': '200px'}
            ),
        ], className="inline-section-header", style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'gap': '16px', 'flexWrap': 'wrap'}),
        
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
            html.Div([
                html.H3('LNG Supply by Destination', className="section-title-inline"),
                html.Button(
                    'Export to Excel',
                    id='export-supply-dest-button',
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
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                dcc.Dropdown(
                    id='supply-dest-view-type',
                    options=[
                        {'label': 'By Destination (mcm/d)', 'value': 'absolute'},
                        {'label': 'Market Share (%)', 'value': 'percentage'}
                    ],
                    value='absolute',
                    clearable=False,
                    style={'width': '200px'}
                ),
                html.Label(
                    'Aggregation Demand:',
                    style={
                        'marginBottom': '0',
                        'fontWeight': 'bold',
                        'fontSize': '14px'
                    }
                ),
                dcc.Dropdown(
                    id='aggregation-demand-dropdown',
                    options=[
                        {'label': 'None', 'value': 'None'},
                        {'label': 'Country', 'value': 'Country'},
                        {'label': 'Classification Level 1', 'value': 'Classification Level 1'}
                    ],
                    value='None',
                    clearable=False,
                    style={'width': '220px'}
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '20px', 'flexWrap': 'wrap'})
        ], className="inline-section-header", style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'gap': '16px', 'flexWrap': 'wrap'}),

        html.Div(id='supply-dest-table-instructions', style={'marginTop': '20px', 'marginBottom': '15px'}),
        dcc.Loading(
            id="supply-dest-table-loading",
            children=[
                html.Div(id='supply-dest-table-container')
            ],
            type="default"
        )
    ], className="main-section-container", style={'marginBottom': '30px'}),

    html.Div([
        html.Div([
            html.Div([
                html.H3('LNG Supply by Destination - Period Analysis', className="section-title-inline"),
                html.Button(
                    'Export to Excel',
                    id='export-supply-dest-period-button',
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
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Span(
                'Uses the same View Type, Aggregation Demand, and Country Grouping settings as the selectors above.',
                style={'fontSize': '13px', 'color': '#64748b'}
            )
        ], className="inline-section-header", style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'gap': '16px', 'flexWrap': 'wrap'}),

        html.Div([
            html.Div('Period Analysis', className='filter-group-header'),
            html.Div([
                html.Div([
                    html.Label('Time View', className='inline-filter-label'),
                    dcc.Dropdown(
                        id='supply-dest-period-view-dropdown',
                        options=SUPPLY_DEST_PERIOD_VIEW_OPTIONS,
                        value='monthly',
                        placeholder='Select period view',
                        clearable=True,
                        className='filter-dropdown',
                        style={'width': '100%'}
                    )
                ], style={'display': 'grid', 'gap': '6px', 'minWidth': '0'}),
                html.Div([
                    html.Label('Periods', className='inline-filter-label'),
                    dcc.Dropdown(
                        id='supply-dest-period-count-dropdown',
                        options=[],
                        value=None,
                        clearable=False,
                        disabled=True,
                        className='filter-dropdown',
                        style={'width': '100%'}
                    )
                ], style={'display': 'grid', 'gap': '6px', 'minWidth': '0'}),
                html.Div([
                    html.Label('Comparison Basis', className='inline-filter-label'),
                    dcc.Dropdown(
                        id='supply-dest-comparison-basis-dropdown',
                        options=SUPPLY_DEST_COMPARISON_BASIS_OPTIONS,
                        value='levels',
                        clearable=False,
                        disabled=True,
                        className='filter-dropdown',
                        style={'width': '100%'}
                    )
                ], style={'display': 'grid', 'gap': '6px', 'minWidth': '0'})
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(180px, 1fr))', 'gap': '12px'})
        ], className='filter-section filter-section-analysis', style={'marginTop': '20px'}),
        html.Div(id='supply-dest-period-table-instructions', style={'marginTop': '16px', 'marginBottom': '15px'}),
        dcc.Loading(
            id="supply-dest-period-table-loading",
            children=[
                html.Div(id='supply-dest-period-table-container')
            ],
            type="default"
        )
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
    Output('latest-upload-timestamp', 'data'),
    Input('initial-load-trigger', 'n_intervals'),
    prevent_initial_call=False
)
def fetch_latest_timestamp(n_intervals):
    """Fetch the latest upload timestamp from kpler_trades table"""
    engine, schema = setup_database_connection()
    timestamp = get_latest_upload_timestamp(engine, schema)
    if timestamp:
        return timestamp.isoformat()
    return None

@callback(
    Output('data-timestamp-display', 'children'),
    Input('latest-upload-timestamp', 'data'),
    prevent_initial_call=False
)
def display_timestamp(timestamp_iso):
    """Display the latest upload timestamp in a formatted way"""
    if timestamp_iso:
        try:
            timestamp = datetime.fromisoformat(timestamp_iso)
            formatted_date = timestamp.strftime('%Y-%m-%d %H:%M UTC')
            return f"Data as of: {formatted_date}"
        except:
            return ""
    return ""

@callback(
    Output('country-classification-mode', 'data'),
    Input('country-classification-dropdown', 'value'),
    prevent_initial_call=False
)
def update_classification_mode(value):
    """Store the selected classification mode"""
    return value


@callback(
    Output('supply-dest-period-count-dropdown', 'options'),
    Output('supply-dest-period-count-dropdown', 'value'),
    Output('supply-dest-period-count-dropdown', 'disabled'),
    Output('supply-dest-comparison-basis-dropdown', 'disabled'),
    Input('supply-dest-period-view-dropdown', 'value'),
    State('supply-dest-period-count-dropdown', 'value'),
    prevent_initial_call=False
)
def update_supply_dest_period_controls(period_view, current_period_count):
    """Enable and populate the historical period controls when a time view is selected."""
    period_view = _normalize_supply_dest_period_view(period_view)
    if not period_view:
        return [], None, True, True

    options = _get_supply_dest_period_options(period_view)
    valid_values = [option['value'] for option in options]
    next_value = current_period_count if current_period_count in valid_values else (
        valid_values[0] if valid_values else None
    )
    return options, next_value, False, False


@callback(
    [Output('supply-charts-data', 'data'),
     Output('continent-charts-data', 'data'),
     Output('summary-data-store', 'data'),
     Output('supply-dest-data-store', 'data'),
     Output('supply-dest-period-data-store', 'data')],
    [Input('initial-load-trigger', 'n_intervals'),
     Input('country-classification-mode', 'data'),
     Input('aggregation-demand-dropdown', 'value')],
    prevent_initial_call=False
)
def refresh_all_data(n_intervals, classification_mode, demand_aggregation_mode):
    """Load all data from database"""
    try:
        # Fetch data
        engine_inst, schema = setup_database_connection()

        # Default to 'Country' if classification_mode is None
        if classification_mode is None:
            classification_mode = 'Country'
        demand_aggregation_mode = normalize_demand_aggregation_mode(demand_aggregation_mode)

        # Dictionary to store all chart data
        charts_data = {}

        # Fetch global supply data
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

        # Fetch summary table data
        summary_df = fetch_summary_table_data(engine_inst, schema, classification_mode)

        supply_dest_base_df = fetch_supply_destination_base_data(engine_inst, schema)
        global_rolling_data = fetch_rolling_windows_data(engine_inst, schema, classification_mode)

        supply_dest_summary_payload = build_supply_dest_summary_store_payload(
            engine_inst,
            schema,
            supply_dest_base_df,
            classification_mode,
            demand_aggregation_mode,
            global_rolling_data
        )

        supply_dest_period_payload = build_supply_dest_period_store_payload(
            supply_dest_base_df,
            classification_mode,
            demand_aggregation_mode
        )

        # For continent charts, pass the same entity names
        continent_charts_entities = list(charts_data.keys())

        return (charts_data,
                continent_charts_entities,
                summary_df.to_dict('records') if not summary_df.empty else [],
                supply_dest_summary_payload,
                supply_dest_period_payload)

    except Exception as e:
        return {}, [], [], _empty_supply_dest_summary_store_payload(), _empty_supply_dest_period_store_payload()


@callback(
    [Output('supply-dest-expanded-classifications', 'data', allow_duplicate=True),
     Output('supply-dest-expanded-countries', 'data', allow_duplicate=True),
     Output('supply-dest-expanded-supply-countries', 'data', allow_duplicate=True)],
    [Input('country-classification-mode', 'data'),
     Input('aggregation-demand-dropdown', 'value'),
     Input('supply-dest-country-grouping-dropdown', 'value')],
    prevent_initial_call=True
)
def reset_supply_dest_expansion_state(classification_mode, demand_aggregation_mode, country_grouping_mode):
    """Reset expanded rows whenever the supply/demand aggregation controls change."""
    return [], [], []


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
        for rolling_col in ['30D', '7D']:
            if rolling_col in display_df.columns:
                conditional_styles.append({
                    'if': {
                        'column_id': rolling_col,
                        'filter_query': '{Country} = "GRAND TOTAL"'
                    },
                    'color': TABLE_COLORS['text_primary']
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
     Input('supply-dest-view-type', 'value'),
     Input('aggregation-demand-dropdown', 'value'),
     Input('supply-dest-country-grouping-dropdown', 'value')],
    [State('supply-dest-sort-by', 'data')],
    prevent_initial_call=False
)
def update_supply_dest_table(supply_dest_data, expanded_classifications, expanded_countries,
                             expanded_supply_countries, classification_mode, view_type,
                             demand_aggregation_mode, country_grouping_mode, supply_dest_sort_by):
    """Update the supply-destination table with expandable rows"""
    show_demand_aggregation = use_demand_classification_mode(classification_mode, demand_aggregation_mode)
    show_demand_country = use_demand_country_mode(demand_aggregation_mode)
    country_grouping_mode = _normalize_supply_dest_country_grouping(country_grouping_mode)
    expanded_classifications = expanded_classifications or []
    expanded_countries = expanded_countries or []
    expanded_supply_countries = expanded_supply_countries or []

    hierarchy_instruction_text = _get_supply_dest_hierarchy_instruction(
        classification_mode,
        demand_aggregation_mode
    )
    instruction_children = [
        html.Span(
            hierarchy_instruction_text,
            style={'fontSize': '14px', 'color': '#666', 'fontStyle': 'italic'}
        )
    ]
    if country_grouping_mode == 'group_small_countries':
        instruction_children.extend([
            html.Br(),
            html.Span(
                "Small countries on the visible country axis are grouped into Rest of countries when their monthly average never exceeds 10 mcm/d across the last 24 months. When demand countries are shown, the threshold is evaluated within each supply-side parent.",
                style={'fontSize': '13px', 'color': '#64748b'}
            )
        ])
    instruction_element = html.Div(instruction_children, style={'lineHeight': '1.5'})
    
    if not supply_dest_data:
        return (html.Div("No data available. Please refresh to load data.", 
                        style={'textAlign': 'center', 'padding': '20px'}),
                instruction_element)
    
    try:
        resolved_supply_dest_data = _resolve_supply_dest_summary_payload(
            supply_dest_data,
            country_grouping_mode
        )
        df = pd.DataFrame(resolved_supply_dest_data)
        
        if df.empty:
            return (html.Div("No data available. Please refresh to load data.", 
                           style={'textAlign': 'center', 'padding': '20px'}),
                    instruction_element)
        
        # Prepare data for display with expandable rows
        display_df, columns = prepare_supply_dest_table_for_display(
            df, 'summary', classification_mode,
            expanded_classifications, expanded_countries, expanded_supply_countries, view_type,
            demand_aggregation_mode
        )
        
        if display_df.empty:
            return (html.Div("No data available", 
                           style={'textAlign': 'center', 'padding': '20px'}),
                    instruction_element)
        
        # Convert to percentages if requested
        if view_type == 'percentage' and classification_mode == 'Classification Level 1' and show_demand_aggregation:
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
        
        elif view_type == 'percentage' and classification_mode == 'Classification Level 1' and show_demand_country:
            text_cols = ['Aggregation Supply', 'Demand Country', 'Supply Country']
            delta_cols = ['Δ 7D-30D', 'Δ 30D Y/Y']
            numeric_cols = [col for col in display_df.columns if col not in text_cols and col not in delta_cols]
            numeric_cols_with_y1 = numeric_cols + (['30D_Y1'] if '30D_Y1' in display_df.columns else [])
            
            current_supply_class = None
            subtotal_values = {}
            
            for _, row in display_df.iterrows():
                agg_supply = str(row.get('Aggregation Supply', ''))
                demand_country = str(row.get('Demand Country', ''))
                if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                    supply_name = agg_supply[2:].strip()
                    subtotal_values[supply_name] = {col: row[col] for col in numeric_cols_with_y1 if col in row}
                elif demand_country == 'Total' and agg_supply:
                    subtotal_values[agg_supply] = {col: row[col] for col in numeric_cols_with_y1 if col in row}
            
            for idx, row in display_df.iterrows():
                agg_supply = str(row.get('Aggregation Supply', ''))
                demand_country = str(row.get('Demand Country', ''))
                
                if agg_supply:
                    if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                        current_supply_class = agg_supply[2:].strip()
                    elif demand_country == 'Total':
                        current_supply_class = agg_supply
                
                if current_supply_class and current_supply_class in subtotal_values:
                    for col in numeric_cols:
                        if col in display_df.columns and col != '30D_Y1' and col in subtotal_values[current_supply_class]:
                            subtotal_val = subtotal_values[current_supply_class][col]
                            if subtotal_val != 0:
                                if agg_supply.startswith('▶') or agg_supply.startswith('▼') or (
                                    demand_country == 'Total' and agg_supply == current_supply_class
                                ):
                                    display_df.at[idx, col] = 1.0
                                else:
                                    display_df.at[idx, col] = (row[col] / subtotal_val)
            
            if 'Δ 7D-30D' in display_df.columns and '7D' in display_df.columns and '30D' in display_df.columns:
                display_df['Δ 7D-30D'] = (display_df['7D'] - display_df['30D'])
            
            if 'Δ 30D Y/Y' in display_df.columns and '30D_Y1' in display_df.columns:
                current_supply_class_for_y1 = None
                for idx, row in display_df.iterrows():
                    agg_supply = str(row.get('Aggregation Supply', ''))
                    demand_country = str(row.get('Demand Country', ''))
                    
                    if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                        current_supply_class_for_y1 = agg_supply[2:].strip()
                    elif demand_country == 'Total' and agg_supply:
                        current_supply_class_for_y1 = agg_supply
                    
                    if agg_supply.startswith('▶') or agg_supply.startswith('▼') or demand_country == 'Total':
                        display_df.at[idx, 'Δ 30D Y/Y'] = 0
                    elif current_supply_class_for_y1 and current_supply_class_for_y1 in subtotal_values:
                        if '30D_Y1' in subtotal_values[current_supply_class_for_y1]:
                            subtotal_y1 = subtotal_values[current_supply_class_for_y1]['30D_Y1']
                            if subtotal_y1 != 0:
                                y1_pct = row['30D_Y1'] / subtotal_y1
                                display_df.at[idx, 'Δ 30D Y/Y'] = display_df.at[idx, '30D'] - y1_pct
                            else:
                                display_df.at[idx, 'Δ 30D Y/Y'] = 0
                        else:
                            display_df.at[idx, 'Δ 30D Y/Y'] = 0
            
            if '30D_Y1' in display_df.columns:
                display_df = display_df.drop('30D_Y1', axis=1)
        
        elif view_type == 'percentage' and classification_mode == 'Classification Level 1':
            text_cols = ['Aggregation Supply', 'Supply Country']
            delta_cols = ['Δ 7D-30D', 'Δ 30D Y/Y']
            numeric_cols = [col for col in display_df.columns if col not in text_cols and col not in delta_cols]
            numeric_cols_with_y1 = numeric_cols + (['30D_Y1'] if '30D_Y1' in display_df.columns else [])
            
            current_supply_class = None
            subtotal_values = {}
            
            for _, row in display_df.iterrows():
                agg_supply = str(row.get('Aggregation Supply', ''))
                if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                    supply_name = agg_supply[2:].strip()
                    subtotal_values[supply_name] = {col: row[col] for col in numeric_cols_with_y1 if col in row}
            
            for idx, row in display_df.iterrows():
                agg_supply = str(row.get('Aggregation Supply', ''))
                supply_country = str(row.get('Supply Country', ''))
                
                if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                    current_supply_class = agg_supply[2:].strip()
                
                if current_supply_class and current_supply_class in subtotal_values:
                    for col in numeric_cols:
                        if col in display_df.columns and col != '30D_Y1' and col in subtotal_values[current_supply_class]:
                            subtotal_val = subtotal_values[current_supply_class][col]
                            if subtotal_val != 0:
                                if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                                    display_df.at[idx, col] = 1.0
                                else:
                                    display_df.at[idx, col] = (row[col] / subtotal_val)
            
            if 'Δ 7D-30D' in display_df.columns and '7D' in display_df.columns and '30D' in display_df.columns:
                display_df['Δ 7D-30D'] = (display_df['7D'] - display_df['30D'])
            
            if 'Δ 30D Y/Y' in display_df.columns and '30D_Y1' in display_df.columns:
                current_supply_class_for_y1 = None
                for idx, row in display_df.iterrows():
                    agg_supply = str(row.get('Aggregation Supply', ''))
                    if agg_supply.startswith('▶') or agg_supply.startswith('▼'):
                        current_supply_class_for_y1 = agg_supply[2:].strip()
                        display_df.at[idx, 'Δ 30D Y/Y'] = 0
                    elif current_supply_class_for_y1 and current_supply_class_for_y1 in subtotal_values:
                        if '30D_Y1' in subtotal_values[current_supply_class_for_y1]:
                            subtotal_y1 = subtotal_values[current_supply_class_for_y1]['30D_Y1']
                            if subtotal_y1 != 0:
                                y1_pct = row['30D_Y1'] / subtotal_y1
                                display_df.at[idx, 'Δ 30D Y/Y'] = display_df.at[idx, '30D'] - y1_pct
                            else:
                                display_df.at[idx, 'Δ 30D Y/Y'] = 0
                        else:
                            display_df.at[idx, 'Δ 30D Y/Y'] = 0
            
            if '30D_Y1' in display_df.columns:
                display_df = display_df.drop('30D_Y1', axis=1)
        
        elif view_type == 'percentage' and classification_mode != 'Classification Level 1' and show_demand_country:
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
        
        elif view_type == 'percentage':
            text_cols = ['Supply Country']
            delta_cols = ['Δ 7D-30D', 'Δ 30D Y/Y']
            numeric_cols = [col for col in display_df.columns if col not in text_cols and col not in delta_cols]
            total_values = {col: display_df[col].sum() for col in numeric_cols if col in display_df.columns}
            
            for idx, row in display_df.iterrows():
                for col in numeric_cols:
                    total_val = total_values.get(col, 0)
                    if total_val != 0:
                        display_df.at[idx, col] = row[col] / total_val
                    else:
                        display_df.at[idx, col] = 0
            
            if 'Δ 7D-30D' in display_df.columns and '7D' in display_df.columns and '30D' in display_df.columns:
                display_df['Δ 7D-30D'] = (display_df['7D'] - display_df['30D'])
            
            if 'Δ 30D Y/Y' in display_df.columns and '30D_Y1' in display_df.columns:
                total_y1 = total_values.get('30D_Y1', 0)
                if total_y1 != 0:
                    display_df['Δ 30D Y/Y'] = display_df['30D'] - (display_df['30D_Y1'] / total_y1)
                else:
                    display_df['Δ 30D Y/Y'] = 0
            
            if '30D_Y1' in display_df.columns:
                display_df = display_df.drop('30D_Y1', axis=1)
        
        # Get conditional styles (includes alternating rows, country totals, grand total)
        conditional_styles = get_table_conditional_styles()
        
        # Add style for indented rows
        if 'Aggregation Supply' in display_df.columns:
            conditional_styles.insert(1, {
                'if': {'filter_query': '{Aggregation Supply} = ""'},
                'backgroundColor': '#f9f9f9',
                'fontSize': '13px'
            })
        
        # Add style for subtotal rows (bold format like Installation Total)
        # Works for both Classification Level 1 mode (Aggregation Demand = Total) and Country mode (Demand Country = Total)
        if 'Aggregation Demand' in display_df.columns:
            conditional_styles.append({
                'if': {'filter_query': '{Aggregation Demand} = "Total"'},
                'backgroundColor': TABLE_COLORS['bg_lighter'],
                'fontWeight': 'bold',
                'color': TABLE_COLORS['text_primary']
            })
        
        if 'Demand Country' in display_df.columns:
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
        for rolling_col in ['30D', '7D']:
            if rolling_col in display_df.columns:
                conditional_styles.append({
                    'if': {
                        'column_id': rolling_col,
                        'filter_query': '{Aggregation Supply} = "GRAND TOTAL" or {Supply Country} = "GRAND TOTAL"'
                    },
                    'color': TABLE_COLORS['text_primary']
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

        default_sort_by = [{'column_id': '30D', 'direction': 'desc'}] if '30D' in display_df.columns else []
        current_sort_by = supply_dest_sort_by or default_sort_by
        current_sort_by = [
            sort_rule for sort_rule in current_sort_by
            if sort_rule.get('column_id') in display_df.columns
        ]
        if not current_sort_by:
            current_sort_by = default_sort_by
        
        # Create the DataTable with pattern matching ID for click handling
        table = dash_table.DataTable(
            id={'type': 'supply-dest-expandable-table', 'index': 'summary'},
            data=display_df.to_dict('records'),
            columns=columns,
            page_size=100,
            sort_action='native',
            sort_mode='multi',
            sort_by=current_sort_by,
            fill_width=False,
            export_format='none',
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


@callback(
    [Output('supply-dest-period-table-container', 'children'),
     Output('supply-dest-period-table-instructions', 'children')],
    [Input('supply-dest-period-data-store', 'data'),
     Input('supply-dest-expanded-classifications', 'data'),
     Input('supply-dest-expanded-countries', 'data'),
     Input('supply-dest-expanded-supply-countries', 'data'),
     Input('country-classification-mode', 'data'),
     Input('supply-dest-view-type', 'value'),
     Input('aggregation-demand-dropdown', 'value'),
     Input('supply-dest-period-view-dropdown', 'value'),
     Input('supply-dest-period-count-dropdown', 'value'),
     Input('supply-dest-comparison-basis-dropdown', 'value'),
     Input('supply-dest-country-grouping-dropdown', 'value'),
     Input('supply-dest-period-sort-by', 'data')],
    prevent_initial_call=False
)
def update_supply_dest_period_table(supply_dest_period_data, expanded_classifications,
                                    expanded_countries, expanded_supply_countries,
                                    classification_mode, view_type, demand_aggregation_mode,
                                    period_view, period_count, comparison_basis,
                                    country_grouping_mode,
                                    supply_dest_period_sort_by):
    """Render the separate historical supply-destination table section."""
    return _render_supply_dest_period_analysis_table(
        supply_dest_period_data,
        classification_mode,
        demand_aggregation_mode,
        view_type,
        period_view,
        period_count,
        comparison_basis,
        expanded_classifications,
        expanded_countries,
        expanded_supply_countries,
        supply_dest_period_sort_by,
        country_grouping_mode
    )


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
    Output('supply-dest-sort-by', 'data', allow_duplicate=True),
    Input({'type': 'supply-dest-expandable-table', 'index': ALL}, 'sort_by'),
    prevent_initial_call=True
)
def store_supply_dest_sort_by(sort_by_values):
    """Persist the supply-destination table sort so refreshes keep the current order."""
    if not sort_by_values:
        return no_update

    for sort_by in sort_by_values:
        if sort_by:
            return sort_by

    return no_update


@callback(
    Output('supply-dest-period-sort-by', 'data', allow_duplicate=True),
    Input({'type': 'supply-dest-period-expandable-table', 'index': ALL}, 'sort_by'),
    State('supply-dest-period-sort-by', 'data'),
    prevent_initial_call=True
)
def store_supply_dest_period_sort_by(sort_by_values, current_sort_by):
    """Persist the historical period-analysis table sort independently from the overview."""
    if not sort_by_values:
        return no_update

    for sort_by in sort_by_values:
        if sort_by is not None:
            if sort_by == current_sort_by:
                return no_update
            return sort_by

    if current_sort_by == []:
        return no_update
    return []


def _toggle_supply_dest_row_expansion(display_df, active_cell, classification_mode,
                                      demand_aggregation_mode, expanded_classifications=None,
                                      expanded_countries=None, expanded_supply_countries=None):
    """Toggle the hierarchical expansion state for a clicked supply-destination row."""
    expanded_classifications = list(expanded_classifications or [])
    expanded_countries = list(expanded_countries or [])
    expanded_supply_countries = list(expanded_supply_countries or [])

    if display_df is None or display_df.empty or not active_cell:
        return expanded_classifications, expanded_countries, expanded_supply_countries

    row_index = active_cell.get('row')
    if row_index is None or row_index >= len(display_df):
        return expanded_classifications, expanded_countries, expanded_supply_countries

    show_demand_aggregation = use_demand_classification_mode(classification_mode, demand_aggregation_mode)
    show_demand_country = use_demand_country_mode(demand_aggregation_mode)
    clicked_row = display_df.iloc[row_index]

    if classification_mode == 'Classification Level 1' and show_demand_aggregation:
        supply_agg = clicked_row.get('Aggregation Supply', '')
        demand_agg = clicked_row.get('Aggregation Demand', '')

        if supply_agg and (supply_agg.startswith('▶') or supply_agg.startswith('▼')):
            supply_class = supply_agg[2:].strip()
            pair_key = f"{supply_class}→{demand_agg}"

            if pair_key in expanded_classifications:
                expanded_classifications.remove(pair_key)
                expanded_countries = [c for c in expanded_countries if not c.startswith(pair_key)]
                expanded_supply_countries = [
                    c for c in expanded_supply_countries if not c.startswith(pair_key)
                ]
            else:
                expanded_classifications.append(pair_key)

        country_demand = clicked_row.get('Country Demand', '')
        if country_demand and country_demand.strip() and (
            country_demand.strip().startswith('▶') or country_demand.strip().startswith('▼')
        ):
            demand_country = country_demand.strip()[2:].strip()

            for i in range(row_index - 1, -1, -1):
                prev_row = display_df.iloc[i]
                prev_supply = prev_row.get('Aggregation Supply', '')
                prev_demand = prev_row.get('Aggregation Demand', '')
                if prev_supply and (prev_supply.startswith('▶') or prev_supply.startswith('▼')):
                    supply_class = prev_supply[2:].strip()
                    pair_key = f"{supply_class}→{prev_demand}"
                    country_key = f"{pair_key}→{demand_country}"

                    if country_key in expanded_countries:
                        expanded_countries.remove(country_key)
                        expanded_supply_countries = [
                            c for c in expanded_supply_countries if not c.startswith(country_key)
                        ]
                    else:
                        expanded_countries.append(country_key)
                    break

    elif classification_mode == 'Classification Level 1' and show_demand_country:
        supply_agg = clicked_row.get('Aggregation Supply', '')
        demand_country = clicked_row.get('Demand Country', '')

        if supply_agg and (supply_agg.startswith('▶') or supply_agg.startswith('▼')):
            supply_class = supply_agg[2:].strip()
            if supply_class in expanded_classifications:
                expanded_classifications.remove(supply_class)
                expanded_countries = [
                    c for c in expanded_countries
                    if not c.startswith(f"{supply_class}→")
                ]
            else:
                expanded_classifications.append(supply_class)

        if demand_country and demand_country.strip() and (
            demand_country.strip().startswith('▶') or demand_country.strip().startswith('▼')
        ):
            demand_country_name = demand_country.strip()[2:].strip()

            for i in range(row_index - 1, -1, -1):
                prev_row = display_df.iloc[i]
                prev_supply = prev_row.get('Aggregation Supply', '')
                if prev_supply and (prev_supply.startswith('▶') or prev_supply.startswith('▼')):
                    supply_class = prev_supply[2:].strip()
                    country_key = f"{supply_class}→{demand_country_name}"
                    if country_key in expanded_countries:
                        expanded_countries.remove(country_key)
                    else:
                        expanded_countries.append(country_key)
                    break

    elif classification_mode == 'Classification Level 1':
        supply_agg = clicked_row.get('Aggregation Supply', '')
        if supply_agg and (supply_agg.startswith('▶') or supply_agg.startswith('▼')):
            supply_class = supply_agg[2:].strip()
            if supply_class in expanded_classifications:
                expanded_classifications.remove(supply_class)
            else:
                expanded_classifications.append(supply_class)

    return expanded_classifications, expanded_countries, expanded_supply_countries


@callback(
    [Output('supply-dest-expanded-classifications', 'data', allow_duplicate=True),
     Output('supply-dest-expanded-countries', 'data', allow_duplicate=True),
     Output('supply-dest-expanded-supply-countries', 'data', allow_duplicate=True)],
    [Input({'type': 'supply-dest-expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'supply-dest-expandable-table', 'index': ALL}, 'derived_viewport_data'),
     State('supply-dest-expanded-classifications', 'data'),
     State('supply-dest-expanded-countries', 'data'),
     State('supply-dest-expanded-supply-countries', 'data'),
     State('supply-dest-data-store', 'data'),
     State('country-classification-mode', 'data'),
     State('aggregation-demand-dropdown', 'value'),
     State('supply-dest-country-grouping-dropdown', 'value'),
     State('supply-dest-view-type', 'value')],
    prevent_initial_call=True
)
def handle_supply_dest_row_expansion(active_cells, table_data_list, expanded_classifications,
                                     expanded_countries, expanded_supply_countries,
                                     supply_dest_data, classification_mode,
                                     demand_aggregation_mode, country_grouping_mode, view_type):
    """Handle clicking on rows to expand/collapse in the overview supply-destination table."""
    ctx = callback_context
    if not ctx.triggered:
        return expanded_classifications, expanded_countries, expanded_supply_countries

    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']

    if 'supply-dest-expandable-table' in prop_id and '.active_cell' in prop_id:
        try:
            active_cell = triggered['value']
            if not active_cell:
                return expanded_classifications, expanded_countries, expanded_supply_countries

            current_table_data = None
            if table_data_list:
                for table_data in table_data_list:
                    if table_data is not None:
                        current_table_data = table_data
                        break

            if current_table_data:
                display_df = pd.DataFrame(current_table_data)
            elif supply_dest_data:
                resolved_supply_dest_data = _resolve_supply_dest_summary_payload(
                    supply_dest_data,
                    country_grouping_mode
                )
                df = pd.DataFrame(resolved_supply_dest_data)
                display_df, _ = prepare_supply_dest_table_for_display(
                    df, 'summary', classification_mode,
                    expanded_classifications, expanded_countries, expanded_supply_countries, view_type,
                    demand_aggregation_mode
                )
            else:
                display_df = pd.DataFrame()

            return _toggle_supply_dest_row_expansion(
                display_df,
                active_cell,
                classification_mode,
                demand_aggregation_mode,
                expanded_classifications,
                expanded_countries,
                expanded_supply_countries
            )
        except Exception:
            pass

    return expanded_classifications, expanded_countries, expanded_supply_countries


@callback(
    [Output('supply-dest-expanded-classifications', 'data', allow_duplicate=True),
     Output('supply-dest-expanded-countries', 'data', allow_duplicate=True),
     Output('supply-dest-expanded-supply-countries', 'data', allow_duplicate=True)],
    [Input({'type': 'supply-dest-period-expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'supply-dest-period-expandable-table', 'index': ALL}, 'derived_viewport_data'),
     State('supply-dest-expanded-classifications', 'data'),
     State('supply-dest-expanded-countries', 'data'),
     State('supply-dest-expanded-supply-countries', 'data'),
     State('supply-dest-period-data-store', 'data'),
     State('country-classification-mode', 'data'),
     State('aggregation-demand-dropdown', 'value'),
     State('supply-dest-period-view-dropdown', 'value'),
     State('supply-dest-period-count-dropdown', 'value'),
     State('supply-dest-comparison-basis-dropdown', 'value'),
     State('supply-dest-country-grouping-dropdown', 'value'),
     State('supply-dest-view-type', 'value')],
    prevent_initial_call=True
)
def handle_supply_dest_period_row_expansion(active_cells, table_data_list, expanded_classifications,
                                            expanded_countries, expanded_supply_countries,
                                            supply_dest_period_data, classification_mode,
                                            demand_aggregation_mode, period_view, period_count,
                                            comparison_basis, country_grouping_mode, view_type):
    """Handle clicking on rows to expand/collapse in the right-side period-analysis table."""
    ctx = callback_context
    if not ctx.triggered:
        return expanded_classifications, expanded_countries, expanded_supply_countries

    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']

    if 'supply-dest-period-expandable-table' in prop_id and '.active_cell' in prop_id:
        try:
            active_cell = triggered['value']
            if not active_cell:
                return expanded_classifications, expanded_countries, expanded_supply_countries

            current_table_data = None
            if table_data_list:
                for table_data in table_data_list:
                    if table_data is not None:
                        current_table_data = table_data
                        break

            if current_table_data:
                display_df = pd.DataFrame(current_table_data)
            else:
                display_df, _, _ = _build_supply_dest_period_view_table(
                    supply_dest_period_data,
                    period_view,
                    period_count,
                    comparison_basis,
                    classification_mode,
                    demand_aggregation_mode,
                    expanded_classifications,
                    expanded_countries,
                    expanded_supply_countries,
                    view_type,
                    country_grouping_mode
                )

            return _toggle_supply_dest_row_expansion(
                display_df,
                active_cell,
                classification_mode,
                demand_aggregation_mode,
                expanded_classifications,
                expanded_countries,
                expanded_supply_countries
            )
        except Exception:
            pass

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

    # Get database connection
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

    # Create chart for each entity
    for entity_name in entities_list:
        # Create the figure using the original chart functions
        if chart_type == 'percentage':
            fig = create_continent_percentage_chart(entity_name, engine_inst, schema, classification_mode or 'Country')
        else:
            fig = create_continent_destination_chart(entity_name, engine_inst, schema, classification_mode or 'Country')

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


def _resolve_table_export_df(derived_virtual_data_list, data_list, columns_list):
    """Resolve the current DataTable payload into an ordered export dataframe."""
    selected_rows = None
    selected_columns = None

    row_sources = derived_virtual_data_list or []
    fallback_sources = data_list or []
    column_sources = columns_list or []

    for idx, column_defs in enumerate(column_sources):
        if column_defs:
            selected_columns = column_defs
            if idx < len(row_sources) and row_sources[idx] is not None:
                selected_rows = row_sources[idx]
            elif idx < len(fallback_sources) and fallback_sources[idx] is not None:
                selected_rows = fallback_sources[idx]
            break

    if selected_columns is None:
        for column_defs in column_sources:
            if column_defs:
                selected_columns = column_defs
                break

    if selected_rows is None:
        for rows in row_sources:
            if rows is not None:
                selected_rows = rows
                break
        if selected_rows is None:
            for rows in fallback_sources:
                if rows is not None:
                    selected_rows = rows
                    break

    if not selected_rows:
        return pd.DataFrame()

    export_df = pd.DataFrame(selected_rows)
    if selected_columns:
        visible_column_ids = [
            column['id'] for column in selected_columns
            if column.get('id') in export_df.columns
        ]
        if visible_column_ids:
            export_df = export_df[visible_column_ids]

    return export_df


def _send_export_dataframe(export_df, filename_prefix, sheet_name):
    """Create a single-sheet Excel download from a dataframe."""
    if export_df is None or export_df.empty:
        return None

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        worksheet = writer.sheets[sheet_name[:31]]
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    max_length = max(max_length, len(str(cell.value)))
                except Exception:
                    pass
            worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)

    output.seek(0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(output.getvalue(), f'{filename_prefix}_{timestamp}.xlsx')


@callback(
    Output('download-supply-dest-excel', 'data'),
    Input('export-supply-dest-button', 'n_clicks'),
    State({'type': 'supply-dest-expandable-table', 'index': ALL}, 'derived_virtual_data'),
    State({'type': 'supply-dest-expandable-table', 'index': ALL}, 'data'),
    State({'type': 'supply-dest-expandable-table', 'index': ALL}, 'columns'),
    prevent_initial_call=True
)
def export_supply_dest_table_to_excel(n_clicks, derived_virtual_data_list, data_list, columns_list):
    """Export the currently rendered LNG Supply by Destination table to Excel."""
    if not n_clicks:
        return None

    export_df = _resolve_table_export_df(derived_virtual_data_list, data_list, columns_list)
    return _send_export_dataframe(
        export_df,
        'LNG_Supply_by_Destination',
        'Supply by Destination'
    )


@callback(
    Output('download-supply-dest-period-excel', 'data'),
    Input('export-supply-dest-period-button', 'n_clicks'),
    State({'type': 'supply-dest-period-expandable-table', 'index': ALL}, 'derived_virtual_data'),
    State({'type': 'supply-dest-period-expandable-table', 'index': ALL}, 'data'),
    State({'type': 'supply-dest-period-expandable-table', 'index': ALL}, 'columns'),
    prevent_initial_call=True
)
def export_supply_dest_period_table_to_excel(n_clicks, derived_virtual_data_list, data_list, columns_list):
    """Export the currently rendered LNG Supply by Destination period-analysis table to Excel."""
    if not n_clicks:
        return None

    export_df = _resolve_table_export_df(derived_virtual_data_list, data_list, columns_list)
    return _send_export_dataframe(
        export_df,
        'LNG_Supply_by_Destination_Period_Analysis',
        'Period Analysis'
    )


# Export callback for Supply Charts
@callback(
    Output('download-supply-charts-excel', 'data'),
    Input('export-supply-charts-button', 'n_clicks'),
    State('supply-charts-data', 'data'),
    State('country-classification-mode', 'data'),
    prevent_initial_call=True
)
def export_supply_charts_to_excel(n_clicks, charts_data, classification_mode):
    """Export LNG Supply 30-Day Rolling Average data to Excel"""
    if n_clicks == 0 or not charts_data:
        return None

    # Convert all entities' data to DataFrames
    all_data = []
    for entity_name, entity_data in charts_data.items():
        if entity_data:
            df = pd.DataFrame(entity_data)
            df['entity'] = entity_name
            all_data.append(df)

    if not all_data:
        return None

    # Create Excel file with BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Combined sheet with all data
        combined_df = safe_concat(all_data, ignore_index=True)
        # Reorder columns for better readability
        cols = ['entity', 'date', 'year', 'month_day', 'rolling_avg', 'is_forecast']
        cols = [c for c in cols if c in combined_df.columns]
        combined_df = combined_df[cols]
        combined_df.to_excel(writer, sheet_name='All Data', index=False)

        # Individual sheets per entity
        for entity_name, entity_data in charts_data.items():
            if entity_data:
                df = pd.DataFrame(entity_data)
                # Excel sheet name limit is 31 characters
                sheet_name = entity_name[:31].replace('/', '-').replace('\\', '-')
                sheet_cols = ['date', 'year', 'month_day', 'rolling_avg', 'is_forecast']
                sheet_cols = [c for c in sheet_cols if c in df.columns]
                df = df[sheet_cols]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    output.seek(0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'LNG_Supply_30D_Rolling_{timestamp}.xlsx'

    return dcc.send_bytes(output.getvalue(), filename)


# Export callback for Continent Charts
@callback(
    Output('download-continent-charts-excel', 'data'),
    Input('export-continent-charts-button', 'n_clicks'),
    State('continent-charts-data', 'data'),
    State('continent-chart-type', 'value'),
    State('country-classification-mode', 'data'),
    prevent_initial_call=True
)
def export_continent_charts_to_excel(n_clicks, entities_list, chart_type, classification_mode):
    """Export LNG Supply by Destination Continent data to Excel"""
    if n_clicks == 0 or not entities_list:
        return None

    engine_inst, schema = setup_database_connection()

    all_data = []
    for entity_name in entities_list:
        try:
            # Handle Global case
            if entity_name == "Global":
                country_filter = ""
            elif classification_mode == 'Classification Level 1':
                country_filter = "AND mc.country_classification_level1 = %(entity_name)s"
            else:
                country_filter = "AND kt.origin_country_name = %(entity_name)s"

            # Add join for classification mode
            join_clause = ""
            internal_flow_filter = """
                        AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                            IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
            """
            if classification_mode == 'Classification Level 1':
                if entity_name != "Global":
                    join_clause = f"""
                    INNER JOIN {schema}.mappings_country mc ON kt.origin_country_name = mc.country
                    LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                    """
                else:
                    join_clause = f"""
                    LEFT JOIN {schema}.mappings_country mc ON kt.origin_country_name = mc.country
                    LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                    """
                internal_flow_filter = """
                        AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                            IS DISTINCT FROM COALESCE(mc.country_classification_level1, 'Unknown')
                """

            # Use different query based on chart_type
            if chart_type == 'percentage':
                # Query with percentage calculation (matches create_continent_percentage_chart)
                query = f"""
                WITH latest_data AS (
                    SELECT MAX(upload_timestamp_utc) as max_timestamp
                    FROM {schema}.kpler_trades
                ),
                all_continents AS (
                    SELECT DISTINCT
                        COALESCE(NULLIF(continent_destination_name, ''), 'Unknown') as continent_destination
                    FROM {schema}.kpler_trades kt
                    {join_clause}
                    , latest_data ld
                    WHERE kt.upload_timestamp_utc = ld.max_timestamp
                        {country_filter}
                        AND kt.start >= '2023-11-01'
                        {internal_flow_filter}
                ),
                all_dates AS (
                    SELECT generate_series(
                        '2023-11-01'::date,
                        (CURRENT_DATE + INTERVAL '14 days')::date,
                        '1 day'::interval
                    )::date as date
                ),
                date_continent_matrix AS (
                    SELECT
                        d.date,
                        c.continent_destination
                    FROM all_dates d
                    CROSS JOIN all_continents c
                ),
                daily_exports_raw AS (
                    SELECT
                        kt.start::date as date,
                        COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination,
                        SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
                    FROM {schema}.kpler_trades kt
                    {join_clause}
                    , latest_data ld
                    WHERE kt.upload_timestamp_utc = ld.max_timestamp
                        {country_filter}
                        AND kt.start >= '2023-11-01'
                        AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                        {internal_flow_filter}
                    GROUP BY kt.start::date, COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown')
                ),
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
            else:
                # Query for absolute values (original query)
                query = f"""
                WITH latest_data AS (
                    SELECT MAX(upload_timestamp_utc) as max_timestamp
                    FROM {schema}.kpler_trades
                ),
                all_continents AS (
                    SELECT DISTINCT
                        COALESCE(NULLIF(continent_destination_name, ''), 'Unknown') as continent_destination
                    FROM {schema}.kpler_trades kt
                    {join_clause}
                    , latest_data ld
                    WHERE kt.upload_timestamp_utc = ld.max_timestamp
                        {country_filter}
                        AND kt.start >= '2023-11-01'
                        {internal_flow_filter}
                ),
                all_dates AS (
                    SELECT generate_series(
                        '2023-11-01'::date,
                        (CURRENT_DATE + INTERVAL '14 days')::date,
                        '1 day'::interval
                    )::date as date
                ),
                date_continent_matrix AS (
                    SELECT
                        d.date,
                        c.continent_destination
                    FROM all_dates d
                    CROSS JOIN all_continents c
                ),
                daily_exports_raw AS (
                    SELECT
                        kt.start::date as date,
                        COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination,
                        SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
                    FROM {schema}.kpler_trades kt
                    {join_clause}
                    , latest_data ld
                    WHERE kt.upload_timestamp_utc = ld.max_timestamp
                        {country_filter}
                        AND kt.start >= '2023-11-01'
                        AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                        {internal_flow_filter}
                    GROUP BY kt.start::date, COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown')
                ),
                daily_exports_complete AS (
                    SELECT
                        m.date,
                        m.continent_destination,
                        COALESCE(e.daily_export_mcmd, 0) as daily_export_mcmd
                    FROM date_continent_matrix m
                    LEFT JOIN daily_exports_raw e
                        ON m.date = e.date AND m.continent_destination = e.continent_destination
                ),
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
            df = pd.read_sql(query, engine_inst, params=params)

            if not df.empty:
                df['entity'] = entity_name
                all_data.append(df)

        except Exception as e:
            continue

    if not all_data:
        return None

    # Create Excel file with BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Combined sheet with all data
        combined_df = safe_concat(all_data, ignore_index=True)
        # Use different columns based on chart_type
        if chart_type == 'percentage':
            cols = ['entity', 'date', 'continent_destination', 'year', 'month_day', 'percentage', 'is_forecast']
        else:
            cols = ['entity', 'date', 'continent_destination', 'year', 'month_day', 'rolling_avg', 'is_forecast']
        cols = [c for c in cols if c in combined_df.columns]
        combined_df = combined_df[cols]
        combined_df.to_excel(writer, sheet_name='All Data', index=False)

        # Individual sheets per entity
        for entity_df in all_data:
            entity_name = entity_df['entity'].iloc[0]
            sheet_name = entity_name[:31].replace('/', '-').replace('\\', '-')
            if chart_type == 'percentage':
                sheet_cols = ['date', 'continent_destination', 'year', 'month_day', 'percentage', 'is_forecast']
            else:
                sheet_cols = ['date', 'continent_destination', 'year', 'month_day', 'rolling_avg', 'is_forecast']
            sheet_cols = [c for c in sheet_cols if c in entity_df.columns]
            entity_df[sheet_cols].to_excel(writer, sheet_name=sheet_name, index=False)

        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    output.seek(0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_type_label = 'Absolute' if chart_type == 'absolute' else 'Percentage'
    filename = f'LNG_Supply_by_Continent_{chart_type_label}_{timestamp}.xlsx'

    return dcc.send_bytes(output.getvalue(), filename)
