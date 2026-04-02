from dash import html, dcc, callback, Output, Input
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import configparser
import os
import sys
from sqlalchemy import create_engine

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

from fundamentals.terminals.scenario_utils import get_available_scenarios

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


# --- Essential Variable Checks ---
if not DB_CONNECTION_STRING:
    raise ValueError(f"Missing DATABASE CONNECTION_STRING in {CONFIG_FILE_PATH}")

# create engine
engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)


# Professional color palette (McKinsey-style) - matching original
PRIMARY_COLORS = {
    'US (Gulf Coast)': '#003A6C',  # Navy Blue
    'Canada (British Columbia)': '#00A3E0',  # Light Blue
    'Mexico': '#6BCABA',  # Teal
    'Argentina': '#90B23C',  # Green
    'Mauritania': '#FFC72C',  # Yellow
    'Senegal': '#FFC72C',  # Yellow
    'Gabon': '#F58220',  # Orange
    'Congo': '#E03C31',  # Red
    'Nigeria': '#7F3F98',  # Purple
    'Qatar': '#A21F5A',  # Burgundy
    'Malaysia': '#005EB8',  # Royal Blue
    'Indonesia': '#00B5E2',  # Sky Blue
    'Australia': '#78BE20',  # Lime Green
    'United States': '#003A6C',  # Navy Blue
    'Canada': '#00A3E0',  # Light Blue
}


def fetch_train_data(scenario='base_view'):
    """
    Fetch train data from database with scenario-based adjustments.

    Start dates are derived from actual volume data:
    - For base_view: First month with volume > 0 in Woodmac baseline
    - For other scenarios: First month with volume > 0 (considering adjustments)

    Args:
        scenario: Scenario name ('base_view', 'best_view', 'test_1', etc.)
                 'base_view' returns Woodmac baseline only
                 Other scenarios derive start dates from volume adjustments

    Returns:
        DataFrame with train information including derived start dates
    """
    if scenario == 'base_view':
        # Base view: Woodmac only, start dates from first volume > 0
        # Optimized query: reduced CTEs, combined operations
        query = f"""
        WITH latest_trains AS (
            SELECT DISTINCT ON (id_plant, id_lng_train)
                id_plant,
                id_lng_train,
                lng_train_date_start_est
            FROM {DB_SCHEMA}.woodmac_lng_plant_train
            WHERE lng_train_date_start_est IS NOT NULL
            ORDER BY id_plant, id_lng_train, upload_timestamp_utc DESC
        ),
        latest_plants AS (
            SELECT DISTINCT ON (id_plant)
                id_plant,
                plant_name,
                country_name
            FROM {DB_SCHEMA}.woodmac_lng_plant_summary
            ORDER BY id_plant, upload_timestamp_utc DESC
        ),
        -- Get latest monthly capacity with max and start date in one pass
        monthly_capacity AS (
            SELECT
                id_plant,
                id_lng_train,
                MAX(metric_value) as max_capacity
            FROM (
                SELECT DISTINCT ON (id_plant, id_lng_train, year, month)
                    id_plant, id_lng_train, metric_value
                FROM {DB_SCHEMA}.woodmac_lng_plant_monthly_capacity_nominal_mta
                ORDER BY id_plant, id_lng_train, year, month, upload_timestamp_utc DESC
            ) c
            GROUP BY id_plant, id_lng_train
        ),
        -- Get first volume date from monthly output
        monthly_start_dates AS (
            SELECT
                id_plant,
                id_lng_train,
                MIN(TO_DATE(year || '-' || LPAD(month::text, 2, '0') || '-01', 'YYYY-MM-DD')) as start_date
            FROM (
                SELECT DISTINCT ON (id_plant, id_lng_train, year, month)
                    id_plant, id_lng_train, year, month, metric_value
                FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_output_mta
                ORDER BY id_plant, id_lng_train, year, month, upload_timestamp_utc DESC
            ) o
            WHERE metric_value > 0
            GROUP BY id_plant, id_lng_train
        ),
        -- Get trains with monthly data (for exclusion)
        trains_with_monthly AS (
            SELECT DISTINCT id_plant, id_lng_train FROM monthly_capacity
            UNION
            SELECT DISTINCT id_plant, id_lng_train FROM monthly_start_dates
        ),
        -- Annual data for trains WITHOUT any monthly data
        annual_data AS (
            SELECT
                a.id_plant,
                a.id_lng_train,
                MAX(a.metric_value) as max_capacity,
                MIN(CASE WHEN a.metric_value > 0
                    THEN TO_DATE(a.year || '-01-01', 'YYYY-MM-DD')
                END) as start_date
            FROM (
                SELECT DISTINCT ON (id_plant, id_lng_train, year)
                    id_plant, id_lng_train, year, metric_value
                FROM {DB_SCHEMA}.woodmac_lng_plant_train_annual_output_mta
                ORDER BY id_plant, id_lng_train, year, upload_timestamp_utc DESC
            ) a
            WHERE NOT EXISTS (
                SELECT 1 FROM trains_with_monthly m
                WHERE m.id_plant = a.id_plant AND m.id_lng_train = a.id_lng_train
            )
            GROUP BY a.id_plant, a.id_lng_train
        )
        SELECT
            p.plant_name,
            p.country_name,
            t.id_plant,
            t.id_lng_train,
            COALESCE(msd.start_date, ad.start_date, t.lng_train_date_start_est::date) as lng_train_date_start_est,
            COALESCE(mc.max_capacity, ad.max_capacity) as capacity,
            t.lng_train_date_start_est::date as woodmac_date,
            NULL::date as internal_date,
            'woodmac' as data_source
        FROM latest_trains t
        JOIN latest_plants p ON t.id_plant = p.id_plant
        LEFT JOIN monthly_capacity mc ON t.id_plant = mc.id_plant AND t.id_lng_train = mc.id_lng_train
        LEFT JOIN monthly_start_dates msd ON t.id_plant = msd.id_plant AND t.id_lng_train = msd.id_lng_train
        LEFT JOIN annual_data ad ON t.id_plant = ad.id_plant AND t.id_lng_train = ad.id_lng_train
        WHERE COALESCE(mc.max_capacity, ad.max_capacity) IS NOT NULL
        ORDER BY p.country_name, p.plant_name, COALESCE(msd.start_date, ad.start_date, t.lng_train_date_start_est::date)
        """
    else:
        # Other scenarios: Derive start dates from volumes including adjustments
        # Integrates annual data for capacity and start dates where monthly data doesn't exist
        query = f"""
        WITH latest_trains AS (
            SELECT DISTINCT ON (id_plant, id_lng_train)
                id_plant,
                id_lng_train,
                lng_train_date_start_est,
                upload_timestamp_utc
            FROM {DB_SCHEMA}.woodmac_lng_plant_train
            ORDER BY id_plant, id_lng_train, upload_timestamp_utc DESC
        ),
        latest_plant_summary AS (
            SELECT DISTINCT ON (id_plant)
                id_plant,
                plant_name,
                country_name,
                upload_timestamp_utc
            FROM {DB_SCHEMA}.woodmac_lng_plant_summary
            ORDER BY id_plant, upload_timestamp_utc DESC
        ),
        latest_monthly_capacity AS (
            SELECT DISTINCT ON (id_plant, id_lng_train, year, month)
                id_plant,
                id_lng_train,
                year,
                month,
                metric_value,
                upload_timestamp_utc
            FROM {DB_SCHEMA}.woodmac_lng_plant_monthly_capacity_nominal_mta
            ORDER BY id_plant, id_lng_train, year, month, upload_timestamp_utc DESC
        ),
        trains_with_monthly_capacity AS (
            SELECT DISTINCT id_plant, id_lng_train
            FROM latest_monthly_capacity
        ),
        capacity_max_monthly AS (
            SELECT
                id_plant,
                id_lng_train,
                MAX(metric_value) as max_capacity
            FROM latest_monthly_capacity
            GROUP BY id_plant, id_lng_train
        ),
        latest_annual_output AS (
            SELECT DISTINCT ON (id_plant, id_lng_train, year)
                id_plant,
                id_lng_train,
                year,
                metric_value
            FROM {DB_SCHEMA}.woodmac_lng_plant_train_annual_output_mta
            ORDER BY id_plant, id_lng_train, year, upload_timestamp_utc DESC
        ),
        capacity_max_annual AS (
            SELECT
                a.id_plant,
                a.id_lng_train,
                MAX(a.metric_value) as max_capacity
            FROM latest_annual_output a
            WHERE NOT EXISTS (
                SELECT 1 FROM trains_with_monthly_capacity mc
                WHERE mc.id_plant = a.id_plant AND mc.id_lng_train = a.id_lng_train
            )
            GROUP BY a.id_plant, a.id_lng_train
        ),
        capacity_max AS (
            SELECT id_plant, id_lng_train, max_capacity FROM capacity_max_monthly
            UNION ALL
            SELECT id_plant, id_lng_train, max_capacity FROM capacity_max_annual
        ),
        woodmac_baseline AS (
            SELECT
                b.plant_name,
                b.country_name,
                a.lng_train_date_start_est,
                cm.max_capacity as capacity,
                a.id_plant,
                a.id_lng_train
            FROM latest_trains a
            JOIN latest_plant_summary b ON a.id_plant = b.id_plant
            JOIN capacity_max cm ON a.id_plant = cm.id_plant AND a.id_lng_train = cm.id_lng_train
            WHERE a.lng_train_date_start_est IS NOT NULL
        ),
        latest_output AS (
            SELECT DISTINCT ON (id_plant, id_lng_train, year, month)
                id_plant,
                id_lng_train,
                year,
                month,
                metric_value,
                upload_timestamp_utc
            FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_output_mta
            ORDER BY id_plant, id_lng_train, year, month, upload_timestamp_utc DESC
        ),
        baseline_volumes AS (
            SELECT
                id_plant,
                id_lng_train,
                year,
                month,
                metric_value as volume
            FROM latest_output
        ),
        latest_adjustments AS (
            SELECT DISTINCT ON (id_plant, id_lng_train, year, month)
                id_plant,
                id_lng_train,
                year,
                month,
                adjusted_output
            FROM {DB_SCHEMA}.fundamentals_terminals_output_adjustments
            WHERE scenario_name = %(scenario)s
            ORDER BY id_plant, id_lng_train, year, month, upload_timestamp_utc DESC
        ),
        trains_with_adjustments AS (
            SELECT DISTINCT id_plant, id_lng_train
            FROM latest_adjustments
        ),
        combined_volumes AS (
            SELECT
                bv.id_plant,
                bv.id_lng_train,
                bv.year,
                bv.month,
                COALESCE(la.adjusted_output, bv.volume) as final_volume
            FROM baseline_volumes bv
            LEFT JOIN latest_adjustments la
                ON bv.id_plant = la.id_plant
                AND bv.id_lng_train = la.id_lng_train
                AND bv.year = la.year
                AND bv.month = la.month
        ),
        trains_with_monthly_output AS (
            SELECT DISTINCT id_plant, id_lng_train
            FROM combined_volumes
        ),
        first_volume_date_monthly AS (
            SELECT
                id_plant,
                id_lng_train,
                MIN(TO_DATE(year || '-' || LPAD(month::text, 2, '0') || '-01', 'YYYY-MM-DD')) as start_date
            FROM combined_volumes
            WHERE final_volume > 0
            GROUP BY id_plant, id_lng_train
        ),
        first_volume_date_annual AS (
            SELECT
                a.id_plant,
                a.id_lng_train,
                MIN(TO_DATE(a.year || '-01-01', 'YYYY-MM-DD')) as start_date
            FROM latest_annual_output a
            LEFT JOIN trains_with_monthly_output mo
                ON a.id_plant = mo.id_plant
                AND a.id_lng_train = mo.id_lng_train
            WHERE a.metric_value > 0
              AND mo.id_plant IS NULL
            GROUP BY a.id_plant, a.id_lng_train
        ),
        first_volume_date AS (
            SELECT * FROM first_volume_date_monthly
            UNION ALL
            SELECT * FROM first_volume_date_annual
        )
        SELECT
            wb.plant_name,
            wb.country_name,
            wb.id_plant,
            wb.id_lng_train,
            COALESCE(fvd.start_date, wb.lng_train_date_start_est::date) as lng_train_date_start_est,
            wb.capacity,
            wb.lng_train_date_start_est::date as woodmac_date,
            fvd.start_date as internal_date,
            CASE WHEN twa.id_plant IS NOT NULL AND fvd.start_date IS NOT NULL AND fvd.start_date != wb.lng_train_date_start_est::date
                 THEN 'adjusted' ELSE 'baseline' END as data_source
        FROM woodmac_baseline wb
        LEFT JOIN first_volume_date fvd ON wb.id_plant = fvd.id_plant AND wb.id_lng_train = fvd.id_lng_train
        LEFT JOIN trains_with_adjustments twa ON wb.id_plant = twa.id_plant AND wb.id_lng_train = twa.id_lng_train
        ORDER BY wb.country_name, wb.plant_name, COALESCE(fvd.start_date, wb.lng_train_date_start_est::date)
        """

    # Execute query and read data
    if scenario == 'base_view':
        df = pd.read_sql_query(query, engine)
    else:
        df = pd.read_sql_query(query, engine, params={'scenario': scenario})

    # Convert start_date to datetime
    df['lng_train_date_start_est'] = pd.to_datetime(df['lng_train_date_start_est'])

    # Filter out trains with zero or null capacity (likely placeholder data)
    df = df[df['capacity'] > 0]

    return df


def convert_to_mcmd(capacity_mtpa):
    """Convert MTPA to Mcm/d using formula: MTPA * 1.36 / 365 * 1000"""
    return capacity_mtpa * 1.36 / 365 * 1000


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple for rgba() formatting."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_new_capacity_filters(scenario='base_view'):
    """
    Get list of new projects and trains from timeline data (starting from current month onwards).

    Args:
        scenario: Scenario name for adjustments ('base_view', 'best_view', etc.)

    Returns:
        DataFrame with columns: plant_name, id_lng_train, lng_train_date_start_est
    """
    df = fetch_train_data(scenario=scenario)

    # Get current date (first day of current month)
    current_date = pd.Timestamp.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # Filter for trains starting from current month onwards
    # lng_train_date_start_est is already a datetime from fetch_train_data()
    new_trains_df = df[df['lng_train_date_start_est'] >= current_date][['plant_name', 'id_lng_train', 'lng_train_date_start_est', 'country_name']].copy()

    return new_trains_df


def fetch_volume_data(start_year=2025, end_year=2040, breakdown='country', new_capacity_only=False, selected_countries=None, scenario='base_view'):
    """
    Fetch monthly output volume data with different breakdown levels and scenario-based adjustments.

    Integrates both monthly and annual data:
    - Monthly data takes precedence where available (higher granularity)
    - Annual data fills gaps beyond monthly coverage (annual value = monthly average)

    Args:
        start_year: Starting year for data (default 2025)
        end_year: Ending year for data (default 2040)
        breakdown: 'country', 'project', or 'train'
        new_capacity_only: If True, filter to show only new capacity from current month onwards
        selected_countries: List of countries to filter by (None = all countries)
        scenario: Scenario name ('base_view', 'best_view', 'test_1', etc.)
                 'base_view' returns Woodmac baseline only
                 Other scenarios apply volume adjustments if they exist

    Returns:
        DataFrame with columns depending on breakdown level
    """
    # Fetch data with scenario-based adjustments
    if scenario == 'base_view':
        # Base view: Woodmac only, no adjustments
        query = f"""
        WITH latest_monthly AS (
            SELECT DISTINCT ON (id_plant, id_lng_train, year, month)
                year,
                month,
                plant_name,
                country_name,
                id_plant,
                id_lng_train,
                lng_train_name_short,
                metric_value,
                upload_timestamp_utc
            FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_output_mta
            ORDER BY id_plant, id_lng_train, year, month, upload_timestamp_utc DESC
        ),
        monthly_coverage_map AS (
            -- Track which year-train combinations have monthly data
            SELECT DISTINCT
                id_plant, id_lng_train, year,
                true as has_monthly_data
            FROM latest_monthly
        ),
        latest_annual AS (
            SELECT DISTINCT ON (id_plant, id_lng_train, year)
                id_plant,
                id_lng_train,
                year,
                plant_name,
                country_name,
                lng_train_name_short,
                metric_value
            FROM {DB_SCHEMA}.woodmac_lng_plant_train_annual_output_mta
            ORDER BY id_plant, id_lng_train, year, upload_timestamp_utc DESC
        ),
        annual_expanded AS (
            -- Expand annual to monthly, but ONLY where monthly doesn't exist
            -- Annual value IS the monthly average, so use it directly (no division)
            SELECT
                a.year,
                month_series.month,
                a.plant_name,
                a.country_name,
                a.id_plant,
                a.id_lng_train,
                a.lng_train_name_short,
                a.metric_value,
                'annual_expanded' as source_type
            FROM latest_annual a
            CROSS JOIN generate_series(1, 12) as month_series(month)
            LEFT JOIN monthly_coverage_map mc
                ON a.id_plant = mc.id_plant
                AND a.id_lng_train = mc.id_lng_train
                AND a.year = mc.year
            WHERE (mc.has_monthly_data IS NULL OR mc.has_monthly_data = false)
              AND a.year >= {start_year} AND a.year <= {end_year}
        ),
        monthly_data AS (
            SELECT
                year,
                month,
                plant_name,
                country_name,
                id_plant,
                id_lng_train,
                lng_train_name_short,
                metric_value,
                'monthly' as source_type
            FROM latest_monthly
            WHERE year >= {start_year} AND year <= {end_year}
        ),
        combined_data AS (
            SELECT * FROM monthly_data
            UNION ALL
            SELECT * FROM annual_expanded
        )
        SELECT
            year,
            month,
            plant_name,
            country_name,
            id_plant,
            id_lng_train,
            lng_train_name_short,
            metric_value as total_output,
            CASE
                WHEN source_type = 'annual_expanded' THEN 'annual_baseline'
                ELSE 'baseline'
            END as data_source
        FROM combined_data
        ORDER BY year, month, plant_name, id_lng_train
        """
        df = pd.read_sql_query(query, engine)
    else:
        # Other scenarios: Apply volume adjustments if they exist
        query = f"""
        WITH latest_monthly AS (
            SELECT DISTINCT ON (id_plant, id_lng_train, year, month)
                year,
                month,
                plant_name,
                country_name,
                id_plant,
                id_lng_train,
                lng_train_name_short,
                metric_value,
                upload_timestamp_utc
            FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_output_mta
            ORDER BY id_plant, id_lng_train, year, month, upload_timestamp_utc DESC
        ),
        monthly_coverage_map AS (
            SELECT DISTINCT
                id_plant, id_lng_train, year,
                true as has_monthly_data
            FROM latest_monthly
        ),
        latest_annual AS (
            SELECT DISTINCT ON (id_plant, id_lng_train, year)
                id_plant,
                id_lng_train,
                year,
                plant_name,
                country_name,
                lng_train_name_short,
                metric_value
            FROM {DB_SCHEMA}.woodmac_lng_plant_train_annual_output_mta
            ORDER BY id_plant, id_lng_train, year, upload_timestamp_utc DESC
        ),
        annual_expanded AS (
            -- Expand annual to monthly, but ONLY where monthly doesn't exist
            SELECT
                a.year,
                month_series.month,
                a.plant_name,
                a.country_name,
                a.id_plant,
                a.id_lng_train,
                a.lng_train_name_short,
                a.metric_value
            FROM latest_annual a
            CROSS JOIN generate_series(1, 12) as month_series(month)
            LEFT JOIN monthly_coverage_map mc
                ON a.id_plant = mc.id_plant
                AND a.id_lng_train = mc.id_lng_train
                AND a.year = mc.year
            WHERE (mc.has_monthly_data IS NULL OR mc.has_monthly_data = false)
              AND a.year >= {start_year} AND a.year <= {end_year}
        ),
        monthly_data AS (
            SELECT
                year,
                month,
                plant_name,
                country_name,
                id_plant,
                id_lng_train,
                lng_train_name_short,
                metric_value
            FROM latest_monthly
            WHERE year >= {start_year} AND year <= {end_year}
        ),
        woodmac_baseline AS (
            SELECT * FROM monthly_data
            UNION ALL
            SELECT * FROM annual_expanded
        ),
        latest_adjustments AS (
            SELECT DISTINCT ON (id_plant, id_lng_train, year, month)
                id_plant,
                id_lng_train,
                year,
                month,
                adjusted_output
            FROM {DB_SCHEMA}.fundamentals_terminals_output_adjustments
            WHERE scenario_name = %(scenario)s
            ORDER BY id_plant, id_lng_train, year, month, upload_timestamp_utc DESC
        )
        SELECT
            wb.year,
            wb.month,
            wb.plant_name,
            wb.country_name,
            wb.id_plant,
            wb.id_lng_train,
            wb.lng_train_name_short,
            COALESCE(la.adjusted_output, wb.metric_value) as total_output,
            CASE
                WHEN la.adjusted_output IS NOT NULL THEN 'adjusted'
                ELSE 'baseline'
            END as data_source
        FROM woodmac_baseline wb
        LEFT JOIN latest_adjustments la
            ON wb.id_plant = la.id_plant
            AND wb.id_lng_train = la.id_lng_train
            AND wb.year = la.year
            AND wb.month = la.month
        ORDER BY wb.year, wb.month, wb.plant_name, wb.id_lng_train
        """
        df = pd.read_sql_query(query, engine, params={'scenario': scenario})

    # Filter by selected countries if provided
    if selected_countries and len(selected_countries) > 0:
        df = df[df['country_name'].isin(selected_countries)]

    # Filter for new capacity if needed
    if new_capacity_only:
        new_trains_df = get_new_capacity_filters(scenario=scenario)

        if new_trains_df.empty:
            return pd.DataFrame()

        # Filter to only include trains that are "new" (started >= current month)
        df = df[df['id_lng_train'].isin(new_trains_df['id_lng_train'])]

        # Filter out zero/null volumes to avoid showing data before train actually starts
        # This handles cases where adjustments start before the baseline start date
        df = df[df['total_output'].notna() & (df['total_output'] > 0)]

    # Now aggregate based on breakdown level
    if breakdown == 'country':
        df = df.groupby(['year', 'month', 'country_name'], as_index=False)['total_output'].sum()
        df = df.rename(columns={'country_name': 'group_name'})
    elif breakdown == 'project':
        df = df.groupby(['year', 'month', 'plant_name', 'country_name'], as_index=False)['total_output'].sum()
        df = df.rename(columns={'plant_name': 'group_name'})
    else:  # train
        # Use lng_train_name_short for proper train naming
        df['group_name'] = df['plant_name'] + ' - ' + df['lng_train_name_short']
        df = df[['year', 'month', 'group_name', 'plant_name', 'country_name', 'total_output']]

    return df


def create_timeline_figure(selected_unit='mtpa', scenario='base_view', year_range=None):
    """
    Create the Plotly timeline figure showing LNG train capacity additions using bar charts.

    Args:
        selected_unit: 'mtpa' or 'mcmd'
        scenario: Scenario name for adjustments ('base_view', 'best_view', etc.)
        year_range: List with [start_year, end_year] for filtering (default: [2025, 2028])

    Returns:
        plotly.graph_objects.Figure
    """
    # Fetch data with scenario
    df = fetch_train_data(scenario=scenario)

    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Apply year range filter
    if year_range is None:
        year_range = [2025, 2028]

    start_year, end_year = year_range
    df['train_year'] = pd.to_datetime(df['lng_train_date_start_est']).dt.year
    df = df[(df['train_year'] >= start_year) & (df['train_year'] <= end_year)].copy()

    if df.empty:
        # Return empty figure with message if no data in range
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data available for {start_year}-{end_year}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Convert to Mcm/d if needed
    if selected_unit == 'mcmd':
        df['capacity'] = convert_to_mcmd(df['capacity'])

    # Create month-year column for aggregation (set to 1st of month)
    df['month_start'] = df['lng_train_date_start_est'].dt.to_period('M').dt.to_timestamp()

    # Sort by country and then by plant name
    df = df.sort_values(['country_name', 'plant_name', 'month_start'])

    # Aggregate by plant and month - sum capacity and count trains
    monthly_df = df.groupby(['country_name', 'plant_name', 'month_start']).agg({
        'capacity': 'sum',
        'lng_train_date_start_est': 'count'  # Count of trains
    }).reset_index()
    monthly_df.rename(columns={'lng_train_date_start_est': 'train_count'}, inplace=True)

    # Get unique plants with their countries, maintaining order
    plants_df = monthly_df.groupby(['country_name', 'plant_name']).first().reset_index()

    # Calculate dynamic date range based on NEW capacity additions (train start dates)
    # Use original df with actual train start dates, not monthly_df which is aggregated
    if not df.empty:
        # Get actual train start dates (when new capacity is added)
        train_start_dates = pd.to_datetime(df['lng_train_date_start_est'])
        data_min_date = train_start_dates.min()
        data_max_date = train_start_dates.max()  # Last date with NEW capacity addition

        # Add minimal buffer: start 15 days before first addition, end 15 days after last addition
        chart_start_date = (data_min_date - pd.DateOffset(days=15)).strftime('%Y-%m-%d')
        chart_end_date = (data_max_date + pd.DateOffset(days=15)).strftime('%Y-%m-%d')

        # For grid lines - get years to iterate (use actual capacity addition years only)
        min_year = data_min_date.year
        max_year = data_max_date.year

        # Title date range
        title_date_range = f"{data_min_date.strftime('%B %Y')} - {data_max_date.strftime('%B %Y')}"
    else:
        # Fallback if no data
        chart_start_date = '2024-12-15'
        chart_end_date = '2028-02-01'
        min_year = 2025
        max_year = 2028
        title_date_range = 'January 2025 - January 2028'

    # Create figure
    fig = go.Figure()

    # Track y positions and labels
    y_position = 0
    y_labels = []
    y_ticks = []
    plant_to_y = {}

    # Add bars for each plant
    for idx, (_, plant_info) in enumerate(plants_df.iterrows()):
        country = plant_info['country_name']
        plant = plant_info['plant_name']

        plant_to_y[plant] = y_position

        # Get all monthly aggregations for this plant
        plant_months = monthly_df[
            (monthly_df['country_name'] == country) &
            (monthly_df['plant_name'] == plant)
        ]

        # Get color for country
        color = PRIMARY_COLORS.get(country, '#666666')

        # Add bars using scatter with mode='markers'
        for _, month_data in plant_months.iterrows():
            capacity = month_data['capacity']
            train_count = month_data['train_count']
            month_date = month_data['month_start']

            # Add bar using marker
            fig.add_trace(go.Bar(
                x=[month_date],
                y=[capacity],
                base=y_position - 0.3,
                width=10 * 24 * 60 * 60 * 1000,  # 10 days in milliseconds
                marker=dict(
                    color=color,
                    line=dict(color=color if capacity > (15 if selected_unit == 'mtpa' else 50) else color, width=0),
                    opacity=0.85
                ),
                orientation='v',
                showlegend=False,
                hovertemplate=f'<b>{plant}</b><br>{country}<br>Capacity: {capacity:.1f} {selected_unit.upper()}<br>Date: %{{x|%b %Y}}<extra></extra>',
                customdata=[[plant, country, capacity, train_count]]
            ))

            # Add capacity label - only for significant capacity (>= 3 MTPA or 10 Mcm/d) to reduce clutter
            min_threshold = 3.0 if selected_unit == 'mtpa' else 10.0
            if capacity >= min_threshold:
                label_text = f"{capacity:.1f}" if selected_unit == 'mtpa' else f"{capacity:.0f}"
                fig.add_annotation(
                    x=month_date,
                    y=y_position + capacity + 0.3,
                    text=label_text,
                    showarrow=False,
                    font=dict(size=9, color=color, family="Arial", weight='bold'),
                    xanchor='center',
                    yanchor='bottom'
                )

            # Add train count if multiple trains (always show this as it's important info)
            if train_count > 1:
                fig.add_annotation(
                    x=month_date,
                    y=y_position - 0.4,
                    text=str(train_count),
                    showarrow=False,
                    font=dict(size=7, color=color, family="Arial"),
                    xanchor='center',
                    yanchor='top',
                    opacity=0.9
                )

        y_labels.append(plant)
        y_ticks.append(y_position)
        y_position += (monthly_df['capacity'].max() * 1.5 if len(monthly_df) > 0 else 10)

    # Add background shading for each plant row based on country color
    for idx, (_, plant_info) in enumerate(plants_df.iterrows()):
        country = plant_info['country_name']
        color = PRIMARY_COLORS.get(country, '#666666')

        # Calculate row boundaries
        y_center = y_ticks[idx]
        if idx < len(y_ticks) - 1:
            y_bottom = (y_ticks[idx] + y_ticks[idx + 1]) / 2
        else:
            y_bottom = y_position

        if idx > 0:
            y_top = (y_ticks[idx - 1] + y_ticks[idx]) / 2
        else:
            y_top = y_ticks[idx] - (monthly_df['capacity'].max() * 0.75 if len(monthly_df) > 0 else 5)

        # Add very light background rectangle for this plant
        fig.add_shape(
            type="rect",
            x0=chart_start_date, x1=chart_end_date,
            y0=y_top, y1=y_bottom,
            fillcolor=color,
            opacity=0.05,  # Very light - only 5% opacity
            line_width=0,
            layer='below'
        )

    # Add horizontal separator lines between plants
    for i in range(len(y_ticks) - 1):
        mid_y = (y_ticks[i] + y_ticks[i+1]) / 2
        fig.add_shape(
            type="line",
            x0=chart_start_date, x1=chart_end_date,
            y0=mid_y, y1=mid_y,
            line=dict(color='#E0E0E0', width=0.5),
            layer='below'
        )

    # Calculate y-axis range (minimum is bottom of first plant)
    if len(y_ticks) > 0:
        y_min = y_ticks[0] - (monthly_df['capacity'].max() * 0.75 if len(monthly_df) > 0 else 5)
        y_max = y_position
    else:
        y_min = 0
        y_max = 10

    # Add vertical grid lines - only yearly for performance (not monthly)
    for year in range(min_year, max_year + 1):
        date = pd.Timestamp(f'{year}-01-01')
        if pd.Timestamp(chart_start_date) <= date <= pd.Timestamp(chart_end_date):
            fig.add_shape(
                type="line",
                x0=date, x1=date,
                y0=y_min, y1=y_max,
                line=dict(color='#999999', width=1, dash='solid'),
                opacity=0.4,
                layer='below'
            )

    # Add "Today" marker
    current_date = datetime.now()
    first_of_month = pd.Timestamp(current_date.replace(day=1))
    if pd.Timestamp(chart_start_date) <= first_of_month <= pd.Timestamp(chart_end_date):
        fig.add_shape(
            type="line",
            x0=first_of_month, x1=first_of_month,
            y0=y_min, y1=y_max,
            line=dict(color='#E74C3C', width=2, dash='dash'),
            layer='above'
        )
        fig.add_annotation(
            x=first_of_month,
            y=y_max * 0.98,
            text='Today',
            showarrow=False,
            font=dict(size=10, color='#E74C3C', family="Arial", weight='bold'),
            bgcolor='white',
            bordercolor='#E74C3C',
            borderwidth=1,
            borderpad=3
        )

    # Add arrows for adjusted trains showing change from Woodmac baseline
    adjusted_trains = df[df['data_source'] == 'adjusted'].copy()

    for _, train in adjusted_trains.iterrows():
        if pd.notna(train['woodmac_date']) and pd.notna(train['internal_date']):
            plant_name = train['plant_name']
            woodmac_date = pd.to_datetime(train['woodmac_date'])
            internal_date = pd.to_datetime(train['internal_date'])

            # Get y position for this plant
            y_pos = plant_to_y.get(plant_name)
            if y_pos is not None:
                # Only show arrow if dates are different and within our chart range
                if (woodmac_date != internal_date and
                    pd.Timestamp(chart_start_date) <= woodmac_date <= pd.Timestamp(chart_end_date) and
                    pd.Timestamp(chart_start_date) <= internal_date <= pd.Timestamp(chart_end_date)):

                    # Position arrow below the baseline
                    arrow_y = y_pos - 1.5

                    # Determine arrow color based on direction
                    if internal_date < woodmac_date:  # Moved earlier
                        arrow_color = '#2E8B57'  # Sea green for earlier
                    else:  # Moved later
                        arrow_color = '#B22222'  # Fire brick for later

                    # Add arrow annotation
                    fig.add_annotation(
                        x=internal_date,
                        y=arrow_y,
                        ax=woodmac_date,
                        ay=arrow_y,
                        xref='x',
                        yref='y',
                        axref='x',
                        ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor=arrow_color,
                        opacity=0.8
                    )

                    # Add label showing the change in months
                    mid_date = woodmac_date + (internal_date - woodmac_date) / 2
                    days_diff = abs((internal_date - woodmac_date).days)
                    months_diff = round(days_diff / 30.44, 1)  # Average days per month
                    label_text = f"{months_diff:.0f}m"

                    fig.add_annotation(
                        x=mid_date,
                        y=arrow_y + 0.8,
                        text=label_text,
                        showarrow=False,
                        font=dict(size=8, color=arrow_color, family="Arial", weight='bold'),
                        bgcolor='white',
                        bordercolor=arrow_color,
                        borderwidth=0.5,
                        borderpad=2,
                        opacity=0.9
                    )

    # Update layout
    unit_label = 'MTPA' if selected_unit == 'mtpa' else 'Mcm/d'

    # Calculate smart tick interval based on date range
    date_range_years = max_year - min_year + 1
    if date_range_years <= 3:
        dtick = 'M3'  # Every 3 months for short ranges
        tickformat = '%b\n%Y'
    elif date_range_years <= 7:
        dtick = 'M6'  # Every 6 months for medium ranges
        tickformat = '%b\n%Y'
    elif date_range_years <= 15:
        dtick = 'M12'  # Annually for longer ranges
        tickformat = '%Y'
    else:
        dtick = 'M24'  # Every 2 years for very long ranges
        tickformat = '%Y'

    fig.update_layout(
        title={
            'text': f'LNG Train Start Dates and Capacity ({unit_label}) | {title_date_range}',
            'font': {'size': 20, 'family': 'Arial', 'color': '#333333'},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.98,
            'yanchor': 'top'
        },
        xaxis=dict(
            title='',
            range=[chart_start_date, chart_end_date],
            type='date',
            tickformat=tickformat,
            dtick=dtick,
            tickfont=dict(size=10, family='Arial', color='#333333'),
            tickangle=0,
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='#CCCCCC',
            zeroline=False
        ),
        yaxis=dict(
            title='',
            range=[y_min, y_max],
            tickmode='array',
            tickvals=y_ticks,
            ticktext=y_labels,
            tickfont=dict(size=11, family='Arial', color='#333333'),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='#CCCCCC',
            zeroline=False,
            side='left'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=max(800, len(y_ticks) * 50),
        margin=dict(l=200, r=200, t=60, b=80),
        hovermode='closest',
        showlegend=False,
        bargap=0,
        barmode='overlay',
        # Performance optimizations
        uirevision='constant',  # Preserve zoom/pan state
        dragmode='pan',  # Default to pan mode (faster than zoom)
    )

    # Optimize for faster rendering
    fig.update_xaxes(automargin=False)
    fig.update_yaxes(automargin=False)

    # Color y-axis labels by country
    country_colors = {}
    for idx, (_, plant_info) in enumerate(plants_df.iterrows()):
        country = plant_info['country_name']
        country_colors[idx] = PRIMARY_COLORS.get(country, '#666666')

    # Add country labels on the right
    country_sections = {}
    current_country = None
    country_start_idx = 0

    for idx, (_, plant_info) in enumerate(plants_df.iterrows()):
        country = plant_info['country_name']
        if country != current_country:
            if current_country is not None:
                # Add label for previous country
                mid_y = (y_ticks[country_start_idx] + y_ticks[idx-1]) / 2
                fig.add_annotation(
                    x=1.005,  # Reduced from 1.05 to 1.005 for minimal spacing
                    y=mid_y,
                    xref='paper',
                    yref='y',
                    text=current_country,
                    showarrow=False,
                    font=dict(size=12, color=PRIMARY_COLORS.get(current_country, '#666666'), family='Arial', weight='bold'),
                    xanchor='left',
                    yanchor='middle'
                )
            current_country = country
            country_start_idx = idx

    # Add last country label
    if current_country is not None:
        mid_y = (y_ticks[country_start_idx] + y_ticks[len(y_ticks)-1]) / 2
        fig.add_annotation(
            x=1.005,  # Reduced from 1.05 to 1.005 for minimal spacing
            y=mid_y,
            xref='paper',
            yref='y',
            text=current_country,
            showarrow=False,
            font=dict(size=12, color=PRIMARY_COLORS.get(current_country, '#666666'), family='Arial', weight='bold'),
            xanchor='left',
            yanchor='middle'
        )

    # Add note at bottom left
    note_text = 'Bar height represents cumulative monthly capacity • Number below bar indicates multiple trains in same month • Arrows show internal adjustments vs Woodmac baseline'
    fig.add_annotation(
        x=0,
        y=-0.08,  # Position below the x-axis with more space
        xref='paper',
        yref='paper',
        text=note_text,
        showarrow=False,
        font=dict(size=9, color='#666666', family='Arial', style='italic'),
        xanchor='left',
        yanchor='top'
    )

    return fig


# Page Layout
layout = html.Div([
    # Page title
    html.H1(
        "LNG Train Timeline - Capacity Additions 2025-2028",
        style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#2E86C1'}
    ),

    # Scenario selector and controls section
    html.Div([
        html.Div([
            html.Label("Scenario:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='scenario-selector-dropdown',
                options=[{'label': s, 'value': s} for s in get_available_scenarios(engine)],
                value='base_view',
                clearable=False,
                className='inline-dropdown',
                style={'width': '200px', 'marginRight': '20px'}
            ),
            html.A(
                html.Button(
                    'Manage Adjustments',
                    style={
                        'padding': '6px 14px',
                        'backgroundColor': '#2E86C1',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'fontWeight': 'bold',
                        'fontSize': '13px'
                    }
                ),
                href='/terminal_adjustments',
                style={'marginRight': '30px'}
            ),
            html.Label("Unit of Measure:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='unit-dropdown',
                options=[
                    {'label': 'MTPA (Million Tonnes Per Annum)', 'value': 'mtpa'},
                    {'label': 'Mcm/d (Million Cubic Meters per Day)', 'value': 'mcmd'}
                ],
                value='mtpa',
                clearable=False,
                className='inline-dropdown',
                style={'width': '350px'}
            )
        ], style={'display': 'inline-flex', 'alignItems': 'center', 'marginBottom': '10px', 'justifyContent': 'center'}),

        # Date range filter
        html.Div([
            html.Label("Year Range:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.RangeSlider(
                id='year-range-slider',
                min=2000,
                max=2055,
                step=1,
                value=[2025, 2028],
                marks={year: str(year) for year in range(2000, 2056, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
                className='year-range-slider'
            )
        ], style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'})
    ], style={'textAlign': 'center'}),

    # Timeline chart
    html.Div([
        dcc.Graph(
            id='train-timeline-graph',
            config={'displayModeBar': True, 'displaylogo': False},
            style={'height': '100%'}
        )
    ]),
    html.Div(
        [
            dcc.Link(
                "Open the cumulative monthly LNG output chart and table on the Production page",
                href='/production',
                className='nav-link-secondary'
            )
        ],
        style={'textAlign': 'center', 'marginTop': '24px'}
    )
])


# Callback to update timeline chart (depends on unit, scenario, and year range)
@callback(
    Output('train-timeline-graph', 'figure'),
    Input('scenario-selector-dropdown', 'value'),
    Input('unit-dropdown', 'value'),
    Input('year-range-slider', 'value'),
    Input('global-refresh-button', 'n_clicks')
)
def update_timeline_chart(scenario, selected_unit, year_range, n_clicks):
    """Update timeline chart based on scenario, unit, and year range."""
    return create_timeline_figure(selected_unit, scenario=scenario, year_range=year_range)
