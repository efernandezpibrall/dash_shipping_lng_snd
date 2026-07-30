import pandas as pd
from fundamentals.terminals.terminal_output_utils import (
    fetch_keyed_terminal_monthly_output,
    fetch_keyed_terminal_train_summary,
)
from utils.database import DB_SCHEMA, engine

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
    return fetch_keyed_terminal_train_summary(engine, scenario=scenario)

    # Legacy provider-ID query retained temporarily for rollback comparison.
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
        DataFrame with columns: plant_name, train_key, lng_train_date_start_est
    """
    df = fetch_train_data(scenario=scenario)

    # Get current date (first day of current month)
    current_date = pd.Timestamp.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # Filter for trains starting from current month onwards
    # lng_train_date_start_est is already a datetime from fetch_train_data()
    new_trains_df = df[df['lng_train_date_start_est'] >= current_date][
        ['plant_name', 'train_key', 'id_lng_train', 'lng_train_date_start_est', 'country_name']
    ].copy()

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
    df = fetch_keyed_terminal_monthly_output(
        engine,
        start_year=start_year,
        end_year=end_year,
        scenario=scenario,
    )
    if selected_countries:
        df = df[df['country_name'].isin(selected_countries)]
    if new_capacity_only:
        new_trains_df = get_new_capacity_filters(scenario=scenario)
        if new_trains_df.empty:
            return pd.DataFrame()
        df = df[df['train_key'].isin(new_trains_df['train_key'])]
        df = df[df['total_output'].notna() & (df['total_output'] > 0)]

    if breakdown == 'country':
        df = df.groupby(['year', 'month', 'country_name'], as_index=False)['total_output'].sum()
        return df.rename(columns={'country_name': 'group_name'})
    if breakdown == 'project':
        df = df.groupby(
            ['year', 'month', 'plant_name', 'country_name'], as_index=False
        )['total_output'].sum()
        return df.rename(columns={'plant_name': 'group_name'})

    df['group_name'] = df['plant_name'] + ' - ' + df['lng_train_name_short']
    return df[
        ['year', 'month', 'group_name', 'plant_name', 'country_name', 'total_output']
    ]

    # Legacy provider-ID query retained temporarily for rollback comparison.
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
              AND a.year >= %(start_year)s AND a.year <= %(end_year)s
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
            WHERE year >= %(start_year)s AND year <= %(end_year)s
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
        df = pd.read_sql_query(
            query,
            engine,
            params={'start_year': start_year, 'end_year': end_year},
        )
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
              AND a.year >= %(start_year)s AND a.year <= %(end_year)s
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
            WHERE year >= %(start_year)s AND year <= %(end_year)s
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
        df = pd.read_sql_query(
            query,
            engine,
            params={
                'scenario': scenario,
                'start_year': start_year,
                'end_year': end_year,
            },
        )

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
