import pandas as pd
from fundamentals.lng.terminals.terminal_output_utils import (
    fetch_keyed_terminal_monthly_output,
    fetch_keyed_terminal_train_summary,
)
from utils.database import engine

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
