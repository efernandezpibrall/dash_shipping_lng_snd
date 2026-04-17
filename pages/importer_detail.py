from dash import html, dcc, dash_table, callback, Output, Input, State, Dash, ALL, ctx, no_update
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
from sqlalchemy import create_engine, text, bindparam

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
    """Get n professional colors, cycling through the palette if needed."""
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
            converted_df.loc[valid_days, value_column] * MCM_PER_CUBIC_METER / period_days[valid_days]
        )
    elif metric_name == 'mtpa':
        converted_df.loc[valid_days, value_column] = (
            converted_df.loc[valid_days, value_column] * 0.45 / 1_000_000 * 365.25 / period_days[valid_days]
        )

    converted_df.loc[~valid_days, value_column] = np.nan
    return converted_df


DESTINATION_AGGREGATION_LABELS = {
    'country': 'Country',
    'continent': 'Continent',
    'subcontinent': 'Subcontinent',
    'basin': 'Basin',
    'country_classification_level1': 'Classification Level 1',
    'country_classification': 'Classification',
    'shipping_region': 'Shipping Region',
}

DESTINATION_AGGREGATION_OPTIONS = [
    {'label': label, 'value': value}
    for value, label in DESTINATION_AGGREGATION_LABELS.items()
]

DESTINATION_CATALOG_COLUMNS = [
    'destination_country_name',
    'country',
    'country_display',
    'continent',
    'subcontinent',
    'basin',
    'country_classification_level1',
    'country_classification',
    'shipping_region',
]


def normalize_destination_countries(destination_countries):
    """Normalize a destination-country filter into a sorted unique tuple."""
    if destination_countries is None:
        return ()
    if isinstance(destination_countries, str):
        raw_values = [destination_countries]
    else:
        raw_values = list(destination_countries)

    normalized_values = []
    for value in raw_values:
        if pd.isna(value):
            continue
        normalized_value = str(value).strip()
        if normalized_value:
            normalized_values.append(normalized_value)

    return tuple(sorted(set(normalized_values)))


def _normalize_mapping_value(value):
    if pd.isna(value):
        return None
    normalized_value = str(value).strip()
    return normalized_value if normalized_value else None


def _collapse_mapping_values(series):
    normalized_values = sorted({
        value for value in (_normalize_mapping_value(item) for item in series)
        if value is not None
    })
    if len(normalized_values) == 1:
        return normalized_values[0]
    return 'Unknown'


def _first_non_empty_value(series, fallback=''):
    for item in series:
        normalized_value = _normalize_mapping_value(item)
        if normalized_value is not None:
            return normalized_value
    return fallback


def get_destination_catalog_dataframe(catalog_records):
    """Return a normalized destination catalog DataFrame."""
    if not catalog_records:
        return pd.DataFrame(columns=DESTINATION_CATALOG_COLUMNS)

    catalog_df = pd.DataFrame(catalog_records)
    for column in DESTINATION_CATALOG_COLUMNS:
        if column not in catalog_df.columns:
            catalog_df[column] = None

    catalog_df = catalog_df[DESTINATION_CATALOG_COLUMNS].copy()
    catalog_df['destination_country_name'] = catalog_df['destination_country_name'].apply(_normalize_mapping_value)
    catalog_df = catalog_df[catalog_df['destination_country_name'].notna()].copy()
    catalog_df['country'] = catalog_df['destination_country_name']
    catalog_df['country_display'] = catalog_df['country_display'].apply(_normalize_mapping_value)
    catalog_df['country_display'] = catalog_df['country_display'].fillna(catalog_df['destination_country_name'])

    for column in DESTINATION_AGGREGATION_LABELS:
        if column == 'country':
            continue
        catalog_df[column] = catalog_df[column].apply(_normalize_mapping_value).fillna('Unknown')

    catalog_df = catalog_df.drop_duplicates(subset=['destination_country_name']).reset_index(drop=True)
    return catalog_df


def build_destination_catalog(engine):
    """Build a deduplicated destination catalog from Kpler destinations plus country mappings."""
    destination_query = text(f"""
        WITH latest_timestamp AS (
            SELECT MAX(upload_timestamp_utc) AS max_ts
            FROM {DB_SCHEMA}.kpler_trades
        )
        SELECT DISTINCT destination_country_name
        FROM {DB_SCHEMA}.kpler_trades kt
        CROSS JOIN latest_timestamp
        WHERE kt.upload_timestamp_utc = latest_timestamp.max_ts
            AND kt.destination_country_name IS NOT NULL
        ORDER BY destination_country_name
    """)
    destinations_df = pd.read_sql(destination_query, engine)
    if destinations_df.empty:
        return []

    destinations_df['destination_country_name'] = destinations_df['destination_country_name'].apply(_normalize_mapping_value)
    destinations_df = destinations_df[destinations_df['destination_country_name'].notna()].drop_duplicates(
        subset=['destination_country_name']
    )
    if destinations_df.empty:
        return []

    mapping_df = pd.read_sql(text(f"SELECT * FROM {DB_SCHEMA}.mappings_country"), engine)
    if mapping_df.empty:
        mapping_df = pd.DataFrame(columns=['country'])

    if 'country' not in mapping_df.columns and 'country_name' in mapping_df.columns:
        mapping_df['country'] = mapping_df['country_name']
    if 'country' not in mapping_df.columns:
        mapping_df['country'] = None

    mapping_df['country'] = mapping_df['country'].apply(_normalize_mapping_value)
    mapping_df = mapping_df[mapping_df['country'].notna()].copy()

    aggregation_spec = {
        'country_display': lambda series: _first_non_empty_value(series),
    }
    for column in DESTINATION_AGGREGATION_LABELS:
        if column == 'country':
            continue
        if column not in mapping_df.columns:
            mapping_df[column] = None
        aggregation_spec[column] = _collapse_mapping_values

    if 'country_name' in mapping_df.columns:
        mapping_df['country_display'] = mapping_df['country_name']
    else:
        mapping_df['country_display'] = mapping_df['country']

    deduped_mapping_df = mapping_df.groupby('country', as_index=False).agg(aggregation_spec)
    catalog_df = destinations_df.merge(
        deduped_mapping_df,
        how='left',
        left_on='destination_country_name',
        right_on='country'
    )

    catalog_df['country'] = catalog_df['destination_country_name']
    catalog_df['country_display'] = catalog_df['country_display'].fillna(catalog_df['destination_country_name'])
    for column in DESTINATION_AGGREGATION_LABELS:
        if column == 'country':
            continue
        catalog_df[column] = catalog_df[column].fillna('Unknown')

    catalog_df = get_destination_catalog_dataframe(catalog_df.to_dict('records'))
    catalog_df = catalog_df.sort_values(
        by=['country_display', 'destination_country_name']
    ).reset_index(drop=True)
    return catalog_df.to_dict('records')


def _sort_destination_group_values(values):
    return sorted(values, key=lambda item: (str(item) == 'Unknown', str(item)))


def build_destination_value_options(aggregation, catalog_records):
    """Build destination value dropdown options for the selected aggregation."""
    catalog_df = get_destination_catalog_dataframe(catalog_records)
    if catalog_df.empty:
        return [{'label': 'China', 'value': 'China'}]

    if aggregation not in DESTINATION_AGGREGATION_LABELS:
        aggregation = 'country'

    if aggregation == 'country':
        country_df = catalog_df[['destination_country_name', 'country_display']].drop_duplicates()
        country_df = country_df.sort_values(by=['country_display', 'destination_country_name'])
        return [
            {
                'label': row['country_display'],
                'value': row['destination_country_name']
            }
            for _, row in country_df.iterrows()
        ]

    distinct_values = _sort_destination_group_values(catalog_df[aggregation].dropna().unique().tolist())
    return [{'label': value, 'value': value} for value in distinct_values]


def get_default_destination_value(aggregation, catalog_records):
    """Return the default dropdown value for the selected aggregation."""
    if aggregation not in DESTINATION_AGGREGATION_LABELS:
        aggregation = 'country'

    options = build_destination_value_options(aggregation, catalog_records)
    if not options:
        return 'China'

    catalog_df = get_destination_catalog_dataframe(catalog_records)
    if aggregation == 'country':
        option_values = {option['value'] for option in options}
        return 'China' if 'China' in option_values else options[0]['value']

    if not catalog_df.empty and 'China' in catalog_df['destination_country_name'].values:
        china_row = catalog_df[catalog_df['destination_country_name'] == 'China'].iloc[0]
        china_group_value = china_row.get(aggregation, 'Unknown')
        option_values = {option['value'] for option in options}
        if china_group_value in option_values:
            return china_group_value

    return options[0]['value']


def resolve_selected_destination_countries(aggregation, selected_value, catalog_records):
    """Resolve the selected destination aggregation/value into destination countries."""
    if not selected_value:
        return []

    catalog_df = get_destination_catalog_dataframe(catalog_records)
    if catalog_df.empty:
        return []

    if aggregation not in DESTINATION_AGGREGATION_LABELS:
        aggregation = 'country'

    if aggregation == 'country':
        matched_df = catalog_df[catalog_df['destination_country_name'] == selected_value]
    else:
        matched_df = catalog_df[catalog_df[aggregation] == selected_value]

    return normalize_destination_countries(matched_df['destination_country_name'].tolist())


def format_destination_selection_label(aggregation, selected_value, catalog_records):
    """Return a user-facing label for the selected destination."""
    if not selected_value:
        return 'Selected Destination'

    if aggregation not in DESTINATION_AGGREGATION_LABELS:
        aggregation = 'country'

    if aggregation == 'country':
        catalog_df = get_destination_catalog_dataframe(catalog_records)
        if not catalog_df.empty:
            matched_df = catalog_df[catalog_df['destination_country_name'] == selected_value]
            if not matched_df.empty:
                return matched_df.iloc[0].get('country_display') or selected_value
        return selected_value

    aggregation_label = DESTINATION_AGGREGATION_LABELS.get(
        aggregation,
        aggregation.replace('_', ' ').title()
    )
    return f"{selected_value} ({aggregation_label})"


def resolve_destination_context(aggregation, selected_value, catalog_records):
    """Resolve the current destination selection into a display label and country list."""
    return {
        'destination_countries': resolve_selected_destination_countries(
            aggregation,
            selected_value,
            catalog_records
        ),
        'display_label': format_destination_selection_label(
            aggregation,
            selected_value,
            catalog_records
        ),
    }


def determine_destination_dropdown_value(aggregation, catalog_records, selection_state):
    """Choose the next destination dropdown value, preserving semantics when possible."""
    if aggregation not in DESTINATION_AGGREGATION_LABELS:
        aggregation = 'country'

    options = build_destination_value_options(aggregation, catalog_records)
    option_values = {option['value'] for option in options}
    if not options:
        return 'China'

    previous_aggregation = (selection_state or {}).get('aggregation')
    previous_value = (selection_state or {}).get('value')
    if previous_aggregation and previous_value:
        previous_countries = resolve_selected_destination_countries(
            previous_aggregation,
            previous_value,
            catalog_records
        )
        if previous_countries:
            catalog_df = get_destination_catalog_dataframe(catalog_records)
            scoped_df = catalog_df[catalog_df['destination_country_name'].isin(previous_countries)]
            if aggregation == 'country':
                unique_values = normalize_destination_countries(
                    scoped_df['destination_country_name'].tolist()
                )
            else:
                unique_values = _sort_destination_group_values(
                    scoped_df[aggregation].dropna().unique().tolist()
                )
            if len(unique_values) == 1 and unique_values[0] in option_values:
                return unique_values[0]

    default_value = get_default_destination_value(aggregation, catalog_records)
    return default_value if default_value in option_values else options[0]['value']


def format_rolling_window_label(window_days):
    return f"{normalize_rolling_window_days(window_days)}D"


def format_rolling_window_title(window_days):
    normalized_window_days = normalize_rolling_window_days(window_days)
    return f"{normalized_window_days}-Day Rolling Average"


IMPORTER_SELECTION_TO_ORIGIN_SCOPE = {
    'country': 'origin_country',
    'continent': 'origin_continent',
    'shipping_region': 'origin_shipping_region',
    'basin': 'origin_basin',
    'subcontinent': 'origin_subcontinent',
    'country_classification_level1': 'origin_classification_level1',
    'country_classification': 'origin_classification',
}

IMPORTER_ORIGIN_LEVEL_TO_SCOPE = {
    'origin_country_name': 'origin_country',
    'continent_origin_name': 'origin_continent',
    'origin_shipping_region': 'origin_shipping_region',
    'origin_basin': 'origin_basin',
    'origin_subcontinent': 'origin_subcontinent',
    'origin_classification_level1': 'origin_classification_level1',
    'origin_classification': 'origin_classification',
}

IMPORTER_MAPPING_RENAME = {
    'continent': 'origin_continent',
    'shipping_region': 'origin_shipping_region',
    'basin': 'origin_basin',
    'subcontinent': 'origin_subcontinent',
    'country_classification_level1': 'origin_classification_level1',
    'country_classification': 'origin_classification',
}


def _normalize_scope_value(value, default='Unknown'):
    normalized_value = _normalize_mapping_value(value)
    return normalized_value if normalized_value is not None else default


def _normalize_scope_series(series, default=None):
    normalized_series = series.apply(_normalize_mapping_value)
    if default is not None:
        normalized_series = normalized_series.fillna(default)
    return normalized_series


def _load_importer_country_mapping_lookup(engine):
    lookup_columns = ['mapping_key'] + list(IMPORTER_MAPPING_RENAME.values())
    mapping_df = pd.read_sql(
        text(f"""
            SELECT
                country_name,
                country,
                continent,
                shipping_region,
                basin,
                subcontinent,
                country_classification_level1,
                country_classification
            FROM {DB_SCHEMA}.mappings_country
            WHERE country_name IS NOT NULL OR country IS NOT NULL
        """),
        engine
    )
    if mapping_df.empty:
        return pd.DataFrame(columns=lookup_columns)

    expected_columns = ['country_name', 'country'] + list(IMPORTER_MAPPING_RENAME.keys())
    for column in expected_columns:
        if column not in mapping_df.columns:
            mapping_df[column] = None
        mapping_df[column] = _normalize_scope_series(mapping_df[column], default=None)

    mapping_df['country_name'] = mapping_df['country_name'].fillna(mapping_df['country'])
    mapping_df = mapping_df[mapping_df['country_name'].notna()].copy()
    if mapping_df.empty:
        return pd.DataFrame(columns=lookup_columns)

    aggregation_spec = {
        'country': lambda series: _first_non_empty_value(series, fallback=''),
    }
    for column in IMPORTER_MAPPING_RENAME:
        aggregation_spec[column] = _collapse_mapping_values

    deduped_mapping_df = mapping_df.groupby('country_name', as_index=False).agg(aggregation_spec)
    deduped_mapping_df['country'] = deduped_mapping_df['country'].replace('', np.nan)
    deduped_mapping_df['country'] = deduped_mapping_df['country'].fillna(deduped_mapping_df['country_name'])

    alias_frames = []
    for alias_col in ['country_name', 'country']:
        alias_df = deduped_mapping_df[[alias_col] + list(IMPORTER_MAPPING_RENAME.keys())].copy()
        alias_df = alias_df.rename(columns={alias_col: 'mapping_key'})
        alias_df['mapping_key'] = _normalize_scope_series(alias_df['mapping_key'], default=None)
        alias_df = alias_df[alias_df['mapping_key'].notna()].copy()
        alias_frames.append(alias_df)

    if not alias_frames:
        return pd.DataFrame(columns=lookup_columns)

    lookup_df = pd.concat(alias_frames, ignore_index=True)
    lookup_df = lookup_df.drop_duplicates(subset=['mapping_key'], keep='first')
    lookup_df = lookup_df.rename(columns=IMPORTER_MAPPING_RENAME)
    for column in IMPORTER_MAPPING_RENAME.values():
        lookup_df[column] = _normalize_scope_series(lookup_df[column], default='Unknown')

    return lookup_df[lookup_columns]


def _fetch_importer_scoped_trades(engine, destination_countries, min_end_date=None, vessel_type=None,
                                  delivered_only=False):
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    expected_columns = [
        'end_date',
        'cargo_mcm',
        'origin_country',
        'origin_continent_chart',
        'origin_continent',
        'origin_shipping_region',
        'origin_basin',
        'origin_subcontinent',
        'origin_classification_level1',
        'origin_classification',
    ]
    if not normalized_destination_countries:
        return pd.DataFrame(columns=expected_columns)

    min_end_date = pd.Timestamp(min_end_date or SUMMARY_LOOKBACK_START).normalize().date()
    where_clauses = [
        "kt.upload_timestamp_utc = (SELECT MAX(upload_timestamp_utc) FROM {schema}.kpler_trades)".format(
            schema=DB_SCHEMA
        ),
        "kt.destination_country_name IN :destination_countries",
        'kt."end" IS NOT NULL',
        "kt.cargo_destination_cubic_meters IS NOT NULL",
        'kt."end"::date >= :min_end_date',
    ]
    params = {
        'destination_countries': normalized_destination_countries,
        'min_end_date': min_end_date,
    }
    if delivered_only:
        where_clauses.append("kt.status = 'Delivered'")
    if vessel_type and vessel_type != 'All':
        where_clauses.append("mv.vessel_type = :vessel_type")
        params['vessel_type'] = vessel_type

    query = text(f"""
        SELECT
            kt."end"::date AS end_date,
            COALESCE(kt.cargo_destination_cubic_meters, 0) * {MCM_PER_CUBIC_METER} AS cargo_mcm,
            COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown') AS origin_country,
            COALESCE(NULLIF(BTRIM(kt.continent_origin_name), ''), 'Unknown') AS origin_continent_chart
        FROM {DB_SCHEMA}.kpler_trades kt
        LEFT JOIN {DB_SCHEMA}.mapping_vessel_type_capacity mv
            ON kt.vessel_capacity_cubic_meters >= mv.capacity_cubic_meters_min
            AND kt.vessel_capacity_cubic_meters < mv.capacity_cubic_meters_max
        WHERE {' AND '.join(where_clauses)}
    """).bindparams(bindparam('destination_countries', expanding=True))

    scoped_trades_df = pd.read_sql(query, engine, params=params)
    if scoped_trades_df.empty:
        return pd.DataFrame(columns=expected_columns)

    scoped_trades_df['end_date'] = pd.to_datetime(scoped_trades_df['end_date'], errors='coerce').dt.normalize()
    scoped_trades_df = scoped_trades_df[scoped_trades_df['end_date'].notna()].copy()
    scoped_trades_df['cargo_mcm'] = pd.to_numeric(scoped_trades_df['cargo_mcm'], errors='coerce').fillna(0.0)
    scoped_trades_df['origin_country'] = _normalize_scope_series(scoped_trades_df['origin_country'], default='Unknown')
    scoped_trades_df['origin_continent_chart'] = _normalize_scope_series(
        scoped_trades_df['origin_continent_chart'],
        default='Unknown'
    )

    mapping_lookup_df = _load_importer_country_mapping_lookup(engine)
    if mapping_lookup_df.empty:
        for column in IMPORTER_MAPPING_RENAME.values():
            scoped_trades_df[column] = 'Unknown'
        return scoped_trades_df[expected_columns]

    scoped_trades_df = pd.merge(
        scoped_trades_df,
        mapping_lookup_df,
        how='left',
        left_on='origin_country',
        right_on='mapping_key'
    ).drop(columns=['mapping_key'])

    for column in IMPORTER_MAPPING_RENAME.values():
        scoped_trades_df[column] = _normalize_scope_series(scoped_trades_df[column], default='Unknown')

    return scoped_trades_df[expected_columns]


def _apply_importer_self_flow_exclusion(scoped_trades_df, selected_destination_aggregation, selected_destination_value):
    if scoped_trades_df is None or scoped_trades_df.empty:
        return pd.DataFrame(columns=scoped_trades_df.columns if scoped_trades_df is not None else [])

    if _normalize_mapping_value(selected_destination_value) is None:
        return scoped_trades_df.copy()

    selection_aggregation = (
        selected_destination_aggregation
        if selected_destination_aggregation in IMPORTER_SELECTION_TO_ORIGIN_SCOPE
        else 'country'
    )
    scope_column = IMPORTER_SELECTION_TO_ORIGIN_SCOPE[selection_aggregation]
    normalized_selected_value = _normalize_scope_value(selected_destination_value)
    return scoped_trades_df[scoped_trades_df[scope_column] != normalized_selected_value].copy()


def _prepare_importer_summary_scope_df(scoped_trades_df, origin_level):
    if scoped_trades_df is None or scoped_trades_df.empty:
        return pd.DataFrame(columns=['end_date', 'cargo_mcm', 'continent', 'country'])

    scope_column = IMPORTER_ORIGIN_LEVEL_TO_SCOPE.get(origin_level, 'origin_shipping_region')
    if scope_column == 'origin_country':
        summary_df = scoped_trades_df[['end_date', 'cargo_mcm', 'origin_country']].copy()
        summary_df['continent'] = summary_df['origin_country']
        summary_df['country'] = summary_df['origin_country']
        summary_df = summary_df[['end_date', 'cargo_mcm', 'continent', 'country']]
    else:
        summary_df = scoped_trades_df[['end_date', 'cargo_mcm', scope_column, 'origin_country']].copy()
        summary_df = summary_df.rename(columns={
            scope_column: 'continent',
            'origin_country': 'country',
        })
    summary_df['continent'] = _normalize_scope_series(summary_df['continent'], default='Unknown')
    summary_df['country'] = _normalize_scope_series(summary_df['country'], default='Unknown')
    return summary_df


def _build_importer_periods_pivot(summary_scope_df, period_type, current_date=None):
    expected_columns = ['continent', 'country']
    if summary_scope_df is None or summary_scope_df.empty:
        return pd.DataFrame(columns=expected_columns)

    reference_date = pd.Timestamp(current_date or dt.date.today()).normalize()
    historical_start = reference_date - pd.DateOffset(years=2)
    historical_df = summary_scope_df[
        (summary_scope_df['end_date'] >= historical_start) &
        (summary_scope_df['end_date'] < reference_date)
    ].copy()
    if historical_df.empty:
        return pd.DataFrame(columns=expected_columns)

    if period_type == 'quarter':
        historical_df['period'] = (
            'Q' + historical_df['end_date'].dt.quarter.astype(str) +
            "'" + historical_df['end_date'].dt.strftime('%y')
        )
        grouped_df = historical_df.groupby(
            ['continent', 'country', 'period'],
            dropna=False,
            as_index=False
        )['cargo_mcm'].sum()
        grouped_df['mcm_d'] = grouped_df['cargo_mcm'] / 91.25
    elif period_type == 'month':
        historical_df['period'] = historical_df['end_date'].dt.strftime("%b'%y")
        historical_df['days_in_period'] = historical_df['end_date'].dt.days_in_month.astype(float)
        grouped_df = historical_df.groupby(
            ['continent', 'country', 'period', 'days_in_period'],
            dropna=False,
            as_index=False
        )['cargo_mcm'].sum()
        grouped_df['mcm_d'] = grouped_df['cargo_mcm'] / grouped_df['days_in_period']
    else:
        historical_df['period'] = (
            'W' + historical_df['end_date'].dt.isocalendar().week.astype(int).astype(str) +
            "'" + historical_df['end_date'].dt.strftime('%y')
        )
        grouped_df = historical_df.groupby(
            ['continent', 'country', 'period'],
            dropna=False,
            as_index=False
        )['cargo_mcm'].sum()
        grouped_df['mcm_d'] = grouped_df['cargo_mcm'] / 7.0

    return grouped_df.pivot_table(
        index=['continent', 'country'],
        columns='period',
        values='mcm_d',
        aggfunc='sum',
        fill_value=0
    ).reset_index()


def _build_importer_rolling_windows_pivot(summary_scope_df, rolling_window_days=30, current_date=None):
    expected_columns = ['continent', 'country', '7D', format_rolling_window_label(rolling_window_days)]
    if summary_scope_df is None or summary_scope_df.empty:
        return pd.DataFrame(columns=expected_columns)

    normalized_window_days = normalize_rolling_window_days(rolling_window_days)
    rolling_window_label = format_rolling_window_label(normalized_window_days)
    reference_date = pd.Timestamp(current_date or dt.date.today()).normalize()
    date_7d_ago = reference_date - pd.Timedelta(days=7)
    date_window_ago = reference_date - pd.Timedelta(days=normalized_window_days)
    date_window_y1_start = reference_date - pd.Timedelta(days=365 + normalized_window_days)
    date_window_y1_end = reference_date - pd.Timedelta(days=365)

    relevant_df = summary_scope_df[
        (summary_scope_df['end_date'] >= date_window_y1_start) &
        (summary_scope_df['end_date'] <= reference_date)
    ].copy()
    if relevant_df.empty:
        return pd.DataFrame(columns=expected_columns)

    all_combinations_df = relevant_df[['continent', 'country']].drop_duplicates().reset_index(drop=True)
    if all_combinations_df.empty:
        return pd.DataFrame(columns=expected_columns)

    all_combinations = pd.MultiIndex.from_frame(all_combinations_df)
    full_date_index = pd.date_range(date_window_y1_start + pd.Timedelta(days=1), reference_date, freq='D')

    daily_pivot = relevant_df.groupby(
        ['end_date', 'continent', 'country'],
        dropna=False,
        as_index=False
    )['cargo_mcm'].sum().pivot(
        index='end_date',
        columns=['continent', 'country'],
        values='cargo_mcm'
    )
    daily_pivot = daily_pivot.reindex(full_date_index, fill_value=0)
    daily_pivot = daily_pivot.reindex(columns=all_combinations, fill_value=0).fillna(0)

    avg_7d = daily_pivot.loc[date_7d_ago + pd.Timedelta(days=1):reference_date].mean()
    avg_window = daily_pivot.loc[date_window_ago + pd.Timedelta(days=1):reference_date].mean()
    avg_window_y1 = daily_pivot.loc[
        date_window_y1_start + pd.Timedelta(days=1):date_window_y1_end
    ].mean()

    rolling_df = pd.concat([
        avg_7d.rename('7D'),
        avg_window.rename(rolling_window_label),
        avg_window_y1.rename(f'{rolling_window_label}_Y1'),
    ], axis=1).reset_index()
    rolling_df.columns = ['continent', 'country', '7D', rolling_window_label, f'{rolling_window_label}_Y1']
    rolling_df[f'Δ 7D-{rolling_window_label}'] = rolling_df['7D'] - rolling_df[rolling_window_label]
    rolling_df[f'Δ {rolling_window_label} Y/Y'] = (
        rolling_df[rolling_window_label] - rolling_df[f'{rolling_window_label}_Y1']
    )
    return rolling_df


def _build_importer_chart_date_index(start_date=None, forecast_days=14, current_date=None):
    reference_date = pd.Timestamp(current_date or dt.date.today()).normalize()
    chart_start_date = pd.Timestamp(start_date or SUMMARY_LOOKBACK_START).normalize()
    chart_end_date = reference_date + pd.Timedelta(days=forecast_days)
    return pd.date_range(chart_start_date, chart_end_date, freq='D'), reference_date


def _build_importer_total_import_df(scoped_trades_df, rolling_window_days=30, current_date=None):
    if scoped_trades_df is None or scoped_trades_df.empty:
        return pd.DataFrame(columns=['date', 'year', 'day_of_year', 'month_day', 'rolling_avg', 'is_forecast'])

    date_index, reference_date = _build_importer_chart_date_index(current_date=current_date)
    daily_series = scoped_trades_df.groupby('end_date')['cargo_mcm'].sum()
    daily_series = daily_series.reindex(date_index, fill_value=0)
    rolling_avg = daily_series.rolling(
        window=normalize_rolling_window_days(rolling_window_days),
        min_periods=1
    ).mean()

    result_df = pd.DataFrame({
        'date': date_index,
        'year': date_index.year.astype(int),
        'day_of_year': date_index.dayofyear.astype(int),
        'month_day': date_index.strftime('%b %d'),
        'rolling_avg': rolling_avg.to_numpy(),
        'is_forecast': date_index > reference_date,
    })
    return result_df[result_df['date'] >= pd.Timestamp('2024-01-01')].reset_index(drop=True)


def _build_importer_continent_chart_df(scoped_trades_df, rolling_window_days=30, current_date=None,
                                       include_percentage=False):
    expected_columns = [
        'date', 'continent_origin', 'year', 'day_of_year', 'month_day', 'rolling_avg', 'is_forecast'
    ]
    if include_percentage:
        expected_columns.insert(6, 'percentage')

    if scoped_trades_df is None or scoped_trades_df.empty:
        return pd.DataFrame(columns=expected_columns)

    continents = sorted(scoped_trades_df['origin_continent'].dropna().unique().tolist())
    if not continents:
        return pd.DataFrame(columns=expected_columns)

    date_index, reference_date = _build_importer_chart_date_index(current_date=current_date)
    daily_matrix = scoped_trades_df.groupby(
        ['end_date', 'origin_continent'],
        dropna=False,
        as_index=False
    )['cargo_mcm'].sum().pivot(
        index='end_date',
        columns='origin_continent',
        values='cargo_mcm'
    )
    daily_matrix = daily_matrix.reindex(date_index, fill_value=0)
    daily_matrix = daily_matrix.reindex(columns=continents, fill_value=0).fillna(0)
    rolling_matrix = daily_matrix.rolling(
        window=normalize_rolling_window_days(rolling_window_days),
        min_periods=1
    ).mean()

    melted_df = rolling_matrix.stack().reset_index()
    melted_df.columns = ['date', 'continent_origin', 'rolling_avg']
    melted_df['year'] = melted_df['date'].dt.year.astype(int)
    melted_df['day_of_year'] = melted_df['date'].dt.dayofyear.astype(int)
    melted_df['month_day'] = melted_df['date'].dt.strftime('%b %d')
    melted_df['is_forecast'] = melted_df['date'] > reference_date
    if include_percentage:
        total_rolling_avg = melted_df.groupby('date')['rolling_avg'].transform('sum')
        melted_df['percentage'] = np.where(
            total_rolling_avg > 0,
            (melted_df['rolling_avg'] / total_rolling_avg) * 100,
            0
        )

    melted_df = melted_df[melted_df['date'] >= pd.Timestamp('2024-01-01')].reset_index(drop=True)
    return melted_df[expected_columns]


def process_trade_and_distance_data(engine, destination_countries):
    """
    Loads trade and distance data from a database for the selected importer destinations,
    joins them, calculates mileage ratios, determines the most likely route based on ratios,
    and flags deviations.

    Args:
        engine: SQLAlchemy engine object.
        destination_countries (list[str] | str): Destination countries to filter by.

    Returns:
        pandas.DataFrame: A DataFrame containing the joined and processed data.
    """
    trades_table_name = "kpler_trades"
    distance_table_name = "kpler_distance_matrix"

    final_df = None
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    if not normalized_destination_countries:
        return pd.DataFrame()

    try:
        # Load trades filtered by destination countries (importer selection)
        try:
            query = text(f'''
                SELECT
                    voyage_id,
                    vessel_name,
                    start,
                    "end",
                    origin_location_name,
                    origin_country_name,
                    zone_origin_name,
                    destination_location_name,
                    zone_destination_name,
                    destination_country_name,
                    mileage_nautical_miles,
                    origin_reload_sts_partial,
                    destination_reload_sts_partial
                FROM {DB_SCHEMA}.{trades_table_name}
                WHERE status = 'Delivered'
                    AND destination_country_name IN :destination_countries
                    AND zone_origin_name <> zone_destination_name
                    AND upload_timestamp_utc = (
                        SELECT MAX(upload_timestamp_utc)
                        FROM {DB_SCHEMA}.{trades_table_name}
                    )
            ''')
            trades_df = pd.read_sql(
                query,
                engine,
                params={'destination_countries': normalized_destination_countries}
            )
        except Exception as e:
            return None

        # Load distance data
        try:
            distance_df = pd.read_sql(f'''SELECT "originLocationName", "destinationLocationName", "distanceDirect", "distanceViaSuez", "distanceViaPanama"
                                          FROM {DB_SCHEMA}.{distance_table_name}''', engine)
            distance_df.columns = ['originLocationName', 'destinationLocationName', 'distanceDirect', 'distanceViaSuez',
                                   'distanceViaPanama']
        except Exception as e:
            distance_df = pd.DataFrame(
                columns=['originLocationName', 'destinationLocationName', 'distanceDirect', 'distanceViaSuez',
                         'distanceViaPanama'])

        # Join DataFrames
        final_df = pd.merge(
            trades_df,
            distance_df,
            how='left',
            left_on=['zone_origin_name', 'zone_destination_name'],
            right_on=['originLocationName', 'destinationLocationName']
        )

        # Extract date components - use "end" date (arrival) for importer
        final_df['year'] = final_df['end'].dt.year
        final_df['month'] = final_df['end'].dt.month
        final_df['season'] = np.where(final_df['month'].isin([10, 11, 12, 1, 2, 3]), 'W', 'S')
        final_df['quarter'] = 'Q' + final_df['end'].dt.quarter.astype(str)

        # Calculate Ratios
        final_df['mileage_nautical_miles'] = pd.to_numeric(final_df['mileage_nautical_miles'], errors='coerce')
        final_df['distanceDirect'] = pd.to_numeric(final_df['distanceDirect'], errors='coerce')
        final_df['distanceViaSuez'] = pd.to_numeric(final_df['distanceViaSuez'], errors='coerce')
        final_df['distanceViaPanama'] = pd.to_numeric(final_df['distanceViaPanama'], errors='coerce')

        final_df['ratio_miles_distancedirect'] = final_df['mileage_nautical_miles'] / final_df['distanceDirect']
        final_df['ratio_miles_distanceviasuez'] = final_df['mileage_nautical_miles'] / final_df['distanceViaSuez']
        final_df['ratio_miles_distanceviapanama'] = final_df['mileage_nautical_miles'] / final_df['distanceViaPanama']
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Calculate differences from 1
        final_df['diff_direct'] = (final_df['ratio_miles_distancedirect'] - 1).abs()
        final_df['diff_suez'] = (final_df['ratio_miles_distanceviasuez'] - 1).abs()
        final_df['diff_panama'] = (final_df['ratio_miles_distanceviapanama'] - 1).abs()

        # Select route
        diff_cols = ['diff_direct', 'diff_suez', 'diff_panama']
        has_valid = final_df[diff_cols].notna().any(axis=1)
        final_df['closest_route_col'] = pd.NA
        final_df.loc[has_valid, 'closest_route_col'] = (
            final_df.loc[has_valid, diff_cols].idxmin(axis=1, skipna=True)
        )
        route_map = {
            'diff_direct': 'Direct',
            'diff_suez': 'ViaSuez',
            'diff_panama': 'ViaPanama'
        }
        final_df['selected_route'] = final_df['closest_route_col'].map(route_map)

        # Closeness check
        closeness_tolerance = 0.2
        lower_bound = 1 - closeness_tolerance
        upper_bound = 1 + closeness_tolerance

        is_direct_close = final_df['ratio_miles_distancedirect'].between(lower_bound, upper_bound, inclusive='both')
        is_suez_close = final_df['ratio_miles_distanceviasuez'].between(lower_bound, upper_bound, inclusive='both')
        is_panama_close = final_df['ratio_miles_distanceviapanama'].between(lower_bound, upper_bound, inclusive='both')
        any_ratio_is_close = is_direct_close | is_suez_close | is_panama_close
        final_df['no_ratio_close_to_1'] = ~any_ratio_is_close

        missing_distance_count = final_df['distanceDirect'].isnull().sum()

    except Exception as e:
        return None

    finally:
        if 'engine' in locals() and engine:
            engine.dispose()

    return final_df



def kpler_analysis(engine,
                   origin_level='origin_shipping_region',
                   destination_countries=None):
    """
    Fetches Kpler trade data for the selected importer destinations, calculates non-laden voyages,
    adds region mappings, and aggregates metrics by origin region/country pair, vessel type,
    year, season, quarter, and status.

    Args:
        engine: SQLAlchemy engine object for database connection.
        origin_level (str): Level of origin aggregation: 'origin_shipping_region' or 'origin_country_name'.
        destination_countries (list[str] | str): Importing countries to filter by.

    Returns:
        pd.DataFrame: Aggregated DataFrame with trade metrics.
    """
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    if not normalized_destination_countries:
        return pd.DataFrame()

    # Fetch laden trades filtered by destination countries (importer selection)
    query_trades = text(f'''
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
          AND a.origin_country_name IS NOT NULL
          AND a.destination_country_name IN :destination_countries
          AND a."end" IS NOT NULL
          AND a."start" IS NOT NULL
          AND status = 'Delivered'
    ''')
    df_trades = pd.read_sql(
        query_trades,
        engine,
        params={'destination_countries': normalized_destination_countries}
    )

    # --- Infer Non-Laden Voyages ---
    df_trades = df_trades.sort_values(['vessel_name', 'start', 'end']).reset_index(drop=True)

    df_trades['prev_end'] = df_trades.groupby('vessel_name')['end'].shift(1)
    df_trades['prev_dest_country'] = df_trades.groupby('vessel_name')['destination_country_name'].shift(1)
    df_trades['prev_dest_location'] = df_trades.groupby('vessel_name')['destination_location_name'].shift(1)
    df_trades['prev_vessel_capacity'] = df_trades.groupby('vessel_name')['vessel_capacity_cubic_meters'].shift(1)

    mask = df_trades['prev_end'].notna()
    new_rows = pd.DataFrame({
        'vessel_name': df_trades.loc[mask, 'vessel_name'],
        'vessel_type': df_trades.loc[mask, 'vessel_type'],
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
    })

    df_trades = pd.concat([df_trades, new_rows], ignore_index=True)
    df_trades = df_trades.drop(columns=[
        'prev_end', 'prev_dest_country', 'prev_dest_location', 'prev_vessel_capacity'
    ])
    df_trades = df_trades.sort_values(['vessel_name', 'start']).reset_index(drop=True)

    # --- Feature Engineering ---
    df_trades['delivery_days'] = (df_trades['end'] - df_trades['start']).dt.days
    df_trades = df_trades[df_trades['delivery_days'] > 0].copy()

    df_trades['speed'] = np.nan
    valid_speed_mask = (df_trades['mileage_nautical_miles'].notna()) & (df_trades['delivery_days'] > 0)
    df_trades.loc[valid_speed_mask, 'speed'] = (
            df_trades.loc[valid_speed_mask, 'mileage_nautical_miles'] /
            df_trades.loc[valid_speed_mask, 'delivery_days'] / 24
    )

    # Use "end" date (arrival) for importer perspective
    df_trades['year'] = df_trades['end'].dt.year
    df_trades['month'] = df_trades['end'].dt.month
    df_trades['week'] = df_trades['end'].dt.isocalendar().week

    df_trades['status'] = np.where(df_trades['status'] != 'non_laden', 'laden', 'non_laden')
    df_trades['season'] = np.where(df_trades['month'].isin([10, 11, 12, 1, 2, 3]), 'W', 'S')
    df_trades['quarter'] = 'Q' + df_trades['end'].dt.quarter.astype(str)

    df_trades['utilization_rate'] = np.nan
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
    # For importer: fixed on destination_country_name, variable grouping on origin level
    group_columns = [
        'vessel_type', 'status',
        'year', 'season', 'quarter', 'month', 'week',
        'destination_country_name'
    ]

    # Add appropriate origin column based on parameter
    if origin_level == 'origin_country_name':
        group_columns.append('origin_country_name')
    elif origin_level in ('origin_basin', 'continent_origin_name', 'origin_subcontinent',
                          'origin_classification_level1', 'origin_classification'):
        level_col_map = {
            'origin_basin':                 'basin',
            'continent_origin_name':        'continent',
            'origin_subcontinent':          'subcontinent',
            'origin_classification_level1': 'country_classification_level1',
            'origin_classification':        'country_classification',
        }
        mapping_col = level_col_map[origin_level]
        mapping_df = pd.read_sql(
            f"SELECT DISTINCT country_name AS origin_country_name, {mapping_col} AS \"{origin_level}\" FROM {DB_SCHEMA}.mappings_country WHERE country_name IS NOT NULL",
            engine
        )
        df_trades = pd.merge(df_trades, mapping_df, on='origin_country_name', how='left')
        group_columns.append(origin_level)
    else:
        # Default to shipping region grouping
        group_columns.append('origin_shipping_region')

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
        count_trades=('vessel_name', 'count')
    ).reset_index()


    return df_trades_shipping_region



def prepare_pivot_table(df, values_col, filters, aggregation_level='Year', add_total_column=False, aggfunc='sum',
                        origin_level='origin_shipping_region'):
    """
    Prepare data pivoted by origin region/country for the tables, with flexible time aggregation.
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

    required_input_cols = [origin_level, values_col] + list(filters.keys()) + index_cols
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

    grouping_cols = index_cols + [origin_level]
    if not all(col in filtered_df.columns for col in grouping_cols):
        return empty_df

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
            columns=origin_level,
            values=values_col,
            aggfunc='first',
            fill_value=np.nan
        )

    except Exception as e:
        return empty_df

    if not pivot_df.empty:
        pivot_df = pivot_df.sort_index()

    if add_total_column and not pivot_df.empty:
        region_cols = pivot_df.columns.tolist()
        if region_cols:
            try:
                pivot_df['Total'] = pivot_df[region_cols].sum(axis=1, skipna=True)
            except Exception as e:
                pass

    if not pivot_df.empty:
        pivot_df = pivot_df.reset_index()

    return pivot_df


def create_stacked_bar_chart(df, metric, title_suffix, selected_status=None, selected_vessel_type=None,
                             aggregation_level='Year', is_intracountry=False,
                             origin_level='origin_shipping_region'):
    """
    Create a Plotly visualization showing data by selected aggregation level and origin regions/countries.
    """

    # Filter the data
    filtered_df = df.copy()

    if selected_status and 'status' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['status'] == selected_status]
    if selected_vessel_type and selected_vessel_type != 'All' and 'vessel_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['vessel_type'] == selected_vessel_type]

    if filtered_df.empty:
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
        if 'destination_country_name' not in filtered_df.columns:
            fig = go.Figure()
            fig.update_layout(
                title="Error: Missing 'destination_country_name' column",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            return fig
        group_field = 'destination_country_name'
        vessel_type_text = f", {selected_vessel_type}" if selected_vessel_type and selected_vessel_type != 'All' else ""
        chart_title = f'Intracountry {title_suffix} by {x_axis_title}{vessel_type_text} and Destination Country'
        legend_title = 'Destination Country'
    else:
        if origin_level not in filtered_df.columns:
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: Missing {origin_level} column",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            return fig

        group_field = origin_level
        origin_label = "Countries" if origin_level == "origin_country_name" else "Shipping Regions"
        vessel_type_text = f", {selected_vessel_type}" if selected_vessel_type and selected_vessel_type != 'All' else ""
        chart_title = f'{title_suffix} by {x_axis_title}{vessel_type_text} and Origin {origin_label}'
        legend_title = 'Origin ' + (
            'Country' if origin_level == "origin_country_name" else 'Shipping Region')

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
    if metric not in filtered_df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=f"Error: Metric '{metric}' not found",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    try:
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

    # Create x-axis labels
    if len(groupby_time_cols) > 1:
        if 'year' in stacked_data.columns:
            if 'season' in stacked_data.columns:
                stacked_data['time_label'] = stacked_data['year'].astype(str) + '-' + stacked_data['season'].astype(str)
            elif 'quarter' in stacked_data.columns:
                stacked_data['time_label'] = stacked_data['year'].astype(str) + '-' + stacked_data['quarter'].astype(str)
            else:
                stacked_data['time_label'] = stacked_data['year'].astype(str)
        else:
            stacked_data['time_label'] = 'Unknown'
    else:
        stacked_data['time_label'] = stacked_data['year'].astype(str)

    time_labels = sorted(stacked_data['time_label'].unique())
    group_values = sorted(stacked_data[group_field].unique())

    # Use professional color palette
    n_groups = len(group_values)
    distinct_colors = get_professional_colors(n_groups)

    fig = go.Figure()

    for i, group_value in enumerate(group_values):
        color = distinct_colors[i % len(distinct_colors)]
        filtered_group_data = stacked_data[stacked_data[group_field] == group_value]
        fig.add_trace(go.Bar(
            x=filtered_group_data['time_label'],
            y=filtered_group_data[metric],
            name=group_value,
            marker_color=color,
            showlegend=True,
            hoverinfo='y+name+x'
        ))

    fig.update_layout(barmode='stack')
    fig.update_xaxes(
        title=x_axis_title,
        type='category',
        categoryorder='category ascending'
    )
    fig.update_yaxes(title=title_suffix)

    fig = apply_professional_chart_styling(
        fig,
        title=chart_title,
        height=700,
        show_legend=True,
        legend_title=legend_title
    )

    return fig


def create_datatable(data, metric_for_format=None, aggregation_level='Year'):
    """Create a formatted DataTable from the provided pivoted data."""
    columns = []
    if data is None or data.empty:
        return dash_table.DataTable(
            columns=[{'name': 'Status', 'id': 'status_col'}],
            data=[{'status_col': 'No data available for the selected filters.'}],
            style_cell={'textAlign': 'center'}
        )

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

    time_cols = [col for col in time_cols if col in data.columns]
    data_cols = [col for col in data.columns if col not in time_cols]

    data_display = data.copy()
    sort_columns = []
    sort_ascending = []
    if 'year' in time_cols:
        sort_columns.append('year')
        sort_ascending.append(False)
    if 'quarter' in time_cols:
        if 'quarter' in data_display.columns:
            data_display['quarter_num'] = data_display['quarter'].str.extract(r'(\d+)').astype(int)
            sort_columns.append('quarter_num')
            sort_ascending.append(False)
    elif 'season' in time_cols:
        sort_columns.append('season')
        sort_ascending.append(False)
    elif 'month' in time_cols:
        if 'month' in data_display.columns:
            sort_columns.append('month')
            sort_ascending.append(False)
    elif 'week' in time_cols:
        if 'week' in data_display.columns:
            sort_columns.append('week')
            sort_ascending.append(False)
    if sort_columns:
        data_display = data_display.sort_values(by=sort_columns, ascending=sort_ascending)
        if 'quarter_num' in data_display.columns:
            data_display = data_display.drop(columns=['quarter_num'])

    is_ton_miles = (metric_for_format == 'sum_ton_miles')
    if is_ton_miles:
        cols_to_divide = [col for col in data_cols if pd.api.types.is_numeric_dtype(data_display[col])]
        if cols_to_divide:
            data_display[cols_to_divide] = data_display[cols_to_divide] / 1_000_000

    for col in data_display.columns:
        col_name = str(col).replace('_', ' ').title()
        col_id = str(col)

        if col in time_cols:
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

            columns.append({
                "name": col_name,
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

    conditional_styles = []
    if 'Total' in data_display.columns:
        conditional_styles.append({
            'if': {'column_id': 'Total'},
            'fontWeight': 'bold',
            'backgroundColor': 'rgb(240, 240, 240)'
        })

    numeric_data_cols = [
        col for col in data_cols
        if pd.api.types.is_numeric_dtype(data_display[col])
    ]
    for col_id in numeric_data_cols:
        conditional_styles.append({
            'if': {'column_id': col_id},
            'textAlign': 'right'
        })

    return dash_table.DataTable(
        columns=columns,
        data=data_display.to_dict('records'),
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
        merge_duplicate_headers=True,
        page_size=20,
        sort_action='native',
        export_format='xlsx',
        export_headers='display',
        export_columns='visible',
        fill_width=False
    )


def create_route_analysis_table(df, aggregation_level, route_scenario_title, include_route_column=True,
                                origin_level='origin_country_name'):
    """
    Creates a table showing aggregated trade counts by origin column (columns) and time periods (rows)
    for a specific route analysis scenario.
    """
    if df is None or df.empty:
        return dash_table.DataTable(
            columns=[{'name': 'Status', 'id': 'status_col'}],
            data=[{'status_col': f'No data available for {route_scenario_title}'}],
            style_cell={'textAlign': 'center'}
        )

    if aggregation_level == 'Year':
        index_cols = ['year']
    elif aggregation_level == 'Year+Season':
        index_cols = ['year', 'season']
    elif aggregation_level == 'Year+Quarter':
        index_cols = ['year', 'quarter']
    else:
        index_cols = ['year']

    required_cols = index_cols + [origin_level, 'voyage_id']
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
        if include_route_column and 'selected_route' not in df.columns:
            df['selected_route'] = 'Unknown'

        unique_routes = ['All Routes']
        if include_route_column:
            unique_routes = df['selected_route'].unique().tolist()

        result_tables = {}

        if include_route_column:
            for route in unique_routes:
                route_df = df[df['selected_route'] == route].copy()
                groupby_cols = index_cols + [origin_level]
                grouped_df = route_df.groupby(groupby_cols, observed=True)['voyage_id'].count().reset_index()
                pivot_df = pd.pivot_table(
                    grouped_df,
                    index=index_cols,
                    columns=origin_level,
                    values='voyage_id',
                    aggfunc='sum',
                    fill_value=0
                ).reset_index()
                result_tables[route] = pivot_df
        else:
            groupby_cols = index_cols + [origin_level]
            grouped_df = df.groupby(groupby_cols, observed=True)['voyage_id'].count().reset_index()
            pivot_df = pd.pivot_table(
                grouped_df,
                index=index_cols,
                columns=origin_level,
                values='voyage_id',
                aggfunc='sum',
                fill_value=0
            ).reset_index()
            result_tables['All Routes'] = pivot_df

        all_pivots = []
        for route, pivot_df in result_tables.items():
            if len(index_cols) > 1:
                if 'season' in index_cols and 'year' in index_cols and 'season' in pivot_df.columns:
                    pivot_df['time_period'] = pivot_df['year'].astype(str) + '-' + pivot_df['season'].astype(str)
                elif 'quarter' in index_cols and 'year' in index_cols and 'quarter' in pivot_df.columns:
                    pivot_df['time_period'] = pivot_df['year'].astype(str) + '-' + pivot_df['quarter'].astype(str)

            time_cols = index_cols + ['time_period']
            time_cols = [col for col in time_cols if col in pivot_df.columns]
            origin_cols = [col for col in pivot_df.columns if col not in time_cols]
            pivot_df_copy = pivot_df.copy()
            col_map = {col: f"{route}||{col}" for col in origin_cols}
            pivot_df_copy = pivot_df_copy.rename(columns=col_map)
            all_pivots.append(pivot_df_copy)

        final_df = all_pivots[0]
        for pivot_df in all_pivots[1:]:
            final_df = pd.merge(
                final_df,
                pivot_df,
                on=time_cols,
                how='outer'
            )

        # Sort descending
        if 'year' in final_df.columns:
            if 'quarter' in final_df.columns:
                final_df['quarter_num'] = final_df['quarter'].str.extract(r'(\d+)').astype(int)
                final_df['sort_key'] = final_df['year'] * 10 + final_df['quarter_num']
                final_df = final_df.sort_values('sort_key', ascending=False)
                final_df = final_df.drop(columns=['quarter_num', 'sort_key'])
            elif 'season' in final_df.columns:
                final_df['season_num'] = final_df['season'].apply(lambda x: 1 if x == 'S' else 2)
                final_df['sort_key'] = final_df['year'] * 10 + final_df['season_num']
                final_df = final_df.sort_values('sort_key', ascending=False)
                final_df = final_df.drop(columns=['season_num', 'sort_key'])
            else:
                final_df = final_df.sort_values('year', ascending=False)
        elif 'time_period' in final_df.columns:
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

        display_cols = []
        if 'time_period' in final_df.columns:
            display_cols.append('time_period')
        else:
            display_cols.extend(index_cols)

        data_cols = [col for col in final_df.columns if col not in display_cols and '||' in col]
        display_cols.extend(data_cols)
        display_cols = [col for col in display_cols if col in final_df.columns]
        final_df = final_df[display_cols]

        table_columns = []
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

        for col in data_cols:
            route, origin = col.split('||', 1)
            table_columns.append({
                "name": [route, origin],
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

        conditional_styles = []
        for col_id in data_cols:
            conditional_styles.append({
                'if': {'column_id': str(col_id)},
                'textAlign': 'right'
            })

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
            merge_duplicate_headers=True,
            page_size=25,
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
    # Store components for importer data
    dcc.Store(id='imp-region-data-store', storage_type='memory'),
    dcc.Store(id='imp-dropdown-options-store', storage_type='local'),
    dcc.Store(id='imp-destination-catalog-store', storage_type='local'),
    dcc.Store(id='imp-destination-selection-store', storage_type='local'),
    dcc.Store(id='imp-refresh-timestamp-store', storage_type='local'),
    dcc.Store(id='imp-diversion-processed-data', storage_type='memory'),
    dcc.Store(id='imp-origin-expanded-continents', data=[]),  # Store for expanded state of continents
    dcc.Store(id='imp-origin-forecast-expanded-continents', data=[]),  # Store for WoodMac forecast table expansion
    dcc.Store(id='imp-maintenance-expanded-plants', data=[]),  # Store for expanded state of plants
    dcc.Download(id='imp-download-importer-detail-supply-excel'),
    dcc.Download(id='imp-download-trade-analysis-excel'),
    dcc.Download(id='imp-download-route-analysis-excel'),
    dcc.Download(id='imp-download-diversion-summary-excel'),

    # Professional Section Header - Importer Analysis Configuration
    html.Div([
        html.Div([

            # --- Group 1: Destination ---
            html.Div([
                html.Div("Destination", className='filter-group-header'),
                html.Div([
                    html.Div([
                        html.Label("Aggregation:", className='filter-label'),
                        dcc.Dropdown(
                            id='imp-destination-aggregation-dropdown',
                            options=DESTINATION_AGGREGATION_OPTIONS,
                            value='country',
                            multi=False,
                            clearable=False,
                            className='filter-dropdown',
                            style={'min-width': '180px'}
                        ),
                    ], className='filter-group'),
                    html.Div("→", className='filter-dependency-arrow'),
                    html.Div([
                        html.Label("Destination:", className='filter-label'),
                        dcc.Dropdown(
                            id='imp-destination-country-dropdown',
                            options=[],
                            value='China',
                            multi=False,
                            clearable=False,
                            className='filter-dropdown',
                            style={'min-width': '180px'}
                        ),
                    ], className='filter-group'),
                ], style={'display': 'flex', 'gap': '8px', 'alignItems': 'flex-end'}),
            ], className='filter-section filter-section-destination'),

            # --- Group 2: Origin ---
            html.Div([
                html.Div("Origin", className='filter-group-header'),
                html.Div([
                    html.Div([
                        html.Label("Origin Level:", className='filter-label'),
                        dcc.Dropdown(
                            id='imp-origin-level-dropdown',
                            options=[
                                {'label': 'Shipping Region',        'value': 'origin_shipping_region'},
                                {'label': 'Country',                'value': 'origin_country_name'},
                                {'label': 'Basin',                  'value': 'origin_basin'},
                                {'label': 'Continent',              'value': 'continent_origin_name'},
                                {'label': 'Subcontinent',           'value': 'origin_subcontinent'},
                                {'label': 'Classification Level 1', 'value': 'origin_classification_level1'},
                                {'label': 'Classification',         'value': 'origin_classification'},
                            ],
                            value='origin_shipping_region',
                            multi=False,
                            clearable=False,
                            className='filter-dropdown',
                            style={'min-width': '180px'}
                        ),
                    ], className='filter-group'),
                ], style={'display': 'flex', 'gap': '8px', 'alignItems': 'flex-end'}),
            ], className='filter-section filter-section-origin'),

            # --- Group 3: Analysis Settings ---
            html.Div([
                html.Div("Analysis Settings", className='filter-group-header'),
                html.Div([
                    html.Div([
                        html.Label("Aggregation:", className='filter-label'),
                        dcc.Dropdown(
                            id='imp-aggregation-dropdown',
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
                            id='imp-region-status-dropdown',
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
                            id='imp-vessel-type-dropdown',
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
                            id='imp-chart-metric-dropdown',
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

    # Country Import Charts Section - Three charts side by side
    html.Div([
        # Section Header
        html.Div([
            html.H3(
                'LNG Import Analysis - 30-Day Rolling Average + WoodMac Forecast',
                id='imp-supply-analysis-title',
                className="section-title-inline"
            ),
            html.Label("Window (days):", className="inline-filter-label", style={'marginLeft': '20px'}),
            dcc.Input(
                id='imp-supply-rolling-window-input',
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
                id='imp-export-supply-analysis-button',
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
            # Left Chart - Country Imports
            html.Div([
                html.H4(id='imp-country-supply-header', children='Total Imports + WoodMac Forecast',
                       style={'fontSize': '14px', 'marginBottom': '10px', 'color': '#4A4A4A'}),
                dcc.Loading(
                    id="imp-country-supply-loading",
                    children=[
                        dcc.Graph(id='imp-country-supply-chart', style={'height': '400px'})
                    ],
                    type="default",
                )
            ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div(style={'width': '2%', 'display': 'inline-block'}),

            # Middle Chart - Continent Origins (Absolute)
            html.Div([
                html.H4(id='imp-continent-origin-header', children='By Origin Continent (mcm/d)',
                       style={'fontSize': '14px', 'marginBottom': '10px', 'color': '#4A4A4A'}),
                dcc.Loading(
                    id="imp-continent-origin-loading",
                    children=[
                        dcc.Graph(id='imp-continent-origin-chart', style={'height': '400px'})
                    ],
                    type="default",
                )
            ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div(style={'width': '2%', 'display': 'inline-block'}),

            # Right Chart - Continent Origins (Percentage)
            html.Div([
                html.H4(id='imp-continent-percentage-header', children='By Origin Continent (%)',
                       style={'fontSize': '14px', 'marginBottom': '10px', 'color': '#4A4A4A'}),
                dcc.Loading(
                    id="imp-continent-percentage-loading",
                    children=[
                        dcc.Graph(id='imp-continent-percentage-chart', style={'height': '400px'})
                    ],
                    type="default",
                )
            ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'padding': '20px'}),

        html.Div(
            "Note: Rolling averages apply only to Kpler-based data. Forecasts from non-Kpler sources, including WoodMac, are shown without a rolling average.",
            style={
                'padding': '0 20px 20px 20px',
                'fontSize': '12px',
                'color': '#6b7280',
                'fontStyle': 'italic'
            }
        )
    ], className="main-section-container", style={'marginBottom': '30px'}),

    # Origin Analysis + Trade Analysis side by side
    html.Div([
        # Left: Origin Analysis Summary
        html.Div([
            html.Div([
                html.H3('Origin Analysis Summary (mcm/d)', className="section-title-inline"),
            ], className="inline-section-header"),
            html.Div([
                dcc.Loading(
                    id="imp-origin-summary-loading",
                    children=[
                        html.Div(id='imp-origin-summary-table-container')
                    ],
                    type="default"
                )
            ], style={'marginTop': '20px'})
        ], className='section-container', style={'flex': '1', 'minWidth': '0', 'margin': '0'}),

        # Right: Trade Analysis
        html.Div([
            html.Div([
                html.H3(id='imp-trade-analysis-header', className='section-title-inline'),
                html.Button('Export to Excel', id='imp-export-trade-analysis-button', n_clicks=0,
                    style={'marginLeft': '20px', 'padding': '5px 15px', 'backgroundColor': '#28a745',
                           'color': 'white', 'border': 'none', 'borderRadius': '4px',
                           'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '12px'}),
            ], className='inline-section-header', style={'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                dcc.Graph(id='imp-trade-count-visualization', style={'height': '600px'})
            ]),
        ], className='section-container', style={'flex': '1', 'minWidth': '0', 'margin': '0'}),

    ], style={'display': 'flex', 'gap': '24px', 'alignItems': 'flex-start', 'marginBottom': '32px'}),

    # Origin Forecast Allocation Summary Section
    html.Div([
        html.Div([
            html.H3('Origin Forecast Allocation Summary (WoodMac, mcm/d)', className="section-title-inline"),
        ], className="inline-section-header"),
        html.Div(
            id='imp-origin-forecast-summary-subtitle',
            style={
                'marginTop': '8px',
                'fontSize': '12px',
                'color': '#6b7280',
                'fontStyle': 'italic'
            }
        ),

        html.Div([
            dcc.Loading(
                id="imp-origin-forecast-summary-loading",
                children=[
                    html.Div(id='imp-origin-forecast-summary-table-container')
                ],
                type="default"
            )
        ], style={'marginTop': '20px'})
    ], className='section-container', style={'margin-bottom': '32px'}),

    # Supplier Maintenance Schedule Section
    html.Div([
        html.Div([
            html.H3('Supplier Maintenance Schedule (MCM/D Impact)', className="section-title-inline"),
        ], className="inline-section-header"),

        html.Div([
            dcc.Loading(
                id="imp-maintenance-summary-loading",
                children=[
                    html.Div(id='imp-maintenance-summary-container')
                ],
                type="default"
            )
        ], style={'marginTop': '20px'})
    ], className='section-container', style={'margin-bottom': '32px'}),

    # Route Analysis Section
    html.Div([
        html.Div([
            html.H3("Route Analysis", className='section-title-inline'),
            html.Button('Export to Excel', id='imp-export-route-analysis-button', n_clicks=0,
                style={'marginLeft': '20px', 'padding': '5px 15px', 'backgroundColor': '#28a745',
                       'color': 'white', 'border': 'none', 'borderRadius': '4px',
                       'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '12px'}),
        ], className='inline-section-header', style={'display': 'flex', 'alignItems': 'center'}),

        html.Div([
            dcc.Graph(id='imp-graph-route-suez-only', style={'height': '600px'})
        ], style={'margin-bottom': '24px'}),
    ], className='section-container', style={'margin-bottom': '0'}),

    # Diversions Analysis Section
    html.Div([
        html.Div([
            html.H2("Diversions Analysis", className='section-title-inline'),
            html.P("Analyze route deviations and cargoes diverted TO the selected importer", className='section-subtitle')
        ], className='header-content'),

        html.Div([
            html.Div([
                html.Label("Select Origin Level:", className='filter-label'),
                dcc.RadioItems(
                    id='imp-diversion-combo-radio',
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
            dcc.Graph(id='imp-diversion-count-chart', style={'height': '600px'})
        ], style={'margin-bottom': '24px'}),

        html.Div([
            html.H2("Diversions Summary Table", className='mckinsey-header', style={'margin': '0'}),
            html.Button('Export to Excel', id='imp-export-diversion-summary-button', n_clicks=0,
                style={'marginLeft': '20px', 'padding': '5px 15px', 'backgroundColor': '#28a745',
                       'color': 'white', 'border': 'none', 'borderRadius': '4px',
                       'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '12px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
        dash_table.DataTable(
            id='imp-diversion-table',
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


SUMMARY_LOOKBACK_START = dt.date(2023, 11, 1)
MCM_PER_CUBIC_METER = 0.6 / 1000
WOODMAC_IMPORT_EXPORTS_TABLE = 'at_lng.woodmac_gas_imports_exports_monthly__mmtpa'
WOODMAC_LNG_CUBIC_METERS_PER_MMTPA_MONTH = 2222 * 1000 / 12
WOODMAC_FORECAST_YEARS_AHEAD = 2
SUPPLY_ALLOCATION_RUNS_TABLE = f'{DB_SCHEMA}.fundamentals_supply_allocation_runs'
SUPPLY_ALLOCATION_COUNTRY_FLOWS_TABLE = f'{DB_SCHEMA}.fundamentals_supply_allocation_country_flows'
SUPPLY_ALLOCATION_DEMAND_DETAIL_TABLE = f'{DB_SCHEMA}.fundamentals_supply_allocation_demand_detail'
SUPPLY_ALLOCATION_DEMAND_SUMMARY_TABLE = f'{DB_SCHEMA}.fundamentals_supply_allocation_demand_summary'


def _build_importer_summary_context(destination_countries, vessel_type, origin_level='origin_shipping_region'):
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    # Always fetch at country level; Python-side grouping applied in fetch_origin_summary_data
    continent_col = "COALESCE(NULLIF(kt.origin_country_name, ''), 'Unknown')"
    country_col   = "COALESCE(NULLIF(kt.origin_country_name, ''), 'Unknown')"
    where_conditions = [
        "kt.upload_timestamp_utc = latest_timestamp.max_ts",
        "kt.destination_country_name IN :destination_countries",
        'kt."end" IS NOT NULL',
        "kt.cargo_destination_cubic_meters IS NOT NULL",
        "kt.status = 'Delivered'",
    ]
    params = {'destination_countries': normalized_destination_countries}

    if vessel_type and vessel_type != 'All':
        where_conditions.append("mv.vessel_type = :vessel_type")
        params['vessel_type'] = vessel_type

    return continent_col, country_col, " AND ".join(where_conditions), params


def fetch_origin_periods_data_hierarchical(conn, continent_col, country_col, where_clause, params, period_type):
    """Fetch importer-side period averages grouped by supplier continent/country."""
    if period_type == 'quarter':
        period_expr = """
            'Q' || EXTRACT(QUARTER FROM kt."end")::text || '''' ||
            RIGHT(EXTRACT(YEAR FROM kt."end")::text, 2) AS period
        """
        order_expr = 'EXTRACT(YEAR FROM kt."end"), EXTRACT(QUARTER FROM kt."end")'
        value_expr = f"SUM(kt.cargo_destination_cubic_meters * {MCM_PER_CUBIC_METER}) / 91.25"
        trailing_sql = ""
    elif period_type == 'month':
        period_expr = """
            TO_CHAR(kt."end", 'Mon') || '''' ||
            RIGHT(EXTRACT(YEAR FROM kt."end")::text, 2) AS period
        """
        order_expr = 'EXTRACT(YEAR FROM kt."end"), EXTRACT(MONTH FROM kt."end")'
        value_expr = f"SUM(kt.cargo_destination_cubic_meters * {MCM_PER_CUBIC_METER})"
        trailing_sql = """
            ),
            with_days AS (
                SELECT
                    continent,
                    country,
                    period,
                    total_mcm / EXTRACT(
                        DAY FROM (
                            DATE_TRUNC('month', MAKE_DATE(year::int, month::int, 1)) +
                            INTERVAL '1 month' - INTERVAL '1 day'
                        )
                    ) AS mcm_d
                FROM period_data
            )
            SELECT continent, country, period, mcm_d
            FROM with_days
        """
    else:
        period_expr = """
            'W' || EXTRACT(WEEK FROM kt."end")::text || '''' ||
            RIGHT(EXTRACT(YEAR FROM kt."end")::text, 2) AS period
        """
        order_expr = 'EXTRACT(YEAR FROM kt."end"), EXTRACT(WEEK FROM kt."end")'
        value_expr = f"SUM(kt.cargo_destination_cubic_meters * {MCM_PER_CUBIC_METER}) / 7.0"
        trailing_sql = ""

    if period_type == 'month':
        query_str = f"""
            WITH latest_timestamp AS (
                SELECT MAX(upload_timestamp_utc) AS max_ts
                FROM {DB_SCHEMA}.kpler_trades
            ),
            period_data AS (
                SELECT
                    {continent_col} AS continent,
                    {country_col} AS country,
                    {period_expr},
                    EXTRACT(YEAR FROM kt."end") AS year,
                    EXTRACT(MONTH FROM kt."end") AS month,
                    {value_expr} AS total_mcm
                FROM {DB_SCHEMA}.kpler_trades kt
                LEFT JOIN {DB_SCHEMA}.mapping_vessel_type_capacity mv
                    ON kt.vessel_capacity_cubic_meters >= mv.capacity_cubic_meters_min
                    AND kt.vessel_capacity_cubic_meters < mv.capacity_cubic_meters_max
                CROSS JOIN latest_timestamp
                WHERE {where_clause}
                    AND kt."end" >= CURRENT_DATE - INTERVAL '2 years'
                    AND kt."end" < CURRENT_DATE
                GROUP BY
                    {continent_col},
                    {country_col},
                    period,
                    EXTRACT(YEAR FROM kt."end"),
                    EXTRACT(MONTH FROM kt."end")
            {trailing_sql}
        """
    else:
        query_str = f"""
            WITH latest_timestamp AS (
                SELECT MAX(upload_timestamp_utc) AS max_ts
                FROM {DB_SCHEMA}.kpler_trades
            ),
            period_data AS (
                SELECT
                    {continent_col} AS continent,
                    {country_col} AS country,
                    {period_expr},
                    {value_expr} AS mcm_d
                FROM {DB_SCHEMA}.kpler_trades kt
                LEFT JOIN {DB_SCHEMA}.mapping_vessel_type_capacity mv
                    ON kt.vessel_capacity_cubic_meters >= mv.capacity_cubic_meters_min
                    AND kt.vessel_capacity_cubic_meters < mv.capacity_cubic_meters_max
                CROSS JOIN latest_timestamp
                WHERE {where_clause}
                    AND kt."end" >= CURRENT_DATE - INTERVAL '2 years'
                    AND kt."end" < CURRENT_DATE
                GROUP BY
                    {continent_col},
                    {country_col},
                    period,
                    {order_expr}
            )
            SELECT continent, country, period, mcm_d
            FROM period_data
        """

    query = text(query_str).bindparams(bindparam('destination_countries', expanding=True))
    df = pd.read_sql(query, conn, params=params)
    if df.empty:
        return pd.DataFrame()

    return df.pivot_table(
        index=['continent', 'country'],
        columns='period',
        values='mcm_d',
        aggfunc='sum',
        fill_value=0
    ).reset_index()


def fetch_origin_rolling_windows_hierarchical(conn, continent_col, country_col, where_clause, params,
                                              rolling_window_days=30):
    """Fetch 7D and configurable rolling windows for importer-side supplier hierarchies."""
    normalized_window_days = normalize_rolling_window_days(rolling_window_days)
    rolling_window_label = format_rolling_window_label(normalized_window_days)

    current_date = dt.datetime.now().date()
    params = {
        **params,
        'date_7d_ago': current_date - dt.timedelta(days=7),
        'date_window_ago': current_date - dt.timedelta(days=normalized_window_days),
        'date_window_y1_start': current_date - dt.timedelta(days=365 + normalized_window_days),
        'date_window_y1_end': current_date - dt.timedelta(days=365),
        'current_date': current_date,
    }

    query = text(f"""
        WITH latest_timestamp AS (
            SELECT MAX(upload_timestamp_utc) AS max_ts
            FROM {DB_SCHEMA}.kpler_trades
        ),
        all_origins AS (
            SELECT DISTINCT
                {continent_col} AS continent,
                {country_col} AS country
            FROM {DB_SCHEMA}.kpler_trades kt
            LEFT JOIN {DB_SCHEMA}.mapping_vessel_type_capacity mv
                ON kt.vessel_capacity_cubic_meters >= mv.capacity_cubic_meters_min
                AND kt.vessel_capacity_cubic_meters < mv.capacity_cubic_meters_max
            CROSS JOIN latest_timestamp
            WHERE {where_clause}
                AND kt."end" >= :date_window_y1_start
                AND kt."end" <= :current_date
        ),
        dates_7d AS (
            SELECT generate_series(
                CAST(:date_7d_ago AS date) + INTERVAL '1 day',
                CAST(:current_date AS date),
                '1 day'::interval
            )::date AS date
        ),
        dates_window AS (
            SELECT generate_series(
                CAST(:date_window_ago AS date) + INTERVAL '1 day',
                CAST(:current_date AS date),
                '1 day'::interval
            )::date AS date
        ),
        dates_window_y1 AS (
            SELECT generate_series(
                CAST(:date_window_y1_start AS date) + INTERVAL '1 day',
                CAST(:date_window_y1_end AS date),
                '1 day'::interval
            )::date AS date
        ),
        matrix_7d AS (
            SELECT d.date, o.continent, o.country
            FROM dates_7d d
            CROSS JOIN all_origins o
        ),
        matrix_window AS (
            SELECT d.date, o.continent, o.country
            FROM dates_window d
            CROSS JOIN all_origins o
        ),
        matrix_window_y1 AS (
            SELECT d.date, o.continent, o.country
            FROM dates_window_y1 d
            CROSS JOIN all_origins o
        ),
        daily_data AS (
            SELECT
                kt."end"::date AS date,
                {continent_col} AS continent,
                {country_col} AS country,
                SUM(kt.cargo_destination_cubic_meters * {MCM_PER_CUBIC_METER}) AS daily_mcmd
            FROM {DB_SCHEMA}.kpler_trades kt
            LEFT JOIN {DB_SCHEMA}.mapping_vessel_type_capacity mv
                ON kt.vessel_capacity_cubic_meters >= mv.capacity_cubic_meters_min
                AND kt.vessel_capacity_cubic_meters < mv.capacity_cubic_meters_max
            CROSS JOIN latest_timestamp
            WHERE {where_clause}
                AND kt."end" >= :date_window_y1_start
                AND kt."end" <= :current_date
            GROUP BY
                kt."end"::date,
                {continent_col},
                {country_col}
        ),
        window_7d AS (
            SELECT
                m.continent,
                m.country,
                AVG(COALESCE(d.daily_mcmd, 0)) AS avg_7d
            FROM matrix_7d m
            LEFT JOIN daily_data d
                ON m.date = d.date
                AND m.continent = d.continent
                AND m.country = d.country
            GROUP BY m.continent, m.country
        ),
        window_current AS (
            SELECT
                m.continent,
                m.country,
                AVG(COALESCE(d.daily_mcmd, 0)) AS avg_window
            FROM matrix_window m
            LEFT JOIN daily_data d
                ON m.date = d.date
                AND m.continent = d.continent
                AND m.country = d.country
            GROUP BY m.continent, m.country
        ),
        window_y1 AS (
            SELECT
                m.continent,
                m.country,
                AVG(COALESCE(d.daily_mcmd, 0)) AS avg_window_y1
            FROM matrix_window_y1 m
            LEFT JOIN daily_data d
                ON m.date = d.date
                AND m.continent = d.continent
                AND m.country = d.country
            GROUP BY m.continent, m.country
        )
        SELECT
            COALESCE(w7.continent, wc.continent, wy1.continent) AS continent,
            COALESCE(w7.country, wc.country, wy1.country) AS country,
            COALESCE(w7.avg_7d, 0) AS "7D",
            COALESCE(wc.avg_window, 0) AS "{rolling_window_label}",
            COALESCE(wy1.avg_window_y1, 0) AS "{rolling_window_label}_Y1",
            COALESCE(w7.avg_7d, 0) - COALESCE(wc.avg_window, 0) AS "Δ 7D-{rolling_window_label}",
            COALESCE(wc.avg_window, 0) - COALESCE(wy1.avg_window_y1, 0) AS "Δ {rolling_window_label} Y/Y"
        FROM window_7d w7
        FULL OUTER JOIN window_current wc
            ON w7.continent = wc.continent
            AND w7.country = wc.country
        FULL OUTER JOIN window_y1 wy1
            ON COALESCE(w7.continent, wc.continent) = wy1.continent
            AND COALESCE(w7.country, wc.country) = wy1.country
    """)
    query = query.bindparams(bindparam('destination_countries', expanding=True))
    return pd.read_sql(query, conn, params=params)


def combine_origin_summary_data_hierarchical(quarters_df, months_df, weeks_df, rolling_df, rolling_window_days=30):
    """Combine supplier-origin summary datasets into the table shown on the importer page."""
    try:
        rolling_window_label = format_rolling_window_label(rolling_window_days)
        all_combinations = set()

        for df in [quarters_df, months_df, weeks_df, rolling_df]:
            if not df.empty and 'continent' in df.columns and 'country' in df.columns:
                all_combinations.update(df[['continent', 'country']].apply(tuple, axis=1))

        if not all_combinations:
            return pd.DataFrame()

        result = pd.DataFrame(list(all_combinations), columns=['continent', 'country'])
        current_date = dt.datetime.now()
        current_quarter = (current_date.month - 1) // 3 + 1
        current_year = current_date.year
        current_week = current_date.isocalendar()[1]
        month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                       'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

        if not quarters_df.empty:
            quarter_cols = [col for col in quarters_df.columns if col not in ['continent', 'country']]
            completed_quarters = [
                col for col in quarter_cols
                if "Q" in col and "'" in col and (
                    int("20" + col.split("'")[1]) < current_year or
                    (
                        int("20" + col.split("'")[1]) == current_year and
                        int(col.split("Q")[1].split("'")[0]) < current_quarter
                    )
                )
            ]
            completed_quarters = sorted(
                completed_quarters,
                key=lambda x: (x.split("'")[1], x.split("Q")[1].split("'")[0])
            )[-5:]
            if completed_quarters:
                result = result.merge(
                    quarters_df[['continent', 'country'] + completed_quarters],
                    on=['continent', 'country'],
                    how='left'
                )

        if not months_df.empty:
            month_cols = [col for col in months_df.columns if col not in ['continent', 'country']]
            completed_months = [
                col for col in month_cols
                if "'" in col and (
                    int("20" + col.split("'")[1]) < current_year or
                    (
                        int("20" + col.split("'")[1]) == current_year and
                        month_order.get(col.split("'")[0], 0) < current_date.month
                    )
                )
            ]
            completed_months = sorted(
                completed_months,
                key=lambda x: (x.split("'")[1], month_order.get(x.split("'")[0], 0))
            )[-3:]
            if completed_months:
                result = result.merge(
                    months_df[['continent', 'country'] + completed_months],
                    on=['continent', 'country'],
                    how='left'
                )

        if not rolling_df.empty and rolling_window_label in rolling_df.columns:
            result = result.merge(
                rolling_df[['continent', 'country', rolling_window_label]],
                on=['continent', 'country'],
                how='left'
            )

        if not weeks_df.empty:
            week_cols = [col for col in weeks_df.columns if col not in ['continent', 'country']]
            completed_weeks = [
                col for col in week_cols
                if "W" in col and "'" in col and (
                    int("20" + col.split("'")[1]) < current_year or
                    (
                        int("20" + col.split("'")[1]) == current_year and
                        int(col.split("W")[1].split("'")[0]) < current_week
                    )
                )
            ]
            completed_weeks = sorted(
                completed_weeks,
                key=lambda x: (x.split("'")[1], x.split("W")[1].split("'")[0].zfill(2))
            )[-3:]
            if completed_weeks:
                result = result.merge(
                    weeks_df[['continent', 'country'] + completed_weeks],
                    on=['continent', 'country'],
                    how='left'
                )

        if not rolling_df.empty:
            for col in ['7D', f'Δ 7D-{rolling_window_label}', f'Δ {rolling_window_label} Y/Y']:
                if col in rolling_df.columns:
                    result = result.merge(
                        rolling_df[['continent', 'country', col]],
                        on=['continent', 'country'],
                        how='left'
                    )

        result = result.fillna(0)
        for col in [col for col in result.columns if col not in ['continent', 'country']]:
            result[col] = result[col].round(1)

        return result
    except Exception as e:
        return pd.DataFrame()


def prepare_origin_table_for_display(df, expanded_continents=None):
    """Prepare importer summary data for display with expandable continent rows."""
    if df.empty:
        return pd.DataFrame()

    expanded_continents = expanded_continents or []
    filtered_rows = []
    continent_totals_for_grand = []
    numeric_cols = [col for col in df.columns if col not in ['continent', 'country']]

    for continent in df['continent'].unique():
        continent_data = df[df['continent'] == continent]
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
            **{col: grand_total_df[col].sum() for col in numeric_cols}
        }]))

    return pd.concat(filtered_rows, ignore_index=True) if filtered_rows else pd.DataFrame()


def fetch_origin_summary_data(engine, destination_countries, status, vessel_type, rolling_window_days=30,
                              origin_level='origin_shipping_region',
                              selected_destination_aggregation='country',
                              selected_destination_value=None,
                              scoped_trades_df=None):
    """Fetch importer-side origin summary data with supplier continent/country hierarchy."""
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    if not normalized_destination_countries or status == 'non_laden':
        return pd.DataFrame()

    try:
        if scoped_trades_df is None:
            scoped_trades_df = _fetch_importer_scoped_trades(
                engine,
                normalized_destination_countries,
                vessel_type=vessel_type,
                delivered_only=True
            )

        filtered_df = _apply_importer_self_flow_exclusion(
            scoped_trades_df,
            selected_destination_aggregation,
            selected_destination_value
        )
        summary_scope_df = _prepare_importer_summary_scope_df(
            filtered_df,
            origin_level or 'origin_shipping_region'
        )
        if summary_scope_df.empty:
            return pd.DataFrame()

        quarters_df = _build_importer_periods_pivot(summary_scope_df, 'quarter')
        months_df = _build_importer_periods_pivot(summary_scope_df, 'month')
        weeks_df = _build_importer_periods_pivot(summary_scope_df, 'week')
        rolling_df = _build_importer_rolling_windows_pivot(
            summary_scope_df,
            rolling_window_days=rolling_window_days
        )
        return combine_origin_summary_data_hierarchical(
            quarters_df,
            months_df,
            weeks_df,
            rolling_df,
            rolling_window_days
        )
    except Exception as e:
        return pd.DataFrame()


def get_origin_forecast_period_config(current_date=None):
    """Return monthly and annual period definitions for the WoodMac origin forecast table."""
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
    if mapping_df is None or mapping_df.empty:
        return pd.DataFrame(columns=['alias', 'country_display', 'continent'])

    alias_frames = []
    for alias_col in ['country', 'country_name']:
        if alias_col not in mapping_df.columns:
            continue

        if alias_col == 'country':
            alias_df = mapping_df[[alias_col, 'country_name', 'continent']].copy()
            alias_df = alias_df.rename(columns={
                'country': 'alias',
                'country_name': 'country_display',
            })
        else:
            alias_df = mapping_df[[alias_col, 'continent']].copy()
            alias_df = alias_df.rename(columns={'country_name': 'alias'})
            alias_df['country_display'] = alias_df['alias']
        alias_df = alias_df[alias_df['alias'].notna()].copy()
        alias_df['alias'] = alias_df['alias'].astype(str).str.strip()
        alias_df = alias_df[alias_df['alias'] != '']
        alias_frames.append(alias_df)

    if not alias_frames:
        return pd.DataFrame(columns=['alias', 'country_display', 'continent'])

    alias_lookup = pd.concat(alias_frames, ignore_index=True)
    alias_lookup['country_display'] = alias_lookup['country_display'].replace('', np.nan)
    alias_lookup['country_display'] = alias_lookup['country_display'].fillna(alias_lookup['alias'])
    alias_lookup['continent'] = alias_lookup['continent'].replace('', np.nan).fillna('Unknown')
    alias_lookup = alias_lookup.drop_duplicates(subset=['alias'], keep='first')
    return alias_lookup[['alias', 'country_display', 'continent']]


def resolve_supply_allocation_destination_aliases(destination_countries, mapping_df):
    """Return destination aliases that can match demand-detail rows stored with display names."""
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    if not normalized_destination_countries:
        return []

    aliases = set(normalized_destination_countries)
    if mapping_df is not None and not mapping_df.empty and 'country' in mapping_df.columns:
        matching_rows = mapping_df[mapping_df['country'].isin(normalized_destination_countries)].copy()
        if 'country_name' in matching_rows.columns:
            aliases.update(
                value.strip()
                for value in matching_rows['country_name'].dropna().astype(str).tolist()
                if value.strip()
            )

    return tuple(sorted(aliases))


def fetch_latest_supply_allocation_run_metadata(engine):
    """Return the latest compatible monthly country-level base-view split-by-contract allocation run."""
    query = text(f"""
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
    """Build the subtitle shown above the SQL-backed WoodMac origin forecast table."""
    if not run_metadata:
        return "No compatible WoodMac supply-allocation SQL run is currently available."

    analysis_date = pd.to_datetime(run_metadata.get('analysis_date'), errors='coerce')
    forecast_start = pd.to_datetime(run_metadata.get('forecast_start'), errors='coerce')
    forecast_end = pd.to_datetime(run_metadata.get('forecast_end'), errors='coerce')
    parts = ["Modeled supplier allocation from SQL outputs"]

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


def build_origin_forecast_period_table(df, value_col, group_cols, current_date=None):
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


def build_origin_forecast_total_values(df, value_col, current_date=None):
    """Return period-value totals for one monthly BCM series."""
    period_config = get_origin_forecast_period_config(current_date)
    total_table = build_origin_forecast_period_table(
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


def prepare_origin_forecast_table_for_display(df, expanded_continents=None, footer_rows=None):
    """Prepare the SQL-backed WoodMac forecast table with expandable continents and footer totals."""
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


def fetch_origin_forecast_summary_data(engine, destination_countries, current_date=None, origin_level='origin_shipping_region'):
    """Fetch SQL-backed WoodMac supplier allocation data for the selected importer destinations."""
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    if not normalized_destination_countries:
        return pd.DataFrame(), [], None

    run_metadata = fetch_latest_supply_allocation_run_metadata(engine)
    if not run_metadata:
        return pd.DataFrame(), [], None

    period_config = get_origin_forecast_period_config(current_date)
    mappings_query = text(f"""
        SELECT DISTINCT
            country,
            country_name,
            continent
        FROM {DB_SCHEMA}.mappings_country
        WHERE country IS NOT NULL
    """)
    mapping_df = pd.read_sql(mappings_query, engine)
    destination_aliases = resolve_supply_allocation_destination_aliases(
        normalized_destination_countries,
        mapping_df
    )

    allocation_query = text(f"""
        SELECT
            date,
            origin AS origin_country,
            destination,
            COALESCE(new_total_allocated_bcm, total_allocated_bcm) AS allocated_volume_bcm
        FROM {SUPPLY_ALLOCATION_DEMAND_DETAIL_TABLE}
        WHERE run_id = :run_id
            AND destination IN :destination_aliases
            AND COALESCE(new_total_allocated_bcm, total_allocated_bcm) IS NOT NULL
            AND date >= :current_month_start
            AND date <= :horizon_end
    """)
    allocation_df = pd.read_sql(
        allocation_query,
        engine,
        params={
            'run_id': run_metadata['run_id'],
            'destination_aliases': destination_aliases,
            'current_month_start': period_config['current_month_start'].date(),
            'horizon_end': period_config['horizon_end'].date(),
        }
    )

    demand_query = text(f"""
        SELECT
            date,
            SUM(forecast_demand_bcm) AS forecast_demand_bcm
        FROM {SUPPLY_ALLOCATION_DEMAND_SUMMARY_TABLE}
        WHERE run_id = :run_id
            AND destination IN :destination_aliases
            AND date >= :current_month_start
            AND date <= :horizon_end
        GROUP BY date
        ORDER BY date
    """)
    demand_totals_df = pd.read_sql(
        demand_query,
        engine,
        params={
            'run_id': run_metadata['run_id'],
            'destination_aliases': destination_aliases,
            'current_month_start': period_config['current_month_start'].date(),
            'horizon_end': period_config['horizon_end'].date(),
        }
    )

    if allocation_df.empty and demand_totals_df.empty:
        return pd.DataFrame(), [], run_metadata

    alias_lookup = build_supply_allocation_country_alias_lookup(mapping_df)
    allocation_df['date'] = pd.to_datetime(allocation_df['date'], errors='coerce')
    allocation_df = allocation_df[allocation_df['date'].notna()].copy()
    allocation_df = allocation_df.groupby(['date', 'origin_country'], as_index=False)['allocated_volume_bcm'].sum()
    allocation_df = pd.merge(
        allocation_df,
        alias_lookup,
        how='left',
        left_on='origin_country',
        right_on='alias'
    )
    allocation_df['continent'] = allocation_df['continent'].replace('', np.nan).fillna('Unknown')
    allocation_df['country'] = allocation_df['country_display'].replace('', np.nan)
    allocation_df['country'] = allocation_df['country'].fillna(allocation_df['origin_country'])
    if origin_level == 'origin_country_name':
        allocation_df['continent'] = allocation_df['country']
    elif origin_level not in ('origin_shipping_region', 'continent_origin_name'):
        level_col_map = {
            'origin_basin':                 'basin',
            'origin_subcontinent':          'subcontinent',
            'origin_classification_level1': 'country_classification_level1',
            'origin_classification':        'country_classification',
        }
        mapping_col = level_col_map.get(origin_level)
        if mapping_col:
            ext_mapping = pd.read_sql(
                f"SELECT DISTINCT country_name AS country_display, {mapping_col} AS level_val FROM {DB_SCHEMA}.mappings_country WHERE country_name IS NOT NULL",
                engine
            )
            allocation_df = pd.merge(allocation_df, ext_mapping, on='country_display', how='left')
            allocation_df['continent'] = allocation_df['level_val'].fillna('Unknown')
            allocation_df = allocation_df.drop(columns=['level_val'])

    summary_df = build_origin_forecast_period_table(
        allocation_df[['date', 'continent', 'country', 'allocated_volume_bcm']],
        'allocated_volume_bcm',
        ['continent', 'country'],
        current_date=current_date
    )
    if not summary_df.empty:
        summary_df = summary_df.sort_values(['continent', 'country']).reset_index(drop=True)
        for col in period_config['ordered_labels']:
            summary_df[col] = summary_df[col].round(1)

    demand_totals_df['date'] = pd.to_datetime(demand_totals_df['date'], errors='coerce')
    demand_values = build_origin_forecast_total_values(
        demand_totals_df[['date', 'forecast_demand_bcm']],
        'forecast_demand_bcm',
        current_date=current_date
    )
    allocated_values = build_origin_forecast_total_values(
        allocation_df[['date', 'allocated_volume_bcm']],
        'allocated_volume_bcm',
        current_date=current_date
    )

    mismatch_values = {}
    for label in period_config['ordered_labels']:
        allocated_value = allocated_values.get(label)
        demand_value = demand_values.get(label)
        if allocated_value is None and demand_value is None:
            mismatch_values[label] = None
        else:
            mismatch_values[label] = round((allocated_value or 0) - (demand_value or 0), 1)

    footer_rows = [
        {'Continent': 'WOODMAC DEMAND TOTAL', 'Country': '', **demand_values},
        {'Continent': 'ALLOCATED SUPPLY TOTAL', 'Country': '', **allocated_values},
        {'Continent': 'MISMATCH (Allocated - Demand)', 'Country': '', **mismatch_values},
    ]

    return summary_df, footer_rows, run_metadata


def fetch_country_import_chart_data(destination_countries, rolling_window_days=30,
                                    selected_destination_aggregation='country',
                                    selected_destination_value=None,
                                    scoped_trades_df=None):
    """Fetch seasonal comparison data for total LNG imports into the selected destinations."""
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    if not normalized_destination_countries:
        return pd.DataFrame()

    if scoped_trades_df is None:
        scoped_trades_df = _fetch_importer_scoped_trades(engine, normalized_destination_countries)

    filtered_df = _apply_importer_self_flow_exclusion(
        scoped_trades_df,
        selected_destination_aggregation,
        selected_destination_value
    )
    return _build_importer_total_import_df(filtered_df, rolling_window_days=rolling_window_days)


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


def fetch_woodmac_country_import_forecast_data(destination_countries):
    """Fetch WoodMac monthly importer forecasts and expand them to daily flat mcm/d values."""
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    if not normalized_destination_countries:
        return pd.DataFrame(columns=['date', 'year', 'day_of_year', 'month_day', 'mcmd', 'is_forecast', 'source'])

    market_outlook_order_expr = """
        TO_DATE(
            (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{4})'))[1]
            || ' ' ||
            (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{4})'))[2],
            'Month YYYY'
        ) DESC,
        MAX(publication_date) DESC
    """
    query = text(f"""
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
                AND direction = 'Import'
                AND measured_at = 'Entry'
                AND metric_name = 'Flow'
                AND country_name IN :destination_countries
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
                AND direction = 'Import'
                AND measured_at = 'Entry'
                AND metric_name = 'Flow'
                AND country_name IN :destination_countries
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
    monthly_df = pd.read_sql(
        query,
        engine,
        params={'destination_countries': normalized_destination_countries}
    )
    forecast_df = expand_woodmac_monthly_forecast_to_daily(monthly_df)
    return filter_woodmac_forecast_horizon(forecast_df)


def fetch_continent_origin_chart_data(destination_countries, rolling_window_days=30, include_percentage=False,
                                      selected_destination_aggregation='country',
                                      selected_destination_value=None,
                                      scoped_trades_df=None):
    """Fetch seasonal comparison data by supplier continent for the selected importer destinations."""
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    if not normalized_destination_countries:
        return pd.DataFrame()

    if scoped_trades_df is None:
        scoped_trades_df = _fetch_importer_scoped_trades(engine, normalized_destination_countries)

    filtered_df = _apply_importer_self_flow_exclusion(
        scoped_trades_df,
        selected_destination_aggregation,
        selected_destination_value
    )
    return _build_importer_continent_chart_df(
        filtered_df,
        rolling_window_days=rolling_window_days,
        include_percentage=include_percentage
    )


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


def _create_seasonal_line_chart(df, series_col, value_col, yaxis_title, metric_label):
    if df.empty:
        return _empty_timeseries_chart("No data available")

    fig = go.Figure()
    years = sorted(df['year'].dropna().unique())

    if series_col is None:
        colors = ['#2E86C1', '#1B4F72', '#5DADE2', '#3498DB', '#76D7C4']
        for i, year in enumerate(years):
            year_data = df[df['year'] == year].copy()
            year_data['plot_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(
                year_data['day_of_year'] - 1,
                unit='d'
            )
            historical_data = year_data[~year_data['is_forecast']]
            forecast_data = year_data[year_data['is_forecast']]
            base_color = colors[i % len(colors)]

            if not historical_data.empty:
                fig.add_trace(go.Scatter(
                    x=historical_data['plot_date'],
                    y=historical_data[value_col],
                    mode='lines',
                    name=str(int(year)),
                    line=dict(color=base_color, width=3 if year == years[-1] else 2),
                    text=historical_data['month_day'],
                    hovertemplate=f'<b>{int(year)} (Historical)</b><br>%{{text}}<br>{metric_label}: %{{y:.1f}}<extra></extra>'
                ))

            if not forecast_data.empty:
                connect_data = pd.concat([historical_data.tail(1), forecast_data]) if not historical_data.empty else forecast_data
                forecast_color = (
                    f"rgba({int(base_color[1:3], 16)}, {int(base_color[3:5], 16)}, {int(base_color[5:7], 16)}, 0.5)"
                )
                fig.add_trace(go.Scatter(
                    x=connect_data['plot_date'],
                    y=connect_data[value_col],
                    mode='lines',
                    name=f"{int(year)} (Forecast)",
                    line=dict(color=forecast_color, width=3 if year == years[-1] else 2),
                    opacity=0.7,
                    text=connect_data['month_day'],
                    hovertemplate=f'<b>{int(year)} (Forecast)</b><br>%{{text}}<br>{metric_label}: %{{y:.1f}}<extra></extra>',
                    showlegend=False
                ))
    else:
        series_values = sorted(df[series_col].dropna().unique())
        color_map = {
            value: get_professional_colors(len(series_values))[idx]
            for idx, value in enumerate(series_values)
        }
        current_year = max(years)
        shown_legend = set()
        for series_value in series_values:
            series_df = df[df[series_col] == series_value]
            for year in years:
                year_data = series_df[series_df['year'] == year].copy()
                if year_data.empty:
                    continue
                year_data['plot_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(
                    year_data['day_of_year'] - 1,
                    unit='d'
                )
                historical_data = year_data[~year_data['is_forecast']]
                forecast_data = year_data[year_data['is_forecast']]
                color = color_map[series_value]
                line_width = 3 if year == current_year else 1.5
                show_legend = series_value not in shown_legend
                if show_legend:
                    shown_legend.add(series_value)

                if not historical_data.empty:
                    fig.add_trace(go.Scatter(
                        x=historical_data['plot_date'],
                        y=historical_data[value_col],
                        mode='lines',
                        name=series_value if show_legend else None,
                        legendgroup=str(series_value),
                        line=dict(color=color, width=line_width),
                        text=historical_data['month_day'],
                        hovertemplate=(
                            f'<b>{series_value} - {int(year)}</b><br>%{{text}}<br>{metric_label}: %{{y:.1f}}<extra></extra>'
                        ),
                        showlegend=show_legend
                    ))

                if not forecast_data.empty:
                    connect_data = pd.concat([historical_data.tail(1), forecast_data]) if not historical_data.empty else forecast_data
                    forecast_color = (
                        f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.4)"
                        if color.startswith('#') else color
                    )
                    fig.add_trace(go.Scatter(
                        x=connect_data['plot_date'],
                        y=connect_data[value_col],
                        mode='lines',
                        name=None,
                        legendgroup=str(series_value),
                        line=dict(color=forecast_color, width=line_width),
                        opacity=0.6,
                        text=connect_data['month_day'],
                        hovertemplate=(
                            f'<b>{series_value} - {int(year)} (Forecast)</b><br>%{{text}}<br>{metric_label}: %{{y:.1f}}<extra></extra>'
                        ),
                        showlegend=False
                    ))

    return _apply_time_series_chart_layout(fig, yaxis_title)


def _hex_to_rgba(color, alpha):
    if not isinstance(color, str) or not color.startswith('#') or len(color) != 7:
        return color
    return f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {alpha})"


def _create_total_import_chart_with_woodmac_forecast(historical_df, forecast_df):
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
                '<br>Imports: %{y:.1f} mcm/d<extra></extra>'
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
                    '<br>Imports: %{y:.1f} mcm/d<extra></extra>'
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
                '<br>Imports: %{y:.1f} mcm/d'
                '<br>Source: %{customdata}<extra></extra>'
            ),
            visible=True if year == default_visible_forecast_year else 'legendonly'
        ))

    return _apply_time_series_chart_layout(fig, 'mcm/d')


def create_country_import_chart(destination_label, destination_countries, rolling_window_days=30,
                                selected_destination_aggregation='country', selected_destination_value=None):
    historical_df = fetch_country_import_chart_data(
        destination_countries,
        rolling_window_days,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination_value
    )

    forecast_df = fetch_woodmac_country_import_forecast_data(destination_countries)
    if historical_df.empty and forecast_df.empty:
        return _empty_timeseries_chart(f"No import data available for {destination_label}")
    return _create_total_import_chart_with_woodmac_forecast(historical_df, forecast_df)


def create_continent_origin_chart(destination_label, destination_countries, rolling_window_days=30,
                                  selected_destination_aggregation='country', selected_destination_value=None):
    df = fetch_continent_origin_chart_data(
        destination_countries,
        rolling_window_days,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination_value
    )
    if df.empty:
        return _empty_timeseries_chart(f"No origin-continent import data available for {destination_label}")
    return _create_seasonal_line_chart(df, 'continent_origin', 'rolling_avg', 'mcm/d', 'Imports')


def create_continent_origin_percentage_chart(destination_label, destination_countries, rolling_window_days=30,
                                             selected_destination_aggregation='country',
                                             selected_destination_value=None):
    df = fetch_continent_origin_chart_data(
        destination_countries,
        rolling_window_days,
        include_percentage=True,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination_value
    )
    if df.empty:
        return _empty_timeseries_chart(f"No origin share data available for {destination_label}")
    return _create_seasonal_line_chart(df, 'continent_origin', 'percentage', '%', 'Share')


@callback(
    Output('imp-supply-analysis-title', 'children'),
    Input('imp-supply-rolling-window-input', 'value')
)
def update_supply_analysis_title(rolling_window_days):
    return f"LNG Import Analysis - {format_rolling_window_title(rolling_window_days)} + WoodMac Forecast"


@callback(
    [Output('imp-country-supply-chart', 'figure'),
     Output('imp-country-supply-header', 'children')],
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-supply-rolling-window-input', 'value'),
    Input('imp-destination-catalog-store', 'data')
)
def update_country_import_chart(selected_destination_aggregation, selected_destination, rolling_window_days,
                                destination_catalog):
    if not selected_destination:
        fig = go.Figure()
        fig.update_layout(height=400)
        return fig, "Total Imports + WoodMac Forecast"

    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    return (
        create_country_import_chart(
            destination_context['display_label'],
            destination_context['destination_countries'],
            rolling_window_days,
            selected_destination_aggregation=selected_destination_aggregation,
            selected_destination_value=selected_destination
        ),
        f"{destination_context['display_label']} - Total Imports + WoodMac Forecast"
    )


@callback(
    [Output('imp-continent-origin-chart', 'figure'),
     Output('imp-continent-origin-header', 'children')],
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-supply-rolling-window-input', 'value'),
    Input('imp-destination-catalog-store', 'data')
)
def update_continent_origin_chart(selected_destination_aggregation, selected_destination, rolling_window_days,
                                  destination_catalog):
    if not selected_destination:
        fig = go.Figure()
        fig.update_layout(height=400)
        return fig, "By Origin Continent (mcm/d)"

    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    return create_continent_origin_chart(
        destination_context['display_label'],
        destination_context['destination_countries'],
        rolling_window_days,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination
    ), (
        f"{destination_context['display_label']} - By Origin Continent"
    )


@callback(
    [Output('imp-continent-percentage-chart', 'figure'),
     Output('imp-continent-percentage-header', 'children')],
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-supply-rolling-window-input', 'value'),
    Input('imp-destination-catalog-store', 'data')
)
def update_continent_origin_percentage_chart(selected_destination_aggregation, selected_destination,
                                             rolling_window_days, destination_catalog):
    if not selected_destination:
        fig = go.Figure()
        fig.update_layout(height=400)
        return fig, "By Origin Continent (%)"

    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    return create_continent_origin_percentage_chart(
        destination_context['display_label'],
        destination_context['destination_countries'],
        rolling_window_days,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination
    ), (
        f"{destination_context['display_label']} - Origin Share (%)"
    )


@callback(
    Output('imp-download-importer-detail-supply-excel', 'data'),
    Input('imp-export-supply-analysis-button', 'n_clicks'),
    State('imp-destination-aggregation-dropdown', 'value'),
    State('imp-destination-country-dropdown', 'value'),
    State('imp-destination-catalog-store', 'data'),
    State('imp-supply-rolling-window-input', 'value'),
    State('imp-region-status-dropdown', 'value'),
    State('imp-vessel-type-dropdown', 'value'),
    State('imp-origin-level-dropdown', 'value'),
    prevent_initial_call=True
)
def export_import_analysis_to_excel(n_clicks, selected_destination_aggregation, selected_destination,
                                    destination_catalog, rolling_window_days, status, vessel_type, origin_level):
    if not n_clicks or not selected_destination:
        raise PreventUpdate

    normalized_window_days = normalize_rolling_window_days(rolling_window_days)
    rolling_window_label = format_rolling_window_label(normalized_window_days)
    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    if not destination_context['destination_countries']:
        raise PreventUpdate

    chart_scoped_trades_df = _fetch_importer_scoped_trades(
        engine,
        destination_context['destination_countries']
    )
    summary_scoped_trades_df = _fetch_importer_scoped_trades(
        engine,
        destination_context['destination_countries'],
        vessel_type=vessel_type,
        delivered_only=True
    )

    supply_df = fetch_country_import_chart_data(
        destination_context['destination_countries'],
        normalized_window_days,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination,
        scoped_trades_df=chart_scoped_trades_df
    )
    continent_df = fetch_continent_origin_chart_data(
        destination_context['destination_countries'],
        normalized_window_days,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination,
        scoped_trades_df=chart_scoped_trades_df
    )
    percentage_df = fetch_continent_origin_chart_data(
        destination_context['destination_countries'],
        normalized_window_days,
        include_percentage=True,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination,
        scoped_trades_df=chart_scoped_trades_df
    )
    summary_df = fetch_origin_summary_data(
        engine,
        destination_context['destination_countries'],
        status,
        vessel_type,
        normalized_window_days,
        origin_level=origin_level or 'origin_shipping_region',
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination,
        scoped_trades_df=summary_scoped_trades_df
    )

    if supply_df.empty and continent_df.empty and percentage_df.empty and summary_df.empty:
        raise PreventUpdate

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not supply_df.empty:
            supply_df.to_excel(writer, sheet_name='Total Imports', index=False)
        if not continent_df.empty:
            continent_df.to_excel(writer, sheet_name='Origin Continent mcmd', index=False)
        if not percentage_df.empty:
            percentage_df.to_excel(writer, sheet_name='Origin Continent Share', index=False)
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='Origin Summary', index=False)

        for worksheet in writer.sheets.values():
            for column_cells in worksheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter
                for cell in column_cells:
                    cell_value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(cell_value))
                worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)

    output.seek(0)
    safe_country = "".join(
        char if char.isalnum() else "_"
        for char in destination_context['display_label']
    ).strip("_") or "destination"
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{safe_country}_LNG_Import_Analysis_{rolling_window_label}_{timestamp}.xlsx"
    return dcc.send_bytes(output.getvalue(), filename)


@callback(
    Output('imp-trade-analysis-header', 'children'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-destination-catalog-store', 'data')
)
def update_trade_analysis_header(selected_destination_aggregation, selected_destination, destination_catalog):
    destination_label = format_destination_selection_label(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    return f'Trade Analysis: Supplier → {destination_label}'


@callback(
    Output('imp-destination-selection-store', 'data'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    prevent_initial_call=False
)
def sync_destination_selection(selected_destination_aggregation, selected_destination):
    return {
        'aggregation': selected_destination_aggregation,
        'value': selected_destination
    }


@callback(
    Output('imp-destination-catalog-store', 'data'),
    Output('imp-destination-country-dropdown', 'options'),
    Output('imp-destination-country-dropdown', 'value'),
    Input('global-refresh-button', 'n_clicks'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    State('imp-destination-catalog-store', 'data'),
    State('imp-destination-selection-store', 'data'),
    prevent_initial_call=False
)
def initialize_country_dropdown(n_clicks, selected_destination_aggregation, existing_catalog, selection_state):
    """Initialize the importer destination controls using the destination catalog."""
    try:
        try:
            triggered_id = ctx.triggered_id
        except Exception:
            triggered_id = None
        if triggered_id == 'imp-destination-aggregation-dropdown' and existing_catalog:
            catalog_records = existing_catalog
            catalog_output = no_update
        else:
            catalog_records = build_destination_catalog(engine)
            catalog_output = catalog_records

        destination_options = build_destination_value_options(
            selected_destination_aggregation,
            catalog_records
        )
        selected_destination_value = determine_destination_dropdown_value(
            selected_destination_aggregation,
            catalog_records,
            selection_state
        )

        return catalog_output, destination_options, selected_destination_value
    except Exception as e:
        fallback_catalog = []
        fallback_options = [{'label': 'China', 'value': 'China'}]
        return fallback_catalog, fallback_options, 'China'


def fetch_supplier_countries_for_importer(engine, destination_countries):
    """Return supplier countries serving the selected importer destinations in the latest Kpler snapshot."""
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    if not normalized_destination_countries:
        return []

    query = text(f"""
        WITH latest_timestamp AS (
            SELECT MAX(upload_timestamp_utc) AS max_ts
            FROM {DB_SCHEMA}.kpler_trades
        )
        SELECT DISTINCT origin_country_name
        FROM {DB_SCHEMA}.kpler_trades kt
        CROSS JOIN latest_timestamp
        WHERE kt.upload_timestamp_utc = latest_timestamp.max_ts
            AND kt.destination_country_name IN :destination_countries
            AND kt.origin_country_name IS NOT NULL
            AND kt.status = 'Delivered'
        ORDER BY origin_country_name
    """)
    countries_df = pd.read_sql(query, engine, params={'destination_countries': normalized_destination_countries})
    if countries_df.empty:
        return ()
    return tuple(countries_df['origin_country_name'].tolist())


def fetch_train_maintenance_data(engine, destination_countries=None):
    """
    Fetch and process maintenance data for supplier countries feeding the selected importer destinations.
    """
    try:
        normalized_destination_countries = normalize_destination_countries(destination_countries)
        supplier_countries = fetch_supplier_countries_for_importer(engine, normalized_destination_countries)
        if normalized_destination_countries and not supplier_countries:
            return pd.DataFrame()

        country_filter = ""
        params = {}
        query = text(f"""
            WITH combined_maintenance AS (
                SELECT
                    plant_name,
                    country_name,
                    lng_train_name_short,
                    year,
                    month,
                    year_actual_forecast,
                    SUM(metric_value) AS total_mtpa,
                    STRING_AGG(metric_comment, '; ') AS metric_comment
                FROM (
                    SELECT
                        plant_name,
                        country_name,
                        lng_train_name_short,
                        year,
                        month,
                        year_actual_forecast,
                        metric_value,
                        metric_comment
                    FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_unplanned_downtime_mta
                    WHERE metric_value > 0
                    UNION ALL
                    SELECT
                        plant_name,
                        country_name,
                        lng_train_name_short,
                        year,
                        month,
                        year_actual_forecast,
                        metric_value,
                        metric_comment
                    FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_planned_maintenance_mta
                    WHERE metric_value > 0
                ) maintenance_data
                WHERE 1=1
                {country_filter}
                GROUP BY
                    plant_name,
                    country_name,
                    lng_train_name_short,
                    year,
                    month,
                    year_actual_forecast
            )
            SELECT *
            FROM combined_maintenance
            ORDER BY country_name, plant_name, lng_train_name_short, year, month
        """)
        if supplier_countries:
            query = text(f"""
                WITH combined_maintenance AS (
                    SELECT
                        plant_name,
                        country_name,
                        lng_train_name_short,
                        year,
                        month,
                        year_actual_forecast,
                        SUM(metric_value) AS total_mtpa,
                        STRING_AGG(metric_comment, '; ') AS metric_comment
                    FROM (
                        SELECT
                            plant_name,
                            country_name,
                            lng_train_name_short,
                            year,
                            month,
                            year_actual_forecast,
                            metric_value,
                            metric_comment
                        FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_unplanned_downtime_mta
                        WHERE metric_value > 0
                        UNION ALL
                        SELECT
                            plant_name,
                            country_name,
                            lng_train_name_short,
                            year,
                            month,
                            year_actual_forecast,
                            metric_value,
                            metric_comment
                        FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_planned_maintenance_mta
                        WHERE metric_value > 0
                    ) maintenance_data
                    WHERE country_name IN :supplier_countries
                    GROUP BY
                        plant_name,
                        country_name,
                        lng_train_name_short,
                        year,
                        month,
                        year_actual_forecast
                )
                SELECT *
                FROM combined_maintenance
                ORDER BY country_name, plant_name, lng_train_name_short, year, month
            """)
            params = {'supplier_countries': supplier_countries}

        df = pd.read_sql(query, engine, params=params)
        if df.empty:
            return df

        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        return df
    except Exception as e:
        return pd.DataFrame()


def process_maintenance_periods_hierarchical(df, expanded_plants=None):
    """
    Process maintenance data into a country -> plant -> train hierarchy.
    Country rows are always shown; plant rows appear when the country is expanded;
    train rows appear when the plant is expanded.
    """
    if df.empty:
        return pd.DataFrame(), {}

    try:
        current_date = pd.Timestamp.now()
        current_quarter = current_date.quarter
        current_year = current_date.year
        mtpa_to_mcm_d = 1.372
        expanded_plants = expanded_plants or []
        period_cols = (
            [f'Q-{i}' for i in range(5, 0, -1)] +
            [f'M-{i}' for i in range(3, 0, -1)] +
            [f'M+{i}' for i in range(1, 4)] +
            [f'Q+{i}' for i in range(1, 5)]
        )

        last_month_end = pd.Timestamp(year=current_date.year, month=current_date.month, day=1) - pd.DateOffset(days=1)
        next_3m_start = pd.Timestamp(year=current_date.year, month=current_date.month, day=1)

        train_data = []
        plant_totals = {}
        country_totals = {}
        comments_data = {}

        for (country, plant, train), group_df in df.groupby(['country_name', 'plant_name', 'lng_train_name_short']):
            plant_key = f"{country}||{plant}"
            row = {
                'Supplier Country': '',
                'Plant': '',
                'Train': train,
                'Type': 'train',
                'PlantKey': plant_key,
            }

            if plant_key not in plant_totals:
                plant_totals[plant_key] = {
                    'country': country,
                    'plant': plant,
                    'totals': {col: 0 for col in period_cols}
                }
                comments_data[plant_key] = {}

            if country not in country_totals:
                country_totals[country] = {col: 0 for col in period_cols}

            comments_data[plant_key][train] = {}

            for q_offset in range(5, 0, -1):
                target_q = current_quarter - q_offset
                target_year = current_year
                while target_q <= 0:
                    target_q += 4
                    target_year -= 1
                q_start_month = (target_q - 1) * 3 + 1
                q_start = pd.Timestamp(year=target_year, month=q_start_month, day=1)
                q_end = q_start + pd.DateOffset(months=3) - pd.DateOffset(days=1)
                q_data = group_df[(group_df['date'] >= q_start) & (group_df['date'] <= q_end)]
                days_in_quarter = (q_end - q_start).days + 1
                total_mtpa = q_data['total_mtpa'].sum()
                avg_mcm_d = (total_mtpa * mtpa_to_mcm_d * 365) / days_in_quarter if days_in_quarter > 0 else 0
                label = f'Q-{q_offset}'
                value = round(avg_mcm_d, 1)
                row[label] = value if value > 0 else None
                plant_totals[plant_key]['totals'][label] += value
                if not q_data.empty and 'metric_comment' in q_data.columns:
                    comments = q_data['metric_comment'].dropna().unique()
                    if len(comments) > 0:
                        comments_data[plant_key][train][label] = '; '.join(comments)

            for m_offset in range(3, 0, -1):
                m_date = last_month_end - pd.DateOffset(months=m_offset - 1)
                m_start = pd.Timestamp(year=m_date.year, month=m_date.month, day=1)
                m_end = m_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                m_data = group_df[(group_df['date'] >= m_start) & (group_df['date'] <= m_end)]
                total_mtpa = m_data['total_mtpa'].sum()
                value = round((total_mtpa * mtpa_to_mcm_d * 365) / m_end.day if m_end.day > 0 else 0, 1)
                label = f'M-{m_offset}'
                row[label] = value if value > 0 else None
                plant_totals[plant_key]['totals'][label] += value
                if not m_data.empty and 'metric_comment' in m_data.columns:
                    comments = m_data['metric_comment'].dropna().unique()
                    if len(comments) > 0:
                        comments_data[plant_key][train][label] = '; '.join(comments)

            for m_offset in range(1, 4):
                m_date = next_3m_start + pd.DateOffset(months=m_offset - 1)
                m_start = pd.Timestamp(year=m_date.year, month=m_date.month, day=1)
                m_end = m_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                m_data = group_df[(group_df['date'] >= m_start) & (group_df['date'] <= m_end)]
                total_mtpa = m_data['total_mtpa'].sum()
                value = round((total_mtpa * mtpa_to_mcm_d * 365) / m_end.day if m_end.day > 0 else 0, 1)
                label = f'M+{m_offset}'
                row[label] = value if value > 0 else None
                plant_totals[plant_key]['totals'][label] += value
                if not m_data.empty and 'metric_comment' in m_data.columns:
                    comments = m_data['metric_comment'].dropna().unique()
                    if len(comments) > 0:
                        comments_data[plant_key][train][label] = '; '.join(comments)

            for q_offset in range(1, 5):
                target_q = current_quarter + q_offset
                target_year = current_year
                while target_q > 4:
                    target_q -= 4
                    target_year += 1
                q_start_month = (target_q - 1) * 3 + 1
                q_start = pd.Timestamp(year=target_year, month=q_start_month, day=1)
                q_end = q_start + pd.DateOffset(months=3) - pd.DateOffset(days=1)
                q_data = group_df[(group_df['date'] >= q_start) & (group_df['date'] <= q_end)]
                days_in_quarter = (q_end - q_start).days + 1
                total_mtpa = q_data['total_mtpa'].sum()
                avg_mcm_d = (total_mtpa * mtpa_to_mcm_d * 365) / days_in_quarter if days_in_quarter > 0 else 0
                label = f'Q+{q_offset}'
                value = round(avg_mcm_d, 1)
                row[label] = value if value > 0 else None
                plant_totals[plant_key]['totals'][label] += value
                if not q_data.empty and 'metric_comment' in q_data.columns:
                    comments = q_data['metric_comment'].dropna().unique()
                    if len(comments) > 0:
                        comments_data[plant_key][train][label] = '; '.join(comments)

            train_data.append(row)

        # Build country_totals by summing all plants per country
        for plant_key, plant_info in plant_totals.items():
            country = plant_info['country']
            for col in period_cols:
                country_totals[country][col] += plant_info['totals'][col]

        final_data = []
        grand_total = {col: 0 for col in period_cols}

        # Group plants by country
        plants_by_country = {}
        for plant_key in sorted(plant_totals.keys()):
            country = plant_totals[plant_key]['country']
            plants_by_country.setdefault(country, []).append(plant_key)

        for country in sorted(plants_by_country.keys()):
            country_expanded = country in expanded_plants
            arrow = '▼ ' if country_expanded else '▶ '
            country_row = {
                'Supplier Country': arrow + country,
                'Plant': '',
                'Train': '',
                'Type': 'country',
                'PlantKey': country,
            }
            for col in period_cols:
                value = round(country_totals[country][col], 1)
                country_row[col] = value if value > 0 else None
                grand_total[col] += country_totals[country][col]
            final_data.append(country_row)

            if country_expanded:
                for plant_key in plants_by_country[country]:
                    plant_info = plant_totals[plant_key]
                    plant_expanded = plant_key in expanded_plants
                    plant_arrow = '▼ ' if plant_expanded else '▶ '
                    plant_row = {
                        'Supplier Country': '',
                        'Plant': plant_arrow + plant_info['plant'],
                        'Train': 'Total',
                        'Type': 'plant',
                        'PlantKey': plant_key,
                    }
                    for col in period_cols:
                        value = round(plant_info['totals'][col], 1)
                        plant_row[col] = value if value > 0 else None
                    final_data.append(plant_row)

                    if plant_expanded:
                        for row in [r.copy() for r in train_data if r['PlantKey'] == plant_key]:
                            row['Supplier Country'] = ''
                            row['Plant'] = ''
                            final_data.append(row)

        grand_total_row = {
            'Supplier Country': '',
            'Plant': 'GRAND TOTAL',
            'Train': '',
            'Type': 'total',
            'PlantKey': 'GRAND_TOTAL',
        }
        for col in period_cols:
            value = round(grand_total[col], 1)
            grand_total_row[col] = value if value > 0 else None
        final_data.append(grand_total_row)

        return pd.DataFrame(final_data), comments_data
    except Exception as e:
        return pd.DataFrame(), {}


def create_maintenance_summary_table(df, comments_data=None):
    """Create an expandable supplier maintenance summary table — McKinsey board style."""
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
        all_period_ids = historical_col_ids + current_col_ids + nearterm_col_ids + outlook_col_ids

        columns = [
            {'name': 'Supplier Country', 'id': 'Supplier Country', 'type': 'text'},
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

        columns.extend([
            {'name': 'Type', 'id': 'Type', 'type': 'text'},
            {'name': 'PlantKey', 'id': 'PlantKey', 'type': 'text'},
        ])

        data = df.to_dict('records')

        # ── Row-level styles ──────────────────────────────────────────────────
        style_data_conditional = [
            # Country: strong navy left-border, off-white bg, dark bold text
            {'if': {'filter_query': '{Type} = "country"'},
             'backgroundColor': '#f0f4f8', 'fontWeight': '700',
             'color': '#1e3a5f', 'borderLeft': '4px solid #1e3a5f'},
            # Plant: very light slate, bold charcoal
            {'if': {'filter_query': '{Type} = "plant"'},
             'backgroundColor': '#f8fafc', 'fontWeight': '600', 'color': '#334155'},
            # Train: white, normal weight, slightly muted
            {'if': {'filter_query': '{Type} = "train"'},
             'backgroundColor': '#ffffff', 'fontWeight': '400',
             'color': '#475569', 'fontSize': '11px'},
            # Grand Total: near-black, white text
            {'if': {'filter_query': '{Plant} = "GRAND TOTAL"'},
             'backgroundColor': '#0f172a', 'color': 'white',
             'fontWeight': '700', 'fontSize': '13px'},
            # Text alignment
            {'if': {'column_id': 'Supplier Country'}, 'textAlign': 'left', 'cursor': 'pointer'},
            {'if': {'column_id': 'Plant'}, 'textAlign': 'left', 'cursor': 'pointer'},
            {'if': {'column_id': 'Train'}, 'textAlign': 'left'},
        ]

        # ── Historical cells: blue tint for non-zero ──────────────────────────
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
            # Small impact (>0, <1)
            style_data_conditional.append({
                'if': {'column_id': col_id, 'filter_query': f'{{{col_id}}} > 0'},
                'backgroundColor': 'rgba(251, 191, 36, 0.20)',
                'color': '#92400e',
            })
            # Medium impact (>=1)
            style_data_conditional.append({
                'if': {'column_id': col_id, 'filter_query': f'{{{col_id}}} >= 1'},
                'backgroundColor': 'rgba(245, 158, 11, 0.35)',
                'color': '#78350f', 'fontWeight': '600',
            })
            # Large impact (>=5) — material, alert red-amber
            style_data_conditional.append({
                'if': {'column_id': col_id, 'filter_query': f'{{{col_id}}} >= 5'},
                'backgroundColor': 'rgba(220, 38, 38, 0.15)',
                'color': '#991b1b', 'fontWeight': '700',
            })

        # ── Outlook: muted amber tiers ────────────────────────────────────────
        for col_id in outlook_col_ids:
            style_data_conditional.append({
                'if': {'column_id': col_id, 'filter_query': f'{{{col_id}}} > 0'},
                'backgroundColor': 'rgba(251, 191, 36, 0.12)',
                'color': '#92400e',
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
        # Boundary: between last historical month (M-1) and current month (M+1)
        style_data_conditional.append(
            {'if': {'column_id': 'M+1'}, 'borderLeft': '3px solid #94a3b8'}
        )
        # Boundary: between near-term and outlook (Q+2)
        style_data_conditional.append(
            {'if': {'column_id': 'Q+2'}, 'borderLeft': '2px solid #cbd5e1'}
        )
        # Boundary: first quarter col (Q-5) and first month col (M-3)
        style_data_conditional.append(
            {'if': {'column_id': 'M-3'}, 'borderLeft': '2px solid #e2e8f0'}
        )

        # ── Header conditional styles: zone colouring ────────────────────────
        header_styles = []
        for col_id in historical_col_ids:
            header_styles.append({'if': {'column_id': col_id}, 'backgroundColor': '#4b5563', 'color': '#f9fafb'})
        for col_id in current_col_ids:
            header_styles.append({'if': {'column_id': col_id}, 'backgroundColor': '#1d4ed8', 'color': 'white',
                                   'borderLeft': '3px solid #e2e8f0'})
        for col_id in nearterm_col_ids:
            header_styles.append({'if': {'column_id': col_id}, 'backgroundColor': '#92400e', 'color': 'white'})
        for col_id in outlook_col_ids:
            header_styles.append({'if': {'column_id': col_id}, 'backgroundColor': '#374151', 'color': '#f9fafb',
                                   'borderLeft': '2px solid #6b7280'})
        # Separator in header at M+1 and Q+2
        header_styles.append({'if': {'column_id': 'M+1'}, 'borderLeft': '3px solid #e2e8f0'})
        header_styles.append({'if': {'column_id': 'Q+2'}, 'borderLeft': '2px solid #6b7280'})
        header_styles.append({'if': {'column_id': 'M-3'}, 'borderLeft': '2px solid #6b7280'})

        # ── Tooltips ──────────────────────────────────────────────────────────
        tooltip_data = []
        if comments_data:
            for row in data:
                tooltip_row = {}
                if row.get('Type') == 'train':
                    plant_key = row.get('PlantKey')
                    train = row.get('Train', '')
                    train_comments = comments_data.get(plant_key, {}).get(train, {})
                    for col in columns:
                        if col['id'] not in ['Supplier Country', 'Plant', 'Train', 'Type', 'PlantKey']:
                            if col['id'] in train_comments and row.get(col['id']):
                                tooltip_row[col['id']] = {'value': train_comments[col['id']], 'type': 'text'}
                tooltip_data.append(tooltip_row)

        return dash_table.DataTable(
            id={'type': 'imp-maintenance-expandable-table', 'index': 0},
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
                'borderBottom': '2px solid #334155',
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
                {'if': {'column_id': 'Supplier Country'}, 'minWidth': '130px', 'maxWidth': '180px'},
                {'if': {'column_id': 'Plant'}, 'minWidth': '160px', 'maxWidth': '220px'},
                {'if': {'column_id': 'Train'}, 'minWidth': '100px', 'maxWidth': '140px'},
            ],
            style_data_conditional=style_data_conditional,
            hidden_columns=['Type', 'PlantKey'],
            sort_action='native',
            page_size=50,
            fill_width=False,
        )
    except Exception as e:
        return html.Div(f"Error creating table: {str(e)}", className="error-message")


@callback(
    Output('imp-region-data-store', 'data'),
    Output('imp-dropdown-options-store', 'data'),
    Output('imp-refresh-timestamp-store', 'data'),
    Output('imp-vessel-type-dropdown', 'options'),
    Output('imp-vessel-type-dropdown', 'value'),
    Input('global-refresh-button', 'n_clicks'),
    Input('imp-origin-level-dropdown', 'value'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-destination-catalog-store', 'data'),
    prevent_initial_call=False
)
def refresh_importer_data(n_clicks, selected_origin_level, selected_destination_aggregation,
                          selected_destination_value, destination_catalog):
    """Fetch base importer trade data and populate store-backed controls."""
    default_vessel_options = []
    default_vessel_value = None
    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination_value,
        destination_catalog
    )
    destination_label = destination_context['display_label']
    destination_countries = destination_context['destination_countries']

    if not destination_countries:
        imp_options_data = {
            'vessel_type_options': default_vessel_options,
            'default_vessel_type': default_vessel_value,
            'origin_level': selected_origin_level,
            'destination_aggregation': selected_destination_aggregation,
            'destination_value': selected_destination_value,
            'destination_label': destination_label,
            'destination_countries': destination_countries,
        }
        return (
            None,
            imp_options_data,
            "Waiting for destination catalog...",
            default_vessel_options,
            default_vessel_value,
        )

    try:
        df_trades_shipping_region = kpler_analysis(
            engine,
            origin_level=selected_origin_level,
            destination_countries=destination_countries
        )
        if df_trades_shipping_region is None or df_trades_shipping_region.empty:
            raise ValueError("kpler_analysis returned empty or None data.")
    except Exception as e:
        return None, {
            'vessel_type_options': default_vessel_options,
            'default_vessel_type': default_vessel_value,
        }, f"Error loading importer data: {e}", default_vessel_options, default_vessel_value

    imp_options_data = {
        'vessel_type_options': default_vessel_options,
        'default_vessel_type': default_vessel_value,
        'origin_level': selected_origin_level,
        'destination_aggregation': selected_destination_aggregation,
        'destination_value': selected_destination_value,
        'destination_label': destination_label,
        'destination_countries': destination_countries,
    }

    if not df_trades_shipping_region.empty and 'vessel_type' in df_trades_shipping_region.columns:
        available_vessel_types = set(df_trades_shipping_region['vessel_type'].dropna().unique())
        ordered_part = [v for v in DESIRED_VESSEL_ORDER if v in available_vessel_types]
        sorted_unexpected_part = sorted(list(available_vessel_types - set(DESIRED_VESSEL_ORDER)))
        final_vessel_order = ordered_part + sorted_unexpected_part
        default_vessel_options = [{'label': 'All', 'value': 'All'}] + [
            {'label': vessel_type, 'value': vessel_type}
            for vessel_type in final_vessel_order
        ]
        default_vessel_value = 'All'
        imp_options_data['vessel_type_options'] = default_vessel_options
        imp_options_data['default_vessel_type'] = default_vessel_value

    imp_shipping_data_json = df_trades_shipping_region.to_json(date_format='iso', orient='split')
    refresh_timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return (
        imp_shipping_data_json,
        imp_options_data,
        refresh_timestamp,
        default_vessel_options,
        default_vessel_value,
    )


@callback(
    Output('imp-trade-count-visualization', 'figure'),
    Input('imp-region-data-store', 'data'),
    Input('imp-dropdown-options-store', 'data'),
    Input('imp-aggregation-dropdown', 'value'),
    Input('imp-region-status-dropdown', 'value'),
    Input('imp-vessel-type-dropdown', 'value'),
    Input('imp-origin-level-dropdown', 'value'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-chart-metric-dropdown', 'value'),
    prevent_initial_call=True
)
def update_importer_region_visualizations(imp_shipping_data, imp_dropdown_options, selected_aggregation,
                                          selected_status, selected_vessel_type, selected_origin_level,
                                          selected_destination_aggregation, destination_value,
                                          selected_chart_metric):
    destination_label = (
        (imp_dropdown_options or {}).get('destination_label') or destination_value or 'Selected Destination'
    )
    no_data_msg = "No data available for the selected filters."
    empty_fig = go.Figure()
    empty_fig.update_layout(title=f"Loading {destination_label} Import Data...", height=600)

    if imp_shipping_data is None or imp_dropdown_options is None:
        error_msg = f"{destination_label} importer data not loaded."
        empty_fig.update_layout(title_text=error_msg)
        return empty_fig

    try:
        df_trades_shipping_region = pd.read_json(StringIO(imp_shipping_data), orient='split')
        if df_trades_shipping_region.empty:
            raise ValueError(f"Loaded {destination_label} importer data is empty.")
        if selected_origin_level not in df_trades_shipping_region.columns:
            raise ValueError(f"Selected origin level column '{selected_origin_level}' not found in data.")

        df_for_charts = df_trades_shipping_region[df_trades_shipping_region['status'] == selected_status].copy()
        if selected_vessel_type and selected_vessel_type != 'All':
            df_for_charts = df_for_charts[df_for_charts['vessel_type'] == selected_vessel_type]
        if df_for_charts.empty:
            empty_fig.update_layout(title_text="No data available after filtering")
            return empty_fig

        metric_mapping = {
            'count_trades': {'column': 'count_trades', 'title': 'Count of Trades', 'unit': 'trades'},
            'mtpa': {'column': 'sum_cargo_destination_cubic_meters', 'title': 'MTPA', 'unit': 'MTPA'},
            'mcm_d': {'column': 'sum_cargo_destination_cubic_meters', 'title': 'mcm/d', 'unit': 'mcm/d'},
            'm3': {'column': 'sum_cargo_destination_cubic_meters', 'title': 'm³', 'unit': 'm³'},
            'median_delivery_days': {'column': 'median_delivery_days', 'title': 'Median Delivery Days', 'unit': 'days'},
            'median_speed': {'column': 'median_speed', 'title': 'Median Speed', 'unit': 'knots'},
            'median_mileage_nautical_miles': {
                'column': 'median_mileage_nautical_miles',
                'title': 'Median Mileage',
                'unit': 'nautical miles'
            },
            'median_ton_miles': {'column': 'median_ton_miles', 'title': 'Median Ton Miles', 'unit': 'ton miles'},
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

        unique_origins = sorted(df_for_charts[selected_origin_level].dropna().unique())
        distinct_colors = get_professional_colors(len(unique_origins))
        color_mapping = {origin: distinct_colors[idx] for idx, origin in enumerate(unique_origins)}

        fig = go.Figure()
        all_groupby_cols = groupby_time_cols + [selected_origin_level]
        if 'median' in selected_chart_metric:
            selected_metric_data = df_for_charts.groupby(all_groupby_cols, observed=False)[metric_column].median().reset_index()
        else:
            selected_metric_data = df_for_charts.groupby(all_groupby_cols, observed=False)[metric_column].sum().reset_index()

        selected_metric_data = convert_trade_analysis_volume_metric(
            selected_metric_data,
            metric_column,
            selected_aggregation,
            selected_chart_metric
        )

        if 'year' in selected_metric_data.columns:
            if 'quarter' in selected_metric_data.columns and 'quarter' in groupby_time_cols:
                selected_metric_data['quarter_num'] = selected_metric_data['quarter'].str.extract(r'(\d+)').astype(int)
                selected_metric_data['time_label'] = (
                    selected_metric_data['year'].astype(str) + '-' + selected_metric_data['quarter'].astype(str)
                )
                selected_metric_data['sort_key'] = selected_metric_data['year'] * 10 + selected_metric_data['quarter_num']
            elif 'season' in selected_metric_data.columns and 'season' in groupby_time_cols:
                selected_metric_data['time_label'] = (
                    selected_metric_data['year'].astype(str) + '-' + selected_metric_data['season'].astype(str)
                )
                selected_metric_data['season_num'] = selected_metric_data['season'].apply(lambda x: 1 if x == 'S' else 2)
                selected_metric_data['sort_key'] = selected_metric_data['year'] * 10 + selected_metric_data['season_num']
            elif 'month' in selected_metric_data.columns and 'month' in groupby_time_cols:
                selected_metric_data['time_label'] = (
                    selected_metric_data['year'].astype(str) + '-' +
                    selected_metric_data['month'].astype(str).str.zfill(2)
                )
                selected_metric_data['sort_key'] = selected_metric_data['year'] * 100 + selected_metric_data['month']
            elif 'week' in selected_metric_data.columns and 'week' in groupby_time_cols:
                selected_metric_data['time_label'] = (
                    selected_metric_data['year'].astype(str) + '-W' +
                    selected_metric_data['week'].astype(str).str.zfill(2)
                )
                selected_metric_data['sort_key'] = selected_metric_data['year'] * 100 + selected_metric_data['week']
            else:
                selected_metric_data['time_label'] = selected_metric_data['year'].astype(str)
                selected_metric_data['sort_key'] = selected_metric_data['year']
        else:
            selected_metric_data['time_label'] = 'Unknown'
            selected_metric_data['sort_key'] = 0

        selected_metric_data = selected_metric_data.sort_values('sort_key', ascending=True)
        sorted_time_labels = selected_metric_data['time_label'].unique()

        for origin in unique_origins:
            origin_data = selected_metric_data[selected_metric_data[selected_origin_level] == origin]
            if not origin_data.empty:
                fig.add_trace(go.Bar(
                    x=origin_data['time_label'],
                    y=origin_data[metric_column],
                    name=origin,
                    marker_color=color_mapping[origin],
                    legendgroup=origin,
                ))

        fig.update_xaxes(categoryorder='array', categoryarray=sorted_time_labels)
        origin_display = "Countries" if selected_origin_level == "origin_country_name" else "Shipping Regions"
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
        fig.update_xaxes(
            title_text=x_axis_title,
            title_font=dict(family='Inter, -apple-system, BlinkMacSystemFont, sans-serif', size=13,
                            color=PROFESSIONAL_COLORS['text_primary']),
            tickfont=dict(family='Inter, -apple-system, BlinkMacSystemFont, sans-serif', size=11,
                          color=PROFESSIONAL_COLORS['text_secondary']),
            gridcolor=PROFESSIONAL_COLORS['grid_color'],
            gridwidth=0.5,
            linecolor=PROFESSIONAL_COLORS['grid_color'],
            linewidth=1,
            showgrid=True,
            zeroline=False
        )
        fig.update_yaxes(
            title_text=metric_unit,
            title_font=dict(family='Inter, -apple-system, BlinkMacSystemFont, sans-serif', size=13,
                            color=PROFESSIONAL_COLORS['text_primary']),
            tickfont=dict(family='Inter, -apple-system, BlinkMacSystemFont, sans-serif', size=11,
                          color=PROFESSIONAL_COLORS['text_secondary']),
            gridcolor=PROFESSIONAL_COLORS['grid_color'],
            gridwidth=0.5,
            linecolor=PROFESSIONAL_COLORS['grid_color'],
            linewidth=1,
            showgrid=True,
            zeroline=False
        )

        table_filters = {'status': selected_status}
        if selected_vessel_type and selected_vessel_type != 'All':
            table_filters['vessel_type'] = selected_vessel_type

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
        table_source_df = convert_trade_analysis_volume_metric(
            df_trades_shipping_region,
            selected_metric_column,
            selected_aggregation,
            selected_chart_metric
        )
        table_data = prepare_pivot_table(
            df=table_source_df,
            values_col=selected_metric_column,
            filters=table_filters,
            aggregation_level=selected_aggregation,
            add_total_column=True,
            aggfunc=agg_func,
            origin_level=selected_origin_level
        )

        return fig
    except Exception as e:
        error_message = f"Error updating importer visuals/tables: {e}"
        empty_fig.update_layout(title_text=error_message)
        return empty_fig


@callback(
    Output('imp-download-trade-analysis-excel', 'data'),
    Input('imp-export-trade-analysis-button', 'n_clicks'),
    State('imp-region-data-store', 'data'),
    State('imp-dropdown-options-store', 'data'),
    State('imp-aggregation-dropdown', 'value'),
    State('imp-region-status-dropdown', 'value'),
    State('imp-vessel-type-dropdown', 'value'),
    State('imp-origin-level-dropdown', 'value'),
    State('imp-destination-aggregation-dropdown', 'value'),
    State('imp-destination-country-dropdown', 'value'),
    State('imp-chart-metric-dropdown', 'value'),
    prevent_initial_call=True
)
def export_importer_trade_analysis_to_excel(n_clicks, imp_shipping_data, imp_dropdown_options,
                                            selected_aggregation, selected_status, selected_vessel_type,
                                            selected_origin_level, selected_destination_aggregation,
                                            destination_value, selected_chart_metric):
    if not n_clicks:
        raise PreventUpdate
    if imp_shipping_data is None or not all([selected_aggregation, selected_status, selected_origin_level]):
        raise PreventUpdate

    try:
        df = pd.read_json(StringIO(imp_shipping_data), orient='split')
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

        table_filters = {'status': selected_status}
        if selected_vessel_type and selected_vessel_type != 'All':
            table_filters['vessel_type'] = selected_vessel_type

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
            origin_level=selected_origin_level
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
        dest = destination_value or "destination"
        safe_dest = "".join(c if c.isalnum() else "_" for c in dest).strip("_") or "destination"
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_dest}_Trade_Analysis_{selected_aggregation}_{timestamp}.xlsx"
        return dcc.send_bytes(output.getvalue(), filename)

    except PreventUpdate:
        raise
    except Exception as e:
        raise PreventUpdate


def create_origin_forecast_summary_table(display_df):
    """Create the SQL-backed WoodMac origin forecast summary table."""
    footer_row_labels = [
        'WOODMAC DEMAND TOTAL',
        'ALLOCATED SUPPLY TOTAL',
        'MISMATCH (Allocated - Demand)',
    ]
    col_display_names = {'Continent': 'Origin Level', 'Country': 'Country'}
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
        'WOODMAC DEMAND TOTAL': {'backgroundColor': '#fff3e0', 'fontWeight': 'bold', 'color': '#8a4b08'},
        'ALLOCATED SUPPLY TOTAL': {'backgroundColor': '#e8f4fd', 'fontWeight': 'bold', 'color': '#1B4F72'},
        'MISMATCH (Allocated - Demand)': {'backgroundColor': '#f3f4f6', 'fontWeight': 'bold', 'color': '#374151'},
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
        id={'type': 'imp-origin-forecast-expandable-table', 'index': 'summary'},
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
    Output('imp-origin-forecast-summary-subtitle', 'children'),
    Output('imp-origin-forecast-summary-table-container', 'children'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-region-status-dropdown', 'value'),
    Input('imp-origin-forecast-expanded-continents', 'data'),
    Input('imp-destination-catalog-store', 'data'),
    Input('imp-origin-level-dropdown', 'value'),
    prevent_initial_call=False
)
def update_origin_forecast_summary_table(selected_destination_aggregation, selected_destination, status,
                                         expanded_continents, destination_catalog, origin_level):
    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    if not destination_context['destination_countries']:
        return (
            "Modeled supplier allocation from SQL outputs.",
            html.Div("Please select a destination.", style={'textAlign': 'center', 'padding': '20px'})
        )
    if status == 'non_laden':
        return (
            "Modeled supplier allocation from SQL outputs.",
            html.Div(
                "WoodMac origin forecast allocation is not shown for non-laden selections.",
                style={'textAlign': 'center', 'padding': '20px'}
            )
        )

    try:
        expanded_continents = expanded_continents or []
        summary_df, footer_rows, run_metadata = fetch_origin_forecast_summary_data(
            engine,
            destination_context['destination_countries'],
            origin_level=origin_level or 'origin_shipping_region'
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

        display_df = prepare_origin_forecast_table_for_display(
            summary_df,
            expanded_continents=expanded_continents,
            footer_rows=footer_rows
        )
        if display_df.empty:
            return (
                subtitle,
                html.Div(
                    f"No WoodMac origin forecast allocation data is available for {destination_context['display_label']}.",
                    style={'textAlign': 'center', 'padding': '20px'}
                )
            )

        return subtitle, create_origin_forecast_summary_table(display_df)
    except Exception as e:
        return (
            "Modeled supplier allocation from SQL outputs.",
            html.Div(
                f"Error loading data: {str(e)}",
                style={'textAlign': 'center', 'padding': '20px', 'color': 'red'}
            )
        )


@callback(
    Output('imp-origin-summary-table-container', 'children'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-supply-rolling-window-input', 'value'),
    Input('imp-region-status-dropdown', 'value'),
    Input('imp-vessel-type-dropdown', 'value'),
    Input('imp-origin-expanded-continents', 'data'),
    Input('imp-destination-catalog-store', 'data'),
    Input('imp-origin-level-dropdown', 'value'),
    prevent_initial_call=False
)
def update_origin_summary_table(selected_destination_aggregation, selected_destination, rolling_window_days, status,
                                vessel_type, expanded_continents, destination_catalog, origin_level):
    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    if not destination_context['destination_countries']:
        return html.Div("Please select a destination.", style={'textAlign': 'center', 'padding': '20px'})
    if status == 'non_laden':
        return html.Div(
            "Origin summary is not available for non-laden trades because the raw SQL source has no discharge volume.",
            style={'textAlign': 'center', 'padding': '20px'}
        )

    try:
        expanded_continents = expanded_continents or []
        df = fetch_origin_summary_data(
            engine,
            destination_context['destination_countries'],
            status,
            vessel_type,
            rolling_window_days,
            origin_level=origin_level or 'origin_shipping_region',
            selected_destination_aggregation=selected_destination_aggregation,
            selected_destination_value=selected_destination
        )
        if df.empty:
            return html.Div("No data available for the selected filters.", style={'textAlign': 'center', 'padding': '20px'})

        display_df = prepare_origin_table_for_display(df, expanded_continents)
        col_display_names = {'Continent': 'Origin Level', 'Country': 'Country'}
        columns = []
        for col in display_df.columns:
            if col in ['Continent', 'Country']:
                columns.append({'name': col_display_names.get(col, col), 'id': col, 'type': 'text'})
            else:
                columns.append({
                    'name': col,
                    'id': col,
                    'type': 'numeric',
                    'format': {'specifier': '.0f'},
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

        quarter_columns = [col for col in display_df.columns if col.startswith('Q') and "'" in col]
        month_columns = [col for col in display_df.columns if "'" in col and not col.startswith('Q')
                         and not col.startswith('W') and col not in ['Continent', 'Country']]
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
                conditional_styles.extend([
                    {'if': {'column_id': col}, 'backgroundColor': '#f5f5f5', 'fontWeight': '600', 'borderLeft': '3px solid white'},
                    {'if': {'column_id': col, 'filter_query': f'{{{col}}} > 0'}, 'color': '#2e7d32'},
                    {'if': {'column_id': col, 'filter_query': f'{{{col}}} < 0'}, 'color': '#c62828'},
                ])
            elif col == delta_yoy_column:
                conditional_styles.extend([
                    {'if': {'column_id': col}, 'backgroundColor': '#e8f5e9', 'fontWeight': '600', 'borderLeft': '3px solid white'},
                    {'if': {'column_id': col, 'filter_query': f'{{{col}}} > 0'}, 'color': '#1b5e20'},
                    {'if': {'column_id': col, 'filter_query': f'{{{col}}} < 0'}, 'color': '#b71c1c'},
                ])

        conditional_styles.append({
            'if': {'filter_query': '{Continent} = "GRAND TOTAL"'},
            'backgroundColor': '#2E86C1',
            'color': 'white',
            'fontWeight': 'bold'
        })

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

        return dash_table.DataTable(
            id={'type': 'imp-origin-expandable-table', 'index': 'summary'},
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
    except Exception as e:
        return html.Div(f"Error loading data: {str(e)}", style={'textAlign': 'center', 'padding': '20px', 'color': 'red'})


@callback(
    Output('imp-graph-route-suez-only', 'figure'),
    [Input('imp-aggregation-dropdown', 'value'),
     Input('imp-origin-level-dropdown', 'value'),
     Input('imp-destination-aggregation-dropdown', 'value'),
     Input('imp-destination-country-dropdown', 'value'),
     Input('imp-destination-catalog-store', 'data')]
)
def update_route_analysis_charts_and_tables(agg_level, origin_level, selected_destination_aggregation,
                                            selected_destination, destination_catalog):
    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    try:
        processed_df = process_trade_and_distance_data(
            engine,
            destination_countries=destination_context['destination_countries']
        )
        if processed_df is None or processed_df.empty:
            raise ValueError("No route data available.")

        if origin_level not in processed_df.columns and origin_level == 'origin_shipping_region':
            region_map_df = pd.read_sql(
                text(f"SELECT DISTINCT country, shipping_region FROM {DB_SCHEMA}.mappings_country"),
                engine
            ).rename(columns={'country': 'origin_country_name', 'shipping_region': 'origin_shipping_region'})
            processed_df = pd.merge(processed_df, region_map_df, how='left', on='origin_country_name')
        elif origin_level not in processed_df.columns:
            level_col_map = {
                'continent_origin_name':        'continent',
                'origin_basin':                 'basin',
                'origin_subcontinent':          'subcontinent',
                'origin_classification_level1': 'country_classification_level1',
                'origin_classification':        'country_classification',
            }
            mapping_col = level_col_map.get(origin_level)
            if mapping_col and 'origin_country_name' in processed_df.columns:
                try:
                    mapping_df = pd.read_sql(
                        f"SELECT DISTINCT country_name AS origin_country_name, {mapping_col} AS \"{origin_level}\" "
                        f"FROM {DB_SCHEMA}.mappings_country WHERE country_name IS NOT NULL",
                        engine
                    )
                    processed_df = pd.merge(processed_df, mapping_df, on='origin_country_name', how='left')
                except Exception:
                    pass
    except Exception as e:
        error_fig = go.Figure().update_layout(title="Error Processing Data", xaxis={'visible': False}, yaxis={'visible': False})
        return error_fig

    if origin_level not in processed_df.columns:
        error_msg = f"Selected origin level column '{origin_level}' not found in processed data."
        error_fig = go.Figure().update_layout(title=error_msg, xaxis={'visible': False}, yaxis={'visible': False})
        return error_fig

    for col in ['origin_country_name', 'origin_shipping_region']:
        if col not in processed_df.columns:
            processed_df[col] = None

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
    if 'selected_route' not in df_direct_only.columns:
        df_direct_only['selected_route'] = 'Direct'

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Route Usage: Suez Available (Not Panama)",
            "Route Usage: Panama Available (Not Suez)",
            "Route Usage: Both Suez & Panama Available"
        ],
        horizontal_spacing=0.12,
        specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]]
    )

    route_colors = {
        'Direct': PROFESSIONAL_COLORS['primary'],
        'ViaSuez': PROFESSIONAL_COLORS['warning'],
        'ViaPanama': PROFESSIONAL_COLORS['success']
    }
    shown_in_legend = set()

    def add_stacked_bar_line_to_subplot(df, row, col):
        if df is None or df.empty:
            return False

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

        required_cols = index_cols + ['selected_route', 'voyage_id']
        if any(col_name not in df.columns for col_name in required_cols):
            return False

        grouped = df.groupby(index_cols + ['selected_route'], observed=True)['voyage_id'].count().unstack(fill_value=0)
        if grouped.empty:
            return False

        total_trades = grouped.sum(axis=1)
        percentage_df = grouped.divide(total_trades, axis=0).fillna(0) * 100

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

        sort_keys = []
        for label in x_labels:
            parts = str(label).split('-')
            try:
                year = int(parts[0])
                if len(parts) > 1 and parts[1].startswith('Q'):
                    sort_key = year * 10 + int(parts[1][1:])
                elif len(parts) > 1 and parts[1] in ['S', 'W']:
                    sort_key = year * 10 + (1 if parts[1] == 'S' else 2)
                else:
                    sort_key = year * 100
            except ValueError:
                sort_key = 0
            sort_keys.append(sort_key)
        sorted_indices = sorted(range(len(sort_keys)), key=lambda idx: sort_keys[idx])
        sorted_x_labels = [x_labels[idx] for idx in sorted_indices]

        for route in percentage_df.columns:
            color = route_colors.get(route, '#808080')
            y_values = [percentage_df.iloc[idx][route] for idx in sorted_indices]
            show_legend = f"route_{route}" not in shown_in_legend
            shown_in_legend.add(f"route_{route}")
            fig.add_trace(go.Bar(
                x=sorted_x_labels,
                y=y_values,
                name=route,
                marker_color=color,
                legendgroup=route,
                showlegend=show_legend
            ), row=row, col=col, secondary_y=False)

        total_y = [total_trades.iloc[idx] for idx in sorted_indices]
        show_total = "total" not in shown_in_legend
        shown_in_legend.add("total")
        fig.add_trace(go.Scatter(
            x=sorted_x_labels,
            y=total_y,
            name="Total Trades Count",
            mode='lines+markers',
            line=dict(color='black', width=3),
            marker=dict(color='black', size=8),
            legendgroup="Total",
            showlegend=show_total
        ), row=row, col=col, secondary_y=True)

        fig.update_xaxes(title_text=agg_level.replace('+', '-'), row=row, col=col,
                         tickangle=45 if len(sorted_x_labels) > 3 else 0)
        return True

    has_data_1 = add_stacked_bar_line_to_subplot(df_suez_only, 1, 1)
    has_data_2 = add_stacked_bar_line_to_subplot(df_panama_only, 1, 2)
    has_data_3 = add_stacked_bar_line_to_subplot(df_both, 1, 3)

    for col in range(1, 4):
        if (col == 1 and has_data_1) or (col == 2 and has_data_2) or (col == 3 and has_data_3):
            fig.update_yaxes(title_text="Percentage of Trades (%)", range=[0, 100], ticksuffix="%", row=1, col=col,
                             secondary_y=False)
            fig.update_yaxes(title_text="Total Trade Count", showgrid=False, row=1, col=col, secondary_y=True)

    fig.update_layout(
        barmode='stack',
        height=500,
        legend=dict(orientation="v", x=1.05, y=0.5, xanchor='left', yanchor='middle', title_text='Route'),
        margin=dict(l=60, r=150, t=70, b=120),
        hovermode="x unified"
    )

    return fig


@callback(
    Output('imp-download-route-analysis-excel', 'data'),
    Input('imp-export-route-analysis-button', 'n_clicks'),
    State('imp-aggregation-dropdown', 'value'),
    State('imp-origin-level-dropdown', 'value'),
    State('imp-destination-aggregation-dropdown', 'value'),
    State('imp-destination-country-dropdown', 'value'),
    State('imp-destination-catalog-store', 'data'),
    prevent_initial_call=True
)
def export_importer_route_analysis_to_excel(n_clicks, agg_level, origin_level,
                                            selected_destination_aggregation, selected_destination,
                                            destination_catalog):
    if not n_clicks:
        raise PreventUpdate

    try:
        destination_context = resolve_destination_context(
            selected_destination_aggregation,
            selected_destination,
            destination_catalog
        )
        processed_df = process_trade_and_distance_data(
            engine,
            destination_countries=destination_context['destination_countries']
        )
        if processed_df is None or processed_df.empty:
            raise PreventUpdate

        if origin_level not in processed_df.columns and origin_level == 'origin_shipping_region':
            region_map_df = pd.read_sql(
                text(f"SELECT DISTINCT country, shipping_region FROM {DB_SCHEMA}.mappings_country"),
                engine
            ).rename(columns={'country': 'origin_country_name', 'shipping_region': 'origin_shipping_region'})
            processed_df = pd.merge(processed_df, region_map_df, how='left', on='origin_country_name')
        elif origin_level not in processed_df.columns:
            level_col_map = {
                'continent_origin_name':        'continent',
                'origin_basin':                 'basin',
                'origin_subcontinent':          'subcontinent',
                'origin_classification_level1': 'country_classification_level1',
                'origin_classification':        'country_classification',
            }
            mapping_col = level_col_map.get(origin_level)
            if mapping_col and 'origin_country_name' in processed_df.columns:
                try:
                    mapping_df = pd.read_sql(
                        f"SELECT DISTINCT country_name AS origin_country_name, {mapping_col} AS \"{origin_level}\" "
                        f"FROM {DB_SCHEMA}.mappings_country WHERE country_name IS NOT NULL",
                        engine
                    )
                    processed_df = pd.merge(processed_df, mapping_df, on='origin_country_name', how='left')
                except Exception:
                    pass
    except PreventUpdate:
        raise
    except Exception as e:
        raise PreventUpdate

    for col in ['origin_country_name', 'origin_shipping_region']:
        if col not in processed_df.columns:
            processed_df[col] = None

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
        'Suez Only': build_route_pivot(df_suez_only, origin_level),
        'Panama Only': build_route_pivot(df_panama_only, origin_level),
        'Both Routes': build_route_pivot(df_both, origin_level),
        'Direct Only': build_route_pivot(df_direct_only, origin_level),
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
    dest = selected_destination or "destination"
    safe_dest = "".join(c if c.isalnum() else "_" for c in dest).strip("_") or "destination"
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{safe_dest}_Route_Analysis_{agg_level}_{timestamp}.xlsx"
    return dcc.send_bytes(output.getvalue(), filename)


def build_importer_diversion_chart_dataframe(df_kpler_diversions, df_mapping_country, df_mapping_location):
    """Build diversion combo/grouping data for charts and pivot tables."""
    needed_columns = [
        'Diversion_month',
        'basin_combo',
        'region_combo',
        'country_combo',
        'Added shipping days',
        'Cubic Meters'
    ]
    if df_kpler_diversions.empty:
        return pd.DataFrame(columns=needed_columns)

    df_kpler_charts = df_kpler_diversions.copy()
    df_kpler_charts = pd.merge(
        df_kpler_charts,
        df_mapping_country[['country', 'basin', 'shipping_region']].rename(
            columns={
                'country': 'Diverted from country',
                'basin': 'Diverted from basin 1',
                'shipping_region': 'Diverted from shipping region 1'
            }
        ),
        on='Diverted from country',
        how='left'
    )
    df_kpler_charts = pd.merge(
        df_kpler_charts,
        df_mapping_location[['destination_location_name', 'basin', 'shipping_region']].rename(
            columns={
                'destination_location_name': 'Diverted from location',
                'basin': 'Diverted from basin 2',
                'shipping_region': 'Diverted from shipping region 2'
            }
        ),
        on='Diverted from location',
        how='left'
    )
    df_kpler_charts = pd.merge(
        df_kpler_charts,
        df_mapping_country[['country', 'basin', 'shipping_region']].rename(
            columns={
                'country': 'New destination country',
                'basin': 'New destination basin 1',
                'shipping_region': 'New destination shipping region 1'
            }
        ),
        on='New destination country',
        how='left'
    )
    df_kpler_charts = pd.merge(
        df_kpler_charts,
        df_mapping_location[['destination_location_name', 'basin', 'shipping_region']].rename(
            columns={
                'destination_location_name': 'New destination location',
                'basin': 'New destination basin 2',
                'shipping_region': 'New destination shipping region 2'
            }
        ),
        on='New destination location',
        how='left'
    )

    df_kpler_charts['Diverted from basin'] = np.where(
        df_kpler_charts['Diverted from basin 1'].isnull(),
        df_kpler_charts['Diverted from basin 2'],
        df_kpler_charts['Diverted from basin 1']
    )
    df_kpler_charts['Diverted from shipping region'] = np.where(
        df_kpler_charts['Diverted from shipping region 1'].isnull(),
        df_kpler_charts['Diverted from shipping region 2'],
        df_kpler_charts['Diverted from shipping region 1']
    )
    df_kpler_charts['New destination basin'] = np.where(
        df_kpler_charts['New destination basin 1'].isnull(),
        df_kpler_charts['New destination basin 2'],
        df_kpler_charts['New destination basin 1']
    )
    df_kpler_charts['New destination shipping region'] = np.where(
        df_kpler_charts['New destination shipping region 1'].isnull(),
        df_kpler_charts['New destination shipping region 2'],
        df_kpler_charts['New destination shipping region 1']
    )

    diversion_month = pd.to_datetime(df_kpler_charts["Diversion date"], errors='coerce')
    df_kpler_charts["Diversion_month"] = diversion_month.dt.to_period("M").dt.to_timestamp().astype(str)
    df_kpler_charts["basin_combo"] = (
        df_kpler_charts["Diverted from basin"].fillna('Unknown') + " -> " +
        df_kpler_charts["New destination basin"].fillna('Unknown')
    )
    df_kpler_charts["region_combo"] = (
        df_kpler_charts["Diverted from shipping region"].fillna('Unknown') + " -> " +
        df_kpler_charts["New destination shipping region"].fillna('Unknown')
    )
    df_kpler_charts["country_combo"] = (
        df_kpler_charts["Diverted from country"].fillna('Unknown') + " -> " +
        df_kpler_charts["New destination country"].fillna('Unknown')
    )

    return df_kpler_charts[needed_columns].where(pd.notnull(df_kpler_charts[needed_columns]), None)


@callback(
    Output('imp-diversion-processed-data', 'data'),
    Input('global-refresh-button', 'n_clicks'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-destination-catalog-store', 'data'),
)
def process_diversion_data(n_clicks, selected_destination_aggregation, selected_destination, destination_catalog):
    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    if not destination_context['destination_countries']:
        return {'main_data': [], 'charts_data': [], 'destination_label': destination_context['display_label']}

    query = text(f"""
        SELECT
            diversion_date AS "Diversion date",
            vessel_name AS "Vessel",
            vessel_state AS "State",
            charterer_name AS "Charterer",
            cargo_origin_cubic_meters AS "Cubic Meters",
            origin_diversion_location_name AS "Origin location",
            origin_diversion_country_name AS "Origin country",
            origin_diversion_date AS "Origin date",
            diverted_from_location_name AS "Diverted from location",
            diverted_from_country_name AS "Diverted from country",
            diverted_from_date AS "Diverted from date",
            new_destination_location_name AS "New destination location",
            new_destination_country_name AS "New destination country",
            new_destination_date AS "New destination date"
        FROM {DB_SCHEMA}.kpler_lng_diversions
        WHERE upload_timestamp_utc = (
            SELECT MAX(upload_timestamp_utc)
            FROM {DB_SCHEMA}.kpler_lng_diversions
        )
            AND new_destination_country_name IN :destination_countries
    """)
    df_kpler_diversions = pd.read_sql(
        query,
        engine,
        params={'destination_countries': destination_context['destination_countries']}
    )
    if df_kpler_diversions.empty:
        return {
            'main_data': [],
            'charts_data': [],
            'destination_label': destination_context['display_label']
        }

    df_kpler_diversions['Added shipping days'] = (
        df_kpler_diversions['New destination date'] - df_kpler_diversions['Diverted from date']
    ).dt.days

    main_df = df_kpler_diversions.copy()
    date_columns = ['Diversion date', 'Origin date', 'Diverted from date', 'New destination date']
    for col in date_columns:
        main_df[col] = pd.to_datetime(main_df[col], errors='coerce').dt.date.astype(str)

    filter_date = dt.date(2024, 1, 1)
    main_df_filtered = main_df[pd.to_datetime(main_df['Diversion date']).dt.date >= filter_date]
    data_kpler_diversions = main_df_filtered.to_dict("records")

    df_kpler_charts = df_kpler_diversions[df_kpler_diversions['State'] == 'Loaded'].copy()
    df_mapping_country = pd.read_sql(text(f"SELECT * FROM {DB_SCHEMA}.mappings_country"), engine)
    df_mapping_location = pd.read_sql(text(f"SELECT * FROM {DB_SCHEMA}.mapping_destination_location_name"), engine)
    charts_df = build_importer_diversion_chart_dataframe(df_kpler_charts, df_mapping_country, df_mapping_location)

    return {
        'main_data': data_kpler_diversions,
        'charts_data': charts_df.to_dict('records'),
        'destination_label': destination_context['display_label']
    }


@callback(
    Output('imp-diversion-table', 'data'),
    Output('imp-diversion-table', 'columns'),
    Output('imp-diversion-count-chart', 'figure'),
    Input('imp-diversion-processed-data', 'data'),
    Input('imp-diversion-combo-radio', 'value'),
)
def update_diversion_ui(stored_data, combo_level):
    if not stored_data:
        empty_fig = go.Figure().update_layout(title="No data available")
        empty_columns = [{"name": "No Data", "id": "no_data"}]
        return [], empty_columns, empty_fig

    data_kpler_diversions = stored_data['main_data']
    diversion_columns = [{"name": col, "id": col} for col in data_kpler_diversions[0].keys()] if data_kpler_diversions else [
        {"name": "No Data", "id": "no_data"}
    ]

    df_kpler_charts = pd.DataFrame(stored_data['charts_data'])
    if df_kpler_charts.empty:
        empty_fig = go.Figure().update_layout(title="No data available")
        return data_kpler_diversions, diversion_columns, empty_fig

    df_kpler_charts["Diversion_month"] = pd.to_datetime(df_kpler_charts["Diversion_month"])
    df_kpler_charts["Month_Display"] = df_kpler_charts["Diversion_month"].dt.strftime('%Y-%m-%d')
    combo_field = combo_level
    all_combo_values = df_kpler_charts[combo_field].dropna().unique()
    color_mapping = {
        combo: get_professional_colors(len(all_combo_values))[idx]
        for idx, combo in enumerate(sorted(all_combo_values))
    }

    df_count = df_kpler_charts.groupby(["Diversion_month", combo_field]).size().reset_index(name='Count')
    df_days = df_kpler_charts.groupby(["Diversion_month", combo_field], as_index=False)["Added shipping days"].sum()
    df_volumes = df_kpler_charts.groupby(["Diversion_month", combo_field], as_index=False)["Cubic Meters"].sum()

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Count of Trades", "Added Shipping Days", "Cargo Volumes"),
        shared_xaxes=True,
        horizontal_spacing=0.08
    )

    for combo in sorted(all_combo_values):
        combo_count = df_count[df_count[combo_field] == combo]
        if not combo_count.empty:
            fig.add_trace(go.Bar(
                x=combo_count["Diversion_month"],
                y=combo_count["Count"],
                name=combo,
                marker_color=color_mapping[combo],
                legendgroup=combo,
            ), row=1, col=1)

        combo_days = df_days[df_days[combo_field] == combo]
        if not combo_days.empty:
            fig.add_trace(go.Bar(
                x=combo_days["Diversion_month"],
                y=combo_days["Added shipping days"],
                name=combo,
                marker_color=color_mapping[combo],
                legendgroup=combo,
                showlegend=False,
            ), row=1, col=2)

        combo_volumes = df_volumes[df_volumes[combo_field] == combo]
        if not combo_volumes.empty:
            fig.add_trace(go.Bar(
                x=combo_volumes["Diversion_month"],
                y=combo_volumes["Cubic Meters"],
                name=combo,
                marker_color=color_mapping[combo],
                legendgroup=combo,
                showlegend=False,
            ), row=1, col=3)

    fig.update_layout(
        barmode='stack',
        height=500,
        paper_bgcolor=PROFESSIONAL_COLORS['bg_white'],
        plot_bgcolor=PROFESSIONAL_COLORS['bg_white'],
        font=dict(family='Inter, -apple-system, BlinkMacSystemFont, sans-serif', size=12,
                  color=PROFESSIONAL_COLORS['text_secondary']),
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
    fig.update_xaxes(
        title_text="Month",
        title_font=dict(family='Inter, -apple-system, BlinkMacSystemFont, sans-serif', size=13,
                        color=PROFESSIONAL_COLORS['text_primary']),
        tickfont=dict(family='Inter, -apple-system, BlinkMacSystemFont, sans-serif', size=11,
                      color=PROFESSIONAL_COLORS['text_secondary']),
        gridcolor=PROFESSIONAL_COLORS['grid_color'],
        gridwidth=0.5,
        linecolor=PROFESSIONAL_COLORS['grid_color'],
        linewidth=1,
        showgrid=True,
        zeroline=False
    )
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Added Days", row=1, col=2)
    fig.update_yaxes(title_text="Cubic Meters", row=1, col=3)

    def create_split_header_columns(df):
        columns = [{"name": ["Month", ""], "id": "Month_Display"}]
        for col in df.columns:
            if col == "Month_Display":
                continue
            from_to_parts = str(col).split(" -> ")
            if len(from_to_parts) == 2:
                columns.append({"name": [from_to_parts[0], from_to_parts[1]], "id": col})
            else:
                columns.append({"name": [str(col), ""], "id": col})
        return columns

    return data_kpler_diversions, diversion_columns, fig


@callback(
    Output('imp-download-diversion-summary-excel', 'data'),
    Input('imp-export-diversion-summary-button', 'n_clicks'),
    State('imp-diversion-table', 'data'),
    State('imp-diversion-table', 'columns'),
    prevent_initial_call=True
)
def export_importer_diversion_summary_to_excel(n_clicks, table_data, table_columns):
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


@callback(
    Output('imp-origin-expanded-continents', 'data', allow_duplicate=True),
    [Input({'type': 'imp-origin-expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'imp-origin-expandable-table', 'index': ALL}, 'data'),
     State('imp-origin-expanded-continents', 'data')],
    prevent_initial_call=True
)
def toggle_origin_continent_expansion(active_cells, table_data_list, expanded_continents):
    if not any(active_cells):
        return expanded_continents or []

    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    if 'imp-origin-expandable-table' in prop_id and '.active_cell' in prop_id:
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
    Output('imp-origin-forecast-expanded-continents', 'data', allow_duplicate=True),
    [Input({'type': 'imp-origin-forecast-expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'imp-origin-forecast-expandable-table', 'index': ALL}, 'data'),
     State('imp-origin-forecast-expanded-continents', 'data')],
    prevent_initial_call=True
)
def toggle_origin_forecast_continent_expansion(active_cells, table_data_list, expanded_continents):
    if not any(active_cells):
        return expanded_continents or []

    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    if 'imp-origin-forecast-expandable-table' in prop_id and '.active_cell' in prop_id:
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
    Output('imp-maintenance-summary-container', 'children'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    State('imp-maintenance-expanded-plants', 'data'),
    Input('imp-destination-catalog-store', 'data')
)
def update_maintenance_table(selected_destination_aggregation, selected_destination, expanded_plants,
                             destination_catalog):
    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    destination_label = destination_context['display_label']
    if not destination_context['destination_countries']:
        return html.Div("Please select a destination.", style={'textAlign': 'center', 'padding': '20px'})

    try:
        raw_data = fetch_train_maintenance_data(engine, destination_context['destination_countries'])
        if raw_data.empty:
            return html.Div(
                f"No supplier maintenance data available for cargoes serving {destination_label}.",
                style={'textAlign': 'center', 'padding': '20px'}
            )

        processed_data, comments_data = process_maintenance_periods_hierarchical(raw_data, expanded_plants)
        if processed_data.empty:
            return html.Div("No maintenance data to display.", style={'textAlign': 'center', 'padding': '20px'})

        return create_maintenance_summary_table(processed_data, comments_data)
    except Exception as e:
        return html.Div(f"Error loading maintenance data: {str(e)}",
                        style={'textAlign': 'center', 'padding': '20px', 'color': 'red'})


@callback(
    Output('imp-maintenance-expanded-plants', 'data', allow_duplicate=True),
    Output('imp-maintenance-summary-container', 'children', allow_duplicate=True),
    [Input({'type': 'imp-maintenance-expandable-table', 'index': ALL}, 'active_cell')],
    [State({'type': 'imp-maintenance-expandable-table', 'index': ALL}, 'data'),
     State('imp-maintenance-expanded-plants', 'data'),
     State('imp-destination-aggregation-dropdown', 'value'),
     State('imp-destination-country-dropdown', 'value'),
     State('imp-destination-catalog-store', 'data')],
    prevent_initial_call=True
)
def toggle_maintenance_plant_expansion(active_cells, table_data_list, expanded_plants,
                                       selected_destination_aggregation, selected_destination,
                                       destination_catalog):
    if not any(active_cells):
        raise PreventUpdate

    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    if 'imp-maintenance-expandable-table' not in prop_id or '.active_cell' not in prop_id:
        raise PreventUpdate

    try:
        active_cell = active_cells[0]
        if not active_cell or active_cell['column_id'] not in ('Supplier Country', 'Plant'):
            raise PreventUpdate

        table_data = table_data_list[0]
        if not table_data:
            raise PreventUpdate

        clicked_row = table_data[active_cell['row']]
        if clicked_row.get('Type') not in ('country', 'plant'):
            raise PreventUpdate

        plant_key = clicked_row.get('PlantKey')
        if not plant_key:
            raise PreventUpdate

        expanded_plants = expanded_plants or []
        if plant_key in expanded_plants:
            expanded_plants.remove(plant_key)
        else:
            expanded_plants.append(plant_key)

        destination_context = resolve_destination_context(
            selected_destination_aggregation,
            selected_destination,
            destination_catalog
        )
        raw_data = fetch_train_maintenance_data(engine, destination_context['destination_countries'])
        if raw_data.empty:
            return expanded_plants, no_update

        processed_data, comments_data = process_maintenance_periods_hierarchical(raw_data, expanded_plants)
        if processed_data.empty:
            return expanded_plants, no_update

        return expanded_plants, create_maintenance_summary_table(processed_data, comments_data)
    except PreventUpdate:
        raise
    except Exception as e:
        raise PreventUpdate
