from dash import html, dcc, callback, clientside_callback, Output, Input, State, ALL, ctx, no_update
from utils.ag_grid_tables import (
    ag_grid_cell_clicked_to_active_cell,
    ag_grid_column_defs_to_datatable_columns,
    create_ag_grid_from_datatable,
    datatable_columns_to_ag_grid_column_defs,
)
from utils.detail_controls import (
    coerce_detail_count as _coerce_detail_count,
    detail_count_options as _detail_count_options,
    format_rolling_window_label,
    format_rolling_window_title,
    normalize_rolling_window_days,
)
from utils.detail_table_formatting import (
    format_table_value_max_one_decimal as _format_table_value_max_one_decimal,
    round_table_value_max_one_decimal as _round_table_value_max_one_decimal,
)
from utils.dataframe_store import (
    load_dataframe_from_payload as _load_store_dataframe,
    serialize_dataframe_split_store as _store_dataframe,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime as dt
import calendar
import re
from io import BytesIO
from dash.exceptions import PreventUpdate

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

MCM_PER_CUBIC_METER = 0.6 / 1000
BCM_PER_MMTPA = 1.36
DAYS_PER_YEAR = 365.25
MCM_PER_BCM = 1000
MCM_PER_MT = BCM_PER_MMTPA * MCM_PER_BCM
MCM_PER_MONTH_PER_MMTPA = BCM_PER_MMTPA * MCM_PER_BCM / 12
MMTPA_PER_MCM_D = DAYS_PER_YEAR / MCM_PER_MT

# Volume unit conversion factors from the page's base mcm/d series.
# Annualized conversions use 1 MMTPA = 1.36 BCM/year.
# MT is period-aware: mcm/d x represented days / 1,360 MCM per MT.
VOLUME_CONVERSIONS = {
    'mcm_d': {'factor': 1.0, 'label': 'mcm/d'},
    'mt': {'factor': None, 'label': 'MT'},
    'mtpa': {'factor': MMTPA_PER_MCM_D, 'label': 'MMTPA'},
}

VOLUME_METRIC_OPTIONS = [
    {'label': 'mcm/d', 'value': 'mcm_d'},
    {'label': 'MT', 'value': 'mt'},
    {'label': 'MMTPA', 'value': 'mtpa'},
]

SUPPLY_CHART_FORECAST_DASH = 'dot'
SUPPLY_CHART_RANGE_FILL = 'rgba(148, 163, 184, 0.20)'
SUPPLY_CHART_RANGE_LOOKBACK_YEARS = 5
DETAIL_CHART_DATA_START_DATE = '2021-01-01'
SUPPLY_CHART_COLOR_SEQUENCE = [
    '#8fb3d9',
    '#5f8fbe',
    '#2f6fbb',
    '#0f4c81',
    '#17324d',
]
DETAIL_DEFAULT_VISIBLE_START_YEAR = 2025
DETAIL_CHART_ANCHOR_YEAR = 2024
IMPORTER_DETAIL_SUPPLY_CHART_HEIGHT = 476
CONTINENT_CHART_COLOR_MAP = {
    'Africa': '#7a5195',
    'Americas': '#2f9e7e',
    'Asia': '#d64550',
    'Europe': '#2f6fbb',
    'Unknown': '#7b8794',
    'Oceania': '#d08b36',
    'Middle East': '#00a0b0',
    'North America': '#b83280',
    'South America': '#c7a12b',
}
CONTINENT_CHART_PREVIOUS_YEAR_WIDTH = 1.15
CONTINENT_CHART_CURRENT_YEAR_WIDTH = 2.8
CONTINENT_CHART_FORECAST_DASH = 'dot'
DETAIL_CHIP_INPUT_STYLE = {
    'position': 'absolute',
    'opacity': 0,
    'width': '1px',
    'height': '1px',
    'margin': 0,
    'pointerEvents': 'none',
}
DETAIL_PERIOD_COMPARISON_BASIS_OPTIONS = [
    {'label': 'Levels', 'value': 'levels'},
    {'label': 'vs Previous Period', 'value': 'previous_period'},
    {'label': 'vs Previous Year', 'value': 'same_period_last_year'},
]
DETAIL_DEFAULT_QUARTER_COUNT = 5
DETAIL_DEFAULT_MONTH_COUNT = 3
DETAIL_DEFAULT_WEEK_COUNT = 3
DETAIL_MAX_QUARTER_COUNT = 8
DETAIL_MAX_MONTH_COUNT = 12
DETAIL_MAX_WEEK_COUNT = 12
MAINTENANCE_DEFAULT_QUARTER_COUNT = 5
MAINTENANCE_DEFAULT_MONTH_COUNT = 3
MAINTENANCE_MAX_FORWARD_QUARTER_COUNT = 4
IMPORTER_DETAIL_DEFAULT_TEXT_WIDTH_LIMITS = (76, 170)
IMPORTER_DETAIL_DEFAULT_NUMERIC_WIDTH_LIMITS = (58, 94)
IMPORTER_DETAIL_PERIOD_WIDTH_LIMITS = {
    'Continent': (108, 156),
    'Country': (70, 132),
}
IMPORTER_DETAIL_MAINTENANCE_WIDTH_LIMITS = {
    'Supplier Country': (126, 192),
    'Plant': (126, 210),
    'Train': (50, 90),
    'Type': (1, 1),
    'PlantKey': (1, 1),
}
DIVERSION_NUMERIC_TABLE_COLUMNS = {'Cubic Meters', 'Added shipping days'}
DIVERSION_DATE_TABLE_COLUMNS = {
    'Diversion date',
    'Origin date',
    'Diverted from date',
    'New destination date',
}
DIVERSION_ROUTE_TABLE_COLUMNS = {
    'Origin location',
    'Origin country',
    'Diverted from location',
    'Diverted from country',
    'New destination location',
    'New destination country',
}
IMPORTER_DETAIL_DIVERSION_WIDTH_LIMITS = {
    'Diversion date': (110, 118),
    'Origin date': (88, 98),
    'Diverted from date': (122, 132),
    'New destination date': (132, 142),
    'Vessel': (118, 168),
    'State': (58, 86),
    'Charterer': (76, 130),
    'Cubic Meters': (84, 108),
    'Added shipping days': (132, 142),
    'Origin location': (98, 158),
    'Origin country': (88, 132),
    'Diverted from location': (126, 178),
    'Diverted from country': (116, 162),
    'New destination location': (132, 186),
    'New destination country': (150, 176),
}
DIVERSION_CHART_COLOR_SEQUENCE = [
    '#26547c',
    '#2f8f7b',
    '#d08b36',
    '#7a5195',
    '#d64550',
    '#607d8b',
    '#1aa6a6',
    '#0f4c81',
    '#b83280',
    '#c7a12b',
]
DIVERSION_CHART_OTHER_COLOR = '#94a3b8'
DIVERSION_CHART_MAX_SERIES_BY_LEVEL = {
    'basin_combo': 12,
    'region_combo': 16,
    'country_combo': 12,
}
DIVERSION_DASHBOARD_ROW_LIMIT = 10
IMPORTER_DETAIL_PERIOD_GRID_OPTIONS = {
    'domLayout': 'autoHeight',
    'rowHeight': 30,
    'headerHeight': 32,
    'groupHeaderHeight': 28,
    'pagination': False,
    'suppressPaginationPanel': True,
    'enableCellTextSelection': True,
    'ensureDomOrder': True,
    'animateRows': False,
    'alwaysShowHorizontalScroll': False,
    'alwaysShowVerticalScroll': False,
}
IMPORTER_DETAIL_PERIOD_DEFAULT_COL_DEF = {
    'wrapHeaderText': True,
    'autoHeaderHeight': True,
    'suppressHeaderMenuButton': True,
    'suppressHeaderFilterButton': True,
    'resizable': True,
}


def _normalize_detail_volume_metric(volume_metric):
    return volume_metric if volume_metric in VOLUME_CONVERSIONS else 'mcm_d'


def _get_detail_volume_metric_info(volume_metric):
    return VOLUME_CONVERSIONS[_normalize_detail_volume_metric(volume_metric)]


def _get_detail_volume_metric_factor(volume_metric, period_days=None):
    normalized_metric = _normalize_detail_volume_metric(volume_metric)
    if normalized_metric == 'mt':
        days = period_days if period_days is not None else DAYS_PER_YEAR
        return days / MCM_PER_MT
    return VOLUME_CONVERSIONS[normalized_metric]['factor']


def _convert_detail_volume_series(series, volume_metric, period_days=None, precision=None):
    numeric_series = pd.to_numeric(series, errors='coerce')
    converted_series = numeric_series * _get_detail_volume_metric_factor(
        volume_metric,
        period_days=period_days
    )
    if precision is not None:
        converted_series = converted_series.round(precision)
    return converted_series.where(pd.notnull(converted_series), None)


def _convert_detail_volume_dataframe(
    df,
    volume_metric,
    columns=None,
    exclude_columns=None,
    precision=None,
    period_days=None,
    period_days_by_column=None
):
    if df is None or df.empty:
        return df

    converted_df = df.copy()
    exclude_columns = set(exclude_columns or [])
    period_days_by_column = period_days_by_column or {}
    if columns is None:
        columns = [
            col for col in converted_df.columns
            if col not in exclude_columns
        ]

    for col in columns:
        if col not in converted_df.columns or col in exclude_columns:
            continue
        converted_df[col] = _convert_detail_volume_series(
            converted_df[col],
            volume_metric,
            period_days=period_days_by_column.get(col, period_days),
            precision=precision
        )
    return converted_df


def get_volume_metric_info(volume_metric):
    """Return conversion metadata for a selected volume metric."""
    return _get_detail_volume_metric_info(volume_metric)


def convert_volume_metric_dataframe(
    df,
    volume_metric,
    columns=None,
    exclude_columns=None,
    precision=None,
    period_days=None,
    period_days_by_column=None
):
    """Convert selected numeric dataframe columns from mcm/d to the display metric."""
    return _convert_detail_volume_dataframe(
        df,
        volume_metric,
        columns=columns,
        exclude_columns=exclude_columns,
        precision=precision,
        period_days=period_days,
        period_days_by_column=period_days_by_column
    )


def _is_detail_quarter_column(column_name):
    return isinstance(column_name, str) and column_name.startswith('Q') and "'" in column_name


def _is_detail_week_column(column_name):
    return isinstance(column_name, str) and column_name.startswith('W') and "'" in column_name


def _is_detail_month_column(column_name):
    return (
        isinstance(column_name, str)
        and "'" in column_name
        and not _is_detail_quarter_column(column_name)
        and not _is_detail_week_column(column_name)
    )


def _is_detail_rolling_column(column_name):
    return (
        isinstance(column_name, str)
        and (column_name == '7D' or (column_name.endswith('D') and column_name[:-1].isdigit()))
    )


def _is_detail_delta_column(column_name):
    return isinstance(column_name, str) and column_name.startswith('Δ ')


def _strip_detail_reference_suffix(column_name):
    column_name = str(column_name)
    for suffix in ('_PP', '_Y1'):
        if column_name.endswith(suffix):
            return column_name[:-len(suffix)]
    return column_name


def _parse_detail_year_suffix(year_suffix):
    try:
        year = int(str(year_suffix).strip())
    except (TypeError, ValueError):
        return None
    return 2000 + year if year < 100 else year


def _get_detail_month_number(month_label):
    month_lookup = {
        calendar.month_abbr[month_num]: month_num
        for month_num in range(1, 13)
    }
    return month_lookup.get(str(month_label).strip())


def _get_detail_period_column_days(column_name, default_rolling_days=None):
    base_column = _strip_detail_reference_suffix(column_name)
    if _is_detail_rolling_column(base_column):
        if base_column == '7D':
            return 7
        try:
            return int(str(base_column).replace('D', ''))
        except (TypeError, ValueError):
            return default_rolling_days

    if _is_detail_quarter_column(base_column):
        return 91.25

    if _is_detail_month_column(base_column):
        try:
            month_label, year_suffix = str(base_column).split("'")
        except ValueError:
            return None
        month_num = _get_detail_month_number(month_label)
        year = _parse_detail_year_suffix(year_suffix)
        if month_num is None or year is None:
            return None
        return calendar.monthrange(year, month_num)[1]

    if _is_detail_week_column(base_column):
        return 7

    if isinstance(base_column, str) and base_column.endswith(' Avg'):
        year_text = base_column.split()[0]
        try:
            year = int(year_text)
        except (TypeError, ValueError):
            return None
        return 366 if calendar.isleap(year) else 365

    return default_rolling_days if base_column == format_rolling_window_label(default_rolling_days) else None


def _build_detail_period_days_map(columns, rolling_window_days=None):
    normalized_window_days = normalize_rolling_window_days(rolling_window_days)
    period_days_by_column = {}
    for column_name in ([] if columns is None else columns):
        period_days = _get_detail_period_column_days(
            column_name,
            default_rolling_days=normalized_window_days
        )
        if period_days is not None:
            period_days_by_column[column_name] = period_days
    return period_days_by_column


def _convert_detail_period_display_df(display_df, volume_metric, rolling_window_days=None, exclude_columns=None):
    if display_df is None or display_df.empty:
        return display_df

    converted_df = display_df.copy()
    text_columns = {
        'Continent', 'Country', 'Zone', 'Destination',
        'continent', 'country', 'Plant', 'Train', 'Type', 'Supplier Country', 'PlantKey'
    }
    text_columns.update(exclude_columns or [])
    delta_columns = [col for col in converted_df.columns if _is_detail_delta_column(col)]
    period_days_by_column = _build_detail_period_days_map(
        converted_df.columns,
        rolling_window_days=rolling_window_days
    )
    convert_columns = [
        col for col in converted_df.columns
        if col not in text_columns and col not in delta_columns
    ]
    converted_df = _convert_detail_volume_dataframe(
        converted_df,
        volume_metric,
        columns=convert_columns,
        exclude_columns=text_columns,
        precision=1,
        period_days_by_column=period_days_by_column
    )

    for delta_col in delta_columns:
        if delta_col.startswith('Δ 7D-'):
            compare_col = delta_col.replace('Δ 7D-', '', 1)
            if {'7D', compare_col}.issubset(converted_df.columns):
                converted_df[delta_col] = (
                    pd.to_numeric(converted_df['7D'], errors='coerce')
                    - pd.to_numeric(converted_df[compare_col], errors='coerce')
                ).round(1)
                continue

        if delta_col.startswith('Δ ') and delta_col.endswith(' Y/Y'):
            base_col = delta_col.replace('Δ ', '', 1)[:-4]
            reference_col = f'{base_col}_Y1'
            if {base_col, reference_col}.issubset(converted_df.columns):
                converted_df[delta_col] = (
                    pd.to_numeric(converted_df[base_col], errors='coerce')
                    - pd.to_numeric(converted_df[reference_col], errors='coerce')
                ).round(1)
                continue
            period_days = period_days_by_column.get(base_col)
        else:
            period_days = period_days_by_column.get(delta_col)

        if delta_col in display_df.columns:
            converted_df[delta_col] = _convert_detail_volume_series(
                display_df[delta_col],
                volume_metric,
                period_days=period_days,
                precision=1
            )

    return converted_df


def _normalize_detail_comparison_basis(comparison_basis):
    if comparison_basis in {'levels', 'previous_period', 'same_period_last_year'}:
        return comparison_basis
    return 'levels'


def _get_detail_previous_period_label(column_name, period_view):
    column_name = str(column_name)
    try:
        if period_view == 'quarter':
            quarter_part, year_suffix = column_name.split("'")
            quarter_num = int(quarter_part.replace('Q', ''))
            year = int(f'20{year_suffix}')
            if quarter_num == 1:
                quarter_num = 4
                year -= 1
            else:
                quarter_num -= 1
            return f"Q{quarter_num}'{str(year)[2:]}"
        if period_view == 'month':
            month_part, year_suffix = column_name.split("'")
            month_order = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            month_lookup = {value: key for key, value in month_order.items()}
            month_num = month_order[month_part]
            year = int(f'20{year_suffix}')
            if month_num == 1:
                month_num = 12
                year -= 1
            else:
                month_num -= 1
            return f"{month_lookup[month_num]}'{str(year)[2:]}"
    except (KeyError, ValueError, TypeError):
        return None
    return None


def _get_detail_prior_year_label(column_name, period_view):
    column_name = str(column_name)
    try:
        label_part, year_suffix = column_name.split("'")
        year = int(f'20{year_suffix}') - 1
        if period_view == 'quarter':
            return f"{label_part}'{str(year)[2:]}"
        if period_view == 'month':
            return f"{label_part}'{str(year)[2:]}"
        if period_view == 'week':
            return f"{label_part}'{str(year)[2:]}"
    except (ValueError, TypeError):
        return None
    return None


def _get_detail_previous_week_label(column_name, week_columns):
    if column_name in week_columns:
        column_index = week_columns.index(column_name)
        if column_index > 0:
            return week_columns[column_index - 1]
    return None


def _build_detail_period_comparison_reference_map(
    visible_period_columns,
    week_columns,
    rolling_columns,
    comparison_basis,
):
    comparison_basis = _normalize_detail_comparison_basis(comparison_basis)
    if comparison_basis not in {'previous_period', 'same_period_last_year'}:
        return {}

    reference_map = {}
    for column_id in visible_period_columns:
        reference_col = None
        if _is_detail_quarter_column(column_id):
            reference_col = (
                _get_detail_previous_period_label(column_id, 'quarter')
                if comparison_basis == 'previous_period'
                else _get_detail_prior_year_label(column_id, 'quarter')
            )
        elif _is_detail_month_column(column_id):
            reference_col = (
                _get_detail_previous_period_label(column_id, 'month')
                if comparison_basis == 'previous_period'
                else _get_detail_prior_year_label(column_id, 'month')
            )
        elif _is_detail_week_column(column_id):
            reference_col = (
                _get_detail_previous_week_label(column_id, week_columns)
                if comparison_basis == 'previous_period'
                else _get_detail_prior_year_label(column_id, 'week')
            )
        if reference_col:
            reference_map[column_id] = reference_col

    for column_id in rolling_columns:
        reference_col = (
            f'{column_id}_PP'
            if comparison_basis == 'previous_period'
            else f'{column_id}_Y1'
        )
        reference_map[column_id] = reference_col

    return reference_map


def _filter_detail_period_display_columns(
    display_df,
    comparison_basis,
    quarter_count,
    month_count,
    week_count,
    return_metadata=False
):
    comparison_basis = _normalize_detail_comparison_basis(comparison_basis)
    empty_metadata = {
        'comparison_basis': comparison_basis,
        'visible_period_cols': [],
        'visible_comparison_cols': [],
        'comparison_reference_map': {},
        'reference_cols': [],
        'comparison_delta_cols': [],
    }
    if display_df is None or display_df.empty:
        if return_metadata:
            return display_df, empty_metadata
        return display_df

    quarter_count = _coerce_detail_count(
        quarter_count,
        DETAIL_DEFAULT_QUARTER_COUNT,
        DETAIL_MAX_QUARTER_COUNT,
        min_count=1
    )
    month_count = _coerce_detail_count(
        month_count,
        DETAIL_DEFAULT_MONTH_COUNT,
        DETAIL_MAX_MONTH_COUNT,
        min_count=1
    )
    week_count = _coerce_detail_count(
        week_count,
        DETAIL_DEFAULT_WEEK_COUNT,
        DETAIL_MAX_WEEK_COUNT,
        min_count=1
    )

    columns = list(display_df.columns)
    text_columns = [col for col in columns if col in {'Continent', 'Country', 'Zone', 'Destination'}]
    quarter_columns = [col for col in columns if _is_detail_quarter_column(col)]
    month_columns = [col for col in columns if _is_detail_month_column(col) and col not in text_columns]
    week_columns = [col for col in columns if _is_detail_week_column(col)]
    rolling_columns = [col for col in columns if _is_detail_rolling_column(col)]
    rolling_columns_before_weeks = [col for col in rolling_columns if col != '7D']
    rolling_columns_after_weeks = [col for col in rolling_columns if col == '7D']
    delta_columns = [col for col in columns if _is_detail_delta_column(col)]

    visible_period_columns = (
        quarter_columns[-quarter_count:]
        + month_columns[-month_count:]
        + week_columns[-week_count:]
    )
    comparison_reference_map = _build_detail_period_comparison_reference_map(
        visible_period_columns,
        week_columns,
        rolling_columns,
        comparison_basis
    )
    visible_comparison_columns = visible_period_columns + rolling_columns
    reference_columns = [
        reference_col
        for reference_col in comparison_reference_map.values()
        if reference_col in columns and reference_col not in visible_comparison_columns
    ]

    selected_columns = list(text_columns)
    selected_columns.extend(quarter_columns[-quarter_count:])
    selected_columns.extend(month_columns[-month_count:])
    selected_columns.extend(rolling_columns_before_weeks)
    selected_columns.extend(week_columns[-week_count:])
    selected_columns.extend(rolling_columns_after_weeks)
    selected_columns.extend(delta_columns)
    selected_columns.extend(reference_columns)

    selected_columns = [col for col in selected_columns if col in columns]
    filtered_df = display_df.loc[:, selected_columns].copy()
    metadata = {
        'comparison_basis': comparison_basis,
        'visible_period_cols': visible_period_columns,
        'visible_comparison_cols': visible_comparison_columns,
        'comparison_reference_map': comparison_reference_map,
        'reference_cols': reference_columns,
        'comparison_delta_cols': (
            visible_comparison_columns
            if comparison_basis in {'previous_period', 'same_period_last_year'}
            else []
        ),
    }
    if return_metadata:
        return filtered_df, metadata
    return filtered_df


def _apply_exporter_detail_period_comparison(display_df, comparison_metadata):
    if display_df is None or display_df.empty:
        return display_df, []

    comparison_metadata = comparison_metadata or {}
    comparison_basis = _normalize_detail_comparison_basis(
        comparison_metadata.get('comparison_basis')
    )
    if comparison_basis not in {'previous_period', 'same_period_last_year'}:
        reference_cols = [
            col for col in comparison_metadata.get('reference_cols', [])
            if col in display_df.columns
        ]
        if reference_cols:
            display_df = display_df.drop(columns=reference_cols, errors='ignore')
        return display_df, []

    comparison_source_df = display_df.copy()
    comparison_delta_cols = []
    reference_map = comparison_metadata.get('comparison_reference_map') or {}
    visible_comparison_cols = [
        col for col in comparison_metadata.get('visible_comparison_cols', [])
        if col in display_df.columns
    ]

    for visible_col in visible_comparison_cols:
        reference_col = reference_map.get(visible_col)
        if reference_col in comparison_source_df.columns:
            visible_values = pd.to_numeric(comparison_source_df[visible_col], errors='coerce')
            reference_values = pd.to_numeric(comparison_source_df[reference_col], errors='coerce')
            display_df[visible_col] = visible_values - reference_values
        else:
            display_df[visible_col] = pd.NA
        comparison_delta_cols.append(visible_col)

    reference_cols = [
        col for col in comparison_metadata.get('reference_cols', [])
        if col not in visible_comparison_cols
    ]
    if reference_cols:
        display_df = display_df.drop(columns=reference_cols, errors='ignore')
    return display_df, comparison_delta_cols


def _get_exporter_detail_period_column_family(column_id, text_columns):
    column_id = str(column_id)
    if column_id in text_columns:
        return 'label'
    if column_id == '7D':
        return 'rolling-7d'
    if _is_detail_rolling_column(column_id):
        return 'rolling-30d'
    if column_id.startswith('Δ 7D-'):
        return 'delta-mom'
    if column_id.startswith('Δ ') and column_id.endswith(' Y/Y'):
        return 'delta-yoy'
    if re.match(r'^\d{4}$', column_id):
        return 'year'
    if _is_detail_quarter_column(column_id):
        return 'quarter'
    if _is_detail_week_column(column_id):
        return 'week'
    if _is_detail_month_column(column_id):
        return 'month'
    return 'numeric'


def _apply_exporter_detail_period_column_classes(
    columns,
    text_columns,
    primary_text_columns=None,
    delta_like_cols=None,
):
    classed_columns = []
    previous_family = None
    text_columns = set(text_columns or [])
    primary_text_columns = set(primary_text_columns or [])
    delta_like_cols = set(delta_like_cols or [])

    for column in columns:
        column = dict(column)
        column_id = column.get('id')
        family = _get_exporter_detail_period_column_family(column_id, text_columns)

        header_classes = [f'supply-dest-header-{family}']
        if family == 'label':
            header_classes.append(
                'supply-dest-header-label-primary'
                if column_id in primary_text_columns
                else 'supply-dest-header-label-secondary'
            )
        elif family != previous_family:
            header_classes.append('supply-dest-header-group-start')

        column['headerClass'] = ' '.join(header_classes)
        if family != 'label':
            cell_classes = ['supply-dest-summary-number-cell']
            if family in {'delta-mom', 'delta-yoy'} or column_id in delta_like_cols:
                cell_classes.append('supply-dest-summary-delta-cell')
            existing_cell_class = str(column.get('cellClass') or '').strip()
            column['cellClass'] = ' '.join(
                class_name for class_name in [existing_cell_class, *cell_classes] if class_name
            )

        classed_columns.append(column)
        previous_family = family

    return classed_columns


def _format_exporter_detail_filter_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '0'

    text = f'{number:.6f}'.rstrip('0').rstrip('.')
    return text or '0'


def _build_exporter_detail_numeric_filter_js(column_id, operator, threshold_text):
    escaped_column_id = str(column_id).replace("\\", "\\\\").replace("'", "\\'")
    return (
        f"(Number(String(params.data && params.data['{escaped_column_id}'] !== undefined "
        f"? params.data['{escaped_column_id}'] : '').replace(/[^0-9.\\-]/g, '')) "
        f"{operator} {threshold_text})"
    )


def _exporter_detail_delta_raw_field(column_id):
    safe_column_id = re.sub(r'[^0-9A-Za-z]+', '_', str(column_id)).strip('_').lower()
    safe_column_id = safe_column_id or 'value'
    return f'__exporter_detail_delta_{safe_column_id}_raw'


def _build_exporter_detail_raw_numeric_filter_js(raw_field, operator, threshold_text):
    escaped_raw_field = str(raw_field).replace("\\", "\\\\").replace("'", "\\'")
    return f"(params.data && params.data['{escaped_raw_field}'] {operator} {threshold_text})"


def _build_exporter_detail_delta_filter_js(column_id, operator, threshold_text, raw_field=None):
    if raw_field:
        return _build_exporter_detail_raw_numeric_filter_js(raw_field, operator, threshold_text)
    return _build_exporter_detail_numeric_filter_js(column_id, operator, threshold_text)


def _get_exporter_detail_delta_thresholds(display_df, column_id, total_column):
    if display_df is None or display_df.empty or column_id not in display_df.columns:
        return []

    total_mask = (
        display_df[total_column].astype(str).str.strip().eq('Global')
        if total_column in display_df.columns
        else pd.Series(False, index=display_df.index)
    )
    values = pd.to_numeric(display_df.loc[~total_mask, column_id], errors='coerce').abs()
    values = values[(values.notna()) & (values > 0)]
    if values.empty:
        return []

    thresholds = []
    for quantile in (0.45, 0.70, 0.88):
        threshold = float(values.quantile(quantile))
        if threshold <= 0:
            continue
        if thresholds and threshold <= thresholds[-1]:
            continue
        thresholds.append(threshold)
    return thresholds


def _build_exporter_detail_delta_heatmap_class_rules(display_df, column_id, total_column):
    thresholds = _get_exporter_detail_delta_thresholds(display_df, column_id, total_column)
    band_thresholds = [0, *thresholds]
    rules = {}
    for band_index, threshold in enumerate(band_thresholds, start=1):
        positive_threshold = _format_exporter_detail_filter_number(threshold)
        negative_threshold = _format_exporter_detail_filter_number(-threshold)
        raw_field = _exporter_detail_delta_raw_field(column_id)
        rules[f'supply-dest-delta-positive-{band_index}'] = {
            'function': _build_exporter_detail_delta_filter_js(
                column_id,
                '>',
                positive_threshold,
                raw_field=raw_field
            )
        }
        rules[f'supply-dest-delta-negative-{band_index}'] = {
            'function': _build_exporter_detail_delta_filter_js(
                column_id,
                '<',
                negative_threshold,
                raw_field=raw_field
            )
        }
    return rules


def _apply_exporter_detail_period_delta_heatmap_class_rules(columns, display_df, total_column, delta_like_cols=None):
    styled_columns = []
    delta_like_cols = set(delta_like_cols or [])
    for column in columns:
        column = dict(column)
        column_id = column.get('id')
        family = _get_exporter_detail_period_column_family(column_id, {'Continent', 'Country', 'Zone', 'Destination'})
        if family in {'delta-mom', 'delta-yoy'} or column_id in delta_like_cols:
            column['cellClassRules'] = _build_exporter_detail_delta_heatmap_class_rules(
                display_df,
                column_id,
                total_column
            )
        styled_columns.append(column)
    return styled_columns


def _build_exporter_detail_delta_gradient_styles(display_df, column_id, total_column, base_bg, border_color):
    styles = [{
        'if': {'column_id': column_id},
        'backgroundColor': base_bg,
        'borderLeft': f'2px solid {border_color}',
        'color': '#334155',
        'fontWeight': '700',
        'textAlign': 'right',
        'paddingRight': '12px',
    }]
    thresholds = _get_exporter_detail_delta_thresholds(display_df, column_id, total_column)
    positive_palette = [
        ('#edf8f1', '#166534', '700'),
        ('#d9f0df', '#14532d', '750'),
        ('#bfe5ca', '#0f3f25', '800'),
        ('#98d3aa', '#0b351f', '850'),
    ]
    negative_palette = [
        ('#fff1f2', '#9f1239', '700'),
        ('#fde1e4', '#9f1239', '750'),
        ('#f8c8cd', '#881337', '800'),
        ('#efa6ad', '#7f1d1d', '850'),
    ]
    for palette, operator, sign in (
        (positive_palette, '>', ''),
        (negative_palette, '<', '-')
    ):
        band_thresholds = [0, *thresholds]
        band_styles = list(zip(band_thresholds, palette[:len(band_thresholds)]))
        for threshold, (background, color, weight) in reversed(band_styles):
            threshold_text = _format_exporter_detail_filter_number(threshold)
            styles.append({
                'if': {
                    'column_id': column_id,
                    'filter_query_js': _build_exporter_detail_delta_filter_js(
                        column_id,
                        operator,
                        f'{sign}{threshold_text}',
                        raw_field=_exporter_detail_delta_raw_field(column_id)
                    )
                },
                'backgroundColor': background,
                'borderLeft': f'2px solid {border_color}',
                'color': color,
                'fontWeight': weight,
                'textAlign': 'right',
                'paddingRight': '12px',
            })
    return styles


def _build_exporter_detail_period_value_styles(display_df, text_columns, total_column, subtotal_column, delta_like_cols=None):
    if display_df is None or display_df.empty:
        return []

    text_columns = set(text_columns or [])
    delta_like_cols = set(delta_like_cols or [])
    styles = [{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8fafc'}]
    for column_id in display_df.columns:
        if column_id in text_columns:
            styles.append({'if': {'column_id': column_id}, 'textAlign': 'left'})
            continue

        family = _get_exporter_detail_period_column_family(column_id, text_columns)
        if family in {'delta-mom', 'delta-yoy'} or column_id in delta_like_cols:
            styles.extend(
                _build_exporter_detail_delta_gradient_styles(
                    display_df,
                    column_id,
                    total_column,
                    base_bg='#f3f5f7' if family == 'delta-mom' or column_id in delta_like_cols else '#eef7ee',
                    border_color='#aeb7c2' if family == 'delta-mom' or column_id in delta_like_cols else '#9abc9a',
                )
            )
            continue

        column_style = {
            'if': {'column_id': column_id},
            'textAlign': 'right',
            'paddingRight': '12px',
        }
        if family == 'year':
            column_style.update({'backgroundColor': '#f8f5ef'})
        elif family == 'quarter':
            column_style.update({'backgroundColor': '#f7f9fc'})
        elif family == 'month':
            column_style.update({'backgroundColor': '#f2faf7'})
        elif family == 'rolling-30d':
            column_style.update({
                'backgroundColor': '#fff4d6',
                'fontWeight': '750',
                'borderLeft': '2px solid #d7a23a',
            })
        elif family == 'week':
            column_style.update({'backgroundColor': '#f7f5fb'})
        elif family == 'rolling-7d':
            column_style.update({
                'backgroundColor': '#fff0e7',
                'fontWeight': '750',
                'borderLeft': '2px solid #d28a63',
            })
        styles.append(column_style)

    if subtotal_column in display_df.columns:
        styles.append({
            'if': {'filter_query': f'{{{subtotal_column}}} = "Total"'},
            'backgroundColor': '#f8fafc',
            'color': '#172033',
            'fontWeight': '750',
        })
    if total_column in display_df.columns:
        styles.append({
            'if': {'filter_query': f'{{{total_column}}} = "Global"'},
            'backgroundColor': '#edf4fb',
            'color': '#0f172a',
            'fontWeight': '850',
        })
    return styles


def _build_exporter_detail_period_grid_display(display_df, columns, delta_like_cols=None):
    """Preformat period-grid numeric values so AG Grid renders like the exporter summary table."""
    grid_df = display_df.copy()
    grid_columns = [dict(column) for column in columns]
    delta_like_cols = set(delta_like_cols or [])
    numeric_ids = {
        column.get('id')
        for column in grid_columns
        if column.get('type') == 'numeric'
    }
    delta_ids = {
        column_id
        for column_id in numeric_ids
        if (
            _get_exporter_detail_period_column_family(column_id, set()) in {'delta-mom', 'delta-yoy'}
            or column_id in delta_like_cols
        )
    }

    for column_id in delta_ids:
        if column_id not in grid_df.columns:
            continue
        raw_field = _exporter_detail_delta_raw_field(column_id)
        grid_df[raw_field] = pd.to_numeric(grid_df[column_id], errors='coerce')

    for column_id in numeric_ids:
        if column_id not in grid_df.columns:
            continue
        grid_df[column_id] = grid_df[column_id].apply(_format_table_value_max_one_decimal)

    for column in grid_columns:
        if column.get('id') in numeric_ids:
            column['type'] = 'text'
            column.pop('format', None)

    return grid_df, grid_columns


def _column_header_text(column):
    name = column.get('name') or column.get('headerName') or column.get('id') or column.get('field') or ''
    if isinstance(name, (list, tuple)):
        return ' '.join(str(part) for part in name)
    return str(name)


def _column_id(column):
    value = column.get('id', column.get('field', column.get('name', '')))
    return str(value)


def _width_sample_text(value):
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return _format_table_value_max_one_decimal(value)
    return str(value)


def _compact_column_width(header_text, value_samples, min_width, max_width, is_numeric=False):
    samples = [str(header_text), *[str(sample) for sample in value_samples if str(sample)]]
    max_header_chars = len(str(header_text))
    max_value_chars = max((len(sample) for sample in samples[1:]), default=0)
    effective_header_chars = max_header_chars
    if not is_numeric and max_header_chars > 16:
        effective_header_chars = int(np.ceil(max_header_chars / 2)) + 2
    header_px = effective_header_chars * 6.4 + (26 if is_numeric else 28)
    value_px = max_value_chars * (6.1 if is_numeric else 6.0) + (22 if is_numeric else 24)
    width = int(round(max(header_px, value_px, min_width)))
    return int(min(max(width, min_width), max_width))


def _build_importer_detail_column_width_styles(
    data,
    columns,
    *,
    numeric_columns=None,
    width_limits=None,
    default_text_limits=IMPORTER_DETAIL_DEFAULT_TEXT_WIDTH_LIMITS,
    default_numeric_limits=IMPORTER_DETAIL_DEFAULT_NUMERIC_WIDTH_LIMITS,
):
    """Build compact AG Grid width styles from displayed headers and values."""
    if data is None:
        records = []
    elif hasattr(data, 'to_dict'):
        records = data.to_dict('records')
    else:
        records = list(data or [])

    numeric_columns = {str(column) for column in (numeric_columns or set())}
    width_limits = width_limits or {}
    styles = []
    for column in columns or []:
        column_id = _column_id(column)
        if not column_id:
            continue

        cell_class = str(column.get('cellClass') or '')
        is_numeric = (
            column.get('type') == 'numeric'
            or column_id in numeric_columns
            or 'mckinsey-ag-grid-number-cell' in cell_class
        )
        min_width, max_width = width_limits.get(
            column_id,
            default_numeric_limits if is_numeric else default_text_limits
        )
        header_text = _column_header_text(column)
        value_samples = [
            _width_sample_text(row.get(column_id))
            for row in records
            if isinstance(row, dict) and row.get(column_id) not in (None, '')
        ][:300]
        width = _compact_column_width(header_text, value_samples, min_width, max_width, is_numeric=is_numeric)
        styles.append({
            'if': {'column_id': column_id},
            'width': f'{width}px',
            'minWidth': f'{width}px',
            'maxWidth': f'{width}px',
        })
    return styles


def _apply_importer_detail_column_widths_to_defs(column_defs, width_styles):
    width_by_field = {}
    for style in width_styles or []:
        condition = style.get('if', {})
        column_id = condition.get('column_id')
        if not column_id:
            continue
        width_by_field[str(column_id)] = {
            'width': int(str(style.get('width', '0')).replace('px', '') or 0),
            'minWidth': int(str(style.get('minWidth', '0')).replace('px', '') or 0),
            'maxWidth': int(str(style.get('maxWidth', '0')).replace('px', '') or 0),
        }

    def apply_to_defs(definitions):
        for definition in definitions or []:
            children = definition.get('children')
            if children:
                apply_to_defs(children)
                continue
            field = str(definition.get('field') or '')
            widths = width_by_field.get(field)
            if widths:
                definition.update(widths)

    apply_to_defs(column_defs)
    return column_defs


def _importer_maintenance_quarter_bounds(current_year, current_quarter, offset, direction):
    target_q = current_quarter + (offset if direction == 'future' else -offset)
    target_year = current_year
    while target_q <= 0:
        target_q += 4
        target_year -= 1
    while target_q > 4:
        target_q -= 4
        target_year += 1

    q_start_month = (target_q - 1) * 3 + 1
    start = pd.Timestamp(year=target_year, month=q_start_month, day=1)
    end = start + pd.DateOffset(months=3) - pd.DateOffset(days=1)
    label = f"Q{target_q}'{str(target_year)[2:]}"
    return start, end, label


def _importer_maintenance_month_bounds(current_date, offset, direction):
    current_month_start = pd.Timestamp(year=current_date.year, month=current_date.month, day=1)
    if direction == 'future':
        month_date = current_month_start + pd.DateOffset(months=offset - 1)
    else:
        last_month_end = current_month_start - pd.DateOffset(days=1)
        month_date = last_month_end - pd.DateOffset(months=offset - 1)

    start = pd.Timestamp(year=month_date.year, month=month_date.month, day=1)
    end = start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    label = f"{calendar.month_abbr[start.month]}'{str(start.year)[2:]}"
    return start, end, label


def _build_importer_maintenance_period_specs(current_date=None):
    current_date = current_date or pd.Timestamp.now()
    current_year = current_date.year
    current_quarter = current_date.quarter

    specs = []
    for q_offset in range(MAINTENANCE_DEFAULT_QUARTER_COUNT, 0, -1):
        start, end, label = _importer_maintenance_quarter_bounds(
            current_year,
            current_quarter,
            q_offset,
            direction='historical'
        )
        specs.append({
            'id': f'Q-{q_offset}',
            'label': label,
            'family': 'historical-quarter',
            'start': start,
            'end': end,
        })

    for m_offset in range(MAINTENANCE_DEFAULT_MONTH_COUNT, 0, -1):
        start, end, label = _importer_maintenance_month_bounds(
            current_date,
            m_offset,
            direction='historical'
        )
        specs.append({
            'id': f'M-{m_offset}',
            'label': label,
            'family': 'historical-month',
            'start': start,
            'end': end,
        })

    for m_offset in range(1, MAINTENANCE_DEFAULT_MONTH_COUNT + 1):
        start, end, label = _importer_maintenance_month_bounds(
            current_date,
            m_offset,
            direction='future'
        )
        specs.append({
            'id': f'M+{m_offset}',
            'label': label,
            'family': 'current-month' if m_offset == 1 else 'nearterm-month',
            'start': start,
            'end': end,
        })

    for q_offset in range(1, MAINTENANCE_MAX_FORWARD_QUARTER_COUNT + 1):
        start, end, label = _importer_maintenance_quarter_bounds(
            current_year,
            current_quarter,
            q_offset,
            direction='future'
        )
        specs.append({
            'id': f'Q+{q_offset}',
            'label': label,
            'family': 'nearterm-quarter' if q_offset == 1 else 'outlook-quarter',
            'start': start,
            'end': end,
        })

    return specs


def _strip_maintenance_expand_marker(value):
    value = str(value or '').strip()
    if value.startswith(('▶ ', '▼ ', '+ ', '- ')):
        return value[2:].strip()
    return value


def _build_maintenance_numeric_filter_js(operator, threshold_text):
    return (
        "(Number(String(params.value !== null && params.value !== undefined "
        f"&& params.value !== '' ? params.value : '').replace(/[^0-9.\\-]/g, '')) {operator} {threshold_text})"
    )


def _build_maintenance_cell_class_rules():
    return {
        'maintenance-impact-active': {
            'function': _build_maintenance_numeric_filter_js('>', '0')
        },
        'maintenance-impact-watch': {
            'function': _build_maintenance_numeric_filter_js('>=', '1')
        },
        'maintenance-impact-high': {
            'function': _build_maintenance_numeric_filter_js('>=', '5')
        },
    }


def _maintenance_value_js():
    return (
        "Number(String(params.value !== null && params.value !== undefined "
        "&& params.value !== '' ? params.value : '').replace(/[^0-9.\\-]/g, ''))"
    )


def _build_maintenance_cell_style_conditions(family_class):
    value_js = _maintenance_value_js()
    non_total_row = "params.data && params.data.Type !== 'total'"
    active_styles = {
        'historical-quarter': {'backgroundColor': '#dce9f8', 'color': '#1e40af', 'fontWeight': '720'},
        'historical-month': {'backgroundColor': '#dce9f8', 'color': '#1e40af', 'fontWeight': '720'},
        'current-month': {'backgroundColor': '#cfe4ff', 'color': '#1d4ed8', 'fontWeight': '780'},
        'nearterm-month': {'backgroundColor': '#fff1c8', 'color': '#92400e', 'fontWeight': '720'},
        'nearterm-quarter': {'backgroundColor': '#fff1c8', 'color': '#92400e', 'fontWeight': '720'},
        'outlook-quarter': {'backgroundColor': '#e4eaf2', 'color': '#374151', 'fontWeight': '720'},
    }
    watch_styles = {
        'nearterm-month': {'backgroundColor': '#f7dc97', 'color': '#78350f', 'fontWeight': '800'},
        'nearterm-quarter': {'backgroundColor': '#f7dc97', 'color': '#78350f', 'fontWeight': '800'},
        'outlook-quarter': {'backgroundColor': '#d5dde8', 'color': '#1f2937', 'fontWeight': '780'},
    }
    conditions = [
        {
            'condition': f"{non_total_row} && {value_js} >= 5",
            'style': {'backgroundColor': '#f2b5bd', 'color': '#7f1d1d', 'fontWeight': '860'},
        }
    ]
    if family_class in watch_styles:
        conditions.append({
            'condition': f"{non_total_row} && {value_js} >= 1 && {value_js} < 5",
            'style': watch_styles[family_class],
        })
        conditions.append({
            'condition': f"{non_total_row} && {value_js} > 0 && {value_js} < 1",
            'style': active_styles[family_class],
        })
    else:
        conditions.append({
            'condition': f"{non_total_row} && {value_js} > 0 && {value_js} < 5",
            'style': active_styles.get(family_class, {}),
        })
    return {
        'styleConditions': conditions,
        'defaultStyle': {},
    }


def _maintenance_period_family_class(family):
    return str(family or 'period').replace('_', '-')


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

ROUTE_ANALYSIS_ROUTE_ORDER = ['Direct', 'ViaSuez', 'ViaPanama']
ROUTE_ANALYSIS_ROUTE_LABELS = {
    'Direct': 'Direct',
    'ViaSuez': 'Via Suez',
    'ViaPanama': 'Via Panama',
    'Unclassified': 'Unclassified',
}
ROUTE_ANALYSIS_ROUTE_COLORS = {
    'Direct': '#26547c',
    'ViaSuez': '#d58a1f',
    'ViaPanama': '#2f8f5b',
    'Unclassified': '#94a3b8',
}
ROUTE_ANALYSIS_TOTAL_COLOR = '#111827'

def get_professional_colors(n_colors):
    """Get n professional colors, cycling through the palette if needed."""
    colors = []
    for i in range(n_colors):
        colors.append(PROFESSIONAL_CHART_COLORS[i % len(PROFESSIONAL_CHART_COLORS)])
    return colors


def _route_analysis_time_columns(agg_level):
    if agg_level == 'Year':
        return ['year'], 'Year'
    if agg_level == 'Year+Season':
        return ['year', 'season'], 'Season'
    if agg_level == 'Year+Quarter':
        return ['year', 'quarter'], 'Quarter'
    if agg_level == 'Month':
        return ['year', 'month'], 'Month'
    if agg_level == 'Week':
        return ['year', 'week'], 'Week'
    return ['year'], 'Year'


def _route_analysis_period_label_and_sort(row, agg_level):
    year = row.get('year')
    try:
        year_int = int(year)
    except (TypeError, ValueError):
        return 'Unknown', 0

    if agg_level == 'Year':
        return str(year_int), year_int

    if agg_level == 'Year+Quarter':
        quarter = str(row.get('quarter') or '')
        try:
            quarter_num = int(quarter.replace('Q', ''))
        except ValueError:
            quarter_num = 0
        label = f"Q{quarter_num} '{str(year_int)[-2:]}" if quarter_num else str(year_int)
        return label, year_int * 10 + quarter_num

    if agg_level == 'Year+Season':
        season = str(row.get('season') or '')
        season_order = {'S': 1, 'W': 2}.get(season, 0)
        label = f"{season} '{str(year_int)[-2:]}" if season else str(year_int)
        return label, year_int * 10 + season_order

    if agg_level == 'Month':
        try:
            month = int(row.get('month'))
        except (TypeError, ValueError):
            month = 0
        month_label = calendar.month_abbr[month] if 1 <= month <= 12 else ''
        label = f"{month_label} '{str(year_int)[-2:]}" if month_label else str(year_int)
        return label, year_int * 100 + month

    if agg_level == 'Week':
        try:
            week = int(row.get('week'))
        except (TypeError, ValueError):
            week = 0
        label = f"W{week:02d} '{str(year_int)[-2:]}" if week else str(year_int)
        return label, year_int * 100 + week

    return str(year_int), year_int


def _route_analysis_tick_values(period_labels, max_ticks=8):
    labels = [label for label in period_labels if label]
    if len(labels) <= max_ticks:
        return labels

    step = max(1, int(np.ceil(len(labels) / max_ticks)))
    tick_values = labels[::step]
    if labels[-1] not in tick_values:
        tick_values.append(labels[-1])
    return tick_values


def _build_route_analysis_panel_frame(df, agg_level):
    index_cols, _ = _route_analysis_time_columns(agg_level)
    required_cols = index_cols + ['selected_route', 'voyage_id']
    if df is None or df.empty or any(col not in df.columns for col in required_cols):
        return pd.DataFrame(), []

    route_df = df[required_cols].copy()
    route_df['selected_route'] = (
        route_df['selected_route']
        .fillna('Unclassified')
        .replace('', 'Unclassified')
    )

    grouped = (
        route_df
        .groupby(index_cols + ['selected_route'], observed=True)['voyage_id']
        .count()
        .unstack(fill_value=0)
    )
    if grouped.empty:
        return pd.DataFrame(), []

    route_columns = [
        route for route in ROUTE_ANALYSIS_ROUTE_ORDER
        if route in grouped.columns
    ] + sorted(
        route for route in grouped.columns
        if route not in ROUTE_ANALYSIS_ROUTE_ORDER
    )

    frame = grouped.reset_index()
    labels_and_sorts = frame.apply(
        lambda row: _route_analysis_period_label_and_sort(row, agg_level),
        axis=1
    )
    frame['period_label'] = [item[0] for item in labels_and_sorts]
    frame['period_sort'] = [item[1] for item in labels_and_sorts]
    frame = frame.sort_values('period_sort', kind='mergesort').reset_index(drop=True)

    frame['total_voyages'] = frame[route_columns].sum(axis=1)
    for route in route_columns:
        frame[f'{route}_pct'] = np.where(
            frame['total_voyages'] > 0,
            frame[route] / frame['total_voyages'] * 100,
            0
        )

    return frame, route_columns


def _empty_route_analysis_figure(message, detail=None):
    subtitle = f"<br><span style='font-size:12px;color:#64748b'>{detail}</span>" if detail else ""
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=f"<b>{message}</b>{subtitle}",
            x=0.02,
            xanchor='left',
            font=dict(size=15, color=PROFESSIONAL_COLORS['text_primary'])
        ),
        height=360,
        paper_bgcolor=PROFESSIONAL_COLORS['bg_white'],
        plot_bgcolor=PROFESSIONAL_COLORS['bg_white'],
        xaxis={'visible': False},
        yaxis={'visible': False},
        margin=dict(l=24, r=24, t=72, b=24),
    )
    return fig


def _route_analysis_year_ago_sort_value(latest_row, agg_level):
    sort_value = latest_row.get('period_sort')
    if pd.isna(sort_value):
        return None
    if agg_level == 'Year':
        return sort_value - 1
    if agg_level in {'Year+Quarter', 'Year+Season'}:
        return sort_value - 10
    if agg_level in {'Month', 'Week'}:
        return sort_value - 100
    return sort_value - 1


def _route_analysis_metric_delta(current_value, comparison_row, metric_getter):
    if comparison_row is None:
        return None
    try:
        comparison_value = metric_getter(comparison_row)
    except Exception:
        return None
    if pd.isna(current_value) or pd.isna(comparison_value):
        return None
    return current_value - comparison_value


def _format_route_analysis_number(value, decimals=0, suffix=''):
    if value is None or pd.isna(value):
        return 'n/a'
    return f"{value:,.{decimals}f}{suffix}"


def _format_route_analysis_delta(value, decimals=0, suffix=''):
    if value is None or pd.isna(value):
        return 'n/a'
    sign = '+' if value > 0 else ''
    return f"{sign}{value:,.{decimals}f}{suffix}"


def _route_analysis_delta_tone(value):
    if value is None or pd.isna(value) or abs(value) < 1e-9:
        return 'flat'
    return 'up' if value > 0 else 'down'


def _route_analysis_delta_cell(value, decimals=0, suffix=''):
    return html.Span(
        _format_route_analysis_delta(value, decimals=decimals, suffix=suffix),
        className=f'route-kpi-delta-cell route-kpi-delta-cell-{_route_analysis_delta_tone(value)}'
    )


def _route_analysis_metric_row(label, value_text, previous_delta, year_ago_delta, decimals=0, suffix=''):
    return html.Div(
        [
            html.Span(label, className='route-kpi-matrix-metric'),
            html.Span(value_text, className='route-kpi-matrix-value'),
            _route_analysis_delta_cell(previous_delta, decimals=decimals, suffix=suffix),
            _route_analysis_delta_cell(year_ago_delta, decimals=decimals, suffix=suffix),
        ],
        className='route-kpi-matrix-row'
    )


def _route_analysis_share_value(row, routes):
    return sum(float(row.get(f'{route}_pct', 0) or 0) for route in routes)


def _route_analysis_legacy_hidden_figure():
    fig = go.Figure()
    fig.update_layout(
        height=10,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis={'visible': False},
        yaxis={'visible': False},
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def _route_analysis_signal_text(total_delta_prev, direct_delta_prev, canal_delta_prev, canal_label):
    signals = []
    if total_delta_prev is not None and not pd.isna(total_delta_prev) and abs(total_delta_prev) >= 1:
        signals.append(f"Vol {_format_route_analysis_delta(total_delta_prev, decimals=0)}")

    mix_moves = [
        ('Direct', direct_delta_prev),
        (canal_label.replace(' share', ''), canal_delta_prev),
    ]
    mix_moves = [
        (label, value)
        for label, value in mix_moves
        if value is not None and not pd.isna(value)
    ]
    material_mix_moves = [
        (label, value)
        for label, value in mix_moves
        if abs(value) >= 1
    ]
    if material_mix_moves:
        label, value = max(material_mix_moves, key=lambda item: abs(item[1]))
        signals.append(f"{label} {_format_route_analysis_delta(value, decimals=1, suffix='pp')}")

    if not signals:
        return "Stable volume and route mix vs previous period"
    return " | ".join(signals[:2])


def _route_analysis_signal_class(direct_delta_prev, canal_delta_prev, total_delta_prev=None):
    ranked = [
        value for value in [direct_delta_prev, canal_delta_prev]
        if value is not None and not pd.isna(value)
    ]
    material = [value for value in ranked if abs(value) >= 1]
    if material:
        strongest = max(material, key=lambda value: abs(value))
        return f'route-kpi-signal route-kpi-signal-{_route_analysis_delta_tone(strongest)}'
    if total_delta_prev is not None and not pd.isna(total_delta_prev) and abs(total_delta_prev) >= 1:
        return f'route-kpi-signal route-kpi-signal-{_route_analysis_delta_tone(total_delta_prev)}'
    return 'route-kpi-signal route-kpi-signal-flat'


def _route_analysis_card_legend(routes):
    legend_routes = list(routes or [])
    if 'Direct' in legend_routes:
        legend_routes = ['Direct'] + [route for route in legend_routes if route != 'Direct']

    items = []
    for route in legend_routes:
        label = ROUTE_ANALYSIS_ROUTE_LABELS.get(route, str(route)).replace('Via ', '')
        route_class = str(route).lower().replace('via', 'via-')
        items.append(
            html.Span(
                [
                    html.Span(
                        className=f'route-kpi-legend-swatch route-kpi-legend-swatch-{route_class}'
                    ),
                    html.Span(label, className='route-kpi-legend-label'),
                ],
                className='route-kpi-legend-item'
            )
        )

    items.append(
        html.Span(
            [
                html.Span(className='route-kpi-legend-line'),
                html.Span('Vol', className='route-kpi-legend-label'),
            ],
            className='route-kpi-legend-item route-kpi-legend-volume'
        )
    )
    return html.Div(items, className='route-kpi-card-legend')


def _build_route_analysis_card_figure(scenario):
    frame = scenario.get('frame')
    routes = scenario.get('routes') or []
    fig = go.Figure()

    if frame is None or frame.empty:
        fig.add_annotation(
            text='No voyages',
            x=0.5,
            y=0.5,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(size=11, color=PROFESSIONAL_COLORS['text_tertiary'])
        )
    else:
        x_values = frame['period_label'].tolist()
        tick_values = _route_analysis_tick_values(x_values, max_ticks=4)
        fallback_colors = get_professional_colors(12)

        for route_idx, route in enumerate(routes):
            route_label = ROUTE_ANALYSIS_ROUTE_LABELS.get(route, str(route))
            route_color = ROUTE_ANALYSIS_ROUTE_COLORS.get(
                route,
                fallback_colors[route_idx % len(fallback_colors)]
            )
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=frame[f'{route}_pct'],
                    name=route_label,
                    marker=dict(
                        color=route_color,
                        line=dict(color='rgba(255,255,255,0.62)', width=0.35),
                    ),
                    customdata=np.stack(
                        [frame[route].to_numpy(), frame['total_voyages'].to_numpy()],
                        axis=-1
                    ),
                    hovertemplate=(
                        "Period: %{x}<br>"
                        f"{route_label}: %{{y:.1f}}%<br>"
                        "Voyages: %{customdata[0]:,.0f} of %{customdata[1]:,.0f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        max_total_voyages = frame['total_voyages'].max()
        indexed_total_voyages = (
            frame['total_voyages'] / max_total_voyages * 100
            if max_total_voyages
            else frame['total_voyages']
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=indexed_total_voyages,
                name='Total voyages',
                mode='lines+markers',
                line=dict(color=ROUTE_ANALYSIS_TOTAL_COLOR, width=1.9, shape='linear'),
                marker=dict(
                    size=4.8,
                    color=ROUTE_ANALYSIS_TOTAL_COLOR,
                    line=dict(color='white', width=0.9),
                ),
                customdata=frame['total_voyages'],
                hovertemplate=(
                    "Period: %{x}<br>"
                    "Total voyages: %{customdata:,.0f}<br>"
                    "Indexed line: %{y:.0f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.update_xaxes(
            tickmode='array',
            tickvals=tick_values,
            ticktext=tick_values,
            tickangle=0,
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=8.6, color=PROFESSIONAL_COLORS['text_tertiary']),
        )
        fig.update_yaxes(
            title_text=None,
            range=[0, 100],
            tickvals=[0, 50, 100],
            ticksuffix='%',
            gridcolor='rgba(148, 163, 184, 0.18)',
            zeroline=False,
            showticklabels=False,
        )

    fig.update_layout(
        barmode='stack',
        bargap=0.1,
        height=212,
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,255,255,0)',
        font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=10,
            color=PROFESSIONAL_COLORS['text_secondary']
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='rgba(15, 23, 42, 0.18)',
            font=dict(size=10, color=PROFESSIONAL_COLORS['text_primary'])
        ),
        margin=dict(l=0, r=0, t=4, b=20),
        showlegend=False,
    )
    fig.update_traces(cliponaxis=False)
    return fig


def _route_analysis_card(scenario, agg_level):
    frame = scenario.get('frame')
    if frame is None or frame.empty:
        return html.Div(
            [
                html.Div(scenario.get('display_title', 'Route bucket'), className='route-kpi-title'),
                html.Div('No delivered voyages', className='route-kpi-empty')
            ],
            className='route-kpi-card route-kpi-card-empty'
        )

    latest_row = frame.iloc[-1]
    previous_row = frame.iloc[-2] if len(frame) > 1 else None
    year_ago_sort = _route_analysis_year_ago_sort_value(latest_row, agg_level)
    year_ago_matches = frame[frame['period_sort'] == year_ago_sort] if year_ago_sort is not None else pd.DataFrame()
    year_ago_row = year_ago_matches.iloc[-1] if not year_ago_matches.empty else None

    total_voyages = float(latest_row.get('total_voyages', 0) or 0)
    direct_share = float(latest_row.get('Direct_pct', 0) or 0)
    canal_routes = scenario.get('canal_routes') or []
    canal_share = _route_analysis_share_value(latest_row, canal_routes)

    total_delta_prev = _route_analysis_metric_delta(
        total_voyages,
        previous_row,
        lambda row: float(row.get('total_voyages', 0) or 0)
    )
    total_delta_yoy = _route_analysis_metric_delta(
        total_voyages,
        year_ago_row,
        lambda row: float(row.get('total_voyages', 0) or 0)
    )
    direct_delta_prev = _route_analysis_metric_delta(
        direct_share,
        previous_row,
        lambda row: float(row.get('Direct_pct', 0) or 0)
    )
    direct_delta_yoy = _route_analysis_metric_delta(
        direct_share,
        year_ago_row,
        lambda row: float(row.get('Direct_pct', 0) or 0)
    )
    canal_label = scenario.get('canal_label', 'Canal share')
    canal_delta_prev = _route_analysis_metric_delta(
        canal_share,
        previous_row,
        lambda row: _route_analysis_share_value(row, canal_routes)
    )
    canal_delta_yoy = _route_analysis_metric_delta(
        canal_share,
        year_ago_row,
        lambda row: _route_analysis_share_value(row, canal_routes)
    )
    signal_text = _route_analysis_signal_text(
        total_delta_prev,
        direct_delta_prev,
        canal_delta_prev,
        canal_label
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(scenario.get('display_title', 'Route bucket'), className='route-kpi-title'),
                    html.Span(str(latest_row.get('period_label', 'Latest')), className='route-kpi-period')
                ],
                className='route-kpi-card-header'
            ),
            html.Div(
                [
                    html.Span('Metric', className='route-kpi-table-heading'),
                    html.Span('Latest', className='route-kpi-table-heading route-kpi-table-heading-right'),
                    html.Span('vs prev', className='route-kpi-table-heading route-kpi-table-heading-right'),
                    html.Span('Y/Y', className='route-kpi-table-heading route-kpi-table-heading-right'),
                ],
                className='route-kpi-matrix-row route-kpi-matrix-head'
            ),
            _route_analysis_metric_row(
                'Voyages',
                _format_route_analysis_number(total_voyages, decimals=0),
                total_delta_prev,
                total_delta_yoy,
                decimals=0
            ),
            _route_analysis_metric_row(
                'Direct',
                _format_route_analysis_number(direct_share, decimals=1, suffix='%'),
                direct_delta_prev,
                direct_delta_yoy,
                decimals=1,
                suffix=' pp'
            ),
            _route_analysis_metric_row(
                canal_label.replace(' share', ''),
                _format_route_analysis_number(canal_share, decimals=1, suffix='%'),
                canal_delta_prev,
                canal_delta_yoy,
                decimals=1,
                suffix=' pp'
            ),
            html.Div(
                [
                    html.Span('Signal', className='route-kpi-signal-label'),
                    html.Span(signal_text, className='route-kpi-signal-text'),
                ],
                className=_route_analysis_signal_class(direct_delta_prev, canal_delta_prev, total_delta_prev)
            ),
            _route_analysis_card_legend(scenario.get('routes')),
            dcc.Graph(
                figure=_build_route_analysis_card_figure(scenario),
                config={'displayModeBar': False, 'responsive': True},
                className='route-kpi-card-graph'
            ),
        ],
        className='route-kpi-card route-kpi-card-with-chart'
    )


def _build_route_analysis_kpi_cards(scenarios, agg_level):
    return html.Div(
        [_route_analysis_card(scenario, agg_level) for scenario in scenarios],
        className='route-kpi-grid'
    )


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

    mapping_columns = ['country', 'country_name', *[
        column for column in DESTINATION_AGGREGATION_LABELS
        if column != 'country'
    ]]
    mapping_df = pd.read_sql(
        text(f"""
            SELECT {', '.join(mapping_columns)}
            FROM {DB_SCHEMA}.mappings_country
        """),
        engine
    )
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

DEFAULT_IMPORTER_ORIGIN_LEVEL = 'origin_classification_level1'
IMPORTER_ORIGIN_LEVEL_OPTIONS = [
    {'label': 'Classification Level 1', 'value': DEFAULT_IMPORTER_ORIGIN_LEVEL},
    {'label': 'Country', 'value': 'origin_country_name'},
    {'label': 'Basin', 'value': 'origin_basin'},
]

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


def _fetch_importer_scoped_trades(engine, destination_countries, min_end_date=None,
                                  delivered_only=False, include_destination_context=False,
                                  mapping_lookup_df=None):
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
    if include_destination_context:
        expected_columns.append('destination_country_name')
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
    query = text(f"""
        SELECT
            kt."end"::date AS end_date,
            COALESCE(kt.cargo_destination_cubic_meters, 0) * {MCM_PER_CUBIC_METER} AS cargo_mcm,
            COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown') AS origin_country,
            COALESCE(NULLIF(BTRIM(kt.continent_origin_name), ''), 'Unknown') AS origin_continent_chart
            {", COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown') AS destination_country_name" if include_destination_context else ""}
        FROM {DB_SCHEMA}.kpler_trades kt
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
    if include_destination_context:
        scoped_trades_df['destination_country_name'] = _normalize_scope_series(
            scoped_trades_df['destination_country_name'],
            default='Unknown'
        )

    if mapping_lookup_df is None:
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

    scope_column = IMPORTER_ORIGIN_LEVEL_TO_SCOPE.get(
        origin_level or DEFAULT_IMPORTER_ORIGIN_LEVEL,
        IMPORTER_ORIGIN_LEVEL_TO_SCOPE[DEFAULT_IMPORTER_ORIGIN_LEVEL]
    )
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


def _build_importer_rolling_windows_pivot(summary_scope_df, rolling_window_days=30, current_date=None,
                                          include_comparison_reference_columns=False):
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
    if include_comparison_reference_columns:
        date_7d_pp_start = date_7d_ago - pd.Timedelta(days=7)
        date_window_pp_start = date_window_ago - pd.Timedelta(days=normalized_window_days)
        relevant_start = min(date_window_y1_start, date_7d_pp_start, date_window_pp_start)
    else:
        relevant_start = date_window_y1_start

    relevant_df = summary_scope_df[
        (summary_scope_df['end_date'] >= relevant_start) &
        (summary_scope_df['end_date'] <= reference_date)
    ].copy()
    if relevant_df.empty:
        return pd.DataFrame(columns=expected_columns)

    all_combinations_df = relevant_df[['continent', 'country']].drop_duplicates().reset_index(drop=True)
    if all_combinations_df.empty:
        return pd.DataFrame(columns=expected_columns)

    all_combinations = pd.MultiIndex.from_frame(all_combinations_df)
    full_date_index = pd.date_range(relevant_start + pd.Timedelta(days=1), reference_date, freq='D')

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

    rolling_series = [
        avg_7d.rename('7D'),
        avg_window.rename(rolling_window_label),
        avg_window_y1.rename(f'{rolling_window_label}_Y1'),
    ]
    if include_comparison_reference_columns:
        avg_7d_pp = daily_pivot.loc[
            date_7d_pp_start + pd.Timedelta(days=1):date_7d_ago
        ].mean()
        avg_window_pp = daily_pivot.loc[
            date_window_pp_start + pd.Timedelta(days=1):date_window_ago
        ].mean()
        avg_7d_y1 = daily_pivot.loc[
            date_window_y1_end - pd.Timedelta(days=6):date_window_y1_end
        ].mean()
        rolling_series.extend([
            avg_7d_pp.rename('7D_PP'),
            avg_window_pp.rename(f'{rolling_window_label}_PP'),
            avg_7d_y1.rename('7D_Y1'),
        ])

    rolling_df = pd.concat(rolling_series, axis=1).reset_index()
    rolling_df.columns = [
        'continent',
        'country',
        *[series.name for series in rolling_series],
    ]
    rolling_df[f'Δ 7D-{rolling_window_label}'] = rolling_df['7D'] - rolling_df[rolling_window_label]
    rolling_df[f'Δ {rolling_window_label} Y/Y'] = (
        rolling_df[rolling_window_label] - rolling_df[f'{rolling_window_label}_Y1']
    )
    return rolling_df


def _build_importer_chart_date_index(start_date=None, forecast_days=14, current_date=None):
    reference_date = pd.Timestamp(current_date or dt.date.today()).normalize()
    chart_start_date = pd.Timestamp(start_date or DETAIL_CHART_DATA_START_DATE).normalize()
    chart_end_date = reference_date + pd.Timedelta(days=forecast_days)
    return pd.date_range(chart_start_date, chart_end_date, freq='D'), reference_date


def _detail_chart_years(current_date=None):
    date_index, _ = _build_importer_chart_date_index(
        start_date=DETAIL_CHART_DATA_START_DATE,
        current_date=current_date
    )
    date_index = date_index[date_index >= pd.Timestamp(DETAIL_CHART_DATA_START_DATE)]
    return sorted(pd.Index(date_index.year.astype(int)).unique().tolist())


def _build_importer_total_import_df(scoped_trades_df, rolling_window_days=30, current_date=None,
                                    chart_start_date=None, display_start_date=None):
    if scoped_trades_df is None or scoped_trades_df.empty:
        return pd.DataFrame(columns=['date', 'year', 'day_of_year', 'month_day', 'rolling_avg', 'is_forecast'])

    date_index, reference_date = _build_importer_chart_date_index(
        start_date=chart_start_date,
        current_date=current_date
    )
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
    display_start = pd.Timestamp(display_start_date or DETAIL_CHART_DATA_START_DATE)
    return result_df[result_df['date'] >= display_start].reset_index(drop=True)


def _build_importer_continent_chart_df(scoped_trades_df, rolling_window_days=30, current_date=None,
                                       include_percentage=False, chart_start_date=None, display_start_date=None):
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

    date_index, reference_date = _build_importer_chart_date_index(
        start_date=chart_start_date,
        current_date=current_date
    )
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

    display_start = pd.Timestamp(display_start_date or DETAIL_CHART_DATA_START_DATE)
    melted_df = melted_df[melted_df['date'] >= display_start].reset_index(drop=True)
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

    normalized_destination_countries = normalize_destination_countries(destination_countries)
    if not normalized_destination_countries:
        return pd.DataFrame()

    try:
        # Load trades filtered by destination countries (importer selection)
        try:
            query = text(f'''
                SELECT
                    kt.voyage_id,
                    kt."end",
                    kt.origin_country_name,
                    kt.destination_country_name,
                    kt.mileage_nautical_miles,
                    dm."distanceDirect",
                    dm."distanceViaSuez",
                    dm."distanceViaPanama"
                FROM {DB_SCHEMA}.{trades_table_name}
                kt
                LEFT JOIN {DB_SCHEMA}.{distance_table_name} dm
                    ON kt.zone_origin_name = dm."originLocationName"
                    AND kt.zone_destination_name = dm."destinationLocationName"
                WHERE kt.status = 'Delivered'
                    AND kt.destination_country_name IN :destination_countries
                    AND kt.zone_origin_name <> kt.zone_destination_name
                    AND kt.upload_timestamp_utc = (
                        SELECT MAX(upload_timestamp_utc)
                        FROM {DB_SCHEMA}.{trades_table_name}
                    )
            ''')
            final_df = pd.read_sql(
                query,
                engine,
                params={'destination_countries': normalized_destination_countries}
            )
        except Exception:
            return None

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

    except Exception:
        return None

    return final_df


def _build_detail_year_selector(selector_id, class_prefix='supply'):
    return html.Div(
        [
            html.Div('Years', className=f'{class_prefix}-year-legend-title'),
            dcc.Checklist(
                id=selector_id,
                options=[],
                value=[],
                inline=True,
                className=f'{class_prefix}-year-checklist',
                inputStyle=DETAIL_CHIP_INPUT_STYLE,
                labelStyle={'marginRight': '0'},
            ),
        ],
        className=f'{class_prefix}-year-legend'
    )


def _build_importer_detail_period_controls():
    return html.Div(
        [
            html.Div(
                [
                    html.Div('Comparison', className='supply-dest-control-label importer-period-control-label'),
                    dcc.RadioItems(
                        id='imp-period-comparison-basis',
                        options=DETAIL_PERIOD_COMPARISON_BASIS_OPTIONS,
                        value='levels',
                        inline=True,
                        className=(
                            'supply-dest-view-selector supply-dest-comparison-selector '
                            'importer-period-view-selector importer-period-comparison-selector'
                        ),
                        inputStyle={'display': 'none'},
                        labelStyle={'marginRight': '0'},
                    ),
                ],
                className=(
                    'supply-dest-control-group supply-dest-comparison-control '
                    'importer-period-control-group importer-period-comparison-control-group'
                )
            ),
            html.Div(
                [
                    html.Div('Periods', className='supply-dest-control-label importer-period-control-label'),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span('Qtrs', className='supply-dest-mini-control-label importer-period-mini-label'),
                                    dcc.Dropdown(
                                        id='imp-period-quarter-count-dropdown',
                                        options=_detail_count_options(DETAIL_MAX_QUARTER_COUNT),
                                        value=DETAIL_DEFAULT_QUARTER_COUNT,
                                        clearable=False,
                                        searchable=False,
                                        className=(
                                            'supply-dest-count-dropdown importer-period-count-dropdown'
                                        ),
                                    ),
                                ],
                                className='supply-dest-count-selector importer-period-count-selector'
                            ),
                            html.Div(
                                [
                                    html.Span('Months', className='supply-dest-mini-control-label importer-period-mini-label'),
                                    dcc.Dropdown(
                                        id='imp-period-month-count-dropdown',
                                        options=_detail_count_options(DETAIL_MAX_MONTH_COUNT),
                                        value=DETAIL_DEFAULT_MONTH_COUNT,
                                        clearable=False,
                                        searchable=False,
                                        className=(
                                            'supply-dest-count-dropdown importer-period-count-dropdown'
                                        ),
                                    ),
                                ],
                                className='supply-dest-count-selector importer-period-count-selector'
                            ),
                            html.Div(
                                [
                                    html.Span('Weeks', className='supply-dest-mini-control-label importer-period-mini-label'),
                                    dcc.Dropdown(
                                        id='imp-period-week-count-dropdown',
                                        options=_detail_count_options(DETAIL_MAX_WEEK_COUNT),
                                        value=DETAIL_DEFAULT_WEEK_COUNT,
                                        clearable=False,
                                        searchable=False,
                                        className=(
                                            'supply-dest-count-dropdown importer-period-count-dropdown'
                                        ),
                                    ),
                                ],
                                className='supply-dest-count-selector importer-period-count-selector'
                            ),
                        ],
                        className='supply-dest-period-count-selectors importer-period-count-selectors'
                    ),
                ],
                className=(
                    'supply-dest-control-group supply-dest-period-count-control importer-period-control-group '
                    'importer-period-count-control-group'
                )
            ),
        ],
        className='supply-dest-controls exporter-detail-period-controls importer-period-controls'
    )


def _detail_year_options(years):
    return [
        {'label': html.Span(str(year), className='supply-year-chip-text'), 'value': str(year)}
        for year in years
    ]


def _default_detail_year_values(years):
    selected_years = [
        str(year)
        for year in years
        if int(year) >= DETAIL_DEFAULT_VISIBLE_START_YEAR
    ]
    return selected_years or [str(years[-1])] if years else []


def _normalize_selected_years(selected_years):
    return {str(year) for year in (selected_years or [])}


def _filter_df_years(df, selected_years):
    selected = _normalize_selected_years(selected_years)
    if df is None or df.empty or not selected or 'year' not in df.columns:
        return df
    return df[df['year'].astype(str).isin(selected)].copy()


def _format_detail_metric_value(value, label):
    if value is None or pd.isna(value):
        return None
    return f"{float(value):,.0f} {label}"


def _calculate_latest_detail_metrics(df, value_col='rolling_avg', value_factor=1.0):
    if df is None or df.empty or value_col not in df.columns:
        return {}

    working = df.copy()
    working['date'] = pd.to_datetime(working['date'], errors='coerce').dt.normalize()
    working[value_col] = pd.to_numeric(working[value_col], errors='coerce')
    if 'is_forecast' in working.columns:
        working = working[~working['is_forecast'].fillna(False)].copy()
    working = working[working['date'].notna() & working[value_col].notna()]
    if working.empty:
        return {}

    by_date = working.groupby('date', as_index=False)[value_col].sum().sort_values('date')
    if by_date.empty:
        return {}

    current_row = by_date.iloc[-1]
    current_date = current_row['date']
    current_value = float(current_row[value_col]) * value_factor

    previous_target = current_date - pd.DateOffset(months=1)
    previous_df = by_date[by_date['date'] <= previous_target]
    previous_value = float(previous_df.iloc[-1][value_col]) * value_factor if not previous_df.empty else None

    yoy_target = current_date - pd.DateOffset(years=1)
    yoy_df = by_date[by_date['date'] <= yoy_target]
    yoy_value = float(yoy_df.iloc[-1][value_col]) * value_factor if not yoy_df.empty else None

    def delta_payload(reference_value):
        if reference_value is None:
            return {'delta': None, 'pct': None}
        delta = current_value - reference_value
        pct = (delta / abs(reference_value) * 100) if reference_value else None
        return {'delta': delta, 'pct': pct}

    return {
        'current_value': current_value,
        'mom': delta_payload(previous_value),
        'previous_year': delta_payload(yoy_value),
    }


def _build_detail_delta_pill(label, delta_payload, unit_label):
    if not delta_payload or delta_payload.get('delta') is None:
        return html.Span(f'{label} n/a', className='supply-rolling-delta-pill supply-rolling-delta-neutral')

    delta = delta_payload.get('delta')
    pct = delta_payload.get('pct')
    direction_class = (
        'supply-rolling-delta-positive'
        if delta > 0
        else 'supply-rolling-delta-negative'
        if delta < 0
        else 'supply-rolling-delta-neutral'
    )
    pct_text = f" ({pct:+.0f}%)" if pct is not None and pd.notna(pct) else ''
    return html.Span(
        [
            html.Span(label, className='supply-rolling-delta-label'),
            html.Span(f"{delta:+,.0f} {unit_label}{pct_text}"),
        ],
        className=f'supply-rolling-delta-pill {direction_class}'
    )


def _build_detail_delta_group(metrics, unit_label):
    return html.Div(
        [
            _build_detail_delta_pill('MoM', metrics.get('mom'), unit_label),
            _build_detail_delta_pill('YoY', metrics.get('previous_year'), unit_label),
        ],
        className='supply-rolling-delta-group'
    )


def _import_analysis_store_payload(selected_destination_aggregation, selected_destination_value,
                                   destination_context, scoped_trades_df=None, error=None):
    destination_context = destination_context or {}
    return {
        'selected_destination_aggregation': selected_destination_aggregation,
        'selected_destination_value': selected_destination_value,
        'destination_label': destination_context.get('display_label') or selected_destination_value,
        'destination_countries': list(destination_context.get('destination_countries') or []),
        'scoped_trades': _store_dataframe(scoped_trades_df) if scoped_trades_df is not None else None,
        'loaded_at': dt.datetime.now().isoformat(timespec='seconds'),
        'error': error,
    }


def _resolve_import_analysis_base_data(base_data):
    if not base_data:
        return pd.DataFrame(), {
            'display_label': None,
            'destination_countries': tuple(),
        }, None, None, None

    scoped_trades_df = _load_store_dataframe(
        base_data,
        'scoped_trades',
        date_columns=['end_date']
    )
    destination_context = {
        'display_label': base_data.get('destination_label'),
        'destination_countries': tuple(base_data.get('destination_countries') or []),
    }
    return (
        scoped_trades_df,
        destination_context,
        base_data.get('selected_destination_aggregation'),
        base_data.get('selected_destination_value'),
        base_data.get('error'),
    )


def _import_analysis_base_data_matches(base_data, destination_context,
                                       selected_destination_aggregation, selected_destination):
    if not base_data or base_data.get('error'):
        return False

    stored_countries = tuple(base_data.get('destination_countries') or [])
    target_countries = tuple(destination_context.get('destination_countries') or [])
    return (
        base_data.get('selected_destination_aggregation') == selected_destination_aggregation
        and base_data.get('selected_destination_value') == selected_destination
        and stored_countries == target_countries
    )


def _build_importer_origin_mix_table(continent_df, volume_metric='mcm_d', rolling_window_days=30):
    vol_info = get_volume_metric_info(volume_metric)
    vol_factor = _get_detail_volume_metric_factor(
        volume_metric,
        period_days=normalize_rolling_window_days(rolling_window_days)
    )

    def empty_state():
        return html.Div(
            'No realized origin mix available',
        className='exporter-detail-continent-mix-empty'
        )

    def direction_class(value):
        if value is None or pd.isna(value):
            return 'continent-kpi-delta-neutral'
        if value > 0:
            return 'continent-kpi-delta-positive'
        if value < 0:
            return 'continent-kpi-delta-negative'
        return 'continent-kpi-delta-neutral'

    def format_volume(value, is_delta=False):
        if value is None or pd.isna(value):
            return 'n/a'
        rounded_value = int(round(float(value)))
        sign = '+' if is_delta and rounded_value > 0 else ''
        return f'{sign}{rounded_value:,}'

    def format_share(value, is_delta=False):
        if value is None or pd.isna(value):
            return 'n/a'
        rounded_value = int(round(float(value)))
        sign = '+' if is_delta and rounded_value > 0 else ''
        suffix = 'pp' if is_delta else '%'
        return f'{sign}{rounded_value}{suffix}'

    def format_delta_pct(delta_value, reference_value):
        if (
            delta_value is None
            or reference_value is None
            or pd.isna(delta_value)
            or pd.isna(reference_value)
            or abs(reference_value) < 0.5
        ):
            return None
        rounded_pct = int(round(float(delta_value) / abs(float(reference_value)) * 100))
        sign = '+' if rounded_pct > 0 else ''
        return f'({sign}{rounded_pct}%)'

    def value_cell(value_text, extra_class=''):
        return html.Td(
            html.Span(value_text, className='continent-kpi-summary-value'),
            className=(
                'continent-kpi-summary-cell continent-kpi-summary-value-cell '
                f'continent-kpi-summary-current-cell {extra_class}'
            ).strip()
        )

    def delta_cell(delta_text, delta_value, role_class, pct_text=None, is_available=True):
        if not is_available or delta_text in (None, 'n/a'):
            return html.Td(
                html.Span('-', className='continent-kpi-summary-empty-value'),
                className=(
                    'continent-kpi-summary-cell continent-kpi-summary-delta-cell '
                    f'continent-kpi-summary-cell-empty {role_class}'
                ).strip()
            )

        return html.Td(
            html.Span(
                [
                    html.Span(delta_text, className='continent-kpi-summary-delta-main'),
                    html.Span(pct_text, className='continent-kpi-summary-delta-pct') if pct_text else None
                ],
                className='continent-kpi-summary-delta-stack'
            ),
            className=(
                'continent-kpi-summary-cell continent-kpi-summary-delta-cell '
                f'{role_class} {direction_class(delta_value)}'
            ).strip(),
            title=delta_text
        )

    if continent_df is None or continent_df.empty:
        return empty_state()

    working = continent_df.copy()
    working['date'] = pd.to_datetime(
        working['date'] if 'date' in working.columns else pd.Series(pd.NaT, index=working.index),
        errors='coerce'
    ).dt.normalize()
    working['rolling_avg'] = pd.to_numeric(
        working['rolling_avg'] if 'rolling_avg' in working.columns else pd.Series(np.nan, index=working.index),
        errors='coerce'
    )
    if 'continent_origin' not in working.columns:
        working['continent_origin'] = 'Unknown'
    working['continent_origin'] = (
        working['continent_origin']
        .replace('', np.nan)
        .fillna('Unknown')
        .astype(str)
    )
    if 'is_forecast' in working.columns:
        working = working[~working['is_forecast'].fillna(False).astype(bool)].copy()
    working = working[working['date'].notna() & working['rolling_avg'].notna()].copy()
    if working.empty:
        return empty_state()

    dated_summary = (
        working
        .groupby(['date', 'continent_origin'], as_index=False)['rolling_avg']
        .sum(min_count=1)
    )
    date_totals = (
        dated_summary
        .groupby('date', as_index=False)['rolling_avg']
        .sum(min_count=1)
    )
    valid_dates = date_totals[date_totals['rolling_avg'].fillna(0) > 0]['date']
    if valid_dates.empty:
        return empty_state()

    def build_snapshot(target_date):
        candidate_dates = date_totals[
            (date_totals['date'] <= target_date)
            & (date_totals['rolling_avg'].fillna(0) > 0)
        ]['date']
        if candidate_dates.empty:
            return None, {}

        snapshot_date = candidate_dates.max()
        snapshot_df = dated_summary[
            (dated_summary['date'] == snapshot_date) & dated_summary['rolling_avg'].notna()
        ].copy()
        snapshot_df['volume'] = snapshot_df['rolling_avg'] * vol_factor
        total_snapshot_volume = snapshot_df['volume'].sum()
        if pd.isna(total_snapshot_volume) or total_snapshot_volume <= 0:
            return None, {}

        snapshot_df['share'] = snapshot_df['volume'] / total_snapshot_volume * 100
        snapshot_map = {
            row.continent_origin: {
                'volume': float(row.volume),
                'share': float(row.share)
            }
            for row in snapshot_df.itertuples(index=False)
        }
        return snapshot_date, snapshot_map

    latest_date = valid_dates.max()
    latest_date, current_snapshot = build_snapshot(latest_date)
    if not current_snapshot:
        return empty_state()
    _, mom_snapshot = build_snapshot(latest_date - pd.DateOffset(months=1))
    _, yoy_snapshot = build_snapshot(latest_date - pd.DateOffset(years=1))

    latest_summary = pd.DataFrame([
        {
            'continent_origin': continent,
            'volume': values['volume'],
            'share': values['share']
        }
        for continent, values in current_snapshot.items()
        if values['volume'] > 0
    ])
    if latest_summary.empty:
        return empty_state()

    latest_summary['_unknown_sort'] = latest_summary['continent_origin'].eq('Unknown')
    latest_summary = latest_summary.sort_values(
        ['_unknown_sort', 'share', 'continent_origin'],
        ascending=[True, False, True]
    )

    rows = []
    metric_rows = [
        ('current', 'Now'),
        ('mom', 'MoM'),
        ('yoy', 'YoY'),
    ]
    for row in latest_summary.itertuples(index=False):
        continent = row.continent_origin
        current_values = current_snapshot.get(continent, {})
        reference_values = {
            'mom': mom_snapshot.get(continent),
            'yoy': yoy_snapshot.get(continent),
        }

        for metric_index, (metric_key, metric_label) in enumerate(metric_rows):
            row_cells = []
            if metric_index == 0:
                row_cells.append(
                    html.Th(
                        [
                            html.Span(
                                className='continent-kpi-summary-swatch',
                                style={'backgroundColor': CONTINENT_CHART_COLOR_MAP.get(continent, '#64748b')}
                            ),
                            html.Span(continent, className='exporter-detail-continent-mix-name')
                        ],
                        rowSpan=len(metric_rows),
                        className='continent-kpi-summary-continent-axis-cell'
                    )
                )

            row_cells.append(
                html.Th(
                    metric_label,
                    className=(
                        'continent-kpi-summary-metric-cell '
                        f'continent-kpi-summary-metric-cell-{metric_key}'
                    ),
                    title=metric_label
                )
            )

            if metric_key == 'current':
                row_cells.extend([
                    value_cell(format_volume(current_values.get('volume'))),
                    value_cell(format_share(current_values.get('share')), 'exporter-detail-continent-mix-share-value')
                ])
            else:
                ref = reference_values.get(metric_key)
                role_class = (
                    'continent-kpi-summary-mom-cell'
                    if metric_key == 'mom'
                    else 'continent-kpi-summary-yoy-cell'
                )
                volume_delta = None
                share_delta = None
                if ref:
                    volume_delta = current_values.get('volume') - ref.get('volume')
                    share_delta = current_values.get('share') - ref.get('share')

                row_cells.extend([
                    delta_cell(
                        format_volume(volume_delta, is_delta=True),
                        volume_delta,
                        role_class,
                        pct_text=format_delta_pct(volume_delta, ref.get('volume') if ref else None),
                        is_available=ref is not None
                    ),
                    delta_cell(
                        format_share(share_delta, is_delta=True),
                        share_delta,
                        role_class,
                        is_available=ref is not None
                    )
                ])

            rows.append(
                html.Tr(
                    row_cells,
                    className=(
                        'continent-kpi-summary-row '
                        f'continent-kpi-summary-row-{metric_key} '
                        + ('continent-kpi-summary-continent-group-start' if metric_index == 0 else '')
                    )
                )
            )

    return html.Div(
        html.Div(
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th(
                                    'Origin',
                                    className='continent-kpi-summary-axis-header continent-kpi-summary-continent-axis-header'
                                ),
                                html.Th(
                                    'Metric',
                                    className='continent-kpi-summary-axis-header continent-kpi-summary-metric-axis-header'
                                ),
                                html.Th(
                                    f"Volume ({vol_info['label']})",
                                    className='continent-kpi-summary-entity-header exporter-detail-continent-mix-metric-header'
                                ),
                                html.Th(
                                    'Share %',
                                    className='continent-kpi-summary-entity-header exporter-detail-continent-mix-metric-header'
                                ),
                            ]
                        )
                    ),
                    html.Tbody(rows)
                ],
                className='continent-kpi-summary-table exporter-detail-continent-mix-table'
            ),
            className='continent-kpi-summary-table-wrap exporter-detail-continent-mix-table-wrap'
        ),
        className='continent-kpi-summary exporter-detail-continent-mix-summary'
    )


def _importer_detail_export_button(button_id, class_name='importer-detail-export-button'):
    return html.Button(
        'Export to Excel',
        id=button_id,
        n_clicks=0,
        className=class_name
    )


def _empty_importer_detail_state(message):
    return html.Div(message, className='exporter-detail-empty-state importer-detail-empty-state')


def _build_importer_route_analysis_aggregation_control():
    return html.Div(
        [
            html.Div('Period', className='importer-detail-route-control-label'),
            dcc.Dropdown(
                id='imp-route-aggregation-dropdown',
                options=[
                    {'label': 'Year', 'value': 'Year'},
                    {'label': 'Season', 'value': 'Year+Season'},
                    {'label': 'Qtr', 'value': 'Year+Quarter'},
                    {'label': 'Month', 'value': 'Month'},
                    {'label': 'Week', 'value': 'Week'},
                ],
                value='Year+Quarter',
                multi=False,
                clearable=False,
                className='filter-dropdown importer-detail-route-aggregation-dropdown',
            ),
        ],
        className='importer-detail-route-control'
    )


def _build_importer_detail_section_header(title_id=None, title='Section', title_class='section-title-inline',
                                          right_children=None, header_class='importer-detail-section-header',
                                          title_row_class='importer-detail-section-title-row'):
    title_kwargs = {'className': title_class}
    if title_id is not None:
        title_kwargs['id'] = title_id
    return html.Div(
        [
            html.Div([html.H3(title, **title_kwargs)], className=title_row_class),
            html.Div(right_children or [], className='importer-detail-section-actions')
        ],
        className=f'inline-section-header {header_class}'
    )


# Dashboard layout
layout = html.Div([
    # Store components for importer data
    dcc.Store(id='imp-destination-catalog-store', storage_type='local'),
    dcc.Store(id='imp-destination-selection-store', storage_type='local'),
    dcc.Store(id='imp-import-analysis-base-data-store', storage_type='memory'),
    dcc.Store(id='imp-diversion-processed-data', storage_type='memory'),
    dcc.Store(id='imp-origin-expanded-continents', data=[]),  # Store for expanded state of continents
    dcc.Store(id='imp-origin-forecast-expanded-continents', data=[]),  # Store for WoodMac forecast table expansion
    dcc.Store(id='imp-maintenance-expanded-plants', data=[]),  # Store for expanded state of plants
    dcc.Store(id='imp-maintenance-raw-data-store', storage_type='memory'),
    dcc.Store(id='imp-maintenance-style-refresh-store', storage_type='memory'),
    dcc.Download(id='imp-download-importer-detail-supply-excel'),
    dcc.Download(id='imp-download-route-analysis-excel'),
    dcc.Download(id='imp-download-diversion-summary-excel'),

    # Professional Section Header - Importer Analysis Configuration
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
                            className='filter-dropdown importer-detail-aggregation-dropdown',
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
                            className='filter-dropdown importer-detail-destination-dropdown',
                        ),
                    ], className='filter-group'),
                ], className='importer-detail-filter-row'),
            ], className='filter-group importer-detail-sticky-filter-group'),

            # --- Group 2: Origin ---
            html.Div([
                html.Div("Origin", className='filter-group-header'),
                dcc.RadioItems(
                    id='imp-origin-level-dropdown',
                    options=IMPORTER_ORIGIN_LEVEL_OPTIONS,
                    value=DEFAULT_IMPORTER_ORIGIN_LEVEL,
                    inline=True,
                    className=(
                        'supply-dest-view-selector exporters-sticky-selector '
                        'exporter-detail-destination-selector importer-detail-origin-selector'
                    ),
                    inputStyle={'display': 'none'},
                    labelStyle={'marginRight': '0'},
                ),
            ], className=(
                'filter-group exporters-sticky-filter-group exporter-detail-sticky-filter-group '
                'importer-detail-sticky-filter-group'
            )),

            # --- Group 2b: Volume Metric ---
            html.Div([
                html.Div("Metric", className='filter-group-header'),
                dcc.RadioItems(
                    id='imp-volume-metric-dropdown',
                    options=VOLUME_METRIC_OPTIONS,
                    value='mcm_d',
                    inline=True,
                    className=(
                        'supply-dest-view-selector exporters-sticky-selector exporters-volume-selector '
                        'exporter-detail-volume-selector importer-detail-volume-selector'
                    ),
                    inputStyle={'display': 'none'},
                    labelStyle={'marginRight': '0'},
                ),
            ], className='filter-group importer-detail-sticky-filter-group'),

            html.Div([
                html.Div('Window', className='filter-group-header'),
                html.Div(
                    [
                        dcc.Input(
                            id='imp-supply-rolling-window-input',
                            type='number',
                            value=30,
                            min=1,
                            step=1,
                            debounce=True,
                            className='dash-input-element importer-detail-rolling-window-input'
                        ),
                        html.Span('days', className='exporters-rolling-window-unit importer-detail-rolling-window-unit')
                    ],
                    className=(
                        'exporters-rolling-window-control exporter-detail-rolling-window-control '
                        'importer-detail-rolling-window-control'
                    )
                )
            ], className=(
                'filter-group exporters-sticky-filter-group exporter-detail-sticky-filter-group '
                'importer-detail-sticky-filter-group exporters-rolling-filter-group '
                'importer-detail-rolling-filter-group'
            )),

    ], className='professional-section-header importer-detail-sticky-filter-bar'),

    # Country Import Charts Section - exporter-style card surface
    html.Div([
        html.Div([
            html.Div(
                [
                    html.H3(
                        'LNG Import Analysis - 30-Day Rolling Average + WoodMac Forecast',
                        id='imp-supply-analysis-title',
                        className='section-title-inline'
                    ),
                    html.Div(
                        [
                            _build_detail_year_selector(
                                'imp-import-analysis-year-selector',
                                class_prefix='supply'
                            ),
                        ],
                        className='exporter-detail-supply-selector-row'
                    ),
                ],
                className='supply-rolling-title-row exporter-detail-supply-title-row importer-detail-import-analysis-title-row'
            ),
            _importer_detail_export_button(
                'imp-export-supply-analysis-button',
                class_name='supply-rolling-export-button importer-detail-export-button'
            ),
        ], className='inline-section-header supply-rolling-section-header importer-detail-section-header'),

        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H5(
                                            id='imp-country-supply-header',
                                            children='Total Imports + WoodMac Forecast',
                                            className='supply-rolling-card-title'
                                        ),
                                        html.Span(
                                            id='imp-country-supply-current-value',
                                            className='supply-rolling-current-value'
                                        )
                                    ],
                                    className='supply-rolling-card-title-group'
                                ),
                                html.Div(
                                    id='imp-country-supply-delta-group',
                                    className='supply-rolling-delta-group'
                                )
                            ],
                            className='supply-rolling-card-header'
                        ),
                        dcc.Loading(
                            id='imp-country-supply-loading',
                            children=[
                                dcc.Graph(
                                    id='imp-country-supply-chart',
                                    config={'displayModeBar': False, 'responsive': True},
                                    className='supply-rolling-graph exporter-detail-rolling-graph importer-detail-import-analysis-graph'
                                )
                            ],
                            type='default',
                        )
                    ],
                    className='supply-rolling-card supply-rolling-card-primary exporter-detail-total-supply-card'
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5(
                                    id='imp-continent-origin-header',
                                    children='Origin Volume',
                                    className='continent-rolling-card-title'
                                ),
                                html.Span(
                                    id='imp-continent-origin-current-value',
                                    className='supply-rolling-current-value'
                                ),
                                html.Div(
                                    id='imp-continent-origin-delta-group',
                                    className='supply-rolling-delta-group'
                                )
                            ],
                            className='continent-rolling-card-header exporter-detail-continent-card-header'
                        ),
                        dcc.Loading(
                            id='imp-continent-origin-loading',
                            children=[
                                dcc.Graph(
                                    id='imp-continent-origin-chart',
                                    config={'displayModeBar': False, 'responsive': True},
                                    className='continent-rolling-graph exporter-detail-rolling-graph importer-detail-import-analysis-graph'
                                )
                            ],
                            type='default',
                        )
                    ],
                    id='imp-continent-origin-card',
                    className='continent-rolling-card continent-rolling-card-primary exporter-detail-continent-card'
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5(
                                    id='imp-continent-percentage-header',
                                    children='Origin Share',
                                    className='continent-rolling-card-title'
                                ),
                                html.Span(
                                    id='imp-continent-percentage-current-value',
                                    className='supply-rolling-current-value'
                                )
                            ],
                            className='continent-rolling-card-header exporter-detail-continent-card-header'
                        ),
                        dcc.Loading(
                            id='imp-continent-percentage-loading',
                            children=[
                                dcc.Graph(
                                    id='imp-continent-percentage-chart',
                                    config={'displayModeBar': False, 'responsive': True},
                                    className='continent-rolling-graph exporter-detail-rolling-graph importer-detail-import-analysis-graph'
                                )
                            ],
                            type='default',
                        )
                    ],
                    id='imp-continent-percentage-card',
                    className='continent-rolling-card exporter-detail-continent-card'
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5(
                                    'Origin Mix',
                                    className='continent-rolling-card-title'
                                )
                            ],
                            className='continent-rolling-card-header exporter-detail-continent-card-header'
                        ),
                        html.Div(
                            id='imp-origin-mix-table',
                            className='exporter-detail-continent-mix-table-host'
                        )
                    ],
                    id='imp-origin-mix-card',
                    className='exporter-detail-continent-mix-card'
                )
            ],
            className='supply-rolling-grid exporter-detail-supply-grid exporter-detail-three-chart-grid'
        ),

        html.Div(
            "Note: Rolling averages apply only to Kpler-based data. Forecasts from non-Kpler sources, including WoodMac, are shown without a rolling average.",
            className='importer-detail-import-analysis-note'
        )
    ], className='main-section-container supply-rolling-section exporter-detail-supply-section importer-detail-section importer-detail-import-analysis-section'),

    # Origin Analysis Summary
    html.Div([
        _build_importer_detail_section_header(
            title='Period Summary Tables',
            right_children=[_build_importer_detail_period_controls()],
            header_class=(
                'importer-detail-section-header importer-period-section-header '
                'supply-dest-section-header'
            ),
            title_row_class=(
                'importer-detail-section-title-row importer-period-title-row '
                'supply-dest-title-row'
            )
        ),
        html.Div([
            html.Div([
                html.Div([
                    html.H3(
                        'Origin Analysis Summary (mcm/d)',
                        id='imp-origin-summary-header',
                        className='section-title-inline importer-detail-panel-title'
                    )
                ], className='importer-detail-panel-header'),
                dcc.Loading(
                    id="imp-origin-summary-loading",
                    children=[
                        html.Div(
                            id='imp-origin-summary-table-container',
                            className='importer-detail-table-container'
                        )
                    ],
                    type="default"
                )
            ], className='importer-detail-panel importer-period-panel'),

        ], className='importer-detail-single-column-grid importer-period-grid-shell'),
    ], className='main-section-container importer-detail-section importer-period-section supply-dest-section'),

    # Origin Forecast Allocation Summary Section
    html.Div([
        _build_importer_detail_section_header(
            title_id='imp-origin-forecast-summary-header',
            title='Origin Forecast Allocation Summary (WoodMac, mcm/d)',
            header_class='importer-detail-section-header importer-detail-forecast-header'
        ),
        html.Div(
            id='imp-origin-forecast-summary-subtitle',
            className='importer-detail-section-subtitle'
        ),
        dcc.Loading(
            id="imp-origin-forecast-summary-loading",
            children=[
                html.Div(
                    id='imp-origin-forecast-summary-table-container',
                    className='importer-detail-table-container'
                )
            ],
            type="default"
        )
    ], className='main-section-container importer-detail-section importer-detail-forecast-section'),

    # Supplier Maintenance Schedule Section
    html.Div([
        _build_importer_detail_section_header(
            title_id='imp-maintenance-summary-header',
            title='Supplier Maintenance Schedule (MCM/D Impact)',
            header_class='importer-detail-section-header importer-detail-maintenance-header'
        ),
        html.Div(
            dcc.Loading(
                id="imp-maintenance-summary-loading",
                children=[
                    html.Div(
                        id='imp-maintenance-summary-container',
                        className='importer-detail-table-container importer-detail-maintenance-table-container'
                    )
                ],
                type="default"
            ),
            className='importer-detail-maintenance-table-panel'
        )
    ], className='main-section-container importer-detail-section importer-detail-maintenance-section'),

    # Route Analysis Section
    html.Div([
        _build_importer_detail_section_header(
            title='Route Analysis',
            right_children=[
                _build_importer_route_analysis_aggregation_control(),
                _importer_detail_export_button('imp-export-route-analysis-button')
            ],
            header_class='importer-detail-section-header importer-detail-route-header'
        ),
        html.Div(
            id='imp-route-analysis-kpi-container',
            className='route-kpi-container'
        ),
        dcc.Graph(
            id='imp-graph-route-suez-only',
            config={'displayModeBar': False, 'responsive': True},
            className='route-analysis-legacy-graph'
        )
    ], className='main-section-container importer-detail-section importer-detail-route-section'),

    # Diversions Analysis Section
    html.Div([
        html.Div([
            html.Div(
                [html.H3('Diversions Analysis', className='section-title-inline')],
                className='importer-detail-section-title-row'
            ),
            html.Div(
                [
                    dcc.RadioItems(
                        id='imp-diversion-combo-radio',
                        options=[
                            {'label': 'Basin', 'value': 'basin_combo'},
                            {'label': 'Region', 'value': 'region_combo'},
                            {'label': 'Country', 'value': 'country_combo'}
                        ],
                        value='basin_combo',
                        inline=True,
                        className='continent-chart-type-selector importer-detail-diversion-selector',
                        inputStyle={'display': 'none'},
                        labelStyle={'marginRight': '0'}
                    ),
                    _importer_detail_export_button('imp-export-diversion-summary-button')
                ],
                className='importer-detail-section-actions'
            )
        ], className='inline-section-header importer-detail-section-header importer-detail-diversion-header'),
        dcc.Graph(
            id='imp-diversion-count-chart',
            config={'displayModeBar': False, 'responsive': True},
            className='importer-detail-large-graph importer-detail-diversion-chart'
        ),
        create_ag_grid_from_datatable(
            id='imp-diversion-table',
            data=[],
            columns=[],
            sort_action="native",
            page_action='none',
            fill_width=False,
            height='348px',
            dashGridOptions={
                'rowHeight': 28,
                'headerHeight': 42,
                'tooltipShowDelay': 250,
            },
            rowClassRules={
                'diversion-row-loaded': "params.data && params.data.State === 'Loaded'",
                'diversion-row-non-loaded': "params.data && params.data.State && params.data.State !== 'Loaded'",
            },
            className='importer-detail-grid importer-detail-diversion-grid'
        )
    ], className='main-section-container importer-detail-section importer-detail-diversion-section'),

], className='importer-detail-page')


SUMMARY_LOOKBACK_START = dt.date(2023, 11, 1)
WOODMAC_IMPORT_EXPORTS_TABLE = 'at_lng.woodmac_gas_imports_exports_monthly__mmtpa'
WOODMAC_LNG_CUBIC_METERS_PER_MMTPA_MONTH = 2222 * 1000 / 12
WOODMAC_FORECAST_YEARS_AHEAD = 2
SUPPLY_ALLOCATION_RUNS_TABLE = f'{DB_SCHEMA}.fundamentals_supply_allocation_runs'
SUPPLY_ALLOCATION_DEMAND_DETAIL_TABLE = f'{DB_SCHEMA}.fundamentals_supply_allocation_demand_detail'
SUPPLY_ALLOCATION_DEMAND_SUMMARY_TABLE = f'{DB_SCHEMA}.fundamentals_supply_allocation_demand_summary'


def combine_origin_summary_data_hierarchical(quarters_df, months_df, weeks_df, rolling_df, rolling_window_days=30,
                                             quarter_count=5, month_count=3, week_count=3,
                                             include_comparison_reference_columns=False):
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
            quarter_keep_count = (
                DETAIL_MAX_QUARTER_COUNT + 4
                if include_comparison_reference_columns
                else quarter_count
            )
            completed_quarters = sorted(
                completed_quarters,
                key=lambda x: (x.split("'")[1], x.split("Q")[1].split("'")[0])
            )[-quarter_keep_count:]
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
            month_keep_count = (
                DETAIL_MAX_MONTH_COUNT + 12
                if include_comparison_reference_columns
                else month_count
            )
            completed_months = sorted(
                completed_months,
                key=lambda x: (x.split("'")[1], month_order.get(x.split("'")[0], 0))
            )[-month_keep_count:]
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
            week_keep_count = (
                DETAIL_MAX_WEEK_COUNT + 53
                if include_comparison_reference_columns
                else week_count
            )
            completed_weeks = sorted(
                completed_weeks,
                key=lambda x: (x.split("'")[1], x.split("W")[1].split("'")[0].zfill(2))
            )[-week_keep_count:]
            if completed_weeks:
                result = result.merge(
                    weeks_df[['continent', 'country'] + completed_weeks],
                    on=['continent', 'country'],
                    how='left'
                )

        if not rolling_df.empty:
            rolling_cols = ['7D', f'Δ 7D-{rolling_window_label}', f'Δ {rolling_window_label} Y/Y']
            if include_comparison_reference_columns:
                rolling_cols.extend([
                    '7D_PP',
                    f'{rolling_window_label}_PP',
                    '7D_Y1',
                    f'{rolling_window_label}_Y1',
                ])
            for col in rolling_cols:
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
    except Exception:
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


def build_importer_origin_summary_from_scoped_trades(scoped_trades_df, rolling_window_days=30,
                                                     origin_level=DEFAULT_IMPORTER_ORIGIN_LEVEL,
                                                     quarter_count=5,
                                                     month_count=3,
                                                     week_count=3,
                                                     include_comparison_reference_columns=False):
    """Build importer-side origin summary data from an already filtered scoped trade frame."""
    summary_scope_df = _prepare_importer_summary_scope_df(
        scoped_trades_df,
        origin_level or DEFAULT_IMPORTER_ORIGIN_LEVEL
    )
    if summary_scope_df.empty:
        return pd.DataFrame()

    quarters_df = _build_importer_periods_pivot(summary_scope_df, 'quarter')
    months_df = _build_importer_periods_pivot(summary_scope_df, 'month')
    weeks_df = _build_importer_periods_pivot(summary_scope_df, 'week')
    rolling_df = _build_importer_rolling_windows_pivot(
        summary_scope_df,
        rolling_window_days=rolling_window_days,
        include_comparison_reference_columns=include_comparison_reference_columns
    )
    return combine_origin_summary_data_hierarchical(
        quarters_df,
        months_df,
        weeks_df,
        rolling_df,
        rolling_window_days,
        quarter_count=quarter_count,
        month_count=month_count,
        week_count=week_count,
        include_comparison_reference_columns=include_comparison_reference_columns
    )


def fetch_origin_summary_data(engine, destination_countries, rolling_window_days=30,
                              origin_level=DEFAULT_IMPORTER_ORIGIN_LEVEL,
                              selected_destination_aggregation='country',
                              selected_destination_value=None,
                              scoped_trades_df=None,
                              quarter_count=5,
                              month_count=3,
                              week_count=3,
                              include_comparison_reference_columns=False):
    """Fetch importer-side origin summary data with supplier continent/country hierarchy."""
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    if not normalized_destination_countries:
        return pd.DataFrame()

    try:
        if scoped_trades_df is None:
            scoped_trades_df = _fetch_importer_scoped_trades(
                engine,
                normalized_destination_countries,
                delivered_only=True
            )

        filtered_df = _apply_importer_self_flow_exclusion(
            scoped_trades_df,
            selected_destination_aggregation,
            selected_destination_value
        )
        return build_importer_origin_summary_from_scoped_trades(
            filtered_df,
            rolling_window_days=rolling_window_days,
            origin_level=origin_level,
            quarter_count=quarter_count,
            month_count=month_count,
            week_count=week_count,
            include_comparison_reference_columns=include_comparison_reference_columns
        )
    except Exception:
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


def fetch_origin_forecast_summary_data(
    engine,
    destination_countries,
    current_date=None,
    origin_level=DEFAULT_IMPORTER_ORIGIN_LEVEL
):
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
            continent,
            basin,
            subcontinent,
            country_classification_level1,
            country_classification
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
            ext_mapping = (
                mapping_df[['country_name', mapping_col]]
                .dropna(subset=['country_name'])
                .drop_duplicates()
                .rename(columns={'country_name': 'country_display', mapping_col: 'level_val'})
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
        scoped_trades_df = _fetch_importer_scoped_trades(
            engine,
            normalized_destination_countries,
            min_end_date=DETAIL_CHART_DATA_START_DATE
        )

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
        scoped_trades_df = _fetch_importer_scoped_trades(
            engine,
            normalized_destination_countries,
            min_end_date=DETAIL_CHART_DATA_START_DATE
        )

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


def _detail_year_sort_key(year):
    try:
        return (0, int(year))
    except (TypeError, ValueError):
        return (1, str(year))


def _get_detail_supply_chart_color_map(years):
    years = sorted(years or [], key=_detail_year_sort_key)
    if not years:
        return {}
    if len(years) <= len(SUPPLY_CHART_COLOR_SEQUENCE):
        visible_colors = SUPPLY_CHART_COLOR_SEQUENCE[-len(years):]
    else:
        repeats = (len(years) // len(SUPPLY_CHART_COLOR_SEQUENCE)) + 1
        visible_colors = (SUPPLY_CHART_COLOR_SEQUENCE * repeats)[-len(years):]
    return {
        year: visible_colors[idx]
        for idx, year in enumerate(years)
    }


def _detail_chart_plot_dates(day_of_year_series):
    return pd.to_datetime(f'{DETAIL_CHART_ANCHOR_YEAR}-01-01') + pd.to_timedelta(
        day_of_year_series.astype(int) - 1,
        unit='d'
    )


def _get_detail_supply_range_years(focus_year, available_years):
    try:
        focus_year_number = int(focus_year)
    except (TypeError, ValueError):
        return []

    previous_years = []
    for year in sorted(available_years, key=_detail_year_sort_key):
        try:
            if int(year) < focus_year_number:
                previous_years.append(year)
        except (TypeError, ValueError):
            continue
    return previous_years[-SUPPLY_CHART_RANGE_LOOKBACK_YEARS:]


def _add_detail_supply_chart_range_band(fig, df, focus_year, available_years, vol_label, value_factor=1.0):
    range_years = _get_detail_supply_range_years(focus_year, available_years)
    if not range_years or df is None or df.empty:
        return

    range_df = df[df['year'].isin(range_years) & df['rolling_avg'].notna()].copy()
    if range_df.empty:
        return

    if 'is_forecast' in range_df.columns:
        range_df = range_df[~range_df['is_forecast'].astype(bool)].copy()
        if range_df.empty:
            return

    range_df['plot_date'] = _detail_chart_plot_dates(range_df['day_of_year'])
    range_df['rolling_avg'] = range_df['rolling_avg'] * value_factor
    range_df = (
        range_df
        .groupby('plot_date', as_index=False)
        .agg(
            range_min=('rolling_avg', 'min'),
            range_max=('rolling_avg', 'max'),
            month_day=('month_day', 'last')
        )
        .sort_values('plot_date')
    )
    if range_df.empty:
        return

    years_label = f"{range_years[0]}-{range_years[-1]}" if len(range_years) > 1 else str(range_years[0])
    fig.add_trace(go.Scatter(
        x=range_df['plot_date'],
        y=range_df['range_min'],
        mode='lines',
        line=dict(color='rgba(148, 163, 184, 0)', width=0),
        hoverinfo='skip',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=range_df['plot_date'],
        y=range_df['range_max'],
        mode='lines',
        name=f'{years_label} range',
        line=dict(color='rgba(148, 163, 184, 0)', width=0),
        fill='tonexty',
        fillcolor=SUPPLY_CHART_RANGE_FILL,
        customdata=range_df[['range_min']].to_numpy(),
        text=range_df['month_day'],
        hovertemplate=(
            f'<b>{years_label} range</b> | '
            '%{text} | '
            f'%{{customdata[0]:,.0f}}-%{{y:,.0f}} {vol_label}<extra></extra>'
        ),
        showlegend=False
    ))


def _detail_continent_chart_line_style(year, current_year, is_forecast=False):
    is_current_year = int(year) == int(current_year)
    return {
        'width': CONTINENT_CHART_CURRENT_YEAR_WIDTH if is_current_year else CONTINENT_CHART_PREVIOUS_YEAR_WIDTH,
        'opacity': 0.94 if is_current_year and not is_forecast else 0.72 if is_current_year else 0.44,
        'dash': CONTINENT_CHART_FORECAST_DASH if is_forecast else 'solid'
    }


def _apply_time_series_chart_layout(fig, yaxis_title):
    fig.update_layout(
        xaxis=dict(
            title=dict(text='', font=dict(size=12, color='#475569')),
            tickformat='%b',
            dtick='M1',
            tickangle=0,
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.18)',
            gridwidth=0.5,
            linecolor='rgba(148, 163, 184, 0.6)',
            linewidth=1,
            tickfont=dict(size=10, color='#64748b'),
            range=[
                pd.Timestamp(year=DETAIL_CHART_ANCHOR_YEAR, month=1, day=1),
                pd.Timestamp(year=DETAIL_CHART_ANCHOR_YEAR, month=12, day=31)
            ],
            showspikes=True,
            spikemode='across',
            spikecolor='rgba(15, 23, 42, 0.18)',
            spikethickness=1
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=11, color='#475569')),
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.22)',
            gridwidth=0.5,
            linecolor='rgba(148, 163, 184, 0.6)',
            linewidth=1,
            tickfont=dict(size=10, color='#64748b'),
            zeroline=True,
            zerolinecolor='rgba(148, 163, 184, 0.28)',
            zerolinewidth=1,
            autorange=True
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.12,
            xanchor='left',
            x=0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            borderwidth=0,
            font=dict(size=11, color='#475569'),
            itemsizing='constant',
            itemwidth=30
        ),
        height=IMPORTER_DETAIL_SUPPLY_CHART_HEIGHT,
        margin=dict(l=44, r=18, t=12, b=36),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.96)',
            bordercolor='rgba(148, 163, 184, 0.7)',
            font=dict(size=11, color='#0f172a'),
            align='left'
        ),
        title=None,
        transition=dict(duration=300, easing='cubic-in-out')
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
        font=dict(size=12, color='#64748b')
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=IMPORTER_DETAIL_SUPPLY_CHART_HEIGHT,
        margin=dict(l=36, r=20, t=12, b=36),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff'
    )
    return fig


def _create_seasonal_line_chart(df, series_col, value_col, yaxis_title, metric_label, selected_years=None):
    df = _filter_df_years(df, selected_years)
    if df.empty:
        return _empty_timeseries_chart("No data available")

    fig = go.Figure()
    years = sorted(df['year'].dropna().unique())

    if series_col is None:
        color_map = _get_detail_supply_chart_color_map([int(year) for year in years])
        current_year = int(max(years))
        for i, year in enumerate(years):
            year = int(year)
            year_data = df[df['year'] == year].copy()
            year_data['plot_date'] = _detail_chart_plot_dates(year_data['day_of_year'])
            historical_data = year_data[~year_data['is_forecast']]
            forecast_data = year_data[year_data['is_forecast']]
            base_color = color_map.get(year, '#0f4c81')
            line_style = _detail_continent_chart_line_style(year, current_year)

            if not historical_data.empty:
                fig.add_trace(go.Scatter(
                    x=historical_data['plot_date'],
                    y=historical_data[value_col],
                    mode='lines',
                    name=str(year),
                    line=dict(color=base_color, width=line_style['width'], dash=line_style['dash']),
                    opacity=line_style['opacity'],
                    text=historical_data['month_day'],
                    hovertemplate=f'<b>{year}</b> | %{{text}} | {metric_label}: %{{y:.1f}}<extra></extra>'
                ))

            if not forecast_data.empty:
                connect_data = pd.concat([historical_data.tail(1), forecast_data]) if not historical_data.empty else forecast_data
                forecast_style = _detail_continent_chart_line_style(year, current_year, is_forecast=True)
                fig.add_trace(go.Scatter(
                    x=connect_data['plot_date'],
                    y=connect_data[value_col],
                    mode='lines',
                    name=f"{year} Forecast",
                    line=dict(color=base_color, width=forecast_style['width'], dash=forecast_style['dash']),
                    opacity=forecast_style['opacity'],
                    text=connect_data['month_day'],
                    hovertemplate=f'<b>{year} forecast</b> | %{{text}} | {metric_label}: %{{y:.1f}}<extra></extra>',
                    showlegend=False
                ))
    else:
        series_values = sorted(df[series_col].dropna().unique())
        color_map = {
            value: CONTINENT_CHART_COLOR_MAP.get(value, get_professional_colors(len(series_values))[idx])
            for idx, value in enumerate(series_values)
        }
        current_year = int(max(years))
        shown_legend = set()
        for series_value in series_values:
            series_df = df[df[series_col] == series_value]
            for year in years:
                year = int(year)
                year_data = series_df[series_df['year'] == year].copy()
                if year_data.empty:
                    continue
                year_data['plot_date'] = _detail_chart_plot_dates(year_data['day_of_year'])
                historical_data = year_data[~year_data['is_forecast']]
                forecast_data = year_data[year_data['is_forecast']]
                color = color_map[series_value]
                historical_style = _detail_continent_chart_line_style(year, current_year)
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
                        line=dict(
                            color=color,
                            width=historical_style['width'],
                            dash=historical_style['dash']
                        ),
                        opacity=historical_style['opacity'],
                        text=historical_data['month_day'],
                        hovertemplate=(
                            f'<b>{series_value}</b> | {year} | %{{text}} | {metric_label}: %{{y:.1f}}<extra></extra>'
                        ),
                        showlegend=show_legend
                    ))

                if not forecast_data.empty:
                    connect_data = pd.concat([historical_data.tail(1), forecast_data]) if not historical_data.empty else forecast_data
                    forecast_style = _detail_continent_chart_line_style(year, current_year, is_forecast=True)
                    fig.add_trace(go.Scatter(
                        x=connect_data['plot_date'],
                        y=connect_data[value_col],
                        mode='lines',
                        name=None,
                        legendgroup=str(series_value),
                        line=dict(
                            color=color,
                            width=forecast_style['width'],
                            dash=forecast_style['dash']
                        ),
                        opacity=forecast_style['opacity'],
                        text=connect_data['month_day'],
                        hovertemplate=(
                            f'<b>{series_value}</b> | {year} forecast | %{{text}} | {metric_label}: %{{y:.1f}}<extra></extra>'
                        ),
                        showlegend=False
                    ))

    return _apply_time_series_chart_layout(fig, yaxis_title)


def _create_total_import_chart_with_woodmac_forecast(
    historical_df,
    forecast_df,
    volume_metric='mcm_d',
    selected_years=None,
    rolling_window_days=30
):
    vol_info = get_volume_metric_info(volume_metric)
    vol_label = vol_info['label']
    vol_factor = _get_detail_volume_metric_factor(
        volume_metric,
        period_days=normalize_rolling_window_days(rolling_window_days)
    )
    full_historical_df = historical_df.copy() if historical_df is not None else pd.DataFrame()
    forecast_df = filter_woodmac_forecast_horizon(forecast_df)
    active_historical_df = _filter_df_years(full_historical_df, selected_years)
    active_forecast_df = _filter_df_years(forecast_df, selected_years)
    active_historical_df = convert_volume_metric_dataframe(
        active_historical_df,
        volume_metric,
        columns=['rolling_avg'],
        period_days=normalize_rolling_window_days(rolling_window_days)
    )
    active_forecast_df = convert_volume_metric_dataframe(
        active_forecast_df,
        volume_metric,
        columns=['mcmd'],
        period_days=normalize_rolling_window_days(rolling_window_days)
    )
    if active_historical_df.empty and active_forecast_df.empty:
        return _empty_timeseries_chart("No data available")

    fig = go.Figure()
    all_years = sorted(set(full_historical_df.get('year', pd.Series(dtype=int)).dropna().astype(int).tolist()) |
                       set(forecast_df.get('year', pd.Series(dtype=int)).dropna().astype(int).tolist()))
    color_map = _get_detail_supply_chart_color_map(all_years)
    latest_historical_year = (
        int(active_historical_df['year'].dropna().max())
        if not active_historical_df.empty and active_historical_df['year'].notna().any()
        else None
    )
    focus_year = latest_historical_year
    if focus_year is None and not active_forecast_df.empty and active_forecast_df['year'].notna().any():
        focus_year = int(active_forecast_df['year'].dropna().max())
    if focus_year is not None:
        _add_detail_supply_chart_range_band(
            fig,
            full_historical_df,
            focus_year,
            all_years,
            vol_label,
            value_factor=vol_factor
        )

    for year in sorted(active_historical_df['year'].dropna().unique()):
        year = int(year)
        year_data = active_historical_df[active_historical_df['year'] == year].copy().sort_values('date')
        if year_data.empty:
            continue
        year_data['plot_date'] = _detail_chart_plot_dates(year_data['day_of_year'])
        base_color = color_map.get(year, '#0f4c81')
        is_focus_year = year == latest_historical_year
        line_width = 2.2 if is_focus_year else 1.15
        line_opacity = 0.95 if is_focus_year else 0.52

        actual_data = year_data[~year_data['is_forecast']] if 'is_forecast' in year_data.columns else year_data
        kpler_fc_data = year_data[year_data['is_forecast']] if 'is_forecast' in year_data.columns else pd.DataFrame()

        if not actual_data.empty:
            fig.add_trace(go.Scatter(
                x=actual_data['plot_date'],
                y=actual_data['rolling_avg'],
                mode='lines',
                name=str(year),
                line=dict(color=base_color, width=line_width, dash='solid'),
                opacity=line_opacity,
                text=actual_data['month_day'],
                hovertemplate=(
                    f'<b>{year}</b> | '
                    '%{text} | '
                    f'%{{y:,.0f}} {vol_label}<extra></extra>'
                )
            ))

        if not kpler_fc_data.empty:
            connect_data = pd.concat([actual_data.tail(1), kpler_fc_data])
            fig.add_trace(go.Scatter(
                x=connect_data['plot_date'],
                y=connect_data['rolling_avg'],
                mode='lines',
                name=f'{year} Kpler Forecast',
                line=dict(color=base_color, width=line_width, dash=SUPPLY_CHART_FORECAST_DASH),
                opacity=0.76 if is_focus_year else 0.36,
                text=connect_data['month_day'],
                hovertemplate=(
                    f'<b>{year} Kpler forecast</b> | '
                    '%{text} | '
                    f'%{{y:,.0f}} {vol_label}<extra></extra>'
                ),
                showlegend=False
            ))

        if is_focus_year:
            latest_point = actual_data.dropna(subset=['rolling_avg']).tail(1)
            if not latest_point.empty:
                point = latest_point.iloc[0]
                fig.add_trace(go.Scatter(
                    x=[point['plot_date']],
                    y=[point['rolling_avg']],
                    mode='markers',
                    marker=dict(
                        color=base_color,
                        size=5.5,
                        line=dict(color='#ffffff', width=1.5)
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))

    forecast_years = sorted(active_forecast_df.get('year', pd.Series(dtype=int)).dropna().astype(int).unique().tolist())
    current_year = dt.date.today().year
    default_visible_forecast_year = (
        current_year if current_year in forecast_years else (forecast_years[0] if forecast_years else None)
    )
    for year in forecast_years:
        year_data = active_forecast_df[active_forecast_df['year'] == year].copy().sort_values('date')
        if year_data.empty:
            continue
        year_data['plot_date'] = _detail_chart_plot_dates(year_data['day_of_year'])
        base_color = color_map.get(year, '#0f4c81')
        fig.add_trace(go.Scatter(
            x=year_data['plot_date'],
            y=year_data['mcmd'],
            mode='lines',
            name=f'{year} WM',
            line=dict(
                color=base_color,
                width=2.2 if year == default_visible_forecast_year else 1.15,
                dash=SUPPLY_CHART_FORECAST_DASH
            ),
            opacity=0.76 if year == default_visible_forecast_year else 0.36,
            text=year_data['month_day'],
            customdata=year_data['source'],
            hovertemplate=(
                f'<b>{year} WoodMac Forecast</b> | '
                '%{text} | '
                f'%{{y:,.0f}} {vol_label}'
                '<br>Source: %{customdata}<extra></extra>'
            ),
            visible=True if year == default_visible_forecast_year else 'legendonly'
        ))

    return _apply_time_series_chart_layout(fig, vol_label)


def create_country_import_chart(destination_label, destination_countries, rolling_window_days=30,
                                selected_destination_aggregation='country', selected_destination_value=None,
                                volume_metric='mcm_d', scoped_trades_df=None, selected_years=None,
                                historical_df=None, forecast_df=None):
    if historical_df is None:
        historical_df = fetch_country_import_chart_data(
            destination_countries,
            rolling_window_days,
            selected_destination_aggregation=selected_destination_aggregation,
            selected_destination_value=selected_destination_value,
            scoped_trades_df=scoped_trades_df
        )

    if forecast_df is None:
        forecast_df = fetch_woodmac_country_import_forecast_data(destination_countries)
    if historical_df.empty and forecast_df.empty:
        return _empty_timeseries_chart(f"No import data available for {destination_label}")
    return _create_total_import_chart_with_woodmac_forecast(
        historical_df,
        forecast_df,
        volume_metric,
        selected_years=selected_years,
        rolling_window_days=rolling_window_days
    )


def create_continent_origin_chart(destination_label, destination_countries, rolling_window_days=30,
                                  selected_destination_aggregation='country', selected_destination_value=None,
                                  volume_metric='mcm_d', scoped_trades_df=None, selected_years=None,
                                  continent_df=None):
    df = continent_df
    if df is None:
        df = fetch_continent_origin_chart_data(
            destination_countries,
            rolling_window_days,
            selected_destination_aggregation=selected_destination_aggregation,
            selected_destination_value=selected_destination_value,
            scoped_trades_df=scoped_trades_df
        )
    if df.empty:
        return _empty_timeseries_chart(f"No origin-continent import data available for {destination_label}")
    vol_info = get_volume_metric_info(volume_metric)
    vol_label = vol_info['label']
    df = convert_volume_metric_dataframe(
        df,
        volume_metric,
        columns=['rolling_avg'],
        period_days=normalize_rolling_window_days(rolling_window_days)
    )
    return _create_seasonal_line_chart(
        df,
        'continent_origin',
        'rolling_avg',
        vol_label,
        f'Imports ({vol_label})',
        selected_years=selected_years
    )


def create_continent_origin_percentage_chart(destination_label, destination_countries, rolling_window_days=30,
                                             selected_destination_aggregation='country',
                                             selected_destination_value=None, scoped_trades_df=None,
                                             selected_years=None, percentage_df=None):
    df = percentage_df
    if df is None:
        df = fetch_continent_origin_chart_data(
            destination_countries,
            rolling_window_days,
            include_percentage=True,
            selected_destination_aggregation=selected_destination_aggregation,
            selected_destination_value=selected_destination_value,
            scoped_trades_df=scoped_trades_df
        )
    if df.empty:
        return _empty_timeseries_chart(f"No origin share data available for {destination_label}")
    return _create_seasonal_line_chart(
        df,
        'continent_origin',
        'percentage',
        '%',
        'Share',
        selected_years=selected_years
    )


@callback(
    Output('imp-supply-analysis-title', 'children'),
    Input('imp-supply-rolling-window-input', 'value')
)
def update_supply_analysis_title(rolling_window_days):
    return f"LNG Import Analysis - {format_rolling_window_title(rolling_window_days)} + WoodMac Forecast"


@callback(
    Output('imp-import-analysis-base-data-store', 'data'),
    Input('global-refresh-button', 'n_clicks'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-destination-catalog-store', 'data'),
    prevent_initial_call=False
)
def refresh_import_analysis_base_data(_n_clicks, selected_destination_aggregation,
                                      selected_destination, destination_catalog):
    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    if not destination_context['destination_countries']:
        return _import_analysis_store_payload(
            selected_destination_aggregation,
            selected_destination,
            destination_context,
            pd.DataFrame()
        )

    try:
        scoped_trades_df = _fetch_importer_scoped_trades(
            engine,
            destination_context['destination_countries'],
            min_end_date=DETAIL_CHART_DATA_START_DATE
        )
        return _import_analysis_store_payload(
            selected_destination_aggregation,
            selected_destination,
            destination_context,
            scoped_trades_df
        )
    except Exception as exc:
        return _import_analysis_store_payload(
            selected_destination_aggregation,
            selected_destination,
            destination_context,
            pd.DataFrame(),
            error=str(exc)
        )


@callback(
    Output('imp-import-analysis-year-selector', 'options'),
    Output('imp-import-analysis-year-selector', 'value'),
    Input('imp-import-analysis-base-data-store', 'data'),
    State('imp-import-analysis-year-selector', 'value'),
    prevent_initial_call=False
)
def update_import_analysis_year_selector(base_data, selected_years):
    scoped_trades_df, destination_context, selected_aggregation, selected_value, error = (
        _resolve_import_analysis_base_data(base_data)
    )
    years = set()
    if not error and not scoped_trades_df.empty:
        filtered_df = _apply_importer_self_flow_exclusion(
            scoped_trades_df,
            selected_aggregation,
            selected_value
        )
        if not filtered_df.empty and 'end_date' in filtered_df.columns:
            end_dates = pd.to_datetime(filtered_df['end_date'], errors='coerce')
            years.update(end_dates.dropna().dt.year.astype(int).tolist())

    if destination_context.get('destination_countries'):
        years.update(_detail_chart_years())
        current_year = dt.date.today().year
        years.update([current_year, current_year + 1, current_year + 2])

    ordered_years = sorted(years)
    options = _detail_year_options(ordered_years)
    selected_set = set(str(year) for year in (selected_years or []))
    value = [str(year) for year in ordered_years if str(year) in selected_set]
    if not value:
        value = _default_detail_year_values(ordered_years)
    return options, value


@callback(
    Output('imp-country-supply-chart', 'figure'),
    Output('imp-country-supply-header', 'children'),
    Output('imp-country-supply-current-value', 'children'),
    Output('imp-country-supply-delta-group', 'children'),
    Output('imp-continent-origin-chart', 'figure'),
    Output('imp-continent-origin-header', 'children'),
    Output('imp-continent-origin-current-value', 'children'),
    Output('imp-continent-origin-delta-group', 'children'),
    Output('imp-continent-percentage-chart', 'figure'),
    Output('imp-continent-percentage-header', 'children'),
    Output('imp-continent-percentage-current-value', 'children'),
    Output('imp-origin-mix-table', 'children'),
    Input('imp-import-analysis-base-data-store', 'data'),
    Input('imp-supply-rolling-window-input', 'value'),
    Input('imp-volume-metric-dropdown', 'value'),
    Input('imp-import-analysis-year-selector', 'value'),
)
def update_import_analysis_charts(base_data, rolling_window_days, volume_metric, selected_years):
    vol_label = get_volume_metric_info(volume_metric)['label']
    normalized_window_days = normalize_rolling_window_days(rolling_window_days)
    vol_factor = _get_detail_volume_metric_factor(
        volume_metric,
        period_days=normalized_window_days
    )
    scoped_trades_df, destination_context, selected_destination_aggregation, selected_destination, error = (
        _resolve_import_analysis_base_data(base_data)
    )

    if error or not destination_context.get('destination_countries'):
        empty_message = "Error loading import data" if error else "No import data available"
        empty_fig = _empty_timeseries_chart(empty_message)
        return (
            empty_fig,
            "Total Imports + WoodMac Forecast",
            None,
            _build_detail_delta_group({}, vol_label),
            empty_fig,
            f"By Origin Continent ({vol_label})",
            None,
            None,
            empty_fig,
            "Origin Share",
            None,
            _build_importer_origin_mix_table(
                pd.DataFrame(),
                volume_metric or 'mcm_d',
                rolling_window_days=normalized_window_days
            )
        )

    destination_countries = destination_context['destination_countries']
    destination_label = destination_context['display_label'] or selected_destination or 'Destination'
    supply_df = fetch_country_import_chart_data(
        destination_countries,
        normalized_window_days,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination,
        scoped_trades_df=scoped_trades_df
    )
    forecast_df = fetch_woodmac_country_import_forecast_data(destination_countries)
    country_fig = create_country_import_chart(
        destination_label,
        destination_countries,
        normalized_window_days,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination,
        volume_metric=volume_metric or 'mcm_d',
        scoped_trades_df=scoped_trades_df,
        selected_years=selected_years,
        historical_df=supply_df,
        forecast_df=forecast_df
    )
    country_metrics = _calculate_latest_detail_metrics(
        supply_df,
        value_col='rolling_avg',
        value_factor=vol_factor
    )

    continent_df = fetch_continent_origin_chart_data(
        destination_countries,
        normalized_window_days,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination,
        scoped_trades_df=scoped_trades_df
    )
    continent_fig = create_continent_origin_chart(
        destination_label,
        destination_countries,
        normalized_window_days,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination,
        volume_metric=volume_metric or 'mcm_d',
        scoped_trades_df=scoped_trades_df,
        selected_years=selected_years,
        continent_df=continent_df
    )
    percentage_df = fetch_continent_origin_chart_data(
        destination_countries,
        normalized_window_days,
        include_percentage=True,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination,
        scoped_trades_df=scoped_trades_df
    )
    percentage_fig = create_continent_origin_percentage_chart(
        destination_label,
        destination_countries,
        normalized_window_days,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination,
        scoped_trades_df=scoped_trades_df,
        selected_years=selected_years,
        percentage_df=percentage_df
    )
    origin_mix_table = _build_importer_origin_mix_table(
        continent_df,
        volume_metric or 'mcm_d',
        rolling_window_days=normalized_window_days
    )

    return (
        country_fig,
        f"{destination_label} - Total Imports",
        _format_detail_metric_value(country_metrics.get('current_value'), vol_label),
        _build_detail_delta_group(country_metrics, vol_label),
        continent_fig,
        f"{destination_label} - Origin Volume",
        None,
        None,
        percentage_fig,
        f"{destination_label} - Origin Share",
        None,
        origin_mix_table,
    )


@callback(
    Output('imp-download-importer-detail-supply-excel', 'data'),
    Input('imp-export-supply-analysis-button', 'n_clicks'),
    State('imp-destination-aggregation-dropdown', 'value'),
    State('imp-destination-country-dropdown', 'value'),
    State('imp-destination-catalog-store', 'data'),
    State('imp-supply-rolling-window-input', 'value'),
    State('imp-origin-level-dropdown', 'value'),
    State('imp-volume-metric-dropdown', 'value'),
    State('imp-import-analysis-base-data-store', 'data'),
    prevent_initial_call=True
)
def export_import_analysis_to_excel(n_clicks, selected_destination_aggregation, selected_destination,
                                    destination_catalog, rolling_window_days, origin_level,
                                    volume_metric, base_data):
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

    mapping_lookup_df = _load_importer_country_mapping_lookup(engine)
    cached_scoped_trades_df, _, _, _, _ = _resolve_import_analysis_base_data(base_data)
    if _import_analysis_base_data_matches(
        base_data,
        destination_context,
        selected_destination_aggregation,
        selected_destination
    ):
        chart_scoped_trades_df = cached_scoped_trades_df
    else:
        chart_scoped_trades_df = _fetch_importer_scoped_trades(
            engine,
            destination_context['destination_countries'],
            min_end_date=DETAIL_CHART_DATA_START_DATE,
            mapping_lookup_df=mapping_lookup_df
        )
    summary_scoped_trades_df = _fetch_importer_scoped_trades(
        engine,
        destination_context['destination_countries'],
        delivered_only=True,
        mapping_lookup_df=mapping_lookup_df
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
        normalized_window_days,
        origin_level=origin_level or DEFAULT_IMPORTER_ORIGIN_LEVEL,
        selected_destination_aggregation=selected_destination_aggregation,
        selected_destination_value=selected_destination,
        scoped_trades_df=summary_scoped_trades_df
    )

    if supply_df.empty and continent_df.empty and percentage_df.empty and summary_df.empty:
        raise PreventUpdate

    vol_info = get_volume_metric_info(volume_metric)
    vol_label = vol_info['label']
    rolling_period_days = normalized_window_days
    for frame in (supply_df, continent_df, percentage_df):
        if not frame.empty and 'rolling_avg' in frame.columns:
            frame['rolling_avg'] = _convert_detail_volume_series(
                frame['rolling_avg'],
                volume_metric,
                period_days=rolling_period_days,
                precision=1
            )
    if not summary_df.empty:
        summary_df = _convert_detail_period_display_df(
            summary_df,
            volume_metric,
            rolling_window_days=rolling_period_days
        )

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not supply_df.empty:
            supply_export_df = supply_df.copy()
            if 'rolling_avg' in supply_export_df.columns:
                supply_export_df = supply_export_df.rename(columns={'rolling_avg': f'rolling_avg ({vol_label})'})
            supply_export_df.to_excel(writer, sheet_name='Total Imports', index=False)
        if not continent_df.empty:
            continent_export_df = continent_df.copy()
            if 'rolling_avg' in continent_export_df.columns:
                continent_export_df = continent_export_df.rename(columns={'rolling_avg': f'rolling_avg ({vol_label})'})
            continent_export_df.to_excel(writer, sheet_name=f"Origin Continent {vol_label.replace('/', '_')}"[:31], index=False)
        if not percentage_df.empty:
            percentage_export_df = percentage_df.copy()
            if 'rolling_avg' in percentage_export_df.columns:
                percentage_export_df = percentage_export_df.rename(columns={'rolling_avg': f'rolling_avg ({vol_label})'})
            percentage_export_df.to_excel(writer, sheet_name='Origin Continent Share', index=False)
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
def initialize_country_dropdown(_n_clicks, selected_destination_aggregation, existing_catalog, selection_state):
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
    except Exception:
        fallback_catalog = []
        fallback_options = [{'label': 'China', 'value': 'China'}]
        return fallback_catalog, fallback_options, 'China'


def fetch_train_maintenance_data(engine, destination_countries=None):
    """
    Fetch and process maintenance data for supplier countries feeding the selected importer destinations.
    """
    try:
        normalized_destination_countries = normalize_destination_countries(destination_countries)
        params = {}
        supplier_ctes = ""
        supplier_filter = ""
        if normalized_destination_countries:
            supplier_ctes = f"""
            latest_timestamp AS (
                SELECT MAX(upload_timestamp_utc) AS max_ts
                FROM {DB_SCHEMA}.kpler_trades
            ),
            supplier_countries AS (
                SELECT DISTINCT origin_country_name
                FROM {DB_SCHEMA}.kpler_trades kt
                CROSS JOIN latest_timestamp
                WHERE kt.upload_timestamp_utc = latest_timestamp.max_ts
                    AND kt.destination_country_name IN :destination_countries
                    AND kt.origin_country_name IS NOT NULL
                    AND kt.status = 'Delivered'
            ),
            """
            supplier_filter = "WHERE country_name IN (SELECT origin_country_name FROM supplier_countries)"
            params = {'destination_countries': tuple(normalized_destination_countries)}

        query = text(f"""
            WITH {supplier_ctes}combined_maintenance AS (
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
                {supplier_filter}
                GROUP BY
                    plant_name,
                    country_name,
                    lng_train_name_short,
                    year,
                    month,
                    year_actual_forecast
            )
            SELECT
                plant_name,
                country_name,
                lng_train_name_short,
                year,
                month,
                year_actual_forecast,
                total_mtpa,
                metric_comment
            FROM combined_maintenance
            ORDER BY country_name, plant_name, lng_train_name_short, year, month
        """)

        df = pd.read_sql(query, engine, params=params)
        if df.empty:
            return df

        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        return df
    except Exception:
        return pd.DataFrame()


def _store_importer_maintenance_raw_data(destination_countries, raw_data):
    normalized_destination_countries = normalize_destination_countries(destination_countries)
    return {
        'destination_countries': list(normalized_destination_countries),
        'raw_data': _store_dataframe(raw_data) if raw_data is not None and not raw_data.empty else None,
        'loaded_at': dt.datetime.now().isoformat(timespec='seconds'),
    }


def _load_importer_maintenance_raw_data(payload, destination_countries):
    normalized_destination_countries = list(normalize_destination_countries(destination_countries))
    if not payload or payload.get('destination_countries') != normalized_destination_countries:
        return pd.DataFrame()
    return _load_store_dataframe(payload, 'raw_data', date_columns=['date'])


def process_maintenance_periods_hierarchical(df, expanded_plants=None):
    """
    Process maintenance data into a country -> plant -> train hierarchy.
    Country rows are always shown; plant rows appear when the country is expanded;
    train rows appear when the plant is expanded.
    """
    if df.empty:
        return pd.DataFrame()

    try:
        current_date = pd.Timestamp.now()
        period_specs = _build_importer_maintenance_period_specs(current_date)
        expanded_plants = expanded_plants or []
        period_cols = [spec['id'] for spec in period_specs]

        train_data = []
        plant_totals = {}
        country_totals = {}

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

            if country not in country_totals:
                country_totals[country] = {col: 0 for col in period_cols}

            for spec in period_specs:
                label = spec['id']
                period_data = group_df[
                    (group_df['date'] >= spec['start'])
                    & (group_df['date'] <= spec['end'])
                ]
                days_in_period = (spec['end'] - spec['start']).days + 1
                # WoodMac maintenance rows are monthly annual-rate values, so convert via monthly MCM before averaging.
                total_monthly_mcm = (
                    pd.to_numeric(period_data['total_mtpa'], errors='coerce').sum()
                    * MCM_PER_MONTH_PER_MMTPA
                )
                avg_mcm_d = (
                    total_monthly_mcm / days_in_period
                    if days_in_period > 0
                    else 0
                )
                value = round(avg_mcm_d, 1)
                row[label] = value if value > 0 else None
                plant_totals[plant_key]['totals'][label] += value

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

        return pd.DataFrame(final_data)
    except Exception:
        return pd.DataFrame()


def create_maintenance_summary_table(df):
    """Create an expandable supplier maintenance summary table using the exporter-detail grid pattern."""
    if df.empty:
        return html.Div("No maintenance data available", className="no-data-message")

    try:
        period_specs = _build_importer_maintenance_period_specs()
        period_col_ids = [spec['id'] for spec in period_specs]

        columns = [
            {
                'name': 'Supplier Country',
                'id': 'Supplier Country',
                'type': 'text',
                'cellClass': 'maintenance-label-cell maintenance-supplier-label-cell',
                'headerClass': 'maintenance-header-label maintenance-header-supplier',
            },
            {
                'name': 'Plant',
                'id': 'Plant',
                'type': 'text',
                'cellClass': 'maintenance-label-cell maintenance-plant-label-cell',
                'headerClass': 'maintenance-header-label maintenance-header-plant',
            },
            {
                'name': 'Train',
                'id': 'Train',
                'type': 'text',
                'cellClass': 'maintenance-label-cell maintenance-train-label-cell',
                'headerClass': 'maintenance-header-label maintenance-header-train',
            },
        ]

        for spec in period_specs:
            family_class = _maintenance_period_family_class(spec['family'])
            columns.append({
                'name': spec['label'],
                'id': spec['id'],
                'type': 'text',
                'cellClass': (
                    'maintenance-period-cell maintenance-period-number-cell '
                    f'maintenance-period-{family_class}'
                ),
                'headerClass': (
                    'maintenance-period-header '
                    f'maintenance-period-header-{family_class}'
                ),
                'cellClassRules': _build_maintenance_cell_class_rules(),
                'cellStyle': _build_maintenance_cell_style_conditions(family_class),
            })

        columns.extend([
            {
                'name': 'Type',
                'id': 'Type',
                'type': 'text',
                'headerClass': 'maintenance-header-hidden',
            },
            {
                'name': 'PlantKey',
                'id': 'PlantKey',
                'type': 'text',
                'headerClass': 'maintenance-header-hidden',
            },
        ])

        display_df = df.copy()
        for col_id in period_col_ids:
            if col_id in display_df.columns:
                display_df[col_id] = display_df[col_id].apply(_format_table_value_max_one_decimal)
        display_df['__maintenance_row_id'] = [
            (
                f"{index}-{row.get('Type', '')}-"
                f"{_strip_maintenance_expand_marker(row.get('Supplier Country', ''))}-"
                f"{_strip_maintenance_expand_marker(row.get('Plant', ''))}-"
                f"{row.get('Train', '')}"
            )
            for index, row in display_df.iterrows()
        ]
        data = display_df.to_dict('records')

        legend = html.Div(
            [
                html.Span(
                    [html.Span(className='maintenance-legend-swatch maintenance-legend-realized'), 'Realized'],
                    className='maintenance-legend-item'
                ),
                html.Span(
                    [html.Span(className='maintenance-legend-swatch maintenance-legend-current'), 'Current month'],
                    className='maintenance-legend-item'
                ),
                html.Span(
                    [html.Span(className='maintenance-legend-swatch maintenance-legend-nearterm'), 'Near-term (0-3M)'],
                    className='maintenance-legend-item'
                ),
                html.Span(
                    [html.Span(className='maintenance-legend-swatch maintenance-legend-outlook'), 'Outlook (Q+2-Q+4)'],
                    className='maintenance-legend-item'
                ),
                html.Span(className='maintenance-legend-divider'),
                html.Span(
                    'Red = >=5 MCM/D impact',
                    className='maintenance-legend-item maintenance-legend-high-impact'
                ),
                html.Span('Click country or plant row to expand trains', className='maintenance-legend-note'),
            ],
            className='maintenance-summary-legend'
        )

        width_styles = _build_importer_detail_column_width_styles(
            data,
            columns,
            numeric_columns=set(period_col_ids),
            width_limits=IMPORTER_DETAIL_MAINTENANCE_WIDTH_LIMITS,
            default_numeric_limits=(58, 82),
            default_text_limits=(60, 180),
        )

        table = create_ag_grid_from_datatable(
            id={'type': 'imp-maintenance-expandable-table', 'index': 0},
            columns=columns,
            data=data,
            style_cell_conditional=width_styles,
            hidden_columns=['Type', 'PlantKey'],
            sort_action='none',
            page_action='none',
            fill_width=False,
            height='auto',
            defaultColDef=IMPORTER_DETAIL_PERIOD_DEFAULT_COL_DEF,
            dashGridOptions={
                **IMPORTER_DETAIL_PERIOD_GRID_OPTIONS,
                'rowHeight': 28,
                'headerHeight': 31,
            },
            rowClassRules={
                'maintenance-summary-total-row': "params.data && params.data['Type'] === 'total'",
                'maintenance-summary-country-row': "params.data && params.data['Type'] === 'country'",
                'maintenance-summary-plant-row': "params.data && params.data['Type'] === 'plant'",
                'maintenance-summary-train-row': "params.data && params.data['Type'] === 'train'",
            },
            getRowId="params.data.__maintenance_row_id",
            className='importer-detail-grid importer-detail-maintenance-grid',
        )
        return html.Div([legend, table])

    except Exception as e:
        return html.Div(f"Error creating table: {str(e)}", className="error-message")


clientside_callback(
    """
    function(children) {
        function numericValue(text) {
            var cleaned = String(text || '').replace(/[^0-9.\\-]/g, '');
            if (!cleaned) {
                return NaN;
            }
            return Number(cleaned);
        }

        function applyImporterMaintenanceStyles() {
            var grid = document.querySelector('.ag-theme-alpine.mckinsey-ag-grid.importer-detail-maintenance-grid');
            if (!grid) {
                return 0;
            }

            var palette = {
                high: {backgroundColor: '#f2b5bd', color: '#7f1d1d', fontWeight: '860'},
                realized: {backgroundColor: '#dce9f8', color: '#1e40af', fontWeight: '720'},
                current: {backgroundColor: '#cfe4ff', color: '#1d4ed8', fontWeight: '780'},
                neartermActive: {backgroundColor: '#fff1c8', color: '#92400e', fontWeight: '720'},
                neartermWatch: {backgroundColor: '#f7dc97', color: '#78350f', fontWeight: '800'},
                outlookActive: {backgroundColor: '#e4eaf2', color: '#374151', fontWeight: '720'},
                outlookWatch: {backgroundColor: '#d5dde8', color: '#1f2937', fontWeight: '780'}
            };

            var styledCount = 0;
            grid.querySelectorAll('.maintenance-period-cell').forEach(function(cell) {
                cell.style.backgroundColor = '';
                cell.style.color = '';
                cell.style.fontWeight = '';

                var row = cell.closest('.ag-row');
                if (!row || row.classList.contains('maintenance-summary-total-row')) {
                    return;
                }

                var value = numericValue(cell.textContent);
                if (!Number.isFinite(value) || value <= 0) {
                    return;
                }

                var style = null;
                if (value >= 5) {
                    style = palette.high;
                } else if (
                    cell.classList.contains('maintenance-period-historical-quarter')
                    || cell.classList.contains('maintenance-period-historical-month')
                ) {
                    style = palette.realized;
                } else if (cell.classList.contains('maintenance-period-current-month')) {
                    style = palette.current;
                } else if (
                    cell.classList.contains('maintenance-period-nearterm-month')
                    || cell.classList.contains('maintenance-period-nearterm-quarter')
                ) {
                    style = value >= 1 ? palette.neartermWatch : palette.neartermActive;
                } else if (cell.classList.contains('maintenance-period-outlook-quarter')) {
                    style = value >= 1 ? palette.outlookWatch : palette.outlookActive;
                }

                if (style) {
                    Object.assign(cell.style, style);
                    styledCount += 1;
                }
            });

            if (!grid.dataset.importerMaintenanceStyleBound) {
                var schedule = function() {
                    window.setTimeout(applyImporterMaintenanceStyles, 0);
                    window.setTimeout(applyImporterMaintenanceStyles, 160);
                };
                grid.addEventListener('scroll', schedule, true);
                grid.addEventListener('wheel', schedule, true);
                grid.dataset.importerMaintenanceStyleBound = '1';
            }
            return styledCount;
        }

        window.setTimeout(applyImporterMaintenanceStyles, 0);
        window.setTimeout(applyImporterMaintenanceStyles, 250);
        window.setTimeout(applyImporterMaintenanceStyles, 800);
        return Date.now();
    }
    """,
    Output('imp-maintenance-style-refresh-store', 'data'),
    Input('imp-maintenance-summary-container', 'children'),
)


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

    return create_ag_grid_from_datatable(
        id={'type': 'imp-origin-forecast-expandable-table', 'index': 'summary'},
        data=display_df.to_dict('records'),
        columns=columns,
        style_data_conditional=conditional_styles,
        sort_action='native',
        page_size=50,
        fill_width=False,
        className='importer-detail-grid importer-period-grid importer-detail-forecast-grid'
    )


@callback(
    Output('imp-origin-forecast-summary-header', 'children'),
    Output('imp-origin-forecast-summary-subtitle', 'children'),
    Output('imp-origin-forecast-summary-table-container', 'children'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-origin-forecast-expanded-continents', 'data'),
    Input('imp-destination-catalog-store', 'data'),
    Input('imp-origin-level-dropdown', 'value'),
    Input('imp-volume-metric-dropdown', 'value'),
    prevent_initial_call=False
)
def update_origin_forecast_summary_table(selected_destination_aggregation, selected_destination,
                                         expanded_continents, destination_catalog, origin_level, volume_metric):
    vol_label = get_volume_metric_info(volume_metric)['label']
    header_text = f'Origin Forecast Allocation Summary (WoodMac, {vol_label})'
    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    if not destination_context['destination_countries']:
        return (
            header_text,
            "Modeled supplier allocation from SQL outputs.",
            html.Div("Please select a destination.", style={'textAlign': 'center', 'padding': '20px'})
        )
    try:
        expanded_continents = expanded_continents or []
        summary_df, footer_rows, run_metadata = fetch_origin_forecast_summary_data(
            engine,
            destination_context['destination_countries'],
            origin_level=origin_level or DEFAULT_IMPORTER_ORIGIN_LEVEL
        )
        subtitle = format_supply_allocation_run_subtitle(run_metadata)
        if run_metadata is None:
            return (
                header_text,
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
        display_df = _convert_detail_period_display_df(
            display_df,
            volume_metric,
            exclude_columns=['Continent', 'Country']
        )
        if display_df.empty:
            return (
                header_text,
                subtitle,
                html.Div(
                    f"No WoodMac origin forecast allocation data is available for {destination_context['display_label']}.",
                    style={'textAlign': 'center', 'padding': '20px'}
                )
            )

        return header_text, subtitle, create_origin_forecast_summary_table(display_df)
    except Exception as e:
        return (
            header_text,
            "Modeled supplier allocation from SQL outputs.",
            html.Div(
                f"Error loading data: {str(e)}",
                style={'textAlign': 'center', 'padding': '20px', 'color': 'red'}
            )
        )


@callback(
    Output('imp-origin-summary-table-container', 'children'),
    Output('imp-origin-summary-header', 'children'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-supply-rolling-window-input', 'value'),
    Input('imp-origin-expanded-continents', 'data'),
    Input('imp-destination-catalog-store', 'data'),
    Input('imp-origin-level-dropdown', 'value'),
    Input('imp-volume-metric-dropdown', 'value'),
    Input('imp-period-comparison-basis', 'value'),
    Input('imp-period-quarter-count-dropdown', 'value'),
    Input('imp-period-month-count-dropdown', 'value'),
    Input('imp-period-week-count-dropdown', 'value'),
    prevent_initial_call=False
)
def update_origin_summary_table(selected_destination_aggregation, selected_destination, rolling_window_days,
                                expanded_continents, destination_catalog, origin_level, volume_metric,
                                comparison_basis, quarter_count, month_count, week_count):
    vol_label = get_volume_metric_info(volume_metric)['label']
    header_text = f'Origin Analysis Summary ({vol_label})'
    comparison_basis = _normalize_detail_comparison_basis(comparison_basis)
    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    if not destination_context['destination_countries']:
        return html.Div("Please select a destination.", style={'textAlign': 'center', 'padding': '20px'}), header_text
    try:
        expanded_continents = expanded_continents or []
        df = fetch_origin_summary_data(
            engine,
            destination_context['destination_countries'],
            rolling_window_days,
            origin_level=origin_level or DEFAULT_IMPORTER_ORIGIN_LEVEL,
            selected_destination_aggregation=selected_destination_aggregation,
            selected_destination_value=selected_destination,
            quarter_count=quarter_count,
            month_count=month_count,
            week_count=week_count,
            include_comparison_reference_columns=comparison_basis != 'levels'
        )
        if df.empty:
            return html.Div("No data available for the selected filters.", style={'textAlign': 'center', 'padding': '20px'}), header_text

        display_df = prepare_origin_table_for_display(df, expanded_continents)
        if 'Continent' in display_df.columns:
            display_df['Continent'] = display_df['Continent'].replace('GRAND TOTAL', 'Global')
        display_df, comparison_metadata = _filter_detail_period_display_columns(
            display_df,
            comparison_basis,
            quarter_count,
            month_count,
            week_count,
            return_metadata=True
        )
        display_df = _convert_detail_period_display_df(
            display_df,
            volume_metric,
            rolling_window_days=rolling_window_days,
            exclude_columns=['Continent', 'Country']
        )
        display_df, comparison_delta_cols = _apply_exporter_detail_period_comparison(
            display_df,
            comparison_metadata
        )
        col_display_names = {'Continent': 'Origin Level', 'Country': 'Country'}
        columns = []
        period_numeric_format = (
            {'specifier': ',.1f'}
            if _normalize_detail_volume_metric(volume_metric) == 'mt'
            else {'specifier': '.0f'}
        )
        for col in display_df.columns:
            if col in ['Continent', 'Country']:
                columns.append({'name': col_display_names.get(col, col), 'id': col, 'type': 'text'})
            else:
                columns.append({
                    'name': col,
                    'id': col,
                    'type': 'numeric',
                    'format': period_numeric_format,
                })

        columns = _apply_exporter_detail_period_column_classes(
            columns,
            text_columns={'Continent', 'Country'},
            primary_text_columns={'Continent'},
            delta_like_cols=comparison_delta_cols
        )
        columns = _apply_exporter_detail_period_delta_heatmap_class_rules(
            columns,
            display_df,
            total_column='Continent',
            delta_like_cols=comparison_delta_cols
        )
        period_value_styles = _build_exporter_detail_period_value_styles(
            display_df,
            text_columns={'Continent', 'Country'},
            total_column='Continent',
            subtotal_column='Country',
            delta_like_cols=comparison_delta_cols
        )
        grid_display_df, grid_columns = _build_exporter_detail_period_grid_display(
            display_df,
            columns,
            delta_like_cols=comparison_delta_cols
        )
        period_numeric_columns = {column.get('id') for column in columns if column.get('type') == 'numeric'}
        width_styles = _build_importer_detail_column_width_styles(
            grid_display_df,
            grid_columns,
            numeric_columns=period_numeric_columns,
            width_limits=IMPORTER_DETAIL_PERIOD_WIDTH_LIMITS,
            default_numeric_limits=(58, 90),
            default_text_limits=(82, 190),
        )

        table = create_ag_grid_from_datatable(
            id={'type': 'imp-origin-expandable-table', 'index': 'summary'},
            data=grid_display_df.to_dict('records'),
            columns=grid_columns,
            page_action='none',
            sort_action='none',
            fill_width=False,
            export_format='none',
            className='importer-detail-grid importer-period-grid supply-dest-summary-grid',
            height='auto',
            defaultColDef=IMPORTER_DETAIL_PERIOD_DEFAULT_COL_DEF,
            dashGridOptions=IMPORTER_DETAIL_PERIOD_GRID_OPTIONS,
            style_cell_conditional=width_styles,
            style_data_conditional=period_value_styles,
            rowClassRules={
                'supply-dest-summary-total-row': (
                    "params.data && params.data['Continent'] === 'Global'"
                ),
                'supply-dest-summary-subtotal-row': (
                    "params.data && (params.data['Country'] === 'Total' "
                    "|| (params.data['Continent'] && "
                    "(String(params.data['Continent']).startsWith('▶') "
                    "|| String(params.data['Continent']).startsWith('▼'))))"
                ),
            }
        )
        return table, header_text
    except Exception as e:
        return html.Div(f"Error loading data: {str(e)}", style={'textAlign': 'center', 'padding': '20px', 'color': 'red'}), header_text


@callback(
    Output('imp-route-analysis-kpi-container', 'children'),
    Output('imp-graph-route-suez-only', 'figure'),
    Input('imp-route-aggregation-dropdown', 'value'),
    Input('imp-origin-level-dropdown', 'value'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-destination-catalog-store', 'data')
)
def update_route_analysis_charts_and_tables(agg_level, origin_level, selected_destination_aggregation,
                                            selected_destination, destination_catalog):
    origin_level = origin_level or DEFAULT_IMPORTER_ORIGIN_LEVEL
    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    if not destination_context['destination_countries']:
        return (
            _empty_importer_detail_state("Please select a destination."),
            _empty_route_analysis_figure("No destination selected")
        )

    try:
        processed_df = process_trade_and_distance_data(
            engine,
            destination_countries=destination_context['destination_countries']
        )
        if processed_df is None or processed_df.empty:
            return (
                _empty_importer_detail_state("No route-analysis data available"),
                _empty_route_analysis_figure(
                    "No route-analysis data available",
                    "The selected destination does not return delivered voyages with distance alternatives."
                )
            )

        if origin_level not in processed_df.columns and origin_level == 'origin_shipping_region':
            region_map_df = pd.read_sql(
                text(f"SELECT DISTINCT country, shipping_region FROM {DB_SCHEMA}.mappings_country"),
                engine
            ).rename(columns={'country': 'origin_country_name', 'shipping_region': 'origin_shipping_region'})
            processed_df = pd.merge(processed_df, region_map_df, how='left', on='origin_country_name')
        elif origin_level not in processed_df.columns:
            level_col_map = {
                'continent_origin_name': 'continent',
                'origin_basin': 'basin',
                'origin_subcontinent': 'subcontinent',
                'origin_classification_level1': 'country_classification_level1',
                'origin_classification': 'country_classification',
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
    except Exception:
        return (
            _empty_importer_detail_state("Unable to load route KPIs."),
            _empty_route_analysis_figure("Error processing route-analysis data")
        )

    if origin_level not in processed_df.columns:
        error_msg = f"Selected origin level column '{origin_level}' not found in processed data."
        return _empty_importer_detail_state(error_msg), _empty_route_analysis_figure(error_msg)

    for col in ['origin_country_name', 'origin_shipping_region']:
        if col not in processed_df.columns:
            processed_df[col] = None

    try:
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
    except KeyError as e:
        return (
            _empty_importer_detail_state(f"Missing route column: {e}"),
            _empty_route_analysis_figure("Missing route-analysis column", str(e))
        )
    except Exception:
        return (
            _empty_importer_detail_state("Unable to filter route-analysis data."),
            _empty_route_analysis_figure("Error filtering route-analysis data")
        )

    try:
        scenarios = [
            {
                'title': 'Suez available, Panama unavailable',
                'display_title': 'Suez available',
                'subtitle': 'Route mix where the Suez alternative exists',
                'canal_routes': ['ViaSuez'],
                'canal_label': 'Suez share',
                'df': df_suez_only,
            },
            {
                'title': 'Panama available, Suez unavailable',
                'display_title': 'Panama available',
                'subtitle': 'Route mix where the Panama alternative exists',
                'canal_routes': ['ViaPanama'],
                'canal_label': 'Panama share',
                'df': df_panama_only,
            },
            {
                'title': 'Suez and Panama available',
                'display_title': 'Both canals available',
                'subtitle': 'Route mix where both canal alternatives exist',
                'canal_routes': ['ViaSuez', 'ViaPanama'],
                'canal_label': 'Canal share',
                'df': df_both,
            },
        ]

        for scenario in scenarios:
            frame, routes = _build_route_analysis_panel_frame(scenario['df'], agg_level)
            scenario['frame'] = frame
            scenario['routes'] = routes
            scenario['voyages'] = int(frame['total_voyages'].sum()) if not frame.empty else 0
            scenario['periods'] = int(frame['period_label'].nunique()) if not frame.empty else 0

        if not any(scenario['voyages'] for scenario in scenarios):
            empty_message = "No route-analysis data available"
            return _empty_importer_detail_state(empty_message), _empty_route_analysis_figure(
                "No route-analysis data available",
                "The selected destination and aggregation do not return delivered voyages with distance alternatives."
            )

        return _build_route_analysis_kpi_cards(scenarios, agg_level), _route_analysis_legacy_hidden_figure()
    except Exception as e:
        return (
            _empty_importer_detail_state("Unable to generate route KPIs."),
            _empty_route_analysis_figure("Error generating route-analysis charts", str(e))
        )


@callback(
    Output('imp-download-route-analysis-excel', 'data'),
    Input('imp-export-route-analysis-button', 'n_clicks'),
    State('imp-route-aggregation-dropdown', 'value'),
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
    origin_level = origin_level or DEFAULT_IMPORTER_ORIGIN_LEVEL

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
    except Exception:
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


def _empty_diversion_analysis_figure(message="No diversions data available"):
    fig = go.Figure()
    fig.update_layout(
        height=472,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin=dict(l=36, r=18, t=38, b=34),
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=13, color='#64748b')
            )
        ]
    )
    return fig


def _append_column_class(column_def, class_name, key='cellClass'):
    existing_class = str(column_def.get(key) or '').strip()
    classes = existing_class.split() if existing_class else []
    if class_name not in classes:
        classes.append(class_name)
    column_def[key] = ' '.join(classes)


def _style_diversion_column_defs(column_defs):
    for column_def in column_defs or []:
        children = column_def.get('children')
        if children:
            _style_diversion_column_defs(children)
            continue

        field = str(column_def.get('field') or '')
        if not field:
            continue

        _append_column_class(column_def, 'diversion-table-header', key='headerClass')
        column_def['tooltipValueGetter'] = {
            'function': "params.value === null || params.value === undefined ? '' : String(params.value)"
        }

        if field in DIVERSION_NUMERIC_TABLE_COLUMNS:
            _append_column_class(column_def, 'diversion-number-cell')
            _append_column_class(column_def, 'diversion-number-header', key='headerClass')
            column_def['valueFormatter'] = {
                'function': "params.value !== null && params.value !== undefined && params.value !== '' ? d3.format(',.0f')(Number(params.value)) : ''"
            }
        elif field in DIVERSION_DATE_TABLE_COLUMNS:
            _append_column_class(column_def, 'diversion-date-cell')
            _append_column_class(column_def, 'diversion-date-header', key='headerClass')
        elif field in DIVERSION_ROUTE_TABLE_COLUMNS:
            _append_column_class(column_def, 'diversion-route-cell')
        elif field == 'State':
            _append_column_class(column_def, 'diversion-state-cell')
        elif field in {'Vessel', 'Charterer'}:
            _append_column_class(column_def, 'diversion-identity-cell')

        if field in {'Diversion date', 'Vessel'}:
            column_def.update({'pinned': 'left', 'lockPinned': True, 'suppressMovable': True})
        if field == 'Diversion date':
            column_def.update({'sort': 'desc', 'sortIndex': 0})

    return column_defs


def _prepare_diversion_table_records(records, row_limit=None):
    prepared_records = []
    for row in records or []:
        formatted_row = dict(row)
        for column_id in DIVERSION_NUMERIC_TABLE_COLUMNS:
            if column_id in formatted_row:
                formatted_row[column_id] = _round_table_value_max_one_decimal(formatted_row[column_id])
        prepared_records.append(formatted_row)

    if not prepared_records:
        return []

    frame = pd.DataFrame(prepared_records)
    if 'Diversion date' in frame.columns:
        frame['_diversion_sort_date'] = pd.to_datetime(frame['Diversion date'], errors='coerce')
        frame = (
            frame
            .sort_values('_diversion_sort_date', ascending=False, kind='mergesort')
            .drop(columns=['_diversion_sort_date'])
        )
    if row_limit is not None:
        frame = frame.head(int(row_limit))
    return frame.to_dict('records')


def _build_diversion_table_columns(records):
    if not records:
        return [{"name": "No Data", "id": "no_data"}]

    columns = []
    for col in records[0].keys():
        column = {"name": col, "id": col}
        if col in DIVERSION_NUMERIC_TABLE_COLUMNS:
            column.update({'type': 'numeric', 'format': {'specifier': ',.0f'}})
        columns.append(column)
    return columns


def _build_diversion_analysis_figure(df_kpler_charts, combo_field):
    if df_kpler_charts is None or df_kpler_charts.empty:
        return _empty_diversion_analysis_figure()
    if combo_field not in df_kpler_charts.columns:
        return _empty_diversion_analysis_figure("No diversion route grouping available")

    chart_df = df_kpler_charts.copy()
    chart_df['Diversion_month'] = pd.to_datetime(chart_df.get('Diversion_month'), errors='coerce')
    chart_df = chart_df.dropna(subset=['Diversion_month'])
    if chart_df.empty:
        return _empty_diversion_analysis_figure()

    chart_df[combo_field] = chart_df[combo_field].fillna('Unknown -> Unknown').astype(str)
    chart_df['Added shipping days'] = pd.to_numeric(chart_df.get('Added shipping days'), errors='coerce').fillna(0.0)
    chart_df['Cubic Meters'] = pd.to_numeric(chart_df.get('Cubic Meters'), errors='coerce').fillna(0.0)

    raw_combo_totals = (
        chart_df
        .groupby(combo_field, as_index=False)
        .size()
        .rename(columns={'size': 'Count'})
        .sort_values(['Count', combo_field], ascending=[False, True], kind='mergesort')
    )
    raw_combo_order = raw_combo_totals[combo_field].tolist()
    max_series = DIVERSION_CHART_MAX_SERIES_BY_LEVEL.get(combo_field, 12)
    if len(raw_combo_order) > max_series:
        top_combos = raw_combo_order[:max_series - 1]
        chart_df['_diversion_combo_display'] = np.where(
            chart_df[combo_field].isin(top_combos),
            chart_df[combo_field],
            'Other routes'
        )
        combo_field = '_diversion_combo_display'
        combo_order = top_combos + ['Other routes']
    else:
        combo_order = raw_combo_order

    df_count = chart_df.groupby(['Diversion_month', combo_field]).size().reset_index(name='Count')
    df_days = chart_df.groupby(['Diversion_month', combo_field], as_index=False)['Added shipping days'].sum()
    df_volumes = chart_df.groupby(['Diversion_month', combo_field], as_index=False)['Cubic Meters'].sum()

    if not combo_order:
        return _empty_diversion_analysis_figure()

    color_mapping = {
        combo: (
            DIVERSION_CHART_OTHER_COLOR
            if combo == 'Other routes'
            else DIVERSION_CHART_COLOR_SEQUENCE[index % len(DIVERSION_CHART_COLOR_SEQUENCE)]
        )
        for index, combo in enumerate(combo_order)
    }
    metric_specs = [
        {
            'title': 'Diversions',
            'data': df_count,
            'value': 'Count',
            'axis': 'Count',
            'hover_label': 'Diversions',
            'hover_suffix': '',
        },
        {
            'title': 'Added Shipping Days',
            'data': df_days,
            'value': 'Added shipping days',
            'axis': 'Days',
            'hover_label': 'Added days',
            'hover_suffix': ' days',
        },
        {
            'title': 'Cargo Volume',
            'data': df_volumes,
            'value': 'Cubic Meters',
            'axis': 'm3 LNG',
            'hover_label': 'Cargo',
            'hover_suffix': ' m3',
        },
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[spec['title'] for spec in metric_specs],
        horizontal_spacing=0.055
    )

    for metric_index, spec in enumerate(metric_specs, start=1):
        metric_df = spec['data']
        value_col = spec['value']
        for combo_index, combo in enumerate(combo_order):
            combo_data = metric_df[metric_df[combo_field] == combo].sort_values('Diversion_month')
            if combo_data.empty:
                continue
            fig.add_trace(
                go.Bar(
                    x=combo_data['Diversion_month'],
                    y=combo_data[value_col],
                    name=combo,
                    legendgroup=combo,
                    legendrank=combo_index,
                    showlegend=metric_index == 1,
                    marker=dict(
                        color=color_mapping[combo],
                        line=dict(color='rgba(255,255,255,0.55)', width=0.4)
                    ),
                    opacity=0.9,
                    hovertemplate=(
                        f'<b>{combo}</b><br>'
                        'Month: %{x|%b %Y}<br>'
                        f"{spec['hover_label']}: %{{y:,.0f}}{spec['hover_suffix']}"
                        '<extra></extra>'
                    )
                ),
                row=1,
                col=metric_index
            )

    fig.update_layout(
        barmode='stack',
        bargap=0.22,
        height=472,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        hovermode='closest',
        font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=11,
            color='#475569'
        ),
        margin=dict(l=48, r=18, t=46, b=92),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.13,
            xanchor='left',
            x=0,
            bgcolor='rgba(255,255,255,0)',
            bordercolor='rgba(255,255,255,0)',
            font=dict(size=10, color='#475569'),
            itemsizing='constant',
            itemwidth=38,
            tracegroupgap=0,
        ),
    )
    fig.update_annotations(
        font=dict(size=12, color='#0f172a', family='Inter, -apple-system, BlinkMacSystemFont, sans-serif')
    )
    fig.update_xaxes(
        title_text='',
        tickformat='%b<br>%Y',
        nticks=7,
        showgrid=False,
        linecolor='rgba(148, 163, 184, 0.55)',
        linewidth=1,
        tickfont=dict(size=10, color='#64748b'),
        zeroline=False,
    )
    for col, spec in enumerate(metric_specs, start=1):
        fig.update_yaxes(
            title_text=spec['axis'],
            title_font=dict(size=11, color='#475569'),
            tickfont=dict(size=10, color='#64748b'),
            tickformat='~s' if col == 3 else ',.0f',
            gridcolor='rgba(148, 163, 184, 0.20)',
            gridwidth=0.5,
            linecolor='rgba(148, 163, 184, 0.45)',
            linewidth=1,
            zeroline=True,
            zerolinecolor='rgba(148, 163, 184, 0.25)',
            row=1,
            col=col
        )
    return fig


@callback(
    Output('imp-diversion-processed-data', 'data'),
    Input('global-refresh-button', 'n_clicks'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-destination-catalog-store', 'data'),
)
def process_diversion_data(_n_clicks, selected_destination_aggregation, selected_destination, destination_catalog):
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
    if df_kpler_charts.empty:
        return {
            'main_data': data_kpler_diversions,
            'charts_data': [],
            'destination_label': destination_context['display_label']
        }

    df_mapping_country = pd.read_sql(
        text(f"""
            SELECT country, basin, shipping_region
            FROM {DB_SCHEMA}.mappings_country
        """),
        engine
    )
    df_mapping_location = pd.read_sql(
        text(f"""
            SELECT destination_location_name, basin, shipping_region
            FROM {DB_SCHEMA}.mapping_destination_location_name
        """),
        engine
    )
    charts_df = build_importer_diversion_chart_dataframe(df_kpler_charts, df_mapping_country, df_mapping_location)

    return {
        'main_data': data_kpler_diversions,
        'charts_data': charts_df.to_dict('records'),
        'destination_label': destination_context['display_label']
    }


@callback(
    Output('imp-diversion-table', 'rowData'),
    Output('imp-diversion-table', 'columnDefs'),
    Output('imp-diversion-count-chart', 'figure'),
    Input('imp-diversion-processed-data', 'data'),
    Input('imp-diversion-combo-radio', 'value'),
)
def update_diversion_ui(stored_data, combo_level):
    if not stored_data:
        empty_columns = [{"name": "No Data", "id": "no_data"}]
        empty_defs = _style_diversion_column_defs(datatable_columns_to_ag_grid_column_defs(empty_columns))
        return [], empty_defs, _empty_diversion_analysis_figure()

    data_kpler_diversions = stored_data.get('main_data') or []
    diversion_table_data = _prepare_diversion_table_records(
        data_kpler_diversions,
        row_limit=DIVERSION_DASHBOARD_ROW_LIMIT
    )
    diversion_columns = _build_diversion_table_columns(diversion_table_data)

    diversion_column_defs = datatable_columns_to_ag_grid_column_defs(diversion_columns)
    diversion_width_styles = _build_importer_detail_column_width_styles(
        diversion_table_data,
        diversion_columns,
        numeric_columns=DIVERSION_NUMERIC_TABLE_COLUMNS,
        width_limits=IMPORTER_DETAIL_DIVERSION_WIDTH_LIMITS,
        default_numeric_limits=(74, 112),
        default_text_limits=(74, 166),
    )
    diversion_column_defs = _apply_importer_detail_column_widths_to_defs(
        diversion_column_defs,
        diversion_width_styles
    )
    diversion_column_defs = _style_diversion_column_defs(diversion_column_defs)

    df_kpler_charts = pd.DataFrame(stored_data.get('charts_data') or [])
    combo_field = combo_level if combo_level in df_kpler_charts.columns else 'basin_combo'
    fig = _build_diversion_analysis_figure(df_kpler_charts, combo_field)

    return diversion_table_data, diversion_column_defs, fig


@callback(
    Output('imp-download-diversion-summary-excel', 'data'),
    Input('imp-export-diversion-summary-button', 'n_clicks'),
    State('imp-diversion-processed-data', 'data'),
    State('imp-diversion-table', 'columnDefs'),
    prevent_initial_call=True
)
def export_importer_diversion_summary_to_excel(n_clicks, stored_data, table_columns):
    if not n_clicks or not stored_data:
        raise PreventUpdate
    table_data = _prepare_diversion_table_records(stored_data.get('main_data') or [])
    if not table_data:
        raise PreventUpdate
    df = pd.DataFrame(table_data)
    table_columns = ag_grid_column_defs_to_datatable_columns(table_columns)
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
    [Input({'type': 'imp-origin-expandable-table', 'index': ALL}, 'cellClicked')],
    [State({'type': 'imp-origin-expandable-table', 'index': ALL}, 'virtualRowData'),
     State('imp-origin-expanded-continents', 'data')],
    prevent_initial_call=True
)
def toggle_origin_continent_expansion(active_cells, table_data_list, expanded_continents):
    if not any(active_cells):
        return expanded_continents or []

    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    if 'imp-origin-expandable-table' in prop_id and '.cellClicked' in prop_id:
        active_cell = ag_grid_cell_clicked_to_active_cell(active_cells[0])
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
    [Input({'type': 'imp-origin-forecast-expandable-table', 'index': ALL}, 'cellClicked')],
    [State({'type': 'imp-origin-forecast-expandable-table', 'index': ALL}, 'virtualRowData'),
     State('imp-origin-forecast-expanded-continents', 'data')],
    prevent_initial_call=True
)
def toggle_origin_forecast_continent_expansion(active_cells, table_data_list, expanded_continents):
    if not any(active_cells):
        return expanded_continents or []

    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    if 'imp-origin-forecast-expandable-table' in prop_id and '.cellClicked' in prop_id:
        active_cell = ag_grid_cell_clicked_to_active_cell(active_cells[0])
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
    Output('imp-maintenance-summary-header', 'children'),
    Output('imp-maintenance-raw-data-store', 'data'),
    Input('imp-destination-aggregation-dropdown', 'value'),
    Input('imp-destination-country-dropdown', 'value'),
    Input('imp-destination-catalog-store', 'data'),
    Input('imp-volume-metric-dropdown', 'value'),
    State('imp-maintenance-expanded-plants', 'data')
)
def update_maintenance_table(selected_destination_aggregation, selected_destination, destination_catalog,
                             volume_metric, expanded_plants):
    vol_label = get_volume_metric_info(volume_metric)['label']
    header_text = f'Supplier Maintenance Schedule ({vol_label.upper()} Impact)'
    destination_context = resolve_destination_context(
        selected_destination_aggregation,
        selected_destination,
        destination_catalog
    )
    destination_label = destination_context['display_label']
    if not destination_context['destination_countries']:
        return (
            html.Div("Please select a destination.", style={'textAlign': 'center', 'padding': '20px'}),
            header_text,
            _store_importer_maintenance_raw_data([], pd.DataFrame()),
        )

    try:
        raw_data = fetch_train_maintenance_data(engine, destination_context['destination_countries'])
        raw_data_store = _store_importer_maintenance_raw_data(
            destination_context['destination_countries'],
            raw_data
        )
        if raw_data.empty:
            return (
                html.Div(
                    f"No supplier maintenance data available for cargoes serving {destination_label}.",
                    style={'textAlign': 'center', 'padding': '20px'}
                ),
                header_text,
                raw_data_store,
            )

        processed_data = process_maintenance_periods_hierarchical(raw_data, expanded_plants)
        if processed_data.empty:
            return (
                html.Div("No maintenance data to display.", style={'textAlign': 'center', 'padding': '20px'}),
                header_text,
                raw_data_store,
            )

        period_specs = _build_importer_maintenance_period_specs()
        period_days_by_column = {
            spec['id']: (spec['end'] - spec['start']).days + 1
            for spec in period_specs
        }
        processed_data = convert_volume_metric_dataframe(
            processed_data,
            volume_metric,
            columns=list(period_days_by_column.keys()),
            period_days_by_column=period_days_by_column,
            precision=1
        )
        return create_maintenance_summary_table(processed_data), header_text, raw_data_store
    except Exception as e:
        return (
            html.Div(
                f"Error loading maintenance data: {str(e)}",
                style={'textAlign': 'center', 'padding': '20px', 'color': 'red'}
            ),
            header_text,
            _store_importer_maintenance_raw_data(destination_context['destination_countries'], pd.DataFrame()),
        )


@callback(
    Output('imp-maintenance-expanded-plants', 'data', allow_duplicate=True),
    Output('imp-maintenance-summary-container', 'children', allow_duplicate=True),
    [Input({'type': 'imp-maintenance-expandable-table', 'index': ALL}, 'cellClicked')],
    [State({'type': 'imp-maintenance-expandable-table', 'index': ALL}, 'virtualRowData'),
     State('imp-maintenance-expanded-plants', 'data'),
     State('imp-destination-aggregation-dropdown', 'value'),
     State('imp-destination-country-dropdown', 'value'),
     State('imp-destination-catalog-store', 'data'),
     State('imp-volume-metric-dropdown', 'value'),
     State('imp-maintenance-raw-data-store', 'data')],
    prevent_initial_call=True
)
def toggle_maintenance_plant_expansion(active_cells, table_data_list, expanded_plants,
                                       selected_destination_aggregation, selected_destination,
                                       destination_catalog, volume_metric, maintenance_raw_data):
    if not any(active_cells):
        raise PreventUpdate

    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    if 'imp-maintenance-expandable-table' not in prop_id or '.cellClicked' not in prop_id:
        raise PreventUpdate

    try:
        active_cell = ag_grid_cell_clicked_to_active_cell(active_cells[0])
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
        raw_data = _load_importer_maintenance_raw_data(
            maintenance_raw_data,
            destination_context['destination_countries']
        )
        if raw_data.empty:
            raw_data = fetch_train_maintenance_data(engine, destination_context['destination_countries'])
        if raw_data.empty:
            return expanded_plants, no_update

        processed_data = process_maintenance_periods_hierarchical(raw_data, expanded_plants)
        if processed_data.empty:
            return expanded_plants, no_update

        period_specs = _build_importer_maintenance_period_specs()
        period_days_by_column = {
            spec['id']: (spec['end'] - spec['start']).days + 1
            for spec in period_specs
        }
        processed_data = convert_volume_metric_dataframe(
            processed_data,
            volume_metric,
            columns=list(period_days_by_column.keys()),
            period_days_by_column=period_days_by_column,
            precision=1
        )
        return expanded_plants, create_maintenance_summary_table(processed_data)
    except PreventUpdate:
        raise
    except Exception:
        raise PreventUpdate
