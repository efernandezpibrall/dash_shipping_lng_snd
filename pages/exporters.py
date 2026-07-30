from dash import html, dcc, callback, Output, Input, State, ALL, callback_context
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate
from utils.ag_grid_tables import (
    ag_grid_cell_clicked_to_active_cell,
    ag_grid_column_defs_to_datatable_columns,
    create_ag_grid_from_datatable,
)
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime, timedelta
from io import BytesIO
import json
import zlib
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import text, bindparam
import calendar
import uuid

from utils.table_styles import StandardTableStyleManager, TABLE_COLORS
from utils.dashboard_snapshot_cache import (
    SnapshotUnavailable as _SnapshotUnavailable,
    build_source_key as _build_source_key,
    get_or_build_snapshot as _get_or_build_snapshot,
    is_snapshot_reference as _is_snapshot_reference,
    pack_record_mapping as _pack_record_mapping,
    resolve_snapshot as _resolve_snapshot,
    snapshot_is_resolvable as _snapshot_is_resolvable,
    unpack_record_mapping as _unpack_record_mapping,
    was_global_refresh_triggered as _was_global_refresh_triggered,
    with_snapshot_slot as _with_snapshot_slot,
)
from utils.database import DB_SCHEMA, engine
from utils.performance import log_callback_timing

# Month order constant for sorting (used in multiple functions)
MONTH_ORDER = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
               'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

SUPPLY_DEST_SUMMARY_COMPARISON_BASIS_OPTIONS = [
    {'label': 'None', 'value': 'levels'},
    {'label': 'vs Previous Period', 'value': 'previous_period'},
    {'label': 'vs Previous Year', 'value': 'same_period_last_year'}
]

SUPPLY_DEST_COUNTRY_GROUPING_OPTIONS = [
    {'label': 'Yes', 'value': 'group_small_countries'},
    {'label': 'No', 'value': 'show_all'}
]

SUPPLY_DEST_TEXT_COLUMNS = [
    'Aggregation Supply',
    'Aggregation Demand',
    'Country Demand',
    'Demand Country',
    'Import Country',
    'Import Classification',
    'Supply Country',
    'Supply Installation'
]
SUPPLY_DEST_TOTAL_LABELS = ('GRAND TOTAL', 'Global')
SUPPLY_DEST_PBD_REFERENCE_COLUMNS = ('30D_PBD', '7D_PBD')
SUPPLY_DEST_PBD_DELTA_COLUMNS = ('Δ 30D vs PBD', 'Δ 7D vs PBD')
SUPPLY_DEST_DELTA_RAW_FIELDS = {
    'Δ 7D-30D': '__supply_dest_delta_7d_30d_raw',
    'Δ 30D Y/Y': '__supply_dest_delta_30d_yoy_raw',
    'Δ 30D vs PBD': '__supply_dest_delta_30d_pbd_raw',
    'Δ 7D vs PBD': '__supply_dest_delta_7d_pbd_raw',
}
SUPPLY_DEST_ROLLING_REFERENCE_COLUMNS = (
    '30D_PP',
    '30D_Y1',
    '7D_PP',
    '7D_Y1',
    *SUPPLY_DEST_PBD_REFERENCE_COLUMNS,
)
SUPPLY_DEST_SOURCE_TEXT_COLUMNS = [
    'supply_classification',
    'demand_classification',
    'supply_country',
    'demand_country',
    'supply_installation'
]
SUPPLY_DEST_DEFAULT_QUARTER_COUNT = 5
SUPPLY_DEST_DEFAULT_MONTH_COUNT = 3
SUPPLY_DEST_DEFAULT_WEEK_COUNT = 3
SUPPLY_DEST_DEFAULT_YEAR_COUNT = 0
SUPPLY_DEST_MAX_QUARTER_COUNT = 8
SUPPLY_DEST_MAX_MONTH_COUNT = 12
SUPPLY_DEST_MAX_WEEK_COUNT = 12
SUPPLY_DEST_MAX_YEAR_COUNT = 5
SUPPLY_DEST_PRELOAD_YEAR_COUNT = SUPPLY_DEST_MAX_YEAR_COUNT + 1
SUPPLY_DEST_PRELOAD_QUARTER_COUNT = SUPPLY_DEST_MAX_QUARTER_COUNT + 4
SUPPLY_DEST_PRELOAD_MONTH_COUNT = SUPPLY_DEST_MAX_MONTH_COUNT + 12
SUPPLY_DEST_PRELOAD_WEEK_COUNT = SUPPLY_DEST_MAX_WEEK_COUNT + 53

BCM_PER_MMTPA = 1.36
DAYS_PER_YEAR = 365.25
MCM_PER_BCM = 1000
MCM_PER_MT = BCM_PER_MMTPA * MCM_PER_BCM
MMTPA_PER_MCM_D = DAYS_PER_YEAR / MCM_PER_MT

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

SUPPLY_CHART_COLOR_SEQUENCE = [
    '#7a5195',
    '#ef5675',
    '#9f7aea',
    '#c44e52',
    '#d08b36',
    '#607d8b',
    '#1aa6a6',
    '#0f4c81',
]
SUPPLY_CHART_FORECAST_DASH = 'dot'
SUPPLY_CHART_ANCHOR_YEAR = 2024
SUPPLY_CHART_QUERY_START_DATE = '2020-11-01'
SUPPLY_CHART_DISPLAY_START_DATE = '2021-01-01'
SUPPLY_CHART_DEFAULT_SELECTED_YEAR_COUNT = 2
SUPPLY_CHART_DEFAULT_DESELECTED_YEARS = {'2024'}
SUPPLY_CHART_RANGE_LOOKBACK_YEARS = 5
SUPPLY_CHART_RANGE_FILL = 'rgba(148, 163, 184, 0.20)'
SUPPLY_CHART_VISIBLE_COUNTRIES = [
    "United States", "Australia", "Qatar", "Russian Federation",
    "Nigeria", "Angola", "Malaysia"
]
SUPPLY_CHART_REST_OF_COUNTRIES_LABEL = 'Rest of the Countries'

CONTINENT_CHART_COLOR_MAP = {
    'Africa': '#7a5195',
    'Americas': '#2f9e7e',
    'Asia': '#d64550',
    'Europe': '#2f6fbb',
    'Unknown': '#7b8794',
    'Oceania': '#d08b36',
    'Middle East': '#00a0b0',
    'North America': '#b83280',
    'South America': '#c7a12b'
}
CONTINENT_CHART_PREVIOUS_YEAR_WIDTH = 1.15
CONTINENT_CHART_CURRENT_YEAR_WIDTH = 2.8
CONTINENT_CHART_FORECAST_DASH = 'dot'
CONTINENT_CHART_ANCHOR_YEAR = 2024
CONTINENT_CHART_QUERY_START_DATE = '2020-11-01'
CONTINENT_CHART_DISPLAY_START_DATE = '2021-01-01'
CONTINENT_CHART_DEFAULT_SELECTED_YEAR_COUNT = 2
DEFAULT_SUPPLY_ROLLING_AVG_DAYS = 30
MIN_SUPPLY_ROLLING_AVG_DAYS = 1
MAX_SUPPLY_ROLLING_AVG_DAYS = 180


def _get_volume_metric_info(volume_metric):
    """Return conversion metadata for the selected overview volume metric."""
    return VOLUME_CONVERSIONS.get(volume_metric or 'mcm_d', VOLUME_CONVERSIONS['mcm_d'])


def _get_volume_metric_factor(volume_metric, period_days=None):
    """Return the mcm/d conversion factor for the selected display metric."""
    normalized_metric = volume_metric if volume_metric in VOLUME_CONVERSIONS else 'mcm_d'
    if normalized_metric == 'mt':
        days = period_days if period_days is not None else DAYS_PER_YEAR
        return float(days) / MCM_PER_MT
    if normalized_metric == 'mtpa':
        return MMTPA_PER_MCM_D
    return 1.0


def _convert_volume_metric_dataframe(
    df,
    volume_metric,
    columns=None,
    exclude_columns=None,
    precision=None,
    period_days=None,
    period_days_by_column=None
):
    """Convert selected numeric dataframe columns from mcm/d to the display metric."""
    if df is None or df.empty:
        return df

    normalized_metric = volume_metric if volume_metric in VOLUME_CONVERSIONS else 'mcm_d'
    if normalized_metric == 'mcm_d' and precision is None:
        return df

    converted_df = df.copy()
    exclude_columns = set(exclude_columns or [])
    period_days_by_column = period_days_by_column or {}
    if columns is None:
        columns = [col for col in converted_df.columns if col not in exclude_columns]

    for col in columns:
        if col not in converted_df.columns or col in exclude_columns:
            continue
        col_period_days = period_days_by_column.get(col, period_days)
        factor = _get_volume_metric_factor(normalized_metric, col_period_days)
        numeric_series = pd.to_numeric(converted_df[col], errors='coerce') * factor
        if precision is not None:
            numeric_series = numeric_series.round(precision)
        converted_df[col] = numeric_series.where(pd.notnull(numeric_series), None)
    return converted_df


def normalize_supply_rolling_avg_days(value):
    """Clamp the first two chart sections' rolling-average window to a practical range."""
    try:
        days = int(round(float(value)))
    except (TypeError, ValueError):
        days = DEFAULT_SUPPLY_ROLLING_AVG_DAYS
    return max(MIN_SUPPLY_ROLLING_AVG_DAYS, min(MAX_SUPPLY_ROLLING_AVG_DAYS, days))


def _supply_rolling_window_preceding_days(value):
    return normalize_supply_rolling_avg_days(value) - 1


def _format_rolling_average_section_title(title_prefix, rolling_avg_days):
    days = normalize_supply_rolling_avg_days(rolling_avg_days)
    return f'{title_prefix} - {days}-Day Rolling Average'


def _empty_supply_dest_summary_store_payload():
    """Return the default store payload for grouped and ungrouped overview tables."""
    return {
        'format': EXPORTERS_SUPPLY_DEST_SUMMARY_FORMAT,
        'show_all': [],
        'group_small_countries': [],
        'comparison': {
            'status': 'unavailable',
            'current_snapshot': None,
            'baseline_snapshot': None,
            'business_day_gap': None,
        },
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


def _normalize_supply_dest_comparison_basis(comparison_basis):
    """Normalize comparison basis while keeping the current table as the default state."""
    if comparison_basis in {'levels', 'previous_period', 'same_period_last_year'}:
        return comparison_basis
    return 'levels'


def _normalize_supply_dest_country_grouping(grouping_mode):
    """Normalize the small-country grouping mode for the destination table."""
    if grouping_mode in {'group_small_countries', 'group_small', 'yes', 'Yes', True}:
        return 'group_small_countries'
    return 'show_all'


def _build_supply_dest_count_options(max_count, min_count=1):
    """Build compact integer options for the summary period-count selectors."""
    return [{'label': str(value), 'value': value} for value in range(min_count, max_count + 1)]


def _coerce_supply_dest_period_count(value, default, max_count, min_count=1):
    """Clamp a period-count selector value to the supported summary-table range."""
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = default
    return max(min_count, min(count, max_count))


def _is_supply_dest_summary_year_column(column_name):
    column_name = str(column_name)
    return len(column_name) == 4 and column_name.isdigit()


def _is_supply_dest_summary_quarter_column(column_name):
    column_name = str(column_name)
    return column_name.startswith('Q') and "'" in column_name


def _is_supply_dest_summary_month_column(column_name):
    column_name = str(column_name)
    return (
        "'" in column_name
        and not column_name.startswith('Q')
        and not column_name.startswith('W')
        and column_name.split("'")[0] in MONTH_ORDER
    )


def _is_supply_dest_summary_week_column(column_name):
    column_name = str(column_name)
    return column_name.startswith('W') and "'" in column_name


def _parse_supply_dest_period_year_suffix(year_suffix):
    try:
        return int(f"20{int(year_suffix):02d}")
    except (TypeError, ValueError):
        return None


def _get_supply_dest_summary_week_days(column_name):
    """Return elapsed days for the active week, otherwise a completed seven-day week."""
    column_name = str(column_name)
    current_date = datetime.now().date()
    current_week = pd.Period(current_date, freq='W')
    current_week_label = (
        f"W{current_week.start_time.isocalendar()[1]}'"
        f"{str(current_week.year)[2:]}"
    )
    if column_name == current_week_label:
        return (current_date - current_week.start_time.date()).days + 1
    return 7


def _get_supply_dest_summary_column_period_days(column_name):
    """Return the represented period length for mcm/d-to-MT conversion."""
    column_name = str(column_name)
    if column_name in {'30D', '30D_PP', '30D_Y1', '30D_PBD'}:
        return 30
    if column_name in {'7D', '7D_PP', '7D_Y1', '7D_PBD'}:
        return 7
    if _is_supply_dest_summary_year_column(column_name):
        year = int(column_name)
        return 366 if calendar.isleap(year) else 365
    if _is_supply_dest_summary_quarter_column(column_name):
        try:
            quarter_part, year_suffix = column_name.split("'")
            quarter = int(quarter_part.replace('Q', ''))
        except (TypeError, ValueError):
            return None
        year = _parse_supply_dest_period_year_suffix(year_suffix)
        if year is None or quarter not in {1, 2, 3, 4}:
            return None
        quarter_start = pd.Timestamp(year=year, month=((quarter - 1) * 3) + 1, day=1)
        return _get_supply_dest_period_days(quarter_start, 'quarterly')
    if _is_supply_dest_summary_month_column(column_name):
        try:
            month_label, year_suffix = column_name.split("'")
        except (TypeError, ValueError):
            return None
        year = _parse_supply_dest_period_year_suffix(year_suffix)
        month = MONTH_ORDER.get(month_label)
        if year is None or month is None:
            return None
        return calendar.monthrange(year, month)[1]
    if _is_supply_dest_summary_week_column(column_name):
        return _get_supply_dest_summary_week_days(column_name)
    return None


def _build_supply_dest_summary_period_days_map(columns):
    """Build a per-column period-length map for destination summary conversions."""
    period_days_by_column = {}
    for column_name in columns:
        period_days = _get_supply_dest_summary_column_period_days(column_name)
        if period_days is not None:
            period_days_by_column[column_name] = period_days
    return period_days_by_column


def _recalculate_supply_dest_absolute_delta_columns(display_df):
    """Refresh absolute delta columns after period-aware unit conversion."""
    recalculated_df = display_df.copy()
    if {'Δ 7D-30D', '7D', '30D'}.issubset(recalculated_df.columns):
        recalculated_df['Δ 7D-30D'] = (
            pd.to_numeric(recalculated_df['7D'], errors='coerce')
            - pd.to_numeric(recalculated_df['30D'], errors='coerce')
        ).round(1)
    if {'Δ 30D Y/Y', '30D', '30D_Y1'}.issubset(recalculated_df.columns):
        recalculated_df['Δ 30D Y/Y'] = (
            pd.to_numeric(recalculated_df['30D'], errors='coerce')
            - pd.to_numeric(recalculated_df['30D_Y1'], errors='coerce')
        ).round(1)
    if {'Δ 30D vs PBD', '30D', '30D_PBD'}.issubset(recalculated_df.columns):
        recalculated_df['Δ 30D vs PBD'] = (
            pd.to_numeric(recalculated_df['30D'], errors='coerce')
            - pd.to_numeric(recalculated_df['30D_PBD'], errors='coerce')
        ).round(1)
    if {'Δ 7D vs PBD', '7D', '7D_PBD'}.issubset(recalculated_df.columns):
        recalculated_df['Δ 7D vs PBD'] = (
            pd.to_numeric(recalculated_df['7D'], errors='coerce')
            - pd.to_numeric(recalculated_df['7D_PBD'], errors='coerce')
        ).round(1)
    return recalculated_df


def _recalculate_supply_dest_pbd_delta_columns(display_df, precision=None):
    """Recalculate prior-business-day deltas from comparable level columns."""
    recalculated_df = display_df.copy()
    pairs = (
        ('Δ 30D vs PBD', '30D', '30D_PBD'),
        ('Δ 7D vs PBD', '7D', '7D_PBD'),
    )
    for delta_column, current_column, baseline_column in pairs:
        if {
            delta_column,
            current_column,
            baseline_column,
        }.issubset(recalculated_df.columns):
            values = (
                pd.to_numeric(
                    recalculated_df[current_column],
                    errors='coerce',
                )
                - pd.to_numeric(
                    recalculated_df[baseline_column],
                    errors='coerce',
                )
            )
            if precision is not None:
                values = values.round(precision)
            recalculated_df[delta_column] = values
    return recalculated_df


def _convert_supply_dest_absolute_volume_metric(display_df, volume_metric):
    """Convert destination summary values from average mcm/d to the selected metric."""
    if display_df is None or display_df.empty:
        return display_df

    delta_cols = set(SUPPLY_DEST_DELTA_RAW_FIELDS)
    period_days_by_column = _build_supply_dest_summary_period_days_map(display_df.columns)
    converted_df = _convert_volume_metric_dataframe(
        display_df,
        volume_metric,
        exclude_columns=set(SUPPLY_DEST_TEXT_COLUMNS) | delta_cols,
        precision=1,
        period_days_by_column=period_days_by_column
    )
    return _recalculate_supply_dest_absolute_delta_columns(converted_df)


def _get_supply_dest_summary_previous_week_label(column_name, week_cols):
    """Return the previous visible weekly label from the preloaded weekly sequence."""
    if column_name in week_cols:
        column_index = week_cols.index(column_name)
        if column_index > 0:
            return week_cols[column_index - 1]
    return None


def _get_supply_dest_summary_prior_year_week_label(column_name):
    """Return the same ISO-week label one year earlier."""
    column_name = str(column_name)
    try:
        week_part, year_suffix = column_name.split("'")
        week_num = int(week_part.replace('W', ''))
        year = int(f'20{year_suffix}')
    except (ValueError, TypeError):
        return None
    return f"W{week_num}'{str(year - 1)[2:]}"


def _build_supply_dest_summary_comparison_reference_map(visible_period_cols, week_cols,
                                                       comparison_basis):
    """Map selected summary-period columns to hidden comparison reference columns."""
    comparison_basis = _normalize_supply_dest_comparison_basis(comparison_basis)
    if comparison_basis not in {'previous_period', 'same_period_last_year'}:
        return {}

    reference_map = {}
    for col in visible_period_cols:
        reference_col = None
        if _is_supply_dest_summary_year_column(col):
            reference_col = _get_supply_dest_previous_period_label(col, 'yearly')
        elif _is_supply_dest_summary_quarter_column(col):
            reference_col = (
                _get_supply_dest_previous_period_label(col, 'quarterly')
                if comparison_basis == 'previous_period'
                else _get_supply_dest_prior_year_label(col, 'quarterly')
            )
        elif _is_supply_dest_summary_month_column(col):
            reference_col = (
                _get_supply_dest_previous_period_label(col, 'monthly')
                if comparison_basis == 'previous_period'
                else _get_supply_dest_prior_year_label(col, 'monthly')
            )
        elif _is_supply_dest_summary_week_column(col):
            reference_col = (
                _get_supply_dest_summary_previous_week_label(col, week_cols)
                if comparison_basis == 'previous_period'
                else _get_supply_dest_summary_prior_year_week_label(col)
            )

        if reference_col:
            reference_map[col] = reference_col

    return reference_map


def _filter_supply_dest_summary_period_columns(df, quarter_count=None, month_count=None,
                                               week_count=None, year_count=None,
                                               comparison_basis='levels',
                                               return_metadata=False):
    """Keep selected periods, plus hidden comparison references when requested."""
    empty_metadata = {
        'comparison_basis': _normalize_supply_dest_comparison_basis(comparison_basis),
        'visible_period_cols': [],
        'comparison_reference_map': {},
        'reference_cols': [],
        'comparison_delta_cols': []
    }
    if df is None or df.empty:
        if return_metadata:
            return df, empty_metadata
        return df

    comparison_basis = _normalize_supply_dest_comparison_basis(comparison_basis)
    year_count = _coerce_supply_dest_period_count(
        year_count,
        SUPPLY_DEST_DEFAULT_YEAR_COUNT,
        SUPPLY_DEST_MAX_YEAR_COUNT,
        min_count=0
    )
    quarter_count = _coerce_supply_dest_period_count(
        quarter_count,
        SUPPLY_DEST_DEFAULT_QUARTER_COUNT,
        SUPPLY_DEST_MAX_QUARTER_COUNT
    )
    month_count = _coerce_supply_dest_period_count(
        month_count,
        SUPPLY_DEST_DEFAULT_MONTH_COUNT,
        SUPPLY_DEST_MAX_MONTH_COUNT
    )
    week_count = _coerce_supply_dest_period_count(
        week_count,
        SUPPLY_DEST_DEFAULT_WEEK_COUNT,
        SUPPLY_DEST_MAX_WEEK_COUNT
    )

    year_cols = [col for col in df.columns if _is_supply_dest_summary_year_column(col)]
    quarter_cols = [col for col in df.columns if _is_supply_dest_summary_quarter_column(col)]
    month_cols = [col for col in df.columns if _is_supply_dest_summary_month_column(col)]
    week_cols = [col for col in df.columns if _is_supply_dest_summary_week_column(col)]
    selected_year_cols = year_cols[-year_count:] if year_count else []
    visible_period_cols = (
        selected_year_cols
        + quarter_cols[-quarter_count:]
        + month_cols[-month_count:]
        + week_cols[-week_count:]
    )
    comparison_reference_map = _build_supply_dest_summary_comparison_reference_map(
        visible_period_cols,
        week_cols,
        comparison_basis
    )
    visible_rolling_cols = [col for col in ['30D', '7D'] if col in df.columns]
    visible_comparison_cols = visible_period_cols + visible_rolling_cols
    if comparison_basis == 'previous_period':
        rolling_reference_map = {
            '30D': '30D_PP',
            '7D': '7D_PP'
        }
    elif comparison_basis == 'same_period_last_year':
        rolling_reference_map = {
            '30D': '30D_Y1',
            '7D': '7D_Y1'
        }
    else:
        rolling_reference_map = {}
    comparison_reference_map.update({
        visible_col: reference_col
        for visible_col, reference_col in rolling_reference_map.items()
        if visible_col in df.columns and reference_col in df.columns
    })
    reference_cols = [
        reference_col
        for reference_col in comparison_reference_map.values()
        if reference_col in df.columns and reference_col not in visible_comparison_cols
    ]
    selected_period_cols = set(visible_period_cols + reference_cols)

    visible_cols = []
    for col in df.columns:
        if (
            _is_supply_dest_summary_year_column(col)
            or _is_supply_dest_summary_quarter_column(col)
            or _is_supply_dest_summary_month_column(col)
            or _is_supply_dest_summary_week_column(col)
        ) and col not in selected_period_cols:
            continue
        visible_cols.append(col)

    filtered_df = df.loc[:, visible_cols].copy()
    metadata = {
        'comparison_basis': comparison_basis,
        'visible_period_cols': visible_period_cols,
        'visible_comparison_cols': visible_comparison_cols,
        'comparison_reference_map': comparison_reference_map,
        'reference_cols': reference_cols,
        'comparison_delta_cols': (
            visible_comparison_cols
            if comparison_basis in {'previous_period', 'same_period_last_year'}
            else []
        )
    }
    if return_metadata:
        return filtered_df, metadata
    return filtered_df


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


def _get_supply_dest_summary_comparison_metadata(summary_payload):
    """Return normalized current/PBD snapshot metadata from the summary payload."""
    if not isinstance(summary_payload, dict):
        summary_payload = {}
    comparison = summary_payload.get('comparison')
    if not isinstance(comparison, dict):
        comparison = {}
    status = comparison.get('status')
    if status not in {'exact', 'fallback', 'unavailable'}:
        status = 'unavailable'
    return {
        'status': status,
        'current_snapshot': (
            comparison.get('current_snapshot')
            if isinstance(comparison.get('current_snapshot'), dict)
            else None
        ),
        'baseline_snapshot': (
            comparison.get('baseline_snapshot')
            if isinstance(comparison.get('baseline_snapshot'), dict)
            else None
        ),
        'business_day_gap': comparison.get('business_day_gap'),
    }


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


def safe_concat(dataframes, **kwargs):
    """Concatenate DataFrames, filtering out empty ones to avoid FutureWarning."""
    non_empty_dfs = [df for df in dataframes if not df.empty]
    if not non_empty_dfs:
        return pd.DataFrame()
    return pd.concat(non_empty_dfs, **kwargs)


def normalize_demand_aggregation_mode(demand_aggregation_mode):
    """Normalize destination aggregation mode to the current selector values."""
    if demand_aggregation_mode in (None, '', 'None'):
        return 'Installation'
    return demand_aggregation_mode


def use_import_classification_mode(demand_aggregation_mode):
    """Import classification granularity groups destination countries by level 1 classification."""
    return normalize_demand_aggregation_mode(demand_aggregation_mode) == 'Classification Level 1'


def use_demand_classification_mode(classification_mode, demand_aggregation_mode):
    """Demand aggregation is only applied when supply is already in Classification Level 1 mode."""
    return (
        classification_mode == 'Classification Level 1'
        and use_import_classification_mode(demand_aggregation_mode)
    )


def use_demand_country_mode(demand_aggregation_mode):
    """Import-country granularity is shown for the Import Country selector mode."""
    return normalize_demand_aggregation_mode(demand_aggregation_mode) == 'Country'


def use_supply_installation_mode(demand_aggregation_mode):
    """Installation granularity is a supply-side drilldown, not a demand aggregation."""
    return normalize_demand_aggregation_mode(demand_aggregation_mode) == 'Installation'


def get_supply_dest_id_cols(classification_mode='Country', demand_aggregation_mode='None'):
    """Return the identifier columns used by the supply-destination table for the active mode."""
    if classification_mode == 'Classification Level 1':
        if use_demand_classification_mode(classification_mode, demand_aggregation_mode):
            return ['supply_classification', 'demand_classification', 'demand_country', 'supply_country']
        if use_demand_country_mode(demand_aggregation_mode):
            return ['supply_classification', 'demand_country', 'supply_country']
        return ['supply_classification', 'supply_country']
    if use_import_classification_mode(demand_aggregation_mode):
        return ['supply_country', 'demand_classification']
    if use_demand_country_mode(demand_aggregation_mode):
        return ['supply_country', 'demand_country']
    if use_supply_installation_mode(demand_aggregation_mode):
        return ['supply_country', 'supply_installation']
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


EXPORTERS_OVERVIEW_NAMESPACE = 'exporters-overview-v1'
EXPORTERS_DESTINATION_BASE_NAMESPACE = 'exporters-destination-base-v2'
EXPORTERS_DESTINATION_PBD_BASE_NAMESPACE = 'exporters-destination-pbd-base-v1'
EXPORTERS_DESTINATION_SUMMARY_NAMESPACE = 'exporters-destination-summary-v2'
EXPORTERS_SUPPLY_CHARTS_NAMESPACE = 'exporters-supply-charts-v1'
EXPORTERS_CONTINENT_DATA_NAMESPACE = 'exporters-continent-data-v1'
EXPORTERS_CONTINENT_EXPORT_NAMESPACE = 'exporters-continent-export-v1'
EXPORTERS_REFERENCE_NAMESPACES = frozenset({
    EXPORTERS_OVERVIEW_NAMESPACE,
    EXPORTERS_DESTINATION_BASE_NAMESPACE,
    EXPORTERS_DESTINATION_PBD_BASE_NAMESPACE,
    EXPORTERS_DESTINATION_SUMMARY_NAMESPACE,
    EXPORTERS_SUPPLY_CHARTS_NAMESPACE,
    EXPORTERS_CONTINENT_DATA_NAMESPACE,
    EXPORTERS_CONTINENT_EXPORT_NAMESPACE,
})
EXPORTERS_SOURCE_STATE_FORMAT = 'exporters-source-state-v2'
EXPORTERS_SUPPLY_DEST_SUMMARY_FORMAT = 'exporters-supply-dest-summary-v2'
EXPORTERS_SNAPSHOT_RECOVERY_MESSAGE = (
    'Cached exporter data is unavailable. Click the global Refresh button '
    'to reload it.'
)
EXPORTERS_CHARTS_CUBE_FORMAT = 'exporters-record-cube-zlib-json-v1'
EXPORTERS_SUPPLY_DEST_FORMAT = 'exporters-supply-dest-zlib-json-v1'
EXPORTERS_SCALAR_TAG = '__exporters_scalar_v1__'


def _tag_exporters_json_value(value):
    value_type = type(value)
    if value is None or value_type in (str, int, float, bool):
        return value
    if value is pd.NaT:
        return {EXPORTERS_SCALAR_TAG: 'pandas.NaT'}
    if value is pd.NA:
        return {EXPORTERS_SCALAR_TAG: 'pandas.NA'}
    if isinstance(value, pd.Timestamp):
        timezone = value.tz
        timezone_name = (
            getattr(timezone, 'zone', None)
            or getattr(timezone, 'key', None)
            or (str(timezone) if timezone is not None else None)
        )
        return {
            EXPORTERS_SCALAR_TAG: 'pandas.Timestamp',
            'nanoseconds': value.value,
            'timezone': timezone_name,
        }
    if isinstance(value, np.generic):
        return {
            EXPORTERS_SCALAR_TAG: 'numpy.scalar',
            'dtype': value.dtype.str,
            'bytes': value.tobytes().hex(),
        }
    if isinstance(value, datetime):
        return {
            EXPORTERS_SCALAR_TAG: 'datetime.datetime',
            'isoformat': value.isoformat(),
            'fold': value.fold,
        }
    if isinstance(value, bytes):
        return {
            EXPORTERS_SCALAR_TAG: 'builtins.bytes',
            'hex': value.hex(),
        }
    if isinstance(value, tuple):
        return {
            EXPORTERS_SCALAR_TAG: 'builtins.tuple',
            'items': [
                _tag_exporters_json_value(item)
                for item in value
            ],
        }
    if isinstance(value, list):
        return [
            _tag_exporters_json_value(item)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _tag_exporters_json_value(item)
            for key, item in value.items()
        }
    raise TypeError(f'{type(value).__name__} is not JSON serializable')


def _decode_exporters_json_object(value):
    scalar_type = value.get(EXPORTERS_SCALAR_TAG)
    if scalar_type == 'pandas.NaT' and len(value) == 1:
        return pd.NaT
    if scalar_type == 'pandas.NA' and len(value) == 1:
        return pd.NA
    if (
        scalar_type == 'pandas.Timestamp'
        and set(value) == {
            EXPORTERS_SCALAR_TAG,
            'nanoseconds',
            'timezone',
        }
    ):
        return pd.Timestamp(
            value['nanoseconds'],
            unit='ns',
            tz=value['timezone'],
        )
    if (
        scalar_type == 'numpy.scalar'
        and set(value) == {
            EXPORTERS_SCALAR_TAG,
            'dtype',
            'bytes',
        }
    ):
        return np.frombuffer(
            bytes.fromhex(value['bytes']),
            dtype=np.dtype(value['dtype']),
            count=1,
        )[0]
    if (
        scalar_type == 'datetime.datetime'
        and set(value) == {
            EXPORTERS_SCALAR_TAG,
            'isoformat',
            'fold',
        }
    ):
        return datetime.fromisoformat(value['isoformat']).replace(
            fold=value['fold']
        )
    if (
        scalar_type == 'builtins.bytes'
        and set(value) == {EXPORTERS_SCALAR_TAG, 'hex'}
    ):
        return bytes.fromhex(value['hex'])
    if (
        scalar_type == 'builtins.tuple'
        and set(value) == {EXPORTERS_SCALAR_TAG, 'items'}
    ):
        return tuple(value['items'])
    return value


def _encode_exporters_json_payload(value, payload_format):
    raw_payload = json.dumps(
        _tag_exporters_json_value(value),
        ensure_ascii=False,
        separators=(',', ':'),
    ).encode('utf-8')
    return {
        'format': payload_format,
        'payload': zlib.compress(raw_payload, level=1),
    }


def _decode_exporters_json_payload(value):
    if not (
        isinstance(value, dict)
        and value.get('format') in {
            EXPORTERS_CHARTS_CUBE_FORMAT,
            EXPORTERS_SUPPLY_DEST_FORMAT,
        }
    ):
        return value
    try:
        encoded_payload = value['payload']
        if not isinstance(encoded_payload, bytes):
            raise TypeError('encoded exporter payload is not bytes')
        return json.loads(
            zlib.decompress(encoded_payload).decode('utf-8'),
            object_hook=_decode_exporters_json_object,
        )
    except Exception as exc:
        raise _SnapshotUnavailable(
            EXPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        ) from exc


def _prepare_exporters_overview_snapshot_payload(payload):
    prepared = dict(payload)
    prepared['charts_cube'] = _encode_exporters_json_payload(
        payload['charts_cube'],
        EXPORTERS_CHARTS_CUBE_FORMAT,
    )
    prepared['supply_dest'] = _encode_exporters_json_payload(
        payload['supply_dest'],
        EXPORTERS_SUPPLY_DEST_FORMAT,
    )
    return prepared


def _prepare_exporters_supply_charts_snapshot_payload(charts_data):
    return {
        'charts_cube': _encode_exporters_json_payload(
            _pack_record_mapping(charts_data),
            EXPORTERS_CHARTS_CUBE_FORMAT,
        ),
        'continent_entities': list(charts_data),
    }


def _prepare_exporters_destination_summary_snapshot_payload(payload):
    return _encode_exporters_json_payload(
        payload,
        EXPORTERS_SUPPLY_DEST_FORMAT,
    )


def _fetch_exporters_source_watermark():
    with engine.connect() as connection:
        row = connection.execute(
            text(f"""
                WITH current_snapshot AS (
                    SELECT
                        snapshot_id,
                        snapshot_date_utc,
                        snapshot_timestamp_utc,
                        facts_retained
                    FROM {DB_SCHEMA}.kpler_trade_snapshots
                    WHERE run_kind = 'canonical'
                      AND status = 'published'
                    ORDER BY
                        snapshot_date_utc DESC,
                        snapshot_timestamp_utc DESC,
                        snapshot_id DESC
                    LIMIT 1
                )
                SELECT
                    current_snapshot.snapshot_id
                        AS current_snapshot_id,
                    current_snapshot.snapshot_date_utc
                        AS current_snapshot_date_utc,
                    current_snapshot.snapshot_timestamp_utc
                        AS current_snapshot_timestamp_utc,
                    current_snapshot.facts_retained
                        AS current_facts_retained,
                    baseline.snapshot_id
                        AS baseline_snapshot_id,
                    baseline.snapshot_date_utc
                        AS baseline_snapshot_date_utc,
                    baseline.snapshot_timestamp_utc
                        AS baseline_snapshot_timestamp_utc,
                    baseline.facts_retained
                        AS baseline_facts_retained
                FROM current_snapshot
                LEFT JOIN LATERAL (
                    SELECT
                        snapshot_id,
                        snapshot_date_utc,
                        snapshot_timestamp_utc,
                        facts_retained
                    FROM {DB_SCHEMA}.kpler_trade_snapshots
                    WHERE run_kind = 'canonical'
                      AND status = 'published'
                      AND facts_retained IS TRUE
                      AND snapshot_date_utc
                          < current_snapshot.snapshot_date_utc
                      AND EXTRACT(
                          ISODOW FROM snapshot_date_utc
                      ) BETWEEN 1 AND 5
                    ORDER BY
                        snapshot_date_utc DESC,
                        snapshot_timestamp_utc DESC,
                        snapshot_id DESC
                    LIMIT 1
                ) AS baseline ON TRUE
            """)
        ).mappings().one_or_none()
    return dict(row) if row is not None else None


def _resolve_exporters_store(value):
    try:
        if _is_snapshot_reference(value):
            namespace = value.get('namespace')
            if namespace not in EXPORTERS_REFERENCE_NAMESPACES:
                raise _SnapshotUnavailable(
                    EXPORTERS_SNAPSHOT_RECOVERY_MESSAGE
                )
        else:
            namespace = None
        resolved = _resolve_snapshot(
            value,
            engine,
            expected_namespace=namespace,
        )
        resolved = _decode_exporters_json_payload(resolved)
    except _SnapshotUnavailable as exc:
        raise _SnapshotUnavailable(
            EXPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        ) from exc
    return _unpack_record_mapping(resolved)


def _resolve_exporters_continent_payload(value):
    resolved = _resolve_exporters_store(value)
    if not (
        isinstance(resolved, dict)
        and isinstance(resolved.get('entities'), list)
        and isinstance(resolved.get('data'), pd.DataFrame)
        and isinstance(resolved.get('source_state'), dict)
        and isinstance(resolved.get('classification_mode'), str)
        and isinstance(resolved.get('rolling_avg_days'), int)
        and isinstance(resolved.get('selected_years'), list)
    ):
        raise _SnapshotUnavailable(
            EXPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        )
    return resolved


def _exporters_snapshot_recovery_notice():
    return html.Div(
        EXPORTERS_SNAPSHOT_RECOVERY_MESSAGE,
        className='exporters-snapshot-recovery-message',
        role='alert',
    )


def _exporters_snapshot_recovery_selector_result():
    return (
        [{
            'label': EXPORTERS_SNAPSHOT_RECOVERY_MESSAGE,
            'value': '__snapshot_unavailable__',
            'disabled': True,
        }],
        [],
    )


def setup_database_connection():
    """Setup database connection using existing configuration"""
    return engine, DB_SCHEMA


def get_all_classification_groups(engine, schema):
    """
    Get all distinct Classification Level 1 groups that have LNG export data
    Returns list of classification names ordered by total volume
    """
    try:
        with engine.connect() as conn:
            query = text(f"""
            WITH latest_data AS (
                SELECT snapshot_timestamp_utc as max_timestamp
                FROM {schema}.kpler_trade_snapshots
                WHERE run_kind = 'canonical' AND status = 'published'
                ORDER BY snapshot_date_utc DESC
                LIMIT 1
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
                    AND kt.start::date <= CURRENT_DATE
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

    except Exception:
        return []


def _build_supply_dest_rolling_windows_from_df(
    df,
    classification_mode='Country',
    demand_aggregation_mode='None',
    as_of_date=None,
):
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

    current_date = pd.Timestamp(
        as_of_date if as_of_date is not None else datetime.now()
    ).date()
    date_7d_ago = current_date - timedelta(days=7)
    date_14d_ago = current_date - timedelta(days=14)
    date_30d_ago = current_date - timedelta(days=30)
    date_60d_ago = current_date - timedelta(days=60)
    date_7d_y1_start = current_date - timedelta(days=365) - timedelta(days=7)
    date_7d_y1_end = current_date - timedelta(days=365)
    date_30d_y1_start = current_date - timedelta(days=365) - timedelta(days=30)
    date_30d_y1_end = current_date - timedelta(days=365)

    group_cols = get_supply_dest_id_cols(classification_mode, demand_aggregation_mode)
    if 'supply_installation' in group_cols and 'supply_installation' not in df.columns:
        df['supply_installation'] = 'Unknown'

    def _rolling_window(start_date, end_date, days, label):
        window_df = df[
            (df['flow_date'].dt.date > start_date)
            & (df['flow_date'].dt.date <= end_date)
        ].copy()
        if window_df.empty:
            return pd.DataFrame(columns=group_cols + [label])
        rolling_df = (window_df.groupby(group_cols)['mcmd'].sum() / days).round(1).reset_index()
        rolling_df.columns = group_cols + [label]
        return rolling_df

    rolling_frames = [
        _rolling_window(date_7d_ago, current_date, 7, '7D'),
        _rolling_window(date_14d_ago, date_7d_ago, 7, '7D_PP'),
        _rolling_window(date_7d_y1_start, date_7d_y1_end, 7, '7D_Y1'),
        _rolling_window(date_30d_ago, current_date, 30, '30D'),
        _rolling_window(date_60d_ago, date_30d_ago, 30, '30D_PP'),
        _rolling_window(date_30d_y1_start, date_30d_y1_end, 30, '30D_Y1')
    ]
    result = rolling_frames[0]
    for rolling_frame in rolling_frames[1:]:
        result = result.merge(rolling_frame, on=group_cols, how='outer')

    final_result = _apply_supply_dest_period_totals(
        result,
        classification_mode,
        demand_aggregation_mode
    )
    final_result = final_result.infer_objects(copy=False).fillna(0)
    for reference_col in ['7D', '7D_PP', '7D_Y1', '30D', '30D_PP', '30D_Y1']:
        if reference_col not in final_result.columns:
            final_result[reference_col] = 0
    final_result['Δ 7D-30D'] = (final_result['7D'] - final_result['30D']).round(1)
    final_result['Δ 30D Y/Y'] = (final_result['30D'] - final_result['30D_Y1']).round(1)
    return final_result


def _merge_supply_dest_pbd_rolling_windows(
    current_rolling,
    baseline_rolling,
    classification_mode='Country',
    demand_aggregation_mode='None',
    baseline_available=False,
):
    """Attach exact prior-vintage 30D/7D values and signed changes."""
    if current_rolling is None:
        current_rolling = pd.DataFrame()
    merged = current_rolling.copy()
    if not baseline_available or baseline_rolling is None or baseline_rolling.empty:
        if merged.empty:
            return merged
        for column_name in (
            *SUPPLY_DEST_PBD_REFERENCE_COLUMNS,
            *SUPPLY_DEST_PBD_DELTA_COLUMNS,
        ):
            merged[column_name] = np.nan
        return merged

    id_cols = get_supply_dest_id_cols(
        classification_mode,
        demand_aggregation_mode,
    )
    if merged.empty:
        merged = pd.DataFrame(
            columns=[
                *id_cols,
                '30D',
                '30D_PP',
                '30D_Y1',
                '7D',
                '7D_PP',
                '7D_Y1',
                'Δ 7D-30D',
                'Δ 30D Y/Y',
            ]
        )
    current_numeric_columns = [
        column_name
        for column_name in merged.columns
        if column_name not in id_cols
    ]
    baseline_columns = [
        column_name
        for column_name in (*id_cols, '30D', '7D')
        if column_name in baseline_rolling.columns
    ]
    baseline_values = baseline_rolling[baseline_columns].copy()
    baseline_values = baseline_values.rename(
        columns={
            '30D': '30D_PBD',
            '7D': '7D_PBD',
        }
    )
    merged = merged.merge(
        baseline_values,
        on=id_cols,
        how='outer',
    )

    for column_name in current_numeric_columns:
        merged[column_name] = pd.to_numeric(
            merged[column_name],
            errors='coerce',
        ).fillna(0)
    for column_name in SUPPLY_DEST_PBD_REFERENCE_COLUMNS:
        merged[column_name] = pd.to_numeric(
            merged[column_name],
            errors='coerce',
        ).fillna(0)

    merged['Δ 30D vs PBD'] = (
        merged['30D'] - merged['30D_PBD']
    ).round(1)
    merged['Δ 7D vs PBD'] = (
        merged['7D'] - merged['7D_PBD']
    ).round(1)
    return merged


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
                    SELECT snapshot_timestamp_utc as max_timestamp
                    FROM {schema}.kpler_trade_snapshots
                    WHERE run_kind = 'canonical' AND status = 'published'
                    ORDER BY snapshot_date_utc DESC
                    LIMIT 1
                )
                SELECT
                    COALESCE(mc_origin.country_classification_level1, 'Unknown') as supply_classification,
                    kt.origin_country_name as supply_country,
                    COALESCE(NULLIF(BTRIM(kt.installation_origin_name), ''), 'Unknown') as supply_installation,
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
                        -- Current and prior rolling-window references
                        (kt.start >= CURRENT_DATE - INTERVAL '60 days' AND kt.start::date <= CURRENT_DATE)
                        OR
                        -- Same 30-day window last year
                        (kt.start >= CURRENT_DATE - INTERVAL '1 year' - INTERVAL '30 days'
                         AND kt.start::date <= (CURRENT_DATE - INTERVAL '1 year')::date)
                    )
                ORDER BY kt.start::date
                """)
            else:
                query = text(f"""
                WITH latest_data AS (
                    SELECT snapshot_timestamp_utc as max_timestamp
                    FROM {schema}.kpler_trade_snapshots
                    WHERE run_kind = 'canonical' AND status = 'published'
                    ORDER BY snapshot_date_utc DESC
                    LIMIT 1
                )
                SELECT
                    kt.origin_country_name as supply_country,
                    COALESCE(NULLIF(BTRIM(kt.installation_origin_name), ''), 'Unknown') as supply_installation,
                    COALESCE(kt.destination_country_name, 'Unknown') as demand_country,
                    kt.start::date as flow_date,
                    kt.cargo_origin_cubic_meters * 0.6 / 1000 as mcmd
                FROM {schema}.kpler_trades kt, latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND (
                        -- Current and prior rolling-window references
                        (kt.start >= CURRENT_DATE - INTERVAL '60 days' AND kt.start::date <= CURRENT_DATE)
                        OR
                        -- Same 30-day window last year
                        (kt.start >= CURRENT_DATE - INTERVAL '1 year' - INTERVAL '30 days'
                         AND kt.start::date <= (CURRENT_DATE - INTERVAL '1 year')::date)
                    )
                ORDER BY kt.start::date
                """)

            df = pd.read_sql(query, conn)

            return _build_supply_dest_rolling_windows_from_df(
                df,
                classification_mode,
                demand_aggregation_mode
            )

    except Exception:
        return pd.DataFrame()


def fetch_supply_dest_summary_data(engine, schema, classification_mode, demand_aggregation_mode,
                                   quarters_df, months_df, weeks_df,
                                   years_df=None, rolling_data=None,
                                   current_date=None):
    """Combine supply-destination years, quarters, months, weeks, and rolling windows into summary format."""
    try:
        years_df = years_df if years_df is not None else pd.DataFrame()
        if years_df.empty and quarters_df.empty and months_df.empty and weeks_df.empty:
            return pd.DataFrame()

        # Get current date to determine what's complete
        current_date = pd.Timestamp(
            current_date if current_date is not None else datetime.now()
        )
        current_quarter = (current_date.month - 1) // 3 + 1
        current_year = current_date.year

        if rolling_data is None:
            rolling_data = fetch_supply_dest_rolling_windows(
                engine,
                schema,
                classification_mode,
                demand_aggregation_mode
            )

        id_cols = get_supply_dest_id_cols(classification_mode, demand_aggregation_mode)

        # Get period columns
        year_cols = [col for col in years_df.columns if col not in id_cols]
        quarter_cols = [col for col in quarters_df.columns if col not in id_cols]
        month_cols = [col for col in months_df.columns if col not in id_cols]
        week_cols = [col for col in weeks_df.columns if col not in id_cols]

        # For years: exclude current/incomplete year.
        year_cols_filtered = []
        for col in year_cols:
            if _is_supply_dest_summary_year_column(col):
                year = int(str(col))
                if year < current_year:
                    year_cols_filtered.append(col)

        # Sort and preload completed years for the summary selector.
        year_cols_sorted = sorted(year_cols_filtered, key=lambda x: int(str(x)))
        selected_year_cols = (
            year_cols_sorted[-SUPPLY_DEST_PRELOAD_YEAR_COUNT:]
            if len(year_cols_sorted) >= SUPPLY_DEST_PRELOAD_YEAR_COUNT
            else year_cols_sorted
        )

        # Filter out current/incomplete periods.
        # For quarters: exclude current quarter.
        quarter_cols_filtered = []
        for col in quarter_cols:
            if "Q" in col and "'" in col:
                q_num = int(col.split("Q")[1].split("'")[0])
                year = int("20" + col.split("'")[1])
                # Exclude if it's the current quarter or future
                if year < current_year or (year == current_year and q_num < current_quarter):
                    quarter_cols_filtered.append(col)

        # Sort and preload completed quarters for the summary selector.
        quarter_cols_sorted = sorted(quarter_cols_filtered,
                                    key=lambda x: (x.split("'")[1], x.split("Q")[1].split("'")[0]))
        selected_quarter_cols = (
            quarter_cols_sorted[-SUPPLY_DEST_PRELOAD_QUARTER_COUNT:]
            if len(quarter_cols_sorted) >= SUPPLY_DEST_PRELOAD_QUARTER_COUNT
            else quarter_cols_sorted
        )

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

        # Sort and preload completed months for the summary selector.
        month_cols_sorted = sorted(month_cols_filtered,
                                  key=lambda x: (x.split("'")[1], month_order.get(x.split("'")[0], 0)))
        selected_month_cols = (
            month_cols_sorted[-SUPPLY_DEST_PRELOAD_MONTH_COUNT:]
            if len(month_cols_sorted) >= SUPPLY_DEST_PRELOAD_MONTH_COUNT
            else month_cols_sorted
        )

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

        # Sort and preload completed weeks for the summary selector.
        week_cols_sorted = sorted(week_cols_filtered,
                                key=lambda x: (x.split("'")[1], x.split("W")[1].split("'")[0].zfill(2)))
        selected_week_cols = (
            week_cols_sorted[-SUPPLY_DEST_PRELOAD_WEEK_COUNT:]
            if len(week_cols_sorted) >= SUPPLY_DEST_PRELOAD_WEEK_COUNT
            else week_cols_sorted
        )

        # Create subsets with selected columns
        years_subset = (
            years_df[id_cols + selected_year_cols].copy()
            if not years_df.empty
            else pd.DataFrame(columns=id_cols)
        )
        quarters_subset = quarters_df[id_cols + selected_quarter_cols].copy()
        months_subset = months_df[id_cols + selected_month_cols].copy()
        weeks_subset = weeks_df[id_cols + selected_week_cols].copy()

        # Merge years, quarters, and months first
        result = years_subset.merge(quarters_subset, on=id_cols, how='outer') if not years_subset.empty else quarters_subset.copy()
        result = result.merge(months_subset, on=id_cols, how='outer')

        # Add 30D column right after months (before weeks), keeping references hidden.
        if not rolling_data.empty:
            rolling_30d_cols = [
                col for col in id_cols + ['30D', '30D_PP', '30D_Y1']
                if col in rolling_data.columns
            ]
            result = result.merge(
                rolling_data[rolling_30d_cols],
                on=id_cols,
                how='left'
            )
        else:
            result['30D'] = 0
            result['30D_PP'] = 0
            result['30D_Y1'] = 0

        # Then merge with weeks
        result = result.merge(weeks_subset, on=id_cols, how='outer')

        # Finally add the remaining rolling window columns and hidden references.
        if not rolling_data.empty:
            final_rolling_cols = [
                col for col in id_cols + [
                    '7D',
                    '7D_PP',
                    '7D_Y1',
                    'Δ 7D-30D',
                    'Δ 30D Y/Y',
                    '30D_PBD',
                    '7D_PBD',
                    'Δ 30D vs PBD',
                    'Δ 7D vs PBD',
                ]
                if col in rolling_data.columns
            ]
            result = result.merge(
                rolling_data[final_rolling_cols],
                on=id_cols,
                how='left'
            )
        else:
            result['7D'] = 0
            result['7D_PP'] = 0
            result['7D_Y1'] = 0
            result['Δ 7D-30D'] = 0
            result['Δ 30D Y/Y'] = 0
            result['30D_PBD'] = np.nan
            result['7D_PBD'] = np.nan
            result['Δ 30D vs PBD'] = np.nan
            result['Δ 7D vs PBD'] = np.nan

        pbd_available = (
            rolling_data is not None
            and not rolling_data.empty
            and any(
                column_name in rolling_data.columns
                and rolling_data[column_name].notna().any()
                for column_name in SUPPLY_DEST_PBD_REFERENCE_COLUMNS
            )
        )

        # Fill valid absent-flow cells with zero while preserving an unavailable
        # comparison snapshot as unavailable.
        result = result.fillna(0)
        if not pbd_available:
            for column_name in (
                *SUPPLY_DEST_PBD_REFERENCE_COLUMNS,
                *SUPPLY_DEST_PBD_DELTA_COLUMNS,
            ):
                if column_name in result.columns:
                    result[column_name] = np.nan

        # Ensure numeric columns are float
        numeric_cols = (
            selected_year_cols
            + selected_quarter_cols
            + selected_month_cols
            + ['30D', '30D_PP', '30D_Y1']
            + selected_week_cols
            + [
                '7D',
                '7D_PP',
                '7D_Y1',
                'Δ 7D-30D',
                'Δ 30D Y/Y',
                '30D_PBD',
                '7D_PBD',
                'Δ 30D vs PBD',
                'Δ 7D vs PBD',
            ]
        )
        for col in numeric_cols:
            if col in result.columns:
                result[col] = result[col].astype(float)

        # Sort by classification/country pairs
        result = result.sort_values(id_cols)

        result = result.reset_index(drop=True)

        if not result.empty:
            rolling_totals = {
                col: rolling_data[col].sum() if rolling_data is not None and not rolling_data.empty and col in rolling_data.columns else 0
                for col in [
                    '30D',
                    '30D_PP',
                    '30D_Y1',
                    '7D',
                    '7D_PP',
                    '7D_Y1',
                    'Δ 7D-30D',
                    'Δ 30D Y/Y',
                    '30D_PBD',
                    '7D_PBD',
                    'Δ 30D vs PBD',
                    'Δ 7D vs PBD',
                ]
            }
            if not pbd_available:
                for column_name in (
                    *SUPPLY_DEST_PBD_REFERENCE_COLUMNS,
                    *SUPPLY_DEST_PBD_DELTA_COLUMNS,
                ):
                    rolling_totals[column_name] = np.nan
            other_cols = {
                col: result[col].sum()
                for col in selected_year_cols + selected_quarter_cols + selected_month_cols + selected_week_cols
                if col in result.columns
            }

            if classification_mode == 'Classification Level 1':
                grand_total_payload = {
                    'supply_classification': 'GRAND TOTAL',
                    'supply_country': '',
                    **other_cols,
                    **rolling_totals
                }
                if use_demand_classification_mode(classification_mode, demand_aggregation_mode):
                    grand_total_payload['demand_classification'] = ''
                    grand_total_payload['demand_country'] = ''
                elif use_demand_country_mode(demand_aggregation_mode):
                    grand_total_payload['demand_country'] = ''
                elif use_import_classification_mode(demand_aggregation_mode):
                    grand_total_payload['demand_classification'] = ''
            else:
                grand_total_payload = {
                    'supply_country': 'GRAND TOTAL',
                    **other_cols,
                    **rolling_totals
                }
                if use_demand_country_mode(demand_aggregation_mode):
                    grand_total_payload['demand_country'] = ''
                elif use_import_classification_mode(demand_aggregation_mode):
                    grand_total_payload['demand_classification'] = ''
                elif use_supply_installation_mode(demand_aggregation_mode):
                    grand_total_payload['supply_installation'] = ''

            result = safe_concat([result, pd.DataFrame([grand_total_payload])], ignore_index=True)

        return result

    except Exception:
        return pd.DataFrame()


def _build_supply_dest_snapshot_comparison_metadata(
    source_state,
    baseline_data_available,
):
    """Build JSON-safe lineage metadata for the current/PBD table pair."""
    source_state = source_state if isinstance(source_state, dict) else {}
    current_snapshot = source_state.get('current_snapshot')
    baseline_snapshot = source_state.get('baseline_snapshot')
    status = source_state.get('baseline_status')
    if (
        status not in {'exact', 'fallback'}
        or not isinstance(current_snapshot, dict)
        or not isinstance(baseline_snapshot, dict)
        or not baseline_data_available
    ):
        status = 'unavailable'
    return {
        'status': status,
        'current_snapshot': (
            dict(current_snapshot)
            if isinstance(current_snapshot, dict)
            else None
        ),
        'baseline_snapshot': (
            dict(baseline_snapshot)
            if isinstance(baseline_snapshot, dict)
            else None
        ),
        'business_day_gap': source_state.get('business_day_gap'),
    }


def build_supply_dest_summary_store_payload(
    engine,
    schema,
    base_df,
    classification_mode='Country',
    demand_aggregation_mode='None',
    previous_business_base_df=None,
    source_state=None,
):
    """Build grouped and ungrouped overview payloads for the supply-destination summary table."""
    if base_df is None or base_df.empty:
        return _empty_supply_dest_summary_store_payload()

    source_state = source_state if isinstance(source_state, dict) else {}
    current_as_of_date = (
        (source_state.get('current_snapshot') or {}).get('snapshot_date_utc')
        or source_state.get('as_of_date')
        or datetime.now().date()
    )
    baseline_as_of_date = (
        (source_state.get('baseline_snapshot') or {}).get('snapshot_date_utc')
    )
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

    filtered_baseline_df = pd.DataFrame()
    if (
        previous_business_base_df is not None
        and not previous_business_base_df.empty
        and baseline_as_of_date is not None
    ):
        filtered_baseline_df = exclude_internal_destination_flows(
            previous_business_base_df.copy(),
            classification_mode,
            origin_country_col='supply_country',
            destination_country_col='demand_country',
            origin_classification_col='supply_classification',
            destination_classification_col='demand_classification',
        )
    baseline_data_available = not filtered_baseline_df.empty
    comparison_metadata = _build_supply_dest_snapshot_comparison_metadata(
        source_state,
        baseline_data_available,
    )

    def _build_payload_records(summary_base_df, summary_baseline_df):
        years_df, quarters_df, months_df, weeks_df = fetch_supply_destination_data(
            engine,
            schema,
            classification_mode,
            demand_aggregation_mode,
            summary_base_df,
            current_as_of_date,
        )
        rolling_df = _build_supply_dest_rolling_windows_from_df(
            summary_base_df,
            classification_mode,
            demand_aggregation_mode,
            current_as_of_date,
        )
        baseline_rolling_df = _build_supply_dest_rolling_windows_from_df(
            summary_baseline_df,
            classification_mode,
            demand_aggregation_mode,
            baseline_as_of_date,
        )
        rolling_df = _merge_supply_dest_pbd_rolling_windows(
            rolling_df,
            baseline_rolling_df,
            classification_mode,
            demand_aggregation_mode,
            baseline_available=baseline_data_available,
        )
        summary_df = fetch_supply_dest_summary_data(
            engine,
            schema,
            classification_mode,
            demand_aggregation_mode,
            quarters_df,
            months_df,
            weeks_df,
            years_df,
            rolling_data=rolling_df,
            current_date=current_as_of_date,
        )
        return summary_df.to_dict('records') if not summary_df.empty else []

    grouped_base_df, grouping_config = group_small_supply_dest_countries(
        filtered_base_df,
        classification_mode,
        demand_aggregation_mode,
        as_of_date=current_as_of_date,
        return_grouping_config=True,
    )
    grouped_baseline_df = group_small_supply_dest_countries(
        filtered_baseline_df,
        classification_mode,
        demand_aggregation_mode,
        grouping_config=grouping_config,
    )

    return {
        'format': EXPORTERS_SUPPLY_DEST_SUMMARY_FORMAT,
        'show_all': _build_payload_records(
            filtered_base_df,
            filtered_baseline_df,
        ),
        'group_small_countries': _build_payload_records(
            grouped_base_df,
            grouped_baseline_df,
        ),
        'comparison': comparison_metadata,
    }


def fetch_global_supply_data(engine, schema, classification_mode='Country', rolling_avg_days=DEFAULT_SUPPLY_ROLLING_AVG_DAYS):
    """Fetch daily global LNG supply data for seasonal chart

    Args:
        engine: Database engine
        schema: Database schema
        classification_mode: 'Country' or 'Classification Level 1'
        rolling_avg_days: Number of days to use for the rolling average
    """
    rolling_window_preceding_days = _supply_rolling_window_preceding_days(rolling_avg_days)

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
                SELECT snapshot_timestamp_utc as max_timestamp
                FROM {schema}.kpler_trade_snapshots
                WHERE run_kind = 'canonical' AND status = 'published'
                ORDER BY snapshot_date_utc DESC
                LIMIT 1
            ),
            -- Get all unique continents globally
            all_continents AS (
                SELECT DISTINCT
                    COALESCE(NULLIF(continent_destination_name, ''), 'Unknown') as continent_destination
                FROM {schema}.kpler_trades kt
                {classification_join_clause}
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
                    {internal_flow_filter}
            ),
            -- Get all dates in our range
            all_dates AS (
                SELECT generate_series(
                    '{SUPPLY_CHART_QUERY_START_DATE}'::date,
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
                    AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
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
                        ROWS BETWEEN {rolling_window_preceding_days} PRECEDING AND CURRENT ROW
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
            WHERE date >= '{SUPPLY_CHART_DISPLAY_START_DATE}'
            ORDER BY date
            """)

            df = pd.read_sql(query, conn)

            if df.empty:
                return pd.DataFrame()

            # Convert to datetime
            df['date'] = pd.to_datetime(df['date'])

            return df

    except Exception:
        return pd.DataFrame()


def fetch_country_supply_data(
    engine,
    schema,
    country_name,
    classification_mode='Country',
    rolling_avg_days=DEFAULT_SUPPLY_ROLLING_AVG_DAYS
):
    """Fetch daily LNG supply data for a specific country or classification

    Args:
        engine: Database engine
        schema: Database schema
        country_name: Country name or classification level name
        classification_mode: 'Country' or 'Classification Level 1'
        rolling_avg_days: Number of days to use for the rolling average
    """
    rolling_window_preceding_days = _supply_rolling_window_preceding_days(rolling_avg_days)

    try:
        with engine.connect() as conn:
            # Get daily aggregated data for specific country/classification with rolling average calculated in SQL
            if classification_mode == 'Classification Level 1':
                query = text(f"""
                WITH latest_data AS (
                    SELECT snapshot_timestamp_utc as max_timestamp
                    FROM {schema}.kpler_trade_snapshots
                    WHERE run_kind = 'canonical' AND status = 'published'
                    ORDER BY snapshot_date_utc DESC
                    LIMIT 1
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
                    AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
                    AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                        IS DISTINCT FROM COALESCE(mc.country_classification_level1, 'Unknown')
                ),
                -- Get all dates in our range
                all_dates AS (
                    SELECT generate_series(
                        '{SUPPLY_CHART_QUERY_START_DATE}'::date,
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
                    AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
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
                            ROWS BETWEEN {rolling_window_preceding_days} PRECEDING AND CURRENT ROW
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
                WHERE date >= '{SUPPLY_CHART_DISPLAY_START_DATE}'
                ORDER BY date
                """)
            else:
                query = text(f"""
                WITH latest_data AS (
                    SELECT snapshot_timestamp_utc as max_timestamp
                    FROM {schema}.kpler_trade_snapshots
                    WHERE run_kind = 'canonical' AND status = 'published'
                    ORDER BY snapshot_date_utc DESC
                    LIMIT 1
                ),
                -- Get all unique continents for this country
                all_continents AS (
                    SELECT DISTINCT
                        COALESCE(NULLIF(continent_destination_name, ''), 'Unknown') as continent_destination
                    FROM {schema}.kpler_trades kt
                    , latest_data ld
                    WHERE kt.upload_timestamp_utc = ld.max_timestamp
                        AND kt.origin_country_name = :country
                        AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
                        AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                            IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
                ),
                -- Get all dates in our range
                all_dates AS (
                    SELECT generate_series(
                        '{SUPPLY_CHART_QUERY_START_DATE}'::date,
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
                        AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
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
                            ROWS BETWEEN {rolling_window_preceding_days} PRECEDING AND CURRENT ROW
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
                WHERE date >= '{SUPPLY_CHART_DISPLAY_START_DATE}'
                ORDER BY date
                """)

            df = pd.read_sql(query, conn, params={"country": country_name})

            if df.empty:
                return pd.DataFrame()

            # Convert to datetime
            df['date'] = pd.to_datetime(df['date'])

            return df

    except Exception:
        return pd.DataFrame()


def build_rest_of_countries_supply_data(global_supply_df, visible_country_dfs):
    """Build residual supply for countries not shown as individual supply cards."""
    if global_supply_df is None or global_supply_df.empty:
        return pd.DataFrame()

    rest_df = global_supply_df.copy()
    if not {'date', 'rolling_avg'}.issubset(rest_df.columns):
        return pd.DataFrame()

    rest_df['date'] = pd.to_datetime(rest_df['date'], errors='coerce')
    rest_df['rolling_avg'] = pd.to_numeric(rest_df['rolling_avg'], errors='coerce').fillna(0)
    rest_df = rest_df[rest_df['date'].notna()].copy()
    if rest_df.empty:
        return pd.DataFrame()

    visible_total = pd.Series(0.0, index=rest_df['date'])
    for country_df in visible_country_dfs or []:
        if country_df is None or country_df.empty or not {'date', 'rolling_avg'}.issubset(country_df.columns):
            continue

        country_series_df = country_df[['date', 'rolling_avg']].copy()
        country_series_df['date'] = pd.to_datetime(country_series_df['date'], errors='coerce')
        country_series_df['rolling_avg'] = pd.to_numeric(
            country_series_df['rolling_avg'],
            errors='coerce'
        ).fillna(0)
        country_series_df = country_series_df[country_series_df['date'].notna()]
        if country_series_df.empty:
            continue

        country_series = country_series_df.groupby('date')['rolling_avg'].sum()
        visible_total = visible_total.add(country_series, fill_value=0)

    rest_df = rest_df.set_index('date')
    rest_df['rolling_avg'] = (
        rest_df['rolling_avg'] - visible_total.reindex(rest_df.index, fill_value=0)
    ).clip(lower=0)
    return rest_df.reset_index()


def _split_supply_chart_batch_df(batch_df, entity_names):
    """Return one chart dataframe per entity from a batched supply query."""
    chart_columns = ['date', 'year', 'day_of_year', 'month_day', 'rolling_avg', 'is_forecast']
    empty_df = pd.DataFrame(columns=chart_columns)
    if batch_df is None or batch_df.empty:
        return {entity_name: empty_df.copy() for entity_name in entity_names}

    prepared_df = batch_df.copy()
    prepared_df['date'] = pd.to_datetime(prepared_df['date'])
    result = {}
    for entity_name in entity_names:
        entity_df = prepared_df[prepared_df['entity_name'] == entity_name].copy()
        if entity_df.empty:
            result[entity_name] = empty_df.copy()
            continue
        result[entity_name] = (
            entity_df[chart_columns]
            .sort_values('date')
            .reset_index(drop=True)
        )
    return result


def _fetch_country_supply_chart_batch(engine, schema, rolling_avg_days=DEFAULT_SUPPLY_ROLLING_AVG_DAYS):
    """Fetch Global and configured country supply chart data in one SQL round trip."""
    rolling_window_preceding_days = _supply_rolling_window_preceding_days(rolling_avg_days)
    entity_names = ['Global'] + SUPPLY_CHART_VISIBLE_COUNTRIES

    query = text(f"""
        WITH latest_data AS (
            SELECT snapshot_timestamp_utc as max_timestamp
            FROM {schema}.kpler_trade_snapshots
            WHERE run_kind = 'canonical' AND status = 'published'
            ORDER BY snapshot_date_utc DESC
            LIMIT 1
        ),
        active_entities AS (
            SELECT DISTINCT
                'Global'::text as entity_name
            FROM {schema}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
                AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                    IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')

            UNION

            SELECT DISTINCT
                kt.origin_country_name as entity_name
            FROM {schema}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name IN :entity_names
                AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
                AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                    IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
        ),
        all_dates AS (
            SELECT generate_series(
                '{SUPPLY_CHART_QUERY_START_DATE}'::date,
                (CURRENT_DATE + INTERVAL '14 days')::date,
                '1 day'::interval
            )::date as date
        ),
        date_entity_matrix AS (
            SELECT
                d.date,
                ae.entity_name
            FROM all_dates d
            CROSS JOIN active_entities ae
        ),
        daily_exports_raw AS (
            SELECT
                'Global'::text as entity_name,
                kt.start::date as date,
                SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
            FROM {schema}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
                AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                    IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
            GROUP BY kt.start::date

            UNION ALL

            SELECT
                kt.origin_country_name as entity_name,
                kt.start::date as date,
                SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
            FROM {schema}.kpler_trades kt, latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.origin_country_name IN :entity_names
                AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
                AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                    IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
            GROUP BY
                kt.origin_country_name,
                kt.start::date
        ),
        daily_supply AS (
            SELECT
                dem.entity_name,
                dem.date,
                COALESCE(der.daily_export_mcmd, 0) as mcmd
            FROM date_entity_matrix dem
            LEFT JOIN daily_exports_raw der
                ON dem.date = der.date
                AND dem.entity_name = der.entity_name
        ),
        rolling_supply AS (
            SELECT
                entity_name,
                date,
                mcmd,
                AVG(mcmd) OVER (
                    PARTITION BY entity_name
                    ORDER BY date
                    ROWS BETWEEN {rolling_window_preceding_days} PRECEDING AND CURRENT ROW
                ) as rolling_avg,
                CASE
                    WHEN date > CURRENT_DATE THEN true
                    ELSE false
                END as is_forecast
            FROM daily_supply
        )
        SELECT
            entity_name,
            date,
            EXTRACT(YEAR FROM date) as year,
            EXTRACT(DOY FROM date) as day_of_year,
            TO_CHAR(date, 'Mon DD') as month_day,
            rolling_avg,
            is_forecast
        FROM rolling_supply
        WHERE date >= '{SUPPLY_CHART_DISPLAY_START_DATE}'
        ORDER BY entity_name, date
    """).bindparams(bindparam('entity_names', expanding=True))

    with engine.connect() as conn:
        batch_df = pd.read_sql(
            query,
            conn,
            params={'entity_names': SUPPLY_CHART_VISIBLE_COUNTRIES}
        )

    return _split_supply_chart_batch_df(batch_df, entity_names)


def _fetch_classification_supply_chart_batch(
    engine,
    schema,
    classification_groups,
    rolling_avg_days=DEFAULT_SUPPLY_ROLLING_AVG_DAYS
):
    """Fetch Global and classification-group supply chart data in one SQL round trip."""
    entity_names = ['Global'] + list(classification_groups or [])
    if not classification_groups:
        return {'Global': fetch_global_supply_data(engine, schema, 'Classification Level 1', rolling_avg_days)}

    rolling_window_preceding_days = _supply_rolling_window_preceding_days(rolling_avg_days)

    query = text(f"""
        WITH latest_data AS (
            SELECT snapshot_timestamp_utc as max_timestamp
            FROM {schema}.kpler_trade_snapshots
            WHERE run_kind = 'canonical' AND status = 'published'
            ORDER BY snapshot_date_utc DESC
            LIMIT 1
        ),
        active_entities AS (
            SELECT DISTINCT
                'Global'::text as entity_name
            FROM {schema}.kpler_trades kt
            LEFT JOIN {schema}.mappings_country mc_origin ON kt.origin_country_name = mc_origin.country
            LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
            , latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
                AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                    IS DISTINCT FROM COALESCE(mc_origin.country_classification_level1, 'Unknown')

            UNION

            SELECT DISTINCT
                mc_origin.country_classification_level1 as entity_name
            FROM {schema}.kpler_trades kt
            INNER JOIN {schema}.mappings_country mc_origin ON kt.origin_country_name = mc_origin.country
            LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
            , latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND mc_origin.country_classification_level1 IN :entity_names
                AND mc_origin.country_classification_level1 IS NOT NULL
                AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
                AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                    IS DISTINCT FROM COALESCE(mc_origin.country_classification_level1, 'Unknown')
        ),
        all_dates AS (
            SELECT generate_series(
                '{SUPPLY_CHART_QUERY_START_DATE}'::date,
                (CURRENT_DATE + INTERVAL '14 days')::date,
                '1 day'::interval
            )::date as date
        ),
        date_entity_matrix AS (
            SELECT
                d.date,
                ae.entity_name
            FROM all_dates d
            CROSS JOIN active_entities ae
        ),
        daily_exports_raw AS (
            SELECT
                'Global'::text as entity_name,
                kt.start::date as date,
                SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
            FROM {schema}.kpler_trades kt
            LEFT JOIN {schema}.mappings_country mc_origin ON kt.origin_country_name = mc_origin.country
            LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
            , latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
                AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                    IS DISTINCT FROM COALESCE(mc_origin.country_classification_level1, 'Unknown')
            GROUP BY kt.start::date

            UNION ALL

            SELECT
                mc_origin.country_classification_level1 as entity_name,
                kt.start::date as date,
                SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
            FROM {schema}.kpler_trades kt
            INNER JOIN {schema}.mappings_country mc_origin ON kt.origin_country_name = mc_origin.country
            LEFT JOIN {schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
            , latest_data ld
            WHERE kt.upload_timestamp_utc = ld.max_timestamp
                AND mc_origin.country_classification_level1 IN :entity_names
                AND mc_origin.country_classification_level1 IS NOT NULL
                AND kt.start >= '{SUPPLY_CHART_QUERY_START_DATE}'
                AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                    IS DISTINCT FROM COALESCE(mc_origin.country_classification_level1, 'Unknown')
            GROUP BY
                mc_origin.country_classification_level1,
                kt.start::date
        ),
        daily_supply AS (
            SELECT
                dem.entity_name,
                dem.date,
                COALESCE(der.daily_export_mcmd, 0) as mcmd
            FROM date_entity_matrix dem
            LEFT JOIN daily_exports_raw der
                ON dem.date = der.date
                AND dem.entity_name = der.entity_name
        ),
        rolling_supply AS (
            SELECT
                entity_name,
                date,
                mcmd,
                AVG(mcmd) OVER (
                    PARTITION BY entity_name
                    ORDER BY date
                    ROWS BETWEEN {rolling_window_preceding_days} PRECEDING AND CURRENT ROW
                ) as rolling_avg,
                CASE
                    WHEN date > CURRENT_DATE THEN true
                    ELSE false
                END as is_forecast
            FROM daily_supply
        )
        SELECT
            entity_name,
            date,
            EXTRACT(YEAR FROM date) as year,
            EXTRACT(DOY FROM date) as day_of_year,
            TO_CHAR(date, 'Mon DD') as month_day,
            rolling_avg,
            is_forecast
        FROM rolling_supply
        WHERE date >= '{SUPPLY_CHART_DISPLAY_START_DATE}'
        ORDER BY entity_name, date
    """).bindparams(bindparam('entity_names', expanding=True))

    with engine.connect() as conn:
        batch_df = pd.read_sql(
            query,
            conn,
            params={'entity_names': list(classification_groups)}
        )

    return _split_supply_chart_batch_df(batch_df, entity_names)


def _fetch_supply_chart_data_legacy(
    engine,
    schema,
    classification_mode='Country',
    rolling_avg_days=DEFAULT_SUPPLY_ROLLING_AVG_DAYS
):
    """Fallback path matching the prior per-entity supply-chart queries."""
    charts_data = {}
    global_supply_df = fetch_global_supply_data(
        engine,
        schema,
        classification_mode,
        rolling_avg_days
    )
    charts_data['Global'] = global_supply_df

    if classification_mode == 'Classification Level 1':
        for group in get_all_classification_groups(engine, schema):
            charts_data[group] = fetch_country_supply_data(
                engine,
                schema,
                group,
                classification_mode,
                rolling_avg_days
            )
        return charts_data

    visible_country_dfs = []
    for country in SUPPLY_CHART_VISIBLE_COUNTRIES:
        country_df = fetch_country_supply_data(
            engine,
            schema,
            country,
            classification_mode,
            rolling_avg_days
        )
        visible_country_dfs.append(country_df)
        charts_data[country] = country_df

    charts_data[SUPPLY_CHART_REST_OF_COUNTRIES_LABEL] = build_rest_of_countries_supply_data(
        global_supply_df,
        visible_country_dfs
    )
    return charts_data


def fetch_supply_chart_data(
    engine,
    schema,
    classification_mode='Country',
    rolling_avg_days=DEFAULT_SUPPLY_ROLLING_AVG_DAYS
):
    """Fetch all supply chart entities with a batched query and legacy fallback."""
    classification_mode = classification_mode or 'Country'
    rolling_avg_days = normalize_supply_rolling_avg_days(rolling_avg_days)
    entity_names = _get_exporter_entity_names(
        engine,
        schema,
        classification_mode,
    )
    return _fetch_supply_chart_data_for_entities(
        engine,
        schema,
        classification_mode,
        rolling_avg_days,
        entity_names,
    )


def _get_exporter_entity_names(
    engine,
    schema,
    classification_mode='Country',
):
    classification_mode = classification_mode or 'Country'
    if classification_mode == 'Classification Level 1':
        return list(dict.fromkeys(
            ['Global'] + get_all_classification_groups(engine, schema)
        ))
    return (
        ['Global']
        + list(SUPPLY_CHART_VISIBLE_COUNTRIES)
        + [SUPPLY_CHART_REST_OF_COUNTRIES_LABEL]
    )


def _fetch_supply_chart_data_for_entities(
    engine,
    schema,
    classification_mode,
    rolling_avg_days,
    entity_names,
):
    """Fetch chart data for a precomputed, deterministically ordered roster."""
    classification_mode = classification_mode or 'Country'
    rolling_avg_days = normalize_supply_rolling_avg_days(rolling_avg_days)
    entity_names = list(dict.fromkeys(entity_names or []))
    try:
        if classification_mode == 'Classification Level 1':
            classification_groups = [
                entity_name
                for entity_name in entity_names
                if entity_name != 'Global'
            ]
            return _fetch_classification_supply_chart_batch(
                engine,
                schema,
                classification_groups,
                rolling_avg_days
            )

        chart_dfs = _fetch_country_supply_chart_batch(engine, schema, rolling_avg_days)
        visible_country_dfs = [
            chart_dfs.get(country, pd.DataFrame())
            for country in SUPPLY_CHART_VISIBLE_COUNTRIES
        ]
        chart_dfs[SUPPLY_CHART_REST_OF_COUNTRIES_LABEL] = build_rest_of_countries_supply_data(
            chart_dfs.get('Global', pd.DataFrame()),
            visible_country_dfs
        )
        return chart_dfs
    except Exception:
        return _fetch_supply_chart_data_legacy(
            engine,
            schema,
            classification_mode,
            rolling_avg_days
        )


def _normalize_supply_destination_base_frame(df):
    """Normalize one exact Kpler snapshot to the destination-table contract."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df['supply_classification'] = (
        df['supply_classification']
        .fillna('Unknown')
        .astype(str)
        .str.strip()
    )
    df['supply_country'] = (
        df['supply_country'].fillna('Unknown').astype(str).str.strip()
    )
    df['supply_installation'] = (
        df['supply_installation']
        .fillna('Unknown')
        .astype(str)
        .str.strip()
    )
    df.loc[df['supply_installation'] == '', 'supply_installation'] = 'Unknown'
    df['demand_classification'] = (
        df['demand_classification']
        .fillna('Unknown')
        .astype(str)
        .str.strip()
    )
    df['demand_country'] = (
        df['demand_country'].fillna('Unknown').astype(str).str.strip()
    )
    df['mcmd'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0) * 0.6 / 1000
    df['flow_date'] = pd.to_datetime(df['flow_date'])
    return df


def _fetch_supply_destination_snapshot_data(
    engine,
    schema,
    snapshot_timestamp_utc=None,
    as_of_date=None,
    window_days=None,
):
    """Fetch one exact snapshot, optionally bounded to an inclusive N-day window."""
    exact_snapshot = (
        snapshot_timestamp_utc is not None
        and as_of_date is not None
    )
    params = {}
    if exact_snapshot:
        as_of_timestamp = pd.Timestamp(as_of_date).normalize()
        params = {
            'snapshot_timestamp_utc': pd.Timestamp(snapshot_timestamp_utc),
            'end_exclusive': as_of_timestamp + pd.Timedelta(days=1),
        }
        if window_days is None:
            start_filter = "AND kt.start >= TIMESTAMP '2022-01-01'"
        else:
            params['window_start'] = (
                as_of_timestamp - pd.Timedelta(days=int(window_days) - 1)
            )
            start_filter = 'AND kt.start >= :window_start'
        source_cte = ''
        source_join = ''
        source_filter = (
            'kt.upload_timestamp_utc = :snapshot_timestamp_utc'
        )
        end_filter = 'AND kt.start < :end_exclusive'
    else:
        source_cte = f"""
            WITH latest_data AS (
                SELECT snapshot_timestamp_utc AS max_timestamp
                FROM {schema}.kpler_trade_snapshots
                WHERE run_kind = 'canonical' AND status = 'published'
                ORDER BY snapshot_date_utc DESC
                LIMIT 1
            )
        """
        source_join = ', latest_data ld'
        source_filter = 'kt.upload_timestamp_utc = ld.max_timestamp'
        start_filter = "AND kt.start >= TIMESTAMP '2022-01-01'"
        end_filter = 'AND kt.start::date <= CURRENT_DATE'

    try:
        with engine.connect() as conn:
            base_query = text(f"""
            {source_cte}
            SELECT
                COALESCE(mc_origin.country_classification_level1, 'Unknown') as supply_classification,
                kt.origin_country_name as supply_country,
                COALESCE(NULLIF(BTRIM(kt.installation_origin_name), ''), 'Unknown') as supply_installation,
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
            {source_join}
            WHERE {source_filter}
                AND kt.start IS NOT NULL
                {start_filter}
                {end_filter}
            """)
            df = pd.read_sql(base_query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    return _normalize_supply_destination_base_frame(df)


def fetch_supply_destination_base_data(
    engine,
    schema,
    snapshot_timestamp_utc=None,
    as_of_date=None,
):
    """Fetch the full current-vintage input for destination summary periods."""
    return _fetch_supply_destination_snapshot_data(
        engine,
        schema,
        snapshot_timestamp_utc,
        as_of_date,
        None,
    )


def fetch_supply_destination_pbd_base_data(
    engine,
    schema,
    snapshot_timestamp_utc,
    as_of_date,
):
    """Fetch only the prior-vintage rows required by its 30D/7D windows."""
    return _fetch_supply_destination_snapshot_data(
        engine,
        schema,
        snapshot_timestamp_utc,
        as_of_date,
        30,
    )


def _build_supply_dest_small_country_grouping(
    df,
    classification_mode='Country',
    demand_aggregation_mode='None',
    threshold_mcmd=10,
    lookback_months=24,
    as_of_date=None,
):
    """Build one current-vintage grouping map reusable by both snapshots."""
    country_col, parent_cols = get_supply_dest_small_country_grouping_config(
        classification_mode,
        demand_aggregation_mode,
    )
    pair_cols = parent_cols + [country_col]
    empty_config = {
        'country_col': country_col,
        'pair_cols': pair_cols,
        'small_pairs': pd.DataFrame(columns=pair_cols),
    }
    if df is None or df.empty or country_col not in df.columns:
        return empty_config

    current_timestamp = pd.Timestamp(
        as_of_date if as_of_date is not None else datetime.now()
    ).normalize()
    current_month = current_timestamp.to_period('M')
    start_month = current_month - (lookback_months - 1)
    lookback_df = df[
        df['flow_date'].dt.to_period('M') >= start_month
    ].copy()
    if lookback_df.empty:
        return empty_config

    lookback_df['__month_period'] = lookback_df['flow_date'].dt.to_period('M')
    monthly_totals = (
        lookback_df.groupby(
            pair_cols + ['__month_period'],
            dropna=False,
        )['mcmd']
        .sum()
        .reset_index()
    )
    if monthly_totals.empty:
        return empty_config

    monthly_totals['__days'] = monthly_totals['__month_period'].apply(
        lambda month_period: (
            current_timestamp.day
            if month_period == current_month
            else month_period.days_in_month
        )
    )
    monthly_totals['__monthly_mcmd'] = (
        monthly_totals['mcmd'] / monthly_totals['__days']
    ).fillna(0)
    max_monthly_by_pair = (
        monthly_totals.groupby(
            pair_cols,
            dropna=False,
        )['__monthly_mcmd']
        .max()
        .reset_index()
    )
    all_pairs = df[pair_cols].drop_duplicates()
    pair_threshold_df = all_pairs.merge(
        max_monthly_by_pair,
        on=pair_cols,
        how='left',
    )
    pair_threshold_df['__monthly_mcmd'] = (
        pair_threshold_df['__monthly_mcmd'].fillna(0)
    )
    small_pairs = pair_threshold_df[
        pair_threshold_df['__monthly_mcmd'] <= threshold_mcmd
    ][pair_cols].copy()
    return {
        'country_col': country_col,
        'pair_cols': pair_cols,
        'small_pairs': small_pairs,
    }


def _apply_supply_dest_small_country_grouping(df, grouping_config):
    """Apply a frozen current-vintage grouping map to a supply frame."""
    if df is None or df.empty:
        return df
    grouping_config = grouping_config or {}
    country_col = grouping_config.get('country_col')
    pair_cols = grouping_config.get('pair_cols') or []
    small_pairs = grouping_config.get('small_pairs')
    if (
        not country_col
        or country_col not in df.columns
        or not pair_cols
        or any(column_name not in df.columns for column_name in pair_cols)
        or not isinstance(small_pairs, pd.DataFrame)
        or small_pairs.empty
    ):
        return df.copy()

    grouped_df = df.copy()
    small_pairs = small_pairs[pair_cols].drop_duplicates().copy()
    small_pairs['__group_small_country'] = True
    grouped_df = grouped_df.merge(
        small_pairs,
        on=pair_cols,
        how='left',
    )
    grouped_df['__group_small_country'] = (
        grouped_df['__group_small_country'].eq(True)
    )
    grouped_df.loc[
        grouped_df['__group_small_country'],
        country_col,
    ] = 'Rest of countries'
    return grouped_df.drop(columns='__group_small_country')


def group_small_supply_dest_countries(
    df,
    classification_mode='Country',
    demand_aggregation_mode='None',
    threshold_mcmd=10,
    lookback_months=24,
    as_of_date=None,
    grouping_config=None,
    return_grouping_config=False,
):
    """Group small countries using one explicit, reusable grouping taxonomy."""
    if grouping_config is None:
        grouping_config = _build_supply_dest_small_country_grouping(
            df,
            classification_mode,
            demand_aggregation_mode,
            threshold_mcmd,
            lookback_months,
            as_of_date,
        )
    grouped_df = _apply_supply_dest_small_country_grouping(
        df,
        grouping_config,
    )
    if return_grouping_config:
        return grouped_df, grouping_config
    return grouped_df


def _apply_supply_dest_period_totals(result, classification_mode='Country', demand_aggregation_mode='None'):
    """Add hierarchy subtotal rows required by the expandable supply-destination table."""
    if result.empty:
        return result

    numeric_cols = [
        col for col in result.columns
        if col not in SUPPLY_DEST_SOURCE_TEXT_COLUMNS
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


def _build_supply_dest_period_matrix(df, current_date, period_view='monthly',
                                     classification_mode='Country', demand_aggregation_mode='None'):
    """Build a period matrix for monthly, quarterly, seasonal, or yearly views."""
    if df.empty:
        return pd.DataFrame()

    period_df = df.copy()
    _, current_period_start, _ = _get_supply_dest_current_period_details(
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
        & (period_df['__period_start'] < current_period_start)
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

    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    for col in pivot.columns:
        days = _get_supply_dest_period_days(col, period_view)
        pivot[col] = (pivot[col] / days).round(1) if days else 0

    if period_view == 'seasonally':
        label_map = (
            period_df[['__period_start', '__period_label']]
            .drop_duplicates(subset=['__period_start'])
            .set_index('__period_start')['__period_label']
            .to_dict()
        )
        pivot.columns = [label_map.get(pd.Timestamp(col), '') for col in pivot.columns]
    else:
        pivot.columns = [_format_supply_dest_period_label(col, period_view) for col in pivot.columns]

    result = pivot.reset_index()
    result = _apply_supply_dest_period_totals(result, classification_mode, demand_aggregation_mode)
    return result


def fetch_supply_destination_data(engine, schema, classification_mode='Country',
                                  demand_aggregation_mode='None', base_df=None,
                                  current_date=None):
    """Fetch bilateral trade flow data for the default supply-destination table."""
    current_date = pd.Timestamp(
        current_date if current_date is not None else datetime.now()
    )
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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    years_df = process_supply_dest_years(df, current_date, classification_mode, demand_aggregation_mode)
    quarters_df = process_supply_dest_quarters(df, current_date, classification_mode, demand_aggregation_mode)
    months_df = process_supply_dest_months(df, current_date, classification_mode, demand_aggregation_mode)
    weeks_df = process_supply_dest_weeks(df, current_date, classification_mode, demand_aggregation_mode)

    return years_df, quarters_df, months_df, weeks_df


def process_supply_dest_years(df, current_date, classification_mode='Country', demand_aggregation_mode='None'):
    """Process supply-destination data for completed years."""
    return _build_supply_dest_period_matrix(
        df,
        current_date,
        'yearly',
        classification_mode,
        demand_aggregation_mode
    )


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

    # Include the current week plus enough completed weeks for selector comparisons.
    current_week = pd.Period(current_date, freq='W')
    start_week = current_week - SUPPLY_DEST_PRELOAD_WEEK_COUNT
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


def _continent_chart_plot_dates(day_of_year_series):
    return pd.to_datetime(f'{CONTINENT_CHART_ANCHOR_YEAR}-01-01') + pd.to_timedelta(
        day_of_year_series.astype(int) - 1,
        unit='d'
    )


def _continent_chart_line_style(year, current_year, is_forecast=False):
    is_current_year = int(year) == int(current_year)
    return {
        'width': CONTINENT_CHART_CURRENT_YEAR_WIDTH if is_current_year else CONTINENT_CHART_PREVIOUS_YEAR_WIDTH,
        'opacity': 0.94 if is_current_year and not is_forecast else 0.72 if is_current_year else 0.44,
        'dash': CONTINENT_CHART_FORECAST_DASH if is_forecast else 'solid'
    }


def _empty_continent_chart_figure(message, height=328):
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
        height=height,
        margin=dict(l=36, r=20, t=12, b=36),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff'
    )
    return fig


def _apply_continent_chart_layout(fig, y_title, yaxis_range=None):
    yaxis_config = dict(
        title=dict(text=y_title, font=dict(size=11, color='#475569')),
        showgrid=True,
        gridcolor='rgba(148, 163, 184, 0.22)',
        gridwidth=0.5,
        linecolor='rgba(148, 163, 184, 0.6)',
        linewidth=1,
        tickfont=dict(size=10, color='#64748b'),
        zeroline=True,
        zerolinecolor='rgba(148, 163, 184, 0.28)',
        zerolinewidth=1
    )
    if yaxis_range is not None:
        yaxis_config['range'] = yaxis_range
    else:
        yaxis_config['autorange'] = True

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
                pd.Timestamp(year=CONTINENT_CHART_ANCHOR_YEAR, month=1, day=1),
                pd.Timestamp(year=CONTINENT_CHART_ANCHOR_YEAR, month=12, day=31)
            ],
            showspikes=True,
            spikemode='across',
            spikecolor='rgba(15, 23, 42, 0.18)',
            spikethickness=1
        ),
        yaxis=yaxis_config,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.12,
            xanchor='left',
            x=0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            borderwidth=0,
            font=dict(size=10, color='#475569'),
            itemsizing='constant',
            itemwidth=30
        ),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        margin=dict(l=44, r=18, t=12, b=42),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.96)',
            bordercolor='rgba(148, 163, 184, 0.7)',
            font=dict(size=11, color='#0f172a'),
            align='left'
        ),
        height=328,
        transition=dict(duration=300, easing='cubic-in-out')
    )


def _get_continent_entity_filter(entity_name, classification_mode):
    if entity_name == "Global":
        return "", {}

    if classification_mode == 'Classification Level 1':
        return "AND mc.country_classification_level1 = %(entity_name)s", {'entity_name': entity_name}

    if entity_name == SUPPLY_CHART_REST_OF_COUNTRIES_LABEL:
        return (
            "AND COALESCE(kt.origin_country_name, '') != ALL(%(excluded_origin_countries)s)",
            {'excluded_origin_countries': SUPPLY_CHART_VISIBLE_COUNTRIES}
        )

    return "AND kt.origin_country_name = %(entity_name)s", {'entity_name': entity_name}


def _continent_chart_year_token(year):
    try:
        if pd.isna(year):
            return None
    except TypeError:
        pass

    try:
        return str(int(float(year)))
    except (TypeError, ValueError):
        token = str(year).strip()
        return token or None


def _continent_chart_year_sort_key(year):
    try:
        return (0, int(year))
    except (TypeError, ValueError):
        return (1, str(year))


def _get_continent_chart_available_years():
    start_year = pd.Timestamp(CONTINENT_CHART_DISPLAY_START_DATE).year
    end_year = (datetime.now() + timedelta(days=14)).year
    if end_year < start_year:
        return [str(start_year)]
    return [str(year) for year in range(start_year, end_year + 1)]


def _default_continent_chart_selected_years(available_years):
    return available_years[-CONTINENT_CHART_DEFAULT_SELECTED_YEAR_COUNT:]


def _normalize_continent_chart_selected_years(selected_years, available_years, use_default=True):
    available_set = set(available_years)
    normalized = [
        token for token in (_continent_chart_year_token(year) for year in (selected_years or []))
        if token in available_set
    ]
    if normalized or not use_default:
        return sorted(set(normalized), key=_continent_chart_year_sort_key)
    return _default_continent_chart_selected_years(available_years)


def _get_continent_chart_selected_window(selected_years):
    available_years = _get_continent_chart_available_years()
    active_years = _normalize_continent_chart_selected_years(
        selected_years,
        available_years,
        use_default=selected_years is None
    )
    if not active_years:
        return active_years, CONTINENT_CHART_QUERY_START_DATE, CONTINENT_CHART_DISPLAY_START_DATE

    first_selected_year = int(active_years[0])
    focus_year = int(active_years[-1])
    earliest_available_year = pd.Timestamp(CONTINENT_CHART_DISPLAY_START_DATE).year
    first_required_year = max(min(first_selected_year, focus_year - 1), earliest_available_year)
    display_start_date = f'{first_required_year}-01-01'
    query_start_date = (
        pd.Timestamp(year=first_required_year, month=1, day=1) - pd.DateOffset(months=2)
    ).strftime('%Y-%m-%d')
    return active_years, query_start_date, display_start_date


def fetch_continent_chart_data_batch(
    engine,
    db_schema,
    entity_names,
    classification_mode='Country',
    selected_years=None,
    rolling_avg_days=DEFAULT_SUPPLY_ROLLING_AVG_DAYS
):
    """Fetch continent chart source data for all visible entities in one SQL query."""
    entity_names = list(dict.fromkeys(entity_names or []))
    if not entity_names:
        return pd.DataFrame()

    requested_years, query_start_date, display_start_date = _get_continent_chart_selected_window(selected_years)
    if not requested_years:
        return pd.DataFrame()

    classification_mode = classification_mode or 'Country'
    rolling_window_preceding_days = _supply_rolling_window_preceding_days(rolling_avg_days)
    bind_params = []
    params = {}

    entity_continent_selects = []
    daily_export_selects = []

    if classification_mode == 'Classification Level 1':
        classification_entities = [entity for entity in entity_names if entity != 'Global']

        if 'Global' in entity_names:
            entity_continent_selects.append(f"""
                SELECT DISTINCT
                    'Global'::text as entity_name,
                    COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination
                FROM {db_schema}.kpler_trades kt
                LEFT JOIN {db_schema}.mappings_country mc_origin ON kt.origin_country_name = mc_origin.country
                LEFT JOIN {db_schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start >= '{query_start_date}'
                    AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                        IS DISTINCT FROM COALESCE(mc_origin.country_classification_level1, 'Unknown')
            """)
            daily_export_selects.append(f"""
                SELECT
                    'Global'::text as entity_name,
                    kt.start::date as date,
                    COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination,
                    SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
                FROM {db_schema}.kpler_trades kt
                LEFT JOIN {db_schema}.mappings_country mc_origin ON kt.origin_country_name = mc_origin.country
                LEFT JOIN {db_schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start >= '{query_start_date}'
                    AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                    AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                        IS DISTINCT FROM COALESCE(mc_origin.country_classification_level1, 'Unknown')
                GROUP BY kt.start::date, COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown')
            """)

        if classification_entities:
            params['classification_entities'] = classification_entities
            bind_params.append(bindparam('classification_entities', expanding=True))
            entity_continent_selects.append(f"""
                SELECT DISTINCT
                    mc_origin.country_classification_level1 as entity_name,
                    COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination
                FROM {db_schema}.kpler_trades kt
                INNER JOIN {db_schema}.mappings_country mc_origin ON kt.origin_country_name = mc_origin.country
                LEFT JOIN {db_schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND mc_origin.country_classification_level1 IN :classification_entities
                    AND mc_origin.country_classification_level1 IS NOT NULL
                    AND kt.start >= '{query_start_date}'
                    AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                        IS DISTINCT FROM COALESCE(mc_origin.country_classification_level1, 'Unknown')
            """)
            daily_export_selects.append(f"""
                SELECT
                    mc_origin.country_classification_level1 as entity_name,
                    kt.start::date as date,
                    COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination,
                    SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
                FROM {db_schema}.kpler_trades kt
                INNER JOIN {db_schema}.mappings_country mc_origin ON kt.origin_country_name = mc_origin.country
                LEFT JOIN {db_schema}.mappings_country mc_dest ON kt.destination_country_name = mc_dest.country
                , latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND mc_origin.country_classification_level1 IN :classification_entities
                    AND mc_origin.country_classification_level1 IS NOT NULL
                    AND kt.start >= '{query_start_date}'
                    AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                    AND COALESCE(mc_dest.country_classification_level1, 'Unknown')
                        IS DISTINCT FROM COALESCE(mc_origin.country_classification_level1, 'Unknown')
                GROUP BY
                    mc_origin.country_classification_level1,
                    kt.start::date,
                    COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown')
            """)
    else:
        country_entities = [
            entity for entity in entity_names
            if entity not in {'Global', SUPPLY_CHART_REST_OF_COUNTRIES_LABEL}
        ]
        include_rest = SUPPLY_CHART_REST_OF_COUNTRIES_LABEL in entity_names

        if 'Global' in entity_names:
            entity_continent_selects.append(f"""
                SELECT DISTINCT
                    'Global'::text as entity_name,
                    COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination
                FROM {db_schema}.kpler_trades kt, latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start >= '{query_start_date}'
                    AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                        IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
            """)
            daily_export_selects.append(f"""
                SELECT
                    'Global'::text as entity_name,
                    kt.start::date as date,
                    COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination,
                    SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
                FROM {db_schema}.kpler_trades kt, latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.start >= '{query_start_date}'
                    AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                    AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                        IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
                GROUP BY kt.start::date, COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown')
            """)

        if country_entities:
            params['country_entities'] = country_entities
            bind_params.append(bindparam('country_entities', expanding=True))
            entity_continent_selects.append(f"""
                SELECT DISTINCT
                    kt.origin_country_name as entity_name,
                    COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination
                FROM {db_schema}.kpler_trades kt, latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.origin_country_name IN :country_entities
                    AND kt.start >= '{query_start_date}'
                    AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                        IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
            """)
            daily_export_selects.append(f"""
                SELECT
                    kt.origin_country_name as entity_name,
                    kt.start::date as date,
                    COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination,
                    SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
                FROM {db_schema}.kpler_trades kt, latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND kt.origin_country_name IN :country_entities
                    AND kt.start >= '{query_start_date}'
                    AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                    AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                        IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
                GROUP BY
                    kt.origin_country_name,
                    kt.start::date,
                    COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown')
            """)

        if include_rest:
            params['excluded_origin_countries'] = SUPPLY_CHART_VISIBLE_COUNTRIES
            bind_params.append(bindparam('excluded_origin_countries', expanding=True))
            entity_continent_selects.append(f"""
                SELECT DISTINCT
                    '{SUPPLY_CHART_REST_OF_COUNTRIES_LABEL}'::text as entity_name,
                    COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination
                FROM {db_schema}.kpler_trades kt, latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND COALESCE(kt.origin_country_name, '') NOT IN :excluded_origin_countries
                    AND kt.start >= '{query_start_date}'
                    AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                        IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
            """)
            daily_export_selects.append(f"""
                SELECT
                    '{SUPPLY_CHART_REST_OF_COUNTRIES_LABEL}'::text as entity_name,
                    kt.start::date as date,
                    COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown') as continent_destination,
                    SUM(kt.cargo_origin_cubic_meters * 0.6 / 1000) as daily_export_mcmd
                FROM {db_schema}.kpler_trades kt, latest_data ld
                WHERE kt.upload_timestamp_utc = ld.max_timestamp
                    AND COALESCE(kt.origin_country_name, '') NOT IN :excluded_origin_countries
                    AND kt.start >= '{query_start_date}'
                    AND kt.start::date <= CURRENT_DATE + INTERVAL '14 days'
                    AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                        IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
                GROUP BY kt.start::date, COALESCE(NULLIF(kt.continent_destination_name, ''), 'Unknown')
            """)

    if not entity_continent_selects or not daily_export_selects:
        return pd.DataFrame()

    entity_continents_sql = "\nUNION\n".join(entity_continent_selects)
    daily_exports_sql = "\nUNION ALL\n".join(daily_export_selects)
    query = text(f"""
        WITH latest_data AS (
            SELECT snapshot_timestamp_utc as max_timestamp
            FROM {db_schema}.kpler_trade_snapshots
            WHERE run_kind = 'canonical' AND status = 'published'
            ORDER BY snapshot_date_utc DESC
            LIMIT 1
        ),
        entity_continents AS (
            {entity_continents_sql}
        ),
        all_dates AS (
            SELECT generate_series(
                '{query_start_date}'::date,
                (CURRENT_DATE + INTERVAL '14 days')::date,
                '1 day'::interval
            )::date as date
        ),
        date_entity_continent_matrix AS (
            SELECT
                d.date,
                ec.entity_name,
                ec.continent_destination
            FROM all_dates d
            CROSS JOIN entity_continents ec
        ),
        daily_exports_raw AS (
            {daily_exports_sql}
        ),
        daily_exports_complete AS (
            SELECT
                decm.date,
                decm.entity_name,
                decm.continent_destination,
                COALESCE(der.daily_export_mcmd, 0) as daily_export_mcmd
            FROM date_entity_continent_matrix decm
            LEFT JOIN daily_exports_raw der
                ON decm.date = der.date
                AND decm.entity_name = der.entity_name
                AND decm.continent_destination = der.continent_destination
        ),
        rolling_continents AS (
            SELECT
                date,
                entity_name,
                continent_destination,
                daily_export_mcmd,
                AVG(daily_export_mcmd) OVER (
                    PARTITION BY entity_name, continent_destination
                    ORDER BY date
                    ROWS BETWEEN {rolling_window_preceding_days} PRECEDING AND CURRENT ROW
                ) as rolling_avg_30d,
                CASE
                    WHEN date > CURRENT_DATE THEN true
                    ELSE false
                END as is_forecast
            FROM daily_exports_complete
        ),
        rolling_with_totals AS (
            SELECT
                rc.*,
                SUM(rolling_avg_30d) OVER (PARTITION BY entity_name, date) as total_rolling_avg_30d
            FROM rolling_continents rc
        )
        SELECT
            entity_name,
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
        FROM rolling_with_totals
        WHERE date >= '{display_start_date}'
        ORDER BY entity_name, continent_destination, date
    """)
    if bind_params:
        query = query.bindparams(*bind_params)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def _get_continent_year_selector_options():
    return [
        {
            'label': html.Span(year, className='continent-year-chip-text'),
            'value': year
        }
        for year in _get_continent_chart_available_years()
    ]


def _format_continent_kpi_value(value, chart_type, vol_label, is_delta=False, include_unit=True):
    if value is None or pd.isna(value):
        return 'n/a'

    rounded_value = int(round(float(value)))
    sign = '+' if is_delta and rounded_value > 0 else ''
    if chart_type == 'percentage':
        return f'{sign}{rounded_value}pp' if is_delta else f'{rounded_value}%'

    suffix = '' if is_delta or not include_unit else f' {vol_label}'
    return f'{sign}{rounded_value:,}{suffix}'


def _format_continent_kpi_pct(delta_pct):
    if delta_pct is None or pd.isna(delta_pct):
        return ''
    sign = '+' if delta_pct > 0 else ''
    return f' ({sign}{delta_pct:.0f}%)'


def _format_continent_kpi_pct_compact(delta_pct):
    if delta_pct is None or pd.isna(delta_pct):
        return None
    rounded_pct = int(round(float(delta_pct)))
    sign = '+' if rounded_pct > 0 else ''
    return f'({sign}{rounded_pct}%)'


def _continent_kpi_direction_class(value):
    if value is None or pd.isna(value):
        return 'continent-kpi-delta-neutral'
    if value > 0:
        return 'continent-kpi-delta-positive'
    if value < 0:
        return 'continent-kpi-delta-negative'
    return 'continent-kpi-delta-neutral'


def _continent_kpi_value_displays_zero(value, chart_type, is_delta_pct=False):
    if value is None or pd.isna(value):
        return True

    display_tolerance = 0.05 if chart_type == 'percentage' and not is_delta_pct else 0.5
    return abs(float(value)) < display_tolerance


def _continent_kpi_all_displayed_values_zero(
    chart_type,
    show_deltas,
    latest_value,
    mom_delta_value,
    mom_delta_pct,
    yoy_delta_value,
    yoy_delta_pct
):
    values_to_check = [
        (latest_value, False)
    ]

    if show_deltas:
        values_to_check.extend([
            (mom_delta_value, False),
            (yoy_delta_value, False)
        ])
        if chart_type != 'percentage':
            values_to_check.extend([
                (mom_delta_pct, True),
                (yoy_delta_pct, True)
            ])

    return all(
        _continent_kpi_value_displays_zero(value, chart_type, is_delta_pct)
        for value, is_delta_pct in values_to_check
    )


def _calculate_continent_kpis(df, active_years, metric_column, chart_type, vol_label):
    if df.empty or not active_years:
        return []

    focus_year = active_years[-1]
    kpi_df = df.copy()
    kpi_df['plot_date'] = _continent_chart_plot_dates(kpi_df['day_of_year'])
    kpi_df = kpi_df[kpi_df[metric_column].notna() & kpi_df['plot_date'].notna()].copy()
    if kpi_df.empty:
        return []

    focus_df = kpi_df[kpi_df['_year_token'] == focus_year].copy()
    if focus_df.empty:
        return []

    if 'is_forecast' in focus_df.columns:
        actual_focus_df = focus_df[~focus_df['is_forecast'].astype(bool)].copy()
        if not actual_focus_df.empty:
            focus_df = actual_focus_df

    try:
        previous_year = str(int(focus_year) - 1)
    except (TypeError, ValueError):
        previous_year = None

    metrics = []
    for continent in sorted(focus_df['continent_destination'].dropna().unique()):
        continent_focus_df = (
            focus_df[focus_df['continent_destination'] == continent]
            .dropna(subset=[metric_column])
            .sort_values('plot_date')
        )
        if continent_focus_df.empty:
            continue

        latest_point = continent_focus_df.tail(1).iloc[0]
        latest_value = latest_point[metric_column]
        latest_plot_date = latest_point['plot_date']

        month_ago_date = latest_plot_date - pd.DateOffset(months=1)
        mom_candidates = continent_focus_df[continent_focus_df['plot_date'] <= month_ago_date]
        mom_delta_value = None
        mom_delta_pct = None
        if not mom_candidates.empty:
            mom_value = mom_candidates.tail(1).iloc[0][metric_column]
            if pd.notna(mom_value):
                mom_delta_value = latest_value - mom_value
                if mom_value != 0 and chart_type != 'percentage':
                    mom_delta_pct = mom_delta_value / abs(mom_value) * 100

        yoy_delta_value = None
        yoy_delta_pct = None
        if previous_year:
            yoy_candidates = (
                kpi_df[
                    (kpi_df['continent_destination'] == continent)
                    & (kpi_df['_year_token'] == previous_year)
                    & (kpi_df['plot_date'] <= latest_plot_date)
                    & kpi_df[metric_column].notna()
                ]
                .sort_values('plot_date')
            )
            if not yoy_candidates.empty:
                yoy_value = yoy_candidates.tail(1).iloc[0][metric_column]
                if pd.notna(yoy_value):
                    yoy_delta_value = latest_value - yoy_value
                    if yoy_value != 0 and chart_type != 'percentage':
                        yoy_delta_pct = yoy_delta_value / abs(yoy_value) * 100

        latest_numeric = float(latest_value) if pd.notna(latest_value) else None
        mom_delta_numeric = float(mom_delta_value) if mom_delta_value is not None and pd.notna(mom_delta_value) else None
        mom_pct_numeric = float(mom_delta_pct) if mom_delta_pct is not None and pd.notna(mom_delta_pct) else None
        yoy_delta_numeric = float(yoy_delta_value) if yoy_delta_value is not None and pd.notna(yoy_delta_value) else None
        yoy_pct_numeric = float(yoy_delta_pct) if yoy_delta_pct is not None and pd.notna(yoy_delta_pct) else None
        show_deltas = continent != 'Unknown'

        if _continent_kpi_all_displayed_values_zero(
            chart_type,
            show_deltas,
            latest_numeric,
            mom_delta_numeric,
            mom_pct_numeric,
            yoy_delta_numeric,
            yoy_pct_numeric
        ):
            continue

        metrics.append({
            'continent': continent,
            'color': CONTINENT_CHART_COLOR_MAP.get(continent, '#64748b'),
            'show_deltas': show_deltas,
            'chart_type': chart_type,
            'unit_label': vol_label,
            'latest_value': latest_numeric,
            'latest_text': _format_continent_kpi_value(latest_value, chart_type, vol_label, include_unit=False),
            'latest_label': latest_point.get('month_day', ''),
            'mom_delta_value': mom_delta_numeric,
            'mom_value_text': (
                _format_continent_kpi_value(mom_delta_value, chart_type, vol_label, is_delta=True)
            ) if mom_delta_value is not None and pd.notna(mom_delta_value) else 'n/a',
            'mom_pct_text': _format_continent_kpi_pct_compact(mom_delta_pct),
            'mom_text': (
                _format_continent_kpi_value(mom_delta_value, chart_type, vol_label, is_delta=True)
                + _format_continent_kpi_pct(mom_delta_pct)
            ) if mom_delta_value is not None and pd.notna(mom_delta_value) else 'n/a',
            'mom_class': _continent_kpi_direction_class(mom_delta_value),
            'mom_delta_pct': mom_pct_numeric,
            'yoy_delta_value': yoy_delta_numeric,
            'yoy_value_text': (
                _format_continent_kpi_value(yoy_delta_value, chart_type, vol_label, is_delta=True)
            ) if yoy_delta_value is not None and pd.notna(yoy_delta_value) else 'n/a',
            'yoy_pct_text': _format_continent_kpi_pct_compact(yoy_delta_pct),
            'yoy_text': (
                _format_continent_kpi_value(yoy_delta_value, chart_type, vol_label, is_delta=True)
                + _format_continent_kpi_pct(yoy_delta_pct)
            ) if yoy_delta_value is not None and pd.notna(yoy_delta_value) else 'n/a',
            'yoy_delta_pct': yoy_pct_numeric,
            'yoy_class': _continent_kpi_direction_class(yoy_delta_value)
        })

    return sorted(metrics, key=lambda item: item['latest_value'], reverse=True)


def _continent_kpi_summary_column_sort_key(continent):
    preferred_order = {
        'Asia': 0,
        'Europe': 1,
        'Americas': 2,
        'Africa': 3,
        'Oceania': 4,
        'Middle East': 5,
        'North America': 6,
        'South America': 7,
        'Unknown': 98
    }
    return (preferred_order.get(continent, 50), continent)


def _build_continent_kpi_summary_value_cell(value_text, class_name=''):
    return html.Td(
        html.Span(value_text, className='continent-kpi-summary-value'),
        className=f'continent-kpi-summary-cell continent-kpi-summary-value-cell {class_name}'.strip()
    )


def _build_continent_kpi_summary_delta_cell(
    value_text,
    class_name,
    is_available=True,
    pct_text=None,
    title_text=None,
    role_class=''
):
    if not is_available or value_text in (None, 'n/a'):
        return html.Td(
            html.Span('-', className='continent-kpi-summary-empty-value'),
            className=(
                'continent-kpi-summary-cell continent-kpi-summary-delta-cell '
                f'continent-kpi-summary-cell-empty {role_class}'
            ).strip()
        )

    cell_content = html.Span(
        [
            html.Span(value_text, className='continent-kpi-summary-delta-main'),
            html.Span(pct_text, className='continent-kpi-summary-delta-pct') if pct_text else None
        ],
        className='continent-kpi-summary-delta-stack'
    )

    return html.Td(
        cell_content,
        className=f'continent-kpi-summary-cell continent-kpi-summary-delta-cell {role_class} {class_name}'.strip(),
        title=title_text or value_text
    )


def _build_continent_kpi_summary_transposed_cell(metric, metric_key, entity_name):
    entity_class = (
        'continent-kpi-summary-entity-cell-primary'
        if entity_name == 'Global'
        else ''
    )

    if metric_key == 'Current':
        value_text = metric['latest_text'] if metric else '-'
        if not metric:
            return html.Td(
                html.Span('-', className='continent-kpi-summary-empty-value'),
                className=(
                    'continent-kpi-summary-cell continent-kpi-summary-value-cell '
                    f'continent-kpi-summary-current-cell continent-kpi-summary-cell-empty {entity_class}'
                ).strip()
            )
        return _build_continent_kpi_summary_value_cell(
            value_text,
            class_name=f'continent-kpi-summary-current-cell {entity_class}'.strip()
        )

    if metric_key == 'MoM':
        return _build_continent_kpi_summary_delta_cell(
            metric.get('mom_value_text', metric['mom_text']) if metric else None,
            f"{metric['mom_class']} {entity_class}".strip() if metric else entity_class,
            is_available=metric.get('show_deltas', True) if metric else False,
            pct_text=metric.get('mom_pct_text') if metric else None,
            title_text=metric.get('mom_text') if metric else None,
            role_class='continent-kpi-summary-mom-cell'
        )

    return _build_continent_kpi_summary_delta_cell(
        metric.get('yoy_value_text', metric['yoy_text']) if metric else None,
        f"{metric['yoy_class']} {entity_class}".strip() if metric else entity_class,
        is_available=metric.get('show_deltas', True) if metric else False,
        pct_text=metric.get('yoy_pct_text') if metric else None,
        title_text=metric.get('yoy_text') if metric else None,
        role_class='continent-kpi-summary-yoy-cell'
    )


def _build_continent_kpi_summary_table(entity_kpi_rows):
    active_rows = [row for row in entity_kpi_rows if row['metrics']]
    if not active_rows:
        return html.Div(
            'KPI data unavailable',
            className='continent-kpi-summary continent-kpi-summary-empty'
        )

    continents = sorted(
        {
            metric['continent']
            for row in active_rows
            for metric in row['metrics']
        },
        key=_continent_kpi_summary_column_sort_key
    )
    subcolumns = [
        ('Current', 'Now'),
        ('MoM', 'MoM'),
        ('YoY', 'YoY')
    ]
    entity_names = [row['entity'] for row in entity_kpi_rows]
    metrics_by_entity = {
        row['entity']: {
            metric['continent']: metric
            for metric in row['metrics']
        }
        for row in entity_kpi_rows
    }

    return html.Div(
        [
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th(
                                            'Destination',
                                            className='continent-kpi-summary-axis-header continent-kpi-summary-continent-axis-header'
                                        ),
                                        html.Th(
                                            'Metric',
                                            className='continent-kpi-summary-axis-header continent-kpi-summary-metric-axis-header'
                                        )
                                    ]
                                    + [
                                        html.Th(
                                            entity_name,
                                            className=(
                                                'continent-kpi-summary-entity-header '
                                                'continent-kpi-summary-entity-header-primary'
                                                if entity_name == 'Global'
                                                else 'continent-kpi-summary-entity-header'
                                            ),
                                            title=entity_name
                                        )
                                        for entity_name in entity_names
                                    ],
                                    className='continent-kpi-summary-entity-header-row'
                                )
                            ]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    (
                                        [
                                            html.Th(
                                                [
                                                    html.Span(
                                                        className='continent-kpi-summary-swatch',
                                                        style={'backgroundColor': CONTINENT_CHART_COLOR_MAP.get(continent, '#64748b')}
                                                    ),
                                                    html.Span(continent)
                                                ],
                                                rowSpan=len(metric_rows),
                                                className='continent-kpi-summary-continent-axis-cell'
                                            )
                                        ]
                                        if metric_index == 0
                                        else []
                                    )
                                    + [
                                        html.Th(
                                            metric_label,
                                            className=(
                                                'continent-kpi-summary-metric-cell '
                                                f'continent-kpi-summary-metric-cell-{metric_key.lower()}'
                                            ),
                                            title=metric_key
                                        )
                                    ]
                                    + [
                                        _build_continent_kpi_summary_transposed_cell(
                                            metrics_by_entity.get(entity_name, {}).get(continent),
                                            metric_key,
                                            entity_name
                                        )
                                        for entity_name in entity_names
                                    ],
                                    className=(
                                        'continent-kpi-summary-row '
                                        f'continent-kpi-summary-row-{metric_key.lower()} '
                                        + ('continent-kpi-summary-continent-group-start' if metric_index == 0 else '')
                                    )
                                )
                                for continent in continents
                                for metric_rows in ([subcolumns[:1]] if continent == 'Unknown' else [subcolumns])
                                for metric_index, (metric_key, metric_label) in enumerate(metric_rows)
                            ]
                        )
                    ],
                    className='continent-kpi-summary-table'
                ),
                className='continent-kpi-summary-table-wrap'
            )
        ],
        className='continent-kpi-summary'
    )


def _create_continent_destination_chart_from_df(
    df,
    volume_metric='mcm_d',
    selected_years=None,
    rolling_avg_days=DEFAULT_SUPPLY_ROLLING_AVG_DAYS
):
    """Create the absolute continent chart from a preloaded dataframe."""
    vol_info = _get_volume_metric_info(volume_metric)
    vol_label = vol_info['label']

    try:
        requested_years, _, _ = _get_continent_chart_selected_window(selected_years)
        if not requested_years:
            return _empty_continent_chart_figure("Select a year above.")

        if df is None or df.empty:
            return _empty_continent_chart_figure("No export data available.")

        df = _convert_volume_metric_dataframe(
            df.copy(),
            volume_metric,
            columns=['rolling_avg'],
            period_days=rolling_avg_days
        )
        df['_year_token'] = df['year'].apply(_continent_chart_year_token)

        data_years = sorted(
            [year for year in df['_year_token'].dropna().unique()],
            key=_continent_chart_year_sort_key
        )
        active_years = [year for year in requested_years if year in set(data_years)]
        if not active_years:
            return _empty_continent_chart_figure("No data for the selected years.")

        kpi_metrics = _calculate_continent_kpis(df, active_years, 'rolling_avg', 'absolute', vol_label)

        df = df[df['_year_token'].isin(active_years)].copy()
        if df.empty:
            return _empty_continent_chart_figure("No data for the selected years.")

        fig = go.Figure()
        years = sorted(active_years, key=_continent_chart_year_sort_key)
        continents = sorted(df['continent_destination'].unique())
        current_year = int(years[-1])
        continent_legend_shown = {}

        for continent in continents:
            continent_data = df[df['continent_destination'] == continent]
            color = CONTINENT_CHART_COLOR_MAP.get(continent, '#64748b')

            for year in years:
                year_continent_data = continent_data[continent_data['_year_token'] == year].copy()
                if year_continent_data.empty:
                    continue

                year_continent_data['plot_date'] = _continent_chart_plot_dates(year_continent_data['day_of_year'])
                historical_data = year_continent_data[~year_continent_data['is_forecast']]
                forecast_data = year_continent_data[year_continent_data['is_forecast']]
                historical_style = _continent_chart_line_style(year, current_year)
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
                        line=dict(color=color, width=historical_style['width'], dash=historical_style['dash']),
                        opacity=historical_style['opacity'],
                        hovertemplate=(
                            f'<b>{continent}</b> | {year} | '
                            '%{text} | '
                            f'%{{y:,.0f}} {vol_label}<extra></extra>'
                        ),
                        text=historical_data['month_day'],
                        showlegend=show_legend
                    ))

                if not forecast_data.empty:
                    if not historical_data.empty:
                        connect_data = pd.concat([historical_data.tail(1), forecast_data])
                    else:
                        connect_data = forecast_data

                    forecast_style = _continent_chart_line_style(year, current_year, is_forecast=True)
                    fig.add_trace(go.Scatter(
                        x=connect_data['plot_date'],
                        y=connect_data['rolling_avg'],
                        mode='lines',
                        name=None,
                        legendgroup=continent,
                        line=dict(color=color, width=forecast_style['width'], dash=forecast_style['dash']),
                        opacity=forecast_style['opacity'],
                        hovertemplate=(
                            f'<b>{continent}</b> | {year} forecast | '
                            '%{text} | '
                            f'%{{y:,.0f}} {vol_label}<extra></extra>'
                        ),
                        text=connect_data['month_day'],
                        showlegend=False
                    ))

        _apply_continent_chart_layout(fig, vol_label)
        fig.update_layout(meta={'continent_kpis': kpi_metrics})
        return fig

    except Exception:
        return _empty_continent_chart_figure("Error loading data")


def _create_continent_percentage_chart_from_df(df, selected_years=None):
    """Create the percentage continent chart from a preloaded dataframe."""
    try:
        requested_years, _, _ = _get_continent_chart_selected_window(selected_years)
        if not requested_years:
            return _empty_continent_chart_figure("Select a year above.")

        if df is None or df.empty:
            return _empty_continent_chart_figure("No export data available.")

        df = df.copy()
        df['_year_token'] = df['year'].apply(_continent_chart_year_token)

        data_years = sorted(
            [year for year in df['_year_token'].dropna().unique()],
            key=_continent_chart_year_sort_key
        )
        active_years = [year for year in requested_years if year in set(data_years)]
        if not active_years:
            return _empty_continent_chart_figure("No data for the selected years.")

        kpi_metrics = _calculate_continent_kpis(df, active_years, 'percentage', 'percentage', '%')

        df = df[df['_year_token'].isin(active_years)].copy()
        if df.empty:
            return _empty_continent_chart_figure("No data for the selected years.")

        fig = go.Figure()
        years = sorted(active_years, key=_continent_chart_year_sort_key)
        continents = sorted(df['continent_destination'].unique())
        current_year = int(years[-1])
        continent_legend_shown = {}

        for continent in continents:
            continent_data = df[df['continent_destination'] == continent]
            color = CONTINENT_CHART_COLOR_MAP.get(continent, '#64748b')

            for year in years:
                year_continent_data = continent_data[continent_data['_year_token'] == year].copy()
                if year_continent_data.empty:
                    continue

                year_continent_data['plot_date'] = _continent_chart_plot_dates(year_continent_data['day_of_year'])
                historical_data = year_continent_data[~year_continent_data['is_forecast']]
                forecast_data = year_continent_data[year_continent_data['is_forecast']]
                historical_style = _continent_chart_line_style(year, current_year)
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
                        line=dict(color=color, width=historical_style['width'], dash=historical_style['dash']),
                        opacity=historical_style['opacity'],
                        hovertemplate=(
                            f'<b>{continent}</b> | {year} | '
                            '%{text} | '
                            '%{y:.1f}%<extra></extra>'
                        ),
                        text=historical_data['month_day'],
                        showlegend=show_legend
                    ))

                if not forecast_data.empty:
                    if not historical_data.empty:
                        connect_data = pd.concat([historical_data.tail(1), forecast_data])
                    else:
                        connect_data = forecast_data

                    forecast_style = _continent_chart_line_style(year, current_year, is_forecast=True)
                    fig.add_trace(go.Scatter(
                        x=connect_data['plot_date'],
                        y=connect_data['percentage'],
                        mode='lines',
                        name=None,
                        legendgroup=continent,
                        line=dict(color=color, width=forecast_style['width'], dash=forecast_style['dash']),
                        opacity=forecast_style['opacity'],
                        hovertemplate=(
                            f'<b>{continent}</b> | {year} forecast | '
                            '%{text} | '
                            '%{y:.1f}%<extra></extra>'
                        ),
                        text=connect_data['month_day'],
                        showlegend=False
                    ))

        _apply_continent_chart_layout(fig, '%', yaxis_range=[0, 100])
        fig.update_layout(meta={'continent_kpis': kpi_metrics})
        return fig

    except Exception:
        return _empty_continent_chart_figure("Error loading data")


def _build_supply_dest_columns(display_df, view_type='absolute', hidden_cols=None, delta_like_cols=None):
    """Build DataTable column definitions for supply-destination tables."""
    hidden_cols = set(hidden_cols or ['30D_Y1'])
    hidden_cols.update(SUPPLY_DEST_ROLLING_REFERENCE_COLUMNS)
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


def _format_supply_dest_width_sample(value, view_type='absolute', is_numeric=False,
                                     is_delta=False, is_preformatted_pp=False):
    """Return a compact display-like string used only for AG Grid width estimation."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''

    if not is_numeric:
        return str(value)

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if view_type == 'percentage' and is_delta:
        pp_value = numeric_value if is_preformatted_pp else numeric_value * 100
        sign = '+' if pp_value > 0 else ''
        return f"{sign}{pp_value:,.0f} pp"
    if view_type == 'percentage':
        return f"{numeric_value:.0%}"
    return f"{numeric_value:,.0f}"


def _build_supply_dest_summary_column_width_styles(display_df, columns, view_type='absolute',
                                                   delta_like_cols=None):
    """Size LNG Supply by Destination columns from their headers and visible values."""
    if display_df.empty:
        return []

    width_styles = []
    delta_like_cols = set(delta_like_cols or [])
    real_delta_ids = {col for col in display_df.columns if str(col).startswith('Δ ')}
    all_delta_ids = real_delta_ids | delta_like_cols
    text_width_limits = {
        'Aggregation Supply': (130, 190),
        'Aggregation Demand': (130, 190),
        'Country Demand': (120, 170),
        'Demand Country': (118, 170),
        'Import Country': (118, 170),
        'Import Classification': (136, 190),
        'Supply Country': (150, 210),
        'Supply Installation': (170, 250),
    }

    for column in columns:
        column_id = column.get('id')
        if not column_id or column_id not in display_df.columns:
            continue

        is_text = column_id in SUPPLY_DEST_TEXT_COLUMNS
        is_delta = column_id in all_delta_ids
        is_preformatted_pp = column_id in delta_like_cols and column_id not in real_delta_ids
        header_text = str(column.get('name') or column_id)
        header_chars = len(header_text)
        samples = [header_text]
        samples.extend(
            _format_supply_dest_width_sample(
                value,
                view_type,
                not is_text,
                is_delta=is_delta,
                is_preformatted_pp=is_preformatted_pp
            )
            for value in display_df[column_id].tolist()
        )
        max_value_chars = max((len(str(sample)) for sample in samples[1:]), default=0)
        max_chars = max(header_chars, max_value_chars)

        if is_text:
            min_width, max_width = text_width_limits.get(column_id, (110, 180))
            width = int(min(max(max_chars * 6.2 + 30, min_width), max_width))
        elif is_delta:
            width = int(min(max(header_chars * 7.2 + 30, max_value_chars * 6.6 + 32, 104), 128))
        else:
            width = int(min(max(header_chars * 7.0 + 30, max_value_chars * 6.8 + 32, 70), 104))

        width_styles.append({
            'if': {'column_id': column_id},
            'width': f'{width}px',
            'minWidth': f'{width}px',
            'maxWidth': f'{width}px'
        })

    return width_styles


def _build_supply_dest_summary_grid_display(display_df, columns, view_type='absolute',
                                            delta_like_cols=None):
    """Return display-only records/columns for the executive summary AG Grid."""
    grid_df = display_df.copy()
    grid_columns = [dict(column) for column in columns]
    delta_like_cols = set(delta_like_cols or [])
    numeric_ids = {
        column.get('id')
        for column in grid_columns
        if column.get('type') == 'numeric'
    }
    real_delta_ids = {column_id for column_id in numeric_ids if str(column_id).startswith('Δ ')}
    delta_ids = real_delta_ids | delta_like_cols
    preformatted_pp_ids = delta_like_cols - real_delta_ids

    for column_id in delta_ids:
        if column_id not in grid_df.columns:
            continue
        raw_field = SUPPLY_DEST_DELTA_RAW_FIELDS.get(column_id)
        if not raw_field:
            continue
        raw_values = pd.to_numeric(grid_df[column_id], errors='coerce')
        if view_type == 'percentage':
            raw_values = raw_values * 100
        raw_values = raw_values.mask(raw_values.abs() < 0.5, 0)
        grid_df[raw_field] = raw_values

    for column_id in numeric_ids:
        if column_id not in grid_df.columns:
            continue

        def format_value(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return (
                    '—'
                    if column_id in SUPPLY_DEST_PBD_DELTA_COLUMNS
                    else ''
                )
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return str(value)
            if view_type == 'percentage' and column_id in delta_ids:
                pp_value = numeric_value if column_id in preformatted_pp_ids else numeric_value * 100
                if abs(pp_value) < 0.5:
                    pp_value = 0
                sign = '+' if pp_value > 0 else ''
                return f'{sign}{pp_value:,.0f} pp'
            if view_type == 'percentage' and column_id not in delta_ids:
                return f'{numeric_value:.0%}'
            if column_id in delta_ids and abs(numeric_value) < 0.5:
                numeric_value = 0
            if (
                column_id in SUPPLY_DEST_PBD_DELTA_COLUMNS
                and numeric_value > 0
            ):
                return f'+{numeric_value:,.0f}'
            return f'{numeric_value:,.0f}'

        grid_df[column_id] = grid_df[column_id].apply(format_value)

    for column in grid_columns:
        if column.get('id') in numeric_ids:
            column['type'] = 'text'
            column.pop('format', None)

    return grid_df, grid_columns


def _apply_supply_dest_summary_comparison(display_df, comparison_metadata, view_type='absolute'):
    """Apply the selected comparison basis to visible summary level columns."""
    if display_df is None or display_df.empty:
        return display_df, []

    comparison_metadata = comparison_metadata or {}
    comparison_basis = _normalize_supply_dest_comparison_basis(
        comparison_metadata.get('comparison_basis')
    )
    if comparison_basis not in {'previous_period', 'same_period_last_year'}:
        return display_df, []

    visible_comparison_cols = comparison_metadata.get(
        'visible_comparison_cols',
        comparison_metadata.get('visible_period_cols', [])
    )
    visible_comparison_cols = [
        col for col in visible_comparison_cols
        if col in display_df.columns
    ]
    comparison_reference_map = comparison_metadata.get('comparison_reference_map') or {}
    comparison_source_df = display_df.copy()
    comparison_delta_cols = []

    for visible_col in visible_comparison_cols:
        reference_col = comparison_reference_map.get(visible_col)
        if reference_col in comparison_source_df.columns:
            visible_values = pd.to_numeric(comparison_source_df[visible_col], errors='coerce')
            reference_values = pd.to_numeric(comparison_source_df[reference_col], errors='coerce')
            display_df[visible_col] = visible_values - reference_values
            if view_type == 'percentage':
                display_df[visible_col] = display_df[visible_col] * 100
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


def _get_supply_dest_summary_column_family(column_id):
    """Classify supply-destination summary columns into visual time-window groups."""
    column_id = str(column_id)
    if column_id in SUPPLY_DEST_TEXT_COLUMNS:
        return 'label'
    if column_id == '30D':
        return 'rolling-30d'
    if column_id == '7D':
        return 'rolling-7d'
    if column_id == 'Δ 7D-30D':
        return 'delta-mom'
    if column_id == 'Δ 30D Y/Y':
        return 'delta-yoy'
    if column_id in SUPPLY_DEST_PBD_DELTA_COLUMNS:
        return 'delta-pbd'
    if _is_supply_dest_summary_year_column(column_id):
        return 'year'
    if column_id.startswith('Q') and "'" in column_id:
        return 'quarter'
    if column_id.startswith('W') and "'" in column_id:
        return 'week'
    if "'" in column_id:
        return 'month'
    return 'numeric'


def _apply_supply_dest_summary_column_classes(columns):
    """Attach scoped AG Grid header classes so period groups are visually distinct."""
    classed_columns = []
    previous_family = None
    for column in columns:
        column = dict(column)
        column_id = column.get('id')
        family = _get_supply_dest_summary_column_family(column_id)
        header_classes = [f'supply-dest-header-{family}']
        if family == 'label':
            header_classes.append(
                'supply-dest-header-label-primary'
                if column_id in {'Aggregation Supply', 'Supply Country'}
                else 'supply-dest-header-label-secondary'
            )
        if family != previous_family and family != 'label':
            header_classes.append('supply-dest-header-group-start')
        column['headerClass'] = ' '.join(header_classes)
        if family != 'label':
            cell_classes = ['supply-dest-summary-number-cell']
            if family in {'delta-mom', 'delta-yoy'}:
                cell_classes.append('supply-dest-summary-delta-cell')
            existing_cell_class = str(column.get('cellClass') or '').strip()
            column['cellClass'] = ' '.join(
                class_name for class_name in [existing_cell_class, *cell_classes] if class_name
            )
        classed_columns.append(column)
        previous_family = family
    return classed_columns


def _build_supply_dest_delta_heatmap_class_rules(display_df, column_id, value_scale=1):
    """Build AG Grid class rules for red/green delta heatmap bands."""
    raw_field = SUPPLY_DEST_DELTA_RAW_FIELDS.get(column_id)
    if not raw_field:
        return {}

    thresholds = _get_supply_dest_delta_thresholds(display_df, column_id, value_scale=value_scale)
    band_thresholds = [0, *thresholds]
    rules = {}
    for band_index, threshold in enumerate(band_thresholds, start=1):
        positive_threshold = _format_supply_dest_filter_number(threshold)
        negative_threshold = _format_supply_dest_filter_number(-threshold)
        rules[f'supply-dest-delta-positive-{band_index}'] = {
            'function': _build_supply_dest_numeric_filter_js(
                column_id,
                '>',
                positive_threshold,
                raw_field=raw_field
            )
        }
        rules[f'supply-dest-delta-negative-{band_index}'] = {
            'function': _build_supply_dest_numeric_filter_js(
                column_id,
                '<',
                negative_threshold,
                raw_field=raw_field
            )
        }
    return rules


def _apply_supply_dest_delta_heatmap_class_rules(columns, display_df, value_scale=1):
    """Attach heatmap class rules to visible delta columns."""
    styled_columns = []
    for column in columns:
        column = dict(column)
        column_id = column.get('id')
        if column_id in SUPPLY_DEST_DELTA_RAW_FIELDS:
            column['cellClassRules'] = _build_supply_dest_delta_heatmap_class_rules(
                display_df,
                column_id,
                value_scale=value_scale
            )
        styled_columns.append(column)
    return styled_columns


def _format_supply_dest_filter_number(value):
    """Format numeric thresholds for Dash filter_query expressions."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '0'

    text = f'{number:.6f}'.rstrip('0').rstrip('.')
    return text or '0'


def _get_supply_dest_delta_thresholds(display_df, column_id, value_scale=1):
    """Return compact quantile thresholds for delta heatmap styling."""
    if display_df is None or display_df.empty or column_id not in display_df.columns:
        return []

    total_mask = _is_supply_dest_grand_total_row(display_df)
    values = pd.to_numeric(display_df.loc[~total_mask, column_id], errors='coerce').abs() * value_scale
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


def _build_supply_dest_numeric_filter_js(column_id, operator, threshold_text, raw_field=None):
    """Build a numeric AG Grid condition from hidden raw values with display-text fallback."""
    escaped_column_id = str(column_id).replace("\\", "\\\\").replace("'", "\\'")
    if raw_field:
        escaped_raw_field = str(raw_field).replace("\\", "\\\\").replace("'", "\\'")
        return f"(params.data && params.data['{escaped_raw_field}'] {operator} {threshold_text})"

    return (
        f"(Number(String(params.data && params.data['{escaped_column_id}'] !== undefined "
        f"? params.data['{escaped_column_id}'] : '').replace(/[^0-9.\\-]/g, '')) "
        f"{operator} {threshold_text})"
    )


def _build_supply_dest_delta_gradient_styles(display_df, column_id, base_bg, border_color, value_scale=1):
    """Build subtle red/green heatmap bands for delta columns."""
    raw_field = SUPPLY_DEST_DELTA_RAW_FIELDS.get(column_id)
    base_style = {
        'if': {'column_id': column_id},
        'backgroundColor': base_bg,
        'borderLeft': f'2px solid {border_color}',
        'color': '#334155',
        'fontWeight': '700',
        'textAlign': 'right',
        'paddingRight': '12px'
    }
    styles = [base_style]
    thresholds = _get_supply_dest_delta_thresholds(display_df, column_id, value_scale=value_scale)

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
            threshold_text = _format_supply_dest_filter_number(threshold)
            styles.append({
                'if': {
                    'column_id': column_id,
                    'filter_query_js': _build_supply_dest_numeric_filter_js(
                        column_id,
                        operator,
                        f'{sign}{threshold_text}',
                        raw_field=raw_field
                    )
                },
                'backgroundColor': background,
                'borderLeft': f'2px solid {border_color}',
                'color': color,
                'fontWeight': weight,
                'textAlign': 'right',
                'paddingRight': '12px'
            })

    return styles


def prepare_supply_dest_table_for_display(df, classification_mode='Country',
                                          expanded_classifications=None, expanded_countries=None,
                                          expanded_supply_countries=None, view_type='absolute',
                                          demand_aggregation_mode='None'):
    """Prepare supply-destination data for display in DataTable with expandable rows

    Args:
        df: DataFrame with the supply-destination data
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
        numeric_cols = [col for col in row_df.columns if col not in SUPPLY_DEST_SOURCE_TEXT_COLUMNS]
        if not numeric_cols:
            return True
        return (row_df[numeric_cols] != 0).any().any()

    show_demand_aggregation = use_demand_classification_mode(classification_mode, demand_aggregation_mode)
    show_demand_country = use_demand_country_mode(demand_aggregation_mode)
    show_import_classification = use_import_classification_mode(demand_aggregation_mode)

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

    elif show_demand_country or show_import_classification:
        # Two-level hierarchy: Supply Country -> selected import-side aggregation
        import_col = 'demand_country' if show_demand_country else 'demand_classification'
        for supply_country in sorted(df['supply_country'].unique()):
            supply_country_rows = []

            for import_bucket in sorted(df[import_col].unique()):
                pair_data = df[(df['supply_country'] == supply_country) &
                              (df[import_col] == import_bucket)]

                if not pair_data.empty and has_non_zero_values(pair_data):
                    entity_totals_for_grand.append(pair_data.copy())
                    supply_country_rows.append(pair_data)

            if supply_country_rows:
                supply_total_df = safe_concat(supply_country_rows, ignore_index=True)
                numeric_cols = [col for col in supply_total_df.columns if col not in
                               SUPPLY_DEST_SOURCE_TEXT_COLUMNS]

                subtotal_payload = {
                    'supply_country': (
                        f"▼ {supply_country}"
                        if supply_country in expanded_supply_countries
                        else f"▶ {supply_country}"
                    ),
                    import_col: '',
                    **{col: supply_total_df[col].sum() for col in numeric_cols}
                }
                subtotal_row = pd.DataFrame([subtotal_payload])
                filtered_rows.append(subtotal_row)

                if supply_country in expanded_supply_countries:
                    detail_rows = supply_total_df.copy()
                    if not detail_rows.empty:
                        if '30D' in detail_rows.columns:
                            detail_rows = detail_rows.assign(
                                __sort_30d=pd.to_numeric(
                                    detail_rows['30D'],
                                    errors='coerce'
                                ).fillna(0)
                            ).sort_values(
                                ['__sort_30d', import_col],
                                ascending=[False, True],
                                kind='mergesort'
                            ).drop(columns='__sort_30d')
                        else:
                            detail_rows = detail_rows.sort_values(
                                import_col,
                                kind='mergesort'
                            )
                        detail_rows.loc[:, 'supply_country'] = ''
                        detail_rows.loc[:, import_col] = "    " + detail_rows[import_col].astype(str)
                        filtered_rows.append(detail_rows)

    else:
        if 'supply_installation' in df.columns:
            for supply_country in sorted(df['supply_country'].unique()):
                supply_country_data = df[df['supply_country'] == supply_country]
                if supply_country_data.empty:
                    continue

                numeric_cols = [
                    col for col in supply_country_data.columns
                    if col not in SUPPLY_DEST_SOURCE_TEXT_COLUMNS
                ]
                country_values = {
                    col: pd.to_numeric(supply_country_data[col], errors='coerce').fillna(0).sum()
                    for col in numeric_cols
                }
                country_total = pd.DataFrame([{
                    'supply_country': (
                        f"▼ {supply_country}"
                        if supply_country in expanded_supply_countries
                        else f"▶ {supply_country}"
                    ),
                    'supply_installation': '',
                    **country_values
                }])

                if not has_non_zero_values(country_total):
                    continue

                entity_totals_for_grand.append(pd.DataFrame([{
                    'supply_country': supply_country,
                    'supply_installation': '',
                    **country_values
                }]))
                filtered_rows.append(country_total)

                if supply_country in expanded_supply_countries:
                    installations = supply_country_data.copy()
                    non_zero_mask = installations.apply(
                        lambda row: has_non_zero_values(row.to_frame().T),
                        axis=1
                    )
                    installations = installations[non_zero_mask]

                    if not installations.empty:
                        if '30D' in installations.columns:
                            installations = installations.assign(
                                __sort_30d=pd.to_numeric(
                                    installations['30D'],
                                    errors='coerce'
                                ).fillna(0)
                            ).sort_values(
                                ['__sort_30d', 'supply_installation'],
                                ascending=[False, True],
                                kind='mergesort'
                            ).drop(columns='__sort_30d')
                        else:
                            installations = installations.sort_values(
                                'supply_installation',
                                kind='mergesort'
                            )
                        installations.loc[:, 'supply_country'] = ''
                        installations.loc[:, 'supply_installation'] = (
                            "    " + installations['supply_installation'].astype(str)
                        )
                        filtered_rows.append(installations)
        else:
            for supply_country in sorted(df['supply_country'].unique()):
                supply_country_data = df[df['supply_country'] == supply_country]
                if not supply_country_data.empty and has_non_zero_values(supply_country_data):
                    entity_totals_for_grand.append(supply_country_data.copy())
                    filtered_rows.append(supply_country_data.copy())

    # Add Grand Total row (only if not in percentage mode)
    if entity_totals_for_grand and view_type != 'percentage':
        grand_total_df = safe_concat(entity_totals_for_grand, ignore_index=True)
        numeric_cols = [col for col in grand_total_df.columns if col not in SUPPLY_DEST_SOURCE_TEXT_COLUMNS]

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
        elif show_import_classification:
            grand_total_row = pd.DataFrame([{
                'supply_country': 'GRAND TOTAL',
                'demand_classification': '',
                **{col: grand_total_df[col].sum() for col in numeric_cols}
            }])
        else:
            grand_total_payload = {
                'supply_country': 'GRAND TOTAL',
                **{col: grand_total_df[col].sum() for col in numeric_cols}
            }
            if 'supply_installation' in df.columns:
                grand_total_payload['supply_installation'] = ''
            grand_total_row = pd.DataFrame([grand_total_payload])

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
            elif col == 'supply_installation':
                new_columns.append('Supply Installation')
            elif col == 'demand_country' and show_demand_country:
                new_columns.append('Import Country')
            elif col == 'demand_classification' and show_import_classification:
                new_columns.append('Import Classification')
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


def _is_supply_dest_grand_total_row(display_df):
    """Identify GRAND TOTAL rows in supply-destination tables after display labels are applied."""
    if display_df is None or display_df.empty:
        return pd.Series(dtype=bool)

    grand_total_mask = pd.Series(False, index=display_df.index)
    for col in ['Aggregation Supply', 'Supply Country']:
        if col in display_df.columns:
            grand_total_mask = grand_total_mask | display_df[col].isin(SUPPLY_DEST_TOTAL_LABELS)
    return grand_total_mask


def _label_supply_dest_total_as_global(display_df):
    """Rename the summary total row to Global for executive presentation."""
    if display_df is None or display_df.empty:
        return display_df

    labeled_df = display_df.copy()
    total_mask = _is_supply_dest_grand_total_row(labeled_df)
    if not total_mask.any():
        return labeled_df

    for col in ['Aggregation Supply', 'Supply Country']:
        if col in labeled_df.columns:
            labeled_df.loc[total_mask & labeled_df[col].isin(SUPPLY_DEST_TOTAL_LABELS), col] = 'Global'
    return labeled_df


def _clean_supply_dest_sort_label(value):
    """Normalize hierarchy labels for stable alpha tie-breaks."""
    return str(value or '').strip().lstrip('▶▼').strip().lower()


def _has_supply_dest_expand_marker(value):
    """Return whether a displayed hierarchy label is expandable."""
    return str(value or '').strip().startswith(('▶', '▼'))


def _strip_supply_dest_expand_marker(value):
    """Return a hierarchy label without expand/collapse markers or indentation."""
    return str(value or '').strip().lstrip('▶▼').strip()


def _get_supply_dest_active_row(display_df, active_cell):
    """Resolve the clicked row, preferring AG Grid's row payload over row-index lookup."""
    if not active_cell:
        return None, None

    row_index = active_cell.get('row')
    try:
        row_index = int(row_index)
    except (TypeError, ValueError):
        row_index = None

    clicked_data = active_cell.get('data')
    if isinstance(clicked_data, dict) and clicked_data:
        return pd.Series(clicked_data), row_index

    if display_df is None or display_df.empty or row_index is None or row_index >= len(display_df):
        return None, row_index

    return display_df.iloc[row_index], row_index


def _sort_supply_dest_classification_blocks(display_df, sort_rules, total_position='top'):
    """Sort classification parent rows while keeping expanded detail rows attached."""
    if (
        display_df is None
        or display_df.empty
        or 'Aggregation Supply' not in display_df.columns
    ):
        return None

    valid_sort_rules = [
        sort_rule for sort_rule in (sort_rules or [])
        if sort_rule.get('column_id') in display_df.columns
    ]
    if not valid_sort_rules:
        return None

    total_mask = _is_supply_dest_grand_total_row(display_df)
    total_df = display_df.loc[total_mask].copy()
    body_df = display_df.loc[~total_mask].copy().reset_index(drop=True)
    if body_df.empty:
        return safe_concat([total_df, body_df], ignore_index=True)

    blocks = []
    row_index = 0
    while row_index < len(body_df):
        block_indices = [row_index]
        parent_label = str(body_df.iloc[row_index].get('Aggregation Supply', '') or '').strip()
        row_index += 1

        if parent_label:
            while row_index < len(body_df):
                next_label = str(body_df.iloc[row_index].get('Aggregation Supply', '') or '').strip()
                if next_label:
                    break
                block_indices.append(row_index)
                row_index += 1

        blocks.append(body_df.loc[block_indices].copy())

    sort_records = []
    sort_cols = []
    ascending = []
    for block_index, block_df in enumerate(blocks):
        parent_row = block_df.iloc[0]
        sort_record = {
            '__block_index': block_index,
            '__original_order': block_index,
            '__label': _clean_supply_dest_sort_label(parent_row.get('Aggregation Supply', ''))
        }
        for sort_index, sort_rule in enumerate(valid_sort_rules):
            column_id = sort_rule.get('column_id')
            sort_col = f'__sort_{sort_index}'
            if column_id in SUPPLY_DEST_TEXT_COLUMNS:
                sort_record[sort_col] = _clean_supply_dest_sort_label(parent_row.get(column_id, ''))
            else:
                sort_record[sort_col] = pd.to_numeric(
                    pd.Series([parent_row.get(column_id)]),
                    errors='coerce'
                ).iloc[0]
            if block_index == 0:
                sort_cols.append(sort_col)
                ascending.append(sort_rule.get('direction', 'asc') != 'desc')
        sort_records.append(sort_record)

    sort_df = pd.DataFrame(sort_records)
    sort_df = sort_df.sort_values(
        by=sort_cols + ['__label', '__original_order'],
        ascending=ascending + [True, True],
        na_position='last',
        kind='mergesort'
    )
    sorted_body_df = safe_concat(
        [blocks[int(block_index)] for block_index in sort_df['__block_index']],
        ignore_index=True
    )

    if total_position == 'bottom':
        return safe_concat([sorted_body_df, total_df], ignore_index=True)
    return safe_concat([total_df, sorted_body_df], ignore_index=True)


def _sort_supply_dest_country_terminal_blocks(display_df, sort_rules, total_position='top'):
    """Sort country parent rows while keeping expanded child rows attached below them."""
    if (
        display_df is None
        or display_df.empty
        or 'Supply Country' not in display_df.columns
    ):
        return None
    child_col = next(
        (
            col for col in ['Supply Installation', 'Import Country', 'Import Classification']
            if col in display_df.columns
        ),
        None
    )
    if child_col is None:
        return None

    valid_sort_rules = [
        sort_rule for sort_rule in (sort_rules or [])
        if sort_rule.get('column_id') in display_df.columns
    ]
    if not valid_sort_rules:
        return None

    total_mask = _is_supply_dest_grand_total_row(display_df)
    total_df = display_df.loc[total_mask].copy()
    body_df = display_df.loc[~total_mask].copy().reset_index(drop=True)
    if body_df.empty:
        return safe_concat([total_df, body_df], ignore_index=True)

    blocks = []
    row_index = 0
    while row_index < len(body_df):
        block_indices = [row_index]
        current_row = body_df.iloc[row_index]
        supply_label = str(current_row.get('Supply Country', '') or '').strip()
        child_label = str(current_row.get(child_col, '') or '').strip()
        row_index += 1

        if supply_label and not child_label:
            while row_index < len(body_df):
                next_row = body_df.iloc[row_index]
                next_supply_label = str(next_row.get('Supply Country', '') or '').strip()
                next_child_label = str(next_row.get(child_col, '') or '').strip()
                if next_supply_label or not next_child_label:
                    break
                block_indices.append(row_index)
                row_index += 1

        blocks.append(body_df.loc[block_indices].copy())

    sort_records = []
    sort_cols = []
    ascending = []
    for block_index, block_df in enumerate(blocks):
        parent_row = block_df.iloc[0]
        sort_record = {
            '__block_index': block_index,
            '__original_order': block_index,
            '__label': _clean_supply_dest_sort_label(parent_row.get('Supply Country', ''))
        }
        for sort_index, sort_rule in enumerate(valid_sort_rules):
            column_id = sort_rule.get('column_id')
            sort_col = f'__sort_{sort_index}'
            if column_id in SUPPLY_DEST_TEXT_COLUMNS:
                sort_record[sort_col] = _clean_supply_dest_sort_label(parent_row.get(column_id, ''))
            else:
                sort_record[sort_col] = pd.to_numeric(
                    pd.Series([parent_row.get(column_id)]),
                    errors='coerce'
                ).iloc[0]
            if block_index == 0:
                sort_cols.append(sort_col)
                ascending.append(sort_rule.get('direction', 'asc') != 'desc')
        sort_records.append(sort_record)

    sort_df = pd.DataFrame(sort_records)
    sort_df = sort_df.sort_values(
        by=sort_cols + ['__label', '__original_order'],
        ascending=ascending + [True, True],
        na_position='last',
        kind='mergesort'
    )
    sorted_body_df = safe_concat(
        [blocks[int(block_index)] for block_index in sort_df['__block_index']],
        ignore_index=True
    )

    if total_position == 'bottom':
        return safe_concat([sorted_body_df, total_df], ignore_index=True)
    return safe_concat([total_df, sorted_body_df], ignore_index=True)


def _sort_supply_dest_summary_display_df(display_df, sort_column='30D'):
    """Pin Global to the top and sort remaining visible rows by 30D descending."""
    if display_df is None or display_df.empty:
        return display_df

    classification_df = _sort_supply_dest_classification_blocks(
        display_df,
        [{'column_id': sort_column, 'direction': 'desc'}],
        total_position='top'
    )
    if classification_df is not None:
        return classification_df

    hierarchical_df = _sort_supply_dest_country_terminal_blocks(
        display_df,
        [{'column_id': sort_column, 'direction': 'desc'}],
        total_position='top'
    )
    if hierarchical_df is not None:
        return hierarchical_df

    total_mask = _is_supply_dest_grand_total_row(display_df)
    total_df = display_df.loc[total_mask].copy()
    sortable_df = display_df.loc[~total_mask].copy()

    if sortable_df.empty or sort_column not in sortable_df.columns:
        return safe_concat([total_df, sortable_df], ignore_index=True)

    sortable_df['__sort_30d'] = pd.to_numeric(sortable_df[sort_column], errors='coerce')
    label_cols = [col for col in SUPPLY_DEST_TEXT_COLUMNS if col in sortable_df.columns]
    if label_cols:
        sortable_df['__sort_label'] = (
            sortable_df[label_cols]
            .fillna('')
            .astype(str)
            .agg(' '.join, axis=1)
            .str.replace(r'^[\s▶▼]+', '', regex=True)
            .str.lower()
        )
        sort_cols = ['__sort_30d', '__sort_label']
        ascending = [False, True]
    else:
        sort_cols = ['__sort_30d']
        ascending = [False]

    sortable_df = sortable_df.sort_values(
        by=sort_cols,
        ascending=ascending,
        na_position='last',
        kind='mergesort'
    ).drop(columns=['__sort_30d', '__sort_label'], errors='ignore')

    return safe_concat([total_df, sortable_df], ignore_index=True)


def _create_top_exporters_selector_region():
    """Render the sticky exporter controls using the shared header format."""
    return [
        html.Div(
            [
                html.Div('Supply Classification', className='filter-group-header'),
                dcc.RadioItems(
                    id='country-classification-dropdown',
                    options=[
                        {'label': 'Country', 'value': 'Country'},
                        {'label': 'Classification Level 1', 'value': 'Classification Level 1'}
                    ],
                    value='Country',
                    inline=True,
                    className='supply-dest-view-selector exporters-sticky-selector exporters-classification-selector',
                    inputStyle={'display': 'none'},
                    labelStyle={'marginRight': '0'}
                )
            ],
            className='filter-group exporters-sticky-filter-group'
        ),
        html.Div(
            [
                html.Div('Group small', className='filter-group-header'),
                dcc.RadioItems(
                    id='supply-dest-country-grouping-dropdown',
                    options=SUPPLY_DEST_COUNTRY_GROUPING_OPTIONS,
                    value='group_small_countries',
                    inline=True,
                    className='supply-dest-view-selector exporters-sticky-selector exporters-grouping-selector',
                    inputStyle={'display': 'none'},
                    labelStyle={'marginRight': '0'}
                )
            ],
            className='filter-group exporters-sticky-filter-group'
        ),
        html.Div(
            [
                html.Div('Metric', className='filter-group-header'),
                dcc.RadioItems(
                    id='exporters-volume-metric-dropdown',
                    options=VOLUME_METRIC_OPTIONS,
                    value='mcm_d',
                    inline=True,
                    className='supply-dest-view-selector exporters-sticky-selector exporters-volume-selector',
                    inputStyle={'display': 'none'},
                    labelStyle={'marginRight': '0'}
                )
            ],
            className='filter-group exporters-sticky-filter-group'
        ),
        html.Div(
            [
                html.Div('Rolling Avg', className='filter-group-header'),
                html.Div(
                    [
                        dcc.Input(
                            id='supply-rolling-window-days-input',
                            type='number',
                            value=DEFAULT_SUPPLY_ROLLING_AVG_DAYS,
                            min=MIN_SUPPLY_ROLLING_AVG_DAYS,
                            max=MAX_SUPPLY_ROLLING_AVG_DAYS,
                            step=1,
                            debounce=0.8,
                            className='exporters-rolling-window-input'
                        ),
                        html.Span('days', className='exporters-rolling-window-unit')
                    ],
                    className='exporters-rolling-window-control'
                )
            ],
            className='filter-group exporters-sticky-filter-group exporters-rolling-filter-group'
        ),
    ]


# Dashboard layout
layout = html.Div([
    # Interval component to trigger initial data load (runs once on page load)
    dcc.Interval(id='initial-load-trigger', interval=1000*60*60*24, n_intervals=0, max_intervals=1),

    # Store components for caching data (memory is faster than local storage)
    dcc.Store(id='exporters-source-state-store', storage_type='memory'),
    dcc.Store(id='supply-charts-data', storage_type='memory'),  # Single store for all supply chart data
    dcc.Store(id='continent-charts-data', storage_type='memory'),  # Store for continent charts data
    dcc.Store(id='supply-dest-data-store', storage_type='memory'),  # Store for supply-destination data

    # Store for expanded states in supply-destination table
    dcc.Store(id='supply-dest-expanded-classifications', data=[]),  # For classification pairs
    dcc.Store(id='supply-dest-expanded-countries', data=[]),  # For demand countries
    dcc.Store(id='supply-dest-expanded-supply-countries', data=[]),  # For supply countries

    # Download components for Excel exports
    dcc.Download(id='download-supply-charts-excel'),
    dcc.Download(id='download-continent-charts-excel'),
    dcc.Download(id='download-supply-dest-excel'),

    html.Div(
        _create_top_exporters_selector_region(),
        className='professional-section-header exporters-sticky-filter-bar',
        style={
            'display': 'flex',
            'gap': '8px',
            'alignItems': 'center',
            'flexWrap': 'wrap',
            'margin': '0',
        }
    ),
    # Supply Charts Section - Dynamic container
    html.Div([
        html.Div([
            html.Div(
                [
                    html.H3(
                        _format_rolling_average_section_title(
                            'LNG Supply',
                            DEFAULT_SUPPLY_ROLLING_AVG_DAYS
                        ),
                        id='supply-rolling-section-title',
                        className="section-title-inline"
                    ),
                    html.Div(
                        [
                            html.Div('Years', className='supply-year-legend-title'),
                            dcc.Checklist(
                                id='supply-year-selector',
                                options=[],
                                value=[],
                                inline=True,
                                className='supply-year-checklist',
                                inputStyle={'display': 'none'},
                                labelStyle={'marginRight': '0'}
                            )
                        ],
                        className='supply-year-legend'
                    )
                ],
                className='supply-rolling-title-row'
            ),
            html.Button(
                'Export to Excel',
                id='export-supply-charts-button',
                n_clicks=0,
                className='supply-rolling-export-button'
            ),
        ], className="inline-section-header supply-rolling-section-header"),
        dcc.Loading(
            id="supply-charts-loading",
            children=[
                html.Div(id='supply-charts-container')
            ],
            type="default",
        )
    ], className="main-section-container supply-rolling-section", style={'marginBottom': '30px'}),

    # Continent Destination Charts Section - 30-Day Rolling Average
    html.Div([
        html.Div([
            html.Div(
                [
                    html.H3(
                        _format_rolling_average_section_title(
                            'LNG Supply by Destination Continent',
                            DEFAULT_SUPPLY_ROLLING_AVG_DAYS
                        ),
                        id='continent-rolling-section-title',
                        className="section-title-inline"
                    ),
                    html.Div(
                        [
                            html.Div('Years', className='continent-year-selector-title'),
                            dcc.Checklist(
                                id='continent-year-selector',
                                options=_get_continent_year_selector_options(),
                                value=_default_continent_chart_selected_years(_get_continent_chart_available_years()),
                                inline=True,
                                className='continent-year-checklist',
                                inputStyle={'display': 'none'},
                                labelStyle={'marginRight': '0'}
                            )
                        ],
                        className='continent-year-selector'
                    ),
                    html.Div(
                        [
                            html.Div('Year style', className='continent-year-style-title'),
                            html.Span(
                                [
                                    html.Span(className='continent-year-style-line continent-year-style-line-current'),
                                    html.Span('Latest')
                                ],
                                className='continent-year-style-item'
                            ),
                            html.Span(
                                [
                                    html.Span(className='continent-year-style-line continent-year-style-line-previous'),
                                    html.Span('Previous')
                                ],
                                className='continent-year-style-item'
                            ),
                            html.Span(
                                [
                                    html.Span(className='continent-year-style-line continent-year-style-line-forecast'),
                                    html.Span('Forecast')
                                ],
                                className='continent-year-style-item'
                            )
                        ],
                        className='continent-year-style-key'
                    )
                ],
                className='continent-rolling-title-row'
            ),
            html.Div(
                [
                    dcc.RadioItems(
                        id='continent-chart-type',
                        options=[
                            {'label': 'Volume', 'value': 'absolute'},
                            {'label': 'Market Share (%)', 'value': 'percentage'}
                        ],
                        value='absolute',
                        inline=True,
                        className='continent-chart-type-selector',
                        inputStyle={'display': 'none'},
                        labelStyle={'marginRight': '0'}
                    ),
                    html.Button(
                        'Export to Excel',
                        id='export-continent-charts-button',
                        n_clicks=0,
                        className='continent-rolling-export-button'
                    )
                ],
                className='continent-rolling-controls'
            ),
        ], className="inline-section-header continent-rolling-section-header"),

        # Dynamic continent charts container - will be populated by callback
        dcc.Loading(
            id="continent-charts-loading",
            children=[
                html.Div(id='continent-charts-container')
            ],
            type="default",
        )
    ], className="main-section-container continent-rolling-section", style={'marginBottom': '30px'}),

    # LNG Supply by Destination Section
    html.Div([
        html.Div([
            html.Div(
                [
                    html.H3('LNG Supply by Destination', className="section-title-inline")
                ],
                className='supply-dest-title-row'
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div('View', className='supply-dest-control-label'),
                            dcc.RadioItems(
                                id='supply-dest-view-type',
                                options=[
                                    {'label': 'Volume', 'value': 'absolute'},
                                    {'label': 'Market Share (%)', 'value': 'percentage'}
                                ],
                                value='absolute',
                                inline=True,
                                className='supply-dest-view-selector',
                                inputStyle={'display': 'none'},
                                labelStyle={'marginRight': '0'}
                            )
                        ],
                        className='supply-dest-control-group supply-dest-view-control'
                    ),
                    html.Div(
                        [
                            html.Div('Aggregation', className='supply-dest-control-label'),
                            dcc.RadioItems(
                                id='aggregation-demand-dropdown',
                                options=[
                                    {'label': 'Installation', 'value': 'Installation'},
                                    {'label': 'Import Country', 'value': 'Country'},
                                    {'label': 'Import Classification Level 1', 'value': 'Classification Level 1'}
                                ],
                                value='Installation',
                                inline=True,
                                className='supply-dest-view-selector supply-dest-aggregation-selector',
                                inputStyle={'display': 'none'},
                                labelStyle={'marginRight': '0'}
                            )
                        ],
                        className='supply-dest-control-group supply-dest-aggregation-control'
                    ),
                    html.Div(
                        [
                            html.Div('Comparison', className='supply-dest-control-label'),
                            dcc.RadioItems(
                                id='supply-dest-summary-comparison-basis',
                                options=SUPPLY_DEST_SUMMARY_COMPARISON_BASIS_OPTIONS,
                                value='levels',
                                inline=True,
                                className='supply-dest-view-selector supply-dest-comparison-selector',
                                inputStyle={'display': 'none'},
                                labelStyle={'marginRight': '0'}
                            )
                        ],
                        className='supply-dest-control-group supply-dest-comparison-control'
                    ),
                    html.Div(
                        [
                            html.Div('Periods', className='supply-dest-control-label'),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span('Years', className='supply-dest-mini-control-label'),
                                            dcc.Dropdown(
                                                id='supply-dest-year-count-dropdown',
                                                options=_build_supply_dest_count_options(
                                                    SUPPLY_DEST_MAX_YEAR_COUNT,
                                                    min_count=0
                                                ),
                                                value=SUPPLY_DEST_DEFAULT_YEAR_COUNT,
                                                clearable=False,
                                                searchable=False,
                                                className='supply-dest-count-dropdown'
                                            )
                                        ],
                                        className='supply-dest-count-selector'
                                    ),
                                    html.Div(
                                        [
                                            html.Span('Qtrs', className='supply-dest-mini-control-label'),
                                            dcc.Dropdown(
                                                id='supply-dest-quarter-count-dropdown',
                                                options=_build_supply_dest_count_options(SUPPLY_DEST_MAX_QUARTER_COUNT),
                                                value=SUPPLY_DEST_DEFAULT_QUARTER_COUNT,
                                                clearable=False,
                                                searchable=False,
                                                className='supply-dest-count-dropdown'
                                            )
                                        ],
                                        className='supply-dest-count-selector'
                                    ),
                                    html.Div(
                                        [
                                            html.Span('Months', className='supply-dest-mini-control-label'),
                                            dcc.Dropdown(
                                                id='supply-dest-month-count-dropdown',
                                                options=_build_supply_dest_count_options(SUPPLY_DEST_MAX_MONTH_COUNT),
                                                value=SUPPLY_DEST_DEFAULT_MONTH_COUNT,
                                                clearable=False,
                                                searchable=False,
                                                className='supply-dest-count-dropdown'
                                            )
                                        ],
                                        className='supply-dest-count-selector'
                                    ),
                                    html.Div(
                                        [
                                            html.Span('Weeks', className='supply-dest-mini-control-label'),
                                            dcc.Dropdown(
                                                id='supply-dest-week-count-dropdown',
                                                options=_build_supply_dest_count_options(SUPPLY_DEST_MAX_WEEK_COUNT),
                                                value=SUPPLY_DEST_DEFAULT_WEEK_COUNT,
                                                clearable=False,
                                                searchable=False,
                                                className='supply-dest-count-dropdown'
                                            )
                                        ],
                                        className='supply-dest-count-selector'
                                    )
                                ],
                                className='supply-dest-period-count-selectors'
                            )
                        ],
                        className='supply-dest-control-group supply-dest-period-count-control'
                    ),
                    html.Button(
                        'Export to Excel',
                        id='export-supply-dest-button',
                        n_clicks=0,
                        className='supply-dest-export-button'
                    )
                ],
                className='supply-dest-controls'
            ),
        ], className="inline-section-header supply-dest-section-header"),

        dcc.Loading(
            id="supply-dest-table-loading",
            children=[
                html.Div(id='supply-dest-table-container', className='supply-dest-table-container')
            ],
            type="default"
        )
    ], className="main-section-container supply-dest-section"),
], className='exporters-page')


# Callbacks

@callback(
    [Output('supply-rolling-section-title', 'children'),
     Output('continent-rolling-section-title', 'children')],
    Input('supply-rolling-window-days-input', 'value'),
    prevent_initial_call=False
)
def update_rolling_average_section_titles(rolling_avg_days):
    """Keep rolling-average section titles synchronized with the sticky selector."""
    return (
        _format_rolling_average_section_title('LNG Supply', rolling_avg_days),
        _format_rolling_average_section_title('LNG Supply by Destination Continent', rolling_avg_days)
    )


def _build_exporters_overview_payload(
    engine_inst,
    schema,
    classification_mode,
    demand_aggregation_mode,
    rolling_avg_days,
):
    chart_dfs = fetch_supply_chart_data(
        engine_inst,
        schema,
        classification_mode,
        rolling_avg_days,
    )
    charts_data = {
        entity_name: entity_df.to_dict('records') if entity_df is not None and not entity_df.empty else []
        for entity_name, entity_df in chart_dfs.items()
    }
    supply_dest_base_df = fetch_supply_destination_base_data(engine_inst, schema)
    supply_dest_summary_payload = build_supply_dest_summary_store_payload(
        engine_inst,
        schema,
        supply_dest_base_df,
        classification_mode,
        demand_aggregation_mode,
    )
    return {
        'charts_cube': _pack_record_mapping(charts_data),
        'continent_entities': list(charts_data.keys()),
        'supply_dest': supply_dest_summary_payload,
    }


def _normalize_exporters_source_watermark(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


def _normalize_exporters_snapshot_metadata(source_pair, prefix):
    """Normalize one snapshot row from the atomic current/baseline lookup."""
    if not isinstance(source_pair, dict):
        return None
    snapshot_id = source_pair.get(f'{prefix}_snapshot_id')
    snapshot_date = source_pair.get(f'{prefix}_snapshot_date_utc')
    snapshot_timestamp = source_pair.get(
        f'{prefix}_snapshot_timestamp_utc'
    )
    if snapshot_id is None or snapshot_date is None or snapshot_timestamp is None:
        return None
    return {
        'snapshot_id': int(snapshot_id),
        'snapshot_date_utc': pd.Timestamp(snapshot_date).date().isoformat(),
        'snapshot_timestamp_utc': _normalize_exporters_source_watermark(
            snapshot_timestamp
        ),
        'facts_retained': bool(
            source_pair.get(f'{prefix}_facts_retained')
        ),
    }


def _previous_weekday_utc(value):
    """Return the prior Monday-Friday date for a UTC snapshot date."""
    candidate = pd.Timestamp(value).date() - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def _business_day_gap(start_date, end_date):
    """Count Monday-Friday business-day steps between two snapshot dates."""
    if start_date is None or end_date is None:
        return None
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if start >= end:
        return 0
    return int(np.busday_count(start.isoformat(), end.isoformat()))


def _build_exporters_source_state(source_pair, refresh_token=None):
    """Build the versioned source contract used by all exporter snapshots."""
    current_snapshot = _normalize_exporters_snapshot_metadata(
        source_pair,
        'current',
    )
    baseline_snapshot = _normalize_exporters_snapshot_metadata(
        source_pair,
        'baseline',
    )
    if current_snapshot is None:
        scalar_watermark = (
            source_pair
            if not isinstance(source_pair, dict)
            else None
        )
        return {
            'format': EXPORTERS_SOURCE_STATE_FORMAT,
            'source_watermark': _normalize_exporters_source_watermark(
                scalar_watermark
            ),
            'as_of_date': datetime.now().date().isoformat(),
            'current_snapshot': None,
            'baseline_snapshot': None,
            'baseline_status': 'unavailable',
            'business_day_gap': None,
            'refresh_token': refresh_token,
        }

    expected_baseline_date = _previous_weekday_utc(
        current_snapshot['snapshot_date_utc']
    )
    baseline_status = 'unavailable'
    business_day_gap = None
    if baseline_snapshot is not None:
        baseline_date = pd.Timestamp(
            baseline_snapshot['snapshot_date_utc']
        ).date()
        baseline_status = (
            'exact'
            if baseline_date == expected_baseline_date
            else 'fallback'
        )
        business_day_gap = _business_day_gap(
            baseline_date,
            current_snapshot['snapshot_date_utc'],
        )

    return {
        'format': EXPORTERS_SOURCE_STATE_FORMAT,
        'source_watermark': current_snapshot['snapshot_timestamp_utc'],
        'as_of_date': current_snapshot['snapshot_date_utc'],
        'current_snapshot': current_snapshot,
        'baseline_snapshot': baseline_snapshot,
        'baseline_status': baseline_status,
        'business_day_gap': business_day_gap,
        'refresh_token': refresh_token,
    }


def _validate_exporters_source_state(source_state):
    if not (
        isinstance(source_state, dict)
        and source_state.get('format') == EXPORTERS_SOURCE_STATE_FORMAT
        and source_state.get('as_of_date')
    ):
        raise _SnapshotUnavailable(
            EXPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        )
    return dict(source_state)


@callback(
    Output('exporters-source-state-store', 'data'),
    [Input('initial-load-trigger', 'n_intervals'),
     Input('global-refresh-button', 'n_clicks')],
    prevent_initial_call=False
)
def refresh_exporters_source_state(_n_intervals, _global_refresh_clicks):
    """Capture one coherent current/PBD snapshot pair on refresh."""
    refresh_token = (
        uuid.uuid4().hex
        if _was_global_refresh_triggered()
        else None
    )
    try:
        source_pair = _fetch_exporters_source_watermark()
    except Exception:
        source_pair = None
        refresh_token = uuid.uuid4().hex
    return _build_exporters_source_state(
        source_pair,
        refresh_token,
    )


def _exporters_destination_base_source_key(source_state):
    return _build_source_key(
        EXPORTERS_DESTINATION_BASE_NAMESPACE,
        _validate_exporters_source_state(source_state),
    )


def _exporters_destination_pbd_base_source_key(source_state):
    source_state = _validate_exporters_source_state(source_state)
    return _build_source_key(
        EXPORTERS_DESTINATION_PBD_BASE_NAMESPACE,
        source_state.get('baseline_snapshot'),
    )


def _exporters_supply_charts_source_key(
    source_state,
    classification_mode,
    rolling_avg_days,
    entity_names,
):
    return _build_source_key(
        EXPORTERS_SUPPLY_CHARTS_NAMESPACE,
        _validate_exporters_source_state(source_state),
        classification_mode,
        rolling_avg_days,
        list(entity_names),
    )


def _exporters_continent_data_source_key(
    source_state,
    classification_mode,
    rolling_avg_days,
    entity_names,
    selected_years,
):
    selected_years, _query_start_date, _display_start_date = (
        _get_continent_chart_selected_window(selected_years)
    )
    return _build_source_key(
        EXPORTERS_CONTINENT_DATA_NAMESPACE,
        _validate_exporters_source_state(source_state),
        classification_mode,
        rolling_avg_days,
        list(entity_names),
        selected_years,
    )


def _exporters_continent_export_source_key(
    source_state,
    classification_mode,
    rolling_avg_days,
    entity_names,
):
    return _build_source_key(
        EXPORTERS_CONTINENT_EXPORT_NAMESPACE,
        _validate_exporters_source_state(source_state),
        classification_mode,
        rolling_avg_days,
        list(entity_names),
        CONTINENT_CHART_QUERY_START_DATE,
        CONTINENT_CHART_DISPLAY_START_DATE,
    )


def _exporters_destination_summary_source_key(
    destination_base_reference,
    classification_mode,
    demand_aggregation_mode,
    destination_pbd_reference=None,
    source_state=None,
):
    source_state = source_state if isinstance(source_state, dict) else {}
    current_dependency = {
        'namespace': destination_base_reference.get('namespace'),
        'source_key': destination_base_reference.get('source_key'),
        'revision': destination_base_reference.get('revision'),
    }
    baseline_dependency = None
    if isinstance(destination_pbd_reference, dict):
        baseline_dependency = {
            'namespace': destination_pbd_reference.get('namespace'),
            'source_key': destination_pbd_reference.get('source_key'),
            'revision': destination_pbd_reference.get('revision'),
        }
    return _build_source_key(
        EXPORTERS_DESTINATION_SUMMARY_NAMESPACE,
        current_dependency,
        baseline_dependency,
        {
            'format': source_state.get('format'),
            'current_snapshot': source_state.get('current_snapshot'),
            'baseline_snapshot': source_state.get('baseline_snapshot'),
            'baseline_status': source_state.get('baseline_status'),
            'business_day_gap': source_state.get('business_day_gap'),
        },
        classification_mode,
        demand_aggregation_mode,
    )


def _build_exporters_supply_charts_snapshot_payload(
    engine_inst,
    schema,
    classification_mode,
    rolling_avg_days,
    entity_names,
):
    chart_dfs = _fetch_supply_chart_data_for_entities(
        engine_inst,
        schema,
        classification_mode,
        rolling_avg_days,
        entity_names,
    )
    charts_data = {
        entity_name: (
            chart_dfs[entity_name].to_dict('records')
            if (
                entity_name in chart_dfs
                and chart_dfs[entity_name] is not None
                and not chart_dfs[entity_name].empty
            )
            else []
        )
        for entity_name in entity_names
    }
    return _prepare_exporters_supply_charts_snapshot_payload(
        charts_data
    )


def _build_exporters_continent_snapshot_payload(
    engine_inst,
    schema,
    source_state,
    classification_mode,
    rolling_avg_days,
    entity_names,
    selected_years,
):
    selected_years, _query_start_date, _display_start_date = (
        _get_continent_chart_selected_window(selected_years)
    )
    source_state = _validate_exporters_source_state(source_state)
    classification_mode = classification_mode or 'Country'
    rolling_avg_days = normalize_supply_rolling_avg_days(
        rolling_avg_days
    )
    continent_df = fetch_continent_chart_data_batch(
        engine_inst,
        schema,
        entity_names,
        classification_mode,
        selected_years=selected_years,
        rolling_avg_days=rolling_avg_days,
    )
    return {
        'entities': list(entity_names),
        'data': continent_df,
        'source_state': source_state,
        'classification_mode': classification_mode,
        'rolling_avg_days': rolling_avg_days,
        'selected_years': selected_years,
    }


def _load_exporters_destination_base_snapshot(
    engine_inst,
    schema,
    source_state,
):
    current_snapshot = source_state.get('current_snapshot') or {}
    return _get_or_build_snapshot(
        engine_inst,
        namespace=EXPORTERS_DESTINATION_BASE_NAMESPACE,
        source_key=_exporters_destination_base_source_key(
            source_state
        ),
        builder=lambda: fetch_supply_destination_base_data(
            engine_inst,
            schema,
            current_snapshot.get('snapshot_timestamp_utc'),
            current_snapshot.get('snapshot_date_utc'),
        ),
        manifest={
            'source_state': source_state,
        },
    )


def _load_exporters_destination_pbd_base_snapshot(
    engine_inst,
    schema,
    source_state,
):
    baseline_snapshot = source_state.get('baseline_snapshot') or {}
    if (
        source_state.get('baseline_status') not in {'exact', 'fallback'}
        or not baseline_snapshot.get('snapshot_timestamp_utc')
        or not baseline_snapshot.get('snapshot_date_utc')
    ):
        return None, pd.DataFrame()
    return _get_or_build_snapshot(
        engine_inst,
        namespace=EXPORTERS_DESTINATION_PBD_BASE_NAMESPACE,
        source_key=_exporters_destination_pbd_base_source_key(
            source_state
        ),
        builder=lambda: fetch_supply_destination_pbd_base_data(
            engine_inst,
            schema,
            baseline_snapshot['snapshot_timestamp_utc'],
            baseline_snapshot['snapshot_date_utc'],
        ),
        manifest={
            'baseline_snapshot': baseline_snapshot,
            'window_days': 30,
        },
    )


def _load_exporters_supply_charts_snapshot(
    engine_inst,
    schema,
    source_state,
    classification_mode,
    rolling_avg_days,
    entity_names,
):
    return _get_or_build_snapshot(
        engine_inst,
        namespace=EXPORTERS_SUPPLY_CHARTS_NAMESPACE,
        source_key=_exporters_supply_charts_source_key(
            source_state,
            classification_mode,
            rolling_avg_days,
            entity_names,
        ),
        builder=lambda: _build_exporters_supply_charts_snapshot_payload(
            engine_inst,
            schema,
            classification_mode,
            rolling_avg_days,
            entity_names,
        ),
        manifest={
            'source_state': source_state,
            'classification_mode': classification_mode,
            'rolling_avg_days': rolling_avg_days,
            'entity_names': list(entity_names),
        },
    )


def _load_exporters_continent_snapshot(
    engine_inst,
    schema,
    source_state,
    classification_mode,
    rolling_avg_days,
    entity_names,
    selected_years,
):
    selected_years, _query_start_date, _display_start_date = (
        _get_continent_chart_selected_window(selected_years)
    )
    return _get_or_build_snapshot(
        engine_inst,
        namespace=EXPORTERS_CONTINENT_DATA_NAMESPACE,
        source_key=_exporters_continent_data_source_key(
            source_state,
            classification_mode,
            rolling_avg_days,
            entity_names,
            selected_years,
        ),
        builder=lambda: _build_exporters_continent_snapshot_payload(
            engine_inst,
            schema,
            source_state,
            classification_mode,
            rolling_avg_days,
            entity_names,
            selected_years,
        ),
        manifest={
            'source_state': source_state,
            'classification_mode': classification_mode,
            'rolling_avg_days': rolling_avg_days,
            'entity_names': list(entity_names),
            'selected_years': selected_years,
        },
    )


def _load_exporters_continent_export_snapshot(
    engine_inst,
    schema,
    source_state,
    classification_mode,
    rolling_avg_days,
    entity_names,
):
    selected_years = _get_continent_chart_available_years()
    return _get_or_build_snapshot(
        engine_inst,
        namespace=EXPORTERS_CONTINENT_EXPORT_NAMESPACE,
        source_key=_exporters_continent_export_source_key(
            source_state,
            classification_mode,
            rolling_avg_days,
            entity_names,
        ),
        builder=lambda: _build_exporters_continent_snapshot_payload(
            engine_inst,
            schema,
            source_state,
            classification_mode,
            rolling_avg_days,
            entity_names,
            selected_years,
        ),
        manifest={
            'source_state': source_state,
            'classification_mode': classification_mode,
            'rolling_avg_days': rolling_avg_days,
            'entity_names': list(entity_names),
            'query_start_date': CONTINENT_CHART_QUERY_START_DATE,
            'display_start_date': CONTINENT_CHART_DISPLAY_START_DATE,
        },
    )


def _resolve_or_load_exporters_continent_payload(
    continent_data,
    classification_mode,
    rolling_avg_days,
    selected_years,
):
    """Resolve the exact selected-year continent snapshot for an interaction."""
    payload = _resolve_exporters_continent_payload(continent_data)
    classification_mode = classification_mode or 'Country'
    rolling_avg_days = normalize_supply_rolling_avg_days(
        rolling_avg_days
    )
    selected_years, _query_start_date, _display_start_date = (
        _get_continent_chart_selected_window(selected_years)
    )
    payload_selected_years, _payload_query_start, _payload_display_start = (
        _get_continent_chart_selected_window(
            payload['selected_years']
        )
    )

    if (
        payload['classification_mode'] != classification_mode
        or payload['rolling_avg_days'] != rolling_avg_days
    ):
        # Classification/rolling changes rebuild the roster/default snapshot.
        # Preserve the current chart until that coordinated refresh publishes.
        raise PreventUpdate

    if payload_selected_years == selected_years:
        return payload

    engine_inst, schema = setup_database_connection()
    reference, _built_payload = _load_exporters_continent_snapshot(
        engine_inst,
        schema,
        payload['source_state'],
        classification_mode,
        rolling_avg_days,
        payload['entities'],
        selected_years,
    )
    if not _snapshot_is_resolvable(reference):
        raise _SnapshotUnavailable(
            EXPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        )
    return _resolve_exporters_continent_payload(reference)


def _resolve_or_load_exporters_continent_export_payload(
    continent_data,
    classification_mode,
    rolling_avg_days,
):
    """Resolve the lazy full-history continent snapshot used only by Excel."""
    payload = _resolve_exporters_continent_payload(continent_data)
    classification_mode = classification_mode or 'Country'
    rolling_avg_days = normalize_supply_rolling_avg_days(
        rolling_avg_days
    )
    if (
        payload['classification_mode'] != classification_mode
        or payload['rolling_avg_days'] != rolling_avg_days
    ):
        raise PreventUpdate

    engine_inst, schema = setup_database_connection()
    reference, _built_payload = (
        _load_exporters_continent_export_snapshot(
            engine_inst,
            schema,
            payload['source_state'],
            classification_mode,
            rolling_avg_days,
            payload['entities'],
        )
    )
    if not _snapshot_is_resolvable(reference):
        raise _SnapshotUnavailable(
            EXPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        )
    return _resolve_exporters_continent_payload(reference)


def _load_exporters_destination_summary_snapshot(
    engine_inst,
    schema,
    destination_base_reference,
    destination_base_df,
    destination_pbd_reference,
    destination_pbd_df,
    source_state,
    classification_mode,
    demand_aggregation_mode,
):
    return _get_or_build_snapshot(
        engine_inst,
        namespace=EXPORTERS_DESTINATION_SUMMARY_NAMESPACE,
        source_key=_exporters_destination_summary_source_key(
            destination_base_reference,
            classification_mode,
            demand_aggregation_mode,
            destination_pbd_reference,
            source_state,
        ),
        builder=lambda: (
            _prepare_exporters_destination_summary_snapshot_payload(
                build_supply_dest_summary_store_payload(
                    engine_inst,
                    schema,
                    destination_base_df,
                    classification_mode,
                    demand_aggregation_mode,
                    destination_pbd_df,
                    source_state,
                )
            )
        ),
        manifest={
            'destination_base_reference': {
                'namespace': destination_base_reference.get('namespace'),
                'source_key': destination_base_reference.get('source_key'),
                'revision': destination_base_reference.get('revision'),
            },
            'destination_pbd_reference': (
                {
                    'namespace': destination_pbd_reference.get('namespace'),
                    'source_key': destination_pbd_reference.get('source_key'),
                    'revision': destination_pbd_reference.get('revision'),
                }
                if isinstance(destination_pbd_reference, dict)
                else None
            ),
            'source_state': source_state,
            'classification_mode': classification_mode,
            'demand_aggregation_mode': demand_aggregation_mode,
        },
    )


def _load_exporters_independent_snapshots(
    engine_inst,
    schema,
    source_state,
    classification_mode,
    rolling_avg_days,
    entity_names,
    continent_selected_years=None,
):
    loaders = {
        'destination_base': lambda: (
            _load_exporters_destination_base_snapshot(
                engine_inst,
                schema,
                source_state,
            )
        ),
        'supply_charts': lambda: (
            _load_exporters_supply_charts_snapshot(
                engine_inst,
                schema,
                source_state,
                classification_mode,
                rolling_avg_days,
                entity_names,
            )
        ),
        'continent_data': lambda: (
            _load_exporters_continent_snapshot(
                engine_inst,
                schema,
                source_state,
                classification_mode,
                rolling_avg_days,
                entity_names,
                continent_selected_years,
            )
        ),
    }
    if (
        source_state.get('baseline_status') in {'exact', 'fallback'}
        and isinstance(source_state.get('baseline_snapshot'), dict)
    ):
        loaders['destination_pbd'] = lambda: (
            _load_exporters_destination_pbd_base_snapshot(
                engine_inst,
                schema,
                source_state,
            )
        )
    with ThreadPoolExecutor(
        max_workers=3,
        thread_name_prefix='exporters-load',
    ) as executor:
        futures = {
            name: executor.submit(loader)
            for name, loader in loaders.items()
        }
        return {
            name: futures[name].result()
            for name in loaders
        }


@callback(
    [Output('supply-charts-data', 'data'),
     Output('continent-charts-data', 'data'),
     Output('supply-dest-data-store', 'data')],
    [Input('exporters-source-state-store', 'data'),
     Input('country-classification-dropdown', 'value'),
     Input('aggregation-demand-dropdown', 'value'),
     Input('supply-rolling-window-days-input', 'value')],
    prevent_initial_call=False
)
@log_callback_timing("exporters.source_load")
def refresh_all_data(
    source_state,
    classification_mode,
    demand_aggregation_mode,
    rolling_avg_days,
):
    """Load all data from database"""
    if not source_state:
        raise PreventUpdate
    try:
        engine_inst, schema = setup_database_connection()
        source_state = _validate_exporters_source_state(
            source_state
        )
        if classification_mode is None:
            classification_mode = 'Country'
        demand_aggregation_mode = normalize_demand_aggregation_mode(demand_aggregation_mode)
        rolling_avg_days = normalize_supply_rolling_avg_days(rolling_avg_days)
        entity_names = _get_exporter_entity_names(
            engine_inst,
            schema,
            classification_mode,
        )
        loaded = _load_exporters_independent_snapshots(
            engine_inst,
            schema,
            source_state,
            classification_mode,
            rolling_avg_days,
            entity_names,
        )
        destination_base_reference, destination_base_df = (
            loaded['destination_base']
        )
        destination_pbd_reference, destination_pbd_df = loaded.get(
            'destination_pbd',
            (None, pd.DataFrame()),
        )
        charts_reference, charts_payload = loaded['supply_charts']
        continent_reference, _continent_payload = (
            loaded['continent_data']
        )
        summary_reference, _summary_payload = (
            _load_exporters_destination_summary_snapshot(
                engine_inst,
                schema,
                destination_base_reference,
                destination_base_df,
                destination_pbd_reference,
                destination_pbd_df,
                source_state,
                classification_mode,
                demand_aggregation_mode,
            )
        )
        references = (
            destination_base_reference,
            destination_pbd_reference,
            charts_reference,
            continent_reference,
            summary_reference,
        )
        if not all(
            _snapshot_is_resolvable(reference)
            for reference in references
            if reference is not None
        ):
            raise _SnapshotUnavailable(
                EXPORTERS_SNAPSHOT_RECOVERY_MESSAGE
            )
        return (
            _with_snapshot_slot(
                charts_reference,
                'charts_cube',
            ),
            continent_reference,
            summary_reference,
        )

    except _SnapshotUnavailable:
        raise
    except Exception:
        return {}, [], _empty_supply_dest_summary_store_payload()


@callback(
    [Output('supply-dest-expanded-classifications', 'data', allow_duplicate=True),
     Output('supply-dest-expanded-countries', 'data', allow_duplicate=True),
     Output('supply-dest-expanded-supply-countries', 'data', allow_duplicate=True)],
    [Input('country-classification-dropdown', 'value'),
     Input('aggregation-demand-dropdown', 'value'),
     Input('supply-dest-country-grouping-dropdown', 'value')],
    prevent_initial_call=True
)
def reset_supply_dest_expansion_state(_classification_mode, _demand_aggregation_mode, _country_grouping_mode):
    """Reset expanded rows whenever the supply/import aggregation controls change."""
    return [], [], []


def _supply_chart_year_token(year):
    """Normalize chart years for checklist values and dataframe filtering."""
    try:
        if pd.isna(year):
            return None
    except TypeError:
        pass

    try:
        return str(int(float(year)))
    except (TypeError, ValueError):
        token = str(year).strip()
        return token or None


def _supply_chart_year_sort_key(year):
    try:
        return (0, int(year))
    except (TypeError, ValueError):
        return (1, str(year))


def _get_supply_chart_years_from_records(records):
    years = set()
    for record in records or []:
        if not isinstance(record, dict):
            continue

        rolling_avg = record.get('rolling_avg')
        try:
            if rolling_avg is None or pd.isna(rolling_avg) or float(rolling_avg) <= 0:
                continue
        except (TypeError, ValueError):
            continue

        token = _supply_chart_year_token(record.get('year'))
        if token:
            years.add(token)
    return sorted(years, key=_supply_chart_year_sort_key)


def _get_supply_chart_available_years(charts_data):
    years = set()
    if isinstance(charts_data, dict):
        for records in charts_data.values():
            years.update(_get_supply_chart_years_from_records(records))
    return sorted(years, key=_supply_chart_year_sort_key)


def _default_supply_chart_selected_years(available_years):
    latest_years = available_years[-SUPPLY_CHART_DEFAULT_SELECTED_YEAR_COUNT:]
    selected_years = [
        year for year in latest_years
        if year not in SUPPLY_CHART_DEFAULT_DESELECTED_YEARS
    ]
    return selected_years or latest_years


def _normalize_supply_chart_selected_years(selected_years, available_years, use_default=True):
    available_set = set(available_years)
    normalized = [
        token for token in (_supply_chart_year_token(year) for year in (selected_years or []))
        if token in available_set
    ]
    if normalized or not use_default:
        return sorted(set(normalized), key=_supply_chart_year_sort_key)
    return _default_supply_chart_selected_years(available_years)


def _normalise_supply_chart_plot_dates(date_series):
    month_day_text = pd.to_datetime(date_series, errors='coerce').dt.strftime('%m-%d')
    return pd.to_datetime(
        month_day_text.map(
            lambda value: f'{SUPPLY_CHART_ANCHOR_YEAR}-{value}' if pd.notna(value) else None
        ),
        errors='coerce'
    )


def _get_supply_chart_color_map(years):
    years = sorted(years or [], key=_supply_chart_year_sort_key)
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


def _get_supply_chart_range_years(focus_year, available_years):
    """Return up to five available years before the focus year for the range band."""
    try:
        focus_year_number = int(focus_year)
    except (TypeError, ValueError):
        return []

    previous_years = []
    for year in sorted(available_years, key=_supply_chart_year_sort_key):
        try:
            if int(year) < focus_year_number:
                previous_years.append(year)
        except (TypeError, ValueError):
            continue
    return previous_years[-SUPPLY_CHART_RANGE_LOOKBACK_YEARS:]


def _add_supply_chart_range_band(fig, df, focus_year, available_years, vol_label):
    range_years = _get_supply_chart_range_years(focus_year, available_years)
    if not range_years:
        return

    range_df = df[
        df['_year_token'].isin(range_years)
        & df['rolling_avg'].notna()
        & df['plot_date'].notna()
    ].copy()
    if range_df.empty:
        return

    if 'is_forecast' in range_df.columns:
        range_df = range_df[~range_df['is_forecast'].astype(bool)].copy()
        if range_df.empty:
            return

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

    years_label = f"{range_years[0]}-{range_years[-1]}" if len(range_years) > 1 else range_years[0]
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


def _prepare_supply_chart_dataframe(
    data,
    volume_metric,
    rolling_avg_days=DEFAULT_SUPPLY_ROLLING_AVG_DAYS
):
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty or not {'date', 'year', 'rolling_avg'}.issubset(df.columns):
        return pd.DataFrame()

    rolling_avg_days = normalize_supply_rolling_avg_days(rolling_avg_days)
    df = _convert_volume_metric_dataframe(
        df,
        volume_metric,
        columns=['rolling_avg'],
        period_days=rolling_avg_days
    )
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    if 'month_day' not in df.columns:
        df['month_day'] = df['date'].dt.strftime('%b %d')

    df['_year_token'] = df['year'].apply(_supply_chart_year_token)
    df['plot_date'] = _normalise_supply_chart_plot_dates(df['date'])
    df = df[
        (df['date'] >= SUPPLY_CHART_DISPLAY_START_DATE)
        & df['_year_token'].notna()
        & df['plot_date'].notna()
    ].copy()
    return df


def _get_supply_chart_previous_year_token(focus_year, available_years, active_years):
    try:
        previous_year = str(int(focus_year) - 1)
        if previous_year in set(available_years):
            return previous_year
    except (TypeError, ValueError):
        pass

    if len(active_years) > 1:
        return active_years[-2]
    return None


def get_supply_chart_header_metrics(
    data,
    volume_metric='mcm_d',
    selected_years=None,
    rolling_avg_days=DEFAULT_SUPPLY_ROLLING_AVG_DAYS
):
    """Return latest selected-year value and same-date previous-year delta for a chart card."""
    df = _prepare_supply_chart_dataframe(data, volume_metric, rolling_avg_days)
    if df.empty:
        return None

    available_years = sorted(
        [year for year in df['_year_token'].dropna().unique()],
        key=_supply_chart_year_sort_key
    )
    active_years = _normalize_supply_chart_selected_years(
        selected_years,
        available_years,
        use_default=selected_years is None
    )
    if not active_years:
        return None

    focus_year = active_years[-1]
    focus_data = df[df['_year_token'] == focus_year].copy()
    if focus_data.empty:
        return None

    if 'is_forecast' in focus_data.columns:
        actual_focus_data = focus_data[~focus_data['is_forecast'].astype(bool)]
        if not actual_focus_data.empty:
            focus_data = actual_focus_data

    latest_point = focus_data.dropna(subset=['rolling_avg']).sort_values('plot_date').tail(1)
    if latest_point.empty:
        return None

    point = latest_point.iloc[0]
    latest_value = point['rolling_avg']
    if pd.isna(latest_value):
        return None

    previous_year = _get_supply_chart_previous_year_token(focus_year, available_years, active_years)
    previous_value = None
    delta_value = None
    delta_pct = None
    mom_value = None
    mom_delta_value = None
    mom_delta_pct = None

    if previous_year:
        previous_data = df[df['_year_token'] == previous_year].dropna(subset=['rolling_avg']).copy()
        if not previous_data.empty:
            previous_data = previous_data.sort_values('plot_date')
            previous_candidates = previous_data[previous_data['plot_date'] <= point['plot_date']]
            if previous_candidates.empty:
                previous_candidates = previous_data
            previous_point = previous_candidates.tail(1)
            if not previous_point.empty:
                previous_value = previous_point.iloc[0]['rolling_avg']
                if pd.notna(previous_value):
                    delta_value = latest_value - previous_value
                    if previous_value != 0:
                        delta_pct = delta_value / abs(previous_value) * 100

    month_ago_date = point['plot_date'] - pd.DateOffset(months=1)
    mom_candidates = focus_data[
        (focus_data['plot_date'] <= month_ago_date)
        & focus_data['rolling_avg'].notna()
    ].copy()
    if not mom_candidates.empty:
        mom_point = mom_candidates.sort_values('plot_date').tail(1)
        if not mom_point.empty:
            mom_value = mom_point.iloc[0]['rolling_avg']
            if pd.notna(mom_value):
                mom_delta_value = latest_value - mom_value
                if mom_value != 0:
                    mom_delta_pct = mom_delta_value / abs(mom_value) * 100

    return {
        'focus_year': focus_year,
        'latest_value': latest_value,
        'latest_label': point.get('month_day', ''),
        'previous_year': previous_year,
        'previous_value': previous_value,
        'delta_value': delta_value,
        'delta_pct': delta_pct,
        'mom_value': mom_value,
        'mom_delta_value': mom_delta_value,
        'mom_delta_pct': mom_delta_pct
    }


def _empty_supply_chart_figure(message, height=328):
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
        height=height,
        margin=dict(l=36, r=24, t=12, b=34),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff'
    )
    return fig


@callback(
    [Output('supply-year-selector', 'options'),
     Output('supply-year-selector', 'value')],
    Input('supply-charts-data', 'data'),
    State('supply-year-selector', 'value'),
    prevent_initial_call=False
)
def update_supply_year_selector_options(charts_data, selected_years):
    """Populate the inline year legend from the loaded rolling-average data."""
    try:
        charts_data = _resolve_exporters_store(charts_data)
    except _SnapshotUnavailable:
        return _exporters_snapshot_recovery_selector_result()
    available_years = _get_supply_chart_available_years(charts_data)
    if not available_years:
        return [], []

    color_by_year = _get_supply_chart_color_map(available_years)
    options = [
        {
            'label': html.Span(
                [
                    html.Span(
                        className='supply-year-chip-swatch',
                        style={'backgroundColor': color_by_year[year]}
                    ),
                    html.Span(year, className='supply-year-chip-text')
                ],
                className='supply-year-chip-label'
            ),
            'value': year
        }
        for year in available_years
    ]
    selected = _normalize_supply_chart_selected_years(selected_years, available_years)
    return options, selected


def create_supply_chart(
    data,
    show_legend=True,
    volume_metric='mcm_d',
    selected_years=None,
    rolling_avg_days=DEFAULT_SUPPLY_ROLLING_AVG_DAYS
):
    """Create seasonal comparison chart for LNG supply with professional styling."""
    vol_info = _get_volume_metric_info(volume_metric)
    vol_label = vol_info['label']

    df = _prepare_supply_chart_dataframe(data, volume_metric, rolling_avg_days)
    if df.empty:
        return _empty_supply_chart_figure("No data available.")

    available_years = sorted(
        [year for year in df['_year_token'].dropna().unique()],
        key=_supply_chart_year_sort_key
    )
    active_years = _normalize_supply_chart_selected_years(
        selected_years,
        available_years,
        use_default=selected_years is None
    )
    if not active_years:
        return _empty_supply_chart_figure("Select a year below.")

    active_df = df[df['_year_token'].isin(active_years)].copy()
    if active_df.empty:
        return _empty_supply_chart_figure("No data for the selected years.")

    fig = go.Figure()

    years = sorted(active_years, key=_supply_chart_year_sort_key)
    focus_year = years[-1]
    color_by_year = _get_supply_chart_color_map(available_years)
    _add_supply_chart_range_band(fig, df, focus_year, available_years, vol_label)

    for year in years:
        year_data = active_df[active_df['_year_token'] == year].copy()
        if year_data.empty:
            continue

        year_data = year_data.dropna(subset=['plot_date']).sort_values('plot_date')
        year_label = str(year)
        is_focus_year = year == focus_year
        line_color = color_by_year[year]
        line_width = 2.2 if is_focus_year else 1.15
        line_opacity = 0.95 if is_focus_year else 0.52

        if 'is_forecast' in year_data.columns:
            historical_data = year_data[~year_data['is_forecast'].astype(bool)]
            forecast_data = year_data[year_data['is_forecast'].astype(bool)]

            if not historical_data.empty:
                fig.add_trace(go.Scatter(
                    x=historical_data['plot_date'],
                    y=historical_data['rolling_avg'],
                    mode='lines',
                    name=year_label,
                    line=dict(
                        color=line_color,
                        width=line_width,
                        dash='solid'
                    ),
                    opacity=line_opacity,
                    hovertemplate=(
                        f'<b>{year_label}</b> | '
                        '%{text} | '
                        f'%{{y:,.0f}} {vol_label}<extra></extra>'
                    ),
                    text=historical_data['month_day'],
                    showlegend=show_legend
                ))

            if not forecast_data.empty:
                if not historical_data.empty:
                    connect_data = pd.concat([historical_data.tail(1), forecast_data])
                else:
                    connect_data = forecast_data

                fig.add_trace(go.Scatter(
                    x=connect_data['plot_date'],
                    y=connect_data['rolling_avg'],
                    mode='lines',
                    name=f'{year_label} forecast',
                    line=dict(
                        color=line_color,
                        width=line_width,
                        dash=SUPPLY_CHART_FORECAST_DASH
                    ),
                    opacity=0.76 if is_focus_year else 0.36,
                    hovertemplate=(
                        f'<b>{year_label} forecast</b> | '
                        '%{text} | '
                        f'%{{y:,.0f}} {vol_label}<extra></extra>'
                    ),
                    text=connect_data['month_day'],
                    showlegend=False
                ))
        else:
            fig.add_trace(go.Scatter(
                x=year_data['plot_date'],
                y=year_data['rolling_avg'],
                mode='lines',
                name=year_label,
                line=dict(
                    color=line_color,
                    width=line_width,
                    dash='solid'
                ),
                opacity=line_opacity,
                hovertemplate=(
                    f'<b>{year_label}</b> | '
                    '%{text} | '
                    f'%{{y:,.0f}} {vol_label}<extra></extra>'
                ),
                text=year_data['month_day'],
                showlegend=show_legend
            ))

        if is_focus_year:
            latest_actual_data = year_data
            if 'is_forecast' in latest_actual_data.columns:
                non_forecast = latest_actual_data[~latest_actual_data['is_forecast'].astype(bool)]
                if not non_forecast.empty:
                    latest_actual_data = non_forecast

            latest_point = latest_actual_data.dropna(subset=['rolling_avg']).tail(1)
            if not latest_point.empty:
                point = latest_point.iloc[0]
                fig.add_trace(go.Scatter(
                    x=[point['plot_date']],
                    y=[point['rolling_avg']],
                    mode='markers',
                    marker=dict(
                        color=line_color,
                        size=5.5,
                        line=dict(color='#ffffff', width=1.5)
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))

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
                pd.Timestamp(year=SUPPLY_CHART_ANCHOR_YEAR, month=1, day=1),
                pd.Timestamp(year=SUPPLY_CHART_ANCHOR_YEAR, month=12, day=31)
            ],
            showspikes=True,
            spikemode='across',
            spikecolor='rgba(15, 23, 42, 0.18)',
            spikethickness=1
        ),
        yaxis=dict(
            title=dict(text=vol_label, font=dict(size=11, color='#475569')),
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
        showlegend=show_legend,
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
        ) if show_legend else None,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        margin=dict(l=44, r=18, t=12, b=36),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.96)',
            bordercolor='rgba(148, 163, 184, 0.7)',
            font=dict(size=11, color='#0f172a'),
            align='left'
        ),
        height=328,
        transition=dict(duration=300, easing='cubic-in-out')
    )

    return fig


# Callback to update supply-destination table
@callback(
    Output('supply-dest-table-container', 'children'),
    [Input('supply-dest-data-store', 'data'),
     Input('supply-dest-expanded-classifications', 'data'),
     Input('supply-dest-expanded-countries', 'data'),
     Input('supply-dest-expanded-supply-countries', 'data'),
     Input('country-classification-dropdown', 'value'),
     Input('supply-dest-view-type', 'value'),
     Input('aggregation-demand-dropdown', 'value'),
     Input('supply-dest-summary-comparison-basis', 'value'),
     Input('supply-dest-country-grouping-dropdown', 'value'),
     Input('exporters-volume-metric-dropdown', 'value'),
     Input('supply-dest-year-count-dropdown', 'value'),
     Input('supply-dest-quarter-count-dropdown', 'value'),
     Input('supply-dest-month-count-dropdown', 'value'),
     Input('supply-dest-week-count-dropdown', 'value')],
    prevent_initial_call=False
)
@log_callback_timing("exporters.destination_table_render")
def update_supply_dest_table(supply_dest_data, expanded_classifications, expanded_countries,
                             expanded_supply_countries, classification_mode, view_type,
                             demand_aggregation_mode, summary_comparison_basis,
                             country_grouping_mode, volume_metric, year_count,
                             quarter_count, month_count, week_count):
    """Update the supply-destination table with expandable rows"""
    try:
        supply_dest_data = _resolve_exporters_store(supply_dest_data)
    except _SnapshotUnavailable:
        return _exporters_snapshot_recovery_notice()
    vol_label = _get_volume_metric_info(volume_metric)['label']
    show_demand_aggregation = use_demand_classification_mode(classification_mode, demand_aggregation_mode)
    show_demand_country = use_demand_country_mode(demand_aggregation_mode)
    show_import_classification = use_import_classification_mode(demand_aggregation_mode)
    country_grouping_mode = _normalize_supply_dest_country_grouping(country_grouping_mode)
    summary_comparison_basis = _normalize_supply_dest_comparison_basis(summary_comparison_basis)
    expanded_classifications = expanded_classifications or []
    expanded_countries = expanded_countries or []
    expanded_supply_countries = expanded_supply_countries or []

    if not supply_dest_data:
        return html.Div(
            "No data available. Please refresh to load data.",
            style={'textAlign': 'center', 'padding': '20px'},
        )

    try:
        snapshot_comparison_metadata = (
            _get_supply_dest_summary_comparison_metadata(
                supply_dest_data
            )
        )
        resolved_supply_dest_data = _resolve_supply_dest_summary_payload(
            supply_dest_data,
            country_grouping_mode
        )
        df = pd.DataFrame(resolved_supply_dest_data)

        if df.empty:
            return html.Div(
                "No data available. Please refresh to load data.",
                style={'textAlign': 'center', 'padding': '20px'},
            )

        df, summary_comparison_metadata = _filter_supply_dest_summary_period_columns(
            df,
            quarter_count,
            month_count,
            week_count,
            year_count,
            summary_comparison_basis,
            return_metadata=True
        )

        # Prepare data for display with expandable rows
        display_df, columns = prepare_supply_dest_table_for_display(
            df, classification_mode,
            expanded_classifications, expanded_countries, expanded_supply_countries, view_type,
            demand_aggregation_mode
        )

        if display_df.empty:
            return html.Div(
                "No data available",
                style={'textAlign': 'center', 'padding': '20px'},
            )
        summary_comparison_delta_cols = []

        # Convert to percentages if requested
        if view_type == 'percentage' and classification_mode == 'Classification Level 1' and show_demand_aggregation:
            # Identify numeric columns (exclude text columns and delta columns)
            text_cols = ['Aggregation Supply', 'Aggregation Demand', 'Country Demand', 'Supply Country']
            delta_cols = list(SUPPLY_DEST_DELTA_RAW_FIELDS)
            # Include 30D_Y1 in numeric columns for subtotal calculation
            numeric_cols = [col for col in display_df.columns if col not in text_cols and col not in delta_cols]
            # Also include 30D_Y1 if present for subtotal storage
            numeric_cols_with_y1 = numeric_cols + (['30D_Y1'] if '30D_Y1' in display_df.columns else [])

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

            # Drop 30D_Y1 unless the active comparison still needs it.
            if '30D_Y1' in display_df.columns and '30D_Y1' not in summary_comparison_metadata.get('reference_cols', []):
                display_df = display_df.drop('30D_Y1', axis=1)

        elif view_type == 'percentage' and classification_mode == 'Classification Level 1' and show_demand_country:
            text_cols = ['Aggregation Supply', 'Demand Country', 'Supply Country']
            delta_cols = list(SUPPLY_DEST_DELTA_RAW_FIELDS)
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

            if '30D_Y1' in display_df.columns and '30D_Y1' not in summary_comparison_metadata.get('reference_cols', []):
                display_df = display_df.drop('30D_Y1', axis=1)

        elif view_type == 'percentage' and classification_mode == 'Classification Level 1':
            text_cols = ['Aggregation Supply', 'Supply Country']
            delta_cols = list(SUPPLY_DEST_DELTA_RAW_FIELDS)
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

            if '30D_Y1' in display_df.columns and '30D_Y1' not in summary_comparison_metadata.get('reference_cols', []):
                display_df = display_df.drop('30D_Y1', axis=1)

        elif (
            view_type == 'percentage'
            and classification_mode != 'Classification Level 1'
            and (show_demand_country or show_import_classification)
        ):
            # For Country mode, calculate percentages relative to supply country subtotals
            import_col = 'Import Country' if show_demand_country else 'Import Classification'
            text_cols = ['Supply Country', import_col]
            delta_cols = list(SUPPLY_DEST_DELTA_RAW_FIELDS)
            # Exclude delta columns from percentage conversion but include 30D_Y1 for subtotal storage
            numeric_cols = [col for col in display_df.columns if col not in text_cols and col not in delta_cols]
            numeric_cols_with_y1 = numeric_cols + (['30D_Y1'] if '30D_Y1' in display_df.columns else [])

            # Calculate percentages relative to supply country subtotals
            current_supply_country = None
            supply_subtotals = {}

            # First pass: find all subtotal rows and store their values
            for idx, row in display_df.iterrows():
                supply_country = str(row.get('Supply Country', '') or '').strip()
                import_bucket = str(row.get(import_col, '') or '').strip()
                is_parent_row = supply_country and (
                    supply_country.startswith('▶')
                    or supply_country.startswith('▼')
                    or import_bucket in {'', 'Total'}
                )
                if is_parent_row:
                    supply_name = (
                        supply_country[2:].strip()
                        if supply_country.startswith(('▶', '▼'))
                        else supply_country
                    )
                    supply_subtotals[supply_name] = {col: row[col] for col in numeric_cols_with_y1 if col in row}

            # Second pass: convert values to percentages
            for idx, row in display_df.iterrows():
                supply_country = str(row.get('Supply Country', '') or '').strip()
                import_bucket = str(row.get(import_col, '') or '').strip()
                is_parent_row = supply_country and (
                    supply_country.startswith('▶')
                    or supply_country.startswith('▼')
                    or import_bucket in {'', 'Total'}
                )

                # Use the supply country as the grouping key
                if supply_country:
                    current_supply_country = (
                        supply_country[2:].strip()
                        if supply_country.startswith(('▶', '▼'))
                        else supply_country
                    )

                # Apply percentage calculation (excluding 30D_Y1)
                if current_supply_country and current_supply_country in supply_subtotals:
                    for col in numeric_cols:
                        if col in display_df.columns and col != '30D_Y1' and col in supply_subtotals[current_supply_country]:
                            subtotal_val = supply_subtotals[current_supply_country][col]
                            if subtotal_val != 0:
                                if is_parent_row:
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
                current_supply_country_for_y1 = None
                for idx, row in display_df.iterrows():
                    supply_country = str(row.get('Supply Country', '') or '').strip()
                    import_bucket = str(row.get(import_col, '') or '').strip()
                    is_parent_row = supply_country and (
                        supply_country.startswith('▶')
                        or supply_country.startswith('▼')
                        or import_bucket in {'', 'Total'}
                    )
                    if supply_country:
                        current_supply_country_for_y1 = (
                            supply_country[2:].strip()
                            if supply_country.startswith(('▶', '▼'))
                            else supply_country
                        )

                    # Convert 30D_Y1 to percentage and calculate delta
                    # Check if this is a subtotal row.
                    if is_parent_row:
                        # Subtotal rows should have Y/Y delta = 0 (100% - 100% = 0)
                        display_df.at[idx, 'Δ 30D Y/Y'] = 0
                    elif (
                        current_supply_country_for_y1
                        and current_supply_country_for_y1 in supply_subtotals
                    ):
                        if '30D_Y1' in supply_subtotals[current_supply_country_for_y1]:
                            subtotal_y1 = supply_subtotals[current_supply_country_for_y1]['30D_Y1']
                            if subtotal_y1 != 0:
                                # Convert Y-1 value to percentage
                                y1_pct = row['30D_Y1'] / subtotal_y1
                                # Calculate percentage point difference
                                display_df.at[idx, 'Δ 30D Y/Y'] = display_df.at[idx, '30D'] - y1_pct
                            else:
                                display_df.at[idx, 'Δ 30D Y/Y'] = 0
                        else:
                            display_df.at[idx, 'Δ 30D Y/Y'] = 0

            # Drop 30D_Y1 unless the active comparison still needs it.
            if '30D_Y1' in display_df.columns and '30D_Y1' not in summary_comparison_metadata.get('reference_cols', []):
                display_df = display_df.drop('30D_Y1', axis=1)

        elif view_type == 'percentage':
            text_cols = ['Supply Country', 'Supply Installation']
            delta_cols = list(SUPPLY_DEST_DELTA_RAW_FIELDS)
            numeric_cols = [col for col in display_df.columns if col not in text_cols and col not in delta_cols]
            if 'Supply Installation' in display_df.columns:
                parent_mask = (
                    display_df['Supply Country'].fillna('').astype(str).str.strip() != ''
                )
                total_source_df = display_df[parent_mask]
            else:
                total_source_df = display_df
            total_values = {col: total_source_df[col].sum() for col in numeric_cols if col in display_df.columns}
            raw_30d_y1 = None
            if '30D_Y1' in display_df.columns:
                raw_30d_y1 = pd.to_numeric(display_df['30D_Y1'], errors='coerce')
            elif '30D' in display_df.columns and 'Δ 30D Y/Y' in display_df.columns:
                raw_30d_y1 = (
                    pd.to_numeric(display_df['30D'], errors='coerce')
                    - pd.to_numeric(display_df['Δ 30D Y/Y'], errors='coerce')
                )

            for idx, row in display_df.iterrows():
                for col in numeric_cols:
                    total_val = total_values.get(col, 0)
                    if total_val != 0:
                        display_df.at[idx, col] = row[col] / total_val
                    else:
                        display_df.at[idx, col] = 0

            if 'Δ 7D-30D' in display_df.columns and '7D' in display_df.columns and '30D' in display_df.columns:
                display_df['Δ 7D-30D'] = (display_df['7D'] - display_df['30D'])

            if 'Δ 30D Y/Y' in display_df.columns and raw_30d_y1 is not None:
                total_y1 = (
                    raw_30d_y1[parent_mask].sum()
                    if 'Supply Installation' in display_df.columns
                    else raw_30d_y1.sum()
                )
                if total_y1 != 0:
                    display_df['Δ 30D Y/Y'] = display_df['30D'] - (raw_30d_y1 / total_y1)
                else:
                    display_df['Δ 30D Y/Y'] = 0

            if '30D_Y1' in display_df.columns and '30D_Y1' not in summary_comparison_metadata.get('reference_cols', []):
                display_df = display_df.drop('30D_Y1', axis=1)

        if view_type != 'percentage':
            display_df = _convert_supply_dest_absolute_volume_metric(
                display_df,
                volume_metric
            )
        elif snapshot_comparison_metadata['status'] in {'exact', 'fallback'}:
            display_df = _recalculate_supply_dest_pbd_delta_columns(
                display_df
            )

        if snapshot_comparison_metadata['status'] == 'unavailable':
            for column_name in (
                *SUPPLY_DEST_PBD_REFERENCE_COLUMNS,
                *SUPPLY_DEST_PBD_DELTA_COLUMNS,
            ):
                if column_name in display_df.columns:
                    display_df[column_name] = np.nan

        display_df, summary_comparison_delta_cols = _apply_supply_dest_summary_comparison(
            display_df,
            summary_comparison_metadata,
            view_type
        )
        display_df = display_df.drop(
            columns=list(SUPPLY_DEST_PBD_REFERENCE_COLUMNS),
            errors='ignore',
        )

        display_df = _label_supply_dest_total_as_global(display_df)
        display_df = _sort_supply_dest_summary_display_df(display_df)
        columns = _build_supply_dest_columns(
            display_df,
            view_type,
            hidden_cols=['30D_Y1'],
            delta_like_cols=summary_comparison_delta_cols
        )

        # Get conditional styles (includes alternating rows, country totals, grand total)
        conditional_styles = get_table_conditional_styles()

        # Add style for indented rows
        if 'Aggregation Supply' in display_df.columns:
            conditional_styles.insert(1, {
                'if': {'filter_query': '{Aggregation Supply} = ""'},
                'backgroundColor': '#f9f9f9',
                'fontSize': '13px'
            })

        if any(col in display_df.columns for col in ['Supply Installation', 'Import Country', 'Import Classification']):
            conditional_styles.insert(1, {
                'if': {'filter_query': '{Supply Country} = ""'},
                'backgroundColor': '#f9fbfd',
                'color': '#475569',
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

        for import_col in ['Import Country', 'Import Classification']:
            if import_col in display_df.columns:
                conditional_styles.append({
                    'if': {'filter_query': f'{{{import_col}}} = "Total"'},
                    'backgroundColor': TABLE_COLORS['bg_lighter'],
                    'fontWeight': 'bold',
                    'color': TABLE_COLORS['text_primary']
                })

        # Add alignment styles for text columns
        text_columns = [col for col in SUPPLY_DEST_TEXT_COLUMNS if col in display_df.columns]
        for col in text_columns:
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

        year_columns = [c for c in display_df.columns if _is_supply_dest_summary_year_column(c)]
        quarter_columns = [c for c in display_df.columns if c.startswith('Q') and "'" in c]
        month_columns = [
            c for c in display_df.columns
            if "'" in c and not c.startswith('Q') and not c.startswith('W') and c not in text_columns
        ]
        week_columns = [c for c in display_df.columns if c.startswith('W') and "'" in c]
        delta_value_scale = 100 if view_type == 'percentage' else 1

        # Period group banding: years, quarters, months, rolling windows, weeks, and deltas.
        for col in display_df.columns:
            if col in year_columns:
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#f8f5ef',
                    'borderLeft': '2px solid #c8b892' if col == year_columns[0] else '1px solid #eee4d3'
                })
            elif col in quarter_columns:
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#f7f9fc',
                    'borderLeft': '2px solid #c4d0df' if col == quarter_columns[0] else '1px solid #e2e8f0'
                })
            elif col in month_columns:
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#f2faf7',
                    'borderLeft': '2px solid #9bcfc1' if col == month_columns[0] else '1px solid #dcefe9'
                })
            elif col == '30D':
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#fff4d6',
                    'fontWeight': '750',
                    'borderLeft': '2px solid #d7a23a'
                })
            elif col in week_columns:
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#f7f5fb',
                    'borderLeft': '2px solid #b8a8d5' if col == week_columns[0] else '1px solid #e7e0f1'
                })
            elif col == '7D':
                conditional_styles.append({
                    'if': {'column_id': col},
                    'backgroundColor': '#fff0e7',
                    'fontWeight': '750',
                    'borderLeft': '2px solid #d28a63'
                })
            elif col == 'Δ 7D-30D':
                conditional_styles.extend(
                    _build_supply_dest_delta_gradient_styles(
                        display_df,
                        col,
                        base_bg='#f3f5f7',
                        border_color='#aeb7c2',
                        value_scale=delta_value_scale
                    )
                )
            elif col == 'Δ 30D Y/Y':
                conditional_styles.extend(
                    _build_supply_dest_delta_gradient_styles(
                        display_df,
                        col,
                        base_bg='#eef7ee',
                        border_color='#9abc9a',
                        value_scale=delta_value_scale
                    )
                )
            elif col in SUPPLY_DEST_PBD_DELTA_COLUMNS:
                conditional_styles.extend(
                    _build_supply_dest_delta_gradient_styles(
                        display_df,
                        col,
                        base_bg='#eef4fb',
                        border_color='#9bb4cf',
                        value_scale=delta_value_scale
                    )
                )

        # Re-add Global style at the end to ensure highest priority
        total_filter_query = (
            '{Aggregation Supply} = "Global" or {Supply Country} = "Global" or '
            '{Aggregation Supply} = "GRAND TOTAL" or {Supply Country} = "GRAND TOTAL"'
        )
        conditional_styles.append({
            'if': {'filter_query': total_filter_query},
            'backgroundColor': '#2E86C1',  # McKinsey blue
            'fontWeight': 'bold',
            'color': 'white'
        })
        for rolling_col in ['30D', '7D']:
            if rolling_col in display_df.columns:
                conditional_styles.append({
                    'if': {
                        'column_id': rolling_col,
                        'filter_query': total_filter_query
                    },
                    'color': TABLE_COLORS['text_primary']
                })

        # Get base table configuration
        table_config = StandardTableStyleManager.get_base_datatable_config()
        table_config['style_data_conditional'] = conditional_styles
        table_config['style_cell_conditional'] = (
            table_config.get('style_cell_conditional', [])
            + _build_supply_dest_summary_column_width_styles(
                display_df,
                columns,
                view_type,
                delta_like_cols=summary_comparison_delta_cols
            )
        )

        grid_display_df, grid_columns = _build_supply_dest_summary_grid_display(
            display_df,
            columns,
            view_type,
            delta_like_cols=summary_comparison_delta_cols
        )
        grid_columns = _apply_supply_dest_summary_column_classes(grid_columns)
        grid_columns = _apply_supply_dest_delta_heatmap_class_rules(
            grid_columns,
            display_df,
            value_scale=delta_value_scale
        )
        # Create the DataTable with pattern matching ID for click handling
        table = create_ag_grid_from_datatable(
            id={'type': 'supply-dest-expandable-table', 'index': 'summary'},
            data=grid_display_df.to_dict('records'),
            columns=grid_columns,
            page_action='none',
            sort_action='none',
            fill_width=False,
            export_format='none',
            className='supply-dest-summary-grid',
            height='auto',
            defaultColDef={
                'wrapHeaderText': False,
                'autoHeaderHeight': False,
                'suppressHeaderMenuButton': True,
                'suppressHeaderFilterButton': True,
                'resizable': True,
            },
            dashGridOptions={
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
            },
            rowClassRules={
                'supply-dest-summary-total-row': (
                    "params.data && (params.data['Aggregation Supply'] === 'Global' "
                    "|| params.data['Supply Country'] === 'Global' "
                    "|| params.data['Aggregation Supply'] === 'GRAND TOTAL' "
                    "|| params.data['Supply Country'] === 'GRAND TOTAL')"
                ),
                'supply-dest-summary-subtotal-row': (
                    "params.data && (params.data['Aggregation Demand'] === 'Total' "
                    "|| params.data['Demand Country'] === 'Total' "
                    "|| params.data['Import Country'] === 'Total' "
                    "|| params.data['Import Classification'] === 'Total' "
                    "|| params.data['Supply Country'] === 'Total')"
                ),
            },
            **table_config
        )

        # Calculate source-aligned date ranges for the footnote.
        current_snapshot = snapshot_comparison_metadata.get(
            'current_snapshot'
        ) or {}
        current_snapshot_date = current_snapshot.get(
            'snapshot_date_utc'
        )
        today = pd.Timestamp(
            current_snapshot_date
            if current_snapshot_date is not None
            else datetime.now()
        ).date()
        date_7d_start = (today - timedelta(days=6)).strftime('%b %d, %Y')
        date_30d_start = (today - timedelta(days=29)).strftime('%b %d, %Y')
        date_today = today.strftime('%b %d, %Y')

        # Previous year 30D window dates
        date_30d_y1_start = (today - timedelta(days=365) - timedelta(days=29)).strftime('%b %d, %Y')
        date_30d_y1_end = (today - timedelta(days=365)).strftime('%b %d, %Y')
        comparison_note = ''
        if summary_comparison_basis == 'previous_period':
            comparison_note = '; level columns shown vs previous period'
        elif summary_comparison_basis == 'same_period_last_year':
            comparison_note = '; level columns shown vs previous year'
        value_note_text = (
            (
                'Market share levels shown as %; comparison and delta columns shown in percentage points (pp)'
                if comparison_note
                else 'Market share levels shown as %; delta columns shown in percentage points (pp)'
            )
            if view_type == 'percentage'
            else f'Values shown are bilateral trade flows in {vol_label}{comparison_note}'
        )

        def _format_snapshot_lineage(snapshot):
            if not isinstance(snapshot, dict):
                return None
            snapshot_date = snapshot.get('snapshot_date_utc')
            snapshot_timestamp = snapshot.get('snapshot_timestamp_utc')
            if snapshot_date is None or snapshot_timestamp is None:
                return None
            parsed_timestamp = pd.Timestamp(snapshot_timestamp)
            fractional_seconds = (
                f'.{parsed_timestamp.microsecond:06d}'
                if parsed_timestamp.microsecond
                else ''
            )
            return (
                f"{pd.Timestamp(snapshot_date).strftime('%b %d, %Y')} "
                f"{parsed_timestamp.strftime('%H:%M:%S')}"
                f"{fractional_seconds} UTC"
            )

        current_lineage = _format_snapshot_lineage(
            snapshot_comparison_metadata.get('current_snapshot')
        )
        baseline_lineage = _format_snapshot_lineage(
            snapshot_comparison_metadata.get('baseline_snapshot')
        )
        baseline_status = snapshot_comparison_metadata.get('status')
        business_day_gap = snapshot_comparison_metadata.get(
            'business_day_gap'
        )
        if baseline_status == 'exact' and baseline_lineage:
            baseline_note_text = (
                f'Current snapshot: {current_lineage} | '
                f'PBD baseline: {baseline_lineage} | '
                'PBD changes are current minus baseline and include '
                'window roll plus Kpler revisions.'
            )
            baseline_note_class = 'supply-dest-baseline-status'
        elif baseline_status == 'fallback' and baseline_lineage:
            baseline_note_text = (
                f'Current snapshot: {current_lineage} | '
                f'Fallback PBD baseline: {baseline_lineage}'
                + (
                    f' ({business_day_gap} Mon–Fri business days earlier)'
                    if business_day_gap is not None
                    else ''
                )
                + ' | PBD changes are current minus baseline and include '
                'window roll plus Kpler revisions.'
            )
            baseline_note_class = (
                'supply-dest-baseline-status '
                'supply-dest-baseline-status-warning'
            )
        else:
            baseline_note_text = (
                (
                    f'Current snapshot: {current_lineage} | '
                    if current_lineage
                    else ''
                )
                + 'PBD baseline unavailable; Δ 30D vs PBD and '
                'Δ 7D vs PBD show —.'
            )
            baseline_note_class = (
                'supply-dest-baseline-status '
                'supply-dest-baseline-status-unavailable'
            )

        # Create footnote
        footnote = html.Div([
            html.P([
                html.Span('Note: ', style={'fontWeight': 'bold'}),
                html.Span(f'30D: {date_30d_start} to {date_today} | ', style={'color': '#666'}),
                html.Span(f'7D: {date_7d_start} to {date_today} | ', style={'color': '#666'}),
                html.Span(f'30D Y-1: {date_30d_y1_start} to {date_30d_y1_end} | ', style={'color': '#666'}),
                html.Span(value_note_text, style={'color': '#666'})
            ], className='supply-dest-table-footnote-text'),
            html.P(
                baseline_note_text,
                className=baseline_note_class,
            ),
        ], className='supply-dest-table-footnote')

        return html.Div([table, footnote], className='supply-dest-table-shell')

    except Exception as e:
        return html.Div(
            f"Error creating supply-destination table: {str(e)}",
            style={'textAlign': 'center', 'padding': '20px', 'color': 'red'},
        )


def _toggle_supply_dest_row_expansion(display_df, active_cell, classification_mode,
                                      demand_aggregation_mode, expanded_classifications=None,
                                      expanded_countries=None, expanded_supply_countries=None):
    """Toggle the hierarchical expansion state for a clicked supply-destination row."""
    expanded_classifications = list(expanded_classifications or [])
    expanded_countries = list(expanded_countries or [])
    expanded_supply_countries = list(expanded_supply_countries or [])

    if display_df is None or display_df.empty or not active_cell:
        return expanded_classifications, expanded_countries, expanded_supply_countries

    clicked_row, row_index = _get_supply_dest_active_row(display_df, active_cell)
    if clicked_row is None:
        return expanded_classifications, expanded_countries, expanded_supply_countries

    show_demand_aggregation = use_demand_classification_mode(classification_mode, demand_aggregation_mode)
    show_demand_country = use_demand_country_mode(demand_aggregation_mode)

    if classification_mode == 'Classification Level 1' and show_demand_aggregation:
        supply_agg = clicked_row.get('Aggregation Supply', '')
        demand_agg = clicked_row.get('Aggregation Demand', '')

        if supply_agg and _has_supply_dest_expand_marker(supply_agg):
            supply_class = _strip_supply_dest_expand_marker(supply_agg)
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
        if country_demand and _has_supply_dest_expand_marker(country_demand):
            demand_country = _strip_supply_dest_expand_marker(country_demand)

            if row_index is not None:
                for i in range(row_index - 1, -1, -1):
                    prev_row = display_df.iloc[i]
                    prev_supply = prev_row.get('Aggregation Supply', '')
                    prev_demand = prev_row.get('Aggregation Demand', '')
                    if prev_supply and _has_supply_dest_expand_marker(prev_supply):
                        supply_class = _strip_supply_dest_expand_marker(prev_supply)
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

        if supply_agg and _has_supply_dest_expand_marker(supply_agg):
            supply_class = _strip_supply_dest_expand_marker(supply_agg)
            if supply_class in expanded_classifications:
                expanded_classifications.remove(supply_class)
                expanded_countries = [
                    c for c in expanded_countries
                    if not c.startswith(f"{supply_class}→")
                ]
            else:
                expanded_classifications.append(supply_class)

        if demand_country and _has_supply_dest_expand_marker(demand_country):
            demand_country_name = _strip_supply_dest_expand_marker(demand_country)

            if row_index is not None:
                for i in range(row_index - 1, -1, -1):
                    prev_row = display_df.iloc[i]
                    prev_supply = prev_row.get('Aggregation Supply', '')
                    if prev_supply and _has_supply_dest_expand_marker(prev_supply):
                        supply_class = _strip_supply_dest_expand_marker(prev_supply)
                        country_key = f"{supply_class}→{demand_country_name}"
                        if country_key in expanded_countries:
                            expanded_countries.remove(country_key)
                        else:
                            expanded_countries.append(country_key)
                        break

    elif classification_mode == 'Classification Level 1':
        supply_agg = clicked_row.get('Aggregation Supply', '')
        if supply_agg and _has_supply_dest_expand_marker(supply_agg):
            supply_class = _strip_supply_dest_expand_marker(supply_agg)
            if supply_class in expanded_classifications:
                expanded_classifications.remove(supply_class)
            else:
                expanded_classifications.append(supply_class)

    elif (
        use_supply_installation_mode(demand_aggregation_mode)
        or use_demand_country_mode(demand_aggregation_mode)
        or use_import_classification_mode(demand_aggregation_mode)
    ):
        supply_country = clicked_row.get('Supply Country', '')
        if supply_country and _has_supply_dest_expand_marker(supply_country):
            supply_country_name = _strip_supply_dest_expand_marker(supply_country)
            if supply_country_name in expanded_supply_countries:
                expanded_supply_countries.remove(supply_country_name)
            else:
                expanded_supply_countries.append(supply_country_name)

    return expanded_classifications, expanded_countries, expanded_supply_countries


@callback(
    [Output('supply-dest-expanded-classifications', 'data', allow_duplicate=True),
     Output('supply-dest-expanded-countries', 'data', allow_duplicate=True),
     Output('supply-dest-expanded-supply-countries', 'data', allow_duplicate=True)],
    [Input({'type': 'supply-dest-expandable-table', 'index': ALL}, 'cellClicked')],
    [State({'type': 'supply-dest-expandable-table', 'index': ALL}, 'virtualRowData'),
     State('supply-dest-expanded-classifications', 'data'),
     State('supply-dest-expanded-countries', 'data'),
     State('supply-dest-expanded-supply-countries', 'data'),
     State('supply-dest-data-store', 'data'),
     State('country-classification-dropdown', 'value'),
     State('aggregation-demand-dropdown', 'value'),
     State('supply-dest-country-grouping-dropdown', 'value'),
     State('supply-dest-view-type', 'value')],
    prevent_initial_call=True
)
def handle_supply_dest_row_expansion(_active_cells, table_data_list, expanded_classifications,
                                     expanded_countries, expanded_supply_countries,
                                     supply_dest_data, classification_mode,
                                     demand_aggregation_mode, country_grouping_mode, view_type):
    """Handle clicking on rows to expand/collapse in the overview supply-destination table."""
    supply_dest_data = _resolve_exporters_store(supply_dest_data)
    ctx = callback_context
    if not ctx.triggered:
        return expanded_classifications, expanded_countries, expanded_supply_countries

    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']

    if 'supply-dest-expandable-table' in prop_id and '.cellClicked' in prop_id:
        try:
            active_cell = ag_grid_cell_clicked_to_active_cell(triggered['value'])
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
                    df, classification_mode,
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
    [Output('continent-year-selector', 'options'),
     Output('continent-year-selector', 'value')],
    Input('continent-charts-data', 'data'),
    State('continent-year-selector', 'value'),
    prevent_initial_call=False
)
def update_continent_year_selector_options(continent_data, selected_years):
    """Populate the continent chart year selector from the configured chart window."""
    if continent_data:
        try:
            _resolve_exporters_continent_payload(continent_data)
        except _SnapshotUnavailable:
            return _exporters_snapshot_recovery_selector_result()

    available_years = _get_continent_chart_available_years()
    if not available_years:
        return [], []

    options = _get_continent_year_selector_options()
    selected = _normalize_continent_chart_selected_years(selected_years, available_years)
    return options, selected


# Callback to dynamically generate continent charts based on classification mode
@callback(
    Output('continent-charts-container', 'children'),
    [Input('continent-charts-data', 'data'),
     Input('continent-chart-type', 'value'),
     Input('country-classification-dropdown', 'value'),
     Input('exporters-volume-metric-dropdown', 'value'),
     Input('continent-year-selector', 'value'),
     Input('supply-rolling-window-days-input', 'value')],
    prevent_initial_call=False
)
@log_callback_timing("exporters.continent_charts_render")
def update_continent_charts(
    continent_data,
    chart_type,
    classification_mode,
    volume_metric,
    selected_years,
    rolling_avg_days
):
    """Dynamically generate continent charts based on classification mode"""
    if not continent_data:
        return html.Div("No data available", className='continent-rolling-empty-state')

    try:
        continent_payload = (
            _resolve_or_load_exporters_continent_payload(
                continent_data,
                classification_mode,
                rolling_avg_days,
                selected_years,
            )
        )
    except _SnapshotUnavailable:
        return _exporters_snapshot_recovery_notice()
    except PreventUpdate:
        raise
    except Exception:
        # Match the frozen callback behavior for transient DB/build failures:
        # preserve the entity cards, but render them from an empty dataframe.
        try:
            continent_payload = dict(
                _resolve_exporters_continent_payload(
                    continent_data
                )
            )
        except _SnapshotUnavailable:
            return _exporters_snapshot_recovery_notice()
        continent_payload['data'] = pd.DataFrame()

    entities_list = continent_payload['entities']
    continent_batch_df = continent_payload['data']
    if not entities_list:
        return html.Div("No data available", className='continent-rolling-empty-state')

    rolling_avg_days = normalize_supply_rolling_avg_days(rolling_avg_days)

    charts = []
    kpi_summary_rows = []

    # Create chart for each entity
    for entity_name in entities_list:
        entity_continent_df = (
            continent_batch_df[continent_batch_df['entity_name'] == entity_name].copy()
            if not continent_batch_df.empty and 'entity_name' in continent_batch_df.columns
            else pd.DataFrame()
        )
        if chart_type == 'percentage':
            fig = _create_continent_percentage_chart_from_df(
                entity_continent_df,
                selected_years=selected_years,
            )
        else:
            fig = _create_continent_destination_chart_from_df(
                entity_continent_df,
                volume_metric or 'mcm_d',
                selected_years=selected_years,
                rolling_avg_days=rolling_avg_days
            )

        card_class_name = (
            'continent-rolling-card continent-rolling-card-primary'
            if entity_name == 'Global'
            else 'continent-rolling-card'
        )
        fig_meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
        kpi_metrics = fig_meta.get('continent_kpis', [])
        kpi_summary_rows.append({
            'entity': entity_name,
            'metrics': kpi_metrics
        })

        chart_div = html.Div([
            html.Div(
                [
                    html.H5(entity_name, className='continent-rolling-card-title')
                ],
                className='continent-rolling-card-header'
            ),
            dcc.Graph(
                id=f'continent-chart-{entity_name.replace(" ", "-").lower()}',
                figure=fig,
                config={'displayModeBar': False, 'responsive': True},
                className='continent-rolling-graph',
                style={'height': '328px', 'width': '100%'}
            )
        ], className=card_class_name)

        charts.append(chart_div)

    return html.Div(
        [
            html.Div(
                _build_continent_kpi_summary_table(kpi_summary_rows),
                className='continent-rolling-summary-panel'
            ),
            html.Div(
                html.Div(
                    charts,
                    className='continent-rolling-grid'
                ),
                className='continent-rolling-charts-panel'
            )
        ],
        className='continent-rolling-content'
    )


def _format_supply_chart_current_value(metrics, vol_label):
    if not metrics or metrics.get('latest_value') is None:
        return None

    latest_label = metrics.get('latest_label') or metrics.get('focus_year') or ''
    return f"{latest_label}: {metrics['latest_value']:,.0f} {vol_label}"


def _build_supply_chart_delta_pill(label, delta_value, delta_pct):
    if delta_value is None or pd.isna(delta_value):
        return html.Span(f'{label} n/a', className='supply-rolling-delta-pill supply-rolling-delta-neutral')

    direction_class = 'supply-rolling-delta-neutral'
    if delta_value > 0:
        direction_class = 'supply-rolling-delta-positive'
    elif delta_value < 0:
        direction_class = 'supply-rolling-delta-negative'

    sign = '+' if delta_value > 0 else ''
    pct_text = ''
    if delta_pct is not None and pd.notna(delta_pct):
        pct_text = f" ({sign}{delta_pct:.0f}%)"

    return html.Span(
        [
            html.Span(label, className='supply-rolling-delta-label'),
            html.Span(f"{sign}{delta_value:,.0f}{pct_text}")
        ],
        className=f'supply-rolling-delta-pill {direction_class}'
    )


def _build_supply_chart_delta_indicators(metrics):
    return html.Div(
        [
            _build_supply_chart_delta_pill(
                'MoM',
                metrics.get('mom_delta_value') if metrics else None,
                metrics.get('mom_delta_pct') if metrics else None
            ),
            _build_supply_chart_delta_pill(
                'YoY',
                metrics.get('delta_value') if metrics else None,
                metrics.get('delta_pct') if metrics else None
            )
        ],
        className='supply-rolling-delta-group'
    )


# Callback to dynamically generate supply charts based on classification mode
@callback(
    Output('supply-charts-container', 'children'),
    [Input('supply-charts-data', 'data'),
     Input('exporters-volume-metric-dropdown', 'value'),
     Input('supply-year-selector', 'value'),
     Input('supply-rolling-window-days-input', 'value')],
    prevent_initial_call=False
)
@log_callback_timing("exporters.supply_charts_render")
def update_supply_charts(
    charts_data,
    volume_metric,
    selected_years,
    rolling_avg_days
):
    """Dynamically generate supply charts based on classification mode"""
    try:
        charts_data = _resolve_exporters_store(charts_data)
    except _SnapshotUnavailable:
        return _exporters_snapshot_recovery_notice()
    if not charts_data:
        return html.Div("No data available", className='supply-rolling-empty-state')

    charts = []
    vol_label = _get_volume_metric_info(volume_metric or 'mcm_d')['label']
    rolling_avg_days = normalize_supply_rolling_avg_days(rolling_avg_days)

    # Create chart for each entity (country or classification group)
    for entity_name, entity_data in charts_data.items():
        fig = create_supply_chart(
            entity_data,
            show_legend=False,
            volume_metric=volume_metric or 'mcm_d',
            selected_years=selected_years,
            rolling_avg_days=rolling_avg_days
        )
        metrics = get_supply_chart_header_metrics(
            entity_data,
            volume_metric=volume_metric or 'mcm_d',
            selected_years=selected_years,
            rolling_avg_days=rolling_avg_days
        )
        current_value = _format_supply_chart_current_value(metrics, vol_label)
        card_class_name = (
            'supply-rolling-card supply-rolling-card-primary'
            if entity_name == 'Global'
            else 'supply-rolling-card'
        )

        chart_div = html.Div([
            html.Div(
                [
                    html.Div(
                        [
                            html.H5(entity_name, className='supply-rolling-card-title'),
                            html.Span(current_value, className='supply-rolling-current-value') if current_value else None
                        ],
                        className='supply-rolling-card-title-group'
                    ),
                    _build_supply_chart_delta_indicators(metrics)
                ],
                className='supply-rolling-card-header'
            ),
            dcc.Graph(
                id=f'supply-chart-{entity_name.replace(" ", "-").lower()}',
                figure=fig,
                config={'displayModeBar': False, 'responsive': True},
                className='supply-rolling-graph',
                style={'height': '328px', 'width': '100%'}
            )
        ], className=card_class_name)

        charts.append(chart_div)

    return html.Div(
        charts,
        className='supply-rolling-grid'
    )


def _resolve_table_export_df(derived_virtual_data_list, data_list, columns_list):
    """Resolve the current AG Grid payload into an ordered export dataframe."""
    selected_rows = None
    selected_columns = None

    row_sources = derived_virtual_data_list or []
    fallback_sources = data_list or []
    column_sources = columns_list or []

    for idx, column_defs in enumerate(column_sources):
        if column_defs:
            selected_columns = ag_grid_column_defs_to_datatable_columns(column_defs)
            if idx < len(row_sources) and row_sources[idx] is not None:
                selected_rows = row_sources[idx]
            elif idx < len(fallback_sources) and fallback_sources[idx] is not None:
                selected_rows = fallback_sources[idx]
            break

    if selected_columns is None:
        for column_defs in column_sources:
            if column_defs:
                selected_columns = ag_grid_column_defs_to_datatable_columns(column_defs)
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
    State({'type': 'supply-dest-expandable-table', 'index': ALL}, 'virtualRowData'),
    State({'type': 'supply-dest-expandable-table', 'index': ALL}, 'rowData'),
    State({'type': 'supply-dest-expandable-table', 'index': ALL}, 'columnDefs'),
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


# Export callback for Supply Charts
@callback(
    Output('download-supply-charts-excel', 'data'),
    Input('export-supply-charts-button', 'n_clicks'),
    State('supply-charts-data', 'data'),
    State('exporters-volume-metric-dropdown', 'value'),
    State('supply-rolling-window-days-input', 'value'),
    prevent_initial_call=True
)
def export_supply_charts_to_excel(n_clicks, charts_data, volume_metric, rolling_avg_days):
    """Export LNG Supply rolling-average data to Excel"""
    charts_data = _resolve_exporters_store(charts_data)
    if n_clicks == 0 or not charts_data:
        return None
    rolling_avg_days = normalize_supply_rolling_avg_days(rolling_avg_days)
    vol_label = _get_volume_metric_info(volume_metric)['label']
    rolling_col = f'rolling_avg_{rolling_avg_days}d ({vol_label})'

    # Convert all entities' data to DataFrames
    all_data = []
    for entity_name, entity_data in charts_data.items():
        if entity_data:
            df = pd.DataFrame(entity_data)
            df = _convert_volume_metric_dataframe(
                df,
                volume_metric,
                columns=['rolling_avg'],
                period_days=rolling_avg_days
            )
            df['entity'] = entity_name
            all_data.append(df)

    if not all_data:
        return None

    # Create Excel file with BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Combined sheet with all data
        combined_df = safe_concat(all_data, ignore_index=True)
        if 'rolling_avg' in combined_df.columns:
            combined_df = combined_df.rename(columns={'rolling_avg': rolling_col})
        # Reorder columns for better readability
        cols = ['entity', 'date', 'year', 'month_day', rolling_col, 'is_forecast']
        cols = [c for c in cols if c in combined_df.columns]
        combined_df = combined_df[cols]
        combined_df.to_excel(writer, sheet_name='All Data', index=False)

        # Individual sheets per entity
        for entity_name, entity_data in charts_data.items():
            if entity_data:
                df = pd.DataFrame(entity_data)
                df = _convert_volume_metric_dataframe(
                    df,
                    volume_metric,
                    columns=['rolling_avg'],
                    period_days=rolling_avg_days
                )
                if 'rolling_avg' in df.columns:
                    df = df.rename(columns={'rolling_avg': rolling_col})
                # Excel sheet name limit is 31 characters
                sheet_name = entity_name[:31].replace('/', '-').replace('\\', '-')
                sheet_cols = ['date', 'year', 'month_day', rolling_col, 'is_forecast']
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
                    except Exception:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    output.seek(0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'LNG_Supply_{rolling_avg_days}D_Rolling_{timestamp}.xlsx'

    return dcc.send_bytes(output.getvalue(), filename)


# Export callback for Continent Charts
@callback(
    Output('download-continent-charts-excel', 'data'),
    Input('export-continent-charts-button', 'n_clicks'),
    State('continent-charts-data', 'data'),
    State('continent-chart-type', 'value'),
    State('country-classification-dropdown', 'value'),
    State('exporters-volume-metric-dropdown', 'value'),
    State('continent-year-selector', 'value'),
    State('supply-rolling-window-days-input', 'value'),
    prevent_initial_call=True
)
def export_continent_charts_to_excel(
    n_clicks,
    continent_data,
    chart_type,
    classification_mode,
    volume_metric,
    selected_years,
    rolling_avg_days
):
    """Export LNG Supply by Destination Continent data to Excel"""
    if n_clicks == 0 or not continent_data:
        return None
    try:
        continent_payload = (
            _resolve_or_load_exporters_continent_export_payload(
                continent_data,
                classification_mode,
                rolling_avg_days,
            )
        )
    except _SnapshotUnavailable:
        raise
    except PreventUpdate:
        raise
    except Exception:
        return None
    entities_list = continent_payload['entities']
    continent_batch_df = continent_payload['data']
    if not entities_list:
        return None

    rolling_avg_days = normalize_supply_rolling_avg_days(rolling_avg_days)
    vol_label = _get_volume_metric_info(volume_metric)['label']

    all_data = []
    for entity_name in entities_list:
        try:
            df = (
                continent_batch_df[
                    continent_batch_df['entity_name'] == entity_name
                ].copy()
                if (
                    not continent_batch_df.empty
                    and 'entity_name' in continent_batch_df.columns
                )
                else pd.DataFrame()
            )
            if df.empty:
                continue

            df['_year_token'] = df['year'].apply(
                _continent_chart_year_token
            )
            available_years = sorted(
                [
                    year
                    for year
                    in df['_year_token'].dropna().unique()
                ],
                key=_continent_chart_year_sort_key
            )
            active_years = _normalize_continent_chart_selected_years(
                selected_years,
                available_years,
                use_default=selected_years is None
            )
            if not active_years:
                continue

            df = (
                df[df['_year_token'].isin(active_years)]
                .drop(columns=['_year_token'])
                .copy()
            )
            if df.empty:
                continue

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(
                    df['date'],
                    errors='coerce'
                ).dt.date

            if chart_type != 'percentage':
                df = _convert_volume_metric_dataframe(
                    df,
                    volume_metric,
                    columns=['rolling_avg'],
                    period_days=rolling_avg_days
                )
            df['entity'] = entity_name
            all_data.append(df)
        except Exception:
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
            rolling_col = f'rolling_avg_{rolling_avg_days}d ({vol_label})'
            if 'rolling_avg' in combined_df.columns:
                combined_df = combined_df.rename(columns={'rolling_avg': rolling_col})
            cols = ['entity', 'date', 'continent_destination', 'year', 'month_day', rolling_col, 'is_forecast']
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
                rolling_col = f'rolling_avg_{rolling_avg_days}d ({vol_label})'
                entity_df = entity_df.rename(columns={'rolling_avg': rolling_col})
                sheet_cols = ['date', 'continent_destination', 'year', 'month_day', rolling_col, 'is_forecast']
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
                    except Exception:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    output.seek(0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_type_label = 'Absolute' if chart_type == 'absolute' else 'Percentage'
    filename = f'LNG_Supply_by_Continent_{chart_type_label}_{rolling_avg_days}D_Rolling_{timestamp}.xlsx'

    return dcc.send_bytes(output.getvalue(), filename)
