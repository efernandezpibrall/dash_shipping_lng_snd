"""Shared LNG volume-metric policy for the exporter/importer detail pages."""

from __future__ import annotations

import re

import pandas as pd

from utils.detail_controls import (
    normalize_rolling_window_days as _normalize_unbounded_rolling_window_days,
)


BCM_PER_MMTPA = 1.36
DAYS_PER_YEAR = 365.25
MCM_PER_BCM = 1000
MCM_PER_MT = BCM_PER_MMTPA * MCM_PER_BCM
MCM_PER_MONTH_PER_MMTPA = MCM_PER_MT / 12
MCM_D_PER_MMTPA = MCM_PER_MT / DAYS_PER_YEAR
MMTPA_PER_MCM_D = DAYS_PER_YEAR / MCM_PER_MT
DETAIL_MAX_ROLLING_WINDOW_DAYS = 180

VOLUME_CONVERSIONS = {
    'mcm_d': {
        'factor': 1.0,
        'label': 'mcm/d',
        'quantity_kind': 'rate',
        'display_precision': 0,
    },
    'bcm': {
        'factor': None,
        'label': 'bcm',
        'quantity_kind': 'period_volume',
        'display_precision': 1,
    },
    'mt': {
        'factor': None,
        'label': 'MT',
        'quantity_kind': 'period_volume',
        'display_precision': 1,
    },
    'mtpa': {
        'factor': MMTPA_PER_MCM_D,
        'label': 'MMTPA',
        'quantity_kind': 'rate',
        'display_precision': 1,
    },
}

VOLUME_METRIC_OPTIONS = [
    {'label': 'mcm/d', 'value': 'mcm_d'},
    {'label': 'bcm', 'value': 'bcm'},
    {'label': 'MT', 'value': 'mt'},
    {'label': 'MMTPA', 'value': 'mtpa'},
]


def normalize_rolling_window_days(window_days, default=30):
    """Clamp detail-page rolling inputs to the supported complete-window range."""
    normalized_days = _normalize_unbounded_rolling_window_days(
        window_days,
        default=default,
    )
    return max(1, min(DETAIL_MAX_ROLLING_WINDOW_DAYS, normalized_days))


def normalize_volume_metric(volume_metric):
    return volume_metric if volume_metric in VOLUME_CONVERSIONS else 'mcm_d'


def get_volume_metric_info(volume_metric):
    return VOLUME_CONVERSIONS[normalize_volume_metric(volume_metric)]


def get_volume_metric_display_precision(volume_metric):
    return int(get_volume_metric_info(volume_metric).get('display_precision', 0))


def get_volume_metric_plotly_format(volume_metric):
    return f",.{get_volume_metric_display_precision(volume_metric)}f"


def get_volume_metric_zero_tolerance(volume_metric):
    return 0.5 * (10 ** -get_volume_metric_display_precision(volume_metric))


def round_volume_metric_display_value(value, volume_metric):
    rounded_value = round(float(value), get_volume_metric_display_precision(volume_metric))
    return 0.0 if rounded_value == 0 else rounded_value


def is_period_volume_metric(volume_metric):
    return get_volume_metric_info(volume_metric).get('quantity_kind') == 'period_volume'


def rolling_measure_name(volume_metric):
    return 'Rolling Volume' if is_period_volume_metric(volume_metric) else 'Rolling Average'


def format_rolling_title(rolling_window_days, volume_metric):
    days = normalize_rolling_window_days(rolling_window_days)
    return f"{days}-Day {rolling_measure_name(volume_metric)}"


def get_rolling_metric_export_column_name(rolling_window_days, volume_metric):
    days = normalize_rolling_window_days(rolling_window_days)
    measure = 'rolling_volume' if is_period_volume_metric(volume_metric) else 'rolling_avg'
    label = get_volume_metric_info(volume_metric)['label']
    return f'{measure}_{days}d ({label})'


def get_excel_number_format(volume_metric):
    precision = get_volume_metric_display_precision(volume_metric)
    return '#,##0' if precision == 0 else '#,##0.' + ('0' * precision)


def apply_excel_metric_format(worksheet, volume_metric, metric_headers=None):
    metric_headers = set(metric_headers or [])
    number_format = get_excel_number_format(volume_metric)
    for column_cells in worksheet.iter_cols(min_row=1, max_row=worksheet.max_row):
        header = column_cells[0].value
        if metric_headers and header not in metric_headers:
            continue
        for cell in column_cells[1:]:
            if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                cell.number_format = number_format


def get_volume_metric_factor(volume_metric, period_days=None):
    normalized_metric = normalize_volume_metric(volume_metric)
    if normalized_metric == 'bcm':
        days = period_days if period_days is not None else DAYS_PER_YEAR
        return days / MCM_PER_BCM
    if normalized_metric == 'mt':
        days = period_days if period_days is not None else DAYS_PER_YEAR
        return days / MCM_PER_MT
    return VOLUME_CONVERSIONS[normalized_metric]['factor']


def convert_volume_metric_series(series, volume_metric, period_days=None, precision=None):
    numeric_series = pd.to_numeric(series, errors='coerce')
    converted_series = numeric_series * get_volume_metric_factor(
        volume_metric,
        period_days=period_days,
    )
    if precision is not None:
        converted_series = converted_series.round(precision)
    return converted_series.where(pd.notnull(converted_series), None)


def convert_volume_metric_dataframe(
    df,
    volume_metric,
    columns=None,
    exclude_columns=None,
    precision=None,
    period_days=None,
    period_days_by_column=None,
):
    if df is None or df.empty:
        return df

    converted_df = df.copy()
    exclude_columns = set(exclude_columns or [])
    period_days_by_column = period_days_by_column or {}
    if columns is None:
        columns = [
            column for column in converted_df.columns
            if column not in exclude_columns
        ]

    for column in columns:
        if column not in converted_df.columns or column in exclude_columns:
            continue
        converted_df[column] = convert_volume_metric_series(
            converted_df[column],
            volume_metric,
            period_days=period_days_by_column.get(column, period_days),
            precision=precision,
        )
    return converted_df


def format_metric_value(value, volume_metric):
    if value is None or pd.isna(value):
        return None
    metric_info = get_volume_metric_info(volume_metric)
    precision = get_volume_metric_display_precision(volume_metric)
    rounded_value = round_volume_metric_display_value(value, volume_metric)
    return f"{rounded_value:,.{precision}f} {metric_info['label']}"


def format_table_metric_value(value, volume_metric, is_delta=False):
    if value is None or value is pd.NA:
        return '—'
    try:
        if pd.isna(value):
            return '—'
        rounded_value = round_volume_metric_display_value(value, volume_metric)
    except (TypeError, ValueError):
        return str(value)
    precision = get_volume_metric_display_precision(volume_metric)
    sign = '+' if is_delta and rounded_value > 0 else ''
    return f'{sign}{rounded_value:,.{precision}f}'


def maintenance_raw_mcmd_field(column_id):
    safe_column_id = re.sub(r'[^0-9A-Za-z]+', '_', str(column_id)).strip('_').lower()
    return f'__maintenance_raw_mcmd_{safe_column_id or "period"}'
