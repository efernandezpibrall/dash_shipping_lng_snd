from dash import (
    html,
    dcc,
    callback,
    Output,
    Input,
    State,
    callback_context,
    no_update,
)
from dash.dash_table.Format import Format, Scheme
from utils.ag_grid_tables import ag_grid_cell_clicked_to_active_cell, create_ag_grid_from_datatable
from dash.exceptions import MissingCallbackContextException, PreventUpdate
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import uuid
import zlib
import calendar
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from datetime import datetime, timedelta
from sqlalchemy import text
from utils.dashboard_snapshot_cache import (
    SnapshotUnavailable as _SnapshotUnavailable,
    build_source_key as _build_source_key,
    get_or_build_snapshot as _get_or_build_snapshot,
    is_snapshot_reference as _is_snapshot_reference,
    pack_record_mapping as _pack_record_mapping,
    resolve_snapshot as _resolve_snapshot,
    snapshot_is_resolvable as _snapshot_is_resolvable,
    snapshot_is_shared as _snapshot_is_shared,
    unpack_record_mapping as _unpack_record_mapping,
    was_global_refresh_triggered as _was_global_refresh_triggered,  # noqa: F401 - compatibility hook
    with_snapshot_slot as _with_snapshot_slot,
)
from utils.performance import log_callback_timing
from utils.performance_flags import (
    revision_aware_refresh_enabled as _revision_aware_refresh_enabled,
)
from utils.arrow_payload import (
    ARROW_RECORD_CUBE_FORMAT,
    decode_arrow_record_cube as _decode_arrow_record_cube,
)

from pages.importer_detail import (
    engine,
    DB_SCHEMA,
    MCM_PER_CUBIC_METER,
    build_destination_catalog,
    get_destination_catalog_dataframe,
    _fetch_importer_scoped_trades,
    _apply_importer_self_flow_exclusion,
    _build_importer_total_import_df,
    _build_importer_continent_chart_df,
    _prepare_importer_summary_scope_df,
    _build_importer_rolling_windows_pivot,
    build_importer_origin_summary_from_scoped_trades,
    DESTINATION_AGGREGATION_LABELS,
    IMPORTER_MAPPING_RENAME,
    IMPORTER_ORIGIN_LEVEL_TO_SCOPE,
    get_volume_metric_info,
    convert_volume_metric_dataframe,
)


DEFAULT_IMPORTER_ROLLING_AVG_DAYS = 30
MIN_IMPORTER_ROLLING_AVG_DAYS = 1
MAX_IMPORTER_ROLLING_AVG_DAYS = 180
VOLUME_METRIC_OPTIONS = [
    {'label': 'mcm/d', 'value': 'mcm_d'},
    {'label': 'bcm', 'value': 'bcm'},
    {'label': 'MT', 'value': 'mt'},
    {'label': 'MMTPA', 'value': 'mtpa'},
]
IMPORTER_VOLUME_METRIC_VALUES = frozenset(
    option['value'] for option in VOLUME_METRIC_OPTIONS
)
TOP_IMPORTER_CHART_COUNT = 12
MONTH_ORDER = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
IMPORTER_GLOBAL_LABEL = 'Global'
IMPORTER_REST_OF_IMPORTERS_LABEL = 'Rest of Importers'
IMPORTER_CHART_COLOR_SEQUENCE = [
    '#7a5195',
    '#ef5675',
    '#9f7aea',
    '#c44e52',
    '#d08b36',
    '#607d8b',
    '#1aa6a6',
    '#0f4c81',
]
IMPORTER_CHART_FORECAST_DASH = 'dot'
IMPORTER_CHART_ANCHOR_YEAR = 2024
IMPORTER_CHART_DISPLAY_START_DATE = '2021-01-01'
IMPORTER_CHART_DEFAULT_SELECTED_YEAR_COUNT = 2
IMPORTER_CHART_DEFAULT_DESELECTED_YEARS = {'2024'}
IMPORTER_CHART_RANGE_LOOKBACK_YEARS = 5
IMPORTER_CHART_RANGE_FILL = 'rgba(148, 163, 184, 0.20)'
IMPORTER_PERIOD_DEFAULT_QUARTER_COUNT = 5
IMPORTER_PERIOD_DEFAULT_MONTH_COUNT = 3
IMPORTER_PERIOD_DEFAULT_WEEK_COUNT = 3
IMPORTER_PERIOD_MAX_QUARTER_COUNT = 8
IMPORTER_PERIOD_MAX_MONTH_COUNT = 48
IMPORTER_PERIOD_MAX_WEEK_COUNT = 12
IMPORTER_PERIOD_ORIGIN_GROUPING_OPTIONS = [
    {'label': 'Yes', 'value': 'group_small_countries'},
    {'label': 'No', 'value': 'show_all'}
]
IMPORTER_PERIOD_VIEW_OPTIONS = [
    {'label': 'Volume', 'value': 'absolute'},
    {'label': 'Market Share (%)', 'value': 'percentage'}
]
IMPORTER_PERIOD_COMPARISON_BASIS_OPTIONS = [
    {'label': 'None', 'value': 'levels'},
    {'label': 'vs Previous Period', 'value': 'previous_period'},
    {'label': 'vs Previous Year', 'value': 'same_period_last_year'}
]
IMPORTER_PERIOD_TEXT_COLUMNS = ['Importer', 'Aggregation']
IMPORTER_PERIOD_PERCENTAGE_DISPLAY_PRECISION = 1
IMPORTER_PERIOD_DELTA_RAW_FIELD_PREFIX = '__importer_period_delta_raw_'
IMPORTER_PERIOD_PBD_CURRENT_COLUMNS = (
    '30D_PBD_CURRENT',
    '7D_PBD_CURRENT',
)
IMPORTER_PERIOD_PBD_REFERENCE_COLUMNS = ('30D_PBD', '7D_PBD')
IMPORTER_PERIOD_PBD_DELTA_COLUMNS = (
    'Δ 30D vs PBD',
    'Δ 7D vs PBD',
)
ORIGIN_CONTINENT_CHART_COLOR_MAP = {
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
ORIGIN_CONTINENT_CHART_TYPE_OPTIONS = [
    {'label': 'Volume', 'value': 'absolute'},
    {'label': 'Market Share (%)', 'value': 'percentage'},
]

IMPORTERS_SOURCE_NAMESPACE = 'importers-source-v4'
IMPORTERS_LEGACY_OVERVIEW_NAMESPACE = 'importers-overview-v5'
IMPORTERS_ARROW_OVERVIEW_NAMESPACE = 'importers-overview-v6'
IMPORTERS_OVERVIEW_NAMESPACE = IMPORTERS_LEGACY_OVERVIEW_NAMESPACE
IMPORTERS_OVERVIEW_NAMESPACES = frozenset({
    IMPORTERS_LEGACY_OVERVIEW_NAMESPACE,
    IMPORTERS_ARROW_OVERVIEW_NAMESPACE,
})
IMPORTERS_PERIOD_NAMESPACE = 'importers-period-v4'
IMPORTERS_SOURCE_STATE_FORMAT = 'importers-source-state-v3'
IMPORTERS_PERIOD_PAYLOAD_FORMAT = 'importers-period-summary-v3'
IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE = (
    'Cached importer data is unavailable. Click the global Refresh button '
    'to reload it.'
)
IMPORTERS_LEGACY_RECORD_CUBE_FORMAT = 'importers-record-cube-zlib-json-v1'
IMPORTERS_RECORD_CUBE_FORMAT = IMPORTERS_LEGACY_RECORD_CUBE_FORMAT
IMPORTERS_SCALAR_TAG = '__importers_scalar_v1__'


def _tag_importers_json_value(value):
    value_type = type(value)
    if value is None or value_type in (str, int, float, bool):
        return value
    if value is pd.NaT:
        return {IMPORTERS_SCALAR_TAG: 'pandas.NaT'}
    if value is pd.NA:
        return {IMPORTERS_SCALAR_TAG: 'pandas.NA'}
    if isinstance(value, pd.Timestamp):
        timezone = value.tz
        timezone_name = (
            getattr(timezone, 'zone', None)
            or getattr(timezone, 'key', None)
            or (str(timezone) if timezone is not None else None)
        )
        return {
            IMPORTERS_SCALAR_TAG: 'pandas.Timestamp',
            'nanoseconds': value.value,
            'timezone': timezone_name,
        }
    if isinstance(value, np.generic):
        return {
            IMPORTERS_SCALAR_TAG: 'numpy.scalar',
            'dtype': value.dtype.str,
            'bytes': value.tobytes().hex(),
        }
    if isinstance(value, datetime):
        return {
            IMPORTERS_SCALAR_TAG: 'datetime.datetime',
            'isoformat': value.isoformat(),
            'fold': value.fold,
        }
    if isinstance(value, bytes):
        return {
            IMPORTERS_SCALAR_TAG: 'builtins.bytes',
            'hex': value.hex(),
        }
    if isinstance(value, tuple):
        return {
            IMPORTERS_SCALAR_TAG: 'builtins.tuple',
            'items': [
                _tag_importers_json_value(item)
                for item in value
            ],
        }
    if isinstance(value, list):
        return [
            _tag_importers_json_value(item)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _tag_importers_json_value(item)
            for key, item in value.items()
        }
    raise TypeError(f'{type(value).__name__} is not JSON serializable')


def _decode_importers_json_object(value):
    scalar_type = value.get(IMPORTERS_SCALAR_TAG)
    if scalar_type == 'pandas.NaT' and len(value) == 1:
        return pd.NaT
    if scalar_type == 'pandas.NA' and len(value) == 1:
        return pd.NA
    if (
        scalar_type == 'pandas.Timestamp'
        and set(value) == {
            IMPORTERS_SCALAR_TAG,
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
            IMPORTERS_SCALAR_TAG,
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
            IMPORTERS_SCALAR_TAG,
            'isoformat',
            'fold',
        }
    ):
        return datetime.fromisoformat(value['isoformat']).replace(
            fold=value['fold']
        )
    if (
        scalar_type == 'builtins.bytes'
        and set(value) == {IMPORTERS_SCALAR_TAG, 'hex'}
    ):
        return bytes.fromhex(value['hex'])
    if (
        scalar_type == 'builtins.tuple'
        and set(value) == {IMPORTERS_SCALAR_TAG, 'items'}
    ):
        return tuple(value['items'])
    return value


def _encode_importers_json_payload(value, payload_format):
    raw_payload = json.dumps(
        _tag_importers_json_value(value),
        ensure_ascii=False,
        separators=(',', ':'),
    ).encode('utf-8')
    return {
        'format': payload_format,
        'payload': zlib.compress(raw_payload, level=1),
    }


def _decode_importers_json_payload(value, payload_format):
    if (
        isinstance(value, dict)
        and value.get('format') == ARROW_RECORD_CUBE_FORMAT
    ):
        try:
            return _decode_arrow_record_cube(value)
        except Exception as exc:
            raise _SnapshotUnavailable(
                IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE
            ) from exc
    if not (
        isinstance(value, dict)
        and value.get('format') == payload_format
    ):
        return value
    try:
        encoded_payload = value['payload']
        if not isinstance(encoded_payload, bytes):
            raise TypeError('encoded importer payload is not bytes')
        return json.loads(
            zlib.decompress(encoded_payload).decode('utf-8'),
            object_hook=_decode_importers_json_object,
        )
    except Exception as exc:
        raise _SnapshotUnavailable(
            IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        ) from exc


def _prepare_importers_overview_snapshot_payload(payload):
    prepared = dict(payload)
    if 'demand_years' not in prepared:
        prepared['demand_years'] = _get_importer_chart_available_years(
            _unpack_record_mapping(payload['demand_cube'])
        )
    if 'origin_years' not in prepared:
        prepared['origin_years'] = _get_importer_chart_available_years(
            _unpack_record_mapping(payload['origin_cube'])
        )
    prepared['demand_cube'] = _encode_importers_json_payload(
        payload['demand_cube'],
        IMPORTERS_LEGACY_RECORD_CUBE_FORMAT,
    )
    prepared['origin_cube'] = _encode_importers_json_payload(
        payload['origin_cube'],
        IMPORTERS_LEGACY_RECORD_CUBE_FORMAT,
    )
    return prepared


def _fetch_importers_source_watermark():
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


def _normalize_importers_source_watermark(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


def _normalize_importers_snapshot_metadata(source_pair, prefix):
    """Normalize one snapshot row from the atomic current/PBD lookup."""
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
        'snapshot_timestamp_utc': _normalize_importers_source_watermark(
            snapshot_timestamp
        ),
        'facts_retained': bool(
            source_pair.get(f'{prefix}_facts_retained')
        ),
    }


def _previous_importer_weekday_utc(value):
    candidate = pd.Timestamp(value).date() - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def _importer_business_day_gap(start_date, end_date):
    if start_date is None or end_date is None:
        return None
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if start >= end:
        return 0
    return int(np.busday_count(start.isoformat(), end.isoformat()))


def _build_importers_source_state(source_pair, refresh_token=None):
    """Build the versioned current/PBD source contract."""
    current_snapshot = _normalize_importers_snapshot_metadata(
        source_pair,
        'current',
    )
    baseline_snapshot = _normalize_importers_snapshot_metadata(
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
            'format': IMPORTERS_SOURCE_STATE_FORMAT,
            'watermark': _normalize_importers_source_watermark(
                scalar_watermark
            ),
            'as_of_date': datetime.now().date().isoformat(),
            'current_snapshot': None,
            'baseline_snapshot': None,
            'baseline_status': 'unavailable',
            'business_day_gap': None,
            'refresh_token': refresh_token,
        }

    expected_baseline_date = _previous_importer_weekday_utc(
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
        business_day_gap = _importer_business_day_gap(
            baseline_date,
            current_snapshot['snapshot_date_utc'],
        )

    return {
        'format': IMPORTERS_SOURCE_STATE_FORMAT,
        'watermark': current_snapshot['snapshot_timestamp_utc'],
        'as_of_date': current_snapshot['snapshot_date_utc'],
        'current_snapshot': current_snapshot,
        'baseline_snapshot': baseline_snapshot,
        'baseline_status': baseline_status,
        'business_day_gap': business_day_gap,
        'refresh_token': refresh_token,
    }


def _resolve_importers_source_store(source_data):
    try:
        resolved = _resolve_snapshot(
            source_data,
            engine,
            expected_namespace=IMPORTERS_SOURCE_NAMESPACE,
        )
    except _SnapshotUnavailable as exc:
        raise _SnapshotUnavailable(
            IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        ) from exc
    if not (
        isinstance(resolved, dict)
        and isinstance(resolved.get('catalog_df'), pd.DataFrame)
        and isinstance(resolved.get('ranking_df'), pd.DataFrame)
        and isinstance(resolved.get('scoped_trades_df'), pd.DataFrame)
    ):
        raise _SnapshotUnavailable(
            IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        )
    return resolved


def _reject_noncurrent_importers_overview_reference(value):
    if (
        _is_snapshot_reference(value)
        and value.get('namespace') not in IMPORTERS_OVERVIEW_NAMESPACES
    ):
        raise _SnapshotUnavailable(
            IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        )


def _resolve_importers_chart_store(charts_data):
    _reject_noncurrent_importers_overview_reference(charts_data)
    expected_namespace = (
        charts_data.get('namespace')
        if _is_snapshot_reference(charts_data)
        else IMPORTERS_OVERVIEW_NAMESPACE
    )
    try:
        resolved = _resolve_snapshot(
            charts_data,
            engine,
            expected_namespace=expected_namespace,
        )
    except _SnapshotUnavailable as exc:
        raise _SnapshotUnavailable(
            IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        ) from exc
    resolved = _decode_arrow_record_cube(resolved)
    resolved = _decode_importers_json_payload(
        resolved,
        IMPORTERS_LEGACY_RECORD_CUBE_FORMAT,
    )
    return _unpack_record_mapping(resolved)


def _resolve_importers_entities_store(entities_data, slot):
    _reject_noncurrent_importers_overview_reference(entities_data)
    expected_namespace = (
        entities_data.get('namespace')
        if _is_snapshot_reference(entities_data)
        else IMPORTERS_OVERVIEW_NAMESPACE
    )
    try:
        return _resolve_snapshot(
            entities_data,
            engine,
            expected_namespace=expected_namespace,
            slot=slot,
        )
    except _SnapshotUnavailable as exc:
        raise _SnapshotUnavailable(
            IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        ) from exc


def _resolve_importers_years_store(charts_data, slot):
    _reject_noncurrent_importers_overview_reference(charts_data)
    expected_namespace = (
        charts_data.get('namespace')
        if _is_snapshot_reference(charts_data)
        else IMPORTERS_OVERVIEW_NAMESPACE
    )
    try:
        years = _resolve_snapshot(
            charts_data,
            engine,
            expected_namespace=expected_namespace,
            slot=slot,
        )
    except _SnapshotUnavailable as exc:
        raise _SnapshotUnavailable(
            IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE
        ) from exc
    if isinstance(years, list):
        return list(years)
    return _get_importer_chart_available_years(
        _resolve_importers_chart_store(charts_data)
    )


def _resolve_importers_period_store(period_data):
    return _resolve_snapshot(
        period_data,
        engine,
        expected_namespace=IMPORTERS_PERIOD_NAMESPACE,
    )


def _importers_snapshot_recovery_notice():
    return html.Div(
        IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE,
        className='importers-snapshot-recovery-message',
        role='alert',
    )


def _importers_snapshot_recovery_selector_result():
    return (
        [{
            'label': IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE,
            'value': '__snapshot_unavailable__',
            'disabled': True,
        }],
        [],
    )

IMPORTER_CLASSIFICATION_OPTIONS = [
    {'label': 'Country', 'value': 'Country'},
    {'label': 'Classification Level 1', 'value': 'Classification Level 1'}
]

ORIGIN_LEVEL_OPTIONS = [
    {'label': 'Shipping Region', 'value': 'origin_shipping_region'},
    {'label': 'Country', 'value': 'origin_country_name'},
    {'label': 'Basin', 'value': 'origin_basin'},
    {'label': 'Continent', 'value': 'continent_origin_name'},
    {'label': 'Subcontinent', 'value': 'origin_subcontinent'},
    {'label': 'Classification Level 1', 'value': 'origin_classification_level1'},
    {'label': 'Classification', 'value': 'origin_classification'},
]

ORIGIN_LEVEL_LABELS = {
    option['value']: option['label']
    for option in ORIGIN_LEVEL_OPTIONS
}


def normalize_importer_rolling_avg_days(value):
    """Clamp the overview rolling-average window to a practical range."""
    try:
        days = int(round(float(value)))
    except (TypeError, ValueError):
        days = DEFAULT_IMPORTER_ROLLING_AVG_DAYS
    return max(MIN_IMPORTER_ROLLING_AVG_DAYS, min(MAX_IMPORTER_ROLLING_AVG_DAYS, days))


def _get_importer_volume_metric_info(volume_metric):
    """Return normalized metadata for an Importers overview metric."""
    normalized_metric = (
        volume_metric
        if volume_metric in IMPORTER_VOLUME_METRIC_VALUES
        else 'mcm_d'
    )
    return get_volume_metric_info(normalized_metric)


def _get_importer_volume_metric_display_precision(volume_metric):
    return int(
        _get_importer_volume_metric_info(volume_metric).get(
            'display_precision',
            0,
        )
    )


def _get_importer_volume_metric_plotly_number_format(volume_metric):
    return f',.{_get_importer_volume_metric_display_precision(volume_metric)}f'


def _round_importer_volume_metric_display_value(value, volume_metric):
    precision = _get_importer_volume_metric_display_precision(volume_metric)
    rounded_value = round(float(value), precision)
    return 0.0 if rounded_value == 0 else rounded_value


def _is_importer_period_volume_metric(volume_metric):
    return (
        _get_importer_volume_metric_info(volume_metric).get('quantity_kind')
        == 'period_volume'
    )


def _get_importer_chart_query_start_date(rolling_avg_days):
    """Return the exact warm-up boundary for the first visible chart point."""
    display_start = pd.Timestamp(IMPORTER_CHART_DISPLAY_START_DATE).normalize()
    preceding_days = normalize_importer_rolling_avg_days(rolling_avg_days) - 1
    return (display_start - pd.Timedelta(days=preceding_days)).strftime('%Y-%m-%d')


def _get_importer_chart_query_end_date(current_date):
    """Include the same 14-day forecast horizon rendered by the chart builders."""
    if current_date is None:
        return None
    return (
        pd.Timestamp(current_date).normalize() + pd.Timedelta(days=14)
    ).strftime('%Y-%m-%d')


def _format_importer_rolling_average_section_title(
    title_prefix,
    rolling_avg_days,
    volume_metric='mcm_d',
):
    days = normalize_importer_rolling_avg_days(rolling_avg_days)
    measure = (
        'Rolling Volume'
        if _is_importer_period_volume_metric(volume_metric)
        else 'Rolling Average'
    )
    return f'{title_prefix} - {days}-Day {measure}'


def _format_importer_rolling_window_label(rolling_avg_days):
    return f'{normalize_importer_rolling_avg_days(rolling_avg_days)}D'


def _get_importer_rolling_metric_export_column_name(
    rolling_avg_days,
    volume_metric,
):
    days = normalize_importer_rolling_avg_days(rolling_avg_days)
    measure = (
        'rolling_volume'
        if _is_importer_period_volume_metric(volume_metric)
        else 'rolling_avg'
    )
    vol_label = _get_importer_volume_metric_info(volume_metric)['label']
    return f'{measure}_{days}d ({vol_label})'


def _build_importer_period_count_options(max_count, min_count=1):
    return [{'label': str(value), 'value': value} for value in range(min_count, max_count + 1)]


def _coerce_importer_period_count(value, default, max_count, min_count=1):
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = default
    return max(min_count, min(max_count, count))


def _normalize_importer_period_origin_grouping(grouping_mode):
    """Normalize the importer period-table small-country grouping mode."""
    if grouping_mode in {'group_small_countries', 'group_small', 'yes', 'Yes', True}:
        return 'group_small_countries'
    return 'show_all'


def _normalize_importer_period_view_type(view_type):
    """Normalize the period-analysis view selector."""
    if view_type == 'percentage':
        return 'percentage'
    return 'absolute'


def _normalize_importer_period_comparison_basis(comparison_basis):
    """Normalize the period-analysis comparison selector."""
    if comparison_basis in {'levels', 'previous_period', 'same_period_last_year'}:
        return comparison_basis
    return 'levels'


def _create_top_importers_selector_region():
    """Render the overview controls using the exporter-page header structure."""
    return [
        html.Div(
            [
                html.Div('Importer Classification', className='filter-group-header'),
                dcc.RadioItems(
                    id='imp-overview-classification-mode',
                    options=IMPORTER_CLASSIFICATION_OPTIONS,
                    value='Country',
                    inline=True,
                    className='importers-sticky-selector importers-classification-selector',
                    inputStyle={'display': 'none'},
                    labelStyle={'marginRight': '0'}
                )
            ],
            className='filter-group importers-sticky-filter-group'
        ),
        html.Div(
            [
                html.Div('Group small', className='filter-group-header'),
                dcc.RadioItems(
                    id='imp-overview-origin-country-grouping-dropdown',
                    options=IMPORTER_PERIOD_ORIGIN_GROUPING_OPTIONS,
                    value='group_small_countries',
                    inline=True,
                    className='importers-sticky-selector importers-grouping-selector',
                    inputStyle={'display': 'none'},
                    labelStyle={'marginRight': '0'}
                )
            ],
            className='filter-group importers-sticky-filter-group'
        ),
        html.Div(
            [
                html.Div('Metric', className='filter-group-header'),
                dcc.RadioItems(
                    id='imp-overview-volume-metric-dropdown',
                    options=VOLUME_METRIC_OPTIONS,
                    value='mcm_d',
                    inline=True,
                    className='importers-sticky-selector importers-volume-selector',
                    inputStyle={'display': 'none'},
                    labelStyle={'marginRight': '0'}
                )
            ],
            className='filter-group importers-sticky-filter-group'
        ),
        html.Div(
            [
                html.Div('Rolling Avg', className='filter-group-header'),
                html.Div(
                    [
                        dcc.Input(
                            id='imp-overview-rolling-window-days-input',
                            type='number',
                            value=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
                            min=MIN_IMPORTER_ROLLING_AVG_DAYS,
                            max=MAX_IMPORTER_ROLLING_AVG_DAYS,
                            step=1,
                            debounce=0.8,
                            className='importers-rolling-window-input'
                        ),
                        html.Span('days', className='importers-rolling-window-unit')
                    ],
                    className='importers-rolling-window-control'
                )
            ],
            className='filter-group importers-sticky-filter-group importers-rolling-filter-group'
        ),
    ]


def _classification_mode_to_destination_aggregation(classification_mode):
    """Map the overview classification mode to importer-detail destination aggregation keys."""
    if classification_mode == 'Classification Level 1':
        return 'country_classification_level1'
    return 'country'


def _slugify_filename_label(label):
    """Create a filesystem-friendly filename fragment."""
    if not label:
        return 'importers'
    cleaned = ''.join(char if char.isalnum() else '_' for char in str(label).strip())
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    return cleaned.strip('_') or 'importers'


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


def _build_chart_export_df(
    charts_data,
    volume_metric='mcm_d',
    selected_years=None,
    chart_type='absolute',
    rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
):
    """Flatten the chart-data store into a single export dataframe."""
    if not charts_data:
        return pd.DataFrame()

    rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
    available_years = _get_importer_chart_available_years(charts_data)
    active_years = set(_normalize_importer_chart_selected_years(selected_years, available_years))
    all_frames = []
    for entity_name, records in charts_data.items():
        if not records:
            continue
        entity_df = pd.DataFrame(records)
        if entity_df.empty:
            continue

        if active_years and 'year' in entity_df.columns:
            entity_df['_year_token'] = entity_df['year'].apply(_importer_chart_year_token)
            entity_df = entity_df[entity_df['_year_token'].isin(active_years)].drop(columns=['_year_token'])
        if entity_df.empty:
            continue

        if chart_type == 'percentage' and 'percentage' in entity_df.columns:
            entity_df = entity_df.rename(columns={'percentage': 'market_share (%)'})
            entity_df = entity_df.drop(columns=['rolling_avg'], errors='ignore')
        else:
            entity_df = convert_volume_metric_dataframe(
                entity_df,
                volume_metric,
                columns=['rolling_avg'],
                period_days=rolling_avg_days,
            )
            if 'rolling_avg' in entity_df.columns:
                entity_df = entity_df.rename(columns={
                    'rolling_avg': _get_importer_rolling_metric_export_column_name(
                        rolling_avg_days,
                        volume_metric,
                    )
                })
        entity_df.insert(0, 'entity', entity_name)
        all_frames.append(entity_df)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def _normalize_importer_mapping_value(value):
    if pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized if normalized else None


def _collapse_importer_mapping_values(series):
    normalized_values = sorted({
        value
        for value in (
            _normalize_importer_mapping_value(item)
            for item in series
        )
        if value is not None
    })
    if len(normalized_values) == 1:
        return normalized_values[0]
    return 'Unknown'


def _first_importer_mapping_value(series, fallback=''):
    for item in series:
        normalized = _normalize_importer_mapping_value(item)
        if normalized is not None:
            return normalized
    return fallback


def _fetch_importers_catalog_ranking_source_df(
    snapshot_timestamp_utc=None,
    as_of_date=None,
):
    """Fetch the destination catalog and exact 30-day ranking inputs once."""
    if snapshot_timestamp_utc is None:
        snapshot_cte = f"""
            SELECT snapshot_timestamp_utc AS max_ts
            FROM {DB_SCHEMA}.kpler_trade_snapshots
            WHERE run_kind = 'canonical' AND status = 'published'
            ORDER BY
                snapshot_date_utc DESC,
                snapshot_timestamp_utc DESC,
                snapshot_id DESC
            LIMIT 1
        """
        ranking_start_clause = (
            "CURRENT_DATE - INTERVAL '29 days'"
        )
        ranking_end_clause = (
            "CURRENT_DATE + INTERVAL '1 day'"
        )
        params = None
    else:
        normalized_as_of_date = pd.Timestamp(
            as_of_date
        ).normalize().date()
        snapshot_cte = (
            'SELECT CAST(:snapshot_timestamp_utc AS timestamptz) '
            'AS max_ts'
        )
        ranking_start_clause = ':ranking_start_date'
        ranking_end_clause = ':ranking_end_date'
        params = {
            'snapshot_timestamp_utc': snapshot_timestamp_utc,
            'ranking_start_date': (
                normalized_as_of_date - timedelta(days=29)
            ),
            'ranking_end_date': (
                normalized_as_of_date + timedelta(days=1)
            ),
        }
    query = text(f"""
        WITH latest_timestamp AS (
            {snapshot_cte}
        ),
        destinations AS (
            SELECT DISTINCT kt.destination_country_name
            FROM {DB_SCHEMA}.kpler_trades kt
            CROSS JOIN latest_timestamp
            WHERE kt.upload_timestamp_utc = latest_timestamp.max_ts
                AND kt.destination_country_name IS NOT NULL
        ),
        ranking AS (
            SELECT
                kt.destination_country_name,
                SUM(
                    COALESCE(
                        kt.cargo_destination_cubic_meters,
                        0
                    ) * {MCM_PER_CUBIC_METER}
                ) / 30.0 AS avg_30d_mcmd
            FROM {DB_SCHEMA}.kpler_trades kt
            CROSS JOIN latest_timestamp
            INNER JOIN {DB_SCHEMA}.mappings_country
                destination_map
                ON destination_map.country =
                    kt.destination_country_name
            LEFT JOIN {DB_SCHEMA}.mappings_country origin_map
                ON origin_map.country = kt.origin_country_name
            WHERE kt.upload_timestamp_utc = latest_timestamp.max_ts
                AND kt.destination_country_name IS NOT NULL
                AND kt."end" IS NOT NULL
                AND kt.cargo_destination_cubic_meters IS NOT NULL
                AND kt."end" >= {ranking_start_clause}
                AND kt."end" < {ranking_end_clause}
                AND (
                    origin_map.country IS NULL
                    OR NULLIF(BTRIM(origin_map.country_name), '')
                        IS DISTINCT FROM NULLIF(BTRIM(
                            destination_map.country_name
                        ), '')
                )
            GROUP BY kt.destination_country_name
        )
        SELECT
            destinations.destination_country_name,
            ranking.avg_30d_mcmd
        FROM destinations
        LEFT JOIN ranking USING (destination_country_name)
        ORDER BY destinations.destination_country_name
    """)
    if params is None:
        return pd.read_sql(query, engine)
    return pd.read_sql(query, engine, params=params)


def _fetch_importers_mapping_source_df():
    """Load the single mapping source shared by catalog and scoped trades."""
    mapping_columns = [
        'country_name',
        'country',
        *[
            column
            for column in DESTINATION_AGGREGATION_LABELS
            if column != 'country'
        ],
    ]
    return pd.read_sql(
        text(f"""
            SELECT {', '.join(mapping_columns)}
            FROM {DB_SCHEMA}.mappings_country
            WHERE country_name IS NOT NULL OR country IS NOT NULL
        """),
        engine,
    )


def _build_destination_catalog_mapping_df(mapping_source_df):
    mapping_df = (
        mapping_source_df.copy()
        if mapping_source_df is not None
        else pd.DataFrame()
    )
    expected_columns = [
        'country_name',
        'country',
        *[
            column
            for column in DESTINATION_AGGREGATION_LABELS
            if column != 'country'
        ],
    ]
    for column in expected_columns:
        if column not in mapping_df.columns:
            mapping_df[column] = None

    mapping_df['country'] = mapping_df['country'].apply(
        _normalize_importer_mapping_value
    )
    mapping_df = mapping_df[mapping_df['country'].notna()].copy()
    if mapping_df.empty:
        return pd.DataFrame(columns=[
            'country',
            'country_display',
            *[
                column
                for column in DESTINATION_AGGREGATION_LABELS
                if column != 'country'
            ],
        ])

    mapping_df['country_display'] = mapping_df['country_name']
    aggregation_spec = {
        'country_display': (
            lambda series: _first_importer_mapping_value(series)
        ),
    }
    for column in DESTINATION_AGGREGATION_LABELS:
        if column == 'country':
            continue
        aggregation_spec[column] = _collapse_importer_mapping_values
    return mapping_df.groupby(
        'country',
        as_index=False,
    ).agg(aggregation_spec)


def _build_importer_mapping_lookup_from_source(mapping_source_df):
    """Build importer-detail's origin lookup without another database read."""
    lookup_columns = [
        'mapping_key',
        *list(IMPORTER_MAPPING_RENAME.values()),
    ]
    mapping_df = (
        mapping_source_df.copy()
        if mapping_source_df is not None
        else pd.DataFrame()
    )
    expected_columns = [
        'country_name',
        'country',
        *list(IMPORTER_MAPPING_RENAME.keys()),
    ]
    for column in expected_columns:
        if column not in mapping_df.columns:
            mapping_df[column] = None
        mapping_df[column] = mapping_df[column].apply(
            _normalize_importer_mapping_value
        )

    mapping_df['country_name'] = mapping_df['country_name'].fillna(
        mapping_df['country']
    )
    mapping_df = mapping_df[
        mapping_df['country_name'].notna()
    ].copy()
    if mapping_df.empty:
        return pd.DataFrame(columns=lookup_columns)

    canonical_spec = {}
    for column in IMPORTER_MAPPING_RENAME:
        canonical_spec[column] = _collapse_importer_mapping_values

    raw_alias_df = mapping_df[
        ['country', *list(IMPORTER_MAPPING_RENAME.keys())]
    ].rename(columns={'country': 'mapping_key'})
    canonical_alias_df = (
        mapping_df.groupby('country_name', as_index=False)
        .agg(canonical_spec)
        .rename(columns={'country_name': 'mapping_key'})
    )
    lookup_df = pd.concat(
        [raw_alias_df, canonical_alias_df],
        ignore_index=True,
    )
    lookup_df['mapping_key'] = lookup_df['mapping_key'].apply(
        _normalize_importer_mapping_value
    )
    lookup_df = lookup_df[lookup_df['mapping_key'].notna()].copy()
    lookup_df = lookup_df.drop_duplicates(
        subset=['mapping_key'],
        keep='first',
    )
    lookup_df = lookup_df.rename(columns=IMPORTER_MAPPING_RENAME)
    for column in IMPORTER_MAPPING_RENAME.values():
        lookup_df[column] = (
            lookup_df[column]
            .apply(_normalize_importer_mapping_value)
            .fillna('Unknown')
        )
    return lookup_df[lookup_columns]


def _build_destination_catalog_and_ranking_from_sources(
    catalog_ranking_source_df,
    mapping_source_df,
):
    source_df = (
        catalog_ranking_source_df.copy()
        if catalog_ranking_source_df is not None
        else pd.DataFrame()
    )
    if source_df.empty:
        empty_catalog = get_destination_catalog_dataframe([])
        return empty_catalog, pd.DataFrame()

    if 'destination_country_name' not in source_df.columns:
        source_df['destination_country_name'] = None
    if 'avg_30d_mcmd' not in source_df.columns:
        source_df['avg_30d_mcmd'] = pd.NA
    source_df['destination_country_name'] = (
        source_df['destination_country_name']
        .apply(_normalize_importer_mapping_value)
    )
    source_df = (
        source_df[source_df['destination_country_name'].notna()]
        .drop_duplicates(subset=['destination_country_name'])
        .copy()
    )
    if source_df.empty:
        empty_catalog = get_destination_catalog_dataframe([])
        return empty_catalog, pd.DataFrame()

    mapping_df = _build_destination_catalog_mapping_df(
        mapping_source_df
    )
    catalog_df = source_df[
        ['destination_country_name']
    ].merge(
        mapping_df,
        how='left',
        left_on='destination_country_name',
        right_on='country',
    )
    catalog_df['country'] = catalog_df['destination_country_name']
    catalog_df['country_display'] = catalog_df[
        'country_display'
    ].fillna(catalog_df['destination_country_name'])
    for column in DESTINATION_AGGREGATION_LABELS:
        if column == 'country':
            continue
        catalog_df[column] = catalog_df[column].fillna('Unknown')
    catalog_df = get_destination_catalog_dataframe(
        catalog_df.to_dict('records')
    )
    catalog_df = catalog_df.sort_values(
        ['country_display', 'destination_country_name']
    ).reset_index(drop=True)

    ranked_source_df = source_df[
        source_df['avg_30d_mcmd'].notna()
    ][['destination_country_name', 'avg_30d_mcmd']].copy()
    if ranked_source_df.empty:
        ranking_df = catalog_df[
            ['destination_country_name', 'country_display']
        ].copy()
        ranking_df['avg_30d_mcmd'] = pd.NA
        return catalog_df, ranking_df

    ranking_df = ranked_source_df.merge(
        catalog_df[
            ['destination_country_name', 'country_display']
        ],
        how='left',
        on='destination_country_name',
    )
    ranking_df['country_display'] = ranking_df[
        'country_display'
    ].fillna(ranking_df['destination_country_name'])
    ranking_df = ranking_df.sort_values(
        ['avg_30d_mcmd', 'destination_country_name'],
        ascending=[False, True],
    ).reset_index(drop=True)
    return catalog_df, ranking_df


def _fetch_destination_ranking_df(catalog_df=None):
    """Return latest-30D destination demand rankings at country level."""
    if catalog_df is None:
        catalog_records = build_destination_catalog(engine)
        catalog_df = get_destination_catalog_dataframe(catalog_records)
    if catalog_df.empty:
        return pd.DataFrame()

    query = text(f"""
        WITH latest_timestamp AS (
            SELECT snapshot_timestamp_utc AS max_ts
            FROM {DB_SCHEMA}.kpler_trade_snapshots
            WHERE run_kind = 'canonical' AND status = 'published'
            ORDER BY snapshot_date_utc DESC
            LIMIT 1
        )
        SELECT
            kt.destination_country_name,
            SUM(COALESCE(kt.cargo_destination_cubic_meters, 0) * {MCM_PER_CUBIC_METER}) / 30.0 AS avg_30d_mcmd
        FROM {DB_SCHEMA}.kpler_trades kt
        CROSS JOIN latest_timestamp
        INNER JOIN {DB_SCHEMA}.mappings_country destination_map
            ON destination_map.country = kt.destination_country_name
        LEFT JOIN {DB_SCHEMA}.mappings_country origin_map
            ON origin_map.country = kt.origin_country_name
        WHERE kt.upload_timestamp_utc = latest_timestamp.max_ts
            AND kt.destination_country_name IS NOT NULL
            AND kt."end" IS NOT NULL
            AND kt.cargo_destination_cubic_meters IS NOT NULL
            AND kt."end" >= CURRENT_DATE - INTERVAL '29 days'
            AND kt."end" < CURRENT_DATE + INTERVAL '1 day'
            AND (
                origin_map.country IS NULL
                OR NULLIF(BTRIM(origin_map.country_name), '')
                    IS DISTINCT FROM NULLIF(BTRIM(
                        destination_map.country_name
                    ), '')
            )
        GROUP BY kt.destination_country_name
        ORDER BY avg_30d_mcmd DESC, kt.destination_country_name
    """)
    ranking_df = pd.read_sql(query, engine)
    if ranking_df.empty:
        ranking_df = catalog_df[['destination_country_name', 'country_display']].copy()
        ranking_df['avg_30d_mcmd'] = pd.NA
        return ranking_df

    ranking_df = ranking_df.merge(
        catalog_df[['destination_country_name', 'country_display']],
        how='left',
        on='destination_country_name'
    )
    ranking_df['country_display'] = ranking_df['country_display'].fillna(
        ranking_df['destination_country_name']
    )
    return ranking_df


def _build_destination_entities(
    classification_mode='Country',
    limit=None,
    catalog_df=None,
    ranking_df=None,
    include_global=False,
    include_rest=False
):
    """Build importer entities for charts or tables from preloaded destination metadata."""
    if catalog_df is None:
        catalog_records = build_destination_catalog(engine)
        catalog_df = get_destination_catalog_dataframe(catalog_records)
    if ranking_df is None:
        ranking_df = _fetch_destination_ranking_df(catalog_df)
    if catalog_df.empty:
        return []

    merged_df = catalog_df.merge(
        ranking_df[['destination_country_name', 'avg_30d_mcmd']],
        how='left',
        on='destination_country_name'
    )
    merged_df['avg_30d_mcmd'] = pd.to_numeric(merged_df['avg_30d_mcmd'], errors='coerce')
    merged_df['avg_30d_mcmd'] = merged_df['avg_30d_mcmd'].fillna(0.0)
    all_destination_countries = sorted(
        value for value in merged_df['destination_country_name'].dropna().unique().tolist()
    )

    def _global_entity():
        return {
            'key': '__global__',
            'label': IMPORTER_GLOBAL_LABEL,
            'destination_countries': all_destination_countries,
            'avg_30d_mcmd': round(float(merged_df['avg_30d_mcmd'].sum()), 1),
            'is_global': True,
        }

    if classification_mode == 'Classification Level 1':
        if 'country_classification_level1' not in merged_df.columns:
            merged_df['country_classification_level1'] = 'Unknown'
        grouped_df = merged_df.groupby('country_classification_level1', dropna=False).agg(
            avg_30d_mcmd=('avg_30d_mcmd', 'sum'),
            destination_countries=('destination_country_name', lambda series: tuple(sorted(set(series.tolist()))))
        ).reset_index()
        grouped_df['label'] = grouped_df['country_classification_level1'].fillna('Unknown')
        grouped_df = grouped_df.sort_values(['avg_30d_mcmd', 'label'], ascending=[False, True]).reset_index(drop=True)
        if limit is not None:
            grouped_df = grouped_df.head(limit).copy()
        entities = [
            {
                'key': row['country_classification_level1'],
                'label': row['label'],
                'destination_countries': list(row['destination_countries']),
                'avg_30d_mcmd': round(float(row['avg_30d_mcmd']), 1) if pd.notna(row['avg_30d_mcmd']) else None
            }
            for _, row in grouped_df.iterrows()
        ]
        if include_global and all_destination_countries:
            entities = [_global_entity()] + entities
        return entities

    country_df = merged_df[['destination_country_name', 'country_display', 'avg_30d_mcmd']].drop_duplicates(
        subset=['destination_country_name']
    )
    country_df = country_df.sort_values(
        ['avg_30d_mcmd', 'country_display', 'destination_country_name'],
        ascending=[False, True, True]
    ).reset_index(drop=True)
    selected_country_df = country_df.head(limit).copy() if limit is not None else country_df.copy()
    entities = [
        {
            'key': row['destination_country_name'],
            'label': row['country_display'],
            'destination_countries': [row['destination_country_name']],
            'avg_30d_mcmd': round(float(row['avg_30d_mcmd']), 1) if pd.notna(row['avg_30d_mcmd']) else None
        }
        for _, row in selected_country_df.iterrows()
    ]
    if include_rest and limit is not None and len(country_df) > len(selected_country_df):
        visible_countries = set(selected_country_df['destination_country_name'].dropna().tolist())
        rest_df = country_df[~country_df['destination_country_name'].isin(visible_countries)].copy()
        if not rest_df.empty:
            entities.append({
                'key': '__rest__',
                'label': IMPORTER_REST_OF_IMPORTERS_LABEL,
                'destination_countries': rest_df['destination_country_name'].dropna().tolist(),
                'avg_30d_mcmd': round(float(rest_df['avg_30d_mcmd'].sum()), 1),
                'is_rest': True,
            })
    if include_global and all_destination_countries:
        entities = [_global_entity()] + entities
    return entities


def _get_entity_destination_country_set(entity):
    return set(entity.get('destination_countries') or [])


def _filter_scoped_trades_for_entity(scoped_trades_df, entity, classification_mode):
    if scoped_trades_df is None or scoped_trades_df.empty:
        return pd.DataFrame(columns=scoped_trades_df.columns if scoped_trades_df is not None else [])

    entity_df = scoped_trades_df.copy()
    if not entity.get('is_global'):
        destination_countries = _get_entity_destination_country_set(entity)
        if 'destination_country_name' in entity_df.columns:
            entity_df = entity_df[entity_df['destination_country_name'].isin(destination_countries)].copy()
        else:
            return pd.DataFrame(columns=scoped_trades_df.columns)

    if entity_df.empty:
        return entity_df

    if entity.get('is_global') or entity.get('is_rest'):
        if 'destination_country_name' in entity_df.columns:
            return entity_df[
                entity_df['origin_country'].fillna('Unknown') !=
                entity_df['destination_country_name'].fillna('Unknown')
            ].copy()
        return entity_df

    return _apply_importer_self_flow_exclusion(
        entity_df,
        _classification_mode_to_destination_aggregation(classification_mode),
        entity['key']
    )


def _build_chart_data_payload(importer_entities, classification_mode='Country',
                              rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
                              scoped_trades_df=None,
                              global_scoped_trades_df=None,
                              current_date=None,
                              snapshot_timestamp_utc=None):
    """Build the demand and origin-continent chart payloads for overview importers."""
    demand_charts_data = {}
    origin_continent_charts_data = {}
    all_destination_countries = sorted({
        country
        for entity in importer_entities or []
        for country in entity.get('destination_countries', [])
    })
    if not all_destination_countries:
        return demand_charts_data, origin_continent_charts_data

    rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
    chart_query_start_date = _get_importer_chart_query_start_date(
        rolling_avg_days
    )
    query_kwargs = {
        'min_end_date': chart_query_start_date,
        'include_destination_context': True,
    }
    if snapshot_timestamp_utc is not None:
        query_kwargs['snapshot_timestamp_utc'] = snapshot_timestamp_utc
    if current_date is not None:
        query_kwargs['max_end_date'] = _get_importer_chart_query_end_date(
            current_date
        )

    if scoped_trades_df is None:
        selected_aggregation = (
            _classification_mode_to_destination_aggregation(
                classification_mode
            )
        )
        scoped_trades_df = _fetch_importer_scoped_trades(
            engine,
            all_destination_countries,
            selected_destination_aggregation=selected_aggregation,
            **query_kwargs,
        )
        if selected_aggregation != 'country':
            global_scoped_trades_df = _fetch_importer_scoped_trades(
                engine,
                all_destination_countries,
                selected_destination_aggregation='country',
                **query_kwargs,
            )
    if global_scoped_trades_df is None:
        global_scoped_trades_df = scoped_trades_df
    for entity in importer_entities:
        entity_label = entity['label']
        try:
            entity_source_df = (
                global_scoped_trades_df
                if entity.get('is_global')
                else scoped_trades_df
            )
            filtered_df = _filter_scoped_trades_for_entity(
                entity_source_df,
                entity,
                classification_mode,
            )

            demand_df = _build_importer_total_import_df(
                filtered_df,
                rolling_window_days=rolling_avg_days,
                chart_start_date=chart_query_start_date,
                display_start_date=IMPORTER_CHART_DISPLAY_START_DATE,
                current_date=current_date,
            )
            demand_charts_data[entity_label] = demand_df.to_dict('records') if not demand_df.empty else []

            origin_continent_df = _build_importer_continent_chart_df(
                filtered_df,
                rolling_window_days=rolling_avg_days,
                include_percentage=True,
                chart_start_date=chart_query_start_date,
                display_start_date=IMPORTER_CHART_DISPLAY_START_DATE,
                current_date=current_date,
            )
            origin_continent_charts_data[entity_label] = (
                origin_continent_df.to_dict('records') if not origin_continent_df.empty else []
            )
        except Exception:
            demand_charts_data[entity_label] = []
            origin_continent_charts_data[entity_label] = []

    return demand_charts_data, origin_continent_charts_data


def _build_small_importer_origin_grouping(
    scoped_trades_df,
    origin_level='origin_shipping_region',
    threshold_mcmd=10,
    lookback_months=24,
    as_of_date=None,
):
    """Derive the small-origin taxonomy once from the current vintage."""
    if scoped_trades_df is None or scoped_trades_df.empty:
        return None
    if 'origin_country' not in scoped_trades_df.columns or 'end_date' not in scoped_trades_df.columns:
        return None

    grouped_df = scoped_trades_df.copy()
    grouped_df['end_date'] = pd.to_datetime(grouped_df['end_date'], errors='coerce').dt.normalize()
    grouped_df = grouped_df[grouped_df['end_date'].notna()].copy()
    if grouped_df.empty:
        return None

    scope_column = IMPORTER_ORIGIN_LEVEL_TO_SCOPE.get(origin_level or 'origin_shipping_region', 'origin_shipping_region')
    parent_cols = []
    if scope_column != 'origin_country':
        if scope_column not in grouped_df.columns:
            return None
        parent_cols = [scope_column]

    current_timestamp = pd.Timestamp(
        as_of_date or datetime.now()
    ).normalize()
    current_month = current_timestamp.to_period('M')
    start_month = current_month - (lookback_months - 1)
    lookback_df = grouped_df[grouped_df['end_date'].dt.to_period('M') >= start_month].copy()
    if lookback_df.empty:
        return None

    lookback_df['__month_period'] = lookback_df['end_date'].dt.to_period('M')
    monthly_totals = (
        lookback_df
        .groupby(parent_cols + ['origin_country', '__month_period'], dropna=False)['cargo_mcm']
        .sum()
        .reset_index()
    )
    if monthly_totals.empty:
        return None

    monthly_totals['__days'] = monthly_totals['__month_period'].apply(
        lambda month_period: (
            current_timestamp.day if month_period == current_month else month_period.days_in_month
        )
    )
    monthly_totals['__monthly_mcmd'] = (
        monthly_totals['cargo_mcm'] / monthly_totals['__days']
    ).fillna(0)

    pair_cols = parent_cols + ['origin_country']
    max_monthly_by_pair = (
        monthly_totals
        .groupby(pair_cols, dropna=False)['__monthly_mcmd']
        .max()
        .reset_index()
    )
    all_pairs = grouped_df[pair_cols].drop_duplicates()
    pair_threshold_df = all_pairs.merge(max_monthly_by_pair, on=pair_cols, how='left')
    pair_threshold_df['__monthly_mcmd'] = pair_threshold_df['__monthly_mcmd'].fillna(0)
    small_pairs = pair_threshold_df[pair_threshold_df['__monthly_mcmd'] <= threshold_mcmd][pair_cols].copy()
    if small_pairs.empty:
        return None

    return {
        'pair_cols': pair_cols,
        'small_pairs': small_pairs.to_dict('records'),
    }


def _apply_small_importer_origin_grouping(
    scoped_trades_df,
    grouping_config,
):
    if (
        scoped_trades_df is None
        or scoped_trades_df.empty
        or not isinstance(grouping_config, dict)
    ):
        return scoped_trades_df
    pair_cols = list(grouping_config.get('pair_cols') or [])
    small_pairs = pd.DataFrame(
        grouping_config.get('small_pairs') or []
    )
    if (
        'origin_country' not in pair_cols
        or small_pairs.empty
        or not set(pair_cols).issubset(scoped_trades_df.columns)
    ):
        return scoped_trades_df

    grouped_df = scoped_trades_df.copy()
    small_pairs['__group_small_country'] = True
    grouped_df = grouped_df.merge(small_pairs, on=pair_cols, how='left')
    grouped_df['__group_small_country'] = grouped_df['__group_small_country'].eq(True)
    grouped_df.loc[grouped_df['__group_small_country'], 'origin_country'] = 'Rest of countries'
    return grouped_df.drop(columns='__group_small_country')


def group_small_importer_origin_countries(
    scoped_trades_df,
    origin_level='origin_shipping_region',
    threshold_mcmd=10,
    lookback_months=24,
    as_of_date=None,
    grouping_config=None,
    return_grouping_config=False,
):
    """Apply one current-vintage small-origin taxonomy to importer facts."""
    if grouping_config is None:
        grouping_config = _build_small_importer_origin_grouping(
            scoped_trades_df,
            origin_level=origin_level,
            threshold_mcmd=threshold_mcmd,
            lookback_months=lookback_months,
            as_of_date=as_of_date,
        )
    grouped_df = _apply_small_importer_origin_grouping(
        scoped_trades_df,
        grouping_config,
    )
    if return_grouping_config:
        return grouped_df, grouping_config
    return grouped_df


def _get_small_importer_destination_countries(scoped_trades_df, importer_entities,
                                              threshold_mcmd=10, lookback_months=24,
                                              as_of_date=None):
    """Return importer countries whose delivered import volume stays below the small-country threshold."""
    if scoped_trades_df is None or scoped_trades_df.empty:
        return set()
    required_columns = {'destination_country_name', 'origin_country', 'end_date', 'cargo_mcm'}
    if not required_columns.issubset(scoped_trades_df.columns):
        return set()

    country_entities = [
        entity for entity in (importer_entities or [])
        if (
            not entity.get('is_global') and
            not entity.get('is_rest') and
            len(entity.get('destination_countries') or []) == 1
        )
    ]
    destination_countries = sorted({
        country
        for entity in country_entities
        for country in entity.get('destination_countries', [])
        if country
    })
    if not destination_countries:
        return set()

    grouped_df = scoped_trades_df.copy()
    grouped_df['end_date'] = pd.to_datetime(grouped_df['end_date'], errors='coerce').dt.normalize()
    grouped_df = grouped_df[grouped_df['end_date'].notna()].copy()
    if grouped_df.empty:
        return set(destination_countries)

    grouped_df['destination_country_name'] = grouped_df['destination_country_name'].fillna('Unknown').astype(str).str.strip()
    grouped_df['origin_country'] = grouped_df['origin_country'].fillna('Unknown').astype(str).str.strip()
    grouped_df = grouped_df[
        grouped_df['destination_country_name'].isin(destination_countries) &
        (grouped_df['origin_country'] != grouped_df['destination_country_name'])
    ].copy()

    current_timestamp = pd.Timestamp(
        as_of_date or datetime.now()
    ).normalize()
    current_month = current_timestamp.to_period('M')
    start_month = current_month - (lookback_months - 1)
    lookback_df = grouped_df[grouped_df['end_date'].dt.to_period('M') >= start_month].copy()

    all_destinations_df = pd.DataFrame({'destination_country_name': destination_countries})
    if lookback_df.empty:
        return set(destination_countries)

    lookback_df['__month_period'] = lookback_df['end_date'].dt.to_period('M')
    monthly_totals = (
        lookback_df
        .groupby(['destination_country_name', '__month_period'], dropna=False)['cargo_mcm']
        .sum()
        .reset_index()
    )
    if monthly_totals.empty:
        return set(destination_countries)

    monthly_totals['__days'] = monthly_totals['__month_period'].apply(
        lambda month_period: (
            current_timestamp.day if month_period == current_month else month_period.days_in_month
        )
    )
    monthly_totals['__monthly_mcmd'] = (
        monthly_totals['cargo_mcm'] / monthly_totals['__days']
    ).fillna(0)

    max_monthly_by_destination = (
        monthly_totals
        .groupby('destination_country_name', dropna=False)['__monthly_mcmd']
        .max()
        .reset_index()
    )
    threshold_df = all_destinations_df.merge(
        max_monthly_by_destination,
        on='destination_country_name',
        how='left'
    )
    threshold_df['__monthly_mcmd'] = threshold_df['__monthly_mcmd'].fillna(0)
    small_destinations = threshold_df[
        threshold_df['__monthly_mcmd'] <= threshold_mcmd
    ]['destination_country_name']
    return set(small_destinations.tolist())


def _group_small_importer_entities(importer_entities, scoped_trades_df, classification_mode='Country',
                                   threshold_mcmd=10, lookback_months=24,
                                   as_of_date=None):
    """Collapse low-volume importer countries into Rest of Importers for the grouped table payload."""
    if classification_mode != 'Country':
        return importer_entities or []

    importer_entities = list(importer_entities or [])
    small_destinations = _get_small_importer_destination_countries(
        scoped_trades_df,
        importer_entities,
        threshold_mcmd=threshold_mcmd,
        lookback_months=lookback_months,
        as_of_date=as_of_date,
    )
    if not small_destinations:
        return importer_entities

    grouped_entities = []
    rest_destination_countries = []
    rest_avg_30d_mcmd = 0.0

    for entity in importer_entities:
        destination_countries = entity.get('destination_countries') or []
        if entity.get('is_global'):
            grouped_entities.append(entity)
            continue

        if entity.get('is_rest'):
            rest_destination_countries.extend(destination_countries)
            rest_avg_30d_mcmd += float(entity.get('avg_30d_mcmd') or 0)
            continue

        if destination_countries and set(destination_countries).issubset(small_destinations):
            rest_destination_countries.extend(destination_countries)
            rest_avg_30d_mcmd += float(entity.get('avg_30d_mcmd') or 0)
            continue

        grouped_entities.append(entity)

    rest_destination_countries = sorted(set(rest_destination_countries))
    if rest_destination_countries:
        grouped_entities.append({
            'key': '__small_importers__',
            'label': IMPORTER_REST_OF_IMPORTERS_LABEL,
            'destination_countries': rest_destination_countries,
            'avg_30d_mcmd': round(rest_avg_30d_mcmd, 1),
            'is_rest': True,
        })

    return grouped_entities


def _empty_importer_period_payload(grouping_mode='group_small_countries'):
    grouping_mode = _normalize_importer_period_origin_grouping(grouping_mode)
    return {
        'format': IMPORTERS_PERIOD_PAYLOAD_FORMAT,
        'active_grouping_mode': grouping_mode,
        'show_all': [],
        'group_small_countries': [],
        'snapshot_comparison': {
            'status': 'unavailable',
            'current_snapshot': None,
            'baseline_snapshot': None,
            'business_day_gap': None,
        },
    }


def _resolve_importer_period_payload(period_payload, grouping_mode='group_small_countries'):
    """Resolve the grouped or ungrouped importer period payload from the store data."""
    grouping_mode = _normalize_importer_period_origin_grouping(grouping_mode)
    if isinstance(period_payload, list):
        return period_payload
    if not isinstance(period_payload, dict):
        return []

    selected_payload = period_payload.get(grouping_mode)
    if isinstance(selected_payload, list):
        return selected_payload

    fallback_payload = period_payload.get('show_all')
    if isinstance(fallback_payload, list):
        return fallback_payload
    return []


def _build_importer_period_snapshot_comparison(
    source_state,
    baseline_data_available,
):
    source_state = (
        source_state if isinstance(source_state, dict) else {}
    )
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


def _build_importer_pbd_rolling_summary(
    scoped_trades_df,
    origin_level,
    as_of_date,
):
    """Build only the exact 30D/7D levels needed for PBD comparison."""
    summary_scope_df = _prepare_importer_summary_scope_df(
        scoped_trades_df,
        origin_level or 'origin_shipping_region',
    )
    rolling_df = _build_importer_rolling_windows_pivot(
        summary_scope_df,
        rolling_window_days=30,
        current_date=as_of_date,
    )
    return rolling_df[
        [
            column_name
            for column_name in (
                'continent',
                'country',
                '30D',
                '7D',
            )
            if column_name in rolling_df.columns
        ]
    ].copy()


def _merge_importer_period_pbd_windows(
    summary_df,
    current_pbd_df,
    baseline_pbd_df,
    baseline_available,
):
    """Outer-join current/PBD rolling levels so additions/removals survive."""
    id_cols = ['continent', 'country']
    merged = (
        summary_df.copy()
        if summary_df is not None
        else pd.DataFrame()
    )
    if not baseline_available:
        if merged.empty:
            return merged
        for column_name in (
            *IMPORTER_PERIOD_PBD_CURRENT_COLUMNS,
            *IMPORTER_PERIOD_PBD_REFERENCE_COLUMNS,
            *IMPORTER_PERIOD_PBD_DELTA_COLUMNS,
        ):
            merged[column_name] = np.nan
        return merged

    if merged.empty:
        merged = pd.DataFrame(columns=id_cols)
    original_numeric_columns = [
        column_name
        for column_name in merged.columns
        if column_name not in id_cols
    ]

    def _rolling_values(frame, rename_map):
        frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        columns = [
            column_name
            for column_name in [*id_cols, '30D', '7D']
            if column_name in frame.columns
        ]
        if not set(id_cols).issubset(columns):
            return pd.DataFrame(columns=[*id_cols, *rename_map.values()])
        result = frame[columns].copy().rename(columns=rename_map)
        for target_column in rename_map.values():
            if target_column not in result.columns:
                result[target_column] = 0.0
        return result[[*id_cols, *rename_map.values()]]

    current_values = _rolling_values(
        current_pbd_df,
        {
            '30D': '30D_PBD_CURRENT',
            '7D': '7D_PBD_CURRENT',
        },
    )
    baseline_values = _rolling_values(
        baseline_pbd_df,
        {
            '30D': '30D_PBD',
            '7D': '7D_PBD',
        },
    )
    merged = merged.merge(current_values, on=id_cols, how='outer')
    merged = merged.merge(baseline_values, on=id_cols, how='outer')
    merged = merged.copy()

    pbd_level_columns = [
        *original_numeric_columns,
        *IMPORTER_PERIOD_PBD_CURRENT_COLUMNS,
        *IMPORTER_PERIOD_PBD_REFERENCE_COLUMNS,
    ]
    missing_level_columns = [
        column_name
        for column_name in pbd_level_columns
        if column_name not in merged.columns
    ]
    if missing_level_columns:
        merged = pd.concat(
            [
                merged,
                pd.DataFrame(
                    0.0,
                    index=merged.index,
                    columns=missing_level_columns,
                ),
            ],
            axis=1,
        )
    merged[pbd_level_columns] = (
        merged[pbd_level_columns]
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0.0)
    )
    merged = merged.copy()

    merged['Δ 30D vs PBD'] = (
        merged['30D_PBD_CURRENT'] - merged['30D_PBD']
    ).round(1)
    merged['Δ 7D vs PBD'] = (
        merged['7D_PBD_CURRENT'] - merged['7D_PBD']
    ).round(1)
    return merged


def _build_period_payload(importer_entities, classification_mode, origin_level,
                          grouping_mode='group_small_countries',
                          rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
                          source_state=None):
    """Build the raw per-importer period-analysis payload."""
    grouping_mode = _normalize_importer_period_origin_grouping(grouping_mode)
    source_state = _normalize_importers_source_state(source_state)
    has_snapshot_pair_contract = (
        source_state.get('format') == IMPORTERS_SOURCE_STATE_FORMAT
        and isinstance(source_state.get('current_snapshot'), dict)
    )
    current_snapshot = source_state.get('current_snapshot') or {}
    baseline_snapshot = source_state.get('baseline_snapshot') or {}
    current_as_of_date = (
        current_snapshot.get('snapshot_date_utc')
        or source_state.get('as_of_date')
        or datetime.now().date().isoformat()
    )
    baseline_as_of_date = baseline_snapshot.get(
        'snapshot_date_utc'
    )
    all_destination_countries = sorted({
        country
        for entity in importer_entities or []
        for country in entity.get('destination_countries', [])
    })
    scoped_trades_df = pd.DataFrame()
    selected_destination_aggregation = (
        _classification_mode_to_destination_aggregation(
            classification_mode
        )
    )
    if all_destination_countries:
        current_query_kwargs = {
            'delivered_only': True,
            'include_destination_context': True,
            'selected_destination_aggregation': (
                selected_destination_aggregation
            ),
        }
        if has_snapshot_pair_contract:
            current_query_kwargs.update({
                'snapshot_timestamp_utc': current_snapshot.get(
                    'snapshot_timestamp_utc'
                ),
                'max_end_date': current_as_of_date,
            })
        scoped_trades_df = _fetch_importer_scoped_trades(
            engine,
            all_destination_countries,
            **current_query_kwargs,
        )

    baseline_scoped_trades_df = pd.DataFrame()
    baseline_candidate_available = (
        has_snapshot_pair_contract
        and source_state.get('baseline_status') in {'exact', 'fallback'}
        and baseline_snapshot.get('snapshot_timestamp_utc')
        and baseline_as_of_date
    )
    if all_destination_countries and baseline_candidate_available:
        baseline_start_date = (
            pd.Timestamp(baseline_as_of_date)
            - pd.Timedelta(days=29)
        ).date()
        try:
            baseline_scoped_trades_df = _fetch_importer_scoped_trades(
                engine,
                all_destination_countries,
                min_end_date=baseline_start_date,
                delivered_only=True,
                include_destination_context=True,
                selected_destination_aggregation=(
                    selected_destination_aggregation
                ),
                snapshot_timestamp_utc=baseline_snapshot[
                    'snapshot_timestamp_utc'
                ],
                max_end_date=baseline_as_of_date,
            )
        except Exception:
            baseline_scoped_trades_df = pd.DataFrame()
    baseline_data_available = (
        baseline_candidate_available
        and not baseline_scoped_trades_df.empty
    )
    snapshot_comparison = _build_importer_period_snapshot_comparison(
        source_state,
        baseline_data_available,
    )

    def _build_payload_variant(entities, group_small_origins=False):
        payload = []
        for entity in entities:
            try:
                entity_scoped_df = _filter_scoped_trades_for_entity(
                    scoped_trades_df,
                    entity,
                    classification_mode
                )
                baseline_entity_scoped_df = _filter_scoped_trades_for_entity(
                    baseline_scoped_trades_df,
                    entity,
                    classification_mode,
                )
                if group_small_origins:
                    (
                        entity_scoped_df,
                        origin_grouping_config,
                    ) = group_small_importer_origin_countries(
                        entity_scoped_df,
                        origin_level or 'origin_shipping_region',
                        as_of_date=current_as_of_date,
                        return_grouping_config=True,
                    )
                    baseline_entity_scoped_df = (
                        group_small_importer_origin_countries(
                            baseline_entity_scoped_df,
                            origin_level or 'origin_shipping_region',
                            grouping_config=origin_grouping_config,
                        )
                    )
                summary_df = build_importer_origin_summary_from_scoped_trades(
                    entity_scoped_df,
                    rolling_window_days=rolling_avg_days,
                    origin_level=origin_level or 'origin_shipping_region',
                    quarter_count=IMPORTER_PERIOD_MAX_QUARTER_COUNT + 4,
                    month_count=IMPORTER_PERIOD_MAX_MONTH_COUNT + 12,
                    week_count=IMPORTER_PERIOD_MAX_WEEK_COUNT + 53,
                    include_comparison_reference_columns=True,
                    current_date=current_as_of_date,
                )
                if has_snapshot_pair_contract:
                    current_pbd_df = _build_importer_pbd_rolling_summary(
                        entity_scoped_df,
                        origin_level,
                        current_as_of_date,
                    )
                    baseline_pbd_df = (
                        _build_importer_pbd_rolling_summary(
                            baseline_entity_scoped_df,
                            origin_level,
                            baseline_as_of_date,
                        )
                        if baseline_data_available
                        else pd.DataFrame()
                    )
                    summary_df = _merge_importer_period_pbd_windows(
                        summary_df,
                        current_pbd_df,
                        baseline_pbd_df,
                        baseline_available=baseline_data_available,
                    )
            except Exception:
                summary_df = pd.DataFrame()
            payload.append({
                'label': entity['label'],
                'key': entity['key'],
                'records': summary_df.to_dict('records') if not summary_df.empty else []
            })
        return payload

    payload = {
        'format': IMPORTERS_PERIOD_PAYLOAD_FORMAT,
        'active_grouping_mode': grouping_mode,
        'show_all': [],
        'group_small_countries': [],
        'snapshot_comparison': snapshot_comparison,
    }
    if grouping_mode == 'show_all':
        payload['show_all'] = _build_payload_variant(importer_entities, group_small_origins=False)
        return payload

    grouped_importer_entities = _group_small_importer_entities(
        importer_entities,
        scoped_trades_df,
        classification_mode,
        as_of_date=current_as_of_date,
    )
    payload['group_small_countries'] = _build_payload_variant(
        grouped_importer_entities,
        group_small_origins=True
    )
    return payload


def _importer_chart_year_token(year):
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


def _importer_chart_year_sort_key(year):
    try:
        return (0, int(year))
    except (TypeError, ValueError):
        return (1, str(year))


def _get_importer_chart_years_from_records(records):
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

        token = _importer_chart_year_token(record.get('year'))
        if token:
            years.add(token)
    return sorted(years, key=_importer_chart_year_sort_key)


def _get_importer_chart_available_years(charts_data):
    years = set()
    if isinstance(charts_data, dict):
        for records in charts_data.values():
            years.update(_get_importer_chart_years_from_records(records))

    start_year = pd.Timestamp(IMPORTER_CHART_DISPLAY_START_DATE).year
    end_year = (datetime.now() + timedelta(days=14)).year
    year_floor = {str(year) for year in range(start_year, max(start_year, end_year) + 1)}
    return sorted(years | year_floor, key=_importer_chart_year_sort_key)


def _default_importer_chart_selected_years(available_years):
    latest_years = available_years[-IMPORTER_CHART_DEFAULT_SELECTED_YEAR_COUNT:]
    selected_years = [
        year for year in latest_years
        if year not in IMPORTER_CHART_DEFAULT_DESELECTED_YEARS
    ]
    return selected_years or latest_years


def _normalize_importer_chart_selected_years(selected_years, available_years, use_default=True):
    available_set = set(available_years)
    normalized = [
        token for token in (_importer_chart_year_token(year) for year in (selected_years or []))
        if token in available_set
    ]
    if normalized or not use_default:
        return sorted(set(normalized), key=_importer_chart_year_sort_key)
    return _default_importer_chart_selected_years(available_years)


def _normalise_importer_chart_plot_dates(date_series):
    month_day_text = pd.to_datetime(date_series, errors='coerce').dt.strftime('%m-%d')
    return pd.to_datetime(
        month_day_text.map(
            lambda value: f'{IMPORTER_CHART_ANCHOR_YEAR}-{value}' if pd.notna(value) else None
        ),
        errors='coerce'
    )


def _get_importer_chart_color_map(years):
    years = sorted(years or [], key=_importer_chart_year_sort_key)
    if not years:
        return {}
    if len(years) <= len(IMPORTER_CHART_COLOR_SEQUENCE):
        visible_colors = IMPORTER_CHART_COLOR_SEQUENCE[-len(years):]
    else:
        repeats = (len(years) // len(IMPORTER_CHART_COLOR_SEQUENCE)) + 1
        visible_colors = (IMPORTER_CHART_COLOR_SEQUENCE * repeats)[-len(years):]
    return {
        year: visible_colors[idx]
        for idx, year in enumerate(years)
    }


def _get_importer_chart_range_years(focus_year, available_years):
    try:
        focus_year_number = int(focus_year)
    except (TypeError, ValueError):
        return []

    previous_years = []
    for year in sorted(available_years, key=_importer_chart_year_sort_key):
        try:
            if int(year) < focus_year_number:
                previous_years.append(year)
        except (TypeError, ValueError):
            continue
    return previous_years[-IMPORTER_CHART_RANGE_LOOKBACK_YEARS:]


def _add_importer_chart_range_band(
    fig,
    df,
    focus_year,
    available_years,
    vol_label,
    volume_metric='mcm_d',
):
    range_years = _get_importer_chart_range_years(focus_year, available_years)
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
    plotly_number_format = _get_importer_volume_metric_plotly_number_format(
        volume_metric
    )
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
        fillcolor=IMPORTER_CHART_RANGE_FILL,
        customdata=range_df[['range_min']].to_numpy(),
        text=range_df['month_day'],
        hovertemplate=(
            f'<b>{years_label} range</b> | '
            '%{text} | '
            f'%{{customdata[0]:{plotly_number_format}}}-'
            f'%{{y:{plotly_number_format}}} {vol_label}<extra></extra>'
        ),
        showlegend=False
    ))


def _prepare_importer_demand_chart_dataframe(
    data,
    volume_metric,
    rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS
):
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty or not {'date', 'year', 'rolling_avg'}.issubset(df.columns):
        return pd.DataFrame()

    rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
    df = convert_volume_metric_dataframe(
        df,
        volume_metric,
        columns=['rolling_avg'],
        period_days=rolling_avg_days,
    )
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    if 'month_day' not in df.columns:
        df['month_day'] = df['date'].dt.strftime('%b %d')

    df['_year_token'] = df['year'].apply(_importer_chart_year_token)
    df['plot_date'] = _normalise_importer_chart_plot_dates(df['date'])
    df = df[
        (df['date'] >= IMPORTER_CHART_DISPLAY_START_DATE)
        & df['_year_token'].notna()
        & df['plot_date'].notna()
    ].copy()
    return df


def _prepare_importer_origin_chart_dataframe(
    data,
    chart_type,
    volume_metric,
    rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
):
    if not data:
        return pd.DataFrame(), 'rolling_avg'

    df = pd.DataFrame(data)
    if df.empty or not {'date', 'year', 'continent_origin', 'rolling_avg'}.issubset(df.columns):
        return pd.DataFrame(), 'rolling_avg'

    metric_column = 'percentage' if chart_type == 'percentage' and 'percentage' in df.columns else 'rolling_avg'
    if metric_column == 'rolling_avg':
        df = convert_volume_metric_dataframe(
            df,
            volume_metric,
            columns=['rolling_avg'],
            period_days=normalize_importer_rolling_avg_days(rolling_avg_days),
        )

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    if df.empty:
        return pd.DataFrame(), metric_column

    if 'month_day' not in df.columns:
        df['month_day'] = df['date'].dt.strftime('%b %d')
    df['_year_token'] = df['year'].apply(_importer_chart_year_token)
    df['plot_date'] = _normalise_importer_chart_plot_dates(df['date'])
    df = df[
        (df['date'] >= IMPORTER_CHART_DISPLAY_START_DATE)
        & df['_year_token'].notna()
        & df['plot_date'].notna()
    ].copy()
    return df, metric_column


def _empty_importer_chart_figure(message, height=328):
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


def _apply_importer_chart_layout(
    fig,
    y_title,
    yaxis_range=None,
    show_legend=False,
    yaxis_tickformat=None,
):
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
    if yaxis_tickformat is not None:
        yaxis_config['tickformat'] = yaxis_tickformat

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
                pd.Timestamp(year=IMPORTER_CHART_ANCHOR_YEAR, month=1, day=1),
                pd.Timestamp(year=IMPORTER_CHART_ANCHOR_YEAR, month=12, day=31)
            ],
            showspikes=True,
            spikemode='across',
            spikecolor='rgba(15, 23, 42, 0.18)',
            spikethickness=1
        ),
        yaxis=yaxis_config,
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
            font=dict(size=10, color='#475569'),
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


def _get_importer_chart_previous_year_token(focus_year, available_years, active_years):
    try:
        previous_year = str(int(focus_year) - 1)
        if previous_year in set(available_years):
            return previous_year
    except (TypeError, ValueError):
        pass

    if len(active_years) > 1:
        return active_years[-2]
    return None


def get_importer_demand_chart_header_metrics(
    data,
    volume_metric='mcm_d',
    selected_years=None,
    rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS
):
    df = _prepare_importer_demand_chart_dataframe(data, volume_metric, rolling_avg_days)
    return _get_importer_demand_chart_header_metrics_from_df(
        df,
        selected_years,
    )


def _get_importer_demand_chart_header_metrics_from_df(
    df,
    selected_years=None,
):
    if df.empty:
        return None

    available_years = sorted(
        [year for year in df['_year_token'].dropna().unique()],
        key=_importer_chart_year_sort_key
    )
    active_years = _normalize_importer_chart_selected_years(
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

    previous_year = _get_importer_chart_previous_year_token(focus_year, available_years, active_years)
    previous_value = None
    delta_value = None
    delta_pct = None
    mom_delta_value = None
    mom_delta_pct = None

    if previous_year:
        previous_data = df[df['_year_token'] == previous_year].dropna(subset=['rolling_avg']).copy()
        if not previous_data.empty:
            previous_candidates = previous_data.sort_values('plot_date')
            previous_candidates = previous_candidates[previous_candidates['plot_date'] <= point['plot_date']]
            if not previous_candidates.empty:
                previous_value = previous_candidates.tail(1).iloc[0]['rolling_avg']
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
        mom_value = mom_candidates.sort_values('plot_date').tail(1).iloc[0]['rolling_avg']
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
        'mom_delta_value': mom_delta_value,
        'mom_delta_pct': mom_delta_pct,
    }


def create_importer_demand_chart(
    data,
    volume_metric='mcm_d',
    selected_years=None,
    rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS
):
    vol_label = get_volume_metric_info(volume_metric)['label']
    df = _prepare_importer_demand_chart_dataframe(data, volume_metric, rolling_avg_days)
    return _create_importer_demand_chart_from_df(
        df,
        vol_label,
        selected_years,
        volume_metric,
    )


def _create_importer_demand_chart_from_df(
    df,
    vol_label,
    selected_years=None,
    volume_metric='mcm_d',
):
    if df.empty:
        return _empty_importer_chart_figure('No data available.')

    available_years = sorted(
        [year for year in df['_year_token'].dropna().unique()],
        key=_importer_chart_year_sort_key
    )
    active_years = _normalize_importer_chart_selected_years(
        selected_years,
        available_years,
        use_default=selected_years is None
    )
    if not active_years:
        return _empty_importer_chart_figure('Select a year above.')

    active_df = df[df['_year_token'].isin(active_years)].copy()
    if active_df.empty:
        return _empty_importer_chart_figure('No data for the selected years.')

    fig = go.Figure()
    years = sorted(active_years, key=_importer_chart_year_sort_key)
    focus_year = years[-1]
    color_by_year = _get_importer_chart_color_map(available_years)
    plotly_number_format = _get_importer_volume_metric_plotly_number_format(
        volume_metric
    )
    _add_importer_chart_range_band(
        fig,
        df,
        focus_year,
        available_years,
        vol_label,
        volume_metric,
    )

    for year in years:
        year_data = active_df[active_df['_year_token'] == year].dropna(subset=['plot_date']).sort_values('plot_date')
        if year_data.empty:
            continue

        is_focus_year = year == focus_year
        line_color = color_by_year.get(year, '#0f4c81')
        line_width = 2.2 if is_focus_year else 1.15
        line_opacity = 0.95 if is_focus_year else 0.52
        forecast_mask = (
            year_data['is_forecast'].astype(bool)
            if 'is_forecast' in year_data.columns
            else pd.Series(False, index=year_data.index)
        )
        historical_data = year_data[~forecast_mask]
        forecast_data = year_data[forecast_mask]

        if not historical_data.empty:
            fig.add_trace(go.Scatter(
                x=historical_data['plot_date'],
                y=historical_data['rolling_avg'],
                mode='lines',
                name=str(year),
                line=dict(color=line_color, width=line_width, dash='solid'),
                opacity=line_opacity,
                hovertemplate=(
                    f'<b>{year}</b> | '
                    '%{text} | '
                    f'%{{y:{plotly_number_format}}} {vol_label}<extra></extra>'
                ),
                text=historical_data['month_day'],
                showlegend=False
            ))

        if not forecast_data.empty:
            connect_data = pd.concat([historical_data.tail(1), forecast_data]) if not historical_data.empty else forecast_data
            fig.add_trace(go.Scatter(
                x=connect_data['plot_date'],
                y=connect_data['rolling_avg'],
                mode='lines',
                name=f'{year} forecast',
                line=dict(color=line_color, width=line_width, dash=IMPORTER_CHART_FORECAST_DASH),
                opacity=0.76 if is_focus_year else 0.36,
                hovertemplate=(
                    f'<b>{year} forecast</b> | '
                    '%{text} | '
                    f'%{{y:{plotly_number_format}}} {vol_label}<extra></extra>'
                ),
                text=connect_data['month_day'],
                showlegend=False
            ))

        if is_focus_year:
            latest_actual_data = historical_data if not historical_data.empty else year_data
            latest_point = latest_actual_data.dropna(subset=['rolling_avg']).tail(1)
            if not latest_point.empty:
                point = latest_point.iloc[0]
                fig.add_trace(go.Scatter(
                    x=[point['plot_date']],
                    y=[point['rolling_avg']],
                    mode='markers',
                    marker=dict(color=line_color, size=5.5, line=dict(color='#ffffff', width=1.5)),
                    hoverinfo='skip',
                    showlegend=False
                ))

    return _apply_importer_chart_layout(
        fig,
        vol_label,
        show_legend=False,
        yaxis_tickformat=plotly_number_format,
    )


def _origin_chart_line_style(year, current_year, is_forecast=False):
    try:
        year_number = int(year)
    except (TypeError, ValueError):
        year_number = current_year
    is_current = year_number == current_year
    return {
        'width': 2.25 if is_current else 1.12,
        'opacity': (0.9 if is_current else 0.42) if not is_forecast else (0.68 if is_current else 0.28),
        'dash': IMPORTER_CHART_FORECAST_DASH if is_forecast else 'solid',
    }


def create_importer_origin_continent_chart(
    data,
    chart_type='absolute',
    volume_metric='mcm_d',
    selected_years=None,
    rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
):
    vol_label = get_volume_metric_info(volume_metric)['label']
    df, metric_column = _prepare_importer_origin_chart_dataframe(
        data,
        chart_type,
        volume_metric,
        rolling_avg_days,
    )
    return _create_importer_origin_continent_chart_from_df(
        df,
        metric_column,
        vol_label,
        selected_years,
        volume_metric,
    )


def _create_importer_origin_continent_chart_from_df(
    df,
    metric_column,
    vol_label,
    selected_years=None,
    volume_metric='mcm_d',
):
    if df.empty:
        return _empty_importer_chart_figure('No data available.')

    available_years = sorted(
        [year for year in df['_year_token'].dropna().unique()],
        key=_importer_chart_year_sort_key
    )
    active_years = _normalize_importer_chart_selected_years(
        selected_years,
        available_years,
        use_default=selected_years is None
    )
    if not active_years:
        return _empty_importer_chart_figure('Select a year above.')

    df = df[df['_year_token'].isin(active_years)].copy()
    if df.empty:
        return _empty_importer_chart_figure('No data for the selected years.')

    fig = go.Figure()
    years = sorted(active_years, key=_importer_chart_year_sort_key)
    current_year = int(years[-1])
    legend_shown = set()
    plotly_number_format = (
        ',.1f'
        if metric_column == 'percentage'
        else _get_importer_volume_metric_plotly_number_format(volume_metric)
    )

    for continent in sorted(df['continent_origin'].dropna().unique()):
        continent_df = df[df['continent_origin'] == continent]
        color = ORIGIN_CONTINENT_CHART_COLOR_MAP.get(continent, '#64748b')
        for year in years:
            year_df = continent_df[continent_df['_year_token'] == year].dropna(subset=['plot_date']).sort_values('plot_date')
            if year_df.empty:
                continue
            forecast_mask = (
                year_df['is_forecast'].astype(bool)
                if 'is_forecast' in year_df.columns
                else pd.Series(False, index=year_df.index)
            )
            historical_data = year_df[~forecast_mask]
            forecast_data = year_df[forecast_mask]
            show_legend = continent not in legend_shown
            if show_legend:
                legend_shown.add(continent)

            historical_style = _origin_chart_line_style(year, current_year)
            if not historical_data.empty:
                fig.add_trace(go.Scatter(
                    x=historical_data['plot_date'],
                    y=historical_data[metric_column],
                    mode='lines',
                    name=continent if show_legend else None,
                    legendgroup=continent,
                    line=dict(color=color, width=historical_style['width'], dash=historical_style['dash']),
                    opacity=historical_style['opacity'],
                    hovertemplate=(
                        f'<b>{continent}</b> | {year} | '
                        '%{text} | '
                        + (
                            f'%{{y:{plotly_number_format}}}%<extra></extra>'
                            if metric_column == 'percentage'
                            else f'%{{y:{plotly_number_format}}} {vol_label}<extra></extra>'
                        )
                    ),
                    text=historical_data['month_day'],
                    showlegend=show_legend
                ))

            if not forecast_data.empty:
                connect_data = pd.concat([historical_data.tail(1), forecast_data]) if not historical_data.empty else forecast_data
                forecast_style = _origin_chart_line_style(year, current_year, is_forecast=True)
                fig.add_trace(go.Scatter(
                    x=connect_data['plot_date'],
                    y=connect_data[metric_column],
                    mode='lines',
                    name=None,
                    legendgroup=continent,
                    line=dict(color=color, width=forecast_style['width'], dash=forecast_style['dash']),
                    opacity=forecast_style['opacity'],
                    hovertemplate=(
                        f'<b>{continent}</b> | {year} forecast | '
                        '%{text} | '
                        + (
                            f'%{{y:{plotly_number_format}}}%<extra></extra>'
                            if metric_column == 'percentage'
                            else f'%{{y:{plotly_number_format}}} {vol_label}<extra></extra>'
                        )
                    ),
                    text=connect_data['month_day'],
                    showlegend=False
                ))

    y_title = '%' if metric_column == 'percentage' else vol_label
    yaxis_range = [0, 100] if metric_column == 'percentage' else None
    return _apply_importer_chart_layout(
        fig,
        y_title,
        yaxis_range=yaxis_range,
        show_legend=True,
        yaxis_tickformat=plotly_number_format,
    )


def _format_importer_chart_current_value(
    metrics,
    vol_label,
    volume_metric='mcm_d',
):
    if not metrics or metrics.get('latest_value') is None:
        return None

    latest_label = metrics.get('latest_label') or metrics.get('focus_year') or ''
    precision = _get_importer_volume_metric_display_precision(volume_metric)
    latest_value = _round_importer_volume_metric_display_value(
        metrics['latest_value'],
        volume_metric,
    )
    return f"{latest_label}: {latest_value:,.{precision}f} {vol_label}"


def _build_importer_chart_delta_pill(
    label,
    delta_value,
    delta_pct,
    volume_metric='mcm_d',
):
    if delta_value is None or pd.isna(delta_value):
        return html.Span(f'{label} n/a', className='importer-rolling-delta-pill importer-rolling-delta-neutral')

    precision = _get_importer_volume_metric_display_precision(volume_metric)
    rounded_delta = _round_importer_volume_metric_display_value(
        delta_value,
        volume_metric,
    )
    direction_class = 'importer-rolling-delta-neutral'
    if rounded_delta > 0:
        direction_class = 'importer-rolling-delta-positive'
    elif rounded_delta < 0:
        direction_class = 'importer-rolling-delta-negative'

    sign = '+' if rounded_delta > 0 else ''
    pct_text = ''
    if delta_pct is not None and pd.notna(delta_pct):
        rounded_pct = int(round(float(delta_pct)))
        rounded_pct = 0 if rounded_pct == 0 else rounded_pct
        pct_sign = '+' if rounded_pct > 0 else ''
        pct_text = f" ({pct_sign}{rounded_pct}%)"

    return html.Span(
        [
            html.Span(label, className='importer-rolling-delta-label'),
            html.Span(
                f"{sign}{rounded_delta:,.{precision}f}{pct_text}"
            )
        ],
        className=f'importer-rolling-delta-pill {direction_class}'
    )


def _build_importer_chart_delta_indicators(
    metrics,
    volume_metric='mcm_d',
):
    return html.Div(
        [
            _build_importer_chart_delta_pill(
                'MoM',
                metrics.get('mom_delta_value') if metrics else None,
                metrics.get('mom_delta_pct') if metrics else None,
                volume_metric,
            ),
            _build_importer_chart_delta_pill(
                'YoY',
                metrics.get('delta_value') if metrics else None,
                metrics.get('delta_pct') if metrics else None,
                volume_metric,
            )
        ],
        className='importer-rolling-delta-group'
    )


def _format_origin_kpi_value(
    value,
    chart_type,
    volume_metric='mcm_d',
    is_delta=False,
):
    if value is None or pd.isna(value):
        return 'n/a'

    precision = (
        1
        if chart_type == 'percentage'
        else _get_importer_volume_metric_display_precision(volume_metric)
    )
    rounded_value = round(float(value), precision)
    rounded_value = 0.0 if rounded_value == 0 else rounded_value
    sign = '+' if is_delta and rounded_value > 0 else ''
    if chart_type == 'percentage':
        suffix = 'pp' if is_delta else '%'
        return f'{sign}{rounded_value:.1f}{suffix}'
    return f'{sign}{rounded_value:,.{precision}f}'


def _format_origin_kpi_pct(delta_pct):
    if delta_pct is None or pd.isna(delta_pct):
        return ''
    rounded_pct = int(round(float(delta_pct)))
    rounded_pct = 0 if rounded_pct == 0 else rounded_pct
    sign = '+' if rounded_pct > 0 else ''
    return f' ({sign}{rounded_pct}%)'


def _format_origin_kpi_pct_compact(delta_pct):
    if delta_pct is None or pd.isna(delta_pct):
        return None
    rounded_pct = int(round(float(delta_pct)))
    rounded_pct = 0 if rounded_pct == 0 else rounded_pct
    sign = '+' if rounded_pct > 0 else ''
    return f'({sign}{rounded_pct}%)'


def _origin_kpi_direction_class(
    value,
    chart_type='absolute',
    volume_metric='mcm_d',
    is_delta_pct=False,
):
    if value is None or pd.isna(value):
        return 'importer-origin-kpi-delta-neutral continent-kpi-delta-neutral'
    precision = (
        0
        if is_delta_pct
        else 1
        if chart_type == 'percentage'
        else _get_importer_volume_metric_display_precision(volume_metric)
    )
    rounded_value = round(float(value), precision)
    if rounded_value > 0:
        return 'importer-origin-kpi-delta-positive continent-kpi-delta-positive'
    if rounded_value < 0:
        return 'importer-origin-kpi-delta-negative continent-kpi-delta-negative'
    return 'importer-origin-kpi-delta-neutral continent-kpi-delta-neutral'


def _origin_kpi_value_displays_zero(
    value,
    chart_type,
    volume_metric='mcm_d',
    is_delta_pct=False,
):
    if value is None or pd.isna(value):
        return True

    precision = (
        0
        if is_delta_pct
        else 1
        if chart_type == 'percentage'
        else _get_importer_volume_metric_display_precision(volume_metric)
    )
    return round(float(value), precision) == 0


def _origin_kpi_all_displayed_values_zero(
    chart_type,
    volume_metric,
    show_deltas,
    latest_value,
    mom_delta_value,
    mom_delta_pct,
    yoy_delta_value,
    yoy_delta_pct
):
    values_to_check = [(latest_value, False)]

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
        _origin_kpi_value_displays_zero(
            value,
            chart_type,
            volume_metric,
            is_delta_pct,
        )
        for value, is_delta_pct in values_to_check
    )


def _calculate_origin_continent_kpis(
    data,
    chart_type='absolute',
    volume_metric='mcm_d',
    selected_years=None,
    rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
):
    vol_label = get_volume_metric_info(volume_metric)['label']
    df, metric_column = _prepare_importer_origin_chart_dataframe(
        data,
        chart_type,
        volume_metric,
        rolling_avg_days,
    )
    return _calculate_origin_continent_kpis_from_df(
        df,
        metric_column,
        vol_label,
        chart_type,
        selected_years,
        volume_metric,
    )


def _calculate_origin_continent_kpis_from_df(
    df,
    metric_column,
    vol_label,
    chart_type='absolute',
    selected_years=None,
    volume_metric='mcm_d',
):
    if df.empty:
        return []

    available_years = sorted(
        [year for year in df['_year_token'].dropna().unique()],
        key=_importer_chart_year_sort_key
    )
    active_years = _normalize_importer_chart_selected_years(
        selected_years,
        available_years,
        use_default=selected_years is None
    )
    if not active_years:
        return []

    focus_year = active_years[-1]
    focus_df = df[df['_year_token'] == focus_year].copy()
    if focus_df.empty:
        return []
    if 'is_forecast' in focus_df.columns:
        actual_focus_df = focus_df[~focus_df['is_forecast'].astype(bool)]
        if not actual_focus_df.empty:
            focus_df = actual_focus_df

    try:
        previous_year = str(int(focus_year) - 1)
    except (TypeError, ValueError):
        previous_year = None

    metrics = []
    for continent in sorted(focus_df['continent_origin'].dropna().unique()):
        continent_focus_df = (
            focus_df[focus_df['continent_origin'] == continent]
            .dropna(subset=[metric_column])
            .sort_values('plot_date')
        )
        if continent_focus_df.empty:
            continue

        latest_point = continent_focus_df.tail(1).iloc[0]
        latest_value = latest_point[metric_column]
        latest_plot_date = latest_point['plot_date']

        mom_delta_value = None
        mom_delta_pct = None
        month_ago_date = latest_plot_date - pd.DateOffset(months=1)
        mom_candidates = continent_focus_df[continent_focus_df['plot_date'] <= month_ago_date]
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
                df[
                    (df['continent_origin'] == continent)
                    & (df['_year_token'] == previous_year)
                    & (df['plot_date'] <= latest_plot_date)
                    & df[metric_column].notna()
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

        if _origin_kpi_all_displayed_values_zero(
            chart_type,
            volume_metric,
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
            'color': ORIGIN_CONTINENT_CHART_COLOR_MAP.get(continent, '#64748b'),
            'show_deltas': show_deltas,
            'chart_type': chart_type,
            'unit_label': vol_label,
            'latest_value': latest_numeric,
            'latest_text': _format_origin_kpi_value(
                latest_value,
                chart_type,
                volume_metric,
            ),
            'latest_label': latest_point.get('month_day', ''),
            'mom_delta_value': mom_delta_numeric,
            'mom_value_text': (
                _format_origin_kpi_value(
                    mom_delta_value,
                    chart_type,
                    volume_metric,
                    is_delta=True,
                )
            ) if mom_delta_value is not None and pd.notna(mom_delta_value) else 'n/a',
            'mom_pct_text': _format_origin_kpi_pct_compact(mom_delta_pct),
            'mom_text': (
                _format_origin_kpi_value(
                    mom_delta_value,
                    chart_type,
                    volume_metric,
                    is_delta=True,
                )
                + _format_origin_kpi_pct(mom_delta_pct)
            ) if mom_delta_value is not None and pd.notna(mom_delta_value) else 'n/a',
            'mom_class': _origin_kpi_direction_class(
                mom_delta_value,
                chart_type,
                volume_metric,
            ),
            'mom_delta_pct': mom_pct_numeric,
            'yoy_delta_value': yoy_delta_numeric,
            'yoy_value_text': (
                _format_origin_kpi_value(
                    yoy_delta_value,
                    chart_type,
                    volume_metric,
                    is_delta=True,
                )
            ) if yoy_delta_value is not None and pd.notna(yoy_delta_value) else 'n/a',
            'yoy_pct_text': _format_origin_kpi_pct_compact(yoy_delta_pct),
            'yoy_text': (
                _format_origin_kpi_value(
                    yoy_delta_value,
                    chart_type,
                    volume_metric,
                    is_delta=True,
                )
                + _format_origin_kpi_pct(yoy_delta_pct)
            ) if yoy_delta_value is not None and pd.notna(yoy_delta_value) else 'n/a',
            'yoy_delta_pct': yoy_pct_numeric,
            'yoy_class': _origin_kpi_direction_class(
                yoy_delta_value,
                chart_type,
                volume_metric,
            ),
        })

    return sorted(metrics, key=lambda item: item['latest_value'] or 0, reverse=True)


def _origin_kpi_summary_column_sort_key(continent):
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


def _build_origin_kpi_summary_value_cell(value_text, class_name=''):
    return html.Td(
        html.Span(value_text, className='importer-origin-kpi-summary-value continent-kpi-summary-value'),
        className=(
            'importer-origin-kpi-summary-cell importer-origin-kpi-summary-value-cell '
            f'continent-kpi-summary-cell continent-kpi-summary-value-cell {class_name}'
        ).strip()
    )


def _build_origin_kpi_summary_delta_cell(
    value_text,
    class_name,
    is_available=True,
    pct_text=None,
    title_text=None,
    role_class=''
):
    if not is_available or value_text in (None, 'n/a'):
        return html.Td(
            html.Span(
                '-',
                className='importer-origin-kpi-summary-empty-value continent-kpi-summary-empty-value'
            ),
            className=(
                'importer-origin-kpi-summary-cell importer-origin-kpi-summary-delta-cell '
                'importer-origin-kpi-summary-cell-empty continent-kpi-summary-cell '
                f'continent-kpi-summary-delta-cell continent-kpi-summary-cell-empty {role_class}'
            ).strip()
        )

    cell_content = html.Span(
        [
            html.Span(
                value_text,
                className='importer-origin-kpi-summary-delta-main continent-kpi-summary-delta-main'
            ),
            html.Span(
                pct_text,
                className='importer-origin-kpi-summary-delta-pct continent-kpi-summary-delta-pct'
            ) if pct_text else None
        ],
        className='importer-origin-kpi-summary-delta-stack continent-kpi-summary-delta-stack'
    )

    return html.Td(
        cell_content,
        className=(
            'importer-origin-kpi-summary-cell importer-origin-kpi-summary-delta-cell '
            f'continent-kpi-summary-cell continent-kpi-summary-delta-cell {role_class} {class_name}'
        ).strip(),
        title=title_text or value_text
    )


def _build_origin_kpi_summary_transposed_cell(metric, metric_key, entity_name):
    entity_class = (
        'importer-origin-kpi-summary-entity-cell-primary continent-kpi-summary-entity-cell-primary'
        if entity_name == IMPORTER_GLOBAL_LABEL
        else ''
    )

    if metric_key == 'Current':
        value_text = metric['latest_text'] if metric else '-'
        if not metric:
            return html.Td(
                html.Span(
                    '-',
                    className='importer-origin-kpi-summary-empty-value continent-kpi-summary-empty-value'
                ),
                className=(
                    'importer-origin-kpi-summary-cell importer-origin-kpi-summary-value-cell '
                    'importer-origin-kpi-summary-current-cell importer-origin-kpi-summary-cell-empty '
                    'continent-kpi-summary-cell continent-kpi-summary-value-cell '
                    f'continent-kpi-summary-current-cell continent-kpi-summary-cell-empty {entity_class}'
                ).strip()
            )
        return _build_origin_kpi_summary_value_cell(
            value_text,
            class_name=(
                'importer-origin-kpi-summary-current-cell continent-kpi-summary-current-cell '
                f'{entity_class}'
            ).strip()
        )

    if metric_key == 'MoM':
        return _build_origin_kpi_summary_delta_cell(
            metric.get('mom_value_text', metric['mom_text']) if metric else None,
            f"{metric['mom_class']} {entity_class}".strip() if metric else entity_class,
            is_available=metric.get('show_deltas', True) if metric else False,
            pct_text=metric.get('mom_pct_text') if metric else None,
            title_text=metric.get('mom_text') if metric else None,
            role_class='importer-origin-kpi-summary-mom-cell continent-kpi-summary-mom-cell'
        )

    return _build_origin_kpi_summary_delta_cell(
        metric.get('yoy_value_text', metric['yoy_text']) if metric else None,
        f"{metric['yoy_class']} {entity_class}".strip() if metric else entity_class,
        is_available=metric.get('show_deltas', True) if metric else False,
        pct_text=metric.get('yoy_pct_text') if metric else None,
        title_text=metric.get('yoy_text') if metric else None,
        role_class='importer-origin-kpi-summary-yoy-cell continent-kpi-summary-yoy-cell'
    )


def _build_origin_kpi_summary_table(entity_kpi_rows):
    active_rows = [row for row in entity_kpi_rows if row['metrics']]
    if not active_rows:
        return html.Div(
            'KPI data unavailable',
            className=(
                'importer-origin-kpi-summary importer-origin-kpi-empty '
                'continent-kpi-summary continent-kpi-summary-empty'
            )
        )

    continents = sorted(
        {
            metric['continent']
            for row in active_rows
            for metric in row['metrics']
        },
        key=_origin_kpi_summary_column_sort_key
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
        html.Div(
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th(
                                    'Origin',
                                    className=(
                                        'importer-origin-kpi-summary-axis-header '
                                        'importer-origin-kpi-summary-origin-axis-header '
                                        'continent-kpi-summary-axis-header '
                                        'continent-kpi-summary-continent-axis-header'
                                    )
                                ),
                                html.Th(
                                    'Metric',
                                    className=(
                                        'importer-origin-kpi-summary-axis-header '
                                        'importer-origin-kpi-summary-metric-axis-header '
                                        'continent-kpi-summary-axis-header '
                                        'continent-kpi-summary-metric-axis-header'
                                    )
                                )
                            ]
                            + [
                                html.Th(
                                    entity_name,
                                    className=(
                                        'importer-origin-kpi-summary-entity-header '
                                        'importer-origin-kpi-summary-entity-header-primary '
                                        'continent-kpi-summary-entity-header '
                                        'continent-kpi-summary-entity-header-primary'
                                        if entity_name == IMPORTER_GLOBAL_LABEL
                                        else (
                                            'importer-origin-kpi-summary-entity-header '
                                            'continent-kpi-summary-entity-header'
                                        )
                                    ),
                                    title=entity_name
                                )
                                for entity_name in entity_names
                            ],
                            className=(
                                'importer-origin-kpi-summary-entity-header-row '
                                'continent-kpi-summary-entity-header-row'
                            )
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                (
                                    [
                                        html.Th(
                                            [
                                                html.Span(
                                                    className=(
                                                        'importer-origin-kpi-summary-swatch '
                                                        'continent-kpi-summary-swatch'
                                                    ),
                                                    style={
                                                        'backgroundColor': ORIGIN_CONTINENT_CHART_COLOR_MAP.get(
                                                            continent,
                                                            '#64748b'
                                                        )
                                                    }
                                                ),
                                                html.Span(continent)
                                            ],
                                            rowSpan=len(metric_rows),
                                            className=(
                                                'importer-origin-kpi-summary-origin-axis-cell '
                                                'continent-kpi-summary-continent-axis-cell'
                                            )
                                        )
                                    ]
                                    if metric_index == 0
                                    else []
                                )
                                + [
                                    html.Th(
                                        metric_label,
                                        className=(
                                            'importer-origin-kpi-summary-metric-cell '
                                            f'importer-origin-kpi-summary-metric-cell-{metric_key.lower()} '
                                            'continent-kpi-summary-metric-cell '
                                            f'continent-kpi-summary-metric-cell-{metric_key.lower()}'
                                        ),
                                        title=metric_key
                                    )
                                ]
                                + [
                                    _build_origin_kpi_summary_transposed_cell(
                                        metrics_by_entity.get(entity_name, {}).get(continent),
                                        metric_key,
                                        entity_name
                                    )
                                    for entity_name in entity_names
                                ],
                                className=(
                                    'importer-origin-kpi-summary-row '
                                    f'importer-origin-kpi-summary-row-{metric_key.lower()} '
                                    'continent-kpi-summary-row '
                                    f'continent-kpi-summary-row-{metric_key.lower()} '
                                    + (
                                        'importer-origin-kpi-summary-origin-group-start '
                                        'continent-kpi-summary-continent-group-start'
                                        if metric_index == 0
                                        else ''
                                    )
                                )
                            )
                            for continent in continents
                            for metric_rows in ([subcolumns[:1]] if continent == 'Unknown' else [subcolumns])
                            for metric_index, (metric_key, metric_label) in enumerate(metric_rows)
                        ]
                    )
                ],
                className='importer-origin-kpi-summary-table continent-kpi-summary-table'
            ),
            className='importer-origin-kpi-summary-table-wrap continent-kpi-summary-table-wrap'
        ),
        className='importer-origin-kpi-summary continent-kpi-summary'
    )


def _build_origin_year_style_key():
    return html.Div(
        [
            html.Div('Year style', className='importer-origin-year-style-title continent-year-style-title'),
            html.Span(
                [
                    html.Span(
                        className=(
                            'importer-origin-year-style-line importer-origin-year-style-line-current '
                            'continent-year-style-line continent-year-style-line-current'
                        )
                    ),
                    html.Span('Latest')
                ],
                className='importer-origin-year-style-item continent-year-style-item'
            ),
            html.Span(
                [
                    html.Span(
                        className=(
                            'importer-origin-year-style-line importer-origin-year-style-line-previous '
                            'continent-year-style-line continent-year-style-line-previous'
                        )
                    ),
                    html.Span('Previous')
                ],
                className='importer-origin-year-style-item continent-year-style-item'
            ),
            html.Span(
                [
                    html.Span(
                        className=(
                            'importer-origin-year-style-line importer-origin-year-style-line-forecast '
                            'continent-year-style-line continent-year-style-line-forecast'
                        )
                    ),
                    html.Span('Forecast')
                ],
                className='importer-origin-year-style-item continent-year-style-item'
            )
        ],
        className='importer-origin-year-style-key continent-year-style-key'
    )


def _get_available_period_columns(period_payload):
    """Return the union of all numeric period columns present in the payload."""
    available_cols = set()
    for importer_payload in period_payload or []:
        for record in importer_payload.get('records', []):
            available_cols.update(
                col for col in record.keys()
                if col not in ['continent', 'country']
            )
    return available_cols


def _is_completed_quarter_label(label, current_date):
    """Return whether the quarter label is completed relative to today."""
    try:
        quarter_part, year_suffix = label.split("'")
        quarter_num = int(quarter_part.replace('Q', ''))
        year = int(f"20{year_suffix}")
    except (IndexError, ValueError):
        return False

    current_quarter = (current_date.month - 1) // 3 + 1
    return year < current_date.year or (year == current_date.year and quarter_num < current_quarter)


def _is_completed_month_label(label, current_date):
    """Return whether the month label is completed relative to today."""
    try:
        month_abbr, year_suffix = label.split("'")
        year = int(f"20{year_suffix}")
        month = MONTH_ORDER[month_abbr]
    except (ValueError, KeyError):
        return False

    return year < current_date.year or (year == current_date.year and month < current_date.month)


def _is_completed_week_label(label, current_date):
    """Return whether the ISO-week label is completed relative to today."""
    try:
        week_part, year_suffix = label.split("'")
        week_num = int(week_part.replace('W', ''))
        year = int(f"20{year_suffix}")
    except (IndexError, ValueError):
        return False

    current_iso = current_date.isocalendar()
    return year < current_iso.year or (year == current_iso.year and week_num < current_iso.week)


def _quarter_sort_key(label):
    """Return a sortable key for quarter labels like Q1'25."""
    quarter_part, year_suffix = label.split("'")
    quarter_num = int(quarter_part.replace('Q', ''))
    year = int(f"20{year_suffix}")
    return (year, quarter_num)


def _month_sort_key(label):
    """Return a sortable key for month labels like Mar'25."""
    month_abbr, year_suffix = label.split("'")
    return (int(f"20{year_suffix}"), MONTH_ORDER[month_abbr])


def _week_sort_key(label):
    """Return a sortable key for week labels like W12'25."""
    week_part, year_suffix = label.split("'")
    week_num = int(week_part.replace('W', ''))
    year = int(f"20{year_suffix}")
    return (year, week_num)


def _parse_importer_period_year_suffix(year_suffix):
    try:
        return int(f"20{int(year_suffix):02d}")
    except (TypeError, ValueError):
        return None


def _get_importer_previous_period_label(column_name, period_type):
    try:
        if period_type == 'quarterly':
            quarter_part, year_suffix = str(column_name).split("'")
            quarter = int(quarter_part.replace('Q', ''))
            year = _parse_importer_period_year_suffix(year_suffix)
            if year is None:
                return None
            if quarter == 1:
                return f"Q4'{str(year - 1)[2:]}"
            return f"Q{quarter - 1}'{str(year)[2:]}"
        if period_type == 'monthly':
            month_label, year_suffix = str(column_name).split("'")
            year = _parse_importer_period_year_suffix(year_suffix)
            month = MONTH_ORDER.get(month_label)
            if year is None or month is None:
                return None
            if month == 1:
                return f"Dec'{str(year - 1)[2:]}"
            previous_month = month - 1
            previous_label = next((label for label, number in MONTH_ORDER.items() if number == previous_month), None)
            return f"{previous_label}'{str(year)[2:]}" if previous_label else None
    except (TypeError, ValueError):
        return None
    return None


def _get_importer_prior_year_label(column_name, period_type):
    try:
        if period_type == 'quarterly':
            quarter_part, year_suffix = str(column_name).split("'")
            year = _parse_importer_period_year_suffix(year_suffix)
            return f"{quarter_part}'{str(year - 1)[2:]}" if year is not None else None
        if period_type == 'monthly':
            month_label, year_suffix = str(column_name).split("'")
            year = _parse_importer_period_year_suffix(year_suffix)
            return f"{month_label}'{str(year - 1)[2:]}" if year is not None else None
        if period_type == 'weekly':
            week_part, year_suffix = str(column_name).split("'")
            year = _parse_importer_period_year_suffix(year_suffix)
            return f"{week_part}'{str(year - 1)[2:]}" if year is not None else None
    except (TypeError, ValueError):
        return None
    return None


def _get_importer_previous_week_label(column_name, week_cols):
    if column_name in week_cols:
        column_index = week_cols.index(column_name)
        if column_index > 0:
            return week_cols[column_index - 1]
    return None


def _build_importer_period_comparison_reference_map(visible_period_cols, week_cols,
                                                    rolling_window_label, available_cols,
                                                    comparison_basis):
    comparison_basis = _normalize_importer_period_comparison_basis(comparison_basis)
    if comparison_basis not in {'previous_period', 'same_period_last_year'}:
        return {}

    reference_map = {}
    for col in visible_period_cols:
        reference_col = None
        if _is_completed_quarter_label(col, datetime.now()):
            reference_col = (
                _get_importer_previous_period_label(col, 'quarterly')
                if comparison_basis == 'previous_period'
                else _get_importer_prior_year_label(col, 'quarterly')
            )
        elif _is_completed_month_label(col, datetime.now()):
            reference_col = (
                _get_importer_previous_period_label(col, 'monthly')
                if comparison_basis == 'previous_period'
                else _get_importer_prior_year_label(col, 'monthly')
            )
        elif _is_completed_week_label(col, datetime.now()):
            reference_col = (
                _get_importer_previous_week_label(col, week_cols)
                if comparison_basis == 'previous_period'
                else _get_importer_prior_year_label(col, 'weekly')
            )
        if reference_col in available_cols:
            reference_map[col] = reference_col

    if comparison_basis == 'previous_period':
        rolling_reference_map = {
            rolling_window_label: f'{rolling_window_label}_PP',
            '7D': '7D_PP',
        }
    else:
        rolling_reference_map = {
            rolling_window_label: f'{rolling_window_label}_Y1',
            '7D': '7D_Y1',
        }
    reference_map.update({
        visible_col: reference_col
        for visible_col, reference_col in rolling_reference_map.items()
        if visible_col in available_cols and reference_col in available_cols
    })
    return reference_map


def _get_period_numeric_columns(
    period_payload,
    rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
    quarter_count=IMPORTER_PERIOD_DEFAULT_QUARTER_COUNT,
    month_count=IMPORTER_PERIOD_DEFAULT_MONTH_COUNT,
    week_count=IMPORTER_PERIOD_DEFAULT_WEEK_COUNT,
    comparison_basis='levels',
    return_metadata=False
):
    """Return the curated period-column order for the combined overview table."""
    available_cols = _get_available_period_columns(period_payload)
    if not available_cols:
        empty_metadata = {
            'comparison_basis': _normalize_importer_period_comparison_basis(comparison_basis),
            'visible_comparison_cols': [],
            'comparison_reference_map': {},
            'reference_cols': [],
            'comparison_delta_cols': [],
        }
        return ([], empty_metadata) if return_metadata else []

    current_date = datetime.now()
    rolling_window_label = _format_importer_rolling_window_label(rolling_avg_days)
    comparison_basis = _normalize_importer_period_comparison_basis(comparison_basis)
    quarter_count = _coerce_importer_period_count(
        quarter_count,
        IMPORTER_PERIOD_DEFAULT_QUARTER_COUNT,
        IMPORTER_PERIOD_MAX_QUARTER_COUNT
    )
    month_count = _coerce_importer_period_count(
        month_count,
        IMPORTER_PERIOD_DEFAULT_MONTH_COUNT,
        IMPORTER_PERIOD_MAX_MONTH_COUNT
    )
    week_count = _coerce_importer_period_count(
        week_count,
        IMPORTER_PERIOD_DEFAULT_WEEK_COUNT,
        IMPORTER_PERIOD_MAX_WEEK_COUNT
    )

    quarter_cols = sorted(
        [
            col for col in available_cols
            if col.startswith('Q') and "'" in col and _is_completed_quarter_label(col, current_date)
        ],
        key=_quarter_sort_key
    )[-quarter_count:]

    month_cols = sorted(
        [
            col for col in available_cols
            if (
                "'" in col and
                not col.startswith('Q') and
                not col.startswith('W') and
                _is_completed_month_label(col, current_date)
            )
        ],
        key=_month_sort_key
    )[-month_count:]

    all_week_cols = sorted(
        [
            col for col in available_cols
            if col.startswith('W') and "'" in col and _is_completed_week_label(col, current_date)
        ],
        key=_week_sort_key
    )
    week_cols = all_week_cols[-week_count:]

    numeric_cols = []
    numeric_cols.extend(quarter_cols)
    numeric_cols.extend(month_cols)
    if rolling_window_label in available_cols:
        numeric_cols.append(rolling_window_label)
    numeric_cols.extend(week_cols)

    if '7D' in available_cols and '7D' not in numeric_cols:
        numeric_cols.append('7D')

    visible_period_cols = quarter_cols + month_cols + week_cols
    visible_comparison_cols = numeric_cols.copy()
    comparison_reference_map = _build_importer_period_comparison_reference_map(
        visible_period_cols,
        all_week_cols,
        rolling_window_label,
        available_cols,
        comparison_basis
    )
    reference_cols = [
        reference_col
        for reference_col in comparison_reference_map.values()
        if reference_col not in numeric_cols
    ]
    numeric_cols.extend(reference_cols)

    if comparison_basis == 'levels':
        for col in [f'Δ 7D-{rolling_window_label}', f'Δ {rolling_window_label} Y/Y']:
            if col in available_cols:
                numeric_cols.append(col)

    for col in (
        *IMPORTER_PERIOD_PBD_CURRENT_COLUMNS,
        *IMPORTER_PERIOD_PBD_REFERENCE_COLUMNS,
        *IMPORTER_PERIOD_PBD_DELTA_COLUMNS,
    ):
        if col in available_cols and col not in numeric_cols:
            numeric_cols.append(col)

    metadata = {
        'comparison_basis': comparison_basis,
        'visible_comparison_cols': visible_comparison_cols,
        'comparison_reference_map': comparison_reference_map,
        'reference_cols': reference_cols,
        'comparison_delta_cols': (
            visible_comparison_cols
            if comparison_basis in {'previous_period', 'same_period_last_year'}
            else []
        ),
    }

    if return_metadata:
        return numeric_cols, metadata
    return numeric_cols


def _sum_numeric_columns(df, numeric_cols):
    """Aggregate numeric columns safely while preserving missing-data semantics."""
    totals = {}
    for col in numeric_cols:
        numeric_series = pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series(dtype=float)
        totals[col] = numeric_series.sum(min_count=1)
    return totals


def _get_importer_period_column_family(column_id):
    column_id = str(column_id)
    if column_id in IMPORTER_PERIOD_TEXT_COLUMNS:
        return 'label'
    if column_id == '7D':
        return 'rolling-7d'
    if column_id.endswith('D') and column_id[:-1].isdigit():
        return 'rolling-window'
    if column_id.startswith('Δ 7D-'):
        return 'delta-mom'
    if column_id.startswith('Δ ') and column_id.endswith(' Y/Y'):
        return 'delta-yoy'
    if column_id in IMPORTER_PERIOD_PBD_DELTA_COLUMNS:
        return 'delta-pbd'
    if column_id.startswith('Q') and "'" in column_id:
        return 'quarter'
    if column_id.startswith('W') and "'" in column_id:
        return 'week'
    if "'" in column_id:
        return 'month'
    return 'numeric'


def _format_importer_period_filter_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '0'
    text = f'{number:.6f}'.rstrip('0').rstrip('.')
    return text or '0'


def _get_importer_period_delta_raw_field(column_id):
    token = ''.join(
        character.lower() if character.isascii() and character.isalnum() else '_'
        for character in str(column_id)
    ).strip('_')
    return f'{IMPORTER_PERIOD_DELTA_RAW_FIELD_PREFIX}{token or "value"}'


def _get_importer_period_delta_thresholds(display_df, column_id):
    if display_df is None or display_df.empty or column_id not in display_df.columns:
        return []

    total_mask = (
        display_df['Importer'].astype(str).eq(IMPORTER_GLOBAL_LABEL)
        if 'Importer' in display_df.columns
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


def _build_importer_period_numeric_filter_js(column_id, operator, threshold_text, raw_field=None):
    data_field = raw_field or column_id
    escaped_data_field = str(data_field).replace("\\", "\\\\").replace("'", "\\'")
    if raw_field:
        return f"(params.data && params.data['{escaped_data_field}'] {operator} {threshold_text})"

    return (
        f"(Number(String(params.data && params.data['{escaped_data_field}'] !== undefined "
        f"? params.data['{escaped_data_field}'] : params.value).replace(/[^0-9.\\-]/g, '')) "
        f"{operator} {threshold_text})"
    )


def _build_importer_period_delta_heatmap_class_rules(display_df, column_id, raw_field=None):
    thresholds = _get_importer_period_delta_thresholds(display_df, column_id)
    band_thresholds = [0, *thresholds]
    rules = {}
    for band_index, threshold in enumerate(band_thresholds, start=1):
        positive_threshold = _format_importer_period_filter_number(threshold)
        negative_threshold = _format_importer_period_filter_number(-threshold)
        rules[f'importer-period-delta-positive-{band_index}'] = {
            'function': _build_importer_period_numeric_filter_js(
                column_id,
                '>',
                positive_threshold,
                raw_field=raw_field
            )
        }
        rules[f'importer-period-delta-negative-{band_index}'] = {
            'function': _build_importer_period_numeric_filter_js(
                column_id,
                '<',
                negative_threshold,
                raw_field=raw_field
            )
        }
    return rules


def _build_importer_period_delta_gradient_styles(display_df, column_id, raw_field=None):
    family = _get_importer_period_column_family(column_id)
    border_color = '#c9d0d9' if family == 'delta-mom' else '#c3d9c2'
    base_bg = '#e9edf2' if family == 'delta-mom' else '#e2f0e3'
    styles = [{
        'if': {'column_id': column_id},
        'backgroundColor': base_bg,
        'borderLeft': f'2px solid {border_color}',
        'color': '#334155',
        'fontWeight': '700',
        'textAlign': 'right',
        'paddingRight': '12px'
    }]
    thresholds = _get_importer_period_delta_thresholds(display_df, column_id)

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
    band_thresholds = [0, *thresholds]

    for palette, operator, sign in (
        (positive_palette, '>', ''),
        (negative_palette, '<', '-')
    ):
        for threshold, (background, color, weight) in reversed(list(zip(band_thresholds, palette))):
            threshold_text = _format_importer_period_filter_number(threshold)
            styles.append({
                'if': {
                    'column_id': column_id,
                    'filter_query_js': _build_importer_period_numeric_filter_js(
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


def _apply_importer_period_column_classes(columns, display_df, delta_like_cols=None, raw_field_map=None):
    delta_like_cols = set(delta_like_cols or [])
    raw_field_map = raw_field_map or {}
    classed_columns = []
    previous_family = None
    for column in columns:
        column = dict(column)
        column_id = column.get('id')
        family = _get_importer_period_column_family(column_id)
        header_classes = [f'importer-period-header-{family}']
        if family == 'label':
            header_classes.append(
                'importer-period-header-label-primary'
                if column_id == 'Importer'
                else 'importer-period-header-label-secondary'
            )
        if family != previous_family and family != 'label':
            header_classes.append('importer-period-header-group-start')
        column['headerClass'] = ' '.join(header_classes)

        existing_cell_class = str(column.get('cellClass') or '').strip()
        extra_cell_classes = []
        if family != 'label':
            if 'importer-period-number-cell' not in existing_cell_class.split():
                extra_cell_classes.append('importer-period-number-cell')
            if family in {'delta-mom', 'delta-yoy', 'delta-pbd'} or column_id in delta_like_cols:
                extra_cell_classes.append('importer-period-delta-cell')
                column['cellClassRules'] = _build_importer_period_delta_heatmap_class_rules(
                    display_df,
                    column_id,
                    raw_field=raw_field_map.get(column_id)
                )
        column['cellClass'] = ' '.join(
            class_name for class_name in [existing_cell_class, *extra_cell_classes] if class_name
        )
        classed_columns.append(column)
        previous_family = family
    return classed_columns


def _build_importer_period_column_width_styles(display_df, columns):
    width_styles = []
    text_width_limits = {
        'Importer': (142, 190),
        'Aggregation': (158, 220),
    }

    for column in columns:
        column_id = column.get('id')
        if not column_id or column_id not in display_df.columns:
            continue

        family = _get_importer_period_column_family(column_id)
        header_text = str(column.get('name') or column_id)
        samples = [header_text, *[str(value) for value in display_df[column_id].head(80).tolist() if pd.notna(value)]]
        max_chars = max((len(sample) for sample in samples), default=len(header_text))
        if family == 'label':
            min_width, max_width = text_width_limits.get(column_id, (124, 180))
            width = int(min(max(max_chars * 6.1 + 30, min_width), max_width))
        elif family in {'delta-mom', 'delta-yoy', 'delta-pbd'}:
            width = int(min(max(len(header_text) * 7.0 + 26, 106), 132))
        elif family in {'rolling-window', 'rolling-7d'}:
            width = int(min(max(len(header_text) * 7.0 + 28, 78), 98))
        else:
            width = int(min(max(len(header_text) * 7.0 + 28, 72), 104))

        width_styles.append({
            'if': {'column_id': column_id},
            'width': f'{width}px',
            'minWidth': f'{width}px',
            'maxWidth': f'{width}px'
        })
    return width_styles


def _build_importer_period_delta_styles(display_df, columns, delta_like_cols=None, raw_field_map=None):
    styles = []
    delta_like_cols = set(delta_like_cols or [])
    raw_field_map = raw_field_map or {}
    for column in columns:
        column_id = column.get('id')
        if _get_importer_period_column_family(column_id) in {'delta-mom', 'delta-yoy', 'delta-pbd'} or column_id in delta_like_cols:
            styles.extend(
                _build_importer_period_delta_gradient_styles(
                    display_df,
                    column_id,
                    raw_field=raw_field_map.get(column_id)
                )
            )
    return styles


def _strip_importer_period_reference_suffix(column_name):
    column_name = str(column_name)
    for suffix in ('_PBD_CURRENT', '_PBD', '_PP', '_Y1'):
        if column_name.endswith(suffix):
            return column_name[:-len(suffix)]
    return column_name


def _get_importer_period_column_days(
    column_name,
    rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
):
    """Return the day basis used to derive a period's average mcm/d."""
    base_column = _strip_importer_period_reference_suffix(column_name)
    if base_column == '7D':
        return 7
    if base_column.endswith('D') and base_column[:-1].isdigit():
        return int(base_column[:-1])
    if base_column.startswith('Q') and "'" in base_column:
        # The shared importer period builder divides every quarter by 91.25.
        return 91.25
    if base_column.startswith('W') and "'" in base_column:
        return 7
    if "'" in base_column:
        try:
            month_label, year_suffix = base_column.split("'")
        except ValueError:
            return None
        month = MONTH_ORDER.get(month_label)
        year = _parse_importer_period_year_suffix(year_suffix)
        if month is not None and year is not None:
            return calendar.monthrange(year, month)[1]
    normalized_days = normalize_importer_rolling_avg_days(rolling_avg_days)
    return normalized_days if base_column == f'{normalized_days}D' else None


def _build_importer_period_days_map(
    columns,
    rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
):
    period_days_by_column = {}
    for column_name in ([] if columns is None else columns):
        period_days = _get_importer_period_column_days(
            column_name,
            rolling_avg_days,
        )
        if period_days is not None:
            period_days_by_column[column_name] = period_days
    return period_days_by_column


def _recalculate_importer_period_absolute_deltas(
    display_df,
    volume_metric,
    rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
):
    recalculated_df = display_df.copy()
    period_days = _build_importer_period_days_map(
        recalculated_df.columns,
        rolling_avg_days,
    )
    for delta_column in [
        column for column in recalculated_df.columns
        if str(column).startswith('Δ 7D-')
    ]:
        comparison_column = str(delta_column).replace('Δ 7D-', '', 1)
        if {'7D', comparison_column}.issubset(recalculated_df.columns):
            if (
                _is_importer_period_volume_metric(volume_metric)
                and period_days.get('7D') != period_days.get(comparison_column)
            ):
                recalculated_df[delta_column] = pd.NA
            else:
                recalculated_df[delta_column] = (
                    pd.to_numeric(recalculated_df['7D'], errors='coerce')
                    - pd.to_numeric(
                        recalculated_df[comparison_column],
                        errors='coerce',
                    )
                ).round(1)

    for delta_column in [
        column for column in recalculated_df.columns
        if str(column).startswith('Δ ') and str(column).endswith(' Y/Y')
    ]:
        base_column = str(delta_column).replace('Δ ', '', 1)[:-4]
        reference_column = f'{base_column}_Y1'
        if {base_column, reference_column}.issubset(recalculated_df.columns):
            recalculated_df[delta_column] = (
                pd.to_numeric(recalculated_df[base_column], errors='coerce')
                - pd.to_numeric(
                    recalculated_df[reference_column],
                    errors='coerce',
                )
            ).round(1)
        elif delta_column in recalculated_df.columns:
            # Levels views do not carry the hidden Y-1 column. Convert the
            # already-computed mcm/d delta using the base horizon instead.
            recalculated_df = convert_volume_metric_dataframe(
                recalculated_df,
                volume_metric,
                columns=[delta_column],
                precision=None,
                period_days=period_days.get(base_column),
            )

    return _recalculate_importer_period_pbd_deltas(recalculated_df)


def _convert_importer_period_absolute_volume_metric(
    display_df,
    volume_metric,
    rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
):
    delta_columns = {
        column for column in display_df.columns
        if str(column).startswith('Δ ')
    }
    converted_df = convert_volume_metric_dataframe(
        display_df,
        volume_metric,
        exclude_columns=set(IMPORTER_PERIOD_TEXT_COLUMNS) | delta_columns,
        precision=None,
        period_days_by_column=_build_importer_period_days_map(
            display_df.columns,
            rolling_avg_days,
        ),
    )
    return _recalculate_importer_period_absolute_deltas(
        converted_df,
        volume_metric,
        rolling_avg_days,
    )


def _get_importer_period_display_precision(view_type, volume_metric):
    if view_type == 'percentage':
        return IMPORTER_PERIOD_PERCENTAGE_DISPLAY_PRECISION
    return _get_importer_volume_metric_display_precision(volume_metric)


def _format_importer_period_grid_value(
    value,
    view_type='absolute',
    is_delta=False,
    is_pbd_delta=False,
    volume_metric='mcm_d',
):
    if value is None or value is pd.NA:
        return '—' if is_delta or is_pbd_delta else ''
    try:
        if pd.isna(value):
            return '—' if is_delta or is_pbd_delta else ''
    except (TypeError, ValueError):
        pass

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    precision = _get_importer_period_display_precision(
        view_type,
        volume_metric,
    )
    numeric_value = round(numeric_value, precision)
    numeric_value = 0.0 if numeric_value == 0 else numeric_value
    if view_type == 'percentage' and is_delta:
        sign = '+' if numeric_value > 0 else ''
        return f'{sign}{numeric_value:,.{precision}f} pp'
    if view_type == 'percentage':
        return f'{numeric_value:.{precision}f}%'
    if is_pbd_delta and numeric_value > 0:
        return f'+{numeric_value:,.{precision}f}'
    return f'{numeric_value:,.{precision}f}'


def _build_importer_period_grid_display(display_df, columns, view_type='absolute',
                                        delta_like_cols=None, raw_field_map=None,
                                        volume_metric='mcm_d'):
    grid_df = display_df.copy()
    grid_columns = [dict(column) for column in columns]
    delta_like_cols = set(delta_like_cols or [])
    raw_field_map = raw_field_map or {}
    display_precision = _get_importer_period_display_precision(
        view_type,
        volume_metric,
    )
    numeric_ids = {
        column.get('id')
        for column in grid_columns
        if column.get('type') == 'numeric'
    }
    delta_ids = {
        column_id for column_id in numeric_ids
        if _get_importer_period_column_family(column_id) in {'delta-mom', 'delta-yoy', 'delta-pbd'}
    } | delta_like_cols

    for column_id, raw_field in raw_field_map.items():
        if column_id in grid_df.columns:
            raw_values = pd.to_numeric(
                grid_df[column_id],
                errors='coerce',
            ).round(display_precision)
            grid_df[raw_field] = raw_values.where(raw_values != 0, 0.0)

    for column_id in numeric_ids:
        if column_id not in grid_df.columns:
            continue
        is_delta = column_id in delta_ids
        grid_df[column_id] = grid_df[column_id].apply(
            lambda value, delta=is_delta, pbd_delta=(
                column_id in IMPORTER_PERIOD_PBD_DELTA_COLUMNS
            ): _format_importer_period_grid_value(
                value,
                view_type=view_type,
                is_delta=delta,
                is_pbd_delta=pbd_delta,
                volume_metric=volume_metric,
            )
        )

    for column in grid_columns:
        if column.get('id') in numeric_ids:
            column['type'] = 'text'
            column.pop('format', None)

    return grid_df, grid_columns


def _format_importer_display_label(value, expanded=False):
    prefix = '▼ ' if expanded else '▶ '
    return f"{prefix}{value}"


def _strip_importer_expand_marker(value):
    value = str(value or '').strip()
    if value.startswith('▶ ') or value.startswith('▼ '):
        return value[2:].strip()
    return value


def _is_importer_expandable_label(value):
    value = str(value or '').strip()
    return value.startswith('▶ ') or value.startswith('▼ ')


def _apply_importer_period_percentage_view(display_df, numeric_cols):
    if display_df is None or display_df.empty:
        return display_df

    percentage_df = display_df.copy()
    numeric_cols = [col for col in numeric_cols if col in percentage_df.columns]
    current_importer_totals = None

    for row_index, row in percentage_df.iterrows():
        importer_label = str(row.get('Importer', '') or '')
        is_importer_total = _is_importer_expandable_label(importer_label)
        is_grand_total = importer_label == IMPORTER_GLOBAL_LABEL

        if is_importer_total or is_grand_total:
            current_importer_totals = {
                col: pd.to_numeric(row.get(col), errors='coerce')
                for col in numeric_cols
            }
            for col in numeric_cols:
                value = current_importer_totals.get(col)
                percentage_df.at[row_index, col] = 100.0 if pd.notna(value) and value != 0 else None
            if is_grand_total:
                current_importer_totals = None
            continue

        if current_importer_totals is None:
            continue

        for col in numeric_cols:
            denominator = current_importer_totals.get(col)
            numerator = pd.to_numeric(row.get(col), errors='coerce')
            if pd.notna(numerator) and pd.notna(denominator) and denominator != 0:
                percentage_df.at[row_index, col] = (numerator / denominator) * 100
            else:
                percentage_df.at[row_index, col] = None

    return percentage_df


def _apply_importer_period_pbd_percentage_view(display_df):
    """Convert hidden current/PBD levels to origin-mix percentage levels."""
    if display_df is None or display_df.empty:
        return display_df

    percentage_df = display_df.copy()
    pbd_level_columns = [
        column_name
        for column_name in (
            *IMPORTER_PERIOD_PBD_CURRENT_COLUMNS,
            *IMPORTER_PERIOD_PBD_REFERENCE_COLUMNS,
        )
        if column_name in percentage_df.columns
    ]
    for column_name in pbd_level_columns:
        current_total = None
        for row_index, row in display_df.iterrows():
            importer_label = str(row.get('Importer', '') or '')
            value = pd.to_numeric(
                row.get(column_name),
                errors='coerce',
            )
            if importer_label == IMPORTER_GLOBAL_LABEL:
                percentage_df.at[row_index, column_name] = (
                    100.0
                    if pd.notna(value) and value != 0
                    else 0.0
                )
                current_total = None
                continue
            if _is_importer_expandable_label(importer_label):
                current_total = value
                percentage_df.at[row_index, column_name] = (
                    100.0
                    if pd.notna(value) and value != 0
                    else 0.0
                )
                continue
            if current_total is None or pd.isna(current_total):
                continue
            percentage_df.at[row_index, column_name] = (
                (value / current_total) * 100
                if current_total != 0 and pd.notna(value)
                else 0.0
            )
    return percentage_df


def _recalculate_importer_period_pbd_deltas(display_df):
    if display_df is None or display_df.empty:
        return display_df
    recalculated_df = display_df.copy()
    for delta_column, current_column, baseline_column in (
        (
            'Δ 30D vs PBD',
            '30D_PBD_CURRENT',
            '30D_PBD',
        ),
        (
            'Δ 7D vs PBD',
            '7D_PBD_CURRENT',
            '7D_PBD',
        ),
    ):
        if {
            current_column,
            baseline_column,
        }.issubset(recalculated_df.columns):
            recalculated_df[delta_column] = (
                pd.to_numeric(
                    recalculated_df[current_column],
                    errors='coerce',
                )
                - pd.to_numeric(
                    recalculated_df[baseline_column],
                    errors='coerce',
                )
            )
    return recalculated_df


def _apply_importer_period_comparison(display_df, comparison_metadata):
    if display_df is None or display_df.empty:
        return display_df, []

    comparison_metadata = comparison_metadata or {}
    comparison_basis = _normalize_importer_period_comparison_basis(
        comparison_metadata.get('comparison_basis')
    )
    reference_cols = [
        col for col in comparison_metadata.get('reference_cols', [])
        if col in display_df.columns
    ]
    if comparison_basis not in {'previous_period', 'same_period_last_year'}:
        return display_df.drop(columns=reference_cols, errors='ignore'), []

    comparison_df = display_df.copy()
    comparison_source_df = display_df.copy()
    delta_cols = []
    comparison_reference_map = comparison_metadata.get('comparison_reference_map') or {}
    for visible_col in comparison_metadata.get('visible_comparison_cols', []):
        reference_col = comparison_reference_map.get(visible_col)
        if visible_col in comparison_source_df.columns and reference_col in comparison_source_df.columns:
            visible_values = pd.to_numeric(comparison_source_df[visible_col], errors='coerce')
            reference_values = pd.to_numeric(comparison_source_df[reference_col], errors='coerce')
            comparison_df[visible_col] = visible_values - reference_values
            delta_cols.append(visible_col)

    comparison_df = comparison_df.drop(columns=reference_cols, errors='ignore')
    return comparison_df, delta_cols


def _build_period_display_df(period_payload, expanded_importers=None,
                             rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS,
                             quarter_count=IMPORTER_PERIOD_DEFAULT_QUARTER_COUNT,
                             month_count=IMPORTER_PERIOD_DEFAULT_MONTH_COUNT,
                             week_count=IMPORTER_PERIOD_DEFAULT_WEEK_COUNT,
                             comparison_basis='levels',
                             return_metadata=False):
    """Create the combined importer -> selected-origin-aggregation period-analysis display dataframe."""
    expanded_importers = expanded_importers or []
    numeric_cols, comparison_metadata = _get_period_numeric_columns(
        period_payload,
        rolling_avg_days=rolling_avg_days,
        quarter_count=quarter_count,
        month_count=month_count,
        week_count=week_count,
        comparison_basis=comparison_basis,
        return_metadata=True
    )

    if not numeric_cols:
        empty_df = pd.DataFrame(columns=['Importer', 'Aggregation'])
        return (empty_df, comparison_metadata) if return_metadata else empty_df

    display_rows = []
    importer_totals = []

    for importer_payload in period_payload or []:
        importer_label = importer_payload.get('label')
        importer_df = pd.DataFrame(importer_payload.get('records', []))
        if importer_df.empty:
            continue

        importer_df = importer_df.sort_values(['continent', 'country']).reset_index(drop=True)
        importer_total_row = {
            'Importer': _format_importer_display_label(importer_label, importer_label in expanded_importers),
            'Aggregation': 'Total',
            **_sum_numeric_columns(importer_df, numeric_cols)
        }
        display_rows.append(importer_total_row)
        importer_totals.append(importer_total_row)

        if importer_label not in expanded_importers:
            continue

        for origin_name in importer_df['continent'].dropna().unique():
            origin_df = importer_df[importer_df['continent'] == origin_name].copy()
            display_rows.append({
                'Importer': '',
                'Aggregation': f"    {origin_name}",
                **_sum_numeric_columns(origin_df, numeric_cols)
            })

    if importer_totals:
        grand_total_row = {
            'Importer': IMPORTER_GLOBAL_LABEL,
            'Aggregation': '',
        }
        importer_totals_df = pd.DataFrame(importer_totals)
        for col in numeric_cols:
            grand_total_row[col] = pd.to_numeric(importer_totals_df[col], errors='coerce').sum(min_count=1)
        display_rows.insert(0, grand_total_row)

    display_df = pd.DataFrame(display_rows, columns=['Importer', 'Aggregation'] + numeric_cols)
    for col in numeric_cols:
        numeric_series = pd.to_numeric(display_df[col], errors='coerce').round(1)
        display_df[col] = numeric_series.where(pd.notnull(numeric_series), None)
    return (display_df, comparison_metadata) if return_metadata else display_df


def _create_period_analysis_table(
    display_df,
    delta_like_cols=None,
    view_type='absolute',
    volume_metric='mcm_d',
):
    """Create the combined overview period-analysis table."""
    delta_like_cols = set(delta_like_cols or [])
    delta_columns = {
        col for col in display_df.columns
        if (
            col not in IMPORTER_PERIOD_TEXT_COLUMNS and
            (
                _get_importer_period_column_family(col) in {'delta-mom', 'delta-yoy', 'delta-pbd'} or
                col in delta_like_cols
            )
        )
    }
    raw_field_map = {
        column_id: _get_importer_period_delta_raw_field(column_id)
        for column_id in delta_columns
    }
    display_precision = _get_importer_period_display_precision(
        view_type,
        volume_metric,
    )
    columns = []
    for col in display_df.columns:
        if col in IMPORTER_PERIOD_TEXT_COLUMNS:
            columns.append({
                'name': col,
                'id': col,
                'type': 'text',
                'cellClass': 'importer-period-text-cell'
            })
        else:
            columns.append({
                'name': col,
                'id': col,
                'type': 'numeric',
                'format': Format(
                    precision=display_precision,
                    scheme=Scheme.fixed,
                ),
                'cellClass': 'importer-period-number-cell'
            })

    columns = _apply_importer_period_column_classes(
        columns,
        display_df,
        delta_like_cols=delta_like_cols,
        raw_field_map=raw_field_map,
    )
    column_width_styles = _build_importer_period_column_width_styles(display_df, columns)
    delta_styles = _build_importer_period_delta_styles(
        display_df,
        columns,
        delta_like_cols=delta_like_cols,
        raw_field_map=raw_field_map
    )
    grid_display_df, grid_columns = _build_importer_period_grid_display(
        display_df,
        columns,
        view_type=view_type,
        delta_like_cols=delta_like_cols,
        raw_field_map=raw_field_map,
        volume_metric=volume_metric,
    )

    page_size = max(len(display_df), 1)
    return create_ag_grid_from_datatable(
        id='imp-overview-period-analysis-table',
        data=grid_display_df.to_dict('records'),
        columns=grid_columns,
        style_cell_conditional=column_width_styles,
        style_data_conditional=delta_styles,
        sort_action='none',
        page_action='none',
        page_size=page_size,
        fill_width=False,
        fixed_columns={'data': 1},
        className='importer-period-grid',
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
            'pagination': False,
            'suppressPaginationPanel': True,
            'enableCellTextSelection': True,
            'ensureDomOrder': True,
            'animateRows': False,
            'groupHeaderHeight': 28,
            'alwaysShowHorizontalScroll': False,
            'alwaysShowVerticalScroll': False,
        },
        rowClassRules={
            'importer-period-grand-total-row': "params.data && params.data['Importer'] === 'Global'",
            'importer-period-importer-total-row': (
                "params.data && params.data['Aggregation'] === 'Total'"
            ),
            'importer-period-child-row': (
                "params.data && params.data['Importer'] === '' && params.data['Aggregation'] !== ''"
            ),
        }
    )


def _build_period_table_footnote(
    rolling_avg_days,
    vol_label,
    comparison_basis='levels',
    snapshot_comparison=None,
    volume_metric='mcm_d',
):
    rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
    comparison_basis = _normalize_importer_period_comparison_basis(comparison_basis)
    snapshot_comparison = (
        snapshot_comparison
        if isinstance(snapshot_comparison, dict)
        else {}
    )
    current_snapshot = snapshot_comparison.get('current_snapshot')
    today = pd.Timestamp(
        (
            current_snapshot.get('snapshot_date_utc')
            if isinstance(current_snapshot, dict)
            else None
        )
        or datetime.now().date()
    ).date()
    date_7d_start = (today - timedelta(days=6)).strftime('%b %d, %Y')
    date_window_start = (today - timedelta(days=rolling_avg_days - 1)).strftime('%b %d, %Y')
    date_today = today.strftime('%b %d, %Y')
    date_window_y1_start = (
        today - timedelta(days=365) - timedelta(days=rolling_avg_days - 1)
    ).strftime('%b %d, %Y')
    date_window_y1_end = (today - timedelta(days=365)).strftime('%b %d, %Y')
    rolling_label = _format_importer_rolling_window_label(rolling_avg_days)
    comparison_note = ''
    if comparison_basis == 'previous_period':
        comparison_note = ' | Comparison: vs previous period'
    elif comparison_basis == 'same_period_last_year':
        comparison_note = ' | Comparison: vs previous year'
    unequal_window_note = (
        f' | Δ 7D-{rolling_label} is unavailable because period totals '
        'cover different horizons.'
        if (
            _is_importer_period_volume_metric(volume_metric)
            and rolling_avg_days != 7
        )
        else ''
    )

    def _format_snapshot_lineage(snapshot):
        if not isinstance(snapshot, dict):
            return None
        snapshot_date = snapshot.get('snapshot_date_utc')
        snapshot_timestamp = snapshot.get(
            'snapshot_timestamp_utc'
        )
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

    current_lineage = _format_snapshot_lineage(current_snapshot)
    baseline_lineage = _format_snapshot_lineage(
        snapshot_comparison.get('baseline_snapshot')
    )
    baseline_status = snapshot_comparison.get('status')
    business_day_gap = snapshot_comparison.get('business_day_gap')
    if baseline_status == 'exact' and baseline_lineage:
        baseline_note_text = (
            f'Current snapshot: {current_lineage} | '
            f'PBD baseline: {baseline_lineage} | '
            'PBD changes are current minus baseline and include '
            'window roll plus Kpler revisions.'
        )
        baseline_note_class = 'importer-period-baseline-status'
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
            'importer-period-baseline-status '
            'importer-period-baseline-status-warning'
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
            'importer-period-baseline-status '
            'importer-period-baseline-status-unavailable'
        )

    return html.Div(
        [
            html.P(
                [
                    html.Span('Note: ', className='importer-period-footnote-strong'),
                    html.Span(f'{rolling_label}: {date_window_start} to {date_today} | '),
                    html.Span(f'7D: {date_7d_start} to {date_today} | '),
                    html.Span(f'{rolling_label} Y-1: {date_window_y1_start} to {date_window_y1_end} | '),
                    html.Span(
                        f'Values shown in {vol_label}{comparison_note}'
                        f'{unequal_window_note}'
                    )
                ],
                className='importer-period-table-footnote-text'
            ),
            html.P(
                baseline_note_text,
                className=baseline_note_class,
                role=(
                    'alert'
                    if baseline_status == 'unavailable'
                    else None
                ),
            ),
        ],
        className='importer-period-table-footnote'
    )


layout = html.Div([
    dcc.Store(id='imp-overview-source-state-store', storage_type='memory'),
    dcc.Store(id='imp-overview-refresh-status-store', storage_type='memory'),
    dcc.Store(id='imp-overview-chart-entities-store', storage_type='memory'),
    dcc.Store(id='imp-overview-table-entities-store', storage_type='memory'),
    dcc.Store(id='imp-overview-demand-data-store', storage_type='memory'),
    dcc.Store(id='imp-overview-origin-continent-data-store', storage_type='memory'),
    dcc.Store(id='imp-overview-period-data-store', storage_type='memory'),
    dcc.Store(id='imp-overview-period-display-store', storage_type='memory'),
    dcc.Store(id='imp-overview-period-expanded-importers', storage_type='memory', data=[]),

    dcc.Download(id='imp-overview-download-demand-excel'),
    dcc.Download(id='imp-overview-download-origin-continent-excel'),
    dcc.Download(id='imp-overview-download-period-analysis-excel'),

    html.Div(
        _create_top_importers_selector_region(),
        className='professional-section-header importers-sticky-filter-bar',
        style={
            'display': 'flex',
            'gap': '12px',
            'alignItems': 'center',
            'flexWrap': 'wrap',
            'margin': '0',
        }
    ),

    html.Div([
        html.Div([
            html.Div(
                [
                    html.H3(
                        _format_importer_rolling_average_section_title(
                            'LNG Demand',
                            DEFAULT_IMPORTER_ROLLING_AVG_DAYS
                        ),
                        id='imp-overview-demand-rolling-section-title',
                        className='section-title-inline'
                    ),
                    html.Div(
                        [
                            html.Div('Years', className='importer-year-legend-title'),
                            dcc.Checklist(
                                id='imp-overview-demand-year-selector',
                                options=[],
                                value=[],
                                inline=True,
                                className='importer-year-checklist',
                                inputStyle={'display': 'none'},
                                labelStyle={'marginRight': '0'}
                            )
                        ],
                        className='importer-year-legend'
                    )
                ],
                className='importer-rolling-title-row'
            ),
            html.Button(
                'Export to Excel',
                id='imp-overview-export-demand-button',
                n_clicks=0,
                className='importer-rolling-export-button'
            ),
        ], className='inline-section-header importer-rolling-section-header'),
        dcc.Loading(
            id='imp-overview-demand-loading',
            children=[html.Div(id='imp-overview-demand-charts-container')],
            type='default'
        )
    ], className='main-section-container importer-rolling-section importer-demand-rolling-section'),

    html.Div([
        html.Div([
            html.Div(
                [
                    html.H3(
                        _format_importer_rolling_average_section_title(
                            'LNG Demand by Origin Continent',
                            DEFAULT_IMPORTER_ROLLING_AVG_DAYS
                        ),
                        id='imp-overview-origin-rolling-section-title',
                        className='section-title-inline'
                    ),
                    html.Div(
                        [
                            html.Div(
                                'Years',
                                className='importer-origin-year-selector-title continent-year-selector-title'
                            ),
                            dcc.Checklist(
                                id='imp-overview-origin-year-selector',
                                options=[],
                                value=[],
                                inline=True,
                                className='importer-origin-year-checklist continent-year-checklist',
                                inputStyle={'display': 'none'},
                                labelStyle={'marginRight': '0'}
                            )
                        ],
                        className='importer-origin-year-selector continent-year-selector'
                    ),
                    _build_origin_year_style_key()
                ],
                className='importer-origin-rolling-title-row continent-rolling-title-row'
            ),
            html.Div(
                [
                    dcc.RadioItems(
                        id='imp-overview-origin-continent-chart-type',
                        options=ORIGIN_CONTINENT_CHART_TYPE_OPTIONS,
                        value='absolute',
                        inline=True,
                        className='importer-origin-chart-type-selector continent-chart-type-selector',
                        inputStyle={'display': 'none'},
                        labelStyle={'marginRight': '0'}
                    ),
                    html.Button(
                        'Export to Excel',
                        id='imp-overview-export-origin-continent-button',
                        n_clicks=0,
                        className='importer-origin-rolling-export-button continent-rolling-export-button'
                    )
                ],
                className='importer-origin-rolling-controls continent-rolling-controls'
            ),
        ], className='inline-section-header importer-origin-rolling-section-header continent-rolling-section-header'),
        dcc.Loading(
            id='imp-overview-origin-continent-loading',
            children=[html.Div(id='imp-overview-origin-continent-charts-container')],
            type='default'
        )
    ], className='main-section-container importer-origin-rolling-section continent-rolling-section'),

    html.Div([
        html.Div([
            html.Div(
                [
                    html.H3('LNG Demand - Period Analysis', className='section-title-inline')
                ],
                className='importer-period-title-row'
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div('View', className='importer-period-control-label'),
                            dcc.RadioItems(
                                id='imp-overview-period-view-type',
                                options=IMPORTER_PERIOD_VIEW_OPTIONS,
                                value='absolute',
                                inline=True,
                                className='importer-period-view-selector',
                                inputStyle={'display': 'none'},
                                labelStyle={'marginRight': '0'}
                            )
                        ],
                        className='importer-period-control-group importer-period-view-control-group'
                    ),
                    html.Div(
                        [
                            html.Div('Aggregation', className='importer-period-control-label'),
                            dcc.Dropdown(
                                id='imp-overview-origin-level-dropdown',
                                options=ORIGIN_LEVEL_OPTIONS,
                                value='origin_shipping_region',
                                clearable=False,
                                className='filter-dropdown importer-period-origin-dropdown'
                            )
                        ],
                        className='importer-period-control-group importer-period-origin-control-group'
                    ),
                    html.Div(
                        [
                            html.Div('Comparison', className='importer-period-control-label'),
                            dcc.RadioItems(
                                id='imp-overview-period-comparison-basis',
                                options=IMPORTER_PERIOD_COMPARISON_BASIS_OPTIONS,
                                value='levels',
                                inline=True,
                                className='importer-period-view-selector importer-period-comparison-selector',
                                inputStyle={'display': 'none'},
                                labelStyle={'marginRight': '0'}
                            )
                        ],
                        className='importer-period-control-group importer-period-comparison-control-group'
                    ),
                    html.Div(
                        [
                            html.Div('Periods', className='importer-period-control-label'),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span('Qtrs', className='importer-period-mini-label'),
                                            dcc.Dropdown(
                                                id='imp-overview-period-quarter-count-dropdown',
                                                options=_build_importer_period_count_options(IMPORTER_PERIOD_MAX_QUARTER_COUNT),
                                                value=IMPORTER_PERIOD_DEFAULT_QUARTER_COUNT,
                                                clearable=False,
                                                searchable=False,
                                                className='importer-period-count-dropdown'
                                            )
                                        ],
                                        className='importer-period-count-selector'
                                    ),
                                    html.Div(
                                        [
                                            html.Span('Months', className='importer-period-mini-label'),
                                            dcc.Dropdown(
                                                id='imp-overview-period-month-count-dropdown',
                                                options=_build_importer_period_count_options(IMPORTER_PERIOD_MAX_MONTH_COUNT),
                                                value=IMPORTER_PERIOD_DEFAULT_MONTH_COUNT,
                                                clearable=False,
                                                searchable=False,
                                                className='importer-period-count-dropdown'
                                            )
                                        ],
                                        className='importer-period-count-selector'
                                    ),
                                    html.Div(
                                        [
                                            html.Span('Weeks', className='importer-period-mini-label'),
                                            dcc.Dropdown(
                                                id='imp-overview-period-week-count-dropdown',
                                                options=_build_importer_period_count_options(IMPORTER_PERIOD_MAX_WEEK_COUNT),
                                                value=IMPORTER_PERIOD_DEFAULT_WEEK_COUNT,
                                                clearable=False,
                                                searchable=False,
                                                className='importer-period-count-dropdown'
                                            )
                                        ],
                                        className='importer-period-count-selector'
                                    )
                                ],
                                className='importer-period-count-selectors'
                            )
                        ],
                        className='importer-period-control-group'
                    ),
                    html.Button(
                        'Export to Excel',
                        id='imp-overview-export-period-analysis-button',
                        n_clicks=0,
                        className='importer-period-export-button'
                    )
                ],
                className='importer-period-controls'
            ),
        ], className='inline-section-header importer-period-section-header'),
        dcc.Loading(
            id='imp-overview-period-analysis-loading',
            children=[html.Div(id='imp-overview-period-analysis-container', className='importer-period-table-container')],
            type='default'
        )
    ], className='main-section-container importer-period-section')
], className='importers-page')


@callback(
    Output('imp-overview-demand-rolling-section-title', 'children'),
    Output('imp-overview-origin-rolling-section-title', 'children'),
    Input('imp-overview-rolling-window-days-input', 'value'),
    Input('imp-overview-volume-metric-dropdown', 'value'),
    prevent_initial_call=False
)
def update_importer_rolling_section_titles(rolling_avg_days, volume_metric):
    return (
        _format_importer_rolling_average_section_title(
            'LNG Demand',
            rolling_avg_days,
            volume_metric,
        ),
        _format_importer_rolling_average_section_title(
            'LNG Demand by Origin Continent',
            rolling_avg_days,
            volume_metric,
        ),
    )


def _build_importers_source_payload(source_state=None):
    """Load immutable overview catalog data in two database reads."""
    source_state = _normalize_importers_source_state(source_state)
    current_snapshot = source_state.get('current_snapshot') or {}
    if (
        source_state.get('format') == IMPORTERS_SOURCE_STATE_FORMAT
        and current_snapshot.get('snapshot_timestamp_utc')
        and current_snapshot.get('snapshot_date_utc')
    ):
        catalog_ranking_loader = (
            lambda: _fetch_importers_catalog_ranking_source_df(
                current_snapshot['snapshot_timestamp_utc'],
                current_snapshot['snapshot_date_utc'],
            )
        )
    else:
        catalog_ranking_loader = (
            _fetch_importers_catalog_ranking_source_df
        )
    loaders = {
        'catalog_ranking': catalog_ranking_loader,
        'mappings': _fetch_importers_mapping_source_df,
    }
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix='importers-source',
    ) as executor:
        futures = {
            name: executor.submit(loader)
            for name, loader in loaders.items()
        }
        loaded = {
            name: futures[name].result()
            for name in loaders
        }

    catalog_df, ranking_df = (
        _build_destination_catalog_and_ranking_from_sources(
            loaded['catalog_ranking'],
            loaded['mappings'],
        )
    )
    return {
        'catalog_df': catalog_df,
        'ranking_df': ranking_df,
        'scoped_trades_df': pd.DataFrame(),
        'source_state': source_state,
    }


def _build_importers_overview_payload_from_source(
    source_payload,
    classification_mode,
    rolling_avg_days,
):
    rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
    chart_query_start_date = _get_importer_chart_query_start_date(
        rolling_avg_days
    )
    normalized_source_state = _normalize_importers_source_state(
        source_payload.get('source_state')
        if isinstance(source_payload, dict)
        else None
    )
    current_snapshot = normalized_source_state.get('current_snapshot') or {}
    snapshot_timestamp_utc = current_snapshot.get('snapshot_timestamp_utc')
    current_date = (
        current_snapshot.get('snapshot_date_utc')
        or normalized_source_state.get('as_of_date')
    )
    catalog_df = source_payload['catalog_df']
    ranking_df = source_payload['ranking_df']
    scoped_trades_df = source_payload.get('scoped_trades_df')
    selected_aggregation = (
        _classification_mode_to_destination_aggregation(
            classification_mode
        )
    )
    destination_countries = sorted(
        catalog_df['destination_country_name']
        .dropna()
        .unique()
        .tolist()
    ) if not catalog_df.empty else []
    if not isinstance(scoped_trades_df, pd.DataFrame):
        scoped_trades_df = pd.DataFrame()
    source_scoped_trades_provided = not scoped_trades_df.empty
    if scoped_trades_df.empty and destination_countries:
        query_kwargs = {
            'min_end_date': chart_query_start_date,
            'include_destination_context': True,
        }
        if snapshot_timestamp_utc is not None:
            query_kwargs['snapshot_timestamp_utc'] = snapshot_timestamp_utc
        if current_date is not None:
            query_kwargs['max_end_date'] = _get_importer_chart_query_end_date(
                current_date
            )
        net_scopes = [selected_aggregation]
        if selected_aggregation != 'country':
            net_scopes.append('country')
        with ThreadPoolExecutor(
            max_workers=len(net_scopes),
            thread_name_prefix='importers-net-scope',
        ) as executor:
            futures = {
                net_scope: executor.submit(
                    _fetch_importer_scoped_trades,
                    engine,
                    destination_countries,
                    selected_destination_aggregation=net_scope,
                    **query_kwargs,
                )
                for net_scope in net_scopes
            }
            scoped_trades_df = futures[
                selected_aggregation
            ].result()
            global_scoped_trades_df = futures.get(
                'country'
            )
            if global_scoped_trades_df is not None:
                global_scoped_trades_df = (
                    global_scoped_trades_df.result()
                )
            else:
                global_scoped_trades_df = scoped_trades_df
    else:
        global_scoped_trades_df = scoped_trades_df
    if (
        selected_aggregation != 'country'
        and destination_countries
        and not source_scoped_trades_provided
        and global_scoped_trades_df is scoped_trades_df
    ):
        global_scoped_trades_df = _fetch_importer_scoped_trades(
            engine,
            destination_countries,
            selected_destination_aggregation='country',
            **query_kwargs,
        )
    table_entities = _build_destination_entities(
        classification_mode,
        limit=None,
        catalog_df=catalog_df,
        ranking_df=ranking_df
    )
    chart_limit = None if classification_mode == 'Classification Level 1' else TOP_IMPORTER_CHART_COUNT
    chart_entities = _build_destination_entities(
        classification_mode,
        limit=chart_limit,
        catalog_df=catalog_df,
        ranking_df=ranking_df,
        include_global=True,
        include_rest=classification_mode != 'Classification Level 1'
    )
    demand_charts_data, origin_continent_charts_data = _build_chart_data_payload(
        chart_entities,
        classification_mode,
        rolling_avg_days,
        scoped_trades_df=scoped_trades_df,
        global_scoped_trades_df=global_scoped_trades_df,
        current_date=current_date,
        snapshot_timestamp_utc=snapshot_timestamp_utc,
    )
    demand_cube = _pack_record_mapping(demand_charts_data)
    origin_cube = _pack_record_mapping(
        origin_continent_charts_data
    )
    return {
        'chart_entities': chart_entities,
        'table_entities': table_entities,
        'demand_cube': demand_cube,
        'origin_cube': origin_cube,
        'demand_years': _get_importer_chart_available_years(
            demand_charts_data
        ),
        'origin_years': _get_importer_chart_available_years(
            origin_continent_charts_data
        ),
    }


def _build_importers_overview_payload(classification_mode, rolling_avg_days):
    """Build a standalone overview payload through the optimized source path."""
    return _build_importers_overview_payload_from_source(
        _build_importers_source_payload(),
        classification_mode,
        rolling_avg_days,
    )


def _normalize_importers_source_state(source_state):
    source_state = (
        dict(source_state)
        if isinstance(source_state, dict)
        else {}
    )
    watermark = source_state.get('watermark')
    if watermark is None:
        watermark = source_state.get('request_token')
    normalized = {
        'watermark': str(watermark) if watermark is not None else None,
        'as_of_date': (
            source_state.get('as_of_date')
            or datetime.now().date().isoformat()
        ),
    }
    if (
        not _revision_aware_refresh_enabled()
        and source_state.get('refresh_generation') is not None
    ):
        normalized['refresh_generation'] = int(
            source_state['refresh_generation']
        )
    if (
        source_state.get('format') == IMPORTERS_SOURCE_STATE_FORMAT
    ):
        normalized.update({
            'format': IMPORTERS_SOURCE_STATE_FORMAT,
            'current_snapshot': (
                dict(source_state['current_snapshot'])
                if isinstance(
                    source_state.get('current_snapshot'),
                    dict,
                )
                else None
            ),
            'baseline_snapshot': (
                dict(source_state['baseline_snapshot'])
                if isinstance(
                    source_state.get('baseline_snapshot'),
                    dict,
                )
                else None
            ),
            'baseline_status': source_state.get(
                'baseline_status',
                'unavailable',
            ),
            'business_day_gap': source_state.get(
                'business_day_gap'
            ),
        })
    return normalized


def _importers_source_snapshot_key(source_state):
    source_state = _normalize_importers_source_state(source_state)
    return _build_source_key(
        IMPORTERS_SOURCE_NAMESPACE,
        source_state,
    )


def _load_importers_source_snapshot(source_state):
    source_state = _normalize_importers_source_state(source_state)

    def source_builder():
        payload = (
            _build_importers_source_payload(source_state)
            if source_state.get('format')
            == IMPORTERS_SOURCE_STATE_FORMAT
            else _build_importers_source_payload()
        )
        if (
            _revision_aware_refresh_enabled()
            and isinstance(source_state.get('current_snapshot'), dict)
        ):
            current_state = _normalize_importers_source_state(
                _build_importers_source_state(
                    _fetch_importers_source_watermark(),
                    refresh_token=None,
                )
            )
            if current_state != source_state:
                raise _SnapshotUnavailable(
                    'Importer sources changed during snapshot construction. '
                    'Refresh and retry.'
                )
        return payload

    return _get_or_build_snapshot(
        engine,
        namespace=IMPORTERS_SOURCE_NAMESPACE,
        source_key=_importers_source_snapshot_key(source_state),
        builder=source_builder,
        manifest={'source_state': source_state},
    )


def _importers_overview_snapshot_key(
    source_reference,
    classification_mode,
    rolling_avg_days,
):
    dependency = {
        'namespace': source_reference.get('namespace'),
        'source_key': source_reference.get('source_key'),
        'revision': source_reference.get('revision'),
    }
    return _build_source_key(
        IMPORTERS_OVERVIEW_NAMESPACE,
        dependency,
        classification_mode,
        normalize_importer_rolling_avg_days(rolling_avg_days),
    )


@callback(
    Output('imp-overview-source-state-store', 'data'),
    Output('imp-overview-refresh-status-store', 'data'),
    Input('global-refresh-button', 'n_clicks'),
    State('imp-overview-source-state-store', 'data'),
)
def load_importers_overview_source_state(
    n_clicks,
    current_source_state=None,
):
    refresh_status = {
        'format': 'dashboard-source-refresh-status-v1',
        'refresh_generation': int(n_clicks or 0),
        'checked_at': datetime.now().astimezone().isoformat(),
    }
    try:
        source_pair = _fetch_importers_source_watermark()
    except Exception:
        source_pair = None
        refresh_status['status'] = 'unavailable'
    else:
        refresh_status['status'] = 'checked'
    source_state = _build_importers_source_state(
        source_pair,
        refresh_token=None,
    )
    if source_pair is None:
        source_state['request_token'] = uuid.uuid4().hex
    if not _revision_aware_refresh_enabled():
        source_state['refresh_generation'] = int(n_clicks or 0)
    if (
        _revision_aware_refresh_enabled()
        and
        isinstance(current_source_state, dict)
        and _normalize_importers_source_state(current_source_state)
        == _normalize_importers_source_state(source_state)
    ):
        return no_update, refresh_status
    return source_state, refresh_status


def _importers_overview_triggered_id():
    try:
        return callback_context.triggered_id
    except MissingCallbackContextException:
        return None


def _source_state_for_importers_overview_load(source_state):
    source_state = (
        dict(source_state)
        if isinstance(source_state, dict)
        else source_state
    )
    if not (
        isinstance(source_state, dict)
        and source_state.get('refresh_token')
    ):
        return source_state
    triggered_id = _importers_overview_triggered_id()
    if triggered_id in (
        None,
        'imp-overview-source-state-store',
    ):
        return source_state
    source_state.pop('refresh_token', None)
    return source_state


@callback(
    Output('imp-overview-chart-entities-store', 'data'),
    Output('imp-overview-table-entities-store', 'data'),
    Output('imp-overview-demand-data-store', 'data'),
    Output('imp-overview-origin-continent-data-store', 'data'),
    Input('imp-overview-source-state-store', 'data'),
    Input('imp-overview-classification-mode', 'value'),
    Input('imp-overview-rolling-window-days-input', 'value'),
    prevent_initial_call=False
)
@log_callback_timing("importers.overview_source_load")
def refresh_overview_data(source_state, classification_mode, rolling_avg_days):
    """Load the importer overview entities and compact server-side chart datasets."""
    if not source_state:
        raise PreventUpdate
    try:
        rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
        source_state = _source_state_for_importers_overview_load(
            source_state
        )
        source_reference, source_payload = (
            _load_importers_source_snapshot(source_state)
        )
        if not _snapshot_is_resolvable(source_reference):
            raise _SnapshotUnavailable(
                IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE
            )
        source_payload = _resolve_importers_source_store(
            source_reference
        )
        source_key = _importers_overview_snapshot_key(
            source_reference,
            classification_mode,
            rolling_avg_days,
        )
        reference, payload = _get_or_build_snapshot(
            engine,
            namespace=IMPORTERS_OVERVIEW_NAMESPACE,
            source_key=source_key,
            builder=lambda: _prepare_importers_overview_snapshot_payload(
                _build_importers_overview_payload_from_source(
                    source_payload,
                    classification_mode,
                    rolling_avg_days,
                )
            ),
            manifest={
                'source_reference': {
                    'namespace': source_reference.get('namespace'),
                    'source_key': source_reference.get('source_key'),
                    'revision': source_reference.get('revision'),
                },
                'classification_mode': classification_mode,
                'rolling_avg_days': rolling_avg_days,
            },
        )

        if not _snapshot_is_resolvable(reference):
            raise _SnapshotUnavailable(
                IMPORTERS_SNAPSHOT_RECOVERY_MESSAGE
            )

        return (
            _with_snapshot_slot(reference, 'chart_entities'),
            _with_snapshot_slot(reference, 'table_entities'),
            _with_snapshot_slot(reference, 'demand_cube'),
            _with_snapshot_slot(reference, 'origin_cube'),
        )
    except _SnapshotUnavailable:
        raise
    except Exception:
        return [], [], {}, {}


@callback(
    Output('imp-overview-demand-year-selector', 'options'),
    Output('imp-overview-demand-year-selector', 'value'),
    Input('imp-overview-demand-data-store', 'data'),
    State('imp-overview-demand-year-selector', 'value'),
    prevent_initial_call=False
)
def update_demand_year_selector_options(charts_data, selected_years):
    try:
        available_years = _resolve_importers_years_store(
            charts_data,
            'demand_years',
        )
    except _SnapshotUnavailable:
        return _importers_snapshot_recovery_selector_result()
    if not available_years:
        return [], []

    color_by_year = _get_importer_chart_color_map(available_years)
    options = [
        {
            'label': html.Span(
                [
                    html.Span(
                        className='importer-year-chip-swatch',
                        style={'backgroundColor': color_by_year.get(year, '#64748b')}
                    ),
                    html.Span(year, className='importer-year-chip-text')
                ],
                className='importer-year-chip-label'
            ),
            'value': year
        }
        for year in available_years
    ]
    selected = _normalize_importer_chart_selected_years(selected_years, available_years)
    return options, selected


@callback(
    Output('imp-overview-demand-charts-container', 'children'),
    Input('imp-overview-demand-data-store', 'data'),
    Input('imp-overview-chart-entities-store', 'data'),
    Input('imp-overview-volume-metric-dropdown', 'value'),
    Input('imp-overview-demand-year-selector', 'value'),
    Input('imp-overview-rolling-window-days-input', 'value'),
    prevent_initial_call=False
)
@log_callback_timing("importers.demand_charts_render")
def update_demand_charts(charts_data, importer_entities, volume_metric, selected_years, rolling_avg_days):
    """Render the demand chart grid using the upgraded exporter-page pattern."""
    try:
        charts_data = _resolve_importers_chart_store(charts_data)
        importer_entities = _resolve_importers_entities_store(
            importer_entities,
            'chart_entities',
        )
    except _SnapshotUnavailable:
        return _importers_snapshot_recovery_notice()
    if not charts_data or not importer_entities:
        return html.Div('No data available', className='importer-rolling-empty-state')

    vol_label = get_volume_metric_info(volume_metric)['label']
    rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
    charts = []
    for entity in importer_entities:
        entity_name = entity['label']
        entity_data = charts_data.get(entity_name, [])
        prepared_df = _prepare_importer_demand_chart_dataframe(
            entity_data,
            volume_metric,
            rolling_avg_days,
        )
        fig = _create_importer_demand_chart_from_df(
            prepared_df,
            vol_label,
            selected_years,
            volume_metric,
        )
        metrics = _get_importer_demand_chart_header_metrics_from_df(
            prepared_df,
            selected_years,
        )
        current_value = _format_importer_chart_current_value(
            metrics,
            vol_label,
            volume_metric,
        )
        card_class_name = (
            'importer-rolling-card importer-rolling-card-primary'
            if entity.get('is_global')
            else 'importer-rolling-card'
        )

        charts.append(
            html.Div([
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5(entity_name, className='importer-rolling-card-title'),
                                html.Span(current_value, className='importer-rolling-current-value') if current_value else None
                            ],
                            className='importer-rolling-card-title-group'
                        ),
                        _build_importer_chart_delta_indicators(
                            metrics,
                            volume_metric,
                        )
                    ],
                    className='importer-rolling-card-header'
                ),
                dcc.Graph(
                    id=f'imp-overview-demand-chart-{_slugify_filename_label(entity_name).lower()}',
                    figure=fig,
                    config={'displayModeBar': False, 'responsive': True},
                    className='importer-rolling-graph',
                    style={'height': '328px', 'width': '100%'}
                )
            ], className=card_class_name)
        )

    return html.Div(charts, className='importer-rolling-grid')


@callback(
    Output('imp-overview-origin-year-selector', 'options'),
    Output('imp-overview-origin-year-selector', 'value'),
    Input('imp-overview-origin-continent-data-store', 'data'),
    State('imp-overview-origin-year-selector', 'value'),
    prevent_initial_call=False
)
def update_origin_year_selector_options(charts_data, selected_years):
    try:
        available_years = _resolve_importers_years_store(
            charts_data,
            'origin_years',
        )
    except _SnapshotUnavailable:
        return _importers_snapshot_recovery_selector_result()
    if not available_years:
        return [], []

    options = [
        {
            'label': html.Span(
                year,
                className='importer-origin-year-chip-text continent-year-chip-text'
            ),
            'value': year
        }
        for year in available_years
    ]
    selected = _normalize_importer_chart_selected_years(selected_years, available_years)
    return options, selected


@callback(
    Output('imp-overview-origin-continent-charts-container', 'children'),
    Input('imp-overview-origin-continent-data-store', 'data'),
    Input('imp-overview-chart-entities-store', 'data'),
    Input('imp-overview-volume-metric-dropdown', 'value'),
    Input('imp-overview-origin-year-selector', 'value'),
    Input('imp-overview-origin-continent-chart-type', 'value'),
    Input('imp-overview-rolling-window-days-input', 'value'),
    prevent_initial_call=False
)
@log_callback_timing("importers.origin_charts_render")
def update_origin_continent_charts(charts_data, importer_entities, volume_metric, selected_years,
                                   chart_type, rolling_avg_days):
    """Render the origin-continent chart grid using the upgraded exporter-page pattern."""
    try:
        charts_data = _resolve_importers_chart_store(charts_data)
        importer_entities = _resolve_importers_entities_store(
            importer_entities,
            'chart_entities',
        )
    except _SnapshotUnavailable:
        return _importers_snapshot_recovery_notice()
    if not charts_data or not importer_entities:
        return html.Div(
            'No data available',
            className='importer-origin-rolling-empty-state continent-rolling-empty-state'
        )

    charts = []
    kpi_rows = []
    vol_label = get_volume_metric_info(volume_metric)['label']
    rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
    for entity in importer_entities:
        entity_name = entity['label']
        entity_data = charts_data.get(entity_name, [])
        prepared_df, metric_column = (
            _prepare_importer_origin_chart_dataframe(
                entity_data,
                chart_type,
                volume_metric,
                rolling_avg_days,
            )
        )
        fig = _create_importer_origin_continent_chart_from_df(
            prepared_df,
            metric_column,
            vol_label,
            selected_years,
            volume_metric,
        )
        kpi_rows.append({
            'entity': entity_name,
            'metrics': _calculate_origin_continent_kpis_from_df(
                prepared_df,
                metric_column,
                vol_label,
                chart_type=chart_type,
                selected_years=selected_years,
                volume_metric=volume_metric,
            )
        })
        card_class_name = (
            'importer-origin-rolling-card importer-origin-rolling-card-primary '
            'continent-rolling-card continent-rolling-card-primary'
            if entity.get('is_global')
            else 'importer-origin-rolling-card continent-rolling-card'
        )
        charts.append(
            html.Div([
                html.Div(
                    [
                        html.H5(
                            entity_name,
                            className='importer-origin-rolling-card-title continent-rolling-card-title'
                        )
                    ],
                    className='importer-origin-rolling-card-header continent-rolling-card-header'
                ),
                dcc.Graph(
                    id=f'imp-overview-origin-continent-chart-{_slugify_filename_label(entity_name).lower()}',
                    figure=fig,
                    config={'displayModeBar': False, 'responsive': True},
                    className='importer-origin-rolling-graph continent-rolling-graph',
                    style={'height': '328px', 'width': '100%'}
                )
            ], className=card_class_name)
        )

    return html.Div(
        [
            html.Div(
                _build_origin_kpi_summary_table(kpi_rows),
                className='importer-origin-kpi-summary-panel continent-rolling-summary-panel'
            ),
            html.Div(
                charts,
                className='importer-origin-rolling-grid importer-origin-rolling-grid-wrap continent-rolling-grid'
            )
        ],
        className='importer-origin-rolling-content importer-origin-rolling-content-wrap continent-rolling-content'
    )


@callback(
    Output('imp-overview-period-expanded-importers', 'data', allow_duplicate=True),
    Input('imp-overview-origin-level-dropdown', 'value'),
    Input('imp-overview-classification-mode', 'value'),
    Input('imp-overview-table-entities-store', 'data'),
    prevent_initial_call=True
)
def reset_period_expansion_state(_origin_level, _classification_mode, _importer_entities):
    """Reset period-analysis expansion when the origin-level control changes."""
    return []


@callback(
    Output('imp-overview-period-data-store', 'data'),
    Input('imp-overview-table-entities-store', 'data'),
    Input('imp-overview-classification-mode', 'value'),
    Input('imp-overview-origin-level-dropdown', 'value'),
    Input('imp-overview-rolling-window-days-input', 'value'),
    Input('imp-overview-origin-country-grouping-dropdown', 'value'),
    State('imp-overview-source-state-store', 'data'),
    prevent_initial_call=False
)
@log_callback_timing("importers.period_source_load")
def refresh_period_data(importer_entities, classification_mode, origin_level, rolling_avg_days,
                        origin_country_grouping_mode, source_state=None):
    """Load the raw period-analysis payload for the overview importers."""
    if not importer_entities:
        return _empty_importer_period_payload(origin_country_grouping_mode)

    try:
        importer_entities = _resolve_importers_entities_store(
            importer_entities,
            'table_entities',
        )
        if not importer_entities:
            return _empty_importer_period_payload(origin_country_grouping_mode)
        normalized_rolling_days = normalize_importer_rolling_avg_days(rolling_avg_days)
        normalized_source_state = _normalize_importers_source_state(
            source_state
        )
        source_key = _build_source_key(
            IMPORTERS_PERIOD_NAMESPACE,
            normalized_source_state,
            importer_entities,
            classification_mode,
            origin_level,
            origin_country_grouping_mode,
            normalized_rolling_days,
        )
        reference, payload = _get_or_build_snapshot(
            engine,
            namespace=IMPORTERS_PERIOD_NAMESPACE,
            source_key=source_key,
            builder=lambda: _build_period_payload(
                importer_entities,
                classification_mode,
                origin_level,
                origin_country_grouping_mode,
                normalized_rolling_days,
                normalized_source_state,
            ),
            manifest={
                'format': IMPORTERS_PERIOD_PAYLOAD_FORMAT,
                'source_state': normalized_source_state,
                'classification_mode': classification_mode,
                'origin_level': origin_level,
                'origin_country_grouping_mode': origin_country_grouping_mode,
                'rolling_avg_days': normalized_rolling_days,
            },
        )
        return reference if _snapshot_is_shared(reference) else payload
    except _SnapshotUnavailable:
        raise
    except Exception:
        return _empty_importer_period_payload(origin_country_grouping_mode)


@callback(
    Output('imp-overview-period-analysis-container', 'children'),
    Output('imp-overview-period-display-store', 'data'),
    Input('imp-overview-period-data-store', 'data'),
    Input('imp-overview-period-expanded-importers', 'data'),
    Input('imp-overview-table-entities-store', 'data'),
    Input('imp-overview-volume-metric-dropdown', 'value'),
    Input('imp-overview-rolling-window-days-input', 'value'),
    Input('imp-overview-origin-country-grouping-dropdown', 'value'),
    Input('imp-overview-period-view-type', 'value'),
    Input('imp-overview-period-comparison-basis', 'value'),
    Input('imp-overview-period-quarter-count-dropdown', 'value'),
    Input('imp-overview-period-month-count-dropdown', 'value'),
    Input('imp-overview-period-week-count-dropdown', 'value'),
    prevent_initial_call=False
)
@log_callback_timing("importers.period_table_render")
def update_period_analysis_table(period_payload, expanded_importers, importer_entities, volume_metric,
                                 rolling_avg_days, origin_country_grouping_mode, view_type, comparison_basis,
                                 quarter_count, month_count, week_count):
    """Render the combined importer overview period-analysis table."""
    try:
        importer_entities = _resolve_importers_entities_store(
            importer_entities,
            'table_entities',
        )
    except _SnapshotUnavailable:
        return _importers_snapshot_recovery_notice(), []
    if not period_payload or not importer_entities:
        message = html.Div('No data available for the selected configuration.', style={'textAlign': 'center', 'padding': '20px'})
        return message, []

    period_payload = _resolve_importers_period_store(period_payload)
    snapshot_comparison = (
        period_payload.get('snapshot_comparison')
        if isinstance(period_payload, dict)
        else {}
    )

    rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
    origin_country_grouping_mode = _normalize_importer_period_origin_grouping(origin_country_grouping_mode)
    view_type = _normalize_importer_period_view_type(view_type)
    comparison_basis = _normalize_importer_period_comparison_basis(comparison_basis)
    if isinstance(period_payload, dict):
        active_grouping_mode = period_payload.get('active_grouping_mode')
        if active_grouping_mode and active_grouping_mode != origin_country_grouping_mode:
            raise PreventUpdate
    resolved_period_payload = _resolve_importer_period_payload(
        period_payload,
        origin_country_grouping_mode
    )
    if not resolved_period_payload:
        message = html.Div('No period-analysis data is available for the overview importers.', style={'textAlign': 'center', 'padding': '20px'})
        return message, []

    display_df, comparison_metadata = _build_period_display_df(
        resolved_period_payload,
        expanded_importers=expanded_importers,
        rolling_avg_days=rolling_avg_days,
        quarter_count=quarter_count,
        month_count=month_count,
        week_count=week_count,
        comparison_basis=comparison_basis,
        return_metadata=True
    )
    if display_df.empty:
        message = html.Div('No period-analysis data is available for the overview importers.', style={'textAlign': 'center', 'padding': '20px'})
        return message, []

    pbd_available = (
        isinstance(snapshot_comparison, dict)
        and snapshot_comparison.get('status') in {
            'exact',
            'fallback',
        }
    )
    if view_type == 'percentage':
        pbd_percentage_df = (
            _apply_importer_period_pbd_percentage_view(
                display_df
            )
            if pbd_available
            else None
        )
        level_delta_cols = [
            col for col in display_df.columns
            if col.startswith('Δ ')
        ]
        if level_delta_cols:
            display_df = display_df.drop(columns=level_delta_cols)
        numeric_cols = [col for col in display_df.columns if col not in IMPORTER_PERIOD_TEXT_COLUMNS]
        display_df = _apply_importer_period_percentage_view(display_df, numeric_cols)
        if pbd_percentage_df is not None:
            for column_name in (
                *IMPORTER_PERIOD_PBD_CURRENT_COLUMNS,
                *IMPORTER_PERIOD_PBD_REFERENCE_COLUMNS,
            ):
                if column_name in display_df.columns:
                    display_df[column_name] = (
                        pbd_percentage_df[column_name]
                    )
            display_df = _recalculate_importer_period_pbd_deltas(
                display_df
            )
        vol_label = 'market share (%)'
    else:
        vol_label = get_volume_metric_info(volume_metric)['label']
        display_df = _convert_importer_period_absolute_volume_metric(
            display_df,
            volume_metric,
            rolling_avg_days,
        )

    if not pbd_available:
        for column_name in (
            *IMPORTER_PERIOD_PBD_CURRENT_COLUMNS,
            *IMPORTER_PERIOD_PBD_REFERENCE_COLUMNS,
            *IMPORTER_PERIOD_PBD_DELTA_COLUMNS,
        ):
            if column_name in display_df.columns:
                display_df[column_name] = np.nan

    display_df, comparison_delta_cols = _apply_importer_period_comparison(
        display_df,
        comparison_metadata
    )
    display_df = display_df.drop(
        columns=[
            *IMPORTER_PERIOD_PBD_CURRENT_COLUMNS,
            *IMPORTER_PERIOD_PBD_REFERENCE_COLUMNS,
        ],
        errors='ignore',
    )
    pbd_delta_cols = [
        column_name
        for column_name in IMPORTER_PERIOD_PBD_DELTA_COLUMNS
        if column_name in display_df.columns
    ]
    if pbd_delta_cols:
        non_pbd_columns = [
            column_name
            for column_name in display_df.columns
            if column_name not in pbd_delta_cols
        ]
        display_df = display_df[
            [*non_pbd_columns, *pbd_delta_cols]
        ]
    all_delta_cols = [
        *comparison_delta_cols,
        *pbd_delta_cols,
    ]
    for col in [col for col in display_df.columns if col not in IMPORTER_PERIOD_TEXT_COLUMNS]:
        numeric_series = pd.to_numeric(display_df[col], errors='coerce').round(1)
        display_df[col] = numeric_series.where(pd.notnull(numeric_series), None)

    table_shell = html.Div(
        [
            _create_period_analysis_table(
                display_df,
                delta_like_cols=all_delta_cols,
                view_type=view_type,
                volume_metric=volume_metric,
            ),
            _build_period_table_footnote(
                rolling_avg_days,
                vol_label,
                comparison_basis,
                snapshot_comparison,
                volume_metric if view_type == 'absolute' else 'mcm_d',
            )
        ],
        className='importer-period-table-shell'
    )
    return table_shell, display_df.to_dict('records')


@callback(
    Output('imp-overview-period-expanded-importers', 'data', allow_duplicate=True),
    Input('imp-overview-period-analysis-table', 'cellClicked'),
    State('imp-overview-period-analysis-table', 'virtualRowData'),
    State('imp-overview-period-expanded-importers', 'data'),
    prevent_initial_call=True
)
def toggle_period_row_expansion(cell_clicked, table_data, expanded_importers):
    """Toggle the single importer expansion level within the overview period table."""
    active_cell = ag_grid_cell_clicked_to_active_cell(cell_clicked)
    if not active_cell or not table_data:
        raise PreventUpdate

    row_index = active_cell.get('row')
    if row_index is None:
        raise PreventUpdate

    display_df = pd.DataFrame(table_data)
    if display_df.empty or row_index >= len(display_df):
        raise PreventUpdate

    expanded_importers = list(expanded_importers or [])
    clicked_row = display_df.iloc[row_index]

    importer_value = str(clicked_row.get('Importer', ''))
    if _is_importer_expandable_label(importer_value):
        importer_name = _strip_importer_expand_marker(importer_value)
        if importer_name in expanded_importers:
            expanded_importers.remove(importer_name)
        else:
            expanded_importers.append(importer_name)
        return expanded_importers

    raise PreventUpdate


@callback(
    Output('imp-overview-download-demand-excel', 'data'),
    Input('imp-overview-export-demand-button', 'n_clicks'),
    State('imp-overview-demand-data-store', 'data'),
    State('imp-overview-volume-metric-dropdown', 'value'),
    State('imp-overview-demand-year-selector', 'value'),
    State('imp-overview-rolling-window-days-input', 'value'),
    prevent_initial_call=True
)
def export_demand_to_excel(n_clicks, charts_data, volume_metric, selected_years, rolling_avg_days):
    """Export the currently rendered demand-chart data."""
    if not n_clicks:
        raise PreventUpdate

    charts_data = _resolve_importers_chart_store(charts_data)

    rolling_label = _format_importer_rolling_window_label(rolling_avg_days)
    export_df = _build_chart_export_df(
        charts_data,
        volume_metric,
        selected_years,
        rolling_avg_days=rolling_avg_days,
    )
    if export_df.empty:
        raise PreventUpdate

    return _send_export_dataframe(
        export_df,
        f'importers_lng_demand_{rolling_label.lower()}_rolling',
        'Demand'
    )


@callback(
    Output('imp-overview-download-origin-continent-excel', 'data'),
    Input('imp-overview-export-origin-continent-button', 'n_clicks'),
    State('imp-overview-origin-continent-data-store', 'data'),
    State('imp-overview-volume-metric-dropdown', 'value'),
    State('imp-overview-origin-year-selector', 'value'),
    State('imp-overview-origin-continent-chart-type', 'value'),
    State('imp-overview-rolling-window-days-input', 'value'),
    prevent_initial_call=True
)
def export_origin_continent_to_excel(n_clicks, charts_data, volume_metric, selected_years,
                                     chart_type, rolling_avg_days):
    """Export the currently rendered origin-continent chart data."""
    if not n_clicks:
        raise PreventUpdate

    charts_data = _resolve_importers_chart_store(charts_data)

    rolling_label = _format_importer_rolling_window_label(rolling_avg_days)
    export_df = _build_chart_export_df(
        charts_data,
        volume_metric,
        selected_years,
        chart_type,
        rolling_avg_days,
    )
    if export_df.empty:
        raise PreventUpdate

    return _send_export_dataframe(
        export_df,
        f'importers_lng_supply_by_origin_continent_{chart_type}_{rolling_label.lower()}_rolling',
        'Origin Continent'
    )


@callback(
    Output('imp-overview-download-period-analysis-excel', 'data'),
    Input('imp-overview-export-period-analysis-button', 'n_clicks'),
    State('imp-overview-period-display-store', 'data'),
    State('imp-overview-origin-level-dropdown', 'value'),
    State('imp-overview-rolling-window-days-input', 'value'),
    State('imp-overview-period-view-type', 'value'),
    State('imp-overview-period-comparison-basis', 'value'),
    State('imp-overview-volume-metric-dropdown', 'value'),
    prevent_initial_call=True
)
def export_period_analysis_to_excel(
    n_clicks,
    period_display_data,
    origin_level,
    rolling_avg_days,
    view_type,
    comparison_basis,
    volume_metric='mcm_d',
):
    """Export the currently rendered period-analysis table."""
    if not n_clicks or not period_display_data:
        raise PreventUpdate

    export_df = pd.DataFrame(period_display_data)
    if export_df.empty:
        raise PreventUpdate

    comparison_basis = _normalize_importer_period_comparison_basis(
        comparison_basis
    )
    for column_name in [
        column for column in export_df.columns
        if column not in IMPORTER_PERIOD_TEXT_COLUMNS
    ]:
        family = _get_importer_period_column_family(column_name)
        is_delta = (
            comparison_basis in {'previous_period', 'same_period_last_year'}
            or family in {'delta-mom', 'delta-yoy', 'delta-pbd'}
        )
        export_df[column_name] = export_df[column_name].apply(
            lambda value, delta=is_delta, pbd=(
                column_name in IMPORTER_PERIOD_PBD_DELTA_COLUMNS
            ): _format_importer_period_grid_value(
                value,
                view_type=view_type,
                is_delta=delta,
                is_pbd_delta=pbd,
                volume_metric=volume_metric,
            )
        )

    safe_origin_level = _slugify_filename_label(ORIGIN_LEVEL_LABELS.get(origin_level, 'origin'))
    rolling_label = _format_importer_rolling_window_label(rolling_avg_days)
    safe_view_type = _slugify_filename_label(_normalize_importer_period_view_type(view_type))
    safe_comparison = _slugify_filename_label(comparison_basis)
    return _send_export_dataframe(
        export_df,
        (
            f'importers_lng_demand_by_origin_period_analysis_{safe_origin_level.lower()}_'
            f'{safe_view_type.lower()}_{safe_comparison.lower()}_{rolling_label.lower()}'
        ),
        'Period Analysis'
    )
