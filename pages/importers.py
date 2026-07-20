from dash import html, dcc, callback, Output, Input, State
from dash.dash_table.Format import Format, Scheme
from utils.ag_grid_tables import ag_grid_cell_clicked_to_active_cell, create_ag_grid_from_datatable
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
from sqlalchemy import text
from utils.dashboard_snapshot_cache import (
    build_source_key as _build_source_key,
    get_or_build_snapshot as _get_or_build_snapshot,
    pack_record_mapping as _pack_record_mapping,
    resolve_snapshot as _resolve_snapshot,
    snapshot_is_shared as _snapshot_is_shared,
    unpack_record_mapping as _unpack_record_mapping,
    was_global_refresh_triggered as _was_global_refresh_triggered,
    with_snapshot_slot as _with_snapshot_slot,
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
    build_importer_origin_summary_from_scoped_trades,
    IMPORTER_ORIGIN_LEVEL_TO_SCOPE,
    VOLUME_METRIC_OPTIONS,
    get_volume_metric_info,
    convert_volume_metric_dataframe,
)


DEFAULT_IMPORTER_ROLLING_AVG_DAYS = 30
MIN_IMPORTER_ROLLING_AVG_DAYS = 1
MAX_IMPORTER_ROLLING_AVG_DAYS = 180
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
IMPORTER_CHART_QUERY_START_DATE = '2020-11-01'
IMPORTER_CHART_DISPLAY_START_DATE = '2021-01-01'
IMPORTER_CHART_DEFAULT_SELECTED_YEAR_COUNT = 2
IMPORTER_CHART_DEFAULT_DESELECTED_YEARS = {'2024'}
IMPORTER_CHART_RANGE_LOOKBACK_YEARS = 5
IMPORTER_CHART_RANGE_FILL = 'rgba(148, 163, 184, 0.20)'
IMPORTER_PERIOD_DEFAULT_QUARTER_COUNT = 5
IMPORTER_PERIOD_DEFAULT_MONTH_COUNT = 3
IMPORTER_PERIOD_DEFAULT_WEEK_COUNT = 3
IMPORTER_PERIOD_MAX_QUARTER_COUNT = 8
IMPORTER_PERIOD_MAX_MONTH_COUNT = 12
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
IMPORTER_PERIOD_DELTA_RAW_FIELD_PREFIX = '__importer_period_delta_raw_'
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

IMPORTERS_OVERVIEW_NAMESPACE = 'importers-overview-v1'
IMPORTERS_PERIOD_NAMESPACE = 'importers-period-v1'


def _fetch_importers_source_watermark():
    query = text(f"""
        SELECT snapshot_timestamp_utc
        FROM {DB_SCHEMA}.kpler_trade_snapshots
        WHERE run_kind = 'canonical' AND status = 'published'
        ORDER BY snapshot_date_utc DESC
        LIMIT 1
    """)
    with engine.connect() as connection:
        return connection.execute(query).scalar()


def _resolve_importers_chart_store(charts_data):
    resolved = _resolve_snapshot(
        charts_data,
        engine,
        expected_namespace=IMPORTERS_OVERVIEW_NAMESPACE,
    )
    return _unpack_record_mapping(resolved)


def _resolve_importers_period_store(period_data):
    return _resolve_snapshot(
        period_data,
        engine,
        expected_namespace=IMPORTERS_PERIOD_NAMESPACE,
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


def _format_importer_rolling_average_section_title(title_prefix, rolling_avg_days):
    days = normalize_importer_rolling_avg_days(rolling_avg_days)
    return f'{title_prefix} - {days}-Day Rolling Average'


def _format_importer_rolling_window_label(rolling_avg_days):
    return f'{normalize_importer_rolling_avg_days(rolling_avg_days)}D'


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


def _build_chart_export_df(charts_data, volume_metric='mcm_d', selected_years=None, chart_type='absolute'):
    """Flatten the chart-data store into a single export dataframe."""
    if not charts_data:
        return pd.DataFrame()

    vol_label = get_volume_metric_info(volume_metric)['label']
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
        else:
            entity_df = convert_volume_metric_dataframe(entity_df, volume_metric, columns=['rolling_avg'])
            if 'rolling_avg' in entity_df.columns:
                entity_df = entity_df.rename(columns={'rolling_avg': f'rolling_avg ({vol_label})'})
        entity_df.insert(0, 'entity', entity_name)
        all_frames.append(entity_df)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


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
        WHERE kt.upload_timestamp_utc = latest_timestamp.max_ts
            AND kt.destination_country_name IS NOT NULL
            AND kt."end" IS NOT NULL
            AND kt.cargo_destination_cubic_meters IS NOT NULL
            AND kt."end"::date > CURRENT_DATE - INTERVAL '30 days'
            AND kt."end"::date <= CURRENT_DATE
            AND COALESCE(NULLIF(BTRIM(kt.destination_country_name), ''), 'Unknown')
                IS DISTINCT FROM COALESCE(NULLIF(BTRIM(kt.origin_country_name), ''), 'Unknown')
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
                              rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS):
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

    scoped_trades_df = _fetch_importer_scoped_trades(
        engine,
        all_destination_countries,
        min_end_date=IMPORTER_CHART_QUERY_START_DATE,
        include_destination_context=True
    )
    for entity in importer_entities:
        entity_label = entity['label']
        try:
            filtered_df = _filter_scoped_trades_for_entity(scoped_trades_df, entity, classification_mode)

            demand_df = _build_importer_total_import_df(
                filtered_df,
                rolling_window_days=rolling_avg_days,
                chart_start_date=IMPORTER_CHART_QUERY_START_DATE,
                display_start_date=IMPORTER_CHART_DISPLAY_START_DATE
            )
            demand_charts_data[entity_label] = demand_df.to_dict('records') if not demand_df.empty else []

            origin_continent_df = _build_importer_continent_chart_df(
                filtered_df,
                rolling_window_days=rolling_avg_days,
                include_percentage=True,
                chart_start_date=IMPORTER_CHART_QUERY_START_DATE,
                display_start_date=IMPORTER_CHART_DISPLAY_START_DATE
            )
            origin_continent_charts_data[entity_label] = (
                origin_continent_df.to_dict('records') if not origin_continent_df.empty else []
            )
        except Exception:
            demand_charts_data[entity_label] = []
            origin_continent_charts_data[entity_label] = []

    return demand_charts_data, origin_continent_charts_data


def group_small_importer_origin_countries(scoped_trades_df, origin_level='origin_shipping_region',
                                          threshold_mcmd=10, lookback_months=24):
    """Group low-volume origin countries into Rest of countries for the importer period table."""
    if scoped_trades_df is None or scoped_trades_df.empty:
        return scoped_trades_df
    if 'origin_country' not in scoped_trades_df.columns or 'end_date' not in scoped_trades_df.columns:
        return scoped_trades_df

    grouped_df = scoped_trades_df.copy()
    grouped_df['end_date'] = pd.to_datetime(grouped_df['end_date'], errors='coerce').dt.normalize()
    grouped_df = grouped_df[grouped_df['end_date'].notna()].copy()
    if grouped_df.empty:
        return grouped_df

    scope_column = IMPORTER_ORIGIN_LEVEL_TO_SCOPE.get(origin_level or 'origin_shipping_region', 'origin_shipping_region')
    parent_cols = []
    if scope_column != 'origin_country':
        if scope_column not in grouped_df.columns:
            return grouped_df
        parent_cols = [scope_column]

    current_timestamp = pd.Timestamp(datetime.now()).normalize()
    current_month = current_timestamp.to_period('M')
    start_month = current_month - (lookback_months - 1)
    lookback_df = grouped_df[grouped_df['end_date'].dt.to_period('M') >= start_month].copy()
    if lookback_df.empty:
        return grouped_df

    lookback_df['__month_period'] = lookback_df['end_date'].dt.to_period('M')
    monthly_totals = (
        lookback_df
        .groupby(parent_cols + ['origin_country', '__month_period'], dropna=False)['cargo_mcm']
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
        return grouped_df

    small_pairs['__group_small_country'] = True
    grouped_df = grouped_df.merge(small_pairs, on=pair_cols, how='left')
    grouped_df['__group_small_country'] = grouped_df['__group_small_country'].eq(True)
    grouped_df.loc[grouped_df['__group_small_country'], 'origin_country'] = 'Rest of countries'
    return grouped_df.drop(columns='__group_small_country')


def _get_small_importer_destination_countries(scoped_trades_df, importer_entities,
                                              threshold_mcmd=10, lookback_months=24):
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

    current_timestamp = pd.Timestamp(datetime.now()).normalize()
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
                                   threshold_mcmd=10, lookback_months=24):
    """Collapse low-volume importer countries into Rest of Importers for the grouped table payload."""
    if classification_mode != 'Country':
        return importer_entities or []

    importer_entities = list(importer_entities or [])
    small_destinations = _get_small_importer_destination_countries(
        scoped_trades_df,
        importer_entities,
        threshold_mcmd=threshold_mcmd,
        lookback_months=lookback_months
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
    return {'active_grouping_mode': grouping_mode, 'show_all': [], 'group_small_countries': []}


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


def _build_period_payload(importer_entities, classification_mode, origin_level,
                          grouping_mode='group_small_countries',
                          rolling_avg_days=DEFAULT_IMPORTER_ROLLING_AVG_DAYS):
    """Build the raw per-importer period-analysis payload."""
    grouping_mode = _normalize_importer_period_origin_grouping(grouping_mode)
    all_destination_countries = sorted({
        country
        for entity in importer_entities or []
        for country in entity.get('destination_countries', [])
    })
    scoped_trades_df = pd.DataFrame()
    if all_destination_countries:
        scoped_trades_df = _fetch_importer_scoped_trades(
            engine,
            all_destination_countries,
            vessel_type='All',
            delivered_only=True,
            include_destination_context=True
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
                if group_small_origins:
                    entity_scoped_df = group_small_importer_origin_countries(
                        entity_scoped_df,
                        origin_level or 'origin_shipping_region'
                    )
                summary_df = build_importer_origin_summary_from_scoped_trades(
                    entity_scoped_df,
                    rolling_window_days=rolling_avg_days,
                    origin_level=origin_level or 'origin_shipping_region',
                    quarter_count=IMPORTER_PERIOD_MAX_QUARTER_COUNT + 4,
                    month_count=IMPORTER_PERIOD_MAX_MONTH_COUNT + 12,
                    week_count=IMPORTER_PERIOD_MAX_WEEK_COUNT + 53,
                    include_comparison_reference_columns=True
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
        'active_grouping_mode': grouping_mode,
        'show_all': [],
        'group_small_countries': [],
    }
    if grouping_mode == 'show_all':
        payload['show_all'] = _build_payload_variant(importer_entities, group_small_origins=False)
        return payload

    grouped_importer_entities = _group_small_importer_entities(
        importer_entities,
        scoped_trades_df,
        classification_mode
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


def _add_importer_chart_range_band(fig, df, focus_year, available_years, vol_label):
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
            f'%{{customdata[0]:,.0f}}-%{{y:,.0f}} {vol_label}<extra></extra>'
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
    df = convert_volume_metric_dataframe(df, volume_metric, columns=['rolling_avg'])
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
    volume_metric
):
    if not data:
        return pd.DataFrame(), 'rolling_avg'

    df = pd.DataFrame(data)
    if df.empty or not {'date', 'year', 'continent_origin', 'rolling_avg'}.issubset(df.columns):
        return pd.DataFrame(), 'rolling_avg'

    metric_column = 'percentage' if chart_type == 'percentage' and 'percentage' in df.columns else 'rolling_avg'
    if metric_column == 'rolling_avg':
        df = convert_volume_metric_dataframe(df, volume_metric, columns=['rolling_avg'])

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


def _apply_importer_chart_layout(fig, y_title, yaxis_range=None, show_legend=False):
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
    _add_importer_chart_range_band(fig, df, focus_year, available_years, vol_label)

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
                    f'%{{y:,.0f}} {vol_label}<extra></extra>'
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
                    f'%{{y:,.0f}} {vol_label}<extra></extra>'
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

    return _apply_importer_chart_layout(fig, vol_label, show_legend=False)


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
    selected_years=None
):
    vol_label = get_volume_metric_info(volume_metric)['label']
    df, metric_column = _prepare_importer_origin_chart_dataframe(
        data,
        chart_type,
        volume_metric
    )
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
                        + ('%{y:.1f}%<extra></extra>' if metric_column == 'percentage' else f'%{{y:,.0f}} {vol_label}<extra></extra>')
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
                        + ('%{y:.1f}%<extra></extra>' if metric_column == 'percentage' else f'%{{y:,.0f}} {vol_label}<extra></extra>')
                    ),
                    text=connect_data['month_day'],
                    showlegend=False
                ))

    y_title = '%' if metric_column == 'percentage' else vol_label
    yaxis_range = [0, 100] if metric_column == 'percentage' else None
    return _apply_importer_chart_layout(fig, y_title, yaxis_range=yaxis_range, show_legend=True)


def _format_importer_chart_current_value(metrics, vol_label):
    if not metrics or metrics.get('latest_value') is None:
        return None

    latest_label = metrics.get('latest_label') or metrics.get('focus_year') or ''
    return f"{latest_label}: {metrics['latest_value']:,.0f} {vol_label}"


def _build_importer_chart_delta_pill(label, delta_value, delta_pct):
    if delta_value is None or pd.isna(delta_value):
        return html.Span(f'{label} n/a', className='importer-rolling-delta-pill importer-rolling-delta-neutral')

    direction_class = 'importer-rolling-delta-neutral'
    if delta_value > 0:
        direction_class = 'importer-rolling-delta-positive'
    elif delta_value < 0:
        direction_class = 'importer-rolling-delta-negative'

    sign = '+' if delta_value > 0 else ''
    pct_text = ''
    if delta_pct is not None and pd.notna(delta_pct):
        pct_text = f" ({sign}{delta_pct:.0f}%)"

    return html.Span(
        [
            html.Span(label, className='importer-rolling-delta-label'),
            html.Span(f"{sign}{delta_value:,.0f}{pct_text}")
        ],
        className=f'importer-rolling-delta-pill {direction_class}'
    )


def _build_importer_chart_delta_indicators(metrics):
    return html.Div(
        [
            _build_importer_chart_delta_pill(
                'MoM',
                metrics.get('mom_delta_value') if metrics else None,
                metrics.get('mom_delta_pct') if metrics else None
            ),
            _build_importer_chart_delta_pill(
                'YoY',
                metrics.get('delta_value') if metrics else None,
                metrics.get('delta_pct') if metrics else None
            )
        ],
        className='importer-rolling-delta-group'
    )


def _format_origin_kpi_value(value, chart_type, is_delta=False):
    if value is None or pd.isna(value):
        return 'n/a'

    rounded_value = int(round(float(value)))
    sign = '+' if is_delta and rounded_value > 0 else ''
    if chart_type == 'percentage':
        return f'{sign}{rounded_value}pp' if is_delta else f'{rounded_value}%'
    return f'{sign}{rounded_value:,}'


def _format_origin_kpi_pct(delta_pct):
    if delta_pct is None or pd.isna(delta_pct):
        return ''
    sign = '+' if delta_pct > 0 else ''
    return f' ({sign}{delta_pct:.0f}%)'


def _format_origin_kpi_pct_compact(delta_pct):
    if delta_pct is None or pd.isna(delta_pct):
        return None
    rounded_pct = int(round(float(delta_pct)))
    sign = '+' if rounded_pct > 0 else ''
    return f'({sign}{rounded_pct}%)'


def _origin_kpi_direction_class(value):
    if value is None or pd.isna(value):
        return 'importer-origin-kpi-delta-neutral continent-kpi-delta-neutral'
    if value > 0:
        return 'importer-origin-kpi-delta-positive continent-kpi-delta-positive'
    if value < 0:
        return 'importer-origin-kpi-delta-negative continent-kpi-delta-negative'
    return 'importer-origin-kpi-delta-neutral continent-kpi-delta-neutral'


def _origin_kpi_value_displays_zero(value, chart_type, is_delta_pct=False):
    if value is None or pd.isna(value):
        return True

    display_tolerance = 0.05 if chart_type == 'percentage' and not is_delta_pct else 0.5
    return abs(float(value)) < display_tolerance


def _origin_kpi_all_displayed_values_zero(
    chart_type,
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
        _origin_kpi_value_displays_zero(value, chart_type, is_delta_pct)
        for value, is_delta_pct in values_to_check
    )


def _calculate_origin_continent_kpis(
    data,
    chart_type='absolute',
    volume_metric='mcm_d',
    selected_years=None
):
    vol_label = get_volume_metric_info(volume_metric)['label']
    df, metric_column = _prepare_importer_origin_chart_dataframe(
        data,
        chart_type,
        volume_metric
    )
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
            'latest_text': _format_origin_kpi_value(latest_value, chart_type),
            'latest_label': latest_point.get('month_day', ''),
            'mom_delta_value': mom_delta_numeric,
            'mom_value_text': (
                _format_origin_kpi_value(mom_delta_value, chart_type, is_delta=True)
            ) if mom_delta_value is not None and pd.notna(mom_delta_value) else 'n/a',
            'mom_pct_text': _format_origin_kpi_pct_compact(mom_delta_pct),
            'mom_text': (
                _format_origin_kpi_value(mom_delta_value, chart_type, is_delta=True)
                + _format_origin_kpi_pct(mom_delta_pct)
            ) if mom_delta_value is not None and pd.notna(mom_delta_value) else 'n/a',
            'mom_class': _origin_kpi_direction_class(mom_delta_value),
            'mom_delta_pct': mom_pct_numeric,
            'yoy_delta_value': yoy_delta_numeric,
            'yoy_value_text': (
                _format_origin_kpi_value(yoy_delta_value, chart_type, is_delta=True)
            ) if yoy_delta_value is not None and pd.notna(yoy_delta_value) else 'n/a',
            'yoy_pct_text': _format_origin_kpi_pct_compact(yoy_delta_pct),
            'yoy_text': (
                _format_origin_kpi_value(yoy_delta_value, chart_type, is_delta=True)
                + _format_origin_kpi_pct(yoy_delta_pct)
            ) if yoy_delta_value is not None and pd.notna(yoy_delta_value) else 'n/a',
            'yoy_delta_pct': yoy_pct_numeric,
            'yoy_class': _origin_kpi_direction_class(yoy_delta_value),
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

    if '7D' in available_cols:
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
            if family in {'delta-mom', 'delta-yoy'} or column_id in delta_like_cols:
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
        elif family in {'delta-mom', 'delta-yoy'}:
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
        if _get_importer_period_column_family(column_id) in {'delta-mom', 'delta-yoy'} or column_id in delta_like_cols:
            styles.extend(
                _build_importer_period_delta_gradient_styles(
                    display_df,
                    column_id,
                    raw_field=raw_field_map.get(column_id)
                )
            )
    return styles


def _format_importer_period_grid_value(value, view_type='absolute', is_delta=False):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if view_type == 'percentage' and is_delta:
        if abs(numeric_value) < 0.5:
            numeric_value = 0
        sign = '+' if numeric_value > 0 else ''
        return f'{sign}{numeric_value:,.0f} pp'
    if view_type == 'percentage':
        return f'{numeric_value:.0f}%'
    if is_delta and abs(numeric_value) < 0.05:
        numeric_value = 0
    return f'{numeric_value:,.1f}'


def _build_importer_period_grid_display(display_df, columns, view_type='absolute',
                                        delta_like_cols=None, raw_field_map=None):
    grid_df = display_df.copy()
    grid_columns = [dict(column) for column in columns]
    delta_like_cols = set(delta_like_cols or [])
    raw_field_map = raw_field_map or {}
    numeric_ids = {
        column.get('id')
        for column in grid_columns
        if column.get('type') == 'numeric'
    }
    delta_ids = {
        column_id for column_id in numeric_ids
        if _get_importer_period_column_family(column_id) in {'delta-mom', 'delta-yoy'}
    } | delta_like_cols

    for column_id, raw_field in raw_field_map.items():
        if column_id in grid_df.columns:
            grid_df[raw_field] = pd.to_numeric(grid_df[column_id], errors='coerce')

    for column_id in numeric_ids:
        if column_id not in grid_df.columns:
            continue
        is_delta = column_id in delta_ids
        grid_df[column_id] = grid_df[column_id].apply(
            lambda value, delta=is_delta: _format_importer_period_grid_value(
                value,
                view_type=view_type,
                is_delta=delta
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


def _create_period_analysis_table(display_df, delta_like_cols=None, view_type='absolute'):
    """Create the combined overview period-analysis table."""
    delta_like_cols = set(delta_like_cols or [])
    delta_columns = {
        col for col in display_df.columns
        if (
            col not in IMPORTER_PERIOD_TEXT_COLUMNS and
            (
                _get_importer_period_column_family(col) in {'delta-mom', 'delta-yoy'} or
                col in delta_like_cols
            )
        )
    }
    raw_field_map = {
        column_id: _get_importer_period_delta_raw_field(column_id)
        for column_id in delta_columns
    }
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
                'format': Format(precision=1, scheme=Scheme.fixed),
                'cellClass': 'importer-period-number-cell'
            })

    columns = _apply_importer_period_column_classes(
        columns,
        display_df,
        delta_like_cols=delta_like_cols,
        raw_field_map=raw_field_map
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
        raw_field_map=raw_field_map
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


def _build_period_table_footnote(rolling_avg_days, vol_label, comparison_basis='levels'):
    rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
    comparison_basis = _normalize_importer_period_comparison_basis(comparison_basis)
    today = datetime.now().date()
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

    return html.Div(
        [
            html.P(
                [
                    html.Span('Note: ', className='importer-period-footnote-strong'),
                    html.Span(f'{rolling_label}: {date_window_start} to {date_today} | '),
                    html.Span(f'7D: {date_7d_start} to {date_today} | '),
                    html.Span(f'{rolling_label} Y-1: {date_window_y1_start} to {date_window_y1_end} | '),
                    html.Span(f'Values shown in {vol_label}{comparison_note}')
                ],
                className='importer-period-table-footnote-text'
            )
        ],
        className='importer-period-table-footnote'
    )


layout = html.Div([
    dcc.Store(id='imp-overview-source-state-store', storage_type='memory'),
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
    prevent_initial_call=False
)
def update_importer_rolling_section_titles(rolling_avg_days):
    return (
        _format_importer_rolling_average_section_title('LNG Demand', rolling_avg_days),
        _format_importer_rolling_average_section_title('LNG Demand by Origin Continent', rolling_avg_days),
    )


def _build_importers_overview_payload(classification_mode, rolling_avg_days):
    rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
    catalog_records = build_destination_catalog(engine)
    catalog_df = get_destination_catalog_dataframe(catalog_records)
    ranking_df = _fetch_destination_ranking_df(catalog_df)
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
        rolling_avg_days
    )
    return {
        'chart_entities': chart_entities,
        'table_entities': table_entities,
        'demand_cube': _pack_record_mapping(demand_charts_data),
        'origin_cube': _pack_record_mapping(origin_continent_charts_data),
    }


@callback(
    Output('imp-overview-source-state-store', 'data'),
    Input('global-refresh-button', 'n_clicks'),
)
def load_importers_overview_source_state(_n_clicks):
    try:
        watermark = _fetch_importers_source_watermark()
        return {'watermark': watermark.isoformat() if hasattr(watermark, 'isoformat') else str(watermark)}
    except Exception:
        return {'request_token': datetime.now().isoformat(timespec='microseconds')}


@callback(
    Output('imp-overview-chart-entities-store', 'data'),
    Output('imp-overview-table-entities-store', 'data'),
    Output('imp-overview-demand-data-store', 'data'),
    Output('imp-overview-origin-continent-data-store', 'data'),
    Input('global-refresh-button', 'n_clicks'),
    Input('imp-overview-classification-mode', 'value'),
    Input('imp-overview-rolling-window-days-input', 'value'),
    State('imp-overview-source-state-store', 'data'),
    prevent_initial_call=False
)
def refresh_overview_data(_n_clicks, classification_mode, rolling_avg_days, source_state=None):
    """Load the importer overview entities and compact server-side chart datasets."""
    try:
        rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
        if source_state and not _was_global_refresh_triggered():
            source_watermark = source_state
        else:
            try:
                watermark = _fetch_importers_source_watermark()
                source_watermark = {
                    'watermark': watermark.isoformat() if hasattr(watermark, 'isoformat') else str(watermark)
                }
            except Exception:
                source_watermark = {
                    'request_token': datetime.now().isoformat(timespec='microseconds')
                }
        source_key = _build_source_key(
            IMPORTERS_OVERVIEW_NAMESPACE,
            source_watermark,
            datetime.now().date(),
            classification_mode,
            rolling_avg_days,
        )
        reference, payload = _get_or_build_snapshot(
            engine,
            namespace=IMPORTERS_OVERVIEW_NAMESPACE,
            source_key=source_key,
            builder=lambda: _build_importers_overview_payload(
                classification_mode,
                rolling_avg_days,
            ),
            force=_was_global_refresh_triggered(),
            manifest={
                'classification_mode': classification_mode,
                'rolling_avg_days': rolling_avg_days,
            },
        )

        if _snapshot_is_shared(reference):
            demand_store = _with_snapshot_slot(reference, 'demand_cube')
            origin_store = _with_snapshot_slot(reference, 'origin_cube')
        else:
            demand_store = _unpack_record_mapping(payload['demand_cube'])
            origin_store = _unpack_record_mapping(payload['origin_cube'])

        return (
            payload['chart_entities'],
            payload['table_entities'],
            demand_store,
            origin_store,
        )
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
    charts_data = _resolve_importers_chart_store(charts_data)
    available_years = _get_importer_chart_available_years(charts_data)
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
def update_demand_charts(charts_data, importer_entities, volume_metric, selected_years, rolling_avg_days):
    """Render the demand chart grid using the upgraded exporter-page pattern."""
    charts_data = _resolve_importers_chart_store(charts_data)
    if not charts_data or not importer_entities:
        return html.Div('No data available', className='importer-rolling-empty-state')

    vol_label = get_volume_metric_info(volume_metric)['label']
    rolling_avg_days = normalize_importer_rolling_avg_days(rolling_avg_days)
    charts = []
    for entity in importer_entities:
        entity_name = entity['label']
        entity_data = charts_data.get(entity_name, [])
        fig = create_importer_demand_chart(
            entity_data,
            volume_metric=volume_metric,
            selected_years=selected_years,
            rolling_avg_days=rolling_avg_days
        )
        metrics = get_importer_demand_chart_header_metrics(
            entity_data,
            volume_metric=volume_metric,
            selected_years=selected_years,
            rolling_avg_days=rolling_avg_days
        )
        current_value = _format_importer_chart_current_value(metrics, vol_label)
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
                        _build_importer_chart_delta_indicators(metrics)
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
    charts_data = _resolve_importers_chart_store(charts_data)
    available_years = _get_importer_chart_available_years(charts_data)
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
def update_origin_continent_charts(charts_data, importer_entities, volume_metric, selected_years,
                                   chart_type, _rolling_avg_days):
    """Render the origin-continent chart grid using the upgraded exporter-page pattern."""
    charts_data = _resolve_importers_chart_store(charts_data)
    if not charts_data or not importer_entities:
        return html.Div(
            'No data available',
            className='importer-origin-rolling-empty-state continent-rolling-empty-state'
        )

    charts = []
    kpi_rows = []
    for entity in importer_entities:
        entity_name = entity['label']
        entity_data = charts_data.get(entity_name, [])
        fig = create_importer_origin_continent_chart(
            entity_data,
            chart_type=chart_type,
            volume_metric=volume_metric,
            selected_years=selected_years
        )
        kpi_rows.append({
            'entity': entity_name,
            'metrics': _calculate_origin_continent_kpis(
                entity_data,
                chart_type=chart_type,
                volume_metric=volume_metric,
                selected_years=selected_years
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
    Input('global-refresh-button', 'n_clicks'),
    State('imp-overview-source-state-store', 'data'),
    prevent_initial_call=False
)
def refresh_period_data(importer_entities, classification_mode, origin_level, rolling_avg_days,
                        origin_country_grouping_mode, global_refresh_clicks, source_state=None):
    """Load the raw period-analysis payload for the overview importers."""
    if not importer_entities:
        return _empty_importer_period_payload(origin_country_grouping_mode)

    try:
        normalized_rolling_days = normalize_importer_rolling_avg_days(rolling_avg_days)
        source_key = _build_source_key(
            IMPORTERS_PERIOD_NAMESPACE,
            source_state,
            global_refresh_clicks or 0,
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
            ),
            manifest={
                'classification_mode': classification_mode,
                'origin_level': origin_level,
                'origin_country_grouping_mode': origin_country_grouping_mode,
                'rolling_avg_days': normalized_rolling_days,
            },
        )
        return reference if _snapshot_is_shared(reference) else payload
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
def update_period_analysis_table(period_payload, expanded_importers, importer_entities, volume_metric,
                                 rolling_avg_days, origin_country_grouping_mode, view_type, comparison_basis,
                                 quarter_count, month_count, week_count):
    """Render the combined importer overview period-analysis table."""
    if not period_payload or not importer_entities:
        message = html.Div('No data available for the selected configuration.', style={'textAlign': 'center', 'padding': '20px'})
        return message, []

    period_payload = _resolve_importers_period_store(period_payload)

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

    if view_type == 'percentage':
        level_delta_cols = [
            col for col in display_df.columns
            if col.startswith('Δ ')
        ]
        if level_delta_cols:
            display_df = display_df.drop(columns=level_delta_cols)
        numeric_cols = [col for col in display_df.columns if col not in IMPORTER_PERIOD_TEXT_COLUMNS]
        display_df = _apply_importer_period_percentage_view(display_df, numeric_cols)
        vol_label = 'market share (%)'
    else:
        vol_label = get_volume_metric_info(volume_metric)['label']
        display_df = convert_volume_metric_dataframe(
            display_df,
            volume_metric,
            exclude_columns=IMPORTER_PERIOD_TEXT_COLUMNS,
            precision=1
        )

    display_df, comparison_delta_cols = _apply_importer_period_comparison(
        display_df,
        comparison_metadata
    )
    for col in [col for col in display_df.columns if col not in IMPORTER_PERIOD_TEXT_COLUMNS]:
        numeric_series = pd.to_numeric(display_df[col], errors='coerce').round(1)
        display_df[col] = numeric_series.where(pd.notnull(numeric_series), None)

    table_shell = html.Div(
        [
            _create_period_analysis_table(
                display_df,
                delta_like_cols=comparison_delta_cols,
                view_type=view_type
            ),
            _build_period_table_footnote(rolling_avg_days, vol_label, comparison_basis)
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
    export_df = _build_chart_export_df(charts_data, volume_metric, selected_years)
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
    export_df = _build_chart_export_df(charts_data, volume_metric, selected_years, chart_type)
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
    prevent_initial_call=True
)
def export_period_analysis_to_excel(n_clicks, period_display_data, origin_level, rolling_avg_days,
                                    view_type, comparison_basis):
    """Export the currently rendered period-analysis table."""
    if not n_clicks or not period_display_data:
        raise PreventUpdate

    export_df = pd.DataFrame(period_display_data)
    if export_df.empty:
        raise PreventUpdate

    safe_origin_level = _slugify_filename_label(ORIGIN_LEVEL_LABELS.get(origin_level, 'origin'))
    rolling_label = _format_importer_rolling_window_label(rolling_avg_days)
    safe_view_type = _slugify_filename_label(_normalize_importer_period_view_type(view_type))
    safe_comparison = _slugify_filename_label(_normalize_importer_period_comparison_basis(comparison_basis))
    return _send_export_dataframe(
        export_df,
        (
            f'importers_lng_demand_by_origin_period_analysis_{safe_origin_level.lower()}_'
            f'{safe_view_type.lower()}_{safe_comparison.lower()}_{rolling_label.lower()}'
        ),
        'Period Analysis'
    )
