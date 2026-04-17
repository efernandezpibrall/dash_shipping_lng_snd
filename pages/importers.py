from dash import html, dcc, dash_table, callback, Output, Input, State
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate
import pandas as pd
from io import BytesIO
from datetime import datetime
from sqlalchemy import text

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
    _create_seasonal_line_chart,
    _empty_timeseries_chart,
    fetch_origin_summary_data,
)


ROLLING_WINDOW_DAYS = 30
TOP_IMPORTER_CHART_COUNT = 12
MONTH_ORDER = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

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


def _create_top_importers_selector_region():
    """Render the overview controls using the exporter-page header structure."""
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
                            html.Div('Importer Classification', className='filter-group-header'),
                            html.Div(
                                [
                                    dcc.RadioItems(
                                        id='imp-overview-classification-mode',
                                        options=IMPORTER_CLASSIFICATION_OPTIONS,
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
                            html.Div('Coverage', className='filter-group-header'),
                            html.Div(
                                id='imp-overview-coverage-display',
                                className='text-tertiary',
                                style={'fontSize': '11px', 'maxWidth': '420px', 'width': '100%'}
                            )
                        ],
                        className='filter-section filter-section-analysis'
                    ),
                    html.Div(
                        [
                            html.Div('Origin Level', className='filter-group-header'),
                            dcc.Dropdown(
                                id='imp-overview-origin-level-dropdown',
                                options=ORIGIN_LEVEL_OPTIONS,
                                value='origin_shipping_region',
                                clearable=False,
                                className='filter-dropdown',
                                style={'minWidth': '220px', 'width': '100%'}
                            )
                        ],
                        className='filter-section filter-section-origin'
                    ),
                    html.Div(
                        [
                            html.Div('Latest Upload', className='filter-group-header'),
                            html.Div(
                                id='imp-overview-data-timestamp-display',
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


def _get_latest_upload_timestamp():
    """Return the latest available Kpler upload timestamp."""
    query = text(f"""
        SELECT MAX(upload_timestamp_utc) AS latest_upload_timestamp
        FROM {DB_SCHEMA}.kpler_trades
    """)
    timestamp_df = pd.read_sql(query, engine)
    if timestamp_df.empty:
        return None

    timestamp = timestamp_df.iloc[0].get('latest_upload_timestamp')
    return pd.to_datetime(timestamp, errors='coerce')


def _format_timestamp_display(timestamp):
    """Format the upload timestamp shown in the header."""
    if pd.isna(timestamp):
        return ''
    return f"Data as of: {timestamp.strftime('%Y-%m-%d %H:%M UTC')}"


def _is_country_origin_level(origin_level):
    """Return whether the selected origin level is already at country granularity."""
    return origin_level == 'origin_country_name'


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


def _build_chart_export_df(charts_data):
    """Flatten the chart-data store into a single export dataframe."""
    if not charts_data:
        return pd.DataFrame()

    all_frames = []
    for entity_name, records in charts_data.items():
        if not records:
            continue
        entity_df = pd.DataFrame(records)
        if entity_df.empty:
            continue
        entity_df.insert(0, 'entity', entity_name)
        all_frames.append(entity_df)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def _fetch_destination_ranking_df():
    """Return latest-30D destination demand rankings at country level."""
    catalog_records = build_destination_catalog(engine)
    catalog_df = get_destination_catalog_dataframe(catalog_records)
    if catalog_df.empty:
        return pd.DataFrame()

    query = text(f"""
        WITH latest_timestamp AS (
            SELECT MAX(upload_timestamp_utc) AS max_ts
            FROM {DB_SCHEMA}.kpler_trades
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


def _build_destination_entities(classification_mode='Country', limit=None):
    """Build importer entities for charts or tables from the destination catalog."""
    catalog_records = build_destination_catalog(engine)
    catalog_df = get_destination_catalog_dataframe(catalog_records)
    ranking_df = _fetch_destination_ranking_df()
    if catalog_df.empty:
        return []

    merged_df = catalog_df.merge(
        ranking_df[['destination_country_name', 'avg_30d_mcmd']],
        how='left',
        on='destination_country_name'
    )
    merged_df['avg_30d_mcmd'] = pd.to_numeric(merged_df['avg_30d_mcmd'], errors='coerce')
    merged_df['avg_30d_mcmd'] = merged_df['avg_30d_mcmd'].fillna(0.0)

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
        return [
            {
                'key': row['country_classification_level1'],
                'label': row['label'],
                'destination_countries': list(row['destination_countries']),
                'avg_30d_mcmd': round(float(row['avg_30d_mcmd']), 1) if pd.notna(row['avg_30d_mcmd']) else None
            }
            for _, row in grouped_df.iterrows()
        ]

    country_df = merged_df[['destination_country_name', 'country_display', 'avg_30d_mcmd']].drop_duplicates(
        subset=['destination_country_name']
    )
    country_df = country_df.sort_values(
        ['avg_30d_mcmd', 'country_display', 'destination_country_name'],
        ascending=[False, True, True]
    ).reset_index(drop=True)
    if limit is not None:
        country_df = country_df.head(limit).copy()
    return [
        {
            'key': row['destination_country_name'],
            'label': row['country_display'],
            'destination_countries': [row['destination_country_name']],
            'avg_30d_mcmd': round(float(row['avg_30d_mcmd']), 1) if pd.notna(row['avg_30d_mcmd']) else None
        }
        for _, row in country_df.iterrows()
    ]


def _build_chart_data_payload(importer_entities, classification_mode='Country'):
    """Build the demand and origin-continent chart payloads for overview importers."""
    demand_charts_data = {}
    origin_continent_charts_data = {}
    destination_aggregation = _classification_mode_to_destination_aggregation(classification_mode)

    for entity in importer_entities:
        destination_countries = entity['destination_countries']
        entity_key = entity['key']
        entity_label = entity['label']
        try:
            scoped_trades_df = _fetch_importer_scoped_trades(engine, destination_countries)
            filtered_df = _apply_importer_self_flow_exclusion(
                scoped_trades_df,
                destination_aggregation,
                entity_key
            )

            demand_df = _build_importer_total_import_df(
                filtered_df,
                rolling_window_days=ROLLING_WINDOW_DAYS
            )
            demand_charts_data[entity_label] = demand_df.to_dict('records') if not demand_df.empty else []

            origin_continent_df = _build_importer_continent_chart_df(
                filtered_df,
                rolling_window_days=ROLLING_WINDOW_DAYS,
                include_percentage=False
            )
            origin_continent_charts_data[entity_label] = (
                origin_continent_df.to_dict('records') if not origin_continent_df.empty else []
            )
        except Exception:
            demand_charts_data[entity_label] = []
            origin_continent_charts_data[entity_label] = []

    return demand_charts_data, origin_continent_charts_data


def _build_period_payload(importer_entities, classification_mode, origin_level):
    """Build the raw per-importer period-analysis payload."""
    payload = []
    destination_aggregation = _classification_mode_to_destination_aggregation(classification_mode)
    for entity in importer_entities:
        try:
            summary_df = fetch_origin_summary_data(
                engine,
                entity['destination_countries'],
                status='laden',
                vessel_type='All',
                rolling_window_days=ROLLING_WINDOW_DAYS,
                origin_level=origin_level or 'origin_shipping_region',
                selected_destination_aggregation=destination_aggregation,
                selected_destination_value=entity['key']
            )
        except Exception:
            summary_df = pd.DataFrame()
        payload.append({
            'label': entity['label'],
            'key': entity['key'],
            'records': summary_df.to_dict('records') if not summary_df.empty else []
        })
    return payload


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


def _get_period_numeric_columns(period_payload):
    """Return the curated period-column order for the combined overview table."""
    available_cols = _get_available_period_columns(period_payload)
    if not available_cols:
        return []

    current_date = datetime.now()
    rolling_window_label = f'{ROLLING_WINDOW_DAYS}D'

    quarter_cols = sorted(
        [
            col for col in available_cols
            if col.startswith('Q') and "'" in col and _is_completed_quarter_label(col, current_date)
        ],
        key=_quarter_sort_key
    )[-5:]

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
    )[-3:]

    week_cols = sorted(
        [
            col for col in available_cols
            if col.startswith('W') and "'" in col and _is_completed_week_label(col, current_date)
        ],
        key=_week_sort_key
    )[-3:]

    numeric_cols = []
    numeric_cols.extend(quarter_cols)
    numeric_cols.extend(month_cols)
    if rolling_window_label in available_cols:
        numeric_cols.append(rolling_window_label)
    numeric_cols.extend(week_cols)

    for col in ['7D', f'Δ 7D-{rolling_window_label}', f'Δ {rolling_window_label} Y/Y']:
        if col in available_cols:
            numeric_cols.append(col)

    return numeric_cols


def _sum_numeric_columns(df, numeric_cols):
    """Aggregate numeric columns safely while preserving missing-data semantics."""
    totals = {}
    for col in numeric_cols:
        numeric_series = pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series(dtype=float)
        totals[col] = numeric_series.sum(min_count=1)
    return totals


def _build_period_display_df(period_payload, origin_level, expanded_importers=None, expanded_origins=None):
    """Create the combined importer -> origin -> country period-analysis display dataframe."""
    expanded_importers = expanded_importers or []
    expanded_origins = expanded_origins or []
    numeric_cols = _get_period_numeric_columns(period_payload)

    if not numeric_cols:
        return pd.DataFrame(columns=['Importer', 'Origin Level', 'Country'])

    display_rows = []
    importer_totals = []
    country_origin_level = _is_country_origin_level(origin_level)

    for importer_payload in period_payload or []:
        importer_label = importer_payload.get('label')
        importer_df = pd.DataFrame(importer_payload.get('records', []))
        if importer_df.empty:
            continue

        importer_df = importer_df.sort_values(['continent', 'country']).reset_index(drop=True)
        importer_total_row = {
            'Importer': f"▼ {importer_label}" if importer_label in expanded_importers else f"▶ {importer_label}",
            'Origin Level': 'Total',
            'Country': '',
            **_sum_numeric_columns(importer_df, numeric_cols)
        }
        display_rows.append(importer_total_row)
        importer_totals.append(importer_total_row)

        if importer_label not in expanded_importers:
            continue

        if country_origin_level:
            for _, row in importer_df.iterrows():
                child_row = {
                    'Importer': '',
                    'Origin Level': f"    {row.get('continent', '')}",
                    'Country': '',
                }
                for col in numeric_cols:
                    child_row[col] = row.get(col)
                display_rows.append(child_row)
            continue

        for origin_name in importer_df['continent'].dropna().unique():
            origin_df = importer_df[importer_df['continent'] == origin_name].copy()
            origin_key = f"{importer_label}→{origin_name}"
            origin_total_row = {
                'Importer': '',
                'Origin Level': (
                    f"    ▼ {origin_name}"
                    if origin_key in expanded_origins else
                    f"    ▶ {origin_name}"
                ),
                'Country': 'Total',
                **_sum_numeric_columns(origin_df, numeric_cols)
            }
            display_rows.append(origin_total_row)

            if origin_key not in expanded_origins:
                continue

            for _, row in origin_df.iterrows():
                child_row = {
                    'Importer': '',
                    'Origin Level': '',
                    'Country': f"        {row.get('country', '')}",
                }
                for col in numeric_cols:
                    child_row[col] = row.get(col)
                display_rows.append(child_row)

    if importer_totals:
        grand_total_row = {
            'Importer': 'GRAND TOTAL',
            'Origin Level': '',
            'Country': '',
        }
        importer_totals_df = pd.DataFrame(importer_totals)
        for col in numeric_cols:
            grand_total_row[col] = pd.to_numeric(importer_totals_df[col], errors='coerce').sum(min_count=1)
        display_rows.append(grand_total_row)

    display_df = pd.DataFrame(display_rows, columns=['Importer', 'Origin Level', 'Country'] + numeric_cols)
    for col in numeric_cols:
        numeric_series = pd.to_numeric(display_df[col], errors='coerce').round(1)
        display_df[col] = numeric_series.where(pd.notnull(numeric_series), None)
    return display_df


def _create_period_analysis_table(display_df, origin_level):
    """Create the combined overview period-analysis table."""
    columns = []
    for col in display_df.columns:
        if col in ['Importer', 'Origin Level', 'Country']:
            columns.append({'name': col, 'id': col, 'type': 'text'})
        else:
            columns.append({
                'name': col,
                'id': col,
                'type': 'numeric',
                'format': Format(precision=1, scheme=Scheme.fixed)
            })

    quarter_columns = [col for col in display_df.columns if col.startswith('Q') and "'" in col]
    month_columns = [
        col for col in display_df.columns
        if "'" in col and not col.startswith('Q') and not col.startswith('W')
        and col not in ['Importer', 'Origin Level', 'Country']
    ]
    week_columns = [col for col in display_df.columns if col.startswith('W') and "'" in col]
    rolling_window_columns = [
        col for col in display_df.columns
        if col == '7D' or (col.endswith('D') and col[:-1].isdigit())
    ]
    delta_vs_window_column = next((col for col in display_df.columns if col.startswith('Δ 7D-')), None)
    delta_yoy_column = next(
        (col for col in display_df.columns if col.startswith('Δ ') and col.endswith(' Y/Y')),
        None
    )

    conditional_styles = [
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': '#fafafa'
        },
        {
            'if': {'filter_query': '{Importer} = "GRAND TOTAL"'},
            'backgroundColor': '#2E86C1',
            'color': 'white',
            'fontWeight': 'bold'
        },
        {
            'if': {'filter_query': '{Origin Level} = "Total" && {Country} = ""'},
            'backgroundColor': '#dbeafe',
            'fontWeight': 'bold'
        },
        {
            'if': {'filter_query': '{Country} = "Total"'},
            'backgroundColor': '#eff6ff',
            'fontWeight': '600'
        },
        {
            'if': {'filter_query': '{Origin Level} = "" && {Country} != ""'},
            'backgroundColor': '#f8fafc',
            'fontSize': '13px'
        },
        {'if': {'column_id': 'Importer'}, 'textAlign': 'left'},
        {'if': {'column_id': 'Origin Level'}, 'textAlign': 'left'},
        {'if': {'column_id': 'Country'}, 'textAlign': 'left'},
    ]

    for col in display_df.columns:
        if col not in ['Importer', 'Origin Level', 'Country']:
            conditional_styles.append({
                'if': {'column_id': col},
                'textAlign': 'right',
                'paddingRight': '12px'
            })

        if col.startswith('Q') and "'" in col and quarter_columns and col == quarter_columns[0]:
            conditional_styles.append({'if': {'column_id': col}, 'borderLeft': '3px solid white'})
        elif col.startswith('W') and "'" in col and week_columns and col == week_columns[0]:
            conditional_styles.append({'if': {'column_id': col}, 'borderLeft': '3px solid white'})
        elif (
            "'" in col and not col.startswith('Q') and not col.startswith('W') and
            month_columns and col == month_columns[0]
        ):
            conditional_styles.append({'if': {'column_id': col}, 'borderLeft': '3px solid white'})
        elif col in rolling_window_columns:
            conditional_styles.append({
                'if': {'column_id': col},
                'backgroundColor': '#fff3e0',
                'fontWeight': '500'
            })
        elif col == delta_vs_window_column:
            conditional_styles.extend([
                {
                    'if': {'column_id': col},
                    'backgroundColor': '#f5f5f5',
                    'fontWeight': '600',
                    'borderLeft': '3px solid white'
                },
                {'if': {'column_id': col, 'filter_query': f'{{{col}}} > 0'}, 'color': '#2e7d32'},
                {'if': {'column_id': col, 'filter_query': f'{{{col}}} < 0'}, 'color': '#c62828'},
            ])
        elif col == delta_yoy_column:
            conditional_styles.extend([
                {
                    'if': {'column_id': col},
                    'backgroundColor': '#e8f5e9',
                    'fontWeight': '600',
                    'borderLeft': '3px solid white'
                },
                {'if': {'column_id': col, 'filter_query': f'{{{col}}} > 0'}, 'color': '#1b5e20'},
                {'if': {'column_id': col, 'filter_query': f'{{{col}}} < 0'}, 'color': '#b71c1c'},
            ])

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
        header_styles.append({'if': {'column_id': delta_vs_window_column}, 'borderLeft': '3px solid white'})
    if delta_yoy_column:
        header_styles.append({'if': {'column_id': delta_yoy_column}, 'backgroundColor': '#e8f5e9'})
        header_styles.append({'if': {'column_id': delta_yoy_column}, 'borderLeft': '3px solid white'})
    if quarter_columns:
        header_styles.append({'if': {'column_id': quarter_columns[0]}, 'borderLeft': '3px solid white'})
    if month_columns:
        header_styles.append({'if': {'column_id': month_columns[0]}, 'borderLeft': '3px solid white'})
    if week_columns:
        header_styles.append({'if': {'column_id': week_columns[0]}, 'borderLeft': '3px solid white'})

    page_size = max(len(display_df), 1)
    return dash_table.DataTable(
        id='imp-overview-period-analysis-table',
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
        sort_action='none',
        page_action='none',
        page_size=page_size,
        fill_width=False
    )


def _update_chart_legend_visibility(fig, show_legend):
    """Trim repeated legends in small-multiple chart grids."""
    fig.update_layout(showlegend=show_legend)
    if show_legend:
        fig.update_layout(margin=dict(l=55, r=40, t=30, b=70))
    else:
        fig.update_layout(margin=dict(l=55, r=20, t=30, b=40))
    return fig


layout = html.Div([
    dcc.Store(id='imp-overview-chart-entities-store', storage_type='memory'),
    dcc.Store(id='imp-overview-table-entities-store', storage_type='memory'),
    dcc.Store(id='imp-overview-demand-data-store', storage_type='memory'),
    dcc.Store(id='imp-overview-origin-continent-data-store', storage_type='memory'),
    dcc.Store(id='imp-overview-period-data-store', storage_type='memory'),
    dcc.Store(id='imp-overview-period-display-store', storage_type='memory'),
    dcc.Store(id='imp-overview-period-expanded-importers', storage_type='memory', data=[]),
    dcc.Store(id='imp-overview-period-expanded-origins', storage_type='memory', data=[]),

    dcc.Download(id='imp-overview-download-demand-excel'),
    dcc.Download(id='imp-overview-download-origin-continent-excel'),
    dcc.Download(id='imp-overview-download-period-analysis-excel'),

    html.Div(
        [_create_top_importers_selector_region()],
        className='professional-section-header',
        style={'margin': '0'}
    ),

    html.Div([
        html.Div([
            html.H3('LNG Demand - 30-Day Rolling Average', className='section-title-inline'),
            html.Button(
                'Export to Excel',
                id='imp-overview-export-demand-button',
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
        ], className='inline-section-header', style={'display': 'flex', 'alignItems': 'center'}),
        html.Div(
            'Overview charts follow the importer classification setting above.',
            style={'marginTop': '8px', 'fontSize': '12px', 'color': '#6b7280'}
        ),
        dcc.Loading(
            id='imp-overview-demand-loading',
            children=[html.Div(id='imp-overview-demand-charts-container')],
            type='default'
        )
    ], className='main-section-container', style={'marginBottom': '30px'}),

    html.Div([
        html.Div([
            html.H3('LNG Supply by Origin Continent - 30-Day Rolling Average', className='section-title-inline'),
            html.Button(
                'Export to Excel',
                id='imp-overview-export-origin-continent-button',
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
        ], className='inline-section-header', style={'display': 'flex', 'alignItems': 'center'}),
        html.Div(
            'Origin-continent composition for the chart entities currently shown.',
            style={'marginTop': '8px', 'fontSize': '12px', 'color': '#6b7280'}
        ),
        dcc.Loading(
            id='imp-overview-origin-continent-loading',
            children=[html.Div(id='imp-overview-origin-continent-charts-container')],
            type='default'
        )
    ], className='main-section-container', style={'marginBottom': '30px'}),

    html.Div([
        html.Div([
            html.H3('LNG Supply by Origin and LNG Demand - Period Analysis', className='section-title-inline'),
            html.Button(
                'Export to Excel',
                id='imp-overview-export-period-analysis-button',
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
        ], className='inline-section-header', style={'display': 'flex', 'alignItems': 'center'}),
        html.Div(id='imp-overview-period-analysis-instructions', style={'marginTop': '16px', 'marginBottom': '15px'}),
        dcc.Loading(
            id='imp-overview-period-analysis-loading',
            children=[html.Div(id='imp-overview-period-analysis-container')],
            type='default'
        )
    ], className='main-section-container', style={'marginBottom': '30px'})
])


@callback(
    Output('imp-overview-data-timestamp-display', 'children'),
    Output('imp-overview-coverage-display', 'children'),
    Output('imp-overview-chart-entities-store', 'data'),
    Output('imp-overview-table-entities-store', 'data'),
    Output('imp-overview-demand-data-store', 'data'),
    Output('imp-overview-origin-continent-data-store', 'data'),
    Input('global-refresh-button', 'n_clicks'),
    Input('imp-overview-classification-mode', 'value'),
    prevent_initial_call=False
)
def refresh_overview_data(n_clicks, classification_mode):
    """Load the importer overview entities and chart datasets."""
    try:
        timestamp = _get_latest_upload_timestamp()
        table_entities = _build_destination_entities(classification_mode, limit=None)
        chart_limit = None if classification_mode == 'Classification Level 1' else TOP_IMPORTER_CHART_COUNT
        chart_entities = _build_destination_entities(classification_mode, limit=chart_limit)
        demand_charts_data, origin_continent_charts_data = _build_chart_data_payload(
            chart_entities,
            classification_mode
        )

        if not table_entities:
            coverage_text = 'No importer coverage available.'
        elif classification_mode == 'Classification Level 1':
            coverage_text = (
                f'Charts and table cover all {len(table_entities)} importer Classification Level 1 groups.'
            )
        else:
            coverage_text = (
                f'Charts show the top {len(chart_entities)} importers by latest 30D demand; '
                f'the table covers all {len(table_entities)} importers.'
            )
        return (
            _format_timestamp_display(timestamp),
            coverage_text,
            chart_entities,
            table_entities,
            demand_charts_data,
            origin_continent_charts_data
        )
    except Exception:
        return '', 'No importer coverage available.', [], [], {}, {}


@callback(
    Output('imp-overview-demand-charts-container', 'children'),
    Input('imp-overview-demand-data-store', 'data'),
    Input('imp-overview-chart-entities-store', 'data'),
    prevent_initial_call=False
)
def update_demand_charts(charts_data, importer_entities):
    """Render the demand chart grid using the exporter-page layout pattern."""
    if not charts_data or not importer_entities:
        return html.Div('No data available', style={'textAlign': 'center', 'padding': '20px'})

    num_charts = len(importer_entities)
    if num_charts <= 4:
        chart_width = '25%'
    elif num_charts <= 8:
        chart_width = '25%'
    elif num_charts <= 12:
        chart_width = '20%'
    else:
        chart_width = '16.66%'

    charts = []
    for idx, entity in enumerate(importer_entities):
        entity_name = entity['label']
        entity_data = charts_data.get(entity_name, [])
        entity_df = pd.DataFrame(entity_data)
        if entity_df.empty:
            fig = _empty_timeseries_chart(f'No LNG demand data available for {entity_name}')
        else:
            fig = _create_seasonal_line_chart(entity_df, None, 'rolling_avg', 'mcm/d', 'Demand')
            fig = _update_chart_legend_visibility(fig, idx == len(importer_entities) - 1)

        charts.append(
            html.Div(
                [
                    html.H5(entity_name, style={'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '14px'}),
                    dcc.Graph(
                        id=f'imp-overview-demand-chart-{_slugify_filename_label(entity_name).lower()}',
                        figure=fig,
                        style={'height': '350px'}
                    )
                ],
                style={'width': chart_width, 'display': 'inline-block', 'padding': '0 10px', 'marginBottom': '20px'}
            )
        )

    return html.Div(
        charts,
        style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'flex-start'}
    )


@callback(
    Output('imp-overview-origin-continent-charts-container', 'children'),
    Input('imp-overview-origin-continent-data-store', 'data'),
    Input('imp-overview-chart-entities-store', 'data'),
    prevent_initial_call=False
)
def update_origin_continent_charts(charts_data, importer_entities):
    """Render the origin-continent chart grid using the exporter-page layout pattern."""
    if not charts_data or not importer_entities:
        return html.Div('No data available', style={'textAlign': 'center', 'padding': '20px'})

    num_charts = len(importer_entities)
    if num_charts <= 4:
        chart_width = '25%'
    elif num_charts <= 8:
        chart_width = '25%'
    elif num_charts <= 12:
        chart_width = '20%'
    else:
        chart_width = '16.66%'

    charts = []
    for idx, entity in enumerate(importer_entities):
        entity_name = entity['label']
        entity_data = charts_data.get(entity_name, [])
        entity_df = pd.DataFrame(entity_data)
        if entity_df.empty:
            fig = _empty_timeseries_chart(f'No origin-continent supply data available for {entity_name}')
        else:
            fig = _create_seasonal_line_chart(
                entity_df,
                'continent_origin',
                'rolling_avg',
                'mcm/d',
                'Supply'
            )
            fig = _update_chart_legend_visibility(fig, idx == len(importer_entities) - 1)

        charts.append(
            html.Div(
                [
                    html.H5(entity_name, style={'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '14px'}),
                    dcc.Graph(
                        id=f'imp-overview-origin-continent-chart-{_slugify_filename_label(entity_name).lower()}',
                        figure=fig,
                        style={'height': '350px'}
                    )
                ],
                style={'width': chart_width, 'display': 'inline-block', 'padding': '0 10px', 'marginBottom': '20px'}
            )
        )

    return html.Div(
        charts,
        style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'flex-start'}
    )


@callback(
    Output('imp-overview-period-expanded-importers', 'data', allow_duplicate=True),
    Output('imp-overview-period-expanded-origins', 'data', allow_duplicate=True),
    Input('imp-overview-origin-level-dropdown', 'value'),
    Input('imp-overview-classification-mode', 'value'),
    Input('imp-overview-table-entities-store', 'data'),
    prevent_initial_call=True
)
def reset_period_expansion_state(origin_level, classification_mode, importer_entities):
    """Reset period-analysis expansion when the origin-level control changes."""
    return [], []


@callback(
    Output('imp-overview-period-data-store', 'data'),
    Input('imp-overview-table-entities-store', 'data'),
    Input('imp-overview-classification-mode', 'value'),
    Input('imp-overview-origin-level-dropdown', 'value'),
    prevent_initial_call=False
)
def refresh_period_data(importer_entities, classification_mode, origin_level):
    """Load the raw period-analysis payload for the overview importers."""
    if not importer_entities:
        return []

    try:
        return _build_period_payload(importer_entities, classification_mode, origin_level)
    except Exception:
        return []


@callback(
    Output('imp-overview-period-analysis-container', 'children'),
    Output('imp-overview-period-analysis-instructions', 'children'),
    Output('imp-overview-period-display-store', 'data'),
    Input('imp-overview-period-data-store', 'data'),
    Input('imp-overview-classification-mode', 'value'),
    Input('imp-overview-origin-level-dropdown', 'value'),
    Input('imp-overview-period-expanded-importers', 'data'),
    Input('imp-overview-period-expanded-origins', 'data'),
    Input('imp-overview-table-entities-store', 'data'),
    prevent_initial_call=False
)
def update_period_analysis_table(period_payload, classification_mode, origin_level,
                                 expanded_importers, expanded_origins, importer_entities):
    """Render the combined importer overview period-analysis table."""
    if not period_payload or not importer_entities:
        message = html.Div('No data available for the selected configuration.', style={'textAlign': 'center', 'padding': '20px'})
        return message, '', []

    display_df = _build_period_display_df(
        period_payload,
        origin_level,
        expanded_importers=expanded_importers,
        expanded_origins=expanded_origins
    )
    if display_df.empty:
        message = html.Div('No period-analysis data is available for the overview importers.', style={'textAlign': 'center', 'padding': '20px'})
        return message, '', []

    origin_level_label = ORIGIN_LEVEL_LABELS.get(origin_level, 'Origin')
    importer_level_label = classification_mode.lower()
    if _is_country_origin_level(origin_level):
        instruction_text = (
            f'Click on ▶ importer rows to expand country-level {origin_level_label.lower()} detail. '
            f'The table covers all {len(importer_entities)} importer {importer_level_label} entries.'
        )
    else:
        instruction_text = (
            f'Click on ▶ importer rows to expand by {origin_level_label.lower()}, and on ▶ origin rows to show countries. '
            f'The table covers all {len(importer_entities)} importer {importer_level_label} entries.'
        )

    instructions = html.Span(
        instruction_text,
        style={'fontSize': '14px', 'color': '#666', 'fontStyle': 'italic'}
    )
    return _create_period_analysis_table(display_df, origin_level), instructions, display_df.to_dict('records')


@callback(
    Output('imp-overview-period-expanded-importers', 'data', allow_duplicate=True),
    Output('imp-overview-period-expanded-origins', 'data', allow_duplicate=True),
    Input('imp-overview-period-analysis-table', 'active_cell'),
    State('imp-overview-period-analysis-table', 'derived_viewport_data'),
    State('imp-overview-origin-level-dropdown', 'value'),
    State('imp-overview-period-expanded-importers', 'data'),
    State('imp-overview-period-expanded-origins', 'data'),
    prevent_initial_call=True
)
def toggle_period_row_expansion(active_cell, table_data, origin_level, expanded_importers, expanded_origins):
    """Toggle importer and origin expansion within the overview period table."""
    if not active_cell or not table_data:
        raise PreventUpdate

    row_index = active_cell.get('row')
    if row_index is None:
        raise PreventUpdate

    display_df = pd.DataFrame(table_data)
    if display_df.empty or row_index >= len(display_df):
        raise PreventUpdate

    expanded_importers = list(expanded_importers or [])
    expanded_origins = list(expanded_origins or [])
    clicked_row = display_df.iloc[row_index]

    importer_value = str(clicked_row.get('Importer', ''))
    if importer_value.startswith('▶ ') or importer_value.startswith('▼ '):
        importer_name = importer_value[2:].strip()
        if importer_name in expanded_importers:
            expanded_importers.remove(importer_name)
            expanded_origins = [key for key in expanded_origins if not key.startswith(f'{importer_name}→')]
        else:
            expanded_importers.append(importer_name)
        return expanded_importers, expanded_origins

    if _is_country_origin_level(origin_level):
        raise PreventUpdate

    origin_value = str(clicked_row.get('Origin Level', '')).strip()
    if not (origin_value.startswith('▶ ') or origin_value.startswith('▼ ')):
        raise PreventUpdate

    origin_name = origin_value[2:].strip()
    importer_name = None
    for scan_idx in range(row_index - 1, -1, -1):
        prior_importer_value = str(display_df.iloc[scan_idx].get('Importer', ''))
        if prior_importer_value.startswith('▶ ') or prior_importer_value.startswith('▼ '):
            importer_name = prior_importer_value[2:].strip()
            break

    if not importer_name:
        raise PreventUpdate

    origin_key = f'{importer_name}→{origin_name}'
    if origin_key in expanded_origins:
        expanded_origins.remove(origin_key)
    else:
        expanded_origins.append(origin_key)
    return expanded_importers, expanded_origins


@callback(
    Output('imp-overview-download-demand-excel', 'data'),
    Input('imp-overview-export-demand-button', 'n_clicks'),
    State('imp-overview-demand-data-store', 'data'),
    prevent_initial_call=True
)
def export_demand_to_excel(n_clicks, charts_data):
    """Export the currently rendered demand-chart data."""
    if not n_clicks:
        raise PreventUpdate

    export_df = _build_chart_export_df(charts_data)
    if export_df.empty:
        raise PreventUpdate

    return _send_export_dataframe(
        export_df,
        'importers_lng_demand_30d_rolling',
        'Demand'
    )


@callback(
    Output('imp-overview-download-origin-continent-excel', 'data'),
    Input('imp-overview-export-origin-continent-button', 'n_clicks'),
    State('imp-overview-origin-continent-data-store', 'data'),
    prevent_initial_call=True
)
def export_origin_continent_to_excel(n_clicks, charts_data):
    """Export the currently rendered origin-continent chart data."""
    if not n_clicks:
        raise PreventUpdate

    export_df = _build_chart_export_df(charts_data)
    if export_df.empty:
        raise PreventUpdate

    return _send_export_dataframe(
        export_df,
        'importers_lng_supply_by_origin_continent_30d_rolling',
        'Origin Continent'
    )


@callback(
    Output('imp-overview-download-period-analysis-excel', 'data'),
    Input('imp-overview-export-period-analysis-button', 'n_clicks'),
    State('imp-overview-period-display-store', 'data'),
    State('imp-overview-origin-level-dropdown', 'value'),
    prevent_initial_call=True
)
def export_period_analysis_to_excel(n_clicks, period_display_data, origin_level):
    """Export the currently rendered period-analysis table."""
    if not n_clicks or not period_display_data:
        raise PreventUpdate

    export_df = pd.DataFrame(period_display_data)
    if export_df.empty:
        raise PreventUpdate

    safe_origin_level = _slugify_filename_label(ORIGIN_LEVEL_LABELS.get(origin_level, 'origin'))
    return _send_export_dataframe(
        export_df,
        f'importers_lng_supply_by_origin_period_analysis_{safe_origin_level.lower()}',
        'Period Analysis'
    )
