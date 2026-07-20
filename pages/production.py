from dash import html, dcc, callback, Input, Output, State, no_update
import plotly.graph_objects as go
import pandas as pd
from datetime import date, datetime
from functools import lru_cache
from io import BytesIO
import json
import os
import sys
from dash.dash_table.Format import Format, Scheme
from utils.ag_grid_tables import create_ag_grid_from_datatable
from dash.exceptions import PreventUpdate

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

from fundamentals.terminals.terminal_output_utils import (
    fetch_capacity_ramp_production_monthly,
    get_capacity_ramp_production_catalog,
)
from utils.export_flow_data import (
    build_export_flow_matrix,
    default_selected_countries,
    get_available_countries,
)
from utils.global_supply_comparison import (
    SUPPLY_SOURCE_ORDER,
    TIME_VIEW_CONFIG,
    aggregate_global_supply_comparison,
    fetch_global_supply_comparison,
    normalize_supply_time_view,
)
from utils.table_styles import (
    StandardTableStyleManager,
    TABLE_COLORS,
    format_table_cell_value_2dp as _format_table_cell_value,
)
from pages.terminals import (
    PRIMARY_COLORS,
    convert_to_mcmd,
    engine,
    hex_to_rgb,
)


def _create_empty_volume_figure(message):
    """Create an empty state figure for the production page."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=18, family='Arial', color='#64748b')
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=80, r=80, t=80, b=80),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig


def _create_empty_supply_comparison_figure(message):
    fig = _create_empty_volume_figure(message)
    fig.update_layout(height=460, margin=dict(l=72, r=36, t=72, b=58))
    return fig


@lru_cache(maxsize=32)
def _fetch_global_supply_comparison_cached(run_id, _refresh_token=None):
    comparison_df, metadata = fetch_global_supply_comparison(
        engine,
        int(run_id),
    )
    return comparison_df, metadata


def create_global_supply_comparison_chart(comparison_df, metadata, time_view='monthly'):
    """Compare the selected ramp run with the latest global provider supply views."""
    if comparison_df is None or comparison_df.empty:
        return _create_empty_supply_comparison_figure(
            "No global supply comparison data is available"
        )

    time_view = normalize_supply_time_view(time_view)
    view_config = TIME_VIEW_CONFIG[time_view]
    if 'supply_mt' not in comparison_df.columns:
        comparison_df = aggregate_global_supply_comparison(comparison_df, time_view)
    source_styles = {
        'Our ramp forecast': {'color': '#111827', 'width': 3.2, 'dash': 'solid'},
        'Energy Aspects': {'color': '#D97706', 'width': 2.2, 'dash': 'solid'},
        'Platts': {'color': '#2563EB', 'width': 2.2, 'dash': 'solid'},
        'WoodMac': {'color': '#16803C', 'width': 2.2, 'dash': 'solid'},
    }
    fig = go.Figure()
    for source_name in source_styles:
        source_df = comparison_df[comparison_df['source'].eq(source_name)].sort_values('period_start')
        if source_df.empty:
            continue
        style = source_styles[source_name]
        fig.add_trace(
            go.Scatter(
                x=source_df['period_start'],
                y=source_df['supply_mt'],
                customdata=source_df['period_label'],
                mode='lines',
                name=source_name,
                connectgaps=False,
                line=style,
                hovertemplate=(
                    f'<b>{source_name}</b><br>'
                    f"{view_config['label']}: %{{customdata}}<br>"
                    f"Supply: %{{y:.2f}} {view_config['unit']}<extra></extra>"
                ),
            )
        )

    window_start = pd.to_datetime(metadata.get('window_start'), errors='coerce')
    forecast_start = pd.to_datetime(metadata.get('forecast_start'), errors='coerce')
    window_end = pd.to_datetime(metadata.get('window_end'), errors='coerce')
    if pd.notna(forecast_start) and pd.notna(window_end):
        fig.add_vrect(
            x0=forecast_start,
            x1=window_end,
            fillcolor='rgba(148, 163, 184, 0.11)',
            line_width=0,
            layer='below',
        )
        fig.add_vline(
            x=forecast_start,
            line_width=1.2,
            line_dash='dash',
            line_color='#64748B',
        )
        fig.add_annotation(
            x=forecast_start,
            y=1.02,
            xref='x',
            yref='paper',
            text='Forecast',
            showarrow=False,
            xanchor='left',
            font={'size': 11, 'color': '#475569'},
        )

    fig.update_layout(
        title={
            'text': (
                'Global LNG Supply: Ramp Scenario vs Providers'
                '<br><sup>Five complete historical years and forecasts through 2031</sup>'
            ),
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial', 'color': '#1F2937'},
        },
        xaxis={
            'title': '',
            'type': 'date',
            'range': [window_start, window_end],
            'tickformat': '%Y' if time_view == 'yearly' else '%b\n%Y',
            'dtick': 'M12' if time_view == 'yearly' else 'M6',
            'showgrid': True,
            'gridcolor': '#E5E7EB',
            'linecolor': '#CBD5E1',
        },
        yaxis={
            'title': f"{view_config['axis_label']} ({view_config['unit']})",
            'rangemode': 'tozero',
            'showgrid': True,
            'gridcolor': '#E5E7EB',
            'linecolor': '#CBD5E1',
            'zeroline': False,
        },
        height=490,
        margin={'l': 72, 'r': 36, 't': 82, 'b': 92},
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend={
            'orientation': 'h',
            'yanchor': 'top',
            'y': -0.14,
            'xanchor': 'center',
            'x': 0.5,
            'font': {'size': 11},
        },
    )
    return fig


def _create_global_supply_comparison_table(comparison_df, time_view='monthly'):
    """Create the period-by-source table shown beneath the comparison chart."""
    if comparison_df is None or comparison_df.empty:
        return _create_empty_table_state("No comparison values are available.")

    time_view = normalize_supply_time_view(time_view)
    view_config = TIME_VIEW_CONFIG[time_view]
    if 'supply_mt' not in comparison_df.columns:
        comparison_df = aggregate_global_supply_comparison(comparison_df, time_view)
    if comparison_df.empty:
        return _create_empty_table_state("No complete periods are available for this time view.")

    table_df = comparison_df.pivot_table(
        index=['period_start', 'period_label'],
        columns='source',
        values='supply_mt',
        aggfunc='sum',
    ).reset_index()
    table_df = table_df.sort_values('period_start').drop(columns='period_start')
    table_df = table_df.rename(columns={'period_label': view_config['label']})
    for source_name in SUPPLY_SOURCE_ORDER:
        if source_name not in table_df.columns:
            table_df[source_name] = None
    table_df = table_df[[view_config['label'], *SUPPLY_SOURCE_ORDER]]
    table_df[list(SUPPLY_SOURCE_ORDER)] = table_df[list(SUPPLY_SOURCE_ORDER)].round(2)

    columns = [{"name": view_config['label'], "id": view_config['label']}]
    columns.extend(
        {
            "name": f"{source_name} ({view_config['unit']})",
            "id": source_name,
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        }
        for source_name in SUPPLY_SOURCE_ORDER
    )
    base_config = StandardTableStyleManager.get_base_datatable_config()
    style_data_conditional = list(base_config['style_data_conditional'])
    style_data_conditional.append(
        {
            "if": {"column_id": view_config['label']},
            "backgroundColor": "#f8fafc",
            "fontWeight": "600",
            "color": TABLE_COLORS['text_primary'],
        }
    )
    column_styles = [
        {
            "if": {"column_id": view_config['label']},
            "minWidth": "132px",
            "width": "20%",
            "textAlign": "left",
        }
    ]
    column_styles.extend(
        {
            "if": {"column_id": source_name},
            "minWidth": "150px",
            "width": "20%",
        }
        for source_name in SUPPLY_SOURCE_ORDER
    )
    return html.Div(
        [
            html.Div(
                f"Comparison Values | {view_config['unit']}",
                style={
                    'fontSize': '14px',
                    'fontWeight': '700',
                    'color': '#1F2937',
                    'margin': '4px 0 8px',
                },
            ),
            create_ag_grid_from_datatable(
                id='production-global-supply-comparison-values-grid',
                columns=columns,
                data=table_df.to_dict('records'),
                sort_action='native',
                page_action='none',
                fill_width=True,
                fixed_columns={"headers": True, "data": 1},
                style_table={
                    "overflowX": "auto",
                    "overflowY": "auto",
                    "maxHeight": "430px",
                    "width": "100%",
                    "minWidth": "100%",
                },
                style_cell_conditional=column_styles,
                style_data_conditional=style_data_conditional,
            ),
        ],
        style={'margin': '0 10px 24px'},
    )


def _build_global_supply_source_note(metadata):
    if not metadata:
        return ""
    parts = []
    for source_name in ('Our ramp forecast', 'Energy Aspects', 'Platts', 'WoodMac'):
        source = (metadata.get('sources') or {}).get(source_name)
        if not source:
            continue
        coverage = f"{source.get('first_month')} to {source.get('last_month')}"
        parts.append(f"{source_name}: {coverage}; {source.get('detail')}")
    parts.extend(metadata.get('warnings') or [])
    return html.Div(
        " | ".join(parts),
        style={
            'fontSize': '11px',
            'lineHeight': '1.45',
            'color': '#64748B',
            'margin': '2px 10px 20px',
        },
    )


def _create_empty_table_state(message):
    """Return a consistent empty state for the country table."""
    return html.Div(message, className="balance-empty-state")


def _get_total_output_label(selected_unit='mtpa'):
    """Return the total-column label matching the active unit."""
    return 'Total MTPA' if selected_unit == 'mtpa' else 'Total Mcm/d'


def _build_volume_table_column_styles(df):
    """Create responsive widths so wide country matrices remain readable."""
    column_styles = []
    column_weights = {}
    column_min_widths = {}

    for column_name in df.columns:
        header_length = len(str(column_name))
        value_lengths = df[column_name].map(_format_table_cell_value).map(len)
        max_length = max([header_length] + value_lengths.tolist()) if not df.empty else header_length

        if column_name == 'Month':
            column_weights[column_name] = max(8, min(max_length, 12))
            column_min_widths[column_name] = 92
        elif str(column_name).startswith('Total '):
            column_weights[column_name] = max(8, min(max_length, 14))
            column_min_widths[column_name] = 96
        else:
            column_weights[column_name] = max(6, min(max_length, 18))
            column_min_widths[column_name] = 72

    total_weight = sum(column_weights.values()) or 1

    for column_name in df.columns:
        width_pct = column_weights[column_name] / total_weight * 100
        style_entry = {
            "if": {"column_id": column_name},
            "minWidth": f"{column_min_widths[column_name]}px",
            "width": f"{width_pct:.2f}%",
        }

        if column_name == 'Month':
            style_entry["textAlign"] = "left"

        column_styles.append(style_entry)

    return column_styles


def _json_safe_value(value):
    """Convert catalog metadata into values accepted by dcc.Store."""
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


@lru_cache(maxsize=64)
def _get_capacity_ramp_catalog_cached(_refresh_token=None):
    catalog_df = get_capacity_ramp_production_catalog(engine)
    return tuple(
        {
            column_name: _json_safe_value(value)
            for column_name, value in row.items()
        }
        for row in catalog_df.to_dict("records")
    )


@lru_cache(maxsize=64)
def _fetch_volume_country_dataframe_cached(
    run_id,
    new_capacity_only,
    start_year,
    end_year,
    breakdown,
    _refresh_token=None,
):
    """Fetch production data once per refresh-sensitive filter set."""
    raw_df = fetch_capacity_ramp_production_monthly(
        run_id=int(run_id),
        engine=engine,
        start_year=start_year,
        end_year=end_year,
        new_capacity_only=new_capacity_only,
    )

    if raw_df.empty:
        return pd.DataFrame(columns=['month', 'country_name', 'total_output'])

    breakdown = breakdown or 'country'
    if breakdown == 'country':
        country_df = raw_df.groupby(
            ['year', 'month', 'country_name'], as_index=False
        )['total_output'].sum()
    elif breakdown == 'project':
        country_df = raw_df.groupby(
            ['year', 'month', 'plant_name', 'country_name'], as_index=False
        )['total_output'].sum()
        country_df = country_df.rename(columns={'plant_name': 'group_name'})
        country_df = country_df[['year', 'month', 'group_name', 'total_output']]
        country_df = country_df.rename(columns={'group_name': 'country_name'})
    else:
        country_df = raw_df[['year', 'month', 'plant_name', 'lng_train_name_short', 'total_output']].copy()
        country_df['country_name'] = (
            country_df['plant_name'].fillna('Unknown terminal')
            + ' - '
            + country_df['lng_train_name_short'].fillna('Train')
        )
        country_df = country_df.groupby(
            ['year', 'month', 'country_name'], as_index=False
        )['total_output'].sum()

    country_df['month'] = pd.to_datetime(country_df[['year', 'month']].assign(day=1))

    return country_df[['month', 'country_name', 'total_output']]


def _prepare_volume_country_dataframe(
    run_id=None,
    selected_unit='mtpa',
    new_capacity_only=False,
    start_year=2025,
    end_year=2040,
    breakdown='country',
    refresh_token=None,
):
    """Return the production-page data in the selected display unit."""
    if run_id is None:
        return pd.DataFrame(columns=['month', 'country_name', 'total_output'])

    country_df = _fetch_volume_country_dataframe_cached(
        int(run_id),
        bool(new_capacity_only),
        int(start_year),
        int(end_year),
        breakdown or 'country',
        refresh_token,
    ).copy()

    if selected_unit == 'mcmd':
        country_df['total_output'] = convert_to_mcmd(country_df['total_output'])

    return country_df


def _find_catalog_scenario(catalog, scenario_id):
    if scenario_id is None:
        return None
    try:
        resolved_id = int(scenario_id)
    except (TypeError, ValueError):
        return None
    return next(
        (row for row in catalog if row.get('scenario_id') == resolved_id),
        None,
    )


def _format_catalog_option_label(row):
    scenario_name = row.get('scenario_name') or f"Scenario {row.get('scenario_id')}"
    run_id = row.get('display_run_id')
    if run_id is None:
        return f"{scenario_name} - No ramp run"
    if int(row.get('display_blocking_qa_count') or 0) > 0:
        return f"{scenario_name} - QA blocked (run {run_id})"
    if row.get('fallback_reason'):
        return f"{scenario_name} - Fallback run {run_id}"
    return f"{scenario_name} - Run {run_id}"


def _format_status_timestamp(value):
    if not value:
        return "Unknown"
    timestamp = pd.to_datetime(value, errors='coerce', utc=True)
    if pd.isna(timestamp):
        return str(value)
    return timestamp.strftime('%Y-%m-%d %H:%M UTC')


def _format_status_month(value):
    if not value:
        return "Unknown"
    timestamp = pd.to_datetime(value, errors='coerce')
    if pd.isna(timestamp):
        return str(value)
    return timestamp.strftime('%b %Y')


def _parse_blocker_summaries(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    return value if isinstance(value, list) else []


def _build_ramp_run_status_banner(metadata):
    """Show run provenance and make analytical QA limitations impossible to miss."""
    base_style = {
        'border': '1px solid',
        'borderRadius': '4px',
        'padding': '10px 14px',
        'margin': '12px 0 18px',
        'fontSize': '13px',
        'lineHeight': '1.45',
    }
    if not metadata:
        return html.Div(
            "Select a Capacity scenario to view its ramp forecast.",
            style={**base_style, 'backgroundColor': '#f8fafc', 'borderColor': '#cbd5e1'},
        )

    scenario_name = metadata.get('scenario_name') or 'Capacity scenario'
    display_run_id = metadata.get('display_run_id')
    if display_run_id is None:
        latest_run_id = metadata.get('latest_attempt_run_id')
        latest_status = metadata.get('latest_attempt_run_status')
        attempt_text = (
            f" Latest attempt: run {latest_run_id} ({latest_status})."
            if latest_run_id is not None
            else ""
        )
        return html.Div(
            [
                html.Strong(f"No ramp output is available for {scenario_name}."),
                html.Span(attempt_text),
                html.Span(" Generate the scenario forecast from "),
                html.A("Capacity", href='/capacity', style={'fontWeight': '600'}),
                html.Span(" before using Production."),
            ],
            style={
                **base_style,
                'backgroundColor': '#fff7ed',
                'borderColor': '#f97316',
                'color': '#9a3412',
            },
        )

    is_blocked = int(metadata.get('display_blocking_qa_count') or 0) > 0
    is_stale = bool(metadata.get('display_is_stale'))
    is_current_published = bool(metadata.get('display_is_current_published'))
    fallback_reason = metadata.get('fallback_reason')
    blocking_count = int(metadata.get('display_blocking_qa_count') or 0)
    generator_version = metadata.get('display_generator_version')
    runtime_version = f"generator {generator_version or 'Unknown'}"
    generated_at = _format_status_timestamp(metadata.get('display_generated_at'))
    horizon_start = _format_status_month(metadata.get('display_horizon_start_month'))
    horizon_end = _format_status_month(metadata.get('display_horizon_end_month'))
    row_count = int(metadata.get('display_monthly_row_count') or 0)
    train_count = int(metadata.get('display_train_count') or 0)

    detail_parts = [
        f"Run {display_run_id}",
        runtime_version,
        f"{train_count:,} trains",
        f"generated {generated_at}",
        f"horizon {horizon_start} to {horizon_end}",
        f"{row_count:,} monthly rows",
        "officially published" if is_current_published else "analytical only",
    ]
    children = [
        html.Strong(
            "QA-blocked ramp output shown for analysis. "
            if is_blocked
            else "Ramp output ready. "
        ),
        html.Span(" | ".join(detail_parts)),
    ]
    if is_blocked:
        children.extend(
            [html.Br(), html.Strong(f"{blocking_count} blocking QA issue(s).")]
        )
        for blocker in _parse_blocker_summaries(
            metadata.get('display_blocking_qa_summaries')
        )[:3]:
            label = " / ".join(
                str(blocker.get(key))
                for key in ('country_name', 'plant_name', 'train_label')
                if blocker.get(key)
            )
            message = blocker.get('message') or blocker.get('qa_type') or 'Blocking QA issue'
            children.extend([html.Br(), html.Span(f"- {label + ': ' if label else ''}{message}")])
    if fallback_reason:
        children.extend([html.Br(), html.Strong("Fallback: "), html.Span(fallback_reason)])
    if is_stale:
        children.extend(
            [
                html.Br(),
                html.Strong("Stale: "),
                html.Span("scenario, SQL profile, baseline, or train registry inputs changed after this run."),
            ]
        )

    has_warning = is_blocked or bool(fallback_reason) or is_stale
    return html.Div(
        children,
        style={
            **base_style,
            'backgroundColor': '#fff7ed' if has_warning else '#f0fdf4',
            'borderColor': '#f97316' if has_warning else '#22c55e',
            'color': '#9a3412' if has_warning else '#166534',
        },
    )


def _build_direct_pivot_matrix(raw_df, selected_unit='mtpa'):
    """Build a month-by-group pivot for plant/train breakdowns (no Rest of the World logic)."""
    total_column_label = _get_total_output_label(selected_unit)
    if raw_df.empty:
        return pd.DataFrame(columns=['Month', total_column_label])

    pivot = raw_df.pivot_table(
        index='month', columns='country_name', values='total_output', aggfunc='sum'
    ).fillna(0)

    col_totals = pivot.sum().sort_values(ascending=False)
    pivot = pivot[col_totals.index]

    pivot[total_column_label] = pivot.sum(axis=1)
    pivot = pivot[pivot[total_column_label] > 0]
    pivot.index = pivot.index.strftime('%Y-%m')

    result = pivot.reset_index().rename(columns={'month': 'Month'})
    cols = ['Month'] + [c for c in result.columns if c not in {'Month', total_column_label}] + [total_column_label]
    return result[cols]


def _get_available_volume_countries(raw_df):
    """Reuse the shared balance-page country ordering for default selections."""
    if raw_df.empty:
        return []

    available_source_df = raw_df.rename(columns={'total_output': 'total_mmtpa'})
    return get_available_countries(
        [available_source_df[['month', 'country_name', 'total_mmtpa']]]
    )


def _build_volume_country_matrix(
    raw_df,
    selected_countries,
    other_countries_mode='rest_of_world',
    selected_unit='mtpa',
):
    """Build the month-by-country matrix shown below the chart."""
    total_column_label = _get_total_output_label(selected_unit)

    if raw_df.empty:
        return pd.DataFrame(columns=['Month', total_column_label])

    matrix_source_df = raw_df.rename(columns={'total_output': 'total_mmtpa'})
    matrix_df = build_export_flow_matrix(
        matrix_source_df[['month', 'country_name', 'total_mmtpa']],
        selected_countries,
        other_countries_mode,
    )

    if matrix_df.empty:
        return pd.DataFrame(columns=['Month', total_column_label])

    matrix_df = matrix_df.rename(columns={'Total MMTPA': total_column_label})
    country_columns = [
        column_name
        for column_name in matrix_df.columns
        if column_name not in {'Month', total_column_label}
    ]

    ordered_columns = ['Month'] + country_columns + [total_column_label]
    return matrix_df[ordered_columns]


def _create_volume_country_table(table_id, df):
    """Create a formatted table for the month-by-country matrix."""
    if df.empty:
        return _create_empty_table_state("No data available for the current selection.")

    base_config = StandardTableStyleManager.get_base_datatable_config()
    numeric_columns = [column for column in df.columns if column != 'Month']

    columns = [{"name": "Month", "id": "Month"}]
    columns.extend(
        {
            "name": column_name,
            "id": column_name,
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        }
        for column_name in numeric_columns
    )

    style_data_conditional = list(base_config['style_data_conditional'])
    style_data_conditional.append(
        {
            "if": {"column_id": "Month"},
            "backgroundColor": "#f8fafc",
            "fontWeight": "600",
            "color": TABLE_COLORS['text_primary'],
        }
    )

    for column_name in numeric_columns:
        if str(column_name).startswith('Total '):
            style_data_conditional.append(
                {
                    "if": {"column_id": column_name},
                    "backgroundColor": "#edf6fd",
                    "fontWeight": "700",
                    "color": TABLE_COLORS['primary_dark'],
                }
            )
        elif column_name == 'Rest of the World':
            style_data_conditional.append(
                {
                    "if": {"column_id": column_name},
                    "backgroundColor": "#f8f9fa",
                    "color": TABLE_COLORS['text_secondary'],
                }
            )

    return create_ag_grid_from_datatable(
        id=table_id,
        columns=columns,
        data=df.to_dict('records'),
        sort_action='native',
        page_action='none',
        fill_width=True,
        fixed_columns={"headers": True, "data": 1},
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": "560px",
            "width": "100%",
            "minWidth": "100%",
            "marginTop": "20px",
        },
        style_cell_conditional=_build_volume_table_column_styles(df),
        style_data_conditional=style_data_conditional,
    )


def create_volume_country_area_chart(matrix_df, selected_unit='mtpa'):
    """Create the stacked area chart using the visible country columns."""
    total_column_label = _get_total_output_label(selected_unit)
    unit_label = 'MTPA' if selected_unit == 'mtpa' else 'Mcm/d'
    if matrix_df.empty:
        return _create_empty_volume_figure("No volume data available")

    country_columns = [
        column_name
        for column_name in matrix_df.columns
        if column_name not in {'Month', total_column_label}
    ]

    if not country_columns:
        return _create_empty_volume_figure(
            "Select at least one country or switch to Rest of the World mode."
        )

    plot_df = matrix_df.copy()
    plot_df['date'] = pd.to_datetime(plot_df['Month'] + '-01')
    pivot_df = plot_df.set_index('date')[country_columns]
    pivot_df = pivot_df[pivot_df.sum(axis=1) > 0]

    if pivot_df.empty:
        return _create_empty_volume_figure("No volume data available")

    column_totals = pivot_df.sum().sort_values(ascending=False)
    pivot_df = pivot_df[column_totals.index]

    fig = go.Figure()
    for group_name in pivot_df.columns:
        color = '#94A3B8' if group_name == 'Rest of the World' else PRIMARY_COLORS.get(group_name, '#666666')
        rgb = hex_to_rgb(color)

        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[group_name],
            mode='lines',
            name=group_name,
            line=dict(width=0.5, color=color),
            fill='tonexty',
            fillcolor=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.7)',
            hovertemplate=f'<b>{group_name}</b><br>Date: %{{x|%b %Y}}<br>Output: %{{y:.1f}} {unit_label}<extra></extra>',
            stackgroup='one'
        ))

    start_date = pivot_df.index.min()
    end_date = pivot_df.index.max()

    fig.update_layout(
        title={
            'text': f'Cumulative Monthly LNG Output by Country ({unit_label}) | {start_date.year}-{end_date.year}',
            'font': {'size': 18, 'family': 'Arial', 'color': '#333333'},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'
        },
        xaxis=dict(
            title='',
            range=[start_date, end_date],
            type='date',
            tickformat='%b\n%Y',
            dtick='M3',
            tickfont=dict(size=9, family='Arial', color='#333333'),
            showgrid=True,
            gridcolor='#E8E8E8',
            showline=True,
            linewidth=1,
            linecolor='#CCCCCC'
        ),
        yaxis=dict(
            title=dict(
                text=f'Monthly Output ({unit_label})',
                font=dict(size=12, family='Arial', color='#333333')
            ),
            showgrid=True,
            gridcolor='#E8E8E8',
            showline=True,
            linewidth=1,
            linecolor='#CCCCCC',
            zeroline=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=80, r=180, t=80, b=80),
        hovermode='x unified',
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02,
            font=dict(size=10, family='Arial'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#CCCCCC',
            borderwidth=1
        )
    )

    return fig


def layout():
    return html.Div([
        html.Div(
            [
                            html.Div(
                                [
                                    html.Div("Scenario", className="filter-group-header"),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id='capacity-scenario-dropdown',
                                                options=[],
                                                value=None,
                                                clearable=False,
                                                className='filter-dropdown',
                                                style={'minWidth': '260px', 'width': '260px'}
                                            ),
                                        ],
                                        style={
                                            'display': 'flex',
                                            'gap': '8px',
                                            'alignItems': 'center',
                                            'flexWrap': 'nowrap',
                                        }
                                    ),
                                ],
                                className="filter-group",
                                style={'minWidth': '280px'},
                            ),
                            html.Div(
                                [
                                    html.Div("Unit of Measure", className="filter-group-header"),
                                    dcc.Dropdown(
                                        id='capacity-unit-dropdown',
                                    options=[
                                        {'label': 'MTPA (Million Tonnes Per Annum)', 'value': 'mtpa'},
                                        {'label': 'Mcm/d (Million Cubic Meters per Day)', 'value': 'mcmd'}
                                    ],
                                    value='mtpa',
                                    clearable=False,
                                    className='filter-dropdown',
                                    style={'width': '100%'}
                                ),
                            ],
                            className="filter-group",
                        ),
                            html.Div(
                                [
                                    html.Div("Scope", className="filter-group-header"),
                                    dcc.Checklist(
                                        id='capacity-new-capacity-checkbox',
                                    options=[{'label': ' New capacity only', 'value': 'new_only'}],
                                    value=[],
                                    style={'fontSize': '14px', 'fontWeight': '600'}
                                ),
                            ],
                            className="filter-group",
                        ),
                            html.Div(
                                [
                                    html.Div("Group By", className="filter-group-header"),
                                    dcc.Dropdown(
                                        id='capacity-breakdown-dropdown',
                                    options=[
                                        {'label': 'Country', 'value': 'country'},
                                        {'label': 'Plant', 'value': 'project'},
                                        {'label': 'Train', 'value': 'train'},
                                    ],
                                    value='country',
                                    clearable=False,
                                    className='filter-dropdown',
                                    style={'width': '100%'}
                                ),
                            ],
                            className="filter-group",
                        ),
                            html.Div(
                                [
                                    html.Div("Country Columns", className="filter-group-header"),
                                    dcc.Dropdown(
                                        id='capacity-country-columns-dropdown',
                                    options=[],
                                    value=None,
                                    multi=True,
                                    placeholder='Select countries to keep as separate columns',
                                    className='filter-dropdown',
                                    style={'width': '100%'}
                                ),
                            ],
                            id='capacity-country-columns-section',
                            className="filter-group",
                        ),
                            html.Div(
                                [
                                    html.Div("Other Countries", className="filter-group-header"),
                                    dcc.RadioItems(
                                        id='capacity-other-country-mode',
                                    options=[
                                        {
                                            'label': 'Include as Rest of the World',
                                            'value': 'rest_of_world',
                                        },
                                        {
                                            'label': 'Exclude from the chart and table',
                                            'value': 'exclude',
                                        },
                                    ],
                                    value='rest_of_world',
                                    className='balance-radio-group',
                                    labelStyle={'display': 'inline-flex', 'alignItems': 'center'},
                                    inputStyle={'marginRight': '6px'},
                                ),
                            ],
                            id='capacity-other-country-section',
                            className="filter-group",
                        ),
                            html.Div(
                                [
                                    html.Div("Export", className="filter-group-header"),
                                    html.Button(
                                        'Export to Excel',
                                    id='capacity-export-excel-button',
                                    n_clicks=0,
                                    style={
                                        'padding': '8px 16px',
                                        'backgroundColor': '#2E86C1',
                                        'color': 'white',
                                        'border': 'none',
                                        'borderRadius': '4px',
                                        'cursor': 'pointer',
                                        'fontWeight': 'bold',
                                        'fontSize': '14px'
                                    }
                                ),
                                dcc.Download(id='capacity-download-excel')
                            ],
                            className="filter-group",
                        ),
                    ],
            className="professional-section-header",
            style={
                'display': 'flex',
                'gap': '12px',
                'alignItems': 'flex-start',
                'flexWrap': 'wrap',
                'margin': '0',
            }
        ),
        dcc.Store(id='production-ramp-run-store', storage_type='memory'),
        html.Div(id='production-ramp-run-status'),
        html.Div([
            html.Div(
                [
                    html.Div("Time View", className="filter-group-header"),
                    dcc.Dropdown(
                        id='production-global-supply-time-view',
                        options=[
                            {'label': 'Month', 'value': 'monthly'},
                            {'label': 'Quarter', 'value': 'quarterly'},
                            {'label': 'Season', 'value': 'season'},
                            {'label': 'Year', 'value': 'yearly'},
                        ],
                        value='monthly',
                        clearable=False,
                        searchable=False,
                        style={'width': '180px'},
                    ),
                ],
                className='filter-group',
                style={'width': '200px', 'margin': '10px 10px 0'},
            ),
            dcc.Graph(
                id='production-global-supply-comparison-chart',
                figure=_create_empty_supply_comparison_figure(
                    "Loading global supply comparison..."
                ),
                config={'displayModeBar': True, 'displaylogo': False},
                style={'height': '100%'},
            ),
            html.Div(id='production-global-supply-comparison-note'),
            html.Div(id='production-global-supply-comparison-table'),
        ]),
        html.Div([
            html.Div([
                html.Label("Year Range:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.RangeSlider(
                    id='capacity-year-range-slider',
                    min=2000,
                    max=2055,
                    step=1,
                    value=[2025, 2040],
                    marks={year: str(year) for year in range(2000, 2056, 5)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    className='year-range-slider'
                )
            ], style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'})
        ], style={'textAlign': 'center'}),
        html.Div([
            dcc.Graph(
                id='capacity-volume-area-chart',
                figure=_create_empty_volume_figure("Loading capacity data..."),
                config={'displayModeBar': True, 'displaylogo': False},
                style={'height': '100%'}
            )
        ], style={'marginTop': '20px'}),
        html.Div(
            id='capacity-country-table-container',
            children=_create_empty_table_state("Loading capacity data..."),
            style={'marginTop': '20px'}
        )
    ])


@callback(
    Output('capacity-scenario-dropdown', 'options'),
    Output('capacity-scenario-dropdown', 'value'),
    Input('global-refresh-button', 'n_clicks'),
    State('capacity-scenario-dropdown', 'value'),
)
def populate_capacity_scenario_options(refresh_clicks, current_scenario_id):
    """List every Capacity scenario and select Base Case by default."""
    catalog = _get_capacity_ramp_catalog_cached(refresh_clicks)
    options = [
        {
            'label': _format_catalog_option_label(row),
            'value': row['scenario_id'],
        }
        for row in catalog
    ]
    scenario_ids = {row['scenario_id'] for row in catalog}
    try:
        resolved_current_id = int(current_scenario_id)
    except (TypeError, ValueError):
        resolved_current_id = None
    if resolved_current_id in scenario_ids:
        selected_scenario_id = resolved_current_id
    else:
        base_case = next(
            (
                row for row in catalog
                if str(row.get('scenario_name') or '').casefold() == 'base case'
            ),
            None,
        )
        selected_scenario_id = (
            base_case['scenario_id']
            if base_case is not None
            else (catalog[0]['scenario_id'] if catalog else None)
        )
    return options, selected_scenario_id


@callback(
    Output('production-ramp-run-store', 'data'),
    Output('production-ramp-run-status', 'children'),
    Input('capacity-scenario-dropdown', 'value'),
    Input('global-refresh-button', 'n_clicks'),
)
def select_capacity_ramp_display_run(scenario_id, refresh_clicks):
    """Freeze the selected scenario's display run for all downstream callbacks."""
    catalog = _get_capacity_ramp_catalog_cached(refresh_clicks)
    metadata = _find_catalog_scenario(catalog, scenario_id)
    store_data = dict(metadata) if metadata else {}
    store_data['refresh_token'] = refresh_clicks
    return store_data, _build_ramp_run_status_banner(metadata)


@callback(
    Output('production-global-supply-comparison-chart', 'figure'),
    Output('production-global-supply-comparison-note', 'children'),
    Output('production-global-supply-comparison-table', 'children'),
    Input('production-ramp-run-store', 'data'),
    Input('production-global-supply-time-view', 'value'),
)
def update_global_supply_comparison(run_metadata, time_view):
    """Render one fixed-window provider comparison for the selected scenario run."""
    run_id = (run_metadata or {}).get('display_run_id')
    if run_id is None:
        return (
            _create_empty_supply_comparison_figure(
                "No ramp run is available for provider comparison"
            ),
            "",
            _create_empty_table_state("No ramp run is available for provider comparison."),
        )
    comparison_df, metadata = _fetch_global_supply_comparison_cached(
        int(run_id),
        (run_metadata or {}).get('refresh_token'),
    )
    time_view = normalize_supply_time_view(time_view)
    period_df = aggregate_global_supply_comparison(comparison_df.copy(), time_view)
    source_note = _build_global_supply_source_note(metadata)
    completeness_note = html.Div(
        (
            "Values are physical monthly supply."
            if time_view == 'monthly'
            else "Only complete periods are shown; blank cells mean the source does not cover every month in that period."
        ),
        style={
            'fontSize': '11px',
            'lineHeight': '1.4',
            'color': '#64748B',
            'margin': '-14px 10px 16px',
        },
    )
    return (
        create_global_supply_comparison_chart(period_df, metadata, time_view),
        [source_note, completeness_note],
        _create_global_supply_comparison_table(period_df, time_view),
    )


@callback(
    Output('capacity-country-columns-dropdown', 'options'),
    Output('capacity-country-columns-dropdown', 'value'),
    Output('capacity-country-columns-section', 'style'),
    Output('capacity-other-country-section', 'style'),
    Input('production-ramp-run-store', 'data'),
    Input('capacity-new-capacity-checkbox', 'value'),
    Input('capacity-year-range-slider', 'value'),
    Input('capacity-breakdown-dropdown', 'value'),
    State('capacity-country-columns-dropdown', 'value')
)
def populate_capacity_country_columns(
    run_metadata,
    new_capacity_checkbox,
    year_range,
    breakdown,
    current_selection,
):
    """Populate the country-columns selector; hide country-only sections for plant/train."""
    is_country = (breakdown or 'country') == 'country'
    hidden = {'display': 'none'}
    visible = None  # let CSS class control the style
    if not is_country:
        return no_update, no_update, hidden, hidden

    run_id = (run_metadata or {}).get('display_run_id')
    if run_id is None:
        return [], [], visible, visible

    new_capacity_only = 'new_only' in (new_capacity_checkbox or [])
    start_year, end_year = year_range if year_range else [2025, 2040]

    raw_df = _prepare_volume_country_dataframe(
        run_id=run_id,
        selected_unit='mtpa',
        new_capacity_only=new_capacity_only,
        start_year=start_year,
        end_year=end_year,
        breakdown='country',
        refresh_token=(run_metadata or {}).get('refresh_token'),
    )
    available_countries = _get_available_volume_countries(raw_df)
    options = [{'label': country, 'value': country} for country in available_countries]

    if current_selection is None:
        selected_values = default_selected_countries(available_countries)
    else:
        selected_values = [
            country for country in current_selection if country in available_countries
        ]
        if current_selection and not selected_values:
            selected_values = default_selected_countries(available_countries)

    section_style = visible if is_country else hidden
    return options, selected_values, section_style, section_style


@callback(
    Output('capacity-volume-area-chart', 'figure'),
    Output('capacity-country-table-container', 'children'),
    Input('production-ramp-run-store', 'data'),
    Input('capacity-unit-dropdown', 'value'),
    Input('capacity-new-capacity-checkbox', 'value'),
    Input('capacity-breakdown-dropdown', 'value'),
    Input('capacity-country-columns-dropdown', 'value'),
    Input('capacity-other-country-mode', 'value'),
    Input('capacity-year-range-slider', 'value'),
    prevent_initial_call=True
)
def update_capacity_section(
    run_metadata,
    selected_unit,
    new_capacity_checkbox,
    breakdown,
    selected_countries,
    other_countries_mode,
    year_range,
):
    """Update the production page chart and table from one shared dataset."""
    if not selected_unit:
        raise PreventUpdate

    run_id = (run_metadata or {}).get('display_run_id')
    if run_id is None:
        empty_message = "No ramp run is available. Generate one from Capacity."
        return _create_empty_volume_figure(empty_message), _create_empty_table_state(empty_message)

    breakdown = breakdown or 'country'
    new_capacity_only = 'new_only' in (new_capacity_checkbox or [])
    start_year, end_year = year_range if year_range else [2025, 2040]

    raw_df = _prepare_volume_country_dataframe(
        run_id=run_id,
        selected_unit=selected_unit,
        new_capacity_only=new_capacity_only,
        start_year=start_year,
        end_year=end_year,
        breakdown=breakdown,
        refresh_token=(run_metadata or {}).get('refresh_token'),
    )

    if raw_df.empty:
        empty_message = "No volume data available for the current selection."
        return _create_empty_volume_figure(empty_message), _create_empty_table_state(empty_message)

    if breakdown != 'country':
        matrix_df = _build_direct_pivot_matrix(raw_df, selected_unit)
        if matrix_df.empty:
            empty_message = "No volume data available for the current selection."
            return _create_empty_volume_figure(empty_message), html.Div()
        return create_volume_country_area_chart(matrix_df, selected_unit), html.Div()

    available_countries = _get_available_volume_countries(raw_df)
    if selected_countries is None:
        resolved_countries = default_selected_countries(available_countries)
    else:
        resolved_countries = [
            country for country in selected_countries if country in available_countries
        ]

    if resolved_countries == [] and other_countries_mode == 'exclude':
        empty_message = "Select at least one country or switch to Rest of the World mode."
        return _create_empty_volume_figure(empty_message), _create_empty_table_state(empty_message)

    matrix_df = _build_volume_country_matrix(
        raw_df,
        resolved_countries,
        other_countries_mode,
        selected_unit,
    )

    if matrix_df.empty:
        empty_message = "No volume data available for the current selection."
        return _create_empty_volume_figure(empty_message), _create_empty_table_state(empty_message)

    return (
        create_volume_country_area_chart(matrix_df, selected_unit),
        _create_volume_country_table('capacity-country-table', matrix_df),
    )


@callback(
    Output('capacity-download-excel', 'data'),
    Input('capacity-export-excel-button', 'n_clicks'),
    State('production-ramp-run-store', 'data'),
    State('capacity-unit-dropdown', 'value'),
    State('capacity-new-capacity-checkbox', 'value'),
    State('capacity-breakdown-dropdown', 'value'),
    State('capacity-country-columns-dropdown', 'value'),
    State('capacity-other-country-mode', 'value'),
    State('capacity-year-range-slider', 'value'),
    prevent_initial_call=True
)
def export_capacity_to_excel(
    n_clicks,
    run_metadata,
    selected_unit,
    new_capacity_checkbox,
    breakdown,
    selected_countries,
    other_countries_mode,
    year_range,
):
    """Export the visible month-by-group matrix to Excel."""
    if n_clicks == 0:
        return None

    run_id = (run_metadata or {}).get('display_run_id')
    if run_id is None:
        return None

    breakdown = breakdown or 'country'
    new_capacity_only = 'new_only' in (new_capacity_checkbox or [])
    start_year, end_year = year_range if year_range else [2025, 2040]

    raw_df = _prepare_volume_country_dataframe(
        run_id=run_id,
        selected_unit=selected_unit,
        new_capacity_only=new_capacity_only,
        start_year=start_year,
        end_year=end_year,
        breakdown=breakdown,
        refresh_token=(run_metadata or {}).get('refresh_token'),
    )

    if raw_df.empty:
        return None

    if breakdown != 'country':
        export_df = _build_direct_pivot_matrix(raw_df, selected_unit)
    else:
        available_countries = _get_available_volume_countries(raw_df)
        if selected_countries is None:
            resolved_countries = default_selected_countries(available_countries)
        else:
            resolved_countries = [
                country for country in selected_countries if country in available_countries
            ]
        if resolved_countries == [] and other_countries_mode == 'exclude':
            return None
        export_df = _build_volume_country_matrix(
            raw_df,
            resolved_countries,
            other_countries_mode,
            selected_unit,
        )

    if export_df.empty:
        return None

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='Volume Data', index=False)

        worksheet = writer.sheets['Volume Data']
        for idx, col in enumerate(export_df.columns):
            max_length = max(
                export_df[col].astype(str).apply(len).max(),
                len(str(col))
            ) + 2

            col_letter = ''
            temp_idx = idx + 1
            while temp_idx > 0:
                temp_idx -= 1
                col_letter = chr(65 + (temp_idx % 26)) + col_letter
                temp_idx //= 26

            worksheet.column_dimensions[col_letter].width = min(max_length, 50)

    output.seek(0)

    unit_label = selected_unit.upper()
    new_cap_label = '_NewCapacity' if new_capacity_only else ''
    breakdown_label = {'country': 'Country', 'project': 'Plant', 'train': 'Train'}[breakdown]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = (
        f'LNG_Production_Run{run_id}_{breakdown_label}Matrix_'
        f'{unit_label}{new_cap_label}_{timestamp}.xlsx'
    )

    return dcc.send_bytes(output.getvalue(), filename)
