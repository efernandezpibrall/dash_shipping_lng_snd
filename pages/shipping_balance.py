from dash import html, dcc, callback, Output, Input, State
import dash_ag_grid as dag
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime as dt
import logging
import threading
from io import StringIO, BytesIO
from dash.exceptions import PreventUpdate
from concurrent.futures import ThreadPoolExecutor

from utils.dashboard_snapshot_cache import (
    build_source_key as _build_source_key,
    get_or_build_snapshot as _get_or_build_snapshot,
    resolve_snapshot as _resolve_snapshot,
    snapshot_is_shared as _snapshot_is_shared,
    was_global_refresh_triggered as _was_global_refresh_triggered,
    with_snapshot_slot as _with_snapshot_slot,
)
from utils.database import engine

from fundamentals.lng.shipping.shipping_balance_calculator import global_shipping_balance as calc_global_shipping_balance, kpler_analysis

logger = logging.getLogger(__name__)

SHIPPING_BALANCE_NAMESPACE = "shipping-balance-v1"

SHIPPING_BALANCE_SOURCE_STATE_QUERY = '''
SELECT
    (SELECT snapshot_timestamp_utc
     FROM at_lng.kpler_trade_snapshots
     WHERE run_kind = 'canonical' AND status = 'published'
     ORDER BY snapshot_date_utc DESC
     LIMIT 1) AS kpler_upload,
    (SELECT MAX(publication_date::timestamp)
     FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa) AS woodmac_publication,
    (SELECT MAX(upload_timestamp_utc) FROM at_lng.syy_newbuilds) AS syy_upload,
    (SELECT MAX(upload_timestamp_utc) FROM at_lng.kpler_vessels_info) AS vessel_upload,
    (SELECT md5(COALESCE(string_agg(
        concat_ws('|', COALESCE(country, ''), COALESCE(shipping_region, '')),
        '||' ORDER BY COALESCE(country, '')
    ), '')) FROM at_lng.mappings_country) AS mapping_hash
'''

SHIPPING_BALANCE_DATE_STATE_QUERY = '''
WITH latest_kpler AS (
    SELECT snapshot_timestamp_utc
    FROM at_lng.kpler_trade_snapshots
    WHERE run_kind = 'canonical' AND status = 'published'
    ORDER BY snapshot_date_utc DESC
    LIMIT 1
)
SELECT
    MIN(s.min_delivered_end) FILTER (WHERE s.facts_retained) AS min_date,
    MAX(s.max_delivered_end) FILTER (WHERE s.facts_retained) AS max_date,
    MAX(s.max_delivered_end) FILTER (
        WHERE s.snapshot_timestamp_utc = latest_kpler.snapshot_timestamp_utc
    ) AS hist_date_max
FROM at_lng.kpler_trade_snapshots s
CROSS JOIN latest_kpler
'''

_SHIPPING_DATE_STATE_CACHE = {}
_SHIPPING_DATE_STATE_LOCK = threading.Lock()


def _fetch_shipping_balance_source_state():
    return pd.read_sql(SHIPPING_BALANCE_SOURCE_STATE_QUERY, engine).iloc[0].to_dict()


def _get_shipping_balance_date_state(kpler_upload):
    cache_key = str(kpler_upload)
    with _SHIPPING_DATE_STATE_LOCK:
        cached = _SHIPPING_DATE_STATE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    date_state = pd.read_sql(SHIPPING_BALANCE_DATE_STATE_QUERY, engine).iloc[0].to_dict()
    with _SHIPPING_DATE_STATE_LOCK:
        _SHIPPING_DATE_STATE_CACHE.clear()
        _SHIPPING_DATE_STATE_CACHE[cache_key] = dict(date_state)
    return date_state


def _resolve_shipping_store(value):
    return _resolve_snapshot(
        value,
        engine,
        expected_namespace=SHIPPING_BALANCE_NAMESPACE,
    )


def _woodmac_validation_period_filter(aggregation_level, selected_year, selected_period):
    selected_year = int(selected_year)
    selected_period = int(selected_period)

    if aggregation_level == 'monthly':
        return (
            "country_name, start_date::DATE, publication_date, source",
            "start_date::DATE",
            (
                "EXTRACT(YEAR FROM start_date) = %(selected_year)s "
                "AND EXTRACT(MONTH FROM start_date) = %(selected_period)s"
            ),
            {'selected_year': selected_year, 'selected_period': selected_period},
        )

    if aggregation_level == 'quarterly':
        quarter_start_month = (selected_period - 1) * 3 + 1
        return (
            "country_name, DATE_TRUNC('quarter', start_date::DATE), publication_date, source",
            "DATE_TRUNC('quarter', start_date::DATE)",
            "start_date = %(period_start)s",
            {'period_start': dt.date(selected_year, quarter_start_month, 1)},
        )

    if aggregation_level == 'seasonal':
        season_date_expr = (
            "CASE WHEN EXTRACT(MONTH FROM start_date::DATE) BETWEEN 1 AND 6 "
            "THEN DATE_TRUNC('year', start_date::DATE) "
            "ELSE DATE_TRUNC('year', start_date::DATE) + INTERVAL '6 months' END"
        )
        return (
            f"country_name, {season_date_expr}, publication_date, source",
            season_date_expr,
            "start_date = %(period_start)s",
            {'period_start': dt.date(selected_year, selected_period, 1)},
        )

    return (
        "country_name, DATE_TRUNC('year', start_date::DATE), publication_date, source",
        "DATE_TRUNC('year', start_date::DATE)",
        "start_date = %(period_start)s",
        {'period_start': dt.date(selected_year, 1, 1)},
    )


def _fetch_woodmac_flow_validation_total(direction, measured_at, aggregation_level, selected_year, selected_period):
    group_by_clause, date_column, date_filter, date_params = _woodmac_validation_period_filter(
        aggregation_level,
        selected_year,
        selected_period,
    )
    params = {
        **date_params,
        'direction': direction,
        'measured_at': measured_at,
    }
    wm_query = f'''
    WITH latest_short_term AS (
        SELECT
            country_name,
            {date_column}::DATE as start_date,
            SUM(metric_value) / 12 * 2222*1000 AS value,
            publication_date,
            'Short Term' as source
        FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
        WHERE market_outlook = (
            SELECT market_outlook
            FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
            WHERE release_type = 'Short Term Outlook'
            GROUP BY market_outlook
            ORDER BY TO_DATE(
                (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{{4}})'))[1]
                || ' ' ||
                (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{{4}})'))[2],
                'Month YYYY'
            ) DESC, MAX(publication_date) DESC
            LIMIT 1
        )
        AND release_type = 'Short Term Outlook'
        AND direction = %(direction)s
        AND measured_at = %(measured_at)s
        AND metric_name = 'Flow'
        AND start_date::DATE < '2036-01-01'
        GROUP BY {group_by_clause}
        HAVING SUM(metric_value) > 0
    ),
    short_term_max_date AS (
        SELECT MAX(start_date::DATE) as max_date
        FROM latest_short_term
    ),
    latest_long_term_raw AS (
        SELECT
            country_name,
            {date_column}::DATE as start_date,
            SUM(metric_value) / 12 * 2222*1000 AS value,
            publication_date,
            'Long Term' as source
        FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
        WHERE market_outlook = (
            SELECT market_outlook
            FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
            WHERE release_type = 'Long Term Outlook'
            GROUP BY market_outlook
            ORDER BY TO_DATE(
                (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{{4}})'))[1]
                || ' ' ||
                (regexp_match(market_outlook, '(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{{4}})'))[2],
                'Month YYYY'
            ) DESC, MAX(publication_date) DESC
            LIMIT 1
        )
        AND release_type = 'Long Term Outlook'
        AND direction = %(direction)s
        AND measured_at = %(measured_at)s
        AND metric_name = 'Flow'
        AND start_date::DATE < '2036-01-01'
        GROUP BY {group_by_clause}
        HAVING SUM(metric_value) > 0
    ),
    latest_long_term AS (
        SELECT *
        FROM latest_long_term_raw
        WHERE start_date > (SELECT max_date FROM short_term_max_date)
    ),
    combined AS (
        SELECT * FROM latest_short_term
        UNION ALL
        SELECT * FROM latest_long_term
    ),
    filtered AS (
        SELECT * FROM combined WHERE {date_filter}
    )
    SELECT COALESCE(SUM(value) / 1000000, 0) as woodmac_total,
           STRING_AGG(DISTINCT source, ' + ') as sources
    FROM filtered
    '''

    wm_result = pd.read_sql(wm_query, engine, params=params)
    woodmac_total = wm_result['woodmac_total'].iloc[0]
    woodmac_sources = wm_result['sources'].iloc[0] if wm_result['sources'].iloc[0] else 'N/A'
    return woodmac_total, woodmac_sources


AG_GRID_THEME = "ag-theme-alpine"

SHIPPING_BALANCE_AG_GRID_DEFAULT_COL_DEF = {
    "sortable": True,
    "filter": False,
    "resizable": True,
    "suppressHeaderMenuButton": True,
    "suppressHeaderFilterButton": True,
    "wrapHeaderText": True,
    "autoHeaderHeight": True,
    "headerClass": "fleet-metrics-grid-header",
    "cellClass": "fleet-metrics-grid-cell",
}

SHIPPING_BALANCE_AG_GRID_OPTIONS = {
    "animateRows": False,
    "pagination": True,
    "paginationPageSizeSelector": [10, 15, 25, 50],
    "suppressRowHoverHighlight": False,
    "suppressCellFocus": True,
    "enableCellTextSelection": True,
    "ensureDomOrder": True,
    "headerHeight": 32,
    "rowHeight": 30,
    "groupHeaderHeight": 28,
    "tooltipShowDelay": 250,
    "rowClassRules": {
        "fleet-metrics-global-row": "params.data && params.data.__row_type === 'total'",
    },
}

SHIPPING_BALANCE_PAGE_STYLE = {
    "backgroundColor": "#f8fafc",
    "paddingBottom": "24px",
}

SHIPPING_BALANCE_SECTION_STYLE = {
    "background": "#ffffff",
    "border": "1px solid #e5e7eb",
    "borderRadius": "8px",
    "padding": "14px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.05)",
}

SHIPPING_BALANCE_PANEL_STYLE = {
    "minWidth": 0,
    "background": "#ffffff",
    "border": "1px solid #e5e7eb",
    "borderRadius": "8px",
    "padding": "12px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.04)",
}

SHIPPING_BALANCE_TWO_COLUMN_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(min(100%, 520px), 1fr))",
    "gap": "12px",
    "alignItems": "start",
}

SHIPPING_BALANCE_CONTROL_ROW_STYLE = {
    "display": "flex",
    "gap": "10px",
    "alignItems": "flex-end",
    "flexWrap": "wrap",
    "marginBottom": "12px",
}

SHIPPING_BALANCE_EXPORT_BUTTON_STYLE = {
    "marginLeft": "12px",
    "padding": "6px 12px",
    "backgroundColor": "#1B4F72",
    "color": "white",
    "border": "none",
    "borderRadius": "6px",
    "cursor": "pointer",
    "fontWeight": "700",
    "fontSize": "12px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.12)",
}

SHIPPING_BALANCE_VALIDATION_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "flexWrap": "wrap",
    "gap": "8px",
    "padding": "8px 10px",
    "marginBottom": "10px",
    "backgroundColor": "#f8fafc",
    "border": "1px solid #e2e8f0",
    "borderRadius": "8px",
    "fontSize": "12px",
    "color": "#475569",
}

SHIPPING_BALANCE_VALIDATION_LABEL_STYLE = {
    "fontSize": "12px",
    "fontWeight": "700",
    "color": "#0f172a",
    "textTransform": "uppercase",
}

SHIPPING_BALANCE_VALIDATION_VALUE_STYLE = {
    "fontSize": "12px",
    "fontWeight": "700",
    "color": "#1B4F72",
}

SHIPPING_BALANCE_VALIDATION_SEPARATOR_STYLE = {
    "color": "#cbd5e1",
}

SHIPPING_BALANCE_CHART_COLORS = {
    "active_ships": "#0b3558",
    "demand": "#37a39c",
    "supply": "#a85534",
    "capacity": "#ff5a1f",
    "positive": "rgba(47, 111, 78, 0.34)",
    "positive_line": "rgba(47, 111, 78, 0.72)",
    "negative": "rgba(194, 65, 12, 0.34)",
    "negative_line": "rgba(194, 65, 12, 0.72)",
    "neutral": "rgba(100, 116, 139, 0.22)",
    "neutral_line": "rgba(100, 116, 139, 0.52)",
    "current_period": "#475569",
}

SHIPPING_BALANCE_ROUTE_COLORS = [
    "#0b3558",
    "#1f5f8b",
    "#2f91d0",
    "#7bc6ef",
    "#37a39c",
    "#2f6f4e",
    "#3f9b3d",
    "#95bf6f",
    "#a85534",
    "#6d5dfc",
    "#c2410c",
    "#64748b",
    "#14b8a6",
    "#7c3aed",
    "#ca8a04",
]

SHIPPING_BALANCE_CHART_FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
SHIPPING_BALANCE_GRID_COLOR = "rgba(148, 163, 184, 0.22)"
SHIPPING_BALANCE_SUBTLE_GRID_COLOR = "rgba(148, 163, 184, 0.14)"


def _numeric_value_formatter(precision=2):
    return {
        "function": (
            "params.value !== null && params.value !== undefined && params.value !== '' "
            f"? d3.format(',.{precision}f')(Number(params.value)) : ''"
        )
    }


def _style_shipping_balance_figure(
    fig,
    *,
    height=None,
    title=None,
    legend_orientation="h",
    legend_x=0,
    legend_y=1.02,
    legend_xanchor="left",
    legend_yanchor="bottom",
    margin=None,
    hovermode="x unified",
    legend_title=None,
):
    margin = margin or {"l": 52, "r": 30, "t": 50 if title else 28, "b": 58}
    title_config = None
    if title:
        title_config = {
            "text": f"<b>{title}</b>",
            "x": 0,
            "xanchor": "left",
            "y": 0.98,
            "font": {"size": 15, "color": "#0f172a", "family": SHIPPING_BALANCE_CHART_FONT},
        }

    fig.update_layout(
        title=title_config,
        template="plotly_white",
        colorway=SHIPPING_BALANCE_ROUTE_COLORS,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font={"family": SHIPPING_BALANCE_CHART_FONT, "size": 11, "color": "#334155"},
        margin=margin,
        height=height,
        hovermode=hovermode,
        hoverdistance=70,
        spikedistance=70,
        hoverlabel={
            "bgcolor": "rgba(255, 255, 255, 0.96)",
            "bordercolor": "rgba(148, 163, 184, 0.55)",
            "font": {"size": 11, "color": "#0f172a"},
            "align": "left",
        },
        legend={
            "orientation": legend_orientation,
            "x": legend_x,
            "y": legend_y,
            "xanchor": legend_xanchor,
            "yanchor": legend_yanchor,
            "bgcolor": "rgba(255,255,255,0)",
            "bordercolor": "rgba(255,255,255,0)",
            "font": {"size": 10, "color": "#334155"},
            "title": {"text": legend_title or "", "font": {"size": 10, "color": "#64748b"}},
            "itemsizing": "constant",
            "itemwidth": 30,
            "groupclick": "togglegroup",
        },
        transition={"duration": 180, "easing": "cubic-in-out"},
        autosize=True,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=SHIPPING_BALANCE_SUBTLE_GRID_COLOR,
        linecolor="rgba(148, 163, 184, 0.42)",
        linewidth=1,
        mirror=False,
        showline=True,
        showspikes=True,
        spikecolor="rgba(71, 85, 105, 0.24)",
        spikethickness=1,
        spikedash="dot",
        tickfont={"size": 10, "color": "#64748b"},
        title_font={"size": 11, "color": "#334155"},
        fixedrange=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=SHIPPING_BALANCE_GRID_COLOR,
        zeroline=True,
        zerolinecolor="rgba(148, 163, 184, 0.45)",
        linecolor="rgba(148, 163, 184, 0.42)",
        linewidth=1,
        mirror=False,
        showline=True,
        tickfont={"size": 10, "color": "#64748b"},
        title_font={"size": 11, "color": "#334155"},
        fixedrange=True,
    )
    if getattr(fig.layout, "yaxis2", None):
        fig.update_layout(
            yaxis2={
                **fig.layout.yaxis2.to_plotly_json(),
                "showgrid": False,
                "zeroline": False,
                "linecolor": "rgba(148, 163, 184, 0.42)",
                "tickfont": {"size": 10, "color": "#64748b"},
                "title": {
                    **(fig.layout.yaxis2.title.to_plotly_json() if fig.layout.yaxis2.title else {}),
                    "font": {"size": 11, "color": "#334155"},
                },
                "fixedrange": True,
            }
        )
    return fig


def _empty_shipping_balance_figure(message, height=400):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        align="center",
        font={"size": 13, "color": "#64748b", "family": SHIPPING_BALANCE_CHART_FONT},
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 34, "r": 28, "t": 32, "b": 34},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _date_axis_settings(aggregation_level):
    if aggregation_level == "quarterly":
        return "%Y Q%q", "M3", "%{x|%Y Q%q}"
    if aggregation_level == "seasonal":
        return "%Y %b", "M6", "%{x|%Y %b}"
    if aggregation_level == "yearly":
        return "%Y", "M12", "%{x|%Y}"
    return "%b %Y", "M3", "%{x|%b %Y}"


def _net_bar_marker_styles(values):
    colors = []
    line_colors = []
    for value in pd.to_numeric(pd.Series(values), errors="coerce").fillna(0):
        if value > 0:
            colors.append(SHIPPING_BALANCE_CHART_COLORS["positive"])
            line_colors.append(SHIPPING_BALANCE_CHART_COLORS["positive_line"])
        elif value < 0:
            colors.append(SHIPPING_BALANCE_CHART_COLORS["negative"])
            line_colors.append(SHIPPING_BALANCE_CHART_COLORS["negative_line"])
        else:
            colors.append(SHIPPING_BALANCE_CHART_COLORS["neutral"])
            line_colors.append(SHIPPING_BALANCE_CHART_COLORS["neutral_line"])
    return colors, line_colors


def _add_current_period_marker(fig, hist_date_max_str):
    if not hist_date_max_str:
        return fig
    hist_date_max = pd.to_datetime(hist_date_max_str, errors="coerce")
    if pd.isna(hist_date_max):
        return fig
    fig.add_shape(
        type="line",
        x0=hist_date_max,
        x1=hist_date_max,
        y0=0,
        y1=1,
        yref="paper",
        line={"color": SHIPPING_BALANCE_CHART_COLORS["current_period"], "width": 1.4, "dash": "dot"},
        layer="above",
    )
    fig.add_annotation(
        x=hist_date_max,
        y=1.01,
        yref="paper",
        text="Current period",
        showarrow=False,
        yanchor="bottom",
        bgcolor="rgba(255, 255, 255, 0.92)",
        bordercolor="rgba(148, 163, 184, 0.45)",
        borderwidth=1,
        borderpad=3,
        font={"color": "#475569", "size": 10, "family": SHIPPING_BALANCE_CHART_FONT},
    )
    return fig


def _build_balance_dual_axis_chart(
    df,
    *,
    scenario_name,
    scenario_column,
    scenario_color,
    net_name,
    aggregation_level,
    hist_date_max_str,
):
    if df.empty or not {"date", "total_active_ships", scenario_column, "net"}.issubset(df.columns):
        return _empty_shipping_balance_figure("No shipping balance data loaded")

    display_df = df.sort_values("date").copy()
    tick_format, dtick, period_hover = _date_axis_settings(aggregation_level)
    bar_colors, bar_line_colors = _net_bar_marker_styles(display_df["net"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=display_df["date"],
            y=display_df["net"],
            name=net_name,
            marker={
                "color": bar_colors,
                "line": {"color": bar_line_colors, "width": 0.9},
            },
            opacity=0.94,
            hovertemplate=f"<b>{net_name}</b><br>{period_hover}<br>%{{y:,.0f}} ships<extra></extra>",
            legendrank=3,
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=display_df["date"],
            y=display_df["total_active_ships"],
            name="Total Active Ships",
            mode="lines",
            line={"color": SHIPPING_BALANCE_CHART_COLORS["active_ships"], "width": 2.7},
            hovertemplate=f"<b>Total Active Ships</b><br>{period_hover}<br>%{{y:,.0f}} ships<extra></extra>",
            legendrank=1,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=display_df["date"],
            y=display_df[scenario_column],
            name=scenario_name,
            mode="lines",
            line={"color": scenario_color, "width": 2.7},
            hovertemplate=f"<b>{scenario_name}</b><br>{period_hover}<br>%{{y:,.0f}} ships<extra></extra>",
            legendrank=2,
        ),
        secondary_y=False,
    )
    fig.update_layout(
        barmode="relative",
        bargap=0.24,
        legend_traceorder="normal",
    )
    fig.update_xaxes(
        title=None,
        tickformat=tick_format,
        dtick=dtick,
        tickmode="auto",
        nticks=11,
    )
    fig.update_yaxes(
        title={"text": "Ships", "font": {"size": 11, "color": "#334155"}},
        rangemode="tozero",
        tickformat=",.0f",
        secondary_y=False,
    )
    fig.update_yaxes(
        title={"text": "Net ships", "font": {"size": 11, "color": "#334155"}},
        tickformat=",.0f",
        secondary_y=True,
    )
    _add_current_period_marker(fig, hist_date_max_str)
    return _style_shipping_balance_figure(
        fig,
        height=400,
        legend_orientation="h",
        legend_x=0,
        legend_y=-0.12,
        legend_xanchor="left",
        legend_yanchor="top",
        margin={"l": 50, "r": 36, "t": 24, "b": 78},
    )


def _build_fleet_statistics_figure(df, *, aggregation_level, hist_date_max_str):
    required_columns = {"date", "total_active_ships", "average_size_cubic_meters"}
    if df.empty or not required_columns.issubset(df.columns):
        return _empty_shipping_balance_figure("No fleet statistics data loaded")

    display_df = df[["date", "total_active_ships", "average_size_cubic_meters"]].sort_values("date").copy()
    tick_format, dtick, period_hover = _date_axis_settings(aggregation_level)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=display_df["date"],
            y=display_df["total_active_ships"],
            name="Total Active Ships",
            marker={
                "color": "rgba(31, 95, 139, 0.34)",
                "line": {"color": "rgba(31, 95, 139, 0.72)", "width": 0.8},
            },
            hovertemplate=f"<b>Total Active Ships</b><br>{period_hover}<br>%{{y:,.0f}} ships<extra></extra>",
            legendrank=1,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=display_df["date"],
            y=display_df["average_size_cubic_meters"],
            name="Avg Capacity (m³)",
            mode="lines",
            line={"color": SHIPPING_BALANCE_CHART_COLORS["capacity"], "width": 2.7},
            hovertemplate=f"<b>Avg Capacity</b><br>{period_hover}<br>%{{y:,.0f}} m³<extra></extra>",
            legendrank=2,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        barmode="relative",
        bargap=0.24,
        legend_traceorder="normal",
    )
    fig.update_xaxes(
        title=None,
        tickformat=tick_format,
        dtick=dtick,
        tickmode="auto",
        nticks=11,
    )
    fig.update_yaxes(
        title={"text": "Active ships", "font": {"size": 11, "color": "#334155"}},
        rangemode="tozero",
        tickformat=",.0f",
        secondary_y=False,
    )
    fig.update_yaxes(
        title={"text": "Avg capacity (m³)", "font": {"size": 11, "color": "#334155"}},
        tickformat=",.0f",
        secondary_y=True,
    )
    _add_current_period_marker(fig, hist_date_max_str)
    return _style_shipping_balance_figure(
        fig,
        height=400,
        legend_orientation="h",
        legend_x=0,
        legend_y=-0.12,
        legend_xanchor="left",
        legend_yanchor="top",
        margin={"l": 50, "r": 38, "t": 24, "b": 78},
    )


def _build_route_days_chart(
    regional_data,
    origin_filter,
    dest_filter,
    dropdown_options,
    *,
    metric_col,
    title,
    y_label,
):
    if regional_data is None:
        return _empty_shipping_balance_figure("No regional route data loaded", height=450)

    df = pd.read_json(StringIO(regional_data), orient="split")
    required_columns = {"date", "origin_shipping_region", "destination_shipping_region", metric_col}
    if df.empty or not required_columns.issubset(df.columns):
        return _empty_shipping_balance_figure("No regional route data loaded", height=450)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", metric_col]).copy()
    if origin_filter:
        df = df[df["origin_shipping_region"].isin(origin_filter)]
    if dest_filter:
        df = df[df["destination_shipping_region"].isin(dest_filter)]
    if df.empty:
        return _empty_shipping_balance_figure("No route data for the selected filters", height=450)

    df["route"] = df["origin_shipping_region"] + " → " + df["destination_shipping_region"]
    ranking_column = "value" if "value" in df.columns else metric_col
    route_order = (
        df.groupby("route")[ranking_column]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index
        .tolist()
    )
    df = df[df["route"].isin(route_order)].sort_values("date")

    aggregation_level = (dropdown_options or {}).get("aggregation_level", "monthly")
    tick_format, dtick, period_hover = _date_axis_settings(aggregation_level)
    fig = go.Figure()

    for idx, route in enumerate(route_order):
        route_df = df[df["route"] == route].sort_values("date")
        if route_df.empty:
            continue
        is_primary = idx < 5
        fig.add_trace(
            go.Scatter(
                x=route_df["date"],
                y=route_df[metric_col],
                name=route,
                mode="lines",
                line={
                    "color": SHIPPING_BALANCE_ROUTE_COLORS[idx % len(SHIPPING_BALANCE_ROUTE_COLORS)],
                    "width": 2.35 if is_primary else 1.55,
                },
                opacity=0.98 if is_primary else 0.68,
                hovertemplate=f"<b>{route}</b><br>{period_hover}<br>%{{y:.1f}} days<extra></extra>",
            )
        )

    if not fig.data:
        return _empty_shipping_balance_figure("No route data for the selected filters", height=450)

    fig.update_xaxes(
        title=None,
        tickformat=tick_format,
        dtick=dtick,
        tickmode="auto",
        nticks=9,
    )
    fig.update_yaxes(
        title={"text": y_label, "font": {"size": 11, "color": "#334155"}},
        rangemode="tozero",
        tickformat=",.1f",
    )
    _add_current_period_marker(fig, (dropdown_options or {}).get("hist_date_max"))
    return _style_shipping_balance_figure(
        fig,
        height=450,
        title=f"{title} (Top 15 Routes)",
        legend_orientation="v",
        legend_x=1.01,
        legend_y=1,
        legend_xanchor="left",
        legend_yanchor="top",
        margin={"l": 52, "r": 224, "t": 54, "b": 44},
        legend_title="Route",
    )


def _is_numeric_column(df, column):
    return pd.api.types.is_numeric_dtype(df[column])


def _grid_height(row_count, page_size, *, minimum=190, maximum=560):
    visible_rows = min(max(row_count, 1), page_size)
    return min(maximum, max(minimum, 72 + visible_rows * 30))


def _build_ag_grid_column_defs(
    df,
    *,
    display_names=None,
    text_columns=None,
    pinned_columns=None,
    numeric_precision=None,
):
    display_names = display_names or {}
    text_columns = set(text_columns or [])
    pinned_columns = set(pinned_columns or [])
    numeric_precision = numeric_precision or {}
    column_defs = []

    for column in df.columns:
        if column == "__row_type":
            continue

        column_def = {
            "field": column,
            "headerName": display_names.get(column, column),
            "minWidth": 92,
        }

        if column in pinned_columns:
            column_def.update(
                {
                    "pinned": "left",
                    "minWidth": 130,
                    "cellClass": "fleet-metrics-left-cell fleet-metrics-strong-cell",
                }
            )
        elif column not in text_columns and _is_numeric_column(df, column):
            column_def.update(
                {
                    "type": "numericColumn",
                    "cellClass": "fleet-metrics-number-cell",
                    "valueFormatter": _numeric_value_formatter(numeric_precision.get(column, 2)),
                    "width": 105,
                    "minWidth": 82,
                }
            )
        else:
            column_def.update(
                {
                    "cellClass": "fleet-metrics-left-cell",
                    "minWidth": max(120, min(260, len(str(display_names.get(column, column))) * 9 + 64)),
                }
            )

        column_defs.append(column_def)

    return column_defs


def create_ag_grid_table(
    data,
    *,
    id_value,
    display_names=None,
    text_columns=None,
    pinned_columns=None,
    numeric_precision=None,
    page_size=25,
    height=None,
    extra_class="",
    column_size="responsiveSizeToFit",
):
    """Create a Fleet Metrics-style AgGrid table from the provided data."""
    if data is None or data.empty:
        return html.Div("No data available for the selected filters.", className="balance-empty-state")

    grid_df = data.copy()
    first_column = next((column for column in grid_df.columns if column != "__row_type"), None)
    if first_column is not None and "__row_type" not in grid_df.columns:
        grid_df["__row_type"] = np.where(grid_df[first_column].astype(str).str.lower() == "total", "total", "")
    grid_df = grid_df.where(pd.notna(grid_df), None)

    grid_options = {
        **SHIPPING_BALANCE_AG_GRID_OPTIONS,
        "pagination": True,
        "paginationPageSize": page_size,
    }

    grid_kwargs = {
        "id": id_value,
        "rowData": grid_df.to_dict("records"),
        "columnDefs": _build_ag_grid_column_defs(
            grid_df,
            display_names=display_names,
            text_columns=text_columns,
            pinned_columns=pinned_columns,
            numeric_precision=numeric_precision,
        ),
        "defaultColDef": SHIPPING_BALANCE_AG_GRID_DEFAULT_COL_DEF,
        "dashGridOptions": grid_options,
        "className": f"{AG_GRID_THEME} fleet-metrics-grid shipping-balance-grid {extra_class}".strip(),
        "style": {"width": "100%", "height": f"{height or _grid_height(len(grid_df), page_size)}px"},
        "dangerously_allow_code": True,
    }
    if column_size:
        grid_kwargs["columnSize"] = column_size

    return dag.AgGrid(**grid_kwargs)


def prepare_table_data(df, metric, selected_regions=None, selected_year=None, selected_statuses=None,
                       is_intracountry=False):
    """
    Prepare data for tables showing the metric values by region pair/country and vessel type.
    Args:
        df: DataFrame with trade data
        metric: The column to aggregate ('count_trades' or 'sum_ton_miles')
        selected_regions: List of origin shipping regions to filter by (only for non-intracountry)
        selected_year: Year to filter by (defaults to latest)
        selected_statuses: List of status values to filter by
        is_intracountry: Whether this is for intracountry data
    Returns:
        DataFrame formatted for display in an AgGrid table
    """
    # Filter the data for years 2019 and later
    filtered_df = df[df['year'] >= 2019]

    # Get the latest year if not specified
    if not selected_year or selected_year == "All Years":
        selected_year = filtered_df['year'].max()
    else:
        selected_year = int(selected_year)

    # Filter by year
    filtered_df = filtered_df[filtered_df['year'] == selected_year]

    # Apply filters specific to trade region or intracountry
    if is_intracountry:
        # Check if origin_country_name exists
        if 'origin_country_name' not in filtered_df.columns:
            # Return empty DataFrame if column doesn't exist
            return pd.DataFrame({'Error': ['origin_country_name column not found in data']})
        index_field = 'origin_country_name'
    else:
        # Check if required columns exist
        if 'origin_shipping_region' not in filtered_df.columns or 'destination_shipping_region' not in filtered_df.columns:
            # Return empty DataFrame if columns don't exist
            return pd.DataFrame({'Error': ['Required shipping region columns not found in data']})
        # Create a new column combining origin and destination regions
        filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
            'destination_shipping_region']
        index_field = 'region_pair'

        # Apply region filter if selected (only for non-intracountry)
        if selected_regions and 'All Regions' not in selected_regions:
            filtered_df = filtered_df[filtered_df['origin_shipping_region'].isin(selected_regions)]

    # Apply status filter if selected
    if selected_statuses and 'All Statuses' not in selected_statuses:
        filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]

    # Check if data is empty after filtering
    if filtered_df.empty:
        return pd.DataFrame({index_field: ['No data'], 'Total': [0]})

    # Determine aggregation method based on metric type
    if metric.startswith('median_'):
        agg_method = 'median'
    elif metric.startswith('mean_'):
        agg_method = 'mean'
    elif metric in ['count_trades', 'sum_ton_miles']:
        agg_method = 'sum'
    else:
        # Default to mean for other metrics that might be averages
        agg_method = 'mean'

    # Aggregate the data for the selected route grouping.
    agg_data = filtered_df.groupby([index_field])[metric].agg(agg_method).reset_index()

    # Create simplified table without vessel type pivoting
    pivot_table = agg_data.copy()

    # Rename metric column to Total for consistency
    pivot_table = pivot_table.rename(columns={metric: 'Total'})

    # Format numeric values based on metric type
    if metric == 'count_trades':
        # Integer formatting for count data
        pivot_table['Total'] = pivot_table['Total'].astype(int)
    elif agg_method in ['median', 'mean']:
        # Round to 2 decimal places for median/mean metrics
        pivot_table['Total'] = pivot_table['Total'].round(2)

    # Add a Total row
    total_row = {index_field: 'Total'}
    if agg_method in ['median', 'mean']:
        # For median/mean metrics, calculate overall median/mean across regions
        value = pivot_table[pivot_table[index_field] != 'Total']['Total'].median() if agg_method == 'median' else pivot_table[pivot_table[index_field] != 'Total']['Total'].mean()
        total_row['Total'] = round(value, 2) if not pd.isna(value) else 0
    else:
        # For sum metrics, sum across regions
        total_row['Total'] = int(pivot_table['Total'].sum()) if metric == 'count_trades' else pivot_table['Total'].sum()
    pivot_table = pd.concat([pivot_table, pd.DataFrame([total_row])], ignore_index=True)

    # Sort by Total value in descending order (except Total row)
    non_total_rows = pivot_table[pivot_table[index_field] != 'Total'].copy()
    total_row = pivot_table[pivot_table[index_field] == 'Total'].copy()
    non_total_rows = non_total_rows.sort_values('Total', ascending=False)

    # Recombine sorted rows with total row at bottom
    pivot_table = pd.concat([non_total_rows, total_row], ignore_index=True)

    return pivot_table


def create_stacked_bar_chart(df, metric, title_suffix, selected_statuses=None, is_intracountry=False):
    """
    Create a Plotly visualization showing data by year and shipping regions/countries.
    Args:
        df: DataFrame with trade data
        metric: The column name to sum and visualize
        title_suffix: Text to use in the title describing the metric
        selected_statuses: List of status values to filter by
        is_intracountry: Whether this is for intracountry data
    Returns:
        A Plotly figure object
    """
    # Filter the data for years 2019 and later
    filtered_df = df[df['year'] >= 2019].copy()

    # Apply status filter if selected
    if selected_statuses and 'All Statuses' not in selected_statuses:
        filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]

    if filtered_df.empty or metric not in filtered_df.columns:
        return _empty_shipping_balance_figure("No intracountry data available", height=600)

    # Set grouping field based on data type
    if is_intracountry:
        # Check if origin_country_name exists
        if 'origin_country_name' not in filtered_df.columns:
            # Return empty figure if column doesn't exist
            return _empty_shipping_balance_figure("No intracountry data available", height=600)
        group_field = 'origin_country_name'
        chart_title = f'Intracountry {title_suffix} by Year and Origin Country (2019+)'
        legend_title = 'Origin Country'
    else:
        # Check if required columns exist
        if 'origin_shipping_region' not in filtered_df.columns or 'destination_shipping_region' not in filtered_df.columns:
            # Return empty figure if columns don't exist
            return _empty_shipping_balance_figure("Required shipping region data not available", height=600)
        # Create a new column combining origin and destination regions
        filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
            'destination_shipping_region']
        group_field = 'region_pair'
        chart_title = f'{title_suffix} by Year and Shipping Regions (2019+)'
        legend_title = 'Shipping Regions (Origin → Destination)'

    filtered_df[metric] = pd.to_numeric(filtered_df[metric], errors="coerce").fillna(0)

    # Aggregate the filtered data for the selected route grouping.
    stacked_data = filtered_df.groupby(['year', group_field])[metric].sum().reset_index()
    if stacked_data.empty:
        return _empty_shipping_balance_figure("No data for the selected filters", height=600)

    years = sorted(stacked_data['year'].unique())
    year_labels = [str(year) for year in years]
    group_values = (
        stacked_data.groupby(group_field)[metric]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    stacked_matrix = (
        stacked_data.pivot_table(index='year', columns=group_field, values=metric, aggfunc='sum')
        .reindex(years)
        .fillna(0)
    )

    # Create figure
    fig = go.Figure()

    for i, group_value in enumerate(group_values):
        values = stacked_matrix[group_value].tolist()
        fig.add_trace(
            go.Bar(
                x=year_labels,
                y=values,
                name=group_value,
                marker={
                    "color": SHIPPING_BALANCE_ROUTE_COLORS[i % len(SHIPPING_BALANCE_ROUTE_COLORS)],
                    "line": {"color": "rgba(255, 255, 255, 0.72)", "width": 0.6},
                },
                hovertemplate=f"<b>{group_value}</b><br>Year %{{x}}<br>{title_suffix}: %{{y:,.0f}}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode='stack',
        bargap=0.28,
    )
    fig.update_xaxes(title=None, type="category")
    fig.update_yaxes(
        title={"text": title_suffix, "font": {"size": 11, "color": "#334155"}},
        rangemode="tozero",
        tickformat=",.0f",
    )

    return _style_shipping_balance_figure(
        fig,
        height=600,
        title=chart_title,
        legend_orientation="h",
        legend_x=0,
        legend_y=-0.18,
        legend_xanchor="left",
        legend_yanchor="top",
        margin={"l": 56, "r": 36, "t": 52, "b": 118},
        legend_title=legend_title,
    )


# Dashboard layout
layout = html.Div([
    # Store components for caching data (using session storage to avoid stale data issues)
    dcc.Store(id='shipping-balance-data-store', storage_type='session'),
    dcc.Store(id='shipping-balance-supply-data-store', storage_type='session'),
    dcc.Store(id='shipping-balance-regional-data-store', storage_type='session'),
    dcc.Store(id='shipping-balance-supply-regional-data-store', storage_type='session'),
    dcc.Store(id='dropdown-options-store', storage_type='session'),
    dcc.Store(id='intracountry-data-store', storage_type='session'),
    dcc.Download(id='download-demand-metrics-excel'),
    dcc.Download(id='download-supply-metrics-excel'),
    dcc.Download(id='download-intracountry-count-excel'),
    dcc.Download(id='download-intracountry-tonmiles-excel'),
    dcc.Download(id='download-fleet-stats-excel'),

    # Global Shipping Balance Overview Section - Sticky Professional Header
    html.Div(
        [
            html.Div(
                [
                    html.Div("Aggregation", className='filter-group-header'),
                    dcc.Dropdown(
                        id='aggregation-dropdown',
                        options=[
                            {'label': 'Year+Quarter', 'value': 'quarterly'},
                            {'label': 'Year+Month', 'value': 'monthly'},
                            {'label': 'Year+Season', 'value': 'seasonal'},
                            {'label': 'Year', 'value': 'yearly'}
                        ],
                        value='quarterly',
                        clearable=False,
                        className='filter-dropdown',
                        style={'width': '100%'},
                    ),
                ],
                className='filter-group',
                style={'flex': '1 1 180px', 'maxWidth': '240px'},
            ),
            html.Div(
                [
                    html.Div("Scenario Window End", className='filter-group-header'),
                    dcc.Input(
                        id='window-end-date-input',
                        type='text',
                        placeholder='YYYY-MM-DD',
                        value=(pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                        className='filter-input',
                        style={'width': '100%', 'height': '36px', 'fontSize': '13px', 'padding': '6px 8px'},
                    ),
                ],
                className='filter-group',
                style={'flex': '1 1 180px', 'maxWidth': '220px'},
            ),
            html.Div(
                [
                    html.Div("Vessel Age", className='filter-group-header'),
                    dcc.Input(
                        id='vessel-age-input',
                        type='number',
                        value=20,
                        min=1,
                        max=50,
                        step=1,
                        className='filter-input',
                        style={'width': '100%', 'height': '36px', 'fontSize': '13px', 'padding': '6px 8px'},
                    ),
                ],
                className='filter-group',
                style={'flex': '0 1 120px', 'maxWidth': '140px'},
            ),
            html.Div(
                [
                    html.Div("Utilization", className='filter-group-header'),
                    dcc.Input(
                        id='utilization-rate-input',
                        type='number',
                        value=85,
                        min=0,
                        max=100,
                        step=1,
                        className='filter-input',
                        style={'width': '100%', 'height': '36px', 'fontSize': '13px', 'padding': '6px 8px'},
                    ),
                ],
                className='filter-group',
                style={'flex': '0 1 120px', 'maxWidth': '140px'},
            ),
            html.Div(
                [
                    html.Div("Historical Data", className='filter-group-header'),
                    dcc.Checklist(
                        id='use-kpler-historical-checkbox',
                        options=[{'label': 'Kpler Historical', 'value': 'kpler'}],
                        value=[],
                        labelStyle={
                            'display': 'inline-flex',
                            'alignItems': 'center',
                            'gap': '6px',
                            'margin': '0',
                            'fontSize': '13px',
                            'fontWeight': '600',
                            'color': '#334155',
                        },
                        inputStyle={'margin': '0'},
                        style={
                            'display': 'inline-flex',
                            'alignItems': 'center',
                            'minHeight': '36px',
                            'padding': '6px 12px',
                            'border': '1px solid #d1d5db',
                            'borderRadius': '999px',
                            'backgroundColor': '#ffffff',
                        },
                    ),
                ],
                className='filter-group',
                style={'flex': '0 1 180px', 'maxWidth': '210px'},
            ),
        ],
        className='professional-section-header',
        style={'display': 'flex', 'gap': '12px', 'alignItems': 'flex-start', 'flexWrap': 'wrap'},
    ),
    # Charts Container with Professional Layout
    html.Div([
        html.Div([
            # Left column - Demand View
            html.Div([
                html.Div([
                    html.H4('Demand View', className="subheader-title-inline"),
                    html.Button(
                        'Export to Excel',
                        id='export-demand-metrics-button',
                        n_clicks=0,
                        style=SHIPPING_BALANCE_EXPORT_BUTTON_STYLE
                    ),
                ], className="fleet-metrics-table-heading"),
                dcc.Graph(
                    id='global-shipping-balance',
                    style={'height': '400px'},
                    config={'displayModeBar': False, 'responsive': True},
                ),
                # Regional Breakdown Section
                html.Div([
                    html.H5('Regional Breakdown', style={'marginTop': '24px', 'marginBottom': '12px', 'color': '#2C3E50'}),
                    html.Div([
                        html.Label("Aggregation:", style={'marginRight': '8px', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='demand-regional-aggregation-dropdown',
                            options=[
                                {'label': 'Year+Quarter', 'value': 'quarterly'},
                                {'label': 'Year+Month', 'value': 'monthly'},
                                {'label': 'Year+Season', 'value': 'seasonal'},
                                {'label': 'Year', 'value': 'yearly'}
                            ],
                            value='quarterly',
                            clearable=False,
                            style={'width': '150px', 'display': 'inline-block', 'marginRight': '20px'}
                        ),
                        html.Label("Year:", style={'marginRight': '8px', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='demand-regional-year-dropdown',
                            clearable=False,
                            style={'width': '120px', 'display': 'inline-block', 'marginRight': '20px'}
                        ),
                        html.Label("Period:", style={'marginRight': '8px', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='demand-regional-period-dropdown',
                            clearable=False,
                            style={'width': '120px', 'display': 'inline-block', 'marginRight': '20px'}
                        ),
                    ], style=SHIPPING_BALANCE_CONTROL_ROW_STYLE),
                    html.Div(id='demand-regional-table-container', style={'overflowX': 'auto'}),

                    # Laden Days Chart
                    html.Div([
                        html.H6('Laden Days Trends by Route', style={'marginTop': '20px', 'marginBottom': '12px', 'color': '#2C3E50'}),
                        html.Div([
                            html.Label("Origin Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='demand-regional-laden-origin-filter',
                                multi=True,
                                placeholder='All Origins',
                                style={'width': '250px', 'display': 'inline-block', 'marginRight': '20px'}
                            ),
                            html.Label("Destination Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='demand-regional-laden-dest-filter',
                                multi=True,
                                placeholder='All Destinations',
                                style={'width': '250px', 'display': 'inline-block'}
                            ),
                        ], style=SHIPPING_BALANCE_CONTROL_ROW_STYLE),
                        dcc.Graph(
                            id='demand-regional-laden-chart',
                            style={'height': '450px'},
                            config={'displayModeBar': False, 'responsive': True},
                        )
                    ]),

                    # Ballast Days Chart
                    html.Div([
                        html.H6('Ballast Days Trends by Route', style={'marginTop': '20px', 'marginBottom': '12px', 'color': '#2C3E50'}),
                        html.Div([
                            html.Label("Origin Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='demand-regional-ballast-origin-filter',
                                multi=True,
                                placeholder='All Origins',
                                style={'width': '250px', 'display': 'inline-block', 'marginRight': '20px'}
                            ),
                            html.Label("Destination Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='demand-regional-ballast-dest-filter',
                                multi=True,
                                placeholder='All Destinations',
                                style={'width': '250px', 'display': 'inline-block'}
                            ),
                        ], style=SHIPPING_BALANCE_CONTROL_ROW_STYLE),
                        dcc.Graph(
                            id='demand-regional-ballast-chart',
                            style={'height': '450px'},
                            config={'displayModeBar': False, 'responsive': True},
                        )
                    ])
                ])
            ], style=SHIPPING_BALANCE_PANEL_STYLE),

            # Right column - Supply View
            html.Div([
                html.Div([
                    html.H4('Supply View', className="subheader-title-inline"),
                    html.Button(
                        'Export to Excel',
                        id='export-supply-metrics-button',
                        n_clicks=0,
                        style=SHIPPING_BALANCE_EXPORT_BUTTON_STYLE
                    ),
                ], className="fleet-metrics-table-heading"),
                dcc.Graph(
                    id='global-shipping-balance-supply',
                    style={'height': '400px'},
                    config={'displayModeBar': False, 'responsive': True},
                ),
                # Regional Breakdown Section
                html.Div([
                    html.H5('Regional Breakdown', style={'marginTop': '24px', 'marginBottom': '12px', 'color': '#2C3E50'}),
                    html.Div([
                        html.Label("Aggregation:", style={'marginRight': '8px', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='supply-regional-aggregation-dropdown',
                            options=[
                                {'label': 'Year+Quarter', 'value': 'quarterly'},
                                {'label': 'Year+Month', 'value': 'monthly'},
                                {'label': 'Year+Season', 'value': 'seasonal'},
                                {'label': 'Year', 'value': 'yearly'}
                            ],
                            value='quarterly',
                            clearable=False,
                            style={'width': '150px', 'display': 'inline-block', 'marginRight': '20px'}
                        ),
                        html.Label("Year:", style={'marginRight': '8px', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='supply-regional-year-dropdown',
                            clearable=False,
                            style={'width': '120px', 'display': 'inline-block', 'marginRight': '20px'}
                        ),
                        html.Label("Period:", style={'marginRight': '8px', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='supply-regional-period-dropdown',
                            clearable=False,
                            style={'width': '120px', 'display': 'inline-block', 'marginRight': '20px'}
                        ),
                    ], style=SHIPPING_BALANCE_CONTROL_ROW_STYLE),
                    html.Div(id='supply-regional-table-container', style={'overflowX': 'auto'}),

                    # Laden Days Chart
                    html.Div([
                        html.H6('Laden Days Trends by Route', style={'marginTop': '20px', 'marginBottom': '12px', 'color': '#2C3E50'}),
                        html.Div([
                            html.Label("Origin Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='supply-regional-laden-origin-filter',
                                multi=True,
                                placeholder='All Origins',
                                style={'width': '250px', 'display': 'inline-block', 'marginRight': '20px'}
                            ),
                            html.Label("Destination Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='supply-regional-laden-dest-filter',
                                multi=True,
                                placeholder='All Destinations',
                                style={'width': '250px', 'display': 'inline-block'}
                            ),
                        ], style=SHIPPING_BALANCE_CONTROL_ROW_STYLE),
                        dcc.Graph(
                            id='supply-regional-laden-chart',
                            style={'height': '450px'},
                            config={'displayModeBar': False, 'responsive': True},
                        )
                    ]),

                    # Ballast Days Chart
                    html.Div([
                        html.H6('Ballast Days Trends by Route', style={'marginTop': '20px', 'marginBottom': '12px', 'color': '#2C3E50'}),
                        html.Div([
                            html.Label("Origin Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='supply-regional-ballast-origin-filter',
                                multi=True,
                                placeholder='All Origins',
                                style={'width': '250px', 'display': 'inline-block', 'marginRight': '20px'}
                            ),
                            html.Label("Destination Regions:", style={'marginRight': '8px', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='supply-regional-ballast-dest-filter',
                                multi=True,
                                placeholder='All Destinations',
                                style={'width': '250px', 'display': 'inline-block'}
                            ),
                        ], style=SHIPPING_BALANCE_CONTROL_ROW_STYLE),
                        dcc.Graph(
                            id='supply-regional-ballast-chart',
                            style={'height': '450px'},
                            config={'displayModeBar': False, 'responsive': True},
                        )
                    ])
                ])
            ], style=SHIPPING_BALANCE_PANEL_STYLE),
        ], style=SHIPPING_BALANCE_TWO_COLUMN_STYLE)
    ], className='section-container', style={**SHIPPING_BALANCE_SECTION_STYLE, 'margin': '14px 12px 12px', 'padding': '12px'}),

    # Fleet Statistics Section
    html.Div([
        html.Div([
            html.H3('Fleet Statistics', className="section-title-inline"),
            html.Button(
                'Export to Excel',
                id='export-fleet-stats-button',
                n_clicks=0,
                style=SHIPPING_BALANCE_EXPORT_BUTTON_STYLE
            ),
        ], className="fleet-metrics-table-heading"),
        dcc.Graph(
            id='fleet-stats-chart',
            style={'height': '400px', 'marginTop': '16px'},
            config={'displayModeBar': False, 'responsive': True},
        )
    ], className='section-container', style={**SHIPPING_BALANCE_SECTION_STYLE, 'margin': '0 12px 12px', 'padding': '12px'}),


    # Intracountry Trade Analysis Section
    html.Div([
        html.Div([
            html.H2("Intracountry Trade Analysis", className='section-title-inline'),
            html.P("Analysis of domestic shipping patterns by origin country", className='section-subtitle')
        ], className='header-content'),

        html.Div([
            html.Div([
                html.Label("Filter by Year:", className='filter-label'),
                dcc.Dropdown(
                    id='intracountry-year-dropdown',
                    options=[{'label': 'All Years', 'value': 'All Years'}],
                    value=None,
                    clearable=False,
                    className='inline-dropdown'
                )
            ], className='filter-group'),

            html.Div([
                html.Label("Filter by Status:", className='filter-label'),
                dcc.Dropdown(
                    id='intracountry-status-dropdown',
                    options=[{'label': 'All Statuses', 'value': 'All Statuses'}],
                    value=['All Statuses'],
                    multi=True,
                    clearable=False,
                    className='inline-dropdown'
                )
            ], className='filter-group')
        ], className='filter-bar')
    ], className='inline-section-header', style={**SHIPPING_BALANCE_SECTION_STYLE, 'margin': '0 12px 12px', 'padding': '12px'}),

    # Intracountry Trade Visualizations Section - Enterprise Standard
    html.Div([
        # Chart Container with Professional Layout
        html.Div([
            # Left column - Trade Count
            html.Div([
                html.Div([
                    html.H4('Count of Intracountry Trades', className="subheader-title-inline"),
                    html.Button(
                        'Export to Excel',
                        id='export-intracountry-count-button',
                        n_clicks=0,
                        style=SHIPPING_BALANCE_EXPORT_BUTTON_STYLE
                    ),
                ], className="fleet-metrics-table-heading"),
                dcc.Graph(
                    id='intracountry-count-visualization',
                    style={'height': '600px'},
                    config={'displayModeBar': False, 'responsive': True},
                )
            ], style=SHIPPING_BALANCE_PANEL_STYLE),

            # Right column - Ton Miles
            html.Div([
                html.Div([
                    html.H4('Intracountry Ton Miles', className="subheader-title-inline"),
                    html.Button(
                        'Export to Excel',
                        id='export-intracountry-tonmiles-button',
                        n_clicks=0,
                        style=SHIPPING_BALANCE_EXPORT_BUTTON_STYLE
                    ),
                ], className="fleet-metrics-table-heading"),
                dcc.Graph(
                    id='intracountry-tonmiles-visualization',
                    style={'height': '600px'},
                    config={'displayModeBar': False, 'responsive': True},
                )
            ], style=SHIPPING_BALANCE_PANEL_STYLE)
        ], style=SHIPPING_BALANCE_TWO_COLUMN_STYLE)
    ], className='section-container', style={**SHIPPING_BALANCE_SECTION_STYLE, 'margin': '0 12px 12px', 'padding': '12px'}),

], style=SHIPPING_BALANCE_PAGE_STYLE)


def _build_shipping_balance_payload(
    *,
    source_state,
    aggregation_level,
    vessel_age,
    utilization_rate,
    window_end_date,
    use_kpler_historical,
):
    date_state = _get_shipping_balance_date_state(source_state.get('kpler_upload'))
    if window_end_date is None:
        yesterday = pd.Timestamp.now() - pd.Timedelta(days=1)
        available_max = date_state.get('max_date')
        window_end_date = (
            min(yesterday, pd.to_datetime(available_max))
            if pd.notna(available_max)
            else yesterday
        )
        window_end_date = pd.Timestamp(window_end_date).normalize()
    utilization_rate_decimal = utilization_rate / 100.0
    historical_max_date = pd.to_datetime(date_state['hist_date_max'])
    hist_date_max = historical_max_date + pd.offsets.MonthEnd(0)

    def build_balance(lng_view):
        return calc_global_shipping_balance(
            engine=engine,
            aggregation_level=aggregation_level,
            life_expectancy=vessel_age,
            lng_view=lng_view,
            utilization_rate=utilization_rate_decimal,
            window_end_date=window_end_date,
            return_regional=True,
            use_kpler_historical=use_kpler_historical,
        )

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix='shipping-balance') as executor:
        demand_future = executor.submit(build_balance, 'demand')
        supply_future = executor.submit(build_balance, 'supply')
        kpler_future = executor.submit(
            kpler_analysis,
            engine,
            None,
            hist_date_max,
            aggregation_level,
        )
        # Resolve in the same logical order as the legacy callback.
        df_regional_demand, df_global_shipping_balance = demand_future.result()
        df_regional_supply, df_global_shipping_balance_supply = supply_future.result()
        df_intracountry_trades, df_trades_shipping_region = kpler_future.result()

    df_filtered = df_trades_shipping_region[df_trades_shipping_region['year'] >= 2019]
    origin_regions = sorted(df_filtered['origin_shipping_region'].unique())
    region_options = [{'label': 'All Regions', 'value': 'All Regions'}] + [
        {'label': region, 'value': region} for region in origin_regions
    ]

    years = sorted(df_filtered['year'].unique())
    latest_year = max(years)
    year_options = [{'label': 'All Years', 'value': 'All Years'}] + [
        {'label': str(year), 'value': str(year)} for year in years
    ]

    region_statuses = sorted(df_trades_shipping_region['status'].unique())
    intracountry_statuses = sorted(df_intracountry_trades['status'].unique())
    status_options_region = [{'label': 'All Statuses', 'value': 'All Statuses'}] + [
        {'label': status.capitalize(), 'value': status} for status in region_statuses
    ]
    status_options_intracountry = [{'label': 'All Statuses', 'value': 'All Statuses'}] + [
        {'label': status.capitalize(), 'value': status} for status in intracountry_statuses
    ]
    status_options_single = [
        {'label': status['label'], 'value': status['value']}
        for status in status_options_intracountry
    ]

    options_data = {
        'region_options': region_options,
        'year_options': year_options,
        'latest_year': str(latest_year),
        'status_options_region': status_options_region,
        'status_options_intracountry': status_options_intracountry,
        'status_options_single': status_options_single,
        'aggregation_level': aggregation_level,
        'vessel_age': vessel_age,
        'utilization_rate': utilization_rate,
        'hist_date_max': hist_date_max.isoformat(),
    }

    max_date = historical_max_date
    min_date = date_state.get('min_date')
    placeholder_text = (
        f"YYYY-MM-DD (Max: {max_date.strftime('%Y-%m-%d')})"
        if pd.notna(max_date)
        else "YYYY-MM-DD"
    )
    min_date_str = min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else ''
    window_end_date_str = (
        window_end_date.strftime('%Y-%m-%d')
        if hasattr(window_end_date, 'strftime')
        else str(window_end_date)[:10]
    )

    return {
        'shipping_balance': df_global_shipping_balance.to_json(date_format='iso', orient='split'),
        'shipping_balance_supply': df_global_shipping_balance_supply.to_json(date_format='iso', orient='split'),
        'shipping_balance_regional': df_regional_demand.to_json(date_format='iso', orient='split'),
        'shipping_balance_supply_regional': df_regional_supply.to_json(date_format='iso', orient='split'),
        'options_data': options_data,
        'intracountry_data': df_intracountry_trades.to_json(date_format='iso', orient='split'),
        'placeholder_text': placeholder_text,
        'min_date_str': min_date_str,
        'window_end_date_str': window_end_date_str,
    }


# Callbacks
# Update the refresh_data callback to include the aggregation dropdown and window date picker
@callback(
    Output('shipping-balance-data-store', 'data'),
    Output('shipping-balance-supply-data-store', 'data'),
    Output('shipping-balance-regional-data-store', 'data'),
    Output('shipping-balance-supply-regional-data-store', 'data'),
    Output('dropdown-options-store', 'data'),
    Output('intracountry-data-store', 'data'),
    Output('window-end-date-input', 'placeholder'),
    Output('window-end-date-input', 'min'),
    Output('window-end-date-input', 'value'),
    Input('global-refresh-button', 'n_clicks'),
    Input('aggregation-dropdown', 'value'),
    Input('vessel-age-input', 'value'),
    Input('utilization-rate-input', 'value'),
    Input('window-end-date-input', 'value'),
    Input('use-kpler-historical-checkbox', 'value'),
    prevent_initial_call=False
)
def refresh_data(_n_clicks, aggregation_level='monthly', vessel_age=20, utilization_rate=85, window_end_date=None, use_kpler_checked=None):
    """Fetch and prepare all data needed for the dashboard."""
    # Parse window_end_date if it's provided as a string
    if window_end_date and isinstance(window_end_date, str) and window_end_date.strip():
        try:
            window_end_date = pd.to_datetime(window_end_date)
        except Exception:
            window_end_date = None
    elif not window_end_date or (isinstance(window_end_date, str) and not window_end_date.strip()):
        window_end_date = None

    try:
        source_state = _fetch_shipping_balance_source_state()
    except Exception:
        logger.warning("Shipping snapshot watermark lookup failed; using live-query fallback", exc_info=True)
        source_state = {"request_token": dt.datetime.now(dt.timezone.utc).isoformat()}

    # Canonicalize the default before keying the snapshot. The callback also
    # writes this value back to the input, so None and the resolved date must
    # address the same prepared revision rather than causing a second build.
    if window_end_date is None:
        date_state = _get_shipping_balance_date_state(source_state.get('kpler_upload'))
        yesterday = pd.Timestamp.now() - pd.Timedelta(days=1)
        available_max = date_state.get('max_date')
        window_end_date = (
            min(yesterday, pd.to_datetime(available_max))
            if pd.notna(available_max)
            else yesterday
        )
        window_end_date = pd.Timestamp(window_end_date).normalize()

    use_kpler_historical = 'kpler' in (use_kpler_checked or [])
    source_key = _build_source_key(
        SHIPPING_BALANCE_NAMESPACE,
        source_state,
        aggregation_level,
        vessel_age,
        utilization_rate,
        window_end_date,
        use_kpler_historical,
    )
    reference, payload = _get_or_build_snapshot(
        engine,
        namespace=SHIPPING_BALANCE_NAMESPACE,
        source_key=source_key,
        builder=lambda: _build_shipping_balance_payload(
            source_state=source_state,
            aggregation_level=aggregation_level,
            vessel_age=vessel_age,
            utilization_rate=utilization_rate,
            window_end_date=window_end_date,
            use_kpler_historical=use_kpler_historical,
        ),
        force=_was_global_refresh_triggered(),
        manifest={
            'aggregation_level': aggregation_level,
            'vessel_age': vessel_age,
            'utilization_rate': utilization_rate,
            'window_end_date': (
                window_end_date.strftime('%Y-%m-%d')
                if hasattr(window_end_date, 'strftime')
                else None
            ),
            'use_kpler_historical': use_kpler_historical,
        },
    )

    if _snapshot_is_shared(reference):
        stores = [
            _with_snapshot_slot(reference, slot)
            for slot in (
                'shipping_balance',
                'shipping_balance_supply',
                'shipping_balance_regional',
                'shipping_balance_supply_regional',
                'options_data',
                'intracountry_data',
            )
        ]
    else:
        stores = [
            payload['shipping_balance'],
            payload['shipping_balance_supply'],
            payload['shipping_balance_regional'],
            payload['shipping_balance_supply_regional'],
            payload['options_data'],
            payload['intracountry_data'],
        ]

    return (
        *stores,
        payload['placeholder_text'],
        payload['min_date_str'],
        payload['window_end_date_str'],
    )


# Update the update_visualizations callback to handle aggregation levels in chart formatting
@callback(
    Output('global-shipping-balance', 'figure'),
    Output('global-shipping-balance-supply', 'figure'),
    Input('shipping-balance-data-store', 'data'),
    Input('shipping-balance-supply-data-store', 'data'),
    Input('dropdown-options-store', 'data'),
)
def update_visualizations(shipping_balance, shipping_balance_supply, dropdown_options):
    """Update visualizations and tables based on selected filters."""
    shipping_balance = _resolve_shipping_store(shipping_balance)
    shipping_balance_supply = _resolve_shipping_store(shipping_balance_supply)
    dropdown_options = _resolve_shipping_store(dropdown_options)
    # Check if data is available
    if shipping_balance is None or shipping_balance_supply is None or dropdown_options is None:
        raise PreventUpdate

    df_global_shipping_balance = pd.read_json(StringIO(shipping_balance), orient='split')
    df_global_shipping_balance_supply = pd.read_json(StringIO(shipping_balance_supply), orient='split')

    # Ensure date columns are datetime type (JSON conversion loses dtype)
    if 'date' in df_global_shipping_balance.columns:
        df_global_shipping_balance['date'] = pd.to_datetime(df_global_shipping_balance['date'])
    if 'date' in df_global_shipping_balance_supply.columns:
        df_global_shipping_balance_supply['date'] = pd.to_datetime(df_global_shipping_balance_supply['date'])

    # Get the aggregation level if available
    aggregation_level = dropdown_options.get('aggregation_level', 'monthly')
    hist_date_max_str = dropdown_options.get('hist_date_max')

    return (
        _build_balance_dual_axis_chart(
            df_global_shipping_balance,
            scenario_name="Ships Demand Total",
            scenario_column="ships_demand",
            scenario_color=SHIPPING_BALANCE_CHART_COLORS["demand"],
            net_name="Net New",
            aggregation_level=aggregation_level,
            hist_date_max_str=hist_date_max_str,
        ),
        _build_balance_dual_axis_chart(
            df_global_shipping_balance_supply,
            scenario_name="Ships Supply Total",
            scenario_column="ships_demand",
            scenario_color=SHIPPING_BALANCE_CHART_COLORS["supply"],
            net_name="Net New",
            aggregation_level=aggregation_level,
            hist_date_max_str=hist_date_max_str,
        ),
    )


@callback(
    Output('download-demand-metrics-excel', 'data'),
    Input('export-demand-metrics-button', 'n_clicks'),
    State('shipping-balance-data-store', 'data'),
    State('dropdown-options-store', 'data'),
    prevent_initial_call=True
)
def export_demand_metrics_to_excel(n_clicks, shipping_balance, dropdown_options):
    """Export Demand View metrics to Excel."""
    if not n_clicks or shipping_balance is None or dropdown_options is None:
        raise PreventUpdate

    shipping_balance = _resolve_shipping_store(shipping_balance)
    dropdown_options = _resolve_shipping_store(dropdown_options)

    df = pd.read_json(StringIO(shipping_balance), orient='split')
    df['date'] = pd.to_datetime(df['date'])

    aggregation_level = dropdown_options.get('aggregation_level', 'monthly')

    table_data = df[['date', 'total_active_ships', 'ships_demand', 'net', 'utilization_ratio', 'value']].copy()
    table_data = table_data.rename(columns={
        'date': 'Date',
        'total_active_ships': 'Total Active Ships',
        'ships_demand': 'Ships Demand Total',
        'net': 'Net Balance',
        'utilization_ratio': 'Utilization (%)',
        'value': 'Volume (M m³)'
    })
    table_data['Date'] = pd.to_datetime(table_data['Date'])
    table_data['Volume (M m³)'] = (table_data['Volume (M m³)'] / 1000000).round(2)
    if aggregation_level == 'quarterly':
        table_data['Date'] = table_data['Date'].dt.to_period('Q').astype(str)
    elif aggregation_level == 'seasonal':
        table_data['Date'] = table_data['Date'].apply(lambda x: f"{x.year}-{'W' if x.month == 1 else 'S'}")
    elif aggregation_level == 'yearly':
        table_data['Date'] = table_data['Date'].dt.year
    else:
        table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m')
    for col in ['Total Active Ships', 'Ships Demand Total', 'Net Balance']:
        table_data[col] = table_data[col].round(1)
    table_data['Utilization (%)'] = table_data['Utilization (%)'].round(2)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        table_data.to_excel(writer, sheet_name='Demand Metrics', index=False)
        for worksheet in writer.sheets.values():
            for column_cells in worksheet.columns:
                max_length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 50)
    output.seek(0)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(output.getvalue(), f'Demand_Metrics_{aggregation_level}_{timestamp}.xlsx')


@callback(
    Output('download-supply-metrics-excel', 'data'),
    Input('export-supply-metrics-button', 'n_clicks'),
    State('shipping-balance-supply-data-store', 'data'),
    State('dropdown-options-store', 'data'),
    prevent_initial_call=True
)
def export_supply_metrics_to_excel(n_clicks, shipping_balance_supply, dropdown_options):
    """Export Supply View metrics to Excel."""
    if not n_clicks or shipping_balance_supply is None or dropdown_options is None:
        raise PreventUpdate

    shipping_balance_supply = _resolve_shipping_store(shipping_balance_supply)
    dropdown_options = _resolve_shipping_store(dropdown_options)

    df = pd.read_json(StringIO(shipping_balance_supply), orient='split')
    df['date'] = pd.to_datetime(df['date'])

    aggregation_level = dropdown_options.get('aggregation_level', 'monthly')

    table_data = df[['date', 'total_active_ships', 'ships_demand', 'net', 'utilization_ratio', 'value']].copy()
    table_data = table_data.rename(columns={
        'date': 'Date',
        'total_active_ships': 'Total Active Ships',
        'ships_demand': 'Ships Supply Total',
        'net': 'Net Balance',
        'utilization_ratio': 'Utilization (%)',
        'value': 'Volume (M m³)'
    })
    table_data['Date'] = pd.to_datetime(table_data['Date'])
    table_data['Volume (M m³)'] = (table_data['Volume (M m³)'] / 1000000).round(2)
    if aggregation_level == 'quarterly':
        table_data['Date'] = table_data['Date'].dt.to_period('Q').astype(str)
    elif aggregation_level == 'seasonal':
        table_data['Date'] = table_data['Date'].apply(lambda x: f"{x.year}-{'W' if x.month == 1 else 'S'}")
    elif aggregation_level == 'yearly':
        table_data['Date'] = table_data['Date'].dt.year
    else:
        table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m')
    for col in ['Total Active Ships', 'Ships Supply Total', 'Net Balance']:
        table_data[col] = table_data[col].round(1)
    table_data['Utilization (%)'] = table_data['Utilization (%)'].round(2)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        table_data.to_excel(writer, sheet_name='Supply Metrics', index=False)
        for worksheet in writer.sheets.values():
            for column_cells in worksheet.columns:
                max_length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 50)
    output.seek(0)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(output.getvalue(), f'Supply_Metrics_{aggregation_level}_{timestamp}.xlsx')


# Fleet Statistics Export Callback
@callback(
    Output('download-fleet-stats-excel', 'data'),
    Input('export-fleet-stats-button', 'n_clicks'),
    State('shipping-balance-data-store', 'data'),
    State('dropdown-options-store', 'data'),
    prevent_initial_call=True
)
def export_fleet_stats_to_excel(n_clicks, shipping_balance, dropdown_options):
    """Export Fleet Statistics data to Excel."""
    if not n_clicks or shipping_balance is None or dropdown_options is None:
        raise PreventUpdate

    shipping_balance = _resolve_shipping_store(shipping_balance)
    dropdown_options = _resolve_shipping_store(dropdown_options)

    df_global = pd.read_json(StringIO(shipping_balance), orient='split')
    df_global['date'] = pd.to_datetime(df_global['date'])

    aggregation_level = dropdown_options.get('aggregation_level', 'monthly')

    fleet_cols = ['date', 'total_active_ships', 'average_size_cubic_meters', 'ships_added', 'ships_removed']
    display_df = df_global[fleet_cols].copy()
    display_df['net_change'] = display_df['ships_added'] - display_df['ships_removed']

    if aggregation_level == 'monthly':
        display_df['date_formatted'] = display_df['date'].dt.strftime('%Y-%m')
    elif aggregation_level == 'quarterly':
        display_df['date_formatted'] = display_df['date'].dt.year.astype(str) + '-Q' + display_df['date'].dt.quarter.astype(str)
    elif aggregation_level == 'seasonal':
        display_df['date_formatted'] = display_df['date'].dt.year.astype(str) + '-' + display_df['date'].dt.month.map({1: 'Winter', 7: 'Summer'})
    elif aggregation_level == 'yearly':
        display_df['date_formatted'] = display_df['date'].dt.year.astype(str)
    else:
        display_df['date_formatted'] = display_df['date'].dt.strftime('%Y-%m')

    display_df = display_df.rename(columns={
        'date_formatted': 'Period',
        'total_active_ships': 'Total Active Ships',
        'average_size_cubic_meters': 'Avg Capacity (m³)',
        'ships_added': 'Ships Added',
        'ships_removed': 'Ships Removed',
        'net_change': 'Net Change'
    })
    display_df = display_df[['Period', 'Total Active Ships', 'Avg Capacity (m³)', 'Ships Added', 'Ships Removed', 'Net Change']]
    display_df = display_df.sort_index(ascending=False)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        display_df.to_excel(writer, sheet_name='Fleet Statistics', index=False)
        for worksheet in writer.sheets.values():
            for column_cells in worksheet.columns:
                max_length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 50)
    output.seek(0)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(output.getvalue(), f'Fleet_Statistics_{aggregation_level}_{timestamp}.xlsx')


# Fleet Statistics Chart Callback
@callback(
    Output('fleet-stats-chart', 'figure'),
    Input('shipping-balance-data-store', 'data'),
    Input('dropdown-options-store', 'data'),
    prevent_initial_call=False
)
def update_fleet_statistics_chart(shipping_balance, dropdown_options):
    """Update fleet statistics chart with dual axes."""
    if shipping_balance is None or dropdown_options is None:
        return {}

    shipping_balance = _resolve_shipping_store(shipping_balance)
    dropdown_options = _resolve_shipping_store(dropdown_options)

    try:
        # Convert stored JSON back to DataFrame
        df_global = pd.read_json(StringIO(shipping_balance), orient='split')

        # Ensure date column is datetime
        if 'date' in df_global.columns:
            df_global['date'] = pd.to_datetime(df_global['date'])

        # Get aggregation level
        aggregation_level = dropdown_options.get('aggregation_level', 'monthly')

        return _build_fleet_statistics_figure(
            df_global,
            aggregation_level=aggregation_level,
            hist_date_max_str=dropdown_options.get('hist_date_max'),
        )

    except Exception as e:
        logger.exception("Error loading shipping balance fleet statistics chart")
        return _empty_shipping_balance_figure(f"Error loading chart: {str(e)}", height=400)




@callback(
    Output('intracountry-count-visualization', 'figure'),
    Output('intracountry-tonmiles-visualization', 'figure'),
    Output('intracountry-year-dropdown', 'options'),
    Output('intracountry-year-dropdown', 'value'),
    Output('intracountry-status-dropdown', 'options'),
    Output('intracountry-status-dropdown', 'value'),
    Input('intracountry-data-store', 'data'),
    Input('dropdown-options-store', 'data'),
    Input('intracountry-year-dropdown', 'value'),
    Input('intracountry-status-dropdown', 'value'),
    prevent_initial_call=False
)
def update_intracountry_visualizations(intracountry_data, dropdown_options, selected_year, selected_statuses):
    """Update intracountry visualizations and tables based on selected filters."""
    # Check if data is available
    if intracountry_data is None or dropdown_options is None:
        raise PreventUpdate

    intracountry_data = _resolve_shipping_store(intracountry_data)
    dropdown_options = _resolve_shipping_store(dropdown_options)

    # Convert stored JSON back to DataFrame
    df_intracountry_trades = pd.read_json(StringIO(intracountry_data), orient='split')

    # Extract dropdown options
    year_options = dropdown_options['year_options']
    status_options = dropdown_options['status_options_intracountry']
    latest_year = dropdown_options['latest_year']

    # Set default values if needed
    if selected_year is None:
        selected_year = latest_year
    if selected_statuses is None:
        selected_statuses = ['All Statuses']

    # Create visualizations
    fig_intracountry_count = create_stacked_bar_chart(
        df_intracountry_trades,
        metric='count_trades',
        title_suffix='Count of Trades',
        selected_statuses=selected_statuses,
        is_intracountry=True
    )

    fig_intracountry_tonmiles = create_stacked_bar_chart(
        df_intracountry_trades,
        metric='sum_ton_miles',
        title_suffix='Ton Miles',
        selected_statuses=selected_statuses,
        is_intracountry=True
    )

    return (
        fig_intracountry_count,
        fig_intracountry_tonmiles,
        year_options,
        selected_year,
        status_options,
        selected_statuses,
    )


@callback(
    Output('download-intracountry-count-excel', 'data'),
    Input('export-intracountry-count-button', 'n_clicks'),
    State('intracountry-data-store', 'data'),
    State('intracountry-year-dropdown', 'value'),
    State('intracountry-status-dropdown', 'value'),
    prevent_initial_call=True
)
def export_intracountry_count_to_excel(n_clicks, intracountry_data, selected_year, selected_statuses):
    """Export Count of Intracountry Trades data to Excel."""
    if not n_clicks or intracountry_data is None:
        raise PreventUpdate

    intracountry_data = _resolve_shipping_store(intracountry_data)

    df = pd.read_json(StringIO(intracountry_data), orient='split')
    table_data = prepare_table_data(
        df, 'count_trades',
        selected_year=selected_year,
        selected_statuses=selected_statuses,
        is_intracountry=True
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        table_data.to_excel(writer, sheet_name='Intracountry Count', index=False)
        for worksheet in writer.sheets.values():
            for column_cells in worksheet.columns:
                max_length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 50)
    output.seek(0)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(output.getvalue(), f'Intracountry_Count_{timestamp}.xlsx')


@callback(
    Output('download-intracountry-tonmiles-excel', 'data'),
    Input('export-intracountry-tonmiles-button', 'n_clicks'),
    State('intracountry-data-store', 'data'),
    State('intracountry-year-dropdown', 'value'),
    State('intracountry-status-dropdown', 'value'),
    prevent_initial_call=True
)
def export_intracountry_tonmiles_to_excel(n_clicks, intracountry_data, selected_year, selected_statuses):
    """Export Intracountry Ton Miles data to Excel."""
    if not n_clicks or intracountry_data is None:
        raise PreventUpdate

    intracountry_data = _resolve_shipping_store(intracountry_data)

    df = pd.read_json(StringIO(intracountry_data), orient='split')
    table_data = prepare_table_data(
        df, 'sum_ton_miles',
        selected_year=selected_year,
        selected_statuses=selected_statuses,
        is_intracountry=True
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        table_data.to_excel(writer, sheet_name='Intracountry Ton Miles', index=False)
        for worksheet in writer.sheets.values():
            for column_cells in worksheet.columns:
                max_length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 50)
    output.seek(0)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(output.getvalue(), f'Intracountry_TonMiles_{timestamp}.xlsx')


# Regional Breakdown Callbacks
def _build_regional_period_dropdown_options(regional_data, aggregation_level):
    """Build year and period dropdown options for regional demand/supply tables."""
    if regional_data is None:
        raise PreventUpdate

    regional_data = _resolve_shipping_store(regional_data)

    df_regional = pd.read_json(StringIO(regional_data), orient='split')

    df_regional['year'] = pd.to_datetime(df_regional['date']).dt.year
    df_regional['month'] = pd.to_datetime(df_regional['date']).dt.month
    df_regional['quarter'] = pd.to_datetime(df_regional['date']).dt.quarter

    years = sorted(df_regional['year'].unique())
    year_options = [{'label': str(year), 'value': year} for year in years]
    default_year = years[-1] if years else None

    if aggregation_level == 'monthly':
        period_options = [{'label': f'Month {i}', 'value': i} for i in range(1, 13)]
        default_period = df_regional[df_regional['year'] == default_year]['month'].max() if default_year else 1
    elif aggregation_level == 'quarterly':
        period_options = [{'label': f'Q{i}', 'value': i} for i in range(1, 5)]
        default_period = df_regional[df_regional['year'] == default_year]['quarter'].max() if default_year else 1
    elif aggregation_level == 'seasonal':
        period_options = [{'label': 'Winter', 'value': 1}, {'label': 'Summer', 'value': 7}]
        default_period = df_regional[df_regional['year'] == default_year]['month'].max() if default_year else 1
        default_period = 1 if default_period in [1, 2, 3, 4, 5, 6] else 7
    else:  # yearly
        period_options = [{'label': 'Full Year', 'value': 1}]
        default_period = 1

    return year_options, default_year, period_options, default_period


@callback(
    Output('demand-regional-year-dropdown', 'options'),
    Output('demand-regional-year-dropdown', 'value'),
    Output('demand-regional-period-dropdown', 'options'),
    Output('demand-regional-period-dropdown', 'value'),
    Input('shipping-balance-regional-data-store', 'data'),
    Input('demand-regional-aggregation-dropdown', 'value'),
    prevent_initial_call=False
)
def update_demand_regional_dropdowns(regional_data, aggregation_level):
    """Update year and period dropdown options for demand regional table."""
    return _build_regional_period_dropdown_options(regional_data, aggregation_level)


@callback(
    Output('supply-regional-year-dropdown', 'options'),
    Output('supply-regional-year-dropdown', 'value'),
    Output('supply-regional-period-dropdown', 'options'),
    Output('supply-regional-period-dropdown', 'value'),
    Input('shipping-balance-supply-regional-data-store', 'data'),
    Input('supply-regional-aggregation-dropdown', 'value'),
    prevent_initial_call=False
)
def update_supply_regional_dropdowns(regional_data, aggregation_level):
    """Update year and period dropdown options for supply regional table."""
    return _build_regional_period_dropdown_options(regional_data, aggregation_level)


def _build_regional_breakdown_table(
    regional_data,
    aggregation_level,
    selected_year,
    selected_period,
    *,
    validation_label,
    woodmac_direction,
    woodmac_measured_at,
    grid_id,
    log_label,
):
    """Build the regional demand/supply breakdown table for a selected period."""
    if regional_data is None or selected_year is None or selected_period is None:
        raise PreventUpdate

    regional_data = _resolve_shipping_store(regional_data)

    try:
        df_regional = pd.read_json(StringIO(regional_data), orient='split')

        df_regional['year'] = pd.to_datetime(df_regional['date']).dt.year
        df_regional['month'] = pd.to_datetime(df_regional['date']).dt.month
        df_regional['quarter'] = pd.to_datetime(df_regional['date']).dt.quarter

        if aggregation_level == 'monthly':
            df_filtered = df_regional[(df_regional['year'] == selected_year) & (df_regional['month'] == selected_period)]
        elif aggregation_level == 'quarterly':
            df_filtered = df_regional[(df_regional['year'] == selected_year) & (df_regional['quarter'] == selected_period)]
        elif aggregation_level == 'seasonal':
            df_filtered = df_regional[(df_regional['year'] == selected_year) & (df_regional['month'] == selected_period)]
        else:
            df_filtered = df_regional[df_regional['year'] == selected_year]

        if df_filtered.empty:
            return html.Div("No data available for the selected period.", style={'padding': '10px', 'color': '#666'})

        cols_to_select = ['origin_shipping_region', 'destination_shipping_region', 'value', 'ships_demand',
                         'mean_vessel_capacity', 'trade_vessel_capacity',
                         'mean_cargo_cubic_meters', 'count_laden_trades', 'count_nonladen_trades',
                         'sum_ton_miles', 'median_vessel_speed_laden', 'median_vessel_speed_nonladen',
                         'utilization_ratio', 'median_laden_days', 'median_nonladen_days']

        if 'sum_cargo' in df_filtered.columns:
            cols_to_select.append('sum_cargo')
        if 'sum_vessel_capacity' in df_filtered.columns:
            cols_to_select.append('sum_vessel_capacity')

        display_df = df_filtered[cols_to_select].copy()
        display_df['value'] = (display_df['value'] / 1000000).round(2)

        total_volume = display_df['value'].sum()
        display_df['volume_pct'] = (display_df['value'] / total_volume * 100).round(2) if total_volume > 0 else 0

        if 'sum_cargo' in display_df.columns:
            display_df['sum_cargo'] = (display_df['sum_cargo'] / 1000000).round(2)
        if 'sum_vessel_capacity' in display_df.columns:
            display_df['sum_vessel_capacity'] = (display_df['sum_vessel_capacity'] / 1000000).round(2)

        rename_dict = {
            'origin_shipping_region': 'Origin Region',
            'destination_shipping_region': 'Destination Region',
            'value': 'LNG Volume (M m³)',
            'volume_pct': 'Volume Share (%)',
            'ships_demand': 'Ships Demand',
            'mean_vessel_capacity': 'Fleet Avg Size (m³)',
            'trade_vessel_capacity': 'Trade Avg Size (m³)',
            'mean_cargo_cubic_meters': 'Avg Cargo Volume (m³)',
            'count_laden_trades': 'Laden Trades',
            'count_nonladen_trades': 'Ballast Trades',
            'sum_ton_miles': 'Total Ton-Miles',
            'median_vessel_speed_laden': 'Speed Laden (kts)',
            'median_vessel_speed_nonladen': 'Speed Ballast (kts)',
            'utilization_ratio': 'Utilization Rate (%)',
            'median_laden_days': 'Laden Days',
            'median_nonladen_days': 'Ballast Days'
        }

        if 'sum_cargo' in display_df.columns:
            rename_dict['sum_cargo'] = 'Sum Cargo (M m³)'
        if 'sum_vessel_capacity' in display_df.columns:
            rename_dict['sum_vessel_capacity'] = 'Sum Vessel Capacity (M m³)'

        display_df = display_df.rename(columns=rename_dict)
        display_df = display_df.sort_values('Ships Demand', ascending=False)

        total_lng_volume = display_df['LNG Volume (M m³)'].sum()

        try:
            woodmac_total, woodmac_sources = _fetch_woodmac_flow_validation_total(
                woodmac_direction,
                woodmac_measured_at,
                aggregation_level,
                selected_year,
                selected_period,
            )

            difference = total_lng_volume - woodmac_total
            pct_diff = (difference / woodmac_total * 100) if woodmac_total > 0 else 0

            if abs(pct_diff) < 5:
                diff_color = '#27AE60'
            elif abs(pct_diff) < 10:
                diff_color = '#F39C12'
            else:
                diff_color = '#E74C3C'

        except Exception:
            woodmac_total = 0
            woodmac_sources = 'Error'
            difference = 0
            pct_diff = 0
            diff_color = '#999'

        validation_summary = html.Div([
            html.Div([
                html.Strong(f"{validation_label} volume check", style=SHIPPING_BALANCE_VALIDATION_LABEL_STYLE),
                html.Span(f"Regional {total_lng_volume:,.2f} M m³", style=SHIPPING_BALANCE_VALIDATION_VALUE_STYLE),
                html.Span("|", style=SHIPPING_BALANCE_VALIDATION_SEPARATOR_STYLE),
                html.Span(f"WoodMac {woodmac_sources}: {woodmac_total:,.2f} M m³", style=SHIPPING_BALANCE_VALIDATION_VALUE_STYLE),
                html.Span("|", style=SHIPPING_BALANCE_VALIDATION_SEPARATOR_STYLE),
                html.Span(f"Diff {difference:+,.2f} M m³ ({pct_diff:+.1f}%)", style={**SHIPPING_BALANCE_VALIDATION_VALUE_STYLE, 'color': diff_color}),
            ], style=SHIPPING_BALANCE_VALIDATION_STYLE)
        ])

        table = create_ag_grid_table(
            display_df,
            id_value=grid_id,
            text_columns={'Origin Region', 'Destination Region'},
            pinned_columns={'Origin Region', 'Destination Region'},
            numeric_precision={
                'LNG Volume (M m³)': 1,
                'Volume Share (%)': 1,
                'Ships Demand': 1,
                'Fleet Avg Size (m³)': 0,
                'Trade Avg Size (m³)': 0,
                'Avg Cargo Volume (m³)': 0,
                'Laden Trades': 0,
                'Ballast Trades': 0,
                'Total Ton-Miles': 0,
                'Speed Laden (kts)': 1,
                'Speed Ballast (kts)': 1,
                'Utilization Rate (%)': 1,
                'Sum Cargo (M m³)': 1,
                'Sum Vessel Capacity (M m³)': 1,
                'Laden Days': 1,
                'Ballast Days': 1,
            },
            page_size=15,
            height=520,
            extra_class="shipping-balance-grid--regional",
            column_size="autoSize",
        )

        return html.Div([validation_summary, table])

    except Exception as e:
        logger.exception(f"Error loading {log_label} regional table")
        return html.Div(f"Error loading regional table: {str(e)}", style={'padding': '10px', 'color': 'red'})


@callback(
    Output('demand-regional-table-container', 'children'),
    Input('shipping-balance-regional-data-store', 'data'),
    Input('demand-regional-aggregation-dropdown', 'value'),
    Input('demand-regional-year-dropdown', 'value'),
    Input('demand-regional-period-dropdown', 'value'),
    prevent_initial_call=False
)
def update_demand_regional_table(regional_data, aggregation_level, selected_year, selected_period):
    """Update demand regional breakdown table."""
    return _build_regional_breakdown_table(
        regional_data,
        aggregation_level,
        selected_year,
        selected_period,
        validation_label="Demand",
        woodmac_direction="Import",
        woodmac_measured_at="Entry",
        grid_id="demand-regional-ag-grid",
        log_label="demand",
    )


@callback(
    Output('supply-regional-table-container', 'children'),
    Input('shipping-balance-supply-regional-data-store', 'data'),
    Input('supply-regional-aggregation-dropdown', 'value'),
    Input('supply-regional-year-dropdown', 'value'),
    Input('supply-regional-period-dropdown', 'value'),
    prevent_initial_call=False
)
def update_supply_regional_table(regional_data, aggregation_level, selected_year, selected_period):
    """Update supply regional breakdown table."""
    return _build_regional_breakdown_table(
        regional_data,
        aggregation_level,
        selected_year,
        selected_period,
        validation_label="Supply",
        woodmac_direction="Export",
        woodmac_measured_at="Exit",
        grid_id="supply-regional-ag-grid",
        log_label="supply",
    )

############################################ Regional Trends Chart Callbacks ###################################################

def _regional_origin_destination_filter_options(regional_data):
    if regional_data is None:
        return [], []
    try:
        regional_data = _resolve_shipping_store(regional_data)
        df_regional = pd.read_json(StringIO(regional_data), orient='split')
        origins = sorted(df_regional['origin_shipping_region'].unique())
        destinations = sorted(df_regional['destination_shipping_region'].unique())
        return [{'label': r, 'value': r} for r in origins], [{'label': r, 'value': r} for r in destinations]
    except Exception:
        return [], []


# Demand Regional Filter Options
@callback(
    Output('demand-regional-laden-origin-filter', 'options'),
    Output('demand-regional-laden-dest-filter', 'options'),
    Output('demand-regional-ballast-origin-filter', 'options'),
    Output('demand-regional-ballast-dest-filter', 'options'),
    Input('shipping-balance-regional-data-store', 'data'),
    prevent_initial_call=False
)
def update_demand_regional_filters(regional_data):
    origin_options, destination_options = _regional_origin_destination_filter_options(regional_data)
    return origin_options, destination_options, origin_options, destination_options


# Supply Regional Filter Options
@callback(
    Output('supply-regional-laden-origin-filter', 'options'),
    Output('supply-regional-laden-dest-filter', 'options'),
    Output('supply-regional-ballast-origin-filter', 'options'),
    Output('supply-regional-ballast-dest-filter', 'options'),
    Input('shipping-balance-supply-regional-data-store', 'data'),
    prevent_initial_call=False
)
def update_supply_regional_filters(regional_data):
    origin_options, destination_options = _regional_origin_destination_filter_options(regional_data)
    return origin_options, destination_options, origin_options, destination_options


def _render_regional_route_days_chart(
    regional_data,
    origin_filter,
    dest_filter,
    dropdown_options,
    *,
    metric_col,
    title,
    y_label,
    log_label,
):
    try:
        regional_data = _resolve_shipping_store(regional_data)
        dropdown_options = _resolve_shipping_store(dropdown_options)
        return _build_route_days_chart(
            regional_data,
            origin_filter,
            dest_filter,
            dropdown_options,
            metric_col=metric_col,
            title=title,
            y_label=y_label,
        )
    except Exception as e:
        logger.exception("Error loading %s chart", log_label)
        return _empty_shipping_balance_figure(f'Error loading chart: {e}', height=450)


# Demand Regional Laden Chart
@callback(
    Output('demand-regional-laden-chart', 'figure'),
    Input('shipping-balance-regional-data-store', 'data'),
    Input('demand-regional-laden-origin-filter', 'value'),
    Input('demand-regional-laden-dest-filter', 'value'),
    Input('dropdown-options-store', 'data'),
    prevent_initial_call=False
)
def update_demand_regional_laden_chart(regional_data, origin_filter, dest_filter, dropdown_options):
    return _render_regional_route_days_chart(
        regional_data,
        origin_filter,
        dest_filter,
        dropdown_options,
        metric_col='median_laden_days',
        title='Laden Days by Route Over Time',
        y_label='Laden days',
        log_label='demand regional laden chart',
    )


# Demand Regional Ballast Chart
@callback(
    Output('demand-regional-ballast-chart', 'figure'),
    Input('shipping-balance-regional-data-store', 'data'),
    Input('demand-regional-ballast-origin-filter', 'value'),
    Input('demand-regional-ballast-dest-filter', 'value'),
    Input('dropdown-options-store', 'data'),
    prevent_initial_call=False
)
def update_demand_regional_ballast_chart(regional_data, origin_filter, dest_filter, dropdown_options):
    return _render_regional_route_days_chart(
        regional_data,
        origin_filter,
        dest_filter,
        dropdown_options,
        metric_col='median_nonladen_days',
        title='Ballast Days by Route Over Time',
        y_label='Ballast days',
        log_label='demand regional ballast chart',
    )


# Supply Regional Laden Chart
@callback(
    Output('supply-regional-laden-chart', 'figure'),
    Input('shipping-balance-supply-regional-data-store', 'data'),
    Input('supply-regional-laden-origin-filter', 'value'),
    Input('supply-regional-laden-dest-filter', 'value'),
    Input('dropdown-options-store', 'data'),
    prevent_initial_call=False
)
def update_supply_regional_laden_chart(regional_data, origin_filter, dest_filter, dropdown_options):
    return _render_regional_route_days_chart(
        regional_data,
        origin_filter,
        dest_filter,
        dropdown_options,
        metric_col='median_laden_days',
        title='Laden Days by Route Over Time',
        y_label='Laden days',
        log_label='supply regional laden chart',
    )


# Supply Regional Ballast Chart
@callback(
    Output('supply-regional-ballast-chart', 'figure'),
    Input('shipping-balance-supply-regional-data-store', 'data'),
    Input('supply-regional-ballast-origin-filter', 'value'),
    Input('supply-regional-ballast-dest-filter', 'value'),
    Input('dropdown-options-store', 'data'),
    prevent_initial_call=False
)
def update_supply_regional_ballast_chart(regional_data, origin_filter, dest_filter, dropdown_options):
    return _render_regional_route_days_chart(
        regional_data,
        origin_filter,
        dest_filter,
        dropdown_options,
        metric_col='median_nonladen_days',
        title='Ballast Days by Route Over Time',
        y_label='Ballast days',
        log_label='supply regional ballast chart',
    )
