import datetime as dt
import json
import logging
import math
from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
import threading

import dash_ag_grid as dag
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import callback, ctx, dcc, html, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import MissingCallbackContextException, PreventUpdate
from plotly.subplots import make_subplots
from sqlalchemy import bindparam, text

from utils.dashboard_snapshot_cache import (
    build_source_key as _build_source_key,
    commit_snapshot_publication_stage as _commit_snapshot_publication_stage,
    get_or_build_snapshot as _get_or_build_snapshot,
    get_snapshot_if_available as _get_snapshot_if_available,
    is_snapshot_reference as _is_snapshot_reference,
    resolve_snapshot as _resolve_snapshot,
    resolve_snapshot_manifest as _resolve_snapshot_manifest,
    snapshot_build_lock as _snapshot_build_lock,
    snapshot_is_resolvable as _snapshot_is_resolvable,
    stage_snapshot_publication as _stage_snapshot_publication,
    was_global_refresh_triggered as _was_global_refresh_triggered,
)
from utils.arrow_payload import (
    pack_dataframe_mapping as _pack_dataframe_mapping,
    unpack_dataframe_mapping as _unpack_dataframe_mapping,
)
from utils.database import DB_SCHEMA, engine
from utils.performance_flags import (
    fleet_arrow_source_enabled as _fleet_arrow_source_enabled,
    fleet_render_snapshot_enabled as _fleet_render_snapshot_enabled,
    fleet_staged_render_enabled as _fleet_staged_render_enabled,
    revision_aware_refresh_enabled as _revision_aware_refresh_enabled,
)
from utils.performance import log_callback_timing

logger = logging.getLogger(__name__)


KPLER_FLEET_METRICS_TABLE = "kpler_lng_fleet_metrics_series"
KPLER_REGIONAL_SIGNAL_TABLE = "kpler_lng_regional_signal_series"
KPLER_DIVERSIONS_TABLE = "kpler_lng_diversions"
PRICE_CURVE_TABLE = "curve"
KPLER_FLEET_DEFAULT_ZONE_FILTER = "asia_pacific_oceans"
KPLER_FLEET_DEFAULT_START_DATE = dt.date(2021, 1, 1)
PRICE_FRESHNESS_DAYS = 7
RELATION_EXISTS_CACHE_SECONDS = 300
CHART_HEIGHT = 500
SIGNAL_REGION_ROW_CHART_HEIGHT = 430
CONGESTION_SIGNAL_CHART_HEIGHT = 720
SEASONAL_CHART_HEIGHT = 481
DIVERSION_SEASONAL_CHART_HEIGHT = 440
REGION_DETAIL_MATRIX_HEIGHT = 1500
SEASONAL_MONTH_TICKVALS = [1, 5, 9, 14, 18, 22, 27, 31, 36, 40, 44, 49]
SEASONAL_MONTH_TICKTEXT = ["Ja", "Fe", "Ma", "Ap", "My", "Jn", "Jl", "Au", "Se", "Oc", "No", "De"]

KPLER_FLEET_ZONE_OPTIONS = [
    {"label": "Asia-Pacific", "value": "asia_pacific_oceans"},
    {"label": "Europe", "value": "europe_basin"},
    {"label": "Americas", "value": "americas_basin"},
    {"label": "Middle East / Indian Ocean", "tab_label": "ME / Indian", "value": "middle_east_indian_ocean"},
    {"label": "Atlantic Basin", "tab_label": "Atlantic", "value": "atlantic_basin"},
    {"label": "Global", "value": "global"},
]
KPLER_FLEET_REGION_ORDER = [option["value"] for option in KPLER_FLEET_ZONE_OPTIONS]
KPLER_FLEET_SEASONAL_REGION_ORDER = ["global"] + [
    zone_filter for zone_filter in KPLER_FLEET_REGION_ORDER if zone_filter != "global"
]
KPLER_FLEET_DIVERSION_REGION_ORDER = [
    zone_filter for zone_filter in KPLER_FLEET_REGION_ORDER if zone_filter != "global"
]

FLEET_METRICS_LEGACY_SNAPSHOT_NAMESPACE = "fleet-metrics-source-v1"
FLEET_METRICS_ARROW_SNAPSHOT_NAMESPACE = "fleet-metrics-source-v2"
FLEET_METRICS_SNAPSHOT_NAMESPACE = (
    FLEET_METRICS_ARROW_SNAPSHOT_NAMESPACE
    if _fleet_arrow_source_enabled()
    else FLEET_METRICS_LEGACY_SNAPSHOT_NAMESPACE
)
FLEET_METRICS_SOURCE_NAMESPACES = frozenset({
    FLEET_METRICS_LEGACY_SNAPSHOT_NAMESPACE,
    FLEET_METRICS_ARROW_SNAPSHOT_NAMESPACE,
})
FLEET_METRICS_RENDER_BUNDLE_NAMESPACE = "fleet-metrics-render-v1"
FLEET_METRICS_RENDER_SUMMARY_NAMESPACE = "fleet-metrics-render-summary-v1"
FLEET_METRICS_RENDER_SIGNALS_NAMESPACE = "fleet-metrics-render-signals-v1"
FLEET_METRICS_RENDER_DETAIL_NAMESPACE = "fleet-metrics-render-detail-v1"
FLEET_METRICS_RENDER_BUNDLE_FORMAT = "fleet-metrics-render-bundle-v1"
FLEET_METRICS_RENDER_SUMMARY_FORMAT = "fleet-metrics-render-summary-v1"
FLEET_METRICS_RENDER_SIGNALS_FORMAT = "fleet-metrics-render-signals-v1"
FLEET_METRICS_RENDER_DETAIL_FORMAT = "fleet-metrics-render-detail-v1"
FLEET_METRICS_RENDER_SCHEMA_VERSION = 1
FLEET_METRICS_SOURCE_FRAME_KEYS = (
    "detail_matrix_floating_days",
    "regional_signals",
    "detail_matrix_weekly",
    "summary_daily",
    "signal_diversions",
    "diversion_history",
    "global_area_weekly",
)


class _FleetMetricsSourceChanged(RuntimeError):
    """Raised before cache publication when Fleet inputs change mid-build."""
_FLEET_RENDER_CACHE = OrderedDict()
_FLEET_COMMON_RENDER_CACHE = OrderedDict()
_FLEET_COMMON_RENDER_FLIGHTS: dict[tuple, Future] = {}
_FLEET_RENDER_CACHE_LOCK = threading.Lock()
_FLEET_REGION_OUTPUT_INDICES = frozenset({0, 1, 3, 5, 6, 13})


def _fetch_fleet_metrics_source_state():
    query = text(f"""
        SELECT
            (SELECT completed_at_utc
             FROM {_table_ref('kpler_ingestion_runs')}
             WHERE module = 'kpler_fleet_metrics' AND status = 'published'
             ORDER BY completed_at_utc DESC LIMIT 1) AS fleet_checked_at,
            (SELECT completed_at_utc
             FROM {_table_ref('kpler_ingestion_runs')}
             WHERE module = 'kpler_regional_signals' AND status = 'published'
             ORDER BY completed_at_utc DESC LIMIT 1) AS signal_checked_at,
            (SELECT MAX(run_id)
             FROM {_table_ref('kpler_ingestion_runs')}
             WHERE module = 'kpler_fleet_metrics' AND status = 'published'
               AND inserted_rows + updated_rows > 0) AS fleet_revision,
            (SELECT MAX(run_id)
             FROM {_table_ref('kpler_ingestion_runs')}
             WHERE module = 'kpler_regional_signals' AND status = 'published'
               AND inserted_rows + updated_rows > 0) AS signal_revision,
            (SELECT MAX(upload_timestamp_utc) FROM {_table_ref(KPLER_FLEET_METRICS_TABLE)}) AS fleet_changed_at,
            (SELECT MAX(upload_timestamp_utc) FROM {_table_ref(KPLER_REGIONAL_SIGNAL_TABLE)}) AS signal_changed_at,
            (SELECT MAX(upload_timestamp_utc) FROM {_table_ref(KPLER_DIVERSIONS_TABLE)}) AS diversion_upload,
            (SELECT MAX(cob) FROM {_table_ref(PRICE_CURVE_TABLE)}
             WHERE code IN ('ICE_JKM_MO', 'ICE_TFU_MO')) AS price_cob
    """)
    with engine.connect() as connection:
        row = connection.execute(query).mappings().first()
    return dict(row or {})


def _freshness_reference_payload(source_state):
    """Serialize operational source freshness outside semantic cache identity."""

    values = {
        "fleet_checked_at": source_state.get("fleet_checked_at")
        or source_state.get("fleet_upload"),
        "signal_checked_at": source_state.get("signal_checked_at")
        or source_state.get("signal_upload"),
        "fleet_changed_at": source_state.get("fleet_changed_at"),
        "signal_changed_at": source_state.get("signal_changed_at"),
    }
    payload = {}
    for key, value in values.items():
        if value is None or pd.isna(value):
            payload[key] = None
        elif hasattr(value, "isoformat"):
            payload[key] = value.isoformat()
        else:
            payload[key] = str(value)
    return payload


_FLEET_SEMANTIC_SOURCE_FIELDS = (
    "fleet_revision",
    "signal_revision",
    "diversion_upload",
    "price_cob",
)


def _fleet_semantic_source_state(source_state):
    """Return only values whose change can alter rendered Fleet output."""

    semantic_state = {
        key: source_state.get(key)
        for key in _FLEET_SEMANTIC_SOURCE_FIELDS
    }
    if "request_token" in source_state:
        semantic_state["request_token"] = source_state["request_token"]
    return semantic_state
KPLER_FLEET_ZONE_SHORT_LABELS = {option["value"]: option["label"] for option in KPLER_FLEET_ZONE_OPTIONS}
KPLER_REGION_DEFINITION_SUMMARIES = {
    "asia_pacific_oceans": "Asian LNG demand basin plus Pacific waiting and approach areas.",
    "europe_basin": "European receiving market plus the nearby seas and Atlantic approach used for delivery pressure.",
    "americas_basin": "North, Central, and South American market anchors with Gulf, Caribbean, and Atlantic approaches.",
    "middle_east_indian_ocean": "Middle East, South-Central Asia, and Indian Ocean corridors used for regional wait and delivery risk.",
    "atlantic_basin": "Atlantic Ocean overlay for cross-basin shipping pressure. It intentionally overlaps Europe and the Americas.",
    "global": "Benchmark view with no regional Kpler zone filter applied.",
}
KPLER_FLEET_ZONE_MEMBERS = {
    "asia_pacific_oceans": {
        "Eastern Asia",
        "South-East Asia",
        "North East Pacific Ocean",
        "North West Pacific Ocean",
        "Central West Pacific Ocean",
        "Central East Pacific Ocean",
        "South East Pacific Ocean",
        "South West Pacific Ocean",
    },
    "europe_basin": {
        "Northern Europe",
        "Western Europe",
        "Southern Europe",
        "Eastern Europe",
        "North Sea",
        "Baltic Sea",
        "English Channel",
        "MED Sea",
        "North East Atlantic Ocean",
    },
    "americas_basin": {
        "Americas",
        "Gulf of Mexico",
        "Caribbean Sea",
        "North West Atlantic Ocean",
        "South West Atlantic Ocean",
    },
    "middle_east_indian_ocean": {
        "Middle East",
        "South-Central Asia",
        "Mideast Gulf",
        "Arabian Sea",
        "Red Sea",
        "West Indian Ocean",
        "Central Indian Ocean",
        "East Indian Ocean",
    },
    "atlantic_basin": {"Atlantic Ocean"},
    "global": set(),
}
KPLER_REGION_COUNTRIES = {
    "asia_pacific_oceans": {
        "Australia",
        "Brunei",
        "Cambodia",
        "China",
        "East Timor",
        "Hong Kong",
        "Indonesia",
        "Japan",
        "Malaysia",
        "Myanmar",
        "New Zealand",
        "Papua New Guinea",
        "Philippines",
        "Singapore",
        "Singapore Republic",
        "South Korea",
        "Taiwan",
        "Thailand",
        "Vietnam",
    },
    "europe_basin": {
        "Belgium",
        "Croatia",
        "Finland",
        "France",
        "Germany",
        "Gibraltar",
        "Greece",
        "Italy",
        "Lithuania",
        "Malta",
        "Netherlands",
        "Norway",
        "Poland",
        "Portugal",
        "Spain",
        "Sweden",
        "Turkey",
        "United Kingdom",
    },
    "americas_basin": {
        "Argentina",
        "Bahamas",
        "Brazil",
        "Canada",
        "Chile",
        "Colombia",
        "Dominican Republic",
        "El Salvador",
        "Jamaica",
        "Mexico",
        "Panama",
        "Puerto Rico",
        "United States",
    },
    "middle_east_indian_ocean": {
        "Bangladesh",
        "Egypt",
        "India",
        "Jordan",
        "Kuwait",
        "Oman",
        "Pakistan",
        "Qatar",
        "United Arab Emirates",
    },
    "atlantic_basin": {
        "Argentina",
        "Bahamas",
        "Belgium",
        "Brazil",
        "Canada",
        "Chile",
        "Colombia",
        "Croatia",
        "Dominican Republic",
        "El Salvador",
        "Finland",
        "France",
        "Germany",
        "Gibraltar",
        "Greece",
        "Italy",
        "Jamaica",
        "Lithuania",
        "Malta",
        "Mexico",
        "Netherlands",
        "Norway",
        "Panama",
        "Poland",
        "Portugal",
        "Puerto Rico",
        "Republic of the Congo",
        "Senegal",
        "Spain",
        "Sweden",
        "Turkey",
        "United Kingdom",
        "United States",
    },
    "global": set(),
}
KPLER_FLEET_METRIC_LABELS = {
    "loaded_vessels": "Commodities on water",
    "floating_storage": "Floating storage",
}
KPLER_FLEET_SPLIT_OPTIONS = [
    {"label": "Subcontinent", "value": "current_subcontinents"},
    {"label": "Country", "value": "current_countries"},
    {"label": "Sea", "value": "current_seas"},
]
KPLER_FLEET_SPLIT_TITLE_LABELS = {
    "current_subcontinents": "current subcontinent",
    "current_countries": "current country",
    "current_seas": "current sea",
    "floating_days": "floating days",
}
KPLER_FLEET_AREA_LABELS = {
    "Unknown": "Others",
}
KPLER_FLEET_DEFAULT_AREAS = {
    ("asia_pacific_oceans", "current_subcontinents"): [
        "South-East Asia",
        "Eastern Asia",
        "Pacific Islands",
        "South America",
        "Northern America",
        "Central America",
        "Eastern Europe",
        "Australia and New Zealand",
        "Unknown",
    ],
    ("europe_basin", "current_subcontinents"): [
        "Northern Europe",
        "Western Europe",
        "Southern Europe",
        "Eastern Europe",
        "Northern Africa",
        "Western Africa",
        "Unknown",
    ],
    ("americas_basin", "current_subcontinents"): [
        "Northern America",
        "South America",
        "Caribbean Islands",
        "Central America",
        "Unknown",
    ],
    ("middle_east_indian_ocean", "current_subcontinents"): [
        "Middle East",
        "South-Central Asia",
        "South-East Asia",
        "Eastern Africa",
        "Southern Africa",
        "Northern Africa",
        "Unknown",
    ],
    ("atlantic_basin", "current_subcontinents"): [
        "Northern America",
        "Southern Europe",
        "South America",
        "Caribbean Islands",
        "Northern Europe",
        "Western Africa",
        "Western Europe",
        "Northern Africa",
        "Southern Africa",
        "Unknown",
    ],
    ("global", "current_subcontinents"): [
        "South-East Asia",
        "Eastern Asia",
        "Northern America",
        "Middle East",
        "Western Europe",
        "Southern Europe",
        "South America",
        "Northern Africa",
        "Unknown",
    ],
}
KPLER_FLEET_GENERIC_DEFAULT_AREAS = {
    "current_countries": [
        "China",
        "Japan",
        "South Korea",
        "India",
        "United States",
        "Spain",
        "Brazil",
        "Turkey",
    ],
    "current_seas": [
        "North East Pacific Ocean",
        "South China Sea",
        "North East Atlantic Ocean",
        "North West Atlantic Ocean",
        "MED Sea",
        "Gulf of Mexico",
        "Mideast Gulf",
        "West Indian Ocean",
    ],
}
KPLER_FLEET_COLORS = [
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
]
KPLER_FLEET_SPECIAL_AREA_COLORS = {
    "Unknown": "#687382",
}
KPLER_FLEET_MA_COLOR = "#ff5a1f"
AG_GRID_THEME = "ag-theme-alpine"

CARD_STYLE = {
    "background": "#ffffff",
    "border": "1px solid #e5e7eb",
    "borderRadius": "8px",
    "padding": "12px 14px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.06)",
    "minHeight": "86px",
}
SECTION_STYLE = {
    "background": "#ffffff",
    "border": "1px solid #e5e7eb",
    "borderRadius": "8px",
    "padding": "14px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.05)",
}

FLEET_METRICS_AG_GRID_DEFAULT_COL_DEF = {
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

FLEET_METRICS_AG_GRID_OPTIONS = {
    "animateRows": False,
    "pagination": True,
    "paginationPageSizeSelector": [6, 12, 20, 50],
    "suppressRowHoverHighlight": False,
    "suppressCellFocus": True,
    "enableCellTextSelection": True,
    "ensureDomOrder": True,
    "headerHeight": 32,
    "rowHeight": 30,
    "groupHeaderHeight": 28,
    "tooltipShowDelay": 250,
    "rowClassRules": {
        "fleet-metrics-selected-region-row": "params.data && params.data.is_selected === true",
        "fleet-metrics-global-row": "params.data && params.data.zone_filter === 'global'",
    },
}
_RELATION_EXISTS_CACHE = {}


def _schema_name():
    return DB_SCHEMA or "at_lng"


def _table_ref(table_name):
    schema = _schema_name()
    safe_schema = schema.replace('"', '""')
    safe_table = table_name.replace('"', '""')
    return f'"{safe_schema}"."{safe_table}"'


def _relation_exists(table_name):
    cache_key = (_schema_name(), table_name)
    cached = _RELATION_EXISTS_CACHE.get(cache_key)
    now = dt.datetime.now(dt.timezone.utc)
    if cached is not None:
        cached_at, cached_value = cached
        if (now - cached_at).total_seconds() <= RELATION_EXISTS_CACHE_SECONDS:
            return cached_value

    try:
        with engine.connect() as connection:
            exists = bool(
                connection.execute(
                    text("SELECT to_regclass(:table_name)"),
                    {"table_name": f"{_schema_name()}.{table_name}"},
                ).scalar()
            )
            _RELATION_EXISTS_CACHE[cache_key] = (now, exists)
            return exists
    except Exception:
        return False


def _fleet_metrics_table_exists():
    return _relation_exists(KPLER_FLEET_METRICS_TABLE)


def _regional_signal_table_exists():
    return _relation_exists(KPLER_REGIONAL_SIGNAL_TABLE)


def _diversions_table_exists():
    return _relation_exists(KPLER_DIVERSIONS_TABLE)


def _empty_figure(message, height=CHART_HEIGHT):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#64748b"),
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=50, r=30, t=40, b=55),
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _area_display_name(area_name):
    return KPLER_FLEET_AREA_LABELS.get(area_name, area_name)


def _region_label(zone_filter):
    return KPLER_FLEET_ZONE_SHORT_LABELS.get(zone_filter, zone_filter.replace("_", " ").title())


def _format_number(value, decimals=1, suffix=""):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{decimals}f}{suffix}"


def _format_delta(value, decimals=2, suffix=""):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):+,.{decimals}f}{suffix}"


def _format_integer(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.0f}"


def _format_percent(value, decimals=1):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{decimals}f}%"


def _latest_complete_week_label(summary_row):
    if not summary_row or not summary_row.get("latest_week"):
        return "No complete week"
    latest_week = pd.to_datetime(summary_row["latest_week"]).date()
    return f"Week of {latest_week:%d %b %Y}"


def _freshness_status(latest_week, end_date):
    if latest_week is None or pd.isna(latest_week):
        return "No data"
    latest_week = pd.to_datetime(latest_week).date()
    if latest_week < end_date - dt.timedelta(days=14):
        return "Stale"
    return "OK"


def _default_area_candidates(zone_filter, split_dimension):
    return (
        KPLER_FLEET_DEFAULT_AREAS.get((zone_filter, split_dimension))
        or KPLER_FLEET_GENERIC_DEFAULT_AREAS.get(split_dimension)
        or KPLER_FLEET_DEFAULT_AREAS.get(("asia_pacific_oceans", "current_subcontinents"), [])
    )


def fetch_area_options(split_dimension, zone_filter):
    if split_dimension not in {option["value"] for option in KPLER_FLEET_SPLIT_OPTIONS}:
        split_dimension = "current_subcontinents"

    if not _fleet_metrics_table_exists():
        return _default_area_candidates(zone_filter, split_dimension)

    query = text(f"""
        SELECT
            area_name,
            SUM(quantity_mtonnes) AS recent_quantity
        FROM {_table_ref(KPLER_FLEET_METRICS_TABLE)}
        WHERE period = 'daily'
          AND zone_filter = :zone_filter
          AND split_dimension = :split_dimension
          AND metric IN ('loaded_vessels', 'floating_storage')
          AND date >= CURRENT_DATE - INTERVAL '120 days'
        GROUP BY area_name
        ORDER BY recent_quantity DESC NULLS LAST, area_name
    """)
    try:
        df = pd.read_sql(
            query,
            engine,
            params={"zone_filter": zone_filter, "split_dimension": split_dimension},
        )
        areas = df["area_name"].dropna().astype(str).tolist()
        return areas or _default_area_candidates(zone_filter, split_dimension)
    except Exception:
        return _default_area_candidates(zone_filter, split_dimension)


def fetch_all_area_options(split_dimension):
    """Return the current per-region 120-day daily ranking in one query."""

    if split_dimension not in {
        option["value"] for option in KPLER_FLEET_SPLIT_OPTIONS
    }:
        split_dimension = "current_subcontinents"

    fallback = {
        zone_filter: list(
            _default_area_candidates(zone_filter, split_dimension)
        )
        for zone_filter in KPLER_FLEET_REGION_ORDER
    }
    if not _fleet_metrics_table_exists():
        return fallback

    query = text(f"""
        SELECT
            zone_filter,
            area_name,
            SUM(quantity_mtonnes) AS recent_quantity
        FROM {_table_ref(KPLER_FLEET_METRICS_TABLE)}
        WHERE period = 'daily'
          AND zone_filter IN :zone_filters
          AND split_dimension = :split_dimension
          AND metric IN ('loaded_vessels', 'floating_storage')
          AND date >= CURRENT_DATE - INTERVAL '120 days'
        GROUP BY zone_filter, area_name
        ORDER BY
            zone_filter,
            recent_quantity DESC NULLS LAST,
            area_name
    """).bindparams(bindparam("zone_filters", expanding=True))
    try:
        frame = pd.read_sql(
            query,
            engine,
            params={
                "zone_filters": tuple(KPLER_FLEET_REGION_ORDER),
                "split_dimension": split_dimension,
            },
        )
        if not {"zone_filter", "area_name"}.issubset(frame.columns):
            raise ValueError("Fleet area options query returned invalid columns")
        options = {}
        for zone_filter in KPLER_FLEET_REGION_ORDER:
            zone_areas = (
                frame.loc[
                    frame["zone_filter"] == zone_filter,
                    "area_name",
                ]
                .dropna()
                .astype(str)
                .tolist()
            )
            options[zone_filter] = zone_areas or fallback[zone_filter]
        return options
    except Exception:
        logger.warning(
            "Fleet area options query failed; using configured defaults",
            exc_info=True,
        )
        return fallback


def fetch_fleet_metrics_weekly(
    split_dimension,
    start_date,
    end_date,
    zone_filter=None,
    zone_filters=None,
    area_names=None,
    metrics=("loaded_vessels", "floating_storage"),
):
    if not _fleet_metrics_table_exists():
        return pd.DataFrame()

    filters = [
        "period = 'daily'",
        "split_dimension = :split_dimension",
        "metric IN :metrics",
        "date BETWEEN :start_date AND :end_date",
    ]
    params = {
        "split_dimension": split_dimension,
        "metrics": tuple(metrics),
        "start_date": start_date,
        "end_date": end_date,
    }

    if zone_filter:
        filters.append("zone_filter = :zone_filter")
        params["zone_filter"] = zone_filter
    elif zone_filters is not None:
        zone_filters = tuple(zone_filters)
        if not zone_filters:
            return pd.DataFrame()
        filters.append("zone_filter IN :zone_filters")
        params["zone_filters"] = zone_filters
    if area_names:
        filters.append("area_name IN :area_names")
        params["area_names"] = tuple(area_names)

    where_clause = "\n              AND ".join(filters)
    bind_params = [bindparam("metrics", expanding=True)]
    if zone_filters is not None and not zone_filter:
        bind_params.append(bindparam("zone_filters", expanding=True))
    if area_names:
        bind_params.append(bindparam("area_names", expanding=True))

    query = text(f"""
        WITH daily AS (
            SELECT
                zone_filter,
                metric,
                date,
                DATE_TRUNC('week', date)::date AS week_start,
                area_name,
                quantity_mtonnes,
                upload_timestamp_utc
            FROM {_table_ref(KPLER_FLEET_METRICS_TABLE)}
            WHERE {where_clause}
        ),
        week_counts AS (
            SELECT
                zone_filter,
                metric,
                area_name,
                week_start,
                COUNT(DISTINCT date) AS day_count
            FROM daily
            GROUP BY zone_filter, metric, area_name, week_start
        ),
        ranked AS (
            SELECT
                daily.*,
                week_counts.day_count,
                ROW_NUMBER() OVER (
                    PARTITION BY daily.zone_filter, daily.metric, daily.area_name, daily.week_start
                    ORDER BY daily.date DESC
                ) AS row_number
            FROM daily
            JOIN week_counts
              ON daily.zone_filter = week_counts.zone_filter
             AND daily.metric = week_counts.metric
             AND daily.area_name = week_counts.area_name
             AND daily.week_start = week_counts.week_start
        )
        SELECT
            zone_filter,
            metric,
            week_start AS date,
            area_name,
            quantity_mtonnes,
            upload_timestamp_utc
        FROM ranked
        WHERE day_count = 7
          AND row_number = 1
        ORDER BY zone_filter, metric, area_name, date
    """).bindparams(*bind_params)

    return pd.read_sql(query, engine, params=params)


def fetch_region_metric_totals_daily(start_date, end_date):
    if not _fleet_metrics_table_exists():
        return pd.DataFrame()

    query = text(f"""
        SELECT
            zone_filter,
            metric,
            date,
            SUM(quantity_mtonnes) AS quantity_mtonnes,
            MAX(upload_timestamp_utc) AS upload_timestamp_utc
        FROM {_table_ref(KPLER_FLEET_METRICS_TABLE)}
        WHERE period = 'daily'
          AND split_dimension = 'current_subcontinents'
          AND metric IN ('loaded_vessels', 'floating_storage')
          AND date BETWEEN :start_date AND :end_date
        GROUP BY zone_filter, metric, date
        ORDER BY zone_filter, metric, date
    """)
    return pd.read_sql(
        query,
        engine,
        params={"start_date": start_date, "end_date": end_date},
    )


def fetch_latest_upload_timestamp():
    if not _fleet_metrics_table_exists():
        return None
    try:
        query = text(f"""
            SELECT MAX(upload_timestamp_utc) AS upload_timestamp_utc
            FROM {_table_ref(KPLER_FLEET_METRICS_TABLE)}
        """)
        value = pd.read_sql(query, engine).iloc[0]["upload_timestamp_utc"]
        return value if pd.notna(value) else None
    except Exception:
        return None


def fetch_latest_signal_upload_timestamp():
    if not _regional_signal_table_exists():
        return None
    try:
        query = text(f"""
            SELECT MAX(upload_timestamp_utc) AS upload_timestamp_utc
            FROM {_table_ref(KPLER_REGIONAL_SIGNAL_TABLE)}
        """)
        value = pd.read_sql(query, engine).iloc[0]["upload_timestamp_utc"]
        return value if pd.notna(value) else None
    except Exception:
        return None


def fetch_regional_signals(start_date, end_date):
    if not _regional_signal_table_exists():
        return pd.DataFrame()

    today = dt.date.today()
    signal_end_date = max(end_date, today + dt.timedelta(days=45))
    query = text(f"""
        SELECT
            endpoint,
            metric,
            date,
            zone_filter,
            area_name,
            value
        FROM {_table_ref(KPLER_REGIONAL_SIGNAL_TABLE)}
        WHERE date BETWEEN :start_date AND :end_date
          AND (
              (
                  zone_filter IN :zone_filters
                  AND endpoint = 'flows'
                  AND metric = 'inbound_tonnes'
              )
              OR (
                  zone_filter IN :zone_filters
                  AND endpoint = 'fleet_utilization'
                  AND metric = 'vessel_count'
              )
              OR (
                  zone_filter IN :zone_filters
                  AND endpoint = 'congestion'
                  AND metric IN ('waiting_count', 'waiting_duration_days')
              )
              OR (
                  zone_filter = 'global'
                  AND endpoint = 'freight_metrics'
                  AND metric IN ('loaded_ton_miles', 'avg_loaded_distance', 'avg_loaded_speed')
              )
          )
        ORDER BY endpoint, metric, zone_filter, area_name, date
    """).bindparams(bindparam("zone_filters", expanding=True))

    return pd.read_sql(
        query,
        engine,
        params={
            "start_date": start_date,
            "end_date": signal_end_date,
            "zone_filters": tuple(KPLER_FLEET_REGION_ORDER),
        },
    )


def fetch_recent_diversions(start_date=None, end_date=None):
    if not _diversions_table_exists():
        return pd.DataFrame()

    today = dt.date.today()
    start_date = start_date or today - dt.timedelta(days=45)
    end_date = end_date or today + dt.timedelta(days=60)
    query = text(f"""
        WITH latest_upload AS (
            SELECT MAX(upload_timestamp_utc) AS upload_timestamp_utc
            FROM {_table_ref(KPLER_DIVERSIONS_TABLE)}
        )
        SELECT
            diversion_date::date AS diversion_date,
            diverted_from_country_name,
            diverted_from_subcontinent_name,
            diverted_from_zone_name,
            diverted_from_date::date AS diverted_from_date,
            new_destination_country_name,
            new_destination_subcontinent_name,
            new_destination_zone_name,
            new_destination_date::date AS new_destination_date
        FROM {_table_ref(KPLER_DIVERSIONS_TABLE)}
        JOIN latest_upload USING (upload_timestamp_utc)
        WHERE COALESCE(new_destination_date::date, diversion_date::date) BETWEEN :start_date AND :end_date
           OR COALESCE(diverted_from_date::date, diversion_date::date) BETWEEN :start_date AND :end_date
        ORDER BY COALESCE(new_destination_date::date, diversion_date::date) DESC NULLS LAST
    """)
    try:
        return pd.read_sql(
            query,
            engine,
            params={"start_date": start_date, "end_date": end_date},
        )
    except Exception:
        return pd.DataFrame()


def _filter_diversions_by_event_window(diversions_df, start_date, end_date):
    if diversions_df.empty:
        return diversions_df.copy()

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()
    working = diversions_df.copy()

    def _date_series(column_name):
        if column_name not in working.columns:
            return pd.Series(pd.NaT, index=working.index)
        return pd.to_datetime(working[column_name], errors="coerce").dt.normalize()

    diversion_dates = _date_series("diversion_date")
    inbound_dates = _date_series("new_destination_date").fillna(diversion_dates)
    outbound_dates = _date_series("diverted_from_date").fillna(diversion_dates)
    mask = (
        inbound_dates.between(start_ts, end_ts, inclusive="both")
        | outbound_dates.between(start_ts, end_ts, inclusive="both")
    )
    return working[mask].copy()


def derive_weekly_fleet_metrics(daily_df, include_incomplete_weeks=False):
    if daily_df.empty:
        return daily_df

    weekly = daily_df.copy()
    weekly["date"] = pd.to_datetime(weekly["date"])
    weekly["week_start"] = weekly["date"] - pd.to_timedelta(weekly["date"].dt.weekday, unit="D")
    group_columns = [
        column
        for column in ["zone_filter", "metric", "area_name", "week_start"]
        if column in weekly.columns
    ]

    if not include_incomplete_weeks:
        counts = weekly.groupby(group_columns)["date"].transform("nunique")
        weekly = weekly[counts == 7]
        if weekly.empty:
            return weekly[[column for column in [
                "zone_filter",
                "metric",
                "date",
                "area_name",
                "quantity_mtonnes",
                "upload_timestamp_utc",
            ] if column in weekly.columns]]

    weekly = weekly.sort_values(group_columns + ["date"], kind="stable")
    weekly = weekly.groupby(group_columns, as_index=False, sort=False).tail(1)
    weekly["date"] = weekly["week_start"]
    return weekly[[column for column in [
        "zone_filter",
        "metric",
        "date",
        "area_name",
        "quantity_mtonnes",
        "upload_timestamp_utc",
    ] if column in weekly.columns]].reset_index(drop=True)


def _weekly_metric_totals(weekly_df):
    if weekly_df.empty:
        return pd.DataFrame()

    totals = (
        weekly_df.groupby(["zone_filter", "date", "metric"], as_index=False)["quantity_mtonnes"]
        .sum()
    )
    return (
        totals.pivot_table(
            index=["zone_filter", "date"],
            columns="metric",
            values="quantity_mtonnes",
            aggfunc="sum",
        )
        .reset_index()
        .sort_values(["zone_filter", "date"])
    )


def _weekly_metric_totals_long(weekly_df):
    if weekly_df.empty:
        return pd.DataFrame()

    return (
        weekly_df.groupby(["zone_filter", "date", "metric"], as_index=False)["quantity_mtonnes"]
        .sum()
        .sort_values(["zone_filter", "metric", "date"])
    )


def _series_delta(series, periods):
    if len(series) <= periods:
        return None
    return float(series.iloc[-1] - series.iloc[-1 - periods])


def _percentile_of_latest(series):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    latest = clean.iloc[-1]
    return float((clean <= latest).mean() * 100.0)


def compute_region_summaries(summary_weekly, end_date):
    totals = _weekly_metric_totals(summary_weekly)
    summaries = {}

    for zone_filter in KPLER_FLEET_REGION_ORDER:
        zone_totals = totals[totals["zone_filter"] == zone_filter].copy() if not totals.empty else pd.DataFrame()
        if zone_totals.empty:
            summaries[zone_filter] = {
                "zone_filter": zone_filter,
                "region": _region_label(zone_filter),
                "status": "No data",
            }
            continue

        zone_totals = zone_totals.sort_values("date")
        loaded = pd.to_numeric(zone_totals.get("loaded_vessels"), errors="coerce")
        floating = pd.to_numeric(zone_totals.get("floating_storage"), errors="coerce")
        latest_loaded = float(loaded.iloc[-1]) if not loaded.empty and pd.notna(loaded.iloc[-1]) else None
        latest_floating = float(floating.iloc[-1]) if not floating.empty and pd.notna(floating.iloc[-1]) else None
        waiting_share = (
            latest_floating / latest_loaded * 100.0
            if latest_loaded and latest_floating is not None
            else None
        )
        latest_week = pd.to_datetime(zone_totals["date"].iloc[-1]).date()
        summaries[zone_filter] = {
            "zone_filter": zone_filter,
            "region": _region_label(zone_filter),
            "latest_week": latest_week,
            "loaded_mt": latest_loaded,
            "floating_mt": latest_floating,
            "waiting_share": waiting_share,
            "loaded_wow": _series_delta(loaded, 1),
            "floating_wow": _series_delta(floating, 1),
            "loaded_4w": _series_delta(loaded, 4),
            "floating_4w": _series_delta(floating, 4),
            "loaded_percentile": _percentile_of_latest(loaded),
            "floating_percentile": _percentile_of_latest(floating),
            "status": _freshness_status(latest_week, end_date),
        }

    return summaries


def _clean_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _matches_region_values(zone_filter, subcontinent=None, zone=None, country=None):
    if zone_filter == "global":
        return True

    members = KPLER_FLEET_ZONE_MEMBERS.get(zone_filter, set())
    countries = KPLER_REGION_COUNTRIES.get(zone_filter, set())
    values = {_clean_text(subcontinent), _clean_text(zone), _clean_text(country)}
    values.discard("")
    return bool(values & members) or bool(values & countries)


def _filter_freight_for_region(signals_df, zone_filter):
    if signals_df.empty:
        return pd.DataFrame()

    freight_df = signals_df[
        (signals_df["endpoint"] == "freight_metrics")
        & (signals_df["zone_filter"] == "global")
    ].copy()
    if freight_df.empty or zone_filter == "global":
        return freight_df

    country_set = KPLER_REGION_COUNTRIES.get(zone_filter, set())
    if not country_set:
        return freight_df.iloc[0:0].copy()

    return freight_df[freight_df["area_name"].isin(country_set)].copy()


def _latest_signal_value(signals_df, zone_filter, endpoint, metric, area_name=None, max_date=None):
    if signals_df.empty:
        return None, None

    df = signals_df[
        (signals_df["zone_filter"] == zone_filter)
        & (signals_df["endpoint"] == endpoint)
        & (signals_df["metric"] == metric)
    ].copy()
    if area_name is not None:
        df = df[df["area_name"] == area_name]
    if max_date is not None:
        df = df[pd.to_datetime(df["date"]).dt.date <= max_date]
    if df.empty:
        return None, None

    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    latest_value = pd.to_numeric(df.loc[df["date"] == latest_date, "value"], errors="coerce").sum()
    return float(latest_value), latest_date.date()


def _fleet_utilization_latest(signals_df, zone_filter, end_date):
    if signals_df.empty:
        return {}

    df = signals_df[
        (signals_df["zone_filter"] == zone_filter)
        & (signals_df["endpoint"] == "fleet_utilization")
        & (signals_df["metric"] == "vessel_count")
    ].copy()
    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.date <= end_date]
    if df.empty:
        return {}

    latest_date = df["date"].max()
    latest = (
        df[df["date"] == latest_date]
        .groupby("area_name", as_index=False)["value"]
        .sum()
    )
    values = {
        str(row["area_name"]): float(row["value"])
        for _, row in latest.iterrows()
    }
    total = sum(values.values())
    loaded = values.get("Loaded", 0.0)
    ballast = values.get("Ballast", 0.0)
    maintenance = values.get("Maintenance", 0.0)
    return {
        "date": latest_date.date(),
        "loaded_count": loaded,
        "ballast_count": ballast,
        "maintenance_count": maintenance,
        "total_count": total,
        "loaded_share": loaded / total * 100.0 if total else None,
    }


def _freight_latest(signals_df, zone_filter, end_date):
    freight_df = _filter_freight_for_region(signals_df, zone_filter)
    if freight_df.empty:
        return {}

    freight_df["date"] = pd.to_datetime(freight_df["date"])
    freight_df = freight_df[freight_df["date"].dt.date <= end_date]
    if freight_df.empty:
        return {}

    latest_date = freight_df["date"].max()
    latest = freight_df[freight_df["date"] == latest_date].copy()
    ton_miles = pd.to_numeric(
        latest.loc[latest["metric"] == "loaded_ton_miles", "value"],
        errors="coerce",
    ).sum()
    avg_distance = pd.to_numeric(
        latest.loc[latest["metric"] == "avg_loaded_distance", "value"],
        errors="coerce",
    )
    avg_distance = avg_distance[avg_distance > 0]
    avg_speed = pd.to_numeric(
        latest.loc[latest["metric"] == "avg_loaded_speed", "value"],
        errors="coerce",
    )
    avg_speed = avg_speed[avg_speed > 0]
    return {
        "date": latest_date.date(),
        "ton_miles_bn": float(ton_miles) / 1_000_000_000.0 if pd.notna(ton_miles) else None,
        "avg_distance": float(avg_distance.mean()) if not avg_distance.empty else None,
        "avg_speed": float(avg_speed.mean()) if not avg_speed.empty else None,
    }


def _diversion_direction_counts(diversions_df, zone_filter):
    if diversions_df.empty:
        return {"into": 0, "out": 0}

    into_count = 0
    out_count = 0
    for _, row in diversions_df.iterrows():
        into_region = _matches_region_values(
            zone_filter,
            subcontinent=row.get("new_destination_subcontinent_name"),
            zone=row.get("new_destination_zone_name"),
            country=row.get("new_destination_country_name"),
        )
        out_region = _matches_region_values(
            zone_filter,
            subcontinent=row.get("diverted_from_subcontinent_name"),
            zone=row.get("diverted_from_zone_name"),
            country=row.get("diverted_from_country_name"),
        )
        if into_region and not out_region:
            into_count += 1
        elif out_region and not into_region:
            out_count += 1
    return {"into": into_count, "out": out_count}


def compute_global_signal_summaries(signals_df, diversions_df, end_date):
    summaries = {}
    today = dt.date.today()
    if not signals_df.empty:
        signals_df = signals_df.copy()
        signals_df["date"] = pd.to_datetime(signals_df["date"])
        signals_df["value"] = pd.to_numeric(signals_df["value"], errors="coerce").fillna(0.0)

    for zone_filter in KPLER_FLEET_REGION_ORDER:
        flows = signals_df[
            (signals_df["zone_filter"] == zone_filter)
            & (signals_df["endpoint"] == "flows")
            & (signals_df["metric"] == "inbound_tonnes")
        ].copy() if not signals_df.empty else pd.DataFrame()
        if not flows.empty:
            flow_dates = pd.to_datetime(flows["date"]).dt.date
            inbound_14d_mt = flows.loc[
                (flow_dates >= today) & (flow_dates <= today + dt.timedelta(days=14)),
                "value",
            ].sum() / 1_000_000.0
            inbound_30d_mt = flows.loc[
                (flow_dates >= today) & (flow_dates <= today + dt.timedelta(days=30)),
                "value",
            ].sum() / 1_000_000.0
            max_flow_date = flow_dates.max()
        else:
            inbound_14d_mt = None
            inbound_30d_mt = None
            max_flow_date = None

        utilization = _fleet_utilization_latest(signals_df, zone_filter, end_date)
        congestion_count, congestion_date = _latest_signal_value(
            signals_df,
            zone_filter,
            "congestion",
            "waiting_count",
            "Total",
            max_date=end_date,
        )
        congestion_duration, _ = _latest_signal_value(
            signals_df,
            zone_filter,
            "congestion",
            "waiting_duration_days",
            "Total",
            max_date=end_date,
        )
        freight = _freight_latest(signals_df, zone_filter, end_date)
        diversion_counts = _diversion_direction_counts(diversions_df, zone_filter)

        latest_dates = [
            value
            for value in [
                max_flow_date,
                utilization.get("date"),
                congestion_date,
                freight.get("date"),
            ]
            if value is not None and pd.notna(value)
        ]
        latest_signal_date = max(latest_dates) if latest_dates else None
        status = "No data"
        if latest_signal_date is not None:
            status = "Stale" if latest_signal_date < end_date - dt.timedelta(days=31) else "OK"

        summaries[zone_filter] = {
            "zone_filter": zone_filter,
            "region": _region_label(zone_filter),
            "inbound_14d_mt": float(inbound_14d_mt) if inbound_14d_mt is not None else None,
            "inbound_30d_mt": float(inbound_30d_mt) if inbound_30d_mt is not None else None,
            "loaded_share": utilization.get("loaded_share"),
            "loaded_count": utilization.get("loaded_count"),
            "ballast_count": utilization.get("ballast_count"),
            "maintenance_count": utilization.get("maintenance_count"),
            "congestion_count": congestion_count,
            "congestion_duration": congestion_duration,
            "diversions_in": diversion_counts["into"],
            "diversions_out": diversion_counts["out"],
            "freight_ton_miles_bn": freight.get("ton_miles_bn"),
            "freight_avg_distance": freight.get("avg_distance"),
            "freight_avg_speed": freight.get("avg_speed"),
            "latest_signal_date": latest_signal_date,
            "status": status,
        }

    return summaries


def fetch_price_context():
    if not _relation_exists(PRICE_CURVE_TABLE):
        return {"status": "No curve table"}

    try:
        query = text(f"""
            WITH latest_cob AS (
                SELECT MAX(cob) AS cob
                FROM {_table_ref(PRICE_CURVE_TABLE)}
                WHERE code IN ('ICE_JKM_MO', 'ICE_TFU_MO')
            ),
            ranked AS (
                SELECT
                    curve.code,
                    curve.cob,
                    curve.contract,
                    curve.expiry,
                    curve.value,
                    curve.currency,
                    curve.units,
                    ROW_NUMBER() OVER (
                        PARTITION BY curve.code
                        ORDER BY curve.expiry
                    ) AS row_number
                FROM {_table_ref(PRICE_CURVE_TABLE)} AS curve
                JOIN latest_cob ON curve.cob = latest_cob.cob
                WHERE curve.code IN ('ICE_JKM_MO', 'ICE_TFU_MO')
                  AND curve.expiry >= curve.cob
            )
            SELECT code, cob, contract, value, currency, units
            FROM ranked
            WHERE row_number = 1
            ORDER BY code
        """)
        df = pd.read_sql(query, engine)
    except Exception as exc:
        return {"status": f"Price error: {exc}"}

    if df.empty:
        return {"status": "No price rows"}

    latest_cob = pd.to_datetime(df["cob"].max()).date()
    today = dt.date.today()
    is_fresh = latest_cob >= today - dt.timedelta(days=PRICE_FRESHNESS_DAYS)
    values = {row["code"]: row for _, row in df.iterrows()}
    jkm = values.get("ICE_JKM_MO")
    ttf = values.get("ICE_TFU_MO")
    spread = None
    if jkm is not None and ttf is not None:
        spread = float(jkm["value"]) - float(ttf["value"])

    return {
        "status": "Fresh" if is_fresh else "Stale",
        "cob": latest_cob,
        "jkm": float(jkm["value"]) if jkm is not None else None,
        "ttf": float(ttf["value"]) if ttf is not None else None,
        "spread": spread,
        "contract": jkm["contract"] if jkm is not None else None,
        "currency": jkm["currency"] if jkm is not None else None,
        "units": jkm["units"] if jkm is not None else None,
    }


def _build_weekly_5y_envelope(metric_df, current_year, value_column="quantity_mtonnes"):
    historical_years = list(range(current_year - 5, current_year))
    historical_df = metric_df[metric_df["year"].isin(historical_years)].copy()
    if historical_df.empty:
        return pd.DataFrame(), None

    envelope = (
        historical_df.groupby("week", as_index=False)[value_column]
        .agg(min_value="min", max_value="max", avg_value="mean")
        .sort_values("week")
    )
    return envelope, f"{current_year - 5}-{current_year - 1}"


def _add_weekly_5y_envelope(
    fig,
    envelope,
    years_label,
    row,
    col,
    show_range_legend,
    show_avg_legend,
    unit_label="mt",
    value_format=".2f",
    average_format=None,
):
    if envelope is None or envelope.empty or not years_label:
        return
    average_format = average_format or value_format

    fig.add_trace(
        go.Scatter(
            x=envelope["week"],
            y=envelope["min_value"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            name=f"{years_label} min",
            legendgroup=f"{years_label} range",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=envelope["week"],
            y=envelope["max_value"],
            customdata=envelope["min_value"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.13)",
            showlegend=show_range_legend,
            hovertemplate=(
                "<b>5Y range</b><br>"
                "Week %{x}<br>"
                f"Min: %{{customdata:{value_format}}} {unit_label}<br>"
                f"Max: %{{y:{value_format}}} {unit_label}<br>"
                f"Years: {years_label}<extra></extra>"
            ),
            name="5Y range",
            legendgroup="5Y range",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=envelope["week"],
            y=envelope["avg_value"],
            mode="lines",
            line=dict(color="rgba(71, 85, 105, 0.92)", width=1.8, dash="dot"),
            hovertemplate=(
                "<b>5Y average</b><br>"
                "Week %{x}<br>"
                f"Avg: %{{y:{average_format}}} {unit_label}<br>"
                f"Years: {years_label}<extra></extra>"
            ),
            name="5Y avg",
            legendgroup="5Y avg",
            showlegend=show_avg_legend,
        ),
        row=row,
        col=col,
    )


def _seasonal_year_trace_style(year, current_year):
    if year == current_year:
        return "#dc2626", 3.0, 1.0, "solid", True
    if year == current_year - 1:
        return "#16a34a", 2.6, 0.98, "solid", True
    return "#94a3b8", 1.4, 0.62, "dot", False


def _seasonal_y_axis_range(metric_df, metric, tick_step):
    max_value = pd.to_numeric(metric_df["quantity_mtonnes"], errors="coerce").max()
    if pd.isna(max_value) or max_value <= 0:
        max_value = tick_step
    y_max = math.ceil((float(max_value) * 1.04) / tick_step) * tick_step
    if metric == "floating_storage":
        y_max = round(y_max, 1)
    else:
        y_max = int(y_max)
    return [0, y_max]


def _seasonal_y_axis_settings(metric_df, metric):
    tick_step = 2.0 if metric == "loaded_vessels" else 0.2
    global_df = metric_df[metric_df["zone_filter"] == "global"]
    regional_df = metric_df[metric_df["zone_filter"] != "global"]
    global_range = _seasonal_y_axis_range(global_df if not global_df.empty else metric_df, metric, tick_step)
    if metric == "loaded_vessels":
        global_range[0] = 12
    regional_range = _seasonal_y_axis_range(
        regional_df if not regional_df.empty else metric_df,
        metric,
        tick_step,
    )
    return tick_step, global_range, regional_range


def build_region_seasonal_chart(summary_weekly, metric):
    totals = _weekly_metric_totals_long(summary_weekly)
    metric_df = totals[totals["metric"] == metric].copy() if not totals.empty else pd.DataFrame()
    title = "Loaded on water" if metric == "loaded_vessels" else "Floating storage"

    if metric_df.empty:
        return _empty_figure(f"No seasonal {title.lower()} data loaded", height=SEASONAL_CHART_HEIGHT)

    metric_df["date"] = pd.to_datetime(metric_df["date"])
    iso_calendar = metric_df["date"].dt.isocalendar()
    metric_df["year"] = iso_calendar.year.astype(int)
    metric_df["week"] = iso_calendar.week.astype(int)
    metric_df = metric_df[metric_df["week"].between(1, 52)].copy()
    years = sorted(metric_df["year"].dropna().unique().tolist())
    if not years:
        return _empty_figure(f"No seasonal {title.lower()} data loaded", height=SEASONAL_CHART_HEIGHT)

    current_year = pd.Timestamp.today().isocalendar().year
    y_tick_step, global_y_axis_range, regional_y_axis_range = _seasonal_y_axis_settings(metric_df, metric)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[f"<b>{_region_label(zone_filter)}</b>" for zone_filter in KPLER_FLEET_SEASONAL_REGION_ORDER],
        horizontal_spacing=0.045,
        vertical_spacing=0.14,
    )

    legend_seen = set()
    for idx, zone_filter in enumerate(KPLER_FLEET_SEASONAL_REGION_ORDER):
        row = idx // 3 + 1
        col = idx % 3 + 1
        zone_df = metric_df[metric_df["zone_filter"] == zone_filter].copy()
        if zone_df.empty:
            continue

        envelope, years_label = _build_weekly_5y_envelope(zone_df, current_year)
        _add_weekly_5y_envelope(
            fig,
            envelope,
            years_label,
            row=row,
            col=col,
            show_range_legend="5y_range" not in legend_seen,
            show_avg_legend="5y_avg" not in legend_seen,
        )
        legend_seen.add("5y_range")
        legend_seen.add("5y_avg")

        displayed_years = [current_year, current_year - 1]
        legend_only_years = [
            year
            for year in sorted(years, reverse=True)
            if year not in set(displayed_years)
        ]
        for year in displayed_years + legend_only_years:
            trace_style = _seasonal_year_trace_style(year, current_year)
            if trace_style is None:
                continue
            year_df = zone_df[zone_df["year"] == year].sort_values("week")
            if year_df.empty:
                continue
            show_legend = year not in legend_seen
            legend_seen.add(year)
            color, width, opacity, dash, visible_by_default = trace_style
            fig.add_trace(
                go.Scatter(
                    x=year_df["week"],
                    y=year_df["quantity_mtonnes"],
                    mode="lines",
                    name=str(year),
                    legendgroup=str(year),
                    showlegend=show_legend,
                    visible=True if visible_by_default else "legendonly",
                    line=dict(
                        color=color,
                        width=width,
                        dash=dash,
                    ),
                    opacity=opacity,
                    hovertemplate=(
                        f"{_region_label(zone_filter)}<br>"
                        f"Year {year}<br>"
                        "Week %{x}<br>"
                        "%{y:.2f} mt<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b> seasonality "
                "<span style='font-size:11px;color:#64748b'>(weekly mt)</span>"
            ),
            x=0.0,
            xanchor="left",
            y=0.98,
            font=dict(size=15, color="#0f172a"),
        ),
        template="plotly_white",
        height=SEASONAL_CHART_HEIGHT,
        margin=dict(l=38, r=14, t=58, b=30),
        plot_bgcolor="rgba(248, 250, 252, 0.9)",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.13,
            xanchor="right",
            x=1.0,
            font=dict(size=9, color="#334155"),
            groupclick="togglegroup",
            itemsizing="constant",
            itemwidth=30,
        ),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.96)",
            bordercolor="rgba(148, 163, 184, 0.65)",
            font=dict(size=11, color="#0f172a"),
        ),
    )
    fig.update_xaxes(
        title=None,
        range=[1, 52],
        tickmode="array",
        tickvals=SEASONAL_MONTH_TICKVALS,
        ticktext=SEASONAL_MONTH_TICKTEXT,
        tickangle=-40,
        gridcolor="rgba(148, 163, 184, 0.14)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(size=8, color="#64748b"),
        fixedrange=True,
    )
    fig.update_yaxes(
        title=None,
        dtick=y_tick_step,
        tickformat=".1f" if metric == "floating_storage" else ",.0f",
        gridcolor="rgba(148, 163, 184, 0.20)",
        zerolinecolor="rgba(148, 163, 184, 0.34)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(size=9, color="#64748b"),
        fixedrange=True,
    )
    for idx, zone_filter in enumerate(KPLER_FLEET_SEASONAL_REGION_ORDER):
        row = idx // 3 + 1
        col = idx % 3 + 1
        fig.update_yaxes(
            range=global_y_axis_range if zone_filter == "global" else regional_y_axis_range,
            row=row,
            col=col,
        )
    for row in (1, 2):
        fig.update_yaxes(title={"text": "mt", "font": dict(size=10, color="#475569")}, row=row, col=1)
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=10, color="#334155")
    return fig


def _first_valid_date(*values):
    for value in values:
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.notna(timestamp):
            return timestamp.normalize()
    return pd.NaT


def _build_diversion_events(diversions_df, start_date=None, end_date=None):
    columns = ["zone_filter", "direction", "date", "diversion_count"]
    if diversions_df.empty:
        return pd.DataFrame(columns=columns)

    start_ts = pd.to_datetime(start_date).normalize() if start_date else None
    end_ts = pd.to_datetime(end_date).normalize() if end_date else None
    rows = []
    for _, row in diversions_df.iterrows():
        for zone_filter in KPLER_FLEET_DIVERSION_REGION_ORDER:
            into_region = _matches_region_values(
                zone_filter,
                subcontinent=row.get("new_destination_subcontinent_name"),
                zone=row.get("new_destination_zone_name"),
                country=row.get("new_destination_country_name"),
            )
            out_region = _matches_region_values(
                zone_filter,
                subcontinent=row.get("diverted_from_subcontinent_name"),
                zone=row.get("diverted_from_zone_name"),
                country=row.get("diverted_from_country_name"),
            )
            if into_region and not out_region:
                event_date = _first_valid_date(row.get("new_destination_date"), row.get("diversion_date"))
                direction = "in"
            elif out_region and not into_region:
                event_date = _first_valid_date(row.get("diverted_from_date"), row.get("diversion_date"))
                direction = "out"
            else:
                continue
            if pd.isna(event_date):
                continue
            if start_ts is not None and event_date < start_ts:
                continue
            if end_ts is not None and event_date > end_ts:
                continue
            rows.append(
                {
                    "zone_filter": zone_filter,
                    "direction": direction,
                    "date": event_date,
                    "diversion_count": 1,
                }
            )

    return pd.DataFrame(rows, columns=columns)


def _build_diversion_weekly_counts(diversions_df, start_date=None, end_date=None):
    events = _build_diversion_events(diversions_df, start_date=start_date, end_date=end_date)
    if events.empty:
        return events

    iso_calendar = events["date"].dt.isocalendar()
    events["year"] = iso_calendar.year.astype(int)
    events["week"] = iso_calendar.week.astype(int)
    events = events[events["week"].between(1, 52)].copy()
    if events.empty:
        return events

    weekly = (
        events.groupby(["zone_filter", "direction", "year", "week"], as_index=False)["diversion_count"]
        .sum()
    )
    current_year = pd.Timestamp.today().isocalendar().year
    default_years = set(range(current_year - 5, current_year + 1))
    years = set(weekly["year"].dropna().astype(int).tolist()) | default_years
    if start_date:
        start_iso_year = pd.to_datetime(start_date).isocalendar().year
        years = {year for year in years if year >= start_iso_year}
    if end_date:
        end_iso = pd.to_datetime(end_date).isocalendar()
        years = {year for year in years if year <= end_iso.year}
    else:
        end_iso = pd.Timestamp.today().isocalendar()

    index_rows = []
    for zone_filter in KPLER_FLEET_DIVERSION_REGION_ORDER:
        for direction in ("in", "out"):
            for year in sorted(years):
                max_week = 52
                if year == end_iso.year:
                    max_week = min(52, int(end_iso.week))
                for week in range(1, max_week + 1):
                    index_rows.append(
                        {
                            "zone_filter": zone_filter,
                            "direction": direction,
                            "year": int(year),
                            "week": int(week),
                        }
                    )
    full_index = pd.DataFrame(index_rows)
    weekly = full_index.merge(
        weekly,
        on=["zone_filter", "direction", "year", "week"],
        how="left",
    )
    weekly["diversion_count"] = pd.to_numeric(weekly["diversion_count"], errors="coerce").fillna(0)
    return weekly


def _diversion_y_axis_settings(weekly_df):
    max_count = pd.to_numeric(weekly_df.get("diversion_count"), errors="coerce").max()
    if pd.isna(max_count) or max_count <= 0:
        max_count = 1
    if max_count <= 8:
        tick_step = 1
    elif max_count <= 20:
        tick_step = 2
    else:
        tick_step = 5
    y_max = int(math.ceil((float(max_count) * 1.10) / tick_step) * tick_step)
    y_max = max(tick_step, y_max)
    return tick_step, [-y_max, y_max]


def _signed_diversion_direction_df(zone_df, direction):
    direction_df = zone_df[zone_df["direction"] == direction].copy()
    if direction_df.empty:
        return direction_df
    direction_df["signed_count"] = pd.to_numeric(
        direction_df["diversion_count"],
        errors="coerce",
    ).fillna(0)
    if direction == "out":
        direction_df["signed_count"] = -direction_df["signed_count"]
    return direction_df


def _add_diversion_direction_envelope(
    fig,
    envelope,
    years_label,
    row,
    col,
    direction,
    show_range_legend,
    show_avg_legend,
):
    if envelope is None or envelope.empty or not years_label:
        return

    label = "In" if direction == "in" else "Out"
    fillcolor = "rgba(37, 99, 235, 0.12)" if direction == "in" else "rgba(249, 115, 22, 0.13)"
    avg_color = "rgba(37, 99, 235, 0.82)" if direction == "in" else "rgba(234, 88, 12, 0.84)"
    low_counts = envelope[["min_value", "max_value"]].abs().min(axis=1)
    high_counts = envelope[["min_value", "max_value"]].abs().max(axis=1)
    range_customdata = list(zip(low_counts, high_counts))

    fig.add_trace(
        go.Scatter(
            x=envelope["week"],
            y=envelope["min_value"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            name=f"{label} {years_label} min",
            legendgroup=f"{label} 5Y range",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=envelope["week"],
            y=envelope["max_value"],
            customdata=range_customdata,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=fillcolor,
            showlegend=show_range_legend,
            hovertemplate=(
                f"<b>{label} 5Y range</b><br>"
                "Week %{x}<br>"
                "Range: %{customdata[0]:.0f}-%{customdata[1]:.0f} diversions<br>"
                f"Years: {years_label}<extra></extra>"
            ),
            name=f"{label} 5Y range",
            legendgroup=f"{label} 5Y range",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=envelope["week"],
            y=envelope["avg_value"],
            customdata=envelope["avg_value"].abs(),
            mode="lines",
            line=dict(color=avg_color, width=1.8, dash="dot"),
            hovertemplate=(
                f"<b>{label} 5Y average</b><br>"
                "Week %{x}<br>"
                "Avg: %{customdata:.1f} diversions<br>"
                f"Years: {years_label}<extra></extra>"
            ),
            name=f"{label} 5Y avg",
            legendgroup=f"{label} 5Y avg",
            showlegend=show_avg_legend,
        ),
        row=row,
        col=col,
    )


def build_diversion_seasonal_chart(diversions_df, start_date=None, end_date=None):
    weekly = _build_diversion_weekly_counts(diversions_df, start_date=start_date, end_date=end_date)
    if weekly.empty:
        return _empty_figure("No diversion seasonality data loaded", height=DIVERSION_SEASONAL_CHART_HEIGHT)

    current_year = pd.Timestamp.today().isocalendar().year
    years = sorted(weekly["year"].dropna().unique().tolist())
    y_tick_step, y_axis_range = _diversion_y_axis_settings(weekly)

    fig = make_subplots(
        rows=1,
        cols=len(KPLER_FLEET_DIVERSION_REGION_ORDER),
        subplot_titles=[f"<b>{_region_label(zone_filter)}</b>" for zone_filter in KPLER_FLEET_DIVERSION_REGION_ORDER],
        horizontal_spacing=0.035,
    )

    legend_seen = set()
    for idx, zone_filter in enumerate(KPLER_FLEET_DIVERSION_REGION_ORDER):
        row = 1
        col = idx + 1
        zone_df = weekly[weekly["zone_filter"] == zone_filter].copy()
        if zone_df.empty:
            continue

        for direction in ("in", "out"):
            direction_zone_df = _signed_diversion_direction_df(zone_df, direction)
            if direction_zone_df.empty:
                continue
            envelope, years_label = _build_weekly_5y_envelope(
                direction_zone_df,
                current_year,
                value_column="signed_count",
            )
            _add_diversion_direction_envelope(
                fig,
                envelope,
                years_label,
                row=row,
                col=col,
                direction=direction,
                show_range_legend=f"{direction}_5y_range" not in legend_seen,
                show_avg_legend=f"{direction}_5y_avg" not in legend_seen,
            )
            legend_seen.add(f"{direction}_5y_range")
            legend_seen.add(f"{direction}_5y_avg")

        displayed_years = [current_year, current_year - 1]
        legend_only_years = [
            year
            for year in sorted(years, reverse=True)
            if year not in set(displayed_years)
        ]
        for direction in ("in", "out"):
            direction_zone_df = _signed_diversion_direction_df(zone_df, direction)
            if direction_zone_df.empty:
                continue
            direction_label = "in" if direction == "in" else "out"
            for year in displayed_years + legend_only_years:
                trace_style = _seasonal_year_trace_style(year, current_year)
                if trace_style is None:
                    continue
                year_df = direction_zone_df[direction_zone_df["year"] == year].sort_values("week")
                if year_df.empty:
                    continue
                show_legend = year not in legend_seen and direction == "in"
                legend_seen.add(year)
                color, width, opacity, dash, visible_by_default = trace_style
                fig.add_trace(
                    go.Scatter(
                        x=year_df["week"],
                        y=year_df["signed_count"],
                        customdata=year_df["diversion_count"],
                        mode="lines",
                        name=str(year),
                        legendgroup=str(year),
                        showlegend=show_legend,
                        visible=True if visible_by_default else "legendonly",
                        line=dict(
                            color=color,
                            width=width,
                            dash=dash if direction == "in" else "dash",
                        ),
                        opacity=opacity,
                        hovertemplate=(
                            f"{_region_label(zone_filter)}<br>"
                            f"Diversions {direction_label}<br>"
                            f"Year {year}<br>"
                            "Week %{x}<br>"
                            "%{customdata:.0f} diversions<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        template="plotly_white",
        height=DIVERSION_SEASONAL_CHART_HEIGHT,
        margin=dict(l=46, r=16, t=86, b=34),
        plot_bgcolor="rgba(248, 250, 252, 0.9)",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.20,
            xanchor="center",
            x=0.5,
            font=dict(size=9, color="#334155"),
            groupclick="togglegroup",
            itemsizing="constant",
            itemwidth=30,
        ),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.96)",
            bordercolor="rgba(148, 163, 184, 0.65)",
            font=dict(size=11, color="#0f172a"),
        ),
    )
    fig.update_xaxes(
        title=None,
        range=[1, 52],
        tickmode="array",
        tickvals=SEASONAL_MONTH_TICKVALS,
        ticktext=SEASONAL_MONTH_TICKTEXT,
        tickangle=-40,
        gridcolor="rgba(148, 163, 184, 0.14)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(size=8, color="#64748b"),
        fixedrange=True,
    )
    fig.update_yaxes(
        title=None,
        dtick=y_tick_step,
        tickformat=",.0f",
        range=y_axis_range,
        gridcolor="rgba(148, 163, 184, 0.20)",
        zerolinecolor="rgba(148, 163, 184, 0.34)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(size=9, color="#64748b"),
        fixedrange=True,
    )
    for col in range(1, len(KPLER_FLEET_DIVERSION_REGION_ORDER) + 1):
        fig.add_hline(
            y=0,
            line_width=1,
            line_color="rgba(71, 85, 105, 0.40)",
            row=1,
            col=col,
        )
        fig.update_yaxes(
            title={"text": "count", "font": dict(size=10, color="#475569")} if col == 1 else None,
            row=1,
            col=col,
        )
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=9, color="#334155")
    return fig


FLOATING_DAY_BUCKET_ORDER = ["5-10d", "11-20d", "21-30d", "31-60d", "60d+", "Unknown"]
FLOATING_DAY_BUCKET_COLORS = {
    "5-10d": "#7bc6ef",
    "11-20d": "#2f91d0",
    "21-30d": "#37a39c",
    "31-60d": "#f59e0b",
    "60d+": "#c2410c",
    "Unknown": "#687382",
}


def _floating_day_bucket(value):
    try:
        days = int(float(value))
    except (TypeError, ValueError):
        return "Unknown"
    if days <= 10:
        return "5-10d"
    if days <= 20:
        return "11-20d"
    if days <= 30:
        return "21-30d"
    if days <= 60:
        return "31-60d"
    return "60d+"


def _detail_matrix_area_order(metric_df, zone_filter, split_dimension):
    if metric_df.empty:
        return []

    available_areas = set(metric_df["area_name"].dropna().astype(str))
    default_areas = [
        area for area in _default_area_candidates(zone_filter, split_dimension)
        if area in available_areas
    ]
    ranked_areas = (
        metric_df.groupby("area_name")["quantity_mtonnes"]
        .sum()
        .sort_values(ascending=False)
        .index
        .astype(str)
        .tolist()
    )
    return default_areas + [area for area in ranked_areas if area not in default_areas]


def _detail_matrix_area_color_map(weekly_df, zone_filter, split_dimension):
    metric_df = weekly_df[weekly_df["zone_filter"] == zone_filter].copy() if not weekly_df.empty else pd.DataFrame()
    if metric_df.empty:
        return {}

    area_order = _detail_matrix_area_order(metric_df, zone_filter, split_dimension)
    color_map = {}
    color_idx = 0
    for area_name in area_order:
        marker_color = KPLER_FLEET_SPECIAL_AREA_COLORS.get(area_name)
        if marker_color is None:
            marker_color = KPLER_FLEET_COLORS[color_idx % len(KPLER_FLEET_COLORS)]
            color_idx += 1
        color_map[area_name] = marker_color
    return color_map


def _detail_matrix_legend_item(label, color=None, line=False, muted=False):
    marker_style = {
        "width": "22px",
        "height": "8px" if line else "10px",
        "borderRadius": "999px" if line else "2px",
        "background": color or "#cbd5e1",
        "display": "inline-block",
        "flex": "0 0 auto",
        "boxShadow": "inset 0 0 0 1px rgba(15, 23, 42, 0.10)",
    }
    if line:
        marker_style.update({
            "height": "3px",
            "boxShadow": "none",
        })

    return html.Span(
        [
            html.Span(style=marker_style),
            html.Span(label),
        ],
        className="fleet-metrics-matrix-legend-item muted" if muted else "fleet-metrics-matrix-legend-item",
    )


def build_region_detail_matrix_legend(weekly_df, split_dimension):
    color_maps = {
        zone_filter: _detail_matrix_area_color_map(
            weekly_df,
            zone_filter,
            split_dimension,
        )
        for zone_filter in KPLER_FLEET_DIVERSION_REGION_ORDER
    }
    return build_region_detail_matrix_legend_from_color_maps(
        color_maps,
        split_dimension,
    )


def build_region_detail_matrix_legend_from_color_maps(
    color_maps,
    split_dimension,
):
    split_label = KPLER_FLEET_SPLIT_TITLE_LABELS.get(split_dimension, split_dimension.replace("_", " "))
    region_cards = []
    for zone_filter in KPLER_FLEET_DIVERSION_REGION_ORDER:
        color_map = dict((color_maps or {}).get(zone_filter) or {})
        if not color_map:
            items = [_detail_matrix_legend_item("No split data", muted=True)]
        else:
            items = [
                _detail_matrix_legend_item(_area_display_name(area_name), color)
                for area_name, color in color_map.items()
            ]
        region_cards.append(
            html.Div(
                [
                    html.Div(_region_label(zone_filter), className="fleet-metrics-matrix-region-legend-title"),
                    html.Div(items, className="fleet-metrics-matrix-region-legend-items"),
                ],
                className="fleet-metrics-matrix-region-legend-card",
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"Split columns: {split_label}", className="fleet-metrics-matrix-legend-label"),
                    _detail_matrix_legend_item("same row colors apply to loaded and floating", muted=True),
                    _detail_matrix_legend_item("4-week MA", KPLER_FLEET_MA_COLOR, line=True),
                ],
                className="fleet-metrics-matrix-legend-group",
            ),
            html.Div(
                [
                    html.Span("Floating-days column", className="fleet-metrics-matrix-legend-label"),
                    *[
                        _detail_matrix_legend_item(bucket, FLOATING_DAY_BUCKET_COLORS[bucket])
                        for bucket in FLOATING_DAY_BUCKET_ORDER
                    ],
                ],
                className="fleet-metrics-matrix-legend-group",
            ),
            html.Div(region_cards, className="fleet-metrics-matrix-region-legend-grid"),
        ],
        className="fleet-metrics-matrix-legend",
    )


def _add_detail_metric_matrix_cell(
    fig,
    weekly_df,
    metric,
    zone_filter,
    row,
    col,
    color_map,
):
    metric_df = weekly_df[
        (weekly_df["zone_filter"] == zone_filter)
        & (weekly_df["metric"] == metric)
    ].copy() if not weekly_df.empty else pd.DataFrame()
    if metric_df.empty:
        return

    metric_df["date"] = pd.to_datetime(metric_df["date"])
    area_order = [area_name for area_name in color_map if area_name in set(metric_df["area_name"])]
    for area_name in area_order:
        area_df = metric_df[metric_df["area_name"] == area_name].sort_values("date")
        if area_df["quantity_mtonnes"].abs().sum() == 0:
            continue
        marker_color = color_map.get(area_name, KPLER_FLEET_SPECIAL_AREA_COLORS["Unknown"])
        display_name = _area_display_name(area_name)
        fig.add_trace(
            go.Bar(
                x=area_df["date"],
                y=area_df["quantity_mtonnes"],
                name=display_name,
                legendgroup=display_name,
                showlegend=False,
                marker_color=marker_color,
                hovertemplate=(
                    f"{_region_label(zone_filter)}<br>"
                    f"{KPLER_FLEET_METRIC_LABELS.get(metric, metric)}<br>"
                    "%{fullData.name}<br>"
                    "%{x|%d %b %Y}<br>"
                    "%{y:.2f} mt<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    total_df = (
        metric_df.groupby("date", as_index=False)["quantity_mtonnes"]
        .sum()
        .sort_values("date")
    )
    total_df["four_week_ma"] = total_df["quantity_mtonnes"].rolling(4, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=total_df["date"],
            y=total_df["four_week_ma"],
            name="4-week MA",
            legendgroup="4-week MA",
            showlegend=False,
            mode="lines",
            line=dict(color=KPLER_FLEET_MA_COLOR, width=2.1),
            hovertemplate=(
                f"{_region_label(zone_filter)}<br>"
                f"{KPLER_FLEET_METRIC_LABELS.get(metric, metric)}<br>"
                "4-week MA<br>%{x|%d %b %Y}<br>%{y:.2f} mt<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )


def _add_detail_floating_days_matrix_cell(fig, weekly_df, zone_filter, row, col):
    metric_df = weekly_df[
        (weekly_df["zone_filter"] == zone_filter)
        & (weekly_df["metric"] == "floating_storage")
    ].copy() if not weekly_df.empty else pd.DataFrame()
    if metric_df.empty:
        return

    metric_df["date"] = pd.to_datetime(metric_df["date"])
    if "age_bucket" in metric_df.columns:
        metric_df["age_bucket"] = metric_df["age_bucket"].fillna("Unknown").astype(str)
    else:
        metric_df["age_bucket"] = metric_df["area_name"].map(_floating_day_bucket)
    metric_df = (
        metric_df.groupby(["date", "age_bucket"], as_index=False)["quantity_mtonnes"]
        .sum()
        .sort_values(["date", "age_bucket"])
    )
    area_order = [
        bucket
        for bucket in FLOATING_DAY_BUCKET_ORDER
        if bucket in set(metric_df["age_bucket"])
    ]
    for area_name in area_order:
        area_df = metric_df[metric_df["age_bucket"] == area_name].sort_values("date")
        fig.add_trace(
            go.Bar(
                x=area_df["date"],
                y=area_df["quantity_mtonnes"],
                name=area_name,
                legendgroup=area_name,
                showlegend=False,
                marker_color=FLOATING_DAY_BUCKET_COLORS.get(area_name, KPLER_FLEET_SPECIAL_AREA_COLORS["Unknown"]),
                hovertemplate=(
                    f"{_region_label(zone_filter)}<br>"
                    "Floating-days age buckets<br>"
                    "%{fullData.name}<br>"
                    "%{x|%d %b %Y}<br>"
                    "%{y:.2f} mt<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    total_df = metric_df.groupby("date", as_index=False)["quantity_mtonnes"].sum().sort_values("date")
    total_df["four_week_ma"] = total_df["quantity_mtonnes"].rolling(4, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=total_df["date"],
            y=total_df["four_week_ma"],
            name="Age 4-week MA",
            legendgroup="Age 4-week MA",
            showlegend=False,
            mode="lines",
            line=dict(color=KPLER_FLEET_MA_COLOR, width=2.1),
            hovertemplate=(
                f"{_region_label(zone_filter)}<br>"
                "Floating-days 4-week MA<br>%{x|%d %b %Y}<br>%{y:.2f} mt<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )


def build_region_detail_matrix(weekly_df, floating_days_weekly_df, split_dimension):
    region_order = KPLER_FLEET_DIVERSION_REGION_ORDER
    if weekly_df.empty and floating_days_weekly_df.empty:
        return _empty_figure("No Kpler fleet metrics data loaded for the regional chart matrix", height=REGION_DETAIL_MATRIX_HEIGHT)

    subplot_titles = []
    subtitle = KPLER_FLEET_SPLIT_TITLE_LABELS.get(split_dimension, split_dimension.replace("_", " "))
    for zone_filter in region_order:
        subplot_titles.extend([
            f"<b>{_region_label(zone_filter)}</b><br><span style='font-size:10px;color:#64748b'>Commodities on water by {subtitle}</span>",
            "<span style='font-size:10px;color:#64748b'>Floating storage by selected split</span>",
            "<span style='font-size:10px;color:#64748b'>Floating storage age buckets</span>",
        ])

    fig = make_subplots(
        rows=len(region_order),
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.045,
        vertical_spacing=0.055,
    )

    color_maps = {
        zone_filter: _detail_matrix_area_color_map(weekly_df, zone_filter, split_dimension)
        for zone_filter in region_order
    }
    for row_idx, zone_filter in enumerate(region_order, start=1):
        _add_detail_metric_matrix_cell(
            fig,
            weekly_df,
            "loaded_vessels",
            zone_filter,
            row_idx,
            1,
            color_maps.get(zone_filter, {}),
        )
        _add_detail_metric_matrix_cell(
            fig,
            weekly_df,
            "floating_storage",
            zone_filter,
            row_idx,
            2,
            color_maps.get(zone_filter, {}),
        )
        _add_detail_floating_days_matrix_cell(
            fig,
            floating_days_weekly_df,
            zone_filter,
            row_idx,
            3,
        )

    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        height=REGION_DETAIL_MATRIX_HEIGHT,
        margin=dict(l=46, r=18, t=56, b=42),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        showlegend=False,
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.96)",
            bordercolor="rgba(148, 163, 184, 0.65)",
            font=dict(size=11, color="#0f172a"),
        ),
    )
    fig.update_xaxes(
        title=None,
        tickformat="%b %Y",
        dtick="M3",
        gridcolor="rgba(148, 163, 184, 0.16)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(size=7, color="#64748b"),
        fixedrange=True,
    )
    fig.update_yaxes(
        title=None,
        rangemode="tozero",
        gridcolor="rgba(148, 163, 184, 0.22)",
        zerolinecolor="rgba(148, 163, 184, 0.42)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(size=8, color="#64748b"),
        fixedrange=True,
    )
    for row_idx in range(1, len(region_order) + 1):
        fig.update_yaxes(title={"text": "mt", "font": dict(size=9, color="#475569")}, row=row_idx, col=1)
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=9, color="#334155")
    return fig


def _signal_region_row_titles():
    return [f"<b>{_region_label(zone_filter)}</b>" for zone_filter in KPLER_FLEET_DIVERSION_REGION_ORDER]


def _style_signal_region_row_figure(fig, *, barmode=None, legend_y=1.18, left_title=None, right_title=None):
    fig.update_layout(
        template="plotly_white",
        height=SIGNAL_REGION_ROW_CHART_HEIGHT,
        margin=dict(l=46, r=42 if right_title else 16, t=82, b=42),
        plot_bgcolor="rgba(248, 250, 252, 0.9)",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        barmode=barmode,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
            font=dict(size=9, color="#334155"),
            groupclick="togglegroup",
            itemsizing="constant",
            itemwidth=30,
        ),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.96)",
            bordercolor="rgba(148, 163, 184, 0.65)",
            font=dict(size=11, color="#0f172a"),
        ),
    )
    fig.update_xaxes(
        title=None,
        gridcolor="rgba(148, 163, 184, 0.14)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(size=7, color="#64748b"),
        fixedrange=True,
    )
    fig.update_yaxes(
        title=None,
        rangemode="tozero",
        gridcolor="rgba(148, 163, 184, 0.20)",
        zerolinecolor="rgba(148, 163, 184, 0.34)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(size=8, color="#64748b"),
        fixedrange=True,
    )
    if left_title:
        fig.update_yaxes(title={"text": left_title, "font": dict(size=10, color="#475569")}, row=1, col=1)
    if right_title:
        fig.update_yaxes(
            title={"text": right_title, "font": dict(size=10, color="#475569")},
            row=1,
            col=len(KPLER_FLEET_DIVERSION_REGION_ORDER),
            secondary_y=True,
        )
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=9, color="#334155")
    return fig


def build_arrival_origin_region_row(signals_df):
    today = dt.date.today()
    region_order = KPLER_FLEET_DIVERSION_REGION_ORDER
    if signals_df.empty:
        return _empty_figure("No Kpler flow signal data loaded", height=SIGNAL_REGION_ROW_CHART_HEIGHT)

    df = signals_df[
        (signals_df["zone_filter"].isin(region_order))
        & (signals_df["endpoint"] == "flows")
        & (signals_df["metric"] == "inbound_tonnes")
    ].copy()
    if df.empty:
        return _empty_figure("No inbound flow signal data for regional basins", height=SIGNAL_REGION_ROW_CHART_HEIGHT)

    df["date"] = pd.to_datetime(df["date"])
    df["area_name"] = df["area_name"].fillna("Unknown").astype(str)
    df["value_mt"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0) / 1_000_000.0
    df = df[
        (df["date"].dt.date >= today - dt.timedelta(days=30))
        & (df["date"].dt.date <= today + dt.timedelta(days=45))
    ].copy()
    if df.empty:
        return _empty_figure("No recent or forward inbound flow signal data", height=SIGNAL_REGION_ROW_CHART_HEIGHT)

    top_origins = (
        df.groupby("area_name")["value_mt"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .index
        .tolist()
    )
    df["origin_group"] = df["area_name"].where(df["area_name"].isin(top_origins), "Others")
    plot_df = (
        df.groupby(["zone_filter", "date", "origin_group"], as_index=False)["value_mt"]
        .sum()
        .sort_values(["zone_filter", "date", "origin_group"])
    )
    origin_order = top_origins + (["Others"] if "Others" in set(plot_df["origin_group"]) else [])
    color_map = {
        origin: KPLER_FLEET_COLORS[idx % len(KPLER_FLEET_COLORS)]
        for idx, origin in enumerate(origin_order)
    }

    fig = make_subplots(
        rows=1,
        cols=len(region_order),
        subplot_titles=_signal_region_row_titles(),
        horizontal_spacing=0.035,
    )
    legend_seen = set()
    for col, zone_filter in enumerate(region_order, start=1):
        zone_df = plot_df[plot_df["zone_filter"] == zone_filter]
        for origin in origin_order:
            area_df = zone_df[zone_df["origin_group"] == origin]
            if area_df.empty or area_df["value_mt"].abs().sum() == 0:
                continue
            display_name = _area_display_name(origin)
            fig.add_trace(
                go.Bar(
                    x=area_df["date"],
                    y=area_df["value_mt"],
                    name=display_name,
                    legendgroup=display_name,
                    showlegend=display_name not in legend_seen,
                    marker_color=color_map.get(origin),
                    hovertemplate=(
                        f"{_region_label(zone_filter)}<br>"
                        "%{fullData.name}<br>%{x|%d %b %Y}<br>%{y:.2f} mt<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
            legend_seen.add(display_name)
        fig.add_vline(
            x=today,
            line_width=1.2,
            line_dash="dash",
            line_color="#64748b",
            row=1,
            col=col,
        )

    _style_signal_region_row_figure(fig, barmode="stack", left_title="mt", legend_y=1.20)
    fig.update_xaxes(tickformat="%d %b", nticks=4)
    return fig


def build_utilization_region_row(signals_df, start_date, end_date):
    region_order = KPLER_FLEET_DIVERSION_REGION_ORDER
    if signals_df.empty:
        return _empty_figure("No fleet utilization signal data loaded", height=SIGNAL_REGION_ROW_CHART_HEIGHT)

    df = signals_df[
        (signals_df["zone_filter"].isin(region_order))
        & (signals_df["endpoint"] == "fleet_utilization")
        & (signals_df["metric"] == "vessel_count")
    ].copy()
    if df.empty:
        return _empty_figure("No fleet utilization signal data for regional basins", height=SIGNAL_REGION_ROW_CHART_HEIGHT)

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    df = df[
        (df["date"].dt.date >= start_date)
        & (df["date"].dt.date <= end_date)
    ].copy()
    if df.empty:
        return _empty_figure("No fleet utilization signal data in this date range", height=SIGNAL_REGION_ROW_CHART_HEIGHT)

    state_order = ["Loaded", "Ballast", "Maintenance"]
    state_colors = {
        "Loaded": "#0b3558",
        "Ballast": "#2f91d0",
        "Maintenance": "#95a3b3",
    }
    plot_df = (
        df.groupby(["zone_filter", "date", "area_name"], as_index=False)["value"]
        .sum()
        .sort_values(["zone_filter", "area_name", "date"])
    )

    fig = make_subplots(
        rows=1,
        cols=len(region_order),
        subplot_titles=_signal_region_row_titles(),
        horizontal_spacing=0.035,
    )
    legend_seen = set()
    for col, zone_filter in enumerate(region_order, start=1):
        zone_df = plot_df[plot_df["zone_filter"] == zone_filter]
        for state in state_order:
            state_df = zone_df[zone_df["area_name"] == state].sort_values("date")
            if state_df.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=state_df["date"],
                    y=state_df["value"],
                    name=state,
                    legendgroup=state,
                    showlegend=state not in legend_seen,
                    mode="lines",
                    stackgroup=f"util_{zone_filter}",
                    line=dict(width=1.2, color=state_colors.get(state)),
                    hovertemplate=(
                        f"{_region_label(zone_filter)}<br>"
                        "%{fullData.name}<br>%{x|%d %b %Y}<br>%{y:.0f} vessels<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
            legend_seen.add(state)

    _style_signal_region_row_figure(fig, left_title="vessels", legend_y=1.16)
    fig.update_xaxes(tickformat="%b %Y", nticks=4)
    return fig


def build_congestion_region_row(signals_df, start_date, end_date):
    region_order = KPLER_FLEET_DIVERSION_REGION_ORDER
    if signals_df.empty:
        return _empty_figure("No congestion signal data loaded", height=CONGESTION_SIGNAL_CHART_HEIGHT)

    df = signals_df[
        (signals_df["zone_filter"].isin(region_order))
        & (signals_df["endpoint"] == "congestion")
        & (signals_df["metric"].isin(["waiting_count", "waiting_duration_days"]))
    ].copy()
    if df.empty:
        return _empty_figure("No congestion signal data for regional basins", height=CONGESTION_SIGNAL_CHART_HEIGHT)

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    df = df[
        (df["date"].dt.date >= start_date)
        & (df["date"].dt.date <= end_date)
    ].copy()
    if df.empty:
        return _empty_figure("No congestion signal data in this date range", height=CONGESTION_SIGNAL_CHART_HEIGHT)

    totals = (
        df.groupby(["zone_filter", "date", "metric"], as_index=False)["value"]
        .sum()
        .pivot_table(
            index=["zone_filter", "date"],
            columns="metric",
            values="value",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
        .sort_values(["zone_filter", "date"])
    )
    for metric in ("waiting_count", "waiting_duration_days"):
        if metric not in totals:
            totals[metric] = 0.0

    def _percentile_series(series):
        clean = pd.to_numeric(series, errors="coerce")
        valid = clean.dropna()
        if valid.empty:
            return pd.Series(0.0, index=series.index)
        if valid.nunique(dropna=True) <= 1:
            value = 50.0 if float(valid.iloc[-1]) > 0 else 0.0
            return pd.Series(value, index=series.index)
        return clean.rank(pct=True, method="average").fillna(0.0) * 100.0

    totals["waiting_count"] = pd.to_numeric(totals["waiting_count"], errors="coerce").fillna(0.0)
    totals["waiting_duration_days"] = pd.to_numeric(
        totals["waiting_duration_days"],
        errors="coerce",
    ).fillna(0.0)
    totals["count_percentile"] = totals.groupby("zone_filter")["waiting_count"].transform(_percentile_series)
    totals["duration_percentile"] = totals.groupby("zone_filter")["waiting_duration_days"].transform(_percentile_series)
    totals["pressure_score"] = 0.6 * totals["count_percentile"] + 0.4 * totals["duration_percentile"]
    totals["waiting_count_14d"] = totals.groupby("zone_filter")["waiting_count"].transform(
        lambda series: series.rolling(14, min_periods=1).mean()
    )
    totals["pressure_score_14d"] = totals.groupby("zone_filter")["pressure_score"].transform(
        lambda series: series.rolling(14, min_periods=1).mean()
    )

    display_start = max(start_date, end_date - dt.timedelta(days=540))
    display_df = totals[totals["date"].dt.date >= display_start].copy()
    if display_df.empty:
        display_df = totals.copy()

    latest_rows = []
    for zone_filter in region_order:
        zone_history = totals[totals["zone_filter"] == zone_filter].sort_values("date")
        if zone_history.empty:
            continue
        latest = zone_history.iloc[-1]
        latest_rows.append(
            {
                "zone_filter": zone_filter,
                "region": _region_label(zone_filter),
                "date": latest["date"],
                "waiting_count": float(latest["waiting_count"]),
                "waiting_duration_days": float(latest["waiting_duration_days"]),
                "pressure_score": float(latest["pressure_score"]),
                "count_percentile": float(latest["count_percentile"]),
                "duration_percentile": float(latest["duration_percentile"]),
            }
        )
    latest_df = pd.DataFrame(latest_rows)
    if latest_df.empty:
        return _empty_figure("No congestion signal data in this date range", height=CONGESTION_SIGNAL_CHART_HEIGHT)

    def _pressure_color(score):
        if score >= 85:
            return "#dc2626"
        if score >= 70:
            return "#f59e0b"
        if score >= 50:
            return "#2563eb"
        return "#64748b"

    latest_df = latest_df.sort_values("pressure_score", ascending=False)
    fig = make_subplots(
        rows=3,
        cols=len(region_order),
        subplot_titles=(
            ["<b>Current pressure ranking</b>"]
            + [f"<b>{_region_label(zone_filter)}</b>" for zone_filter in region_order]
            + ["" for _ in region_order]
        ),
        specs=[
            [{"type": "bar", "colspan": len(region_order)}] + [None for _ in region_order[1:]],
            [{"type": "xy"} for _ in region_order],
            [{"type": "xy"} for _ in region_order],
        ],
        row_heights=[0.28, 0.34, 0.38],
        vertical_spacing=0.09,
        horizontal_spacing=0.035,
    )

    fig.add_vrect(
        x0=70,
        x1=85,
        fillcolor="rgba(245, 158, 11, 0.08)",
        line_width=0,
        layer="below",
        row=1,
        col=1,
    )
    fig.add_vrect(
        x0=85,
        x1=100,
        fillcolor="rgba(220, 38, 38, 0.08)",
        line_width=0,
        layer="below",
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=latest_df["pressure_score"],
            y=latest_df["region"],
            orientation="h",
            marker_color=[_pressure_color(score) for score in latest_df["pressure_score"]],
            text=[
                f"{score:.0f}%  |  {count:.0f} waiting  |  {duration:.1f}d"
                for score, count, duration in zip(
                    latest_df["pressure_score"],
                    latest_df["waiting_count"],
                    latest_df["waiting_duration_days"],
                )
            ],
            textposition="auto",
            customdata=list(
                zip(
                    latest_df["waiting_count"],
                    latest_df["waiting_duration_days"],
                    latest_df["count_percentile"],
                    latest_df["duration_percentile"],
                    latest_df["date"],
                )
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Pressure score: %{x:.0f}%<br>"
                "Waiting vessels: %{customdata[0]:.0f}<br>"
                "Wait-duration signal: %{customdata[1]:.2f} days<br>"
                "Count percentile: %{customdata[2]:.0f}%<br>"
                "Duration percentile: %{customdata[3]:.0f}%<br>"
                "Latest date: %{customdata[4]|%d %b %Y}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    count_axis_max = max(1.0, float(display_df["waiting_count"].max()) * 1.18)
    for col, zone_filter in enumerate(region_order, start=1):
        zone_display = display_df[display_df["zone_filter"] == zone_filter].sort_values("date")
        if zone_display.empty:
            continue
        latest = zone_display.iloc[-1]
        count_p90 = totals.loc[totals["zone_filter"] == zone_filter, "waiting_count"].quantile(0.90)
        region_label = _region_label(zone_filter)
        customdata = list(
            zip(
                zone_display["waiting_count"],
                zone_display["waiting_duration_days"],
                zone_display["pressure_score"],
            )
        )
        fig.add_trace(
            go.Scatter(
                x=zone_display["date"],
                y=zone_display["waiting_count"],
                name="Daily vessels",
                legendgroup="Daily vessels",
                showlegend=col == 1,
                mode="lines",
                line=dict(color="rgba(100, 116, 139, 0.30)", width=1.0),
                customdata=customdata,
                hovertemplate=(
                    f"{region_label}<br>"
                    "Daily waiting vessels<br>%{x|%d %b %Y}<br>"
                    "%{y:.0f} vessels<br>"
                    "Wait-duration signal: %{customdata[1]:.2f} days<br>"
                    "Pressure score: %{customdata[2]:.0f}%<extra></extra>"
                ),
            ),
            row=2,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=zone_display["date"],
                y=zone_display["waiting_count_14d"],
                name="14D avg vessels",
                legendgroup="14D avg vessels",
                showlegend=col == 1,
                mode="lines",
                line=dict(color="#0b3558", width=2.4),
                hovertemplate=(
                    f"{region_label}<br>"
                    "14D average waiting vessels<br>%{x|%d %b %Y}<br>"
                    "%{y:.1f} vessels<extra></extra>"
                ),
            ),
            row=2,
            col=col,
        )
        if pd.notna(count_p90):
            fig.add_trace(
                go.Scatter(
                    x=[zone_display["date"].min(), zone_display["date"].max()],
                    y=[count_p90, count_p90],
                    name="90th pct",
                    legendgroup="90th pct",
                    showlegend=col == 1,
                    mode="lines",
                    line=dict(color="rgba(220, 38, 38, 0.55)", width=1.2, dash="dot"),
                    hovertemplate=f"{region_label}<br>90th pct waiting vessels<br>%{{y:.1f}} vessels<extra></extra>",
                ),
                row=2,
                col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=[latest["date"]],
                y=[latest["waiting_count"]],
                mode="markers+text",
                marker=dict(color="#0b3558", size=7, line=dict(color="#ffffff", width=1.2)),
                text=[f"{latest['waiting_count']:.0f}"],
                textposition="top center",
                textfont=dict(size=9, color="#0f172a"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=zone_display["date"],
                y=zone_display["pressure_score"],
                name="Daily pressure",
                legendgroup="Daily pressure",
                showlegend=col == 1,
                mode="lines",
                line=dict(color="rgba(15, 118, 110, 0.28)", width=1.0),
                customdata=customdata,
                hovertemplate=(
                    f"{region_label}<br>"
                    "Daily pressure score<br>%{x|%d %b %Y}<br>"
                    "%{y:.0f}%<br>"
                    "Waiting vessels: %{customdata[0]:.0f}<br>"
                    "Wait-duration signal: %{customdata[1]:.2f} days<extra></extra>"
                ),
            ),
            row=3,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=zone_display["date"],
                y=zone_display["pressure_score_14d"],
                name="14D avg pressure",
                legendgroup="14D avg pressure",
                showlegend=col == 1,
                mode="lines",
                line=dict(color="#0f766e", width=2.4),
                hovertemplate=f"{region_label}<br>14D average pressure<br>%{{x|%d %b %Y}}<br>%{{y:.0f}}%<extra></extra>",
            ),
            row=3,
            col=col,
        )
        fig.add_hrect(
            y0=70,
            y1=85,
            fillcolor="rgba(245, 158, 11, 0.08)",
            line_width=0,
            row=3,
            col=col,
        )
        fig.add_hrect(
            y0=85,
            y1=100,
            fillcolor="rgba(220, 38, 38, 0.08)",
            line_width=0,
            row=3,
            col=col,
        )
        fig.add_hline(
            y=85,
            line_width=1,
            line_dash="dot",
            line_color="rgba(220, 38, 38, 0.45)",
            row=3,
            col=col,
        )

    fig.update_layout(
        template="plotly_white",
        height=CONGESTION_SIGNAL_CHART_HEIGHT,
        margin=dict(l=58, r=22, t=78, b=42),
        plot_bgcolor="rgba(248, 250, 252, 0.9)",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        bargap=0.28,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            font=dict(size=9, color="#334155"),
            groupclick="togglegroup",
            itemsizing="constant",
            itemwidth=30,
        ),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.96)",
            bordercolor="rgba(148, 163, 184, 0.65)",
            font=dict(size=11, color="#0f172a"),
        ),
    )
    fig.update_xaxes(
        title=None,
        range=[0, 104],
        ticksuffix="%",
        gridcolor="rgba(148, 163, 184, 0.18)",
        tickfont=dict(size=9, color="#64748b"),
        fixedrange=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=10, color="#334155"),
        fixedrange=True,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title=None,
        tickformat="%b %Y",
        nticks=4,
        gridcolor="rgba(148, 163, 184, 0.14)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(size=7, color="#64748b"),
        fixedrange=True,
        row=2,
    )
    fig.update_xaxes(
        title=None,
        tickformat="%b %Y",
        nticks=4,
        gridcolor="rgba(148, 163, 184, 0.14)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(size=7, color="#64748b"),
        fixedrange=True,
        row=3,
    )
    fig.update_yaxes(
        title=None,
        range=[0, count_axis_max],
        rangemode="tozero",
        gridcolor="rgba(148, 163, 184, 0.20)",
        zerolinecolor="rgba(148, 163, 184, 0.34)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(size=8, color="#64748b"),
        fixedrange=True,
        row=2,
    )
    fig.update_yaxes(
        title=None,
        range=[0, 100],
        ticksuffix="%",
        dtick=25,
        gridcolor="rgba(148, 163, 184, 0.20)",
        zerolinecolor="rgba(148, 163, 184, 0.34)",
        showline=True,
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(size=8, color="#64748b"),
        fixedrange=True,
        row=3,
    )
    for col in range(2, len(region_order) + 1):
        fig.update_yaxes(showticklabels=False, row=2, col=col)
        fig.update_yaxes(showticklabels=False, row=3, col=col)
    fig.update_yaxes(title={"text": "vessels", "font": dict(size=10, color="#475569")}, row=2, col=1)
    fig.update_yaxes(title={"text": "pressure", "font": dict(size=10, color="#475569")}, row=3, col=1)
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=9, color="#334155")
    return fig


def build_freight_region_row(signals_df, start_date, end_date):
    region_order = KPLER_FLEET_DIVERSION_REGION_ORDER
    if signals_df.empty:
        return _empty_figure("No freight demand signal data loaded", height=SIGNAL_REGION_ROW_CHART_HEIGHT)

    fig = make_subplots(
        rows=1,
        cols=len(region_order),
        subplot_titles=_signal_region_row_titles(),
        horizontal_spacing=0.035,
        specs=[[{"secondary_y": True} for _ in region_order]],
    )
    has_data = False
    for col, zone_filter in enumerate(region_order, start=1):
        freight_df = _filter_freight_for_region(signals_df, zone_filter)
        if freight_df.empty:
            continue
        freight_df = freight_df.copy()
        freight_df["date"] = pd.to_datetime(freight_df["date"])
        freight_df["value"] = pd.to_numeric(freight_df["value"], errors="coerce").fillna(0.0)
        freight_df = freight_df[
            (freight_df["date"].dt.date >= start_date)
            & (freight_df["date"].dt.date <= end_date)
        ].copy()
        if freight_df.empty:
            continue

        ton_miles = (
            freight_df[freight_df["metric"] == "loaded_ton_miles"]
            .groupby("date", as_index=False)["value"]
            .sum()
            .sort_values("date")
        )
        ton_miles["value_bn"] = ton_miles["value"] / 1_000_000_000.0
        avg_distance = freight_df[freight_df["metric"] == "avg_loaded_distance"].copy()
        avg_distance = avg_distance[avg_distance["value"] > 0]
        avg_distance = (
            avg_distance.groupby("date", as_index=False)["value"]
            .mean()
            .sort_values("date")
        )
        if not ton_miles.empty:
            has_data = True
            fig.add_trace(
                go.Bar(
                    x=ton_miles["date"],
                    y=ton_miles["value_bn"],
                    name="Loaded ton-miles",
                    legendgroup="Loaded ton-miles",
                    showlegend=col == 1,
                    marker_color="#1f5f8b",
                    hovertemplate=(
                        f"{_region_label(zone_filter)}<br>"
                        "Loaded ton-miles<br>%{x|%b %Y}<br>%{y:.1f} bn t-nmi<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
                secondary_y=False,
            )
        if not avg_distance.empty:
            has_data = True
            fig.add_trace(
                go.Scatter(
                    x=avg_distance["date"],
                    y=avg_distance["value"],
                    name="Avg distance",
                    legendgroup="Avg distance",
                    showlegend=col == 1,
                    mode="lines+markers",
                    marker=dict(size=4),
                    line=dict(color=KPLER_FLEET_MA_COLOR, width=2.0),
                    hovertemplate=(
                        f"{_region_label(zone_filter)}<br>"
                        "Avg distance<br>%{x|%b %Y}<br>%{y:.0f} nmi<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
                secondary_y=True,
            )

    if not has_data:
        return _empty_figure("No freight demand signal data in this date range", height=SIGNAL_REGION_ROW_CHART_HEIGHT)

    _style_signal_region_row_figure(fig, left_title="bn t-nmi", right_title="nmi", legend_y=1.16)
    fig.update_xaxes(tickformat="%b %Y", nticks=4)
    for col in range(1, len(region_order)):
        fig.update_yaxes(showticklabels=False, secondary_y=True, row=1, col=col)
    return fig


def _build_kpi_card(label, value, subtitle=None, tone="neutral"):
    tone_colors = {
        "neutral": "#1f2937",
        "good": "#047857",
        "warning": "#b45309",
        "danger": "#b91c1c",
    }
    return html.Div(
        [
            html.Div(label, style={"fontSize": "11px", "fontWeight": "700", "color": "#64748b", "textTransform": "uppercase"}),
            html.Div(value, style={"fontSize": "24px", "fontWeight": "700", "color": tone_colors.get(tone, "#1f2937"), "lineHeight": "1.15"}),
            html.Div(subtitle or "", style={"fontSize": "12px", "color": "#64748b", "marginTop": "6px"}),
        ],
        style=CARD_STYLE,
    )


def build_summary_cards(summary_row):
    if not summary_row or summary_row.get("status") == "No data":
        return [
            _build_kpi_card("Loaded on water", "-", "No data loaded for this basin"),
            _build_kpi_card("Floating storage", "-", "Run the FleetMetrics backfill for this region"),
            _build_kpi_card("Waiting share", "-", "Floating / loaded"),
            _build_kpi_card("4w loaded change", "-", ""),
            _build_kpi_card("4w floating change", "-", ""),
            _build_kpi_card("2y floating percentile", "-", ""),
        ]

    percentile = summary_row.get("floating_percentile")
    percentile_tone = "danger" if percentile and percentile >= 80 else "warning" if percentile and percentile >= 65 else "neutral"
    waiting_share = summary_row.get("waiting_share")
    waiting_tone = "danger" if waiting_share and waiting_share >= 6 else "warning" if waiting_share and waiting_share >= 4 else "neutral"
    latest_label = _latest_complete_week_label(summary_row)
    return [
        _build_kpi_card("Loaded on water", f"{_format_number(summary_row.get('loaded_mt'), 2)} mt", latest_label),
        _build_kpi_card("Floating storage", f"{_format_number(summary_row.get('floating_mt'), 2)} mt", latest_label),
        _build_kpi_card("Waiting share", _format_percent(waiting_share, 1), "Floating / loaded", waiting_tone),
        _build_kpi_card("4w loaded change", f"{_format_delta(summary_row.get('loaded_4w'), 2)} mt", "Latest vs four weeks prior"),
        _build_kpi_card("4w floating change", f"{_format_delta(summary_row.get('floating_4w'), 2)} mt", "Latest vs four weeks prior"),
        _build_kpi_card("2y floating percentile", _format_percent(percentile, 0), "Relative to completed weeks in range", percentile_tone),
    ]


def build_global_signal_cards(signal_row):
    if not signal_row or signal_row.get("status") == "No data":
        return [
            _build_kpi_card("LNG arrivals", "-", "No regional signal data loaded"),
            _build_kpi_card("Loaded fleet share", "-", "Loaded / active LNG fleet"),
            _build_kpi_card("Congestion duration", "-", "Waiting before discharge"),
            _build_kpi_card("Diversions", "-", "Into / out of basin"),
            _build_kpi_card("Freight demand", "-", "Loaded ton-miles"),
        ]

    loaded_share = signal_row.get("loaded_share")
    congestion_duration = signal_row.get("congestion_duration")
    inbound = signal_row.get("inbound_14d_mt")
    freight = signal_row.get("freight_ton_miles_bn")
    diversion_text = f"{_format_integer(signal_row.get('diversions_in'))} in / {_format_integer(signal_row.get('diversions_out'))} out"
    return [
        _build_kpi_card(
            "LNG arrivals",
            f"{_format_number(inbound, 2)} mt",
            "Kpler flows next 14 days",
            "good" if inbound and inbound > 0 else "neutral",
        ),
        _build_kpi_card(
            "Loaded fleet share",
            _format_percent(loaded_share, 1),
            f"{_format_integer(signal_row.get('loaded_count'))} loaded vessels",
        ),
        _build_kpi_card(
            "Congestion duration",
            f"{_format_number(congestion_duration, 2)} d",
            f"{_format_integer(signal_row.get('congestion_count'))} waiting vessels",
            "warning" if congestion_duration and congestion_duration >= 3 else "neutral",
        ),
        _build_kpi_card(
            "Diversions",
            diversion_text,
            "Recent and forward Kpler diversions",
            "warning" if signal_row.get("diversions_in") or signal_row.get("diversions_out") else "neutral",
        ),
        _build_kpi_card(
            "Freight demand",
            f"{_format_number(freight, 1)} bn t-nmi",
            f"{_format_number(signal_row.get('freight_avg_distance'), 0)} nmi avg distance",
        ),
    ]


def build_price_card(price_context):
    status = price_context.get("status")
    if status != "Fresh":
        cob = price_context.get("cob")
        subtitle = f"Latest COB {cob:%d %b %Y}" if cob else status
        return _build_kpi_card("JKM-TTF prompt spread", "Stale", subtitle, "warning")

    spread = price_context.get("spread")
    contract = price_context.get("contract") or "prompt"
    units = price_context.get("units") or "MMBtu"
    return _build_kpi_card(
        "JKM-TTF prompt spread",
        f"{_format_delta(spread, 2)} $/{units}",
        f"{contract}; price context only",
        "good" if spread and spread > 0 else "warning",
    )


def build_status_strip(
    fleet_checked_at,
    signal_checked_at,
    selected_region,
    price_context,
    fleet_changed_at=None,
    signal_changed_at=None,
):
    fleet_checked_text = "FleetMetrics checked: unavailable"
    if fleet_checked_at is not None and pd.notna(fleet_checked_at):
        checked_dt = pd.to_datetime(fleet_checked_at)
        fleet_checked_text = f"FleetMetrics checked: {checked_dt:%d %b %Y %H:%M UTC}"
    signal_checked_text = "Signals checked: unavailable"
    if signal_checked_at is not None and pd.notna(signal_checked_at):
        checked_dt = pd.to_datetime(signal_checked_at)
        signal_checked_text = f"Signals checked: {checked_dt:%d %b %Y %H:%M UTC}"
    fleet_changed_text = "FleetMetrics changed: unavailable"
    if fleet_changed_at is not None and pd.notna(fleet_changed_at):
        changed_dt = pd.to_datetime(fleet_changed_at)
        fleet_changed_text = f"FleetMetrics changed: {changed_dt:%d %b %Y %H:%M UTC}"
    signal_changed_text = "Signals changed: unavailable"
    if signal_changed_at is not None and pd.notna(signal_changed_at):
        changed_dt = pd.to_datetime(signal_changed_at)
        signal_changed_text = f"Signals changed: {changed_dt:%d %b %Y %H:%M UTC}"
    price_status = price_context.get("status", "Unavailable")
    return html.Div(
        [
            html.Span(fleet_checked_text),
            html.Span(signal_checked_text),
            html.Span(fleet_changed_text),
            html.Span(signal_changed_text),
            html.Span(f"Selected basin: {_region_label(selected_region)}"),
            html.Span("Basin presets overlap and are not additive"),
            html.Span(f"Price context: {price_status}"),
        ],
        style={
            "display": "flex",
            "gap": "18px",
            "alignItems": "center",
            "flexWrap": "wrap",
            "fontSize": "12px",
            "color": "#475569",
        },
    )


KPLER_FLEET_TABLE_REGION_LABELS = {
    "asia_pacific_oceans": "Asia-Pac",
    "europe_basin": "Europe",
    "americas_basin": "Americas",
    "middle_east_indian_ocean": "ME/Indian",
    "atlantic_basin": "Atlantic",
    "global": "Global",
}


def _region_table_label(zone_filter):
    return KPLER_FLEET_TABLE_REGION_LABELS.get(zone_filter, _region_label(zone_filter))


def build_comparison_rows(summaries, selected_region=None):
    rows = []
    for zone_filter in KPLER_FLEET_REGION_ORDER:
        row = summaries.get(zone_filter, {"region": _region_label(zone_filter), "status": "No data"})
        rows.append(
            {
                "zone_filter": zone_filter,
                "is_selected": zone_filter == selected_region,
                "region": _region_table_label(zone_filter),
                "loaded_mt": _format_number(row.get("loaded_mt"), 2),
                "loaded_wow": _format_delta(row.get("loaded_wow"), 2),
                "loaded_wow_raw": row.get("loaded_wow"),
                "loaded_4w": _format_delta(row.get("loaded_4w"), 2),
                "loaded_4w_raw": row.get("loaded_4w"),
                "loaded_percentile": _format_percent(row.get("loaded_percentile"), 0),
                "loaded_percentile_raw": row.get("loaded_percentile"),
                "floating_mt": _format_number(row.get("floating_mt"), 2),
                "floating_wow": _format_delta(row.get("floating_wow"), 2),
                "floating_wow_raw": row.get("floating_wow"),
                "floating_4w": _format_delta(row.get("floating_4w"), 2),
                "floating_4w_raw": row.get("floating_4w"),
                "floating_percentile": _format_percent(row.get("floating_percentile"), 0),
                "floating_percentile_raw": row.get("floating_percentile"),
                "waiting_share": _format_percent(row.get("waiting_share"), 1),
                "waiting_share_raw": row.get("waiting_share"),
                "status": row.get("status", "No data"),
            }
        )
    return rows


def build_global_signal_rows(signal_summaries, selected_region=None):
    rows = []
    for zone_filter in KPLER_FLEET_REGION_ORDER:
        row = signal_summaries.get(zone_filter, {"region": _region_label(zone_filter), "status": "No data"})
        rows.append(
            {
                "zone_filter": zone_filter,
                "is_selected": zone_filter == selected_region,
                "region": _region_table_label(zone_filter),
                "inbound_14d_mt": _format_number(row.get("inbound_14d_mt"), 2),
                "inbound_30d_mt": _format_number(row.get("inbound_30d_mt"), 2),
                "loaded_share": _format_percent(row.get("loaded_share"), 1),
                "loaded_share_raw": row.get("loaded_share"),
                "loaded_count": _format_integer(row.get("loaded_count")),
                "ballast_count": _format_integer(row.get("ballast_count")),
                "congestion_count": _format_integer(row.get("congestion_count")),
                "congestion_duration": _format_number(row.get("congestion_duration"), 2),
                "congestion_duration_raw": row.get("congestion_duration"),
                "diversions_in": _format_integer(row.get("diversions_in")),
                "diversions_out": _format_integer(row.get("diversions_out")),
                "freight_ton_miles_bn": _format_number(row.get("freight_ton_miles_bn"), 1),
                "freight_avg_distance": _format_number(row.get("freight_avg_distance"), 0),
                "status": row.get("status", "No data"),
            }
        )
    return rows


def build_movers_rows(weekly_df):
    if weekly_df.empty:
        return []

    totals = (
        weekly_df.groupby(["area_name", "date", "metric"], as_index=False)["quantity_mtonnes"]
        .sum()
        .pivot_table(index=["area_name", "date"], columns="metric", values="quantity_mtonnes", aggfunc="sum")
        .reset_index()
        .sort_values(["area_name", "date"])
    )
    rows = []
    for area_name, area_df in totals.groupby("area_name", sort=False):
        area_df = area_df.sort_values("date")
        loaded = pd.to_numeric(area_df.get("loaded_vessels"), errors="coerce")
        floating = pd.to_numeric(area_df.get("floating_storage"), errors="coerce")
        latest_loaded = float(loaded.iloc[-1]) if not loaded.empty and pd.notna(loaded.iloc[-1]) else None
        latest_floating = float(floating.iloc[-1]) if not floating.empty and pd.notna(floating.iloc[-1]) else None
        loaded_wow = _series_delta(loaded, 1)
        floating_wow = _series_delta(floating, 1)
        floating_4w = _series_delta(floating, 4)
        rank_value = abs(loaded_wow or 0) + abs(floating_wow or 0)
        if rank_value == 0 and not latest_loaded and not latest_floating:
            continue
        rows.append(
            {
                "area": _area_display_name(area_name),
                "loaded_mt": _format_number(latest_loaded, 2),
                "loaded_wow": _format_delta(loaded_wow, 2),
                "floating_mt": _format_number(latest_floating, 2),
                "floating_wow": _format_delta(floating_wow, 2),
                "floating_4w": _format_delta(floating_4w, 2),
                "_rank": rank_value,
            }
        )

    rows = sorted(rows, key=lambda row: row["_rank"], reverse=True)[:12]
    for row in rows:
        row.pop("_rank", None)
    return rows


def _join_cell_classes(*classes):
    return " ".join(class_name for class_name in classes if class_name)


def _number_column(field, header_name, width=88, min_width=66, cell_class=None):
    return {
        "field": field,
        "headerName": header_name,
        "width": width,
        "minWidth": min_width,
        "type": "rightAligned",
        "cellClass": _join_cell_classes("fleet-metrics-number-cell", cell_class),
    }


def _delta_column(field, header_name, width=84, min_width=66, cell_class=None, raw_field=None):
    column = _number_column(field, header_name, width, min_width, cell_class)
    raw_field = raw_field or f"{field}_raw"
    column["cellClassRules"] = {
        "fleet-metrics-positive-cell": f"params.data && params.data.{raw_field} > 0.0001",
        "fleet-metrics-negative-cell": f"params.data && params.data.{raw_field} < -0.0001",
        "fleet-metrics-muted-cell": (
            f"!params.data || params.data.{raw_field} == null "
            f"|| (params.data.{raw_field} >= -0.0001 && params.data.{raw_field} <= 0.0001)"
        ),
    }
    return column


def _threshold_column(field, header_name, width=88, min_width=66, high=80, medium=65, cell_class=None, raw_field=None):
    column = _number_column(field, header_name, width, min_width, cell_class)
    raw_field = raw_field or f"{field}_raw"
    column["cellClassRules"] = {
        "fleet-metrics-pressure-high": (
            f"params.data && params.data.{raw_field} != null && params.data.{raw_field} >= {high}"
        ),
        "fleet-metrics-pressure-medium": (
            f"params.data && params.data.{raw_field} != null "
            f"&& params.data.{raw_field} >= {medium} && params.data.{raw_field} < {high}"
        ),
        "fleet-metrics-muted-cell": (
            f"!params.data || params.data.{raw_field} == null"
        ),
    }
    return column


def _metric_column_group(header_name, children):
    return {
        "headerName": header_name,
        "headerClass": "fleet-metrics-grid-header-group",
        "marryChildren": True,
        "children": children,
    }


FLEET_METRICS_REGION_COMPARISON_COLUMN_DEFS = [
    {
        "field": "region",
        "headerName": "Region",
        "pinned": "left",
        "width": 78,
        "minWidth": 70,
        "lockPinned": True,
        "suppressMovable": True,
        "cellClass": "fleet-metrics-left-cell fleet-metrics-strong-cell",
    },
    _metric_column_group(
        "Loaded",
        [
            _number_column("loaded_mt", "Mt", 50, 44, "fleet-metrics-group-start fleet-metrics-metric-anchor"),
            _delta_column("loaded_wow", "WoW", 44, 38),
            _delta_column("loaded_4w", "4W", 44, 38),
            _threshold_column("loaded_percentile", "Pct", 44, 38, high=80, medium=65),
        ],
    ),
    _metric_column_group(
        "Floating",
        [
            _number_column("floating_mt", "Mt", 50, 44, "fleet-metrics-group-start fleet-metrics-metric-anchor"),
            _delta_column("floating_wow", "WoW", 44, 38),
            _delta_column("floating_4w", "4W", 44, 38),
            _threshold_column("floating_percentile", "Pct", 44, 38, high=80, medium=65),
        ],
    ),
    _metric_column_group(
        "Wait",
        [
            _threshold_column("waiting_share", "Share", 52, 46, high=6, medium=4, cell_class="fleet-metrics-group-start"),
        ],
    ),
]

FLEET_METRICS_GLOBAL_SIGNALS_COLUMN_DEFS = [
    {
        "field": "region",
        "headerName": "Region",
        "pinned": "left",
        "width": 78,
        "minWidth": 70,
        "lockPinned": True,
        "suppressMovable": True,
        "cellClass": "fleet-metrics-left-cell fleet-metrics-strong-cell",
    },
    _metric_column_group(
        "Arrivals",
        [
            _number_column("inbound_14d_mt", "14d", 46, 40, "fleet-metrics-group-start fleet-metrics-metric-anchor"),
            _number_column("inbound_30d_mt", "30d", 46, 40),
        ],
    ),
    _metric_column_group(
        "Util",
        [
            _threshold_column("loaded_share", "L%", 44, 38, high=70, medium=55, cell_class="fleet-metrics-group-start"),
            _number_column("loaded_count", "L", 34, 30),
            _number_column("ballast_count", "B", 34, 30),
        ],
    ),
    _metric_column_group(
        "Congestion",
        [
            _number_column("congestion_count", "Vsl", 40, 34, "fleet-metrics-group-start"),
            _threshold_column("congestion_duration", "Days", 66, 54, high=3, medium=2),
        ],
    ),
    _metric_column_group(
        "Div",
        [
            _number_column("diversions_in", "In", 30, 28, "fleet-metrics-group-start"),
            _number_column("diversions_out", "Out", 34, 30),
        ],
    ),
    _metric_column_group(
        "Freight",
        [
            _number_column("freight_ton_miles_bn", "t-mi", 50, 42, "fleet-metrics-group-start"),
            _number_column("freight_avg_distance", "nmi", 48, 40),
        ],
    ),
]


FLEET_METRICS_COMPACT_FIELD_LIMITS = {
    "region": (70, 78),
    "loaded_mt": (44, 52),
    "floating_mt": (44, 52),
    "loaded_wow": (38, 48),
    "loaded_4w": (38, 48),
    "floating_wow": (38, 48),
    "floating_4w": (38, 48),
    "loaded_percentile": (38, 48),
    "floating_percentile": (38, 48),
    "waiting_share": (46, 56),
    "inbound_14d_mt": (40, 50),
    "inbound_30d_mt": (40, 50),
    "loaded_share": (38, 48),
    "loaded_count": (30, 38),
    "ballast_count": (30, 38),
    "congestion_count": (34, 42),
    "congestion_duration": (54, 70),
    "diversions_in": (28, 34),
    "diversions_out": (30, 38),
    "freight_ton_miles_bn": (42, 54),
    "freight_avg_distance": (40, 52),
}


def _grid_text_length(value):
    if value is None:
        return 0
    text_value = str(value).strip()
    if text_value == "-":
        return 1
    return len(text_value)


def _compact_column_width(column, rows):
    field = column.get("field")
    header_name = column.get("headerName", field or "")
    min_width, max_width = FLEET_METRICS_COMPACT_FIELD_LIMITS.get(
        field,
        (int(column.get("minWidth") or 50), int(column.get("width") or 90)),
    )
    value_length = max([_grid_text_length(row.get(field)) for row in rows] or [0])
    header_length = max(_grid_text_length(part) for part in str(header_name).replace("/", " / ").split())
    if field == "region":
        header_length = _grid_text_length(header_name)
        value_length = max([_grid_text_length(row.get(field)) for row in rows] or [header_length])
    target_length = max(header_length, value_length)
    width = int(target_length * 6.2 + 24)
    width = max(min_width, min(max_width, width))
    return width


def build_compact_column_defs(column_defs, rows):
    rows = rows or []
    compact_defs = []
    for column in column_defs:
        compact_column = {key: value for key, value in column.items() if key != "children"}
        if "children" in column:
            compact_column["children"] = build_compact_column_defs(column["children"], rows)
        elif compact_column.get("field"):
            width = _compact_column_width(compact_column, rows)
            compact_column["width"] = width
            compact_column["minWidth"] = min(width, int(compact_column.get("minWidth") or width))
            compact_column["suppressSizeToFit"] = True
        compact_defs.append(compact_column)
    return compact_defs

FLEET_METRICS_MOVERS_COLUMN_DEFS = [
    {
        "field": "area",
        "headerName": "Area",
        "pinned": "left",
        "width": 166,
        "minWidth": 128,
        "cellClass": "fleet-metrics-left-cell fleet-metrics-strong-cell",
    },
    _number_column("loaded_mt", "Loaded mt"),
    _delta_column("loaded_wow", "Loaded WoW"),
    _number_column("floating_mt", "Floating mt"),
    _delta_column("floating_wow", "Floating WoW"),
    _delta_column("floating_4w", "Floating 4w"),
]

def _ag_grid_table(
    id_value,
    column_defs,
    page_size=8,
    height=260,
    pagination=True,
    extra_class="",
    column_size="responsiveSizeToFit",
):
    grid_options = {
        **FLEET_METRICS_AG_GRID_OPTIONS,
        "pagination": pagination,
    }
    if pagination:
        grid_options["paginationPageSize"] = page_size
    else:
        grid_options.pop("paginationPageSizeSelector", None)
    grid_kwargs = {
        "id": id_value,
        "rowData": [],
        "columnDefs": column_defs,
        "defaultColDef": FLEET_METRICS_AG_GRID_DEFAULT_COL_DEF,
        "dashGridOptions": grid_options,
        "className": f"{AG_GRID_THEME} fleet-metrics-grid {extra_class}".strip(),
        "style": {"width": "100%", "height": f"{height}px"},
        "dangerously_allow_code": True,
    }
    if column_size:
        grid_kwargs["columnSize"] = column_size
    return dag.AgGrid(**grid_kwargs)


def _seasonality_section():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Regional seasonality", style={"margin": "0", "fontSize": "16px"}),
                    html.Div(
                        "5Y weekly range, 5Y average, current year, and prior year. Older years stay available in the legend.",
                        className="fleet-metrics-table-subtitle",
                    ),
                ],
                className="fleet-metrics-table-heading",
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="fleet-metrics-loaded-seasonal-chart",
                        style={"height": f"{SEASONAL_CHART_HEIGHT}px"},
                        config={"displayModeBar": False, "responsive": True},
                    ),
                    dcc.Graph(
                        id="fleet-metrics-floating-seasonal-chart",
                        style={"height": f"{SEASONAL_CHART_HEIGHT}px"},
                        config={"displayModeBar": False, "responsive": True},
                    ),
                ],
                className="fleet-metrics-seasonal-grid",
            ),
        ],
        style={
            **SECTION_STYLE,
            "margin": "14px 12px 12px",
            "padding": "10px",
        },
    )


def _diversion_seasonality_section():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Diversion seasonality", style={"margin": "0", "fontSize": "16px"}),
                    html.Div(
                        "Weekly Kpler diversion counts by basin. Inbound is above zero, outbound is below zero; internal basin reroutes are excluded.",
                        className="fleet-metrics-table-subtitle",
                    ),
                ],
                className="fleet-metrics-table-heading",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                id="fleet-metrics-diversion-seasonal-chart",
                                style={"height": f"{DIVERSION_SEASONAL_CHART_HEIGHT}px"},
                                config={"displayModeBar": False, "responsive": True},
                            ),
                        ],
                        className="fleet-metrics-diversion-chart-panel",
                    ),
                ],
                className="fleet-metrics-diversion-seasonal-grid",
            ),
        ],
        style={**SECTION_STYLE, "margin": "0 12px 12px", "padding": "12px"},
    )


def _signal_chart_section(title, subtitle, graph_id, height=SIGNAL_REGION_ROW_CHART_HEIGHT):
    return html.Div(
        [
            html.Div(
                [
                    html.H3(title, style={"margin": "0", "fontSize": "16px"}),
                    html.Div(subtitle, className="fleet-metrics-table-subtitle"),
                ],
                className="fleet-metrics-table-heading",
            ),
            dcc.Graph(
                id=graph_id,
                style={"height": f"{height}px"},
                config={"displayModeBar": False, "responsive": True},
            ),
        ],
        style={**SECTION_STYLE, "margin": "0 12px 12px", "padding": "12px"},
    )


def _signal_chart_sections():
    return [
        _signal_chart_section(
            "LNG arrivals by origin",
            "Daily inbound Kpler flows by regional basin. Recent history and forward arrivals are stacked by origin.",
            "fleet-metrics-arrival-pipeline-chart",
        ),
        _signal_chart_section(
            "Fleet utilization",
            "Loaded, ballast, and maintenance vessel counts by regional basin.",
            "fleet-metrics-utilization-chart",
        ),
        _signal_chart_section(
            "Congestion and waiting",
            "Current pressure, waiting-vessel trend, and abnormality versus recent history by regional basin.",
            "fleet-metrics-congestion-signal-chart",
            height=CONGESTION_SIGNAL_CHART_HEIGHT,
        ),
        _signal_chart_section(
            "Freight demand",
            "Loaded ton-miles and average loaded distance by regional basin.",
            "fleet-metrics-freight-signal-chart",
        ),
    ]


def _regional_detail_matrix_section():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Regional fleet metrics matrix", style={"margin": "0", "fontSize": "16px"}),
                    html.Div(
                        "One row per basin using the three detail views: loaded, floating, and floating-days.",
                        className="fleet-metrics-table-subtitle",
                    ),
                ],
                className="fleet-metrics-table-heading",
            ),
            html.Div(
                id="fleet-metrics-region-detail-matrix-legend",
                children=build_region_detail_matrix_legend(pd.DataFrame(), "current_subcontinents"),
            ),
            dcc.Graph(
                id="fleet-metrics-region-detail-matrix-chart",
                style={"height": f"{REGION_DETAIL_MATRIX_HEIGHT}px"},
                config={"displayModeBar": False, "responsive": True},
            ),
        ],
        style={
            **SECTION_STYLE,
            "margin": "14px 12px 12px",
            "padding": "10px",
        },
    )


def _definition_pill_list(items, variant, empty_label):
    display_items = [_area_display_name(item) for item in items if item]
    if not display_items:
        display_items = [empty_label]

    return html.Div(
        [
            html.Span(
                item,
                className=f"fleet-region-definition-pill fleet-region-definition-pill--{variant}",
            )
            for item in display_items
        ],
        className="fleet-region-definition-pill-list",
    )


def _region_definition_group(title, meta, items, variant, empty_label):
    return html.Div(
        [
            html.Div(
                [
                    html.Div(title, className="fleet-region-definition-group-title"),
                    html.Div(meta, className="fleet-region-definition-group-meta"),
                ],
                className="fleet-region-definition-group-header",
            ),
            _definition_pill_list(items, variant, empty_label),
        ],
        className="fleet-region-definition-group",
    )


def _region_definition_card(zone_filter):
    region_label = KPLER_FLEET_ZONE_SHORT_LABELS.get(zone_filter, zone_filter)
    zone_members = sorted(KPLER_FLEET_ZONE_MEMBERS.get(zone_filter, set()), key=str.casefold)
    country_members = sorted(KPLER_REGION_COUNTRIES.get(zone_filter, set()), key=str.casefold)
    default_areas = KPLER_FLEET_DEFAULT_AREAS.get((zone_filter, "current_subcontinents"), [])

    if zone_filter == "global":
        zone_members = ["No regional Kpler zone filter", "All LNG fleet metric observations"]
        country_members = ["All countries available in the Kpler aggregate feed"]
        zone_badge = "Global benchmark"
        country_meta = "All markets"
    else:
        zone_badge = f"{len(zone_members)} Kpler areas"
        country_meta = f"{len(country_members)} country anchors"

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H4(region_label, className="fleet-region-definition-title"),
                            html.Div(
                                KPLER_REGION_DEFINITION_SUMMARIES.get(zone_filter, ""),
                                className="fleet-region-definition-summary",
                            ),
                        ],
                        style={"minWidth": 0},
                    ),
                    html.Span(zone_badge, className="fleet-region-definition-badge"),
                ],
                className="fleet-region-definition-card-header",
            ),
            _region_definition_group(
                "Kpler zone filter",
                "Subcontinents, seas, and oceans used to build the basin preset.",
                zone_members,
                "zone",
                "No filter",
            ),
            _region_definition_group(
                "Country anchors",
                country_meta,
                country_members,
                "country",
                "All countries",
            ),
            _region_definition_group(
                "Default chart areas",
                "Default subcontinent split shown before the user changes the area selection.",
                default_areas,
                "area",
                "Dynamic by selected split",
            ),
        ],
        className=f"fleet-region-definition-card fleet-region-definition-card--{zone_filter}",
    )


def _region_definitions_section():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Methodology appendix", className="fleet-region-definitions-kicker"),
                            html.H3("Region definitions used in Fleet Metrics", className="fleet-region-definitions-title"),
                            html.Div(
                                "These presets combine destination markets with nearby sea and ocean areas where LNG can wait before delivery. "
                                "They are designed for short-term basin pressure monitoring, not for additive regional accounting.",
                                className="fleet-region-definitions-subtitle",
                            ),
                        ],
                        style={"minWidth": 0},
                    ),
                    html.Div(
                        [
                            html.Span("Regional basins are not additive", className="fleet-region-definition-principle"),
                            html.Span("Sea and ocean waiting areas included", className="fleet-region-definition-principle"),
                            html.Span("Global is the unfiltered benchmark", className="fleet-region-definition-principle"),
                        ],
                        className="fleet-region-definition-principles",
                    ),
                ],
                className="fleet-region-definitions-header",
            ),
            html.Div(
                [_region_definition_card(zone_filter) for zone_filter in KPLER_FLEET_REGION_ORDER],
                className="fleet-region-definition-card-grid",
            ),
        ],
        className="fleet-region-definitions-section",
        style={**SECTION_STYLE, "margin": "0 12px 24px", "padding": "16px"},
    )


layout = html.Div(
    [
        dcc.Store(
            id="fleet-metrics-source-ref-store",
            storage_type="memory",
        ),
        dcc.Store(
            id="fleet-metrics-refresh-status-store",
            storage_type="memory",
        ),
        dcc.Store(
            id="fleet-metrics-render-ready-store",
            storage_type="memory",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Fleet Metrics", className="filter-group-header"),
                        html.Div(
                            "Regional LNG fleet pressure",
                            style={"fontWeight": "700", "fontSize": "15px", "color": "#1e3a5f"},
                        ),
                    ],
                    className="filter-group",
                    style={"minWidth": "180px"},
                ),
                html.Div(
                    [
                        html.Div("Region", className="filter-group-header"),
                        dcc.Tabs(
                            id="fleet-metrics-region-tabs",
                            value=KPLER_FLEET_DEFAULT_ZONE_FILTER,
                            children=[
                                dcc.Tab(
                                    label=option.get("tab_label", option["label"]),
                                    value=option["value"],
                                    style={"fontSize": "12px", "padding": "7px 8px", "whiteSpace": "nowrap"},
                                    selected_style={"fontSize": "12px", "padding": "7px 8px", "whiteSpace": "nowrap"},
                                )
                                for option in KPLER_FLEET_ZONE_OPTIONS
                            ],
                            style={"height": "34px"},
                        ),
                    ],
                    style={"flex": "1", "minWidth": "520px"},
                ),
                html.Div(
                    [
                        html.Div("Split", className="filter-group-header"),
                        dcc.Dropdown(
                            id="fleet-metrics-split-dropdown",
                            options=KPLER_FLEET_SPLIT_OPTIONS,
                            value="current_subcontinents",
                            clearable=False,
                            style={"width": "180px"},
                        ),
                    ],
                    className="filter-group",
                ),
                html.Div(
                    [
                        html.Div("Areas", className="filter-group-header"),
                        dcc.Dropdown(
                            id="fleet-metrics-area-dropdown",
                            multi=True,
                            placeholder="Select areas",
                            style={"minWidth": "360px"},
                        ),
                    ],
                    className="filter-group",
                    style={"flex": "1", "minWidth": "360px"},
                ),
                html.Div(
                    [
                        html.Div("Date Range", className="filter-group-header"),
                        dcc.DatePickerRange(
                            id="fleet-metrics-date-range",
                            start_date=KPLER_FLEET_DEFAULT_START_DATE.isoformat(),
                            end_date=dt.date.today().isoformat(),
                            display_format="YYYY-MM-DD",
                            min_date_allowed=KPLER_FLEET_DEFAULT_START_DATE.isoformat(),
                            minimum_nights=0,
                        ),
                    ],
                    className="filter-group",
                    style={"minWidth": "280px"},
                ),
            ],
            className="professional-section-header",
            style={"display": "flex", "gap": "12px", "alignItems": "flex-end", "flexWrap": "wrap"},
        ),
        _seasonality_section(),
        _regional_detail_matrix_section(),
        _diversion_seasonality_section(),
        html.Div(
            [
                html.Div(id="fleet-metrics-status-strip", style={"marginBottom": "12px"}),
                html.Div(
                    [
                        html.Div(id="fleet-metrics-summary-cards", style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(145px, 1fr))",
                            "gap": "12px",
                        }),
                        html.Div(id="fleet-metrics-price-card"),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(0, 5fr) minmax(190px, 1fr)",
                        "gap": "12px",
                        "alignItems": "stretch",
                    },
                ),
            ],
            style={**SECTION_STYLE, "margin": "14px 12px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Regional comparison", style={"margin": "0", "fontSize": "16px"}),
                                html.Div(
                                    "Completed week, momentum, and percentile.",
                                    className="fleet-metrics-table-subtitle",
                                ),
                            ],
                            className="fleet-metrics-table-heading",
                        ),
                        _ag_grid_table(
                            "fleet-metrics-region-comparison-table",
                            FLEET_METRICS_REGION_COMPARISON_COLUMN_DEFS,
                            page_size=6,
                            height=260,
                            pagination=False,
                            extra_class="fleet-metrics-grid--comparison",
                            column_size=None,
                        ),
                    ],
                    style={**SECTION_STYLE, "minWidth": 0, "padding": "12px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Global signals", style={"margin": "0", "fontSize": "16px"}),
                                html.Div(
                                    "Flows, utilization, congestion, diversions, freight.",
                                    className="fleet-metrics-table-subtitle",
                                ),
                            ],
                            className="fleet-metrics-table-heading",
                        ),
                        _ag_grid_table(
                            "fleet-metrics-global-signals-table",
                            FLEET_METRICS_GLOBAL_SIGNALS_COLUMN_DEFS,
                            page_size=6,
                            height=260,
                            pagination=False,
                            extra_class="fleet-metrics-grid--signals",
                            column_size=None,
                        ),
                    ],
                    style={**SECTION_STYLE, "minWidth": 0, "padding": "12px"},
                ),
            ],
            className="fleet-metrics-top-table-grid",
            style={"margin": "0 12px 12px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Selected basin signal cards", style={"margin": "0", "fontSize": "16px"}),
                        html.Div(
                            "Kpler aggregate signal summary for the active regional tab.",
                            className="fleet-metrics-table-subtitle",
                        ),
                    ],
                    className="fleet-metrics-table-heading",
                ),
                html.Div(id="fleet-metrics-global-signal-cards", style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(155px, 1fr))",
                    "gap": "12px",
                }),
            ],
            style={**SECTION_STYLE, "margin": "0 12px 12px", "padding": "12px"},
        ),
        *_signal_chart_sections(),
        html.Div(
            [
                html.H3("Top area movers", style={"margin": "0 0 10px", "fontSize": "16px"}),
                _ag_grid_table(
                    "fleet-metrics-movers-table",
                    FLEET_METRICS_MOVERS_COLUMN_DEFS,
                    page_size=6,
                    height=250,
                ),
            ],
            style={**SECTION_STYLE, "margin": "0 12px 12px"},
        ),
        _region_definitions_section(),
    ],
    style={"backgroundColor": "#f8fafc", "paddingBottom": "24px"},
)


def _pack_fleet_metrics_source_bundle(source_bundle):
    if not _fleet_arrow_source_enabled():
        return source_bundle
    packed = _pack_dataframe_mapping(
        source_bundle,
        dataframe_keys=FLEET_METRICS_SOURCE_FRAME_KEYS,
    )
    packed["format"] = "fleet-metrics-source-arrow-v1"
    return packed


def _unpack_fleet_metrics_source_bundle(source_bundle):
    if not isinstance(source_bundle, dict):
        return source_bundle
    return _unpack_dataframe_mapping(
        source_bundle,
        dataframe_keys=FLEET_METRICS_SOURCE_FRAME_KEYS,
    )


def _resolve_fleet_metrics_source_bundle(
    source_reference,
    *,
    decode_frames=False,
):
    if not (
        _is_snapshot_reference(source_reference)
        and source_reference.get("namespace")
        in FLEET_METRICS_SOURCE_NAMESPACES
        and _snapshot_is_resolvable(source_reference)
    ):
        raise RuntimeError("Fleet metrics source snapshot is unavailable")
    source_bundle = _resolve_snapshot(
        source_reference,
        engine,
        expected_namespace=source_reference["namespace"],
    )
    if decode_frames:
        source_bundle = _unpack_fleet_metrics_source_bundle(source_bundle)
    if (
        not isinstance(source_bundle, dict)
        or not isinstance(source_bundle.get("context"), dict)
        or not {
            "start_date",
            "end_date",
            "split_dimension",
            "today",
        }.issubset(source_bundle["context"])
    ):
        raise TypeError("Fleet metrics source snapshot is invalid")
    return source_bundle


@callback(
    Output("fleet-metrics-area-dropdown", "options"),
    Output("fleet-metrics-area-dropdown", "value"),
    Input("fleet-metrics-source-ref-store", "data"),
    Input("fleet-metrics-region-tabs", "value"),
    prevent_initial_call=False,
)
def update_area_options(source_reference, zone_filter):
    zone_filter = zone_filter or KPLER_FLEET_DEFAULT_ZONE_FILTER
    try:
        source_bundle = _resolve_fleet_metrics_source_bundle(
            source_reference
        )
        context = source_bundle.get("context") or {}
        split_dimension = context.get(
            "split_dimension",
            "current_subcontinents",
        )
        areas = (
            source_bundle.get("area_options_by_region", {}).get(zone_filter)
            or _default_area_candidates(zone_filter, split_dimension)
        )
    except Exception:
        split_dimension = "current_subcontinents"
        areas = _default_area_candidates(zone_filter, split_dimension)
    options = [{"label": _area_display_name(area), "value": area} for area in areas]

    default_candidates = _default_area_candidates(zone_filter, split_dimension)
    default_values = [area for area in default_candidates if area in areas]
    if not default_values:
        default_values = areas[:9]

    return options, default_values


def _build_fleet_metrics_source_bundle(
    *,
    start_date_val,
    end_date_val,
    split_dimension,
    today,
    source_state,
):
    signal_diversion_start = today - dt.timedelta(days=45)
    signal_diversion_end = today + dt.timedelta(days=60)
    overlaps_signal_window = (
        start_date_val <= signal_diversion_end
        and end_date_val >= signal_diversion_start
    )

    if overlaps_signal_window:
        diversion_tasks = {
            "all_diversions": lambda: fetch_recent_diversions(
                start_date=min(start_date_val, signal_diversion_start),
                end_date=max(end_date_val, signal_diversion_end),
            )
        }
    else:
        diversion_tasks = {
            "signal_diversions": lambda: fetch_recent_diversions(
                start_date=signal_diversion_start,
                end_date=signal_diversion_end,
            ),
            "diversion_history": lambda: fetch_recent_diversions(
                start_date=start_date_val,
                end_date=end_date_val,
            ),
        }

    tasks = {
        "detail_matrix_floating_days": lambda: fetch_fleet_metrics_weekly(
            split_dimension="floating_days",
            start_date=start_date_val,
            end_date=end_date_val,
            zone_filters=KPLER_FLEET_DIVERSION_REGION_ORDER,
            metrics=("floating_storage",),
        ),
        "regional_signals": lambda: fetch_regional_signals(
            start_date=start_date_val,
            end_date=end_date_val,
        ),
        "detail_matrix_all_weekly": lambda: fetch_fleet_metrics_weekly(
            split_dimension=split_dimension,
            start_date=start_date_val,
            end_date=end_date_val,
            zone_filters=KPLER_FLEET_REGION_ORDER,
            metrics=("loaded_vessels", "floating_storage"),
        ),
        "summary_daily": lambda: fetch_region_metric_totals_daily(
            start_date=start_date_val,
            end_date=end_date_val,
        ),
        "area_options_by_region": lambda: fetch_all_area_options(
            split_dimension
        ),
        "price_context": fetch_price_context,
        **diversion_tasks,
    }

    results = {}
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="fleet-metrics") as executor:
        futures = {name: executor.submit(task) for name, task in tasks.items()}
        for name in tasks:
            results[name] = futures[name].result()

    if overlaps_signal_window:
        all_diversions_df = results["all_diversions"]
        results["signal_diversions"] = _filter_diversions_by_event_window(
            all_diversions_df,
            signal_diversion_start,
            signal_diversion_end,
        )
        results["diversion_history"] = _filter_diversions_by_event_window(
            all_diversions_df,
            start_date_val,
            end_date_val,
        )

    detail_matrix_all = results.pop("detail_matrix_all_weekly")
    results["detail_matrix_weekly"] = detail_matrix_all[
        detail_matrix_all["zone_filter"].isin(KPLER_FLEET_DIVERSION_REGION_ORDER)
    ].copy()
    results["global_area_weekly"] = detail_matrix_all[
        detail_matrix_all["zone_filter"] == "global"
    ].copy()

    results["fleet_checked_at"] = (
        source_state.get("fleet_checked_at") or source_state.get("fleet_upload")
    )
    results["signal_checked_at"] = (
        source_state.get("signal_checked_at") or source_state.get("signal_upload")
    )
    results["fleet_changed_at"] = source_state.get("fleet_changed_at")
    results["signal_changed_at"] = source_state.get("signal_changed_at")
    results["context"] = {
        "start_date": start_date_val.isoformat(),
        "end_date": end_date_val.isoformat(),
        "split_dimension": split_dimension,
        "today": today.isoformat(),
    }
    return results


def _build_fleet_metrics_common_render(
    source_bundle,
    *,
    start_date_val,
    end_date_val,
    split_dimension,
):
    """Build zone-independent derived data and figures once per source bundle."""
    summary_weekly = derive_weekly_fleet_metrics(source_bundle["summary_daily"])
    summaries = compute_region_summaries(summary_weekly, end_date_val)
    regional_signals = source_bundle["regional_signals"]
    signal_summaries = compute_global_signal_summaries(
        regional_signals,
        source_bundle["signal_diversions"],
        end_date_val,
    )
    detail_matrix_weekly = source_bundle["detail_matrix_weekly"]
    detail_matrix_floating_days = source_bundle["detail_matrix_floating_days"]

    figure_tasks = {
        "detail_matrix_fig": lambda: build_region_detail_matrix(
            detail_matrix_weekly,
            detail_matrix_floating_days,
            split_dimension,
        ),
        "arrival_pipeline_fig": lambda: build_arrival_origin_region_row(regional_signals),
        "utilization_fig": lambda: build_utilization_region_row(
            regional_signals,
            start_date_val,
            end_date_val,
        ),
        "congestion_signal_fig": lambda: build_congestion_region_row(
            regional_signals,
            start_date_val,
            end_date_val,
        ),
        "freight_signal_fig": lambda: build_freight_region_row(
            regional_signals,
            start_date_val,
            end_date_val,
        ),
        "diversion_seasonal_fig": lambda: build_diversion_seasonal_chart(
            source_bundle["diversion_history"],
            start_date=start_date_val,
            end_date=end_date_val,
        ),
        "loaded_seasonal_fig": lambda: build_region_seasonal_chart(
            summary_weekly,
            "loaded_vessels",
        ),
        "floating_seasonal_fig": lambda: build_region_seasonal_chart(
            summary_weekly,
            "floating_storage",
        ),
        "detail_matrix_legend": lambda: build_region_detail_matrix_legend(
            detail_matrix_weekly,
            split_dimension,
        ),
    }
    figures = {}
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="fleet-render") as executor:
        futures = {name: executor.submit(task) for name, task in figure_tasks.items()}
        for name in figure_tasks:
            figures[name] = futures[name].result()

    return {
        "summary_weekly": summary_weekly,
        "summaries": summaries,
        "regional_signals": regional_signals,
        "signal_summaries": signal_summaries,
        "detail_matrix_weekly": detail_matrix_weekly,
        "price_card": build_price_card(source_bundle["price_context"]),
        **figures,
    }


def _figure_to_snapshot(figure):
    return json.loads(
        pio.to_json(
            figure,
            validate=False,
            pretty=False,
            remove_uids=False,
        )
    )


def _figure_from_snapshot(value):
    if isinstance(value, go.Figure):
        return value
    if not isinstance(value, dict):
        raise TypeError("Fleet render snapshot figure is invalid")
    # Dash accepts the final Plotly mapping directly. Avoid reparsing the
    # immutable artifact into graph objects in every web worker.
    return value


def _fleet_render_dependency(source_reference):
    return {
        "namespace": source_reference.get("namespace"),
        "source_key": source_reference.get("source_key"),
        "revision": source_reference.get("revision"),
    }


def _fleet_render_bundle_source_key(source_reference):
    return _build_source_key(
        FLEET_METRICS_RENDER_BUNDLE_NAMESPACE,
        _fleet_render_dependency(source_reference),
        FLEET_METRICS_RENDER_SCHEMA_VERSION,
    )


def _fleet_render_artifact_source_key(
    bundle_source_key,
    section,
):
    return _build_source_key(
        f"fleet-metrics-render-{section}-v1",
        bundle_source_key,
        section,
        FLEET_METRICS_RENDER_SCHEMA_VERSION,
    )


def _build_fleet_render_artifacts(source_reference, source_bundle):
    context = dict(source_bundle.get("context") or {})
    split_dimension = context.get(
        "split_dimension",
        "current_subcontinents",
    )
    start_date_val = pd.to_datetime(context["start_date"]).date()
    end_date_val = pd.to_datetime(context["end_date"]).date()
    common_render = _build_fleet_metrics_common_render(
        source_bundle,
        start_date_val=start_date_val,
        end_date_val=end_date_val,
        split_dimension=split_dimension,
    )

    detail_matrix_weekly = common_render["detail_matrix_weekly"]
    movers_by_region = {}
    for zone_filter in KPLER_FLEET_REGION_ORDER:
        if zone_filter in KPLER_FLEET_DIVERSION_REGION_ORDER:
            area_weekly = detail_matrix_weekly[
                detail_matrix_weekly["zone_filter"] == zone_filter
            ].copy()
        else:
            area_weekly = source_bundle["global_area_weekly"]
        movers_by_region[zone_filter] = build_movers_rows(area_weekly)

    summary_payload = {
        "format": FLEET_METRICS_RENDER_SUMMARY_FORMAT,
        "source_reference": _fleet_render_dependency(source_reference),
        "context": context,
        "fleet_checked_at": source_bundle.get(
            "fleet_checked_at",
            source_bundle.get("upload_timestamp"),
        ),
        "signal_checked_at": source_bundle.get(
            "signal_checked_at",
            source_bundle.get("signal_upload_timestamp"),
        ),
        "fleet_changed_at": source_bundle.get("fleet_changed_at"),
        "signal_changed_at": source_bundle.get("signal_changed_at"),
        "upload_timestamp": source_bundle.get("upload_timestamp"),
        "signal_upload_timestamp": source_bundle.get(
            "signal_upload_timestamp"
        ),
        "price_context": source_bundle.get("price_context") or {},
        "summaries": common_render["summaries"],
        "signal_summaries": common_render["signal_summaries"],
        "movers_by_region": movers_by_region,
    }
    signals_payload = {
        "format": FLEET_METRICS_RENDER_SIGNALS_FORMAT,
        "source_reference": _fleet_render_dependency(source_reference),
        "figures": {
            name: _figure_to_snapshot(common_render[name])
            for name in (
                "arrival_pipeline_fig",
                "utilization_fig",
                "congestion_signal_fig",
                "freight_signal_fig",
                "diversion_seasonal_fig",
            )
        },
    }
    color_maps = {
        zone_filter: _detail_matrix_area_color_map(
            detail_matrix_weekly,
            zone_filter,
            split_dimension,
        )
        for zone_filter in KPLER_FLEET_DIVERSION_REGION_ORDER
    }
    detail_payload = {
        "format": FLEET_METRICS_RENDER_DETAIL_FORMAT,
        "source_reference": _fleet_render_dependency(source_reference),
        "split_dimension": split_dimension,
        "color_maps": color_maps,
        "figures": {
            name: _figure_to_snapshot(common_render[name])
            for name in (
                "loaded_seasonal_fig",
                "floating_seasonal_fig",
                "detail_matrix_fig",
            )
        },
    }
    return summary_payload, signals_payload, detail_payload


def _require_fleet_render_artifact(
    value,
    *,
    expected_format,
):
    if not isinstance(value, dict) or value.get("format") != expected_format:
        raise TypeError("Fleet render snapshot artifact is invalid")
    return value


def _resolve_fleet_render_bundle(bundle_reference):
    if not (
        _is_snapshot_reference(
            bundle_reference,
            FLEET_METRICS_RENDER_BUNDLE_NAMESPACE,
        )
        and _snapshot_is_resolvable(bundle_reference)
    ):
        raise RuntimeError("Fleet render snapshot bundle is unavailable")
    bundle = _resolve_snapshot(
        bundle_reference,
        engine,
        expected_namespace=FLEET_METRICS_RENDER_BUNDLE_NAMESPACE,
    )
    if not (
        isinstance(bundle, dict)
        and bundle.get("format") == FLEET_METRICS_RENDER_BUNDLE_FORMAT
        and isinstance(bundle.get("artifacts"), dict)
    ):
        raise TypeError("Fleet render snapshot bundle is invalid")
    return bundle


def _resolve_fleet_render_artifact(
    bundle_reference,
    section,
):
    bundle = _resolve_fleet_render_bundle(bundle_reference)
    artifact_reference = bundle["artifacts"].get(section)
    contracts = {
        "summary": (
            FLEET_METRICS_RENDER_SUMMARY_NAMESPACE,
            FLEET_METRICS_RENDER_SUMMARY_FORMAT,
        ),
        "signals": (
            FLEET_METRICS_RENDER_SIGNALS_NAMESPACE,
            FLEET_METRICS_RENDER_SIGNALS_FORMAT,
        ),
        "detail": (
            FLEET_METRICS_RENDER_DETAIL_NAMESPACE,
            FLEET_METRICS_RENDER_DETAIL_FORMAT,
        ),
    }
    try:
        namespace, expected_format = contracts[section]
    except KeyError as exc:
        raise ValueError(f"Unknown Fleet render section {section!r}") from exc
    if not (
        _is_snapshot_reference(artifact_reference, namespace)
        and _snapshot_is_resolvable(artifact_reference)
    ):
        raise RuntimeError(f"Fleet render {section} artifact is unavailable")
    artifact = _resolve_snapshot(
        artifact_reference,
        engine,
        expected_namespace=namespace,
    )
    return _require_fleet_render_artifact(
        artifact,
        expected_format=expected_format,
    )


def _get_or_build_fleet_render_bundle(source_reference):
    if not _fleet_render_snapshot_enabled():
        raise RuntimeError("Fleet render snapshots are disabled")
    bundle_source_key = _fleet_render_bundle_source_key(source_reference)
    available = _get_snapshot_if_available(
        engine,
        namespace=FLEET_METRICS_RENDER_BUNDLE_NAMESPACE,
        source_key=bundle_source_key,
    )
    if available is not None:
        return available[0]

    with _snapshot_build_lock(
        FLEET_METRICS_RENDER_BUNDLE_NAMESPACE,
        bundle_source_key,
    ):
        available = _get_snapshot_if_available(
            engine,
            namespace=FLEET_METRICS_RENDER_BUNDLE_NAMESPACE,
            source_key=bundle_source_key,
        )
        if available is not None:
            return available[0]

        source_manifest = _resolve_snapshot_manifest(
            source_reference,
            engine,
            expected_namespace=source_reference["namespace"],
        )
        expected_source_state = source_manifest.get("source_state")
        source_bundle = _unpack_fleet_metrics_source_bundle(
            _resolve_fleet_metrics_source_bundle(source_reference)
        )
        with _stage_snapshot_publication(
            f"fleet-render:{bundle_source_key}"
        ) as publication_stage:
            summary_payload, signals_payload, detail_payload = (
                _build_fleet_render_artifacts(
                    source_reference,
                    source_bundle,
                )
            )
            artifact_specs = {
                "summary": (
                    FLEET_METRICS_RENDER_SUMMARY_NAMESPACE,
                    summary_payload,
                ),
                "signals": (
                    FLEET_METRICS_RENDER_SIGNALS_NAMESPACE,
                    signals_payload,
                ),
                "detail": (
                    FLEET_METRICS_RENDER_DETAIL_NAMESPACE,
                    detail_payload,
                ),
            }
            artifact_references = {}
            for section, (namespace, payload) in artifact_specs.items():
                source_key = _fleet_render_artifact_source_key(
                    bundle_source_key,
                    section,
                )
                reference, _ = _get_or_build_snapshot(
                    engine,
                    namespace=namespace,
                    source_key=source_key,
                    builder=lambda payload=payload: payload,
                    manifest={
                        "section": section,
                        "source_reference": _fleet_render_dependency(
                            source_reference
                        ),
                        "render_schema_version": (
                            FLEET_METRICS_RENDER_SCHEMA_VERSION
                        ),
                    },
                )
                artifact_references[section] = reference

            if (
                isinstance(expected_source_state, dict)
                and _build_source_key(
                    "fleet-metrics-source-state-validation",
                    _fleet_semantic_source_state(
                        _fetch_fleet_metrics_source_state()
                    ),
                )
                != _build_source_key(
                    "fleet-metrics-source-state-validation",
                    expected_source_state,
                )
            ):
                raise RuntimeError(
                    "Fleet metrics sources changed during render precompute"
                )

            return _commit_snapshot_publication_stage(
                publication_stage,
                bundle_namespace=FLEET_METRICS_RENDER_BUNDLE_NAMESPACE,
                bundle_source_key=bundle_source_key,
                bundle_payload={
                    "format": FLEET_METRICS_RENDER_BUNDLE_FORMAT,
                    "source_reference": _fleet_render_dependency(
                        source_reference
                    ),
                    "render_schema_version": (
                        FLEET_METRICS_RENDER_SCHEMA_VERSION
                    ),
                    "artifacts": artifact_references,
                },
                bundle_manifest={
                    "source_reference": _fleet_render_dependency(
                        source_reference
                    ),
                    "source_state": expected_source_state,
                    "render_schema_version": (
                        FLEET_METRICS_RENDER_SCHEMA_VERSION
                    ),
                },
            )


def _get_fleet_metrics_common_render(
    source_key,
    source_bundle,
    *,
    start_date_val,
    end_date_val,
    split_dimension,
):
    with _FLEET_RENDER_CACHE_LOCK:
        cached = _FLEET_COMMON_RENDER_CACHE.get(source_key)
        if cached is not None:
            _FLEET_COMMON_RENDER_CACHE.move_to_end(source_key)
            return cached
        flight = _FLEET_COMMON_RENDER_FLIGHTS.get(source_key)
        if flight is None:
            flight = Future()
            _FLEET_COMMON_RENDER_FLIGHTS[source_key] = flight
            owns_flight = True
        else:
            owns_flight = False

    if not owns_flight:
        return flight.result()

    try:
        prepared = _build_fleet_metrics_common_render(
            source_bundle,
            start_date_val=start_date_val,
            end_date_val=end_date_val,
            split_dimension=split_dimension,
        )
        with _FLEET_RENDER_CACHE_LOCK:
            _FLEET_COMMON_RENDER_CACHE[source_key] = prepared
            _FLEET_COMMON_RENDER_CACHE.move_to_end(source_key)
            while len(_FLEET_COMMON_RENDER_CACHE) > 32:
                _FLEET_COMMON_RENDER_CACHE.popitem(last=False)
        flight.set_result(prepared)
        return prepared
    except BaseException as exc:
        flight.set_exception(exc)
        raise
    finally:
        with _FLEET_RENDER_CACHE_LOCK:
            _FLEET_COMMON_RENDER_FLIGHTS.pop(source_key, None)


def _normalize_fleet_metrics_source_controls(
    split_dimension,
    start_date,
    end_date,
):
    if split_dimension not in {
        option["value"] for option in KPLER_FLEET_SPLIT_OPTIONS
    }:
        split_dimension = "current_subcontinents"
    today = dt.date.today()
    start_date_val = (
        pd.to_datetime(start_date).date()
        if start_date
        else KPLER_FLEET_DEFAULT_START_DATE
    )
    end_date_val = (
        pd.to_datetime(end_date).date()
        if end_date
        else today
    )
    if start_date_val > end_date_val:
        start_date_val, end_date_val = end_date_val, start_date_val
    return split_dimension, start_date_val, end_date_val, today


@callback(
    Output("fleet-metrics-source-ref-store", "data"),
    Output("fleet-metrics-refresh-status-store", "data"),
    Input("fleet-metrics-split-dropdown", "value"),
    Input("fleet-metrics-date-range", "start_date"),
    Input("fleet-metrics-date-range", "end_date"),
    Input("global-refresh-button", "n_clicks"),
    State("fleet-metrics-source-ref-store", "data"),
    prevent_initial_call=False,
)
@log_callback_timing("fleet_metrics.source_load")
def load_fleet_metrics_source(
    split_dimension,
    start_date,
    end_date,
    global_refresh_clicks=None,
    current_source_reference=None,
):
    (
        split_dimension,
        start_date_val,
        end_date_val,
        today,
    ) = _normalize_fleet_metrics_source_controls(
        split_dimension,
        start_date,
        end_date,
    )
    refresh_status = {
        "format": "dashboard-source-refresh-status-v1",
        "refresh_generation": int(global_refresh_clicks or 0),
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    try:
        source_lookup_available = True
        try:
            source_state = _fetch_fleet_metrics_source_state()
        except Exception:
            logger.warning(
                "Fleet snapshot watermark lookup failed; "
                "using live-query fallback",
                exc_info=True,
            )
            source_state = {
                "request_token": dt.datetime.now(
                    dt.timezone.utc
                ).isoformat()
            }
            source_lookup_available = False
        refresh_status["status"] = (
            "checked" if source_lookup_available else "unavailable"
        )
        refresh_status["kpler_freshness"] = (
            _freshness_reference_payload(source_state)
        )

        for _attempt in range(3):
            semantic_source_state = _fleet_semantic_source_state(
                source_state
            )
            source_key = _build_source_key(
                FLEET_METRICS_SNAPSHOT_NAMESPACE,
                semantic_source_state,
                split_dimension,
                start_date_val,
                end_date_val,
                today,
            )
            if (
                _revision_aware_refresh_enabled()
                and _is_snapshot_reference(
                    current_source_reference,
                    FLEET_METRICS_SNAPSHOT_NAMESPACE,
                )
                and current_source_reference.get("source_key")
                == source_key
            ):
                return no_update, refresh_status

            def build_stable_source_bundle():
                source_bundle = _build_fleet_metrics_source_bundle(
                    start_date_val=start_date_val,
                    end_date_val=end_date_val,
                    split_dimension=split_dimension,
                    today=today,
                    source_state=source_state,
                )
                if (
                    source_lookup_available
                    and _build_source_key(
                        "fleet-metrics-source-state-validation",
                        _fleet_semantic_source_state(
                            _fetch_fleet_metrics_source_state()
                        ),
                    )
                    != _build_source_key(
                        "fleet-metrics-source-state-validation",
                        semantic_source_state,
                    )
                ):
                    raise _FleetMetricsSourceChanged
                return _pack_fleet_metrics_source_bundle(source_bundle)

            try:
                source_reference, _source_bundle = _get_or_build_snapshot(
                    engine,
                    namespace=FLEET_METRICS_SNAPSHOT_NAMESPACE,
                    source_key=source_key,
                    builder=build_stable_source_bundle,
                    force=(
                        not _revision_aware_refresh_enabled()
                        and _was_global_refresh_triggered()
                    ),
                    manifest={
                        "start_date": start_date_val.isoformat(),
                        "end_date": end_date_val.isoformat(),
                        "split_dimension": split_dimension,
                        "today": today.isoformat(),
                        "source_state": semantic_source_state,
                        "source_freshness": (
                            _freshness_reference_payload(source_state)
                        ),
                    },
                )
            except _FleetMetricsSourceChanged:
                source_state = _fetch_fleet_metrics_source_state()
                refresh_status["kpler_freshness"] = (
                    _freshness_reference_payload(source_state)
                )
                continue
            if not _snapshot_is_resolvable(source_reference):
                raise RuntimeError(
                    "Fleet metrics source snapshot is unavailable"
                )
            return source_reference, refresh_status
        raise RuntimeError(
            "Fleet metrics sources changed during snapshot construction"
        )
    except Exception as exc:
        logger.exception("Error loading Kpler fleet metrics source")
        refresh_status["status"] = "error"
        return (
            {
                "format": "fleet-metrics-source-error-v1",
                "error": str(exc),
            },
            refresh_status,
        )


def _fleet_region_only_triggered():
    try:
        return ctx.triggered_id == "fleet-metrics-region-tabs"
    except MissingCallbackContextException:
        return False


def _fleet_refresh_status_only_triggered():
    try:
        triggered_ids = {
            item["prop_id"].split(".", 1)[0]
            for item in ctx.triggered
            if item.get("prop_id")
        }
    except MissingCallbackContextException:
        return False
    return triggered_ids == {"fleet-metrics-refresh-status-store"}


def _fleet_refresh_freshness(refresh_status):
    if not isinstance(refresh_status, dict):
        return {}
    freshness = refresh_status.get("kpler_freshness")
    return freshness if isinstance(freshness, dict) else {}


def _fleet_render_response(result, *, region_only):
    if not region_only:
        return result
    return tuple(
        value if index in _FLEET_REGION_OUTPUT_INDICES else no_update
        for index, value in enumerate(result)
    )


@log_callback_timing("fleet_metrics.page_render")
def update_fleet_metrics_page(
    source_reference,
    zone_filter,
    refresh_status=None,
):
    zone_filter = zone_filter or KPLER_FLEET_DEFAULT_ZONE_FILTER
    if zone_filter not in KPLER_FLEET_REGION_ORDER:
        zone_filter = KPLER_FLEET_DEFAULT_ZONE_FILTER
    if _fleet_refresh_status_only_triggered():
        return (no_update,) * 18

    try:
        source_bundle = _unpack_fleet_metrics_source_bundle(
            _resolve_fleet_metrics_source_bundle(source_reference)
        )
        context = source_bundle.get("context") or {}
        split_dimension = context.get(
            "split_dimension",
            "current_subcontinents",
        )
        start_date_val = pd.to_datetime(
            context["start_date"]
        ).date()
        end_date_val = pd.to_datetime(context["end_date"]).date()
        render_source_key = (
            str(source_reference["namespace"]),
            str(source_reference["source_key"]),
            source_reference["revision"],
        )
        render_cache_key = (render_source_key, zone_filter)
        region_only = _fleet_region_only_triggered()
        with _FLEET_RENDER_CACHE_LOCK:
            cached_render = _FLEET_RENDER_CACHE.get(render_cache_key)
            if cached_render is not None:
                _FLEET_RENDER_CACHE.move_to_end(render_cache_key)
                return _fleet_render_response(
                    cached_render,
                    region_only=region_only,
                )

        price_context = source_bundle["price_context"]
        freshness = _fleet_refresh_freshness(refresh_status)
        if not freshness and isinstance(source_reference, dict):
            freshness = source_reference.get("kpler_freshness") or {}
        fleet_checked_at = freshness.get("fleet_checked_at") or source_bundle.get(
            "fleet_checked_at", source_bundle.get("upload_timestamp")
        )
        signal_checked_at = freshness.get("signal_checked_at") or source_bundle.get(
            "signal_checked_at", source_bundle.get("signal_upload_timestamp")
        )
        common_render = _get_fleet_metrics_common_render(
            render_source_key,
            source_bundle,
            start_date_val=start_date_val,
            end_date_val=end_date_val,
            split_dimension=split_dimension,
        )
        summaries = common_render["summaries"]
        selected_summary = summaries.get(zone_filter)
        status_strip = build_status_strip(
            fleet_checked_at,
            signal_checked_at,
            zone_filter,
            price_context,
            freshness.get("fleet_changed_at")
            or source_bundle.get("fleet_changed_at"),
            freshness.get("signal_changed_at")
            or source_bundle.get("signal_changed_at"),
        )
        summary_cards = build_summary_cards(selected_summary)
        price_card = common_render["price_card"]
        comparison_rows = build_comparison_rows(summaries, zone_filter)
        comparison_column_defs = build_compact_column_defs(
            FLEET_METRICS_REGION_COMPARISON_COLUMN_DEFS,
            comparison_rows,
        )
        signal_summaries = common_render["signal_summaries"]
        selected_signal_summary = signal_summaries.get(zone_filter)
        signal_cards = build_global_signal_cards(selected_signal_summary)
        signal_rows = build_global_signal_rows(signal_summaries, zone_filter)
        signal_column_defs = build_compact_column_defs(
            FLEET_METRICS_GLOBAL_SIGNALS_COLUMN_DEFS,
            signal_rows,
        )
        loaded_seasonal_fig = common_render["loaded_seasonal_fig"]
        floating_seasonal_fig = common_render["floating_seasonal_fig"]
        arrival_pipeline_fig = common_render["arrival_pipeline_fig"]
        utilization_fig = common_render["utilization_fig"]
        congestion_signal_fig = common_render["congestion_signal_fig"]
        freight_signal_fig = common_render["freight_signal_fig"]
        diversion_seasonal_fig = common_render["diversion_seasonal_fig"]
        detail_matrix_weekly = common_render["detail_matrix_weekly"]
        if zone_filter in KPLER_FLEET_DIVERSION_REGION_ORDER:
            all_area_weekly = detail_matrix_weekly[
                detail_matrix_weekly["zone_filter"] == zone_filter
            ].copy()
        else:
            all_area_weekly = source_bundle["global_area_weekly"]
        detail_matrix_fig = common_render["detail_matrix_fig"]
        detail_matrix_legend = common_render["detail_matrix_legend"]
        movers_rows = build_movers_rows(all_area_weekly)

        result = (
            status_strip,
            summary_cards,
            price_card,
            comparison_rows,
            comparison_column_defs,
            signal_cards,
            signal_rows,
            signal_column_defs,
            arrival_pipeline_fig,
            utilization_fig,
            congestion_signal_fig,
            freight_signal_fig,
            diversion_seasonal_fig,
            movers_rows,
            loaded_seasonal_fig,
            floating_seasonal_fig,
            detail_matrix_legend,
            detail_matrix_fig,
        )
        with _FLEET_RENDER_CACHE_LOCK:
            _FLEET_RENDER_CACHE[render_cache_key] = result
            _FLEET_RENDER_CACHE.move_to_end(render_cache_key)
            while len(_FLEET_RENDER_CACHE) > 32:
                _FLEET_RENDER_CACHE.popitem(last=False)
        return _fleet_render_response(
            result,
            region_only=region_only,
        )
    except Exception as exc:
        logger.exception("Error loading Kpler fleet metrics page")
        error_fig = _empty_figure(f"Error loading Kpler fleet metrics: {exc}")
        return (
            html.Div(f"Error loading FleetMetrics page: {exc}", style={"color": "#b91c1c"}),
            build_summary_cards(None),
            _build_kpi_card("JKM-TTF prompt spread", "-", "Unavailable", "warning"),
            [],
            build_compact_column_defs(FLEET_METRICS_REGION_COMPARISON_COLUMN_DEFS, []),
            build_global_signal_cards(None),
            [],
            build_compact_column_defs(FLEET_METRICS_GLOBAL_SIGNALS_COLUMN_DEFS, []),
            error_fig,
            error_fig,
            error_fig,
            error_fig,
            error_fig,
            [],
            error_fig,
            error_fig,
            build_region_detail_matrix_legend(
                pd.DataFrame(),
                "current_subcontinents",
            ),
            error_fig,
        )


def _fleet_summary_outputs_from_artifact(
    summary_artifact,
    zone_filter,
    refresh_status=None,
):
    summaries = summary_artifact.get("summaries") or {}
    signal_summaries = summary_artifact.get("signal_summaries") or {}
    price_context = summary_artifact.get("price_context") or {}
    freshness = _fleet_refresh_freshness(refresh_status)
    comparison_rows = build_comparison_rows(summaries, zone_filter)
    signal_rows = build_global_signal_rows(signal_summaries, zone_filter)
    return (
        build_status_strip(
            freshness.get("fleet_checked_at")
            or summary_artifact.get("fleet_checked_at")
            or summary_artifact.get("upload_timestamp"),
            freshness.get("signal_checked_at")
            or summary_artifact.get("signal_checked_at")
            or summary_artifact.get("signal_upload_timestamp"),
            zone_filter,
            price_context,
            freshness.get("fleet_changed_at")
            or summary_artifact.get("fleet_changed_at"),
            freshness.get("signal_changed_at")
            or summary_artifact.get("signal_changed_at"),
        ),
        build_summary_cards(summaries.get(zone_filter)),
        build_price_card(price_context),
        comparison_rows,
        build_compact_column_defs(
            FLEET_METRICS_REGION_COMPARISON_COLUMN_DEFS,
            comparison_rows,
        ),
        build_global_signal_cards(signal_summaries.get(zone_filter)),
        signal_rows,
        build_compact_column_defs(
            FLEET_METRICS_GLOBAL_SIGNALS_COLUMN_DEFS,
            signal_rows,
        ),
        list(
            (summary_artifact.get("movers_by_region") or {}).get(
                zone_filter
            )
            or []
        ),
    )


def _fleet_figure_outputs_from_artifacts(
    signals_artifact,
    detail_artifact,
):
    signal_figures = signals_artifact.get("figures") or {}
    detail_figures = detail_artifact.get("figures") or {}
    split_dimension = detail_artifact.get(
        "split_dimension",
        "current_subcontinents",
    )
    return (
        _figure_from_snapshot(signal_figures["arrival_pipeline_fig"]),
        _figure_from_snapshot(signal_figures["utilization_fig"]),
        _figure_from_snapshot(signal_figures["congestion_signal_fig"]),
        _figure_from_snapshot(signal_figures["freight_signal_fig"]),
        _figure_from_snapshot(signal_figures["diversion_seasonal_fig"]),
        _figure_from_snapshot(detail_figures["loaded_seasonal_fig"]),
        _figure_from_snapshot(detail_figures["floating_seasonal_fig"]),
        build_region_detail_matrix_legend_from_color_maps(
            detail_artifact.get("color_maps") or {},
            split_dimension,
        ),
        _figure_from_snapshot(detail_figures["detail_matrix_fig"]),
    )


def _fleet_summary_error_outputs(exc):
    return (
        html.Div(
            f"Error loading FleetMetrics page: {exc}",
            style={"color": "#b91c1c"},
        ),
        build_summary_cards(None),
        _build_kpi_card(
            "JKM-TTF prompt spread",
            "-",
            "Unavailable",
            "warning",
        ),
        [],
        build_compact_column_defs(
            FLEET_METRICS_REGION_COMPARISON_COLUMN_DEFS,
            [],
        ),
        build_global_signal_cards(None),
        [],
        build_compact_column_defs(
            FLEET_METRICS_GLOBAL_SIGNALS_COLUMN_DEFS,
            [],
        ),
        [],
    )


def _fleet_figure_error_outputs(exc):
    error_fig = _empty_figure(
        f"Error loading Kpler fleet metrics: {exc}"
    )
    return (
        error_fig,
        error_fig,
        error_fig,
        error_fig,
        error_fig,
        error_fig,
        error_fig,
        build_region_detail_matrix_legend_from_color_maps(
            {},
            "current_subcontinents",
        ),
        error_fig,
    )


@log_callback_timing("fleet_metrics.summary_render")
def update_fleet_metrics_summary(
    source_reference,
    zone_filter,
    refresh_status=None,
):
    zone_filter = zone_filter or KPLER_FLEET_DEFAULT_ZONE_FILTER
    if zone_filter not in KPLER_FLEET_REGION_ORDER:
        zone_filter = KPLER_FLEET_DEFAULT_ZONE_FILTER
    try:
        bundle_reference = _get_or_build_fleet_render_bundle(
            source_reference
        )
        summary_artifact = _resolve_fleet_render_artifact(
            bundle_reference,
            "summary",
        )
        summary_outputs = _fleet_summary_outputs_from_artifact(
            summary_artifact,
            zone_filter,
            refresh_status,
        )
        if _fleet_refresh_status_only_triggered():
            return (
                summary_outputs[0],
                *((no_update,) * 9),
            )
        if _fleet_region_only_triggered():
            summary_outputs = tuple(
                no_update if index in {2, 4, 7} else value
                for index, value in enumerate(summary_outputs)
            )
            return (*summary_outputs, no_update)
        return (*summary_outputs, bundle_reference)
    except Exception as exc:
        logger.exception("Error loading staged Fleet metrics summary")
        return (
            *_fleet_summary_error_outputs(exc),
            {
                "format": "fleet-metrics-render-error-v1",
                "error": str(exc),
            },
        )


@log_callback_timing("fleet_metrics.figure_render")
def update_fleet_metrics_figures(bundle_reference):
    if not bundle_reference:
        raise PreventUpdate
    if (
        isinstance(bundle_reference, dict)
        and bundle_reference.get("format")
        == "fleet-metrics-render-error-v1"
    ):
        return _fleet_figure_error_outputs(
            bundle_reference.get("error") or "unavailable"
        )
    try:
        signals_artifact = _resolve_fleet_render_artifact(
            bundle_reference,
            "signals",
        )
        detail_artifact = _resolve_fleet_render_artifact(
            bundle_reference,
            "detail",
        )
        return _fleet_figure_outputs_from_artifacts(
            signals_artifact,
            detail_artifact,
        )
    except Exception as exc:
        logger.exception("Error loading staged Fleet metrics figures")
        return _fleet_figure_error_outputs(exc)


@log_callback_timing("fleet_metrics.render_snapshot_page")
def update_fleet_metrics_page_from_render_snapshot(
    source_reference,
    zone_filter,
    refresh_status=None,
):
    """Serve the legacy 18-output contract from immutable render artifacts."""

    zone_filter = zone_filter or KPLER_FLEET_DEFAULT_ZONE_FILTER
    if zone_filter not in KPLER_FLEET_REGION_ORDER:
        zone_filter = KPLER_FLEET_DEFAULT_ZONE_FILTER
    try:
        bundle_reference = _get_or_build_fleet_render_bundle(
            source_reference
        )
        summary_artifact = _resolve_fleet_render_artifact(
            bundle_reference,
            "summary",
        )
        summary_outputs = _fleet_summary_outputs_from_artifact(
            summary_artifact,
            zone_filter,
            refresh_status,
        )
        if _fleet_refresh_status_only_triggered():
            return (
                summary_outputs[0],
                *((no_update,) * 17),
            )
        signals_artifact = _resolve_fleet_render_artifact(
            bundle_reference,
            "signals",
        )
        detail_artifact = _resolve_fleet_render_artifact(
            bundle_reference,
            "detail",
        )
        figure_outputs = _fleet_figure_outputs_from_artifacts(
            signals_artifact,
            detail_artifact,
        )
        result = (
            *summary_outputs[:8],
            *figure_outputs[:5],
            summary_outputs[8],
            *figure_outputs[5:],
        )
        return _fleet_render_response(
            result,
            region_only=_fleet_region_only_triggered(),
        )
    except Exception as exc:
        logger.exception("Error loading Fleet render snapshot page")
        summary_outputs = _fleet_summary_error_outputs(exc)
        figure_outputs = _fleet_figure_error_outputs(exc)
        return (
            *summary_outputs[:8],
            *figure_outputs[:5],
            summary_outputs[8],
            *figure_outputs[5:],
        )


def precompute_default_fleet_metrics():
    """Build the exact default-navigation source and render snapshots."""

    today = dt.date.today()
    source_reference, refresh_status = load_fleet_metrics_source(
        "current_subcontinents",
        KPLER_FLEET_DEFAULT_START_DATE.isoformat(),
        today.isoformat(),
        0,
        None,
    )
    if not (
        _is_snapshot_reference(source_reference)
        and _snapshot_is_resolvable(source_reference)
    ):
        raise RuntimeError("Default Fleet source snapshot could not be prepared")
    bundle_reference = _get_or_build_fleet_render_bundle(source_reference)
    bundle = _resolve_fleet_render_bundle(bundle_reference)
    for section in ("summary", "signals", "detail"):
        _resolve_fleet_render_artifact(bundle_reference, section)
    return {
        "format": "fleet-metrics-precompute-result-v1",
        "status": "ready",
        "source_reference": source_reference,
        "bundle_reference": bundle_reference,
        "artifact_references": dict(bundle["artifacts"]),
        "refresh_status": refresh_status,
    }


_FLEET_SUMMARY_CALLBACK_OUTPUTS = (
    Output("fleet-metrics-status-strip", "children"),
    Output("fleet-metrics-summary-cards", "children"),
    Output("fleet-metrics-price-card", "children"),
    Output("fleet-metrics-region-comparison-table", "rowData"),
    Output("fleet-metrics-region-comparison-table", "columnDefs"),
    Output("fleet-metrics-global-signal-cards", "children"),
    Output("fleet-metrics-global-signals-table", "rowData"),
    Output("fleet-metrics-global-signals-table", "columnDefs"),
    Output("fleet-metrics-movers-table", "rowData"),
)
_FLEET_FIGURE_CALLBACK_OUTPUTS = (
    Output("fleet-metrics-arrival-pipeline-chart", "figure"),
    Output("fleet-metrics-utilization-chart", "figure"),
    Output("fleet-metrics-congestion-signal-chart", "figure"),
    Output("fleet-metrics-freight-signal-chart", "figure"),
    Output("fleet-metrics-diversion-seasonal-chart", "figure"),
    Output("fleet-metrics-loaded-seasonal-chart", "figure"),
    Output("fleet-metrics-floating-seasonal-chart", "figure"),
    Output("fleet-metrics-region-detail-matrix-legend", "children"),
    Output("fleet-metrics-region-detail-matrix-chart", "figure"),
)


if _fleet_staged_render_enabled() and _fleet_render_snapshot_enabled():
    callback(
        *_FLEET_SUMMARY_CALLBACK_OUTPUTS,
        Output("fleet-metrics-render-ready-store", "data"),
        Input("fleet-metrics-source-ref-store", "data"),
        Input("fleet-metrics-region-tabs", "value"),
        Input("fleet-metrics-refresh-status-store", "data"),
        prevent_initial_call=False,
    )(update_fleet_metrics_summary)
    callback(
        *_FLEET_FIGURE_CALLBACK_OUTPUTS,
        Input("fleet-metrics-render-ready-store", "data"),
        prevent_initial_call=False,
    )(update_fleet_metrics_figures)
elif _fleet_render_snapshot_enabled():
    callback(
        *_FLEET_SUMMARY_CALLBACK_OUTPUTS[:8],
        *_FLEET_FIGURE_CALLBACK_OUTPUTS[:5],
        _FLEET_SUMMARY_CALLBACK_OUTPUTS[8],
        *_FLEET_FIGURE_CALLBACK_OUTPUTS[5:],
        Input("fleet-metrics-source-ref-store", "data"),
        Input("fleet-metrics-region-tabs", "value"),
        Input("fleet-metrics-refresh-status-store", "data"),
        prevent_initial_call=False,
    )(update_fleet_metrics_page_from_render_snapshot)
else:
    callback(
        *_FLEET_SUMMARY_CALLBACK_OUTPUTS[:8],
        *_FLEET_FIGURE_CALLBACK_OUTPUTS[:5],
        _FLEET_SUMMARY_CALLBACK_OUTPUTS[8],
        *_FLEET_FIGURE_CALLBACK_OUTPUTS[5:],
        Input("fleet-metrics-source-ref-store", "data"),
        Input("fleet-metrics-region-tabs", "value"),
        Input("fleet-metrics-refresh-status-store", "data"),
        prevent_initial_call=False,
    )(update_fleet_metrics_page)
