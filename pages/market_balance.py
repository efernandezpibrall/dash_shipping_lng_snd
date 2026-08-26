from __future__ import annotations

import datetime as dt
from io import BytesIO
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme
from utils.ag_grid_tables import create_ag_grid_from_datatable
from plotly.subplots import make_subplots

from utils.balance_time import TIME_GROUP_LABELS, normalize_time_group
from utils.balance_components import create_balance_empty_state as _create_empty_state
from utils.snapshot_controls import (
    build_ea_upload_dropdown_options as _build_ea_run_dropdown_options,
    deserialize_snapshot_value as _deserialize_snapshot_value,
    format_metadata_timestamp as _format_metadata_timestamp,
    resolve_snapshot_control_values as _resolve_snapshot_control_values,
)
from utils.dashboard_snapshot_cache import (
    SnapshotUnavailable as _SnapshotUnavailable,
    build_source_key as _build_source_key,
    decode_snapshot_payload as _decode_snapshot_payload,
    encode_snapshot_payload as _encode_snapshot_payload,
    get_or_build_snapshot as _get_or_build_snapshot,
    is_snapshot_reference as _is_snapshot_reference,
    resolve_snapshot as _resolve_snapshot,
    snapshot_is_resolvable as _snapshot_is_resolvable,
    snapshot_is_shared as _snapshot_is_shared,
    was_global_refresh_triggered as _was_global_refresh_triggered,  # noqa: F401 - compatibility hook
)
from utils.provider_flow_snapshot import fetch_provider_flow_source_state as _fetch_provider_flow_source_state
from utils.historical_comparison_snapshot import (
    get_historical_comparison_frame as _get_historical_comparison_frame,
)
from utils.ea_run_interface import normalize_ea_run_id
from utils.performance import log_callback_timing
from utils.performance_flags import (
    revision_aware_refresh_enabled as _revision_aware_refresh_enabled,
)
from utils.market_balance_data import (
    COUNTRY_GROUP_LABELS,
    build_period_delta_table,
    deserialize_frame as _deserialize_frame,
    fetch_country_balance_meta_payload,
    fetch_country_balance_payload,
    fetch_net_balance_for_ea_upload,
    fetch_net_balance_for_woodmac_publications,
    fetch_provider_overview_payload,
    fetch_trade_balance_payload,
    serialize_frame,
    engine,
)

if TYPE_CHECKING:
    import dash_ag_grid as dag
from utils.table_styles import (
    StandardTableStyleManager,
    format_table_cell_value_1dp as _format_table_cell_value,
)


EXPORT_BUTTON_STYLE = {
    "marginLeft": "12px",
    "padding": "3px 10px",
    "backgroundColor": "#28a745",
    "color": "white",
    "border": "none",
    "borderRadius": "3px",
    "cursor": "pointer",
    "fontWeight": "bold",
    "fontSize": "11px",
}

STICKY_CONTROL_SHELL_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "gap": "10px",
    "padding": "3px 8px 3px 12px",
    "backgroundColor": "#ffffff",
    "border": "1px solid #dbe4ee",
    "borderRadius": "999px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.05)",
}

MARKET_BALANCE_STICKY_HEADER_STYLE = {
    "display": "flex",
    "gap": "12px",
    "alignItems": "flex-end",
    "flexWrap": "wrap",
}

MARKET_BALANCE_REFERENCE_NAMESPACES = frozenset(
    {
        "market-balance-overview-v2",
        "market-balance-trade-v1",
        "market-balance-country-meta-v2",
    }
)
MARKET_BALANCE_SNAPSHOT_FORMATS = {
    namespace: f"{namespace}-zlib-json-v1"
    for namespace in MARKET_BALANCE_REFERENCE_NAMESPACES
}
MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE = (
    "Cached market-balance data is unavailable. Click the global Refresh "
    "button to reload it."
)


def _prepare_market_store_payload(namespace, payload):
    return {
        "format": MARKET_BALANCE_SNAPSHOT_FORMATS[namespace],
        "payload": _encode_snapshot_payload(payload),
    }


def _decode_market_store_payload(namespace, payload):
    expected_format = MARKET_BALANCE_SNAPSHOT_FORMATS.get(namespace)
    if not (
        expected_format
        and isinstance(payload, dict)
        and payload.get("format") == expected_format
    ):
        if (
            isinstance(payload, dict)
            and payload.get("format") in MARKET_BALANCE_SNAPSHOT_FORMATS.values()
        ):
            raise _SnapshotUnavailable(
                MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE
            )
        return payload

    try:
        encoded_payload = payload["payload"]
        if not isinstance(encoded_payload, (bytes, bytearray, memoryview)):
            raise TypeError("encoded market-balance payload is not bytes")
        return _decode_snapshot_payload(encoded_payload)
    except Exception as exc:
        raise _SnapshotUnavailable(
            MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE
        ) from exc


def _resolve_market_store(store_payload, namespace):
    if store_payload is None:
        raise _SnapshotUnavailable(
            MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE
        )
    if (
        _is_snapshot_reference(store_payload)
        and not _is_snapshot_reference(store_payload, namespace)
    ):
        raise _SnapshotUnavailable(
            MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE
        )
    try:
        resolved = _resolve_snapshot(
            store_payload,
            engine,
            expected_namespace=namespace,
        )
        if _is_snapshot_reference(resolved):
            raise _SnapshotUnavailable(
                MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE
            )
        return _decode_market_store_payload(namespace, resolved)
    except _SnapshotUnavailable as exc:
        raise _SnapshotUnavailable(
            MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE
        ) from exc


def _load_cached_market_store(namespace, key_parts, builder, source_state=None):
    if source_state is None:
        try:
            source_state = _fetch_provider_flow_source_state()
        except Exception:
            source_state = {"request_token": dt.datetime.now(dt.timezone.utc).isoformat()}
    source_key = _build_source_key(namespace, source_state, key_parts)
    converted_namespace = namespace in MARKET_BALANCE_REFERENCE_NAMESPACES
    reference, payload = _get_or_build_snapshot(
        engine,
        namespace=namespace,
        source_key=source_key,
        builder=(
            lambda: _prepare_market_store_payload(namespace, builder())
            if converted_namespace
            else builder()
        ),
        manifest={"source_state": source_state, "filters": key_parts},
    )
    if converted_namespace:
        if not _snapshot_is_resolvable(reference):
            raise _SnapshotUnavailable(
                MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE
            )
        return reference
    return reference if _snapshot_is_shared(reference) else payload

STICKY_RADIO_LABEL_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "marginRight": "10px",
    "fontSize": "12px",
    "fontWeight": "600",
    "color": "#334155",
}

TAB_STYLE = {
    "padding": "12px 16px",
    "fontWeight": "600",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "borderTop": "2px solid #2E86C1",
    "backgroundColor": "#ffffff",
    "color": "#1B4F72",
}

TIME_GROUP_OPTIONS = [
    {"label": TIME_GROUP_LABELS["monthly"], "value": "monthly"},
    {"label": TIME_GROUP_LABELS["quarterly"], "value": "quarterly"},
    {"label": TIME_GROUP_LABELS["yearly"], "value": "yearly"},
    {"label": TIME_GROUP_LABELS["season"], "value": "season"},
]

TRADE_UNIT_OPTIONS = [
    {"label": "bcm", "value": "bcm"},
    {"label": "Mt", "value": "mt"},
    {"label": "mcm/d", "value": "mcm_d"},
]

COUNTRY_GROUP_OPTIONS = [
    {"label": label, "value": value}
    for value, label in COUNTRY_GROUP_LABELS.items()
]

DIFF_TYPE_OPTIONS = [
    {"label": "Percentage (%)", "value": "percentage"},
    {"label": "Absolute", "value": "absolute"},
]

MAINTENANCE_METRIC_OPTIONS = [
    {"label": "Total", "value": "Total"},
    {"label": "Planned", "value": "Planned"},
    {"label": "Unplanned", "value": "Unplanned"},
]

COUNTRY_LEVEL_OPTIONS = [
    {"label": "Subtype", "value": "subtype"},
    {"label": "Subsubtype", "value": "subsubtype"},
]


def _default_market_balance_start_date() -> str:
    today = dt.date.today()
    return dt.date(today.year - 3, 1, 1).isoformat()


def _default_market_balance_end_date() -> str:
    today = dt.date.today()
    return dt.date(today.year + 5, 12, 31).isoformat()


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=message,
        xaxis={"visible": False},
        yaxis={"visible": False},
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return figure


def _build_status_block(lines: list[str]) -> html.Div:
    visible_lines = [line for line in lines if line]
    if not visible_lines:
        return html.Div()

    return html.Div(
        [html.Div(line, className="text-tertiary", style={"fontSize": "12px"}) for line in visible_lines],
        style={"display": "grid", "gap": "4px", "marginTop": "12px"},
    )


def _build_error_banner(error_message: str | None) -> html.Div:
    if not error_message:
        return html.Div()

    return html.Div(
        error_message,
        style={
            "backgroundColor": "#fef2f2",
            "border": "1px solid #fecaca",
            "color": "#991b1b",
            "padding": "10px 12px",
            "borderRadius": "6px",
            "marginTop": "12px",
        },
    )


def _market_balance_snapshot_recovery_notice() -> html.Div:
    return _build_error_banner(
        MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE,
    )


def _market_balance_snapshot_recovery_option() -> list[dict]:
    return [
        {
            "label": MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE,
            "value": "__snapshot_unavailable__",
            "disabled": True,
        }
    ]


def _market_balance_comparison_recovery_result(
    comparison_source,
) -> tuple:
    options = _market_balance_snapshot_recovery_option()
    woodmac_style = (
        {"display": "none"}
        if comparison_source == "ea"
        else {
            "display": "flex",
            "gap": "12px",
            "flexWrap": "wrap",
            "alignItems": "flex-end",
        }
    )
    ea_style = (
        {
            "display": "flex",
            "gap": "12px",
            "flexWrap": "wrap",
            "alignItems": "flex-end",
        }
        if comparison_source == "ea"
        else {"display": "none"}
    )
    return (
        options,
        None,
        options,
        None,
        options,
        None,
        woodmac_style,
        ea_style,
    )


def _get_country_snapshot_values(meta_data: dict, country: str | None) -> list[dict]:
    country_snapshots = meta_data.get("country_snapshots") or {}
    if country and country in country_snapshots:
        return country_snapshots.get(country) or []
    return meta_data.get("snapshots") or []


def _choose_country_snapshot_value(
    snapshots: list[dict],
    current_snapshot: int | None,
) -> int | None:
    run_ids = [run.get("run_id") for run in snapshots]
    try:
        normalized_current = normalize_ea_run_id(current_snapshot)
    except (TypeError, ValueError):
        normalized_current = None
    if normalized_current in run_ids:
        return normalized_current
    return run_ids[1] if len(run_ids) > 1 else None


def _split_wrapped_header_name(column_name: str) -> list[str]:
    words = [word for word in str(column_name).split() if word]
    return words or [str(column_name)]


def _header_width_length(column_name: str, *, wrap_multi_word_headers: bool) -> int:
    header_parts = (
        _split_wrapped_header_name(column_name)
        if wrap_multi_word_headers
        else [str(column_name)]
    )
    return max((len(line) for line in header_parts), default=0)


def _build_responsive_column_styles(
    df: pd.DataFrame,
    *,
    compact: bool = False,
    wrap_multi_word_headers: bool = False,
) -> list[dict]:
    column_styles = []

    for column_name in df.columns:
        header_length = _header_width_length(
            column_name,
            wrap_multi_word_headers=wrap_multi_word_headers,
        )
        value_lengths = df[column_name].map(_format_table_cell_value).map(len)
        max_value_length = max(value_lengths.tolist()) if not df.empty else 0
        header_width_px = (header_length * 7) + (18 if compact else 24)
        value_width_px = (max_value_length * 7) + (12 if compact else 18)
        content_width_px = max(header_width_px, value_width_px)

        if column_name in {"Period", "Date", "Month"}:
            min_width = 76 if compact else 82
            max_width = 112 if compact else 132
            width_px = max(min_width, min(content_width_px, max_width))
        elif str(column_name).startswith("Total"):
            min_width = 72 if compact else 88
            max_width = 132 if compact else 190
            width_px = max(min_width, min(content_width_px, max_width))
        else:
            min_width = 58 if compact else 64
            max_width = 122 if compact else 180
            width_px = max(min_width, min(content_width_px, max_width))

        style_entry = {
            "if": {"column_id": column_name},
            "minWidth": f"{width_px}px",
            "width": f"{width_px}px",
            "maxWidth": f"{width_px}px",
        }

        if column_name in {"Period", "Date", "Month"}:
            style_entry["textAlign"] = "left"

        column_styles.append(style_entry)

    return column_styles


def _build_export_header(title: str, button_id: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                html.H3(
                    title,
                    className="section-title-inline",
                    style={
                        "whiteSpace": "nowrap",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                    },
                ),
                style={"flex": "1 1 auto", "minWidth": "0"},
            ),
            html.Button(
                "Export to Excel",
                id=button_id,
                n_clicks=0,
                style={**EXPORT_BUTTON_STYLE, "flexShrink": "0"},
            ),
        ],
        className="inline-section-header",
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "gap": "12px",
            "flexWrap": "nowrap",
        },
    )


def _build_market_table(
    table_id: str,
    payload: dict | None,
    *,
    table_mode: str = "absolute",
    column_styles: list[dict] | None = None,
    empty_message: str = "No data available for the current selection.",
    page_size: int = 18,
    compact: bool = False,
    wrap_multi_word_headers: bool = False,
) -> html.Div | dag.AgGrid:
    df = _deserialize_frame(payload)
    if df.empty:
        return html.Div(empty_message, className="text-tertiary", style={"padding": "16px"})

    base_config = StandardTableStyleManager.get_base_datatable_config()
    columns = []
    numeric_columns = set(payload.get("numeric_columns") or [])
    for column_name in df.columns:
        if column_name in numeric_columns:
            columns.append(
                {
                    "name": column_name,
                    "id": column_name,
                    "type": "numeric",
                    "format": Format(precision=1, scheme=Scheme.fixed),
                }
            )
        else:
            columns.append({"name": column_name, "id": column_name})

    style_data_conditional = list(base_config["style_data_conditional"])
    style_data_conditional.extend(
        [
            {
                "if": {"column_id": "Total"},
                "fontWeight": "bold",
                "backgroundColor": "#eff6ff",
            },
            {
                "if": {"column_id": "Supply"},
                "backgroundColor": "#ecfccb",
            },
            {
                "if": {"column_id": "Demand"},
                "backgroundColor": "#fee2e2",
            },
            {
                "if": {"column_id": "Delta"},
                "fontWeight": "bold",
            },
        ]
    )
    if column_styles:
        style_data_conditional.extend(column_styles)

    if table_mode in {"delta", "net", "provider_gap"}:
        columns_to_color = (
            ["Delta", "Delta %"]
            if table_mode == "provider_gap"
            else list(numeric_columns)
        )
        for column_name in columns_to_color:
            if column_name not in df.columns:
                continue
            if not str(column_name).strip():
                continue
            style_data_conditional.extend(
                [
                    {
                        "if": {
                            "column_id": column_name,
                            "filter_query": f"{{{column_name}}} > 0",
                        },
                        "color": "#166534",
                        "fontWeight": "bold",
                    },
                    {
                        "if": {
                            "column_id": column_name,
                            "filter_query": f"{{{column_name}}} < 0",
                        },
                        "color": "#991b1b",
                        "fontWeight": "bold",
                    },
                ]
            )

    return create_ag_grid_from_datatable(
        id=table_id,
        columns=columns,
        data=df.to_dict("records"),
        style_cell_conditional=_build_responsive_column_styles(
            df,
            compact=compact,
            wrap_multi_word_headers=wrap_multi_word_headers,
        ),
        style_data_conditional=style_data_conditional,
        fill_width=False,
        page_size=page_size,
    )


def _normalize_maintenance_metric(value: str | None) -> str:
    metric_value = str(value or "Unplanned").strip()
    valid_values = {option["value"] for option in MAINTENANCE_METRIC_OPTIONS}
    return metric_value if metric_value in valid_values else "Unplanned"


def _build_selected_maintenance_payload(
    payload: dict | None,
    selected_metric: str | None,
) -> dict:
    df = _deserialize_frame(payload)
    if df.empty:
        return serialize_frame(pd.DataFrame(columns=["Period", "Total"]))

    metric_value = _normalize_maintenance_metric(selected_metric)
    if "Metric" not in df.columns:
        return serialize_frame(df)

    filtered_df = df[df["Metric"].astype(str) == metric_value].copy()
    if filtered_df.empty and metric_value != "Total":
        filtered_df = df[df["Metric"].astype(str) == "Total"].copy()
    if filtered_df.empty:
        return serialize_frame(pd.DataFrame(columns=["Period", "Total"]))

    filtered_df = filtered_df.drop(columns=["Metric"])
    return serialize_frame(filtered_df)


def _build_labeled_table(title: str, child) -> html.Div:
    return html.Div(
        [
            html.Div(
                title,
                className="text-tertiary",
                style={"fontSize": "12px", "fontWeight": "600", "marginTop": "16px"},
            ),
            child,
        ]
    )


def _format_maintenance_kpi_value(value, *, suffix: str = " Mt") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.1f}{suffix}"


def _sum_numeric_column(df: pd.DataFrame, column_name: str) -> float:
    if df.empty or column_name not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[column_name], errors="coerce").fillna(0.0).sum())


def _build_maintenance_kpi_card(
    label: str,
    value: str,
    subtitle: str,
    *,
    accent_color: str = "#1f4e79",
) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "11px", "fontWeight": "700", "color": "#64748b", "textTransform": "uppercase", "letterSpacing": "0.04em"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "800", "color": "#0f172a", "marginTop": "4px"}),
            html.Div(subtitle, style={"fontSize": "12px", "color": "#475569", "marginTop": "4px", "lineHeight": "1.35"}),
        ],
        style={
            "backgroundColor": "#ffffff",
            "border": "1px solid #dbe4ee",
            "borderTop": f"4px solid {accent_color}",
            "borderRadius": "12px",
            "padding": "12px 14px",
            "boxShadow": "0 1px 3px rgba(15, 23, 42, 0.08)",
            "minWidth": "160px",
            "flex": "1 1 160px",
        },
    )


def _build_maintenance_kpi_cards(comparison_df: pd.DataFrame) -> html.Div:
    if comparison_df.empty:
        return _create_empty_state("No provider maintenance comparison available for the current selection.")

    woodmac_unplanned = _sum_numeric_column(comparison_df, "WoodMac Unplanned")
    ea_unplanned = _sum_numeric_column(comparison_df, "Energy Aspects Unplanned")
    woodmac_planned = _sum_numeric_column(comparison_df, "WoodMac Planned")
    delta = woodmac_unplanned - ea_unplanned
    delta_pct = None if ea_unplanned == 0 else (delta / ea_unplanned) * 100.0
    delta_subtitle = (
        "WoodMac higher outage estimate"
        if delta > 0
        else "EA higher outage estimate"
        if delta < 0
        else "Providers aligned on unplanned outage"
    )
    delta_color = "#166534" if delta > 0 else "#991b1b" if delta < 0 else "#475569"

    return html.Div(
        [
            _build_maintenance_kpi_card(
                "WoodMac unplanned",
                _format_maintenance_kpi_value(woodmac_unplanned),
                "Comparable outage basis",
                accent_color="#1f77b4",
            ),
            _build_maintenance_kpi_card(
                "EA unplanned",
                _format_maintenance_kpi_value(ea_unplanned),
                "Dataset 15522, world level",
                accent_color="#ff7f0e",
            ),
            _build_maintenance_kpi_card(
                "Delta",
                _format_maintenance_kpi_value(delta),
                delta_subtitle,
                accent_color=delta_color,
            ),
            _build_maintenance_kpi_card(
                "Delta %",
                _format_maintenance_kpi_value(delta_pct, suffix="%"),
                "Relative to EA unplanned",
                accent_color=delta_color,
            ),
            _build_maintenance_kpi_card(
                "WoodMac planned",
                _format_maintenance_kpi_value(woodmac_planned),
                "Context, not in provider delta",
                accent_color="#64748b",
            ),
        ],
        style={
            "display": "flex",
            "gap": "12px",
            "flexWrap": "wrap",
            "marginBottom": "14px",
        },
    )


def _period_coverage_line(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None

    period_col = "Period" if "Period" in df.columns else df.columns[0]
    group_column_count = max(len(df.columns) - 2, 0)
    start_period = df[period_col].iloc[0]
    end_period = df[period_col].iloc[-1]
    return (
        f"Coverage: {len(df):,} periods | {start_period} to {end_period} | "
        f"{group_column_count} group columns"
    )


def _build_overview_net_summary(
    provider_label: str,
    df: pd.DataFrame,
    metadata: dict,
    overview_net_metadata: dict,
) -> html.Div:
    lines = []
    coverage_line = _period_coverage_line(df)
    if coverage_line:
        lines.append(coverage_line)
    else:
        lines.append("No baseline net balance data available for the current selection.")

    if provider_label == "WoodMac":
        lines.append(
            "Current exports publication: "
            f"{_format_metadata_timestamp(metadata.get('woodmac_export', {}).get('short_term_publication_timestamp')) or 'N/A'}"
        )
        lines.append(
            "Current imports publication: "
            f"{_format_metadata_timestamp(metadata.get('woodmac_import', {}).get('short_term_publication_timestamp')) or 'N/A'}"
        )
    else:
        lines.append(
            "Current exports upload_timestamp_utc: "
            f"{_format_metadata_timestamp(metadata.get('ea_export', {}).get('upload_timestamp_utc')) or 'N/A'}"
        )
        lines.append(
            "Current imports upload_timestamp_utc: "
            f"{_format_metadata_timestamp(metadata.get('ea_import', {}).get('upload_timestamp_utc')) or 'N/A'}"
        )

    lines.append(
        "Basis: "
        f"{TIME_GROUP_LABELS.get(normalize_time_group(overview_net_metadata.get('time_group')), 'Yearly')} | "
        f"{overview_net_metadata.get('country_group_label', 'Classification')} | "
        f"{overview_net_metadata.get('unit', 'bcm')}"
    )
    return _build_status_block(lines)


def _build_overview_net_balance_section(
    *,
    title: str,
    export_button_id: str,
    baseline_summary_id: str,
    comparison_source_dropdown_id: str,
    comparison_st_dropdown_id: str,
    comparison_lt_dropdown_id: str,
    comparison_ea_upload_dropdown_id: str,
    comparison_woodmac_controls_id: str,
    comparison_ea_controls_id: str,
    baseline_table_container_id: str,
    comparison_summary_id: str,
    comparison_table_container_id: str,
) -> html.Div:
    default_comparison_source = "woodmac" if "WoodMac" in title else "ea"
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                html.H3(
                                    title,
                                    className="section-title-inline",
                                    style={
                                        "whiteSpace": "nowrap",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                    },
                                ),
                                style={"flex": "1 1 auto", "minWidth": "0"},
                            ),
                            html.Button(
                                "Export to Excel",
                                id=export_button_id,
                                n_clicks=0,
                                style={**EXPORT_BUTTON_STYLE, "flexShrink": "0"},
                            ),
                        ],
                        className="inline-section-header",
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "space-between",
                            "gap": "12px",
                            "flexWrap": "nowrap",
                        },
                    ),
                ],
                className="balance-section-header",
            ),
            html.Div(
                [
                    html.Div(
                        "Baseline Table",
                        className="balance-panel-title balance-panel-title-left",
                    ),
                    html.Div(
                        "Delta vs Selected Snapshot",
                        className="balance-panel-title balance-panel-title-right",
                        title="Delta formula: left baseline table - selected snapshot",
                        style={
                            "textDecoration": "underline dotted",
                            "textUnderlineOffset": "3px",
                            "cursor": "help",
                        },
                    ),
                    html.Div(
                        [html.Div(id=baseline_summary_id, className="balance-pane-summary")],
                        className="balance-pane-top-area balance-pane-top-area-left",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Source:", className="filter-label"),
                                    dcc.Dropdown(
                                        id=comparison_source_dropdown_id,
                                        options=[
                                            {"label": "WoodMac", "value": "woodmac"},
                                            {"label": "Energy Aspects", "value": "ea"},
                                        ],
                                        value=default_comparison_source,
                                        clearable=False,
                                        className="filter-dropdown",
                                        style={"minWidth": "180px"},
                                    ),
                                ],
                                className="filter-group",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("ST publication:", className="filter-label"),
                                            dcc.Dropdown(
                                                id=comparison_st_dropdown_id,
                                                options=[],
                                                value=None,
                                                clearable=False,
                                                className="filter-dropdown",
                                                style={"minWidth": "260px"},
                                            ),
                                        ],
                                        className="filter-group",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("LT publication:", className="filter-label"),
                                            dcc.Dropdown(
                                                id=comparison_lt_dropdown_id,
                                                options=[],
                                                value=None,
                                                clearable=False,
                                                className="filter-dropdown",
                                                style={"minWidth": "260px"},
                                            ),
                                        ],
                                        className="filter-group",
                                    ),
                                ],
                                id=comparison_woodmac_controls_id,
                                className="balance-comparison-control-row",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("upload_timestamp_utc:", className="filter-label"),
                                            dcc.Dropdown(
                                                id=comparison_ea_upload_dropdown_id,
                                                options=[],
                                                value=None,
                                                clearable=False,
                                                className="filter-dropdown",
                                                style={"minWidth": "280px"},
                                            ),
                                        ],
                                        className="filter-group",
                                    ),
                                ],
                                id=comparison_ea_controls_id,
                                className="balance-comparison-control-row",
                            ),
                        ],
                        className="balance-comparison-controls balance-pane-top-area balance-pane-top-area-right",
                    ),
                    html.Div(
                        id=baseline_table_container_id,
                        className="balance-table-container balance-table-container-left",
                    ),
                    html.Div(
                        [
                            html.Div(id=comparison_summary_id),
                            html.Div(
                                id=comparison_table_container_id,
                                className="balance-table-container",
                            ),
                        ],
                        className="balance-table-shell balance-table-shell-right",
                    ),
                ],
                className="balance-comparison-grid",
            ),
        ],
        className="balance-section-card",
    )


def _build_provider_overview_section() -> html.Div:
    return html.Div(
        [
            html.Div(
                [_build_export_header("Provider Overview", "market-balance-overview-export")],
                className="balance-section-header",
            ),
            html.Div(id="market-balance-overview-status", className="balance-pane-summary"),
            html.Div(id="market-balance-overview-error"),
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(id="market-balance-overview-supply-figure", style={"height": "420px"})],
                        className="section-container",
                        style={"flex": "1", "minWidth": "0"},
                    ),
                    html.Div(
                        [dcc.Graph(id="market-balance-overview-demand-figure", style={"height": "420px"})],
                        className="section-container",
                        style={"flex": "1", "minWidth": "0"},
                    ),
                ],
                style={"display": "flex", "gap": "24px", "marginTop": "20px"},
            ),
        ],
        id="market-balance-provider-overview-section",
        className="balance-section-card",
        style={"minWidth": "0"},
    )


def _build_overview_chart_section(
    title: str,
    *,
    section_id: str,
    children,
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.H3(
                            title,
                            className="section-title-inline",
                            style={
                                "whiteSpace": "nowrap",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                            },
                        ),
                        style={"flex": "1 1 auto", "minWidth": "0"},
                    ),
                ],
                className="balance-section-header",
            ),
            html.Div(children, style={"padding": "16px"}),
        ],
        id=section_id,
        className="balance-section-card",
        style={"minWidth": "0"},
    )


def _fetch_net_balance_comparison_frame(
    *,
    comparison_source: str,
    short_term_value: str | None,
    long_term_value: str | None,
    ea_upload_value: int | None,
    country_group: str,
    time_group: str,
    unit: str,
    start_date: str | None,
    end_date: str | None,
    base_reference=None,
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        if comparison_source == "ea":
            if not ea_upload_value:
                return None, "No Energy Aspects comparison run available."
            _, comparison_df = _get_historical_comparison_frame(
                direction="net-balance",
                base_reference=base_reference,
                selection={
                    "source": "ea",
                    "run_id": normalize_ea_run_id(ea_upload_value),
                },
                query_dependencies={
                    "start_date": start_date,
                    "end_date": end_date,
                    "country_group": country_group,
                    "time_group": time_group,
                    "unit": unit,
                },
                builder=lambda: fetch_net_balance_for_ea_upload(
                    ea_as_of_run_id=ea_upload_value,
                    country_group=country_group,
                    time_group=time_group,
                    unit=unit,
                    start_date=start_date,
                    end_date=end_date,
                ),
            )
            return comparison_df, None

        short_term_snapshot = _deserialize_snapshot_value(short_term_value)
        long_term_snapshot = _deserialize_snapshot_value(long_term_value)
        if not short_term_snapshot or not long_term_snapshot:
            return None, "No WoodMac comparison publications available."

        selection = {
            "source": "woodmac",
            "short_term_market_outlook": short_term_snapshot.get(
                "market_outlook"
            ),
            "short_term_publication_timestamp": short_term_snapshot.get(
                "publication_timestamp"
            ),
            "long_term_market_outlook": long_term_snapshot.get(
                "market_outlook"
            ),
            "long_term_publication_timestamp": long_term_snapshot.get(
                "publication_timestamp"
            ),
        }
        _, comparison_df = _get_historical_comparison_frame(
            direction="net-balance",
            base_reference=base_reference,
            selection=selection,
            query_dependencies={
                "start_date": start_date,
                "end_date": end_date,
                "country_group": country_group,
                "time_group": time_group,
                "unit": unit,
            },
            builder=lambda: fetch_net_balance_for_woodmac_publications(
                short_term_market_outlook=short_term_snapshot.get("market_outlook"),
                short_term_publication_timestamp=short_term_snapshot.get("publication_timestamp"),
                long_term_market_outlook=long_term_snapshot.get("market_outlook"),
                long_term_publication_timestamp=long_term_snapshot.get("publication_timestamp"),
                country_group=country_group,
                time_group=time_group,
                unit=unit,
                start_date=start_date,
                end_date=end_date,
            ),
        )
        return comparison_df, None
    except Exception as exc:
        return None, f"Comparison load failed: {exc}"


def _render_overview_net_delta(
    *,
    active_tab: str,
    store_payload: dict | None,
    baseline_key: str,
    comparison_source: str,
    short_term_value: str | None,
    long_term_value: str | None,
    ea_upload_value: int | None,
    start_date: str | None,
    end_date: str | None,
    time_group: str,
    unit: str,
    country_group: str,
    table_id: str,
) -> tuple[html.Div, html.Div | dag.AgGrid]:
    if active_tab != "overview":
        return html.Div(), html.Div()

    overview_reference = store_payload
    try:
        store_payload = _resolve_market_store(
            store_payload,
            "market-balance-overview-v2",
        )
    except _SnapshotUnavailable:
        return (
            _market_balance_snapshot_recovery_notice(),
            _create_empty_state(
                MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE
            ),
        )

    error_message = (store_payload or {}).get("error")
    if error_message:
        return html.Div(), html.Div()

    data = (store_payload or {}).get("data", {})
    baseline_df = _deserialize_frame(data.get(baseline_key))
    if baseline_df.empty:
        return (
            html.Div(),
            _create_empty_state("No baseline net balance data available for the current selection."),
        )

    comparison_df, comparison_error = _fetch_net_balance_comparison_frame(
        comparison_source=comparison_source,
        short_term_value=short_term_value,
        long_term_value=long_term_value,
        ea_upload_value=ea_upload_value,
        country_group=country_group,
        time_group=time_group,
        unit=unit,
        start_date=start_date,
        end_date=end_date,
        base_reference=overview_reference,
    )

    if comparison_error:
        return (
            html.Div(),
            _create_empty_state(
                "Unable to load comparison snapshot."
                if comparison_error.startswith("Comparison load failed:")
                else comparison_error
            ),
        )

    if comparison_df is None or comparison_df.empty:
        return (
            html.Div(),
            _create_empty_state("No comparison snapshot data available for the current selection."),
        )

    delta_df = build_period_delta_table(baseline_df, comparison_df)
    return (
        html.Div(),
        _build_market_table(
            table_id,
            serialize_frame(delta_df),
            table_mode="delta",
            page_size=10,
        ),
    )


def build_workbook_bytes(sheet_map: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheet_map.items():
            safe_sheet_name = str(sheet_name or "Sheet1")[:31] or "Sheet1"
            export_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
            if export_df.empty:
                export_df = pd.DataFrame({"Message": ["No data available"]})
            export_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
    return output.getvalue()


def _build_provider_metric_comparison_figure(
    woodmac_df: pd.DataFrame,
    ea_df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    yaxis_title: str = "Mt",
) -> go.Figure:
    if woodmac_df.empty and ea_df.empty:
        return _empty_figure(title)

    figure = go.Figure()

    if not woodmac_df.empty and metric in woodmac_df.columns:
        woodmac_period_col = "Period" if "Period" in woodmac_df.columns else "Month"
        figure.add_trace(
            go.Scatter(
                x=woodmac_df[woodmac_period_col],
                y=woodmac_df[metric],
                name="WoodMac",
                mode="lines+markers",
                line={"width": 2},
            )
        )

    if not ea_df.empty and metric in ea_df.columns:
        ea_period_col = "Period" if "Period" in ea_df.columns else "Month"
        figure.add_trace(
            go.Scatter(
                x=ea_df[ea_period_col],
                y=ea_df[metric],
                name="Energy Aspects",
                mode="lines+markers",
                line={"width": 2},
            )
        )

    figure.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        xaxis={"title": None},
        yaxis={"title": yaxis_title},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.15},
    )
    return figure


def _build_trade_stacked_figure(
    df: pd.DataFrame,
    *,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    import plotly.express as px

    if df.empty or len(df.columns) <= 2:
        return _empty_figure(title)

    value_columns = [
        column_name for column_name in df.columns if column_name not in {"Period", "Total"}
    ]
    if not value_columns:
        return _empty_figure(title)

    chart_df = df[["Period", *value_columns]].melt(
        id_vars=["Period"], var_name="Category", value_name="Value"
    )
    figure = px.bar(
        chart_df,
        x="Period",
        y="Value",
        color="Category",
        barmode="stack",
    )
    figure.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        xaxis_title=None,
        yaxis_title=yaxis_title,
        legend_title_text="",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.15},
    )
    return figure


def _build_maintenance_figure(df: pd.DataFrame) -> go.Figure:
    title = "Provider Outage Gap"
    if df.empty:
        return _empty_figure(title)

    required_columns = {
        "Period",
        "WoodMac Unplanned",
        "Energy Aspects Unplanned",
        "Delta",
    }
    if not required_columns.issubset(df.columns):
        return _empty_figure(title)

    working_df = df.copy()
    for column_name in required_columns - {"Period"}:
        working_df[column_name] = pd.to_numeric(
            working_df[column_name],
            errors="coerce",
        ).fillna(0.0)

    delta_colors = [
        "#166534" if value >= 0 else "#991b1b"
        for value in working_df["Delta"]
    ]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.64, 0.36],
        subplot_titles=(
            "Unplanned outage level (Mt)",
            "Provider gap: WoodMac - Energy Aspects (Mt)",
        ),
    )
    figure.add_trace(
        go.Bar(
            x=working_df["Period"],
            y=working_df["WoodMac Unplanned"],
            name="WoodMac Unplanned",
            marker_color="#1f77b4",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=working_df["Period"],
            y=working_df["Energy Aspects Unplanned"],
            name="Energy Aspects Unplanned",
            marker_color="#ff7f0e",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=working_df["Period"],
            y=working_df["Delta"],
            name="Delta",
            marker_color=delta_colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    figure.add_hline(y=0, line_color="#94a3b8", line_width=1, row=2, col=1)
    figure.update_layout(
        title={
            "text": (
                "Provider Outage Gap<br>"
                "<sup>Provider comparison is unplanned outage only; "
                "WoodMac planned maintenance shown as context.</sup>"
            )
        },
        template="plotly_white",
        margin={"l": 48, "r": 24, "t": 92, "b": 44},
        barmode="group",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.18},
    )
    figure.update_yaxes(title_text="Mt", row=1, col=1)
    figure.update_yaxes(title_text="Mt", zeroline=True, row=2, col=1)
    figure.update_xaxes(title_text=None)
    return figure


def _build_pacific_supply_figure(df: pd.DataFrame) -> go.Figure:
    import plotly.express as px

    if df.empty:
        return _empty_figure("Pacific Locked Supply")

    period_col = "Period" if "Period" in df.columns else "Month"
    figure = px.bar(
        df,
        x=period_col,
        y="Supply",
        color="Country",
        facet_row="Provider",
        barmode="stack",
        title="Pacific Locked Supply",
    )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        xaxis_title=None,
        yaxis_title="Supply (MMTPA)",
        legend_title_text="",
    )
    return figure


def _build_pacific_total_figure(df: pd.DataFrame) -> go.Figure:
    import plotly.express as px

    if df.empty:
        return _empty_figure("Pacific Supply Equivalent (mcm/d)")

    period_col = "Period" if "Period" in df.columns else "Month"
    figure = px.line(
        df,
        x=period_col,
        y="Equivalent MCM/D",
        color="Provider",
        markers=True,
        title="Pacific Supply Equivalent (mcm/d)",
    )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        xaxis_title=None,
        yaxis_title="mcm/d",
        legend_title_text="",
    )
    return figure


def _build_country_balance_figure(df: pd.DataFrame, country: str | None) -> go.Figure:
    if df.empty:
        return _empty_figure("Country Balance")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=df["Date"], y=df["Demand"], name="Demand", mode="lines", line={"width": 2})
    )
    figure.add_trace(
        go.Scatter(x=df["Date"], y=df["Supply"], name="Supply", mode="lines", line={"width": 2})
    )
    if "Fcst Margin" in df.columns:
        figure.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["Fcst Margin"],
                name="Fcst Margin",
                yaxis="y2",
                marker_opacity=0.45,
            )
        )
    figure.update_layout(
        title=f"{country or 'Country'} Balance",
        template="plotly_white",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        xaxis={"title": None},
        yaxis={"title": "bcm"},
        yaxis2={"title": "Fcst Margin", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.15},
    )
    return figure


def _build_country_category_figure(df: pd.DataFrame, *, title: str, chart_type: str) -> go.Figure:
    import plotly.express as px

    if df.empty:
        return _empty_figure(title)

    has_series_names = (
        "series_name" in df.columns
        and df["series_name"].map(lambda value: bool(str(value).strip()) if pd.notna(value) else False).any()
    )
    if chart_type == "area" and has_series_names:
        figure = px.area(df, x="Date", y="value", color="series_name", title=title)
    else:
        grouped_df = df.groupby("Date", as_index=False)["value"].sum()
        figure = px.line(grouped_df, x="Date", y="value", title=title)

    figure.update_layout(
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        xaxis_title=None,
        yaxis_title="bcm",
        legend_title_text="",
    )
    return figure


layout = html.Div(
    [
        dcc.Store(id="market-balance-source-state-store", storage_type="memory"),
        dcc.Store(id="market-balance-refresh-status-store", storage_type="memory"),
        dcc.Store(id="market-balance-overview-store", storage_type="memory"),
        dcc.Store(id="market-balance-trade-store", storage_type="memory"),
        dcc.Store(id="market-balance-country-meta-store", storage_type="memory"),
        dcc.Store(id="market-balance-country-store", storage_type="memory"),
        dcc.Store(id="market-balance-date-range-init-store", storage_type="memory"),
        dcc.Download(id="market-balance-overview-woodmac-net-download"),
        dcc.Download(id="market-balance-overview-ea-net-download"),
        dcc.Download(id="market-balance-overview-download"),
        dcc.Download(id="market-balance-trade-download"),
        dcc.Download(id="market-balance-country-download"),

        html.Div(
            [
                html.Div(
                    [
                        html.Div("Date Range", className="filter-group-header"),
                        html.Div(
                            [
                                dcc.DatePickerRange(
                                    id="market-balance-date-range",
                                    start_date=_default_market_balance_start_date(),
                                    end_date=_default_market_balance_end_date(),
                                    minimum_nights=0,
                                    display_format="YYYY-MM",
                                    month_format="YYYY-MM",
                                    start_date_placeholder_text="Start month",
                                    end_date_placeholder_text="End month",
                                    clearable=False,
                                    number_of_months_shown=2,
                                )
                            ],
                            className="professional-date-picker",
                        ),
                    ],
                    className="filter-group",
                    style={"minWidth": "280px"},
                ),
                html.Div(
                    [
                        html.Div("Time View", className="filter-group-header"),
                        html.Div(
                            [
                                dcc.RadioItems(
                                    id="market-balance-trade-time-group",
                                    options=TIME_GROUP_OPTIONS,
                                    value="yearly",
                                    inline=True,
                                    labelStyle=STICKY_RADIO_LABEL_STYLE,
                                    inputStyle={"marginRight": "6px"},
                                    style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"},
                                )
                            ],
                            style=STICKY_CONTROL_SHELL_STYLE,
                        ),
                    ],
                    className="filter-group",
                    style={"minWidth": "360px"},
                ),
                html.Div(
                    [
                        html.Div("Unit", className="filter-group-header"),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="market-balance-trade-unit",
                                    options=TRADE_UNIT_OPTIONS,
                                    value="bcm",
                                    clearable=False,
                                    className="capacity-scenario-sticky-dropdown",
                                    style={"minWidth": "120px", "width": "100%"},
                                )
                            ],
                            className="capacity-scenario-sticky-shell",
                            style={"width": "100%"},
                        ),
                    ],
                    className="filter-group",
                    style={"minWidth": "120px"},
                ),
                html.Div(
                    [
                        html.Div("Country Grouping", className="filter-group-header"),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="market-balance-trade-country-group",
                                    options=COUNTRY_GROUP_OPTIONS,
                                    value="country_classification_level1",
                                    clearable=False,
                                    className="capacity-scenario-sticky-dropdown",
                                    style={"minWidth": "220px", "width": "100%"},
                                )
                            ],
                            className="capacity-scenario-sticky-shell",
                            style={"width": "100%"},
                        ),
                    ],
                    className="filter-group",
                    style={"minWidth": "220px"},
                ),
            ],
            className="professional-section-header",
            style=MARKET_BALANCE_STICKY_HEADER_STYLE,
        ),
        html.Div(
            [
                _build_overview_net_balance_section(
                    title="WoodMac Net Balance",
                    export_button_id="market-balance-overview-woodmac-export",
                    baseline_summary_id="market-balance-overview-woodmac-net-summary",
                    comparison_source_dropdown_id="market-balance-overview-woodmac-comparison-source",
                    comparison_st_dropdown_id="market-balance-overview-woodmac-comparison-st",
                    comparison_lt_dropdown_id="market-balance-overview-woodmac-comparison-lt",
                    comparison_ea_upload_dropdown_id="market-balance-overview-woodmac-comparison-ea-upload",
                    comparison_woodmac_controls_id="market-balance-overview-woodmac-comparison-woodmac-controls",
                    comparison_ea_controls_id="market-balance-overview-woodmac-comparison-ea-controls",
                    baseline_table_container_id="market-balance-overview-woodmac-net-table",
                    comparison_summary_id="market-balance-overview-woodmac-delta-summary",
                    comparison_table_container_id="market-balance-overview-woodmac-delta-table",
                ),
                html.Div(
                    [
                        _build_overview_net_balance_section(
                            title="Energy Aspects Net Balance",
                            export_button_id="market-balance-overview-ea-export",
                            baseline_summary_id="market-balance-overview-ea-net-summary",
                            comparison_source_dropdown_id="market-balance-overview-ea-comparison-source",
                            comparison_st_dropdown_id="market-balance-overview-ea-comparison-st",
                            comparison_lt_dropdown_id="market-balance-overview-ea-comparison-lt",
                            comparison_ea_upload_dropdown_id="market-balance-overview-ea-comparison-ea-upload",
                            comparison_woodmac_controls_id="market-balance-overview-ea-comparison-woodmac-controls",
                            comparison_ea_controls_id="market-balance-overview-ea-comparison-ea-controls",
                            baseline_table_container_id="market-balance-overview-ea-net-table",
                            comparison_summary_id="market-balance-overview-ea-delta-summary",
                            comparison_table_container_id="market-balance-overview-ea-delta-table",
                        ),
                        _build_provider_overview_section(),
                        _build_overview_chart_section(
                            "Maintenance Comparison",
                            section_id="market-balance-maintenance-section",
                            children=[
                                html.Div(id="market-balance-maintenance-kpis"),
                                dcc.Graph(id="market-balance-maintenance-figure", style={"height": "420px"}),
                                html.Div(id="market-balance-maintenance-provider-table"),
                                html.Div(
                                    [
                                        html.Label("WoodMac Table Metric", className="filter-label"),
                                        dcc.Dropdown(
                                            id="market-balance-maintenance-metric",
                                            options=MAINTENANCE_METRIC_OPTIONS,
                                            value="Unplanned",
                                            clearable=False,
                                            className="filter-dropdown",
                                            style={"minWidth": "180px"},
                                        ),
                                    ],
                                    className="filter-group",
                                    style={"maxWidth": "240px", "marginTop": "12px"},
                                ),
                                html.Div(id="market-balance-maintenance-table"),
                            ],
                        ),
                        _build_overview_chart_section(
                            "Pacific Locked Supply",
                            section_id="market-balance-pacific-section",
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [dcc.Graph(id="market-balance-pacific-figure", style={"height": "420px"})],
                                            className="section-container",
                                            style={"flex": "1", "minWidth": "0"},
                                        ),
                                        html.Div(
                                            [dcc.Graph(id="market-balance-pacific-total-figure", style={"height": "420px"})],
                                            className="section-container",
                                            style={"flex": "1", "minWidth": "0"},
                                        ),
                                    ],
                                    style={"display": "flex", "gap": "24px"},
                                )
                            ],
                        ),
                    ],
                    id="market-balance-overview-second-row",
                    style={"display": "grid", "gap": "24px"},
                ),
            ],
            id="market-balance-overview-top-row",
            style={"display": "none"},
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Div("Market Balance", className="filter-group-header"),
                        html.Div(
                            "Provider overview, trade balance, and country drilldown in one place.",
                            style={"fontWeight": "600", "fontSize": "15px", "color": "#1e3a5f"},
                        ),
                    ],
                    className="filter-section filter-section-destination",
                ),
                html.Div(
                    [
                        html.Div("Data Model", className="filter-group-header"),
                        html.Div(
                            "Each section loads independently with structured store payloads.",
                            className="text-tertiary",
                            style={"fontSize": "12px", "maxWidth": "320px"},
                        ),
                    ],
                    className="filter-section filter-section-analysis",
                ),
            ],
            className="filter-bar-grouped",
        ),

        dcc.Tabs(
            id="market-balance-tabs",
            value="overview",
            children=[
                dcc.Tab(
                    label="Overview",
                    value="overview",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[html.Div(style={"minHeight": "1px"})],
                ),
                dcc.Tab(
                    label="Trade Balance",
                    value="trade_balance",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div("Trade Balance Controls", className="filter-group-header"),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Label("Difference Type", className="filter-label"),
                                                                dcc.Dropdown(
                                                                    id="market-balance-trade-diff-type",
                                                                    options=DIFF_TYPE_OPTIONS,
                                                                    value="percentage",
                                                                    clearable=False,
                                                                    className="filter-dropdown",
                                                                    style={"minWidth": "180px"},
                                                                ),
                                                            ],
                                                            className="filter-group",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label("Years", className="filter-label"),
                                                                dcc.Dropdown(
                                                                    id="market-balance-trade-years",
                                                                    value=[],
                                                                    multi=True,
                                                                    placeholder="Select years",
                                                                    className="filter-dropdown",
                                                                    style={"minWidth": "320px"},
                                                                ),
                                                            ],
                                                            className="filter-group",
                                                        ),
                                                    ],
                                                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                                                ),
                                            ],
                                            className="filter-section filter-section-analysis",
                                        ),
                                    ],
                                    className="filter-bar-grouped",
                                ),
                                html.Div(
                                    _build_export_header(
                                        "Trade Balance Summary",
                                        "market-balance-trade-export",
                                    ),
                                    style={"marginTop": "20px"},
                                ),
                                html.Div(id="market-balance-trade-status"),
                                html.Div(id="market-balance-trade-error"),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3("Exports", className="section-title-inline"),
                                                dcc.Graph(id="market-balance-trade-export-figure", style={"height": "420px"}),
                                                html.Div(id="market-balance-trade-export-table"),
                                                html.Div(id="market-balance-trade-export-diff-table"),
                                                html.Div(id="market-balance-trade-export-flex-table"),
                                            ],
                                            className="section-container",
                                        ),
                                        html.Div(
                                            [
                                                html.H3("Imports", className="section-title-inline"),
                                                dcc.Graph(id="market-balance-trade-import-figure", style={"height": "420px"}),
                                                html.Div(id="market-balance-trade-import-table"),
                                                html.Div(id="market-balance-trade-import-diff-table"),
                                                html.Div(id="market-balance-trade-import-flex-table"),
                                            ],
                                            className="section-container",
                                            style={"marginTop": "24px"},
                                        ),
                                    ],
                                    style={"marginTop": "20px"},
                                ),
                            ],
                            className="dashboard-container",
                        )
                    ],
                ),
                dcc.Tab(
                    label="Country Drilldown",
                    value="country_drilldown",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div("Country Drilldown Controls", className="filter-group-header"),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Label("Country", className="filter-label"),
                                                                dcc.Dropdown(
                                                                    id="market-balance-country-dropdown",
                                                                    clearable=False,
                                                                    className="filter-dropdown",
                                                                    style={"minWidth": "220px"},
                                                                ),
                                                            ],
                                                            className="filter-group",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label("Hierarchy", className="filter-label"),
                                                                dcc.Dropdown(
                                                                    id="market-balance-country-level",
                                                                    options=COUNTRY_LEVEL_OPTIONS,
                                                                    value="subtype",
                                                                    clearable=False,
                                                                    className="filter-dropdown",
                                                                    style={"minWidth": "140px"},
                                                                ),
                                                            ],
                                                            className="filter-group",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label("Compare Against", className="filter-label"),
                                                                dcc.Dropdown(
                                                                    id="market-balance-country-snapshot",
                                                                    clearable=True,
                                                                    className="filter-dropdown",
                                                                    style={"minWidth": "280px"},
                                                                ),
                                                            ],
                                                            className="filter-group",
                                                        ),
                                                    ],
                                                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                                                ),
                                            ],
                                            className="filter-section filter-section-analysis",
                                        ),
                                    ],
                                    className="filter-bar-grouped",
                                ),
                                html.Div(
                                    _build_export_header(
                                        "Country Drilldown",
                                        "market-balance-country-export",
                                    ),
                                    style={"marginTop": "20px"},
                                ),
                                html.Div(id="market-balance-country-status"),
                                html.Div(id="market-balance-country-error"),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3("Current Data", className="section-title-inline"),
                                                html.Div(id="market-balance-country-current-table"),
                                            ],
                                            className="section-container",
                                            style={"flex": "1", "minWidth": "0"},
                                        ),
                                        html.Div(
                                            [
                                                html.H3("Delta vs Snapshot", className="section-title-inline"),
                                                html.Div(id="market-balance-country-delta-table"),
                                            ],
                                            className="section-container",
                                            style={"flex": "1", "minWidth": "0"},
                                        ),
                                    ],
                                    style={"display": "flex", "gap": "24px", "marginTop": "20px"},
                                ),
                                html.Div(
                                    [dcc.Graph(id="market-balance-country-balance-figure", style={"height": "440px"})],
                                    className="section-container",
                                    style={"marginTop": "24px"},
                                ),
                                html.Div(
                                    id="market-balance-country-category-container",
                                    style={"marginTop": "24px"},
                                ),
                            ],
                            className="dashboard-container",
                        )
                    ],
                ),
            ],
        ),
    ]
)


@callback(
    Output("market-balance-source-state-store", "data"),
    Output("market-balance-refresh-status-store", "data"),
    Input("global-refresh-button", "n_clicks"),
    State("market-balance-source-state-store", "data"),
)
def load_market_balance_source_state(n_clicks, current_source_state=None):
    refresh_status = {
        "format": "dashboard-source-refresh-status-v1",
        "refresh_generation": int(n_clicks or 0),
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    try:
        source_state = _fetch_provider_flow_source_state()
    except Exception:
        source_state = {
            "request_token": dt.datetime.now(dt.timezone.utc).isoformat()
        }
        refresh_status["status"] = "unavailable"
    else:
        refresh_status["status"] = "checked"
    if not _revision_aware_refresh_enabled():
        source_state = {
            **source_state,
            "refresh_generation": int(n_clicks or 0),
        }
    if (
        _revision_aware_refresh_enabled()
        and source_state == current_source_state
    ):
        return no_update, refresh_status
    return source_state, refresh_status


@callback(
    Output("market-balance-overview-store", "data"),
    Input("market-balance-source-state-store", "data"),
    Input("market-balance-date-range", "start_date"),
    Input("market-balance-date-range", "end_date"),
    Input("market-balance-trade-time-group", "value"),
    Input("market-balance-trade-unit", "value"),
    Input("market-balance-trade-country-group", "value"),
)
@log_callback_timing("market_balance.overview_source_load")
def load_overview_store(source_state, start_date, end_date, time_group, unit, country_group):
    key_parts = {
        "start_date": start_date,
        "end_date": end_date,
        "time_group": time_group,
        "unit": unit,
        "country_group": country_group,
    }
    return _load_cached_market_store(
        "market-balance-overview-v2",
        key_parts,
        lambda: fetch_provider_overview_payload(
            start_date=start_date,
            end_date=end_date,
            time_group=time_group,
            unit=unit,
            country_group=country_group,
            force_source_refresh=False,
            source_state=source_state,
        ),
        source_state=source_state,
    )


@callback(
    Output("market-balance-date-range", "start_date"),
    Output("market-balance-date-range", "end_date"),
    Output("market-balance-date-range-init-store", "data"),
    Input("market-balance-tabs", "value"),
    State("market-balance-date-range-init-store", "data"),
)
def initialize_market_balance_date_range(_, initialized):
    if initialized:
        return no_update, no_update, initialized

    return (
        _default_market_balance_start_date(),
        _default_market_balance_end_date(),
        True,
    )


@callback(
    Output("market-balance-overview-top-row", "style"),
    Input("market-balance-tabs", "value"),
)
def toggle_overview_top_row(active_tab):
    if active_tab == "overview":
        return {"display": "grid", "gap": "24px", "marginBottom": "20px"}
    return {"display": "none"}


@callback(
    Output("market-balance-overview-status", "children"),
    Output("market-balance-overview-error", "children"),
    Output("market-balance-overview-woodmac-net-summary", "children"),
    Output("market-balance-overview-woodmac-net-table", "children"),
    Output("market-balance-overview-ea-net-summary", "children"),
    Output("market-balance-overview-ea-net-table", "children"),
    Output("market-balance-overview-supply-figure", "figure"),
    Output("market-balance-overview-demand-figure", "figure"),
    Output("market-balance-maintenance-kpis", "children"),
    Output("market-balance-maintenance-figure", "figure"),
    Output("market-balance-maintenance-provider-table", "children"),
    Output("market-balance-maintenance-table", "children"),
    Output("market-balance-pacific-figure", "figure"),
    Output("market-balance-pacific-total-figure", "figure"),
    Input("market-balance-tabs", "value"),
    Input("market-balance-overview-store", "data"),
    Input("market-balance-maintenance-metric", "value"),
)
@log_callback_timing("market_balance.overview_render")
def render_overview(active_tab, store_payload, maintenance_metric="Unplanned"):
    if active_tab != "overview":
        empty = _empty_figure("Overview unavailable")
        return (
            html.Div(),
            html.Div(),
            html.Div(),
            html.Div(),
            html.Div(),
            html.Div(),
            empty,
            empty,
            html.Div(),
            empty,
            html.Div(),
            html.Div(),
            empty,
            empty,
        )

    try:
        store_payload = _resolve_market_store(
            store_payload,
            "market-balance-overview-v2",
        )
    except _SnapshotUnavailable:
        empty = _empty_figure(
            MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE
        )
        return (
            html.Div(),
            _market_balance_snapshot_recovery_notice(),
            html.Div(),
            html.Div(),
            html.Div(),
            html.Div(),
            empty,
            empty,
            html.Div(),
            empty,
            html.Div(),
            html.Div(),
            empty,
            empty,
        )

    error_message = (store_payload or {}).get("error")
    if error_message:
        empty = _empty_figure("Overview unavailable")
        return (
            html.Div(),
            _build_error_banner(error_message),
            html.Div(),
            html.Div(),
            html.Div(),
            html.Div(),
            empty,
            empty,
            html.Div(),
            empty,
            html.Div(),
            html.Div(),
            empty,
            empty,
        )

    data = (store_payload or {}).get("data", {})
    metadata = (store_payload or {}).get("metadata", {})

    woodmac_df = _deserialize_frame(data.get("woodmac_balance"))
    ea_df = _deserialize_frame(data.get("ea_balance"))
    maintenance_comparison_payload = data.get("maintenance_provider_comparison")
    maintenance_comparison_df = _deserialize_frame(maintenance_comparison_payload)
    maintenance_grouped_payload = data.get("maintenance_grouped")
    pacific_detail_df = _deserialize_frame(data.get("pacific_detail"))
    pacific_totals_df = _deserialize_frame(data.get("pacific_totals"))
    overview_net_metadata = metadata.get("overview_net", {})
    overview_unit = overview_net_metadata.get("unit", "bcm")
    overview_yaxis_title = overview_unit if overview_unit != "mcm_d" else "mcm/d"
    selected_maintenance_metric = _normalize_maintenance_metric(maintenance_metric)
    selected_maintenance_payload = _build_selected_maintenance_payload(
        maintenance_grouped_payload,
        selected_maintenance_metric,
    )

    status_lines = [
        f"WoodMac exports snapshot: {_format_metadata_timestamp(metadata.get('woodmac_export', {}).get('short_term_publication_timestamp')) or 'N/A'}",
        f"WoodMac imports snapshot: {_format_metadata_timestamp(metadata.get('woodmac_import', {}).get('short_term_publication_timestamp')) or 'N/A'}",
        f"Energy Aspects exports upload: {_format_metadata_timestamp(metadata.get('ea_export', {}).get('upload_timestamp_utc')) or 'N/A'}",
        f"Energy Aspects imports upload: {_format_metadata_timestamp(metadata.get('ea_import', {}).get('upload_timestamp_utc')) or 'N/A'}",
    ]

    return (
        _build_status_block(status_lines),
        html.Div(),
        _build_overview_net_summary("WoodMac", _deserialize_frame(data.get("woodmac_net_balance")), metadata, overview_net_metadata),
        _build_market_table(
            "market-balance-overview-woodmac-net-table-grid",
            data.get("woodmac_net_balance"),
            table_mode="net",
            page_size=10,
        ),
        _build_overview_net_summary("Energy Aspects", _deserialize_frame(data.get("ea_net_balance")), metadata, overview_net_metadata),
        _build_market_table(
            "market-balance-overview-ea-net-table-grid",
            data.get("ea_net_balance"),
            table_mode="net",
            page_size=10,
        ),
        _build_provider_metric_comparison_figure(
            woodmac_df,
            ea_df,
            metric="Supply",
            title="Supply Overview",
            yaxis_title=overview_yaxis_title,
        ),
        _build_provider_metric_comparison_figure(
            woodmac_df,
            ea_df,
            metric="Demand",
            title="Demand Overview",
            yaxis_title=overview_yaxis_title,
        ),
        _build_maintenance_kpi_cards(maintenance_comparison_df),
        _build_maintenance_figure(maintenance_comparison_df),
        _build_labeled_table(
            "Provider Outage Gap (Unplanned, Mt)",
            _build_market_table(
                "market-balance-maintenance-provider-gap-table-grid",
                maintenance_comparison_payload,
                table_mode="provider_gap",
                empty_message="No provider outage gap data available for the current selection.",
                page_size=10,
            ),
        ),
        html.Div(
            [
                html.Div(
                    _build_labeled_table(
                        f"WoodMac Maintenance Detail by {overview_net_metadata.get('country_group_label', 'Country')}",
                        _build_market_table(
                            "market-balance-maintenance-table-grid",
                            selected_maintenance_payload,
                            table_mode="absolute",
                            empty_message="No WoodMac maintenance detail available for the current selection.",
                            page_size=10,
                        ),
                    ),
                    className="section-container",
                    style={"flex": "1", "minWidth": "0"},
                ),
            ],
            style={"display": "flex", "gap": "24px", "flexWrap": "wrap", "marginTop": "12px"},
        ),
        _build_pacific_supply_figure(pacific_detail_df),
        _build_pacific_total_figure(pacific_totals_df),
    )


@callback(
    Output("market-balance-overview-woodmac-comparison-st", "options"),
    Output("market-balance-overview-woodmac-comparison-st", "value"),
    Output("market-balance-overview-woodmac-comparison-lt", "options"),
    Output("market-balance-overview-woodmac-comparison-lt", "value"),
    Output("market-balance-overview-woodmac-comparison-ea-upload", "options"),
    Output("market-balance-overview-woodmac-comparison-ea-upload", "value"),
    Output("market-balance-overview-woodmac-comparison-woodmac-controls", "style"),
    Output("market-balance-overview-woodmac-comparison-ea-controls", "style"),
    Input("market-balance-overview-store", "data"),
    Input("market-balance-overview-woodmac-comparison-source", "value"),
    State("market-balance-overview-woodmac-comparison-st", "value"),
    State("market-balance-overview-woodmac-comparison-lt", "value"),
    State("market-balance-overview-woodmac-comparison-ea-upload", "value"),
)
def sync_woodmac_overview_comparison_controls(
    store_payload,
    comparison_source,
    current_st_value,
    current_lt_value,
    current_ea_upload_value,
):
    try:
        store_payload = _resolve_market_store(
            store_payload,
            "market-balance-overview-v2",
        )
    except _SnapshotUnavailable:
        return _market_balance_comparison_recovery_result(
            comparison_source
        )
    comparison_options = ((store_payload or {}).get("metadata", {}) or {}).get(
        "comparison_options", {}
    )
    return _resolve_snapshot_control_values(
        comparison_source,
        comparison_options,
        current_st_value,
        current_lt_value,
        current_ea_upload_value,
    )


@callback(
    Output("market-balance-overview-ea-comparison-st", "options"),
    Output("market-balance-overview-ea-comparison-st", "value"),
    Output("market-balance-overview-ea-comparison-lt", "options"),
    Output("market-balance-overview-ea-comparison-lt", "value"),
    Output("market-balance-overview-ea-comparison-ea-upload", "options"),
    Output("market-balance-overview-ea-comparison-ea-upload", "value"),
    Output("market-balance-overview-ea-comparison-woodmac-controls", "style"),
    Output("market-balance-overview-ea-comparison-ea-controls", "style"),
    Input("market-balance-overview-store", "data"),
    Input("market-balance-overview-ea-comparison-source", "value"),
    State("market-balance-overview-ea-comparison-st", "value"),
    State("market-balance-overview-ea-comparison-lt", "value"),
    State("market-balance-overview-ea-comparison-ea-upload", "value"),
)
def sync_ea_overview_comparison_controls(
    store_payload,
    comparison_source,
    current_st_value,
    current_lt_value,
    current_ea_upload_value,
):
    try:
        store_payload = _resolve_market_store(
            store_payload,
            "market-balance-overview-v2",
        )
    except _SnapshotUnavailable:
        return _market_balance_comparison_recovery_result(
            comparison_source
        )
    comparison_options = ((store_payload or {}).get("metadata", {}) or {}).get(
        "comparison_options", {}
    )
    return _resolve_snapshot_control_values(
        comparison_source,
        comparison_options,
        current_st_value,
        current_lt_value,
        current_ea_upload_value,
    )


@callback(
    Output("market-balance-overview-woodmac-delta-summary", "children"),
    Output("market-balance-overview-woodmac-delta-table", "children"),
    Input("market-balance-tabs", "value"),
    Input("market-balance-overview-store", "data"),
    Input("market-balance-date-range", "start_date"),
    Input("market-balance-date-range", "end_date"),
    Input("market-balance-trade-time-group", "value"),
    Input("market-balance-trade-unit", "value"),
    Input("market-balance-trade-country-group", "value"),
    Input("market-balance-overview-woodmac-comparison-source", "value"),
    Input("market-balance-overview-woodmac-comparison-st", "value"),
    Input("market-balance-overview-woodmac-comparison-lt", "value"),
    Input("market-balance-overview-woodmac-comparison-ea-upload", "value"),
)
def render_woodmac_overview_delta(
    active_tab,
    store_payload,
    start_date,
    end_date,
    time_group,
    unit,
    country_group,
    comparison_source,
    short_term_value,
    long_term_value,
    ea_upload_value,
):
    return _render_overview_net_delta(
        active_tab=active_tab,
        store_payload=store_payload,
        baseline_key="woodmac_net_balance",
        comparison_source=comparison_source,
        short_term_value=short_term_value,
        long_term_value=long_term_value,
        ea_upload_value=ea_upload_value,
        start_date=start_date,
        end_date=end_date,
        time_group=time_group,
        unit=unit,
        country_group=country_group,
        table_id="market-balance-overview-woodmac-delta-grid",
    )


@callback(
    Output("market-balance-overview-ea-delta-summary", "children"),
    Output("market-balance-overview-ea-delta-table", "children"),
    Input("market-balance-tabs", "value"),
    Input("market-balance-overview-store", "data"),
    Input("market-balance-date-range", "start_date"),
    Input("market-balance-date-range", "end_date"),
    Input("market-balance-trade-time-group", "value"),
    Input("market-balance-trade-unit", "value"),
    Input("market-balance-trade-country-group", "value"),
    Input("market-balance-overview-ea-comparison-source", "value"),
    Input("market-balance-overview-ea-comparison-st", "value"),
    Input("market-balance-overview-ea-comparison-lt", "value"),
    Input("market-balance-overview-ea-comparison-ea-upload", "value"),
)
def render_ea_overview_delta(
    active_tab,
    store_payload,
    start_date,
    end_date,
    time_group,
    unit,
    country_group,
    comparison_source,
    short_term_value,
    long_term_value,
    ea_upload_value,
):
    return _render_overview_net_delta(
        active_tab=active_tab,
        store_payload=store_payload,
        baseline_key="ea_net_balance",
        comparison_source=comparison_source,
        short_term_value=short_term_value,
        long_term_value=long_term_value,
        ea_upload_value=ea_upload_value,
        start_date=start_date,
        end_date=end_date,
        time_group=time_group,
        unit=unit,
        country_group=country_group,
        table_id="market-balance-overview-ea-delta-grid",
    )


def _export_overview_net_section(
    *,
    n_clicks,
    store_payload,
    baseline_key: str,
    comparison_source: str,
    short_term_value: str | None,
    long_term_value: str | None,
    ea_upload_value: int | None,
    start_date: str | None,
    end_date: str | None,
    time_group: str,
    unit: str,
    country_group: str,
    download_name: str,
    baseline_sheet_name: str,
    delta_sheet_name: str,
):
    if not n_clicks:
        return no_update

    overview_reference = store_payload
    store_payload = _resolve_market_store(
        store_payload,
        "market-balance-overview-v2",
    )

    if (store_payload or {}).get("error"):
        return no_update

    data = (store_payload or {}).get("data", {})
    baseline_df = _deserialize_frame(data.get(baseline_key))
    if baseline_df.empty:
        return no_update

    comparison_df, comparison_error = _fetch_net_balance_comparison_frame(
        comparison_source=comparison_source,
        short_term_value=short_term_value,
        long_term_value=long_term_value,
        ea_upload_value=ea_upload_value,
        country_group=country_group,
        time_group=time_group,
        unit=unit,
        start_date=start_date,
        end_date=end_date,
        base_reference=overview_reference,
    )

    delta_df = (
        build_period_delta_table(baseline_df, comparison_df)
        if comparison_error is None and comparison_df is not None and not comparison_df.empty
        else pd.DataFrame()
    )
    workbook_bytes = build_workbook_bytes(
        {
            baseline_sheet_name: baseline_df,
            delta_sheet_name: delta_df,
        }
    )
    return dcc.send_bytes(workbook_bytes, download_name)


@callback(
    Output("market-balance-overview-woodmac-net-download", "data"),
    Input("market-balance-overview-woodmac-export", "n_clicks"),
    State("market-balance-overview-store", "data"),
    State("market-balance-date-range", "start_date"),
    State("market-balance-date-range", "end_date"),
    State("market-balance-trade-time-group", "value"),
    State("market-balance-trade-unit", "value"),
    State("market-balance-trade-country-group", "value"),
    State("market-balance-overview-woodmac-comparison-source", "value"),
    State("market-balance-overview-woodmac-comparison-st", "value"),
    State("market-balance-overview-woodmac-comparison-lt", "value"),
    State("market-balance-overview-woodmac-comparison-ea-upload", "value"),
    prevent_initial_call=True,
)
def export_woodmac_overview_net_workbook(
    n_clicks,
    store_payload,
    start_date,
    end_date,
    time_group,
    unit,
    country_group,
    comparison_source,
    short_term_value,
    long_term_value,
    ea_upload_value,
):
    return _export_overview_net_section(
        n_clicks=n_clicks,
        store_payload=store_payload,
        baseline_key="woodmac_net_balance",
        comparison_source=comparison_source,
        short_term_value=short_term_value,
        long_term_value=long_term_value,
        ea_upload_value=ea_upload_value,
        start_date=start_date,
        end_date=end_date,
        time_group=time_group,
        unit=unit,
        country_group=country_group,
        download_name="market_balance_woodmac_net_balance.xlsx",
        baseline_sheet_name="WoodMac Net Balance",
        delta_sheet_name="Delta vs Snapshot",
    )


@callback(
    Output("market-balance-overview-ea-net-download", "data"),
    Input("market-balance-overview-ea-export", "n_clicks"),
    State("market-balance-overview-store", "data"),
    State("market-balance-date-range", "start_date"),
    State("market-balance-date-range", "end_date"),
    State("market-balance-trade-time-group", "value"),
    State("market-balance-trade-unit", "value"),
    State("market-balance-trade-country-group", "value"),
    State("market-balance-overview-ea-comparison-source", "value"),
    State("market-balance-overview-ea-comparison-st", "value"),
    State("market-balance-overview-ea-comparison-lt", "value"),
    State("market-balance-overview-ea-comparison-ea-upload", "value"),
    prevent_initial_call=True,
)
def export_ea_overview_net_workbook(
    n_clicks,
    store_payload,
    start_date,
    end_date,
    time_group,
    unit,
    country_group,
    comparison_source,
    short_term_value,
    long_term_value,
    ea_upload_value,
):
    return _export_overview_net_section(
        n_clicks=n_clicks,
        store_payload=store_payload,
        baseline_key="ea_net_balance",
        comparison_source=comparison_source,
        short_term_value=short_term_value,
        long_term_value=long_term_value,
        ea_upload_value=ea_upload_value,
        start_date=start_date,
        end_date=end_date,
        time_group=time_group,
        unit=unit,
        country_group=country_group,
        download_name="market_balance_energy_aspects_net_balance.xlsx",
        baseline_sheet_name="Energy Aspects Net",
        delta_sheet_name="Delta vs Snapshot",
    )


@callback(
    Output("market-balance-overview-download", "data"),
    Input("market-balance-overview-export", "n_clicks"),
    State("market-balance-overview-store", "data"),
    prevent_initial_call=True,
)
def export_overview_workbook(n_clicks, store_payload):
    if not n_clicks:
        return no_update

    store_payload = _resolve_market_store(
        store_payload,
        "market-balance-overview-v2",
    )

    data = (store_payload or {}).get("data", {})
    if not data:
        return no_update

    workbook_bytes = build_workbook_bytes(
        {
            "WoodMac Net Balance": _deserialize_frame(data.get("woodmac_net_balance")),
            "EA Net Balance": _deserialize_frame(data.get("ea_net_balance")),
            "WoodMac Balance": _deserialize_frame(data.get("woodmac_balance")),
            "EA Balance": _deserialize_frame(data.get("ea_balance")),
            "Maintenance Provider Gap": _deserialize_frame(
                data.get("maintenance_provider_comparison")
            ),
            "Maintenance": _deserialize_frame(data.get("maintenance")),
            "WM Maintenance Grouped": _deserialize_frame(data.get("maintenance_grouped")),
            "EA Maintenance": _deserialize_frame(data.get("maintenance_ea")),
            "Pacific Detail": _deserialize_frame(data.get("pacific_detail")),
            "Pacific Totals": _deserialize_frame(data.get("pacific_totals")),
        }
    )
    return dcc.send_bytes(workbook_bytes, "market_balance_overview.xlsx")


@callback(
    Output("market-balance-trade-store", "data"),
    Input("market-balance-source-state-store", "data"),
    Input("market-balance-date-range", "start_date"),
    Input("market-balance-date-range", "end_date"),
    Input("market-balance-trade-time-group", "value"),
    Input("market-balance-trade-diff-type", "value"),
    Input("market-balance-trade-country-group", "value"),
    Input("market-balance-trade-years", "value"),
    Input("market-balance-trade-unit", "value"),
)
@log_callback_timing("market_balance.trade_source_load")
def load_trade_store(source_state, start_date, end_date, time_group, diff_type, country_group, selected_years, unit):
    key_parts = {
        "start_date": start_date,
        "end_date": end_date,
        "time_group": time_group,
        "diff_type": diff_type,
        "country_group": country_group,
        "selected_years": selected_years,
        "unit": unit,
    }
    return _load_cached_market_store(
        "market-balance-trade-v1",
        key_parts,
        lambda: fetch_trade_balance_payload(
            start_date=start_date,
            end_date=end_date,
            time_group=time_group,
            diff_type=diff_type,
            country_group=country_group,
            selected_years=selected_years,
            unit=unit,
            force_source_refresh=False,
            source_state=source_state,
        ),
        source_state=source_state,
    )


@callback(
    Output("market-balance-trade-years", "options"),
    Input("market-balance-trade-store", "data"),
)
def sync_trade_years(store_payload):
    try:
        store_payload = _resolve_market_store(
            store_payload,
            "market-balance-trade-v1",
        )
    except _SnapshotUnavailable:
        return _market_balance_snapshot_recovery_option()
    metadata = (store_payload or {}).get("metadata", {})
    available_years = metadata.get("available_years") or []
    return [{"label": str(year), "value": year} for year in available_years]


@callback(
    Output("market-balance-trade-status", "children"),
    Output("market-balance-trade-error", "children"),
    Output("market-balance-trade-export-figure", "figure"),
    Output("market-balance-trade-import-figure", "figure"),
    Output("market-balance-trade-export-table", "children"),
    Output("market-balance-trade-export-diff-table", "children"),
    Output("market-balance-trade-export-flex-table", "children"),
    Output("market-balance-trade-import-table", "children"),
    Output("market-balance-trade-import-diff-table", "children"),
    Output("market-balance-trade-import-flex-table", "children"),
    Input("market-balance-trade-store", "data"),
)
@log_callback_timing("market_balance.trade_render")
def render_trade_balance(store_payload):
    try:
        store_payload = _resolve_market_store(
            store_payload,
            "market-balance-trade-v1",
        )
    except _SnapshotUnavailable:
        empty = _empty_figure(
            MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE
        )
        return (
            html.Div(),
            _market_balance_snapshot_recovery_notice(),
            empty,
            empty,
            html.Div(),
            html.Div(),
            html.Div(),
            html.Div(),
            html.Div(),
            html.Div(),
        )
    error_message = (store_payload or {}).get("error")
    if error_message:
        empty = _empty_figure("Trade balance unavailable")
        return (
            html.Div(),
            _build_error_banner(error_message),
            empty,
            empty,
            html.Div(),
            html.Div(),
            html.Div(),
            html.Div(),
            html.Div(),
            html.Div(),
        )

    data = (store_payload or {}).get("data", {})
    metadata = (store_payload or {}).get("metadata", {})
    unit = metadata.get("unit", "bcm")
    yaxis_title = unit if unit != "mcm_d" else "mcm/d"

    exports_df = _deserialize_frame(data.get("exports"))
    imports_df = _deserialize_frame(data.get("imports"))
    export_metadata_ts = _format_metadata_timestamp(
        metadata.get("export_metadata", {}).get("upload_timestamp_utc")
    )
    import_metadata_ts = _format_metadata_timestamp(
        metadata.get("import_metadata", {}).get("upload_timestamp_utc")
    )
    status_lines = [
        f"Source: {metadata.get('source', 'Energy Aspects')}",
        f"Export upload_timestamp_utc: {export_metadata_ts or 'N/A'}",
        f"Import upload_timestamp_utc: {import_metadata_ts or 'N/A'}",
        f"Grouping: {metadata.get('country_group_label', 'Country')} | Time view: {TIME_GROUP_LABELS.get(normalize_time_group(metadata.get('time_group')), 'Monthly')} | Unit: {unit}",
        *(metadata.get("warnings") or []),
    ]

    time_group = normalize_time_group(metadata.get("time_group"))
    if time_group == "yearly":
        status_lines.append(
            "Flex volumes are calculated as Energy Aspects actual trade minus WoodMac annual contracted demand."
        )
    flex_notice = html.Div(
        "Flex volumes are available only for yearly view.",
        className="text-tertiary",
        style={"padding": "16px"},
    )

    return (
        _build_status_block(status_lines),
        html.Div(),
        _build_trade_stacked_figure(exports_df, title="Exports", yaxis_title=yaxis_title),
        _build_trade_stacked_figure(imports_df, title="Imports", yaxis_title=yaxis_title),
        _build_market_table("market-balance-trade-exports-table", data.get("exports")),
        _build_market_table(
            "market-balance-trade-exports-diff-table",
            data.get("exports_diff"),
            table_mode="delta",
        ),
        _build_labeled_table(
            "Exports Flex Volumes (Actual - WoodMac Contracts)",
            _build_market_table("market-balance-trade-exports-flex-table", data.get("exports_flex")),
        )
        if time_group == "yearly"
        else _build_labeled_table("Exports Flex Volumes", flex_notice),
        _build_market_table("market-balance-trade-imports-table", data.get("imports")),
        _build_market_table(
            "market-balance-trade-imports-diff-table",
            data.get("imports_diff"),
            table_mode="delta",
        ),
        _build_labeled_table(
            "Imports Flex Volumes (Actual - WoodMac Contracts)",
            _build_market_table("market-balance-trade-imports-flex-table", data.get("imports_flex")),
        )
        if time_group == "yearly"
        else _build_labeled_table("Imports Flex Volumes", flex_notice),
    )


@callback(
    Output("market-balance-trade-download", "data"),
    Input("market-balance-trade-export", "n_clicks"),
    State("market-balance-trade-store", "data"),
    prevent_initial_call=True,
)
def export_trade_workbook(n_clicks, store_payload):
    if not n_clicks:
        return no_update

    store_payload = _resolve_market_store(
        store_payload,
        "market-balance-trade-v1",
    )

    data = (store_payload or {}).get("data", {})
    if not data:
        return no_update

    workbook_bytes = build_workbook_bytes(
        {
            "Exports": _deserialize_frame(data.get("exports")),
            "Exports Diff": _deserialize_frame(data.get("exports_diff")),
            "Exports Flex": _deserialize_frame(data.get("exports_flex")),
            "Imports": _deserialize_frame(data.get("imports")),
            "Imports Diff": _deserialize_frame(data.get("imports_diff")),
            "Imports Flex": _deserialize_frame(data.get("imports_flex")),
        }
    )
    return dcc.send_bytes(workbook_bytes, "market_balance_trade.xlsx")


@callback(
    Output("market-balance-country-meta-store", "data"),
    Input("market-balance-source-state-store", "data"),
)
def load_country_meta_store(source_state=None):
    return _load_cached_market_store(
        "market-balance-country-meta-v2",
        {},
        lambda: fetch_country_balance_meta_payload(
            current_ea=(source_state or {}).get("current_ea")
        ),
        source_state=source_state,
    )


@callback(
    Output("market-balance-country-dropdown", "options"),
    Output("market-balance-country-dropdown", "value"),
    Input("market-balance-country-meta-store", "data"),
    State("market-balance-country-dropdown", "value"),
)
def sync_country_controls(store_payload, current_country):
    try:
        store_payload = _resolve_market_store(
            store_payload,
            "market-balance-country-meta-v2",
        )
    except _SnapshotUnavailable:
        return _market_balance_snapshot_recovery_option(), None
    data = (store_payload or {}).get("data", {})
    metadata = (store_payload or {}).get("metadata", {})
    countries = data.get("countries") or []

    country_options = [{"label": country, "value": country} for country in countries]

    next_country = current_country if current_country in countries else metadata.get("default_country")
    return country_options, next_country


@callback(
    Output("market-balance-country-snapshot", "options"),
    Output("market-balance-country-snapshot", "value"),
    Input("market-balance-country-meta-store", "data"),
    Input("market-balance-country-dropdown", "value"),
    State("market-balance-country-snapshot", "value"),
)
def sync_country_snapshot_control(store_payload, country, current_snapshot):
    try:
        store_payload = _resolve_market_store(
            store_payload,
            "market-balance-country-meta-v2",
        )
    except _SnapshotUnavailable:
        return _market_balance_snapshot_recovery_option(), None
    data = (store_payload or {}).get("data", {})
    snapshots = _get_country_snapshot_values(data, country)

    snapshot_options = _build_ea_run_dropdown_options(snapshots)
    next_snapshot = _choose_country_snapshot_value(
        snapshots,
        current_snapshot,
    )
    return snapshot_options, next_snapshot


@callback(
    Output("market-balance-country-store", "data"),
    Input("market-balance-source-state-store", "data"),
    Input("market-balance-country-dropdown", "value"),
    Input("market-balance-country-level", "value"),
    Input("market-balance-trade-time-group", "value"),
    Input("market-balance-date-range", "start_date"),
    Input("market-balance-date-range", "end_date"),
    Input("market-balance-country-snapshot", "value"),
)
def load_country_store(source_state, country, level, time_group, start_date, end_date, comparison_snapshot):
    key_parts = {
        "country": country,
        "level": level,
        "time_group": time_group,
        "start_date": start_date,
        "end_date": end_date,
        "comparison_snapshot": comparison_snapshot,
    }
    return _load_cached_market_store(
        "market-balance-country-v2",
        key_parts,
        lambda: fetch_country_balance_payload(
            country=country,
            level=level,
            time_group=time_group,
            start_date=start_date,
            end_date=end_date,
            comparison_timestamp=comparison_snapshot,
            ea_as_of_run_id=((source_state or {}).get("current_ea") or {}).get("run_id"),
        ),
        source_state=source_state,
    )


@callback(
    Output("market-balance-country-status", "children"),
    Output("market-balance-country-error", "children"),
    Output("market-balance-country-current-table", "children"),
    Output("market-balance-country-delta-table", "children"),
    Output("market-balance-country-balance-figure", "figure"),
    Output("market-balance-country-category-container", "children"),
    Input("market-balance-country-store", "data"),
)
def render_country_balance(store_payload):
    try:
        store_payload = _resolve_market_store(
            store_payload,
            "market-balance-country-v2",
        )
    except _SnapshotUnavailable:
        return (
            html.Div(),
            _market_balance_snapshot_recovery_notice(),
            html.Div(),
            html.Div(),
            _empty_figure(
                MARKET_BALANCE_SNAPSHOT_RECOVERY_MESSAGE
            ),
            html.Div(),
        )
    error_message = (store_payload or {}).get("error")
    if error_message:
        return (
            html.Div(),
            _build_error_banner(error_message),
            html.Div(),
            html.Div(),
            _empty_figure("Country drilldown unavailable"),
            html.Div(),
        )

    data = (store_payload or {}).get("data", {})
    metadata = (store_payload or {}).get("metadata", {})
    current_payload = data.get("current_table")
    delta_payload = data.get("delta_table")
    chart_payload = data.get("balance_chart")
    requested_comparison_snapshot = metadata.get("requested_comparison_snapshot")
    comparison_snapshot = metadata.get("comparison_snapshot")
    comparison_label = comparison_snapshot or requested_comparison_snapshot

    status_lines = [
        f"Country: {metadata.get('country') or 'N/A'}",
        f"Current snapshot: {metadata.get('current_snapshot') or 'N/A'}",
        f"Comparison snapshot: {comparison_label or 'No comparison selected'}",
        f"Hierarchy: {metadata.get('level', 'subtype')} | Time view: {TIME_GROUP_LABELS.get(normalize_time_group(metadata.get('time_group')), 'Monthly')}",
        *(metadata.get("warnings") or []),
    ]
    if (
        requested_comparison_snapshot
        and comparison_snapshot
        and requested_comparison_snapshot != comparison_snapshot
    ):
        status_lines.insert(
            3,
            f"Requested comparison: {requested_comparison_snapshot}",
        )

    category_children = []
    for idx, chart_payload_dict in enumerate(data.get("category_charts") or []):
        chart_df = _deserialize_frame(chart_payload_dict.get("frame"))
        category_children.append(
            html.Div(
                [
                    dcc.Graph(
                        id=f"market-balance-country-category-{idx}",
                        figure=_build_country_category_figure(
                            chart_df,
                            title=chart_payload_dict.get("title", "Category"),
                            chart_type=chart_payload_dict.get("chart_type", "line"),
                        ),
                        style={"height": "360px"},
                    )
                ],
                className="section-container",
                style={"flex": "1", "minWidth": "0"},
            )
        )

    if category_children:
        category_children = [
            html.Div(category_children, style={"display": "flex", "gap": "24px", "flexWrap": "wrap"})
        ]

    return (
        _build_status_block(status_lines),
        html.Div(),
        _build_market_table(
            "market-balance-country-current-grid",
            current_payload,
            column_styles=metadata.get("column_styles"),
            compact=True,
            wrap_multi_word_headers=True,
        ),
        _build_market_table(
            "market-balance-country-delta-grid",
            delta_payload,
            table_mode="delta",
            column_styles=metadata.get("column_styles"),
            compact=True,
            wrap_multi_word_headers=True,
            empty_message=(
                "No Delta vs Snapshot data available for the selected comparison."
                if requested_comparison_snapshot
                else "Select a comparison snapshot to calculate deltas."
            ),
        ),
        _build_country_balance_figure(
            _deserialize_frame(chart_payload),
            metadata.get("country"),
        ),
        category_children,
    )


@callback(
    Output("market-balance-country-download", "data"),
    Input("market-balance-country-export", "n_clicks"),
    State("market-balance-country-store", "data"),
    State("market-balance-country-dropdown", "value"),
    prevent_initial_call=True,
)
def export_country_workbook(n_clicks, store_payload, country):
    if not n_clicks:
        return no_update

    store_payload = _resolve_market_store(
        store_payload,
        "market-balance-country-v2",
    )

    data = (store_payload or {}).get("data", {})
    if not data:
        return no_update

    sheet_map = {
        "Current Data": _deserialize_frame(data.get("current_table")),
        "Delta vs Snapshot": _deserialize_frame(data.get("delta_table")),
        "Balance Chart Data": _deserialize_frame(data.get("balance_chart")),
    }
    for idx, chart_payload in enumerate(data.get("category_charts") or [], start=1):
        sheet_map[f"Category {idx}"] = _deserialize_frame(chart_payload.get("frame"))

    workbook_bytes = build_workbook_bytes(sheet_map)
    safe_country = (country or "country").replace(" ", "_").lower()
    return dcc.send_bytes(workbook_bytes, f"{safe_country}_market_balance.xlsx")
