import datetime as dt

import pandas as pd
from dash import dcc, html, callback, Input, Output, State
import dash_ag_grid as dag
from dash.dash_table.Format import Format, Scheme
from utils.ag_grid_tables import create_ag_grid_from_datatable
from dash.exceptions import PreventUpdate
from sqlalchemy import text

from utils.balance_time import (
    build_lng_season_periods as _build_lng_season_periods,
    filter_by_month_date_range as _filter_by_date_range,
    get_default_interval_window as _get_default_interval_window,
    get_month_date_bounds as _get_date_bounds,
    normalize_month_date as _normalize_month_date,
)
from utils.balance_matrix import (
    align_matrix_to_reference_months as _align_matrix_to_reference_months,
    build_delta_matrix as _build_delta_matrix,
    export_matrix_to_excel_bytes as _export_matrix_to_excel_bytes,
)
from utils.balance_components import (
    build_balance_section_summary as _build_section_summary,
    create_balance_comparison_section as _create_comparison_section,
    create_balance_empty_state as _create_empty_state,
)
from utils.dataframe_store import (
    deserialize_dataframe_store as _deserialize_dataframe_payload,
    serialize_dataframe_store as _serialize_dataframe,
)
from utils.dashboard_snapshot_cache import (
    SnapshotUnavailable as _SnapshotUnavailable,
    is_snapshot_reference as _is_snapshot_reference,
    snapshot_is_resolvable as _snapshot_is_resolvable,
    was_global_refresh_triggered as _was_global_refresh_triggered,
    with_snapshot_slot as _with_snapshot_slot,
)
from utils.provider_flow_snapshot import (
    get_provider_flow_snapshot as _get_provider_flow_snapshot,
    resolve_provider_flow_snapshot as _resolve_provider_flow_snapshot,
)
from utils.historical_comparison_snapshot import (
    get_historical_comparison_frame as _get_historical_comparison_frame,
)
from utils.performance import log_callback_timing
from utils.performance_flags import (
    revision_aware_refresh_enabled as _revision_aware_refresh_enabled,
)
from utils.snapshot_controls import (
    build_ea_metadata_lines as _build_ea_metadata_lines,
    build_woodmac_metadata_lines as _build_woodmac_metadata_lines,
    deserialize_snapshot_value as _deserialize_snapshot_value,
    ea_metadata_from_upload_options as _ea_metadata_from_upload_options,
    resolve_snapshot_control_values as _resolve_snapshot_control_values,
    woodmac_metadata_from_publication_options as _woodmac_metadata_from_publication_options,
)
from utils.import_flow_data import (
    DB_SCHEMA,
    build_import_flow_matrix,
    default_selected_countries,
    engine,
    fetch_ea_import_flow_raw_data,
    fetch_ea_import_flow_metadata,
    fetch_ea_import_flow_raw_data_for_upload,
    fetch_ea_upload_options,
    fetch_woodmac_import_flow_raw_data_for_publications,
    fetch_woodmac_import_flow_raw_data,
    fetch_woodmac_import_flow_metadata,
    fetch_woodmac_publication_options,
    get_available_countries,
)
from utils.flow_country_selection import (
    normalize_selected_flow_values as _normalize_selected_destination_columns,
    resolve_flow_destination_selection,
    sanitize_monthly_country_flow_data,
)
from utils.table_styles import (
    StandardTableStyleManager,
    TABLE_COLORS,
    build_responsive_column_styles as _build_responsive_column_styles,
)


TIME_VIEW_LABELS = {
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "seasonally": "Seasonally",
    "yearly": "Yearly",
}

TIME_VIEW_PERIOD_LABELS = {
    "monthly": "month",
    "quarterly": "quarter",
    "seasonally": "season",
    "yearly": "year",
}

SEASONAL_TIME_VIEW_TOOLTIP = (
    "Seasonally: Summer (Y-S) runs from April to September of year Y. "
    "Winter (Y-W) runs from October to December of year Y and January to March of year Y+1."
)

TIME_VIEW_CONTROL_SHELL_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "gap": "10px",
    "padding": "3px 8px 3px 12px",
    "backgroundColor": "#ffffff",
    "border": "1px solid #dbe4ee",
    "borderRadius": "999px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.05)",
}

DEMAND_STICKY_HEADER_STYLE = {
    "display": "flex",
    "gap": "12px",
    "alignItems": "flex-end",
    "flexWrap": "wrap",
}

UNKNOWN_DESTINATION_GROUP = "Unknown"

DESTINATION_AGGREGATION_LABELS = {
    "country": "Country",
    "continent": "Continent",
    "subcontinent": "Subcontinent",
    "basin": "Basin",
    "country_classification_level1": "Classification Level 1",
    "country_classification": "Classification",
    "shipping_region": "Shipping Region",
}

DESTINATION_AGGREGATION_OPTIONS = [
    {"label": label, "value": value}
    for value, label in DESTINATION_AGGREGATION_LABELS.items()
]

DESTINATION_AGGREGATION_LOOKUP_COLUMNS = [
    "country_name",
    "country",
    "continent",
    "subcontinent",
    "basin",
    "country_classification_level1",
    "country_classification",
    "shipping_region",
]

DEMAND_SNAPSHOT_RECOVERY_MESSAGE = (
    "Cached demand data is unavailable. Click the global Refresh button "
    "to reload it."
)


def _deserialize_dataframe(value):
    if value is None:
        raise _SnapshotUnavailable(
            DEMAND_SNAPSHOT_RECOVERY_MESSAGE
        )
    try:
        resolved = _resolve_provider_flow_snapshot(value)
        if _is_snapshot_reference(resolved):
            raise _SnapshotUnavailable(
                DEMAND_SNAPSHOT_RECOVERY_MESSAGE
            )
    except _SnapshotUnavailable as exc:
        raise _SnapshotUnavailable(
            DEMAND_SNAPSHOT_RECOVERY_MESSAGE
        ) from exc
    if isinstance(resolved, pd.DataFrame):
        return resolved.copy()
    return _deserialize_dataframe_payload(resolved)


def _snapshot_recovery_notice():
    return html.Div(
        DEMAND_SNAPSHOT_RECOVERY_MESSAGE,
        className="balance-error-banner",
        role="alert",
    )


def _snapshot_recovery_option():
    return [
        {
            "label": DEMAND_SNAPSHOT_RECOVERY_MESSAGE,
            "value": "__snapshot_unavailable__",
            "disabled": True,
        }
    ]


def _normalize_mapping_value(value):
    if pd.isna(value):
        return None

    normalized_value = str(value).strip()
    return normalized_value if normalized_value else None


def _collapse_mapping_values(series):
    normalized_values = sorted(
        {
            value
            for value in (_normalize_mapping_value(item) for item in series)
            if value is not None
        }
    )
    if len(normalized_values) == 1:
        return normalized_values[0]

    return UNKNOWN_DESTINATION_GROUP


def _first_non_empty_value(series, fallback=""):
    for item in series:
        normalized_value = _normalize_mapping_value(item)
        if normalized_value is not None:
            return normalized_value

    return fallback


def _sanitize_supply_raw_export_flow(raw_df: pd.DataFrame) -> pd.DataFrame:
    return sanitize_monthly_country_flow_data(
        raw_df,
        unknown_group=UNKNOWN_DESTINATION_GROUP,
    )


def _get_destination_aggregation_lookup_dataframe(
    lookup_records,
) -> pd.DataFrame:
    if not lookup_records:
        return pd.DataFrame(columns=DESTINATION_AGGREGATION_LOOKUP_COLUMNS)

    lookup_df = pd.DataFrame(lookup_records)
    for column_name in DESTINATION_AGGREGATION_LOOKUP_COLUMNS:
        if column_name not in lookup_df.columns:
            lookup_df[column_name] = None

    lookup_df = lookup_df[DESTINATION_AGGREGATION_LOOKUP_COLUMNS].copy()
    lookup_df["country_name"] = lookup_df["country_name"].apply(_normalize_mapping_value)
    lookup_df = lookup_df[lookup_df["country_name"].notna()].copy()
    lookup_df["country"] = lookup_df["country"].apply(_normalize_mapping_value)
    lookup_df["country"] = lookup_df["country"].fillna(lookup_df["country_name"])

    for column_name in DESTINATION_AGGREGATION_LABELS:
        if column_name == "country":
            continue
        lookup_df[column_name] = (
            lookup_df[column_name]
            .apply(_normalize_mapping_value)
            .fillna(UNKNOWN_DESTINATION_GROUP)
        )

    return lookup_df.drop_duplicates(subset=["country_name"]).reset_index(drop=True)


def _build_destination_aggregation_lookup_records(mapping_df: pd.DataFrame) -> list[dict]:
    if mapping_df.empty:
        return []

    if "country_name" not in mapping_df.columns:
        mapping_df["country_name"] = mapping_df.get("country")
    if "country" not in mapping_df.columns:
        mapping_df["country"] = mapping_df["country_name"]

    mapping_df["country_name"] = mapping_df["country_name"].apply(_normalize_mapping_value)
    mapping_df = mapping_df[mapping_df["country_name"].notna()].copy()
    if mapping_df.empty:
        return []

    aggregation_spec = {
        "country": lambda series: _first_non_empty_value(series),
    }
    for column_name in DESTINATION_AGGREGATION_LABELS:
        if column_name == "country":
            continue
        if column_name not in mapping_df.columns:
            mapping_df[column_name] = None
        aggregation_spec[column_name] = _collapse_mapping_values

    deduped_df = mapping_df.groupby("country_name", as_index=False).agg(aggregation_spec)
    deduped_df["country"] = deduped_df["country"].apply(_normalize_mapping_value)
    deduped_df["country"] = deduped_df["country"].fillna(deduped_df["country_name"])
    for column_name in DESTINATION_AGGREGATION_LABELS:
        if column_name == "country":
            continue
        deduped_df[column_name] = deduped_df[column_name].fillna(UNKNOWN_DESTINATION_GROUP)

    return deduped_df[DESTINATION_AGGREGATION_LOOKUP_COLUMNS].to_dict("records")


def _fetch_destination_aggregation_lookup_records() -> list[dict]:
    query = text(
        f"""
        SELECT
            country,
            country_name,
            continent,
            subcontinent,
            basin,
            country_classification_level1,
            country_classification,
            shipping_region
        FROM {DB_SCHEMA}.mappings_country
        """
    )

    with engine.connect() as connection:
        mapping_df = pd.read_sql_query(query, connection)
    return _build_destination_aggregation_lookup_records(mapping_df)


def _sort_destination_group_values(values) -> list[str]:
    normalized_values = sorted(
        {
            value
            for value in (_normalize_mapping_value(item) for item in values)
            if value is not None
        },
        key=lambda item: (item == UNKNOWN_DESTINATION_GROUP, item),
    )
    return normalized_values


def _enrich_export_flow_with_destination_aggregation(
    raw_df: pd.DataFrame,
    destination_aggregation: str,
    lookup_records,
) -> pd.DataFrame:
    sanitized_df = _sanitize_supply_raw_export_flow(raw_df)
    if sanitized_df.empty:
        return sanitized_df.assign(destination_group=pd.Series(dtype="object"))

    if destination_aggregation not in DESTINATION_AGGREGATION_LABELS:
        destination_aggregation = "country"

    if destination_aggregation == "country":
        enriched_df = sanitized_df.copy()
        enriched_df["destination_group"] = enriched_df["country_name"]
        return enriched_df

    lookup_df = _get_destination_aggregation_lookup_dataframe(lookup_records)
    if lookup_df.empty:
        enriched_df = sanitized_df.copy()
        enriched_df["destination_group"] = UNKNOWN_DESTINATION_GROUP
        return enriched_df

    enriched_df = sanitized_df.merge(
        lookup_df[["country_name", destination_aggregation]].rename(
            columns={destination_aggregation: "destination_group"}
        ),
        how="left",
        on="country_name",
    )
    enriched_df["destination_group"] = (
        enriched_df["destination_group"]
        .apply(_normalize_mapping_value)
        .fillna(UNKNOWN_DESTINATION_GROUP)
    )
    return enriched_df


def _get_available_destination_group_values(
    destination_aggregation: str,
    available_countries: list[str] | None,
    lookup_records,
) -> list[str]:
    available_countries = [
        country
        for country in (available_countries or [])
        if _normalize_mapping_value(country) is not None
    ]
    if destination_aggregation == "country":
        return available_countries

    if not available_countries:
        return []

    lookup_df = _get_destination_aggregation_lookup_dataframe(lookup_records)
    if lookup_df.empty:
        return [UNKNOWN_DESTINATION_GROUP]

    filtered_lookup_df = lookup_df[lookup_df["country_name"].isin(available_countries)].copy()
    group_values = []
    if not filtered_lookup_df.empty and destination_aggregation in filtered_lookup_df.columns:
        group_values.extend(filtered_lookup_df[destination_aggregation].tolist())

    mapped_countries = set(filtered_lookup_df["country_name"].tolist())
    if set(available_countries) - mapped_countries:
        group_values.append(UNKNOWN_DESTINATION_GROUP)

    return _sort_destination_group_values(group_values)


def _build_supply_matrix(
    raw_df: pd.DataFrame,
    destination_aggregation: str,
    selected_destination_columns,
    other_countries_mode: str,
    lookup_records,
) -> pd.DataFrame:
    if destination_aggregation not in DESTINATION_AGGREGATION_LABELS:
        destination_aggregation = "country"

    if destination_aggregation == "country":
        return build_import_flow_matrix(
            raw_df,
            _normalize_selected_destination_columns(selected_destination_columns),
            other_countries_mode,
        )

    enriched_df = _enrich_export_flow_with_destination_aggregation(
        raw_df,
        destination_aggregation,
        lookup_records,
    )
    if enriched_df.empty:
        return pd.DataFrame(columns=["Month", "Total MMTPA"])

    visible_columns = _normalize_selected_destination_columns(selected_destination_columns)
    if not visible_columns:
        visible_columns = _sort_destination_group_values(
            enriched_df["destination_group"].tolist()
        )

    bucketed_df = (
        enriched_df.groupby(["month", "destination_group"], as_index=False)["total_mmtpa"]
        .sum()
        .sort_values(["month", "destination_group"])
    )

    pivot_df = (
        bucketed_df.pivot(
            index="month",
            columns="destination_group",
            values="total_mmtpa",
        )
        .fillna(0.0)
        .sort_index()
    )

    month_index = pd.date_range(
        start=enriched_df["month"].min(),
        end=enriched_df["month"].max(),
        freq="MS",
    )
    pivot_df = pivot_df.reindex(month_index, fill_value=0.0)
    pivot_df.index.name = "month"

    for column_name in visible_columns:
        if column_name not in pivot_df.columns:
            pivot_df[column_name] = 0.0

    pivot_df["Total MMTPA"] = pivot_df[visible_columns].sum(axis=1)

    result_df = pivot_df.reset_index().rename(columns={"month": "Month"})
    result_df["Month"] = pd.to_datetime(result_df["Month"]).dt.strftime("%Y-%m")

    ordered_columns = ["Month", "Total MMTPA"] + visible_columns
    result_df = result_df[ordered_columns]
    numeric_columns = [column for column in ordered_columns if column != "Month"]
    result_df[numeric_columns] = result_df[numeric_columns].round(2)
    return result_df


def _resolve_destination_columns_selection(
    destination_aggregation: str,
    selected_values,
    available_countries: list[str],
    lookup_records,
) -> list[str]:
    return resolve_flow_destination_selection(
        destination_aggregation,
        selected_values,
        available_countries,
        lookup_records,
        default_selected_countries_fn=default_selected_countries,
        available_group_values_fn=_get_available_destination_group_values,
    )


def _apply_supply_time_view(matrix_df: pd.DataFrame, time_view: str) -> pd.DataFrame:
    if matrix_df.empty:
        return matrix_df.copy()

    view_df = matrix_df.copy()
    view_df["__axis_date"] = pd.to_datetime(
        view_df["Month"].astype(str),
        errors="coerce",
    ).dt.to_period("M").dt.to_timestamp()
    view_df = view_df.dropna(subset=["__axis_date"]).sort_values("__axis_date").reset_index(drop=True)
    if view_df.empty or time_view == "monthly":
        return view_df.drop(columns=["__axis_date"], errors="ignore").reset_index(drop=True)

    if time_view == "quarterly":
        view_df["__period_start"] = view_df["__axis_date"].dt.to_period("Q").dt.start_time
        view_df["__period_label"] = (
            view_df["__axis_date"].dt.year.astype(str)
            + "-Q"
            + view_df["__axis_date"].dt.quarter.astype(str)
        )
    elif time_view == "seasonally":
        (
            view_df["__period_start"],
            view_df["__period_label"],
        ) = _build_lng_season_periods(view_df["__axis_date"])
    elif time_view == "yearly":
        view_df["__period_start"] = view_df["__axis_date"].dt.to_period("Y").dt.start_time
        view_df["__period_label"] = view_df["__axis_date"].dt.year.astype(str)
    else:
        return view_df.drop(columns=["__axis_date"], errors="ignore").reset_index(drop=True)

    numeric_columns = [column for column in matrix_df.columns if column != "Month"]
    non_total_numeric_columns = [
        column for column in numeric_columns if column != "Total MMTPA"
    ]
    weighted_columns = (
        non_total_numeric_columns
        if non_total_numeric_columns
        else [column for column in numeric_columns if column == "Total MMTPA"]
    )

    view_df["__days_in_month"] = view_df["__axis_date"].dt.days_in_month.astype(float)
    for column_name in weighted_columns:
        numeric_series = pd.to_numeric(view_df[column_name], errors="coerce")
        view_df[f"__weighted__{column_name}"] = (
            numeric_series.fillna(0.0) * view_df["__days_in_month"]
        )
        view_df[f"__available__{column_name}"] = numeric_series.notna().astype(int)

    aggregation_kwargs = {
        "__month_count": ("Month", "size"),
        "__days_in_month": ("__days_in_month", "sum"),
    }
    for column_name in weighted_columns:
        aggregation_kwargs[f"__weighted__{column_name}"] = (
            f"__weighted__{column_name}",
            "sum",
        )
        aggregation_kwargs[f"__available__{column_name}"] = (
            f"__available__{column_name}",
            "sum",
        )

    grouped = (
        view_df.groupby(["__period_start", "__period_label"], as_index=False)
        .agg(**aggregation_kwargs)
        .sort_values("__period_start")
        .reset_index(drop=True)
    )

    result_df = pd.DataFrame({"Month": grouped["__period_label"]})
    for column_name in non_total_numeric_columns:
        result_df[column_name] = (
            grouped[f"__weighted__{column_name}"] / grouped["__days_in_month"]
        )
        result_df.loc[
            grouped[f"__available__{column_name}"] < grouped["__month_count"],
            column_name,
        ] = float("nan")

    if "Total MMTPA" in numeric_columns:
        if non_total_numeric_columns:
            result_df["Total MMTPA"] = result_df[non_total_numeric_columns].sum(
                axis=1,
                min_count=len(non_total_numeric_columns),
            )
        else:
            result_df["Total MMTPA"] = (
                grouped["__weighted__Total MMTPA"] / grouped["__days_in_month"]
            )
            result_df.loc[
                grouped["__available__Total MMTPA"] < grouped["__month_count"],
                "Total MMTPA",
            ] = float("nan")

    ordered_columns = [column for column in matrix_df.columns if column in result_df.columns]
    result_df = result_df[ordered_columns]
    result_numeric_columns = [column for column in result_df.columns if column != "Month"]
    if result_numeric_columns:
        result_df[result_numeric_columns] = result_df[result_numeric_columns].round(2)

    return result_df.reset_index(drop=True)


def _create_balance_table(
    table_id: str,
    df: pd.DataFrame,
    table_mode: str = "absolute",
) -> dag.AgGrid | html.Div:
    if df.empty:
        return _create_empty_state("No data available for the current selection.")

    base_config = StandardTableStyleManager.get_base_datatable_config()
    numeric_columns = [column for column in df.columns if column != "Month"]

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

    style_data_conditional = list(base_config["style_data_conditional"])
    style_data_conditional.append(
        {
            "if": {"column_id": "Month"},
            "backgroundColor": "#f8fafc",
            "fontWeight": "600",
            "color": TABLE_COLORS["text_primary"],
        }
    )

    if table_mode == "delta":
        for column_name in numeric_columns:
            style_data_conditional.extend(
                [
                    {
                        "if": {
                            "column_id": column_name,
                            "filter_query": f"{{{column_name}}} > 0",
                        },
                        "backgroundColor": "#ecfdf5",
                        "color": "#166534",
                        "fontWeight": "600",
                    },
                    {
                        "if": {
                            "column_id": column_name,
                            "filter_query": f"{{{column_name}}} < 0",
                        },
                        "backgroundColor": "#fef2f2",
                        "color": "#991b1b",
                        "fontWeight": "600",
                    },
                    {
                        "if": {
                            "column_id": column_name,
                            "filter_query": f"{{{column_name}}} = 0",
                        },
                        "backgroundColor": "#f8fafc",
                        "color": "#64748b",
                    },
                ]
            )
    else:
        style_data_conditional.extend(
            [
                {
                    "if": {"column_id": "Total MMTPA"},
                    "backgroundColor": "#edf6fd",
                    "fontWeight": "700",
                    "color": TABLE_COLORS["primary_dark"],
                },
                {
                    "if": {"column_id": "Rest of the World"},
                    "backgroundColor": "#f8f9fa",
                    "color": TABLE_COLORS["text_secondary"],
                },
            ]
        )

    return create_ag_grid_from_datatable(
        id=table_id,
        columns=columns,
        data=df.where(pd.notna(df), None).to_dict("records"),
        sort_action="native",
        page_action="none",
        fill_width=True,
        fixed_columns={"headers": True, "data": 1},
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": "560px",
            "width": "100%",
            "minWidth": "100%",
        },
        style_cell_conditional=_build_responsive_column_styles(df),
        style_data_conditional=style_data_conditional,
    )


def _fetch_comparison_raw_df(
    comparison_source: str,
    short_term_value: str | None,
    long_term_value: str | None,
    ea_upload_value: int | None,
    start_date: str | None = None,
    end_date: str | None = None,
    base_reference=None,
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        if comparison_source == "ea":
            if not ea_upload_value:
                return None, "No Energy Aspects comparison run available."
            _, comparison_df = _get_historical_comparison_frame(
                direction="demand",
                base_reference=base_reference,
                selection={
                    "source": "ea",
                    "run_id": int(ea_upload_value),
                },
                query_dependencies={
                    "start_date": start_date,
                    "end_date": end_date,
                },
                builder=lambda: fetch_ea_import_flow_raw_data_for_upload(
                    ea_upload_value,
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
            direction="demand",
            base_reference=base_reference,
            selection=selection,
            query_dependencies={},
            builder=lambda: fetch_woodmac_import_flow_raw_data_for_publications(
                short_term_snapshot.get("market_outlook"),
                short_term_snapshot.get("publication_timestamp"),
                long_term_snapshot.get("market_outlook"),
                long_term_snapshot.get("publication_timestamp"),
            ),
        )
        return comparison_df, None
    except Exception as exc:
        return None, f"Comparison load failed: {exc}"


def _build_delta_comparison_output(
    baseline_raw_df: pd.DataFrame,
    comparison_raw_df: pd.DataFrame | None,
    destination_aggregation: str,
    selected_destination_columns,
    other_countries_mode: str,
    destination_aggregation_lookup,
    time_view: str,
    empty_baseline_message: str,
    comparison_table_id: str,
    comparison_error_message: str | None = None,
):
    baseline_matrix = _build_supply_matrix(
        baseline_raw_df,
        destination_aggregation,
        selected_destination_columns,
        other_countries_mode,
        destination_aggregation_lookup,
    )
    baseline_month_labels = baseline_matrix["Month"].tolist()

    if baseline_matrix.empty:
        return (
            html.Div(),
            _create_empty_state(empty_baseline_message),
        )

    if (
        destination_aggregation == "country"
        and selected_destination_columns == []
        and other_countries_mode == "exclude"
    ):
        return (
            html.Div(),
            _create_empty_state(
                "Select at least one country or switch to Rest of the World mode."
            ),
        )

    if comparison_error_message:
        return (
            html.Div(),
            _create_empty_state(
                "Unable to load comparison snapshot."
                if comparison_error_message.startswith("Comparison load failed:")
                else comparison_error_message
            ),
        )

    comparison_matrix = _build_supply_matrix(
        comparison_raw_df if comparison_raw_df is not None else pd.DataFrame(),
        destination_aggregation,
        selected_destination_columns,
        other_countries_mode,
        destination_aggregation_lookup,
    )
    comparison_matrix = _align_matrix_to_reference_months(
        comparison_matrix,
        baseline_month_labels,
    )
    baseline_matrix = _apply_supply_time_view(baseline_matrix, time_view)
    comparison_matrix = _apply_supply_time_view(comparison_matrix, time_view)
    delta_matrix = _build_delta_matrix(baseline_matrix, comparison_matrix)

    comparison_table = _create_balance_table(
        comparison_table_id,
        delta_matrix,
        table_mode="delta",
    )

    return html.Div(), comparison_table


def _create_woodmac_comparison_section() -> html.Div:
    return _create_comparison_section(
        title="WoodMac Imports Flow",
        export_button_id="demand-export-woodmac-button",
        baseline_summary_id="demand-woodmac-summary",
        default_comparison_source="woodmac",
        comparison_source_dropdown_id="demand-comparison-source-dropdown",
        comparison_st_dropdown_id="demand-comparison-st-dropdown",
        comparison_lt_dropdown_id="demand-comparison-lt-dropdown",
        comparison_ea_upload_dropdown_id="demand-comparison-ea-upload-dropdown",
        comparison_woodmac_controls_id="demand-comparison-woodmac-controls",
        comparison_ea_controls_id="demand-comparison-ea-controls",
        baseline_table_container_id="demand-woodmac-table-container",
        comparison_summary_id="demand-comparison-summary",
        comparison_table_container_id="demand-comparison-table-container",
    )


def _create_ea_comparison_section() -> html.Div:
    return _create_comparison_section(
        title="Energy Aspects Imports Flow",
        export_button_id="demand-export-ea-button",
        baseline_summary_id="demand-ea-summary",
        default_comparison_source="ea",
        comparison_source_dropdown_id="demand-ea-comparison-source-dropdown",
        comparison_st_dropdown_id="demand-ea-comparison-st-dropdown",
        comparison_lt_dropdown_id="demand-ea-comparison-lt-dropdown",
        comparison_ea_upload_dropdown_id="demand-ea-comparison-ea-upload-dropdown",
        comparison_woodmac_controls_id="demand-ea-comparison-woodmac-controls",
        comparison_ea_controls_id="demand-ea-comparison-ea-controls",
        baseline_table_container_id="demand-ea-table-container",
        comparison_summary_id="demand-ea-comparison-summary",
        comparison_table_container_id="demand-ea-comparison-table-container",
    )


layout = html.Div(
    [
        dcc.Store(id="demand-woodmac-data-store", storage_type="memory"),
        dcc.Store(id="demand-ea-data-store", storage_type="memory"),
        dcc.Store(id="demand-country-options-store", storage_type="memory"),
        dcc.Store(id="demand-destination-aggregation-lookup-store", storage_type="memory"),
        dcc.Store(id="demand-country-columns-selection-store", storage_type="memory"),
        dcc.Store(id="demand-load-error-store", storage_type="memory"),
        dcc.Store(id="demand-woodmac-metadata-store", storage_type="memory"),
        dcc.Store(id="demand-ea-metadata-store", storage_type="memory"),
        dcc.Store(id="demand-comparison-options-store", storage_type="memory"),
        dcc.Download(id="demand-download-woodmac-excel"),
        dcc.Download(id="demand-download-ea-excel"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Date Range",
                            htmlFor="demand-date-range",
                            className="filter-group-header",
                        ),
                        html.Div(
                            [
                                dcc.DatePickerRange(
                                    id="demand-date-range",
                                    start_date=None,
                                    end_date=None,
                                    min_date_allowed=None,
                                    max_date_allowed=None,
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
                        html.Label(
                            "Time View",
                            htmlFor="demand-time-view",
                            className="filter-group-header",
                            title=SEASONAL_TIME_VIEW_TOOLTIP,
                            style={"cursor": "help"},
                        ),
                        html.Div(
                            [
                                dcc.RadioItems(
                                    id="demand-time-view",
                                    options=[
                                        {"label": "Monthly", "value": "monthly"},
                                        {"label": "Quarterly", "value": "quarterly"},
                                        {"label": "Seasonally", "value": "seasonally"},
                                        {"label": "Yearly", "value": "yearly"},
                                    ],
                                    value="yearly",
                                    inline=True,
                                    labelStyle={
                                        "display": "inline-flex",
                                        "alignItems": "center",
                                        "marginRight": "10px",
                                        "fontSize": "12px",
                                        "fontWeight": "600",
                                        "color": "#334155",
                                    },
                                    inputStyle={"marginRight": "6px"},
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "flexWrap": "wrap",
                                    },
                                )
                            ],
                            style=TIME_VIEW_CONTROL_SHELL_STYLE,
                        ),
                    ],
                    className="filter-group",
                    style={"minWidth": "360px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Destination Aggregation",
                            htmlFor="demand-destination-aggregation-dropdown",
                            className="filter-group-header",
                        ),
                        dcc.Dropdown(
                            id="demand-destination-aggregation-dropdown",
                            options=DESTINATION_AGGREGATION_OPTIONS,
                            value="country",
                            clearable=False,
                            className="filter-dropdown",
                            style={"minWidth": "220px"},
                        ),
                    ],
                    className="filter-group",
                    style={"minWidth": "220px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Country Columns",
                            id="demand-country-columns-header",
                            htmlFor="demand-country-dropdown",
                            className="filter-group-header",
                        ),
                        dcc.Dropdown(
                            id="demand-country-dropdown",
                            options=[],
                            value=None,
                            multi=True,
                            placeholder="Select countries to keep as separate columns",
                            className="filter-dropdown",
                            style={"minWidth": "340px"},
                        ),
                    ],
                    className="filter-group",
                    style={"flex": "1", "minWidth": "340px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Other Countries",
                            htmlFor="demand-other-country-mode",
                            className="filter-group-header",
                        ),
                        dcc.RadioItems(
                            id="demand-other-country-mode",
                            options=[
                                {
                                    "label": "Include as Rest of the World",
                                    "value": "rest_of_world",
                                },
                                {
                                    "label": "Exclude from the table",
                                    "value": "exclude",
                                },
                            ],
                            value="rest_of_world",
                            className="balance-radio-group",
                            labelStyle={
                                "display": "inline-flex",
                                "alignItems": "center",
                                "marginRight": "10px",
                            },
                            inputStyle={"marginRight": "6px"},
                            style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"},
                        ),
                    ],
                    className="filter-group",
                    style={"minWidth": "300px"},
                ),
            ],
            className="professional-section-header balance-desktop-filter-grid",
            style=DEMAND_STICKY_HEADER_STYLE,
        ),
        html.Div(
            [
                html.Div(id="demand-load-error-banner"),
                dcc.Loading(
                    children=[
                        html.Div(
                            [
                                _create_woodmac_comparison_section(),
                                _create_ea_comparison_section(),
                            ],
                            className="balance-results-stack",
                        )
                    ],
                    type="default",
                ),
            ],
            className="balance-page-shell",
        ),
    ]
)


@callback(
    Output("demand-woodmac-data-store", "data"),
    Output("demand-ea-data-store", "data"),
    Output("demand-country-options-store", "data"),
    Output("demand-destination-aggregation-lookup-store", "data"),
    Output("demand-load-error-store", "data"),
    Output("demand-woodmac-metadata-store", "data"),
    Output("demand-ea-metadata-store", "data"),
    Output("demand-comparison-options-store", "data"),
    Input("global-refresh-button", "n_clicks"),
)
@log_callback_timing("demand.source_load")
def load_balance_source_data(_):
    woodmac_df = pd.DataFrame()
    ea_df = pd.DataFrame()
    woodmac_metadata = {}
    ea_metadata = {}
    destination_aggregation_lookup = []
    comparison_options = {
        "woodmac": {"short_term": [], "long_term": []},
        "ea_comparison_runs": [],
    }
    errors = []

    try:
        provider_reference, provider_payload = _get_provider_flow_snapshot(
            force=(
                not _revision_aware_refresh_enabled()
                and _was_global_refresh_triggered()
            )
        )
        if not _snapshot_is_resolvable(provider_reference):
            raise _SnapshotUnavailable(
                DEMAND_SNAPSHOT_RECOVERY_MESSAGE
            )
    except _SnapshotUnavailable:
        raise
    except Exception as exc:
        raise _SnapshotUnavailable(
            DEMAND_SNAPSHOT_RECOVERY_MESSAGE
        ) from exc

    provider_errors = provider_payload.get("errors") or {}

    woodmac_df = provider_payload.get("woodmac_import", pd.DataFrame())
    ea_df = provider_payload.get("ea_import", pd.DataFrame())
    if "woodmac_import" in provider_errors:
        errors.append(f"WoodMac load failed: {provider_errors['woodmac_import']}")
    if "ea_import" in provider_errors:
        errors.append(f"Energy Aspects load failed: {provider_errors['ea_import']}")

    try:
        if "woodmac_import_options" in provider_errors:
            raise RuntimeError(provider_errors["woodmac_import_options"])
        comparison_options["woodmac"] = provider_payload.get("woodmac_import_options") or {}
        woodmac_metadata = _woodmac_metadata_from_publication_options(comparison_options["woodmac"])
    except Exception as exc:
        errors.append(f"WoodMac comparison options load failed: {exc}")
        try:
            woodmac_metadata = fetch_woodmac_import_flow_metadata()
        except Exception as metadata_exc:
            errors.append(f"WoodMac metadata load failed: {metadata_exc}")

    try:
        if "ea_import_options" in provider_errors:
            raise RuntimeError(provider_errors["ea_import_options"])
        comparison_options["ea_comparison_runs"] = (
            provider_payload.get("ea_comparison_runs")
            or provider_payload.get("ea_import_options")
            or []
        )
        ea_metadata = _ea_metadata_from_upload_options(
            provider_payload.get("current_ea")
        )
    except Exception as exc:
        errors.append(f"Energy Aspects comparison options load failed: {exc}")
        try:
            ea_metadata = fetch_ea_import_flow_metadata()
        except Exception as metadata_exc:
            errors.append(f"Energy Aspects metadata load failed: {metadata_exc}")

    try:
        if "mapping" in provider_errors:
            raise RuntimeError(provider_errors["mapping"])
        mapping_df = provider_payload.get("mapping", pd.DataFrame())
        destination_aggregation_lookup = _build_destination_aggregation_lookup_records(mapping_df)
    except Exception as exc:
        errors.append(f"Destination aggregation lookup load failed: {exc}")

    available_countries = get_available_countries([woodmac_df, ea_df])
    error_message = " | ".join(errors) if errors else None

    woodmac_store = _with_snapshot_slot(
        provider_reference,
        "woodmac_import",
    )
    ea_store = _with_snapshot_slot(
        provider_reference,
        "ea_import",
    )

    return (
        woodmac_store,
        ea_store,
        available_countries,
        destination_aggregation_lookup,
        error_message,
        woodmac_metadata,
        ea_metadata,
        comparison_options,
    )


@callback(
    Output("demand-country-dropdown", "options"),
    Output("demand-country-dropdown", "value"),
    Output("demand-country-dropdown", "disabled"),
    Output("demand-country-columns-header", "children"),
    Output("demand-country-dropdown", "placeholder"),
    Output("demand-other-country-mode", "disabled"),
    Output("demand-country-columns-selection-store", "data"),
    Input("demand-country-options-store", "data"),
    Input("demand-destination-aggregation-dropdown", "value"),
    Input("demand-destination-aggregation-lookup-store", "data"),
    Input("demand-woodmac-data-store", "data"),
    Input("demand-ea-data-store", "data"),
    Input("demand-date-range", "start_date"),
    Input("demand-date-range", "end_date"),
    State("demand-country-dropdown", "value"),
    State("demand-country-columns-selection-store", "data"),
)
def update_balance_country_options(
    available_countries,
    destination_aggregation,
    destination_aggregation_lookup,
    woodmac_data,
    ea_data,
    start_date,
    end_date,
    current_selection,
    remembered_country_selection,
):
    available_countries = available_countries or []
    available_country_set = set(available_countries)
    default_country_selection = default_selected_countries(available_countries)
    remembered_selection_provided = remembered_country_selection is not None
    remembered_country_selection = [
        country
        for country in _normalize_selected_destination_columns(remembered_country_selection)
        if country in available_country_set
    ]
    normalized_current_selection = _normalize_selected_destination_columns(current_selection)
    current_country_selection = [
        country
        for country in normalized_current_selection
        if country in available_country_set
    ]
    current_selection_is_country_like = (
        current_selection is not None
        and len(normalized_current_selection) == len(current_country_selection)
    )

    country_options = [
        {"label": country, "value": country}
        for country in available_countries
    ]

    destination_aggregation = (
        destination_aggregation
        if destination_aggregation in DESTINATION_AGGREGATION_LABELS
        else "country"
    )

    if destination_aggregation == "country":
        if current_selection_is_country_like:
            selected_values = current_country_selection
        elif remembered_selection_provided:
            selected_values = remembered_country_selection
        else:
            selected_values = default_country_selection

        return (
            country_options,
            selected_values,
            False,
            "Country Columns",
            "Select countries to keep as separate columns",
            False,
            selected_values,
        )

    try:
        filtered_woodmac_df = _filter_by_date_range(
            _deserialize_dataframe(woodmac_data),
            start_date,
            end_date,
        )
        filtered_ea_df = _filter_by_date_range(
            _deserialize_dataframe(ea_data),
            start_date,
            end_date,
        )
    except _SnapshotUnavailable:
        return (
            _snapshot_recovery_option(),
            [],
            True,
            "Country Columns",
            DEMAND_SNAPSHOT_RECOVERY_MESSAGE,
            True,
            [],
        )
    filtered_available_countries = get_available_countries(
        [filtered_woodmac_df, filtered_ea_df]
    )
    aggregation_values = _get_available_destination_group_values(
        destination_aggregation,
        filtered_available_countries,
        destination_aggregation_lookup,
    )
    aggregation_options = [
        {"label": value, "value": value}
        for value in aggregation_values
    ]
    preserved_country_selection = (
        current_country_selection
        or remembered_country_selection
        or default_country_selection
    )
    aggregation_label = DESTINATION_AGGREGATION_LABELS.get(
        destination_aggregation,
        "Destination",
    )
    if current_selection_is_country_like:
        preserved_country_selection = current_country_selection
    elif remembered_selection_provided:
        preserved_country_selection = remembered_country_selection
    else:
        preserved_country_selection = default_country_selection

    return (
        aggregation_options,
        aggregation_values,
        True,
        "Destination Columns",
        f"All {aggregation_label.lower()} groups are shown as columns",
        True,
        preserved_country_selection,
    )


@callback(
    Output("demand-comparison-st-dropdown", "options"),
    Output("demand-comparison-st-dropdown", "value"),
    Output("demand-comparison-lt-dropdown", "options"),
    Output("demand-comparison-lt-dropdown", "value"),
    Output("demand-comparison-ea-upload-dropdown", "options"),
    Output("demand-comparison-ea-upload-dropdown", "value"),
    Output("demand-comparison-woodmac-controls", "style"),
    Output("demand-comparison-ea-controls", "style"),
    Input("demand-comparison-source-dropdown", "value"),
    Input("demand-comparison-options-store", "data"),
    State("demand-comparison-st-dropdown", "value"),
    State("demand-comparison-lt-dropdown", "value"),
    State("demand-comparison-ea-upload-dropdown", "value"),
)
def update_comparison_snapshot_controls(
    comparison_source,
    comparison_options,
    current_st_value,
    current_lt_value,
    current_ea_upload_value,
):
    return _resolve_snapshot_control_values(
        comparison_source,
        comparison_options,
        current_st_value,
        current_lt_value,
        current_ea_upload_value,
    )


@callback(
    Output("demand-ea-comparison-st-dropdown", "options"),
    Output("demand-ea-comparison-st-dropdown", "value"),
    Output("demand-ea-comparison-lt-dropdown", "options"),
    Output("demand-ea-comparison-lt-dropdown", "value"),
    Output("demand-ea-comparison-ea-upload-dropdown", "options"),
    Output("demand-ea-comparison-ea-upload-dropdown", "value"),
    Output("demand-ea-comparison-woodmac-controls", "style"),
    Output("demand-ea-comparison-ea-controls", "style"),
    Input("demand-ea-comparison-source-dropdown", "value"),
    Input("demand-comparison-options-store", "data"),
    State("demand-ea-comparison-st-dropdown", "value"),
    State("demand-ea-comparison-lt-dropdown", "value"),
    State("demand-ea-comparison-ea-upload-dropdown", "value"),
)
def update_ea_comparison_snapshot_controls(
    comparison_source,
    comparison_options,
    current_st_value,
    current_lt_value,
    current_ea_upload_value,
):
    return _resolve_snapshot_control_values(
        comparison_source,
        comparison_options,
        current_st_value,
        current_lt_value,
        current_ea_upload_value,
    )


@callback(
    Output("demand-date-range", "min_date_allowed"),
    Output("demand-date-range", "max_date_allowed"),
    Output("demand-date-range", "start_date"),
    Output("demand-date-range", "end_date"),
    Input("demand-woodmac-data-store", "data"),
    Input("demand-ea-data-store", "data"),
    State("demand-date-range", "start_date"),
    State("demand-date-range", "end_date"),
)
def update_balance_date_range(woodmac_data, ea_data, current_start_date, current_end_date):
    try:
        woodmac_raw_df = _deserialize_dataframe(woodmac_data)
        ea_raw_df = _deserialize_dataframe(ea_data)
    except _SnapshotUnavailable:
        return None, None, None, None

    min_date, max_date = _get_date_bounds([woodmac_raw_df, ea_raw_df])
    if min_date is None or max_date is None:
        return None, None, None, None

    normalized_min = _normalize_month_date(min_date)
    normalized_max = _normalize_month_date(max_date)
    default_start, default_end = _get_default_interval_window()

    normalized_start = _normalize_month_date(current_start_date) or default_start
    normalized_end = _normalize_month_date(current_end_date) or default_end

    if normalized_start < normalized_min:
        normalized_start = normalized_min
    if normalized_end > normalized_max:
        normalized_end = normalized_max
    if normalized_start > normalized_end:
        normalized_start = normalized_min
        normalized_end = normalized_max

    return (
        normalized_min.strftime("%Y-%m-%d"),
        normalized_max.strftime("%Y-%m-%d"),
        normalized_start.strftime("%Y-%m-%d"),
        normalized_end.strftime("%Y-%m-%d"),
    )


@callback(
    Output("demand-load-error-banner", "children"),
    Input("demand-load-error-store", "data"),
)
def update_balance_error_banner(error_message):
    if not error_message:
        return html.Div()

    return html.Div(error_message, className="balance-error-banner")


@callback(
    Output("demand-woodmac-summary", "children"),
    Output("demand-woodmac-table-container", "children"),
    Output("demand-ea-summary", "children"),
    Output("demand-ea-table-container", "children"),
    Input("demand-woodmac-data-store", "data"),
    Input("demand-ea-data-store", "data"),
    Input("demand-woodmac-metadata-store", "data"),
    Input("demand-ea-metadata-store", "data"),
    Input("demand-country-dropdown", "value"),
    Input("demand-other-country-mode", "value"),
    Input("demand-date-range", "start_date"),
    Input("demand-date-range", "end_date"),
    Input("demand-time-view", "value"),
    Input("demand-destination-aggregation-dropdown", "value"),
    Input("demand-destination-aggregation-lookup-store", "data"),
)
@log_callback_timing("demand.tables_render")
def render_balance_tables(
    woodmac_data,
    ea_data,
    woodmac_metadata,
    ea_metadata,
    selected_destination_columns,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    destination_aggregation,
    destination_aggregation_lookup,
):
    try:
        woodmac_raw_df = _filter_by_date_range(
            _deserialize_dataframe(woodmac_data),
            start_date,
            end_date,
        )
        ea_raw_df = _filter_by_date_range(
            _deserialize_dataframe(ea_data),
            start_date,
            end_date,
        )
    except _SnapshotUnavailable:
        return (
            _snapshot_recovery_notice(),
            _create_empty_state(
                DEMAND_SNAPSHOT_RECOVERY_MESSAGE
            ),
            _snapshot_recovery_notice(),
            _create_empty_state(
                DEMAND_SNAPSHOT_RECOVERY_MESSAGE
            ),
        )

    available_countries = get_available_countries([woodmac_raw_df, ea_raw_df])
    resolved_destination_columns = _resolve_destination_columns_selection(
        destination_aggregation,
        selected_destination_columns,
        available_countries,
        destination_aggregation_lookup,
    )

    woodmac_matrix = _build_supply_matrix(
        woodmac_raw_df,
        destination_aggregation,
        resolved_destination_columns,
        other_countries_mode,
        destination_aggregation_lookup,
    )
    woodmac_matrix = _apply_supply_time_view(woodmac_matrix, time_view)
    ea_matrix = _build_supply_matrix(
        ea_raw_df,
        destination_aggregation,
        resolved_destination_columns,
        other_countries_mode,
        destination_aggregation_lookup,
    )
    ea_matrix = _apply_supply_time_view(ea_matrix, time_view)

    woodmac_summary = _build_section_summary(
        woodmac_raw_df,
        woodmac_matrix,
        destination_aggregation,
        other_countries_mode,
        _build_woodmac_metadata_lines(woodmac_metadata),
        time_view=time_view,
        time_view_period_labels=TIME_VIEW_PERIOD_LABELS,
        destination_aggregation_labels=DESTINATION_AGGREGATION_LABELS,
    )
    ea_summary = _build_section_summary(
        ea_raw_df,
        ea_matrix,
        destination_aggregation,
        other_countries_mode,
        _build_ea_metadata_lines(ea_metadata),
        time_view=time_view,
        time_view_period_labels=TIME_VIEW_PERIOD_LABELS,
        destination_aggregation_labels=DESTINATION_AGGREGATION_LABELS,
    )

    if (
        destination_aggregation == "country"
        and resolved_destination_columns == []
        and other_countries_mode == "exclude"
    ):
        empty_message = _create_empty_state(
            "Select at least one country or switch to Rest of the World mode."
        )
        return woodmac_summary, empty_message, ea_summary, empty_message

    woodmac_table = _create_balance_table("demand-woodmac-table", woodmac_matrix)
    ea_table = _create_balance_table("demand-ea-table", ea_matrix)

    return woodmac_summary, woodmac_table, ea_summary, ea_table


@callback(
    Output("demand-comparison-summary", "children"),
    Output("demand-comparison-table-container", "children"),
    Input("demand-woodmac-data-store", "data"),
    Input("demand-ea-data-store", "data"),
    Input("demand-country-dropdown", "value"),
    Input("demand-other-country-mode", "value"),
    Input("demand-date-range", "start_date"),
    Input("demand-date-range", "end_date"),
    Input("demand-time-view", "value"),
    Input("demand-destination-aggregation-dropdown", "value"),
    Input("demand-destination-aggregation-lookup-store", "data"),
    Input("demand-comparison-source-dropdown", "value"),
    Input("demand-comparison-st-dropdown", "value"),
    Input("demand-comparison-lt-dropdown", "value"),
    Input("demand-comparison-ea-upload-dropdown", "value"),
)
@log_callback_timing("demand.woodmac_comparison_render")
def render_comparison_delta_table(
    woodmac_data,
    ea_data,
    selected_destination_columns,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    destination_aggregation,
    destination_aggregation_lookup,
    comparison_source,
    short_term_value,
    long_term_value,
    ea_upload_value,
):
    try:
        baseline_raw_df = _filter_by_date_range(
            _deserialize_dataframe(woodmac_data),
            start_date,
            end_date,
        )
        ea_filtered_df = _filter_by_date_range(
            _deserialize_dataframe(ea_data),
            start_date,
            end_date,
        )
    except _SnapshotUnavailable:
        return (
            _snapshot_recovery_notice(),
            _create_empty_state(
                DEMAND_SNAPSHOT_RECOVERY_MESSAGE
            ),
        )

    available_countries = get_available_countries([baseline_raw_df, ea_filtered_df])
    resolved_destination_columns = _resolve_destination_columns_selection(
        destination_aggregation,
        selected_destination_columns,
        available_countries,
        destination_aggregation_lookup,
    )

    comparison_raw_df, comparison_error_message = _fetch_comparison_raw_df(
        comparison_source,
        short_term_value,
        long_term_value,
        ea_upload_value,
        start_date,
        end_date,
        base_reference=woodmac_data,
    )

    comparison_filtered_df = _filter_by_date_range(
        comparison_raw_df if comparison_raw_df is not None else pd.DataFrame(),
        start_date,
        end_date,
    )
    return _build_delta_comparison_output(
        baseline_raw_df,
        comparison_filtered_df,
        destination_aggregation,
        resolved_destination_columns,
        other_countries_mode,
        destination_aggregation_lookup,
        time_view,
        "No baseline WoodMac import data available for the current selection.",
        "demand-comparison-delta-table",
        comparison_error_message=comparison_error_message,
    )

 

@callback(
    Output("demand-ea-comparison-summary", "children"),
    Output("demand-ea-comparison-table-container", "children"),
    Input("demand-woodmac-data-store", "data"),
    Input("demand-ea-data-store", "data"),
    Input("demand-country-dropdown", "value"),
    Input("demand-other-country-mode", "value"),
    Input("demand-date-range", "start_date"),
    Input("demand-date-range", "end_date"),
    Input("demand-time-view", "value"),
    Input("demand-destination-aggregation-dropdown", "value"),
    Input("demand-destination-aggregation-lookup-store", "data"),
    Input("demand-ea-comparison-source-dropdown", "value"),
    Input("demand-ea-comparison-st-dropdown", "value"),
    Input("demand-ea-comparison-lt-dropdown", "value"),
    Input("demand-ea-comparison-ea-upload-dropdown", "value"),
)
@log_callback_timing("demand.ea_comparison_render")
def render_ea_comparison_delta_table(
    woodmac_data,
    ea_data,
    selected_destination_columns,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    destination_aggregation,
    destination_aggregation_lookup,
    comparison_source,
    short_term_value,
    long_term_value,
    ea_upload_value,
):
    try:
        baseline_raw_df = _filter_by_date_range(
            _deserialize_dataframe(ea_data),
            start_date,
            end_date,
        )
        woodmac_filtered_df = _filter_by_date_range(
            _deserialize_dataframe(woodmac_data),
            start_date,
            end_date,
        )
    except _SnapshotUnavailable:
        return (
            _snapshot_recovery_notice(),
            _create_empty_state(
                DEMAND_SNAPSHOT_RECOVERY_MESSAGE
            ),
        )

    available_countries = get_available_countries([baseline_raw_df, woodmac_filtered_df])
    resolved_destination_columns = _resolve_destination_columns_selection(
        destination_aggregation,
        selected_destination_columns,
        available_countries,
        destination_aggregation_lookup,
    )

    comparison_raw_df, comparison_error_message = _fetch_comparison_raw_df(
        comparison_source,
        short_term_value,
        long_term_value,
        ea_upload_value,
        start_date,
        end_date,
        base_reference=ea_data,
    )

    comparison_filtered_df = _filter_by_date_range(
        comparison_raw_df if comparison_raw_df is not None else pd.DataFrame(),
        start_date,
        end_date,
    )
    return _build_delta_comparison_output(
        baseline_raw_df,
        comparison_filtered_df,
        destination_aggregation,
        resolved_destination_columns,
        other_countries_mode,
        destination_aggregation_lookup,
        time_view,
        "No baseline Energy Aspects import data available for the current selection.",
        "demand-ea-comparison-delta-table",
        comparison_error_message=comparison_error_message,
    )


def _build_filtered_matrix_for_export(
    source_data,
    selected_destination_columns,
    other_countries_mode,
    start_date,
    end_date,
    time_view: str,
    destination_aggregation: str,
    destination_aggregation_lookup,
) -> pd.DataFrame:
    raw_df = _filter_by_date_range(
        _deserialize_dataframe(source_data),
        start_date,
        end_date,
    )
    return _apply_supply_time_view(
        _build_supply_matrix(
            raw_df,
            destination_aggregation,
            selected_destination_columns,
            other_countries_mode,
            destination_aggregation_lookup,
        ),
        time_view,
    )


@callback(
    Output("demand-download-woodmac-excel", "data"),
    Input("demand-export-woodmac-button", "n_clicks"),
    State("demand-woodmac-data-store", "data"),
    State("demand-country-dropdown", "value"),
    State("demand-other-country-mode", "value"),
    State("demand-date-range", "start_date"),
    State("demand-date-range", "end_date"),
    State("demand-time-view", "value"),
    State("demand-destination-aggregation-dropdown", "value"),
    State("demand-destination-aggregation-lookup-store", "data"),
    prevent_initial_call=True,
)
def export_woodmac_balance_excel(
    n_clicks,
    woodmac_data,
    selected_destination_columns,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    destination_aggregation,
    destination_aggregation_lookup,
):
    if not n_clicks:
        raise PreventUpdate

    export_df = _build_filtered_matrix_for_export(
        woodmac_data,
        selected_destination_columns,
        other_countries_mode,
        start_date,
        end_date,
        time_view,
        destination_aggregation,
        destination_aggregation_lookup,
    )
    if export_df.empty:
        raise PreventUpdate

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"WoodMac_Imports_Flow_{timestamp}.xlsx"
    return dcc.send_bytes(
        _export_matrix_to_excel_bytes(export_df, "Imports Flow"),
        filename,
    )


@callback(
    Output("demand-download-ea-excel", "data"),
    Input("demand-export-ea-button", "n_clicks"),
    State("demand-ea-data-store", "data"),
    State("demand-country-dropdown", "value"),
    State("demand-other-country-mode", "value"),
    State("demand-date-range", "start_date"),
    State("demand-date-range", "end_date"),
    State("demand-time-view", "value"),
    State("demand-destination-aggregation-dropdown", "value"),
    State("demand-destination-aggregation-lookup-store", "data"),
    prevent_initial_call=True,
)
def export_ea_balance_excel(
    n_clicks,
    ea_data,
    selected_destination_columns,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    destination_aggregation,
    destination_aggregation_lookup,
):
    if not n_clicks:
        raise PreventUpdate

    export_df = _build_filtered_matrix_for_export(
        ea_data,
        selected_destination_columns,
        other_countries_mode,
        start_date,
        end_date,
        time_view,
        destination_aggregation,
        destination_aggregation_lookup,
    )
    if export_df.empty:
        raise PreventUpdate

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"EA_Imports_Flow_{timestamp}.xlsx"
    return dcc.send_bytes(
        _export_matrix_to_excel_bytes(export_df, "Imports Flow"),
        filename,
    )
