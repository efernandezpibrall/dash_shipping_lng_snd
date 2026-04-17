from io import BytesIO, StringIO
import datetime as dt
import json

import pandas as pd
from dash import dcc, html, dash_table, callback, Input, Output, State
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate
from sqlalchemy import text

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
from utils.table_styles import StandardTableStyleManager, TABLE_COLORS


EXPORT_BUTTON_STYLE = {
    "marginLeft": "20px",
    "padding": "5px 15px",
    "backgroundColor": "#28a745",
    "color": "white",
    "border": "none",
    "borderRadius": "4px",
    "cursor": "pointer",
    "fontWeight": "bold",
    "fontSize": "12px",
}

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
    "padding": "6px 8px 6px 12px",
    "backgroundColor": "#ffffff",
    "border": "1px solid #dbe4ee",
    "borderRadius": "999px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.05)",
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
    if raw_df.empty:
        return pd.DataFrame(columns=["month", "country_name", "total_mmtpa"])

    cleaned_df = raw_df.copy()
    cleaned_df["month"] = (
        pd.to_datetime(cleaned_df["month"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    cleaned_df["country_name"] = (
        cleaned_df["country_name"]
        .fillna(UNKNOWN_DESTINATION_GROUP)
        .astype(str)
        .str.strip()
        .replace("", UNKNOWN_DESTINATION_GROUP)
    )
    cleaned_df["total_mmtpa"] = pd.to_numeric(
        cleaned_df["total_mmtpa"],
        errors="coerce",
    ).fillna(0.0)
    cleaned_df = cleaned_df.dropna(subset=["month"])

    return (
        cleaned_df.groupby(["month", "country_name"], as_index=False)["total_mmtpa"]
        .sum()
        .sort_values(["month", "country_name"])
        .reset_index(drop=True)
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


def _normalize_selected_destination_columns(selected_values) -> list[str]:
    if selected_values is None:
        return []
    if isinstance(selected_values, str):
        raw_values = [selected_values]
    else:
        raw_values = list(selected_values)

    normalized_values = []
    for value in raw_values:
        normalized_value = _normalize_mapping_value(value)
        if normalized_value is not None and normalized_value not in normalized_values:
            normalized_values.append(normalized_value)

    return normalized_values


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


def _resolve_country_columns_selection(
    selected_values,
    available_countries: list[str],
) -> list[str]:
    available_country_set = set(available_countries or [])
    normalized_selection = _normalize_selected_destination_columns(selected_values)
    if normalized_selection:
        resolved_selection = [
            country
            for country in normalized_selection
            if country in available_country_set
        ]
        if resolved_selection:
            return resolved_selection

    return default_selected_countries(available_countries or [])


def _resolve_destination_columns_selection(
    destination_aggregation: str,
    selected_values,
    available_countries: list[str],
    lookup_records,
) -> list[str]:
    if destination_aggregation == "country":
        return _resolve_country_columns_selection(
            selected_values,
            available_countries,
        )

    normalized_selection = _normalize_selected_destination_columns(selected_values)
    if normalized_selection:
        return normalized_selection

    return _get_available_destination_group_values(
        destination_aggregation,
        available_countries,
        lookup_records,
    )


def _serialize_dataframe(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None

    return df.to_json(date_format="iso", orient="split")


def _deserialize_dataframe(data: str | None) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()

    return pd.read_json(StringIO(data), orient="split")


def _normalize_month_date(value) -> pd.Timestamp | None:
    if not value:
        return None

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None

    return timestamp.to_period("M").to_timestamp()


def _get_date_bounds(dataframes: list[pd.DataFrame]) -> tuple[str | None, str | None]:
    non_empty_frames = [df for df in dataframes if df is not None and not df.empty]
    if not non_empty_frames:
        return None, None

    combined_df = pd.concat(non_empty_frames, ignore_index=True)
    min_month = pd.to_datetime(combined_df["month"]).min()
    max_month = pd.to_datetime(combined_df["month"]).max()

    return min_month.strftime("%Y-%m-%d"), max_month.strftime("%Y-%m-%d")


def _get_default_interval_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    current_year = pd.Timestamp.now().year
    default_start = pd.Timestamp(year=current_year, month=1, day=1)
    default_end = pd.Timestamp(year=current_year + 5, month=12, day=1)
    return default_start, default_end


def _filter_by_date_range(
    raw_df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    filtered_df = raw_df.copy()
    filtered_df["month"] = pd.to_datetime(filtered_df["month"]).dt.to_period("M").dt.to_timestamp()

    start_month = _normalize_month_date(start_date)
    end_month = _normalize_month_date(end_date)

    if start_month is not None:
        filtered_df = filtered_df[filtered_df["month"] >= start_month]
    if end_month is not None:
        filtered_df = filtered_df[filtered_df["month"] <= end_month]

    return filtered_df


def _build_lng_season_periods(
    dates: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    normalized_dates = pd.to_datetime(dates, errors="coerce").dt.to_period("M").dt.to_timestamp()
    is_summer = normalized_dates.dt.month.between(4, 9)
    season_year = (
        normalized_dates.dt.year - normalized_dates.dt.month.isin([1, 2, 3]).astype(int)
    ).astype("Int64")

    season_start_month = pd.Series(10, index=normalized_dates.index, dtype="int64")
    season_start_month.loc[is_summer] = 4

    season_code = pd.Series("W", index=normalized_dates.index, dtype="object")
    season_code.loc[is_summer] = "S"

    season_start = pd.to_datetime(
        {
            "year": season_year,
            "month": season_start_month,
            "day": 1,
        },
        errors="coerce",
    )
    season_label = season_year.astype(str)
    season_label = season_label.where(normalized_dates.notna(), "")
    season_label = season_label + "-" + season_code.where(normalized_dates.notna(), "")
    return season_start, season_label


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


def _align_matrix_to_reference_months(
    matrix_df: pd.DataFrame,
    reference_month_labels: list[str],
) -> pd.DataFrame:
    reference_index = (
        pd.Series(reference_month_labels, dtype="object")
        .pipe(pd.to_datetime, errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    reference_index = pd.Index(reference_index.dropna().unique())

    numeric_columns = [column for column in matrix_df.columns if column != "Month"]
    if reference_index.empty:
        return pd.DataFrame(columns=["Month"] + numeric_columns)

    if matrix_df.empty:
        aligned_df = pd.DataFrame(index=reference_index)
    else:
        aligned_df = matrix_df.copy()
        aligned_df["Month"] = pd.to_datetime(
            aligned_df["Month"].astype(str),
            errors="coerce",
        ).dt.to_period("M").dt.to_timestamp()
        aligned_df = (
            aligned_df.dropna(subset=["Month"])
            .drop_duplicates(subset=["Month"], keep="last")
            .set_index("Month")
            .sort_index()
            .reindex(reference_index)
        )

    for column_name in numeric_columns:
        source_series = (
            aligned_df[column_name]
            if column_name in aligned_df.columns
            else pd.Series(float("nan"), index=aligned_df.index, dtype="float64")
        )
        aligned_df[column_name] = pd.to_numeric(
            source_series,
            errors="coerce",
        )

    aligned_df.index.name = "Month"
    result_df = aligned_df.reset_index()
    result_df["Month"] = pd.to_datetime(result_df["Month"]).dt.strftime("%Y-%m")

    return result_df[["Month"] + numeric_columns]


def _create_empty_state(message: str) -> html.Div:
    return html.Div(message, className="balance-empty-state")


def _serialize_snapshot_value(payload: dict[str, str | None]) -> str:
    return json.dumps(payload, sort_keys=True)


def _deserialize_snapshot_value(value: str | dict | None) -> dict[str, str | None]:
    if not value:
        return {}

    if isinstance(value, dict):
        return value

    try:
        parsed_value = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}

    return parsed_value if isinstance(parsed_value, dict) else {}


def _default_previous_option_value(options: list[dict]) -> str | None:
    if len(options) > 1:
        return options[1]["value"]
    if options:
        return options[0]["value"]
    return None


def _build_woodmac_snapshot_dropdown_options(
    publication_options: list[dict[str, str | None]],
) -> list[dict[str, str]]:
    dropdown_options = []
    for option in publication_options:
        publication_label = option.get("market_outlook", "Unknown publication")
        publication_timestamp = _format_metadata_timestamp(
            option.get("publication_timestamp")
        )
        label = publication_label
        if publication_timestamp:
            label = f"{publication_label} | {publication_timestamp}"

        dropdown_options.append(
            {
                "label": label,
                "value": _serialize_snapshot_value(option),
            }
        )

    return dropdown_options


def _build_ea_upload_dropdown_options(
    upload_timestamps: list[str],
) -> list[dict[str, str]]:
    dropdown_options = []
    for upload_timestamp in upload_timestamps:
        formatted_timestamp = _format_metadata_timestamp(upload_timestamp) or upload_timestamp
        dropdown_options.append(
            {
                "label": formatted_timestamp,
                "value": upload_timestamp,
            }
        )

    return dropdown_options


def _build_delta_matrix(
    baseline_matrix: pd.DataFrame,
    comparison_matrix: pd.DataFrame,
) -> pd.DataFrame:
    if baseline_matrix.empty:
        return pd.DataFrame(columns=["Month", "Total MMTPA"])

    numeric_columns = [column for column in baseline_matrix.columns if column != "Month"]
    delta_df = baseline_matrix.copy()
    delta_df[numeric_columns] = delta_df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    comparison_aligned = comparison_matrix.copy()
    for column in numeric_columns:
        if column not in comparison_aligned.columns:
            comparison_aligned[column] = float("nan")

    comparison_aligned = comparison_aligned[["Month"] + numeric_columns]
    comparison_aligned[numeric_columns] = comparison_aligned[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    comparison_aligned = comparison_aligned.set_index("Month").reindex(delta_df["Month"])

    delta_index = delta_df.set_index("Month")
    delta_index[numeric_columns] = (
        delta_index[numeric_columns] - comparison_aligned[numeric_columns]
    ).round(2)

    return delta_index.reset_index()


def _format_table_cell_value(value) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"

    return str(value)


def _build_responsive_column_styles(df: pd.DataFrame) -> list[dict]:
    column_styles = []
    column_weights = {}
    column_min_widths = {}

    for column_name in df.columns:
        header_length = len(str(column_name))
        value_lengths = df[column_name].map(_format_table_cell_value).map(len)
        max_length = max([header_length] + value_lengths.tolist()) if not df.empty else header_length

        if column_name == "Month":
            column_weights[column_name] = max(8, min(max_length, 12))
            column_min_widths[column_name] = 92
        elif column_name == "Total MMTPA":
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

        if column_name == "Month":
            style_entry["textAlign"] = "left"

        column_styles.append(style_entry)

    return column_styles


def _format_metadata_timestamp(value) -> str | None:
    if not value:
        return None

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return str(value)

    return timestamp.strftime("%Y-%m-%d %H:%M")


def _build_woodmac_metadata_lines(metadata: dict | None) -> list[str]:
    if not metadata:
        return []

    lines = []
    short_term_line = metadata.get("short_term_market_outlook")
    short_term_timestamp = _format_metadata_timestamp(
        metadata.get("short_term_publication_timestamp")
    )
    if short_term_line:
        if short_term_timestamp:
            lines.append(
                f"ST publication: {short_term_line} | publication_date: {short_term_timestamp}"
            )
        else:
            lines.append(f"ST publication: {short_term_line}")

    long_term_line = metadata.get("long_term_market_outlook")
    long_term_timestamp = _format_metadata_timestamp(
        metadata.get("long_term_publication_timestamp")
    )
    if long_term_line:
        if long_term_timestamp:
            lines.append(
                f"LT publication: {long_term_line} | publication_date: {long_term_timestamp}"
            )
        else:
            lines.append(f"LT publication: {long_term_line}")

    return lines


def _build_ea_metadata_lines(metadata: dict | None) -> list[str]:
    if not metadata:
        return []

    upload_timestamp = _format_metadata_timestamp(metadata.get("upload_timestamp_utc"))
    if not upload_timestamp:
        return []

    return [f"upload_timestamp_utc: {upload_timestamp}"]
def _create_balance_table(
    table_id: str,
    df: pd.DataFrame,
    table_mode: str = "absolute",
) -> dash_table.DataTable | html.Div:
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

    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=df.where(pd.notna(df), None).to_dict("records"),
        sort_action="native",
        page_action="none",
        fill_width=True,
        fixed_rows={"headers": True},
        fixed_columns={"headers": True, "data": 1},
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": "560px",
            "width": "100%",
            "minWidth": "100%",
        },
        style_header=base_config["style_header"],
        style_cell={
            **base_config["style_cell"],
            "minWidth": "72px",
            "width": "72px",
            "maxWidth": "none",
            "border": f"1px solid {TABLE_COLORS['border_light']}",
            "padding": "6px 8px",
        },
        style_cell_conditional=_build_responsive_column_styles(df),
        style_data_conditional=style_data_conditional,
    )


def _build_section_summary(
    raw_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    destination_aggregation: str,
    other_countries_mode: str,
    metadata_lines: list[str] | None = None,
    time_view: str = "monthly",
) -> html.Div:
    summary_children = []

    if raw_df.empty:
        summary_children.append(
            html.Div("No source data returned.", className="balance-summary-row")
        )
    else:
        month_start = (
            matrix_df["Month"].iloc[0]
            if not matrix_df.empty
            else raw_df["month"].min().strftime("%Y-%m")
        )
        month_end = (
            matrix_df["Month"].iloc[-1]
            if not matrix_df.empty
            else raw_df["month"].max().strftime("%Y-%m")
        )
        visible_country_count = max(len(matrix_df.columns) - 2, 0)
        standardized_country_count = raw_df["country_name"].nunique()
        period_label = TIME_VIEW_PERIOD_LABELS.get(time_view, "month")
        count_label = period_label if len(matrix_df) == 1 else f"{period_label}s"
        aggregation_label = DESTINATION_AGGREGATION_LABELS.get(
            destination_aggregation,
            "Country",
        )
        visible_column_label = (
            "visible country columns"
            if destination_aggregation == "country"
            else f"visible {aggregation_label.lower()} columns"
        )
        visibility_note = (
            "Other countries grouped into Rest of the World."
            if destination_aggregation == "country"
            and other_countries_mode == "rest_of_world"
            else (
                "Only selected countries are included in the totals."
                if destination_aggregation == "country"
                else (
                    f"Destination aggregation: {aggregation_label}. "
                    f"All groups shown as explicit columns."
                )
            )
        )

        summary_children.append(
            html.Div(
                [
                    html.Span(f"{len(matrix_df):,} {count_label}"),
                    html.Span(f"{month_start} to {month_end}"),
                    html.Span(f"{visible_country_count} {visible_column_label}"),
                    html.Span(f"{standardized_country_count} standardized source countries"),
                    html.Span(visibility_note),
                ],
                className="balance-summary-row",
            )
        )

    if metadata_lines:
        summary_children.append(
            html.Div(
                [html.Span(line) for line in metadata_lines],
                className="balance-metadata-row",
            )
        )

    return html.Div(
        summary_children
    )


def _build_comparison_metadata_lines(
    comparison_source: str,
    short_term_value: str | None,
    long_term_value: str | None,
    ea_upload_value: str | None,
) -> list[str]:
    if comparison_source == "woodmac":
        short_term_snapshot = _deserialize_snapshot_value(short_term_value)
        long_term_snapshot = _deserialize_snapshot_value(long_term_value)
        lines = ["Delta formula: left baseline table - selected snapshot"]

        short_term_line = short_term_snapshot.get("market_outlook")
        short_term_timestamp = _format_metadata_timestamp(
            short_term_snapshot.get("publication_timestamp")
        )
        if short_term_line:
            if short_term_timestamp:
                lines.append(
                    f"Comparison source: WoodMac | ST publication: {short_term_line} | publication_date: {short_term_timestamp}"
                )
            else:
                lines.append(
                    f"Comparison source: WoodMac | ST publication: {short_term_line}"
                )

        long_term_line = long_term_snapshot.get("market_outlook")
        long_term_timestamp = _format_metadata_timestamp(
            long_term_snapshot.get("publication_timestamp")
        )
        if long_term_line:
            if long_term_timestamp:
                lines.append(
                    f"LT publication: {long_term_line} | publication_date: {long_term_timestamp}"
                )
            else:
                lines.append(f"LT publication: {long_term_line}")

        return lines
    ea_upload_label = _format_metadata_timestamp(ea_upload_value) or ea_upload_value
    if ea_upload_label:
        return [
            "Delta formula: left baseline table - selected snapshot",
            f"Comparison source: Energy Aspects | upload_timestamp_utc: {ea_upload_label}",
        ]

    return ["Delta formula: left baseline table - selected snapshot"]


def _build_comparison_summary(
    delta_df: pd.DataFrame,
    metadata_lines: list[str],
) -> html.Div:
    return html.Div()


def _resolve_snapshot_control_values(
    comparison_source,
    comparison_options,
    current_st_value,
    current_lt_value,
    current_ea_upload_value,
):
    comparison_options = comparison_options or {}
    woodmac_options = comparison_options.get("woodmac", {})
    short_term_options = _build_woodmac_snapshot_dropdown_options(
        woodmac_options.get("short_term", [])
    )
    long_term_options = _build_woodmac_snapshot_dropdown_options(
        woodmac_options.get("long_term", [])
    )
    ea_upload_options = _build_ea_upload_dropdown_options(
        comparison_options.get("ea_uploads", [])
    )

    short_term_values = {option["value"] for option in short_term_options}
    long_term_values = {option["value"] for option in long_term_options}
    ea_upload_values = {option["value"] for option in ea_upload_options}

    short_term_value = (
        current_st_value
        if current_st_value in short_term_values
        else _default_previous_option_value(short_term_options)
    )
    long_term_value = (
        current_lt_value
        if current_lt_value in long_term_values
        else _default_previous_option_value(long_term_options)
    )
    ea_upload_value = (
        current_ea_upload_value
        if current_ea_upload_value in ea_upload_values
        else _default_previous_option_value(ea_upload_options)
    )

    if comparison_source == "ea":
        return (
            short_term_options,
            short_term_value,
            long_term_options,
            long_term_value,
            ea_upload_options,
            ea_upload_value,
            {"display": "none"},
            {"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "flex-end"},
        )

    return (
        short_term_options,
        short_term_value,
        long_term_options,
        long_term_value,
        ea_upload_options,
        ea_upload_value,
        {"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "flex-end"},
        {"display": "none"},
    )


def _fetch_comparison_raw_df(
    comparison_source: str,
    short_term_value: str | None,
    long_term_value: str | None,
    ea_upload_value: str | None,
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        if comparison_source == "ea":
            if not ea_upload_value:
                return None, "No Energy Aspects upload_timestamp_utc available."
            return (
                fetch_ea_import_flow_raw_data_for_upload(ea_upload_value),
                None,
            )

        short_term_snapshot = _deserialize_snapshot_value(short_term_value)
        long_term_snapshot = _deserialize_snapshot_value(long_term_value)
        if not short_term_snapshot or not long_term_snapshot:
            return None, "No WoodMac comparison publications available."

        return (
            fetch_woodmac_import_flow_raw_data_for_publications(
                short_term_snapshot.get("market_outlook"),
                short_term_snapshot.get("publication_timestamp"),
                long_term_snapshot.get("market_outlook"),
                long_term_snapshot.get("publication_timestamp"),
            ),
            None,
        )
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
    metadata_lines: list[str],
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
            _build_comparison_summary(pd.DataFrame(columns=["Month", "Total MMTPA"]), metadata_lines),
            _create_empty_state(empty_baseline_message),
        )

    if (
        destination_aggregation == "country"
        and selected_destination_columns == []
        and other_countries_mode == "exclude"
    ):
        return (
            _build_comparison_summary(
                pd.DataFrame(columns=baseline_matrix.columns),
                metadata_lines,
            ),
            _create_empty_state(
                "Select at least one country or switch to Rest of the World mode."
            ),
        )

    if comparison_error_message:
        return (
            _build_comparison_summary(
                pd.DataFrame(columns=baseline_matrix.columns),
                metadata_lines + [comparison_error_message],
            ),
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

    comparison_summary = _build_comparison_summary(delta_matrix, metadata_lines)
    comparison_table = _create_balance_table(
        comparison_table_id,
        delta_matrix,
        table_mode="delta",
    )

    return comparison_summary, comparison_table


def _create_source_section(
    title: str,
    subtitle: str,
    summary_id: str,
    table_container_id: str,
    export_button_id: str,
) -> html.Div:
    header_children = [
        html.Div(
            [
                html.H3(title, className="balance-section-title"),
                html.Button(
                    "Export to Excel",
                    id=export_button_id,
                    n_clicks=0,
                    style=EXPORT_BUTTON_STYLE,
                ),
            ],
            className="inline-section-header",
            style={"display": "flex", "alignItems": "center"},
        )
    ]

    if subtitle:
        header_children.append(
            html.P(subtitle, className="balance-section-subtitle")
        )

    header_children.append(html.Div(id=summary_id))

    return html.Div(
        [
            html.Div(
                header_children,
                className="balance-section-header",
            ),
            html.Div(id=table_container_id, className="balance-table-container"),
        ],
        className="balance-section-card",
    )


def _create_comparison_section(
    title: str,
    export_button_id: str,
    baseline_summary_id: str,
    default_comparison_source: str,
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
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(title, className="balance-section-title"),
                            html.Button(
                                "Export to Excel",
                                id=export_button_id,
                                n_clicks=0,
                                style=EXPORT_BUTTON_STYLE,
                            ),
                        ],
                        className="inline-section-header",
                        style={"display": "flex", "alignItems": "center"},
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
                        [
                            html.Div(
                                id=baseline_summary_id,
                                className="balance-pane-summary",
                            )
                        ],
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
                                            {
                                                "label": "WoodMac",
                                                "value": "woodmac",
                                            },
                                            {
                                                "label": "Energy Aspects",
                                                "value": "ea",
                                            },
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
                                            html.Label(
                                                "ST publication:",
                                                className="filter-label",
                                            ),
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
                                            html.Label(
                                                "LT publication:",
                                                className="filter-label",
                                            ),
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
                                            html.Label(
                                                "upload_timestamp_utc:",
                                                className="filter-label",
                                            ),
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
        dcc.Store(id="demand-refresh-timestamp-store", storage_type="memory"),
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
                        html.Div(
                            [
                                html.Div("Date Range", className="filter-group-header"),
                                html.Label("Month interval:", className="filter-label"),
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
                            className="filter-section filter-section-destination",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Time View",
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
                                            style={"display": "flex", "alignItems": "center"},
                                        )
                                    ],
                                    style=TIME_VIEW_CONTROL_SHELL_STYLE,
                                ),
                            ],
                            className="filter-section filter-section-destination",
                        ),
                        html.Div(
                            [
                                html.Div("Destination Aggregation", className="filter-group-header"),
                                html.Label("Group by:", className="filter-label"),
                                dcc.Dropdown(
                                    id="demand-destination-aggregation-dropdown",
                                    options=DESTINATION_AGGREGATION_OPTIONS,
                                    value="country",
                                    clearable=False,
                                    className="filter-dropdown",
                                    style={"minWidth": "240px"},
                                ),
                            ],
                            className="filter-section filter-section-destination",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Country Columns",
                                    id="demand-country-columns-header",
                                    className="filter-group-header",
                                ),
                                html.Label(
                                    "Countries:",
                                    id="demand-country-columns-label",
                                    className="filter-label",
                                ),
                                dcc.Dropdown(
                                    id="demand-country-dropdown",
                                    options=[],
                                    value=None,
                                    multi=True,
                                    placeholder="Select countries to keep as separate columns",
                                    className="filter-dropdown",
                                    style={"minWidth": "380px"},
                                ),
                            ],
                            className="filter-section filter-section-origin-exp",
                        ),
                        html.Div(
                            [
                                html.Div("Other Countries", className="filter-group-header"),
                                html.Label("Handling:", className="filter-label"),
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
                                    labelStyle={"display": "inline-flex", "alignItems": "center"},
                                    inputStyle={"marginRight": "6px"},
                                ),
                            ],
                            className="filter-section filter-section-volume",
                        ),
                        html.Div(
                            [
                                html.Div("Status", className="filter-group-header"),
                                html.Div(
                                    id="demand-refresh-indicator",
                                    className="text-tertiary",
                                    style={"fontSize": "12px", "whiteSpace": "nowrap"},
                                ),
                                html.Div(
                                    id="demand-meta-indicator",
                                    className="text-tertiary",
                                    style={"fontSize": "12px", "maxWidth": "260px"},
                                ),
                            ],
                            className="filter-section filter-section-analysis",
                        ),
                    ],
                    className="filter-bar-grouped",
                )
            ],
            className="professional-section-header",
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
    Output("demand-refresh-timestamp-store", "data"),
    Output("demand-load-error-store", "data"),
    Output("demand-woodmac-metadata-store", "data"),
    Output("demand-ea-metadata-store", "data"),
    Output("demand-comparison-options-store", "data"),
    Input("global-refresh-button", "n_clicks"),
)
def load_balance_source_data(_):
    woodmac_df = pd.DataFrame()
    ea_df = pd.DataFrame()
    woodmac_metadata = {}
    ea_metadata = {}
    destination_aggregation_lookup = []
    comparison_options = {
        "woodmac": {"short_term": [], "long_term": []},
        "ea_uploads": [],
    }
    errors = []

    try:
        woodmac_df = fetch_woodmac_import_flow_raw_data()
    except Exception as exc:
        errors.append(f"WoodMac load failed: {exc}")

    try:
        woodmac_metadata = fetch_woodmac_import_flow_metadata()
    except Exception as exc:
        errors.append(f"WoodMac metadata load failed: {exc}")

    try:
        ea_df = fetch_ea_import_flow_raw_data()
    except Exception as exc:
        errors.append(f"Energy Aspects load failed: {exc}")

    try:
        ea_metadata = fetch_ea_import_flow_metadata()
    except Exception as exc:
        errors.append(f"Energy Aspects metadata load failed: {exc}")

    try:
        comparison_options["woodmac"] = fetch_woodmac_publication_options()
    except Exception as exc:
        errors.append(f"WoodMac comparison options load failed: {exc}")

    try:
        comparison_options["ea_uploads"] = fetch_ea_upload_options()
    except Exception as exc:
        errors.append(f"Energy Aspects comparison options load failed: {exc}")

    try:
        destination_aggregation_lookup = _fetch_destination_aggregation_lookup_records()
    except Exception as exc:
        errors.append(f"Destination aggregation lookup load failed: {exc}")

    available_countries = get_available_countries([woodmac_df, ea_df])
    refresh_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = " | ".join(errors) if errors else None

    return (
        _serialize_dataframe(woodmac_df),
        _serialize_dataframe(ea_df),
        available_countries,
        destination_aggregation_lookup,
        refresh_timestamp,
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
    Output("demand-country-columns-label", "children"),
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
            "Countries:",
            "Select countries to keep as separate columns",
            False,
            selected_values,
        )

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
        f"{aggregation_label} groups:",
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
    woodmac_raw_df = _deserialize_dataframe(woodmac_data)
    ea_raw_df = _deserialize_dataframe(ea_data)

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
    Output("demand-refresh-indicator", "children"),
    Output("demand-meta-indicator", "children"),
    Input("demand-refresh-timestamp-store", "data"),
    Input("demand-country-options-store", "data"),
    Input("demand-date-range", "start_date"),
    Input("demand-date-range", "end_date"),
    Input("demand-time-view", "value"),
    Input("demand-destination-aggregation-dropdown", "value"),
)
def update_balance_status(
    refresh_timestamp,
    available_countries,
    start_date,
    end_date,
    time_view,
    destination_aggregation,
):
    refresh_text = (
        f"Last refreshed: {refresh_timestamp}"
        if refresh_timestamp
        else "Last refreshed: waiting for data"
    )
    time_view_label = TIME_VIEW_LABELS.get(time_view, "Monthly")
    aggregation_label = DESTINATION_AGGREGATION_LABELS.get(
        destination_aggregation,
        "Country",
    )
    if start_date and end_date:
        start_label = _normalize_month_date(start_date).strftime("%Y-%m")
        end_label = _normalize_month_date(end_date).strftime("%Y-%m")
        range_text = (
            f"Showing {start_label} to {end_label} in {time_view_label} view. "
            f"Destination aggregation: {aggregation_label}."
        )
    else:
        range_text = (
            f"EA dataset attributes are resolved from Energy Aspects metadata, "
            f"and country names are standardized with at_lng.mappings_country. "
            f"Time view: {time_view_label}. "
            f"Destination aggregation: {aggregation_label}."
        )

    meta_text = (
        f"{len(available_countries or []):,} countries available after standardization. {range_text}"
        if available_countries
        else range_text
    )
    return refresh_text, meta_text


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
    )
    ea_summary = _build_section_summary(
        ea_raw_df,
        ea_matrix,
        destination_aggregation,
        other_countries_mode,
        _build_ea_metadata_lines(ea_metadata),
        time_view=time_view,
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
    )

    metadata_lines = _build_comparison_metadata_lines(
        comparison_source,
        short_term_value,
        long_term_value,
        ea_upload_value,
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
        metadata_lines,
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
    )

    metadata_lines = _build_comparison_metadata_lines(
        comparison_source,
        short_term_value,
        long_term_value,
        ea_upload_value,
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
        metadata_lines,
        "No baseline Energy Aspects import data available for the current selection.",
        "demand-ea-comparison-delta-table",
        comparison_error_message=comparison_error_message,
    )


def _export_matrix_to_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        worksheet = writer.sheets[sheet_name]
        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter
            for cell in column_cells:
                cell_value = "" if cell.value is None else str(cell.value)
                if len(cell_value) > max_length:
                    max_length = len(cell_value)
            worksheet.column_dimensions[column_letter].width = min(max_length + 2, 24)

    output.seek(0)
    return output.getvalue()


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
