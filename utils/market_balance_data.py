from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd
from sqlalchemy import text

from utils.balance_time import (
    align_frames_on_period,
    annualized_mmtpa_to_monthly_bcm,
    annualized_mmtpa_to_monthly_mt,
    calculate_differences,
    get_days_in_period,
    get_time_period,
    group_numeric_frame_by_period,
    normalize_time_group,
    sort_period_labels,
    validate_complete_seasons,
)
from utils.export_flow_data import (
    DB_SCHEMA,
    engine,
    fetch_ea_export_flow_metadata,
    fetch_ea_export_flow_raw_data,
    fetch_ea_export_flow_raw_data_for_upload,
    fetch_ea_upload_options as fetch_ea_export_upload_options,
    fetch_woodmac_export_flow_metadata,
    fetch_woodmac_export_flow_raw_data,
    fetch_woodmac_export_flow_raw_data_for_publications,
    fetch_woodmac_publication_options as fetch_woodmac_export_publication_options,
)
from utils.import_flow_data import (
    fetch_ea_import_flow_metadata,
    fetch_ea_import_flow_raw_data,
    fetch_ea_import_flow_raw_data_for_upload,
    fetch_woodmac_import_flow_metadata,
    fetch_woodmac_import_flow_raw_data,
    fetch_woodmac_import_flow_raw_data_for_publications,
)


COUNTRY_GROUP_COLUMN_MAP = {
    "country": "country_name",
    "continent": "continent",
    "subcontinent": "subcontinent",
    "basin": "basin",
    "shipping_region": "shipping_region",
    "country_classification_level1": "country_classification_level1",
    "country_classification": "country_classification",
}

COUNTRY_GROUP_LABELS = {
    "country": "Country",
    "continent": "Continent",
    "subcontinent": "Subcontinent",
    "basin": "Basin",
    "shipping_region": "Shipping Region",
    "country_classification_level1": "Classification Level 1",
    "country_classification": "Classification",
}

PACIFIC_LOCKED_COUNTRIES = [
    "Australia",
    "Brunei",
    "Indonesia",
    "Malaysia",
    "Papua New Guinea",
]

OVERVIEW_NET_DEFAULTS = {
    "country_group": "country_classification_level1",
    "time_group": "yearly",
    "unit": "bcm",
}

COUNTRY_CATEGORY_CONFIG = [
    {
        "title": "Pipelines",
        "aliases": {"Pipeline", "Pipelines"},
        "chart_type": "area",
    },
    {
        "title": "Production",
        "aliases": {"Production"},
        "chart_type": "line",
    },
    {
        "title": "LNG Imports",
        "aliases": {"LNG"},
        "chart_type": "line",
    },
    {
        "title": "Stocks",
        "aliases": {"Stock", "Stocks"},
        "chart_type": "line",
    },
]

_relation_support_cache: dict[tuple[int, str, str], bool] = {}


def _payload(
    *,
    data: dict | None = None,
    metadata: dict | None = None,
    error: str | None = None,
) -> dict:
    return {
        "data": data or {},
        "metadata": metadata or {},
        "error": error,
    }


def _normalize_text_value(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized if normalized else None


def _relation_exists(relation_name: str, *, schema: str = DB_SCHEMA) -> bool:
    cache_key = (id(engine), schema, relation_name)
    if cache_key in _relation_support_cache:
        return _relation_support_cache[cache_key]

    query = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema_name
              AND table_name = :relation_name
        )
        """
    )
    with engine.connect() as connection:
        exists = bool(
            connection.execute(
                query,
                {"schema_name": schema, "relation_name": relation_name},
            ).scalar()
        )

    _relation_support_cache[cache_key] = exists
    return exists


def _serialize_scalar(value):
    if value is None or pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        if (
            value.hour == 0
            and value.minute == 0
            and value.second == 0
            and value.microsecond == 0
        ):
            return value.strftime("%Y-%m-%d")
        if value.second or value.microsecond:
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return value.strftime("%Y-%m-%d %H:%M")

    if hasattr(value, "strftime"):
        try:
            timestamp = pd.Timestamp(value)
            return _serialize_scalar(timestamp)
        except Exception:
            return str(value)

    return value


def serialize_frame(df: pd.DataFrame | None) -> dict:
    if df is None:
        return {"records": [], "columns": [], "numeric_columns": []}

    working_df = df.copy()
    numeric_columns = [
        column_name
        for column_name in working_df.columns
        if pd.api.types.is_numeric_dtype(working_df[column_name])
    ]

    for column_name in working_df.columns:
        if pd.api.types.is_datetime64_any_dtype(working_df[column_name]):
            working_df[column_name] = working_df[column_name].map(_serialize_scalar)
        elif working_df[column_name].dtype == "object":
            working_df[column_name] = working_df[column_name].map(_serialize_scalar)

    working_df = working_df.where(pd.notna(working_df), None)
    return {
        "records": working_df.to_dict("records"),
        "columns": working_df.columns.tolist(),
        "numeric_columns": numeric_columns,
    }


def _sanitize_flow_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["month", "country_name", "total_mmtpa"])

    working_df = raw_df.copy()
    working_df["month"] = pd.to_datetime(working_df["month"], errors="coerce")
    working_df["country_name"] = (
        working_df["country_name"].map(_normalize_text_value).fillna("Unknown")
    )
    working_df["total_mmtpa"] = pd.to_numeric(
        working_df["total_mmtpa"], errors="coerce"
    ).fillna(0.0)
    working_df = working_df.dropna(subset=["month"])

    return (
        working_df.groupby(["month", "country_name"], as_index=False)["total_mmtpa"]
        .sum()
        .sort_values(["month", "country_name"])
    )


def _normalize_range_start(value) -> pd.Timestamp | None:
    if not value:
        return None

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.to_period("M").to_timestamp()


def _normalize_range_end(value) -> pd.Timestamp | None:
    if not value:
        return None

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.to_period("M").to_timestamp() + pd.offsets.MonthEnd(0)


def _filter_frame_by_date_range(
    df: pd.DataFrame,
    *,
    date_col: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    working_df = df.copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")
    working_df = working_df.dropna(subset=[date_col])

    start_bound = _normalize_range_start(start_date)
    end_bound = _normalize_range_end(end_date)

    if start_bound is not None:
        working_df = working_df[working_df[date_col] >= start_bound]
    if end_bound is not None:
        working_df = working_df[working_df[date_col] <= end_bound]

    return working_df.copy()


def fetch_country_mapping_df() -> pd.DataFrame:
    query = text(
        f"""
        SELECT
            country_name,
            country,
            continent,
            subcontinent,
            basin,
            shipping_region,
            country_classification_level1,
            country_classification
        FROM {DB_SCHEMA}.mappings_country
        """
    )
    with engine.connect() as connection:
        mapping_df = pd.read_sql_query(query, connection)

    if mapping_df.empty:
        return pd.DataFrame(columns=["country_name", "country", *COUNTRY_GROUP_COLUMN_MAP.values()])

    for column_name in mapping_df.columns:
        mapping_df[column_name] = mapping_df[column_name].map(_normalize_text_value)

    return mapping_df


def _build_country_group_lookup_maps(mapping_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    lookup_maps = {group_name: {} for group_name in COUNTRY_GROUP_COLUMN_MAP}
    if mapping_df.empty:
        return lookup_maps

    for _, row in mapping_df.iterrows():
        normalized_keys = {
            key
            for key in (
                _normalize_text_value(row.get("country_name")),
                _normalize_text_value(row.get("country")),
            )
            if key
        }
        if not normalized_keys:
            continue

        for group_name, column_name in COUNTRY_GROUP_COLUMN_MAP.items():
            group_value = _normalize_text_value(row.get(column_name)) or "Unknown"
            for normalized_key in normalized_keys:
                lookup_maps[group_name][normalized_key] = group_value

    return lookup_maps


def _resolve_group_values(
    country_series: pd.Series,
    country_group: str,
    lookup_maps: dict[str, dict[str, str]],
) -> pd.Series:
    normalized_group = country_group if country_group in COUNTRY_GROUP_COLUMN_MAP else "country"
    if normalized_group == "country":
        return country_series.map(_normalize_text_value).fillna("Unknown")

    group_lookup = lookup_maps.get(normalized_group, {})
    return country_series.map(
        lambda value: group_lookup.get(_normalize_text_value(value) or "", "Unknown")
    )


def _prepare_trade_flow_table(
    raw_df: pd.DataFrame,
    *,
    country_group: str,
    time_group: str,
    unit: str,
    selected_years: Iterable[int] | None,
    mapping_df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    normalized_time_group = normalize_time_group(time_group)
    working_df = _sanitize_flow_raw_df(raw_df)
    working_df = _filter_frame_by_date_range(
        working_df,
        date_col="month",
        start_date=start_date,
        end_date=end_date,
    )
    if working_df.empty:
        return pd.DataFrame(columns=["Period", "Total"])

    selected_years = {
        int(year)
        for year in (selected_years or [])
        if str(year).strip()
    }
    if selected_years:
        working_df = working_df[working_df["month"].dt.year.isin(selected_years)].copy()
    if working_df.empty:
        return pd.DataFrame(columns=["Period", "Total"])

    lookup_maps = _build_country_group_lookup_maps(mapping_df)
    working_df["group_value"] = _resolve_group_values(
        working_df["country_name"], country_group, lookup_maps
    )
    working_df["Period"] = working_df["month"].apply(
        lambda value: get_time_period(value, normalized_time_group)
    )

    if normalized_time_group == "season":
        working_df = validate_complete_seasons(
            working_df, date_col="month", period_col="Period"
        )
        if working_df.empty:
            return pd.DataFrame(columns=["Period", "Total"])

    working_df["monthly_mt"] = annualized_mmtpa_to_monthly_mt(working_df["total_mmtpa"])
    working_df["monthly_bcm"] = annualized_mmtpa_to_monthly_bcm(working_df["total_mmtpa"])

    normalized_unit = str(unit or "bcm").strip().lower()
    value_column = "monthly_bcm"
    if normalized_unit in {"mt", "mmt"}:
        value_column = "monthly_mt"

    period_df = (
        working_df.groupby(["Period", "group_value"], as_index=False)[value_column]
        .sum()
        .rename(columns={value_column: "value"})
    )

    if normalized_unit == "mcm_d":
        period_df["value"] = period_df["value"] * 1000.0
        period_df["value"] = period_df["value"] / period_df["Period"].apply(
            lambda period: get_days_in_period(period, normalized_time_group)
        )

    pivot_df = (
        period_df.pivot(index="Period", columns="group_value", values="value")
        .fillna(0.0)
        .reset_index()
    )
    if pivot_df.empty:
        return pd.DataFrame(columns=["Period", "Total"])

    group_columns = [column_name for column_name in pivot_df.columns if column_name != "Period"]
    group_columns = [
        column_name for column_name in group_columns if pivot_df[column_name].abs().sum() > 0
    ]
    if not group_columns:
        return pd.DataFrame(columns=["Period", "Total"])

    group_columns = sorted(
        group_columns,
        key=lambda column_name: (-pivot_df[column_name].sum(), column_name),
    )
    pivot_df["Total"] = pivot_df[group_columns].sum(axis=1)

    ordered_periods = sort_period_labels(
        pd.unique(pivot_df["Period"]).tolist(),
        normalized_time_group,
    )
    pivot_df["Period"] = pd.Categorical(
        pivot_df["Period"], categories=ordered_periods, ordered=True
    )
    pivot_df = pivot_df.sort_values("Period").reset_index(drop=True)
    pivot_df["Period"] = pivot_df["Period"].astype(str)

    ordered_columns = ["Period", "Total", *group_columns]
    pivot_df = pivot_df[ordered_columns]
    numeric_columns = [column_name for column_name in ordered_columns if column_name != "Period"]
    pivot_df[numeric_columns] = pivot_df[numeric_columns].round(2)
    return pivot_df


def _build_net_balance_table(
    exports_df: pd.DataFrame,
    imports_df: pd.DataFrame,
    *,
    time_group: str,
) -> pd.DataFrame:
    if exports_df.empty and imports_df.empty:
        return pd.DataFrame(columns=["Period", "Total"])

    normalized_time_group = normalize_time_group(time_group)
    all_periods = sort_period_labels(
        sorted(
            set(exports_df.get("Period", pd.Series(dtype=str)).astype(str).tolist())
            | set(imports_df.get("Period", pd.Series(dtype=str)).astype(str).tolist())
        ),
        normalized_time_group,
    )
    if not all_periods:
        return pd.DataFrame(columns=["Period", "Total"])

    value_columns = sorted(
        {
            column_name
            for column_name in [*exports_df.columns.tolist(), *imports_df.columns.tolist()]
            if column_name != "Period"
        }
    )
    result_df = pd.DataFrame({"Period": all_periods})
    exports_index = exports_df.set_index("Period") if not exports_df.empty else pd.DataFrame()
    imports_index = imports_df.set_index("Period") if not imports_df.empty else pd.DataFrame()

    def _aligned_numeric_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
        if column_name not in frame.columns:
            return pd.Series(0.0, index=all_periods, dtype=float)

        series = frame[column_name]
        if isinstance(series, pd.DataFrame):
            series = series.sum(axis=1)

        series = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return series.reindex(all_periods, fill_value=0.0).astype(float)

    for column_name in value_columns:
        export_series = _aligned_numeric_series(exports_index, column_name)
        import_series = _aligned_numeric_series(imports_index, column_name)
        result_df[column_name] = (export_series - import_series).round(2).values

    ordered_columns = ["Period"]
    if "Total" in result_df.columns:
        ordered_columns.append("Total")
    ordered_columns.extend(
        column_name for column_name in value_columns if column_name != "Total"
    )
    return result_df[ordered_columns]


def _build_provider_net_balance_table(
    export_raw_df: pd.DataFrame,
    import_raw_df: pd.DataFrame,
    *,
    mapping_df: pd.DataFrame,
    country_group: str,
    time_group: str,
    unit: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    exports_df = _prepare_trade_flow_table(
        export_raw_df,
        country_group=country_group,
        time_group=time_group,
        unit=unit,
        selected_years=None,
        mapping_df=mapping_df,
        start_date=start_date,
        end_date=end_date,
    )
    imports_df = _prepare_trade_flow_table(
        import_raw_df,
        country_group=country_group,
        time_group=time_group,
        unit=unit,
        selected_years=None,
        mapping_df=mapping_df,
        start_date=start_date,
        end_date=end_date,
    )
    return _build_net_balance_table(exports_df, imports_df, time_group=time_group)


def fetch_net_balance_comparison_options() -> dict[str, object]:
    return {
        "woodmac": fetch_woodmac_export_publication_options(),
        "ea_uploads": fetch_ea_export_upload_options(),
    }


def fetch_net_balance_for_woodmac_publications(
    *,
    short_term_market_outlook: str,
    short_term_publication_timestamp: str,
    long_term_market_outlook: str,
    long_term_publication_timestamp: str,
    country_group: str,
    time_group: str,
    unit: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    mapping_df = fetch_country_mapping_df()
    export_raw_df = fetch_woodmac_export_flow_raw_data_for_publications(
        short_term_market_outlook,
        short_term_publication_timestamp,
        long_term_market_outlook,
        long_term_publication_timestamp,
    )
    import_raw_df = fetch_woodmac_import_flow_raw_data_for_publications(
        short_term_market_outlook,
        short_term_publication_timestamp,
        long_term_market_outlook,
        long_term_publication_timestamp,
    )
    return _build_provider_net_balance_table(
        export_raw_df,
        import_raw_df,
        mapping_df=mapping_df,
        country_group=country_group,
        time_group=time_group,
        unit=unit,
        start_date=start_date,
        end_date=end_date,
    )


def fetch_net_balance_for_ea_upload(
    *,
    upload_timestamp_utc: str,
    country_group: str,
    time_group: str,
    unit: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    mapping_df = fetch_country_mapping_df()
    export_raw_df = fetch_ea_export_flow_raw_data_for_upload(upload_timestamp_utc)
    import_raw_df = fetch_ea_import_flow_raw_data_for_upload(upload_timestamp_utc)
    return _build_provider_net_balance_table(
        export_raw_df,
        import_raw_df,
        mapping_df=mapping_df,
        country_group=country_group,
        time_group=time_group,
        unit=unit,
        start_date=start_date,
        end_date=end_date,
    )


def build_period_delta_table(
    current_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    *,
    period_col: str = "Period",
) -> pd.DataFrame:
    if current_df.empty:
        return current_df.copy()

    numeric_columns = current_df.select_dtypes(include=["number"]).columns.tolist()
    result_df = current_df.copy()
    if comparison_df.empty:
        result_df[numeric_columns] = 0.0
        return result_df

    aligned_previous_df = align_frames_on_period(
        current_df,
        comparison_df,
        period_col=period_col,
    )
    for column_name in numeric_columns:
        result_df[column_name] = (
            pd.to_numeric(result_df[column_name], errors="coerce").fillna(0.0)
            - pd.to_numeric(aligned_previous_df[column_name], errors="coerce").fillna(0.0)
        ).round(2)

    return result_df


def calculate_flex_volumes(actual_df: pd.DataFrame, contract_df: pd.DataFrame) -> pd.DataFrame:
    if actual_df.empty or contract_df.empty:
        return pd.DataFrame(columns=actual_df.columns if not actual_df.empty else ["Period", "Total"])

    actual = actual_df.copy()
    contracts = contract_df.copy()
    actual["Period"] = actual["Period"].astype(str)
    contracts["Period"] = contracts["Period"].astype(str)

    all_periods = sort_period_labels(
        sorted(set(actual["Period"].tolist()) | set(contracts["Period"].tolist())),
        "yearly",
    )
    group_columns = sorted(
        {
            column_name
            for column_name in [*actual.columns.tolist(), *contracts.columns.tolist()]
            if column_name != "Period"
        }
    )

    result_df = pd.DataFrame({"Period": all_periods})
    actual_index = actual.set_index("Period")
    contract_index = contracts.set_index("Period")

    def _aligned_numeric_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
        if column_name not in frame.columns:
            return pd.Series(0.0, index=all_periods, dtype=float)

        series = frame[column_name]
        if isinstance(series, pd.DataFrame):
            series = series.sum(axis=1)

        series = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return series.reindex(all_periods, fill_value=0.0).astype(float)

    for column_name in group_columns:
        actual_series = _aligned_numeric_series(actual_index, column_name)
        contract_series = _aligned_numeric_series(contract_index, column_name)
        result_df[column_name] = (actual_series - contract_series).round(2).values

    ordered_columns = ["Period"]
    if "Total" in result_df.columns:
        ordered_columns.append("Total")
    ordered_columns.extend(
        column_name
        for column_name in group_columns
        if column_name != "Total"
    )
    return result_df[ordered_columns]


def fetch_contract_volume_tables(
    *,
    country_group: str,
    selected_years: Iterable[int] | None,
    unit: str,
    mapping_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_df = mapping_df.copy() if mapping_df is not None else fetch_country_mapping_df()
    query = text(
        f"""
        SELECT
            CAST(demand_table.year AS INTEGER) AS contract_year,
            COALESCE(contract_table.country_name_source, 'Unknown') AS exporter_country_name,
            COALESCE(contract_table.country_name_delivery, 'Unknown') AS importer_country_name,
            COALESCE(demand_table.acq_volume__mmtpa, 0) AS contracted_mmtpa
        FROM {DB_SCHEMA}.woodmac_lng_contract_annual_contracted_demand_mta demand_table
        LEFT JOIN {DB_SCHEMA}.woodmac_lng_contract contract_table
            ON demand_table.id_contract = contract_table.id_contract
        WHERE demand_table.id_contract IS NOT NULL
          AND demand_table.year IS NOT NULL
        """
    )
    with engine.connect() as connection:
        contracts_df = pd.read_sql_query(query, connection)

    if contracts_df.empty:
        empty_df = pd.DataFrame(columns=["Period", "Total"])
        return empty_df, empty_df

    selected_years = {
        int(year)
        for year in (selected_years or [])
        if str(year).strip()
    }
    if selected_years:
        contracts_df = contracts_df[contracts_df["contract_year"].isin(selected_years)].copy()
    if contracts_df.empty:
        empty_df = pd.DataFrame(columns=["Period", "Total"])
        return empty_df, empty_df

    lookup_maps = _build_country_group_lookup_maps(mapping_df)
    contracts_df["export_group"] = _resolve_group_values(
        contracts_df["exporter_country_name"], country_group, lookup_maps
    )
    contracts_df["import_group"] = _resolve_group_values(
        contracts_df["importer_country_name"], country_group, lookup_maps
    )

    def _pivot_contracts(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        grouped_df = (
            df.groupby(["contract_year", group_col], as_index=False)["contracted_mmtpa"]
            .sum()
            .rename(columns={"contract_year": "Period"})
        )
        grouped_df["Period"] = grouped_df["Period"].astype(str)
        pivot_df = (
            grouped_df.pivot(index="Period", columns=group_col, values="contracted_mmtpa")
            .fillna(0.0)
            .reset_index()
        )
        group_columns = [column_name for column_name in pivot_df.columns if column_name != "Period"]
        group_columns = [
            column_name for column_name in group_columns if pivot_df[column_name].abs().sum() > 0
        ]
        if not group_columns:
            return pd.DataFrame(columns=["Period", "Total"])

        group_columns = sorted(
            group_columns,
            key=lambda column_name: (-pivot_df[column_name].sum(), column_name),
        )
        pivot_df["Total"] = pivot_df[group_columns].sum(axis=1)
        ordered_periods = sort_period_labels(
            pd.unique(pivot_df["Period"]).tolist(),
            "yearly",
        )
        pivot_df["Period"] = pd.Categorical(
            pivot_df["Period"], categories=ordered_periods, ordered=True
        )
        pivot_df = pivot_df.sort_values("Period").reset_index(drop=True)
        pivot_df["Period"] = pivot_df["Period"].astype(str)

        ordered_columns = ["Period", "Total", *group_columns]
        pivot_df = pivot_df[ordered_columns]

        normalized_unit = str(unit or "bcm").strip().lower()
        numeric_columns = [column_name for column_name in ordered_columns if column_name != "Period"]
        if normalized_unit == "bcm":
            pivot_df[numeric_columns] = pivot_df[numeric_columns] * 1.36
        elif normalized_unit == "mcm_d":
            pivot_df[numeric_columns] = pivot_df[numeric_columns] * 1.36 * 1000.0
            pivot_df[numeric_columns] = pivot_df[numeric_columns].div(
                pivot_df["Period"].apply(lambda value: get_days_in_period(value, "yearly")),
                axis=0,
            )

        pivot_df[numeric_columns] = pivot_df[numeric_columns].round(2)
        return pivot_df

    return _pivot_contracts(contracts_df, "export_group"), _pivot_contracts(
        contracts_df, "import_group"
    )


def fetch_trade_balance_payload(
    *,
    time_group: str = "monthly",
    diff_type: str = "percentage",
    country_group: str = "country",
    selected_years: Iterable[int] | None = None,
    unit: str = "bcm",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    try:
        warnings: list[str] = []
        mapping_df = fetch_country_mapping_df()
        export_raw_df = fetch_ea_export_flow_raw_data()
        import_raw_df = fetch_ea_import_flow_raw_data()

        exports_df = _prepare_trade_flow_table(
            export_raw_df,
            country_group=country_group,
            time_group=time_group,
            unit=unit,
            selected_years=selected_years,
            mapping_df=mapping_df,
            start_date=start_date,
            end_date=end_date,
        )
        imports_df = _prepare_trade_flow_table(
            import_raw_df,
            country_group=country_group,
            time_group=time_group,
            unit=unit,
            selected_years=selected_years,
            mapping_df=mapping_df,
            start_date=start_date,
            end_date=end_date,
        )

        available_years = sorted(
            {
                int(timestamp.year)
                for timestamp in pd.concat(
                    [
                        _filter_frame_by_date_range(
                            _sanitize_flow_raw_df(export_raw_df),
                            date_col="month",
                            start_date=start_date,
                            end_date=end_date,
                        )["month"],
                        _filter_frame_by_date_range(
                            _sanitize_flow_raw_df(import_raw_df),
                            date_col="month",
                            start_date=start_date,
                            end_date=end_date,
                        )["month"],
                    ],
                    ignore_index=True,
                )
            }
        )

        if exports_df.empty and imports_df.empty:
            return _payload(
                metadata={
                    "available_years": available_years,
                    "unit": unit,
                    "country_group": country_group,
                },
                error="No trade balance data is available for the current selection.",
            )

        net_df = _build_net_balance_table(
            exports_df,
            imports_df,
            time_group=time_group,
        )

        exports_diff_df = calculate_differences(exports_df, diff_type)
        imports_diff_df = calculate_differences(imports_df, diff_type)

        exports_flex_df = pd.DataFrame(columns=["Period", "Total"])
        imports_flex_df = pd.DataFrame(columns=["Period", "Total"])
        if normalize_time_group(time_group) == "yearly":
            try:
                contract_exports_df, contract_imports_df = fetch_contract_volume_tables(
                    country_group=country_group,
                    selected_years=selected_years,
                    unit=unit,
                    mapping_df=mapping_df,
                )
                exports_flex_df = calculate_flex_volumes(exports_df, contract_exports_df)
                imports_flex_df = calculate_flex_volumes(imports_df, contract_imports_df)
            except Exception as exc:
                warnings.append(f"Flex volumes unavailable: {exc}")

        return _payload(
            data={
                "exports": serialize_frame(exports_df),
                "imports": serialize_frame(imports_df),
                "net": serialize_frame(net_df),
                "exports_diff": serialize_frame(exports_diff_df),
                "imports_diff": serialize_frame(imports_diff_df),
                "exports_flex": serialize_frame(exports_flex_df),
                "imports_flex": serialize_frame(imports_flex_df),
            },
            metadata={
                "available_years": available_years,
                "unit": unit,
                "country_group": country_group,
                "country_group_label": COUNTRY_GROUP_LABELS.get(country_group, "Country"),
                "time_group": normalize_time_group(time_group),
                "diff_type": diff_type,
                "source": "Energy Aspects",
                "warnings": warnings,
                "export_metadata": fetch_ea_export_flow_metadata(),
                "import_metadata": fetch_ea_import_flow_metadata(),
                "start_date": _serialize_scalar(_normalize_range_start(start_date)),
                "end_date": _serialize_scalar(_normalize_range_end(end_date)),
            },
        )
    except Exception as exc:
        return _payload(
            metadata={
                "unit": unit,
                "country_group": country_group,
                "time_group": normalize_time_group(time_group),
                "diff_type": diff_type,
            },
            error=f"Trade balance load failed: {exc}",
        )


def _build_provider_balance_table(
    export_raw_df: pd.DataFrame,
    import_raw_df: pd.DataFrame,
    *,
    time_group: str = "monthly",
    unit: str = "mt",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    normalized_time_group = normalize_time_group(time_group)
    normalized_unit = str(unit or "mt").strip().lower()
    export_df = _filter_frame_by_date_range(
        _sanitize_flow_raw_df(export_raw_df),
        date_col="month",
        start_date=start_date,
        end_date=end_date,
    )
    import_df = _filter_frame_by_date_range(
        _sanitize_flow_raw_df(import_raw_df),
        date_col="month",
        start_date=start_date,
        end_date=end_date,
    )
    if export_df.empty and import_df.empty:
        return pd.DataFrame(columns=["Period", "Supply", "Demand", "Delta"])

    def _group_provider_series(flow_df: pd.DataFrame) -> pd.Series:
        if flow_df.empty:
            return pd.Series(dtype=float)

        working_df = flow_df.copy()
        working_df["Period"] = working_df["month"].apply(
            lambda value: get_time_period(value, normalized_time_group)
        )
        if normalized_time_group == "season":
            working_df = validate_complete_seasons(
                working_df, date_col="month", period_col="Period"
            )
            if working_df.empty:
                return pd.Series(dtype=float)

        working_df["monthly_mt"] = annualized_mmtpa_to_monthly_mt(working_df["total_mmtpa"])
        working_df["monthly_bcm"] = annualized_mmtpa_to_monthly_bcm(working_df["total_mmtpa"])
        value_column = "monthly_mt" if normalized_unit in {"mt", "mmt"} else "monthly_bcm"
        grouped_series = working_df.groupby("Period")[value_column].sum()
        if normalized_unit == "mcm_d":
            grouped_series = grouped_series * 1000.0
            grouped_series = grouped_series / grouped_series.index.to_series().apply(
                lambda period: get_days_in_period(period, normalized_time_group)
            )

        return grouped_series.astype(float)

    export_totals = _group_provider_series(export_df)
    import_totals = _group_provider_series(import_df)
    periods = sort_period_labels(
        sorted(set(export_totals.index.tolist()) | set(import_totals.index.tolist())),
        normalized_time_group,
    )
    if not periods:
        return pd.DataFrame(columns=["Period", "Supply", "Demand", "Delta"])

    result_df = pd.DataFrame({"Period": periods})
    result_df["Supply"] = export_totals.reindex(periods, fill_value=0.0).values
    result_df["Demand"] = import_totals.reindex(periods, fill_value=0.0).values
    result_df["Delta"] = result_df["Supply"] - result_df["Demand"]
    numeric_columns = ["Supply", "Demand", "Delta"]
    result_df[numeric_columns] = result_df[numeric_columns].round(2)
    return result_df


def _build_pacific_supply_detail(
    raw_df: pd.DataFrame,
    provider: str,
    *,
    time_group: str = "monthly",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    normalized_time_group = normalize_time_group(time_group)
    working_df = _filter_frame_by_date_range(
        _sanitize_flow_raw_df(raw_df),
        date_col="month",
        start_date=start_date,
        end_date=end_date,
    )
    working_df = working_df[working_df["country_name"].isin(PACIFIC_LOCKED_COUNTRIES)].copy()
    if working_df.empty:
        return pd.DataFrame(columns=["Period", "Country", "Provider", "Supply"])

    working_df["month"] = pd.to_datetime(working_df["month"], errors="coerce")
    working_df = working_df.dropna(subset=["month"])
    working_df["Period"] = working_df["month"].apply(
        lambda value: get_time_period(value, normalized_time_group)
    )
    if normalized_time_group == "season":
        period_months_df = (
            working_df[["month", "Period"]]
            .assign(month=lambda df: df["month"].dt.to_period("M").dt.to_timestamp())
            .drop_duplicates()
        )
        period_months_df = validate_complete_seasons(
            period_months_df,
            date_col="month",
            period_col="Period",
        )
        complete_periods = period_months_df["Period"].astype(str).unique().tolist()
        if not complete_periods:
            return pd.DataFrame(columns=["Period", "Country", "Provider", "Supply"])
        working_df = working_df[working_df["Period"].isin(complete_periods)].copy()

    period_month_counts = (
        working_df[["Period", "month"]]
        .assign(month=lambda df: df["month"].dt.to_period("M").dt.to_timestamp())
        .drop_duplicates()
        .groupby("Period")["month"]
        .count()
        .to_dict()
    )
    working_df = (
        working_df.groupby(["Period", "country_name"], as_index=False)["total_mmtpa"]
        .sum()
        .rename(columns={"country_name": "Country", "total_mmtpa": "Supply"})
    )
    working_df["Provider"] = provider
    working_df["Supply"] = pd.to_numeric(working_df["Supply"], errors="coerce").fillna(0.0)
    working_df["Supply"] = working_df["Supply"] / working_df["Period"].map(period_month_counts).fillna(1)

    ordered_periods = sort_period_labels(
        pd.unique(working_df["Period"]).tolist(),
        normalized_time_group,
    )
    working_df["Period"] = pd.Categorical(
        working_df["Period"],
        categories=ordered_periods,
        ordered=True,
    )
    working_df = working_df.sort_values(["Period", "Country"]).reset_index(drop=True)
    working_df["Period"] = working_df["Period"].astype(str)
    working_df["Supply"] = working_df["Supply"].round(2)
    return working_df[["Period", "Country", "Provider", "Supply"]]


def _build_pacific_supply_totals(
    detail_df: pd.DataFrame,
    *,
    time_group: str = "monthly",
) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(columns=["Period", "Provider", "Supply", "Equivalent MCM/D"])

    total_df = (
        detail_df.groupby(["Period", "Provider"], as_index=False)["Supply"]
        .sum()
        .sort_values(["Period", "Provider"])
    )
    total_df["Equivalent MCM/D"] = total_df["Supply"] * 1.36 * 1000.0 / 365.0
    total_df["Equivalent MCM/D"] = total_df["Equivalent MCM/D"].round(2)
    ordered_periods = sort_period_labels(
        pd.unique(total_df["Period"]).tolist(),
        normalize_time_group(time_group),
    )
    total_df["Period"] = pd.Categorical(
        total_df["Period"],
        categories=ordered_periods,
        ordered=True,
    )
    total_df = total_df.sort_values(["Period", "Provider"]).reset_index(drop=True)
    total_df["Period"] = total_df["Period"].astype(str)
    return total_df


def filter_complete_ea_monthly_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["month", "value", "upload_timestamp_utc"])

    working_df = raw_df.copy()
    working_df["n_assets"] = pd.to_numeric(
        working_df.get("n_assets"), errors="coerce"
    ).fillna(0)
    max_assets = working_df["n_assets"].max()
    working_df = working_df[working_df["n_assets"] == max_assets].copy()
    if "n_assets" in working_df.columns:
        working_df = working_df.drop(columns=["n_assets"])
    return working_df.reset_index(drop=True)


def convert_wm_maintenance_to_monthly_mt(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["month", "metric", "value"])

    working_df = raw_df.copy()
    working_df["value"] = pd.to_numeric(working_df["value"], errors="coerce").fillna(0.0) / 12.0
    return working_df


def _build_maintenance_grouped_table(
    raw_df: pd.DataFrame,
    *,
    mapping_df: pd.DataFrame,
    country_group: str,
    time_group: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["Period", "Metric", "Total"])

    normalized_time_group = normalize_time_group(time_group)
    working_df = _filter_frame_by_date_range(
        raw_df.copy(),
        date_col="month",
        start_date=start_date,
        end_date=end_date,
    )
    if working_df.empty:
        return pd.DataFrame(columns=["Period", "Metric", "Total"])

    working_df = convert_wm_maintenance_to_monthly_mt(working_df)
    working_df["month"] = pd.to_datetime(working_df["month"], errors="coerce")
    working_df["value"] = pd.to_numeric(working_df["value"], errors="coerce").fillna(0.0)
    working_df["country_name"] = working_df["country_name"].map(_normalize_text_value).fillna("Unknown")
    working_df["metric"] = working_df["metric"].map(_normalize_text_value).fillna("Unknown")
    working_df = working_df.dropna(subset=["month"])
    if working_df.empty:
        return pd.DataFrame(columns=["Period", "Metric", "Total"])

    lookup_maps = _build_country_group_lookup_maps(mapping_df)
    working_df["group_value"] = _resolve_group_values(
        working_df["country_name"],
        country_group,
        lookup_maps,
    )
    working_df["Period"] = working_df["month"].apply(
        lambda value: get_time_period(value, normalized_time_group)
    )

    if normalized_time_group == "season":
        period_months_df = (
            working_df[["month", "Period"]]
            .assign(month=lambda df: df["month"].dt.to_period("M").dt.to_timestamp())
            .drop_duplicates()
        )
        period_months_df = validate_complete_seasons(
            period_months_df,
            date_col="month",
            period_col="Period",
        )
        complete_periods = period_months_df["Period"].astype(str).unique().tolist()
        if not complete_periods:
            return pd.DataFrame(columns=["Period", "Metric", "Total"])
        working_df = working_df[working_df["Period"].isin(complete_periods)].copy()

    grouped_df = (
        working_df.groupby(["Period", "metric", "group_value"], as_index=False)["value"]
        .sum()
        .rename(columns={"metric": "Metric"})
    )
    if grouped_df.empty:
        return pd.DataFrame(columns=["Period", "Metric", "Total"])

    metric_rows_df = (
        grouped_df.groupby(["Period", "Metric", "group_value"], as_index=False)["value"]
        .sum()
    )
    total_rows_df = (
        grouped_df.groupby(["Period", "group_value"], as_index=False)["value"]
        .sum()
    )
    total_rows_df["Metric"] = "Total"
    combined_rows_df = pd.concat([total_rows_df, metric_rows_df], ignore_index=True)

    pivot_df = (
        combined_rows_df.pivot(index=["Period", "Metric"], columns="group_value", values="value")
        .fillna(0.0)
        .reset_index()
    )
    group_columns = [
        column_name
        for column_name in pivot_df.columns
        if column_name not in {"Period", "Metric"}
    ]
    group_columns = [
        column_name
        for column_name in group_columns
        if pivot_df[column_name].abs().sum() > 0
    ]
    if not group_columns:
        return pd.DataFrame(columns=["Period", "Metric", "Total"])

    group_columns = sorted(
        group_columns,
        key=lambda column_name: (-pivot_df[column_name].sum(), column_name),
    )
    pivot_df["Total"] = pivot_df[group_columns].sum(axis=1)

    ordered_periods = sort_period_labels(
        pd.unique(pivot_df["Period"]).tolist(),
        normalized_time_group,
    )
    metric_order = [
        metric_name
        for metric_name in ["Total", "Planned", "Unplanned"]
        if metric_name in pivot_df["Metric"].tolist()
    ]
    remaining_metrics = [
        metric_name
        for metric_name in sorted(pd.unique(pivot_df["Metric"]).tolist())
        if metric_name not in metric_order
    ]
    pivot_df["Period"] = pd.Categorical(
        pivot_df["Period"],
        categories=ordered_periods,
        ordered=True,
    )
    pivot_df["Metric"] = pd.Categorical(
        pivot_df["Metric"],
        categories=[*metric_order, *remaining_metrics],
        ordered=True,
    )
    pivot_df = pivot_df.sort_values(["Period", "Metric"]).reset_index(drop=True)
    pivot_df["Period"] = pivot_df["Period"].astype(str)
    pivot_df["Metric"] = pivot_df["Metric"].astype(str)

    ordered_columns = ["Period", "Metric", "Total", *group_columns]
    pivot_df = pivot_df[ordered_columns]
    numeric_columns = [
        column_name
        for column_name in ordered_columns
        if column_name not in {"Period", "Metric"}
    ]
    pivot_df[numeric_columns] = pivot_df[numeric_columns].round(2)
    return pivot_df


def _periodize_maintenance_frame(
    maintenance_df: pd.DataFrame,
    *,
    time_group: str,
) -> pd.DataFrame:
    if maintenance_df is None or maintenance_df.empty:
        return pd.DataFrame(columns=["Period", "Provider", "Metric", "Value"])

    normalized_time_group = normalize_time_group(time_group)
    working_df = maintenance_df.copy()
    working_df["Month"] = pd.to_datetime(working_df["Month"], errors="coerce")
    working_df["Value"] = pd.to_numeric(working_df["Value"], errors="coerce").fillna(0.0)
    working_df = working_df.dropna(subset=["Month"])
    if working_df.empty:
        return pd.DataFrame(columns=["Period", "Provider", "Metric", "Value"])

    working_df["Period"] = working_df["Month"].apply(
        lambda value: get_time_period(value, normalized_time_group)
    )
    if normalized_time_group == "season":
        period_months_df = (
            working_df[["Month", "Period"]]
            .assign(Month=lambda df: df["Month"].dt.to_period("M").dt.to_timestamp())
            .drop_duplicates()
        )
        period_months_df = validate_complete_seasons(
            period_months_df,
            date_col="Month",
            period_col="Period",
        )
        complete_periods = period_months_df["Period"].astype(str).unique().tolist()
        if not complete_periods:
            return pd.DataFrame(columns=["Period", "Provider", "Metric", "Value"])
        working_df = working_df[working_df["Period"].isin(complete_periods)].copy()

    grouped_df = (
        working_df.groupby(["Period", "Provider", "Metric"], as_index=False)["Value"]
        .sum()
    )
    ordered_periods = sort_period_labels(
        pd.unique(grouped_df["Period"]).tolist(),
        normalized_time_group,
    )
    grouped_df["Period"] = pd.Categorical(
        grouped_df["Period"],
        categories=ordered_periods,
        ordered=True,
    )
    grouped_df = grouped_df.sort_values(["Period", "Provider", "Metric"]).reset_index(drop=True)
    grouped_df["Period"] = grouped_df["Period"].astype(str)
    grouped_df["Value"] = grouped_df["Value"].round(2)
    return grouped_df[["Period", "Provider", "Metric", "Value"]]


MAINTENANCE_PROVIDER_COMPARISON_COLUMNS = [
    "Period",
    "WoodMac Unplanned",
    "Energy Aspects Unplanned",
    "Delta",
    "Delta %",
    "WoodMac Planned",
    "WoodMac Total",
]


def build_maintenance_provider_comparison_table(
    maintenance_df: pd.DataFrame,
    *,
    time_group: str = "monthly",
) -> pd.DataFrame:
    if maintenance_df is None or maintenance_df.empty:
        return pd.DataFrame(columns=MAINTENANCE_PROVIDER_COMPARISON_COLUMNS)

    normalized_time_group = normalize_time_group(time_group)
    working_df = maintenance_df.copy()
    if "Period" not in working_df.columns and "Month" in working_df.columns:
        working_df["Month"] = pd.to_datetime(working_df["Month"], errors="coerce")
        working_df = working_df.dropna(subset=["Month"])
        working_df["Period"] = working_df["Month"].apply(
            lambda value: get_time_period(value, normalized_time_group)
        )

    required_columns = {"Period", "Provider", "Metric", "Value"}
    if not required_columns.issubset(working_df.columns):
        return pd.DataFrame(columns=MAINTENANCE_PROVIDER_COMPARISON_COLUMNS)

    working_df["Period"] = working_df["Period"].astype(str)
    working_df["Provider"] = working_df["Provider"].astype(str)
    working_df["Metric"] = working_df["Metric"].astype(str)
    working_df["Value"] = pd.to_numeric(working_df["Value"], errors="coerce").fillna(0.0)

    grouped_df = (
        working_df.groupby(["Period", "Provider", "Metric"], as_index=False)["Value"]
        .sum()
    )
    if grouped_df.empty:
        return pd.DataFrame(columns=MAINTENANCE_PROVIDER_COMPARISON_COLUMNS)

    pivot_df = grouped_df.pivot_table(
        index="Period",
        columns=["Provider", "Metric"],
        values="Value",
        aggfunc="sum",
    )

    ordered_periods = sort_period_labels(
        pd.unique(grouped_df["Period"]).tolist(),
        normalized_time_group,
    )

    def _series_for(provider: str, metric: str) -> pd.Series:
        if (provider, metric) not in pivot_df.columns:
            return pd.Series(index=ordered_periods, dtype="float64")
        return pivot_df[(provider, metric)].reindex(ordered_periods)

    woodmac_unplanned = _series_for("WoodMac", "Unplanned")
    woodmac_planned = _series_for("WoodMac", "Planned")
    ea_unplanned = _series_for("Energy Aspects", "Unplanned")

    comparison_df = pd.DataFrame(
        {
            "Period": ordered_periods,
            "WoodMac Unplanned": woodmac_unplanned.fillna(0.0).to_numpy(),
            "Energy Aspects Unplanned": ea_unplanned.fillna(0.0).to_numpy(),
            "WoodMac Planned": woodmac_planned.fillna(0.0).to_numpy(),
        }
    )
    comparison_df["WoodMac Total"] = (
        comparison_df["WoodMac Unplanned"] + comparison_df["WoodMac Planned"]
    )
    comparison_df["Delta"] = (
        comparison_df["WoodMac Unplanned"]
        - comparison_df["Energy Aspects Unplanned"]
    )
    comparison_df["Delta %"] = pd.NA
    valid_delta_pct = ea_unplanned.notna() & (ea_unplanned != 0)
    if valid_delta_pct.any():
        comparison_df.loc[valid_delta_pct.to_numpy(), "Delta %"] = (
            comparison_df.loc[valid_delta_pct.to_numpy(), "Delta"]
            / ea_unplanned.loc[valid_delta_pct].to_numpy()
            * 100.0
        )

    numeric_columns = [
        column_name
        for column_name in MAINTENANCE_PROVIDER_COMPARISON_COLUMNS
        if column_name != "Period"
    ]
    comparison_df[numeric_columns] = comparison_df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    comparison_df[numeric_columns] = comparison_df[numeric_columns].round(1)
    return comparison_df[MAINTENANCE_PROVIDER_COMPARISON_COLUMNS]


def _build_ea_maintenance_total_table(maintenance_df: pd.DataFrame) -> pd.DataFrame:
    if maintenance_df is None or maintenance_df.empty:
        return pd.DataFrame(columns=["Period", "Total"])

    working_df = maintenance_df.copy()
    provider_series = working_df.get("Provider", pd.Series(dtype=str)).astype(str)
    working_df = working_df[provider_series == "Energy Aspects"].copy()
    if working_df.empty:
        return pd.DataFrame(columns=["Period", "Total"])

    working_df["Value"] = pd.to_numeric(working_df["Value"], errors="coerce").fillna(0.0)
    grouped_df = (
        working_df.groupby("Period", as_index=False)["Value"]
        .sum()
        .rename(columns={"Value": "Total"})
    )
    grouped_df["Total"] = grouped_df["Total"].round(2)
    return grouped_df[["Period", "Total"]]


def fetch_ea_dataset_metadata(dataset_id: str) -> dict:
    query = text(
        f"""
        WITH ranked_metadata AS (
            SELECT
                CAST(dataset_id AS INTEGER) AS dataset_id,
                type,
                NULLIF(TRIM(value), '') AS value,
                upload_timestamp_utc,
                ROW_NUMBER() OVER (
                    PARTITION BY dataset_id, type
                    ORDER BY upload_timestamp_utc DESC NULLS LAST
                ) AS row_num
            FROM {DB_SCHEMA}.ea_metadata
            WHERE dataset_id = :dataset_id
        ),
        latest_metadata AS (
            SELECT dataset_id, type, value, upload_timestamp_utc
            FROM ranked_metadata
            WHERE row_num = 1
        )
        SELECT
            dataset_id,
            MAX(CASE WHEN type = 'description' THEN value END) AS description,
            MAX(CASE WHEN type = 'aspect' THEN value END) AS aspect,
            MAX(CASE WHEN type = 'aspect_subtype' THEN value END) AS aspect_subtype,
            MAX(CASE WHEN type = 'category' THEN value END) AS category,
            MAX(CASE WHEN type = 'category_subtype' THEN value END) AS category_subtype,
            MAX(CASE WHEN type = 'frequency' THEN value END) AS frequency,
            MAX(CASE WHEN type = 'region' THEN value END) AS region,
            MAX(CASE WHEN type = 'source' THEN value END) AS source,
            MAX(CASE WHEN type = 'unit' THEN value END) AS unit,
            MAX(CASE WHEN type = 'release_date' THEN value END) AS release_date,
            MAX(upload_timestamp_utc) AS metadata_upload_timestamp_utc
        FROM latest_metadata
        GROUP BY dataset_id
        """
    )
    with engine.connect() as connection:
        metadata_df = pd.read_sql_query(query, connection, params={"dataset_id": dataset_id})

    if metadata_df.empty:
        return {}
    return {
        key: _serialize_scalar(value)
        for key, value in metadata_df.iloc[0].to_dict().items()
    }


def fetch_global_maintenance_overview(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    time_group: str = "monthly",
) -> tuple[pd.DataFrame, dict]:
    wm_query = text(
        f"""
        WITH maintenance_union AS (
            SELECT
                make_date(year::int, month::int, 1) AS month,
                'Unplanned' AS metric,
                SUM(metric_value) AS value
            FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_unplanned_downtime_mta
            WHERE metric_value > 0
            GROUP BY 1

            UNION ALL

            SELECT
                make_date(year::int, month::int, 1) AS month,
                'Planned' AS metric,
                SUM(metric_value) AS value
            FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_planned_maintenance_mta
            WHERE metric_value > 0
            GROUP BY 1
        )
        SELECT month, metric, value
        FROM maintenance_union
        ORDER BY month, metric
        """
    )
    ea_query = text(
        f"""
        WITH latest_snapshot AS (
            SELECT MAX(upload_timestamp_utc) AS upload_timestamp_utc
            FROM {DB_SCHEMA}.ea_values
            WHERE dataset_id = '15522'
        )
        SELECT
            date::date AS month,
            ROUND(SUM(value)::numeric, 2) AS value,
            MAX(upload_timestamp_utc) AS upload_timestamp_utc,
            COUNT(*) AS n_assets
        FROM {DB_SCHEMA}.ea_values
        WHERE dataset_id = '15522'
          AND upload_timestamp_utc = (SELECT upload_timestamp_utc FROM latest_snapshot)
        GROUP BY 1
        ORDER BY month
        """
    )

    with engine.connect() as connection:
        wm_df = pd.read_sql_query(wm_query, connection)
        ea_df = pd.read_sql_query(ea_query, connection)

    wm_df = _filter_frame_by_date_range(
        wm_df,
        date_col="month",
        start_date=start_date,
        end_date=end_date,
    )
    ea_df = _filter_frame_by_date_range(
        ea_df,
        date_col="month",
        start_date=start_date,
        end_date=end_date,
    )

    if not wm_df.empty:
        wm_df = convert_wm_maintenance_to_monthly_mt(wm_df)
        wm_df["Provider"] = "WoodMac"
        wm_df["Month"] = pd.to_datetime(wm_df["month"])
        wm_df["Metric"] = wm_df["metric"]
        wm_df["Value"] = pd.to_numeric(wm_df["value"], errors="coerce").fillna(0.0)
        wm_df = wm_df[["Month", "Provider", "Metric", "Value"]]
    else:
        wm_df = pd.DataFrame(columns=["Month", "Provider", "Metric", "Value"])

    ea_upload_timestamp = None
    if not ea_df.empty:
        ea_df = filter_complete_ea_monthly_rows(ea_df)
        ea_upload_timestamp = ea_df["upload_timestamp_utc"].max()
        ea_df["Provider"] = "Energy Aspects"
        ea_df["Metric"] = "Unplanned"
        ea_df["Month"] = pd.to_datetime(ea_df["month"])
        ea_df["Value"] = pd.to_numeric(ea_df["value"], errors="coerce").fillna(0.0)
        ea_df = ea_df[["Month", "Provider", "Metric", "Value"]]
    else:
        ea_df = pd.DataFrame(columns=["Month", "Provider", "Metric", "Value"])

    ea_dataset_metadata = {}
    try:
        ea_dataset_metadata = fetch_ea_dataset_metadata("15522")
    except Exception:
        ea_dataset_metadata = {}

    combined_df = pd.concat([wm_df, ea_df], ignore_index=True)
    if not combined_df.empty:
        combined_df = _periodize_maintenance_frame(
            combined_df,
            time_group=time_group,
        )

    metadata = {
        "ea_upload_timestamp_utc": _serialize_scalar(ea_upload_timestamp),
        "ea_dataset": ea_dataset_metadata,
        "notes": [
            "WoodMac maintenance is converted from annualized MMTPA-style values to monthly Mt for comparison.",
            "Energy Aspects maintenance uses dataset_id 15522 and keeps only complete monthly rows with max asset count.",
        ],
    }
    return combined_df, metadata


def fetch_woodmac_maintenance_grouped_table(
    *,
    mapping_df: pd.DataFrame,
    country_group: str,
    time_group: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    query = text(
        f"""
        WITH maintenance_union AS (
            SELECT
                make_date(year::int, month::int, 1) AS month,
                COALESCE(country_name, 'Unknown') AS country_name,
                'Unplanned' AS metric,
                SUM(metric_value) AS value
            FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_unplanned_downtime_mta
            WHERE metric_value > 0
            GROUP BY 1, 2

            UNION ALL

            SELECT
                make_date(year::int, month::int, 1) AS month,
                COALESCE(country_name, 'Unknown') AS country_name,
                'Planned' AS metric,
                SUM(metric_value) AS value
            FROM {DB_SCHEMA}.woodmac_lng_plant_train_monthly_planned_maintenance_mta
            WHERE metric_value > 0
            GROUP BY 1, 2
        )
        SELECT month, country_name, metric, value
        FROM maintenance_union
        ORDER BY month, country_name, metric
        """
    )
    with engine.connect() as connection:
        raw_df = pd.read_sql_query(query, connection)

    return _build_maintenance_grouped_table(
        raw_df,
        mapping_df=mapping_df,
        country_group=country_group,
        time_group=time_group,
        start_date=start_date,
        end_date=end_date,
    )


def fetch_provider_overview_payload(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    time_group: str = "yearly",
    unit: str = "bcm",
    country_group: str = "country_classification_level1",
) -> dict:
    try:
        woodmac_export_df = fetch_woodmac_export_flow_raw_data()
        woodmac_import_df = fetch_woodmac_import_flow_raw_data()
        ea_export_df = fetch_ea_export_flow_raw_data()
        ea_import_df = fetch_ea_import_flow_raw_data()
        mapping_df = fetch_country_mapping_df()

        woodmac_balance_df = _build_provider_balance_table(
            woodmac_export_df,
            woodmac_import_df,
            time_group=time_group,
            unit=unit,
            start_date=start_date,
            end_date=end_date,
        )
        ea_balance_df = _build_provider_balance_table(
            ea_export_df,
            ea_import_df,
            time_group=time_group,
            unit=unit,
            start_date=start_date,
            end_date=end_date,
        )
        overview_net_config = OVERVIEW_NET_DEFAULTS.copy()
        overview_net_config.update(
            {
                "country_group": country_group,
                "time_group": time_group,
                "unit": unit,
            }
        )
        woodmac_net_balance_df = _build_provider_net_balance_table(
            woodmac_export_df,
            woodmac_import_df,
            mapping_df=mapping_df,
            start_date=start_date,
            end_date=end_date,
            **overview_net_config,
        )
        ea_net_balance_df = _build_provider_net_balance_table(
            ea_export_df,
            ea_import_df,
            mapping_df=mapping_df,
            start_date=start_date,
            end_date=end_date,
            **overview_net_config,
        )
        maintenance_df, maintenance_metadata = fetch_global_maintenance_overview(
            start_date=start_date,
            end_date=end_date,
            time_group=time_group,
        )
        maintenance_grouped_df = fetch_woodmac_maintenance_grouped_table(
            mapping_df=mapping_df,
            country_group=country_group,
            time_group=time_group,
            start_date=start_date,
            end_date=end_date,
        )
        maintenance_ea_df = _build_ea_maintenance_total_table(maintenance_df)
        maintenance_provider_comparison_df = build_maintenance_provider_comparison_table(
            maintenance_df,
            time_group=time_group,
        )

        pacific_detail_df = pd.concat(
            [
                _build_pacific_supply_detail(
                    woodmac_export_df,
                    "WoodMac",
                    time_group=time_group,
                    start_date=start_date,
                    end_date=end_date,
                ),
                _build_pacific_supply_detail(
                    ea_export_df,
                    "Energy Aspects",
                    time_group=time_group,
                    start_date=start_date,
                    end_date=end_date,
                ),
            ],
            ignore_index=True,
        )
        pacific_totals_df = _build_pacific_supply_totals(
            pacific_detail_df,
            time_group=time_group,
        )

        metadata = {
            "woodmac_export": fetch_woodmac_export_flow_metadata(),
            "woodmac_import": fetch_woodmac_import_flow_metadata(),
            "ea_export": fetch_ea_export_flow_metadata(),
            "ea_import": fetch_ea_import_flow_metadata(),
            "maintenance": maintenance_metadata,
            "pacific_countries": PACIFIC_LOCKED_COUNTRIES,
            "comparison_options": fetch_net_balance_comparison_options(),
            "overview_net": {
                "country_group": overview_net_config["country_group"],
                "country_group_label": COUNTRY_GROUP_LABELS.get(
                    overview_net_config["country_group"], "Country"
                ),
                "time_group": normalize_time_group(overview_net_config["time_group"]),
                "unit": overview_net_config["unit"],
                "start_date": _serialize_scalar(_normalize_range_start(start_date)),
                "end_date": _serialize_scalar(_normalize_range_end(end_date)),
            },
        }

        return _payload(
            data={
                "woodmac_balance": serialize_frame(woodmac_balance_df),
                "ea_balance": serialize_frame(ea_balance_df),
                "woodmac_net_balance": serialize_frame(woodmac_net_balance_df),
                "ea_net_balance": serialize_frame(ea_net_balance_df),
                "maintenance": serialize_frame(maintenance_df),
                "maintenance_grouped": serialize_frame(maintenance_grouped_df),
                "maintenance_ea": serialize_frame(maintenance_ea_df),
                "maintenance_provider_comparison": serialize_frame(
                    maintenance_provider_comparison_df
                ),
                "pacific_detail": serialize_frame(pacific_detail_df),
                "pacific_totals": serialize_frame(pacific_totals_df),
            },
            metadata=metadata,
        )
    except Exception as exc:
        return _payload(error=f"Overview load failed: {exc}")


def ensure_balance_hierarchy_columns(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()
    column_tuples = []

    if not isinstance(working_df.columns, pd.MultiIndex):
        for column_name in working_df.columns:
            if isinstance(column_name, tuple):
                raw_levels = list(column_name[:3])
            else:
                raw_levels = [column_name]

            while len(raw_levels) < 3:
                raw_levels.append("")

            column_tuples.append(
                tuple(_normalize_hierarchy_level(value) for value in raw_levels[:3])
            )
    else:
        for column_name in working_df.columns:
            raw_levels = list(column_name[:3])
            while len(raw_levels) < 3:
                raw_levels.append("")

            column_tuples.append(
                tuple(_normalize_hierarchy_level(value) for value in raw_levels[:3])
            )

    working_df.columns = pd.MultiIndex.from_tuples(
        column_tuples,
        names=["type", "subtype", "subsubtype"],
    )
    return working_df


def _normalize_hierarchy_level(value) -> str:
    if isinstance(value, slice):
        return ""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    normalized = str(value).strip()
    if normalized.lower() in {"", "nan", "none", "nat"}:
        return ""
    return normalized


def sort_balance_columns(
    columns: Iterable[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    normalized_columns = [
        tuple(_normalize_hierarchy_level(value) for value in column_tuple[:3])
        for column_tuple in columns
    ]
    return sorted(
        normalized_columns,
        key=lambda column_tuple: tuple(value.lower() for value in column_tuple),
    )


def compress_balance_column_levels(
    columns: Iterable[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    compressed_columns = []
    for level_1, level_2, level_3 in columns:
        compressed_columns.append(
            (
                level_1,
                "" if level_2 in {"", level_1} else level_2,
                "" if level_3 in {"", level_2} else level_3,
            )
        )
    return compressed_columns


def determine_balance_column_color(column_tuple: tuple[str, str, str]) -> str:
    level_1, level_2, level_3 = column_tuple
    non_empty_levels = sum(1 for value in column_tuple if value)

    if level_3:
        return "#FFFFFF"
    if level_1 == "Demand":
        return "#ADD8E6" if non_empty_levels == 1 else "#E0FFFF"
    if level_1 == "Supply":
        return "#FFD700" if non_empty_levels == 1 else "#FFF8DC"
    if level_1 == "Fcst Margin":
        return "#F7D2FC"
    return "#FFFFFF"


def get_balance_column_styles(
    display_columns: list[str],
    formatted_columns: list[tuple[str, str, str]],
) -> list[dict]:
    balance_columns = [
        column_name
        for column_name in display_columns
        if column_name not in {"Date", "Period", "Month", "Timestamp UTC"}
    ]
    return [
        {
            "if": {"column_id": column_name},
            "backgroundColor": determine_balance_column_color(column_tuple),
            "color": "black",
        }
        for column_name, column_tuple in zip(balance_columns, formatted_columns)
    ]


def build_balance_display_columns(
    formatted_columns: list[tuple[str, str, str]],
) -> list[str]:
    candidate_labels = [
        level_3 or level_2 or level_1 or "Unmapped"
        for level_1, level_2, level_3 in formatted_columns
    ]
    candidate_counts = Counter(candidate_labels)

    display_columns: list[str] = []
    for candidate_label, (level_1, level_2, level_3) in zip(
        candidate_labels, formatted_columns
    ):
        if candidate_counts[candidate_label] == 1 and candidate_label != "Unmapped":
            display_columns.append(candidate_label)
            continue

        expanded_parts = [value for value in (level_1, level_2, level_3) if value]
        display_columns.append(" | ".join(expanded_parts) if expanded_parts else "Unmapped")

    unique_labels: list[str] = []
    seen_counts: Counter[str] = Counter()
    for label in display_columns:
        base_label = label or "Unmapped"
        seen_counts[base_label] += 1
        if seen_counts[base_label] == 1:
            unique_labels.append(base_label)
        else:
            unique_labels.append(f"{base_label} ({seen_counts[base_label]})")

    return unique_labels


def format_country_balance_table(
    df: pd.DataFrame,
    level: str,
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    if df.empty:
        return pd.DataFrame(columns=["Date", "Timestamp UTC"]), []

    pivot_frames = [
        ensure_balance_hierarchy_columns(
            pd.pivot_table(
                df,
                values="value",
                index=["Date", "Timestamp UTC"],
                columns=["type"],
                aggfunc="sum",
                fill_value=0,
            )
        ),
        ensure_balance_hierarchy_columns(
            pd.pivot_table(
                df,
                values="value",
                index=["Date", "Timestamp UTC"],
                columns=["type", "subtype"],
                aggfunc="sum",
                fill_value=0,
            )
        ),
    ]

    if level != "subtype":
        pivot_frames.append(
            ensure_balance_hierarchy_columns(
                pd.pivot_table(
                    df,
                    values="value",
                    index=["Date", "Timestamp UTC"],
                    columns=["type", "subtype", "subsubtype"],
                    aggfunc="sum",
                    fill_value=0,
                )
            )
        )

    combined_df = pd.concat(pivot_frames, axis=1)

    combined_df = ensure_balance_hierarchy_columns(combined_df)
    combined_df = combined_df.T.groupby(level=[0, 1, 2]).first().T
    combined_df = combined_df.reindex(
        columns=pd.MultiIndex.from_tuples(
            sort_balance_columns(list(combined_df.columns)),
            names=["type", "subtype", "subsubtype"],
        )
    )
    compressed_columns = compress_balance_column_levels(combined_df.columns)
    combined_df.columns = pd.MultiIndex.from_tuples(
        compressed_columns, names=["type", "subtype", "subsubtype"]
    )

    combined_df = combined_df.T.groupby(level=[0, 1, 2]).first().T
    formatted_columns = list(combined_df.columns)

    if ("Supply", "", "") not in formatted_columns:
        combined_df[("Supply", "", "")] = 0.0
        formatted_columns.append(("Supply", "", ""))
    if ("Demand", "", "") not in formatted_columns:
        combined_df[("Demand", "", "")] = 0.0
        formatted_columns.append(("Demand", "", ""))
    if ("Fcst Margin", "", "") not in formatted_columns:
        combined_df[("Fcst Margin", "", "")] = (
            combined_df[("Supply", "", "")] - combined_df[("Demand", "", "")]
        )
        formatted_columns.append(("Fcst Margin", "", ""))

    non_margin_columns = [
        column_name for column_name in formatted_columns if column_name[0] != "Fcst Margin"
    ]
    margin_columns = [
        column_name for column_name in formatted_columns if column_name[0] == "Fcst Margin"
    ]
    formatted_columns = non_margin_columns + margin_columns
    combined_df = combined_df[formatted_columns].reset_index().round(2)

    display_columns = ["Date", "Timestamp UTC"] + build_balance_display_columns(
        formatted_columns
    )
    combined_df.columns = display_columns
    return combined_df, formatted_columns


def _build_country_balance_delta_table(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame,
) -> pd.DataFrame:
    if current_df.empty:
        return current_df.copy()

    numeric_columns = current_df.select_dtypes(include=["number"]).columns.tolist()
    result_df = current_df.copy()
    if previous_df.empty:
        return result_df.iloc[0:0].copy()

    period_col = "Date" if "Date" in current_df.columns else "Period"
    aligned_previous_df = align_frames_on_period(
        current_df,
        previous_df,
        period_col=period_col,
    )
    for column_name in numeric_columns:
        result_df[column_name] = (
            pd.to_numeric(result_df[column_name], errors="coerce").fillna(0.0)
            - pd.to_numeric(aligned_previous_df[column_name], errors="coerce").fillna(0.0)
        ).round(2)

    return result_df


def _drop_country_display_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.drop(columns=["Timestamp UTC"], errors="ignore")
    return df.drop(columns=["Timestamp UTC"], errors="ignore")


def fetch_country_balance_meta_payload() -> dict:
    try:
        countries_query = text(
            f"""
            SELECT DISTINCT country
            FROM {DB_SCHEMA}.fundamentals_global_balance_ea_datasets
            WHERE used != 'N'
              AND balance = 'Gas'
            ORDER BY country
            """
        )
        snapshots_query = text(
            f"""
            WITH selected_datasets AS (
                SELECT DISTINCT
                    country,
                    CAST(dataset_id AS TEXT) AS dataset_id
                FROM {DB_SCHEMA}.fundamentals_global_balance_ea_datasets
                WHERE used != 'N'
                  AND balance = 'Gas'
            )
            SELECT
                selected_datasets.country,
                values_table.upload_timestamp_utc::timestamp AS upload_timestamp_utc
            FROM {DB_SCHEMA}.ea_values values_table
            JOIN selected_datasets
                ON CAST(values_table.dataset_id AS TEXT) = selected_datasets.dataset_id
            WHERE upload_timestamp_utc IS NOT NULL
            GROUP BY selected_datasets.country, values_table.upload_timestamp_utc
            ORDER BY selected_datasets.country, values_table.upload_timestamp_utc DESC
            """
        )
        with engine.connect() as connection:
            countries_df = pd.read_sql_query(countries_query, connection)
            snapshots_df = pd.read_sql_query(snapshots_query, connection)

        countries = [
            _normalize_text_value(value)
            for value in countries_df.get("country", pd.Series(dtype=object)).tolist()
        ]
        countries = [value for value in countries if value]
        country_snapshots: dict[str, list[str]] = {}
        all_snapshot_values = []
        for _, snapshot_row in snapshots_df.iterrows():
            country = _normalize_text_value(snapshot_row.get("country"))
            timestamp = pd.to_datetime(
                snapshot_row.get("upload_timestamp_utc"), errors="coerce"
            )
            if not country or pd.isna(timestamp):
                continue

            snapshot = _serialize_scalar(timestamp)
            if not snapshot:
                continue
            country_snapshots.setdefault(country, []).append(snapshot)
            all_snapshot_values.append(timestamp)

        snapshots = [
            _serialize_scalar(value)
            for value in sorted(set(all_snapshot_values), reverse=True)
        ]
        snapshots = [value for value in snapshots if value]
        default_country = countries[0] if countries else None
        default_country_snapshots = country_snapshots.get(default_country, snapshots)
        return _payload(
            data={
                "countries": countries,
                "snapshots": snapshots,
                "country_snapshots": country_snapshots,
            },
            metadata={
                "default_country": default_country,
                "default_snapshot": (
                    default_country_snapshots[1]
                    if len(default_country_snapshots) > 1
                    else None
                ),
                "latest_snapshot": snapshots[0] if snapshots else None,
            },
        )
    except Exception as exc:
        return _payload(error=f"Country drilldown options load failed: {exc}")


def build_country_conversion_factors_cte() -> tuple[str, list[str]]:
    if _relation_exists("fundamentals_ea_global_datasets"):
        return (
            f"""
            conversion_factors AS (
                SELECT
                    CAST(id AS TEXT) AS dataset_id,
                    COALESCE(conversion_factor_bcm_gas, 1) AS conversion_factor_bcm_gas
                FROM {DB_SCHEMA}.fundamentals_ea_global_datasets
            )
            """,
            [],
        )

    return (
        """
        conversion_factors AS (
            SELECT
                NULL::TEXT AS dataset_id,
                NULL::DOUBLE PRECISION AS conversion_factor_bcm_gas
            WHERE FALSE
        )
        """,
        [
            "Legacy EA conversion-factor table unavailable; country balance is using raw EA values."
        ],
    )


def _fetch_country_balance_raw_data(
    country: str,
    *,
    snapshot_timestamp: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    conversion_factors_cte, warnings = build_country_conversion_factors_cte()
    query = text(
        f"""
        WITH selected_datasets AS (
            SELECT
                CAST(dataset_id AS TEXT) AS dataset_id,
                type,
                subtype,
                subsubtype,
                operation
            FROM {DB_SCHEMA}.fundamentals_global_balance_ea_datasets
            WHERE country = :country
              AND balance = 'Gas'
              AND used != 'N'
        ),
        latest_snapshot AS (
            SELECT MAX(values_table.upload_timestamp_utc) AS upload_timestamp_utc
            FROM {DB_SCHEMA}.ea_values values_table
            JOIN selected_datasets
                ON CAST(values_table.dataset_id AS TEXT) = selected_datasets.dataset_id
        ),
        {conversion_factors_cte}
        SELECT
            CAST(values_table.dataset_id AS TEXT) AS dataset_id,
            values_table.date::date AS date,
            values_table.value,
            values_table.upload_timestamp_utc::timestamp AS upload_timestamp_utc,
            selected_datasets.type,
            selected_datasets.subtype,
            selected_datasets.subsubtype,
            selected_datasets.operation,
            COALESCE(conversion_factors.conversion_factor_bcm_gas, 1) AS conversion_factor_bcm_gas
        FROM {DB_SCHEMA}.ea_values values_table
        JOIN selected_datasets
            ON CAST(values_table.dataset_id AS TEXT) = selected_datasets.dataset_id
        LEFT JOIN conversion_factors
            ON CAST(values_table.dataset_id AS TEXT) = conversion_factors.dataset_id
        WHERE values_table.upload_timestamp_utc = COALESCE(
            CAST(:snapshot_timestamp AS timestamp),
            (SELECT upload_timestamp_utc FROM latest_snapshot)
        )
        ORDER BY values_table.date
        """
    )
    with engine.connect() as connection:
        raw_df = pd.read_sql_query(
            query,
            connection,
            params={"country": country, "snapshot_timestamp": snapshot_timestamp},
        )

    if raw_df.empty:
        return raw_df, warnings

    raw_df["value"] = pd.to_numeric(raw_df["value"], errors="coerce").fillna(0.0)
    raw_df["conversion_factor_bcm_gas"] = pd.to_numeric(
        raw_df["conversion_factor_bcm_gas"], errors="coerce"
    ).fillna(1.0)
    raw_df.loc[raw_df["operation"] == "-", "value"] *= -1.0
    raw_df["value"] = raw_df["value"] * raw_df["conversion_factor_bcm_gas"]
    raw_df["Date"] = pd.to_datetime(raw_df["date"]).dt.date
    raw_df["Timestamp UTC"] = pd.to_datetime(
        raw_df["upload_timestamp_utc"], errors="coerce"
    ).map(_serialize_scalar)
    return raw_df, warnings


def _build_country_chart_payloads(
    raw_df: pd.DataFrame,
    *,
    country: str,
    time_group: str,
) -> list[dict]:
    chart_payloads: list[dict] = []
    if raw_df.empty:
        return chart_payloads

    for config in COUNTRY_CATEGORY_CONFIG:
        category_df = raw_df[raw_df["subtype"].isin(config["aliases"])].copy()
        if category_df.empty:
            continue

        category_df["Date"] = pd.to_datetime(category_df["Date"], errors="coerce")
        category_df["subtype"] = category_df["subtype"].map(_normalize_text_value)
        category_df["subsubtype"] = category_df["subsubtype"].map(_normalize_text_value)
        category_df["series_name"] = category_df["subsubtype"].where(
            category_df["subsubtype"].notna(),
            category_df["subtype"].fillna(config["title"]),
        )
        category_df["Period"] = category_df["Date"].apply(
            lambda value: get_time_period(value, time_group)
        )
        if normalize_time_group(time_group) == "season":
            category_df = validate_complete_seasons(
                category_df, date_col="Date", period_col="Period"
            )
            if category_df.empty:
                continue

        grouped_df = (
            category_df.groupby(["Period", "series_name"], as_index=False)["value"]
            .sum()
            .rename(columns={"Period": "Date"})
        )
        grouped_df["Date"] = pd.Categorical(
            grouped_df["Date"],
            categories=sort_period_labels(grouped_df["Date"].unique().tolist(), time_group),
            ordered=True,
        )
        grouped_df = grouped_df.sort_values(["Date", "series_name"]).reset_index(drop=True)
        grouped_df["Date"] = grouped_df["Date"].astype(str)
        grouped_df["value"] = grouped_df["value"].round(2)

        chart_payloads.append(
            {
                "title": f"{country} {config['title']}",
                "chart_type": config["chart_type"],
                "frame": serialize_frame(grouped_df),
            }
        )

    return chart_payloads


def fetch_country_balance_payload(
    *,
    country: str | None,
    level: str = "subtype",
    time_group: str = "monthly",
    comparison_timestamp: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    if not country:
        return _payload(error="Select a country to load the drilldown.")

    try:
        current_raw_df, current_warnings = _fetch_country_balance_raw_data(country)
        current_raw_df = _filter_frame_by_date_range(
            current_raw_df,
            date_col="Date",
            start_date=start_date,
            end_date=end_date,
        )
        if current_raw_df.empty:
            return _payload(error=f"No country balance data is available for {country}.")

        comparison_requested = bool(comparison_timestamp)
        previous_raw_df, previous_warnings = (
            _fetch_country_balance_raw_data(country, snapshot_timestamp=comparison_timestamp)
            if comparison_requested
            else (pd.DataFrame(), [])
        )
        previous_raw_df = _filter_frame_by_date_range(
            previous_raw_df,
            date_col="Date",
            start_date=start_date,
            end_date=end_date,
        )

        current_table_df, formatted_columns = format_country_balance_table(current_raw_df, level)
        current_grouped_df = group_numeric_frame_by_period(
            current_table_df,
            time_group,
            date_col="Date",
            preserve_first=["Timestamp UTC"],
        )

        previous_table_df, _ = format_country_balance_table(previous_raw_df, level)
        previous_grouped_df = group_numeric_frame_by_period(
            previous_table_df,
            time_group,
            date_col="Date",
            preserve_first=["Timestamp UTC"],
        )
        delta_df = _build_country_balance_delta_table(current_grouped_df, previous_grouped_df)
        comparison_snapshot_used = (
            previous_raw_df["Timestamp UTC"].iloc[0]
            if not previous_raw_df.empty and "Timestamp UTC" in previous_raw_df.columns
            else None
        )

        warnings = list(dict.fromkeys([*current_warnings, *previous_warnings]))
        delta_nonzero_cells = 0
        if comparison_requested and previous_raw_df.empty:
            warnings.append(
                f"No comparison rows found for {country} at {comparison_timestamp}; "
                "Delta vs Snapshot is unavailable for that selection."
            )
        elif comparison_requested and not delta_df.empty:
            numeric_delta_df = delta_df.select_dtypes(include=["number"])
            if not numeric_delta_df.empty:
                delta_nonzero_cells = int(
                    numeric_delta_df.fillna(0.0).ne(0.0).sum().sum()
                )
            if delta_nonzero_cells == 0:
                warnings.append(
                    "Selected comparison snapshot loaded, but no numeric deltas were "
                    "found for the current date range and time view."
                )

        chart_df = current_grouped_df.copy()
        for required_column in ["Demand", "Supply", "Fcst Margin"]:
            if required_column not in chart_df.columns:
                chart_df[required_column] = 0.0

        current_display_df = _drop_country_display_metadata_columns(current_grouped_df)
        delta_display_df = _drop_country_display_metadata_columns(delta_df)

        chart_payloads = _build_country_chart_payloads(
            current_raw_df, country=country, time_group=time_group
        )

        metadata = {
            "country": country,
            "level": level,
            "time_group": normalize_time_group(time_group),
            "current_snapshot": current_raw_df["Timestamp UTC"].iloc[0]
            if not current_raw_df.empty
            else None,
            "comparison_snapshot": comparison_snapshot_used,
            "requested_comparison_snapshot": comparison_timestamp,
            "comparison_row_count": int(len(previous_raw_df)) if comparison_requested else 0,
            "delta_nonzero_cells": delta_nonzero_cells,
            "warnings": warnings,
            "column_styles": get_balance_column_styles(
                current_display_df.columns.tolist(),
                formatted_columns,
            ),
            "start_date": _serialize_scalar(_normalize_range_start(start_date)),
            "end_date": _serialize_scalar(_normalize_range_end(end_date)),
        }

        return _payload(
            data={
                "current_table": serialize_frame(current_display_df),
                "delta_table": serialize_frame(delta_display_df),
                "balance_chart": serialize_frame(
                    chart_df[["Date", "Demand", "Supply", "Fcst Margin"]]
                ),
                "category_charts": chart_payloads,
            },
            metadata=metadata,
        )
    except Exception as exc:
        return _payload(error=f"Country drilldown load failed: {exc}")
