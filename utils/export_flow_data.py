import configparser
import os

import pandas as pd
from sqlalchemy import create_engine, text

from utils.ea_balance_catalog import build_resolved_ea_lng_balance_ctes
from utils.ea_run_interface import (
    ea_values_at_run_source_sql,
    fetch_current_ea_run,
    fetch_ea_comparison_runs,
    normalize_ea_run_id,
)
from utils.flow_country_selection import (
    build_woodmac_flow_metadata,
    build_woodmac_publication_options,
    resolve_available_countries,
    resolve_default_selected_countries,
    sanitize_raw_flow_data,
)


try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    CONFIG_FILE_PATH = os.path.join(config_dir, "config.ini")
except Exception:
    CONFIG_FILE_PATH = "config.ini"


config_reader = configparser.ConfigParser(interpolation=None)
config_reader.read(CONFIG_FILE_PATH)

DB_CONNECTION_STRING = config_reader.get("DATABASE", "CONNECTION_STRING", fallback=None)
DB_SCHEMA = config_reader.get("DATABASE", "SCHEMA", fallback="at_lng")

if not DB_CONNECTION_STRING:
    raise ValueError(f"Missing DATABASE CONNECTION_STRING in {CONFIG_FILE_PATH}")


engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)


DEFAULT_SELECTED_COUNTRIES = [
    "United States",
    "Qatar",
    "United Arab Emirates",
    "Australia",
    "Russia",
    "Canada",
    "Mozambique",
]

MONTH_YEAR_REGEX = (
    "(January|February|March|April|May|June|July|August|September|October|"
    "November|December)\\s+(\\d{4})"
)

COUNTRY_MAPPING_CTE = f"""
country_mapping AS (
    SELECT
        raw_country_key,
        MIN(country_name) AS country_name
    FROM (
        SELECT
            UPPER(TRIM(country)) AS raw_country_key,
            country_name
        FROM {DB_SCHEMA}.mappings_country
        WHERE COALESCE(TRIM(country), '') <> ''
          AND COALESCE(TRIM(country_name), '') <> ''

        UNION ALL

        SELECT
            UPPER(TRIM(country_name)) AS raw_country_key,
            country_name
        FROM {DB_SCHEMA}.mappings_country
        WHERE COALESCE(TRIM(country_name), '') <> ''
    ) standardized_mapping
    GROUP BY raw_country_key
)
"""

WOODMAC_EXPORT_FLOW_QUERY = f"""
WITH
{COUNTRY_MAPPING_CTE},
latest_short_term_market AS (
    SELECT
        market_outlook,
        MAX(publication_date::timestamp) AS publication_timestamp
    FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa
    WHERE release_type = 'Short Term Outlook'
      AND direction = 'Export'
      AND measured_at = 'Exit'
      AND metric_name = 'Flow'
    GROUP BY market_outlook
    ORDER BY TO_DATE(
        (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[1]
        || ' ' ||
        (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[2],
        'Month YYYY'
    ) DESC NULLS LAST,
    MAX(publication_date::timestamp) DESC
    LIMIT 1
),
latest_long_term_market AS (
    SELECT
        market_outlook,
        MAX(publication_date::timestamp) AS publication_timestamp
    FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa
    WHERE release_type = 'Long Term Outlook'
      AND direction = 'Export'
      AND measured_at = 'Exit'
      AND metric_name = 'Flow'
    GROUP BY market_outlook
    ORDER BY TO_DATE(
        (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[1]
        || ' ' ||
        (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[2],
        'Month YYYY'
    ) DESC NULLS LAST,
    MAX(publication_date::timestamp) DESC
    LIMIT 1
),
short_term_raw AS (
    SELECT
        start_date::date AS month,
        country_name,
        SUM(metric_value) AS total_mmtpa
    FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa
    WHERE market_outlook = (SELECT market_outlook FROM latest_short_term_market)
      AND publication_date::timestamp = (
          SELECT publication_timestamp FROM latest_short_term_market
      )
      AND release_type = 'Short Term Outlook'
      AND direction = 'Export'
      AND measured_at = 'Exit'
      AND metric_name = 'Flow'
    GROUP BY start_date::date, country_name
    HAVING SUM(metric_value) > 0
),
short_term AS (
    SELECT
        raw.month,
        COALESCE(mapping.country_name, raw.country_name) AS country_name,
        SUM(raw.total_mmtpa) AS total_mmtpa
    FROM short_term_raw raw
    LEFT JOIN country_mapping mapping
        ON UPPER(TRIM(raw.country_name)) = mapping.raw_country_key
    GROUP BY raw.month, COALESCE(mapping.country_name, raw.country_name)
),
short_term_max_month AS (
    SELECT MAX(month) AS max_month
    FROM short_term
),
long_term_raw AS (
    SELECT
        start_date::date AS month,
        country_name,
        SUM(metric_value) AS total_mmtpa
    FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa
    WHERE market_outlook = (SELECT market_outlook FROM latest_long_term_market)
      AND publication_date::timestamp = (
          SELECT publication_timestamp FROM latest_long_term_market
      )
      AND release_type = 'Long Term Outlook'
      AND direction = 'Export'
      AND measured_at = 'Exit'
      AND metric_name = 'Flow'
    GROUP BY start_date::date, country_name
    HAVING SUM(metric_value) > 0
),
long_term AS (
    SELECT
        raw.month,
        COALESCE(mapping.country_name, raw.country_name) AS country_name,
        SUM(raw.total_mmtpa) AS total_mmtpa
    FROM long_term_raw raw
    LEFT JOIN country_mapping mapping
        ON UPPER(TRIM(raw.country_name)) = mapping.raw_country_key
    WHERE raw.month > COALESCE(
        (SELECT max_month FROM short_term_max_month),
        DATE '1900-01-01'
    )
    GROUP BY raw.month, COALESCE(mapping.country_name, raw.country_name)
)
SELECT month, country_name, total_mmtpa
FROM short_term
UNION ALL
SELECT month, country_name, total_mmtpa
FROM long_term
ORDER BY month, country_name
"""

def _build_ea_export_flow_query() -> str:
    balance_ctes, resolved_reference = build_resolved_ea_lng_balance_ctes(
        engine, DB_SCHEMA
    )
    return f"""
WITH
{COUNTRY_MAPPING_CTE},
{balance_ctes},
export_mappings AS (
    SELECT
        dataset_id,
        country,
        unit,
        frequency
    FROM {resolved_reference}
    WHERE aspect = 'exports'
      AND category_subtype = 'LNG'
      AND frequency = 'monthly'
      AND unit = 'Mt'
      AND country IS NOT NULL
      AND country <> ''
),
latest_values AS (
    SELECT
        values_table.dataset_id,
        values_table.date::date AS month,
        values_table.value
    FROM {DB_SCHEMA}.ea_values_current values_table
    WHERE values_table.dataset_id IN (SELECT dataset_id FROM export_mappings)
),
country_monthly AS (
    SELECT
        values_table.month,
        COALESCE(mapping.country_name, mappings.country) AS country_name,
        SUM(values_table.value * 12.0) AS total_mmtpa
    FROM latest_values values_table
    JOIN export_mappings mappings
        ON values_table.dataset_id = mappings.dataset_id
    LEFT JOIN country_mapping mapping
        ON UPPER(TRIM(mappings.country)) = mapping.raw_country_key
    GROUP BY values_table.month, COALESCE(mapping.country_name, mappings.country)
)
SELECT month, country_name, total_mmtpa
FROM country_monthly
ORDER BY month, country_name
"""

WOODMAC_PARAMETERIZED_EXPORT_FLOW_QUERY = f"""
WITH
{COUNTRY_MAPPING_CTE},
short_term_raw AS (
    SELECT
        start_date::date AS month,
        country_name,
        SUM(metric_value) AS total_mmtpa
    FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa
    WHERE market_outlook = :short_term_market_outlook
      AND publication_date::timestamp = CAST(:short_term_publication_timestamp AS timestamp)
      AND release_type = 'Short Term Outlook'
      AND direction = 'Export'
      AND measured_at = 'Exit'
      AND metric_name = 'Flow'
    GROUP BY start_date::date, country_name
    HAVING SUM(metric_value) > 0
),
short_term AS (
    SELECT
        raw.month,
        COALESCE(mapping.country_name, raw.country_name) AS country_name,
        SUM(raw.total_mmtpa) AS total_mmtpa
    FROM short_term_raw raw
    LEFT JOIN country_mapping mapping
        ON UPPER(TRIM(raw.country_name)) = mapping.raw_country_key
    GROUP BY raw.month, COALESCE(mapping.country_name, raw.country_name)
),
short_term_max_month AS (
    SELECT MAX(month) AS max_month
    FROM short_term
),
long_term_raw AS (
    SELECT
        start_date::date AS month,
        country_name,
        SUM(metric_value) AS total_mmtpa
    FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa
    WHERE market_outlook = :long_term_market_outlook
      AND publication_date::timestamp = CAST(:long_term_publication_timestamp AS timestamp)
      AND release_type = 'Long Term Outlook'
      AND direction = 'Export'
      AND measured_at = 'Exit'
      AND metric_name = 'Flow'
    GROUP BY start_date::date, country_name
    HAVING SUM(metric_value) > 0
),
long_term AS (
    SELECT
        raw.month,
        COALESCE(mapping.country_name, raw.country_name) AS country_name,
        SUM(raw.total_mmtpa) AS total_mmtpa
    FROM long_term_raw raw
    LEFT JOIN country_mapping mapping
        ON UPPER(TRIM(raw.country_name)) = mapping.raw_country_key
    WHERE raw.month > COALESCE(
        (SELECT max_month FROM short_term_max_month),
        DATE '1900-01-01'
    )
    GROUP BY raw.month, COALESCE(mapping.country_name, raw.country_name)
)
SELECT month, country_name, total_mmtpa
FROM short_term
UNION ALL
SELECT month, country_name, total_mmtpa
FROM long_term
ORDER BY month, country_name
"""

def _build_ea_parameterized_export_flow_query() -> str:
    balance_ctes, resolved_reference = build_resolved_ea_lng_balance_ctes(
        engine, DB_SCHEMA
    )
    return f"""
WITH
{COUNTRY_MAPPING_CTE},
{balance_ctes},
export_mappings AS (
    SELECT
        dataset_id,
        country,
        unit,
        frequency
    FROM {resolved_reference}
    WHERE aspect = 'exports'
      AND category_subtype = 'LNG'
      AND frequency = 'monthly'
      AND unit = 'Mt'
      AND country IS NOT NULL
      AND country <> ''
),
latest_values AS (
    SELECT
        values_table.dataset_id,
        values_table.date::date AS month,
        values_table.value
    FROM {ea_values_at_run_source_sql(
        schema=DB_SCHEMA,
        dataset_ids_sql="export_mappings",
    )} values_table
),
country_monthly AS (
    SELECT
        values_table.month,
        COALESCE(mapping.country_name, mappings.country) AS country_name,
        SUM(values_table.value * 12.0) AS total_mmtpa
    FROM latest_values values_table
    JOIN export_mappings mappings
        ON values_table.dataset_id = mappings.dataset_id
    LEFT JOIN country_mapping mapping
        ON UPPER(TRIM(mappings.country)) = mapping.raw_country_key
    GROUP BY values_table.month, COALESCE(mapping.country_name, mappings.country)
)
SELECT month, country_name, total_mmtpa
FROM country_monthly
ORDER BY month, country_name
"""

WOODMAC_EXPORT_FLOW_METADATA_QUERY = f"""
WITH latest_short_term_market AS (
    SELECT
        market_outlook,
        MAX(publication_date::timestamp) AS publication_timestamp
    FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa
    WHERE release_type = 'Short Term Outlook'
      AND direction = 'Export'
      AND measured_at = 'Exit'
      AND metric_name = 'Flow'
    GROUP BY market_outlook
    ORDER BY TO_DATE(
        (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[1]
        || ' ' ||
        (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[2],
        'Month YYYY'
    ) DESC NULLS LAST,
    MAX(publication_date::timestamp) DESC
    LIMIT 1
),
latest_long_term_market AS (
    SELECT
        market_outlook,
        MAX(publication_date::timestamp) AS publication_timestamp
    FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa
    WHERE release_type = 'Long Term Outlook'
      AND direction = 'Export'
      AND measured_at = 'Exit'
      AND metric_name = 'Flow'
    GROUP BY market_outlook
    ORDER BY TO_DATE(
        (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[1]
        || ' ' ||
        (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[2],
        'Month YYYY'
    ) DESC NULLS LAST,
    MAX(publication_date::timestamp) DESC
    LIMIT 1
)
SELECT
    st.market_outlook AS short_term_market_outlook,
    st.publication_timestamp AS short_term_publication_timestamp,
    lt.market_outlook AS long_term_market_outlook,
    lt.publication_timestamp AS long_term_publication_timestamp
FROM latest_short_term_market st
CROSS JOIN latest_long_term_market lt
"""

WOODMAC_PUBLICATION_OPTIONS_QUERY = f"""
SELECT
    CASE
        WHEN release_type = 'Short Term Outlook' THEN 'short_term'
        ELSE 'long_term'
    END AS publication_kind,
    market_outlook,
    MAX(publication_date::timestamp) AS publication_timestamp
FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa
WHERE release_type IN ('Short Term Outlook', 'Long Term Outlook')
  AND direction = 'Export'
  AND measured_at = 'Exit'
  AND metric_name = 'Flow'
GROUP BY release_type, market_outlook
ORDER BY publication_kind,
    TO_DATE(
        (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[1]
        || ' ' ||
        (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[2],
        'Month YYYY'
    ) DESC NULLS LAST,
    MAX(publication_date::timestamp) DESC
"""

def _sanitize_raw_export_flow(df: pd.DataFrame) -> pd.DataFrame:
    return sanitize_raw_flow_data(df)


def fetch_woodmac_export_flow_raw_data() -> pd.DataFrame:
    with engine.connect() as connection:
        woodmac_df = pd.read_sql_query(WOODMAC_EXPORT_FLOW_QUERY, connection)

    return _sanitize_raw_export_flow(woodmac_df)


def fetch_ea_export_flow_raw_data(
    *,
    ea_as_of_run_id: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if ea_as_of_run_id is not None:
        return fetch_ea_export_flow_raw_data_at_run(
            ea_as_of_run_id,
            start_date=start_date,
            end_date=end_date,
        )
    with engine.connect() as connection:
        ea_df = pd.read_sql_query(_build_ea_export_flow_query(), connection)

    return _sanitize_raw_export_flow(ea_df)


def fetch_woodmac_export_flow_raw_data_for_publications(
    short_term_market_outlook: str,
    short_term_publication_timestamp: str,
    long_term_market_outlook: str,
    long_term_publication_timestamp: str,
) -> pd.DataFrame:
    params = {
        "short_term_market_outlook": short_term_market_outlook,
        "short_term_publication_timestamp": short_term_publication_timestamp,
        "long_term_market_outlook": long_term_market_outlook,
        "long_term_publication_timestamp": long_term_publication_timestamp,
    }

    with engine.connect() as connection:
        woodmac_df = pd.read_sql_query(
            text(WOODMAC_PARAMETERIZED_EXPORT_FLOW_QUERY),
            connection,
            params=params,
        )

    return _sanitize_raw_export_flow(woodmac_df)


def fetch_ea_export_flow_raw_data_at_run(
    ea_as_of_run_id: int,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    run_id = normalize_ea_run_id(ea_as_of_run_id)
    with engine.connect() as connection:
        ea_df = pd.read_sql_query(
            text(_build_ea_parameterized_export_flow_query()),
            connection,
            params={
                "ea_as_of_run_id": run_id,
                "ea_start_date": start_date,
                "ea_end_date": end_date,
            },
        )

    return _sanitize_raw_export_flow(ea_df)


def fetch_woodmac_export_flow_metadata() -> dict[str, str | None]:
    with engine.connect() as connection:
        metadata_df = pd.read_sql_query(WOODMAC_EXPORT_FLOW_METADATA_QUERY, connection)

    if metadata_df.empty:
        return {}

    return build_woodmac_flow_metadata(metadata_df)


def fetch_ea_export_flow_metadata() -> dict[str, str | None]:
    current_run = fetch_current_ea_run(engine, schema=DB_SCHEMA)
    return {
        "run_id": current_run["run_id"],
        "snapshot_at": current_run["snapshot_at"],
        "upload_timestamp_utc": current_run["snapshot_at"],
    }


def fetch_woodmac_publication_options() -> dict[str, list[dict[str, str | None]]]:
    with engine.connect() as connection:
        options_df = pd.read_sql_query(WOODMAC_PUBLICATION_OPTIONS_QUERY, connection)

    if options_df.empty:
        return {"short_term": [], "long_term": []}

    return build_woodmac_publication_options(options_df)


def fetch_ea_upload_options(*, max_run_id: int | None = None) -> list[dict]:
    current_run = (
        fetch_current_ea_run(engine, schema=DB_SCHEMA)
        if max_run_id is None
        else {"run_id": normalize_ea_run_id(max_run_id)}
    )
    return fetch_ea_comparison_runs(
        engine,
        max_run_id=current_run["run_id"],
        schema=DB_SCHEMA,
    )


# Temporary compatibility name for callers being migrated in the same release.
fetch_ea_export_flow_raw_data_for_upload = fetch_ea_export_flow_raw_data_at_run


def get_available_countries(dataframes: list[pd.DataFrame]) -> list[str]:
    return resolve_available_countries(dataframes)


def default_selected_countries(available_countries: list[str]) -> list[str]:
    return resolve_default_selected_countries(
        available_countries,
        DEFAULT_SELECTED_COUNTRIES,
    )


def build_export_flow_matrix(
    raw_df: pd.DataFrame,
    selected_countries: list[str] | None,
    other_countries_mode: str = "rest_of_world",
) -> pd.DataFrame:
    sanitized_df = _sanitize_raw_export_flow(raw_df)
    if sanitized_df.empty:
        return pd.DataFrame(columns=["Month", "Total MMTPA"])

    selected_countries = [
        country for country in (selected_countries or []) if country
    ]

    working_df = sanitized_df.copy()

    if selected_countries:
        if other_countries_mode == "exclude":
            working_df = working_df[working_df["country_name"].isin(selected_countries)].copy()
            working_df["country_bucket"] = working_df["country_name"]
            visible_columns = selected_countries
        else:
            working_df["country_bucket"] = working_df["country_name"].where(
                working_df["country_name"].isin(selected_countries),
                "Rest of the World",
            )
            visible_columns = selected_countries + ["Rest of the World"]
    else:
        if other_countries_mode == "exclude":
            return pd.DataFrame(columns=["Month", "Total MMTPA"])

        working_df["country_bucket"] = "Rest of the World"
        visible_columns = ["Rest of the World"]

    bucketed_df = (
        working_df.groupby(["month", "country_bucket"], as_index=False)["total_mmtpa"]
        .sum()
        .sort_values(["month", "country_bucket"])
    )

    pivot_df = (
        bucketed_df.pivot(index="month", columns="country_bucket", values="total_mmtpa")
        .fillna(0.0)
        .sort_index()
    )

    month_index = pd.date_range(
        start=sanitized_df["month"].min(),
        end=sanitized_df["month"].max(),
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
