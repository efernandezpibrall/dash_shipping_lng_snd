"""Data contracts for the LNG Physical Snapshot page."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import logging
import time
from functools import lru_cache
from typing import Iterable

import pandas as pd
from sqlalchemy import bindparam, text

from utils.ea_run_interface import (
    ea_values_at_run_source_sql,
    fetch_current_ea_run,
)
from utils.import_flow_data import DB_SCHEMA, engine


LOGGER = logging.getLogger(__name__)

DISPLAY_YEARS = tuple(range(2026, 2031))
DISPLAY_COUNTRIES = (
    "China",
    "Japan",
    "South Korea",
    "Thailand",
    "India",
    "Egypt",
    "Turkey",
)
PROVIDER_ORDER = ("Energy Aspects", "WoodMac", "Platts")
DEMAND_COLUMNS = (
    "country_name",
    "provider",
    "year",
    "annual_mmt",
    "source_vintage",
    "source_upload",
    "release_type",
    "complete_year",
)
CACHE_TTL_SECONDS = 300


EA_VALUES_SOURCE = ea_values_at_run_source_sql(
    schema=DB_SCHEMA,
    dataset_ids_sql="provider_mapping",
)

EA_DEMAND_QUERY = text(
    f"""
    WITH provider_mapping AS MATERIALIZED (
        SELECT DISTINCT
            datasets.dataset_id::text AS dataset_id,
            country_mapping.country_name,
            datasets.unit
        FROM {DB_SCHEMA}.fundamentals_ea_lng_balance_datasets datasets
        JOIN {DB_SCHEMA}.mappings_country country_mapping
          ON country_mapping.country = datasets.country
        WHERE datasets.aspect = 'imports'
          AND country_mapping.country_name = ANY(:countries)
    ),
    latest_values AS (
        SELECT
            values_table.dataset_id,
            values_table.date::date AS month,
            values_table.value,
            values_table.upload_timestamp_utc
        FROM {EA_VALUES_SOURCE} values_table
    )
    SELECT
        provider_mapping.country_name,
        'Energy Aspects' AS provider,
        DATE_TRUNC('month', latest_values.month)::date AS month,
        SUM(
            CASE
                WHEN provider_mapping.unit IN ('Mt', 'MMt')
                    THEN latest_values.value
                WHEN provider_mapping.unit = 'bcm'
                    THEN latest_values.value / 1.36
                ELSE NULL
            END
        )::double precision AS monthly_mmt,
        CAST(:ea_source_vintage AS timestamptz) AS source_vintage,
        MAX(latest_values.upload_timestamp_utc) AS source_upload,
        'Forecast' AS release_type
    FROM latest_values
    JOIN provider_mapping USING (dataset_id)
    WHERE latest_values.month >= DATE '2026-01-01'
      AND latest_values.month < DATE '2031-01-01'
    GROUP BY
        provider_mapping.country_name,
        DATE_TRUNC('month', latest_values.month)
    ORDER BY provider_mapping.country_name, month
    """
)

WOODMAC_DEMAND_QUERY = text(
    f"""
    WITH candidate_headers AS MATERIALIZED (
        SELECT
            source.release_type,
            source.market_outlook,
            source.publication_date::timestamp AS publication_timestamp,
            source.upload_timestamp_utc,
            TO_DATE(
                (regexp_match(
                    source.market_outlook,
                    '(January|February|March|April|May|June|July|August|September|October|November|December)[[:space:]]+([0-9]{{4}})'
                ))[1] || ' ' ||
                (regexp_match(
                    source.market_outlook,
                    '(January|February|March|April|May|June|July|August|September|October|November|December)[[:space:]]+([0-9]{{4}})'
                ))[2],
                'Month YYYY'
            ) AS outlook_month
        FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa source
        WHERE source.release_type IN (
                'Short Term Outlook',
                'Long Term Outlook'
              )
          AND source.direction = 'Import'
          AND source.measured_at = 'Entry'
          AND source.metric_name = 'Flow'
          AND source.unit = 'mmtpa'
        GROUP BY
            source.release_type,
            source.market_outlook,
            source.publication_date::timestamp,
            source.upload_timestamp_utc
    ),
    selected_headers AS MATERIALIZED (
        SELECT *
        FROM (
            SELECT
                candidate_headers.*,
                ROW_NUMBER() OVER (
                    PARTITION BY release_type
                    ORDER BY
                        outlook_month DESC NULLS LAST,
                        publication_timestamp DESC,
                        upload_timestamp_utc DESC,
                        market_outlook DESC
                ) AS header_rank
            FROM candidate_headers
        ) ranked_headers
        WHERE header_rank = 1
    ),
    provider_monthly AS MATERIALIZED (
        SELECT
            country_mapping.country_name,
            source.start_date::date AS month,
            SUM(source.metric_value)::double precision / 12.0 AS monthly_mmt,
            selected_headers.market_outlook AS source_vintage,
            selected_headers.upload_timestamp_utc AS source_upload,
            selected_headers.release_type
        FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa source
        JOIN selected_headers
          ON selected_headers.release_type = source.release_type
         AND selected_headers.market_outlook = source.market_outlook
         AND selected_headers.publication_timestamp =
                source.publication_date::timestamp
         AND selected_headers.upload_timestamp_utc =
                source.upload_timestamp_utc
        JOIN {DB_SCHEMA}.mappings_country country_mapping
          ON country_mapping.country = source.country_name
        WHERE source.direction = 'Import'
          AND source.measured_at = 'Entry'
          AND source.metric_name = 'Flow'
          AND source.unit = 'mmtpa'
          AND country_mapping.country_name = ANY(:countries)
          AND source.start_date::date >= DATE '2026-01-01'
          AND source.start_date::date < DATE '2031-01-01'
        GROUP BY
            country_mapping.country_name,
            source.start_date::date,
            selected_headers.market_outlook,
            selected_headers.upload_timestamp_utc,
            selected_headers.release_type
    ),
    short_term_horizon AS (
        SELECT MAX(month) AS final_short_term_month
        FROM provider_monthly
        WHERE release_type = 'Short Term Outlook'
    ),
    stitched_monthly AS (
        SELECT provider_monthly.*
        FROM provider_monthly
        WHERE release_type = 'Short Term Outlook'

        UNION ALL

        SELECT provider_monthly.*
        FROM provider_monthly
        CROSS JOIN short_term_horizon
        WHERE release_type = 'Long Term Outlook'
          AND month > short_term_horizon.final_short_term_month
    )
    SELECT
        country_name,
        'WoodMac' AS provider,
        month,
        monthly_mmt,
        source_vintage,
        source_upload,
        release_type
    FROM stitched_monthly
    ORDER BY country_name, month
    """
)

PLATTS_DEMAND_QUERY = text(
    f"""
    WITH latest_header AS MATERIALIZED (
        SELECT
            upload_timestamp_utc,
            vintage_date
        FROM {DB_SCHEMA}.platts_lng_supply_demand_forecasts
        WHERE dataset_key = 'lnga_short_term_demand'
        ORDER BY
            upload_timestamp_utc DESC NULLS LAST,
            vintage_date DESC NULLS LAST
        LIMIT 1
    )
    SELECT
        country_mapping.country_name,
        'Platts' AS provider,
        source.period_start::date AS month,
        SUM(source.value)::double precision AS monthly_mmt,
        MAX(source.vintage_date) AS source_vintage,
        MAX(source.upload_timestamp_utc) AS source_upload,
        'Short Term Outlook' AS release_type
    FROM {DB_SCHEMA}.platts_lng_supply_demand_forecasts source
    CROSS JOIN latest_header
    JOIN {DB_SCHEMA}.mappings_country country_mapping
      ON country_mapping.country = source.country_or_market
    WHERE source.dataset_key = 'lnga_short_term_demand'
      AND source.metric = 'lng_demand_forecast'
      AND source.flow_type = 'demand'
      AND source.unit = 'MMt'
      AND source.upload_timestamp_utc =
            latest_header.upload_timestamp_utc
      AND source.vintage_date = latest_header.vintage_date
      AND country_mapping.country_name = ANY(:countries)
      AND source.period_start >= DATE '2026-01-01'
      AND source.period_start < DATE '2031-01-01'
    GROUP BY country_mapping.country_name, source.period_start::date
    ORDER BY country_mapping.country_name, month
    """
)

STORAGE_QUERY = text(
    f"""
    WITH latest AS MATERIALIZED (
        SELECT MAX(upload_timestamp_utc) AS upload_timestamp_utc
        FROM {DB_SCHEMA}.fundamentals_forecast_daily_balance
        WHERE balance = 'EU'
          AND balance_adjustment = :scenario
    )
    SELECT
        source.date,
        source."Storage [%]"::double precision AS storage_pct,
        source.storage::double precision / 1000.0 AS storage_bcm,
        source.upload_timestamp_utc
    FROM {DB_SCHEMA}.fundamentals_forecast_daily_balance source
    JOIN latest
      ON source.upload_timestamp_utc = latest.upload_timestamp_utc
    WHERE source.balance = 'EU'
      AND source.balance_adjustment = :scenario
      AND source.date IN :target_dates
    ORDER BY source.date
    """
).bindparams(bindparam("target_dates", expanding=True))


def _read_sql(query, db_engine, params=None):
    with db_engine.connect() as connection:
        return pd.read_sql_query(query, connection, params=params or {})


def empty_monthly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "country_name",
            "provider",
            "month",
            "monthly_mmt",
            "source_vintage",
            "source_upload",
            "release_type",
        ]
    )


def fetch_ea_monthly(db_engine=engine) -> pd.DataFrame:
    current_run = fetch_current_ea_run(db_engine, schema=DB_SCHEMA)
    return _read_sql(
        EA_DEMAND_QUERY,
        db_engine,
        params={
            "countries": list(DISPLAY_COUNTRIES),
            "ea_as_of_run_id": current_run["run_id"],
            "ea_start_date": "2026-01-01",
            "ea_end_date": "2030-12-31",
            "ea_source_vintage": current_run["snapshot_at"],
        },
    )


def fetch_woodmac_monthly(db_engine=engine) -> pd.DataFrame:
    return _read_sql(
        WOODMAC_DEMAND_QUERY,
        db_engine,
        params={"countries": list(DISPLAY_COUNTRIES)},
    )


def fetch_platts_monthly(db_engine=engine) -> pd.DataFrame:
    return _read_sql(
        PLATTS_DEMAND_QUERY,
        db_engine,
        params={"countries": list(DISPLAY_COUNTRIES)},
    )


def _join_unique(values: Iterable[object]) -> str:
    normalized = sorted(
        {
            str(value)
            for value in values
            if value is not None and not pd.isna(value) and str(value)
        }
    )
    return " | ".join(normalized)


def annualize_monthly_demand(monthly_frame: pd.DataFrame) -> pd.DataFrame:
    if monthly_frame is None or monthly_frame.empty:
        return pd.DataFrame(columns=DEMAND_COLUMNS)

    frame = monthly_frame.copy()
    frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
    frame["monthly_mmt"] = pd.to_numeric(
        frame["monthly_mmt"], errors="coerce"
    )
    frame = frame.dropna(
        subset=["country_name", "provider", "month"]
    ).copy()
    frame = frame[
        frame["country_name"].isin(DISPLAY_COUNTRIES)
        & frame["provider"].isin(PROVIDER_ORDER)
        & frame["month"].dt.year.isin(DISPLAY_YEARS)
    ]
    if frame.empty:
        return pd.DataFrame(columns=DEMAND_COLUMNS)

    frame["year"] = frame["month"].dt.year.astype(int)
    grouped = (
        frame.groupby(["country_name", "provider", "year"], as_index=False)
        .agg(
            annual_mmt=("monthly_mmt", "sum"),
            available_months=("month", "nunique"),
            numeric_months=("monthly_mmt", "count"),
            source_vintage=("source_vintage", _join_unique),
            source_upload=("source_upload", "max"),
            release_type=("release_type", _join_unique),
        )
    )
    grouped["complete_year"] = (
        grouped["available_months"].eq(12)
        & grouped["numeric_months"].eq(12)
    )
    grouped.loc[~grouped["complete_year"], "annual_mmt"] = pd.NA
    grouped["annual_mmt"] = pd.to_numeric(
        grouped["annual_mmt"], errors="coerce"
    ).round(2)
    return grouped.loc[:, DEMAND_COLUMNS]


def fetch_demand_snapshot(db_engine=engine):
    provider_loaders = (
        ("Energy Aspects", fetch_ea_monthly),
        ("WoodMac", fetch_woodmac_monthly),
        ("Platts", fetch_platts_monthly),
    )
    monthly_frames = []
    warnings = []
    with ThreadPoolExecutor(
        max_workers=3,
        thread_name_prefix="physical-snapshot-demand",
    ) as executor:
        futures = {
            provider: executor.submit(loader, db_engine)
            for provider, loader in provider_loaders
        }
        for provider, _loader in provider_loaders:
            try:
                provider_frame = futures[provider].result()
            except Exception:
                LOGGER.exception(
                    "%s physical-snapshot demand is unavailable", provider
                )
                warnings.append(f"{provider} demand is unavailable.")
                continue
            if provider_frame is None or provider_frame.empty:
                warnings.append(
                    f"{provider} returned no mapped demand data."
                )
                continue
            monthly_frames.append(provider_frame)

    if not monthly_frames:
        return pd.DataFrame(columns=DEMAND_COLUMNS), warnings
    monthly = pd.concat(monthly_frames, ignore_index=True)
    return annualize_monthly_demand(monthly), warnings


def _cache_bucket() -> int:
    return int(time.time() // CACHE_TTL_SECONDS)


@lru_cache(maxsize=32)
def _cached_demand_snapshot(cache_bucket: int, refresh_token: int):
    del cache_bucket, refresh_token
    frame, warnings = fetch_demand_snapshot(engine)
    return frame, tuple(warnings)


def get_demand_snapshot(refresh_token=0):
    frame, warnings = _cached_demand_snapshot(
        _cache_bucket(), int(refresh_token or 0)
    )
    return frame.copy(deep=True), list(warnings)


def build_demand_matrix(annual_frame: pd.DataFrame) -> list[dict]:
    annual = (
        annual_frame.copy()
        if annual_frame is not None
        else pd.DataFrame(columns=DEMAND_COLUMNS)
    )
    rows = []
    for country_name in DISPLAY_COUNTRIES:
        for provider_index, provider in enumerate(PROVIDER_ORDER):
            row = {
                "Country": country_name,
                "Provider": provider,
                "__country_group_start": provider_index == 0,
            }
            for year in DISPLAY_YEARS:
                column_id = f"{year}E"
                matching = annual[
                    annual["country_name"].eq(country_name)
                    & annual["provider"].eq(provider)
                    & annual["year"].eq(year)
                ]
                record = matching.iloc[0] if not matching.empty else None
                complete = bool(record["complete_year"]) if record is not None else False
                value = record["annual_mmt"] if record is not None else None
                row[column_id] = (
                    float(value)
                    if complete and value is not None and pd.notna(value)
                    else None
                )
                release_type = (
                    str(record["release_type"])
                    if record is not None and pd.notna(record["release_type"])
                    else ""
                )
                vintage = (
                    str(record["source_vintage"])
                    if record is not None and pd.notna(record["source_vintage"])
                    else ""
                )
                upload = (
                    pd.to_datetime(
                        record["source_upload"],
                        errors="coerce",
                        utc=True,
                    )
                    if record is not None
                    else pd.NaT
                )
                upload_text = (
                    upload.strftime("%d %b %Y %H:%M UTC")
                    if pd.notna(upload)
                    else "Unknown upload"
                )
                completeness = (
                    "Complete 12-month year"
                    if complete
                    else "Incomplete year — annual value withheld"
                )
                row[f"__{column_id}_is_lto"] = (
                    provider == "WoodMac"
                    and "Long Term Outlook" in release_type
                )
                row[f"__{column_id}_tooltip"] = (
                    f"{country_name} · {provider} · {year}\n"
                    f"{vintage or 'Current forecast'}\n"
                    f"{release_type or 'Release type unavailable'} · "
                    f"{upload_text}\n{completeness}"
                )
            rows.append(row)
    return rows


def build_provider_metadata(annual_frame: pd.DataFrame) -> list[dict]:
    annual = (
        annual_frame.copy()
        if annual_frame is not None
        else pd.DataFrame(columns=DEMAND_COLUMNS)
    )
    metadata = []
    for provider in PROVIDER_ORDER:
        provider_frame = annual[annual["provider"].eq(provider)]
        if provider_frame.empty:
            metadata.append(
                {
                    "provider": provider,
                    "vintage": "Unavailable",
                    "upload": "Unavailable",
                }
            )
            continue
        vintages = _join_unique(provider_frame["source_vintage"])
        uploads = pd.to_datetime(
            provider_frame["source_upload"],
            errors="coerce",
            utc=True,
        ).dropna()
        upload_text = (
            uploads.max().strftime("%d %b %Y %H:%M UTC")
            if not uploads.empty
            else "Unknown upload"
        )
        if provider in {"Energy Aspects", "Platts"}:
            parsed_vintages = pd.to_datetime(
                provider_frame["source_vintage"],
                errors="coerce",
                utc=True,
            ).dropna()
            if not parsed_vintages.empty:
                vintages = " | ".join(
                    timestamp.strftime("%d %b %Y %H:%M UTC")
                    for timestamp in sorted(parsed_vintages.unique())
                )
        metadata.append(
            {
                "provider": provider,
                "vintage": vintages or "Current forecast",
                "upload": upload_text,
            }
        )
    return metadata


def next_storage_endpoints(reference_date=None, count=4) -> list[dict]:
    reference = pd.Timestamp(
        reference_date if reference_date is not None else dt.date.today()
    ).normalize()
    candidates = []
    for year in range(reference.year, reference.year + count + 2):
        winter_end = pd.Timestamp(year=year, month=3, day=31)
        summer_end = pd.Timestamp(year=year, month=10, day=31)
        if winter_end >= reference:
            candidates.append(
                {
                    "date": winter_end.date(),
                    "label": f"End Winter {year - 1}/{str(year)[-2:]}",
                }
            )
        if summer_end >= reference:
            candidates.append(
                {
                    "date": summer_end.date(),
                    "label": f"End Summer {year}",
                }
            )
    return sorted(candidates, key=lambda item: item["date"])[:count]


def fetch_storage_snapshot(
    scenario: str,
    target_dates: tuple[dt.date, ...],
    db_engine=engine,
) -> pd.DataFrame:
    return _read_sql(
        STORAGE_QUERY,
        db_engine,
        params={
            "scenario": scenario,
            "target_dates": target_dates,
        },
    )


@lru_cache(maxsize=64)
def _cached_storage_snapshot(
    cache_bucket: int,
    refresh_token: int,
    scenario: str,
    target_dates: tuple[dt.date, ...],
):
    del cache_bucket, refresh_token
    return fetch_storage_snapshot(scenario, target_dates, engine)


def get_storage_snapshot(scenario, endpoints, refresh_token=0):
    target_dates = tuple(item["date"] for item in endpoints)
    frame = _cached_storage_snapshot(
        _cache_bucket(),
        int(refresh_token or 0),
        scenario,
        target_dates,
    )
    return frame.copy(deep=True)


def format_storage_records(
    storage_frame: pd.DataFrame,
    endpoints: list[dict],
    scenario: str,
) -> list[dict]:
    frame = (
        storage_frame.copy()
        if storage_frame is not None
        else pd.DataFrame()
    )
    if not frame.empty:
        frame["date"] = pd.to_datetime(
            frame["date"], errors="coerce"
        ).dt.date
        frame["storage_pct"] = pd.to_numeric(
            frame["storage_pct"], errors="coerce"
        )
        frame["storage_bcm"] = pd.to_numeric(
            frame["storage_bcm"], errors="coerce"
        )
    lookup = {
        row["date"]: row
        for row in frame.to_dict("records")
        if row.get("date") is not None
    }
    records = []
    for endpoint in endpoints:
        row = lookup.get(endpoint["date"], {})
        storage_pct = row.get("storage_pct")
        storage_bcm = row.get("storage_bcm")
        stockout = bool(
            (
                storage_pct is not None
                and pd.notna(storage_pct)
                and storage_pct < 0
            )
            or (
                storage_bcm is not None
                and pd.notna(storage_bcm)
                and storage_bcm < 0
            )
        )
        records.append(
            {
                "label": endpoint["label"],
                "date": endpoint["date"],
                "scenario": scenario,
                "storage_pct": (
                    float(storage_pct)
                    if storage_pct is not None and pd.notna(storage_pct)
                    else None
                ),
                "storage_bcm": (
                    float(storage_bcm)
                    if storage_bcm is not None and pd.notna(storage_bcm)
                    else None
                ),
                "stockout": stockout,
                "source_upload": row.get("upload_timestamp_utc"),
            }
        )
    return records


def clear_snapshot_caches() -> None:
    """Clear page caches for focused tests and process-level maintenance."""
    _cached_demand_snapshot.cache_clear()
    _cached_storage_snapshot.cache_clear()
