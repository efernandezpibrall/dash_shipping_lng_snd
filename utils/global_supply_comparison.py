"""Build a like-for-like monthly global LNG supply comparison."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
import math

import pandas as pd
from sqlalchemy import text

LOGGER = logging.getLogger(__name__)
DB_SCHEMA = "at_lng"
EA_GLOBAL_SUPPLY_DATASET_ID = "15666"
PLATTS_SHORT_TERM_DATASET_KEY = "lnga_short_term_supply"
PLATTS_LONG_TERM_DATASET_KEY = "lnga_long_term_supply"
PLATTS_LONG_TERM_SCENARIO = "Inflections"
COMPARISON_END_MONTH = pd.Timestamp("2031-12-01")
SUPPLY_SOURCE_ORDER = (
    "Our ramp forecast",
    "Energy Aspects",
    "Platts",
    "WoodMac",
)
TIME_VIEW_CONFIG = {
    "monthly": {
        "label": "Month",
        "axis_label": "Monthly LNG supply",
        "unit": "Mt/month",
        "months": 1,
    },
    "quarterly": {
        "label": "Quarter",
        "axis_label": "Quarterly LNG supply",
        "unit": "Mt/quarter",
        "months": 3,
    },
    "season": {
        "label": "Season",
        "axis_label": "Seasonal LNG supply",
        "unit": "Mt/season",
        "months": 6,
    },
    "yearly": {
        "label": "Year",
        "axis_label": "Annual LNG supply",
        "unit": "Mt/year",
        "months": 12,
    },
}
MONTH_YEAR_REGEX = (
    "(January|February|March|April|May|June|July|August|September|October|"
    "November|December)\\s+(\\d{4})"
)


EA_GLOBAL_SUPPLY_CURRENT_QUERY = text(f"""
    SELECT source.date::date AS month,
           source.value::double precision AS monthly_mt,
           source.upload_timestamp_utc
    FROM {DB_SCHEMA}.ea_values_current source
    WHERE source.dataset_id = :dataset_id
      AND source.date >= CAST(:start_month AS timestamp)
      AND source.date < CAST(:end_month AS timestamp) + INTERVAL '1 month'
    ORDER BY source.date
""")


PLATTS_SHORT_TERM_SUPPLY_QUERY = text(f"""
    WITH latest_upload AS (
        SELECT MAX(upload_timestamp_utc) AS upload_timestamp_utc
        FROM {DB_SCHEMA}.platts_lng_supply_demand_forecasts
        WHERE dataset_key = :dataset_key
    ),
    latest_vintage AS (
        SELECT MAX(vintage_date) AS vintage_date
        FROM {DB_SCHEMA}.platts_lng_supply_demand_forecasts source
        CROSS JOIN latest_upload
        WHERE source.dataset_key = :dataset_key
          AND source.upload_timestamp_utc = latest_upload.upload_timestamp_utc
    )
    SELECT source.period_start AS month,
           SUM(source.value)::double precision AS monthly_mt,
           MAX(source.upload_timestamp_utc) AS upload_timestamp_utc,
           MAX(source.vintage_date) AS vintage_date
    FROM {DB_SCHEMA}.platts_lng_supply_demand_forecasts source
    CROSS JOIN latest_upload
    CROSS JOIN latest_vintage
    WHERE source.dataset_key = :dataset_key
      AND source.upload_timestamp_utc = latest_upload.upload_timestamp_utc
      AND source.vintage_date = latest_vintage.vintage_date
      AND source.metric = 'lng_supply_forecast'
      AND source.unit = 'MMt'
      AND source.period_start BETWEEN :start_month AND :end_month
    GROUP BY source.period_start
    ORDER BY source.period_start
""")


PLATTS_LONG_TERM_2031_QUERY = text(f"""
    WITH latest_upload AS (
        SELECT MAX(upload_timestamp_utc) AS upload_timestamp_utc
        FROM {DB_SCHEMA}.platts_lng_supply_demand_forecasts
        WHERE dataset_key = :dataset_key
    ),
    latest_vintage AS (
        SELECT MAX(vintage_date) AS vintage_date
        FROM {DB_SCHEMA}.platts_lng_supply_demand_forecasts source
        CROSS JOIN latest_upload
        WHERE source.dataset_key = :dataset_key
          AND source.upload_timestamp_utc = latest_upload.upload_timestamp_utc
          AND source.scenario = :scenario
    )
    SELECT SUM(source.value)::double precision AS annual_mt,
           MAX(source.upload_timestamp_utc) AS upload_timestamp_utc,
           MAX(source.vintage_date) AS vintage_date
    FROM {DB_SCHEMA}.platts_lng_supply_demand_forecasts source
    CROSS JOIN latest_upload
    CROSS JOIN latest_vintage
    WHERE source.dataset_key = :dataset_key
      AND source.upload_timestamp_utc = latest_upload.upload_timestamp_utc
      AND source.vintage_date = latest_vintage.vintage_date
      AND source.scenario = :scenario
      AND source.metric = 'lng_supply_forecast'
      AND source.unit = 'MMt'
      AND source.period_year = 2031
""")


RAMP_GLOBAL_SUPPLY_QUERY = text(f"""
    SELECT forecast_month AS month,
           SUM(adjusted_output)::double precision / 12.0 AS monthly_mt
    FROM {DB_SCHEMA}.fundamentals_terminal_ramp_forecast_monthly
    WHERE run_id = :run_id
      AND forecast_month BETWEEN :start_month AND :end_month
    GROUP BY forecast_month
    ORDER BY forecast_month
""")


WOODMAC_GLOBAL_SUPPLY_QUERY = text(f"""
    WITH latest_short_term_market AS (
        SELECT market_outlook,
               MAX(publication_date::timestamp) AS publication_timestamp
        FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa
        WHERE release_type = 'Short Term Outlook'
          AND direction = 'Export'
          AND measured_at = 'Exit'
          AND metric_name = 'Flow'
        GROUP BY market_outlook
        ORDER BY TO_DATE(
            (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[1] || ' ' ||
            (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[2],
            'Month YYYY'
        ) DESC NULLS LAST,
        MAX(publication_date::timestamp) DESC
        LIMIT 1
    ),
    latest_long_term_market AS (
        SELECT market_outlook,
               MAX(publication_date::timestamp) AS publication_timestamp
        FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa
        WHERE release_type = 'Long Term Outlook'
          AND direction = 'Export'
          AND measured_at = 'Exit'
          AND metric_name = 'Flow'
        GROUP BY market_outlook
        ORDER BY TO_DATE(
            (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[1] || ' ' ||
            (regexp_match(market_outlook, '{MONTH_YEAR_REGEX}'))[2],
            'Month YYYY'
        ) DESC NULLS LAST,
        MAX(publication_date::timestamp) DESC
        LIMIT 1
    ),
    short_term AS (
        SELECT source.start_date::date AS month,
               SUM(source.metric_value)::double precision / 12.0 AS monthly_mt
        FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa source
        CROSS JOIN latest_short_term_market latest
        WHERE source.market_outlook = latest.market_outlook
          AND source.publication_date::timestamp = latest.publication_timestamp
          AND source.release_type = 'Short Term Outlook'
          AND source.direction = 'Export'
          AND source.measured_at = 'Exit'
          AND source.metric_name = 'Flow'
          AND source.start_date::date BETWEEN :start_month AND :end_month
        GROUP BY source.start_date::date
    ),
    long_term AS (
        SELECT source.start_date::date AS month,
               SUM(source.metric_value)::double precision / 12.0 AS monthly_mt
        FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa source
        CROSS JOIN latest_long_term_market latest
        WHERE source.market_outlook = latest.market_outlook
          AND source.publication_date::timestamp = latest.publication_timestamp
          AND source.release_type = 'Long Term Outlook'
          AND source.direction = 'Export'
          AND source.measured_at = 'Exit'
          AND source.metric_name = 'Flow'
          AND source.start_date::date BETWEEN :start_month AND :end_month
          AND source.start_date::date > COALESCE(
              (SELECT MAX(month) FROM short_term), DATE '1900-01-01'
          )
        GROUP BY source.start_date::date
    ),
    combined AS (
        SELECT month, monthly_mt FROM short_term
        UNION ALL
        SELECT month, monthly_mt FROM long_term
    )
    SELECT combined.month,
           combined.monthly_mt,
           short_market.market_outlook AS short_term_market_outlook,
           short_market.publication_timestamp AS short_term_publication_timestamp,
           long_market.market_outlook AS long_term_market_outlook,
           long_market.publication_timestamp AS long_term_publication_timestamp
    FROM combined
    CROSS JOIN latest_short_term_market short_market
    CROSS JOIN latest_long_term_market long_market
    ORDER BY combined.month
""")


def comparison_month_bounds(as_of_month=None) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return five complete historical years, current forecast month, and Dec-2031."""
    if as_of_month is None:
        as_of_month = datetime.now(timezone.utc)
    forecast_start = pd.Timestamp(as_of_month)
    if forecast_start.tzinfo is not None:
        forecast_start = forecast_start.tz_convert("UTC").tz_localize(None)
    forecast_start = forecast_start.to_period("M").to_timestamp()
    window_start = forecast_start - pd.DateOffset(years=5)
    return window_start, forecast_start, COMPARISON_END_MONTH


def normalize_supply_time_view(value: str | None) -> str:
    """Normalize UI aliases onto the four supported supply time views."""
    normalized = str(value or "monthly").strip().lower()
    aliases = {
        "month": "monthly",
        "quarter": "quarterly",
        "seasonal": "season",
        "seasonally": "season",
        "year": "yearly",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in TIME_VIEW_CONFIG else "monthly"


def _supply_period_columns(months: pd.Series, time_view: str) -> pd.DataFrame:
    normalized_months = pd.to_datetime(months, errors="coerce").dt.to_period("M").dt.to_timestamp()
    result = pd.DataFrame(index=months.index)

    if time_view == "monthly":
        result["period_start"] = normalized_months
        result["period_label"] = normalized_months.dt.strftime("%b %Y")
    elif time_view == "quarterly":
        result["period_start"] = normalized_months.dt.to_period("Q").dt.start_time
        result["period_label"] = (
            result["period_start"].dt.year.astype(str)
            + " Q"
            + result["period_start"].dt.quarter.astype(str)
        )
    elif time_view == "yearly":
        result["period_start"] = normalized_months.dt.to_period("Y").dt.start_time
        result["period_label"] = result["period_start"].dt.year.astype(str)
    else:
        is_summer = normalized_months.dt.month.between(4, 9)
        season_year = normalized_months.dt.year - normalized_months.dt.month.isin([1, 2, 3]).astype(int)
        season_month = pd.Series(10, index=months.index)
        season_month.loc[is_summer] = 4
        result["period_start"] = pd.to_datetime(
            {"year": season_year, "month": season_month, "day": 1},
            errors="coerce",
        )
        result["period_label"] = "Winter " + season_year.astype(str) + "/" + (
            season_year + 1
        ).astype(str).str[-2:]
        result.loc[is_summer, "period_label"] = "Summer " + season_year.loc[is_summer].astype(str)

    return result


def aggregate_global_supply_comparison(
    comparison_df: pd.DataFrame,
    time_view: str = "monthly",
) -> pd.DataFrame:
    """Aggregate physical monthly supply into complete comparable periods."""
    output_columns = ["period_start", "period_label", "source", "supply_mt"]
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame(columns=output_columns)

    time_view = normalize_supply_time_view(time_view)
    expected_month_count = TIME_VIEW_CONFIG[time_view]["months"]
    working_df = comparison_df[["month", "source", "monthly_mt"]].copy()
    working_df["month"] = pd.to_datetime(working_df["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    working_df["monthly_mt"] = pd.to_numeric(working_df["monthly_mt"], errors="coerce")
    working_df = working_df.dropna(subset=["month", "source", "monthly_mt"])
    working_df = working_df[working_df["monthly_mt"].map(math.isfinite)]
    if working_df.empty:
        return pd.DataFrame(columns=output_columns)

    period_columns = _supply_period_columns(working_df["month"], time_view)
    working_df = pd.concat([working_df, period_columns], axis=1)

    rows = []
    for (source, period_start, period_label), period_df in working_df.groupby(
        ["source", "period_start", "period_label"],
        sort=False,
    ):
        expected_months = set(
            pd.date_range(period_start, periods=expected_month_count, freq="MS")
        )
        actual_months = set(period_df["month"].drop_duplicates())
        if actual_months != expected_months:
            continue
        rows.append(
            {
                "period_start": period_start,
                "period_label": period_label,
                "source": source,
                "supply_mt": period_df.groupby("month")["monthly_mt"].sum().sum(),
            }
        )

    if not rows:
        return pd.DataFrame(columns=output_columns)
    return (
        pd.DataFrame(rows, columns=output_columns)
        .sort_values(["period_start", "source"])
        .reset_index(drop=True)
    )


def _series_frame(raw_df: pd.DataFrame, source: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["month", "source", "monthly_mt"])
    frame = raw_df[["month", "monthly_mt"]].copy()
    frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
    frame["monthly_mt"] = pd.to_numeric(frame["monthly_mt"], errors="coerce")
    frame = frame.dropna(subset=["month", "monthly_mt"])
    frame = frame[frame["monthly_mt"].map(math.isfinite)]
    frame["source"] = source
    return frame[["month", "source", "monthly_mt"]].sort_values("month")


def _coverage(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if frame.empty:
        return None, None
    return frame["month"].min().strftime("%b %Y"), frame["month"].max().strftime("%b %Y")


def _timestamp_label(value) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp.strftime("%Y-%m-%d")


def _fetch_ramp(engine, run_id: int, start_month, end_month):
    with engine.connect() as connection:
        return pd.read_sql_query(
            RAMP_GLOBAL_SUPPLY_QUERY,
            connection,
            params={"run_id": int(run_id), "start_month": start_month, "end_month": end_month},
        )


def _fetch_ea(engine, start_month, end_month):
    with engine.connect() as connection:
        return pd.read_sql_query(
            EA_GLOBAL_SUPPLY_CURRENT_QUERY,
            connection,
            params={
                "dataset_id": EA_GLOBAL_SUPPLY_DATASET_ID,
                "start_month": start_month,
                "end_month": end_month,
            },
        )


def _fetch_platts(engine, start_month, end_month):
    with engine.connect() as connection:
        short_term = pd.read_sql_query(
            PLATTS_SHORT_TERM_SUPPLY_QUERY,
            connection,
            params={
                "dataset_key": PLATTS_SHORT_TERM_DATASET_KEY,
                "start_month": start_month,
                "end_month": min(end_month, pd.Timestamp("2030-12-01")),
            },
        )
        long_term = pd.read_sql_query(
            PLATTS_LONG_TERM_2031_QUERY,
            connection,
            params={
                "dataset_key": PLATTS_LONG_TERM_DATASET_KEY,
                "scenario": PLATTS_LONG_TERM_SCENARIO,
            },
        )

    long_term_months = pd.DataFrame(columns=["month", "monthly_mt"])
    if not long_term.empty and pd.notna(long_term.iloc[0].get("annual_mt")):
        long_term_months = pd.DataFrame(
            {
                "month": pd.date_range("2031-01-01", "2031-12-01", freq="MS"),
                "monthly_mt": float(long_term.iloc[0]["annual_mt"]) / 12.0,
            }
        )
    return short_term, long_term, long_term_months


def _fetch_woodmac(engine, start_month, end_month):
    with engine.connect() as connection:
        raw_df = pd.read_sql_query(
            WOODMAC_GLOBAL_SUPPLY_QUERY,
            connection,
            params={"start_month": start_month, "end_month": end_month},
        )
    if raw_df.empty:
        return pd.DataFrame(columns=["month", "monthly_mt"]), {}
    row = raw_df.iloc[0]
    metadata = {
        "short_term_market_outlook": row.get("short_term_market_outlook"),
        "short_term_publication_timestamp": row.get("short_term_publication_timestamp"),
        "long_term_market_outlook": row.get("long_term_market_outlook"),
        "long_term_publication_timestamp": row.get("long_term_publication_timestamp"),
    }
    return raw_df[["month", "monthly_mt"]], metadata


def fetch_global_supply_comparison(
    engine,
    run_id: int,
    *,
    as_of_month=None,
) -> tuple[pd.DataFrame, dict]:
    """Return ramp, EA, Platts and WoodMac global supply in monthly Mt."""
    window_start, forecast_start, window_end = comparison_month_bounds(as_of_month)
    frames: list[pd.DataFrame] = []
    metadata = {
        "window_start": window_start.isoformat(),
        "forecast_start": forecast_start.isoformat(),
        "window_end": window_end.isoformat(),
        "sources": {},
        "warnings": [],
    }

    def add_source(source, raw_df, detail):
        frame = _series_frame(raw_df, source)
        frames.append(frame)
        first_month, last_month = _coverage(frame)
        metadata["sources"][source] = {
            "first_month": first_month,
            "last_month": last_month,
            "detail": detail,
        }

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="supply-comparison") as executor:
        ramp_future = executor.submit(
            _fetch_ramp, engine, run_id, window_start, window_end
        )
        ea_future = executor.submit(_fetch_ea, engine, window_start, window_end)
        platts_future = executor.submit(_fetch_platts, engine, window_start, window_end)
        woodmac_future = executor.submit(
            _fetch_woodmac, engine, window_start, window_end
        )

        try:
            add_source(
                "Our ramp forecast",
                ramp_future.result(),
                f"run {int(run_id)}; annualized train output converted to monthly Mt",
            )
        except Exception as exc:
            LOGGER.exception("Ramp supply comparison load failed")
            metadata["warnings"].append(f"Ramp forecast unavailable: {exc}")

        try:
            ea_df = ea_future.result()
            ea_upload = _timestamp_label(
                ea_df["upload_timestamp_utc"].max()
            ) if not ea_df.empty else None
            add_source(
                "Energy Aspects",
                ea_df,
                f"global exports incl. unplanned outages; upload {ea_upload or 'unknown'}",
            )
        except Exception as exc:
            LOGGER.exception("Energy Aspects supply comparison load failed")
            metadata["warnings"].append(f"Energy Aspects unavailable: {exc}")

        try:
            platts_short, platts_long, platts_2031 = platts_future.result()
            short_upload = _timestamp_label(
                platts_short["upload_timestamp_utc"].max()
            ) if not platts_short.empty else None
            short_vintage = _timestamp_label(
                platts_short["vintage_date"].max()
            ) if not platts_short.empty else None
            long_vintage = _timestamp_label(
                platts_long["vintage_date"].max()
            ) if not platts_long.empty else None
            platts_df = pd.concat(
                [platts_short[["month", "monthly_mt"]], platts_2031],
                ignore_index=True,
            )
            add_source(
                "Platts",
                platts_df,
                (
                    f"short-term vintage {short_vintage or 'unknown'}; "
                    f"2031 {PLATTS_LONG_TERM_SCENARIO} annual vintage "
                    f"{long_vintage or 'unknown'} divided by 12; upload {short_upload or 'unknown'}"
                ),
            )
        except Exception as exc:
            LOGGER.exception("Platts supply comparison load failed")
            metadata["warnings"].append(f"Platts unavailable: {exc}")

        try:
            woodmac_df, woodmac_metadata = woodmac_future.result()
            add_source(
                "WoodMac",
                woodmac_df,
                (
                    f"{woodmac_metadata.get('short_term_market_outlook') or 'latest short-term'} + "
                    f"{woodmac_metadata.get('long_term_market_outlook') or 'latest long-term'}"
                ),
            )
        except Exception as exc:
            LOGGER.exception("WoodMac supply comparison load failed")
            metadata["warnings"].append(f"WoodMac unavailable: {exc}")

    comparison_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if comparison_df.empty:
        comparison_df = pd.DataFrame(columns=["month", "source", "monthly_mt"])
    else:
        comparison_df = comparison_df[
            comparison_df["month"].between(window_start, window_end)
        ].sort_values(["month", "source"]).reset_index(drop=True)
    return comparison_df, metadata
