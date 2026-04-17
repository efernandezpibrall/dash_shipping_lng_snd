from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


SUPPORTED_TIME_GROUPS = {"monthly", "quarterly", "yearly", "season"}

TIME_GROUP_LABELS = {
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "yearly": "Yearly",
    "season": "Seasonally",
}


def normalize_time_group(value: str | None) -> str:
    normalized = str(value or "monthly").strip().lower()
    if normalized in {"season", "seasonal", "seasonally"}:
        return "season"
    if normalized not in SUPPORTED_TIME_GROUPS:
        return "monthly"
    return normalized


def infer_period_type(period: str | None, fallback: str | None = None) -> str:
    normalized_fallback = normalize_time_group(fallback)
    label = str(period or "").strip()

    if re.fullmatch(r"\d{4}-\d{2}", label):
        return "monthly"
    if re.fullmatch(r"\d{4}-Q[1-4]", label):
        return "quarterly"
    if re.fullmatch(r"\d{4}", label):
        return "yearly"
    if re.fullmatch(r"\d{4}-[SW]", label):
        return "season"
    return normalized_fallback


def get_time_period(date_value, period_type: str) -> str:
    period_type = normalize_time_group(period_type)
    timestamp = pd.to_datetime(date_value)

    if period_type == "monthly":
        return timestamp.strftime("%Y-%m")
    if period_type == "quarterly":
        return f"{timestamp.year}-Q{((timestamp.month - 1) // 3) + 1}"
    if period_type == "yearly":
        return str(timestamp.year)

    if 4 <= timestamp.month <= 9:
        return f"{timestamp.year}-S"
    if timestamp.month >= 10:
        return f"{timestamp.year}-W"
    return f"{timestamp.year - 1}-W"


def get_days_in_period(period: str, period_type: str) -> int:
    period_type = infer_period_type(period, period_type)

    if period_type == "monthly":
        year, month = map(int, str(period).split("-"))
        return pd.Period(year=year, month=month, freq="M").days_in_month

    if period_type == "quarterly":
        year_text, quarter_text = str(period).split("-Q")
        year = int(year_text)
        first_month = (int(quarter_text) - 1) * 3 + 1
        return sum(
            pd.Period(year=year, month=month, freq="M").days_in_month
            for month in range(first_month, first_month + 3)
        )

    if period_type == "yearly":
        year = int(period)
        return 366 if pd.Period(year=year, freq="Y").is_leap_year else 365

    year_text, season_code = str(period).split("-")
    year = int(year_text)
    if season_code == "S":
        months = range(4, 10)
        return sum(
            pd.Period(year=year, month=month, freq="M").days_in_month
            for month in months
        )

    current_year_days = sum(
        pd.Period(year=year, month=month, freq="M").days_in_month
        for month in range(10, 13)
    )
    next_year_days = sum(
        pd.Period(year=year + 1, month=month, freq="M").days_in_month
        for month in range(1, 4)
    )
    return current_year_days + next_year_days


def sort_period_labels(labels: Iterable[str], period_type: str) -> list[str]:
    period_type = normalize_time_group(period_type)

    def sort_key(label: str):
        label = str(label)
        if period_type == "monthly":
            year_text, month_text = label.split("-")
            return int(year_text), int(month_text)
        if period_type == "quarterly":
            year_text, quarter_text = label.split("-Q")
            return int(year_text), int(quarter_text)
        if period_type == "yearly":
            return (int(label),)

        year_text, season_code = label.split("-")
        return int(year_text), 0 if season_code == "S" else 1

    return sorted((str(label) for label in labels), key=sort_key)


def validate_complete_seasons(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    period_col: str = "time_period",
) -> pd.DataFrame:
    if df.empty:
        return df

    working_df = df.copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")
    working_df = working_df.dropna(subset=[date_col, period_col])

    valid_periods: list[str] = []
    for period, period_df in working_df.groupby(period_col):
        year_text, season_code = str(period).split("-")
        year = int(year_text)

        if season_code == "S":
            expected = {(year, month) for month in range(4, 10)}
        else:
            expected = {(year, month) for month in range(10, 13)}
            expected.update({(year + 1, month) for month in range(1, 4)})

        observed_dates = pd.to_datetime(period_df[date_col], errors="coerce")
        observed = {
            (int(year), int(month))
            for year, month in zip(
                observed_dates.dt.year.fillna(0),
                observed_dates.dt.month.fillna(0),
            )
            if year and month
        }
        if expected.issubset(observed):
            valid_periods.append(str(period))

    if not valid_periods:
        return working_df.iloc[0:0].copy()

    return working_df[working_df[period_col].isin(valid_periods)].copy()


def calculate_differences(
    df: pd.DataFrame,
    diff_type: str = "percentage",
    *,
    period_col: str = "Period",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    working_df = df.copy()
    numeric_columns = [
        column_name for column_name in working_df.columns if column_name != period_col
    ]
    numeric_columns = [
        column_name
        for column_name in numeric_columns
        if pd.api.types.is_numeric_dtype(working_df[column_name])
    ]

    for column_name in numeric_columns:
        if diff_type == "absolute":
            working_df[column_name] = working_df[column_name].diff()
        else:
            working_df[column_name] = working_df[column_name].pct_change() * 100.0

    return working_df.iloc[1:].round(2)


def align_frames_on_period(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    *,
    period_col: str = "Period",
) -> pd.DataFrame:
    if reference_df.empty:
        return comparison_df.copy()

    if comparison_df.empty:
        result_df = pd.DataFrame({period_col: reference_df[period_col].tolist()})
        for column_name in reference_df.columns:
            if column_name == period_col:
                continue
            result_df[column_name] = np.nan
        return result_df

    working_df = comparison_df.copy()
    for column_name in reference_df.columns:
        if column_name not in working_df.columns:
            working_df[column_name] = np.nan

    ordered_columns = list(reference_df.columns)
    working_df = working_df[ordered_columns]
    working_df = (
        working_df.set_index(period_col)
        .reindex(reference_df[period_col].tolist())
        .reset_index()
    )
    working_df.columns = ordered_columns
    return working_df


def group_numeric_frame_by_period(
    df: pd.DataFrame,
    period_type: str,
    *,
    date_col: str = "Date",
    preserve_first: Iterable[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    preserve_first = list(preserve_first or [])
    working_df = df.copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")
    working_df = working_df.dropna(subset=[date_col])
    working_df["__period"] = working_df[date_col].apply(
        lambda value: get_time_period(value, period_type)
    )

    aggregation_map: dict[str, str] = {}
    for column_name in working_df.columns:
        if column_name in {date_col, "__period"}:
            continue
        if column_name in preserve_first:
            aggregation_map[column_name] = "first"
        elif pd.api.types.is_numeric_dtype(working_df[column_name]):
            aggregation_map[column_name] = "sum"

    grouped_df = working_df.groupby("__period", as_index=False).agg(aggregation_map)
    grouped_df = grouped_df.rename(columns={"__period": date_col})
    ordered_periods = sort_period_labels(grouped_df[date_col].tolist(), period_type)
    grouped_df[date_col] = pd.Categorical(
        grouped_df[date_col], categories=ordered_periods, ordered=True
    )
    grouped_df = grouped_df.sort_values(date_col).reset_index(drop=True)
    grouped_df[date_col] = grouped_df[date_col].astype(str)

    numeric_columns = grouped_df.select_dtypes(include=["number"]).columns
    grouped_df[numeric_columns] = grouped_df[numeric_columns].round(2)

    return grouped_df


def annualized_mmtpa_to_monthly_mt(values) -> pd.Series:
    series = pd.Series(values, copy=False)
    return pd.to_numeric(series, errors="coerce").fillna(0.0) / 12.0


def annualized_mmtpa_to_monthly_bcm(values) -> pd.Series:
    return annualized_mmtpa_to_monthly_mt(values) * 1.36


def convert_periodized_bcm_frame(
    df: pd.DataFrame,
    to_unit: str,
    period_type: str,
    *,
    period_col: str = "Period",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    working_df = df.copy()
    target_unit = str(to_unit or "bcm").strip().lower()
    numeric_columns = [
        column_name
        for column_name in working_df.columns
        if column_name != period_col
        and pd.api.types.is_numeric_dtype(working_df[column_name])
    ]

    if target_unit == "bcm":
        return working_df.round(2)

    if target_unit in {"mt", "mmt"}:
        working_df[numeric_columns] = working_df[numeric_columns] / 1.36
        return working_df.round(2)

    if target_unit != "mcm_d":
        return working_df.round(2)

    days = working_df[period_col].apply(lambda value: get_days_in_period(value, period_type))
    working_df[numeric_columns] = working_df[numeric_columns].mul(1000.0, axis=0)
    for column_name in numeric_columns:
        working_df[column_name] = working_df[column_name] / days

    return working_df.round(2)
