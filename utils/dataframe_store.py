"""Shared helpers for storing pandas frames in Dash memory stores."""

from __future__ import annotations

from io import StringIO

import pandas as pd


def serialize_dataframe_store(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None
    return df.to_json(date_format="iso", orient="split")


def deserialize_dataframe_store(data: str | None) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    return pd.read_json(StringIO(data), orient="split")


def serialize_dataframe_split_store(df: pd.DataFrame | None) -> str | None:
    if df is None:
        return None
    return df.to_json(date_format="iso", orient="split")


def load_dataframe_from_payload(
    payload: dict | None,
    key: str,
    date_columns: list[str] | None = None,
) -> pd.DataFrame:
    if not payload or not payload.get(key):
        return pd.DataFrame()
    try:
        df = pd.read_json(StringIO(payload[key]), orient="split")
    except Exception:
        return pd.DataFrame()

    for column in date_columns or []:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce").dt.normalize()
    return df
