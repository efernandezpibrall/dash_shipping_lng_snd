"""Shared scalar formatting helpers for exporter/importer detail tables."""

import numpy as np
import pandas as pd


def format_table_value_max_one_decimal(value):
    """Format table display values with at most one decimal place."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return str(value)

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(numeric_value):
        return ""
    if abs(numeric_value) < 0.05:
        numeric_value = 0

    text = f"{numeric_value:,.1f}"
    return text.rstrip("0").rstrip(".")


def round_table_value_max_one_decimal(value):
    """Round raw rowData values so table renderers cannot leak long float precision."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return value

    if not np.isfinite(numeric_value):
        return None
    rounded_value = round(numeric_value, 1)
    return int(rounded_value) if float(rounded_value).is_integer() else rounded_value
