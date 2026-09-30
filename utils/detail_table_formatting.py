"""Shared scalar formatting helpers for exporter/importer detail tables."""

import numpy as np
import pandas as pd


def column_header_text(column):
    name = column.get('name') or column.get('headerName') or column.get('id') or column.get('field') or ''
    if isinstance(name, (list, tuple)):
        return ' '.join(str(part) for part in name)
    return str(name)


def column_id(column):
    value = column.get('id', column.get('field', column.get('name', '')))
    return str(value)


def width_sample_text(value):
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return format_table_value_max_one_decimal(value)
    return str(value)


def compact_column_width(header_text, value_samples, min_width, max_width, is_numeric=False):
    samples = [str(header_text), *[str(sample) for sample in value_samples if str(sample)]]
    max_header_chars = len(str(header_text))
    max_value_chars = max((len(sample) for sample in samples[1:]), default=0)
    effective_header_chars = max_header_chars
    if not is_numeric and max_header_chars > 16:
        effective_header_chars = int(np.ceil(max_header_chars / 2)) + 2
    header_px = effective_header_chars * 6.4 + (26 if is_numeric else 28)
    value_px = max_value_chars * (6.1 if is_numeric else 6.0) + (22 if is_numeric else 24)
    width = int(round(max(header_px, value_px, min_width)))
    return int(min(max(width, min_width), max_width))


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
