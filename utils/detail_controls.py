"""Shared controls helpers for exporter/importer detail pages."""


def normalize_rolling_window_days(window_days, default=30):
    """Ensure the rolling window input is always a positive integer."""
    try:
        normalized_window_days = int(window_days)
        return normalized_window_days if normalized_window_days > 0 else default
    except (TypeError, ValueError):
        return default


def format_rolling_window_label(window_days):
    return f"{normalize_rolling_window_days(window_days)}D"


def format_rolling_window_title(window_days):
    normalized_window_days = normalize_rolling_window_days(window_days)
    return f"{normalized_window_days}-Day Rolling Average"


def detail_count_options(max_count, min_count=1):
    return [
        {"label": str(value), "value": value}
        for value in range(min_count, max_count + 1)
    ]


def coerce_detail_count(value, default, max_count, min_count=0):
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    return max(min_count, min(max_count, coerced))
