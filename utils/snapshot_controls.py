"""Shared snapshot dropdown helpers for balance comparison controls."""

from __future__ import annotations

import json

import pandas as pd


def format_metadata_timestamp(value) -> str | None:
    if not value:
        return None

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return str(value)

    return timestamp.strftime("%Y-%m-%d %H:%M")


def serialize_snapshot_value(payload: dict[str, str | None]) -> str:
    return json.dumps(payload, sort_keys=True)


def deserialize_snapshot_value(value: str | dict | None) -> dict[str, str | None]:
    if not value:
        return {}

    if isinstance(value, dict):
        return value

    try:
        parsed_value = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}

    return parsed_value if isinstance(parsed_value, dict) else {}


def default_previous_option_value(options: list[dict]) -> str | None:
    if len(options) > 1:
        return options[1]["value"]
    if options:
        return options[0]["value"]
    return None


def build_woodmac_snapshot_dropdown_options(
    publication_options: list[dict[str, str | None]],
) -> list[dict[str, str]]:
    dropdown_options = []
    for option in publication_options:
        publication_label = option.get("market_outlook", "Unknown publication")
        publication_timestamp = format_metadata_timestamp(
            option.get("publication_timestamp")
        )
        label = publication_label
        if publication_timestamp:
            label = f"{publication_label} | {publication_timestamp}"

        dropdown_options.append(
            {
                "label": label,
                "value": serialize_snapshot_value(option),
            }
        )

    return dropdown_options


def build_ea_upload_dropdown_options(
    upload_timestamps: list[str],
) -> list[dict[str, str]]:
    dropdown_options = []
    for upload_timestamp in upload_timestamps:
        formatted_timestamp = format_metadata_timestamp(upload_timestamp) or upload_timestamp
        dropdown_options.append(
            {
                "label": formatted_timestamp,
                "value": upload_timestamp,
            }
        )

    return dropdown_options


def woodmac_metadata_from_publication_options(
    publication_options: dict | None,
) -> dict[str, str | None]:
    publication_options = publication_options or {}
    short_term_options = publication_options.get("short_term") or []
    long_term_options = publication_options.get("long_term") or []
    if not short_term_options or not long_term_options:
        return {}

    short_term = short_term_options[0]
    long_term = long_term_options[0]
    return {
        "short_term_market_outlook": short_term.get("market_outlook"),
        "short_term_publication_timestamp": short_term.get("publication_timestamp"),
        "long_term_market_outlook": long_term.get("market_outlook"),
        "long_term_publication_timestamp": long_term.get("publication_timestamp"),
    }


def ea_metadata_from_upload_options(
    upload_options: list[str] | None,
) -> dict[str, str | None]:
    if not upload_options:
        return {}
    return {"upload_timestamp_utc": upload_options[0]}


def build_woodmac_metadata_lines(metadata: dict | None) -> list[str]:
    if not metadata:
        return []

    lines = []
    short_term_line = metadata.get("short_term_market_outlook")
    short_term_timestamp = format_metadata_timestamp(
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
    long_term_timestamp = format_metadata_timestamp(
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


def build_ea_metadata_lines(metadata: dict | None) -> list[str]:
    if not metadata:
        return []

    upload_timestamp = format_metadata_timestamp(metadata.get("upload_timestamp_utc"))
    if not upload_timestamp:
        return []

    return [f"upload_timestamp_utc: {upload_timestamp}"]


def resolve_snapshot_control_values(
    comparison_source,
    comparison_options,
    current_st_value,
    current_lt_value,
    current_ea_upload_value,
):
    comparison_options = comparison_options or {}
    woodmac_options = comparison_options.get("woodmac", {})
    short_term_options = build_woodmac_snapshot_dropdown_options(
        woodmac_options.get("short_term", [])
    )
    long_term_options = build_woodmac_snapshot_dropdown_options(
        woodmac_options.get("long_term", [])
    )
    ea_upload_options = build_ea_upload_dropdown_options(
        comparison_options.get("ea_uploads", [])
    )

    short_term_values = {option["value"] for option in short_term_options}
    long_term_values = {option["value"] for option in long_term_options}
    ea_upload_values = {option["value"] for option in ea_upload_options}

    short_term_value = (
        current_st_value
        if current_st_value in short_term_values
        else default_previous_option_value(short_term_options)
    )
    long_term_value = (
        current_lt_value
        if current_lt_value in long_term_values
        else default_previous_option_value(long_term_options)
    )
    ea_upload_value = (
        current_ea_upload_value
        if current_ea_upload_value in ea_upload_values
        else default_previous_option_value(ea_upload_options)
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
