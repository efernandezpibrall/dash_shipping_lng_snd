"""Shared country-selection helpers for import/export flow pages."""

import pandas as pd


def resolve_available_countries(dataframes: list[pd.DataFrame]) -> list[str]:
    non_empty_frames = [df for df in dataframes if df is not None and not df.empty]
    if not non_empty_frames:
        return []

    combined_df = pd.concat(non_empty_frames, ignore_index=True)
    country_totals = (
        combined_df.groupby("country_name", as_index=False)["total_mmtpa"]
        .sum()
        .sort_values(["total_mmtpa", "country_name"], ascending=[False, True])
    )

    return country_totals["country_name"].tolist()


def resolve_default_selected_countries(
    available_countries: list[str],
    default_countries: list[str],
) -> list[str]:
    defaults = [country for country in default_countries if country in available_countries]
    if defaults:
        return defaults

    return available_countries[: min(7, len(available_countries))]


def normalize_selected_flow_values(selected_values) -> list[str]:
    if selected_values is None:
        return []
    if isinstance(selected_values, str):
        raw_values = [selected_values]
    else:
        raw_values = list(selected_values)

    normalized_values = []
    for value in raw_values:
        if pd.isna(value):
            normalized_value = None
        else:
            normalized_value = str(value).strip()
            normalized_value = normalized_value if normalized_value else None
        if normalized_value is not None and normalized_value not in normalized_values:
            normalized_values.append(normalized_value)

    return normalized_values


def resolve_flow_country_selection(
    selected_values,
    available_countries: list[str],
    default_selected_countries_fn,
) -> list[str]:
    available_country_set = set(available_countries or [])
    normalized_selection = normalize_selected_flow_values(selected_values)
    if normalized_selection:
        resolved_selection = [
            country
            for country in normalized_selection
            if country in available_country_set
        ]
        if resolved_selection:
            return resolved_selection

    return default_selected_countries_fn(available_countries or [])


def resolve_flow_destination_selection(
    destination_aggregation: str,
    selected_values,
    available_countries: list[str],
    lookup_records,
    *,
    default_selected_countries_fn,
    available_group_values_fn,
) -> list[str]:
    if destination_aggregation == "country":
        return resolve_flow_country_selection(
            selected_values,
            available_countries,
            default_selected_countries_fn,
        )

    normalized_selection = normalize_selected_flow_values(selected_values)
    if normalized_selection:
        return normalized_selection

    return available_group_values_fn(
        destination_aggregation,
        available_countries,
        lookup_records,
    )


def sanitize_raw_flow_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "country_name", "total_mmtpa"])

    cleaned_df = df.copy()
    cleaned_df["month"] = pd.to_datetime(cleaned_df["month"])
    cleaned_df["country_name"] = (
        cleaned_df["country_name"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )
    cleaned_df["total_mmtpa"] = pd.to_numeric(
        cleaned_df["total_mmtpa"], errors="coerce"
    ).fillna(0.0)

    cleaned_df = (
        cleaned_df.groupby(["month", "country_name"], as_index=False)["total_mmtpa"]
        .sum()
        .sort_values(["month", "country_name"])
    )

    return cleaned_df


def sanitize_monthly_country_flow_data(
    raw_df: pd.DataFrame,
    *,
    unknown_group: str = "Unknown",
) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["month", "country_name", "total_mmtpa"])

    cleaned_df = raw_df.copy()
    cleaned_df["month"] = (
        pd.to_datetime(cleaned_df["month"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    cleaned_df["country_name"] = (
        cleaned_df["country_name"]
        .fillna(unknown_group)
        .astype(str)
        .str.strip()
        .replace("", unknown_group)
    )
    cleaned_df["total_mmtpa"] = pd.to_numeric(
        cleaned_df["total_mmtpa"],
        errors="coerce",
    ).fillna(0.0)
    cleaned_df = cleaned_df.dropna(subset=["month"])

    return (
        cleaned_df.groupby(["month", "country_name"], as_index=False)["total_mmtpa"]
        .sum()
        .sort_values(["month", "country_name"])
        .reset_index(drop=True)
    )


def serialize_timestamp(value) -> str | None:
    if value is None or pd.isna(value):
        return None

    return pd.Timestamp(value).isoformat()


def build_woodmac_flow_metadata(metadata_df: pd.DataFrame) -> dict[str, str | None]:
    if metadata_df.empty:
        return {}

    row = metadata_df.iloc[0]
    return {
        "short_term_market_outlook": row.get("short_term_market_outlook"),
        "short_term_publication_timestamp": serialize_timestamp(
            row.get("short_term_publication_timestamp")
        ),
        "long_term_market_outlook": row.get("long_term_market_outlook"),
        "long_term_publication_timestamp": serialize_timestamp(
            row.get("long_term_publication_timestamp")
        ),
    }


def build_woodmac_publication_options(
    options_df: pd.DataFrame,
) -> dict[str, list[dict[str, str | None]]]:
    if options_df.empty:
        return {"short_term": [], "long_term": []}

    result = {"short_term": [], "long_term": []}
    for _, row in options_df.iterrows():
        result[row["publication_kind"]].append(
            {
                "market_outlook": row["market_outlook"],
                "publication_timestamp": serialize_timestamp(
                    row["publication_timestamp"]
                ),
            }
        )

    return result
