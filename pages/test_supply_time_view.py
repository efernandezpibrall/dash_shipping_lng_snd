import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "dash_shipping_lng_snd"))

from pages import supply  # noqa: E402


DESTINATION_LOOKUP_RECORDS = [
    {
        "country_name": "United States",
        "country": "United States",
        "continent": "North America",
        "subcontinent": "Northern America",
        "basin": "Atlantic",
        "country_classification_level1": "Atlantic Basin",
        "country_classification": "Atlantic Producer",
        "shipping_region": "US Gulf",
    },
    {
        "country_name": "Qatar",
        "country": "Qatar",
        "continent": "Asia",
        "subcontinent": "Western Asia",
        "basin": "Middle East",
        "country_classification_level1": "Middle East",
        "country_classification": "Middle East Producer",
        "shipping_region": "Middle East",
    },
    {
        "country_name": "Peru",
        "country": "Peru",
        "continent": "South America",
        "subcontinent": "South America",
        "basin": "Pacific",
        "country_classification_level1": "Pacific Basin",
        "country_classification": "Pacific Producer",
        "shipping_region": "Pacific",
    },
]


def test_apply_supply_time_view_quarterly_uses_day_weighted_average():
    monthly_df = pd.DataFrame(
        {
            "Month": ["2024-01", "2024-02", "2024-03"],
            "Total MMTPA": [15.0, 25.0, 35.0],
            "United States": [10.0, 20.0, 30.0],
            "Qatar": [5.0, 5.0, 5.0],
        }
    )

    result = supply._apply_supply_time_view(monthly_df, "quarterly")

    assert result.to_dict("records") == [
        {
            "Month": "2024-Q1",
            "Total MMTPA": 25.0,
            "United States": 20.0,
            "Qatar": 5.0,
        }
    ]


def test_apply_supply_time_view_seasonally_keeps_capacity_season_labels():
    monthly_df = pd.DataFrame(
        {
            "Month": ["2024-09", "2024-10", "2025-03", "2025-04"],
            "Total MMTPA": [10.0, 10.0, 10.0, 10.0],
            "United States": [10.0, 10.0, 10.0, 10.0],
        }
    )

    result = supply._apply_supply_time_view(monthly_df, "seasonally")

    assert result["Month"].tolist() == ["2024-S", "2024-W", "2025-S"]
    assert result["Total MMTPA"].tolist() == [10.0, 10.0, 10.0]


def test_apply_supply_time_view_yearly_respects_leap_year_day_weights():
    monthly_df = pd.DataFrame(
        {
            "Month": ["2024-01", "2024-02", "2025-01", "2025-02"],
            "Total MMTPA": [0.0, 100.0, 0.0, 100.0],
            "United States": [0.0, 100.0, 0.0, 100.0],
        }
    )

    result = supply._apply_supply_time_view(monthly_df, "yearly")

    assert result.to_dict("records") == [
        {
            "Month": "2024",
            "Total MMTPA": 48.33,
            "United States": 48.33,
        },
        {
            "Month": "2025",
            "Total MMTPA": 47.46,
            "United States": 47.46,
        },
    ]


def test_apply_supply_time_view_uses_only_visible_months_for_partial_periods():
    monthly_df = pd.DataFrame(
        {
            "Month": ["2024-02", "2024-03"],
            "Total MMTPA": [0.0, 100.0],
            "United States": [0.0, 100.0],
        }
    )

    result = supply._apply_supply_time_view(monthly_df, "quarterly")

    assert result.to_dict("records") == [
        {
            "Month": "2024-Q1",
            "Total MMTPA": 51.67,
            "United States": 51.67,
        }
    ]


def test_delta_alignment_keeps_missing_comparison_periods_blank():
    baseline_matrix = pd.DataFrame(
        {
            "Month": ["2024-01", "2024-02", "2024-03"],
            "Total MMTPA": [100.0, 100.0, 100.0],
            "United States": [100.0, 100.0, 100.0],
        }
    )
    comparison_matrix = pd.DataFrame(
        {
            "Month": ["2024-02"],
            "Total MMTPA": [100.0],
            "United States": [100.0],
        }
    )

    comparison_aligned = supply._align_matrix_to_reference_months(
        comparison_matrix,
        baseline_matrix["Month"].tolist(),
    )

    baseline_quarterly = supply._apply_supply_time_view(baseline_matrix, "quarterly")
    comparison_quarterly = supply._apply_supply_time_view(comparison_aligned, "quarterly")
    delta_quarterly = supply._build_delta_matrix(
        baseline_quarterly,
        comparison_quarterly,
    )

    assert delta_quarterly["Month"].tolist() == ["2024-Q1"]
    assert pd.isna(comparison_quarterly.loc[0, "Total MMTPA"])
    assert pd.isna(comparison_quarterly.loc[0, "United States"])
    assert pd.isna(delta_quarterly.loc[0, "Total MMTPA"])
    assert pd.isna(delta_quarterly.loc[0, "United States"])


def test_enrich_export_flow_with_destination_aggregation_fills_unknown_for_missing_mapping():
    raw_df = pd.DataFrame(
        {
            "month": ["2024-01-01", "2024-01-01"],
            "country_name": ["United States", "Unknownland"],
            "total_mmtpa": [10.0, 5.0],
        }
    )

    enriched_df = supply._enrich_export_flow_with_destination_aggregation(
        raw_df,
        "basin",
        DESTINATION_LOOKUP_RECORDS,
    )

    assert enriched_df["destination_group"].tolist() == ["Atlantic", "Unknown"]


def test_build_supply_matrix_country_mode_matches_country_builder():
    raw_df = pd.DataFrame(
        {
            "month": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"],
            "country_name": ["United States", "Qatar", "United States", "Qatar"],
            "total_mmtpa": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = supply._build_supply_matrix(
        raw_df,
        "country",
        ["United States"],
        "rest_of_world",
        DESTINATION_LOOKUP_RECORDS,
    )

    assert result.to_dict("records") == [
        {
            "Month": "2024-01",
            "Total MMTPA": 30.0,
            "United States": 10.0,
            "Rest of the World": 20.0,
        },
        {
            "Month": "2024-02",
            "Total MMTPA": 70.0,
            "United States": 30.0,
            "Rest of the World": 40.0,
        },
    ]


def test_build_supply_matrix_non_country_mode_creates_explicit_group_columns():
    raw_df = pd.DataFrame(
        {
            "month": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-02-01"],
            "country_name": ["United States", "Qatar", "Unknownland", "United States"],
            "total_mmtpa": [10.0, 20.0, 5.0, 30.0],
        }
    )

    result = supply._build_supply_matrix(
        raw_df,
        "basin",
        ["Atlantic", "Middle East", "Unknown"],
        "exclude",
        DESTINATION_LOOKUP_RECORDS,
    )

    assert result.to_dict("records") == [
        {
            "Month": "2024-01",
            "Total MMTPA": 35.0,
            "Atlantic": 10.0,
            "Middle East": 20.0,
            "Unknown": 5.0,
        },
        {
            "Month": "2024-02",
            "Total MMTPA": 30.0,
            "Atlantic": 30.0,
            "Middle East": 0.0,
            "Unknown": 0.0,
        },
    ]


def test_update_balance_country_options_disables_country_controls_for_non_country_aggregation():
    source_df = pd.DataFrame(
        {
            "month": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "country_name": ["United States", "Qatar", "Peru"],
            "total_mmtpa": [10.0, 20.0, 5.0],
        }
    )

    result = supply.update_balance_country_options(
        ["United States", "Qatar", "Peru"],
        "basin",
        DESTINATION_LOOKUP_RECORDS,
        supply._serialize_dataframe(source_df),
        supply._serialize_dataframe(pd.DataFrame()),
        "2024-01-01",
        "2024-01-31",
        ["United States"],
        ["United States", "Qatar"],
    )

    assert result[0] == [
        {"label": "Atlantic", "value": "Atlantic"},
        {"label": "Middle East", "value": "Middle East"},
        {"label": "Pacific", "value": "Pacific"},
    ]
    assert result[1] == ["Atlantic", "Middle East", "Pacific"]
    assert result[2] is True
    assert result[3] == "Destination Columns"
    assert result[4] == "Basin groups:"
    assert result[6] is True
    assert result[7] == ["United States"]


def test_update_balance_country_options_restores_previous_country_selection():
    result = supply.update_balance_country_options(
        ["United States", "Qatar", "Peru"],
        "country",
        DESTINATION_LOOKUP_RECORDS,
        None,
        None,
        None,
        None,
        ["Atlantic", "Middle East"],
        ["United States", "Qatar"],
    )

    assert result[0][:2] == [
        {"label": "United States", "value": "United States"},
        {"label": "Qatar", "value": "Qatar"},
    ]
    assert result[1] == ["United States", "Qatar"]
    assert result[2] is False
    assert result[3] == "Country Columns"
    assert result[4] == "Countries:"
    assert result[6] is False
    assert result[7] == ["United States", "Qatar"]


def test_update_balance_country_options_preserves_explicit_empty_country_selection():
    result = supply.update_balance_country_options(
        ["United States", "Qatar", "Peru"],
        "basin",
        DESTINATION_LOOKUP_RECORDS,
        supply._serialize_dataframe(
            pd.DataFrame(
                {
                    "month": ["2024-01-01"],
                    "country_name": ["United States"],
                    "total_mmtpa": [10.0],
                }
            )
        ),
        supply._serialize_dataframe(pd.DataFrame()),
        "2024-01-01",
        "2024-01-31",
        [],
        ["United States", "Qatar"],
    )

    assert result[7] == []
