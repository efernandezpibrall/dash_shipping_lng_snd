import pandas as pd

from utils import global_supply_comparison as comparison


def test_energy_aspects_query_uses_bounded_current_state_view():
    current_query = str(comparison.EA_GLOBAL_SUPPLY_CURRENT_QUERY)

    assert "at_lng.ea_values_current source" in current_query
    assert "at_lng.ea_values source" not in current_query
    assert "MAX(upload_timestamp_utc)" not in current_query
    assert "source.dataset_id = :dataset_id" in current_query
    assert "source.date >= CAST(:start_month AS timestamp)" in current_query
    assert "source.date < CAST(:end_month AS timestamp) + INTERVAL '1 month'" in current_query


def test_comparison_bounds_use_five_complete_trailing_years():
    start, forecast_start, end = comparison.comparison_month_bounds("2026-07-16")

    assert start == pd.Timestamp("2021-07-01")
    assert forecast_start == pd.Timestamp("2026-07-01")
    assert end == pd.Timestamp("2031-12-01")


def test_comparison_keeps_real_source_horizons_and_builds_platts_2031(monkeypatch):
    monkeypatch.setattr(
        comparison,
        "_fetch_ramp",
        lambda *args: pd.DataFrame(
            {"month": [pd.Timestamp("2026-07-01")], "monthly_mt": [40.0]}
        ),
    )
    monkeypatch.setattr(
        comparison,
        "_fetch_ea",
        lambda *args: pd.DataFrame(
            {
                "month": [pd.Timestamp("2029-12-01")],
                "monthly_mt": [50.0],
                "upload_timestamp_utc": [pd.Timestamp("2026-07-16")],
            }
        ),
    )
    monkeypatch.setattr(
        comparison,
        "_fetch_platts",
        lambda *args: (
            pd.DataFrame(
                {
                    "month": [pd.Timestamp("2030-12-01")],
                    "monthly_mt": [52.0],
                    "upload_timestamp_utc": [pd.Timestamp("2026-05-05")],
                    "vintage_date": [pd.Timestamp("2026-04-01")],
                }
            ),
            pd.DataFrame(
                {
                    "annual_mt": [660.0],
                    "vintage_date": [pd.Timestamp("2026-01-01")],
                }
            ),
            pd.DataFrame(
                {
                    "month": pd.date_range("2031-01-01", "2031-12-01", freq="MS"),
                    "monthly_mt": [55.0] * 12,
                }
            ),
        ),
    )
    monkeypatch.setattr(
        comparison,
        "_fetch_woodmac",
        lambda *args: (
            pd.DataFrame(
                {"month": [pd.Timestamp("2031-12-01")], "monthly_mt": [56.0]}
            ),
            {
                "short_term_market_outlook": "Short",
                "long_term_market_outlook": "Long",
            },
        ),
    )

    result, metadata = comparison.fetch_global_supply_comparison(
        object(), 41, as_of_month="2026-07-01"
    )

    ea = result[result["source"].eq("Energy Aspects")]
    platts = result[result["source"].eq("Platts")]
    ramp = result[result["source"].eq("Our ramp forecast")]
    assert ea["month"].max() == pd.Timestamp("2029-12-01")
    assert ramp["month"].max() == pd.Timestamp("2026-07-01")
    assert len(platts[platts["month"].dt.year.eq(2031)]) == 12
    assert set(platts.loc[platts["month"].dt.year.eq(2031), "monthly_mt"]) == {55.0}
    assert metadata["sources"]["Platts"]["last_month"] == "Dec 2031"
    assert metadata["warnings"] == []


def test_time_views_sum_only_complete_periods():
    monthly = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2027-01-01",
                    "2027-02-01",
                    "2027-03-01",
                    "2027-04-01",
                    "2027-05-01",
                ]
            ),
            "source": ["WoodMac"] * 5,
            "monthly_mt": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    quarterly = comparison.aggregate_global_supply_comparison(monthly, "quarter")
    yearly = comparison.aggregate_global_supply_comparison(monthly, "year")

    assert quarterly[["period_label", "supply_mt"]].to_dict("records") == [
        {"period_label": "2027 Q1", "supply_mt": 6.0}
    ]
    assert yearly.empty


def test_lng_season_view_uses_april_to_september_and_october_to_march():
    monthly = pd.DataFrame(
        {
            "month": pd.date_range("2026-10-01", "2027-03-01", freq="MS"),
            "source": ["Our ramp forecast"] * 6,
            "monthly_mt": [2.0] * 6,
        }
    )

    seasonal = comparison.aggregate_global_supply_comparison(monthly, "season")

    assert seasonal[["period_label", "supply_mt"]].to_dict("records") == [
        {"period_label": "Winter 2026/27", "supply_mt": 12.0}
    ]
