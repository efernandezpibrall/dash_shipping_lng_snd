import pandas as pd
import pytest

from pages import production


def _component_text(component):
    if component is None:
        return ""
    if isinstance(component, str):
        return component
    if isinstance(component, (list, tuple)):
        return " ".join(_component_text(child) for child in component)
    return _component_text(getattr(component, "children", None))


def test_production_groupings_reconcile_to_the_exact_run_total(monkeypatch):
    source_df = pd.DataFrame(
        [
            {
                "run_id": 12,
                "train_key": "train-1",
                "year": 2027,
                "month": 1,
                "country_name": "Mexico",
                "plant_name": "Terminal A",
                "lng_train_name_short": "Train 1",
                "total_output": 1.0,
            },
            {
                "run_id": 12,
                "train_key": "train-2",
                "year": 2027,
                "month": 1,
                "country_name": "Mexico",
                "plant_name": "Terminal A",
                "lng_train_name_short": "Train 2",
                "total_output": 2.0,
            },
            {
                "run_id": 12,
                "train_key": "train-3",
                "year": 2027,
                "month": 1,
                "country_name": "Canada",
                "plant_name": "Terminal B",
                "lng_train_name_short": "Train 1",
                "total_output": 3.0,
            },
        ]
    )
    calls = []

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        return source_df.copy()

    monkeypatch.setattr(production, "fetch_capacity_ramp_production_monthly", fake_fetch)
    production._fetch_volume_country_dataframe_cached.cache_clear()

    grouped = {}
    for breakdown in ("country", "project", "train"):
        grouped[breakdown] = production._prepare_volume_country_dataframe(
            run_id=12,
            selected_unit="mtpa",
            new_capacity_only=True,
            start_year=2026,
            end_year=2031,
            breakdown=breakdown,
            refresh_token=1,
        )

    assert all(frame["total_output"].sum() == pytest.approx(6.0) for frame in grouped.values())
    assert set(grouped["country"]["country_name"]) == {"Canada", "Mexico"}
    assert set(grouped["project"]["country_name"]) == {"Terminal A", "Terminal B"}
    assert set(grouped["train"]["country_name"]) == {
        "Terminal A - Train 1",
        "Terminal A - Train 2",
        "Terminal B - Train 1",
    }
    assert all(call["run_id"] == 12 for call in calls)
    assert all(call["new_capacity_only"] is True for call in calls)


def test_capacity_scenario_selector_defaults_to_base_case(monkeypatch):
    catalog = (
        {
            "scenario_id": 7,
            "scenario_name": "Alternative",
            "display_run_id": None,
        },
        {
            "scenario_id": 2,
            "scenario_name": "Base Case",
            "display_run_id": 12,
            "display_run_status": "completed",
            "display_blocking_qa_count": 1,
        },
    )
    monkeypatch.setattr(production, "_get_capacity_ramp_catalog_cached", lambda _: catalog)

    options, selected = production.populate_capacity_scenario_options(0, None)

    assert selected == 2
    assert [option["value"] for option in options] == [7, 2]
    assert options[0]["label"] == "Alternative - No ramp run"
    assert options[1]["label"] == "Base Case - QA blocked (run 12)"


def test_blocked_fallback_banner_identifies_both_runs_and_stale_state():
    banner = production._build_ramp_run_status_banner(
        {
            "scenario_name": "Base Case",
            "latest_attempt_run_id": 13,
            "latest_attempt_run_status": "failed",
            "display_run_id": 12,
            "display_run_status": "completed",
            "display_is_current_published": False,
            "display_generator_version": "terminal_ramp_sql_inputs_v1",
            "display_generated_at": "2026-07-13T10:00:00Z",
            "display_horizon_start_month": "2026-07-01",
            "display_horizon_end_month": "2031-07-01",
            "display_monthly_row_count": 7198,
            "display_train_count": 118,
            "display_blocking_qa_count": 1,
            "display_blocking_qa_summaries": [
                {"plant_name": "Tango", "message": "Approved profile is missing."}
            ],
            "display_is_stale": True,
            "fallback_reason": "Latest attempt run 13 is failed; showing materialized run 12.",
        }
    )

    text = _component_text(banner)
    assert "QA-blocked ramp output shown for analysis" in text
    assert "Run 12" in text
    assert "generator terminal_ramp_sql_inputs_v1" in text
    assert "118 trains" in text
    assert "Latest attempt run 13 is failed" in text
    assert "Approved profile is missing" in text
    assert "Stale" in text


def test_excel_export_uses_store_run_id(monkeypatch):
    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "month": pd.to_datetime(["2027-01-01"]),
                "country_name": ["Terminal A"],
                "total_output": [1.25],
            }
        )

    monkeypatch.setattr(production, "_prepare_volume_country_dataframe", fake_prepare)

    download = production.export_capacity_to_excel(
        1,
        {"display_run_id": 12, "refresh_token": 4},
        "mtpa",
        [],
        "project",
        None,
        "rest_of_world",
        [2026, 2031],
    )

    assert captured["run_id"] == 12
    assert captured["refresh_token"] == 4
    assert download["filename"].startswith("LNG_Production_Run12_PlantMatrix_MTPA_")
    assert download["base64"] is True


def test_global_supply_chart_has_one_trace_per_available_source():
    comparison_df = pd.DataFrame(
        {
            "month": pd.to_datetime(
                ["2026-07-01", "2026-08-01"] * 4
            ),
            "source": [
                "Our ramp forecast",
                "Our ramp forecast",
                "Energy Aspects",
                "Energy Aspects",
                "Platts",
                "Platts",
                "WoodMac",
                "WoodMac",
            ],
            "monthly_mt": [40.0, 41.0, 39.0, 40.0, 40.5, 41.5, 38.5, 39.5],
        }
    )
    metadata = {
        "window_start": "2021-07-01",
        "forecast_start": "2026-07-01",
        "window_end": "2031-12-01",
    }

    figure = production.create_global_supply_comparison_chart(comparison_df, metadata)

    assert [trace.name for trace in figure.data] == [
        "Our ramp forecast",
        "Energy Aspects",
        "Platts",
        "WoodMac",
    ]
    assert figure.layout.yaxis.title.text == "Monthly LNG supply (Mt/month)"
    assert pd.Timestamp(figure.layout.xaxis.range[0]) == pd.Timestamp("2021-07-01")
    assert pd.Timestamp(figure.layout.xaxis.range[1]) == pd.Timestamp("2031-12-01")


def test_global_supply_quarter_table_and_chart_use_same_period_totals():
    comparison_df = pd.DataFrame(
        {
            "month": list(pd.date_range("2027-01-01", "2027-03-01", freq="MS")) * 2,
            "source": ["Our ramp forecast"] * 3 + ["WoodMac"] * 3,
            "monthly_mt": [40.0, 41.0, 42.0, 39.0, 40.0, 41.0],
        }
    )
    metadata = {
        "window_start": "2021-07-01",
        "forecast_start": "2026-07-01",
        "window_end": "2031-12-01",
    }
    period_df = production.aggregate_global_supply_comparison(comparison_df, "quarterly")

    figure = production.create_global_supply_comparison_chart(
        period_df,
        metadata,
        "quarterly",
    )
    table = production._create_global_supply_comparison_table(period_df, "quarterly")

    assert figure.layout.yaxis.title.text == "Quarterly LNG supply (Mt/quarter)"
    assert list(figure.data[0].y) == [123.0]
    assert list(figure.data[1].y) == [120.0]
    table_grid = table.children[1]
    assert table_grid.rowData[0]["Quarter"] == "2027 Q1"
    assert table_grid.rowData[0]["Our ramp forecast"] == pytest.approx(123.0)
    assert table_grid.rowData[0]["WoodMac"] == pytest.approx(120.0)
