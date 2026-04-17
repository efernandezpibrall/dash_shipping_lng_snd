import sys
import datetime as dt
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "dash_shipping_lng_snd"))

from pages import market_balance  # noqa: E402
from utils import balance_time  # noqa: E402
from utils import market_balance_data  # noqa: E402


def _walk_components(component):
    if component is None:
        return
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk_components(child)
    else:
        yield from _walk_components(children)


def _find_component_by_id(component, component_id):
    for child in _walk_components(component):
        if getattr(child, "id", None) == component_id:
            return child
    return None


def test_validate_complete_seasons_keeps_only_complete_periods():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-04-01",
                    "2024-05-01",
                    "2024-06-01",
                    "2024-07-01",
                    "2024-08-01",
                    "2024-09-01",
                    "2024-10-01",
                    "2024-11-01",
                    "2024-12-01",
                    "2025-01-01",
                ]
            ),
            "time_period": [
                "2024-S",
                "2024-S",
                "2024-S",
                "2024-S",
                "2024-S",
                "2024-S",
                "2024-W",
                "2024-W",
                "2024-W",
                "2024-W",
            ],
            "value": range(10),
        }
    )

    result = balance_time.validate_complete_seasons(df)

    assert sorted(result["time_period"].unique().tolist()) == ["2024-S"]


def test_convert_periodized_bcm_frame_to_mcmd_uses_exact_days():
    df = pd.DataFrame(
        {
            "Period": ["2024-Q1", "2024"],
            "Total": [9.1, 36.6],
            "Europe": [9.1, 36.6],
        }
    )

    result = balance_time.convert_periodized_bcm_frame(df, "mcm_d", "quarterly")

    assert result.loc[0, "Total"] == 100.0
    assert result.loc[0, "Europe"] == 100.0
    assert result.loc[1, "Total"] == 100.0
    assert result.loc[1, "Europe"] == 100.0


def test_align_frames_on_period_preserves_reference_order_and_blanks_missing():
    reference_df = pd.DataFrame(
        {
            "Period": ["2024-Q1", "2024-Q2"],
            "Total": [1.0, 2.0],
        }
    )
    comparison_df = pd.DataFrame(
        {
            "Period": ["2024-Q2"],
            "Total": [3.0],
        }
    )

    aligned_df = balance_time.align_frames_on_period(reference_df, comparison_df)

    assert aligned_df["Period"].tolist() == ["2024-Q1", "2024-Q2"]
    assert pd.isna(aligned_df.loc[0, "Total"])
    assert aligned_df.loc[1, "Total"] == 3.0


def test_calculate_flex_volumes_subtracts_contracts_from_actual():
    actual_df = pd.DataFrame(
        {
            "Period": ["2024", "2025"],
            "Total": [10.0, 12.0],
            "Europe": [4.0, 5.0],
        }
    )
    contract_df = pd.DataFrame(
        {
            "Period": ["2024", "2025"],
            "Total": [6.0, 7.0],
            "Europe": [1.0, 2.5],
        }
    )

    result = market_balance_data.calculate_flex_volumes(actual_df, contract_df)

    assert result.to_dict("records") == [
        {"Period": "2024", "Total": 4.0, "Europe": 3.0},
        {"Period": "2025", "Total": 5.0, "Europe": 2.5},
    ]


def test_build_net_balance_table_subtracts_imports_from_exports():
    exports_df = pd.DataFrame(
        {
            "Period": ["2024", "2025"],
            "Total": [12.0, 11.0],
            "Europe": [7.0, 6.0],
        }
    )
    imports_df = pd.DataFrame(
        {
            "Period": ["2024", "2025"],
            "Total": [9.0, 13.0],
            "Europe": [4.0, 8.0],
        }
    )

    result = market_balance_data._build_net_balance_table(
        exports_df,
        imports_df,
        time_group="yearly",
    )

    assert result.to_dict("records") == [
        {"Period": "2024", "Total": 3.0, "Europe": 3.0},
        {"Period": "2025", "Total": -2.0, "Europe": -2.0},
    ]


def test_periodize_maintenance_frame_applies_selected_time_view():
    maintenance_df = pd.DataFrame(
        {
            "Month": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-01-01",
                    "2024-02-01",
                ]
            ),
            "Provider": [
                "WoodMac",
                "WoodMac",
                "WoodMac",
                "Energy Aspects",
                "Energy Aspects",
            ],
            "Metric": [
                "Planned",
                "Planned",
                "Unplanned",
                "Unplanned",
                "Unplanned",
            ],
            "Value": [1.0, 2.0, 3.0, 0.5, 0.75],
        }
    )

    result = market_balance_data._periodize_maintenance_frame(
        maintenance_df,
        time_group="quarterly",
    )

    assert result.to_dict("records") == [
        {
            "Period": "2024-Q1",
            "Provider": "Energy Aspects",
            "Metric": "Unplanned",
            "Value": 1.25,
        },
        {
            "Period": "2024-Q1",
            "Provider": "WoodMac",
            "Metric": "Planned",
            "Value": 3.0,
        },
        {
            "Period": "2024-Q1",
            "Provider": "WoodMac",
            "Metric": "Unplanned",
            "Value": 3.0,
        },
    ]


def test_build_maintenance_provider_comparison_table_calculates_provider_gap():
    maintenance_df = pd.DataFrame(
        {
            "Period": ["2024-Q2", "2024-Q2", "2024-Q2", "2024-Q1", "2024-Q1", "2024-Q1"],
            "Provider": [
                "WoodMac",
                "WoodMac",
                "Energy Aspects",
                "WoodMac",
                "WoodMac",
                "Energy Aspects",
            ],
            "Metric": [
                "Unplanned",
                "Planned",
                "Unplanned",
                "Unplanned",
                "Planned",
                "Unplanned",
            ],
            "Value": [5.0, 2.0, 0.0, 3.0, 1.0, 2.0],
        }
    )

    result = market_balance_data.build_maintenance_provider_comparison_table(
        maintenance_df,
        time_group="quarterly",
    )

    assert result.columns.tolist() == [
        "Period",
        "WoodMac Unplanned",
        "Energy Aspects Unplanned",
        "Delta",
        "Delta %",
        "WoodMac Planned",
        "WoodMac Total",
    ]
    assert result["Period"].tolist() == ["2024-Q1", "2024-Q2"]
    assert result.loc[0].drop(labels=["Delta %"]).to_dict() == {
        "Period": "2024-Q1",
        "WoodMac Unplanned": 3.0,
        "Energy Aspects Unplanned": 2.0,
        "Delta": 1.0,
        "WoodMac Planned": 1.0,
        "WoodMac Total": 4.0,
    }
    assert result.loc[0, "Delta %"] == 50.0
    assert result.loc[1, "Delta"] == 5.0
    assert pd.isna(result.loc[1, "Delta %"])


def test_build_pacific_supply_detail_applies_selected_time_view():
    raw_df = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-01-01",
                    "2024-03-01",
                    "2024-04-01",
                ]
            ),
            "country_name": [
                "Australia",
                "Australia",
                "Australia",
                "Malaysia",
                "Malaysia",
                "Malaysia",
            ],
            "total_mmtpa": [12.0, 24.0, 36.0, 24.0, 12.0, 48.0],
        }
    )

    result = market_balance_data._build_pacific_supply_detail(
        raw_df,
        "WoodMac",
        time_group="quarterly",
    )

    assert result.to_dict("records") == [
        {
            "Period": "2024-Q1",
            "Country": "Australia",
            "Provider": "WoodMac",
            "Supply": 24.0,
        },
        {
            "Period": "2024-Q1",
            "Country": "Malaysia",
            "Provider": "WoodMac",
            "Supply": 12.0,
        },
        {
            "Period": "2024-Q2",
            "Country": "Malaysia",
            "Provider": "WoodMac",
            "Supply": 48.0,
        },
    ]


def test_build_maintenance_grouped_table_applies_country_grouping():
    raw_df = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-02-01",
                    "2024-02-01",
                ]
            ),
            "country_name": ["Australia", "Malaysia", "Australia", "Malaysia"],
            "metric": ["Planned", "Unplanned", "Planned", "Unplanned"],
            "value": [12.0, 24.0, 36.0, 12.0],
        }
    )
    mapping_df = pd.DataFrame(
        {
            "country_name": ["Australia", "Malaysia"],
            "country": ["Australia", "Malaysia"],
            "continent": ["Oceania", "Asia"],
            "subcontinent": ["Oceania", "South East Asia"],
            "basin": ["Pacific", "Pacific"],
            "shipping_region": ["Pacific", "Pacific"],
            "country_classification_level1": ["Pacific", "Pacific"],
            "country_classification": ["Australia", "SE Asia"],
        }
    )

    result = market_balance_data._build_maintenance_grouped_table(
        raw_df,
        mapping_df=mapping_df,
        country_group="country_classification",
        time_group="quarterly",
    )

    assert result.to_dict("records") == [
        {
            "Period": "2024-Q1",
            "Metric": "Total",
            "Total": 7.0,
            "Australia": 4.0,
            "SE Asia": 3.0,
        },
        {
            "Period": "2024-Q1",
            "Metric": "Planned",
            "Total": 4.0,
            "Australia": 4.0,
            "SE Asia": 0.0,
        },
        {
            "Period": "2024-Q1",
            "Metric": "Unplanned",
            "Total": 3.0,
            "Australia": 0.0,
            "SE Asia": 3.0,
        },
    ]
    assert result["Period"].nunique() == 1


def test_build_selected_maintenance_payload_filters_metric_rows():
    payload = {
        "records": [
            {"Period": "2024-Q1", "Metric": "Total", "Total": 7.0, "Australia": 4.0, "SE Asia": 3.0},
            {"Period": "2024-Q1", "Metric": "Planned", "Total": 4.0, "Australia": 4.0, "SE Asia": 0.0},
            {"Period": "2024-Q1", "Metric": "Unplanned", "Total": 3.0, "Australia": 0.0, "SE Asia": 3.0},
        ],
        "columns": ["Period", "Metric", "Total", "Australia", "SE Asia"],
        "numeric_columns": ["Total", "Australia", "SE Asia"],
    }

    result = market_balance._build_selected_maintenance_payload(payload, "Planned")

    assert result["columns"] == ["Period", "Total", "Australia", "SE Asia"]
    assert result["numeric_columns"] == ["Total", "Australia", "SE Asia"]
    assert result["records"] == [
        {"Period": "2024-Q1", "Total": 4.0, "Australia": 4.0, "SE Asia": 0.0}
    ]


def test_build_maintenance_kpi_cards_returns_empty_state():
    result = market_balance._build_maintenance_kpi_cards(pd.DataFrame())

    assert "No provider maintenance comparison available" in result.children


def test_build_ea_maintenance_total_table_stays_global():
    maintenance_df = pd.DataFrame(
        {
            "Period": ["2024-Q1", "2024-Q1", "2024-Q2"],
            "Provider": ["Energy Aspects", "WoodMac", "Energy Aspects"],
            "Metric": ["Unplanned", "Planned", "Unplanned"],
            "Value": [1.5, 3.0, 2.25],
        }
    )

    result = market_balance_data._build_ea_maintenance_total_table(maintenance_df)

    assert result.to_dict("records") == [
        {"Period": "2024-Q1", "Total": 1.5},
        {"Period": "2024-Q2", "Total": 2.25},
    ]


def test_calculate_flex_volumes_handles_columns_present_on_only_one_side():
    actual_df = pd.DataFrame(
        {
            "Period": ["2024"],
            "Total": [10.0],
            "Europe": [4.0],
        }
    )
    contract_df = pd.DataFrame(
        {
            "Period": ["2024"],
            "Total": [6.0],
            "Asia": [2.0],
        }
    )

    result = market_balance_data.calculate_flex_volumes(actual_df, contract_df)

    assert result.to_dict("records") == [
        {"Period": "2024", "Total": 4.0, "Asia": -2.0, "Europe": 4.0},
    ]


def test_fetch_contract_volume_tables_uses_contract_demand_source(monkeypatch):
    class _FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeEngine:
        def connect(self):
            return _FakeConnection()

    contracts_df = pd.DataFrame(
        {
            "contract_year": [2024, 2024],
            "exporter_country_name": ["United States", "Qatar"],
            "importer_country_name": ["Japan", "India"],
            "contracted_mmtpa": [6.0, 4.0],
        }
    )
    mapping_df = pd.DataFrame(
        {
            "country_name": ["United States", "Qatar", "Japan", "India"],
            "country": ["United States", "Qatar", "Japan", "India"],
            "continent": ["North America", "Middle East", "Asia", "Asia"],
            "subcontinent": ["North America", "Middle East", "East Asia", "South Asia"],
            "basin": ["Atlantic", "Middle East", "Pacific", "Indian"],
            "shipping_region": ["Atlantic", "Middle East", "Pacific", "Indian"],
            "country_classification_level1": ["Americas", "Middle East", "Asia", "Asia"],
            "country_classification": ["Americas", "Middle East", "Asia", "Asia"],
        }
    )

    monkeypatch.setattr(market_balance_data, "engine", _FakeEngine())
    monkeypatch.setattr(
        market_balance_data.pd,
        "read_sql_query",
        lambda query, connection: contracts_df.copy(),
    )

    exports_df, imports_df = market_balance_data.fetch_contract_volume_tables(
        country_group="country_classification",
        selected_years=[2024],
        unit="bcm",
        mapping_df=mapping_df,
    )

    assert exports_df.to_dict("records") == [
        {"Period": "2024", "Total": 13.6, "Americas": 8.16, "Middle East": 5.44}
    ]
    assert imports_df.to_dict("records") == [
        {"Period": "2024", "Total": 13.6, "Asia": 13.6}
    ]


def test_filter_complete_ea_monthly_rows_keeps_max_asset_count_only():
    raw_df = pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "value": [10.0, 11.0, 12.0],
            "upload_timestamp_utc": [
                "2024-04-01 00:00",
                "2024-04-01 00:00",
                "2024-04-01 00:00",
            ],
            "n_assets": [5, 4, 5],
        }
    )

    filtered_df = market_balance_data.filter_complete_ea_monthly_rows(raw_df)

    assert filtered_df["month"].dt.strftime("%Y-%m").tolist() == ["2024-01", "2024-03"]
    assert "n_assets" not in filtered_df.columns


def test_convert_wm_maintenance_to_monthly_mt_divides_annualized_values():
    raw_df = pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01"]),
            "metric": ["Unplanned"],
            "value": [12.0],
        }
    )

    converted_df = market_balance_data.convert_wm_maintenance_to_monthly_mt(raw_df)

    assert converted_df.loc[0, "value"] == 1.0


def test_format_country_balance_table_adds_fcst_margin():
    raw_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "Timestamp UTC": ["2024-02-01 00:00", "2024-02-01 00:00"],
            "type": ["Demand", "Supply"],
            "subtype": ["LNG", "Production"],
            "subsubtype": ["", ""],
            "value": [3.0, 5.0],
        }
    )

    formatted_df, formatted_columns = market_balance_data.format_country_balance_table(
        raw_df, "subtype"
    )

    assert "Fcst Margin" in formatted_df.columns
    assert formatted_df.loc[0, "Fcst Margin"] == 2.0
    assert ("Fcst Margin", "", "") in formatted_columns


def test_format_country_balance_table_disambiguates_duplicate_leaf_names():
    raw_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "Timestamp UTC": ["2024-02-01 00:00", "2024-02-01 00:00"],
            "type": ["Demand", "Supply"],
            "subtype": ["Stocks", "Stocks"],
            "subsubtype": ["", ""],
            "value": [3.0, 5.0],
        }
    )

    formatted_df, _ = market_balance_data.format_country_balance_table(raw_df, "subtype")

    assert "Demand | Stocks" in formatted_df.columns
    assert "Supply | Stocks" in formatted_df.columns


def test_format_country_balance_table_handles_subsubtype_overlap_without_merge_errors():
    raw_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "Timestamp UTC": ["2024-02-01 00:00", "2024-02-01 00:00", "2024-02-01 00:00"],
            "type": ["Demand", "Supply", "Supply"],
            "subtype": ["LNG", "Pipeline", "Pipeline"],
            "subsubtype": ["", "", "Nord Stream"],
            "value": [3.0, 5.0, 2.0],
        }
    )

    formatted_df, _ = market_balance_data.format_country_balance_table(raw_df, "subsubtype")

    assert "Demand" in formatted_df.columns
    assert "LNG" in formatted_df.columns
    assert "Supply" in formatted_df.columns
    assert "Pipeline" in formatted_df.columns
    assert "Nord Stream" in formatted_df.columns
    assert formatted_df.loc[0, "Fcst Margin"] == 4.0


def test_build_balance_display_columns_never_returns_blank_labels():
    labels = market_balance_data.build_balance_display_columns(
        [("", "", ""), ("", "", ""), ("Supply", "", "")]
    )

    assert labels == ["Unmapped", "Unmapped (2)", "Supply"]


def test_build_country_balance_delta_table_aligns_on_date():
    current_df = pd.DataFrame(
        {
            "Date": ["2024-Q1", "2024-Q2"],
            "Demand": [10.0, 12.0],
            "Supply": [14.0, 15.0],
        }
    )
    previous_df = pd.DataFrame(
        {
            "Date": ["2024-Q2"],
            "Demand": [11.0],
            "Supply": [13.5],
        }
    )

    delta_df = market_balance_data._build_country_balance_delta_table(
        current_df, previous_df
    )

    assert delta_df.to_dict("records") == [
        {"Date": "2024-Q1", "Demand": 10.0, "Supply": 14.0},
        {"Date": "2024-Q2", "Demand": 1.0, "Supply": 1.5},
    ]


def test_build_country_balance_delta_table_is_empty_without_comparison_rows():
    current_df = pd.DataFrame(
        {
            "Date": ["2024-Q1", "2024-Q2"],
            "Demand": [10.0, 12.0],
            "Supply": [14.0, 15.0],
        }
    )

    delta_df = market_balance_data._build_country_balance_delta_table(
        current_df, pd.DataFrame()
    )

    assert delta_df.columns.tolist() == current_df.columns.tolist()
    assert delta_df.empty


def test_drop_country_display_metadata_columns_removes_timestamp_utc():
    df = pd.DataFrame(
        {
            "Date": ["2024"],
            "Timestamp UTC": ["2026-04-15 06:55:14"],
            "Demand": [10.0],
        }
    )

    result = market_balance_data._drop_country_display_metadata_columns(df)

    assert result.columns.tolist() == ["Date", "Demand"]


def test_get_balance_column_styles_handles_tables_without_timestamp_column():
    styles = market_balance_data.get_balance_column_styles(
        ["Date", "Demand", "Supply"],
        [("Demand", "", ""), ("Supply", "", "")],
    )

    assert styles[0]["if"]["column_id"] == "Demand"
    assert styles[1]["if"]["column_id"] == "Supply"


def test_sync_country_snapshot_control_uses_selected_country_snapshots():
    payload = {
        "data": {
            "snapshots": ["2024-03-01 00:00", "2024-02-01 00:00"],
            "country_snapshots": {
                "Japan": ["2024-03-01 00:00", "2024-01-01 00:00"],
                "France": ["2024-03-02 00:00", "2024-02-02 00:00"],
            },
        }
    }

    options, value = market_balance.sync_country_snapshot_control(
        payload, "France", "2024-01-01 00:00"
    )

    assert options == [
        {"label": "2024-03-02 00:00", "value": "2024-03-02 00:00"},
        {"label": "2024-02-02 00:00", "value": "2024-02-02 00:00"},
    ]
    assert value == "2024-02-02 00:00"


def test_serialize_scalar_preserves_snapshot_seconds_when_present():
    assert (
        market_balance_data._serialize_scalar(pd.Timestamp("2026-01-28 08:30:45"))
        == "2026-01-28 08:30:45"
    )
    assert (
        market_balance_data._serialize_scalar(pd.Timestamp("2026-01-28 08:30:00"))
        == "2026-01-28 08:30"
    )


def test_build_country_conversion_factors_cte_falls_back_when_legacy_table_missing(monkeypatch):
    monkeypatch.setattr(
        market_balance_data,
        "_relation_exists",
        lambda relation_name, schema=market_balance_data.DB_SCHEMA: False,
    )

    cte_sql, warnings = market_balance_data.build_country_conversion_factors_cte()

    assert "fundamentals_ea_global_datasets" not in cte_sql
    assert "NULL::DOUBLE PRECISION AS conversion_factor_bcm_gas" in cte_sql
    assert warnings == [
        "Legacy EA conversion-factor table unavailable; country balance is using raw EA values."
    ]


def test_ensure_balance_hierarchy_columns_sanitizes_mixed_labels():
    df = pd.DataFrame(
        [[1.0, 2.0]],
        columns=pd.Index([("Supply", "Pipeline", ""), slice(None, None, None)], dtype=object),
    )

    normalized_df = market_balance_data.ensure_balance_hierarchy_columns(df)

    assert isinstance(normalized_df.columns, pd.MultiIndex)
    assert ("Supply", "Pipeline", "") in normalized_df.columns.tolist()
    assert ("", "", "") in normalized_df.columns.tolist()


def test_fetch_trade_balance_payload_survives_flex_source_error(monkeypatch):
    mapping_df = pd.DataFrame(
        {
            "country_name": ["United States", "Japan"],
            "country": ["United States", "Japan"],
            "continent": ["North America", "Asia"],
            "subcontinent": ["North America", "East Asia"],
            "basin": ["Atlantic", "Pacific"],
            "shipping_region": ["Atlantic", "Pacific"],
            "country_classification_level1": ["Americas", "Asia"],
            "country_classification": ["Americas", "Asia"],
        }
    )
    export_raw_df = pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "country_name": ["United States", "United States"],
            "total_mmtpa": [12.0, 12.0],
        }
    )
    import_raw_df = pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "country_name": ["Japan", "Japan"],
            "total_mmtpa": [10.0, 10.0],
        }
    )

    monkeypatch.setattr(market_balance_data, "fetch_country_mapping_df", lambda: mapping_df)
    monkeypatch.setattr(
        market_balance_data, "fetch_ea_export_flow_raw_data", lambda: export_raw_df
    )
    monkeypatch.setattr(
        market_balance_data, "fetch_ea_import_flow_raw_data", lambda: import_raw_df
    )
    monkeypatch.setattr(
        market_balance_data,
        "fetch_ea_export_flow_metadata",
        lambda: {"upload_timestamp_utc": "2024-03-01 00:00"},
    )
    monkeypatch.setattr(
        market_balance_data,
        "fetch_ea_import_flow_metadata",
        lambda: {"upload_timestamp_utc": "2024-03-01 00:00"},
    )
    monkeypatch.setattr(
        market_balance_data,
        "fetch_contract_volume_tables",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("contract source missing")),
    )

    payload = market_balance_data.fetch_trade_balance_payload(
        time_group="yearly",
        diff_type="absolute",
        country_group="country_classification",
        selected_years=[2024],
        unit="bcm",
    )

    assert payload["error"] is None
    assert payload["data"]["exports"]["records"]
    assert payload["data"]["imports"]["records"]
    assert payload["metadata"]["warnings"] == [
        "Flex volumes unavailable: contract source missing"
    ]


def test_build_workbook_bytes_returns_xlsx_payload():
    workbook_bytes = market_balance.build_workbook_bytes(
        {
            "Sheet One": pd.DataFrame({"A": [1, 2]}),
            "Sheet Two": pd.DataFrame({"B": ["x", "y"]}),
        }
    )

    assert workbook_bytes[:2] == b"PK"
    assert len(workbook_bytes) > 100


def test_export_overview_workbook_includes_maintenance_provider_gap(monkeypatch):
    captured = {}

    def fake_build_workbook_bytes(sheet_map):
        captured["sheet_map"] = sheet_map
        return b"xlsx"

    def fake_send_bytes(workbook_bytes, filename):
        captured["workbook_bytes"] = workbook_bytes
        captured["filename"] = filename
        return {"filename": filename}

    monkeypatch.setattr(market_balance, "build_workbook_bytes", fake_build_workbook_bytes)
    monkeypatch.setattr(market_balance.dcc, "send_bytes", fake_send_bytes)

    result = market_balance.export_overview_workbook(
        1,
        {
            "data": {
                "maintenance_provider_comparison": {
                    "records": [{"Period": "2024", "Delta": 1.0}],
                    "columns": ["Period", "Delta"],
                    "numeric_columns": ["Delta"],
                }
            }
        },
    )

    assert result == {"filename": "market_balance_overview.xlsx"}
    assert captured["workbook_bytes"] == b"xlsx"
    assert "Maintenance Provider Gap" in captured["sheet_map"]
    assert captured["sheet_map"]["Maintenance Provider Gap"].to_dict("records") == [
        {"Period": "2024", "Delta": 1.0}
    ]


def test_build_market_table_skips_blank_column_filter_queries():
    payload = {
        "records": [{"": 1.0, "Total": 2.0}],
        "columns": ["", "Total"],
        "numeric_columns": ["", "Total"],
    }

    table = market_balance._build_market_table(
        "test-table",
        payload,
        table_mode="delta",
    )

    filter_queries = [
        rule.get("if", {}).get("filter_query")
        for rule in table.style_data_conditional
        if isinstance(rule, dict)
    ]

    assert "{} > 0" not in filter_queries
    assert "{} < 0" not in filter_queries


def test_build_market_table_uses_one_decimal_and_responsive_widths():
    payload = {
        "records": [
            {
                "Period": "2024",
                "Total": 12.34,
                "Europe": -5.67,
                "Total Supply Pipeline": 1.23,
            }
        ],
        "columns": ["Period", "Total", "Europe", "Total Supply Pipeline"],
        "numeric_columns": ["Total", "Europe", "Total Supply Pipeline"],
    }

    table = market_balance._build_market_table("test-table", payload, table_mode="net")

    assert table.columns[1]["format"].to_plotly_json()["specifier"] == ".1f"
    assert table.columns[2]["format"].to_plotly_json()["specifier"] == ".1f"
    assert any(
        rule.get("if", {}).get("column_id") == "Period"
        and rule.get("width") == "82px"
        for rule in table.style_cell_conditional
    )
    assert any(
        rule.get("if", {}).get("column_id") == "Total"
        and rule.get("width") == "88px"
        for rule in table.style_cell_conditional
    )
    assert any(
        rule.get("if", {}).get("column_id") == "Total Supply Pipeline"
        and int(rule.get("width", "0px").removesuffix("px")) >= 160
        for rule in table.style_cell_conditional
    )
    assert not hasattr(table, "export_format")


def test_build_market_table_can_wrap_multi_word_headers_compactly():
    payload = {
        "records": [
            {
                "Date": "2026",
                "LNG Imports": -10.38,
                "Fcst Margin": -0.27,
                "Total Supply Pipeline": -0.6,
            }
        ],
        "columns": ["Date", "LNG Imports", "Fcst Margin", "Total Supply Pipeline"],
        "numeric_columns": ["LNG Imports", "Fcst Margin", "Total Supply Pipeline"],
    }

    table = market_balance._build_market_table(
        "test-country-table",
        payload,
        table_mode="delta",
        compact=True,
        wrap_multi_word_headers=True,
    )

    assert table.columns[1]["name"] == "LNG Imports"
    assert table.columns[2]["name"] == "Fcst Margin"
    assert table.columns[3]["name"] == "Total Supply Pipeline"
    assert any(
        rule.get("selector") == ".dash-header div"
        and "white-space: normal" in rule.get("rule", "")
        for rule in table.css
    )
    assert table.style_cell["fontSize"] == "11px"
    assert table.style_header["padding"] == "5px 4px"
    assert any(
        rule.get("if", {}).get("column_id") == "Total Supply Pipeline"
        and int(rule.get("width", "0px").removesuffix("px")) <= 132
        for rule in table.style_cell_conditional
    )


def test_market_balance_layout_has_expected_defaults():
    tabs = _find_component_by_id(market_balance.layout, "market-balance-tabs")
    overview_top_row = _find_component_by_id(
        market_balance.layout, "market-balance-overview-top-row"
    )
    overview_second_row = _find_component_by_id(
        market_balance.layout, "market-balance-overview-second-row"
    )
    provider_overview_section = _find_component_by_id(
        market_balance.layout, "market-balance-provider-overview-section"
    )
    maintenance_section = _find_component_by_id(
        market_balance.layout, "market-balance-maintenance-section"
    )
    maintenance_table = _find_component_by_id(
        market_balance.layout, "market-balance-maintenance-table"
    )
    maintenance_kpis = _find_component_by_id(
        market_balance.layout, "market-balance-maintenance-kpis"
    )
    maintenance_provider_table = _find_component_by_id(
        market_balance.layout, "market-balance-maintenance-provider-table"
    )
    maintenance_metric = _find_component_by_id(
        market_balance.layout, "market-balance-maintenance-metric"
    )
    pacific_section = _find_component_by_id(
        market_balance.layout, "market-balance-pacific-section"
    )
    sticky_status = _find_component_by_id(
        market_balance.layout, "market-balance-sticky-status"
    )
    date_range = _find_component_by_id(market_balance.layout, "market-balance-date-range")
    trade_time_group = _find_component_by_id(
        market_balance.layout, "market-balance-trade-time-group"
    )
    trade_unit = _find_component_by_id(market_balance.layout, "market-balance-trade-unit")
    trade_country_group = _find_component_by_id(
        market_balance.layout, "market-balance-trade-country-group"
    )
    country_level = _find_component_by_id(
        market_balance.layout, "market-balance-country-level"
    )
    woodmac_comparison_source = _find_component_by_id(
        market_balance.layout, "market-balance-overview-woodmac-comparison-source"
    )
    ea_comparison_source = _find_component_by_id(
        market_balance.layout, "market-balance-overview-ea-comparison-source"
    )
    woodmac_export_button = _find_component_by_id(
        market_balance.layout, "market-balance-overview-woodmac-export"
    )
    ea_export_button = _find_component_by_id(
        market_balance.layout, "market-balance-overview-ea-export"
    )

    assert tabs is not None
    assert overview_top_row is not None
    assert overview_second_row is not None
    assert provider_overview_section is not None
    assert maintenance_section is not None
    assert maintenance_table is not None
    assert maintenance_kpis is not None
    assert maintenance_provider_table is not None
    assert maintenance_metric is not None
    assert pacific_section is not None
    assert sticky_status is not None
    assert date_range is not None
    assert woodmac_comparison_source is not None
    assert ea_comparison_source is not None
    assert woodmac_export_button is not None
    assert ea_export_button is not None
    assert tabs.value == "overview"
    assert overview_second_row.style == {"display": "grid", "gap": "24px"}
    assert getattr(overview_second_row.children[1], "id", None) == "market-balance-provider-overview-section"
    assert getattr(overview_second_row.children[2], "id", None) == "market-balance-maintenance-section"
    assert getattr(overview_second_row.children[3], "id", None) == "market-balance-pacific-section"
    assert date_range.start_date == dt.date(dt.date.today().year - 3, 1, 1).isoformat()
    assert date_range.end_date == dt.date(dt.date.today().year + 5, 12, 31).isoformat()
    assert trade_time_group.value == "yearly"
    assert trade_unit.value == "bcm"
    assert trade_country_group.value == "country_classification_level1"
    assert maintenance_metric.value == "Unplanned"
    assert country_level.value == "subtype"


def test_overview_net_balance_section_uses_inline_export_header():
    section = market_balance._build_overview_net_balance_section(
        title="WoodMac Net Balance",
        export_button_id="test-export-button",
        baseline_summary_id="test-baseline-summary",
        comparison_source_dropdown_id="test-comparison-source",
        comparison_st_dropdown_id="test-comparison-st",
        comparison_lt_dropdown_id="test-comparison-lt",
        comparison_ea_upload_dropdown_id="test-comparison-ea-upload",
        comparison_woodmac_controls_id="test-woodmac-controls",
        comparison_ea_controls_id="test-ea-controls",
        baseline_table_container_id="test-baseline-table",
        comparison_summary_id="test-comparison-summary",
        comparison_table_container_id="test-comparison-table",
    )

    header = section.children[0]
    inline_header = header.children[0]

    assert header.className == "balance-section-header"
    assert inline_header.className == "inline-section-header"
    assert inline_header.children[1].id == "test-export-button"


def test_toggle_overview_top_row_only_shows_for_overview_tab():
    assert market_balance.toggle_overview_top_row("overview") == {
        "display": "grid",
        "gap": "24px",
        "marginBottom": "20px",
    }
    assert market_balance.toggle_overview_top_row("trade_balance") == {"display": "none"}


def test_initialize_market_balance_date_range_sets_runtime_defaults_once():
    start_date, end_date, initialized = market_balance.initialize_market_balance_date_range(
        "overview",
        None,
    )

    assert start_date == dt.date(dt.date.today().year - 3, 1, 1).isoformat()
    assert end_date == dt.date(dt.date.today().year + 5, 12, 31).isoformat()
    assert initialized is True

    start_date, end_date, initialized = market_balance.initialize_market_balance_date_range(
        "overview",
        True,
    )

    assert start_date is market_balance.no_update
    assert end_date is market_balance.no_update
    assert initialized is True


def test_render_sticky_status_summarizes_active_tab():
    result = market_balance.render_sticky_status(
        "trade_balance",
        "2024-01-01",
        "2024-12-01",
        "yearly",
        "bcm",
        "country_classification",
        {"metadata": {}},
        {
            "metadata": {
                "source": "Energy Aspects",
                "export_metadata": {"upload_timestamp_utc": "2024-02-01 00:00"},
                "import_metadata": {"upload_timestamp_utc": "2024-02-02 00:00"},
            }
        },
        {"metadata": {}},
    )

    assert "Active tab: Trade Balance" in result.children[0].children
    assert "Date Range: 2024-01 to 2024-12" in result.children[1].children
    assert "Trade source: Energy Aspects" in result.children[2].children


def test_render_trade_balance_handles_error_payload():
    result = market_balance.render_trade_balance({"error": "boom"})

    assert "boom" in result[1].children
    assert result[2].layout.title.text == "Trade balance unavailable"


def test_render_trade_balance_labels_flex_tables_for_yearly_view():
    payload = {
        "data": {
            "exports": {
                "records": [{"Period": "2024", "Total": 10.0}],
                "columns": ["Period", "Total"],
                "numeric_columns": ["Total"],
            },
            "imports": {
                "records": [{"Period": "2024", "Total": 9.0}],
                "columns": ["Period", "Total"],
                "numeric_columns": ["Total"],
            },
            "net": {
                "records": [{"Period": "2024", "Total": 1.0}],
                "columns": ["Period", "Total"],
                "numeric_columns": ["Total"],
            },
            "exports_diff": {
                "records": [{"Period": "2024", "Total": 0.0}],
                "columns": ["Period", "Total"],
                "numeric_columns": ["Total"],
            },
            "imports_diff": {
                "records": [{"Period": "2024", "Total": 0.0}],
                "columns": ["Period", "Total"],
                "numeric_columns": ["Total"],
            },
            "exports_flex": {
                "records": [{"Period": "2024", "Total": 2.0}],
                "columns": ["Period", "Total"],
                "numeric_columns": ["Total"],
            },
            "imports_flex": {
                "records": [{"Period": "2024", "Total": 1.0}],
                "columns": ["Period", "Total"],
                "numeric_columns": ["Total"],
            },
        },
        "metadata": {
            "unit": "bcm",
            "country_group_label": "Country",
            "time_group": "yearly",
            "source": "Energy Aspects",
            "warnings": [],
            "export_metadata": {"upload_timestamp_utc": "2024-01-01 00:00"},
            "import_metadata": {"upload_timestamp_utc": "2024-01-01 00:00"},
        },
        "error": None,
    }

    result = market_balance.render_trade_balance(payload)

    assert result[6].children[0].children == "Exports Flex Volumes (Actual - WoodMac Contracts)"
    assert result[9].children[0].children == "Imports Flex Volumes (Actual - WoodMac Contracts)"


def test_render_overview_surfaces_ea_maintenance_metadata():
    payload = {
        "data": {
            "woodmac_net_balance": {
                "records": [{"Period": "2024", "Total": 1.0}],
                "columns": ["Period", "Total"],
                "numeric_columns": ["Total"],
            },
            "ea_net_balance": {
                "records": [{"Period": "2024", "Total": 2.0}],
                "columns": ["Period", "Total"],
                "numeric_columns": ["Total"],
            },
            "woodmac_balance": {
                "records": [{"Period": "2024", "Supply": 10.0, "Demand": 8.0, "Delta": 2.0}],
                "columns": ["Period", "Supply", "Demand", "Delta"],
                "numeric_columns": ["Supply", "Demand", "Delta"],
            },
            "ea_balance": {
                "records": [{"Period": "2024", "Supply": 9.0, "Demand": 7.0, "Delta": 2.0}],
                "columns": ["Period", "Supply", "Demand", "Delta"],
                "numeric_columns": ["Supply", "Demand", "Delta"],
            },
            "maintenance": {"records": [], "columns": [], "numeric_columns": []},
            "maintenance_grouped": {
                "records": [
                    {
                        "Period": "2024",
                        "Metric": "Total",
                        "Total": 3.0,
                        "Europe": 3.0,
                    },
                    {
                        "Period": "2024",
                        "Metric": "Planned",
                        "Total": 1.0,
                        "Europe": 1.0,
                    },
                    {
                        "Period": "2024",
                        "Metric": "Unplanned",
                        "Total": 2.0,
                        "Europe": 2.0,
                    },
                ],
                "columns": ["Period", "Metric", "Total", "Europe"],
                "numeric_columns": ["Total", "Europe"],
            },
            "maintenance_ea": {
                "records": [{"Period": "2024", "Total": 2.0}],
                "columns": ["Period", "Total"],
                "numeric_columns": ["Total"],
            },
            "maintenance_provider_comparison": {
                "records": [
                    {
                        "Period": "2024",
                        "WoodMac Unplanned": 2.0,
                        "Energy Aspects Unplanned": 1.5,
                        "Delta": 0.5,
                        "Delta %": 33.3,
                        "WoodMac Planned": 1.0,
                        "WoodMac Total": 3.0,
                    }
                ],
                "columns": [
                    "Period",
                    "WoodMac Unplanned",
                    "Energy Aspects Unplanned",
                    "Delta",
                    "Delta %",
                    "WoodMac Planned",
                    "WoodMac Total",
                ],
                "numeric_columns": [
                    "WoodMac Unplanned",
                    "Energy Aspects Unplanned",
                    "Delta",
                    "Delta %",
                    "WoodMac Planned",
                    "WoodMac Total",
                ],
            },
            "pacific_detail": {"records": [], "columns": [], "numeric_columns": []},
            "pacific_totals": {"records": [], "columns": [], "numeric_columns": []},
        },
        "metadata": {
            "woodmac_export": {"short_term_publication_timestamp": "2024-01-01 00:00"},
            "woodmac_import": {"short_term_publication_timestamp": "2024-01-01 00:00"},
            "ea_export": {"upload_timestamp_utc": "2024-01-02 00:00"},
            "ea_import": {"upload_timestamp_utc": "2024-01-02 00:00"},
            "overview_net": {
                "country_group": "country_classification",
                "country_group_label": "Classification",
                "time_group": "yearly",
                "unit": "bcm",
            },
            "maintenance": {
                "ea_dataset": {
                    "dataset_id": 15522,
                    "unit": "Mt",
                    "aspect_subtype": "unplanned_outage",
                    "description": "Monthly EA forecast for unplanned outages at liquefaction plants in World in Mt",
                    "release_date": "2025-06-02T15:55:00",
                },
                "notes": ["WoodMac maintenance is converted from annualized MMTPA-style values to monthly Mt for comparison."],
            },
        },
        "error": None,
    }

    result = market_balance.render_overview("overview", payload)
    status_block = result[0]

    assert len(status_block.children) == 4
    assert "WoodMac exports snapshot" in status_block.children[0].children
    assert "Energy Aspects imports upload" in status_block.children[3].children
    assert "Current exports publication" in result[2].children[1].children
    assert result[3].id == "market-balance-overview-woodmac-net-table-grid"
    assert "Current exports upload_timestamp_utc" in result[4].children[1].children
    assert result[5].id == "market-balance-overview-ea-net-table-grid"
    assert result[6].layout.title.text == "Supply Overview"
    assert [trace.name for trace in result[6].data] == ["WoodMac", "Energy Aspects"]
    assert result[7].layout.title.text == "Demand Overview"
    assert [trace.name for trace in result[7].data] == ["WoodMac", "Energy Aspects"]
    assert len(result[8].children) == 5
    assert result[9].layout.title.text.startswith("Provider Outage Gap")
    assert [trace.name for trace in result[9].data[:2]] == [
        "WoodMac Unplanned",
        "Energy Aspects Unplanned",
    ]
    assert result[10].children[0].children == "Provider Outage Gap (Unplanned, Mt)"
    assert result[10].children[1].id == "market-balance-maintenance-provider-gap-table-grid"
    assert result[10].children[1].data == [
        {
            "Period": "2024",
            "WoodMac Unplanned": 2.0,
            "Energy Aspects Unplanned": 1.5,
            "Delta": 0.5,
            "Delta %": 33.3,
            "WoodMac Planned": 1.0,
            "WoodMac Total": 3.0,
        }
    ]
    assert result[11].children[0].children.children[0].children == "WoodMac Maintenance Detail by Classification"
    assert result[11].children[0].children.children[1].id == "market-balance-maintenance-table-grid"
    assert result[11].children[0].children.children[1].data == [
        {"Period": "2024", "Total": 2.0, "Europe": 2.0}
    ]


def test_render_overview_returns_lightweight_placeholders_when_tab_inactive():
    result = market_balance.render_overview("trade_balance", {"data": {}, "metadata": {}, "error": None})

    assert result[0].children is None
    assert result[2].children is None
    assert result[3].children is None
    assert result[4].children is None
    assert result[5].children is None
    assert result[6].layout.title.text == "Overview unavailable"
    assert result[8].children is None
    assert result[9].layout.title.text == "Overview unavailable"
    assert result[10].children is None
    assert result[11].children is None


def test_resolve_snapshot_control_values_defaults_to_previous_snapshot():
    comparison_options = {
        "woodmac": {
            "short_term": [
                {"market_outlook": "Apr", "publication_timestamp": "2024-04-01 00:00"},
                {"market_outlook": "Mar", "publication_timestamp": "2024-03-01 00:00"},
            ],
            "long_term": [
                {"market_outlook": "Apr LT", "publication_timestamp": "2024-04-01 00:00"},
                {"market_outlook": "Mar LT", "publication_timestamp": "2024-03-01 00:00"},
            ],
        },
        "ea_uploads": ["2024-04-10 00:00", "2024-03-10 00:00"],
    }

    result = market_balance._resolve_snapshot_control_values(
        "woodmac",
        comparison_options,
        None,
        None,
        None,
    )

    assert result[1] == market_balance._serialize_snapshot_value(
        comparison_options["woodmac"]["short_term"][1]
    )
    assert result[3] == market_balance._serialize_snapshot_value(
        comparison_options["woodmac"]["long_term"][1]
    )
    assert result[5] == "2024-03-10 00:00"
    assert result[6]["display"] == "flex"
    assert result[7]["display"] == "none"


def test_render_woodmac_overview_delta_builds_delta_table(monkeypatch):
    def _fake_fetch(**_kwargs):
        return pd.DataFrame(
            {
                "Period": ["2024"],
                "Total": [1.5],
                "Europe": [0.5],
            }
        )

    monkeypatch.setattr(
        market_balance,
        "fetch_net_balance_for_woodmac_publications",
        _fake_fetch,
    )

    store_payload = {
        "data": {
            "woodmac_net_balance": market_balance_data.serialize_frame(
                pd.DataFrame(
                    {
                        "Period": ["2024"],
                        "Total": [2.0],
                        "Europe": [1.0],
                    }
                )
            )
        },
        "metadata": {},
        "error": None,
    }

    summary, table = market_balance.render_woodmac_overview_delta(
        "overview",
        store_payload,
        "2024-01-01",
        "2024-12-31",
        "yearly",
        "bcm",
        "country_classification",
        "woodmac",
        market_balance._serialize_snapshot_value(
            {"market_outlook": "Apr", "publication_timestamp": "2024-04-01 00:00"}
        ),
        market_balance._serialize_snapshot_value(
            {"market_outlook": "Apr LT", "publication_timestamp": "2024-04-01 00:00"}
        ),
        None,
    )

    assert summary.children is None
    assert table.id == "market-balance-overview-woodmac-delta-grid"
    assert table.data == [{"Period": "2024", "Total": 0.5, "Europe": 0.5}]


def test_render_country_balance_accepts_structured_payload():
    payload = {
        "data": {
            "current_table": {
                "records": [{"Date": "2024-01", "Demand": 1.0, "Supply": 2.0, "Fcst Margin": 1.0}],
                "columns": ["Date", "Demand", "Supply", "Fcst Margin"],
                "numeric_columns": ["Demand", "Supply", "Fcst Margin"],
            },
            "delta_table": {
                "records": [{"Date": "2024-01", "Demand": 0.2, "Supply": 0.5, "Fcst Margin": 0.3}],
                "columns": ["Date", "Demand", "Supply", "Fcst Margin"],
                "numeric_columns": ["Demand", "Supply", "Fcst Margin"],
            },
            "balance_chart": {
                "records": [{"Date": "2024-01", "Demand": 1.0, "Supply": 2.0, "Fcst Margin": 1.0}],
                "columns": ["Date", "Demand", "Supply", "Fcst Margin"],
                "numeric_columns": ["Demand", "Supply", "Fcst Margin"],
            },
            "category_charts": [
                {
                    "title": "Japan LNG Imports",
                    "chart_type": "line",
                    "frame": {
                        "records": [{"Date": "2024-01", "value": 1.0}],
                        "columns": ["Date", "value"],
                        "numeric_columns": ["value"],
                    },
                }
            ],
        },
        "metadata": {
            "country": "Japan",
            "current_snapshot": "2024-02-01 00:00",
            "comparison_snapshot": "2024-01-01 00:00",
            "level": "subtype",
            "time_group": "monthly",
            "warnings": ["Legacy EA conversion-factor table unavailable; country balance is using raw EA values."],
            "column_styles": [],
        },
        "error": None,
    }

    result = market_balance.render_country_balance(payload)

    assert "Japan" in result[0].children[0].children
    assert "Legacy EA conversion-factor table unavailable" in result[0].children[-1].children
    assert result[4].layout.title.text == "Japan Balance"
    assert result[5]


def test_build_country_chart_payloads_uses_series_name_when_subsubtype_is_blank():
    raw_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "subtype": ["Pipeline", "Pipeline"],
            "subsubtype": ["", ""],
            "value": [1.0, 2.0],
        }
    )

    payloads = market_balance_data._build_country_chart_payloads(
        raw_df, country="Japan", time_group="monthly"
    )

    assert payloads
    frame = payloads[0]["frame"]
    assert "series_name" in frame["columns"]
    assert frame["records"][0]["series_name"] == "Pipeline"


def test_index_shipping_snd_includes_market_balance_route():
    index_file = PROJECT_ROOT / "dash_shipping_lng_snd" / "index_shipping_snd.py"
    content = index_file.read_text()

    assert "/market_balance" in content
    assert "nav-market-balance" in content
