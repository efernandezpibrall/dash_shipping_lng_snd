import pandas as pd

from pages import supply
from utils import market_balance_data


def test_supply_shared_provider_failure_uses_legacy_queries(monkeypatch):
    woodmac = pd.DataFrame(
        {"month": pd.to_datetime(["2025-01-01"]), "country_name": ["Qatar"], "total_mmtpa": [1.0]}
    )
    ea = pd.DataFrame(
        {"month": pd.to_datetime(["2025-01-01"]), "country_name": ["Qatar"], "total_mmtpa": [2.0]}
    )
    mapping_records = [{"country": "Qatar", "country_name": "Qatar"}]

    monkeypatch.setattr(
        supply,
        "_get_provider_flow_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("shared unavailable")),
    )
    monkeypatch.setattr(
        supply,
        "_build_provider_flow_payload",
        lambda: (
            {"current_ea": {"run_id": 42}},
            {
                "woodmac_export": woodmac,
                "ea_export": ea,
                "woodmac_export_options": {"short_term": [], "long_term": []},
                "ea_comparison_runs": [],
                "current_ea": {"run_id": 42, "snapshot_at": "2026-07-16T00:00:00Z"},
                "mapping": pd.DataFrame(mapping_records),
                "errors": {},
            },
        ),
    )

    result = supply.load_balance_source_data(None)

    pd.testing.assert_frame_equal(
        supply._deserialize_dataframe(result[0]),
        supply._deserialize_dataframe(supply._serialize_dataframe(woodmac)),
    )
    pd.testing.assert_frame_equal(
        supply._deserialize_dataframe(result[1]),
        supply._deserialize_dataframe(supply._serialize_dataframe(ea)),
    )
    assert result[3] == supply._build_destination_aggregation_lookup_records(
        pd.DataFrame(mapping_records)
    )
    assert result[4] is None


def test_market_provider_failure_uses_original_source_functions(monkeypatch):
    frames = {
        "woodmac_export": pd.DataFrame({"value": [1]}),
        "woodmac_import": pd.DataFrame({"value": [2]}),
        "ea_export": pd.DataFrame({"value": [3]}),
        "ea_import": pd.DataFrame({"value": [4]}),
        "mapping": pd.DataFrame({"country_name": ["Qatar"]}),
    }
    monkeypatch.setattr(
        market_balance_data,
        "get_provider_flow_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("shared unavailable")),
    )
    monkeypatch.setattr(
        market_balance_data,
        "build_provider_flow_payload",
        lambda: (
            {"current_ea": {"run_id": 42}},
            {
                **frames,
                "woodmac_export_options": {"short_term": [], "long_term": []},
                "woodmac_import_options": {"short_term": [], "long_term": []},
                "ea_export_options": [],
                "ea_import_options": [],
                "ea_comparison_runs": [],
                "current_ea": {"run_id": 42, "snapshot_at": "2026-07-16T00:00:00Z"},
                "errors": {},
            },
        ),
    )

    payload = market_balance_data._resolve_latest_provider_flow_payload()

    for key, expected in frames.items():
        pd.testing.assert_frame_equal(payload[key], expected)
    assert payload["errors"] == {}
