from threading import Barrier, Event, Lock, enumerate as enumerate_threads

import pandas as pd
import pytest

from utils import market_balance_data


def _mapping_frame():
    return pd.DataFrame(
        {
            "country_name": ["Qatar", "United Kingdom"],
            "continent": ["Asia", "Europe"],
            "subcontinent": ["Western Asia", "Northern Europe"],
            "basin": ["Pacific", "Atlantic"],
            "shipping_region": ["Middle East", "NWE"],
            "country_classification_level1": ["Exporter", "Importer"],
            "country_classification": ["Exporter", "Importer"],
        }
    )


def _flow_frame(country, values):
    return pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "country_name": [country, country],
            "total_mmtpa": values,
        }
    )


@pytest.mark.parametrize("provider", ("woodmac", "ea"))
def test_comparison_fetches_overlap_and_match_sequential_output(
    monkeypatch,
    provider,
):
    barrier = Barrier(3, timeout=2)
    active = 0
    peak_active = 0
    active_lock = Lock()
    mapping_df = _mapping_frame()
    export_df = _flow_frame("Qatar", [12.0, 14.0])
    import_df = _flow_frame("United Kingdom", [8.0, 9.0])

    def concurrent_result(value):
        nonlocal active, peak_active
        with active_lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            barrier.wait()
            return value.copy(deep=True)
        finally:
            with active_lock:
                active -= 1

    monkeypatch.setattr(
        market_balance_data,
        "fetch_country_mapping_df",
        lambda: concurrent_result(mapping_df),
    )
    if provider == "woodmac":
        monkeypatch.setattr(
            market_balance_data,
            "fetch_woodmac_export_flow_raw_data_for_publications",
            lambda *_args: concurrent_result(export_df),
        )
        monkeypatch.setattr(
            market_balance_data,
            "fetch_woodmac_import_flow_raw_data_for_publications",
            lambda *_args: concurrent_result(import_df),
        )
        result = market_balance_data.fetch_net_balance_for_woodmac_publications(
            short_term_market_outlook="ST",
            short_term_publication_timestamp="2026-07-20T00:00:00",
            long_term_market_outlook="LT",
            long_term_publication_timestamp="2026-07-01T00:00:00",
            country_group="country",
            time_group="monthly",
            unit="mt",
        )
    else:
        prewarmed = Event()

        def prewarm(*_args):
            prewarmed.set()
            return "", "resolved_lng_balance_datasets"

        def checked_result(value):
            assert prewarmed.is_set()
            return concurrent_result(value)

        monkeypatch.setattr(
            market_balance_data,
            "build_resolved_ea_lng_balance_ctes",
            prewarm,
        )
        monkeypatch.setattr(
            market_balance_data,
            "fetch_country_mapping_df",
            lambda: checked_result(mapping_df),
        )
        monkeypatch.setattr(
            market_balance_data,
            "fetch_ea_export_flow_raw_data_for_upload",
            lambda *_args, **_kwargs: checked_result(export_df),
        )
        monkeypatch.setattr(
            market_balance_data,
            "fetch_ea_import_flow_raw_data_for_upload",
            lambda *_args, **_kwargs: checked_result(import_df),
        )
        result = market_balance_data.fetch_net_balance_for_ea_upload(
            ea_as_of_run_id=42,
            country_group="country",
            time_group="monthly",
            unit="mt",
        )

    expected = market_balance_data._build_provider_net_balance_table(
        export_df,
        import_df,
        mapping_df=mapping_df,
        country_group="country",
        time_group="monthly",
        unit="mt",
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    assert peak_active == 3
    assert active == 0
    assert not any(
        thread.name.startswith("market-balance-comparison")
        for thread in enumerate_threads()
    )


def test_comparison_preserves_deterministic_failure_order(monkeypatch):
    barrier = Barrier(3, timeout=2)

    def fail(message):
        barrier.wait()
        raise RuntimeError(message)

    monkeypatch.setattr(
        market_balance_data,
        "fetch_country_mapping_df",
        lambda: fail("mapping failed"),
    )
    monkeypatch.setattr(
        market_balance_data,
        "fetch_woodmac_export_flow_raw_data_for_publications",
        lambda *_args: fail("exports failed"),
    )
    monkeypatch.setattr(
        market_balance_data,
        "fetch_woodmac_import_flow_raw_data_for_publications",
        lambda *_args: fail("imports failed"),
    )

    with pytest.raises(RuntimeError, match="mapping failed"):
        market_balance_data.fetch_net_balance_for_woodmac_publications(
            short_term_market_outlook="ST",
            short_term_publication_timestamp="2026-07-20T00:00:00",
            long_term_market_outlook="LT",
            long_term_publication_timestamp="2026-07-01T00:00:00",
            country_group="country",
            time_group="monthly",
            unit="mt",
        )

    assert not any(
        thread.name.startswith("market-balance-comparison")
        for thread in enumerate_threads()
    )
