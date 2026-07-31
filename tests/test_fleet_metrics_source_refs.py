from concurrent.futures import ThreadPoolExecutor
import threading
import time

from dash import no_update
import pandas as pd
import pytest

from pages import fleet_metrics


def _source_reference(revision="opaque-revision-token"):
    return {
        "format": "dashboard_source_ref_v1",
        "namespace": fleet_metrics.FLEET_METRICS_SNAPSHOT_NAMESPACE,
        "source_key": "fleet-source",
        "revision": revision,
        "shared": True,
    }


def _source_bundle():
    return {
        "context": {
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "split_dimension": "current_subcontinents",
            "today": "2026-07-26",
        },
        "area_options_by_region": {
            zone_filter: [f"{zone_filter}-area"]
            for zone_filter in fleet_metrics.KPLER_FLEET_REGION_ORDER
        },
        "price_context": {},
        "upload_timestamp": "2026-07-25T00:00:00",
        "signal_upload_timestamp": "2026-07-25T00:00:00",
        "global_area_weekly": pd.DataFrame(),
    }


def _common_render():
    marker = object()
    return {
        "summaries": {
            zone_filter: {"zone_filter": zone_filter}
            for zone_filter in fleet_metrics.KPLER_FLEET_REGION_ORDER
        },
        "signal_summaries": {
            zone_filter: {"zone_filter": zone_filter}
            for zone_filter in fleet_metrics.KPLER_FLEET_REGION_ORDER
        },
        "price_card": marker,
        "loaded_seasonal_fig": marker,
        "floating_seasonal_fig": marker,
        "arrival_pipeline_fig": marker,
        "utilization_fig": marker,
        "congestion_signal_fig": marker,
        "freight_signal_fig": marker,
        "diversion_seasonal_fig": marker,
        "detail_matrix_weekly": pd.DataFrame(
            columns=["zone_filter"]
        ),
        "detail_matrix_fig": marker,
        "detail_matrix_legend": marker,
    }


@pytest.fixture(autouse=True)
def clear_render_caches():
    with fleet_metrics._FLEET_RENDER_CACHE_LOCK:
        fleet_metrics._FLEET_RENDER_CACHE.clear()
        fleet_metrics._FLEET_COMMON_RENDER_CACHE.clear()
        fleet_metrics._FLEET_COMMON_RENDER_FLIGHTS.clear()
    yield
    with fleet_metrics._FLEET_RENDER_CACHE_LOCK:
        fleet_metrics._FLEET_RENDER_CACHE.clear()
        fleet_metrics._FLEET_COMMON_RENDER_CACHE.clear()
        fleet_metrics._FLEET_COMMON_RENDER_FLIGHTS.clear()


def _patch_render_builders(monkeypatch):
    monkeypatch.setattr(
        fleet_metrics,
        "_resolve_fleet_metrics_source_bundle",
        lambda _reference: _source_bundle(),
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_get_fleet_metrics_common_render",
        lambda *_args, **_kwargs: _common_render(),
    )
    monkeypatch.setattr(
        fleet_metrics,
        "build_status_strip",
        lambda *_args: "status",
    )
    monkeypatch.setattr(
        fleet_metrics,
        "build_summary_cards",
        lambda summary: ("summary", summary),
    )
    monkeypatch.setattr(
        fleet_metrics,
        "build_comparison_rows",
        lambda _summaries, zone: [f"comparison-{zone}"],
    )
    monkeypatch.setattr(
        fleet_metrics,
        "build_compact_column_defs",
        lambda columns, rows: [len(columns), len(rows)],
    )
    monkeypatch.setattr(
        fleet_metrics,
        "build_global_signal_cards",
        lambda summary: ("signals", summary),
    )
    monkeypatch.setattr(
        fleet_metrics,
        "build_global_signal_rows",
        lambda _summaries, zone: [f"signal-{zone}"],
    )
    monkeypatch.setattr(
        fleet_metrics,
        "build_movers_rows",
        lambda _frame: ["mover"],
    )


def test_region_only_render_updates_exact_six_outputs_including_cache_hit(
    monkeypatch,
):
    _patch_render_builders(monkeypatch)
    monkeypatch.setattr(
        fleet_metrics,
        "_fleet_region_only_triggered",
        lambda: False,
    )
    reference = _source_reference()
    full_result = fleet_metrics.update_fleet_metrics_page(
        reference,
        "europe_basin",
    )
    assert len(full_result) == 18
    assert all(value is not no_update for value in full_result)

    monkeypatch.setattr(
        fleet_metrics,
        "_fleet_region_only_triggered",
        lambda: True,
    )
    cached_result = fleet_metrics.update_fleet_metrics_page(
        reference,
        "europe_basin",
    )
    assert len(cached_result) == 18
    assert {
        index
        for index, value in enumerate(cached_result)
        if value is not no_update
    } == fleet_metrics._FLEET_REGION_OUTPUT_INDICES


@pytest.mark.parametrize(
    "split_dimension",
    (
        "current_subcontinents",
        "current_basins",
        "current_shipping_regions",
    ),
)
@pytest.mark.parametrize(
    "zone_filter",
    fleet_metrics.KPLER_FLEET_REGION_ORDER,
)
def test_region_only_render_matches_full_render_for_every_region_and_split(
    monkeypatch,
    split_dimension,
    zone_filter,
):
    _patch_render_builders(monkeypatch)
    bundle = _source_bundle()
    bundle["context"]["split_dimension"] = split_dimension
    monkeypatch.setattr(
        fleet_metrics,
        "_resolve_fleet_metrics_source_bundle",
        lambda _reference: bundle,
    )
    reference = _source_reference(
        revision=f"{split_dimension}-revision"
    )

    monkeypatch.setattr(
        fleet_metrics,
        "_fleet_region_only_triggered",
        lambda: False,
    )
    full_result = fleet_metrics.update_fleet_metrics_page(
        reference,
        zone_filter,
    )

    monkeypatch.setattr(
        fleet_metrics,
        "_fleet_region_only_triggered",
        lambda: True,
    )
    region_result = fleet_metrics.update_fleet_metrics_page(
        reference,
        zone_filter,
    )

    assert {
        index
        for index, value in enumerate(region_result)
        if value is not no_update
    } == fleet_metrics._FLEET_REGION_OUTPUT_INDICES
    for index in fleet_metrics._FLEET_REGION_OUTPUT_INDICES:
        assert region_result[index] == full_result[index]


@pytest.mark.parametrize("worker_count", (1, 4, 8))
def test_common_render_is_single_flight(monkeypatch, worker_count):
    build_count = 0
    build_lock = threading.Lock()

    def build(*_args, **_kwargs):
        nonlocal build_count
        with build_lock:
            build_count += 1
        time.sleep(0.03)
        return {"built": True}

    monkeypatch.setattr(
        fleet_metrics,
        "_build_fleet_metrics_common_render",
        build,
    )

    def load():
        return fleet_metrics._get_fleet_metrics_common_render(
            ("fleet-source", "opaque-revision-token"),
            {},
            start_date_val=pd.Timestamp("2026-01-01").date(),
            end_date_val=pd.Timestamp("2026-12-31").date(),
            split_dimension="current_subcontinents",
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(lambda _index: load(), range(worker_count)))

    assert build_count == 1
    assert results == [{"built": True}] * worker_count


def test_area_options_resolve_from_source_without_sql(monkeypatch):
    monkeypatch.setattr(
        fleet_metrics,
        "_resolve_fleet_metrics_source_bundle",
        lambda _reference: _source_bundle(),
    )
    monkeypatch.setattr(
        fleet_metrics,
        "fetch_area_options",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("region switch must not query SQL")
        ),
    )
    options, values = fleet_metrics.update_area_options(
        _source_reference(),
        "americas_basin",
    )

    assert options == [
        {
            "label": "americas_basin-area",
            "value": "americas_basin-area",
        }
    ]
    assert values == ["americas_basin-area"]


def test_source_callback_publishes_only_small_reference(monkeypatch):
    reference = _source_reference()
    monkeypatch.setattr(
        fleet_metrics,
        "_fetch_fleet_metrics_source_state",
        lambda: {"fleet_upload": "2026-07-25T00:00:00"},
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_get_or_build_snapshot",
        lambda *_args, **_kwargs: (reference, _source_bundle()),
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_snapshot_is_resolvable",
        lambda value: value == reference,
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_was_global_refresh_triggered",
        lambda: False,
    )

    result = fleet_metrics.load_fleet_metrics_source(
        "current_subcontinents",
        "2026-01-01",
        "2026-12-31",
        0,
    )

    assert result == reference
    assert len(str(result).encode("utf-8")) < 10_000


def test_render_error_returns_all_outputs(monkeypatch):
    monkeypatch.setattr(
        fleet_metrics,
        "_resolve_fleet_metrics_source_bundle",
        lambda _reference: (_ for _ in ()).throw(
            RuntimeError("unavailable")
        ),
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_fleet_region_only_triggered",
        lambda: True,
    )
    result = fleet_metrics.update_fleet_metrics_page(
        _source_reference(),
        "europe_basin",
    )

    assert len(result) == 18
    assert all(value is not no_update for value in result)


def test_source_resolver_rejects_wrong_namespace_and_corrupt_reference(
    monkeypatch,
):
    monkeypatch.setattr(
        fleet_metrics,
        "_snapshot_is_resolvable",
        lambda _value: True,
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_resolve_snapshot",
        lambda *_args, **_kwargs: _source_bundle(),
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        fleet_metrics._resolve_fleet_metrics_source_bundle(
            {
                **_source_reference(),
                "namespace": "wrong-namespace",
            }
        )
    with pytest.raises(RuntimeError, match="unavailable"):
        fleet_metrics._resolve_fleet_metrics_source_bundle(
            {
                "format": "corrupt",
                "namespace": fleet_metrics.FLEET_METRICS_SNAPSHOT_NAMESPACE,
                "source_key": "fleet-source",
                "revision": "opaque",
                "shared": True,
            }
        )


def test_malformed_area_query_is_isolated_to_defaults(monkeypatch):
    monkeypatch.setattr(
        fleet_metrics,
        "_fleet_metrics_table_exists",
        lambda: True,
    )
    monkeypatch.setattr(
        fleet_metrics.pd,
        "read_sql",
        lambda *_args, **_kwargs: pd.DataFrame({"unexpected": [1]}),
    )

    result = fleet_metrics.fetch_all_area_options(
        "current_subcontinents"
    )

    assert set(result) == set(fleet_metrics.KPLER_FLEET_REGION_ORDER)
    for zone_filter in fleet_metrics.KPLER_FLEET_REGION_ORDER:
        assert result[zone_filter] == list(
            fleet_metrics._default_area_candidates(
                zone_filter,
                "current_subcontinents",
            )
        )


def test_common_render_failure_releases_waiters_and_allows_retry(
    monkeypatch,
):
    build_count = 0
    build_lock = threading.Lock()

    def build(*_args, **_kwargs):
        nonlocal build_count
        with build_lock:
            build_count += 1
            current_count = build_count
        time.sleep(0.05)
        if current_count == 1:
            raise RuntimeError("render failed")
        return {"built": current_count}

    monkeypatch.setattr(
        fleet_metrics,
        "_build_fleet_metrics_common_render",
        build,
    )

    def load():
        return fleet_metrics._get_fleet_metrics_common_render(
            ("fleet-source", "failure-retry"),
            {},
            start_date_val=pd.Timestamp("2026-01-01").date(),
            end_date_val=pd.Timestamp("2026-12-31").date(),
            split_dimension="current_subcontinents",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(load) for _ in range(4)]
        for future in futures:
            with pytest.raises(RuntimeError, match="render failed"):
                future.result()

    assert build_count == 1
    assert load() == {"built": 2}
    assert build_count == 2
