from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading
import time

from dash import no_update
from dash._utils import to_json
import pandas as pd
import plotly.graph_objects as go
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


def _semantic_fingerprint(value):
    canonical = json.dumps(
        json.loads(to_json(value)),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
@pytest.mark.parametrize(
    "start_date,end_date,upload_timestamp",
    (
        ("2026-01-01", "2026-12-31", "2026-07-25T00:00:00"),
        ("2026-03-15", "2026-04-30", "2025-01-01T00:00:00"),
    ),
)
def test_staged_render_has_exact_legacy_semantic_fingerprint(
    monkeypatch,
    split_dimension,
    zone_filter,
    start_date,
    end_date,
    upload_timestamp,
):
    _patch_render_builders(monkeypatch)
    source_bundle = _source_bundle()
    source_bundle["context"].update({
        "split_dimension": split_dimension,
        "start_date": start_date,
        "end_date": end_date,
    })
    source_bundle["upload_timestamp"] = upload_timestamp
    source_bundle["signal_upload_timestamp"] = upload_timestamp
    reference = _source_reference(
        revision=(
            f"{split_dimension}:{start_date}:{end_date}:"
            f"{upload_timestamp}"
        )
    )
    figures = {
        name: go.Figure(
            data=[go.Scatter(
                x=[start_date, end_date],
                y=[index, index + 1],
                customdata=[[split_dimension], [zone_filter]],
                hovertemplate="%{customdata[0]}<extra></extra>",
            )],
            layout={"title": name},
        )
        for index, name in enumerate((
            "arrival_pipeline_fig",
            "utilization_fig",
            "congestion_signal_fig",
            "freight_signal_fig",
            "diversion_seasonal_fig",
            "loaded_seasonal_fig",
            "floating_seasonal_fig",
            "detail_matrix_fig",
        ))
    }
    common_render = {
        **_common_render(),
        **figures,
        "price_card": ("price", source_bundle["price_context"]),
        "detail_matrix_legend": ("legend", split_dimension),
    }
    monkeypatch.setattr(
        fleet_metrics,
        "_resolve_fleet_metrics_source_bundle",
        lambda _reference: source_bundle,
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_get_fleet_metrics_common_render",
        lambda *_args, **_kwargs: common_render,
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_build_fleet_metrics_common_render",
        lambda *_args, **_kwargs: common_render,
    )
    monkeypatch.setattr(
        fleet_metrics,
        "build_price_card",
        lambda context: ("price", context),
    )
    monkeypatch.setattr(
        fleet_metrics,
        "build_status_strip",
        lambda *args: ("status", *args),
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_detail_matrix_area_color_map",
        lambda _frame, region, split: {
            f"{region}:{split}": "#123456"
        },
    )
    monkeypatch.setattr(
        fleet_metrics,
        "build_region_detail_matrix_legend_from_color_maps",
        lambda _maps, split: ("legend", split),
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_fleet_region_only_triggered",
        lambda: False,
    )

    legacy = fleet_metrics.update_fleet_metrics_page(
        reference,
        zone_filter,
    )
    summary, signals, detail = (
        fleet_metrics._build_fleet_render_artifacts(
            reference,
            source_bundle,
        )
    )
    summary_outputs = fleet_metrics._fleet_summary_outputs_from_artifact(
        summary,
        zone_filter,
    )
    figure_outputs = fleet_metrics._fleet_figure_outputs_from_artifacts(
        signals,
        detail,
    )
    staged = (
        *summary_outputs[:8],
        *figure_outputs[:5],
        summary_outputs[8],
        *figure_outputs[5:],
    )
    artifacts = {
        "summary": summary,
        "signals": signals,
        "detail": detail,
    }
    monkeypatch.setattr(
        fleet_metrics,
        "_get_or_build_fleet_render_bundle",
        lambda _reference: {"bundle": "reference"},
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_resolve_fleet_render_artifact",
        lambda _bundle, section: artifacts[section],
    )
    render_snapshot = (
        fleet_metrics.update_fleet_metrics_page_from_render_snapshot(
            reference,
            zone_filter,
        )
    )

    assert len(legacy) == len(staged) == len(render_snapshot) == 18
    legacy_fingerprints = [
        _semantic_fingerprint(value) for value in legacy
    ]
    assert [
        _semantic_fingerprint(value) for value in staged
    ] == legacy_fingerprints
    assert [
        _semantic_fingerprint(value) for value in render_snapshot
    ] == legacy_fingerprints


def test_staged_render_error_contract_preserves_all_outputs():
    summary = fleet_metrics._fleet_summary_error_outputs(
        RuntimeError("stale source")
    )
    figures = fleet_metrics._fleet_figure_error_outputs(
        RuntimeError("stale source")
    )

    assert len(summary) == 9
    assert len(figures) == 9
    assert all(value is not no_update for value in summary + figures)


def test_staged_signal_and_detail_callbacks_resolve_independently(monkeypatch):
    resolved_sections = []

    def resolve(_bundle_reference, section):
        resolved_sections.append(section)
        return {"section": section}

    monkeypatch.setattr(
        fleet_metrics,
        "_resolve_fleet_render_artifact",
        resolve,
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_fleet_signal_outputs_from_artifact",
        lambda artifact: (artifact["section"],) * 5,
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_fleet_detail_outputs_from_artifact",
        lambda artifact: (artifact["section"],) * 4,
    )

    signals = fleet_metrics.update_fleet_metrics_signals(
        {"bundle": "reference"}
    )
    assert signals == ("signals",) * 5
    assert resolved_sections == ["signals"]

    resolved_sections.clear()
    detail = fleet_metrics.update_fleet_metrics_detail(
        {"bundle": "reference"}
    )
    assert detail == ("detail",) * 4
    assert resolved_sections == ["detail"]


def test_render_snapshot_region_switch_preserves_exact_no_update_contract(
    monkeypatch,
):
    summary_artifact = {
        "summaries": {},
        "signal_summaries": {},
        "price_context": {},
        "movers_by_region": {},
    }
    monkeypatch.setattr(
        fleet_metrics,
        "_get_or_build_fleet_render_bundle",
        lambda _reference: {"bundle": "reference"},
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_resolve_fleet_render_artifact",
        lambda _bundle, section: summary_artifact,
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_fleet_summary_outputs_from_artifact",
        lambda *_args: tuple(range(9)),
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_fleet_figure_outputs_from_artifacts",
        lambda *_args: tuple(range(9, 18)),
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_fleet_region_only_triggered",
        lambda: True,
    )

    result = fleet_metrics.update_fleet_metrics_page_from_render_snapshot(
        _source_reference(),
        "europe_basin",
    )

    assert {
        index
        for index, value in enumerate(result)
        if value is not no_update
    } == fleet_metrics._FLEET_REGION_OUTPUT_INDICES


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

    result, refresh_status = fleet_metrics.load_fleet_metrics_source(
        "current_subcontinents",
        "2026-01-01",
        "2026-12-31",
        0,
    )

    assert {
        key: result[key]
        for key in ("format", "namespace", "source_key", "revision", "shared")
    } == reference
    assert refresh_status["kpler_freshness"]["fleet_checked_at"] == (
        "2026-07-25T00:00:00"
    )
    assert len(str(result).encode("utf-8")) < 10_000
    assert refresh_status["status"] == "checked"
    assert refresh_status["refresh_generation"] == 0


def test_noop_check_refreshes_status_without_changing_data_cache_key(monkeypatch):
    states = iter(
        (
            {
                "fleet_revision": 12,
                "signal_revision": 18,
                "fleet_checked_at": "2026-07-31T10:00:00+00:00",
                "signal_checked_at": "2026-07-31T10:01:00+00:00",
            },
            {
                "fleet_revision": 12,
                "signal_revision": 18,
                "fleet_checked_at": "2026-07-31T11:00:00+00:00",
                "signal_checked_at": "2026-07-31T11:01:00+00:00",
            },
        )
    )
    source_keys = []
    build_count = 0

    monkeypatch.setattr(
        fleet_metrics,
        "_fetch_fleet_metrics_source_state",
        lambda: next(states),
    )

    def snapshot(_engine, *, source_key, **_kwargs):
        nonlocal build_count
        build_count += 1
        source_keys.append(source_key)
        return {
            **_source_reference(),
            "source_key": source_key,
        }, _source_bundle()

    monkeypatch.setattr(fleet_metrics, "_get_or_build_snapshot", snapshot)
    monkeypatch.setattr(fleet_metrics, "_snapshot_is_resolvable", lambda _value: True)
    monkeypatch.setattr(fleet_metrics, "_was_global_refresh_triggered", lambda: False)

    first_reference, first_status = fleet_metrics.load_fleet_metrics_source(
        "current_subcontinents", "2026-01-01", "2026-12-31", 0
    )
    second_reference, second_status = fleet_metrics.load_fleet_metrics_source(
        "current_subcontinents",
        "2026-01-01",
        "2026-12-31",
        1,
        first_reference,
    )

    assert build_count == 1
    assert len(source_keys) == 1
    assert second_reference is no_update
    assert first_status["kpler_freshness"]["fleet_checked_at"] != (
        second_status["kpler_freshness"]["fleet_checked_at"]
    )


def test_changed_revision_builds_exactly_one_new_generation(monkeypatch):
    states = iter(
        (
            {"fleet_revision": 12, "signal_revision": 18},
            {"fleet_revision": 13, "signal_revision": 18},
        )
    )
    source_keys = []

    monkeypatch.setattr(
        fleet_metrics,
        "_fetch_fleet_metrics_source_state",
        lambda: next(states),
    )

    def snapshot(_engine, *, source_key, **_kwargs):
        source_keys.append(source_key)
        return {
            **_source_reference(revision=f"revision-{len(source_keys)}"),
            "source_key": source_key,
        }, _source_bundle()

    monkeypatch.setattr(fleet_metrics, "_get_or_build_snapshot", snapshot)
    monkeypatch.setattr(
        fleet_metrics,
        "_snapshot_is_resolvable",
        lambda _value: True,
    )
    monkeypatch.setattr(
        fleet_metrics,
        "_was_global_refresh_triggered",
        lambda: False,
    )

    first_reference, _ = fleet_metrics.load_fleet_metrics_source(
        "current_subcontinents", "2026-01-01", "2026-12-31", 0
    )
    second_reference, _ = fleet_metrics.load_fleet_metrics_source(
        "current_subcontinents",
        "2026-01-01",
        "2026-12-31",
        1,
        first_reference,
    )

    assert len(source_keys) == 2
    assert source_keys[0] != source_keys[1]
    assert second_reference is not no_update
    assert second_reference["revision"] == "revision-2"


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
