import pandas as pd
import pytest
from dash._callback import GLOBAL_CALLBACK_MAP
from dash.exceptions import PreventUpdate

import index_shipping_snd
from pages import capacity


SOURCE_KEY = "capacity-source-2026-07-26"
BASE_REVISION = "2026-07-25T00:00:00"


def _source_ref(component, source_key=SOURCE_KEY):
    return {
        "format": capacity.CAPACITY_SOURCE_REF_FORMAT,
        "source_key": source_key,
        "snapshot_key": source_key,
        "component": component,
    }


def _source_revision(source_key=SOURCE_KEY):
    return {
        "format": "capacity_source_render_revision_v1",
        "source_key": source_key,
    }


def _scenario_state():
    selected = 7
    options = [
        {
            "scenario_id": selected,
            "scenario_name": "Base Case",
            "current_snapshot_timestamp_utc": BASE_REVISION,
        }
    ]
    working = {
        "format": capacity.CAPACITY_SCENARIO_WORKING_FORMAT,
        "scenario_id": selected,
        "base_revision": BASE_REVISION,
        "mode": "clean",
        "upserts": None,
        "deleted_keys": [],
        "row_order": [],
        "full_override": None,
    }
    revision = {
        "format": "capacity_scenario_render_revision_v1",
        "source_key": SOURCE_KEY,
        "scenario_id": str(selected),
        "base_revision": BASE_REVISION,
        "working_mode": "clean",
        "dirty": False,
        "dirty_source": "",
        "dirty_updated_at": "",
        "refresh_revision": "",
        "option_name": "Base Case",
        "option_revision": BASE_REVISION,
    }
    return revision, selected, working, options


def _walk(component):
    if component is None:
        return
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk(child)
        return
    yield component
    yield from _walk(getattr(component, "children", None))


def test_capacity_source_revision_requires_all_four_exact_components():
    woodmac = _source_ref("woodmac_store")
    train = _source_ref("train_store")
    ea = _source_ref("ea_store")
    metadata = {"snapshot_key": SOURCE_KEY}

    assert capacity._capacity_source_revision_is_coherent(
        _source_revision(),
        woodmac,
        train,
        ea,
        metadata,
    )
    assert not capacity._capacity_source_revision_is_coherent(
        _source_revision(),
        woodmac,
        _source_ref("train_store", "different-source"),
        ea,
        metadata,
    )
    assert not capacity._capacity_source_revision_is_coherent(
        _source_revision(),
        woodmac,
        train,
        ea,
        {},
    )


def test_capacity_scenario_revision_requires_selected_working_option_parity():
    revision, selected, working, options = _scenario_state()

    assert capacity._capacity_scenario_revision_is_coherent(
        revision,
        selected,
        working,
        {"dirty": False},
        "",
        options,
    )
    assert not capacity._capacity_scenario_revision_is_coherent(
        revision,
        selected + 1,
        working,
        {"dirty": False},
        "",
        options,
    )
    assert not capacity._capacity_scenario_revision_is_coherent(
        revision,
        selected,
        {**working, "base_revision": "different"},
        {"dirty": False},
        "",
        options,
    )
    assert not capacity._capacity_scenario_revision_is_coherent(
        revision,
        selected,
        working,
        {"dirty": False},
        "",
        [
            {
                **options[0],
                "current_snapshot_timestamp_utc": "different",
            }
        ],
    )


def test_empty_scenario_revision_clears_a_removed_selection():
    revision = {
        "format": "capacity_scenario_render_revision_v1",
        "source_key": SOURCE_KEY,
        "scenario_id": None,
        "empty": True,
        "dirty": False,
        "dirty_source": "",
        "dirty_updated_at": "",
        "refresh_revision": "removed",
    }

    assert capacity._capacity_scenario_revision_is_coherent(
        revision,
        None,
        None,
        {"dirty": False},
        "removed",
        [],
    )
    assert not capacity._capacity_scenario_revision_is_coherent(
        revision,
        None,
        {"format": capacity.CAPACITY_SCENARIO_WORKING_FORMAT},
        {"dirty": False},
        "removed",
        [],
    )


def test_mismatched_source_revision_stops_before_heavy_render(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "_get_prepared_country_capacity_view",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("heavy render should not run")
        ),
    )

    with pytest.raises(PreventUpdate):
        capacity.render_capacity_table(
            _source_revision(),
            ["Qatar"],
            "exclude",
            "2026-01-01",
            "2026-12-31",
            "monthly",
            "detail",
            _source_ref("woodmac_store"),
            _source_ref("train_store", "different-source"),
            _source_ref("ea_store"),
            {"snapshot_key": SOURCE_KEY},
        )


def test_capacity_layout_and_callback_topology_use_two_revision_stores():
    component_ids = {
        getattr(component, "id", None)
        for component in _walk(capacity.layout)
    }
    assert "capacity-page-source-render-revision-store" in component_ids
    assert "capacity-page-scenario-render-revision-store" in component_ids

    callback_map = {
        **GLOBAL_CALLBACK_MAP,
        **index_shipping_snd.app.callback_map,
    }
    assert "capacity-page-source-render-revision-store.data" in callback_map
    assert "capacity-page-scenario-render-revision-store.data" in callback_map

    source_markers = (
        "capacity-page-woodmac-chart.figure",
        "capacity-page-ea-chart.figure",
        "capacity-page-unmapped-train-summary.children",
    )
    scenario_markers = (
        "capacity-page-yearly-capacity-comparison-chart.figure",
        "capacity-page-yearly-woodmac-capacity-discrepancy-table-container.children",
        "capacity-page-internal-scenario-chart.figure",
        "capacity-page-train-change-summary.children",
        "capacity-page-train-timeline-table-container.children",
    )
    lifecycle_ids = {
        "capacity-page-woodmac-data-store",
        "capacity-page-train-capacity-data-store",
        "capacity-page-ea-capacity-data-store",
        "capacity-page-metadata-store",
        "capacity-page-capacity-scenario-selected-store",
        "capacity-page-capacity-scenario-options-store",
        "capacity-page-capacity-scenario-working-store",
        "capacity-page-capacity-scenario-dirty-store",
        "capacity-page-capacity-scenario-refresh-store",
    }

    for callback_key, definition in callback_map.items():
        if any(marker in callback_key for marker in source_markers):
            input_ids = {
                dependency["id"]
                for dependency in definition["inputs"]
            }
            state_ids = {
                dependency["id"]
                for dependency in definition["state"]
            }
            assert "capacity-page-source-render-revision-store" in input_ids
            assert not input_ids.intersection(lifecycle_ids)
            assert state_ids.intersection(lifecycle_ids)
        if any(marker in callback_key for marker in scenario_markers):
            input_ids = {
                dependency["id"]
                for dependency in definition["inputs"]
            }
            state_ids = {
                dependency["id"]
                for dependency in definition["state"]
            }
            assert "capacity-page-scenario-render-revision-store" in input_ids
            assert not input_ids.intersection(lifecycle_ids)
            assert state_ids.intersection(lifecycle_ids)


def test_relational_capacity_metadata_carries_source_identity(monkeypatch):
    events = pd.DataFrame(
        {
            "provider_name": ["woodmac"],
            "country_name": ["Qatar"],
            "effective_month": pd.to_datetime(["2026-01-01"]),
        }
    )
    components = {
        "woodmac_store": events,
        "train_store": pd.DataFrame(),
        "ea_store": pd.DataFrame(),
    }
    monkeypatch.setattr(
        capacity,
        "_get_cached_capacity_source_snapshot",
        lambda _source_key: components,
    )
    monkeypatch.setattr(
        capacity,
        "fetch_capacity_source_state",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        capacity,
        "capacity_source_event_bounds",
        lambda _events: (
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-12-01"),
        ),
    )

    snapshot = capacity._fetch_relational_capacity_source_snapshot(
        SOURCE_KEY,
        {},
    )

    assert snapshot["snapshot_key"] == SOURCE_KEY
    assert snapshot["metadata"]["snapshot_key"] == SOURCE_KEY
