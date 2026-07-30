from threading import Barrier

import pandas as pd
import pytest

from pages import terminal_adjustments


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            yield from _walk(child)


def _components_by_id(layout):
    return {
        component.id: component
        for component in _walk(layout)
        if getattr(component, "id", None)
    }


def test_layout_loaders_overlap_and_preserve_exact_options(monkeypatch):
    loader_barrier = Barrier(3, timeout=2)
    scenarios = ["base_view", "best_view", "stress_case"]
    plants = pd.DataFrame(
        [
            {"plant_name": "Alpha", "country_name": "Country A"},
            {"plant_name": "Beta", "country_name": "Country B"},
        ]
    )
    trains = pd.DataFrame(
        [
            {
                "plant_name": "Alpha",
                "lng_train_name_short": "T1",
                "country_name": "Country A",
                "id_plant": 1,
                "id_lng_train": 10,
            },
            {
                "plant_name": "Beta",
                "lng_train_name_short": "T2",
                "country_name": "Country B",
                "id_plant": None,
                "id_lng_train": None,
            },
        ]
    )

    def load(value):
        loader_barrier.wait()
        return value

    monkeypatch.setattr(
        terminal_adjustments,
        "get_available_scenarios",
        lambda _engine: load(scenarios),
    )
    monkeypatch.setattr(
        terminal_adjustments,
        "get_plants_list",
        lambda: load(plants),
    )
    monkeypatch.setattr(
        terminal_adjustments,
        "get_trains_list",
        lambda: load(trains),
    )

    components = _components_by_id(terminal_adjustments.layout())

    assert components["scenario-selector"].options == [
        {"label": "base_view", "value": "base_view"},
        {"label": "best_view", "value": "best_view"},
        {"label": "stress_case", "value": "stress_case"},
    ]
    assert components["scenario-selector"].value == "best_view"
    assert components["plant-filter"].options == [
        {"label": "Alpha (Country A)", "value": "Alpha"},
        {"label": "Beta (Country B)", "value": "Beta"},
    ]
    assert components["train-filter"].options == [
        {"label": "Alpha - T1", "value": "Alpha|T1"},
        {"label": "Beta - T2", "value": "Beta|T2"},
    ]
    assert components["trains-to-copy"].options == [
        {"label": "Alpha - T1", "value": "Alpha|T1"},
    ]
    assert [
        child.value
        for child in components["scenario-list"].children
    ] == ["best_view", "stress_case"]


def test_layout_propagates_option_loader_failure(monkeypatch):
    monkeypatch.setattr(
        terminal_adjustments,
        "get_available_scenarios",
        lambda _engine: ["base_view", "best_view"],
    )
    monkeypatch.setattr(
        terminal_adjustments,
        "get_plants_list",
        lambda: (_ for _ in ()).throw(RuntimeError("plants unavailable")),
    )
    monkeypatch.setattr(
        terminal_adjustments,
        "get_trains_list",
        lambda: pd.DataFrame(),
    )

    with pytest.raises(RuntimeError, match="plants unavailable"):
        terminal_adjustments.layout()
