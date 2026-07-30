from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
import time

import pandas as pd

from pages import contracts


REVISION_V1 = (
    ("contracts", 1, "2026-07-27T00:00:00+00:00"),
    ("demand", 1, "2026-07-27T00:00:00+00:00"),
    ("price_assumptions", 1, "2026-07-27T00:00:00+00:00"),
    ("price_formula", 1, "2026-07-27T00:00:00+00:00"),
)
REVISION_V2 = tuple(
    (name, row_count, "2026-07-28T00:00:00+00:00")
    for name, row_count, _watermark in REVISION_V1
)


def _snapshot(revision_key=REVISION_V1, marker="v1"):
    return contracts.ContractsSnapshot(
        revision_key=revision_key,
        contracts=pd.DataFrame({"state": [f"contracts-{marker}"]}),
        demand=pd.DataFrame({"state": [f"demand-{marker}"]}),
        price_assumptions=pd.DataFrame(
            {"state": [f"assumptions-{marker}"]}
        ),
        price_formula=pd.DataFrame({"state": [f"formula-{marker}"]}),
        year_settings={"marker": marker},
    )


def _reset_contracts_state(monkeypatch):
    monkeypatch.setattr(contracts, "_contracts_snapshot", None)
    monkeypatch.setattr(
        contracts,
        "_contracts_snapshot_status",
        "unavailable",
    )
    monkeypatch.setattr(
        contracts,
        "_contracts_snapshot_message",
        "not loaded",
    )


def test_contract_loaders_overlap_and_assign_fixed_named_results(monkeypatch):
    loader_barrier = Barrier(4, timeout=2)
    expected = {
        "contracts": pd.DataFrame({"source": ["contracts"]}),
        "demand": pd.DataFrame({"source": ["demand"]}),
        "price_assumptions": pd.DataFrame(
            {"source": ["price_assumptions"]}
        ),
        "price_formula": pd.DataFrame({"source": ["price_formula"]}),
    }

    def loader(name):
        loader_barrier.wait()
        return expected[name]

    monkeypatch.setattr(
        contracts,
        "load_contracts_data",
        lambda: loader("contracts"),
    )
    monkeypatch.setattr(
        contracts,
        "load_annual_demand_data",
        lambda: loader("demand"),
    )
    monkeypatch.setattr(
        contracts,
        "load_price_assumptions_data",
        lambda: loader("price_assumptions"),
    )
    monkeypatch.setattr(
        contracts,
        "load_price_formula_data",
        lambda: loader("price_formula"),
    )
    monkeypatch.setattr(
        contracts,
        "_enhance_contracts_data",
        lambda contracts_df, assumptions_df: pd.DataFrame(
            {
                "contracts_source": contracts_df["source"],
                "assumptions_source": assumptions_df["source"],
            }
        ),
    )
    monkeypatch.setattr(
        contracts,
        "_calculate_contract_year_settings",
        lambda frame: {
            "rows": len(frame),
            "contracts_source": frame["contracts_source"].iloc[0],
        },
    )

    snapshot = contracts._build_contracts_snapshot(REVISION_V1)

    assert snapshot.revision_key == REVISION_V1
    assert snapshot.contracts.to_dict("records") == [
        {
            "contracts_source": "contracts",
            "assumptions_source": "price_assumptions",
        }
    ]
    pd.testing.assert_frame_equal(snapshot.demand, expected["demand"])
    pd.testing.assert_frame_equal(
        snapshot.price_assumptions,
        expected["price_assumptions"],
    )
    pd.testing.assert_frame_equal(
        snapshot.price_formula,
        expected["price_formula"],
    )
    assert snapshot.year_settings == {
        "rows": 1,
        "contracts_source": "contracts",
    }


def test_contract_snapshot_load_is_single_flight_across_callers(monkeypatch):
    _reset_contracts_state(monkeypatch)
    call_count = 0
    count_lock = Lock()

    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V1,
    )

    def build_snapshot(_revision_key):
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.03)
        return _snapshot()

    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        build_snapshot,
    )

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(
            executor.map(
                lambda _value: contracts._ensure_contracts_snapshot(),
                range(6),
            )
        )

    assert call_count == 1
    assert all(result is results[0] for result in results)
    assert contracts._contracts_snapshot_status == "fresh"


def test_unchanged_revision_reuses_snapshot(monkeypatch):
    existing = _snapshot()
    monkeypatch.setattr(contracts, "_contracts_snapshot", existing)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V1,
    )
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: (_ for _ in ()).throw(
            AssertionError("unexpected reload")
        ),
    )

    result = contracts._ensure_contracts_snapshot()

    assert result is existing
    assert contracts._contracts_snapshot_status == "fresh"


def test_changed_revision_atomically_replaces_snapshot(monkeypatch):
    existing = _snapshot()
    replacement = _snapshot(REVISION_V2, marker="v2")
    monkeypatch.setattr(contracts, "_contracts_snapshot", existing)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V2,
    )
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda revision: replacement
        if revision == REVISION_V2
        else None,
    )

    result = contracts._ensure_contracts_snapshot()

    assert result is replacement
    assert contracts._contracts_snapshot is replacement
    assert contracts._contracts_snapshot_status == "fresh"


def test_forced_refresh_reloads_unchanged_revision(monkeypatch):
    existing = _snapshot()
    replacement = _snapshot(marker="forced")
    monkeypatch.setattr(contracts, "_contracts_snapshot", existing)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V1,
    )
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: replacement,
    )

    result = contracts._ensure_contracts_snapshot(force=True)

    assert result is replacement


def test_contract_refresh_failure_keeps_last_good_snapshot(monkeypatch):
    existing = _snapshot()
    monkeypatch.setattr(contracts, "_contracts_snapshot", existing)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V2,
    )
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: (_ for _ in ()).throw(
            RuntimeError("demand unavailable")
        ),
    )

    result = contracts._ensure_contracts_snapshot()

    assert result is existing
    assert contracts._contracts_snapshot is existing
    assert contracts._contracts_snapshot_status == "stale"
    assert "last verified" in contracts._contracts_snapshot_message


def test_initial_contract_failure_returns_visible_unavailable_state(
    monkeypatch,
):
    _reset_contracts_state(monkeypatch)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: (_ for _ in ()).throw(RuntimeError("revision unavailable")),
    )

    result = contracts._ensure_contracts_snapshot()
    message, class_name = contracts._contracts_source_status(result)

    assert result.contracts.empty
    assert contracts._contracts_snapshot_status == "unavailable"
    assert "unavailable" in message
    assert class_name.endswith("-unavailable")
