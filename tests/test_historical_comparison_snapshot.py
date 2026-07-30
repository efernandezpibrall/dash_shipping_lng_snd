from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pandas as pd
import pytest

from utils import dashboard_snapshot_cache as snapshots
from utils import historical_comparison_snapshot as comparisons


@pytest.fixture
def historical_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(snapshots.LOCAL_PERSISTENCE_ENV, "1")
    monkeypatch.setenv(
        snapshots.LOCAL_CACHE_DIR_ENV,
        str(tmp_path / "historical-cache"),
    )
    snapshots.close_persistent_snapshot_cache()
    snapshots.clear_local_snapshots()
    comparisons.clear_historical_comparison_source_state()
    yield
    comparisons.clear_historical_comparison_source_state()
    snapshots.clear_local_snapshots()
    snapshots.close_persistent_snapshot_cache()


def _reference(revision=None):
    return {
        "format": snapshots.REFERENCE_FORMAT,
        "namespace": "provider-flow-source-v2",
        "source_key": "provider-source",
        "revision": revision or snapshots._new_local_revision_token(),
        "shared": True,
        "slot": "woodmac_export",
    }


def test_warm_comparison_reuses_exact_frame_and_mapping_state(
    monkeypatch,
    historical_cache,
):
    manifest_calls = 0
    live_mapping_calls = 0
    build_calls = 0

    def manifest_state(*_args, **_kwargs):
        nonlocal manifest_calls
        manifest_calls += 1
        return {
            "source_state": {
                "mapping_hash": "country-v1",
                "ea_balance_mapping_hash": "ea-v1",
            }
        }

    def live_mapping_state():
        nonlocal live_mapping_calls
        live_mapping_calls += 1
        return {
            "mapping_hash": "country-v1",
            "ea_balance_mapping_hash": "ea-v1",
        }

    def builder():
        nonlocal build_calls
        build_calls += 1
        return pd.DataFrame(
            {
                "month": pd.to_datetime(["2026-01-01"]),
                "value": [12.5],
            }
        )

    monkeypatch.setattr(
        comparisons,
        "resolve_snapshot_manifest",
        manifest_state,
    )
    monkeypatch.setattr(
        comparisons,
        "fetch_provider_flow_mapping_state",
        live_mapping_state,
    )
    base_reference = _reference()
    first_reference, first_frame = comparisons.get_historical_comparison_frame(
        direction="supply",
        base_reference=base_reference,
        selection={"source": "ea", "run_id": 42},
        query_dependencies={
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
        },
        builder=builder,
    )
    second_reference, second_frame = comparisons.get_historical_comparison_frame(
        direction="supply",
        base_reference=base_reference,
        selection={"source": "ea", "run_id": 42},
        query_dependencies={
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
        },
        builder=builder,
    )

    assert first_reference == second_reference
    pd.testing.assert_frame_equal(first_frame, second_frame, check_exact=True)
    assert build_calls == 1
    assert manifest_calls == 2
    assert live_mapping_calls == 1


def test_base_revision_and_exact_selection_partition_snapshots(
    monkeypatch,
    historical_cache,
):
    monkeypatch.setattr(
        comparisons,
        "resolve_snapshot_manifest",
        lambda *_args, **_kwargs: {
            "source_state": {
                "mapping_hash": "country-v1",
                "ea_balance_mapping_hash": "ea-v1",
            }
        },
    )
    monkeypatch.setattr(
        comparisons,
        "fetch_provider_flow_mapping_state",
        lambda: {
            "mapping_hash": "country-v1",
            "ea_balance_mapping_hash": "ea-v1",
        },
    )
    build_calls = 0

    def builder():
        nonlocal build_calls
        build_calls += 1
        return pd.DataFrame({"value": [build_calls]})

    first_base = _reference()
    second_base = _reference()
    references = []
    for base_reference, run_id in (
        (first_base, 41),
        (first_base, 42),
        (second_base, 42),
    ):
        reference, _ = comparisons.get_historical_comparison_frame(
            direction="demand",
            base_reference=base_reference,
            selection={"source": "ea", "run_id": run_id},
            query_dependencies={"start_date": "2026-01-01"},
            builder=builder,
        )
        references.append(reference)

    assert build_calls == 3
    assert len({reference["source_key"] for reference in references}) == 3


@pytest.mark.parametrize("worker_count", (1, 4, 8))
def test_comparison_build_is_single_flight(
    monkeypatch,
    historical_cache,
    worker_count,
):
    monkeypatch.setattr(
        comparisons,
        "resolve_snapshot_manifest",
        lambda *_args, **_kwargs: {
            "source_state": {
                "mapping_hash": "country-v1",
                "ea_balance_mapping_hash": "ea-v1",
            }
        },
    )
    monkeypatch.setattr(
        comparisons,
        "fetch_provider_flow_mapping_state",
        lambda: {
            "mapping_hash": "country-v1",
            "ea_balance_mapping_hash": "ea-v1",
        },
    )
    build_calls = 0
    build_lock = threading.Lock()

    def builder():
        nonlocal build_calls
        with build_lock:
            build_calls += 1
        time.sleep(0.03)
        return pd.DataFrame({"value": [7.0]})

    base_reference = _reference()

    def load():
        return comparisons.get_historical_comparison_frame(
            direction="net-balance",
            base_reference=base_reference,
            selection={
                "source": "woodmac",
                "short_term_publication_timestamp": "2026-07-20",
                "long_term_publication_timestamp": "2026-07-01",
            },
            query_dependencies={
                "country_group": "country",
                "time_group": "yearly",
                "unit": "bcm",
            },
            builder=builder,
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(lambda _index: load(), range(worker_count)))

    assert build_calls == 1
    assert all(result[0] == results[0][0] for result in results)
    assert all(result[1].equals(results[0][1]) for result in results)


def test_changed_mapping_during_build_refuses_publication(
    monkeypatch,
    historical_cache,
):
    monkeypatch.setattr(
        comparisons,
        "resolve_snapshot_manifest",
        lambda *_args, **_kwargs: {
            "source_state": {
                "mapping_hash": "country-v1",
                "ea_balance_mapping_hash": "ea-v1",
            }
        },
    )
    monkeypatch.setattr(
        comparisons,
        "fetch_provider_flow_mapping_state",
        lambda: {
            "mapping_hash": "country-v2",
            "ea_balance_mapping_hash": "ea-v1",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="mappings changed during snapshot construction",
    ):
        comparisons.get_historical_comparison_frame(
            direction="supply",
            base_reference=_reference(),
            selection={"source": "ea", "run_id": 42},
            query_dependencies={"start_date": "2026-01-01"},
            builder=lambda: pd.DataFrame({"value": [1.0]}),
        )
