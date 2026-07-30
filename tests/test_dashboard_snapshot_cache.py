import datetime as dt
import json
import multiprocessing
import os
import stat
import threading
import time

import numpy as np
import pandas as pd
import pytest

from utils import dashboard_snapshot_cache as snapshots


def _cross_process_snapshot_worker(
    cache_directory,
    start_event,
    builder_calls,
    result_queue,
):
    os.environ[snapshots.LOCAL_PERSISTENCE_ENV] = "1"
    os.environ[snapshots.LOCAL_CACHE_DIR_ENV] = cache_directory
    snapshots.close_persistent_snapshot_cache()
    snapshots.clear_local_snapshots()
    start_event.wait(timeout=10)

    def builder():
        with builder_calls.get_lock():
            builder_calls.value += 1
        time.sleep(0.2)
        return {"rows": [1, 2, 3]}

    try:
        reference, payload = snapshots.get_or_build_snapshot(
            _UnavailableEngine(),
            namespace="cross-process-test",
            source_key="same-source",
            builder=builder,
        )
        result_queue.put(("ok", reference, payload))
    except BaseException as exc:
        result_queue.put(("error", type(exc).__name__, str(exc)))
    finally:
        snapshots.close_persistent_snapshot_cache()


@pytest.fixture(autouse=True)
def _isolate_snapshot_backend(monkeypatch):
    monkeypatch.setenv(snapshots.LOCAL_PERSISTENCE_ENV, "0")
    snapshots.close_persistent_snapshot_cache()
    snapshots.clear_local_snapshots()
    yield
    snapshots.clear_local_snapshots()
    snapshots.close_persistent_snapshot_cache()


def _enable_persistence(monkeypatch, tmp_path, **limits):
    cache_directory = tmp_path / "persistent-snapshots"
    monkeypatch.setenv(snapshots.LOCAL_PERSISTENCE_ENV, "1")
    monkeypatch.setenv(
        snapshots.LOCAL_CACHE_DIR_ENV,
        str(cache_directory),
    )
    if "memory" in limits:
        monkeypatch.setenv(
            snapshots.MEMORY_MAX_BYTES_ENV,
            str(limits["memory"]),
        )
    if "disk" in limits:
        monkeypatch.setenv(
            snapshots.DISK_MAX_BYTES_ENV,
            str(limits["disk"]),
        )
    snapshots.close_persistent_snapshot_cache()
    snapshots.clear_local_snapshots()
    return cache_directory


class _UnavailableEngine:
    def connect(self):
        raise RuntimeError("migration unavailable")


class _LockConnection:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, *_args, **_kwargs):
        return None


class _LockEngine:
    def connect(self):
        return _LockConnection()


def test_codec_round_trip_preserves_dataframe_values_and_dtypes():
    frame = pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-01-01", None, "2025-03-01"]),
            "volume": pd.Series([1.5, np.nan, 3.25], dtype="float64"),
            "count": pd.Series([1, pd.NA, 3], dtype="Int64"),
            "label": pd.Series(["A", None, "C"], dtype="object"),
        },
        index=pd.Index([10, 20, 30], name="row_id"),
    )
    payload = {
        "frame": frame,
        "as_of": pd.Timestamp("2026-07-14T12:30:00"),
        "date": dt.date(2026, 7, 14),
        "tuple": (1, "two"),
    }

    restored = snapshots.decode_snapshot_payload(snapshots.encode_snapshot_payload(payload))

    pd.testing.assert_frame_equal(restored["frame"], frame)
    assert restored["as_of"] == payload["as_of"]
    assert restored["date"] == payload["date"]
    assert restored["tuple"] == payload["tuple"]


def test_codec_is_deterministic_and_preserves_dtype_order_index_and_null_contracts():
    index = pd.Index([30, 10, 20], name="row_id")
    frame = pd.DataFrame(
        {
            "label": ["C", None, "B"],
            "nullable_int": pd.array([3, pd.NA, 2], dtype="Int64"),
            "volume": np.array([3.5, np.nan, 2.25], dtype="float64"),
            "month": pd.to_datetime(
                ["2026-03-01", None, "2026-02-01"]
            ),
        },
        index=index,
    )
    payload = {
        "frame": frame,
        "series": pd.Series(
            pd.array([True, pd.NA, False], dtype="boolean"),
            index=index,
            name="active",
        ),
        "as_of": pd.Timestamp("2026-07-24T12:30:00"),
        "tuple": ("first", None, 3),
    }

    first = snapshots.encode_snapshot_payload(payload)
    second = snapshots.encode_snapshot_payload(payload)
    restored = snapshots.decode_snapshot_payload(first)

    assert first == second
    assert list(restored["frame"].columns) == list(frame.columns)
    pd.testing.assert_frame_equal(restored["frame"], frame)
    pd.testing.assert_series_equal(restored["series"], payload["series"])
    assert restored["as_of"] == payload["as_of"]
    assert restored["tuple"] == payload["tuple"]


def test_record_cube_round_trip_preserves_entity_and_record_order():
    records = {
        "Global": [{"date": "2025-01-01", "value": 1.0}],
        "Japan": [
            {"date": "2025-01-01", "value": 2.0, "share": 50.0},
            {"date": "2025-01-02", "value": None, "share": 40.0},
        ],
        "Empty": [],
    }
    packed = snapshots.pack_record_mapping(records)
    assert snapshots.unpack_record_mapping(packed) == records
    assert list(snapshots.unpack_record_mapping(packed)) == list(records)


def test_local_snapshot_single_flight_and_slot_resolution(monkeypatch):
    snapshots.clear_local_snapshots("single-flight-test")
    monkeypatch.setattr(snapshots, "shared_snapshot_schema_available", lambda _engine: False)
    calls = 0
    calls_lock = threading.Lock()

    def builder():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return {"left": [1, 2], "right": {"ok": True}}

    results = []

    def run():
        results.append(
            snapshots.get_or_build_snapshot(
                _UnavailableEngine(),
                namespace="single-flight-test",
                source_key="same-key",
                builder=builder,
            )
        )

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert len(results) == 4
    reference = snapshots.with_snapshot_slot(results[0][0], "right")
    assert snapshots.resolve_snapshot(reference, _UnavailableEngine()) == {"ok": True}


def test_get_snapshot_if_available_reuses_local_payload_without_builder(
    monkeypatch,
):
    namespace = "last-good-local-test"
    monkeypatch.setattr(
        snapshots,
        "shared_snapshot_schema_available",
        lambda _engine: False,
    )
    reference, payload = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace=namespace,
        source_key="verified-revision",
        builder=lambda: {"rows": [1, 2, 3]},
    )

    available = snapshots.get_snapshot_if_available(
        _UnavailableEngine(),
        namespace=namespace,
        source_key="verified-revision",
    )

    assert available == (reference, payload)


def test_get_snapshot_if_available_returns_none_when_no_snapshot_exists(
    monkeypatch,
):
    monkeypatch.setattr(
        snapshots,
        "shared_snapshot_schema_available",
        lambda _engine: False,
    )

    assert (
        snapshots.get_snapshot_if_available(
            _UnavailableEngine(),
            namespace="last-good-miss-test",
            source_key="missing-revision",
        )
        is None
    )


def test_source_key_is_deterministic_and_filter_sensitive():
    first = snapshots.build_source_key("flows", {"b": 2, "a": 1}, ["x", "y"])
    second = snapshots.build_source_key("flows", {"a": 1, "b": 2}, ["x", "y"])
    changed = snapshots.build_source_key("flows", {"a": 1, "b": 2}, ["x", "z"])
    assert first == second
    assert changed != first
    assert len(first) == 64


def test_manifest_is_normalized_for_postgres_json(monkeypatch):
    namespace = "manifest-test"
    snapshots.clear_local_snapshots(namespace)
    monkeypatch.setattr(snapshots, "shared_snapshot_schema_available", lambda _engine: False)
    snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace=namespace,
        source_key="source",
        builder=lambda: {"ok": True},
        manifest={
            "timestamp": pd.Timestamp("2026-07-14T12:30:00"),
            "date": dt.date(2026, 7, 14),
            "number": np.int64(7),
        },
    )

    cached = snapshots._get_local(namespace, "source")
    assert cached[2] == {
        "timestamp": "2026-07-14T12:30:00",
        "date": "2026-07-14",
        "number": 7,
    }


def test_reference_is_small_and_legacy_values_pass_through(monkeypatch):
    snapshots.clear_local_snapshots("size-test")
    monkeypatch.setattr(snapshots, "shared_snapshot_schema_available", lambda _engine: False)
    reference, payload = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="size-test",
        source_key="source",
        builder=lambda: {"data": list(range(1000))},
    )
    assert len(json.dumps(reference).encode("utf-8")) < 250
    assert snapshots.resolve_snapshot({"legacy": True}, _UnavailableEngine()) == {"legacy": True}
    assert payload["data"][-1] == 999


def test_shared_read_failure_falls_back_to_original_builder(monkeypatch):
    namespace = "read-fallback-test"
    snapshots.clear_local_snapshots(namespace)
    monkeypatch.setattr(snapshots, "shared_snapshot_schema_available", lambda _engine: True)
    monkeypatch.setattr(
        snapshots,
        "_read_shared",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("read failed")),
    )
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return {"ok": True}

    reference, payload = snapshots.get_or_build_snapshot(
        _LockEngine(),
        namespace=namespace,
        source_key="source",
        builder=builder,
    )

    assert calls == 1
    assert reference["shared"] is False
    assert payload == {"ok": True}


def test_shared_publish_failure_reuses_built_payload_locally(monkeypatch):
    namespace = "publish-fallback-test"
    snapshots.clear_local_snapshots(namespace)
    monkeypatch.setattr(snapshots, "shared_snapshot_schema_available", lambda _engine: True)
    monkeypatch.setattr(snapshots, "_read_shared", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(snapshots, "_next_revision", lambda *_args, **_kwargs: 7)
    monkeypatch.setattr(
        snapshots,
        "_write_shared",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return {"rows": [1, 2, 3]}

    reference, payload = snapshots.get_or_build_snapshot(
        _LockEngine(),
        namespace=namespace,
        source_key="source",
        builder=builder,
    )

    assert calls == 1
    assert reference["shared"] is False
    assert snapshots.resolve_snapshot(reference, _LockEngine()) == payload
    cached_reference, cached_payload = snapshots.get_or_build_snapshot(
        _LockEngine(),
        namespace=namespace,
        source_key="source",
        builder=builder,
    )
    assert calls == 1
    assert cached_reference["shared"] is False
    assert cached_payload == payload


def test_builder_failure_is_not_retried_by_storage_fallback(monkeypatch):
    namespace = "builder-failure-test"
    snapshots.clear_local_snapshots(namespace)
    monkeypatch.setattr(snapshots, "shared_snapshot_schema_available", lambda _engine: True)
    monkeypatch.setattr(snapshots, "_read_shared", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(snapshots, "_next_revision", lambda *_args, **_kwargs: 1)
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        raise ValueError("source failed")

    with pytest.raises(ValueError, match="source failed"):
        snapshots.get_or_build_snapshot(
            _LockEngine(),
            namespace=namespace,
            source_key="source",
            builder=builder,
        )

    assert calls == 1


def test_persistent_memory_hit_is_resolvable_and_skips_postgres(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    monkeypatch.setattr(
        snapshots,
        "shared_snapshot_schema_available",
        lambda _engine: (_ for _ in ()).throw(
            AssertionError("Postgres probe must be skipped")
        ),
    )
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return {"rows": [1, 2, 3]}

    first_reference, first_payload = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="persistent-memory-hit",
        source_key="source",
        builder=builder,
    )
    second_reference, second_payload = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="persistent-memory-hit",
        source_key="source",
        builder=builder,
    )

    assert calls == 1
    assert first_reference == second_reference
    assert second_payload is first_payload
    assert snapshots.snapshot_is_shared(first_reference)
    assert snapshots.snapshot_is_resolvable(first_reference)
    assert (
        snapshots._validate_disk_revision_token(
            first_reference["revision"]
        )
        == first_reference["revision"]
    )
    assert int(first_reference["revision"]) > 0


def test_persistent_disk_hit_survives_memory_clear_and_handle_restart(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    payload = {
        "frame": pd.DataFrame(
            {"value": pd.Series([1, pd.NA], dtype="Int64")}
        ),
        "slot": {"ok": True},
    }
    reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="restart-test",
        source_key="source",
        builder=lambda: payload,
        manifest={"watermark": "2026-07-24"},
    )
    stores = snapshots._get_persistent_stores()
    raw_record = stores.cache.get(
        snapshots._disk_record_key(
            "restart-test",
            "source",
            reference["revision"],
        )
    )
    assert isinstance(raw_record, bytes)
    assert not raw_record.startswith(b"\x80")

    snapshots.clear_local_snapshots()
    snapshots.close_persistent_snapshot_cache()
    restored_reference, restored_payload = (
        snapshots.get_or_build_snapshot(
            _UnavailableEngine(),
            namespace="restart-test",
            source_key="source",
            builder=lambda: (_ for _ in ()).throw(
                AssertionError("disk hit must not rebuild")
            ),
        )
    )

    assert restored_reference == reference
    json_reference = json.loads(json.dumps(reference))
    assert json_reference == reference
    pd.testing.assert_frame_equal(
        restored_payload["frame"],
        payload["frame"],
    )
    assert restored_payload["slot"] == payload["slot"]
    assert (
        snapshots.resolve_snapshot(
            snapshots.with_snapshot_slot(reference, "slot"),
            _UnavailableEngine(),
        )
        == {"ok": True}
    )
    snapshots.clear_local_snapshots()
    json_restored_payload = snapshots.resolve_snapshot(
        json_reference,
        _UnavailableEngine(),
    )
    pd.testing.assert_frame_equal(
        json_restored_payload["frame"],
        payload["frame"],
    )
    assert json_restored_payload["slot"] == payload["slot"]


def test_historical_exact_disk_read_does_not_downgrade_latest(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    first_reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="historical-resolution-test",
        source_key="source",
        builder=lambda: {"version": 1},
    )
    second_reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="historical-resolution-test",
        source_key="source",
        builder=lambda: {"version": 2},
        force=True,
    )

    snapshots.clear_local_snapshots()
    snapshots.close_persistent_snapshot_cache()
    assert snapshots.resolve_snapshot(
        first_reference,
        _UnavailableEngine(),
    ) == {"version": 1}

    latest_reference, latest_payload = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="historical-resolution-test",
        source_key="source",
        builder=lambda: (_ for _ in ()).throw(
            AssertionError("latest revision must be read from disk")
        ),
    )

    assert latest_reference == second_reference
    assert latest_payload == {"version": 2}


def test_force_publish_is_atomic_and_preserves_exact_revisions_and_slots(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    first_reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="force-test",
        source_key="source",
        builder=lambda: {"selected": "first", "rows": [1]},
    )

    with pytest.raises(ValueError, match="builder failed"):
        snapshots.get_or_build_snapshot(
            _UnavailableEngine(),
            namespace="force-test",
            source_key="source",
            builder=lambda: (_ for _ in ()).throw(
                ValueError("builder failed")
            ),
            force=True,
        )

    snapshots.clear_local_snapshots()
    unchanged_reference, unchanged_payload = (
        snapshots.get_or_build_snapshot(
            _UnavailableEngine(),
            namespace="force-test",
            source_key="source",
            builder=lambda: (_ for _ in ()).throw(
                AssertionError("failed force must not advance latest")
            ),
        )
    )
    assert unchanged_reference == first_reference
    assert unchanged_payload["selected"] == "first"

    second_reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="force-test",
        source_key="source",
        builder=lambda: {"selected": "second", "rows": [2]},
        force=True,
    )
    assert second_reference["revision"] != first_reference["revision"]
    assert (
        snapshots.resolve_snapshot(
            snapshots.with_snapshot_slot(
                first_reference,
                "selected",
            ),
            _UnavailableEngine(),
        )
        == "first"
    )
    assert (
        snapshots.resolve_snapshot(
            snapshots.with_snapshot_slot(
                second_reference,
                "selected",
            ),
            _UnavailableEngine(),
        )
        == "second"
    )


def test_initial_builder_failure_does_not_create_or_increment_revision(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="source failed"):
        snapshots.get_or_build_snapshot(
            _UnavailableEngine(),
            namespace="initial-failure-test",
            source_key="source",
            builder=lambda: (_ for _ in ()).throw(
                RuntimeError("source failed")
            ),
        )

    stores = snapshots._get_persistent_stores()
    assert (
        snapshots._disk_latest_revision(
            stores,
            "initial-failure-test",
            "source",
        )
        is None
    )
    reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="initial-failure-test",
        source_key="source",
        builder=lambda: {"ok": True},
    )
    assert isinstance(reference["revision"], str)
    snapshots._validate_disk_revision_token(reference["revision"])


def test_persistent_builder_oserror_propagates_without_retry(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        raise OSError("source transport failed")

    with pytest.raises(OSError, match="source transport failed"):
        snapshots.get_or_build_snapshot(
            _UnavailableEngine(),
            namespace="builder-oserror-test",
            source_key="source",
            builder=builder,
        )

    assert calls == 1


def test_failed_latest_pointer_swap_preserves_unique_revision_identity(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    stores = snapshots._get_persistent_stores()
    real_cache_set = stores.cache.set
    latest_key = snapshots._disk_latest_key(
        "pointer-failure-test",
        "source",
    )

    def reject_latest_pointer(key, *args, **kwargs):
        if key == latest_key:
            return False
        return real_cache_set(key, *args, **kwargs)

    monkeypatch.setattr(stores.cache, "set", reject_latest_pointer)

    fallback_reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="pointer-failure-test",
        source_key="source",
        builder=lambda: {"version": "orphaned"},
    )

    assert fallback_reference["shared"] is False
    assert (
        snapshots._disk_latest_revision(
            stores,
            "pointer-failure-test",
            "source",
        )
        is None
    )

    monkeypatch.setattr(stores.cache, "set", real_cache_set)
    snapshots.clear_local_snapshots("pointer-failure-test")
    published_reference, published_payload = (
        snapshots.get_or_build_snapshot(
            _UnavailableEngine(),
            namespace="pointer-failure-test",
            source_key="source",
            builder=lambda: {"version": "published"},
        )
    )

    assert published_reference["shared"] is True
    assert (
        published_reference["revision"]
        != fallback_reference["revision"]
    )
    assert published_payload == {"version": "published"}


def test_pointer_only_eviction_builds_new_token_without_aliasing_old_reference(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    old_reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="pointer-eviction-test",
        source_key="source",
        builder=lambda: {"version": "old"},
    )
    stores = snapshots._get_persistent_stores()
    stores.cache.delete(
        snapshots._disk_latest_key(
            "pointer-eviction-test",
            "source",
        ),
        retry=True,
    )
    snapshots.clear_local_snapshots()
    snapshots.close_persistent_snapshot_cache()

    new_reference, new_payload = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="pointer-eviction-test",
        source_key="source",
        builder=lambda: {"version": "new"},
    )

    assert new_reference["revision"] != old_reference["revision"]
    assert new_payload == {"version": "new"}
    assert snapshots.resolve_snapshot(
        old_reference,
        _UnavailableEngine(),
    ) == {"version": "old"}
    assert snapshots.resolve_snapshot(
        new_reference,
        _UnavailableEngine(),
    ) == {"version": "new"}


def test_record_and_pointer_eviction_never_reuses_old_reference_identity(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    old_reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="full-eviction-test",
        source_key="source",
        builder=lambda: {"version": "old"},
    )
    stores = snapshots._get_persistent_stores()
    stores.cache.delete(
        snapshots._disk_record_key(
            "full-eviction-test",
            "source",
            old_reference["revision"],
        ),
        retry=True,
    )
    stores.cache.delete(
        snapshots._disk_latest_key(
            "full-eviction-test",
            "source",
        ),
        retry=True,
    )
    snapshots.clear_local_snapshots()
    snapshots.close_persistent_snapshot_cache()

    new_reference, new_payload = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="full-eviction-test",
        source_key="source",
        builder=lambda: {"version": "new"},
    )

    assert new_reference["revision"] != old_reference["revision"]
    assert new_payload == {"version": "new"}
    with pytest.raises(
        snapshots.SnapshotUnavailable,
        match="missing or has been evicted",
    ):
        snapshots.resolve_snapshot(
            old_reference,
            _UnavailableEngine(),
        )
    assert snapshots.resolve_snapshot(
        new_reference,
        _UnavailableEngine(),
    ) == {"version": "new"}


def test_revision_collision_never_overwrites_immutable_record(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    first_reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="revision-collision-test",
        source_key="source",
        builder=lambda: {"version": "first"},
    )
    stores = snapshots._get_persistent_stores()
    record_key = snapshots._disk_record_key(
        "revision-collision-test",
        "source",
        first_reference["revision"],
    )
    original_record = stores.cache.get(record_key)
    monkeypatch.setattr(
        snapshots,
        "_disk_next_revision",
        lambda *_args: first_reference["revision"],
    )

    fallback_reference, fallback_payload = (
        snapshots.get_or_build_snapshot(
            _UnavailableEngine(),
            namespace="revision-collision-test",
            source_key="source",
            builder=lambda: {"version": "second"},
            force=True,
        )
    )

    assert fallback_reference["shared"] is False
    assert (
        fallback_reference["revision"]
        != first_reference["revision"]
    )
    assert fallback_payload == {"version": "second"}
    assert stores.cache.get(record_key) == original_record
    snapshots.clear_local_snapshots()
    latest_reference, latest_payload = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="revision-collision-test",
        source_key="source",
        builder=lambda: (_ for _ in ()).throw(
            AssertionError("original latest must remain published")
        ),
    )
    assert latest_reference == first_reference
    assert latest_payload == {"version": "first"}


def test_corrupt_and_evicted_exact_references_raise_without_rebuild(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    corrupt_reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="corrupt-test",
        source_key="source",
        builder=lambda: {"ok": True},
    )
    stores = snapshots._get_persistent_stores()
    stores.cache.set(
        snapshots._disk_record_key(
            "corrupt-test",
            "source",
            corrupt_reference["revision"],
        ),
        b"broken",
        retry=True,
    )
    snapshots.clear_local_snapshots()

    with pytest.raises(
        snapshots.SnapshotUnavailable,
        match="corrupt or unreadable",
    ):
        snapshots.resolve_snapshot(
            corrupt_reference,
            _UnavailableEngine(),
        )
    with pytest.raises(
        snapshots.SnapshotUnavailable,
        match="corrupt or unreadable",
    ):
        snapshots.get_or_build_snapshot(
            _UnavailableEngine(),
            namespace="corrupt-test",
            source_key="source",
            builder=lambda: (_ for _ in ()).throw(
                AssertionError("corrupt latest must not auto-rebuild")
            ),
        )

    evicted_reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="evicted-test",
        source_key="source",
        builder=lambda: {"ok": True},
    )
    stores.cache.delete(
        snapshots._disk_record_key(
            "evicted-test",
            "source",
            evicted_reference["revision"],
        ),
        retry=True,
    )
    snapshots.clear_local_snapshots()
    with pytest.raises(
        snapshots.SnapshotUnavailable,
        match="missing or has been evicted",
    ):
        snapshots.resolve_snapshot(
            evicted_reference,
            _UnavailableEngine(),
        )


def test_tampered_manifest_header_fails_record_integrity_check(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="manifest-integrity-test",
        source_key="source",
        builder=lambda: {"ok": True},
        manifest={"watermark": "2026-07-24"},
    )
    stores = snapshots._get_persistent_stores()
    record_key = snapshots._disk_record_key(
        "manifest-integrity-test",
        "source",
        reference["revision"],
    )
    raw_record = stores.cache.get(record_key)
    header_size = int.from_bytes(
        raw_record[: snapshots._DISK_HEADER_LENGTH_BYTES],
        "big",
    )
    header_start = (
        snapshots._DISK_HEADER_LENGTH_BYTES
        + snapshots._DISK_RECORD_CHECKSUM_BYTES
    )
    header_end = header_start + header_size
    header = json.loads(raw_record[header_start:header_end])
    header["manifest"]["watermark"] = "2026-07-25"
    tampered_header = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    assert len(tampered_header) == header_size
    tampered_record = (
        raw_record[:header_start]
        + tampered_header
        + raw_record[header_end:]
    )
    stores.cache.set(record_key, tampered_record, retry=True)
    snapshots.clear_local_snapshots()

    with pytest.raises(
        snapshots.SnapshotUnavailable,
        match="corrupt or unreadable",
    ):
        snapshots.resolve_snapshot(
            reference,
            _UnavailableEngine(),
        )


def test_disk_publish_failure_falls_back_once_without_postgres_probe(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    monkeypatch.setattr(
        snapshots,
        "shared_snapshot_schema_available",
        lambda _engine: (_ for _ in ()).throw(
            AssertionError("Postgres probe must be skipped")
        ),
    )
    monkeypatch.setattr(
        snapshots,
        "_disk_publish",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            snapshots._PersistentStorageError("disk full")
        ),
    )
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return {"rows": [1, 2, 3]}

    reference, payload = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="disk-failure-test",
        source_key="source",
        builder=builder,
    )

    assert calls == 1
    assert reference["shared"] is False
    assert not snapshots.snapshot_is_resolvable(reference)
    assert payload == {"rows": [1, 2, 3]}


def test_memory_is_bounded_by_decoded_bytes_and_disk_resolves_eviction(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path, memory=1_200)
    first_reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="memory-bound-test",
        source_key="first",
        builder=lambda: {"blob": "x" * 2_000},
    )
    snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="memory-bound-test",
        source_key="second",
        builder=lambda: {"blob": "y" * 2_000},
    )

    assert snapshots._LOCAL_TOTAL_BYTES <= 1_200
    assert (
        snapshots.resolve_snapshot(
            first_reference,
            _UnavailableEngine(),
        )["blob"]
        == "x" * 2_000
    )
    assert snapshots._LOCAL_TOTAL_BYTES <= 1_200


def test_memory_eviction_prunes_latest_pointers_for_unique_source_keys(
    monkeypatch,
):
    monkeypatch.setenv(snapshots.MEMORY_MAX_BYTES_ENV, "1024")

    for index in range(250):
        snapshots._cache_local(
            "latest-pointer-bound-test",
            f"source-{index}",
            f"revision-{index}",
            {"blob": "x" * 2_000},
            {},
        )

    assert snapshots._LOCAL_TOTAL_BYTES == 0
    assert snapshots._LOCAL_PAYLOADS == {}
    assert not [
        key
        for key in snapshots._LOCAL_LATEST
        if key[0] == "latest-pointer-bound-test"
    ]


def test_evicting_old_revision_does_not_remove_newer_latest_pointer(
    monkeypatch,
):
    monkeypatch.setenv(snapshots.MEMORY_MAX_BYTES_ENV, "2500")
    namespace = "latest-pointer-revision-test"
    source_key = "same-source"

    snapshots._cache_local(
        namespace,
        source_key,
        "old",
        {"blob": "x" * 800},
        {},
    )
    snapshots._cache_local(
        namespace,
        source_key,
        "new",
        {"blob": "y" * 800},
        {},
    )

    assert (namespace, source_key, "old") not in snapshots._LOCAL_PAYLOADS
    assert (namespace, source_key, "new") in snapshots._LOCAL_PAYLOADS
    assert snapshots._LOCAL_LATEST[(namespace, source_key)] == "new"


def test_persistent_directories_are_external_owner_only_and_defaults_are_bounded(
    monkeypatch,
    tmp_path,
):
    disk_limit = 8 * 1024 * 1024
    cache_directory = _enable_persistence(
        monkeypatch,
        tmp_path,
        disk=disk_limit,
    )
    snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="permission-test",
        source_key="source",
        builder=lambda: {"ok": True},
    )
    stores = snapshots._get_persistent_stores()

    assert snapshots.DEFAULT_MEMORY_MAX_BYTES == 512 * 1024 * 1024
    assert snapshots.DEFAULT_DISK_MAX_BYTES == 2 * 1024 * 1024 * 1024
    assert stores.cache.size_limit == disk_limit
    assert snapshots._REPOSITORY_ROOT not in cache_directory.resolve().parents
    for directory in (
        cache_directory,
        cache_directory / "locks",
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    marker = cache_directory / snapshots._CACHE_MARKER_NAME
    assert marker.read_bytes() == snapshots._CACHE_MARKER_CONTENT
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    keys = list(stores.cache.iterkeys())
    assert any(str(key).startswith("record:") for key in keys)
    assert any(str(key).startswith("latest:") for key in keys)
    lock_files = list((cache_directory / "locks").glob("*.lock"))
    assert len(lock_files) == 1
    assert stat.S_IMODE(lock_files[0].stat().st_mode) == 0o600


def test_existing_unmarked_shared_directory_is_rejected_without_chmod(
    monkeypatch,
    tmp_path,
):
    shared_directory = tmp_path / "shared-cache"
    shared_directory.mkdir(mode=0o755)
    (shared_directory / "unrelated.txt").write_text(
        "not a dashboard cache",
        encoding="utf-8",
    )
    original_mode = stat.S_IMODE(shared_directory.stat().st_mode)
    monkeypatch.setenv(snapshots.LOCAL_PERSISTENCE_ENV, "1")
    monkeypatch.setenv(
        snapshots.LOCAL_CACHE_DIR_ENV,
        str(shared_directory),
    )
    snapshots.close_persistent_snapshot_cache()

    with pytest.raises(snapshots._PersistentStorageError):
        snapshots._get_persistent_stores()

    assert stat.S_IMODE(shared_directory.stat().st_mode) == original_mode
    assert not (shared_directory / snapshots._CACHE_MARKER_NAME).exists()


def test_existing_empty_owner_only_directory_becomes_dedicated_cache(
    monkeypatch,
    tmp_path,
):
    dedicated_directory = tmp_path / "dedicated-cache"
    dedicated_directory.mkdir(mode=0o700)
    monkeypatch.setenv(snapshots.LOCAL_PERSISTENCE_ENV, "1")
    monkeypatch.setenv(
        snapshots.LOCAL_CACHE_DIR_ENV,
        str(dedicated_directory),
    )
    snapshots.close_persistent_snapshot_cache()

    stores = snapshots._get_persistent_stores()

    assert stores.root == dedicated_directory
    assert stat.S_IMODE(dedicated_directory.stat().st_mode) == 0o700
    marker = dedicated_directory / snapshots._CACHE_MARKER_NAME
    assert marker.read_bytes() == snapshots._CACHE_MARKER_CONTENT
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600


def test_cross_process_lock_files_use_fixed_bounded_stripes(
    monkeypatch,
    tmp_path,
):
    cache_directory = _enable_persistence(monkeypatch, tmp_path)
    stores = snapshots._get_persistent_stores()

    for index in range(512):
        with snapshots._disk_source_lock(
            stores,
            "lock-stripe-test",
            f"source-{index}",
        ):
            pass

    lock_names = {
        path.name for path in (cache_directory / "locks").glob("*.lock")
    }
    allowed_names = {
        f"stripe-{index:02d}.lock"
        for index in range(snapshots._LOCK_STRIPE_COUNT)
    }
    assert 1 <= len(lock_names) <= snapshots._LOCK_STRIPE_COUNT
    assert lock_names <= allowed_names


def test_clear_defaults_to_memory_only_and_explicit_persistent_clear_deletes(
    monkeypatch,
    tmp_path,
):
    _enable_persistence(monkeypatch, tmp_path)
    reference, _ = snapshots.get_or_build_snapshot(
        _UnavailableEngine(),
        namespace="clear-test",
        source_key="source",
        builder=lambda: {"ok": True},
    )

    snapshots.clear_local_snapshots("clear-test")
    assert snapshots.resolve_snapshot(
        reference,
        _UnavailableEngine(),
    ) == {"ok": True}

    snapshots.clear_local_snapshots(
        "clear-test",
        persistent=True,
    )
    snapshots.clear_local_snapshots("clear-test")
    with pytest.raises(
        snapshots.SnapshotUnavailable,
        match="missing or has been evicted",
    ):
        snapshots.resolve_snapshot(
            reference,
            _UnavailableEngine(),
        )


def test_cross_process_single_flight_builds_once_and_reuses_exact_revision(
    monkeypatch,
    tmp_path,
):
    cache_directory = _enable_persistence(monkeypatch, tmp_path)
    snapshots.close_persistent_snapshot_cache()
    snapshots.clear_local_snapshots()
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    builder_calls = context.Value("i", 0)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_cross_process_snapshot_worker,
            args=(
                str(cache_directory),
                start_event,
                builder_calls,
                result_queue,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    results = [result_queue.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    assert [result[0] for result in results] == ["ok", "ok"]
    references = [result[1] for result in results]
    assert references[0] == references[1]
    assert isinstance(references[0]["revision"], str)
    snapshots._validate_disk_revision_token(
        references[0]["revision"]
    )
    assert builder_calls.value == 1
    assert [result[2] for result in results] == [
        {"rows": [1, 2, 3]},
        {"rows": [1, 2, 3]},
    ]


def test_atomic_cache_initialization_spawn_stress_uses_shared_disk(
    tmp_path,
):
    context = multiprocessing.get_context("spawn")

    for round_index in range(20):
        cache_directory = tmp_path / f"initialization-round-{round_index}"
        start_event = context.Event()
        builder_calls = context.Value("i", 0)
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_cross_process_snapshot_worker,
                args=(
                    str(cache_directory),
                    start_event,
                    builder_calls,
                    result_queue,
                ),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        start_event.set()
        results = [
            result_queue.get(timeout=15)
            for _ in processes
        ]
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        result_queue.close()
        result_queue.join_thread()

        assert [result[0] for result in results] == ["ok", "ok"]
        references = [result[1] for result in results]
        assert references[0] == references[1]
        assert all(reference["shared"] for reference in references)
        assert builder_calls.value == 1
        marker = cache_directory / snapshots._CACHE_MARKER_NAME
        assert marker.read_bytes() == snapshots._CACHE_MARKER_CONTENT
