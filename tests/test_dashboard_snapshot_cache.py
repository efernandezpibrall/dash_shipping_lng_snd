import datetime as dt
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

from utils import dashboard_snapshot_cache as snapshots


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
