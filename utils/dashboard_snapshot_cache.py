"""Versioned prepared-data snapshots for large Dash callback stores.

The public browser contract is intentionally tiny: callbacks receive a
``dashboard_source_ref_v1`` mapping and resolve the immutable payload on the
server.  The cache has three layers:

* a bounded in-process LRU for the normal warm path;
* an optional Postgres snapshot table for cross-process resolution; and
* the caller's legacy payload fallback when the migration is unavailable.

No request-time DDL is performed.  Payloads use a deterministic, compressed
JSON codec with explicit pandas dtype metadata instead of pickle.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from contextlib import suppress
import base64
import datetime as dt
import hashlib
import json
import logging
import math
import threading
import time
import zlib
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from dash import ctx
from dash.exceptions import MissingCallbackContextException
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import BYTEA, JSONB


LOGGER = logging.getLogger(__name__)

REFERENCE_FORMAT = "dashboard_source_ref_v1"
PAYLOAD_CODEC = "zlib-json-v1"
SNAPSHOT_TABLE = "at_lng.dashboard_prepared_snapshots"

_MAX_LOCAL_ENTRIES = 32
_LOCAL_PAYLOADS: "OrderedDict[tuple[str, str, int], Any]" = OrderedDict()
_LOCAL_MANIFESTS: dict[tuple[str, str, int], dict[str, Any]] = {}
_LOCAL_SHARED: dict[tuple[str, str, int], bool] = {}
_LOCAL_LATEST: dict[tuple[str, str], int] = {}
_LOCAL_LOCK = threading.RLock()
_SINGLE_FLIGHTS: dict[tuple[str, str, bool], Future] = {}
_SHARED_SCHEMA_STATE: dict[int, bool] = {}


class SnapshotUnavailable(RuntimeError):
    """Raised when a browser reference cannot be resolved safely."""


def _normalize_key_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_key_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_normalize_key_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return "__nan__"
    return value


def build_source_key(namespace: str, *parts: Any, **named_parts: Any) -> str:
    """Return a stable source key from watermarks, mappings, and filters."""
    payload = {
        "namespace": namespace,
        "parts": _normalize_key_value(parts),
        "named": _normalize_key_value(named_parts),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prepare_manifest(
    manifest: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]] | None,
    payload: Any,
) -> dict[str, Any]:
    raw_manifest = manifest(payload) if callable(manifest) else dict(manifest or {})
    normalized = _normalize_key_value(raw_manifest)
    return dict(normalized or {})


def was_global_refresh_triggered() -> bool:
    """Return whether the active Dash callback was triggered by global refresh."""
    try:
        return ctx.triggered_id == "global-refresh-button"
    except MissingCallbackContextException:
        return False


def _pack_scalar(value: Any) -> Any:
    try:
        if value is not None and pd.isna(value):
            if isinstance(value, (pd.Timestamp, dt.datetime, dt.date, np.datetime64)):
                return {"__dashboard_type__": "nat"}
    except (TypeError, ValueError):
        pass
    if value is pd.NA:
        return {"__dashboard_type__": "pd_na"}
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return {"__dashboard_type__": "nat"}
        return {"__dashboard_type__": "timestamp", "value": value.isoformat()}
    if isinstance(value, dt.datetime):
        return {"__dashboard_type__": "datetime", "value": value.isoformat()}
    if isinstance(value, dt.date):
        return {"__dashboard_type__": "date", "value": value.isoformat()}
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return {"__dashboard_type__": "nat"}
        return {"__dashboard_type__": "timestamp", "value": pd.Timestamp(value).isoformat()}
    if isinstance(value, np.generic):
        return _pack_scalar(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return {"__dashboard_type__": "nan"}
        if math.isinf(value):
            return {"__dashboard_type__": "infinity", "sign": 1 if value > 0 else -1}
    if isinstance(value, bytes):
        return {
            "__dashboard_type__": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    return value


def _pack_payload(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "__dashboard_type__": "dataframe",
            "columns": [_pack_payload(item) for item in value.columns.tolist()],
            "index": [_pack_payload(item) for item in value.index.tolist()],
            "data": [
                [_pack_payload(item) for item in row]
                for row in value.itertuples(index=False, name=None)
            ],
            "dtypes": [str(dtype) for dtype in value.dtypes.tolist()],
            "index_dtype": str(value.index.dtype),
            "index_names": [_pack_payload(item) for item in value.index.names],
        }
    if isinstance(value, pd.Series):
        return {
            "__dashboard_type__": "series",
            "name": _pack_payload(value.name),
            "index": [_pack_payload(item) for item in value.index.tolist()],
            "data": [_pack_payload(item) for item in value.tolist()],
            "dtype": str(value.dtype),
            "index_dtype": str(value.index.dtype),
            "index_names": [_pack_payload(item) for item in value.index.names],
        }
    if isinstance(value, tuple):
        return {"__dashboard_type__": "tuple", "items": [_pack_payload(item) for item in value]}
    if isinstance(value, Mapping):
        return {str(key): _pack_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pack_payload(item) for item in value]
    return _pack_scalar(value)


def _restore_dtype(series: pd.Series, dtype_name: str) -> pd.Series:
    try:
        if dtype_name.startswith("datetime64"):
            return pd.to_datetime(series, errors="coerce")
        if dtype_name.startswith("timedelta64"):
            return pd.to_timedelta(series, errors="coerce")
        if dtype_name == "category":
            return series.astype("category")
        return series.astype(dtype_name)
    except (TypeError, ValueError):
        return series


def _restore_index(values: list[Any], dtype_name: str, names: list[Any]) -> pd.Index:
    index = pd.Index(values)
    if dtype_name.startswith("datetime64"):
        index = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    else:
        with suppress(TypeError, ValueError):
            index = index.astype(dtype_name)
    if len(names) == 1:
        index.name = names[0]
    return index


def _unpack_payload(value: Any) -> Any:
    if isinstance(value, list):
        return [_unpack_payload(item) for item in value]
    if not isinstance(value, dict):
        return value

    value_type = value.get("__dashboard_type__")
    if value_type == "pd_na":
        return pd.NA
    if value_type == "nat":
        return pd.NaT
    if value_type == "timestamp":
        return pd.Timestamp(value["value"])
    if value_type == "datetime":
        return dt.datetime.fromisoformat(value["value"])
    if value_type == "date":
        return dt.date.fromisoformat(value["value"])
    if value_type == "nan":
        return float("nan")
    if value_type == "infinity":
        return float("inf") if value.get("sign", 1) > 0 else float("-inf")
    if value_type == "bytes":
        return base64.b64decode(value["value"])
    if value_type == "tuple":
        return tuple(_unpack_payload(item) for item in value.get("items", []))
    if value_type == "dataframe":
        columns = [_unpack_payload(item) for item in value.get("columns", [])]
        rows = [
            [_unpack_payload(item) for item in row]
            for row in value.get("data", [])
        ]
        frame = pd.DataFrame(rows, columns=columns)
        for column, dtype_name in zip(frame.columns, value.get("dtypes", [])):
            frame[column] = _restore_dtype(frame[column], dtype_name)
        frame.index = _restore_index(
            [_unpack_payload(item) for item in value.get("index", [])],
            value.get("index_dtype", "object"),
            [_unpack_payload(item) for item in value.get("index_names", [None])],
        )
        return frame
    if value_type == "series":
        series = pd.Series(
            [_unpack_payload(item) for item in value.get("data", [])],
            name=_unpack_payload(value.get("name")),
        )
        series = _restore_dtype(series, value.get("dtype", "object"))
        series.index = _restore_index(
            [_unpack_payload(item) for item in value.get("index", [])],
            value.get("index_dtype", "object"),
            [_unpack_payload(item) for item in value.get("index_names", [None])],
        )
        return series
    return {key: _unpack_payload(item) for key, item in value.items()}


def encode_snapshot_payload(payload: Any) -> bytes:
    packed = _pack_payload(payload)
    raw = json.dumps(
        packed,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return zlib.compress(raw, level=6)


def decode_snapshot_payload(encoded: bytes | bytearray | memoryview) -> Any:
    raw_bytes = bytes(encoded)
    packed = json.loads(zlib.decompress(raw_bytes).decode("utf-8"))
    return _unpack_payload(packed)


def _snapshot_ref(namespace: str, source_key: str, revision: int, *, shared: bool) -> dict[str, Any]:
    return {
        "format": REFERENCE_FORMAT,
        "namespace": namespace,
        "source_key": source_key,
        "revision": int(revision),
        "shared": bool(shared),
    }


def with_snapshot_slot(reference: Mapping[str, Any], slot: str) -> dict[str, Any]:
    result = dict(reference)
    result["slot"] = slot
    return result


def is_snapshot_reference(value: Any, namespace: str | None = None) -> bool:
    if not isinstance(value, Mapping) or value.get("format") != REFERENCE_FORMAT:
        return False
    return namespace is None or value.get("namespace") == namespace


def snapshot_is_shared(reference: Mapping[str, Any] | None) -> bool:
    return bool(reference and reference.get("shared"))


def _cache_local(
    namespace: str,
    source_key: str,
    revision: int,
    payload: Any,
    manifest: Mapping[str, Any] | None,
    *,
    shared: bool = False,
) -> None:
    cache_key = (namespace, source_key, int(revision))
    with _LOCAL_LOCK:
        _LOCAL_PAYLOADS[cache_key] = payload
        _LOCAL_PAYLOADS.move_to_end(cache_key)
        _LOCAL_MANIFESTS[cache_key] = dict(manifest or {})
        _LOCAL_SHARED[cache_key] = bool(shared)
        _LOCAL_LATEST[(namespace, source_key)] = int(revision)
        while len(_LOCAL_PAYLOADS) > _MAX_LOCAL_ENTRIES:
            oldest_key, _ = _LOCAL_PAYLOADS.popitem(last=False)
            _LOCAL_MANIFESTS.pop(oldest_key, None)
            _LOCAL_SHARED.pop(oldest_key, None)


def _get_local(namespace: str, source_key: str, revision: int | None = None):
    with _LOCAL_LOCK:
        if revision is None:
            revision = _LOCAL_LATEST.get((namespace, source_key))
        if revision is None:
            return None
        cache_key = (namespace, source_key, int(revision))
        payload = _LOCAL_PAYLOADS.get(cache_key)
        if payload is None:
            return None
        _LOCAL_PAYLOADS.move_to_end(cache_key)
        return (
            int(revision),
            payload,
            dict(_LOCAL_MANIFESTS.get(cache_key, {})),
            bool(_LOCAL_SHARED.get(cache_key, False)),
        )


def shared_snapshot_schema_available(engine) -> bool:
    engine_key = id(engine)
    if engine_key in _SHARED_SCHEMA_STATE:
        return _SHARED_SCHEMA_STATE[engine_key]
    try:
        with engine.connect() as connection:
            available = connection.execute(
                text("SELECT to_regclass(:table_name) IS NOT NULL"),
                {"table_name": SNAPSHOT_TABLE},
            ).scalar()
    except Exception:
        LOGGER.debug("Shared dashboard snapshot schema check failed", exc_info=True)
        return False
    _SHARED_SCHEMA_STATE[engine_key] = bool(available)
    return bool(available)


_READ_LATEST_SQL = text(f"""
    SELECT revision, codec, manifest_jsonb, payload_bytea
    FROM {SNAPSHOT_TABLE}
    WHERE namespace = :namespace AND source_key = :source_key
    ORDER BY revision DESC
    LIMIT 1
""")

_READ_EXACT_SQL = text(f"""
    SELECT revision, codec, manifest_jsonb, payload_bytea
    FROM {SNAPSHOT_TABLE}
    WHERE namespace = :namespace
      AND source_key = :source_key
      AND revision = :revision
""")

_MAX_REVISION_SQL = text(f"""
    SELECT COALESCE(MAX(revision), 0)
    FROM {SNAPSHOT_TABLE}
    WHERE namespace = :namespace AND source_key = :source_key
""")

_UPSERT_SQL = text(f"""
    INSERT INTO {SNAPSHOT_TABLE} (
        namespace, source_key, revision, codec, manifest_jsonb, payload_bytea,
        created_at, updated_at
    ) VALUES (
        :namespace, :source_key, :revision, :codec, :manifest_jsonb,
        :payload_bytea, now(), now()
    )
    ON CONFLICT (namespace, source_key, revision) DO UPDATE SET
        codec = EXCLUDED.codec,
        manifest_jsonb = EXCLUDED.manifest_jsonb,
        payload_bytea = EXCLUDED.payload_bytea,
        updated_at = now()
""").bindparams(
    bindparam("manifest_jsonb", type_=JSONB),
    bindparam("payload_bytea", type_=BYTEA),
)


def _decode_shared_row(row, namespace: str, source_key: str):
    if row is None:
        return None
    mapping = row._mapping if hasattr(row, "_mapping") else row
    codec = mapping["codec"]
    if codec != PAYLOAD_CODEC:
        raise SnapshotUnavailable(f"Unsupported dashboard snapshot codec: {codec}")
    revision = int(mapping["revision"])
    payload = decode_snapshot_payload(mapping["payload_bytea"])
    manifest = dict(mapping["manifest_jsonb"] or {})
    _cache_local(namespace, source_key, revision, payload, manifest, shared=True)
    return revision, payload, manifest


def _read_shared(engine, namespace: str, source_key: str, revision: int | None = None):
    query = _READ_EXACT_SQL if revision is not None else _READ_LATEST_SQL
    params = {"namespace": namespace, "source_key": source_key}
    if revision is not None:
        params["revision"] = int(revision)
    with engine.connect() as connection:
        row = connection.execute(query, params).first()
    return _decode_shared_row(row, namespace, source_key)


def _next_revision(engine, namespace: str, source_key: str) -> int:
    with engine.connect() as connection:
        latest = connection.execute(
            _MAX_REVISION_SQL,
            {"namespace": namespace, "source_key": source_key},
        ).scalar()
    return int(latest or 0) + 1


def _write_shared(
    engine,
    namespace: str,
    source_key: str,
    revision: int,
    payload: Any,
    manifest: Mapping[str, Any] | None,
) -> None:
    encoded = encode_snapshot_payload(payload)
    LOGGER.info(
        "dashboard_snapshot publish namespace=%s source_key=%s revision=%s codec=%s payload_bytes=%s",
        namespace,
        source_key,
        revision,
        PAYLOAD_CODEC,
        len(encoded),
    )
    with engine.begin() as connection:
        connection.execute(
            _UPSERT_SQL,
            {
                "namespace": namespace,
                "source_key": source_key,
                "revision": int(revision),
                "codec": PAYLOAD_CODEC,
                "manifest_jsonb": dict(manifest or {}),
                "payload_bytea": encoded,
            },
        )


def _run_single_flight(key: tuple[str, str, bool], builder: Callable[[], Any]) -> Any:
    with _LOCAL_LOCK:
        future = _SINGLE_FLIGHTS.get(key)
        if future is None:
            future = Future()
            _SINGLE_FLIGHTS[key] = future
            leader = True
        else:
            leader = False
    if not leader:
        return future.result()
    try:
        result = builder()
        future.set_result(result)
        return result
    except BaseException as exc:
        future.set_exception(exc)
        raise
    finally:
        with _LOCAL_LOCK:
            _SINGLE_FLIGHTS.pop(key, None)


def get_or_build_snapshot(
    engine,
    *,
    namespace: str,
    source_key: str,
    builder: Callable[[], Any],
    manifest: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]] | None = None,
    force: bool = False,
) -> tuple[dict[str, Any], Any]:
    """Resolve or build an immutable snapshot.

    The second return value is the exact in-memory payload so callers can
    return their legacy browser data if shared storage is unavailable.
    """
    request_started = time.perf_counter()
    if not force:
        local = _get_local(namespace, source_key)
        if local is not None:
            revision, payload, _, local_shared = local
            LOGGER.info(
                "dashboard_snapshot hit namespace=%s source_key=%s revision=%s cache=local duration_ms=%.1f",
                namespace,
                source_key,
                revision,
                (time.perf_counter() - request_started) * 1000,
            )
            return _snapshot_ref(namespace, source_key, revision, shared=local_shared), payload

    shared_available = shared_snapshot_schema_available(engine)

    def build_or_load():
        def build_local(prebuilt_payload=None, *, has_prebuilt=False):
            local = _get_local(namespace, source_key)
            revision = (local[0] + 1) if local is not None else 1
            build_started = time.perf_counter()
            payload = prebuilt_payload if has_prebuilt else builder()
            build_ms = 0.0 if has_prebuilt else (time.perf_counter() - build_started) * 1000
            payload_manifest = _prepare_manifest(manifest, payload)
            _cache_local(namespace, source_key, revision, payload, payload_manifest, shared=False)
            LOGGER.info(
                "dashboard_snapshot build namespace=%s source_key=%s revision=%s cache=local build_ms=%.1f duration_ms=%.1f",
                namespace,
                source_key,
                revision,
                build_ms,
                (time.perf_counter() - request_started) * 1000,
            )
            return _snapshot_ref(namespace, source_key, revision, shared=False), payload

        if not shared_available:
            return build_local()

        if not force:
            try:
                shared = _read_shared(engine, namespace, source_key)
            except Exception:
                LOGGER.warning(
                    "Shared dashboard snapshot read failed; using local fallback",
                    exc_info=True,
                )
                return build_local()
            if shared is not None:
                revision, payload, _ = shared
                LOGGER.info(
                    "dashboard_snapshot hit namespace=%s source_key=%s revision=%s cache=postgres duration_ms=%.1f",
                    namespace,
                    source_key,
                    revision,
                    (time.perf_counter() - request_started) * 1000,
                )
                return _snapshot_ref(namespace, source_key, revision, shared=True), payload

        lock_name = f"dashboard-snapshot:{namespace}:{source_key}"
        payload = None
        builder_started = False
        payload_built = False
        try:
            with engine.connect() as lock_connection:
                lock_connection.execute(
                    text("SELECT pg_advisory_lock(hashtextextended(:lock_name, 0))"),
                    {"lock_name": lock_name},
                )
                try:
                    if not force:
                        shared = _read_shared(engine, namespace, source_key)
                        if shared is not None:
                            revision, payload, _ = shared
                            LOGGER.info(
                                "dashboard_snapshot hit namespace=%s source_key=%s revision=%s cache=postgres-after-lock duration_ms=%.1f",
                                namespace,
                                source_key,
                                revision,
                                (time.perf_counter() - request_started) * 1000,
                            )
                            return _snapshot_ref(namespace, source_key, revision, shared=True), payload
                    revision = _next_revision(engine, namespace, source_key)
                    # Builder failures are data-path failures and must propagate.
                    # Only shared persistence failures fall back to local storage.
                    builder_started = True
                    build_started = time.perf_counter()
                    payload = builder()
                    payload_built = True
                    build_ms = (time.perf_counter() - build_started) * 1000
                    payload_manifest = _prepare_manifest(manifest, payload)
                    _write_shared(
                        engine,
                        namespace,
                        source_key,
                        revision,
                        payload,
                        payload_manifest,
                    )
                    _cache_local(namespace, source_key, revision, payload, payload_manifest, shared=True)
                    LOGGER.info(
                        "dashboard_snapshot build namespace=%s source_key=%s revision=%s cache=postgres build_ms=%.1f duration_ms=%.1f",
                        namespace,
                        source_key,
                        revision,
                        build_ms,
                        (time.perf_counter() - request_started) * 1000,
                    )
                    return _snapshot_ref(namespace, source_key, revision, shared=True), payload
                finally:
                    with suppress(Exception):
                        lock_connection.execute(
                            text("SELECT pg_advisory_unlock(hashtextextended(:lock_name, 0))"),
                            {"lock_name": lock_name},
                        )
        except Exception:
            if builder_started and not payload_built:
                raise
            if not payload_built:
                LOGGER.warning(
                    "Shared dashboard snapshot storage unavailable; using local fallback",
                    exc_info=True,
                )
                return build_local()
            LOGGER.warning(
                "Shared dashboard snapshot publish failed; retaining prepared data locally",
                exc_info=True,
            )
            return build_local(payload, has_prebuilt=True)

    return _run_single_flight((namespace, source_key, bool(force)), build_or_load)


def resolve_snapshot(
    value: Any,
    engine,
    *,
    expected_namespace: str | None = None,
    slot: str | None = None,
) -> Any:
    """Resolve a browser reference, or return a legacy value unchanged."""
    if not is_snapshot_reference(value, expected_namespace):
        return value
    namespace = str(value["namespace"])
    source_key = str(value["source_key"])
    revision = int(value["revision"])
    local = _get_local(namespace, source_key, revision)
    if local is None:
        if not shared_snapshot_schema_available(engine):
            raise SnapshotUnavailable(
                f"Snapshot {namespace}/{source_key}/{revision} is not available in this process"
            )
        local = _read_shared(engine, namespace, source_key, revision)
    if local is None:
        raise SnapshotUnavailable(
            f"Snapshot {namespace}/{source_key}/{revision} does not exist"
        )
    payload = local[1]
    requested_slot = slot if slot is not None else value.get("slot")
    if requested_slot is None:
        return payload
    if not isinstance(payload, Mapping) or requested_slot not in payload:
        raise SnapshotUnavailable(
            f"Snapshot {namespace}/{source_key}/{revision} has no slot {requested_slot!r}"
        )
    return payload[requested_slot]


def clear_local_snapshots(namespace: str | None = None) -> None:
    """Clear local entries, primarily for deterministic tests and refreshes."""
    with _LOCAL_LOCK:
        if namespace is None:
            _LOCAL_PAYLOADS.clear()
            _LOCAL_MANIFESTS.clear()
            _LOCAL_SHARED.clear()
            _LOCAL_LATEST.clear()
            return
        for key in [key for key in _LOCAL_PAYLOADS if key[0] == namespace]:
            _LOCAL_PAYLOADS.pop(key, None)
            _LOCAL_MANIFESTS.pop(key, None)
            _LOCAL_SHARED.pop(key, None)
        for key in [key for key in _LOCAL_LATEST if key[0] == namespace]:
            _LOCAL_LATEST.pop(key, None)


def pack_record_mapping(record_mapping: Mapping[str, list[Mapping[str, Any]]] | None) -> dict[str, Any]:
    """Column-pack an ordered entity -> record-list mapping losslessly."""
    entities: list[str] = []
    columns: list[str] = []
    column_seen: set[str] = set()
    rows: list[list[Any]] = []
    present_columns: list[list[int]] = []
    entity_row_counts: list[int] = []

    for entity, records in (record_mapping or {}).items():
        entities.append(entity)
        entity_records = list(records or [])
        entity_row_counts.append(len(entity_records))
        for record in entity_records:
            for column in record:
                if column not in column_seen:
                    column_seen.add(column)
                    columns.append(column)

    for records in (record_mapping or {}).values():
        for record in records or []:
            rows.append([record.get(column) for column in columns])
            present_columns.append([
                index for index, column in enumerate(columns) if column in record
            ])

    return {
        "format": "entity_record_cube_v1",
        "entities": entities,
        "columns": columns,
        "entity_row_counts": entity_row_counts,
        "rows": rows,
        "present_columns": present_columns,
    }


def unpack_record_mapping(payload: Any) -> Any:
    """Restore an entity record cube; accept legacy mappings unchanged."""
    if not isinstance(payload, Mapping) or payload.get("format") != "entity_record_cube_v1":
        return payload
    columns = list(payload.get("columns") or [])
    rows = list(payload.get("rows") or [])
    present_columns = list(payload.get("present_columns") or [])
    if len(present_columns) != len(rows):
        present_columns = [list(range(len(columns))) for _ in rows]
    result: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for entity, row_count in zip(
        payload.get("entities") or [],
        payload.get("entity_row_counts") or [],
    ):
        count = int(row_count)
        entity_rows = rows[offset:offset + count]
        entity_presence = present_columns[offset:offset + count]
        result[entity] = [
            {
                columns[index]: row[index]
                for index in (presence if presence is not None else range(len(columns)))
            }
            for row, presence in zip(entity_rows, entity_presence)
        ]
        offset += count
    return result
