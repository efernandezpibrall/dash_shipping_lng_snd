"""Versioned prepared-data snapshots for large Dash callback stores.

The public browser contract is intentionally tiny: callbacks receive a
``dashboard_source_ref_v1`` mapping and resolve the immutable payload on the
server.  The cache has three layers:

* a byte-bounded in-process LRU for the normal warm path;
* a bounded same-host diskcache for restart and worker-process resolution; and
* the caller's legacy payload fallback when persistence is unavailable.

The rollback switch restores the prior optional Postgres-table path. No
request-time DDL is performed. Payloads use deterministic compressed JSON
bytes with explicit pandas dtype metadata instead of arbitrary payload pickle.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
import base64
import datetime as dt
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import stat
import sys
import threading
import time
import uuid
import zlib
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from dash import ctx
from dash.exceptions import MissingCallbackContextException
try:
    from diskcache import Cache as DiskCache
except ImportError:  # pragma: no cover - exercised through the disabled fallback
    DiskCache = None
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import BYTEA, JSONB

from utils.database import DB_SCHEMA


LOGGER = logging.getLogger(__name__)

REFERENCE_FORMAT = "dashboard_source_ref_v1"
PAYLOAD_CODEC = "zlib-json-v1"
SNAPSHOT_TABLE = f"{DB_SCHEMA}.dashboard_prepared_snapshots"

LOCAL_PERSISTENCE_ENV = "DASHBOARD_SNAPSHOT_LOCAL_PERSISTENCE_ENABLED"
LOCAL_CACHE_DIR_ENV = "DASHBOARD_SNAPSHOT_CACHE_DIR"
MEMORY_MAX_BYTES_ENV = "DASHBOARD_SNAPSHOT_MEMORY_MAX_BYTES"
DISK_MAX_BYTES_ENV = "DASHBOARD_SNAPSHOT_DISK_MAX_BYTES"

DEFAULT_MEMORY_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_DISK_MAX_BYTES = 2 * 1024 * 1024 * 1024
DISK_RECORD_FORMAT = "dashboard_snapshot_disk_record_v2"
SNAPSHOT_EVENT_FORMAT = "dashboard_snapshot_event_v1"
PUBLICATION_BUNDLE_FORMAT = "dashboard_snapshot_bundle_v1"
_DISK_HEADER_LENGTH_BYTES = 8
_DISK_RECORD_CHECKSUM_BYTES = 32
_CACHE_MARKER_NAME = ".dashboard-snapshot-cache-v1"
_CACHE_MARKER_CONTENT = b"dash_shipping_lng_snd dashboard snapshot cache v1\n"
_CACHE_INIT_LOCK_NAME = ".dashboard-snapshot-init.lock"
_COMPOUND_LOCK_NAME = "compound.lock"
_LOCK_STRIPE_COUNT = 64
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()

_LOCAL_PAYLOADS: "OrderedDict[tuple[str, str, int | str], Any]" = OrderedDict()
_LOCAL_MANIFESTS: dict[tuple[str, str, int | str], dict[str, Any]] = {}
_LOCAL_SHARED: dict[tuple[str, str, int | str], bool] = {}
_LOCAL_BACKENDS: dict[tuple[str, str, int | str], str] = {}
_LOCAL_SIZES: dict[tuple[str, str, int | str], int] = {}
_LOCAL_TOTAL_BYTES = 0
_LOCAL_LATEST: dict[tuple[str, str], int | str] = {}
_LOCAL_LOCK = threading.RLock()
_SINGLE_FLIGHTS: dict[tuple[str, str, bool], Future] = {}
_SHARED_SCHEMA_STATE: dict[int, bool] = {}
_PERSISTENT_STORES = None
_PERSISTENT_STORES_CONFIG: tuple[str, int] | None = None
_PERSISTENT_STORES_LOCK = threading.RLock()
_DISK_STORES: set[Any] = set()
_DISK_STORES_LOCK = threading.RLock()
_FORK_GUARD_STORES: tuple[Any, ...] = ()
_ACTIVE_PUBLICATION_STAGE: ContextVar["SnapshotPublicationStage | None"] = (
    ContextVar("dashboard_snapshot_publication_stage", default=None)
)
_SNAPSHOT_LOCK_WAIT_MS: ContextVar[float] = ContextVar(
    "dashboard_snapshot_lock_wait_ms",
    default=0.0,
)
_SNAPSHOT_READ_BACKEND: ContextVar[str | None] = ContextVar(
    "dashboard_snapshot_read_backend",
    default=None,
)


class SnapshotUnavailable(RuntimeError):
    """Raised when a browser reference cannot be resolved safely."""


class _PersistentStorageError(SnapshotUnavailable):
    """Raised when local persistence itself is unavailable."""


@dataclass(frozen=True)
class SnapshotResult:
    """Detailed outcome for one immutable snapshot lookup or build."""

    reference: dict[str, Any]
    payload: Any
    status: str
    backend: str
    read_ms: float
    build_ms: float
    lock_wait_ms: float
    encoded_bytes: int | None
    decoded_bytes: int


@dataclass(frozen=True)
class _StagedPublicationRecord:
    namespace: str
    source_key: str
    revision: str


@dataclass
class SnapshotPublicationStage:
    """Collect immutable records before atomically advancing their pointers."""

    name: str
    stage_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    records: dict[tuple[str, str], _StagedPublicationRecord] = field(
        default_factory=dict
    )
    committed: bool = False

    def register(
        self,
        namespace: str,
        source_key: str,
        revision: str,
    ) -> None:
        identity = (namespace, source_key)
        existing = self.records.get(identity)
        if existing is not None and existing.revision != revision:
            raise SnapshotUnavailable(
                "A publication stage cannot contain two revisions for the "
                f"same snapshot identity: {namespace}/{source_key}"
            )
        self.records[identity] = _StagedPublicationRecord(
            namespace=namespace,
            source_key=source_key,
            revision=revision,
        )


def _emit_snapshot_event(event: str, **fields: Any) -> None:
    payload = {
        "format": SNAPSHOT_EVENT_FORMAT,
        "event": event,
        **fields,
    }
    LOGGER.info(
        "dashboard_snapshot_event %s",
        json.dumps(
            _normalize_key_value(payload),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    )


def _validate_cache_marker(root: Path, marker: Path) -> None:
    try:
        root_stat = root.stat()
        marker_stat = marker.lstat()
        marker_content = marker.read_bytes()
    except OSError as exc:
        raise _PersistentStorageError(
            f"Dashboard snapshot cache marker is unreadable at {marker}"
        ) from exc
    if root_stat.st_uid != os.getuid():
        raise _PersistentStorageError(
            f"Dashboard snapshot cache is not owned by the current user: {root}"
        )
    if (
        not stat.S_ISREG(marker_stat.st_mode)
        or marker_stat.st_uid != os.getuid()
        or marker_content != _CACHE_MARKER_CONTENT
    ):
        raise _PersistentStorageError(
            f"Dashboard snapshot cache marker is invalid at {marker}"
        )


@contextmanager
def _cache_initialization_lock(root: Path):
    lock_path = root / _CACHE_INIT_LOCK_NAME
    open_flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    try:
        file_descriptor = os.open(
            lock_path,
            open_flags,
            0o600,
        )
    except OSError as exc:
        raise _PersistentStorageError(
            f"Dashboard snapshot initialization lock failed at {lock_path}"
        ) from exc
    try:
        lock_stat = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_uid != os.getuid()
        ):
            raise _PersistentStorageError(
                f"Dashboard snapshot initialization lock is unsafe: {lock_path}"
            )
        os.fchmod(file_descriptor, 0o600)
        fcntl.flock(file_descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            with suppress(OSError):
                fcntl.flock(file_descriptor, fcntl.LOCK_UN)
    finally:
        os.close(file_descriptor)


def _write_cache_marker_atomically(marker: Path) -> None:
    temporary_marker = marker.parent / (
        f"{marker.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    file_descriptor = -1
    directory_descriptor = -1
    try:
        file_descriptor = os.open(
            temporary_marker,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        os.fchmod(file_descriptor, 0o600)
        remaining = memoryview(_CACHE_MARKER_CONTENT)
        while remaining:
            written = os.write(file_descriptor, remaining)
            if written <= 0:
                raise OSError("cache marker write made no progress")
            remaining = remaining[written:]
        os.fsync(file_descriptor)
        os.close(file_descriptor)
        file_descriptor = -1
        os.replace(temporary_marker, marker)
        directory_descriptor = os.open(marker.parent, os.O_RDONLY)
        os.fsync(directory_descriptor)
    except OSError as exc:
        raise _PersistentStorageError(
            f"Dashboard snapshot cache marker could not be published at {marker}"
        ) from exc
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        with suppress(FileNotFoundError):
            temporary_marker.unlink()


def _is_safe_marker_temporary_file(path: Path) -> bool:
    if not path.name.startswith(f"{_CACHE_MARKER_NAME}.tmp-"):
        return False
    try:
        path_stat = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(path_stat.st_mode)
        and path_stat.st_uid == os.getuid()
    )


def _prepare_dedicated_cache_root(root: Path) -> None:
    marker = root / _CACHE_MARKER_NAME
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        root_was_created = True
    except FileExistsError:
        root_was_created = False
    except OSError as exc:
        raise _PersistentStorageError(
            f"Dashboard snapshot cache directory could not be created at {root}"
        ) from exc

    if not root.is_dir():
        raise _PersistentStorageError(
            f"Dashboard snapshot cache path is not a directory: {root}"
        )

    with _cache_initialization_lock(root):
        try:
            root_stat = root.stat()
            entries = list(root.iterdir())
        except OSError as exc:
            raise _PersistentStorageError(
                f"Dashboard snapshot cache directory is unreadable at {root}"
            ) from exc
        if root_stat.st_uid != os.getuid():
            raise _PersistentStorageError(
                "Dashboard snapshot cache is not owned by the current user: "
                f"{root}"
            )

        if marker in entries:
            _validate_cache_marker(root, marker)
            # A valid owner-controlled marker identifies this as our
            # dedicated leaf, so repairing its permissions is safe.
            root.chmod(0o700)
            marker.chmod(0o600)
        else:
            allowed_initialization_entries = {
                root / _CACHE_INIT_LOCK_NAME,
            }
            unexpected_entries = [
                entry
                for entry in entries
                if entry not in allowed_initialization_entries
                and not _is_safe_marker_temporary_file(entry)
            ]
            if unexpected_entries:
                raise _PersistentStorageError(
                    "Refusing unmarked nonempty dashboard snapshot cache "
                    f"directory: {root}"
                )
            if (
                not root_was_created
                and stat.S_IMODE(root_stat.st_mode) != 0o700
            ):
                raise _PersistentStorageError(
                    "An existing unmarked dashboard snapshot cache directory "
                    f"must already be owner-only (0700): {root}"
                )
            if root_was_created:
                root.chmod(0o700)
            entries = [
                entry
                for entry in root.iterdir()
                if entry.name != _CACHE_INIT_LOCK_NAME
            ]
            unexpected_entries = [
                entry
                for entry in entries
                if not _is_safe_marker_temporary_file(entry)
            ]
            if unexpected_entries:
                raise _PersistentStorageError(
                    "Refusing dashboard snapshot cache initialization with "
                    f"unexpected files at {root}"
                )
            for entry in entries:
                entry.unlink()
            _write_cache_marker_atomically(marker)
            _validate_cache_marker(root, marker)
    root.chmod(0o700)
    marker.chmod(0o600)


class _DiskStores:
    def __init__(self, root: Path, size_limit: int):
        _prepare_dedicated_cache_root(root)
        locks_directory = root / "locks"
        locks_directory.mkdir(mode=0o700, exist_ok=True)
        locks_stat = locks_directory.lstat()
        if (
            not stat.S_ISDIR(locks_stat.st_mode)
            or locks_stat.st_uid != os.getuid()
        ):
            raise _PersistentStorageError(
                f"Dashboard snapshot lock directory is unsafe: {locks_directory}"
            )
        locks_directory.chmod(0o700)

        self.root = root
        self.locks_directory = locks_directory
        self.process_id = os.getpid()
        self.active_lock_fds: set[int] = set()
        self.active_lock_fds_guard = threading.RLock()
        self.closed = False
        # Records and latest pointers share one bounded cache, so the
        # configured limit covers the complete DiskCache footprint.
        self.cache = DiskCache(
            str(root),
            size_limit=int(size_limit),
            eviction_policy="least-recently-used",
        )
        with _DISK_STORES_LOCK:
            prior_generation = next(
                (
                    stores
                    for stores in _DISK_STORES
                    if stores.process_id == self.process_id
                    and stores.locks_directory == self.locks_directory
                ),
                None,
            )
            if prior_generation is None:
                self.compound_lock = threading.RLock()
                self.source_locks = tuple(
                    threading.RLock()
                    for _ in range(_LOCK_STRIPE_COUNT)
                )
                self.lock_depths = threading.local()
            else:
                self.compound_lock = prior_generation.compound_lock
                self.source_locks = prior_generation.source_locks
                self.lock_depths = prior_generation.lock_depths
            _DISK_STORES.add(self)

    def close(self) -> None:
        with _DISK_STORES_LOCK:
            with self.active_lock_fds_guard:
                self.closed = True
        try:
            self.cache.close()
        finally:
            _forget_disk_store_if_quiescent(self)


def _forget_disk_store_if_quiescent(stores: _DiskStores) -> None:
    with _DISK_STORES_LOCK:
        with stores.active_lock_fds_guard:
            if stores.closed and not stores.active_lock_fds:
                _DISK_STORES.discard(stores)


def _prepare_snapshot_state_for_fork() -> None:
    global _FORK_GUARD_STORES

    _DISK_STORES_LOCK.acquire()
    _FORK_GUARD_STORES = tuple(sorted(_DISK_STORES, key=id))
    for stores in _FORK_GUARD_STORES:
        stores.active_lock_fds_guard.acquire()


def _release_snapshot_state_after_fork() -> None:
    global _FORK_GUARD_STORES

    for stores in reversed(_FORK_GUARD_STORES):
        stores.active_lock_fds_guard.release()
    _FORK_GUARD_STORES = ()
    _DISK_STORES_LOCK.release()


def _reset_snapshot_state_after_fork() -> None:
    """Discard inherited locks and handles in a newly forked process."""

    global _ACTIVE_PUBLICATION_STAGE
    global _DISK_STORES
    global _DISK_STORES_LOCK
    global _FORK_GUARD_STORES
    global _LOCAL_LOCK
    global _PERSISTENT_STORES
    global _PERSISTENT_STORES_CONFIG
    global _PERSISTENT_STORES_LOCK
    global _SNAPSHOT_LOCK_WAIT_MS
    global _SNAPSHOT_READ_BACKEND

    for inherited_stores in _FORK_GUARD_STORES:
        for file_descriptor in tuple(inherited_stores.active_lock_fds):
            with suppress(OSError):
                os.close(file_descriptor)
    _PERSISTENT_STORES = None
    _PERSISTENT_STORES_CONFIG = None
    _PERSISTENT_STORES_LOCK = threading.RLock()
    _LOCAL_LOCK = threading.RLock()
    _SINGLE_FLIGHTS.clear()
    _ACTIVE_PUBLICATION_STAGE = ContextVar(
        "dashboard_snapshot_publication_stage",
        default=None,
    )
    _SNAPSHOT_LOCK_WAIT_MS = ContextVar(
        "dashboard_snapshot_lock_wait_ms",
        default=0.0,
    )
    _SNAPSHOT_READ_BACKEND = ContextVar(
        "dashboard_snapshot_read_backend",
        default=None,
    )
    _DISK_STORES = set()
    _DISK_STORES_LOCK = threading.RLock()
    _FORK_GUARD_STORES = ()


os.register_at_fork(
    before=_prepare_snapshot_state_for_fork,
    after_in_parent=_release_snapshot_state_after_fork,
    after_in_child=_reset_snapshot_state_after_fork,
)


def _env_flag(name: str, *, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() not in {
        "0",
        "false",
        "no",
        "off",
        "disabled",
    }


def _positive_env_bytes(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return int(default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SnapshotUnavailable(
            f"{name} must be a positive integer number of bytes"
        ) from exc
    if value <= 0:
        raise SnapshotUnavailable(
            f"{name} must be a positive integer number of bytes"
        )
    return value


def local_snapshot_persistence_enabled() -> bool:
    """Return whether same-host persistent snapshots are active."""
    return _env_flag(LOCAL_PERSISTENCE_ENV, default=True)


def _memory_max_bytes() -> int:
    return _positive_env_bytes(
        MEMORY_MAX_BYTES_ENV,
        DEFAULT_MEMORY_MAX_BYTES,
    )


def _disk_max_bytes() -> int:
    return _positive_env_bytes(
        DISK_MAX_BYTES_ENV,
        DEFAULT_DISK_MAX_BYTES,
    )


def _persistent_cache_root() -> Path:
    configured = os.environ.get(LOCAL_CACHE_DIR_ENV)
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_symlink():
            raise SnapshotUnavailable(
                "Dashboard snapshot persistence directory cannot be a symlink"
            )
        root = candidate
    else:
        root = (
            Path.home()
            / ".cache"
            / "dash_shipping_lng_snd"
            / "dashboard_snapshots"
            / "cache-v1"
        )
    resolved = root.resolve(strict=False)
    if resolved == _REPOSITORY_ROOT or _REPOSITORY_ROOT in resolved.parents:
        raise SnapshotUnavailable(
            "Dashboard snapshot persistence directory must be outside the repository"
        )
    return resolved


def _get_persistent_stores() -> _DiskStores:
    global _PERSISTENT_STORES
    global _PERSISTENT_STORES_CONFIG

    if DiskCache is None:
        raise _PersistentStorageError(
            "Local dashboard snapshot persistence requires diskcache"
        )
    if (
        _PERSISTENT_STORES is not None
        and _PERSISTENT_STORES.process_id != os.getpid()
    ):
        _reset_snapshot_state_after_fork()

    root = _persistent_cache_root()
    size_limit = _disk_max_bytes()
    config = (str(root), size_limit)
    with _PERSISTENT_STORES_LOCK:
        if (
            _PERSISTENT_STORES is not None
            and _PERSISTENT_STORES_CONFIG == config
        ):
            return _PERSISTENT_STORES
        if _PERSISTENT_STORES is not None:
            with suppress(Exception):
                _PERSISTENT_STORES.close()
        try:
            stores = _DiskStores(root, size_limit)
        except Exception as exc:
            _PERSISTENT_STORES = None
            _PERSISTENT_STORES_CONFIG = None
            raise _PersistentStorageError(
                f"Dashboard snapshot persistence is unavailable at {root}"
            ) from exc
        _PERSISTENT_STORES = stores
        _PERSISTENT_STORES_CONFIG = config
        return stores


def close_persistent_snapshot_cache() -> None:
    """Close process-local disk handles without deleting persistent entries."""
    global _PERSISTENT_STORES
    global _PERSISTENT_STORES_CONFIG

    with _PERSISTENT_STORES_LOCK:
        if _PERSISTENT_STORES is not None:
            with suppress(Exception):
                _PERSISTENT_STORES.close()
        _PERSISTENT_STORES = None
        _PERSISTENT_STORES_CONFIG = None


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


def _snapshot_ref(
    namespace: str,
    source_key: str,
    revision: int | str,
    *,
    shared: bool,
) -> dict[str, Any]:
    return {
        "format": REFERENCE_FORMAT,
        "namespace": namespace,
        "source_key": source_key,
        "revision": revision,
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


def snapshot_is_resolvable(reference: Mapping[str, Any] | None) -> bool:
    """Return whether a browser reference is safe to resolve server-side."""
    return snapshot_is_shared(reference)


def snapshot_reference_is_available(
    reference: Mapping[str, Any] | None,
    engine,
) -> bool:
    """Verify that an exact shared reference still exists in its backend."""

    if not is_snapshot_reference(reference) or not snapshot_is_shared(reference):
        return False
    try:
        namespace = str(reference["namespace"])
        source_key = str(reference["source_key"])
        if local_snapshot_persistence_enabled():
            revision = _validate_disk_revision_token(reference.get("revision"))
            stores = _get_persistent_stores()
            raw_record = stores.cache.get(
                _disk_record_key(namespace, source_key, revision),
                default=_MISSING,
                retry=True,
            )
            if raw_record is _MISSING:
                return False
            header, encoded_payload = _decode_disk_record_header(raw_record)
            return (
                header.get("format") == DISK_RECORD_FORMAT
                and header.get("codec") == PAYLOAD_CODEC
                and header.get("namespace") == namespace
                and header.get("source_key") == source_key
                and header.get("revision") == revision
                and int(header.get("payload_bytes"))
                == len(encoded_payload)
            )
        revision = int(reference["revision"])
        with engine.connect() as connection:
            return bool(connection.execute(
                _SHARED_REFERENCE_EXISTS_SQL,
                {
                    "namespace": namespace,
                    "source_key": source_key,
                    "revision": revision,
                },
            ).scalar())
    except Exception:
        return False


def _estimate_decoded_bytes(
    value: Any,
    seen: set[int] | None = None,
) -> int:
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)

    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum()) + int(
            value.columns.memory_usage(deep=True)
        )
    if isinstance(value, pd.Series):
        return int(value.memory_usage(index=True, deep=True))
    if isinstance(value, pd.Index):
        return int(value.memory_usage(deep=True))
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sys.getsizeof(value) + sum(
            _estimate_decoded_bytes(key, seen)
            + _estimate_decoded_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return sys.getsizeof(value) + sum(
            _estimate_decoded_bytes(item, seen) for item in value
        )
    if isinstance(value, str):
        return sys.getsizeof(value) + len(value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return sys.getsizeof(value) + len(value)
    return sys.getsizeof(value)


def _cache_local(
    namespace: str,
    source_key: str,
    revision: int | str,
    payload: Any,
    manifest: Mapping[str, Any] | None,
    *,
    shared: bool = False,
    backend: str = "memory",
    advance_latest: bool = True,
) -> None:
    global _LOCAL_TOTAL_BYTES

    cache_key = (namespace, source_key, revision)
    normalized_manifest = dict(manifest or {})
    decoded_bytes = _estimate_decoded_bytes(
        (payload, normalized_manifest)
    )
    max_bytes = _memory_max_bytes()
    with _LOCAL_LOCK:
        previous_size = _LOCAL_SIZES.get(cache_key, 0)
        _LOCAL_TOTAL_BYTES -= previous_size
        _LOCAL_PAYLOADS[cache_key] = payload
        _LOCAL_PAYLOADS.move_to_end(cache_key)
        _LOCAL_MANIFESTS[cache_key] = normalized_manifest
        _LOCAL_SHARED[cache_key] = bool(shared)
        _LOCAL_BACKENDS[cache_key] = backend
        _LOCAL_SIZES[cache_key] = decoded_bytes
        _LOCAL_TOTAL_BYTES += decoded_bytes
        if advance_latest:
            _LOCAL_LATEST[(namespace, source_key)] = revision
        while _LOCAL_TOTAL_BYTES > max_bytes and _LOCAL_PAYLOADS:
            oldest_key, _ = _LOCAL_PAYLOADS.popitem(last=False)
            evicted_bytes = _LOCAL_SIZES.pop(oldest_key, 0)
            _LOCAL_TOTAL_BYTES -= evicted_bytes
            _LOCAL_MANIFESTS.pop(oldest_key, None)
            _LOCAL_SHARED.pop(oldest_key, None)
            _LOCAL_BACKENDS.pop(oldest_key, None)
            oldest_namespace, oldest_source_key, oldest_revision = oldest_key
            latest_key = (oldest_namespace, oldest_source_key)
            if _LOCAL_LATEST.get(latest_key) == oldest_revision:
                _LOCAL_LATEST.pop(latest_key, None)
            _emit_snapshot_event(
                "memory_eviction",
                namespace=oldest_namespace,
                source_key=oldest_source_key,
                revision=oldest_revision,
                decoded_bytes=evicted_bytes,
                memory_bytes=_LOCAL_TOTAL_BYTES,
                memory_limit_bytes=max_bytes,
            )


def _get_local(
    namespace: str,
    source_key: str,
    revision: int | str | None = None,
):
    with _LOCAL_LOCK:
        if revision is None:
            revision = _LOCAL_LATEST.get((namespace, source_key))
        if revision is None:
            return None
        cache_key = (namespace, source_key, revision)
        payload = _LOCAL_PAYLOADS.get(cache_key)
        if payload is None:
            return None
        _LOCAL_PAYLOADS.move_to_end(cache_key)
        return (
            revision,
            payload,
            dict(_LOCAL_MANIFESTS.get(cache_key, {})),
            bool(_LOCAL_SHARED.get(cache_key, False)),
        )


def _get_local_backend(
    namespace: str,
    source_key: str,
    revision: int | str,
) -> str | None:
    with _LOCAL_LOCK:
        return _LOCAL_BACKENDS.get(
            (namespace, source_key, revision)
        )


def _disk_namespace_token(namespace: str) -> str:
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()


def _disk_source_token(namespace: str, source_key: str) -> str:
    identity = json.dumps(
        [namespace, source_key],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def _disk_record_prefix(namespace: str, source_key: str) -> str:
    return (
        f"record:{_disk_namespace_token(namespace)}:"
        f"{_disk_source_token(namespace, source_key)}:"
    )


def _disk_record_key(
    namespace: str,
    source_key: str,
    revision: str,
) -> str:
    return (
        f"{_disk_record_prefix(namespace, source_key)}"
        f"{_validate_disk_revision_token(revision)}"
    )


def _disk_latest_key(namespace: str, source_key: str) -> str:
    return (
        f"latest:{_disk_namespace_token(namespace)}:"
        f"{_disk_source_token(namespace, source_key)}"
    )


def _validate_disk_revision_token(value: Any) -> str:
    if not isinstance(value, str):
        raise SnapshotUnavailable(
            "Local dashboard snapshot revision must be an opaque UUID string"
        )
    try:
        if not value.isascii() or not value.isdecimal():
            raise ValueError
        parsed = uuid.UUID(int=int(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise SnapshotUnavailable(
            "Local dashboard snapshot revision is not a valid UUID"
        ) from exc
    canonical = str(parsed.int)
    if parsed.version != 4 or value != canonical:
        raise SnapshotUnavailable(
            "Local dashboard snapshot revision is not a canonical UUID4 "
            "decimal token"
        )
    return canonical


def _new_local_revision_token() -> str:
    return str(uuid.uuid4().int)


def _encode_disk_record(
    namespace: str,
    source_key: str,
    revision: str,
    payload: Any,
    manifest: Mapping[str, Any] | None,
    *,
    publication_stage: str | None = None,
) -> bytes:
    encoded_payload = encode_snapshot_payload(payload)
    header = {
        "codec": PAYLOAD_CODEC,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "format": DISK_RECORD_FORMAT,
        "manifest": dict(manifest or {}),
        "namespace": namespace,
        "payload_bytes": len(encoded_payload),
        "payload_sha256": hashlib.sha256(encoded_payload).hexdigest(),
        "revision": _validate_disk_revision_token(revision),
        "source_key": source_key,
    }
    if publication_stage is not None:
        header["publication_stage"] = str(publication_stage)
    header_bytes = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    record_checksum = hashlib.sha256(
        header_bytes + encoded_payload
    ).digest()
    return (
        len(header_bytes).to_bytes(_DISK_HEADER_LENGTH_BYTES, "big")
        + record_checksum
        + header_bytes
        + encoded_payload
    )


def _decode_disk_record_header(
    raw_record: Any,
    *,
    verify_checksum: bool = True,
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(raw_record, bytes):
        raise TypeError("record is not raw bytes")
    minimum_record_bytes = (
        _DISK_HEADER_LENGTH_BYTES + _DISK_RECORD_CHECKSUM_BYTES
    )
    if len(raw_record) < minimum_record_bytes:
        raise ValueError("record header is truncated")
    header_size = int.from_bytes(
        raw_record[:_DISK_HEADER_LENGTH_BYTES],
        "big",
    )
    checksum_start = _DISK_HEADER_LENGTH_BYTES
    checksum_end = checksum_start + _DISK_RECORD_CHECKSUM_BYTES
    expected_checksum = raw_record[checksum_start:checksum_end]
    header_start = checksum_end
    header_end = header_start + header_size
    if header_size <= 0 or header_end > len(raw_record):
        raise ValueError("record header length is invalid")
    header_bytes = raw_record[header_start:header_end]
    encoded_payload = raw_record[header_end:]
    if verify_checksum:
        actual_checksum = hashlib.sha256(
            header_bytes + encoded_payload
        ).digest()
        if expected_checksum != actual_checksum:
            raise ValueError(
                "record header or payload checksum does not match"
            )
    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, Mapping):
        raise ValueError("record header is invalid")
    return dict(header), encoded_payload


def _decode_disk_record(
    raw_record: Any,
    *,
    namespace: str,
    source_key: str,
    revision: str,
) -> tuple[Any, dict[str, Any]]:
    revision = _validate_disk_revision_token(revision)
    identity = f"{namespace}/{source_key}/{revision}"
    try:
        header, encoded_payload = _decode_disk_record_header(raw_record)
        if header.get("format") != DISK_RECORD_FORMAT:
            raise ValueError("record format is unsupported")
        if header.get("codec") != PAYLOAD_CODEC:
            raise ValueError("record codec is unsupported")
        if header.get("namespace") != namespace:
            raise ValueError("record namespace does not match")
        if header.get("source_key") != source_key:
            raise ValueError("record source key does not match")
        if header.get("revision") != revision:
            raise ValueError("record revision does not match")
        if int(header.get("payload_bytes")) != len(encoded_payload):
            raise ValueError("record payload length does not match")
        if header.get("payload_sha256") != hashlib.sha256(
            encoded_payload
        ).hexdigest():
            raise ValueError("record payload checksum does not match")
        manifest = header.get("manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError("record manifest is invalid")
        payload = decode_snapshot_payload(encoded_payload)
    except Exception as exc:
        raise SnapshotUnavailable(
            f"Snapshot {identity} is corrupt or unreadable"
        ) from exc
    return payload, dict(manifest)


def _disk_latest_revision(
    stores: _DiskStores,
    namespace: str,
    source_key: str,
) -> str | None:
    try:
        raw_revision = stores.cache.get(
            _disk_latest_key(namespace, source_key),
            default=_MISSING,
            retry=True,
        )
    except Exception as exc:
        raise _PersistentStorageError(
            "Dashboard snapshot latest-pointer read failed"
        ) from exc
    if raw_revision is _MISSING:
        return None
    try:
        if not isinstance(raw_revision, bytes):
            raise TypeError
        revision = _validate_disk_revision_token(
            raw_revision.decode("ascii")
        )
    except (
        SnapshotUnavailable,
        TypeError,
        UnicodeDecodeError,
    ) as exc:
        _emit_snapshot_event(
            "corruption",
            namespace=namespace,
            source_key=source_key,
            backend="disk",
            target="latest_pointer",
        )
        raise SnapshotUnavailable(
            f"Snapshot {namespace}/{source_key} has a corrupt latest pointer"
        ) from exc
    return revision


def _disk_read_exact(
    stores: _DiskStores,
    namespace: str,
    source_key: str,
    revision: str,
    *,
    advance_latest: bool = False,
):
    try:
        raw_record = stores.cache.get(
            _disk_record_key(namespace, source_key, revision),
            default=_MISSING,
            retry=True,
        )
    except Exception as exc:
        raise _PersistentStorageError(
            f"Snapshot {namespace}/{source_key}/{revision} could not be read"
        ) from exc
    if raw_record is _MISSING:
        return None
    decode_started = time.perf_counter()
    try:
        payload, manifest = _decode_disk_record(
            raw_record,
            namespace=namespace,
            source_key=source_key,
            revision=revision,
        )
    except SnapshotUnavailable:
        _emit_snapshot_event(
            "corruption",
            namespace=namespace,
            source_key=source_key,
            revision=revision,
            backend="disk",
            encoded_bytes=(
                len(raw_record) if isinstance(raw_record, bytes) else None
            ),
        )
        raise
    _emit_snapshot_event(
        "decoding",
        namespace=namespace,
        source_key=source_key,
        revision=revision,
        backend="disk",
        decode_ms=round(
            (time.perf_counter() - decode_started) * 1000,
            3,
        ),
        encoded_bytes=len(raw_record),
        decoded_bytes=_estimate_decoded_bytes(payload),
    )
    _cache_local(
        namespace,
        source_key,
        revision,
        payload,
        manifest,
        shared=True,
        backend="disk",
        advance_latest=advance_latest,
    )
    return revision, payload, manifest


def _disk_read_latest(
    stores: _DiskStores,
    namespace: str,
    source_key: str,
):
    revision = _disk_latest_revision(stores, namespace, source_key)
    if revision is None:
        return None
    return _disk_read_exact(
        stores,
        namespace,
        source_key,
        revision,
        advance_latest=True,
    )


def _disk_next_revision(
    _stores: _DiskStores,
    _namespace: str,
    _source_key: str,
) -> str:
    # UUID4 identity does not depend on evictable cache state. Even if every
    # pointer and record disappears, a later build cannot alias an old
    # browser reference. Surviving records are also protected by ADD at
    # publication time.
    return _new_local_revision_token()


def _disk_publish(
    stores: _DiskStores,
    namespace: str,
    source_key: str,
    revision: str,
    payload: Any,
    manifest: Mapping[str, Any] | None,
) -> int:
    publication_stage = _ACTIVE_PUBLICATION_STAGE.get()
    raw_record = _encode_disk_record(
        namespace,
        source_key,
        revision,
        payload,
        manifest,
        publication_stage=(
            publication_stage.stage_id
            if publication_stage is not None
            else None
        ),
    )
    record_key = _disk_record_key(namespace, source_key, revision)
    try:
        with stores.cache.transact(retry=True):
            stored = stores.cache.add(
                record_key,
                raw_record,
                retry=True,
            )
            if not stored:
                persisted = stores.cache.get(
                    record_key,
                    default=_MISSING,
                    retry=True,
                )
                if persisted is not _MISSING and persisted != raw_record:
                    raise OSError(
                        "snapshot revision collision; existing immutable "
                        "record was preserved"
                    )
                raise OSError(
                    "diskcache rejected duplicate snapshot revision"
                )
            if publication_stage is None:
                published = stores.cache.set(
                    _disk_latest_key(namespace, source_key),
                    revision.encode("ascii"),
                    retry=True,
                )
                if not published:
                    raise OSError("diskcache rejected the latest pointer")
            persisted = stores.cache.get(
                record_key,
                default=_MISSING,
                retry=True,
            )
            if persisted is _MISSING or persisted != raw_record:
                raise OSError(
                    "snapshot record was evicted before publication"
                )
            if publication_stage is None:
                persisted_revision = stores.cache.get(
                    _disk_latest_key(namespace, source_key),
                    default=_MISSING,
                    retry=True,
                )
                if persisted_revision != revision.encode("ascii"):
                    raise OSError(
                        "latest pointer was evicted before publication"
                    )
    except Exception as exc:
        raise _PersistentStorageError(
            f"Snapshot {namespace}/{source_key}/{revision} could not be persisted"
        ) from exc
    if publication_stage is not None:
        publication_stage.register(namespace, source_key, revision)
    return len(raw_record)


@contextmanager
def stage_snapshot_publication(name: str):
    """Stage immutable records until a bundle commit advances all pointers."""

    if not local_snapshot_persistence_enabled():
        raise SnapshotUnavailable(
            "Atomic snapshot publication requires local persistent storage"
        )
    if _ACTIVE_PUBLICATION_STAGE.get() is not None:
        raise SnapshotUnavailable("Nested snapshot publication stages are unsupported")
    _get_persistent_stores()
    stage = SnapshotPublicationStage(name=str(name))
    token = _ACTIVE_PUBLICATION_STAGE.set(stage)
    try:
        yield stage
    finally:
        _ACTIVE_PUBLICATION_STAGE.reset(token)
        if not stage.committed and stage.records:
            _emit_snapshot_event(
                "publication_abandoned",
                stage_id=stage.stage_id,
                stage_name=stage.name,
                record_count=len(stage.records),
            )


def _advance_local_latest(
    namespace: str,
    source_key: str,
    revision: int | str,
) -> None:
    with _LOCAL_LOCK:
        if (namespace, source_key, revision) in _LOCAL_PAYLOADS:
            _LOCAL_LATEST[(namespace, source_key)] = revision


def commit_snapshot_publication_stage(
    stage: SnapshotPublicationStage,
    *,
    bundle_namespace: str,
    bundle_source_key: str,
    bundle_payload: Mapping[str, Any],
    bundle_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish staged records and one immutable bundle manifest."""

    if stage is not _ACTIVE_PUBLICATION_STAGE.get():
        raise SnapshotUnavailable(
            "The snapshot publication stage is not active in this context"
        )
    if stage.committed:
        raise SnapshotUnavailable("The snapshot publication stage is already committed")

    stores = _get_persistent_stores()
    bundle_revision = _new_local_revision_token()
    normalized_bundle = {
        "format": PUBLICATION_BUNDLE_FORMAT,
        **dict(bundle_payload),
    }
    normalized_manifest = _prepare_manifest(
        bundle_manifest,
        normalized_bundle,
    )
    raw_bundle_record = _encode_disk_record(
        bundle_namespace,
        bundle_source_key,
        bundle_revision,
        normalized_bundle,
        normalized_manifest,
        publication_stage=stage.stage_id,
    )
    bundle_record_key = _disk_record_key(
        bundle_namespace,
        bundle_source_key,
        bundle_revision,
    )
    staged_records = list(stage.records.values())
    try:
        with stores.cache.transact(retry=True):
            for record in staged_records:
                persisted = stores.cache.get(
                    _disk_record_key(
                        record.namespace,
                        record.source_key,
                        record.revision,
                    ),
                    default=_MISSING,
                    retry=True,
                )
                if persisted is _MISSING:
                    raise OSError(
                        "staged snapshot record was evicted before bundle commit"
                    )

            stored = stores.cache.add(
                bundle_record_key,
                raw_bundle_record,
                retry=True,
            )
            if not stored:
                raise OSError("diskcache rejected the bundle record")

            for record in staged_records:
                if not stores.cache.set(
                    _disk_latest_key(record.namespace, record.source_key),
                    record.revision.encode("ascii"),
                    retry=True,
                ):
                    raise OSError(
                        "diskcache rejected a staged latest pointer"
                    )
            if not stores.cache.set(
                _disk_latest_key(bundle_namespace, bundle_source_key),
                bundle_revision.encode("ascii"),
                retry=True,
            ):
                raise OSError("diskcache rejected the bundle latest pointer")

            if stores.cache.get(
                bundle_record_key,
                default=_MISSING,
                retry=True,
            ) != raw_bundle_record:
                raise OSError("bundle record was evicted before publication")
            for record in staged_records:
                if stores.cache.get(
                    _disk_latest_key(record.namespace, record.source_key),
                    default=_MISSING,
                    retry=True,
                ) != record.revision.encode("ascii"):
                    raise OSError("staged latest pointer verification failed")
    except Exception as exc:
        raise _PersistentStorageError(
            f"Snapshot bundle {bundle_namespace}/{bundle_source_key} "
            "could not be committed"
        ) from exc

    _cache_local(
        bundle_namespace,
        bundle_source_key,
        bundle_revision,
        normalized_bundle,
        normalized_manifest,
        shared=True,
        backend="disk",
    )
    for record in staged_records:
        _advance_local_latest(
            record.namespace,
            record.source_key,
            record.revision,
        )
    stage.committed = True
    _emit_snapshot_event(
        "publication_committed",
        stage_id=stage.stage_id,
        stage_name=stage.name,
        bundle_namespace=bundle_namespace,
        bundle_source_key=bundle_source_key,
        bundle_revision=bundle_revision,
        record_count=len(staged_records),
        encoded_bytes=len(raw_bundle_record),
    )
    return _snapshot_ref(
        bundle_namespace,
        bundle_source_key,
        bundle_revision,
        shared=True,
    )


@contextmanager
def _disk_reentrant_lock(
    stores: _DiskStores,
    *,
    lock_name: str,
    process_lock: threading.RLock,
    depth_key: str | int,
    namespace: str,
    source_key: str,
):
    if stores.process_id != os.getpid():
        raise _PersistentStorageError(
            "Dashboard snapshot store was inherited across a process fork"
        )
    lock_path = stores.locks_directory / lock_name
    wait_started = time.perf_counter()
    with process_lock:
        depths = getattr(stores.lock_depths, "values", None)
        if depths is None:
            depths = {}
            stores.lock_depths.values = depths
        depth = depths.get(depth_key, 0)
        if depth:
            depths[depth_key] = depth + 1
            try:
                yield
            finally:
                depths[depth_key] = depth
            return

        with stores.active_lock_fds_guard:
            if stores.closed:
                raise _PersistentStorageError(
                    "Dashboard snapshot store is already closed"
                )
            try:
                file_descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_RDWR,
                    0o600,
                )
            except OSError as exc:
                raise _PersistentStorageError(
                    f"Dashboard snapshot lock is unavailable at {lock_path}"
                ) from exc
            tracked_file_descriptor = file_descriptor
            stores.active_lock_fds.add(tracked_file_descriptor)
        lock_file = None
        try:
            try:
                os.fchmod(file_descriptor, 0o600)
            except OSError as exc:
                raise _PersistentStorageError(
                    f"Dashboard snapshot lock permissions failed at {lock_path}"
                ) from exc
            lock_file = os.fdopen(file_descriptor, "a+b", buffering=0)
            file_descriptor = -1
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                lock_wait_ms = (
                    time.perf_counter() - wait_started
                ) * 1000
                _SNAPSHOT_LOCK_WAIT_MS.set(
                    _SNAPSHOT_LOCK_WAIT_MS.get() + lock_wait_ms
                )
                _emit_snapshot_event(
                    "lock_wait",
                    namespace=namespace,
                    source_key=source_key,
                    lock_wait_ms=round(lock_wait_ms, 3),
                )
            except OSError as exc:
                raise _PersistentStorageError(
                    "Dashboard snapshot lock acquisition failed at "
                    f"{lock_path}"
                ) from exc
            depths[depth_key] = 1
            try:
                yield
            finally:
                depths.pop(depth_key, None)
                with suppress(OSError):
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            with stores.active_lock_fds_guard:
                stores.active_lock_fds.discard(tracked_file_descriptor)
                if lock_file is not None:
                    lock_file.close()
                elif file_descriptor >= 0:
                    os.close(file_descriptor)
            _forget_disk_store_if_quiescent(stores)


@contextmanager
def _disk_source_lock(
    stores: _DiskStores,
    namespace: str,
    source_key: str,
):
    stripe = (
        int(_disk_source_token(namespace, source_key)[:16], 16)
        % _LOCK_STRIPE_COUNT
    )
    depths = getattr(stores.lock_depths, "values", {})
    held_stripes = {
        key
        for key, depth in depths.items()
        if isinstance(key, int) and depth
    }
    if (
        held_stripes
        and stripe not in held_stripes
        and not depths.get("compound")
    ):
        raise _PersistentStorageError(
            "Nested dashboard snapshot stripes require snapshot_build_lock"
        )
    with _disk_reentrant_lock(
        stores,
        lock_name=f"stripe-{stripe:02d}.lock",
        process_lock=stores.source_locks[stripe],
        depth_key=stripe,
        namespace=namespace,
        source_key=source_key,
    ):
        yield


@contextmanager
def _disk_compound_lock(
    stores: _DiskStores,
    namespace: str,
    source_key: str,
):
    depths = getattr(stores.lock_depths, "values", {})
    if not depths.get("compound") and any(
        isinstance(key, int) and depth
        for key, depth in depths.items()
    ):
        raise _PersistentStorageError(
            "snapshot_build_lock must be acquired before snapshot source locks"
        )
    with _disk_reentrant_lock(
        stores,
        lock_name=_COMPOUND_LOCK_NAME,
        process_lock=stores.compound_lock,
        depth_key="compound",
        namespace=namespace,
        source_key=source_key,
    ):
        yield


@contextmanager
def snapshot_build_lock(namespace: str, source_key: str):
    """Acquire the shared same-host build lock for a compound artifact."""

    if not local_snapshot_persistence_enabled():
        yield
        return
    stores = _get_persistent_stores()
    with _disk_compound_lock(stores, namespace, source_key):
        yield


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

_SHARED_REFERENCE_EXISTS_SQL = text(f"""
    SELECT EXISTS (
        SELECT 1
        FROM {SNAPSHOT_TABLE}
        WHERE namespace = :namespace
          AND source_key = :source_key
          AND revision = :revision
    )
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
    encoded_payload = mapping["payload_bytea"]
    decode_started = time.perf_counter()
    try:
        payload = decode_snapshot_payload(encoded_payload)
    except Exception:
        _emit_snapshot_event(
            "corruption",
            namespace=namespace,
            source_key=source_key,
            revision=revision,
            backend="postgres",
            encoded_bytes=len(encoded_payload),
        )
        raise
    _emit_snapshot_event(
        "decoding",
        namespace=namespace,
        source_key=source_key,
        revision=revision,
        backend="postgres",
        decode_ms=round(
            (time.perf_counter() - decode_started) * 1000,
            3,
        ),
        encoded_bytes=len(encoded_payload),
        decoded_bytes=_estimate_decoded_bytes(payload),
    )
    manifest = dict(mapping["manifest_jsonb"] or {})
    _cache_local(
        namespace,
        source_key,
        revision,
        payload,
        manifest,
        shared=True,
        backend="postgres",
    )
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
        wait_started = time.perf_counter()
        try:
            return future.result()
        finally:
            lock_wait_ms = (
                time.perf_counter() - wait_started
            ) * 1000
            _SNAPSHOT_LOCK_WAIT_MS.set(
                _SNAPSHOT_LOCK_WAIT_MS.get() + lock_wait_ms
            )
            _emit_snapshot_event(
                "lock_wait",
                namespace=key[0],
                source_key=key[1],
                backend="memory_single_flight",
                lock_wait_ms=round(lock_wait_ms, 3),
            )
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


def _next_local_revision(namespace: str, source_key: str) -> int:
    with _LOCAL_LOCK:
        latest = _LOCAL_LATEST.get((namespace, source_key), 0)
        return (latest + 1) if isinstance(latest, int) else 1


def _get_or_build_legacy_snapshot(
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
        if local is not None and isinstance(local[0], int):
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
            build_started = time.perf_counter()
            payload = prebuilt_payload if has_prebuilt else builder()
            build_ms = 0.0 if has_prebuilt else (time.perf_counter() - build_started) * 1000
            payload_manifest = _prepare_manifest(manifest, payload)
            revision = _next_local_revision(namespace, source_key)
            _cache_local(
                namespace,
                source_key,
                revision,
                payload,
                payload_manifest,
                shared=False,
                backend="memory",
            )
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
                _emit_snapshot_event(
                    "fallback",
                    namespace=namespace,
                    source_key=source_key,
                    from_backend="postgres",
                    to_backend="memory",
                    reason="read_failed",
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
                    _cache_local(
                        namespace,
                        source_key,
                        revision,
                        payload,
                        payload_manifest,
                        shared=True,
                        backend="postgres",
                    )
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
                _emit_snapshot_event(
                    "fallback",
                    namespace=namespace,
                    source_key=source_key,
                    from_backend="postgres",
                    to_backend="memory",
                    reason="storage_failed",
                )
                return build_local()
            LOGGER.warning(
                "Shared dashboard snapshot publish failed; retaining prepared data locally",
                exc_info=True,
            )
            _emit_snapshot_event(
                "fallback",
                namespace=namespace,
                source_key=source_key,
                from_backend="postgres",
                to_backend="memory",
                reason="publish_failed",
            )
            return build_local(payload, has_prebuilt=True)

    return _run_single_flight((namespace, source_key, bool(force)), build_or_load)


def _build_memory_snapshot_result(
    *,
    namespace: str,
    source_key: str,
    builder: Callable[[], Any],
    manifest: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]] | None,
    force: bool,
) -> tuple[dict[str, Any], Any]:
    if not force:
        local = _get_local(namespace, source_key)
        if local is not None and isinstance(local[0], str):
            revision, payload, _, local_shared = local
            return (
                _snapshot_ref(
                    namespace,
                    source_key,
                    revision,
                    shared=local_shared,
                ),
                payload,
            )

    payload = builder()
    payload_manifest = _prepare_manifest(manifest, payload)
    revision = _new_local_revision_token()
    _cache_local(
        namespace,
        source_key,
        revision,
        payload,
        payload_manifest,
        shared=False,
        backend="memory",
    )
    return (
        _snapshot_ref(
            namespace,
            source_key,
            revision,
            shared=False,
        ),
        payload,
    )


def _get_or_build_memory_only_snapshot(
    *,
    namespace: str,
    source_key: str,
    builder: Callable[[], Any],
    manifest: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]] | None,
    force: bool,
) -> tuple[dict[str, Any], Any]:
    if not force:
        local = _get_local(namespace, source_key)
        if local is not None and isinstance(local[0], str):
            revision, payload, _, local_shared = local
            return (
                _snapshot_ref(
                    namespace,
                    source_key,
                    revision,
                    shared=local_shared,
                ),
                payload,
            )

    def build_local():
        result = _build_memory_snapshot_result(
            namespace=namespace,
            source_key=source_key,
            builder=builder,
            manifest=manifest,
            force=force,
        )
        reference, payload = result
        return (
            reference,
            payload,
        )

    return _run_single_flight(
        (namespace, source_key, bool(force)),
        build_local,
    )


def _get_or_build_persistent_snapshot(
    *,
    namespace: str,
    source_key: str,
    builder: Callable[[], Any],
    manifest: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]] | None,
    force: bool,
) -> tuple[dict[str, Any], Any]:
    request_started = time.perf_counter()
    _memory_max_bytes()
    publication_stage = _ACTIVE_PUBLICATION_STAGE.get()
    if publication_stage is not None:
        staged_record = publication_stage.records.get(
            (namespace, source_key)
        )
        if staged_record is not None:
            staged_local = _get_local(
                namespace,
                source_key,
                staged_record.revision,
            )
            if staged_local is not None:
                return (
                    _snapshot_ref(
                        namespace,
                        source_key,
                        staged_record.revision,
                        shared=True,
                    ),
                    staged_local[1],
                )
    if not force:
        local = _get_local(namespace, source_key)
        if local is not None:
            revision, payload, _, local_shared = local
            backend = _get_local_backend(
                namespace,
                source_key,
                revision,
            )
            if backend != "postgres" and isinstance(revision, str):
                return (
                    _snapshot_ref(
                        namespace,
                        source_key,
                        revision,
                        shared=local_shared,
                    ),
                    payload,
                )

    try:
        stores = _get_persistent_stores()
    except _PersistentStorageError:
        LOGGER.warning(
            "Local dashboard snapshot persistence is unavailable; "
            "using process memory without probing Postgres",
            exc_info=True,
        )
        _emit_snapshot_event(
            "fallback",
            namespace=namespace,
            source_key=source_key,
            from_backend="disk",
            to_backend="memory",
            reason="initialization_failed",
        )
        return _get_or_build_memory_only_snapshot(
            namespace=namespace,
            source_key=source_key,
            builder=builder,
            manifest=manifest,
            force=force,
        )

    try:
        observed_revision = _disk_latest_revision(
            stores,
            namespace,
            source_key,
        )
        if not force:
            persisted = _disk_read_latest(
                stores,
                namespace,
                source_key,
            )
            if persisted is not None:
                revision, payload, _ = persisted
                LOGGER.info(
                    "dashboard_snapshot hit namespace=%s source_key=%s "
                    "revision=%s cache=disk duration_ms=%.1f",
                    namespace,
                    source_key,
                    revision,
                    (time.perf_counter() - request_started) * 1000,
                )
                return (
                    _snapshot_ref(
                        namespace,
                        source_key,
                        revision,
                        shared=True,
                    ),
                    payload,
                )
    except _PersistentStorageError:
        LOGGER.warning(
            "Local dashboard snapshot read failed; using process memory "
            "without probing Postgres",
            exc_info=True,
        )
        _emit_snapshot_event(
            "fallback",
            namespace=namespace,
            source_key=source_key,
            from_backend="disk",
            to_backend="memory",
            reason="read_failed",
        )
        return _get_or_build_memory_only_snapshot(
            namespace=namespace,
            source_key=source_key,
            builder=builder,
            manifest=manifest,
            force=force,
        )

    def build_or_load():
        with ExitStack() as stack:
            try:
                stack.enter_context(
                    _disk_source_lock(
                        stores,
                        namespace,
                        source_key,
                    )
                )
                latest_revision = _disk_latest_revision(
                    stores,
                    namespace,
                    source_key,
                )
                if not force:
                    persisted = _disk_read_latest(
                        stores,
                        namespace,
                        source_key,
                    )
                    if persisted is not None:
                        revision, payload, _ = persisted
                        return (
                            _snapshot_ref(
                                namespace,
                                source_key,
                                revision,
                                shared=True,
                            ),
                            payload,
                        )
                elif (
                    latest_revision is not None
                    and latest_revision != observed_revision
                ):
                    persisted = _disk_read_exact(
                        stores,
                        namespace,
                        source_key,
                        latest_revision,
                        advance_latest=True,
                    )
                    if persisted is not None:
                        revision, payload, _ = persisted
                        return (
                            _snapshot_ref(
                                namespace,
                                source_key,
                                revision,
                                shared=True,
                            ),
                            payload,
                        )

                revision = _disk_next_revision(
                    stores,
                    namespace,
                    source_key,
                )
            except _PersistentStorageError:
                LOGGER.warning(
                    "Local dashboard snapshot storage failed; using process "
                    "memory without probing Postgres",
                    exc_info=True,
                )
                _emit_snapshot_event(
                    "fallback",
                    namespace=namespace,
                    source_key=source_key,
                    from_backend="disk",
                    to_backend="memory",
                    reason="lock_or_storage_failed",
                )
                return _build_memory_snapshot_result(
                    namespace=namespace,
                    source_key=source_key,
                    builder=builder,
                    manifest=manifest,
                    force=force,
                )

            # Builder and manifest failures are data-path failures. Keep them
            # outside storage-error handling so they propagate exactly once.
            build_started = time.perf_counter()
            payload = builder()
            payload_manifest = _prepare_manifest(manifest, payload)
            try:
                persisted_bytes = _disk_publish(
                    stores,
                    namespace,
                    source_key,
                    revision,
                    payload,
                    payload_manifest,
                )
            except _PersistentStorageError:
                LOGGER.warning(
                    "Local dashboard snapshot publish failed; "
                    "retaining prepared data in process memory",
                    exc_info=True,
                )
                _emit_snapshot_event(
                    "fallback",
                    namespace=namespace,
                    source_key=source_key,
                    from_backend="disk",
                    to_backend="memory",
                    reason="publish_failed",
                )
                local_revision = _new_local_revision_token()
                _cache_local(
                    namespace,
                    source_key,
                    local_revision,
                    payload,
                    payload_manifest,
                    shared=False,
                    backend="memory",
                )
                return (
                    _snapshot_ref(
                        namespace,
                        source_key,
                        local_revision,
                        shared=False,
                    ),
                    payload,
                )

            _cache_local(
                namespace,
                source_key,
                revision,
                payload,
                payload_manifest,
                shared=True,
                backend="disk",
                advance_latest=publication_stage is None,
            )
            LOGGER.info(
                "dashboard_snapshot build namespace=%s source_key=%s "
                "revision=%s cache=disk codec=%s payload_bytes=%s "
                "build_ms=%.1f duration_ms=%.1f",
                namespace,
                source_key,
                revision,
                PAYLOAD_CODEC,
                persisted_bytes,
                (time.perf_counter() - build_started) * 1000,
                (time.perf_counter() - request_started) * 1000,
            )
            return (
                _snapshot_ref(
                    namespace,
                    source_key,
                    revision,
                    shared=True,
                ),
                payload,
            )

    return _run_single_flight(
        (namespace, source_key, bool(force)),
        build_or_load,
    )


def _get_or_build_snapshot_tuple(
    engine,
    *,
    namespace: str,
    source_key: str,
    builder: Callable[[], Any],
    manifest: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]] | None = None,
    force: bool = False,
) -> tuple[dict[str, Any], Any]:
    """Resolve or build an immutable snapshot and return ``(reference, payload)``."""
    if local_snapshot_persistence_enabled():
        return _get_or_build_persistent_snapshot(
            namespace=namespace,
            source_key=source_key,
            builder=builder,
            manifest=manifest,
            force=force,
        )
    return _get_or_build_legacy_snapshot(
        engine,
        namespace=namespace,
        source_key=source_key,
        builder=builder,
        manifest=manifest,
        force=force,
    )


def _get_snapshot_if_available_tuple(
    engine,
    *,
    namespace: str,
    source_key: str,
) -> tuple[dict[str, Any], Any] | None:
    """Return an existing immutable snapshot without invoking a builder."""
    persistent = local_snapshot_persistence_enabled()
    local = _get_local(namespace, source_key)
    if local is not None:
        revision, payload, _, local_shared = local
        if (persistent and isinstance(revision, str)) or (
            not persistent and isinstance(revision, int)
        ):
            _SNAPSHOT_READ_BACKEND.set("memory")
            return (
                _snapshot_ref(
                    namespace,
                    source_key,
                    revision,
                    shared=local_shared,
                ),
                payload,
            )

    if persistent:
        try:
            stores = _get_persistent_stores()
            persisted = _disk_read_latest(stores, namespace, source_key)
        except _PersistentStorageError:
            return None
        if persisted is None:
            return None
        revision, payload, _ = persisted
        _SNAPSHOT_READ_BACKEND.set("disk")
        return (
            _snapshot_ref(
                namespace,
                source_key,
                revision,
                shared=True,
            ),
            payload,
        )

    if not shared_snapshot_schema_available(engine):
        return None
    try:
        shared = _read_shared(engine, namespace, source_key)
    except Exception:
        LOGGER.warning(
            "Shared dashboard snapshot read failed while resolving last-good data",
            exc_info=True,
        )
        return None
    if shared is None:
        return None
    revision, payload, _ = shared
    _SNAPSHOT_READ_BACKEND.set("postgres")
    return (
        _snapshot_ref(
            namespace,
            source_key,
            revision,
            shared=True,
        ),
        payload,
    )


def _snapshot_result_backend(reference: Mapping[str, Any]) -> str:
    namespace = str(reference.get("namespace") or "")
    source_key = str(reference.get("source_key") or "")
    revision = reference.get("revision")
    backend = _get_local_backend(namespace, source_key, revision)
    if backend:
        return backend
    if not reference.get("shared"):
        return "memory"
    if local_snapshot_persistence_enabled() and isinstance(revision, str):
        return "disk"
    return "postgres"


def _snapshot_encoded_bytes(
    reference: Mapping[str, Any],
) -> int | None:
    if not local_snapshot_persistence_enabled():
        return None
    revision = reference.get("revision")
    if not isinstance(revision, str):
        return None
    try:
        stores = _get_persistent_stores()
        raw_record = stores.cache.get(
            _disk_record_key(
                str(reference["namespace"]),
                str(reference["source_key"]),
                revision,
            ),
            default=_MISSING,
            retry=True,
        )
        if raw_record is _MISSING:
            return None
        header, _encoded_payload = _decode_disk_record_header(
            raw_record,
            verify_checksum=False,
        )
        return int(header.get("payload_bytes"))
    except Exception:
        return None


def get_snapshot_if_available_result(
    engine,
    *,
    namespace: str,
    source_key: str,
) -> SnapshotResult | None:
    """Return a detailed immutable hit without invoking a builder."""

    started = time.perf_counter()
    backend_token = _SNAPSHOT_READ_BACKEND.set(None)
    try:
        resolved = _get_snapshot_if_available_tuple(
            engine,
            namespace=namespace,
            source_key=source_key,
        )
        backend = _SNAPSHOT_READ_BACKEND.get()
    finally:
        _SNAPSHOT_READ_BACKEND.reset(backend_token)
    read_ms = (time.perf_counter() - started) * 1000
    if resolved is None:
        _emit_snapshot_event(
            "miss",
            namespace=namespace,
            source_key=source_key,
            read_ms=round(read_ms, 3),
        )
        return None
    reference, payload = resolved
    backend = backend or _snapshot_result_backend(reference)
    result = SnapshotResult(
        reference=reference,
        payload=payload,
        status="hit",
        backend=backend,
        read_ms=read_ms,
        build_ms=0.0,
        lock_wait_ms=0.0,
        encoded_bytes=_snapshot_encoded_bytes(reference),
        decoded_bytes=_estimate_decoded_bytes(payload),
    )
    _emit_snapshot_event(
        f"{backend}_hit",
        namespace=namespace,
        source_key=source_key,
        revision=reference.get("revision"),
        backend=backend,
        read_ms=round(read_ms, 3),
        encoded_bytes=result.encoded_bytes,
        decoded_bytes=result.decoded_bytes,
    )
    return result


def get_snapshot_if_available(
    engine,
    *,
    namespace: str,
    source_key: str,
) -> tuple[dict[str, Any], Any] | None:
    """Return an existing immutable snapshot without invoking a builder."""

    result = get_snapshot_if_available_result(
        engine,
        namespace=namespace,
        source_key=source_key,
    )
    if result is None:
        return None
    return result.reference, result.payload


def get_or_build_snapshot_result(
    engine,
    *,
    namespace: str,
    source_key: str,
    builder: Callable[[], Any],
    manifest: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]] | None = None,
    force: bool = False,
) -> SnapshotResult:
    """Resolve or build a snapshot and return detailed cache telemetry."""

    request_started = time.perf_counter()
    lock_wait_token = _SNAPSHOT_LOCK_WAIT_MS.set(0.0)
    try:
        if not force:
            available = get_snapshot_if_available_result(
                engine,
                namespace=namespace,
                source_key=source_key,
            )
            if available is not None:
                return available

        builder_called = False
        build_ms = 0.0

        def measured_builder():
            nonlocal builder_called, build_ms
            builder_called = True
            build_started = time.perf_counter()
            try:
                return builder()
            finally:
                build_ms = (time.perf_counter() - build_started) * 1000

        reference, payload = _get_or_build_snapshot_tuple(
            engine,
            namespace=namespace,
            source_key=source_key,
            builder=measured_builder,
            manifest=manifest,
            force=force,
        )
        duration_ms = (time.perf_counter() - request_started) * 1000
        status = "built" if builder_called else "hit_after_wait"
        backend = _snapshot_result_backend(reference)
        lock_wait_ms = _SNAPSHOT_LOCK_WAIT_MS.get()
        read_ms = max(0.0, duration_ms - build_ms - lock_wait_ms)
        result = SnapshotResult(
            reference=reference,
            payload=payload,
            status=status,
            backend=backend,
            read_ms=read_ms,
            build_ms=build_ms,
            lock_wait_ms=lock_wait_ms,
            encoded_bytes=_snapshot_encoded_bytes(reference),
            decoded_bytes=_estimate_decoded_bytes(payload),
        )
        _emit_snapshot_event(
            status,
            namespace=namespace,
            source_key=source_key,
            revision=reference.get("revision"),
            backend=backend,
            read_ms=round(result.read_ms, 3),
            build_ms=round(result.build_ms, 3),
            lock_wait_ms=round(result.lock_wait_ms, 3),
            encoded_bytes=result.encoded_bytes,
            decoded_bytes=result.decoded_bytes,
        )
        return result
    finally:
        _SNAPSHOT_LOCK_WAIT_MS.reset(lock_wait_token)


def get_or_build_snapshot(
    engine,
    *,
    namespace: str,
    source_key: str,
    builder: Callable[[], Any],
    manifest: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]] | None = None,
    force: bool = False,
) -> tuple[dict[str, Any], Any]:
    """Resolve or build an immutable snapshot and return ``(reference, payload)``."""

    result = get_or_build_snapshot_result(
        engine,
        namespace=namespace,
        source_key=source_key,
        builder=builder,
        manifest=manifest,
        force=force,
    )
    return result.reference, result.payload


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
    persistent = local_snapshot_persistence_enabled()
    if persistent:
        revision = _validate_disk_revision_token(value.get("revision"))
    else:
        try:
            revision = int(value["revision"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotUnavailable(
                "Legacy dashboard snapshot revision must be an integer"
            ) from exc
    local = _get_local(namespace, source_key, revision)
    if local is None:
        if persistent:
            try:
                stores = _get_persistent_stores()
                local = _disk_read_exact(
                    stores,
                    namespace,
                    source_key,
                    revision,
                )
            except _PersistentStorageError as exc:
                raise SnapshotUnavailable(
                    f"Snapshot {namespace}/{source_key}/{revision} "
                    "cannot be resolved because local persistence is unavailable"
                ) from exc
        else:
            if not shared_snapshot_schema_available(engine):
                raise SnapshotUnavailable(
                    f"Snapshot {namespace}/{source_key}/{revision} "
                    "is not available in this process"
                )
            local = _read_shared(
                engine,
                namespace,
                source_key,
                revision,
            )
    if local is None:
        raise SnapshotUnavailable(
            f"Snapshot {namespace}/{source_key}/{revision} "
            "is missing or has been evicted"
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


def resolve_snapshot_manifest(
    value: Any,
    engine,
    *,
    expected_namespace: str | None = None,
) -> dict[str, Any]:
    """Resolve the immutable manifest associated with one exact reference."""

    if not is_snapshot_reference(value, expected_namespace):
        raise SnapshotUnavailable(
            "An exact dashboard snapshot reference is required"
        )
    namespace = str(value["namespace"])
    source_key = str(value["source_key"])
    persistent = local_snapshot_persistence_enabled()
    if persistent:
        revision = _validate_disk_revision_token(value.get("revision"))
    else:
        try:
            revision = int(value["revision"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotUnavailable(
                "Legacy dashboard snapshot revision must be an integer"
            ) from exc

    local = _get_local(namespace, source_key, revision)
    if local is None:
        if persistent:
            try:
                stores = _get_persistent_stores()
                local = _disk_read_exact(
                    stores,
                    namespace,
                    source_key,
                    revision,
                )
            except _PersistentStorageError as exc:
                raise SnapshotUnavailable(
                    f"Snapshot {namespace}/{source_key}/{revision} "
                    "cannot be resolved because local persistence is unavailable"
                ) from exc
        else:
            if not shared_snapshot_schema_available(engine):
                raise SnapshotUnavailable(
                    f"Snapshot {namespace}/{source_key}/{revision} "
                    "is not available in this process"
                )
            local = _read_shared(
                engine,
                namespace,
                source_key,
                revision,
            )
    if local is None:
        raise SnapshotUnavailable(
            f"Snapshot {namespace}/{source_key}/{revision} "
            "is missing or has been evicted"
        )
    manifest = local[2]
    if not isinstance(manifest, Mapping):
        raise SnapshotUnavailable(
            f"Snapshot {namespace}/{source_key}/{revision} "
            "has an invalid manifest"
        )
    return dict(manifest)


def _timestamp_to_utc_iso(value: Any) -> str | None:
    try:
        return dt.datetime.fromtimestamp(
            float(value),
            tz=dt.timezone.utc,
        ).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def inspect_persistent_snapshot_cache() -> dict[str, Any]:
    """Return read-only operator diagnostics for the shared DiskCache."""

    stores = _get_persistent_stores()
    rows = list(
        stores.cache._sql(
            "SELECT key, store_time, access_time, mode, filename, value "
            "FROM Cache"
        )
    )
    raw_values: dict[str, Any] = {}
    row_metadata: dict[str, tuple[float, float]] = {}
    invalid_values: list[dict[str, Any]] = []
    for key, store_time, access_time, mode, filename, value in rows:
        if not isinstance(key, str):
            continue
        try:
            raw_values[key] = stores.cache.disk.fetch(
                mode,
                filename,
                value,
                read=False,
            )
            row_metadata[key] = (store_time, access_time)
        except Exception as exc:
            invalid_values.append({
                "record_key": key,
                "error": type(exc).__name__,
            })
    records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = list(invalid_values)
    latest_pointer_count = 0
    inspected_at = dt.datetime.now(dt.timezone.utc)
    for key, raw_record in raw_values.items():
        if key.startswith("latest:"):
            latest_pointer_count += 1
            continue
        if not key.startswith("record:"):
            continue
        try:
            store_time, access_time = row_metadata[key]
            header, _encoded_payload = _decode_disk_record_header(
                raw_record
            )
            namespace = str(header["namespace"])
            source_key = str(header["source_key"])
            revision = _validate_disk_revision_token(header["revision"])
            raw_latest_revision = raw_values.get(
                _disk_latest_key(namespace, source_key)
            )
            latest_revision = None
            if isinstance(raw_latest_revision, bytes):
                latest_revision = _validate_disk_revision_token(
                    raw_latest_revision.decode("ascii")
                )
            is_latest = latest_revision == revision
            publication_stage = header.get("publication_stage")
            created_at_utc = (
                header.get("created_at_utc")
                or _timestamp_to_utc_iso(store_time)
            )
            last_access_at_utc = _timestamp_to_utc_iso(access_time)
            created_at = _parse_utc_datetime(created_at_utc)
            last_access_at = _parse_utc_datetime(last_access_at_utc)
            records.append({
                "namespace": namespace,
                "source_key": source_key,
                "revision": revision,
                "record_bytes": len(raw_record),
                "payload_bytes": int(header.get("payload_bytes") or 0),
                "created_at_utc": created_at_utc,
                "age_seconds": (
                    max(0.0, (inspected_at - created_at).total_seconds())
                    if created_at is not None
                    else None
                ),
                "last_access_at_utc": last_access_at_utc,
                "last_access_age_seconds": (
                    max(
                        0.0,
                        (inspected_at - last_access_at).total_seconds(),
                    )
                    if last_access_at is not None
                    else None
                ),
                "is_latest": is_latest,
                "publication_stage": publication_stage,
                "orphaned_staging": bool(
                    publication_stage and not is_latest
                ),
                "_record_key": key,
            })
        except Exception as exc:
            invalid_records.append({
                "record_key": key,
                "error": type(exc).__name__,
            })

    records.sort(key=lambda item: (
        item["namespace"],
        item["source_key"],
        item["created_at_utc"] or "",
        item["revision"],
    ))
    namespace_summaries: dict[str, dict[str, Any]] = {}
    for record in records:
        summary = namespace_summaries.setdefault(
            record["namespace"],
            {
                "namespace": record["namespace"],
                "record_count": 0,
                "latest_count": 0,
                "record_bytes": 0,
                "payload_bytes": 0,
                "orphaned_staging_count": 0,
            },
        )
        summary["record_count"] += 1
        summary["latest_count"] += int(record["is_latest"])
        summary["record_bytes"] += int(record["record_bytes"])
        summary["payload_bytes"] += int(record["payload_bytes"])
        summary["orphaned_staging_count"] += int(
            record["orphaned_staging"]
        )

    return {
        "format": "dashboard_snapshot_cache_inspection_v1",
        "cache_root": str(stores.root),
        "volume_bytes": int(stores.cache.volume()),
        "size_limit_bytes": int(stores.cache.size_limit),
        "latest_pointer_count": latest_pointer_count,
        "latest_pointers": [
            {
                "namespace": record["namespace"],
                "source_key": record["source_key"],
                "revision": record["revision"],
            }
            for record in records
            if record["is_latest"]
        ],
        "record_count": len(records),
        "invalid_record_count": len(invalid_records),
        "orphaned_staging_count": sum(
            int(record["orphaned_staging"])
            for record in records
        ),
        "namespaces": sorted(
            namespace_summaries.values(),
            key=lambda item: (-item["record_bytes"], item["namespace"]),
        ),
        "records": records,
        "invalid_records": invalid_records,
    }


def _parse_utc_datetime(value: Any) -> dt.datetime | None:
    try:
        parsed = dt.datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def prune_persistent_snapshot_cache(
    *,
    apply: bool = False,
    staged_older_than_days: int = 7,
    retired_namespaces: Sequence[str] = (),
    retired_older_than_days: int = 30,
    coordinated_restart_confirmed: bool = False,
) -> dict[str, Any]:
    """Safely prune abandoned staging or retired, non-latest records."""

    if staged_older_than_days < 1 or retired_older_than_days < 1:
        raise ValueError("Cache retention windows must be at least one day")
    inspection = inspect_persistent_snapshot_cache()
    now = dt.datetime.now(dt.timezone.utc)
    staged_cutoff = now - dt.timedelta(days=staged_older_than_days)
    retired_cutoff = now - dt.timedelta(days=retired_older_than_days)
    retired = {str(namespace) for namespace in retired_namespaces}
    if apply and retired and not coordinated_restart_confirmed:
        raise ValueError(
            "Retired-namespace pruning requires a coordinated worker "
            "restart confirmation"
        )
    candidates: list[dict[str, Any]] = []
    for record in inspection["records"]:
        if record["is_latest"]:
            continue
        created_at = _parse_utc_datetime(record.get("created_at_utc"))
        accessed_at = _parse_utc_datetime(record.get("last_access_at_utc"))
        reason = None
        if (
            record.get("orphaned_staging")
            and created_at is not None
            and created_at < staged_cutoff
        ):
            reason = "abandoned_staging"
        elif (
            record["namespace"] in retired
            and accessed_at is not None
            and accessed_at < retired_cutoff
        ):
            reason = "retired_namespace"
        if reason is not None:
            candidates.append({
                **record,
                "reason": reason,
            })

    deleted: list[dict[str, Any]] = []
    if apply and candidates:
        stores = _get_persistent_stores()
        for candidate in candidates:
            latest_revision = _disk_latest_revision(
                stores,
                candidate["namespace"],
                candidate["source_key"],
            )
            if latest_revision == candidate["revision"]:
                continue
            if stores.cache.delete(
                candidate["_record_key"],
                retry=True,
            ):
                deleted.append(candidate)
                _emit_snapshot_event(
                    "pruned",
                    namespace=candidate["namespace"],
                    source_key=candidate["source_key"],
                    revision=candidate["revision"],
                    reason=candidate["reason"],
                    encoded_bytes=candidate["record_bytes"],
                )

    public_candidates = [
        {
            key: value
            for key, value in candidate.items()
            if not key.startswith("_")
        }
        for candidate in candidates
    ]
    public_deleted = [
        {
            key: value
            for key, value in candidate.items()
            if not key.startswith("_")
        }
        for candidate in deleted
    ]
    return {
        "format": "dashboard_snapshot_cache_prune_v1",
        "apply": bool(apply),
        "staged_older_than_days": staged_older_than_days,
        "retired_older_than_days": retired_older_than_days,
        "retired_namespaces": sorted(retired),
        "coordinated_restart_confirmed": bool(
            coordinated_restart_confirmed
        ),
        "candidate_count": len(public_candidates),
        "deleted_count": len(public_deleted),
        "candidates": public_candidates,
        "deleted": public_deleted,
    }


def _clear_persistent_snapshots(namespace: str | None) -> None:
    stores = _get_persistent_stores()
    if namespace is None:
        stores.cache.clear(retry=True)
        return

    namespace_token = _disk_namespace_token(namespace)
    record_prefix = f"record:{namespace_token}:"
    latest_prefix = f"latest:{namespace_token}:"
    keys = [
        key
        for key in stores.cache.iterkeys()
        if isinstance(key, str)
        and (
            key.startswith(record_prefix)
            or key.startswith(latest_prefix)
        )
    ]
    for key in keys:
        stores.cache.delete(key, retry=True)


def clear_local_snapshots(
    namespace: str | None = None,
    *,
    persistent: bool = False,
) -> None:
    """Clear decoded memory; persistent entries survive unless requested."""
    global _LOCAL_TOTAL_BYTES

    with _LOCAL_LOCK:
        if namespace is None:
            _LOCAL_PAYLOADS.clear()
            _LOCAL_MANIFESTS.clear()
            _LOCAL_SHARED.clear()
            _LOCAL_BACKENDS.clear()
            _LOCAL_SIZES.clear()
            _LOCAL_TOTAL_BYTES = 0
            _LOCAL_LATEST.clear()
        else:
            for key in [
                key for key in _LOCAL_PAYLOADS if key[0] == namespace
            ]:
                _LOCAL_PAYLOADS.pop(key, None)
                _LOCAL_MANIFESTS.pop(key, None)
                _LOCAL_SHARED.pop(key, None)
                _LOCAL_BACKENDS.pop(key, None)
                _LOCAL_TOTAL_BYTES -= _LOCAL_SIZES.pop(key, 0)
            for key in [
                key for key in _LOCAL_LATEST if key[0] == namespace
            ]:
                _LOCAL_LATEST.pop(key, None)
    if persistent:
        _clear_persistent_snapshots(namespace)


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
