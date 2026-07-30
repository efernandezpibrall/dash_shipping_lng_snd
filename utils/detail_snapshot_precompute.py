"""Precompute exporter/importer detail snapshots outside Dash requests.

The detail pages remain the canonical owners of snapshot keys and builders.
This module deliberately calls their existing source-loader functions instead
of duplicating key construction.  That keeps normal page loads and precompute
runs on exactly the same cache contract.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import event

from utils import dashboard_snapshot_cache as snapshots


class DetailSnapshotPrecomputeError(RuntimeError):
    """Raised when a detail snapshot cannot be prepared safely."""


_READ_ONLY_PREFIX = re.compile(r"^\s*(select|with|show|explain)\b", re.I)
_SQL_COMMENT = re.compile(r"/\*.*?\*/|--[^\r\n]*", re.S)
_SQL_WRITE_TOKEN = re.compile(
    r"\b("
    r"insert|update|delete|merge|copy|create|alter|drop|truncate|"
    r"grant|revoke|comment|vacuum|refresh|call|do"
    r")\b",
    re.I,
)


def sql_statement_is_read_only(statement: str) -> bool:
    """Conservatively accept only SELECT-like SQL without write/DDL tokens."""

    normalized = _SQL_COMMENT.sub(" ", str(statement or "")).strip()
    return bool(
        _READ_ONLY_PREFIX.match(normalized)
        and not _SQL_WRITE_TOKEN.search(normalized)
    )


@dataclass
class ReadOnlySqlAudit:
    statement_count: int = 0
    rejected_statement_count: int = 0

    def inspect(
        self,
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        self.statement_count += 1
        if sql_statement_is_read_only(statement):
            return
        self.rejected_statement_count += 1
        first_token = str(statement or "").strip().split(None, 1)
        statement_type = first_token[0].upper() if first_token else "UNKNOWN"
        raise DetailSnapshotPrecomputeError(
            "Precompute SQL guard rejected a non-read-only statement "
            f"of type {statement_type}"
        )


@contextmanager
def audit_read_only_sql(engines: Iterable[Any]):
    """Reject DML/DDL issued through the supplied SQLAlchemy engines."""

    audit = ReadOnlySqlAudit()
    unique_engines = []
    seen_engine_ids = set()
    for engine in engines:
        if engine is None or id(engine) in seen_engine_ids:
            continue
        seen_engine_ids.add(id(engine))
        unique_engines.append(engine)
        event.listen(engine, "before_cursor_execute", audit.inspect)
    try:
        yield audit
    finally:
        for engine in unique_engines:
            event.remove(engine, "before_cursor_execute", audit.inspect)


@dataclass(frozen=True)
class PreparedSnapshot:
    page: str
    selection: str
    section: str
    namespace: str
    source_key: str
    revision: int | str
    duration_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_token(namespace: str, value: Any) -> str:
    return snapshots.build_source_key(namespace, value)


def _require_persistent_disk_cache() -> None:
    if not snapshots.local_snapshot_persistence_enabled():
        raise DetailSnapshotPrecomputeError(
            "Precompute requires "
            f"{snapshots.LOCAL_PERSISTENCE_ENV}=1. The disabled mode may "
            "write snapshots to the legacy Postgres backend."
        )
    # Opening the store before any source query makes persistence failures
    # fail closed instead of falling back to process memory after expensive
    # data preparation.
    snapshots._get_persistent_stores()


def _require_normal_source_context(
    source_context: Mapping[str, Any],
    *,
    page: str,
) -> dict[str, Any]:
    if not isinstance(source_context, Mapping):
        raise DetailSnapshotPrecomputeError(
            f"{page} source context is unavailable"
        )
    normalized = dict(source_context)
    if normalized.get("refresh_generation") is not None:
        raise DetailSnapshotPrecomputeError(
            f"{page} precompute only supports normal navigation keys; "
            "refresh_generation must be None"
        )
    if normalized.get("source_watermark") is None:
        raise DetailSnapshotPrecomputeError(
            f"{page} source watermark is unavailable"
        )
    return normalized


def build_exporter_source_context(exporter_page) -> dict[str, Any]:
    return _require_normal_source_context(
        exporter_page._build_exporter_detail_source_context(
            exporter_page._fetch_exporter_detail_source_watermark(),
            force_refresh=False,
            maintenance_source_version=(
                exporter_page._fetch_exporter_maintenance_source_version()
            ),
        ),
        page="exporter",
    )


def build_importer_source_context(importer_page) -> dict[str, Any]:
    return _require_normal_source_context(
        importer_page._build_importer_detail_source_context(
            importer_page._fetch_importer_detail_source_watermark(),
            force_refresh=False,
            maintenance_source_version=(
                importer_page._fetch_importer_maintenance_source_version()
            ),
        ),
        page="importer",
    )


def _normalized_run_metadata(metadata: Any) -> Any:
    if not isinstance(metadata, Mapping):
        return metadata
    normalized = dict(metadata)
    if normalized.get("run_id") is not None:
        normalized["run_id"] = str(normalized["run_id"])
    return normalized


def _exporter_source_sentinel(exporter_page) -> dict[str, Any]:
    return {
        "source_context": build_exporter_source_context(exporter_page),
        "allocation_run": _normalized_run_metadata(
            exporter_page.fetch_latest_supply_allocation_run_metadata(
                exporter_page.engine
            )
        ),
        "diversion_version": (
            exporter_page._fetch_exporter_diversion_source_version()
        ),
        "forecast_month": (
            exporter_page._exporter_detail_forecast_month_token()
        ),
    }


def _importer_source_sentinel(importer_page) -> dict[str, Any]:
    return {
        "source_context": build_importer_source_context(importer_page),
        "allocation_run": _normalized_run_metadata(
            importer_page.fetch_latest_supply_allocation_run_metadata(
                importer_page.engine
            )
        ),
        "diversion_version": (
            importer_page._fetch_importer_diversion_source_version()
        ),
        "forecast_month": (
            importer_page._importer_detail_forecast_month_token()
        ),
    }


def _require_reference(
    reference: Any,
    *,
    page: str,
    selection: str,
    section: str,
    expected_namespace: str,
    started_at: float,
) -> tuple[dict[str, Any], PreparedSnapshot]:
    if not snapshots.is_snapshot_reference(reference, expected_namespace):
        error = (
            reference.get("error")
            if isinstance(reference, Mapping)
            else None
        )
        raise DetailSnapshotPrecomputeError(
            f"{page} {selection!r} {section} did not produce a "
            f"{expected_namespace!r} reference"
            + (f": {error}" if error else "")
        )
    if not snapshots.snapshot_is_resolvable(reference):
        raise DetailSnapshotPrecomputeError(
            f"{page} {selection!r} {section} produced a non-persistent "
            "reference"
        )
    normalized = dict(reference)
    result = PreparedSnapshot(
        page=page,
        selection=selection,
        section=section,
        namespace=str(normalized["namespace"]),
        source_key=str(normalized["source_key"]),
        revision=normalized["revision"],
        duration_seconds=round(time.perf_counter() - started_at, 6),
    )
    return normalized, result


def _run_reference_loader(
    loader,
    *,
    page: str,
    selection: str,
    section: str,
    expected_namespace: str,
) -> tuple[dict[str, Any], PreparedSnapshot]:
    started_at = time.perf_counter()
    reference = loader()
    return _require_reference(
        reference,
        page=page,
        selection=selection,
        section=section,
        expected_namespace=expected_namespace,
        started_at=started_at,
    )


def precompute_exporter_selection(
    exporter_page,
    country: str,
    source_context: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[PreparedSnapshot]]:
    """Build the five normal-navigation snapshots for one exporter."""

    context = _require_normal_source_context(
        source_context,
        page="exporter",
    )
    selection = str(country or "").strip()
    if not selection:
        raise DetailSnapshotPrecomputeError("Exporter country is required")

    references: list[dict[str, Any]] = []
    results: list[PreparedSnapshot] = []

    def add(loader, section, namespace):
        reference, result = _run_reference_loader(
            loader,
            page="exporter",
            selection=selection,
            section=section,
            expected_namespace=namespace,
        )
        references.append(reference)
        results.append(result)

    add(
        lambda: exporter_page.refresh_exporter_detail_base_data(
            selection,
            source_context=context,
        ),
        "base",
        exporter_page.EXPORTER_DETAIL_BASE_NAMESPACE,
    )
    add(
        lambda: exporter_page.refresh_destination_forecast_source(
            selection,
            context,
        ),
        "allocation",
        exporter_page.EXPORTER_ALLOCATION_SOURCE_NAMESPACE,
    )
    add(
        lambda: exporter_page._get_exporter_maintenance_source_reference(
            selection,
            context,
        )[0],
        "maintenance",
        exporter_page.EXPORTER_MAINTENANCE_SOURCE_NAMESPACE,
    )
    # Normal Dash navigation passes source context, not the base reference.
    # Keeping base_reference=None is necessary for exact route-key parity.
    add(
        lambda: exporter_page.refresh_exporter_route_analysis_source(
            selection,
            source_context=context,
        ),
        "route",
        exporter_page.EXPORTER_ROUTE_SOURCE_NAMESPACE,
    )
    add(
        lambda: exporter_page.refresh_exporter_diversion_source(
            context,
            selection,
        ),
        "diversion",
        exporter_page.EXPORTER_DIVERSION_SOURCE_NAMESPACE,
    )
    return references, results


def precompute_importer_selection(
    importer_page,
    aggregation: str,
    selected_value: str,
    destination_catalog: Sequence[Mapping[str, Any]],
    source_context: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[PreparedSnapshot]]:
    """Build the five normal-navigation snapshots for one importer scope."""

    context = _require_normal_source_context(
        source_context,
        page="importer",
    )
    requested_aggregation = str(aggregation or "").strip()
    aggregation_by_case = {
        str(value).casefold(): str(value)
        for value in importer_page.DESTINATION_AGGREGATION_LABELS
    }
    aggregation = aggregation_by_case.get(
        requested_aggregation.casefold()
    )
    selected_value = str(selected_value or "").strip()
    if aggregation is None:
        valid_aggregations = ", ".join(
            importer_page.DESTINATION_AGGREGATION_LABELS
        )
        raise DetailSnapshotPrecomputeError(
            f"Unknown importer aggregation {requested_aggregation!r}; "
            f"expected one of: {valid_aggregations}"
        )
    if not selected_value:
        raise DetailSnapshotPrecomputeError(
            "Importer selected value is required"
        )
    destination_context = importer_page.resolve_destination_context(
        aggregation,
        selected_value,
        destination_catalog,
    )
    destination_countries = destination_context.get(
        "destination_countries"
    )
    if not destination_countries:
        raise DetailSnapshotPrecomputeError(
            f"Importer selection {aggregation}={selected_value!r} "
            "does not resolve to destination countries"
        )
    selection = f"{aggregation}={selected_value}"

    references: list[dict[str, Any]] = []
    results: list[PreparedSnapshot] = []

    def add(loader, section, namespace):
        reference, result = _run_reference_loader(
            loader,
            page="importer",
            selection=selection,
            section=section,
            expected_namespace=namespace,
        )
        references.append(reference)
        results.append(result)

    add(
        lambda: importer_page.refresh_import_analysis_base_data(
            0,
            aggregation,
            selected_value,
            destination_catalog,
            source_context=context,
        ),
        "base",
        importer_page.IMPORTER_DETAIL_BASE_NAMESPACE,
    )
    add(
        lambda: importer_page.refresh_origin_forecast_source(
            context,
            aggregation,
            selected_value,
            destination_catalog,
        ),
        "allocation",
        importer_page.IMPORTER_ALLOCATION_SOURCE_NAMESPACE,
    )
    add(
        lambda: importer_page._get_importer_maintenance_source_reference(
            destination_countries,
            context,
        )[0],
        "maintenance",
        importer_page.IMPORTER_MAINTENANCE_SOURCE_NAMESPACE,
    )
    add(
        lambda: importer_page.refresh_importer_route_analysis_source(
            aggregation,
            selected_value,
            destination_catalog,
            context,
        ),
        "route",
        importer_page.IMPORTER_ROUTE_SOURCE_NAMESPACE,
    )
    add(
        lambda: importer_page.refresh_importer_diversion_source(
            context,
            aggregation,
            selected_value,
            destination_catalog,
        ),
        "diversion",
        importer_page.IMPORTER_DIVERSION_SOURCE_NAMESPACE,
    )
    return references, results


def _validate_reopened_persistence(
    references: Sequence[Mapping[str, Any]],
    engines: Sequence[Any],
) -> None:
    # This intentionally clears only decoded process memory. Persistent
    # records are never removed by the precompute runner.
    snapshots.clear_local_snapshots()
    snapshots.close_persistent_snapshot_cache()
    stores = snapshots._get_persistent_stores()
    engine_by_page = {
        "exporter": engines[0],
        "importer": engines[1],
    }
    for reference in references:
        page = str(reference.get("_precompute_page"))
        section = str(reference.get("_precompute_section"))
        selection = str(reference.get("_precompute_selection"))
        engine = engine_by_page[page]
        clean_reference = {
            key: value
            for key, value in reference.items()
            if not key.startswith("_precompute_")
        }
        persisted = snapshots._disk_read_latest(
            stores,
            str(clean_reference["namespace"]),
            str(clean_reference["source_key"]),
        )
        if (
            persisted is None
            or persisted[0] != clean_reference["revision"]
        ):
            raise DetailSnapshotPrecomputeError(
                f"{page} {selection!r} {section} latest pointer does not "
                "resolve to the prepared revision"
            )
        payload = snapshots.resolve_snapshot(clean_reference, engine)
        manifest = snapshots.resolve_snapshot_manifest(
            clean_reference,
            engine,
        )
        if not isinstance(payload, Mapping) or not payload:
            raise DetailSnapshotPrecomputeError(
                f"{page} {selection!r} {section} has an invalid payload"
            )
        if page == "exporter":
            if manifest.get("origin_country") != selection:
                raise DetailSnapshotPrecomputeError(
                    f"Exporter {selection!r} {section} manifest does not "
                    "match its selection"
                )
        elif not manifest.get("destination_countries"):
            raise DetailSnapshotPrecomputeError(
                f"Importer {selection!r} {section} manifest has no "
                "destination countries"
            )


def _cache_metrics() -> dict[str, int]:
    stores = snapshots._get_persistent_stores()
    return {
        "volume_bytes": int(stores.cache.volume()),
        "size_limit_bytes": int(stores.cache.size_limit),
        "headroom_bytes": max(
            0,
            int(stores.cache.size_limit) - int(stores.cache.volume()),
        ),
    }


def precompute_detail_snapshots(
    exporter_page,
    importer_page,
    *,
    exporter_countries: Sequence[str],
    importer_targets: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Precompute selected detail pages and prove cross-process resolution."""

    _require_persistent_disk_cache()
    started_at = time.perf_counter()
    references: list[dict[str, Any]] = []
    results: list[PreparedSnapshot] = []

    with audit_read_only_sql(
        [exporter_page.engine, importer_page.engine]
    ) as sql_audit:
        exporter_before = (
            _exporter_source_sentinel(exporter_page)
            if exporter_countries
            else None
        )
        importer_catalog = (
            importer_page.build_destination_catalog(importer_page.engine)
            if importer_targets
            else []
        )
        importer_before = (
            _importer_source_sentinel(importer_page)
            if importer_targets
            else None
        )

        for country in exporter_countries:
            built_references, built_results = (
                precompute_exporter_selection(
                    exporter_page,
                    country,
                    exporter_before["source_context"],
                )
            )
            for reference, result in zip(
                built_references,
                built_results,
                strict=True,
            ):
                reference["_precompute_page"] = "exporter"
                reference["_precompute_section"] = result.section
                reference["_precompute_selection"] = result.selection
            references.extend(built_references)
            results.extend(built_results)

        for aggregation, selected_value in importer_targets:
            built_references, built_results = (
                precompute_importer_selection(
                    importer_page,
                    aggregation,
                    selected_value,
                    importer_catalog,
                    importer_before["source_context"],
                )
            )
            for reference, result in zip(
                built_references,
                built_results,
                strict=True,
            ):
                reference["_precompute_page"] = "importer"
                reference["_precompute_section"] = result.section
                reference["_precompute_selection"] = result.selection
            references.extend(built_references)
            results.extend(built_results)

        exporter_after = (
            _exporter_source_sentinel(exporter_page)
            if exporter_countries
            else None
        )
        importer_after = (
            _importer_source_sentinel(importer_page)
            if importer_targets
            else None
        )
        if (
            exporter_before is not None
            and _stable_token(
                "exporter-precompute-source-audit",
                exporter_before,
            )
            != _stable_token(
                "exporter-precompute-source-audit",
                exporter_after,
            )
        ):
            raise DetailSnapshotPrecomputeError(
                "Exporter source versions changed during precompute; "
                "discard this run and retry"
            )
        if (
            importer_before is not None
            and _stable_token(
                "importer-precompute-source-audit",
                importer_before,
            )
            != _stable_token(
                "importer-precompute-source-audit",
                importer_after,
            )
        ):
            raise DetailSnapshotPrecomputeError(
                "Importer source versions changed during precompute; "
                "discard this run and retry"
            )

        _validate_reopened_persistence(
            references,
            [exporter_page.engine, importer_page.engine],
        )
        cache_metrics = _cache_metrics()

    return {
        "status": "ready",
        "duration_seconds": round(
            time.perf_counter() - started_at,
            6,
        ),
        "sql_audit": {
            "read_only_statement_count": sql_audit.statement_count,
            "rejected_statement_count": (
                sql_audit.rejected_statement_count
            ),
        },
        "cache": cache_metrics,
        "snapshots": [result.as_dict() for result in results],
    }
