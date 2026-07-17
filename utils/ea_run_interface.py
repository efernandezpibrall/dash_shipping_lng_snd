"""Validated application interface to the Energy Aspects change ledger.

Dash consumers may read run metadata plus the public current view and
historical reconstruction function.  They must not read the physical event
table directly.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sqlalchemy import text


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def normalize_ea_run_id(value: Any, *, field_name: str = "ea_as_of_run_id") -> int:
    """Return a built-in positive int, rejecting bools and coercions."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _qualified(schema: str, relation: str) -> str:
    if not _IDENTIFIER.fullmatch(schema) or not _IDENTIFIER.fullmatch(relation):
        raise ValueError("Invalid database identifier")
    return f'"{schema}"."{relation}"'


def _iso_snapshot(value: Any) -> str | None:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.isoformat().replace("+00:00", "Z")


def _run_record(row: Any) -> dict[str, Any]:
    mapping = dict(row._mapping if hasattr(row, "_mapping") else row)
    return {
        "run_id": int(mapping["run_id"]),
        "snapshot_at": _iso_snapshot(mapping.get("snapshot_at")),
        "change_count": int(mapping.get("change_count") or 0),
        "delete_count": int(mapping.get("delete_count") or 0),
    }


def fetch_current_ea_run(engine, *, schema: str = "at_lng") -> dict[str, Any]:
    """Return the latest accepted/full EA run, including zero-change runs."""
    runs = _qualified(schema, "ea_ingestion_runs")
    query = text(
        f"""
        SELECT run_id, snapshot_at, change_count, delete_count
        FROM {runs}
        WHERE status = 'accepted'
          AND coverage = 'full'
        ORDER BY run_id DESC
        LIMIT 1
        """
    )
    with engine.connect() as connection:
        row = connection.execute(query).first()
    if row is None:
        raise RuntimeError("No accepted full Energy Aspects run is available")
    return _run_record(row)


def fetch_ea_run(
    engine,
    ea_as_of_run_id: int,
    *,
    schema: str = "at_lng",
) -> dict[str, Any]:
    """Validate and return one accepted/full run."""
    run_id = normalize_ea_run_id(ea_as_of_run_id)
    runs = _qualified(schema, "ea_ingestion_runs")
    query = text(
        f"""
        SELECT run_id, snapshot_at, change_count, delete_count
        FROM {runs}
        WHERE run_id = :run_id
          AND status = 'accepted'
          AND coverage = 'full'
        """
    )
    with engine.connect() as connection:
        row = connection.execute(query, {"run_id": run_id}).first()
    if row is None:
        raise ValueError("Unknown or noncanonical Energy Aspects run ID")
    return _run_record(row)


def fetch_ea_comparison_runs(
    engine,
    *,
    max_run_id: int,
    schema: str = "at_lng",
    min_snapshot_at: str | None = None,
) -> list[dict[str, Any]]:
    """Return changed accepted/full runs bounded to a captured current run."""
    bound_run_id = normalize_ea_run_id(max_run_id, field_name="max_run_id")
    runs = _qualified(schema, "ea_ingestion_runs")
    query = text(
        f"""
        SELECT run_id, snapshot_at, change_count, delete_count
        FROM {runs}
        WHERE status = 'accepted'
          AND coverage = 'full'
          AND change_count + delete_count > 0
          AND run_id <= :max_run_id
          AND (
              CAST(:min_snapshot_at AS timestamptz) IS NULL
              OR snapshot_at >= CAST(:min_snapshot_at AS timestamptz)
          )
        ORDER BY run_id DESC
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(
            query,
            {"max_run_id": bound_run_id, "min_snapshot_at": min_snapshot_at},
        ).fetchall()
    return [_run_record(row) for row in rows]


def ea_values_at_run_source_sql(
    *,
    schema: str,
    dataset_ids_sql: str,
) -> str:
    """Return the fixed bounded-function source used inside larger SQL queries."""
    function = _qualified(schema, "ea_values_at_run")
    # The one-row ``wanted`` relation disappears when the mapping is empty,
    # so PostgreSQL never invokes the strict reconstruction function with an
    # empty array. Empty provider mappings therefore return an empty result.
    return f"""
        (
            SELECT reconstructed.*
            FROM (
                SELECT array_agg(
                    DISTINCT CAST(dataset_id AS text)
                    ORDER BY CAST(dataset_id AS text)
                ) AS dataset_ids
                FROM {dataset_ids_sql}
                WHERE dataset_id IS NOT NULL
                  AND pg_catalog.btrim(CAST(dataset_id AS text)) <> ''
                HAVING count(*) > 0
            ) AS wanted
            CROSS JOIN LATERAL {function}(
                :ea_as_of_run_id,
                wanted.dataset_ids,
                CAST(:ea_start_date AS timestamp),
                CAST(:ea_end_date AS timestamp)
            ) AS reconstructed
        )
    """
