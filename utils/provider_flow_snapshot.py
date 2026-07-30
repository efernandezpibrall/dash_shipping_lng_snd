"""Shared WoodMac/Energy Aspects flow snapshot for balance pages."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging

import pandas as pd
from sqlalchemy import text

from utils.dashboard_snapshot_cache import (
    build_source_key,
    get_or_build_snapshot,
    resolve_snapshot,
)
from utils.ea_balance_catalog import build_resolved_ea_lng_balance_ctes
from utils.export_flow_data import (
    DB_SCHEMA,
    engine,
    fetch_ea_export_flow_raw_data,
    fetch_ea_upload_options as fetch_ea_export_upload_options,
    fetch_woodmac_export_flow_raw_data,
    fetch_woodmac_publication_options as fetch_woodmac_export_publication_options,
)
from utils.ea_run_interface import fetch_current_ea_run
from utils.import_flow_data import (
    fetch_ea_import_flow_raw_data,
    fetch_ea_upload_options as fetch_ea_import_upload_options,
    fetch_woodmac_import_flow_raw_data,
    fetch_woodmac_publication_options as fetch_woodmac_import_publication_options,
)


LOGGER = logging.getLogger(__name__)
NAMESPACE = "provider-flow-source-v2"
MAX_SOURCE_STATE_RETRIES = 3

_SOURCE_STATE_QUERIES = {
    "woodmac_publication": text(f"""
        SELECT MAX(publication_date::timestamp)
        FROM {DB_SCHEMA}.woodmac_gas_imports_exports_monthly__mmtpa
        WHERE metric_name = 'Flow'
    """),
    "mapping_hash": text(f"""
        SELECT md5(COALESCE(string_agg(
            concat_ws('|',
                COALESCE(country, ''),
                COALESCE(country_name, ''),
                COALESCE(continent, ''),
                COALESCE(subcontinent, ''),
                COALESCE(basin, ''),
                COALESCE(shipping_region, ''),
                COALESCE(country_classification_level1, ''),
                COALESCE(country_classification, '')
            ),
            '||' ORDER BY COALESCE(country_name, ''), COALESCE(country, '')
        ), ''))
        FROM {DB_SCHEMA}.mappings_country
    """),
}

_MAPPING_QUERY = text(f"""
    SELECT
        country,
        country_name,
        continent,
        subcontinent,
        basin,
        country_classification_level1,
        country_classification,
        shipping_region
    FROM {DB_SCHEMA}.mappings_country
""")


def _fetch_ea_balance_mapping_hash() -> str:
    """Hash the effective catalog and selection used by both EA flow queries."""

    balance_ctes, resolved_reference = build_resolved_ea_lng_balance_ctes(
        engine, DB_SCHEMA
    )
    query = text(f"""
        WITH
        {balance_ctes}
        SELECT md5(COALESCE(string_agg(
            concat_ws('|',
                COALESCE(dataset_id, ''),
                COALESCE(country, ''),
                COALESCE(country_iso, ''),
                COALESCE(region, ''),
                COALESCE(sub_region, ''),
                COALESCE(description, ''),
                COALESCE(aspect, ''),
                COALESCE(aspect_subtype, ''),
                COALESCE(category, ''),
                COALESCE(category_subtype, ''),
                COALESCE(frequency, ''),
                COALESCE(lifecycle_stage, ''),
                COALESCE(source, ''),
                COALESCE(unit, '')
            ),
            '||' ORDER BY dataset_id
        ), ''))
        FROM {resolved_reference}
    """)
    with engine.connect() as connection:
        return str(connection.execute(query).scalar() or "")


def fetch_provider_flow_source_state() -> dict[str, object]:
    def fetch_scalar(query):
        with engine.connect() as connection:
            return connection.execute(query).scalar()

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="provider-state") as executor:
        futures = {
            name: executor.submit(fetch_scalar, query)
            for name, query in _SOURCE_STATE_QUERIES.items()
        }
        current_ea_future = executor.submit(
            fetch_current_ea_run,
            engine,
            schema=DB_SCHEMA,
        )
        ea_mapping_future = executor.submit(_fetch_ea_balance_mapping_hash)
        state = {name: futures[name].result() for name in _SOURCE_STATE_QUERIES}
        state["ea_balance_mapping_hash"] = ea_mapping_future.result()
        state["current_ea"] = current_ea_future.result()
        return state


def fetch_provider_flow_mapping_state() -> dict[str, str]:
    """Return the two effective mappings used by historical flow queries."""

    def fetch_country_mapping_hash():
        with engine.connect() as connection:
            return str(
                connection.execute(
                    _SOURCE_STATE_QUERIES["mapping_hash"]
                ).scalar()
                or ""
            )

    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="provider-mapping-state",
    ) as executor:
        country_mapping_future = executor.submit(fetch_country_mapping_hash)
        ea_mapping_future = executor.submit(_fetch_ea_balance_mapping_hash)
        return {
            "mapping_hash": country_mapping_future.result(),
            "ea_balance_mapping_hash": ea_mapping_future.result(),
        }


def _fetch_mapping_df() -> pd.DataFrame:
    with engine.connect() as connection:
        return pd.read_sql_query(_MAPPING_QUERY, connection)


def _build_provider_payload(current_ea: dict[str, object]) -> dict[str, object]:
    ea_as_of_run_id = current_ea["run_id"]
    # Longest reads are submitted first so the three-worker bound has the
    # smallest possible critical path.
    tasks = [
        (
            "ea_import",
            lambda: fetch_ea_import_flow_raw_data(
                ea_as_of_run_id=ea_as_of_run_id
            ),
        ),
        (
            "ea_export",
            lambda: fetch_ea_export_flow_raw_data(
                ea_as_of_run_id=ea_as_of_run_id
            ),
        ),
        ("woodmac_import", fetch_woodmac_import_flow_raw_data),
        ("woodmac_export", fetch_woodmac_export_flow_raw_data),
        ("mapping", _fetch_mapping_df),
        ("woodmac_import_options", fetch_woodmac_import_publication_options),
        ("woodmac_export_options", fetch_woodmac_export_publication_options),
        (
            "ea_comparison_runs",
            lambda: fetch_ea_export_upload_options(max_run_id=ea_as_of_run_id),
        ),
    ]
    results: dict[str, object] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="provider-flow") as executor:
        futures = [(name, executor.submit(function)) for name, function in tasks]
        # Consume in declaration order so error reporting is deterministic.
        for name, future in futures:
            try:
                results[name] = future.result()
            except Exception as exc:
                LOGGER.exception("Provider flow snapshot task %s failed", name)
                errors[name] = str(exc)

    if errors:
        details = " | ".join(f"{name}: {message}" for name, message in errors.items())
        raise RuntimeError(f"Provider flow snapshot build failed: {details}")
    results["current_ea"] = dict(current_ea)
    results["ea_import_options"] = results["ea_comparison_runs"]
    results["ea_export_options"] = results["ea_comparison_runs"]
    results["errors"] = errors
    return results


def build_provider_flow_payload() -> tuple[dict[str, object], dict[str, object]]:
    """Build one internally consistent payload without using snapshot storage."""
    for _attempt in range(MAX_SOURCE_STATE_RETRIES):
        source_state = fetch_provider_flow_source_state()
        payload = _build_provider_payload(source_state["current_ea"])
        if fetch_provider_flow_source_state() == source_state:
            return source_state, payload
    raise RuntimeError("Provider flow sources changed during payload construction")


def get_provider_flow_snapshot(*, force: bool = False):
    for _attempt in range(MAX_SOURCE_STATE_RETRIES):
        source_state = fetch_provider_flow_source_state()
        source_key = build_source_key(NAMESPACE, source_state)
        reference, payload = get_or_build_snapshot(
            engine,
            namespace=NAMESPACE,
            source_key=source_key,
            builder=lambda: _build_provider_payload(source_state["current_ea"]),
            force=force,
            manifest={
                "source_state": source_state,
                "current_ea": source_state["current_ea"],
            },
        )
        if fetch_provider_flow_source_state() == source_state:
            return reference, payload
    raise RuntimeError("Provider flow sources changed during snapshot construction")


def get_provider_flow_snapshot_for_state(
    source_state: dict[str, object], *, force: bool = False
):
    """Resolve or build data for the caller's exact captured source state."""

    current_ea = source_state.get("current_ea")
    if not isinstance(current_ea, dict) or "run_id" not in current_ea:
        raise ValueError("A captured current_ea state is required")
    source_key = build_source_key(NAMESPACE, source_state)
    reference, payload = get_or_build_snapshot(
        engine,
        namespace=NAMESPACE,
        source_key=source_key,
        builder=lambda: _build_provider_payload(current_ea),
        force=force,
        manifest={"source_state": source_state, "current_ea": current_ea},
    )
    if fetch_provider_flow_source_state() != source_state:
        raise RuntimeError("Provider flow sources changed from the captured page state")
    return reference, payload


def resolve_provider_flow_snapshot(value):
    return resolve_snapshot(value, engine, expected_namespace=NAMESPACE)
