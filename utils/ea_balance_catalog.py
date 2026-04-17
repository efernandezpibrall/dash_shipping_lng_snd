from __future__ import annotations

from typing import Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine


EA_DATASET_CATALOG_VIEW = "ea_dataset_catalog"
EA_LNG_BALANCE_SELECTION_TABLE = "fundamentals_ea_lng_balance_dataset_selection"
EA_LNG_BALANCE_RESOLVED_VIEW = "fundamentals_ea_lng_balance_datasets_resolved"
EA_LNG_BALANCE_LEGACY_TABLE = "fundamentals_ea_lng_balance_datasets"

_object_support_cache: dict[tuple[int, str], tuple[bool, bool, bool]] = {}


def build_ea_dataset_catalog_view_sql(schema: str) -> str:
    return f"""
CREATE OR REPLACE VIEW {schema}.{EA_DATASET_CATALOG_VIEW} AS
WITH ranked_metadata AS (
    SELECT
        CAST(dataset_id AS INTEGER) AS dataset_id,
        type,
        NULLIF(TRIM(value), '') AS value,
        upload_timestamp_utc,
        ROW_NUMBER() OVER (
            PARTITION BY dataset_id, type
            ORDER BY upload_timestamp_utc DESC NULLS LAST
        ) AS row_num
    FROM {schema}.ea_metadata
    WHERE dataset_id ~ '^[0-9]+$'
),
latest_metadata AS (
    SELECT
        dataset_id,
        type,
        value,
        upload_timestamp_utc
    FROM ranked_metadata
    WHERE row_num = 1
)
SELECT
    dataset_id,
    MAX(CASE WHEN type = 'country' THEN value END) AS country,
    MAX(CASE WHEN type = 'country_iso' THEN value END) AS country_iso,
    MAX(CASE WHEN type = 'region' THEN value END) AS region,
    MAX(CASE WHEN type = 'sub_region' THEN value END) AS sub_region,
    MAX(CASE WHEN type = 'description' THEN value END) AS description,
    MAX(CASE WHEN type = 'aspect' THEN value END) AS aspect,
    MAX(CASE WHEN type = 'aspect_subtype' THEN value END) AS aspect_subtype,
    MAX(CASE WHEN type = 'category' THEN value END) AS category,
    MAX(CASE WHEN type = 'category_subtype' THEN value END) AS category_subtype,
    MAX(CASE WHEN type = 'frequency' THEN value END) AS frequency,
    MAX(CASE WHEN type = 'lifecycle_stage' THEN value END) AS lifecycle_stage,
    MAX(CASE WHEN type = 'source' THEN value END) AS source,
    MAX(CASE WHEN type = 'unit' THEN value END) AS unit,
    MAX(CASE WHEN type = 'forecast_start_date' THEN value END) AS forecast_start_date,
    MAX(CASE WHEN type = 'release_date' THEN value END) AS release_date,
    MAX(upload_timestamp_utc) AS metadata_upload_timestamp_utc
FROM latest_metadata
GROUP BY dataset_id
"""


def build_ea_lng_balance_selection_table_sql(schema: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {schema}.{EA_LNG_BALANCE_SELECTION_TABLE} (
    dataset_id INTEGER PRIMARY KEY
)
"""


def build_ea_lng_balance_resolved_view_sql(schema: str) -> str:
    return f"""
CREATE OR REPLACE VIEW {schema}.{EA_LNG_BALANCE_RESOLVED_VIEW} AS
SELECT
    catalog.dataset_id,
    catalog.country,
    catalog.country_iso,
    catalog.region,
    catalog.sub_region,
    catalog.description,
    catalog.aspect,
    catalog.aspect_subtype,
    catalog.category,
    catalog.category_subtype,
    catalog.frequency,
    catalog.lifecycle_stage,
    catalog.source,
    catalog.unit
FROM {schema}.{EA_LNG_BALANCE_SELECTION_TABLE} selection
JOIN {schema}.{EA_DATASET_CATALOG_VIEW} catalog
    ON catalog.dataset_id = selection.dataset_id
"""


def _relation_exists(
    engine: Engine,
    schema: str,
    relation_name: str,
    relation_types: tuple[str, ...],
) -> bool:
    placeholders = ", ".join(f":type_{index}" for index, _ in enumerate(relation_types))
    params = {"schema_name": schema, "relation_name": relation_name}
    params.update({f"type_{index}": relation_type for index, relation_type in enumerate(relation_types)})

    query = text(
        f"""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema_name
              AND table_name = :relation_name
              AND table_type IN ({placeholders})
        )
        """
    )
    with engine.connect() as connection:
        return bool(connection.execute(query, params).scalar())


def _catalog_view_exists(engine: Engine, schema: str) -> bool:
    return _relation_exists(engine, schema, EA_DATASET_CATALOG_VIEW, ("VIEW",))


def _selection_table_exists(engine: Engine, schema: str) -> bool:
    return _relation_exists(engine, schema, EA_LNG_BALANCE_SELECTION_TABLE, ("BASE TABLE",))


def _legacy_balance_table_exists(engine: Engine, schema: str) -> bool:
    return _relation_exists(engine, schema, EA_LNG_BALANCE_LEGACY_TABLE, ("BASE TABLE",))


def build_inline_ea_dataset_catalog_cte() -> str:
    return """
inline_ranked_ea_metadata AS (
    SELECT
        CAST(dataset_id AS INTEGER) AS dataset_id,
        type,
        NULLIF(TRIM(value), '') AS value,
        upload_timestamp_utc,
        ROW_NUMBER() OVER (
            PARTITION BY dataset_id, type
            ORDER BY upload_timestamp_utc DESC NULLS LAST
        ) AS row_num
    FROM {schema}.ea_metadata
    WHERE dataset_id ~ '^[0-9]+$'
),
inline_latest_ea_metadata AS (
    SELECT
        dataset_id,
        type,
        value,
        upload_timestamp_utc
    FROM inline_ranked_ea_metadata
    WHERE row_num = 1
),
ea_dataset_catalog AS (
    SELECT
        dataset_id,
        MAX(CASE WHEN type = 'country' THEN value END) AS country,
        MAX(CASE WHEN type = 'country_iso' THEN value END) AS country_iso,
        MAX(CASE WHEN type = 'region' THEN value END) AS region,
        MAX(CASE WHEN type = 'sub_region' THEN value END) AS sub_region,
        MAX(CASE WHEN type = 'description' THEN value END) AS description,
        MAX(CASE WHEN type = 'aspect' THEN value END) AS aspect,
        MAX(CASE WHEN type = 'aspect_subtype' THEN value END) AS aspect_subtype,
        MAX(CASE WHEN type = 'category' THEN value END) AS category,
        MAX(CASE WHEN type = 'category_subtype' THEN value END) AS category_subtype,
        MAX(CASE WHEN type = 'frequency' THEN value END) AS frequency,
        MAX(CASE WHEN type = 'lifecycle_stage' THEN value END) AS lifecycle_stage,
        MAX(CASE WHEN type = 'source' THEN value END) AS source,
        MAX(CASE WHEN type = 'unit' THEN value END) AS unit
    FROM inline_latest_ea_metadata
    GROUP BY dataset_id
)
"""


def build_resolved_ea_lng_balance_ctes(engine: Engine, schema: str) -> Tuple[str, str]:
    cache_key = (id(engine), schema)
    if cache_key not in _object_support_cache:
        _object_support_cache[cache_key] = (
            _catalog_view_exists(engine, schema),
            _selection_table_exists(engine, schema),
            _legacy_balance_table_exists(engine, schema),
        )

    (
        catalog_view_exists,
        selection_table_exists,
        legacy_balance_table_exists,
    ) = _object_support_cache[cache_key]

    cte_parts: list[str] = []

    if catalog_view_exists:
        catalog_reference = f"{schema}.{EA_DATASET_CATALOG_VIEW}"
    else:
        cte_parts.append(build_inline_ea_dataset_catalog_cte().format(schema=schema))
        catalog_reference = "ea_dataset_catalog"

    if selection_table_exists:
        selection_reference = f"{schema}.{EA_LNG_BALANCE_SELECTION_TABLE}"
        selection_sql = f"SELECT dataset_id FROM {selection_reference}"
    elif legacy_balance_table_exists:
        selection_sql = (
            f"SELECT DISTINCT dataset_id FROM {schema}.{EA_LNG_BALANCE_LEGACY_TABLE}"
        )
    else:
        selection_sql = "SELECT NULL::INTEGER AS dataset_id WHERE FALSE"

    cte_parts.append(
        f"""
resolved_lng_balance_datasets AS (
    SELECT
        CAST(catalog.dataset_id AS TEXT) AS dataset_id,
        catalog.country,
        catalog.country_iso,
        catalog.region,
        catalog.sub_region,
        catalog.description,
        catalog.aspect,
        catalog.aspect_subtype,
        catalog.category,
        catalog.category_subtype,
        catalog.frequency,
        catalog.lifecycle_stage,
        catalog.source,
        catalog.unit
    FROM ({selection_sql}) selection
    JOIN {catalog_reference} catalog
        ON catalog.dataset_id = selection.dataset_id
)
"""
    )

    return ",\n".join(cte_parts), "resolved_lng_balance_datasets"
