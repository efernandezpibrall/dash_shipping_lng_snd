import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import datetime as dt
import hashlib
import json
import math
import re
from threading import RLock
import uuid

import pandas as pd
import plotly.graph_objects as go
import dash_ag_grid as dag
from dash import (
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    ctx,
    dcc,
    html,
    no_update,
)
from dash.dash_table.Format import Format, Scheme
from utils.ag_grid_tables import create_ag_grid_from_datatable
from dash.exceptions import PreventUpdate
from openpyxl.styles import Border, Side
from sqlalchemy import text

from fundamentals.terminals.capacity_source_cache_utils import (
    capacity_events_to_monthly_country,
    capacity_source_event_bounds,
    compile_capacity_source_events,
    complete_capacity_source_refresh,
    fetch_capacity_source_events,
    fetch_capacity_source_state,
    mark_capacity_source_failed,
    mark_capacity_source_running,
    read_capacity_refresh_job,
)
from fundamentals.terminals.capacity_scenario_utils import (
    create_capacity_scenario_from_source,
    delete_capacity_scenario,
    fetch_capacity_scenario_rows,
    get_available_capacity_scenarios,
    save_capacity_scenario_rows,
)
from fundamentals.terminals.ramp_forecast_utils import (
    generate_ramp_forecast_for_capacity_scenario,
    get_active_ramp_profile_catalog,
    get_active_seasonal_profile_options,
    get_latest_ramp_forecast_qa_issues,
    get_latest_ramp_forecast_status,
    publish_ramp_forecast_run,
)
from fundamentals.terminals.terminal_registry_utils import (
    assign_terminal_train_profiles,
    attach_registry_keys_to_capacity_rows,
    backfill_terminal_keys,
    build_energy_aspects_provider_identity,
    create_draft_registry_for_unresolved_non_woodmac_rows,
    find_terminal_train_candidates,
    list_canonical_country_names,
    normalize_train_label,
    replace_provider_source_allocations,
    register_terminal_train,
)
from utils.balance_time import (
    build_lng_season_periods as _build_lng_season_periods,
    filter_by_month_date_range as _filter_by_date_range,
    get_month_date_bounds as _get_date_bounds,
    normalize_month_date as _normalize_month_date,
)
from utils.balance_components import create_balance_empty_state as _create_empty_state
from utils.dataframe_store import (
    deserialize_dataframe_store as _deserialize_dataframe,
    serialize_dataframe_store as _serialize_dataframe,
)
from utils.export_flow_data import (
    COUNTRY_MAPPING_CTE,
    DB_SCHEMA,
    build_export_flow_matrix,
    default_selected_countries,
    engine,
    get_available_countries,
)
from utils.flow_country_selection import serialize_timestamp
from utils.table_styles import (
    StandardTableStyleManager,
    TABLE_COLORS,
    format_table_cell_value_1dp as _format_table_cell_value,
)


EXPORT_BUTTON_STYLE = {
    "marginLeft": "12px",
    "padding": "3px 10px",
    "backgroundColor": "#28a745",
    "color": "white",
    "border": "none",
    "borderRadius": "3px",
    "cursor": "pointer",
    "fontWeight": "bold",
    "fontSize": "11px",
}

TRAIN_CHANGE_CONTROL_SHELL_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "gap": "10px",
    "padding": "6px 8px 6px 12px",
    "backgroundColor": "#ffffff",
    "border": "1px solid #dbe4ee",
    "borderRadius": "999px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.05)",
}

TRAIN_CHANGE_CONTROL_LABEL_STYLE = {
    "fontSize": "11px",
    "fontWeight": "700",
    "letterSpacing": "0.06em",
    "textTransform": "uppercase",
    "color": "#64748b",
}

PRIMARY_CONTROL_BUTTON_STYLE = {
    "padding": "7px 14px",
    "backgroundColor": "#1e3a5f",
    "color": "white",
    "border": "1px solid #1e3a5f",
    "borderRadius": "999px",
    "cursor": "pointer",
    "fontWeight": "600",
    "fontSize": "12px",
}

CAPACITY_SOURCE_TABLE = f"{DB_SCHEMA}.woodmac_lng_plant_monthly_capacity_nominal_mta"
WOODMAC_ANNUAL_OUTPUT_SOURCE_TABLE = f"{DB_SCHEMA}.woodmac_lng_plant_train_annual_output_mta"
EA_CAPACITY_SOURCE_TABLE = f"{DB_SCHEMA}.ea_lng_liquefaction_projects"
EA_EXCLUDED_STATUSES = {"cancelled", "retired"}
EA_NEGATIVE_STATUSES: set[str] = set()
EA_TOP_PANEL_EXCLUDED_STATUSES = EA_EXCLUDED_STATUSES
WOODMAC_ANNUAL_CARRY_FORWARD_START = "2029-01-01"
WOODMAC_LEGACY_CAPACITY_NOTE = (
    "Dashboard assumption: some legacy annual-only Woodmac reductions around 2029 "
    "(for example Algeria LNG (Skikda), Algeria LNG (Bethioua), and Bontang) "
    "appear to reflect source timing or coverage rather than a true 2029 shutdown, "
    f"so their last available proxy capacity is carried forward from {WOODMAC_ANNUAL_CARRY_FORWARD_START[:4]} onward."
)
EA_SCHEDULE_CAPACITY_NOTE = (
    "Derived cumulative schedule from the latest Energy Aspects snapshot using start_date month; "
    "this is not a historical monthly capacity feed."
)
TRAIN_TIMELINE_CHART_SOURCE_CONFIG = {
    "woodmac": {
        "label": "Woodmac",
        "date_column": "Woodmac First Date",
        "capacity_column": "Woodmac Capacity Change",
        "out_of_range_flag": "__woodmac_out_of_range",
    },
    "energy_aspects": {
        "label": "Energy Aspects",
        "date_column": "Energy Aspects First Date",
        "capacity_column": "Energy Aspects Capacity Change",
        "out_of_range_flag": "__ea_out_of_range",
    },
    "internal_scenario": {
        "label": "Internal Scenario",
        "date_column": "Scenario First Date",
        "capacity_column": "Scenario Capacity",
        "out_of_range_flag": "__scenario_out_of_range",
    },
}

COUNTRY_COLORS = {
    "United States": "#003A6C",
    "Qatar": "#A21F5A",
    "Australia": "#78BE20",
    "Canada": "#00A3E0",
    "Russia": "#7F3F98",
    "Mozambique": "#F58220",
    "Mexico": "#6BCABA",
    "Nigeria": "#E03C31",
    "Malaysia": "#005EB8",
    "Indonesia": "#00B5E2",
    "United Arab Emirates": "#0F766E",
}

FALLBACK_COLORS = [
    "#003A6C",
    "#00A3E0",
    "#6BCABA",
    "#90B23C",
    "#FFC72C",
    "#F58220",
    "#E03C31",
    "#7F3F98",
    "#A21F5A",
    "#005EB8",
    "#00B5E2",
]

TRAIN_CHANGE_TIME_VIEW_PERIOD_LABELS = {
    "monthly": "monthly",
    "quarterly": "quarterly",
    "seasonally": "seasonal",
    "yearly": "yearly",
}

SEASONAL_TIME_VIEW_TOOLTIP = (
    "Seasonally: Summer (Y-S) runs from April to September of year Y. "
    "Winter (Y-W) runs from October to December of year Y and January to March of year Y+1."
)

TRAIN_CHANGE_DETAIL_VIEW_LABELS = {
    "total": "Total",
    "country": "Country",
    "plants": "Plants View",
    "plants_trains": "Plants + Trains View",
}
TRAIN_TIMELINE_SHEET_NAME = "Train Timeline"
TRAIN_TIMELINE_IMPORT_META_SHEET_NAME = "__TrainTimelineImportMeta"
TRAIN_TIMELINE_IMPORT_TEMPLATE_VERSION = "train_timeline_upload_v3"
CAPACITY_PROVIDER_MAPPER_VERSION = "capacity_provider_mapper_v4_provider_links_only"
TRAIN_TIMELINE_IMPORT_KEY_COLUMN = "__scenario_row_key"
TRAIN_TIMELINE_IMPORT_EXPORT_ROW_COLUMN = "__export_row_number"
TRAIN_TIMELINE_IMPORT_HAS_SCENARIO_ROW_COLUMN = "__has_saved_scenario_row"
TRAIN_TIMELINE_IMPORT_SAVED_ROW_COLUMN = "__saved_scenario_row_json"
TRAIN_TIMELINE_IMPORT_META_COLUMNS = {
    "__template_version",
    "__scenario_id",
    "__scenario_name",
    "__original_name_visibility",
    TRAIN_TIMELINE_IMPORT_HAS_SCENARIO_ROW_COLUMN,
    TRAIN_TIMELINE_IMPORT_SAVED_ROW_COLUMN,
    TRAIN_TIMELINE_IMPORT_EXPORT_ROW_COLUMN,
    TRAIN_TIMELINE_IMPORT_KEY_COLUMN,
}
TRAIN_TIMELINE_EDITABLE_COLUMNS = {
    "Scenario First Date",
    "Scenario Capacity",
    "Scenario Note",
}
TRAIN_TIMELINE_PROVIDER_COLUMNS = {
    "Woodmac Original Name",
    "Woodmac FID Date",
    "Woodmac First Date",
    "Woodmac Capacity Change",
    "Energy Aspects Original Plant",
    "Energy Aspects Original Train",
    "Energy Aspects First Date",
    "Energy Aspects Capacity Change",
}
TRAIN_TIMELINE_DATE_COLUMNS = {
    "Woodmac FID Date",
    "Woodmac First Date",
    "Energy Aspects First Date",
    "Scenario First Date",
}
TRAIN_TIMELINE_NUMERIC_COLUMNS = {
    "Woodmac Capacity Change",
    "Energy Aspects Capacity Change",
    "Scenario Capacity",
}
INTERNAL_SCENARIO_ADDS_COLUMN = "Internal Scenario Adds (MTPA)"
INTERNAL_SCENARIO_REDUCTIONS_COLUMN = "Internal Scenario Reductions (MTPA)"
INTERNAL_SCENARIO_NET_COLUMN = "Internal Scenario Net Delta (MTPA)"
INTERNAL_SCENARIO_EMPTY_MESSAGE = (
    "Select or create an internal scenario from Train Timeline to populate this section."
)
YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE = (
    "Select an internal scenario from the Internal Scenario dropdown at the top of the page to populate this section."
)

WOODMAC_CAPACITY_QUERY = f"""
WITH latest_plant_summary AS (
    SELECT DISTINCT ON (plant_row.id_plant)
        plant_row.id_plant,
        plant_row.country_name
    FROM {DB_SCHEMA}.woodmac_lng_plant_summary AS plant_row
    ORDER BY plant_row.id_plant, plant_row.upload_timestamp_utc DESC
),
latest_monthly_capacity AS (
    SELECT DISTINCT ON (
        source_row.id_plant,
        source_row.id_lng_train,
        source_row.year,
        source_row.month
    )
        source_row.id_plant,
        source_row.id_lng_train,
        source_row.year,
        TO_DATE(
            source_row.year || '-' || LPAD(source_row.month::text, 2, '0') || '-01',
            'YYYY-MM-DD'
        ) AS month,
        COALESCE(
            NULLIF(BTRIM(plant_summary.country_name), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'country_name'), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'country'), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'country_nm'), ''),
            'Unknown'
        ) AS country_name,
        source_row.metric_value AS capacity_mtpa,
        source_row.upload_timestamp_utc
    FROM {CAPACITY_SOURCE_TABLE} AS source_row
    LEFT JOIN latest_plant_summary AS plant_summary
        ON source_row.id_plant = plant_summary.id_plant
    WHERE source_row.metric_value IS NOT NULL
    ORDER BY
        source_row.id_plant,
        source_row.id_lng_train,
        source_row.year,
        source_row.month,
        source_row.upload_timestamp_utc DESC
),
monthly_exact_coverage AS (
    SELECT DISTINCT id_plant, id_lng_train, month
    FROM latest_monthly_capacity
),
latest_annual_output AS (
    SELECT DISTINCT ON (annual_row.id_plant, annual_row.id_lng_train, annual_row.year)
        annual_row.id_plant,
        annual_row.id_lng_train,
        annual_row.year,
        COALESCE(
            NULLIF(BTRIM(plant_summary.country_name), ''),
            NULLIF(BTRIM(annual_row.country_name), ''),
            'Unknown'
        ) AS country_name,
        annual_row.metric_value AS capacity_mtpa,
        annual_row.upload_timestamp_utc
    FROM {WOODMAC_ANNUAL_OUTPUT_SOURCE_TABLE} AS annual_row
    LEFT JOIN latest_plant_summary AS plant_summary
        ON annual_row.id_plant = plant_summary.id_plant
    WHERE annual_row.metric_value IS NOT NULL
    ORDER BY
        annual_row.id_plant,
        annual_row.id_lng_train,
        annual_row.year,
        annual_row.upload_timestamp_utc DESC
),
monthly_train_bounds AS (
    SELECT
        id_plant,
        id_lng_train,
        MAX(month) AS last_monthly_month
    FROM latest_monthly_capacity
    GROUP BY id_plant, id_lng_train
),
annual_train_bounds AS (
    SELECT
        annual_row.id_plant,
        annual_row.id_lng_train,
        MAX(annual_row.country_name) AS country_name,
        MIN(
            CASE
                WHEN annual_row.capacity_mtpa > 0 THEN TO_DATE(
                    annual_row.year || '-01-01',
                    'YYYY-MM-DD'
                )
            END
        ) AS first_active_month,
        TO_DATE(MAX(annual_row.year)::text || '-12-01', 'YYYY-MM-DD') AS last_annual_month,
        MAX(annual_row.capacity_mtpa) AS capacity_mtpa,
        MAX(annual_row.upload_timestamp_utc) AS upload_timestamp_utc
    FROM latest_annual_output AS annual_row
    GROUP BY annual_row.id_plant, annual_row.id_lng_train
    HAVING MAX(annual_row.capacity_mtpa) > 0
),
coverage_horizon AS (
    SELECT GREATEST(
        COALESCE(
            (SELECT MAX(month) FROM latest_monthly_capacity),
            DATE '1900-01-01'
        ),
        COALESCE(
            (
                SELECT TO_DATE(MAX(year)::text || '-12-01', 'YYYY-MM-DD')
                FROM latest_annual_output
            ),
            DATE '1900-01-01'
        )
    ) AS max_month
),
monthly_carry_forward_base AS (
    SELECT
        monthly_row.id_plant,
        monthly_row.id_lng_train,
        monthly_row.country_name,
        monthly_row.capacity_mtpa,
        monthly_bounds.last_monthly_month
    FROM monthly_train_bounds AS monthly_bounds
    JOIN latest_monthly_capacity AS monthly_row
        ON monthly_bounds.id_plant = monthly_row.id_plant
       AND monthly_bounds.id_lng_train = monthly_row.id_lng_train
       AND monthly_bounds.last_monthly_month = monthly_row.month
),
monthly_carry_forward AS (
    SELECT
        monthly_row.id_plant,
        monthly_row.id_lng_train,
        month_series.month::date AS month,
        monthly_row.country_name,
        monthly_row.capacity_mtpa
    FROM monthly_carry_forward_base AS monthly_row
    CROSS JOIN coverage_horizon AS horizon
    CROSS JOIN LATERAL generate_series(
        (monthly_row.last_monthly_month + INTERVAL '1 month')::date,
        horizon.max_month,
        INTERVAL '1 month'
    ) AS month_series(month)
    LEFT JOIN monthly_exact_coverage AS monthly_map
        ON monthly_row.id_plant = monthly_map.id_plant
       AND monthly_row.id_lng_train = monthly_map.id_lng_train
       AND month_series.month::date = monthly_map.month
    WHERE monthly_map.id_plant IS NULL
),
annual_only_fallback_base AS (
    SELECT
        annual_row.id_plant,
        annual_row.id_lng_train,
        annual_row.country_name,
        annual_row.capacity_mtpa,
        annual_row.first_active_month,
        annual_row.last_annual_month
    FROM annual_train_bounds AS annual_row
    LEFT JOIN monthly_train_bounds AS monthly_bounds
        ON annual_row.id_plant = monthly_bounds.id_plant
       AND annual_row.id_lng_train = monthly_bounds.id_lng_train
    WHERE monthly_bounds.id_plant IS NULL
),
annual_only_fallback AS (
    SELECT
        annual_row.id_plant,
        annual_row.id_lng_train,
        month_series.month::date AS month,
        annual_row.country_name,
        annual_row.capacity_mtpa
    FROM annual_only_fallback_base AS annual_row
    CROSS JOIN LATERAL generate_series(
        annual_row.first_active_month,
        annual_row.last_annual_month,
        INTERVAL '1 month'
    ) AS month_series(month)
    LEFT JOIN monthly_exact_coverage AS monthly_map
        ON annual_row.id_plant = monthly_map.id_plant
       AND annual_row.id_lng_train = monthly_map.id_lng_train
       AND month_series.month::date = monthly_map.month
    WHERE monthly_map.id_plant IS NULL
),
annual_only_carry_forward AS (
    SELECT
        annual_row.id_plant,
        annual_row.id_lng_train,
        month_series.month::date AS month,
        annual_row.country_name,
        annual_row.capacity_mtpa
    FROM annual_only_fallback_base AS annual_row
    CROSS JOIN coverage_horizon AS horizon
    CROSS JOIN LATERAL generate_series(
        GREATEST(
            DATE '{WOODMAC_ANNUAL_CARRY_FORWARD_START}',
            (annual_row.last_annual_month + INTERVAL '1 month')::date
        ),
        horizon.max_month,
        INTERVAL '1 month'
    ) AS month_series(month)
    LEFT JOIN monthly_exact_coverage AS monthly_map
        ON annual_row.id_plant = monthly_map.id_plant
       AND annual_row.id_lng_train = monthly_map.id_lng_train
       AND month_series.month::date = monthly_map.month
    WHERE monthly_map.id_plant IS NULL
),
combined_capacity AS (
    SELECT
        id_plant,
        id_lng_train,
        month,
        country_name,
        capacity_mtpa
    FROM latest_monthly_capacity

    UNION ALL

    SELECT
        id_plant,
        id_lng_train,
        month,
        country_name,
        capacity_mtpa
    FROM monthly_carry_forward

    UNION ALL

    SELECT
        id_plant,
        id_lng_train,
        month,
        country_name,
        capacity_mtpa
    FROM annual_only_fallback

    UNION ALL

    SELECT
        id_plant,
        id_lng_train,
        month,
        country_name,
        capacity_mtpa
    FROM annual_only_carry_forward
)
SELECT
    month,
    country_name,
    SUM(capacity_mtpa) AS total_mmtpa
FROM combined_capacity
GROUP BY month, country_name
HAVING SUM(capacity_mtpa) <> 0
ORDER BY month, country_name
"""

WOODMAC_TRAIN_CAPACITY_QUERY = f"""
WITH latest_plant_summary AS (
    SELECT DISTINCT ON (plant_row.id_plant)
        plant_row.id_plant,
        plant_row.plant_name,
        plant_row.country_name
    FROM {DB_SCHEMA}.woodmac_lng_plant_summary AS plant_row
    ORDER BY plant_row.id_plant, plant_row.upload_timestamp_utc DESC
),
latest_train_upload AS (
    SELECT MAX(upload_timestamp_utc) AS upload_timestamp_utc
    FROM {DB_SCHEMA}.woodmac_lng_plant_train
),
latest_train_metadata AS (
    SELECT DISTINCT ON (train_row.id_plant, train_row.id_lng_train)
        train_row.id_plant,
        train_row.id_lng_train,
        train_row.lng_train_date_fid,
        to_jsonb(train_row) AS train_json
    FROM {DB_SCHEMA}.woodmac_lng_plant_train AS train_row
    JOIN latest_train_upload AS latest_upload
        ON train_row.upload_timestamp_utc = latest_upload.upload_timestamp_utc
    ORDER BY
        train_row.id_plant,
        train_row.id_lng_train,
        train_row.lng_train_date_fid DESC NULLS LAST
),
latest_capacity AS (
    SELECT
        source_row.id_plant,
        source_row.id_lng_train,
        source_row.year,
        TO_DATE(
            source_row.year || '-' || LPAD(source_row.month::text, 2, '0') || '-01',
            'YYYY-MM-DD'
        ) AS month,
        COALESCE(
            NULLIF(BTRIM(plant_summary.country_name), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'country_name'), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'country'), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'country_nm'), ''),
            'Unknown'
        ) AS country_name,
        COALESCE(
            NULLIF(BTRIM(plant_summary.plant_name), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'plant_name'), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'lng_plant_name'), ''),
            CONCAT('Plant ', source_row.id_plant::text)
        ) AS plant_name,
        COALESCE(
            NULLIF(BTRIM(train_metadata.train_json ->> 'lng_train_name'), ''),
            NULLIF(BTRIM(train_metadata.train_json ->> 'lng_train_name_short'), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'lng_train_name'), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'lng_train_name_short'), ''),
            CONCAT('Train ', source_row.id_lng_train::text)
        ) AS lng_train_name,
        COALESCE(
            NULLIF(BTRIM(train_metadata.train_json ->> 'lng_train_name_short'), ''),
            NULLIF(BTRIM(train_metadata.train_json ->> 'lng_train_name'), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'lng_train_name_short'), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'lng_train_name'), ''),
            CONCAT('Train ', source_row.id_lng_train::text)
        ) AS lng_train_name_short,
        train_metadata.lng_train_date_fid AS woodmac_fid_date,
        source_row.metric_value AS capacity_mtpa
    FROM (
        SELECT DISTINCT ON (
            monthly_row.id_plant,
            monthly_row.id_lng_train,
            monthly_row.year,
            monthly_row.month
        )
            monthly_row.*
        FROM {CAPACITY_SOURCE_TABLE} AS monthly_row
        WHERE monthly_row.metric_value IS NOT NULL
        ORDER BY
            monthly_row.id_plant,
            monthly_row.id_lng_train,
            monthly_row.year,
            monthly_row.month,
            monthly_row.upload_timestamp_utc DESC
    ) AS source_row
    LEFT JOIN latest_plant_summary AS plant_summary
        ON source_row.id_plant = plant_summary.id_plant
    LEFT JOIN latest_train_metadata AS train_metadata
        ON source_row.id_plant = train_metadata.id_plant
       AND source_row.id_lng_train = train_metadata.id_lng_train
),
monthly_coverage_map AS (
    SELECT DISTINCT id_plant, id_lng_train, month
    FROM latest_capacity
),
latest_annual_output AS (
    SELECT DISTINCT ON (
        annual_row.id_plant,
        annual_row.id_lng_train,
        annual_row.year
    )
        annual_row.id_plant,
        annual_row.id_lng_train,
        annual_row.year,
        COALESCE(
            NULLIF(BTRIM(plant_summary.country_name), ''),
            NULLIF(BTRIM(annual_row.country_name), ''),
            'Unknown'
        ) AS country_name,
        COALESCE(
            NULLIF(BTRIM(plant_summary.plant_name), ''),
            NULLIF(BTRIM(annual_row.plant_name), ''),
            CONCAT('Plant ', annual_row.id_plant::text)
        ) AS plant_name,
        COALESCE(
            NULLIF(BTRIM(train_metadata.train_json ->> 'lng_train_name'), ''),
            NULLIF(BTRIM(train_metadata.train_json ->> 'lng_train_name_short'), ''),
            NULLIF(BTRIM(annual_row.lng_train_name), ''),
            NULLIF(BTRIM(annual_row.lng_train_name_short), ''),
            CONCAT('Train ', annual_row.id_lng_train::text)
        ) AS lng_train_name,
        COALESCE(
            NULLIF(BTRIM(train_metadata.train_json ->> 'lng_train_name_short'), ''),
            NULLIF(BTRIM(train_metadata.train_json ->> 'lng_train_name'), ''),
            NULLIF(BTRIM(annual_row.lng_train_name_short), ''),
            CONCAT('Train ', annual_row.id_lng_train::text)
        ) AS lng_train_name_short,
        train_metadata.lng_train_date_fid AS woodmac_fid_date,
        annual_row.metric_value AS capacity_mtpa
    FROM {WOODMAC_ANNUAL_OUTPUT_SOURCE_TABLE} AS annual_row
    LEFT JOIN latest_plant_summary AS plant_summary
        ON annual_row.id_plant = plant_summary.id_plant
    LEFT JOIN latest_train_metadata AS train_metadata
        ON annual_row.id_plant = train_metadata.id_plant
       AND annual_row.id_lng_train = train_metadata.id_lng_train
    WHERE annual_row.metric_value IS NOT NULL
    ORDER BY
        annual_row.id_plant,
        annual_row.id_lng_train,
        annual_row.year,
        annual_row.upload_timestamp_utc DESC
),
monthly_train_bounds AS (
    SELECT
        id_plant,
        id_lng_train,
        MAX(month) AS last_monthly_month
    FROM latest_capacity
    GROUP BY id_plant, id_lng_train
),
annual_train_bounds AS (
    SELECT
        annual_row.id_plant,
        annual_row.id_lng_train,
        MAX(annual_row.country_name) AS country_name,
        MAX(annual_row.plant_name) AS plant_name,
        MAX(annual_row.lng_train_name) AS lng_train_name,
        MAX(annual_row.lng_train_name_short) AS lng_train_name_short,
        MAX(annual_row.woodmac_fid_date) AS woodmac_fid_date,
        MIN(
            CASE
                WHEN annual_row.capacity_mtpa > 0 THEN TO_DATE(
                    annual_row.year || '-01-01',
                    'YYYY-MM-DD'
                )
            END
        ) AS first_active_month,
        TO_DATE(MAX(annual_row.year)::text || '-12-01', 'YYYY-MM-DD') AS last_annual_month,
        MAX(annual_row.capacity_mtpa) AS capacity_mtpa
    FROM latest_annual_output AS annual_row
    GROUP BY annual_row.id_plant, annual_row.id_lng_train
    HAVING MAX(annual_row.capacity_mtpa) > 0
),
coverage_horizon AS (
    SELECT GREATEST(
        COALESCE(
            (SELECT MAX(month) FROM latest_capacity),
            DATE '1900-01-01'
        ),
        COALESCE(
            (
                SELECT TO_DATE(MAX(year)::text || '-12-01', 'YYYY-MM-DD')
                FROM latest_annual_output
            ),
            DATE '1900-01-01'
        )
    ) AS max_month
),
monthly_carry_forward_base AS (
    SELECT
        monthly_row.id_plant,
        monthly_row.id_lng_train,
        monthly_row.country_name,
        monthly_row.plant_name,
        monthly_row.lng_train_name,
        monthly_row.lng_train_name_short,
        monthly_row.woodmac_fid_date,
        monthly_row.capacity_mtpa,
        monthly_bounds.last_monthly_month
    FROM monthly_train_bounds AS monthly_bounds
    JOIN latest_capacity AS monthly_row
        ON monthly_bounds.id_plant = monthly_row.id_plant
       AND monthly_bounds.id_lng_train = monthly_row.id_lng_train
       AND monthly_bounds.last_monthly_month = monthly_row.month
),
monthly_carry_forward AS (
    SELECT
        monthly_row.id_plant,
        monthly_row.id_lng_train,
        month_series.month::date AS month,
        monthly_row.country_name,
        monthly_row.plant_name,
        monthly_row.lng_train_name,
        monthly_row.lng_train_name_short,
        monthly_row.woodmac_fid_date,
        monthly_row.capacity_mtpa
    FROM monthly_carry_forward_base AS monthly_row
    CROSS JOIN coverage_horizon AS horizon
    CROSS JOIN LATERAL generate_series(
        (monthly_row.last_monthly_month + INTERVAL '1 month')::date,
        horizon.max_month,
        INTERVAL '1 month'
    ) AS month_series(month)
),
annual_only_fallback_base AS (
    SELECT
        annual_row.id_plant,
        annual_row.id_lng_train,
        annual_row.country_name,
        annual_row.plant_name,
        annual_row.lng_train_name,
        annual_row.lng_train_name_short,
        annual_row.woodmac_fid_date,
        annual_row.capacity_mtpa,
        annual_row.first_active_month,
        annual_row.last_annual_month
    FROM annual_train_bounds AS annual_row
    LEFT JOIN monthly_train_bounds AS monthly_bounds
        ON annual_row.id_plant = monthly_bounds.id_plant
       AND annual_row.id_lng_train = monthly_bounds.id_lng_train
    WHERE monthly_bounds.id_plant IS NULL
),
annual_only_fallback AS (
    SELECT
        annual_row.id_plant,
        annual_row.id_lng_train,
        month_series.month::date AS month,
        annual_row.country_name,
        annual_row.plant_name,
        annual_row.lng_train_name,
        annual_row.lng_train_name_short,
        annual_row.woodmac_fid_date,
        annual_row.capacity_mtpa
    FROM annual_only_fallback_base AS annual_row
    CROSS JOIN LATERAL generate_series(
        annual_row.first_active_month,
        annual_row.last_annual_month,
        INTERVAL '1 month'
    ) AS month_series(month)
),
annual_only_carry_forward AS (
    SELECT
        annual_row.id_plant,
        annual_row.id_lng_train,
        month_series.month::date AS month,
        annual_row.country_name,
        annual_row.plant_name,
        annual_row.lng_train_name,
        annual_row.lng_train_name_short,
        annual_row.woodmac_fid_date,
        annual_row.capacity_mtpa
    FROM annual_only_fallback_base AS annual_row
    CROSS JOIN coverage_horizon AS horizon
    CROSS JOIN LATERAL generate_series(
        GREATEST(
            DATE '{WOODMAC_ANNUAL_CARRY_FORWARD_START}',
            (annual_row.last_annual_month + INTERVAL '1 month')::date
        ),
        horizon.max_month,
        INTERVAL '1 month'
    ) AS month_series(month)
),
combined_capacity AS (
    SELECT
        id_plant,
        id_lng_train,
        month,
        country_name,
        plant_name,
        lng_train_name,
        lng_train_name_short,
        woodmac_fid_date,
        capacity_mtpa,
        'monthly'::text AS capacity_source
    FROM latest_capacity

    UNION ALL

    SELECT
        id_plant,
        id_lng_train,
        month,
        country_name,
        plant_name,
        lng_train_name,
        lng_train_name_short,
        woodmac_fid_date,
        capacity_mtpa,
        'monthly_carry_forward'::text AS capacity_source
    FROM monthly_carry_forward

    UNION ALL

    SELECT
        id_plant,
        id_lng_train,
        month,
        country_name,
        plant_name,
        lng_train_name,
        lng_train_name_short,
        woodmac_fid_date,
        capacity_mtpa,
        'annual_fallback'::text AS capacity_source
    FROM annual_only_fallback

    UNION ALL

    SELECT
        id_plant,
        id_lng_train,
        month,
        country_name,
        plant_name,
        lng_train_name,
        lng_train_name_short,
        woodmac_fid_date,
        capacity_mtpa,
        'annual_carry_forward'::text AS capacity_source
    FROM annual_only_carry_forward
)
SELECT
    month,
    country_name,
    plant_name,
    lng_train_name,
    lng_train_name_short,
    woodmac_fid_date,
    id_plant,
    id_lng_train,
    capacity_mtpa,
    capacity_source
FROM combined_capacity
ORDER BY month, country_name, plant_name, lng_train_name
"""

WOODMAC_TRAIN_CAPACITY_ARRAY_QUERY = f"""
SELECT
    ARRAY_AGG(month ORDER BY month, country_name, plant_name, lng_train_name) AS month,
    ARRAY_AGG(country_name ORDER BY month, country_name, plant_name, lng_train_name) AS country_name,
    ARRAY_AGG(plant_name ORDER BY month, country_name, plant_name, lng_train_name) AS plant_name,
    ARRAY_AGG(lng_train_name ORDER BY month, country_name, plant_name, lng_train_name) AS lng_train_name,
    ARRAY_AGG(lng_train_name_short ORDER BY month, country_name, plant_name, lng_train_name)
        AS lng_train_name_short,
    ARRAY_AGG(woodmac_fid_date ORDER BY month, country_name, plant_name, lng_train_name)
        AS woodmac_fid_date,
    ARRAY_AGG(id_plant ORDER BY month, country_name, plant_name, lng_train_name) AS id_plant,
    ARRAY_AGG(id_lng_train ORDER BY month, country_name, plant_name, lng_train_name)
        AS id_lng_train,
    ARRAY_AGG(capacity_mtpa ORDER BY month, country_name, plant_name, lng_train_name)
        AS capacity_mtpa,
    ARRAY_AGG(capacity_source ORDER BY month, country_name, plant_name, lng_train_name)
        AS capacity_source
FROM ({WOODMAC_TRAIN_CAPACITY_QUERY}) AS capacity_rows
"""

WOODMAC_CAPACITY_METADATA_QUERY = f"""
WITH latest_monthly_capacity AS (
    SELECT DISTINCT ON (source_row.id_plant, source_row.id_lng_train, source_row.year, source_row.month)
        source_row.id_plant,
        source_row.id_lng_train,
        source_row.year,
        TO_DATE(
            source_row.year || '-' || LPAD(source_row.month::text, 2, '0') || '-01',
            'YYYY-MM-DD'
        ) AS month,
        source_row.upload_timestamp_utc
    FROM {CAPACITY_SOURCE_TABLE} AS source_row
    WHERE source_row.metric_value IS NOT NULL
    ORDER BY
        source_row.id_plant,
        source_row.id_lng_train,
        source_row.year,
        source_row.month,
        source_row.upload_timestamp_utc DESC
),
monthly_exact_coverage AS (
    SELECT DISTINCT id_plant, id_lng_train, month
    FROM latest_monthly_capacity
),
latest_annual_output AS (
    SELECT DISTINCT ON (annual_row.id_plant, annual_row.id_lng_train, annual_row.year)
        annual_row.id_plant,
        annual_row.id_lng_train,
        annual_row.year,
        annual_row.metric_value,
        annual_row.upload_timestamp_utc
    FROM {WOODMAC_ANNUAL_OUTPUT_SOURCE_TABLE} AS annual_row
    WHERE annual_row.metric_value IS NOT NULL
    ORDER BY
        annual_row.id_plant,
        annual_row.id_lng_train,
        annual_row.year,
        annual_row.upload_timestamp_utc DESC
),
monthly_train_bounds AS (
    SELECT
        id_plant,
        id_lng_train,
        MAX(month) AS last_monthly_month
    FROM latest_monthly_capacity
    GROUP BY id_plant, id_lng_train
),
annual_train_bounds AS (
    SELECT
        annual_row.id_plant,
        annual_row.id_lng_train,
        MIN(
            CASE
                WHEN annual_row.metric_value > 0 THEN TO_DATE(
                    annual_row.year || '-01-01',
                    'YYYY-MM-DD'
                )
            END
        ) AS first_active_month,
        TO_DATE(MAX(annual_row.year)::text || '-12-01', 'YYYY-MM-DD') AS last_annual_month,
        MAX(annual_row.metric_value) AS capacity_mtpa,
        MAX(annual_row.upload_timestamp_utc) AS upload_timestamp_utc
    FROM latest_annual_output AS annual_row
    GROUP BY annual_row.id_plant, annual_row.id_lng_train
    HAVING MAX(annual_row.metric_value) > 0
),
coverage_horizon AS (
    SELECT GREATEST(
        COALESCE(
            (SELECT MAX(month) FROM latest_monthly_capacity),
            DATE '1900-01-01'
        ),
        COALESCE(
            (
                SELECT TO_DATE(MAX(year)::text || '-12-01', 'YYYY-MM-DD')
                FROM latest_annual_output
            ),
            DATE '1900-01-01'
        )
    ) AS max_month
),
monthly_carry_forward_base AS (
    SELECT
        monthly_row.id_plant,
        monthly_row.id_lng_train,
        monthly_bounds.last_monthly_month,
        monthly_row.upload_timestamp_utc
    FROM monthly_train_bounds AS monthly_bounds
    JOIN latest_monthly_capacity AS monthly_row
        ON monthly_bounds.id_plant = monthly_row.id_plant
       AND monthly_bounds.id_lng_train = monthly_row.id_lng_train
       AND monthly_bounds.last_monthly_month = monthly_row.month
),
monthly_carry_forward AS (
    SELECT
        monthly_row.id_plant,
        monthly_row.id_lng_train,
        month_series.month::date AS month
    FROM monthly_carry_forward_base AS monthly_row
    CROSS JOIN coverage_horizon AS horizon
    CROSS JOIN LATERAL generate_series(
        (monthly_row.last_monthly_month + INTERVAL '1 month')::date,
        horizon.max_month,
        INTERVAL '1 month'
    ) AS month_series(month)
    LEFT JOIN monthly_exact_coverage AS monthly_map
        ON monthly_row.id_plant = monthly_map.id_plant
       AND monthly_row.id_lng_train = monthly_map.id_lng_train
       AND month_series.month::date = monthly_map.month
    WHERE monthly_map.id_plant IS NULL
),
annual_only_fallback_base AS (
    SELECT
        annual_row.id_plant,
        annual_row.id_lng_train,
        annual_row.first_active_month,
        annual_row.last_annual_month,
        annual_row.upload_timestamp_utc
    FROM annual_train_bounds AS annual_row
    LEFT JOIN monthly_train_bounds AS monthly_bounds
        ON annual_row.id_plant = monthly_bounds.id_plant
       AND annual_row.id_lng_train = monthly_bounds.id_lng_train
    WHERE monthly_bounds.id_plant IS NULL
),
annual_only_fallback AS (
    SELECT
        annual_row.id_plant,
        annual_row.id_lng_train,
        month_series.month::date AS month
    FROM annual_only_fallback_base AS annual_row
    CROSS JOIN LATERAL generate_series(
        annual_row.first_active_month,
        annual_row.last_annual_month,
        INTERVAL '1 month'
    ) AS month_series(month)
    LEFT JOIN monthly_exact_coverage AS monthly_map
        ON annual_row.id_plant = monthly_map.id_plant
       AND annual_row.id_lng_train = monthly_map.id_lng_train
       AND month_series.month::date = monthly_map.month
    WHERE monthly_map.id_plant IS NULL
),
annual_only_carry_forward AS (
    SELECT
        annual_row.id_plant,
        annual_row.id_lng_train,
        month_series.month::date AS month
    FROM annual_only_fallback_base AS annual_row
    CROSS JOIN coverage_horizon AS horizon
    CROSS JOIN LATERAL generate_series(
        GREATEST(
            DATE '{WOODMAC_ANNUAL_CARRY_FORWARD_START}',
            (annual_row.last_annual_month + INTERVAL '1 month')::date
        ),
        horizon.max_month,
        INTERVAL '1 month'
    ) AS month_series(month)
    LEFT JOIN monthly_exact_coverage AS monthly_map
        ON annual_row.id_plant = monthly_map.id_plant
       AND annual_row.id_lng_train = monthly_map.id_lng_train
       AND month_series.month::date = monthly_map.month
    WHERE monthly_map.id_plant IS NULL
),
combined_capacity AS (
    SELECT
        id_plant,
        id_lng_train,
        month
    FROM latest_monthly_capacity

    UNION ALL

    SELECT
        id_plant,
        id_lng_train,
        month
    FROM monthly_carry_forward

    UNION ALL

    SELECT
        id_plant,
        id_lng_train,
        month
    FROM annual_only_fallback

    UNION ALL

    SELECT
        id_plant,
        id_lng_train,
        month
    FROM annual_only_carry_forward
)
SELECT
    (SELECT MAX(upload_timestamp_utc) FROM latest_monthly_capacity) AS monthly_upload_timestamp_utc,
    (SELECT MAX(upload_timestamp_utc) FROM annual_train_bounds) AS annual_upload_timestamp_utc,
    NULLIF(
        GREATEST(
            COALESCE(
                (SELECT MAX(upload_timestamp_utc) FROM latest_monthly_capacity),
                TIMESTAMP '1900-01-01'
            ),
            COALESCE(
                (SELECT MAX(upload_timestamp_utc) FROM annual_train_bounds),
                TIMESTAMP '1900-01-01'
            )
        ),
        TIMESTAMP '1900-01-01'
    ) AS upload_timestamp_utc,
    (SELECT COUNT(*) FROM combined_capacity) AS source_rows,
    (SELECT COUNT(*) FROM latest_monthly_capacity) AS monthly_source_rows,
    (SELECT COUNT(*) FROM monthly_carry_forward) AS monthly_carry_forward_train_months,
    (
        SELECT COUNT(*)
        FROM (
            SELECT month FROM annual_only_fallback
            UNION ALL
            SELECT month FROM annual_only_carry_forward
        ) AS annual_proxy_rows
    ) AS annual_fallback_train_months,
    (SELECT COUNT(DISTINCT (id_plant, id_lng_train)) FROM combined_capacity) AS source_trains,
    (SELECT MIN(month) FROM combined_capacity) AS min_month,
    (SELECT MAX(month) FROM combined_capacity) AS max_month
"""

COUNTRY_MAPPING_QUERY = f"""
WITH
{COUNTRY_MAPPING_CTE},
canonical_country AS (
    SELECT country_name, MIN(iso3) AS iso3
    FROM {DB_SCHEMA}.mappings_country
    WHERE country_name IS NOT NULL
      AND iso3 IS NOT NULL
    GROUP BY country_name
)
SELECT mapping.raw_country_key, mapping.country_name, canonical.iso3
FROM country_mapping mapping
LEFT JOIN canonical_country canonical USING (country_name)
ORDER BY country_name, raw_country_key
"""

EA_CAPACITY_QUERY = f"""
WITH latest_snapshot AS (
    SELECT MAX(publication_date) AS publication_date
    FROM {EA_CAPACITY_SOURCE_TABLE}
)
SELECT
    DATE_TRUNC('month', start_date)::date AS month,
    country AS country_name,
    project_name,
    train_name,
    capacity_mtpa,
    status,
    publication_date
FROM {EA_CAPACITY_SOURCE_TABLE}
WHERE publication_date = (SELECT publication_date FROM latest_snapshot)
  AND start_date IS NOT NULL
  AND capacity_mtpa IS NOT NULL
ORDER BY month, country, project_name, train_name
"""

EA_CAPACITY_METADATA_QUERY = f"""
WITH latest_snapshot AS (
    SELECT MAX(publication_date) AS publication_date
    FROM {EA_CAPACITY_SOURCE_TABLE}
)
SELECT
    projects.publication_date,
    MAX(projects.upload_timestamp_utc) AS upload_timestamp_utc,
    COUNT(*) AS source_rows,
    COUNT(*) FILTER (
        WHERE projects.start_date IS NOT NULL
          AND projects.capacity_mtpa IS NOT NULL
    ) AS dated_rows,
    COUNT(DISTINCT projects.country) AS source_countries
FROM {EA_CAPACITY_SOURCE_TABLE} AS projects
WHERE projects.publication_date = (SELECT publication_date FROM latest_snapshot)
GROUP BY projects.publication_date
"""

CAPACITY_SOURCE_WATERMARK_QUERY = f"""
SELECT
    (SELECT MAX(upload_timestamp_utc) FROM {CAPACITY_SOURCE_TABLE})
        AS monthly_upload_timestamp_utc,
    (SELECT MAX(upload_timestamp_utc) FROM {WOODMAC_ANNUAL_OUTPUT_SOURCE_TABLE})
        AS annual_upload_timestamp_utc,
    (SELECT MAX(upload_timestamp_utc) FROM {DB_SCHEMA}.woodmac_lng_plant_summary)
        AS plant_summary_upload_timestamp_utc,
    (SELECT MAX(upload_timestamp_utc) FROM {DB_SCHEMA}.woodmac_lng_plant_train)
        AS train_metadata_upload_timestamp_utc,
    (SELECT MAX(publication_date) FROM {EA_CAPACITY_SOURCE_TABLE})
        AS ea_publication_date,
    (SELECT MAX(upload_timestamp_utc) FROM {EA_CAPACITY_SOURCE_TABLE})
        AS ea_upload_timestamp_utc,
    (
        SELECT MD5(
            COALESCE(
                STRING_AGG(
                    TO_JSONB(mapping_row)::text,
                    '|' ORDER BY TO_JSONB(mapping_row)::text
                ),
                ''
            )
        )
        FROM {DB_SCHEMA}.mappings_country AS mapping_row
    ) AS country_mapping_hash,
    (
        SELECT MD5(
            COALESCE(
                STRING_AGG(
                    CONCAT_WS(
                        '|', provider_name, provider_plant_id,
                        provider_train_id, train_key::text,
                        allocation_share::text
                    ),
                    '|' ORDER BY provider_name, provider_plant_id,
                                 provider_train_id, train_key
                ),
                ''
            )
        )
        FROM {DB_SCHEMA}.fundamentals_terminal_train_provider_links
    ) AS provider_link_hash,
    (
        SELECT MD5(
            COALESCE(
                STRING_AGG(
                    CONCAT_WS(
                        '|', terminals.terminal_key::text,
                        terminals.country_name, terminals.terminal_name,
                        trains.train_key::text, trains.train_label
                    ),
                    '|' ORDER BY terminals.terminal_key, trains.train_key
                ),
                ''
            )
        )
        FROM {DB_SCHEMA}.fundamentals_terminal_registry terminals
        JOIN {DB_SCHEMA}.fundamentals_terminal_train_registry trains
          USING (terminal_key)
    ) AS canonical_registry_hash,
    '{CAPACITY_PROVIDER_MAPPER_VERSION}'::text AS mapper_version
"""

CAPACITY_SOURCE_REFRESH_JOB_TABLE = f"{DB_SCHEMA}.fundamentals_capacity_source_refresh_jobs"
CAPACITY_SOURCE_EVENT_TABLE = f"{DB_SCHEMA}.fundamentals_capacity_source_events"

CAPACITY_SOURCE_REF_FORMAT = "capacity_source_ref_v3"
CAPACITY_SCENARIO_WORKING_FORMAT = "capacity_scenario_working_v3"
CAPACITY_SOURCE_CACHE_MAX_ENTRIES = 1
CAPACITY_DERIVED_CACHE_MAX_ENTRIES = 32
_CAPACITY_SOURCE_CACHE: OrderedDict[str, dict[str, object]] = OrderedDict()
_CAPACITY_SOURCE_CACHE_LOCK = RLock()
_CAPACITY_SOURCE_BUILD_LOCK = RLock()
_CAPACITY_DERIVED_CACHE: OrderedDict[str, dict[str, object]] = OrderedDict()
_CAPACITY_DERIVED_CACHE_LOCK = RLock()
_CAPACITY_VIEW_CACHE: OrderedDict[str, object] = OrderedDict()
_CAPACITY_VIEW_CACHE_LOCK = RLock()
_CAPACITY_VIEW_CACHE_KEY_LOCKS: dict[str, RLock] = {}
_CAPACITY_SCENARIO_CACHE: OrderedDict[str, pd.DataFrame] = OrderedDict()
_CAPACITY_SCENARIO_CACHE_LOCK = RLock()
_CAPACITY_REFRESH_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="capacity-source-refresh",
)


def fetch_country_mapping_df() -> pd.DataFrame:
    with engine.connect() as connection:
        mapping_df = pd.read_sql_query(COUNTRY_MAPPING_QUERY, connection)

    if mapping_df.empty:
        return pd.DataFrame(columns=["raw_country_key", "country_name", "iso3"])

    mapping_df["raw_country_key"] = (
        mapping_df["raw_country_key"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    mapping_df["country_name"] = (
        mapping_df["country_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    mapping_df["iso3"] = (
        mapping_df["iso3"].fillna("").astype(str).str.strip().str.upper()
    )
    mapping_df = mapping_df[
        (mapping_df["raw_country_key"] != "") & (mapping_df["country_name"] != "")
    ].drop_duplicates(subset=["raw_country_key"], keep="first")

    return mapping_df


def fetch_provider_link_mapping_df() -> pd.DataFrame:
    with engine.connect() as connection:
        mapping_df = pd.read_sql_query(
            f"""
            SELECT links.provider_name, links.provider_plant_id,
                   links.provider_train_id, links.train_key,
                   links.allocation_share, trains.terminal_key,
                   terminals.country_name AS canonical_country_name,
                   terminals.terminal_name AS canonical_terminal_name,
                   trains.train_label AS canonical_train_label
            FROM {DB_SCHEMA}.fundamentals_terminal_train_provider_links links
            JOIN {DB_SCHEMA}.fundamentals_terminal_train_registry trains
              USING (train_key)
            JOIN {DB_SCHEMA}.fundamentals_terminal_registry terminals
              USING (terminal_key)
            ORDER BY links.provider_name, links.provider_plant_id,
                     links.provider_train_id, links.train_key
            """,
            connection,
        )
    if not mapping_df.empty:
        mapping_df["train_key"] = mapping_df["train_key"].astype(str)
        mapping_df["terminal_key"] = mapping_df["terminal_key"].astype(str)
        mapping_df["allocation_share"] = pd.to_numeric(
            mapping_df["allocation_share"], errors="coerce"
        )
    return mapping_df


def _standardize_country_names(
    raw_df: pd.DataFrame,
    country_mapping_df: pd.DataFrame | None,
    source_column: str = "country_name",
) -> pd.DataFrame:
    if raw_df.empty or source_column not in raw_df.columns:
        return raw_df

    standardized_df = raw_df.copy()
    standardized_df[source_column] = (
        standardized_df[source_column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    if country_mapping_df is None or country_mapping_df.empty:
        return standardized_df

    standardized_df["__raw_country_key"] = standardized_df[source_column].str.upper()
    mapping_df = country_mapping_df.rename(
        columns={"country_name": "__mapped_country_name"}
    )
    standardized_df = standardized_df.merge(
        mapping_df,
        how="left",
        left_on="__raw_country_key",
        right_on="raw_country_key",
    )
    standardized_df[source_column] = standardized_df["__mapped_country_name"].where(
        standardized_df["__mapped_country_name"].notna(),
        standardized_df[source_column],
    )
    standardized_df = standardized_df.drop(
        columns=["__raw_country_key", "raw_country_key", "__mapped_country_name"],
        errors="ignore",
    )

    return standardized_df


def _apply_train_mapping(
    raw_df: pd.DataFrame,
    train_mapping_df: pd.DataFrame | None,
    provider: str,
    parent_source_field: str,
    parent_source_column: str,
    source_field: str,
    source_column: str,
    capacity_column: str = "capacity_mtpa",
    output_column: str = "train",
    mapping_applied_column: str | None = None,
) -> pd.DataFrame:
    if raw_df.empty or source_column not in raw_df.columns or parent_source_column not in raw_df.columns:
        return raw_df

    standardized_df = raw_df.copy()
    standardized_df[parent_source_column] = (
        standardized_df[parent_source_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    standardized_df[source_column] = (
        standardized_df[source_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    relevant_mapping_df = pd.DataFrame()
    if train_mapping_df is not None and not train_mapping_df.empty:
        relevant_mapping_df = train_mapping_df[
            (train_mapping_df["provider"] == provider)
            & (train_mapping_df["parent_source_field"] == parent_source_field)
            & (train_mapping_df["source_field"] == source_field)
        ][
            [
                "country_name",
                "plant_name",
                "__parent_source_name_key",
                "__source_name_key",
                "train",
                "allocation_share",
            ]
        ].rename(
            columns={
                "train": "__mapped_train",
                "allocation_share": "__mapped_allocation_share",
            }
        )

    standardized_df["__parent_source_name_key"] = standardized_df[parent_source_column].str.upper()
    standardized_df["__source_name_key"] = standardized_df[source_column].str.upper()
    if not relevant_mapping_df.empty:
        standardized_df = standardized_df.merge(
            relevant_mapping_df,
            how="left",
            on=["country_name", "plant_name", "__parent_source_name_key", "__source_name_key"],
        )
    else:
        standardized_df["__mapped_train"] = pd.NA
        standardized_df["__mapped_allocation_share"] = pd.NA

    mapped_train_series = pd.to_numeric(
        standardized_df["__mapped_train"],
        errors="coerce",
    ).astype("Int64")
    inferred_train_series = _infer_direct_train_series(
        standardized_df,
        parent_source_column=parent_source_column,
        source_column=source_column,
        mapped_train_series=mapped_train_series,
    )
    mapped_mask = mapped_train_series.notna()
    standardized_df[output_column] = mapped_train_series.where(mapped_mask, inferred_train_series)
    standardized_df["allocation_share"] = pd.to_numeric(
        standardized_df["__mapped_allocation_share"],
        errors="coerce",
    ).fillna(1.0)
    if capacity_column in standardized_df.columns:
        standardized_df[capacity_column] = (
            pd.to_numeric(standardized_df[capacity_column], errors="coerce").fillna(0.0)
            * standardized_df["allocation_share"]
        )
    if mapping_applied_column:
        standardized_df[mapping_applied_column] = mapped_mask.astype(bool)
    standardized_df = standardized_df.drop(
        columns=[
            "__parent_source_name_key",
            "__source_name_key",
            "__mapped_train",
            "__mapped_allocation_share",
        ],
        errors="ignore",
    )

    return standardized_df


def _canonical_train_number(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric_value) and math.isfinite(float(numeric_value)):
        integer_value = int(float(numeric_value))
        if integer_value > 0 and float(numeric_value) == integer_value:
            return integer_value

    label = str(value).strip()
    match = re.search(r"\btrain\s+(\d+)\b", label, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    numeric_label = re.fullmatch(r"(\d+)(?:\s*\([^)]*\))?", label)
    return int(numeric_label.group(1)) if numeric_label else None


def _apply_provider_link_mapping(
    raw_df: pd.DataFrame,
    provider_links_df: pd.DataFrame,
    *,
    provider_name: str,
    provider_plant_id_column: str,
    provider_train_id_column: str,
    parent_source_column: str,
    source_column: str,
    capacity_column: str = "capacity_mtpa",
) -> pd.DataFrame:
    """Expand provider sources to canonical trains and apply allocation shares."""
    if raw_df.empty:
        return raw_df.copy()
    mapped = raw_df.copy()
    mapped["__provider_plant_id"] = (
        mapped[provider_plant_id_column].fillna("").astype(str).str.strip()
    )
    mapped["__provider_train_id"] = (
        mapped[provider_train_id_column].fillna("").astype(str).str.strip()
    )
    relevant = provider_links_df[
        provider_links_df["provider_name"].astype(str).str.casefold().eq(
            provider_name.casefold()
        )
    ].copy()
    if relevant.empty:
        for column_name in [
            "train_key",
            "terminal_key",
            "canonical_country_name",
            "canonical_terminal_name",
            "canonical_train_label",
        ]:
            mapped[column_name] = None
        mapped["allocation_share"] = 1.0
    else:
        relevant = relevant.rename(
            columns={
                "provider_plant_id": "__provider_plant_id",
                "provider_train_id": "__provider_train_id",
            }
        )
        mapped = mapped.merge(
            relevant,
            on=["__provider_plant_id", "__provider_train_id"],
            how="left",
            validate="many_to_many",
        )
        mapped["allocation_share"] = pd.to_numeric(
            mapped["allocation_share"], errors="coerce"
        ).fillna(1.0)

    mapped["source_capacity_mtpa"] = pd.to_numeric(
        mapped[capacity_column], errors="coerce"
    ).fillna(0.0)
    mapped[capacity_column] = (
        mapped["source_capacity_mtpa"] * mapped["allocation_share"]
    )
    resolved_mask = mapped["train_key"].notna()
    if "country_name" in mapped.columns:
        mapped["country_name"] = mapped["canonical_country_name"].where(
            resolved_mask
            & mapped["canonical_country_name"].fillna("").astype(str).str.strip().ne(""),
            mapped["country_name"],
        )
    if "plant_name" in mapped.columns:
        provider_plant_name = mapped["plant_name"].fillna("").astype(str).str.strip()
        canonical_plant_name = (
            mapped["canonical_terminal_name"].fillna("").astype(str).str.strip()
        )
        canonical_plant_mask = resolved_mask & canonical_plant_name.ne("")
        mapped["plant_name"] = canonical_plant_name.where(
            canonical_plant_mask,
            provider_plant_name,
        )
    canonical_numbers = mapped["canonical_train_label"].map(
        _canonical_train_number
    )
    inferred_numbers = pd.Series(
        [
            _infer_contextual_train_number(source_value, parent_value)
            for source_value, parent_value in zip(
                mapped[source_column], mapped[parent_source_column]
            )
        ],
        index=mapped.index,
        dtype="Int64",
    )
    mapped["train"] = pd.to_numeric(
        canonical_numbers.where(canonical_numbers.notna(), inferred_numbers),
        errors="coerce",
    ).astype("Int64")
    mapped["train_mapping_applied"] = mapped["train_key"].notna()
    mapped["__provider_plant_id"] = mapped["__provider_plant_id"].astype(str)
    mapped["__provider_train_id"] = mapped["__provider_train_id"].astype(str)
    return mapped


DIRECT_TRAIN_PATTERN = re.compile(
    r"^\s*train\s+(\d+)(?:\s*(?:bolt[\s-]*on))?(?:\s*\([^)]*\))?\s*$",
    flags=re.IGNORECASE,
)

PARENT_TRAIN_PATTERN = re.compile(
    r"\btrain\s+(\d+)\b",
    flags=re.IGNORECASE,
)


def _infer_direct_train_number(value: object) -> int | None:
    if pd.isna(value):
        return None

    text_value = str(value).strip()
    if not text_value:
        return None

    match = DIRECT_TRAIN_PATTERN.match(text_value)
    if not match:
        return None

    try:
        return int(match.group(1))
    except ValueError:
        return None


def _infer_contextual_train_number(source_value: object, parent_value: object) -> int | None:
    direct_train = _infer_direct_train_number(source_value)

    if pd.isna(parent_value):
        return direct_train

    parent_text = str(parent_value).strip()
    if not parent_text:
        return direct_train

    parent_match = PARENT_TRAIN_PATTERN.search(parent_text)
    if not parent_match:
        return direct_train

    try:
        parent_train = int(parent_match.group(1))
    except ValueError:
        return direct_train

    if direct_train is None:
        return parent_train
    if direct_train == parent_train:
        return direct_train
    if direct_train == 1 and parent_train > 1:
        return parent_train

    return direct_train


def _infer_direct_train_series(
    standardized_df: pd.DataFrame,
    parent_source_column: str,
    source_column: str,
    mapped_train_series: pd.Series,
) -> pd.Series:
    inferred_series = pd.Series(pd.NA, index=standardized_df.index, dtype="Int64")
    if standardized_df.empty:
        return inferred_series

    identity_columns = ["country_name", "plant_name", parent_source_column, source_column]
    identity_df = standardized_df[identity_columns].copy()
    identity_df["__row_id"] = range(len(standardized_df))
    identity_df["__direct_train"] = [
        _infer_contextual_train_number(source_value, parent_value)
        for source_value, parent_value in zip(
            identity_df[source_column],
            identity_df[parent_source_column],
        )
    ]

    candidate_df = identity_df.loc[mapped_train_series.isna()].drop_duplicates(
        subset=identity_columns,
        keep="first",
    )
    candidate_df = candidate_df[candidate_df["__direct_train"].notna()].copy()
    if candidate_df.empty:
        return inferred_series

    candidate_parent_counts = (
        candidate_df.groupby(["country_name", "plant_name", "__direct_train"])[parent_source_column]
        .nunique()
        .rename("__candidate_parent_count")
        .reset_index()
    )

    reserved_df = standardized_df.loc[mapped_train_series.notna(), identity_columns].copy()
    reserved_df["__mapped_train"] = mapped_train_series[mapped_train_series.notna()].astype("Int64")
    reserved_df = reserved_df.drop_duplicates(
        subset=identity_columns + ["__mapped_train"],
        keep="first",
    )
    if reserved_df.empty:
        reserved_counts = pd.DataFrame(
            columns=["country_name", "plant_name", "__direct_train", "__reserved_count"]
        )
    else:
        reserved_counts = (
            reserved_df.groupby(["country_name", "plant_name", "__mapped_train"])
            .size()
            .rename("__reserved_count")
            .reset_index()
            .rename(columns={"__mapped_train": "__direct_train"})
        )

    candidate_df = candidate_df.merge(
        candidate_parent_counts,
        how="left",
        on=["country_name", "plant_name", "__direct_train"],
    )
    candidate_df = candidate_df.merge(
        reserved_counts,
        how="left",
        on=["country_name", "plant_name", "__direct_train"],
    )
    candidate_df["__reserved_count"] = (
        pd.to_numeric(candidate_df["__reserved_count"], errors="coerce").fillna(0).astype(int)
    )
    candidate_df["__candidate_parent_count"] = (
        pd.to_numeric(candidate_df["__candidate_parent_count"], errors="coerce").fillna(0).astype(int)
    )
    candidate_df["__can_infer"] = (
        candidate_df["__candidate_parent_count"].eq(1)
        & candidate_df["__reserved_count"].eq(0)
    )

    allowed_df = candidate_df[candidate_df["__can_infer"]][
        identity_columns + ["__direct_train"]
    ].drop_duplicates(subset=identity_columns, keep="first")
    if allowed_df.empty:
        return inferred_series

    assigned_df = identity_df.merge(
        allowed_df,
        how="left",
        on=identity_columns,
        suffixes=("", "_allowed"),
    ).sort_values("__row_id", kind="mergesort")

    return pd.to_numeric(assigned_df["__direct_train_allowed"], errors="coerce").astype("Int64")


def _format_train_label(value: object) -> str:
    if pd.isna(value) or value in (None, ""):
        return ""
    try:
        return str(int(float(value)))
    except Exception:
        return str(value).strip()


def _coerce_positive_train_label(
    value: object,
    field_name: str = "Train",
) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and value.strip() == "":
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    numeric_value = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(numeric_value):
        raise ValueError(f"{field_name} must be blank or a positive whole number.")

    numeric_value = float(numeric_value)
    if numeric_value <= 0 or not numeric_value.is_integer():
        raise ValueError(f"{field_name} must be blank or a positive whole number.")

    return str(int(numeric_value))


def fetch_woodmac_capacity_raw_data(
    country_mapping_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    with engine.connect() as connection:
        raw_df = pd.read_sql_query(WOODMAC_CAPACITY_QUERY, connection)

    if raw_df.empty:
        return pd.DataFrame(columns=["month", "country_name", "total_mmtpa"])

    raw_df["month"] = pd.to_datetime(raw_df["month"])
    raw_df["country_name"] = (
        raw_df["country_name"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )
    raw_df["total_mmtpa"] = pd.to_numeric(raw_df["total_mmtpa"], errors="coerce").fillna(0.0)

    return _standardize_country_names(raw_df, country_mapping_df)


def fetch_woodmac_capacity_metadata() -> dict[str, str | int | None]:
    with engine.connect() as connection:
        metadata_df = pd.read_sql_query(WOODMAC_CAPACITY_METADATA_QUERY, connection)

    if metadata_df.empty:
        return {}

    row = metadata_df.iloc[0]
    return {
        "monthly_upload_timestamp_utc": _serialize_timestamp(row.get("monthly_upload_timestamp_utc")),
        "annual_upload_timestamp_utc": _serialize_timestamp(row.get("annual_upload_timestamp_utc")),
        "upload_timestamp_utc": _serialize_timestamp(row.get("upload_timestamp_utc")),
        "source_rows": int(row["source_rows"]) if pd.notna(row.get("source_rows")) else 0,
        "monthly_source_rows": int(row["monthly_source_rows"]) if pd.notna(row.get("monthly_source_rows")) else 0,
        "monthly_carry_forward_train_months": int(row["monthly_carry_forward_train_months"]) if pd.notna(row.get("monthly_carry_forward_train_months")) else 0,
        "annual_fallback_train_months": int(row["annual_fallback_train_months"]) if pd.notna(row.get("annual_fallback_train_months")) else 0,
        "source_trains": int(row["source_trains"]) if pd.notna(row.get("source_trains")) else 0,
        "min_month": _serialize_timestamp(row.get("min_month")),
        "max_month": _serialize_timestamp(row.get("max_month")),
    }


def fetch_woodmac_train_capacity_raw_data(
    country_mapping_df: pd.DataFrame | None = None,
    provider_links_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    with engine.connect() as connection:
        array_row = connection.execute(
            text(WOODMAC_TRAIN_CAPACITY_ARRAY_QUERY)
        ).mappings().first()

    raw_df = pd.DataFrame(
        {
            column_name: (array_row.get(column_name) or []) if array_row else []
            for column_name in [
                "month",
                "country_name",
                "plant_name",
                "lng_train_name",
                "lng_train_name_short",
                "woodmac_fid_date",
                "id_plant",
                "id_lng_train",
                "capacity_mtpa",
                "capacity_source",
            ]
        }
    )

    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "country_name",
                "raw_plant_name",
                "plant_name",
                "raw_train_name",
                "raw_train_display_name",
                "train",
                "allocation_share",
                "train_mapping_applied",
                "lng_train_name",
                "lng_train_name_short",
                "woodmac_fid_date",
                "id_plant",
                "id_lng_train",
                "capacity_mtpa",
                "terminal_key",
                "train_key",
                "canonical_terminal_name",
                "canonical_train_label",
            ]
        )

    raw_df["month"] = pd.to_datetime(raw_df["month"])
    for column_name in ["country_name", "plant_name", "lng_train_name", "lng_train_name_short"]:
        raw_df[column_name] = (
            raw_df[column_name]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    raw_df["capacity_mtpa"] = pd.to_numeric(raw_df["capacity_mtpa"], errors="coerce").fillna(0.0)
    raw_df["id_plant"] = pd.to_numeric(raw_df["id_plant"], errors="coerce")
    raw_df["id_lng_train"] = pd.to_numeric(raw_df["id_lng_train"], errors="coerce")
    if "lng_train_name" not in raw_df.columns:
        raw_df["lng_train_name"] = raw_df.get("lng_train_name_short", "Unknown")
    if "woodmac_fid_date" not in raw_df.columns:
        raw_df["woodmac_fid_date"] = None
    raw_df["woodmac_fid_date"] = pd.to_datetime(
        raw_df["woodmac_fid_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    raw_df["woodmac_fid_date"] = raw_df["woodmac_fid_date"].where(
        raw_df["woodmac_fid_date"].notna(),
        "",
    )
    raw_df["raw_plant_name"] = raw_df["plant_name"]
    raw_df["raw_train_name"] = raw_df["lng_train_name_short"]
    raw_df["raw_train_display_name"] = raw_df["lng_train_name"]

    raw_df = _standardize_country_names(raw_df, country_mapping_df)
    raw_df["provider_plant_id"] = raw_df["id_plant"].astype("Int64").astype(str)
    raw_df["provider_train_id"] = raw_df["id_lng_train"].astype("Int64").astype(str)
    raw_df = _apply_provider_link_mapping(
        raw_df,
        fetch_provider_link_mapping_df()
        if provider_links_df is None
        else provider_links_df,
        provider_name="woodmac",
        provider_plant_id_column="provider_plant_id",
        provider_train_id_column="provider_train_id",
        parent_source_column="raw_plant_name",
        source_column="raw_train_name",
    )

    return raw_df


def fetch_ea_capacity_raw_data(
    country_mapping_df: pd.DataFrame | None = None,
    provider_links_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    with engine.connect() as connection:
        raw_df = pd.read_sql_query(EA_CAPACITY_QUERY, connection)

    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "country_name",
                "project_name",
                "plant_name",
                "train_name",
                "train",
                "allocation_share",
                "train_mapping_applied",
                "capacity_mtpa",
                "status",
                "publication_date",
            ]
        )

    raw_df["month"] = pd.to_datetime(raw_df["month"])
    raw_df["project_name"] = (
        raw_df["project_name"]
        .fillna("Unknown Project")
        .astype(str)
        .str.strip()
        .replace("", "Unknown Project")
    )
    raw_df["train_name"] = (
        raw_df["train_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    raw_df["status"] = (
        raw_df["status"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )
    raw_df["capacity_mtpa"] = pd.to_numeric(raw_df["capacity_mtpa"], errors="coerce").fillna(0.0)
    raw_df["publication_date"] = pd.to_datetime(raw_df["publication_date"], errors="coerce")

    raw_df = _standardize_country_names(raw_df, country_mapping_df)
    raw_df["plant_name"] = raw_df["project_name"]
    if "iso3" not in raw_df.columns or raw_df["iso3"].fillna("").eq("").any():
        unresolved_countries = sorted(
            raw_df.loc[
                raw_df.get("iso3", pd.Series("", index=raw_df.index)).fillna("").eq(""),
                "country_name",
            ].astype(str).unique()
        )
        raise ValueError(
            "Energy Aspects country mapping is missing ISO3 for: "
            + ", ".join(unresolved_countries[:10])
        )
    identities = raw_df.apply(
        lambda row: build_energy_aspects_provider_identity(
            row["iso3"], row["project_name"], row["train_name"]
        ),
        axis=1,
    )
    raw_df["provider_plant_id"] = identities.map(lambda value: value[0])
    raw_df["provider_train_id"] = identities.map(lambda value: value[1])
    raw_df = _apply_provider_link_mapping(
        raw_df,
        fetch_provider_link_mapping_df()
        if provider_links_df is None
        else provider_links_df,
        provider_name="energy_aspects",
        provider_plant_id_column="provider_plant_id",
        provider_train_id_column="provider_train_id",
        parent_source_column="project_name",
        source_column="train_name",
    )

    return raw_df


def fetch_ea_capacity_metadata() -> dict[str, str | int | None]:
    with engine.connect() as connection:
        metadata_df = pd.read_sql_query(EA_CAPACITY_METADATA_QUERY, connection)

    if metadata_df.empty:
        return {}

    row = metadata_df.iloc[0]
    return {
        "publication_date": _serialize_timestamp(row.get("publication_date")),
        "upload_timestamp_utc": _serialize_timestamp(row.get("upload_timestamp_utc")),
        "source_rows": int(row["source_rows"]) if pd.notna(row.get("source_rows")) else 0,
        "dated_rows": int(row["dated_rows"]) if pd.notna(row.get("dated_rows")) else 0,
        "source_countries": int(row["source_countries"]) if pd.notna(row.get("source_countries")) else 0,
    }


def _build_ea_capacity_schedule(
    raw_df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    empty_df = pd.DataFrame(columns=["month", "country_name", "total_mmtpa"])
    if raw_df.empty:
        return empty_df

    schedule_df = raw_df.copy()
    schedule_df["month"] = pd.to_datetime(schedule_df["month"]).dt.to_period("M").dt.to_timestamp()
    schedule_df["status_key"] = (
        schedule_df["status"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    schedule_df = schedule_df[
        ~schedule_df["status_key"].isin(EA_TOP_PANEL_EXCLUDED_STATUSES)
    ].copy()
    if schedule_df.empty:
        return empty_df

    start_month = _normalize_month_date(start_date) or schedule_df["month"].min()
    end_month = _normalize_month_date(end_date) or schedule_df["month"].max()

    if start_month is None or end_month is None or start_month > end_month:
        return empty_df

    schedule_df = schedule_df[schedule_df["month"] <= end_month].copy()
    if schedule_df.empty:
        return empty_df

    expanded_frames = []
    for row in schedule_df.itertuples(index=False):
        active_start = max(pd.Timestamp(row.month), start_month)
        if active_start > end_month:
            continue

        expanded_months = pd.date_range(active_start, end_month, freq="MS")
        if expanded_months.empty:
            continue

        expanded_frames.append(
            pd.DataFrame(
                {
                    "month": expanded_months,
                    "country_name": row.country_name,
                    "total_mmtpa": float(row.capacity_mtpa),
                }
            )
        )

    if not expanded_frames:
        return empty_df

    expanded_df = pd.concat(expanded_frames, ignore_index=True)
    expanded_df = (
        expanded_df.groupby(["month", "country_name"], as_index=False)["total_mmtpa"]
        .sum()
        .sort_values(["month", "country_name"])
        .reset_index(drop=True)
    )

    return expanded_df


def _serialize_timestamp(value) -> str | None:
    return serialize_timestamp(value)


CAPACITY_SOURCE_TIMESTAMP_COLUMNS = {
    "monthly_upload_timestamp_utc",
    "annual_upload_timestamp_utc",
    "plant_summary_upload_timestamp_utc",
    "train_metadata_upload_timestamp_utc",
    "ea_publication_date",
    "ea_upload_timestamp_utc",
}


def _normalize_capacity_watermark_values(
    row,
    column_names,
) -> dict[str, str | None]:
    return {
        column_name: (
            _serialize_timestamp(row.get(column_name))
            if column_name in CAPACITY_SOURCE_TIMESTAMP_COLUMNS
            else (
                str(row.get(column_name))
                if pd.notna(row.get(column_name))
                else None
            )
        )
        for column_name in column_names
    }


def _fetch_capacity_source_watermarks() -> dict[str, str | None]:
    with engine.connect() as connection:
        watermark_df = pd.read_sql_query(
            CAPACITY_SOURCE_WATERMARK_QUERY,
            connection,
        )

    if watermark_df.empty:
        return {}

    row = watermark_df.iloc[0]
    return _normalize_capacity_watermark_values(row, watermark_df.columns)


def _fetch_capacity_source_state() -> dict[str, object]:
    watermarks = _fetch_capacity_source_watermarks()
    if not watermarks:
        return {}
    source_key = _build_capacity_source_snapshot_key(watermarks)
    state = fetch_capacity_source_state(engine, source_key)
    return {
        **state,
        "source_key": source_key,
        "display_source_key": str(state.get("display_source_key") or ""),
        "last_success_at": _serialize_timestamp(state.get("display_finished_at")),
        "watermarks": watermarks,
    }


def _build_capacity_source_snapshot_key(
    watermarks: dict[str, str | None],
) -> str:
    serialized_watermarks = json.dumps(
        watermarks,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized_watermarks.encode("utf-8")).hexdigest()


def _build_woodmac_capacity_from_train_data(
    train_capacity_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["month", "country_name", "total_mmtpa"]
    if train_capacity_df.empty:
        return pd.DataFrame(columns=columns)

    required_columns = {"month", "country_name", "capacity_mtpa"}
    if not required_columns.issubset(train_capacity_df.columns):
        return pd.DataFrame(columns=columns)

    capacity_df = train_capacity_df.copy()
    capacity_df["month"] = pd.to_datetime(capacity_df["month"], errors="coerce")
    capacity_df["capacity_mtpa"] = pd.to_numeric(
        capacity_df["capacity_mtpa"],
        errors="coerce",
    ).fillna(0.0)
    capacity_df = (
        capacity_df.dropna(subset=["month"])
        .groupby(["month", "country_name"], as_index=False)["capacity_mtpa"]
        .sum()
        .rename(columns={"capacity_mtpa": "total_mmtpa"})
    )
    capacity_df = capacity_df[capacity_df["total_mmtpa"].ne(0)].copy()
    return capacity_df.sort_values(["month", "country_name"]).reset_index(drop=True)[columns]


def _deserialize_woodmac_capacity_store(
    payload: dict | str | None,
    start_date: str | None = None,
    end_date: str | None = None,
    selected_countries: list[str] | None = None,
) -> pd.DataFrame:
    payload = _resolve_capacity_source_ref(payload, "woodmac_store")
    if isinstance(payload, pd.DataFrame) and payload.attrs.get(
        "capacity_data_kind"
    ) == "source_events":
        return capacity_events_to_monthly_country(
            payload,
            "woodmac",
            start_month=start_date,
            end_month=end_date,
            countries=selected_countries,
        )
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    return pd.DataFrame(columns=["month", "country_name", "total_mmtpa"])


def _deserialize_ea_capacity_store(
    payload: dict | str | None,
) -> pd.DataFrame:
    resolved = _resolve_capacity_source_ref(payload, "ea_store")
    return resolved.copy() if isinstance(resolved, pd.DataFrame) else pd.DataFrame()


def _deserialize_train_capacity_store(
    payload: dict | str | None,
) -> pd.DataFrame:
    payload = _resolve_capacity_source_ref(payload, "train_store")
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    return pd.DataFrame()


def _get_train_capacity_snapshot_key(payload: dict | str | None) -> str:
    if _is_capacity_source_ref(payload):
        return str(payload.get("snapshot_key") or "empty")
    if isinstance(payload, pd.DataFrame):
        if payload.empty:
            return "empty"
        digest = hashlib.sha256()
        digest.update("|".join(map(str, payload.columns)).encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(payload.astype(str), index=True).values.tobytes()
        )
        return digest.hexdigest()
    if isinstance(payload, dict) and payload.get("snapshot_key"):
        return str(payload["snapshot_key"])
    if not payload:
        return "empty"
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()


def _get_cached_capacity_source_snapshot(
    snapshot_key: str,
) -> dict[str, object] | None:
    with _CAPACITY_SOURCE_CACHE_LOCK:
        snapshot = _CAPACITY_SOURCE_CACHE.get(snapshot_key)
        if snapshot is None:
            return None
        _CAPACITY_SOURCE_CACHE.move_to_end(snapshot_key)
        return snapshot


def _cache_capacity_source_snapshot(
    snapshot_key: str,
    snapshot: dict[str, object],
) -> None:
    with _CAPACITY_SOURCE_CACHE_LOCK:
        _CAPACITY_SOURCE_CACHE[snapshot_key] = snapshot
        _CAPACITY_SOURCE_CACHE.move_to_end(snapshot_key)
        while len(_CAPACITY_SOURCE_CACHE) > CAPACITY_SOURCE_CACHE_MAX_ENTRIES:
            _CAPACITY_SOURCE_CACHE.popitem(last=False)


def _build_capacity_source_ref(
    snapshot: dict[str, object],
    component: str,
) -> dict[str, object]:
    source_key = str(snapshot.get("snapshot_key") or "")
    if not source_key:
        return None
    source_ref = {
        "format": CAPACITY_SOURCE_REF_FORMAT,
        "source_key": source_key,
        "snapshot_key": source_key,
        "component": component,
    }
    if snapshot.get("last_success_at"):
        source_ref["last_success_at"] = snapshot["last_success_at"]
    return source_ref


def _is_capacity_source_ref(payload: object) -> bool:
    return isinstance(payload, dict) and payload.get("format") == CAPACITY_SOURCE_REF_FORMAT


def _capacity_source_revision_is_coherent(
    revision,
    woodmac_ref,
    train_ref,
    ea_ref,
    metadata,
) -> bool:
    if not (
        isinstance(revision, dict)
        and revision.get("format")
        == "capacity_source_render_revision_v1"
        and revision.get("source_key")
    ):
        return False
    expected_key = str(revision["source_key"])
    expected_components = (
        (woodmac_ref, "woodmac_store"),
        (train_ref, "train_store"),
        (ea_ref, "ea_store"),
    )
    if any(
        not _is_capacity_source_ref(reference)
        or str(reference.get("component") or "") != component
        or str(reference.get("source_key") or "") != expected_key
        or str(reference.get("snapshot_key") or "") != expected_key
        for reference, component in expected_components
    ):
        return False
    return (
        isinstance(metadata, dict)
        and str(metadata.get("snapshot_key") or "") == expected_key
    )


def _capacity_scenario_revision_is_coherent(
    revision,
    selected_scenario_id,
    working_store,
    dirty_store,
    refresh_revision,
    scenario_options,
) -> bool:
    if not (
        isinstance(revision, dict)
        and revision.get("format")
        == "capacity_scenario_render_revision_v1"
        and revision.get("source_key")
    ):
        return False
    selected_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    dirty_payload = dirty_store if isinstance(dirty_store, dict) else {}
    if revision.get("empty") is True:
        return (
            pd.isna(selected_value)
            and working_store is None
            and revision.get("scenario_id") is None
            and bool(revision.get("dirty"))
            == bool(dirty_payload.get("dirty"))
            and str(revision.get("dirty_source") or "")
            == str(dirty_payload.get("source") or "")
            and str(revision.get("dirty_updated_at") or "")
            == str(dirty_payload.get("updated_at") or "")
            and str(revision.get("refresh_revision") or "")
            == str(refresh_revision or "")
        )
    if not (
        isinstance(working_store, dict)
        and working_store.get("format")
        == CAPACITY_SCENARIO_WORKING_FORMAT
    ):
        return False
    working_value = pd.to_numeric(
        working_store.get("scenario_id"),
        errors="coerce",
    )
    if (
        pd.isna(selected_value)
        or pd.isna(working_value)
        or int(selected_value) != int(working_value)
        or str(revision.get("scenario_id")) != str(int(selected_value))
    ):
        return False
    base_revision = str(working_store.get("base_revision") or "")
    if (
        not base_revision
        or str(revision.get("base_revision") or "") != base_revision
    ):
        return False
    option = _get_capacity_scenario_option_map(scenario_options).get(
        int(selected_value),
        {},
    )
    option_revision = str(
        option.get("current_snapshot_timestamp_utc") or ""
    )
    return (
        option_revision == base_revision
        and str(revision.get("working_mode") or "")
        == str(working_store.get("mode") or "")
        and bool(revision.get("dirty"))
        == bool(dirty_payload.get("dirty"))
        and str(revision.get("dirty_source") or "")
        == str(dirty_payload.get("source") or "")
        and str(revision.get("dirty_updated_at") or "")
        == str(dirty_payload.get("updated_at") or "")
        and str(revision.get("refresh_revision") or "")
        == str(refresh_revision or "")
        and str(revision.get("option_name") or "")
        == str(option.get("scenario_name") or "")
        and str(revision.get("option_revision") or "")
        == option_revision
    )


def _capacity_scenario_source_refs_are_coherent(
    scenario_revision,
    woodmac_ref,
    train_ref,
    ea_ref,
    metadata,
) -> bool:
    if not isinstance(scenario_revision, dict):
        return False
    return _capacity_source_revision_is_coherent(
        {
            "format": "capacity_source_render_revision_v1",
            "source_key": scenario_revision.get("source_key"),
        },
        woodmac_ref,
        train_ref,
        ea_ref,
        metadata,
    )


def _capacity_events_to_woodmac_change_log(events: pd.DataFrame) -> pd.DataFrame:
    source = events[
        events["provider_name"].astype(str).str.casefold().eq("woodmac")
    ].copy()
    if source.empty:
        return _build_train_change_log(pd.DataFrame(), None, "rest_of_world", None, None)
    source["Delta MTPA"] = (
        pd.to_numeric(source["source_capacity_change_mtpa"], errors="coerce")
        * pd.to_numeric(source["allocation_share"], errors="coerce")
    ).round(6)
    source["Abs Delta"] = source["Delta MTPA"].abs()
    source["Effective Date"] = pd.to_datetime(
        source["effective_month"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    source["Country"] = source["country_name"]
    source["Plant"] = source["plant_name"]
    source["Canonical Train Label"] = source["train_label"].fillna("")
    source["Train"] = source["Canonical Train Label"].map(_canonical_train_number)
    source["Provider Plant ID"] = source["provider_plant_id"]
    source["Provider Train ID"] = source["provider_train_id"]
    source["Provider Sources"] = source.apply(
        lambda row: [
            {
                "provider_name": "woodmac",
                "provider_plant_id": row["provider_plant_id"],
                "provider_train_id": row["provider_train_id"],
                "allocation_share": float(row["allocation_share"]),
            }
        ],
        axis=1,
    )
    source["series_key"] = source.apply(
        lambda row: (
            f"CANONICAL|{row['train_key']}"
            if pd.notna(row.get("train_key")) and str(row.get("train_key")).strip()
            else "PROVIDER|woodmac|"
            f"{row['provider_plant_id']}|{row['provider_train_id']}"
        ),
        axis=1,
    )
    source["Source Field"] = "plant_name"
    source["Source Name"] = source["provider_plant_name"]
    source["Parent Source Field"] = "plant_name"
    source["Parent Source Name"] = source["provider_plant_name"]
    source["Train Source Field"] = "lng_train_name_short"
    source["Train Source Name"] = source["provider_train_name"].fillna("")
    source["Train Display Source Name"] = source["provider_train_name"].fillna("")
    source["Woodmac FID Date"] = pd.to_datetime(
        source["woodmac_fid_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d").fillna("")
    source["Mapping Applied"] = source["train_key"].notna()
    source["Train Mapping Applied"] = source["train_key"].notna()
    return source[
        [
            "series_key", "terminal_key", "train_key", "Canonical Train Label",
            "Provider Plant ID", "Provider Train ID", "Provider Sources",
            "Effective Date", "Country", "Plant", "Train", "Delta MTPA",
            "Abs Delta", "Source Field", "Source Name", "Parent Source Field",
            "Parent Source Name", "Train Source Field", "Train Source Name",
            "Train Display Source Name", "Woodmac FID Date", "Mapping Applied",
            "Train Mapping Applied",
        ]
    ].reset_index(drop=True)


def _capacity_events_to_ea_change_log(events: pd.DataFrame) -> pd.DataFrame:
    source = events[
        events["provider_name"].astype(str).str.casefold().eq("energy_aspects")
    ].copy()
    if source.empty:
        return _build_ea_change_log(pd.DataFrame(), None, "rest_of_world", None, None)
    source["EA Net Delta (MTPA)"] = (
        pd.to_numeric(source["source_capacity_change_mtpa"], errors="coerce")
        * pd.to_numeric(source["allocation_share"], errors="coerce")
    ).round(6)
    source["EA Adds (MTPA)"] = source["EA Net Delta (MTPA)"].clip(lower=0)
    source["EA Reductions (MTPA)"] = source["EA Net Delta (MTPA)"].clip(upper=0)
    source["EA Abs Delta"] = source["EA Net Delta (MTPA)"].abs()
    source["Effective Date"] = pd.to_datetime(
        source["effective_month"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    source["month"] = pd.to_datetime(source["effective_month"], errors="coerce")
    source["Country"] = source["country_name"]
    source["Plant"] = source["plant_name"]
    source["Canonical Train Label"] = source["train_label"].fillna("")
    source["Train"] = source["Canonical Train Label"].map(_canonical_train_number)
    source["Provider Plant ID"] = source["provider_plant_id"]
    source["Provider Train ID"] = source["provider_train_id"]
    source["Provider Sources"] = source.apply(
        lambda row: [
            {
                "provider_name": "energy_aspects",
                "provider_plant_id": row["provider_plant_id"],
                "provider_train_id": row["provider_train_id"],
                "allocation_share": float(row["allocation_share"]),
            }
        ],
        axis=1,
    )
    source["project_name"] = source["provider_plant_name"]
    source["train_name"] = source["provider_train_name"].fillna("")
    source["status"] = source["provider_status"].fillna("Unknown")
    source["capacity_mtpa"] = source["EA Net Delta (MTPA)"]
    source["Source Field"] = "project_name"
    source["Source Name"] = source["provider_plant_name"]
    source["Parent Source Field"] = "project_name"
    source["Parent Source Name"] = source["provider_plant_name"]
    source["Train Source Field"] = "train_name"
    source["Train Source Name"] = source["provider_train_name"].fillna("")
    source["Mapping Applied"] = source["train_key"].notna()
    source["Train Mapping Applied"] = source["train_key"].notna()
    return source[
        [
            "terminal_key", "train_key", "Canonical Train Label",
            "Provider Plant ID", "Provider Train ID", "Provider Sources",
            "Effective Date", "Country", "Plant", "Train", "project_name",
            "train_name", "status", "EA Adds (MTPA)", "EA Reductions (MTPA)",
            "EA Net Delta (MTPA)", "EA Abs Delta", "Source Field", "Source Name",
            "Parent Source Field", "Parent Source Name", "Train Source Field",
            "Train Source Name", "Mapping Applied", "Train Mapping Applied",
            "month", "country_name", "capacity_mtpa",
        ]
    ].reset_index(drop=True)


def _build_capacity_source_components(events: pd.DataFrame) -> dict[str, object]:
    events = events.copy()
    events.attrs["capacity_data_kind"] = "source_events"
    woodmac_change_df = _capacity_events_to_woodmac_change_log(events)
    woodmac_scope = events[
        events["provider_name"].astype(str).str.casefold().eq("woodmac")
    ].copy()
    woodmac_scope["month"] = pd.to_datetime(
        woodmac_scope["effective_month"], errors="coerce"
    )
    woodmac_scope["total_mmtpa"] = (
        pd.to_numeric(
            woodmac_scope["source_capacity_change_mtpa"], errors="coerce"
        )
        * pd.to_numeric(woodmac_scope["allocation_share"], errors="coerce")
    ).abs()
    woodmac_change_df.attrs["capacity_country_scope_df"] = woodmac_scope[
        ["month", "country_name", "total_mmtpa"]
    ]
    return {
        "woodmac_store": events,
        "train_store": woodmac_change_df,
        "ea_store": _capacity_events_to_ea_change_log(events),
    }


def _fetch_relational_capacity_source_snapshot(
    source_key: str,
    watermarks: dict[str, str | None] | None = None,
    source_state: dict[str, object] | None = None,
) -> dict[str, object] | None:
    components = _get_cached_capacity_source_snapshot(source_key)
    if components is None:
        with _CAPACITY_SOURCE_BUILD_LOCK:
            components = _get_cached_capacity_source_snapshot(source_key)
            if components is None:
                events = fetch_capacity_source_events(engine, source_key)
                if events.empty:
                    return None
                components = _build_capacity_source_components(events)
                _cache_capacity_source_snapshot(source_key, components)
            else:
                events = components["woodmac_store"]
    else:
        events = components["woodmac_store"]
    state = source_state or fetch_capacity_source_state(engine, source_key)
    min_month, max_month = capacity_source_event_bounds(events)
    metadata = {
        "snapshot_key": source_key,
        "woodmac": {
            "monthly_upload_timestamp_utc": (watermarks or {}).get(
                "monthly_upload_timestamp_utc"
            ),
            "annual_upload_timestamp_utc": (watermarks or {}).get(
                "annual_upload_timestamp_utc"
            ),
            "upload_timestamp_utc": max(
                filter(
                    None,
                    [
                        (watermarks or {}).get("monthly_upload_timestamp_utc"),
                        (watermarks or {}).get("annual_upload_timestamp_utc"),
                    ],
                ),
                default=None,
            ),
        },
        "ea": {
            "publication_date": (watermarks or {}).get("ea_publication_date"),
            "upload_timestamp_utc": (watermarks or {}).get(
                "ea_upload_timestamp_utc"
            ),
        },
    }
    return {
        "snapshot_key": source_key,
        "event_count": int(len(events)),
        "available_countries": sorted(events["country_name"].dropna().astype(str).unique()),
        "metadata": metadata,
        "error_message": None,
        "last_success_at": _serialize_timestamp(
            state.get("display_finished_at") or state.get("finished_at")
        ),
        "min_date": _serialize_timestamp(min_month),
        "max_date": _serialize_timestamp(max_month),
    }


def _resolve_capacity_source_ref(
    payload: object,
    component: str,
):
    if not _is_capacity_source_ref(payload):
        return payload
    if str(payload.get("component") or "") != component:
        raise ValueError(
            f"Capacity source reference expected {component}, received {payload.get('component')}."
        )

    source_key = str(payload.get("source_key") or "")
    if not source_key:
        raise ValueError("Capacity source reference is missing its source key.")
    cached = _get_cached_capacity_source_snapshot(source_key)
    if cached is None:
        with _CAPACITY_SOURCE_BUILD_LOCK:
            cached = _get_cached_capacity_source_snapshot(source_key)
            if cached is None:
                events = fetch_capacity_source_events(engine, source_key)
                if events.empty:
                    raise RuntimeError(
                        f"Capacity source {source_key} is no longer available."
                    )
                cached = _build_capacity_source_components(events)
                _cache_capacity_source_snapshot(source_key, cached)
    return cached.get(component)


def _serialize_capacity_refresh_job(row) -> dict[str, object] | None:
    if row is None:
        return None
    payload = dict(row)
    for column_name in ["requested_at", "finished_at"]:
        payload[column_name] = _serialize_timestamp(payload.get(column_name))
    return payload


def _read_capacity_refresh_job(source_key: str) -> dict[str, object] | None:
    return _serialize_capacity_refresh_job(
        read_capacity_refresh_job(engine, source_key)
    )


def _build_relational_capacity_source_events(source_key: str) -> pd.DataFrame:
    mapping_loaders = {
        "country": fetch_country_mapping_df,
        "links": fetch_provider_link_mapping_df,
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {name: executor.submit(loader) for name, loader in mapping_loaders.items()}
        mappings = {name: future.result() for name, future in futures.items()}
    with ThreadPoolExecutor(max_workers=2) as executor:
        woodmac_future = executor.submit(
            fetch_woodmac_train_capacity_raw_data,
            mappings["country"],
            mappings["links"],
        )
        ea_future = executor.submit(
            fetch_ea_capacity_raw_data,
            mappings["country"],
            mappings["links"],
        )
        woodmac_df = woodmac_future.result()
        ea_df = ea_future.result()
    return compile_capacity_source_events(
        woodmac_df,
        ea_df,
        source_key,
        ea_excluded_statuses=EA_EXCLUDED_STATUSES,
        ea_negative_statuses=EA_NEGATIVE_STATUSES,
    )


def _run_capacity_source_refresh_job(
    source_key: str,
    watermarks: dict[str, str | None] | None = None,
) -> None:
    lock_key = f"capacity-source-events:{source_key}"
    try:
        with engine.connect() as lock_connection:
            lock_connection.execute(
                text("SELECT pg_advisory_lock(hashtextextended(:lock_key, 0))"),
                {"lock_key": lock_key},
            )
            lock_connection.commit()
            try:
                job = read_capacity_refresh_job(engine, source_key)
                if job is None or job.get("status") != "running":
                    return
                current_state = _fetch_capacity_source_state()
                if str(current_state.get("source_key") or "") != source_key:
                    message = (
                        "Source inputs changed during refresh; refresh restarted for the current source."
                    )
                    mark_capacity_source_failed(engine, source_key, message)
                    _queue_capacity_source_refresh(
                        str(current_state.get("source_key") or ""),
                        start_worker=True,
                    )
                    return
                events = _build_relational_capacity_source_events(source_key)
                refreshed_state = _fetch_capacity_source_state()
                if str(refreshed_state.get("source_key") or "") != source_key:
                    message = (
                        "Source inputs changed during refresh; refresh restarted for the current source."
                    )
                    mark_capacity_source_failed(engine, source_key, message)
                    _queue_capacity_source_refresh(
                        str(refreshed_state.get("source_key") or ""),
                        start_worker=True,
                    )
                    return
                complete_capacity_source_refresh(engine, source_key, events)
                with _CAPACITY_SOURCE_CACHE_LOCK:
                    _CAPACITY_SOURCE_CACHE.pop(source_key, None)
                with _CAPACITY_DERIVED_CACHE_LOCK:
                    _CAPACITY_DERIVED_CACHE.clear()
                with _CAPACITY_VIEW_CACHE_LOCK:
                    _CAPACITY_VIEW_CACHE.clear()
                    _CAPACITY_VIEW_CACHE_KEY_LOCKS.clear()
            finally:
                lock_connection.execute(
                    text("SELECT pg_advisory_unlock(hashtextextended(:lock_key, 0))"),
                    {"lock_key": lock_key},
                )
                lock_connection.commit()
    except Exception as exc:
        mark_capacity_source_failed(engine, source_key, str(exc))


def _queue_capacity_source_refresh(
    source_key: str | None = None,
    *,
    start_worker: bool = True,
) -> dict[str, object]:
    normalized_source_key = str(source_key or "").strip()
    if not normalized_source_key:
        source_state = _fetch_capacity_source_state()
        normalized_source_key = str(source_state.get("source_key") or "")
    if not normalized_source_key:
        raise RuntimeError("Capacity source watermarks are unavailable.")
    existing = read_capacity_refresh_job(engine, normalized_source_key)
    if existing and existing.get("status") == "running":
        requested_at = pd.to_datetime(existing.get("requested_at"), errors="coerce", utc=True)
        if pd.notna(requested_at) and requested_at < pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=5):
            mark_capacity_source_failed(
                engine,
                normalized_source_key,
                "Refresh worker did not complete within five minutes.",
            )
    row, created = mark_capacity_source_running(engine, normalized_source_key)
    job = _serialize_capacity_refresh_job(row) or {}
    job["_created"] = created
    if created and start_worker:
        try:
            _CAPACITY_REFRESH_EXECUTOR.submit(
                _run_capacity_source_refresh_job,
                normalized_source_key,
            )
        except Exception as exc:
            message = f"Capacity source refresh worker could not start: {exc}"
            mark_capacity_source_failed(engine, normalized_source_key, message)
            raise RuntimeError(message) from exc
    return job


def _build_capacity_refresh_status(
    status: str,
    *,
    snapshot_key: str | None = None,
    error_message: str | None = None,
    last_success_at: str | None = None,
):
    normalized_status = str(status or "idle").casefold()
    revision_label = f"Source {snapshot_key[:12]}" if snapshot_key else "No active source"
    last_success_label = ""
    normalized_success_at = pd.to_datetime(last_success_at, errors="coerce", utc=True)
    if pd.notna(normalized_success_at):
        last_success_label = (
            f" · Last success {normalized_success_at.strftime('%Y-%m-%d %H:%M UTC')}"
        )
    if normalized_status == "running":
        text_value = (
            f"{revision_label} · Refreshing source data; current data remains available"
            f"{last_success_label}."
        )
        color = "#9a3412"
    elif normalized_status == "completed":
        text_value = f"{revision_label} · Refresh completed successfully{last_success_label}."
        color = "#166534"
    elif normalized_status == "failed":
        text_value = (
            f"{revision_label} · Refresh did not replace the active data"
            f"{': ' + error_message if error_message else ''}{last_success_label}."
        )
        color = "#991b1b"
    elif normalized_status == "loading":
        text_value = "Preparing the first capacity source snapshot..."
        color = "#1d4ed8"
    else:
        text_value = f"{revision_label} · Data ready{last_success_label}."
        color = "#475569"
    return html.Div(text_value, style={"color": color})


def _format_capacity_source_timestamp(value) -> str:
    normalized_value = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(normalized_value):
        return "Unavailable"
    return normalized_value.strftime("%Y-%m-%d %H:%M UTC")


def _build_capacity_source_metadata_text(
    metadata: dict | None,
    selected_scenario_id,
    scenario_options_data: list[dict] | None,
) -> str:
    normalized_metadata = metadata if isinstance(metadata, dict) else {}
    ea_metadata = normalized_metadata.get("ea")
    if not isinstance(ea_metadata, dict):
        ea_metadata = {}
    woodmac_metadata = normalized_metadata.get("woodmac")
    if not isinstance(woodmac_metadata, dict):
        woodmac_metadata = {}

    woodmac_timestamp_values = pd.to_datetime(
        [
            woodmac_metadata.get("upload_timestamp_utc"),
            woodmac_metadata.get("monthly_upload_timestamp_utc"),
            woodmac_metadata.get("annual_upload_timestamp_utc"),
        ],
        errors="coerce",
        utc=True,
    )
    valid_woodmac_timestamps = woodmac_timestamp_values[
        woodmac_timestamp_values.notna()
    ]
    latest_woodmac_timestamp = (
        valid_woodmac_timestamps.max()
        if len(valid_woodmac_timestamps)
        else None
    )

    selected_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    selected_option = None
    if pd.notna(selected_value):
        selected_option = _get_capacity_scenario_option_map(
            scenario_options_data
        ).get(int(selected_value))
    scenario_name = (
        str(selected_option.get("scenario_name") or "").strip()
        if selected_option
        else ""
    )

    return (
        "EA uploaded: "
        f"{_format_capacity_source_timestamp(ea_metadata.get('upload_timestamp_utc'))}"
        " · WoodMac uploaded: "
        f"{_format_capacity_source_timestamp(latest_woodmac_timestamp)}"
        " · Internal scenario: "
        f"{scenario_name or 'None selected'}"
    )


def _resolve_capacity_date_values(
    min_date,
    max_date,
    current_start_date,
    current_end_date,
) -> tuple[str | None, str | None, str | None, str | None]:
    normalized_min = _normalize_month_date(min_date)
    normalized_max = _normalize_month_date(max_date)
    if normalized_min is None or normalized_max is None:
        return None, None, None, None

    default_start, default_end = _get_default_interval_window()
    normalized_start = _normalize_month_date(current_start_date) or default_start
    normalized_end = _normalize_month_date(current_end_date) or default_end
    if normalized_start < normalized_min:
        normalized_start = normalized_min
    if normalized_end > normalized_max:
        normalized_end = normalized_max
    if normalized_start > normalized_end:
        normalized_start = normalized_min
        normalized_end = normalized_max
    return (
        normalized_min.strftime("%Y-%m-%d"),
        normalized_max.strftime("%Y-%m-%d"),
        normalized_start.strftime("%Y-%m-%d"),
        normalized_end.strftime("%Y-%m-%d"),
    )


def _build_capacity_source_activation_values(
    snapshot: dict[str, object],
    scenario_options: list[dict],
    current_countries,
    current_start_date,
    current_end_date,
) -> tuple[object, ...]:
    country_options, selected_countries = update_capacity_country_options(
        snapshot["available_countries"],
        current_countries,
    )
    min_date, max_date, start_date, end_date = _resolve_capacity_date_values(
        snapshot.get("min_date"),
        snapshot.get("max_date"),
        current_start_date,
        current_end_date,
    )
    return (
        _build_capacity_source_ref(snapshot, "woodmac_store"),
        _build_capacity_source_ref(snapshot, "train_store"),
        _build_capacity_source_ref(snapshot, "ea_store"),
        snapshot["available_countries"],
        dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        snapshot["error_message"],
        snapshot["metadata"],
        country_options,
        selected_countries,
        min_date,
        max_date,
        start_date,
        end_date,
        scenario_options,
    )


def _load_capacity_source_snapshot(
    *,
    force_refresh: bool = False,
) -> dict[str, object]:
    if force_refresh:
        with _CAPACITY_DERIVED_CACHE_LOCK:
            _CAPACITY_DERIVED_CACHE.clear()
        with _CAPACITY_VIEW_CACHE_LOCK:
            _CAPACITY_VIEW_CACHE.clear()
            _CAPACITY_VIEW_CACHE_KEY_LOCKS.clear()

    source_state = _fetch_capacity_source_state()
    current_source_key = str(source_state.get("source_key") or "")
    display_source_key = str(source_state.get("display_source_key") or "")
    if force_refresh and current_source_key:
        _queue_capacity_source_refresh(current_source_key)
    if not display_source_key:
        raise RuntimeError("No completed relational Capacity source cache is available.")
    snapshot = _fetch_relational_capacity_source_snapshot(
        display_source_key,
        source_state.get("watermarks") or {},
        source_state=source_state,
    )
    if snapshot is None:
        raise RuntimeError(
            f"Completed Capacity source {display_source_key} has no event rows."
        )
    if display_source_key != current_source_key:
        snapshot["error_message"] = (
            "Displaying the preceding completed Capacity source while the current source refreshes."
        )
    return snapshot


def _get_default_interval_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    current_year = pd.Timestamp.now().year
    default_start = pd.Timestamp(year=current_year - 1, month=12, day=1)
    default_end = pd.Timestamp(year=current_year + 5, month=12, day=1)
    return default_start, default_end


def _apply_capacity_time_view(matrix_df: pd.DataFrame, time_view: str) -> pd.DataFrame:
    if matrix_df.empty:
        return matrix_df.copy()

    view_df = matrix_df.copy()
    view_df["__axis_date"] = pd.to_datetime(
        view_df["Month"].astype(str),
        errors="coerce",
    ).dt.to_period("M").dt.to_timestamp()
    view_df = view_df.dropna(subset=["__axis_date"]).sort_values("__axis_date").reset_index(drop=True)

    if time_view == "quarterly":
        view_df = view_df[view_df["__axis_date"].dt.month.isin([3, 6, 9, 12])].copy()
        view_df["Month"] = (
            view_df["__axis_date"].dt.year.astype(str)
            + "-Q"
            + view_df["__axis_date"].dt.quarter.astype(str)
        )
    elif time_view == "seasonally":
        view_df = view_df[view_df["__axis_date"].dt.month.isin([3, 9])].copy()
        _, season_labels = _build_lng_season_periods(view_df["__axis_date"])
        view_df["Month"] = season_labels
    elif time_view == "yearly":
        view_df = view_df[view_df["__axis_date"].dt.month.eq(12)].copy()
        view_df["Month"] = view_df["__axis_date"].dt.year.astype(str)

    return view_df.reset_index(drop=True)


def _create_empty_capacity_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=18, family="Arial", color="#64748b"),
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        margin=dict(l=80, r=80, t=80, b=80),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _resolve_country_color(country_name: str, color_index: int) -> str:
    if country_name == "Rest of the World":
        return "#94A3B8"

    return COUNTRY_COLORS.get(country_name, FALLBACK_COLORS[color_index % len(FALLBACK_COLORS)])


def _format_yoy_delta(value: float) -> str:
    return f"{value:+,.0f} MTPA"


def _format_total_capacity(value: float) -> str:
    return f"{value:,.0f} MTPA"


def _format_yoy_delta_with_percent(current_value: float, previous_value: float) -> str:
    delta_value = current_value - previous_value
    delta_text = _format_yoy_delta(delta_value)
    if abs(previous_value) < 1e-9:
        return delta_text

    percent_change = (delta_value / previous_value) * 100
    return f"{delta_text} ({percent_change:+.1f}%)"


def _build_december_yoy_annotations(total_series: pd.Series) -> list[dict]:
    if total_series.empty:
        return []

    december_points = total_series[total_series.index.month == 12]
    if december_points.empty:
        return []

    annotations = []
    offset_pattern = [-34, -48, -34, -48]

    for annotation_index, (current_date, current_value) in enumerate(december_points.items()):
        previous_date = current_date - pd.DateOffset(years=1)
        if previous_date not in total_series.index:
            continue

        previous_value = float(total_series.loc[previous_date])
        current_value = float(current_value)
        delta_value = current_value - previous_value
        is_positive = delta_value >= 0
        accent_color = "#166534" if is_positive else "#991b1b"
        border_color = "#bbf7d0" if is_positive else "#fecaca"

        annotations.append(
            dict(
                x=current_date,
                y=current_value,
                text=(
                    f"<span style='color:#111827'>{_format_total_capacity(current_value)}</span>"
                    f"<br><span style='color:{accent_color}'>"
                    f"{_format_yoy_delta_with_percent(current_value, previous_value)}"
                    f"</span>"
                ),
                showarrow=True,
                arrowhead=0,
                arrowsize=0.8,
                arrowwidth=1,
                arrowcolor=accent_color,
                ax=0,
                ay=offset_pattern[annotation_index % len(offset_pattern)],
                bgcolor="rgba(255, 255, 255, 0.92)",
                bordercolor=border_color,
                borderwidth=1,
                borderpad=4,
                align="center",
                font=dict(size=10, family="Arial", color="#111827"),
            )
        )

    return annotations


def _resolve_selected_countries(
    available_countries: list[str],
    selected_countries: list[str] | None,
) -> list[str]:
    if selected_countries is None:
        return default_selected_countries(available_countries)

    return [
        country for country in selected_countries if country in available_countries
    ]


def _resolve_capacity_country_scope(
    train_raw_df: pd.DataFrame,
    ea_raw_df: pd.DataFrame,
    selected_countries: list[str] | None,
    *,
    scenario_rows_df: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[str]:
    country_scope_frames = []
    if not train_raw_df.empty:
        cached_country_scope_df = train_raw_df.attrs.get(
            "capacity_country_scope_df"
        )
        if isinstance(cached_country_scope_df, pd.DataFrame) and not cached_country_scope_df.empty:
            country_scope_frames.append(cached_country_scope_df)
        elif {"month", "country_name", "capacity_mtpa"}.issubset(train_raw_df.columns):
            country_scope_frames.append(
                train_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
                    ["month", "country_name", "total_mmtpa"]
                ]
            )
        elif {"Effective Date", "Country", "Abs Delta"}.issubset(train_raw_df.columns):
            country_scope_frames.append(
                train_raw_df.rename(
                    columns={
                        "Effective Date": "month",
                        "Country": "country_name",
                        "Abs Delta": "total_mmtpa",
                    }
                )[["month", "country_name", "total_mmtpa"]]
            )
    if not ea_raw_df.empty:
        country_scope_frames.append(
            ea_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
                ["month", "country_name", "total_mmtpa"]
            ]
        )
    if scenario_rows_df is not None:
        internal_scope_df = _get_cached_internal_scenario_monthly_schedule(
            scenario_rows_df,
            start_date,
            end_date,
        )
        if not internal_scope_df.empty:
            country_scope_frames.append(internal_scope_df)

    return _resolve_selected_countries(
        get_available_countries(country_scope_frames),
        selected_countries,
    )


def _build_capacity_derived_cache_key(
    train_capacity_data: dict | str | None,
    ea_capacity_data: str | None,
    selected_countries: list[str] | None,
    other_countries_mode: str,
    start_date: str | None,
    end_date: str | None,
    scenario_rows_df: pd.DataFrame | None,
) -> str:
    scenario_rows_payload = _serialize_dataframe(
        _prepare_capacity_scenario_rows_df(scenario_rows_df)
        if scenario_rows_df is not None
        else pd.DataFrame()
    )
    key_payload = {
        "snapshot_key": _get_train_capacity_snapshot_key(train_capacity_data),
        "ea_hash": hashlib.sha256(
            str(ea_capacity_data or "").encode("utf-8")
        ).hexdigest(),
        "selected_countries": list(selected_countries or []),
        "other_countries_mode": other_countries_mode,
        "start_date": _serialize_timestamp(_normalize_month_date(start_date)),
        "end_date": _serialize_timestamp(_normalize_month_date(end_date)),
        "scenario_hash": hashlib.sha256(
            str(scenario_rows_payload or "").encode("utf-8")
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _get_prepared_capacity_provider_data(
    train_capacity_data: dict | str | None,
    ea_capacity_data: str | None,
    selected_countries: list[str] | None,
    other_countries_mode: str,
    start_date: str | None,
    end_date: str | None,
    *,
    scenario_rows_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    prepared_scenario_rows_df = (
        _prepare_capacity_scenario_rows_df(scenario_rows_df)
        if scenario_rows_df is not None
        else pd.DataFrame()
    )
    # Provider change logs, lookup snapshots, and timeline rows do not depend on
    # the selected internal scenario. Keep one provider-only core per source and
    # filter revision, then overlay the scenario-aware country scope below.
    cache_key = _build_capacity_derived_cache_key(
        train_capacity_data,
        ea_capacity_data,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
        None,
    )
    with _CAPACITY_DERIVED_CACHE_LOCK:
        cached_payload = _CAPACITY_DERIVED_CACHE.get(cache_key)
        if cached_payload is not None:
            _CAPACITY_DERIVED_CACHE.move_to_end(cache_key)
        else:
            train_raw_df = _deserialize_train_capacity_store(train_capacity_data)
            ea_raw_df = _deserialize_ea_capacity_store(ea_capacity_data)
            resolved_countries = _resolve_capacity_country_scope(
                train_raw_df,
                ea_raw_df,
                selected_countries,
                start_date=start_date,
                end_date=end_date,
            )
            full_change_payload = _get_cached_capacity_view_payload(
                "provider_full_change_logs",
                {
                    "train": _get_capacity_payload_signature(train_capacity_data),
                    "ea": _get_capacity_payload_signature(ea_capacity_data),
                },
                lambda: {
                    "woodmac": _build_train_change_log(
                        train_raw_df,
                        None,
                        "rest_of_world",
                        None,
                        None,
                    ),
                    "ea": _build_ea_change_log(
                        ea_raw_df,
                        None,
                        "rest_of_world",
                        None,
                        None,
                    ),
                },
            )
            full_woodmac_change_df = _build_train_change_log(
                full_change_payload["woodmac"],
                resolved_countries,
                other_countries_mode,
                None,
                None,
            )
            full_ea_change_df = _build_ea_change_log(
                full_change_payload["ea"],
                resolved_countries,
                other_countries_mode,
                None,
                None,
            )
            woodmac_change_df = _build_train_change_log(
                full_woodmac_change_df,
                resolved_countries,
                other_countries_mode,
                start_date,
                end_date,
            )
            ea_change_df = _build_ea_change_log(
                full_ea_change_df,
                resolved_countries,
                other_countries_mode,
                start_date,
                end_date,
            )
            timeline_df = _build_train_timeline_df(
                woodmac_change_df,
                ea_change_df,
                aggregate_from_date=start_date,
            )
            woodmac_lookup_df = _build_provider_timeline_lookup_snapshot(
                full_woodmac_change_df,
                "woodmac",
                start_date,
                end_date,
            )
            ea_lookup_df = _build_provider_timeline_lookup_snapshot(
                full_ea_change_df,
                "energy_aspects",
                start_date,
                end_date,
            )
            cached_payload = {
                "train_raw_df": train_raw_df,
                "ea_raw_df": ea_raw_df,
                "scenario_rows_df": pd.DataFrame(),
                "resolved_countries": resolved_countries,
                "woodmac_change_df": woodmac_change_df,
                "ea_change_df": ea_change_df,
                "timeline_df": timeline_df,
                "woodmac_lookup_df": woodmac_lookup_df,
                "ea_lookup_df": ea_lookup_df,
            }
            _CAPACITY_DERIVED_CACHE[cache_key] = cached_payload
            _CAPACITY_DERIVED_CACHE.move_to_end(cache_key)
            while len(_CAPACITY_DERIVED_CACHE) > CAPACITY_DERIVED_CACHE_MAX_ENTRIES:
                _CAPACITY_DERIVED_CACHE.popitem(last=False)

    if scenario_rows_df is None:
        return cached_payload

    scenario_scope_key = _build_capacity_derived_cache_key(
        train_capacity_data,
        ea_capacity_data,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
        prepared_scenario_rows_df,
    )
    scenario_scope = _get_cached_capacity_view_payload(
        "provider_scenario_scope",
        {"derived": scenario_scope_key},
        lambda: {
            "scenario_rows_df": prepared_scenario_rows_df,
            "resolved_countries": _resolve_capacity_country_scope(
                cached_payload["train_raw_df"],
                cached_payload["ea_raw_df"],
                selected_countries,
                scenario_rows_df=prepared_scenario_rows_df,
                start_date=start_date,
                end_date=end_date,
            ),
        },
    )
    scenario_payload = dict(cached_payload)
    scenario_payload.update(scenario_scope)
    return scenario_payload


def _get_capacity_payload_signature(payload: object) -> str:
    if _is_capacity_source_ref(payload):
        return str(payload.get("snapshot_key") or "empty")
    if isinstance(payload, pd.DataFrame):
        return _get_train_capacity_snapshot_key(payload)
    if isinstance(payload, dict) and payload.get("snapshot_key"):
        return str(payload.get("snapshot_key"))
    return hashlib.sha256(str(payload or "").encode("utf-8")).hexdigest()


def _get_cached_capacity_view_payload(
    cache_kind: str,
    key_payload: dict[str, object],
    builder,
):
    cache_key = hashlib.sha256(
        json.dumps(
            {"kind": cache_kind, **key_payload},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    with _CAPACITY_VIEW_CACHE_LOCK:
        cached_payload = _CAPACITY_VIEW_CACHE.get(cache_key)
        if cached_payload is not None:
            _CAPACITY_VIEW_CACHE.move_to_end(cache_key)
            return cached_payload
        key_lock = _CAPACITY_VIEW_CACHE_KEY_LOCKS.setdefault(cache_key, RLock())

    with key_lock:
        with _CAPACITY_VIEW_CACHE_LOCK:
            cached_payload = _CAPACITY_VIEW_CACHE.get(cache_key)
            if cached_payload is not None:
                _CAPACITY_VIEW_CACHE.move_to_end(cache_key)
                return cached_payload
        prepared_payload = builder()
        with _CAPACITY_VIEW_CACHE_LOCK:
            _CAPACITY_VIEW_CACHE[cache_key] = prepared_payload
            _CAPACITY_VIEW_CACHE.move_to_end(cache_key)
            while len(_CAPACITY_VIEW_CACHE) > 64:
                evicted_key, _evicted_payload = _CAPACITY_VIEW_CACHE.popitem(last=False)
                _CAPACITY_VIEW_CACHE_KEY_LOCKS.pop(evicted_key, None)
        return prepared_payload


def _get_prepared_country_capacity_view(
    source_kind: str,
    source_data,
    selected_countries,
    other_countries_mode: str,
    start_date: str | None,
    end_date: str | None,
    time_view: str,
) -> dict[str, object]:
    def _build() -> dict[str, object]:
        if source_kind == "woodmac":
            source_df = _deserialize_woodmac_capacity_store(
                source_data,
                start_date=start_date,
                end_date=end_date,
                selected_countries=selected_countries,
            )
        else:
            source_df = _build_ea_capacity_schedule(
                _deserialize_ea_capacity_store(source_data),
                start_date,
                end_date,
            )
        resolved_countries = _resolve_selected_countries(
            get_available_countries([source_df]),
            selected_countries,
        )
        matrix_df = _apply_capacity_time_view(
            _rename_total_column(
                build_export_flow_matrix(
                    source_df,
                    resolved_countries,
                    other_countries_mode,
                )
            ),
            time_view,
        )
        return {
            "source_df": source_df,
            "resolved_countries": resolved_countries,
            "matrix_df": matrix_df,
        }

    return _get_cached_capacity_view_payload(
        f"country_capacity:{source_kind}",
        {
            "source": _get_capacity_payload_signature(source_data),
            "countries": list(selected_countries or []),
            "other_mode": other_countries_mode,
            "start_date": start_date,
            "end_date": end_date,
            "time_view": time_view,
        },
        _build,
    )


def _get_prepared_internal_capacity_view(
    working_store_data,
    selected_countries,
    other_countries_mode: str,
    start_date: str | None,
    end_date: str | None,
    time_view: str,
) -> dict[str, object]:
    def _build() -> dict[str, object]:
        working_rows_df = _resolve_active_capacity_scenario_rows(working_store_data)
        schedule_df = _get_cached_internal_scenario_monthly_schedule(
            working_rows_df,
            start_date,
            end_date,
        )
        resolved_countries = _resolve_selected_countries(
            get_available_countries([schedule_df]),
            selected_countries,
        )
        matrix_df = _apply_capacity_time_view(
            _rename_total_column(
                build_export_flow_matrix(
                    schedule_df,
                    resolved_countries,
                    other_countries_mode,
                )
            ),
            time_view,
        )
        return {
            "working_rows_df": working_rows_df,
            "schedule_df": schedule_df,
            "resolved_countries": resolved_countries,
            "matrix_df": matrix_df,
        }

    return _get_cached_capacity_view_payload(
        "internal_capacity",
        {
            "working": hashlib.sha256(
                json.dumps(
                    working_store_data,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest(),
            "countries": list(selected_countries or []),
            "other_mode": other_countries_mode,
            "start_date": start_date,
            "end_date": end_date,
            "time_view": time_view,
        },
        _build,
    )


def _build_responsive_column_styles(df: pd.DataFrame) -> list[dict]:
    column_styles = []

    for column_name in df.columns:
        header_length = len(str(column_name))
        value_lengths = df[column_name].map(_format_table_cell_value).map(len)
        max_length = max([header_length] + value_lengths.tolist()) if not df.empty else header_length

        if column_name == "Month":
            width_px = max(82, min((max_length * 7) + 18, 108))
        elif str(column_name).startswith("Total "):
            width_px = max(88, min((max_length * 7) + 18, 112))
        else:
            width_px = max(64, min((max_length * 6) + 16, 112))

        style_entry = {
            "if": {"column_id": column_name},
            "minWidth": f"{width_px}px",
            "width": f"{width_px}px",
            "maxWidth": f"{width_px}px",
        }

        if column_name == "Month":
            style_entry["textAlign"] = "left"

        column_styles.append(style_entry)

    return column_styles


def _build_fixed_column_width_styles(
    column_widths: dict[str, int],
    left_align_columns: set[str] | None = None,
) -> list[dict]:
    left_align_columns = left_align_columns or set()
    column_styles = []

    for column_name, width_px in column_widths.items():
        style_entry = {
            "if": {"column_id": column_name},
            "minWidth": f"{width_px}px",
            "width": f"{width_px}px",
            "maxWidth": f"{width_px}px",
        }
        if column_name in left_align_columns:
            style_entry["textAlign"] = "left"
        column_styles.append(style_entry)

    return column_styles


def _get_provider_discrepancy_config(provider: str) -> dict[str, object]:
    provider_key = str(provider or "").strip().casefold()
    provider_config = {
        "woodmac": {
            "provider_key": "woodmac",
            "display_name": "Woodmac",
            "entity_input_column": "Plant",
            "entity_output_column": "Plant",
            "provider_date_column": "Woodmac First Date",
            "provider_capacity_change_column": "Woodmac Capacity Change",
            "provider_capacity_display_column": "Woodmac Capacity",
            "provider_color": "#1d4ed8",
            "capacity_columns": [
                "Country",
                "Plant",
                "Train",
                "Woodmac First Date",
                "Woodmac Capacity",
                "Scenario First Date",
                "Scenario Capacity",
                "Abs Capacity Delta",
            ],
            "timeline_columns": [
                "Country",
                "Plant",
                "Train",
                "Woodmac First Date",
                "Scenario First Date",
                "Abs Timeline Delta (Months)",
            ],
            "missing_columns": [
                "Country",
                "Plant",
                "Train",
                "Woodmac First Date",
                "Woodmac Capacity",
            ],
            "capacity_empty_message": (
                "No Woodmac capacity discrepancies were found in the current selection."
            ),
            "timeline_empty_message": (
                "No Woodmac timeline discrepancies were found in the current selection."
            ),
            "missing_empty_message": (
                "No Woodmac plants and trains are missing in the current internal scenario."
            ),
        },
        "energy_aspects": {
            "provider_key": "energy_aspects",
            "display_name": "Energy Aspects",
            "entity_input_column": "Plant",
            "entity_output_column": "Project",
            "provider_date_column": "Energy Aspects First Date",
            "provider_capacity_change_column": "Energy Aspects Capacity Change",
            "provider_capacity_display_column": "Energy Aspects Capacity",
            "provider_color": "#b45309",
            "capacity_columns": [
                "Country",
                "Project",
                "Train",
                "Energy Aspects First Date",
                "Energy Aspects Capacity",
                "Scenario First Date",
                "Scenario Capacity",
                "Abs Capacity Delta",
            ],
            "timeline_columns": [
                "Country",
                "Project",
                "Train",
                "Energy Aspects First Date",
                "Scenario First Date",
                "Abs Timeline Delta (Months)",
            ],
            "missing_columns": [
                "Country",
                "Project",
                "Train",
                "Energy Aspects First Date",
                "Energy Aspects Capacity",
            ],
            "capacity_empty_message": (
                "No Energy Aspects capacity discrepancies were found in the current selection."
            ),
            "timeline_empty_message": (
                "No Energy Aspects timeline discrepancies were found in the current selection."
            ),
            "missing_empty_message": (
                "No Energy Aspects plants and trains are missing in the current internal scenario."
            ),
        },
    }.get(provider_key)
    if provider_config is None:
        raise ValueError(f"Unsupported provider '{provider}'.")

    return provider_config


def _build_yearly_capacity_comparison_column_styles() -> list[dict]:
    return _build_fixed_column_width_styles(
        {
            "Year": 52,
            "Internal Scenario": 76,
            "Woodmac": 68,
            "Energy Aspects": 78,
            "Delta vs Woodmac": 86,
            "Delta vs Energy Aspects": 92,
        }
    )


def _build_yearly_discrepancy_column_styles(provider: str) -> list[dict]:
    provider_config = _get_provider_discrepancy_config(provider)
    entity_column = str(provider_config["entity_output_column"])
    provider_date_column = str(provider_config["provider_date_column"])
    provider_capacity_column = str(provider_config["provider_capacity_display_column"])

    return _build_fixed_column_width_styles(
        {
            "Country": 74,
            entity_column: 102,
            "Train": 46,
            provider_date_column: 88,
            provider_capacity_column: 78,
            "Scenario First Date": 88,
            "Scenario Capacity": 78,
            "Abs Capacity Delta": 80,
        },
        left_align_columns={"Country", entity_column},
    )


def _build_yearly_timeline_discrepancy_column_styles(provider: str) -> list[dict]:
    provider_config = _get_provider_discrepancy_config(provider)
    entity_column = str(provider_config["entity_output_column"])
    provider_date_column = str(provider_config["provider_date_column"])

    return _build_fixed_column_width_styles(
        {
            "Country": 74,
            entity_column: 108,
            "Train": 46,
            provider_date_column: 88,
            "Scenario First Date": 88,
            "Abs Timeline Delta (Months)": 94,
        },
        left_align_columns={"Country", entity_column},
    )


def _build_yearly_missing_internal_column_styles(provider: str) -> list[dict]:
    provider_config = _get_provider_discrepancy_config(provider)
    entity_column = str(provider_config["entity_output_column"])
    provider_date_column = str(provider_config["provider_date_column"])
    provider_capacity_column = str(provider_config["provider_capacity_display_column"])

    return _build_fixed_column_width_styles(
        {
            "Country": 74,
            entity_column: 108,
            "Train": 46,
            provider_date_column: 88,
            provider_capacity_column: 78,
        },
        left_align_columns={"Country", entity_column},
    )


YEARLY_PROVIDER_DISCREPANCY_TABLE_MAX_HEIGHT = "390px"


def _rename_total_column(matrix_df: pd.DataFrame) -> pd.DataFrame:
    if matrix_df.empty:
        return pd.DataFrame(columns=["Month", "Total MTPA"])

    renamed_df = matrix_df.rename(columns={"Total MMTPA": "Total MTPA"})
    if "Total MTPA" not in renamed_df.columns:
        renamed_df["Total MTPA"] = 0.0

    return renamed_df


def _build_capacity_metadata_lines(metadata: dict | None) -> list[str]:
    if not metadata:
        return []

    metadata_lines = []

    metadata_lines.append(WOODMAC_LEGACY_CAPACITY_NOTE)

    return metadata_lines


def _build_ea_capacity_metadata_lines(
    metadata: dict | None,
) -> list[str]:
    if not metadata:
        return [EA_SCHEDULE_CAPACITY_NOTE]

    metadata_lines = []

    metadata_lines.append(EA_SCHEDULE_CAPACITY_NOTE)
    metadata_lines.append(
        "Cancelled and retired projects are excluded from this top schedule view."
    )

    return metadata_lines


def _build_section_summary(
    raw_df: pd.DataFrame,
    metadata_lines: list[str] | None = None,
) -> html.Div:
    summary_children = []

    if raw_df.empty:
        summary_children.append(
            html.Div("No source data returned.", className="balance-summary-row")
        )

    if metadata_lines:
        summary_children.append(
            html.Div(
                [html.Span(line) for line in metadata_lines],
                className="balance-metadata-row",
            )
        )

    return html.Div(summary_children)


def _build_train_change_summary(
    woodmac_change_df: pd.DataFrame,
    ea_change_df: pd.DataFrame,
    time_view: str = "monthly",
    detail_view: str = "country",
) -> html.Div:
    time_view_period_label = TRAIN_CHANGE_TIME_VIEW_PERIOD_LABELS.get(time_view, "monthly")
    detail_view_label = TRAIN_CHANGE_DETAIL_VIEW_LABELS.get(detail_view, "Country")

    if woodmac_change_df.empty and ea_change_df.empty:
        return html.Div(
            [
                html.Div(
                    "No provider capacity changes detected in the selected range.",
                    className="balance-summary-row",
                ),
                html.Div(
                    [
                        html.Span(
                            f"Time View aggregates changes into {time_view_period_label} periods. Detail View is set to {detail_view_label}. Total groups all visible countries together, Country shows one row per period-country, Plants View keeps one row per period-plant, and Plants + Trains View adds shared canonical train rows only when the visible provider changes are fully resolved at train level."
                        )
                    ],
                    className="balance-metadata-row",
                ),
            ]
        )

    return html.Div()


def _build_train_change_footer_notes() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        f"Woodmac Effective Date reflects the first day of each monthly series point. Monthly nominal capacity is used first. For trains with monthly history, the last monthly capacity is carried forward after monthly coverage ends. Annual proxy is only used for trains with no monthly capacity history, and annual-only legacy trains are carried forward from {WOODMAC_ANNUAL_CARRY_FORWARD_START[:4]} onward."
                    )
                ],
                className="balance-metadata-row",
            ),
            html.Div(
                [html.Span(WOODMAC_LEGACY_CAPACITY_NOTE)],
                className="balance-metadata-row",
            ),
        ],
        style={"paddingTop": "12px"},
    )


def _build_capacity_scenario_row_key(
    country_name: object,
    plant_name: object,
    train_label: object,
    effective_date: object | None = None,
    capacity_value: object | None = None,
    provider: str | None = None,
) -> str:
    normalized_parts = [
        " ".join(str(country_name or "").strip().split()).casefold(),
        " ".join(str(plant_name or "").strip().split()).casefold(),
        " ".join(str(train_label or "").strip().split()).casefold(),
    ]
    if effective_date is not None:
        normalized_effective_date = pd.to_datetime(effective_date, errors="coerce")
        normalized_parts.append(
            ""
            if pd.isna(normalized_effective_date)
            else normalized_effective_date.strftime("%Y-%m-%d")
        )
    if capacity_value is not None:
        numeric_capacity = pd.to_numeric([capacity_value], errors="coerce")[0]
        normalized_parts.append("" if pd.isna(numeric_capacity) else f"{float(numeric_capacity):.6f}")
    if provider is not None:
        normalized_parts.append(" ".join(str(provider).strip().split()).casefold())
    digest = hashlib.sha1("|".join(normalized_parts).encode("utf-8")).hexdigest()[:16]
    return f"capacity-scenario-{digest}"


def _normalize_capacity_change_direction(value: object) -> str | None:
    numeric_value = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(numeric_value) or round(float(numeric_value), 6) == 0:
        return None
    return "reduction" if float(numeric_value) < 0 else "addition"


def _normalize_train_timeline_identity_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value).strip().split()).casefold()


def _normalize_train_timeline_fallback_label(value: object) -> str:
    normalized_value = _normalize_train_timeline_identity_text(value)
    match = re.fullmatch(r"(?:train\s*)?(\d+)", normalized_value)
    if match:
        return str(int(match.group(1)))
    return normalized_value


def _build_train_timeline_identity_key(row: pd.Series) -> str:
    direction = _normalize_train_timeline_identity_text(row.get("timeline_direction"))
    train_key = _normalize_train_timeline_identity_text(row.get("train_key"))
    if train_key:
        return f"canonical|{train_key}|{direction}"

    fallback_parts = [
        _normalize_train_timeline_identity_text(row.get("Country")),
        _normalize_train_timeline_identity_text(row.get("Plant")),
        _normalize_train_timeline_fallback_label(row.get("Train")),
        direction,
    ]
    return "display|" + "|".join(fallback_parts)


def _ensure_train_timeline_identity_key(rows_df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = rows_df.copy()
    if prepared_df.empty:
        if "timeline_identity_key" not in prepared_df.columns:
            prepared_df["timeline_identity_key"] = pd.Series(dtype="object")
        return prepared_df

    for column_name in ["Country", "Plant", "Train", "train_key", "timeline_direction"]:
        if column_name not in prepared_df.columns:
            prepared_df[column_name] = ""
    prepared_df["timeline_identity_key"] = prepared_df.apply(
        _build_train_timeline_identity_key,
        axis=1,
    )
    return prepared_df


def _build_train_timeline_reference_key(
    country_name: object,
    plant_name: object,
    train_label: object,
    effective_date: object,
    capacity_value: object,
    aggregate_from_date: str | None = None,
) -> str | None:
    normalized_effective_date = pd.to_datetime(effective_date, errors="coerce")
    direction = _normalize_capacity_change_direction(capacity_value)
    if pd.isna(normalized_effective_date) or direction is None:
        return None

    split_month = _normalize_month_date(aggregate_from_date)
    bucket_kind = (
        "future"
        if split_month is not None and normalized_effective_date >= split_month
        else "history"
    )

    normalized_parts = [
        " ".join(str(country_name or "").strip().split()).casefold(),
        " ".join(str(plant_name or "").strip().split()).casefold(),
        " ".join(str(train_label or "").strip().split()).casefold(),
        bucket_kind,
        direction,
    ]
    if bucket_kind == "history":
        normalized_parts.append(normalized_effective_date.strftime("%Y-%m-%d"))

    digest = hashlib.sha1("|".join(normalized_parts).encode("utf-8")).hexdigest()[:16]
    return f"capacity-timeline-{digest}"


def _prepare_provider_timeline_event_rows(
    change_df: pd.DataFrame,
    provider: str,
) -> pd.DataFrame:
    columns = [
        "Country",
        "Plant",
        "Train",
        "terminal_key",
        "train_key",
        "Canonical Train Label",
        "Provider Plant ID",
        "Provider Train ID",
        "Provider Sources",
        "Provider Identity Key",
        "Original Plant",
        "Original Train",
        "Woodmac FID Date",
        "Effective Date",
        "Capacity Change",
        "timeline_direction",
        "timeline_identity_key",
        "display_sort_plant",
        "display_sort_train",
    ]
    if change_df.empty:
        return pd.DataFrame(columns=columns)

    provider_key = provider.strip().casefold()
    provider_df = change_df.copy()
    provider_df["Country"] = provider_df["Country"].fillna("").astype(str).str.strip()
    provider_df["Plant"] = provider_df["Plant"].fillna("").astype(str).str.strip()
    provider_df["Train"] = provider_df.get("Train").map(_format_train_label)
    provider_df["Effective Date"] = pd.to_datetime(
        provider_df["Effective Date"],
        errors="coerce",
    )
    provider_df["display_sort_train"] = pd.to_numeric(
        provider_df["Train"],
        errors="coerce",
    )
    for column_name in [
        "terminal_key",
        "train_key",
        "Canonical Train Label",
        "Provider Plant ID",
        "Provider Train ID",
    ]:
        if column_name not in provider_df.columns:
            provider_df[column_name] = ""
        provider_df[column_name] = (
            provider_df[column_name].fillna("").astype(str).str.strip()
        )
    if "Provider Sources" not in provider_df.columns:
        provider_df["Provider Sources"] = [[] for _ in range(len(provider_df))]
    provider_df["Provider Sources"] = provider_df["Provider Sources"].map(
        lambda value: value if isinstance(value, list) else []
    )

    if provider_key == "woodmac":
        provider_df["Capacity Change"] = pd.to_numeric(
            provider_df["Delta MTPA"],
            errors="coerce",
        ).fillna(0.0)
        provider_df["Original Plant"] = provider_df.get("Source Name", "").fillna("").astype(str).str.strip()
        provider_df["Original Train"] = (
            provider_df.get("Train Display Source Name", provider_df.get("Train Source Name", ""))
            .fillna("")
            .astype(str)
            .str.strip()
        )
        fid_series = (
            provider_df["Woodmac FID Date"]
            if "Woodmac FID Date" in provider_df.columns
            else pd.Series("", index=provider_df.index)
        )
        provider_df["Woodmac FID Date"] = fid_series.fillna("").astype(str).str.strip()
    elif provider_key == "energy_aspects":
        provider_df["Capacity Change"] = pd.to_numeric(
            provider_df["EA Net Delta (MTPA)"],
            errors="coerce",
        ).fillna(0.0)
        provider_df["Original Plant"] = provider_df.get("Source Name", "").fillna("").astype(str).str.strip()
        provider_df["Original Train"] = provider_df.get("Train Source Name", "").fillna("").astype(str).str.strip()
        provider_df["Woodmac FID Date"] = ""
    else:
        raise ValueError(f"Unsupported provider '{provider}'.")

    provider_df = provider_df[
        provider_df["Effective Date"].notna()
        & provider_df["Capacity Change"].round(6).ne(0)
    ].copy()
    if provider_df.empty:
        return pd.DataFrame(columns=columns)

    canonical_identity = "canonical|" + provider_df["train_key"]
    provider_identity = (
        "provider|"
        + provider_key
        + "|"
        + provider_df["Provider Plant ID"]
        + "|"
        + provider_df["Provider Train ID"]
    )
    fallback_identity = (
        "display|"
        + provider_df["Country"]
        + "|"
        + provider_df["Plant"]
        + "|"
        + provider_df["Train"]
        + "|"
        + provider_df["Original Train"]
    )
    provider_df["Provider Identity Key"] = canonical_identity.where(
        provider_df["train_key"].ne(""),
        provider_identity.where(
            provider_df["Provider Plant ID"].ne("")
            & provider_df["Provider Train ID"].ne(""),
            fallback_identity,
        ),
    )

    provider_df = (
        provider_df.groupby(
            [
                "Country",
                "Plant",
                "Train",
                "Provider Identity Key",
                "terminal_key",
                "train_key",
                "Canonical Train Label",
                "Effective Date",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            {
                "Capacity Change": "sum",
                "display_sort_train": "last",
                "Original Plant": _combine_distinct_text_values,
                "Original Train": _combine_distinct_text_values,
                "Provider Plant ID": _combine_distinct_text_values,
                "Provider Train ID": _combine_distinct_text_values,
                "Woodmac FID Date": _combine_distinct_text_values,
                "Provider Sources": _combine_provider_sources,
            }
        )
    )
    provider_df = provider_df[
        provider_df["Capacity Change"].round(6).ne(0)
    ].copy()
    if provider_df.empty:
        return pd.DataFrame(columns=columns)

    provider_df["timeline_direction"] = provider_df["Capacity Change"].map(
        _normalize_capacity_change_direction
    )
    provider_df = provider_df[provider_df["timeline_direction"].notna()].copy()
    if provider_df.empty:
        return pd.DataFrame(columns=columns)

    provider_df["display_sort_plant"] = provider_df["Plant"]
    provider_df["Capacity Change"] = pd.to_numeric(
        provider_df["Capacity Change"],
        errors="coerce",
    ).round(6)
    provider_df = _ensure_train_timeline_identity_key(provider_df)
    return provider_df[columns]


def _build_provider_timeline_snapshot(
    change_df: pd.DataFrame,
    provider: str,
    aggregate_from_date: str | None = None,
) -> pd.DataFrame:
    columns = [
        "timeline_reference_key",
        "scenario_row_key",
        "Country",
        "Plant",
        "Train",
        "terminal_key",
        "train_key",
        "Canonical Train Label",
        "Provider Plant ID",
        "Provider Train ID",
        "Provider Sources",
        "Provider Identity Key",
        "Original Plant",
        "Original Train",
        "Woodmac FID Date",
        "First Date",
        "Capacity Change",
        "timeline_direction",
        "timeline_identity_key",
        "display_sort_plant",
        "display_sort_train",
        "display_sort_effective_date",
        "display_sort_direction",
    ]
    provider_df = _prepare_provider_timeline_event_rows(change_df, provider)
    if provider_df.empty:
        return pd.DataFrame(columns=columns)

    split_month = _normalize_month_date(aggregate_from_date)
    historical_df = provider_df.copy()
    future_df = pd.DataFrame(columns=provider_df.columns)
    if split_month is not None:
        historical_df = provider_df[provider_df["Effective Date"] < split_month].copy()
        future_source_df = provider_df[provider_df["Effective Date"] >= split_month].copy()
        if not future_source_df.empty:
            future_df = (
                future_source_df.groupby(
                    [
                        "Country",
                        "Plant",
                        "Train",
                        "Provider Identity Key",
                        "terminal_key",
                        "train_key",
                        "Canonical Train Label",
                        "timeline_direction",
                    ],
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    {
                        "Capacity Change": "sum",
                        "Effective Date": "min",
                        "display_sort_train": "last",
                        "Original Plant": _combine_distinct_text_values,
                        "Original Train": _combine_distinct_text_values,
                        "Provider Plant ID": _combine_distinct_text_values,
                        "Provider Train ID": _combine_distinct_text_values,
                        "Provider Sources": _combine_provider_sources,
                        "Woodmac FID Date": _combine_distinct_text_values,
                        "display_sort_plant": "last",
                    }
                )
            )

    snapshot_frames = [
        frame for frame in (historical_df, future_df) if not frame.empty
    ]
    snapshot_df = pd.concat(snapshot_frames, ignore_index=True, sort=False)
    snapshot_df = snapshot_df[
        snapshot_df["Capacity Change"].round(6).ne(0)
        & snapshot_df["timeline_direction"].notna()
    ].copy()
    if snapshot_df.empty:
        return pd.DataFrame(columns=columns)

    snapshot_df["timeline_reference_key"] = snapshot_df.apply(
        lambda row: _build_train_timeline_reference_key(
            row["Country"],
            row["Plant"],
            row.get("Canonical Train Label")
            or row.get("Provider Identity Key")
            or row["Train"],
            row["Effective Date"],
            row["Capacity Change"],
            aggregate_from_date,
        ),
        axis=1,
    )
    snapshot_df["scenario_row_key"] = snapshot_df["timeline_reference_key"]
    snapshot_df["First Date"] = snapshot_df["Effective Date"].dt.strftime("%Y-%m-%d")
    snapshot_df["display_sort_plant"] = snapshot_df["Plant"]
    snapshot_df["display_sort_effective_date"] = snapshot_df["Effective Date"]
    snapshot_df["display_sort_direction"] = snapshot_df["timeline_direction"].map(
        {"reduction": 0, "addition": 1}
    ).fillna(2)
    snapshot_df = _ensure_train_timeline_identity_key(snapshot_df)
    snapshot_df["__train_blank_sort"] = snapshot_df["Train"].eq("").astype(int)
    snapshot_df = snapshot_df.sort_values(
        [
            "Country",
            "display_sort_plant",
            "__train_blank_sort",
            "display_sort_train",
            "Train",
            "display_sort_effective_date",
            "display_sort_direction",
        ],
        ascending=[True, True, False, True, True, True, True],
        na_position="last",
    ).drop(columns=["__train_blank_sort"], errors="ignore").reset_index(drop=True)
    return snapshot_df[columns]


def _build_provider_scenario_rows_from_change_log(
    change_df: pd.DataFrame,
    provider: str,
    aggregate_from_date: str | None = None,
) -> pd.DataFrame:
    columns = _get_capacity_scenario_row_columns()
    provider_key = provider.strip().casefold()
    snapshot_df = _build_provider_timeline_snapshot(
        change_df,
        provider,
        aggregate_from_date=aggregate_from_date,
    )
    if snapshot_df.empty:
        return pd.DataFrame(columns=columns)

    canonical_train_labels = (
        snapshot_df["Canonical Train Label"].fillna("").astype(str).str.strip()
    )
    snapshot_df["Scenario Train Label"] = canonical_train_labels.where(
        canonical_train_labels.ne(""),
        snapshot_df["Train"],
    )
    snapshot_df["scenario_row_key"] = snapshot_df.apply(
        lambda row: _build_capacity_scenario_row_key(
            row["Country"],
            row["Plant"],
            row["Scenario Train Label"],
            effective_date=row["First Date"],
            capacity_value=row["Capacity Change"],
            provider=provider_key,
        ),
        axis=1,
    )

    scenario_rows_df = pd.DataFrame(
        {
            "scenario_row_key": snapshot_df["scenario_row_key"],
            "terminal_key": snapshot_df["terminal_key"],
            "train_key": snapshot_df["train_key"],
            "country_name": snapshot_df["Country"],
            "plant_name": snapshot_df["Plant"],
            "train_label": snapshot_df["Scenario Train Label"],
            "base_provider": provider_key,
            "base_first_date": pd.to_datetime(snapshot_df["First Date"], errors="coerce"),
            "base_capacity_mtpa": pd.to_numeric(snapshot_df["Capacity Change"], errors="coerce").round(6),
            "scenario_first_date": pd.to_datetime(snapshot_df["First Date"], errors="coerce"),
            "scenario_capacity_mtpa": pd.to_numeric(snapshot_df["Capacity Change"], errors="coerce").round(6),
            "scenario_note": "",
            "display_sort_plant": snapshot_df["display_sort_plant"],
            "display_sort_train": snapshot_df["display_sort_train"],
        }
    )
    return _prepare_capacity_scenario_rows_df(scenario_rows_df)


def _month_distance_to_boundary(
    start_value,
    end_value,
) -> int | None:
    month_delta = _month_difference(start_value, end_value)
    if month_delta is None:
        return None
    return abs(int(month_delta))


def _select_train_timeline_out_of_range_candidates(
    event_df: pd.DataFrame,
    identity_columns: list[str],
    start_month: pd.Timestamp | None,
    end_month: pd.Timestamp | None,
    effective_date_column: str = "Effective Date",
) -> pd.DataFrame:
    if event_df.empty:
        return event_df.copy()

    candidate_frames = []
    if start_month is not None:
        before_df = event_df[event_df[effective_date_column] < start_month].copy()
        if not before_df.empty:
            before_df = before_df.sort_values(
                identity_columns + [effective_date_column],
                ascending=[True] * len(identity_columns) + [False],
                na_position="last",
            ).drop_duplicates(identity_columns, keep="first")
            before_df["lookup_bucket"] = "before_range"
            before_df["lookup_month_distance"] = before_df[effective_date_column].map(
                lambda value: _month_distance_to_boundary(value, start_month)
            )
            candidate_frames.append(before_df)

    if end_month is not None:
        after_df = event_df[event_df[effective_date_column] > end_month].copy()
        if not after_df.empty:
            after_df = after_df.sort_values(
                identity_columns + [effective_date_column],
                ascending=[True] * len(identity_columns) + [True],
                na_position="last",
            ).drop_duplicates(identity_columns, keep="first")
            after_df["lookup_bucket"] = "after_range"
            after_df["lookup_month_distance"] = after_df[effective_date_column].map(
                lambda value: _month_distance_to_boundary(end_month, value)
            )
            candidate_frames.append(after_df)

    if not candidate_frames:
        return event_df.iloc[0:0].copy()

    outside_df = pd.concat(candidate_frames, ignore_index=True, sort=False)
    outside_df["__lookup_bucket_sort"] = outside_df["lookup_bucket"].map(
        {"before_range": 0, "after_range": 1}
    ).fillna(2)
    outside_df = outside_df.sort_values(
        identity_columns + ["lookup_month_distance", "__lookup_bucket_sort"],
        ascending=[True] * len(identity_columns) + [True, True],
        na_position="last",
    ).drop_duplicates(identity_columns, keep="first")
    return outside_df.drop(columns=["__lookup_bucket_sort"], errors="ignore").reset_index(drop=True)


def _build_provider_timeline_lookup_snapshot(
    change_df: pd.DataFrame,
    provider: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    columns = [
        "Country",
        "Plant",
        "Train",
        "train_key",
        "timeline_direction",
        "timeline_identity_key",
        "First Date",
        "Capacity Change",
        "Original Plant",
        "Original Train",
        "lookup_bucket",
        "lookup_is_out_of_range",
        "display_sort_plant",
        "display_sort_train",
        "display_sort_effective_date",
        "display_sort_direction",
    ]
    event_df = _prepare_provider_timeline_event_rows(change_df, provider)
    if event_df.empty:
        return pd.DataFrame(columns=columns)

    start_month = _normalize_month_date(start_date)
    end_month = _normalize_month_date(end_date)
    if start_month is None:
        start_month = event_df["Effective Date"].min()
    if end_month is None:
        end_month = event_df["Effective Date"].max()

    identity_columns = ["timeline_identity_key"]
    in_range_df = event_df[
        (event_df["Effective Date"] >= start_month)
        & (event_df["Effective Date"] <= end_month)
    ].copy()
    if not in_range_df.empty:
        in_range_df = (
            in_range_df.groupby(identity_columns, as_index=False, dropna=False)
            .agg(
                {
                    "Country": "last",
                    "Plant": "last",
                    "Train": "last",
                    "train_key": "last",
                    "timeline_direction": "last",
                    "Capacity Change": "sum",
                    "Effective Date": "min",
                    "display_sort_plant": "last",
                    "display_sort_train": "last",
                    "Original Plant": _combine_distinct_text_values,
                    "Original Train": _combine_distinct_text_values,
                }
            )
        )
        in_range_df["lookup_bucket"] = "in_range"
        in_range_df["lookup_is_out_of_range"] = False
    else:
        in_range_df = pd.DataFrame(columns=identity_columns + ["lookup_bucket", "lookup_is_out_of_range"])

    outside_df = _select_train_timeline_out_of_range_candidates(
        event_df,
        identity_columns,
        start_month,
        end_month,
    )
    if not outside_df.empty:
        outside_df["lookup_is_out_of_range"] = True
        outside_df = outside_df[
            ~outside_df.set_index(identity_columns).index.isin(
                in_range_df.set_index(identity_columns).index
            )
        ].copy()

    lookup_df = pd.concat([in_range_df, outside_df], ignore_index=True, sort=False)
    if lookup_df.empty:
        return pd.DataFrame(columns=columns)

    lookup_df["First Date"] = pd.to_datetime(
        lookup_df["Effective Date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    lookup_df["display_sort_effective_date"] = pd.to_datetime(
        lookup_df["Effective Date"],
        errors="coerce",
    )
    lookup_df["display_sort_direction"] = lookup_df["timeline_direction"].map(
        {"reduction": 0, "addition": 1}
    ).fillna(2)
    lookup_df["Capacity Change"] = pd.to_numeric(
        lookup_df["Capacity Change"],
        errors="coerce",
    ).round(6)
    return lookup_df[columns]


def _build_train_timeline_df(
    woodmac_change_df: pd.DataFrame,
    ea_change_df: pd.DataFrame,
    aggregate_from_date: str | None = None,
) -> pd.DataFrame:
    columns = [
        "scenario_row_key",
        "timeline_reference_key",
        "Country",
        "Plant",
        "Train",
        "train_key",
        "timeline_identity_key",
        "Woodmac Original Name",
        "Woodmac FID Date",
        "Woodmac First Date",
        "Woodmac Capacity Change",
        "Energy Aspects Original Plant",
        "Energy Aspects Original Train",
        "Energy Aspects First Date",
        "Energy Aspects Capacity Change",
        "timeline_direction",
        "display_sort_plant",
        "display_sort_train",
        "display_sort_effective_date",
        "display_sort_direction",
    ]
    if woodmac_change_df.empty and ea_change_df.empty:
        return pd.DataFrame(columns=columns)

    woodmac_snapshot_df = _build_provider_timeline_snapshot(
        woodmac_change_df,
        "woodmac",
        aggregate_from_date=aggregate_from_date,
    )
    ea_snapshot_df = _build_provider_timeline_snapshot(
        ea_change_df,
        "energy_aspects",
        aggregate_from_date=aggregate_from_date,
    )
    if woodmac_snapshot_df.empty and ea_snapshot_df.empty:
        return pd.DataFrame(columns=columns)

    woodmac_merge_df = woodmac_snapshot_df[
        [
            "timeline_reference_key",
            "train_key",
            "Country",
            "Plant",
            "Train",
            "Original Train",
            "Woodmac FID Date",
            "First Date",
            "Capacity Change",
            "timeline_direction",
            "display_sort_plant",
            "display_sort_train",
            "display_sort_effective_date",
            "display_sort_direction",
        ]
    ].rename(
        columns={
            "Country": "Country_woodmac",
            "Plant": "Plant_woodmac",
            "Train": "Train_woodmac",
            "Original Train": "Woodmac Original Name",
            "First Date": "Woodmac First Date",
            "Capacity Change": "Woodmac Capacity Change",
            "timeline_direction": "timeline_direction_woodmac",
            "display_sort_plant": "display_sort_plant_woodmac",
            "display_sort_train": "display_sort_train_woodmac",
            "display_sort_effective_date": "display_sort_effective_date_woodmac",
            "display_sort_direction": "display_sort_direction_woodmac",
        }
    )
    ea_merge_df = ea_snapshot_df[
        [
            "timeline_reference_key",
            "train_key",
            "Country",
            "Plant",
            "Train",
            "Original Plant",
            "Original Train",
            "First Date",
            "Capacity Change",
            "timeline_direction",
            "display_sort_plant",
            "display_sort_train",
            "display_sort_effective_date",
            "display_sort_direction",
        ]
    ].rename(
        columns={
            "Country": "Country_ea",
            "Plant": "Plant_ea",
            "Train": "Train_ea",
            "Original Plant": "Energy Aspects Original Plant",
            "Original Train": "Energy Aspects Original Train",
            "First Date": "Energy Aspects First Date",
            "Capacity Change": "Energy Aspects Capacity Change",
            "timeline_direction": "timeline_direction_ea",
            "display_sort_plant": "display_sort_plant_ea",
            "display_sort_train": "display_sort_train_ea",
            "display_sort_effective_date": "display_sort_effective_date_ea",
            "display_sort_direction": "display_sort_direction_ea",
        }
    )
    merged_df = pd.merge(
        woodmac_merge_df,
        ea_merge_df,
        on=["timeline_reference_key", "train_key"],
        how="outer",
        validate="one_to_one",
    )

    for column_name in ["Country", "Plant", "Train"]:
        woodmac_values = merged_df.get(f"{column_name}_woodmac", pd.Series(index=merged_df.index, dtype="object"))
        ea_values = merged_df.get(f"{column_name}_ea", pd.Series(index=merged_df.index, dtype="object"))
        merged_df[column_name] = woodmac_values.replace("", pd.NA).combine_first(
            ea_values.replace("", pd.NA)
        ).fillna("")

    merged_df["scenario_row_key"] = merged_df["timeline_reference_key"]
    merged_df["timeline_direction"] = (
        merged_df.get("timeline_direction_woodmac")
        .combine_first(merged_df.get("timeline_direction_ea"))
    )
    merged_df["display_sort_plant"] = (
        merged_df.get("display_sort_plant_woodmac")
        .combine_first(merged_df.get("display_sort_plant_ea"))
        .combine_first(merged_df["Plant"])
    )
    woodmac_train_sort = pd.to_numeric(
        merged_df.get("display_sort_train_woodmac"), errors="coerce"
    )
    ea_train_sort = pd.to_numeric(
        merged_df.get("display_sort_train_ea"), errors="coerce"
    )
    merged_df["display_sort_train"] = woodmac_train_sort.where(
        woodmac_train_sort.notna(), ea_train_sort
    )
    merged_df["display_sort_effective_date"] = pd.concat(
        [
            pd.to_datetime(merged_df.get("display_sort_effective_date_woodmac"), errors="coerce"),
            pd.to_datetime(merged_df.get("display_sort_effective_date_ea"), errors="coerce"),
        ],
        axis=1,
    ).min(axis=1)
    woodmac_direction_sort = pd.to_numeric(
        merged_df.get("display_sort_direction_woodmac"), errors="coerce"
    )
    ea_direction_sort = pd.to_numeric(
        merged_df.get("display_sort_direction_ea"), errors="coerce"
    )
    merged_df["display_sort_direction"] = woodmac_direction_sort.where(
        woodmac_direction_sort.notna(), ea_direction_sort
    )
    merged_df = _ensure_train_timeline_identity_key(merged_df)

    for column_name in columns:
        if column_name not in merged_df.columns:
            merged_df[column_name] = None

    merged_df["__train_blank_sort"] = merged_df["Train"].fillna("").astype(str).eq("").astype(int)
    merged_df = merged_df.sort_values(
        [
            "Country",
            "display_sort_plant",
            "__train_blank_sort",
            "display_sort_train",
            "Train",
            "display_sort_effective_date",
            "display_sort_direction",
        ],
        ascending=[True, True, False, True, True, True, True],
        na_position="last",
    ).drop(columns=["__train_blank_sort"], errors="ignore").reset_index(drop=True)
    return merged_df[columns]


def _get_capacity_scenario_row_columns() -> list[str]:
    return [
        "scenario_row_key",
        "terminal_key",
        "train_key",
        "country_name",
        "plant_name",
        "train_label",
        "base_provider",
        "base_first_date",
        "base_capacity_mtpa",
        "scenario_first_date",
        "scenario_capacity_mtpa",
        "scenario_note",
        "display_sort_plant",
        "display_sort_train",
    ]


def _prepare_capacity_scenario_rows_df(rows_df: pd.DataFrame | None) -> pd.DataFrame:
    columns = _get_capacity_scenario_row_columns()
    if rows_df is None or rows_df.empty:
        return pd.DataFrame(columns=columns)

    prepared_df = rows_df.copy()
    for column_name in columns:
        if column_name not in prepared_df.columns:
            prepared_df[column_name] = None

    for column_name in [
        "scenario_row_key",
        "terminal_key",
        "train_key",
        "country_name",
        "plant_name",
        "train_label",
        "base_provider",
        "display_sort_plant",
    ]:
        prepared_df[column_name] = (
            prepared_df[column_name]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    for column_name in ["terminal_key", "train_key"]:
        prepared_df[column_name] = prepared_df[column_name].replace("", None)
    prepared_df["scenario_note"] = (
        prepared_df["scenario_note"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    for column_name in ["base_first_date", "scenario_first_date"]:
        prepared_df[column_name] = pd.to_datetime(
            prepared_df[column_name],
            errors="coerce",
        ).dt.to_period("M").dt.to_timestamp()

    for column_name in ["base_capacity_mtpa", "scenario_capacity_mtpa", "display_sort_train"]:
        prepared_df[column_name] = pd.to_numeric(
            prepared_df[column_name],
            errors="coerce",
        )

    prepared_df = prepared_df.drop_duplicates(
        subset=["scenario_row_key"],
        keep="last",
    ).reset_index(drop=True)
    return prepared_df[columns]


def _build_capacity_scenario_rows_from_internal_rows(
    source_rows_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = _get_capacity_scenario_row_columns()
    source_rows_df = _prepare_capacity_scenario_rows_df(source_rows_df)
    if source_rows_df.empty:
        return pd.DataFrame(columns=columns)

    duplicated_rows_df = source_rows_df.copy()
    duplicated_rows_df["base_provider"] = "internal_scenario"
    duplicated_rows_df["base_first_date"] = duplicated_rows_df["scenario_first_date"]
    duplicated_rows_df["base_capacity_mtpa"] = duplicated_rows_df["scenario_capacity_mtpa"]
    return duplicated_rows_df[columns]


def _build_new_capacity_scenario_rows(
    base_type: str,
    train_raw_df: pd.DataFrame,
    ea_raw_df: pd.DataFrame,
    source_scenario_id: int | None = None,
    aggregate_from_date: str | None = None,
) -> pd.DataFrame:
    normalized_base_type = (base_type or "").strip().casefold()
    if normalized_base_type == "woodmac":
        return _build_provider_scenario_rows_from_change_log(
            _build_train_change_log(
                train_raw_df,
                None,
                "rest_of_world",
                None,
                None,
            ),
            "woodmac",
            aggregate_from_date=aggregate_from_date,
        )

    if normalized_base_type == "energy_aspects":
        return _build_provider_scenario_rows_from_change_log(
            _build_ea_change_log(
                ea_raw_df,
                None,
                "rest_of_world",
                None,
                None,
            ),
            "energy_aspects",
            aggregate_from_date=aggregate_from_date,
        )

    if normalized_base_type == "internal_scenario":
        if source_scenario_id is None:
            return pd.DataFrame(columns=_get_capacity_scenario_row_columns())
        return _build_capacity_scenario_rows_from_internal_rows(
            fetch_capacity_scenario_rows(int(source_scenario_id), engine)
        )

    return pd.DataFrame(columns=_get_capacity_scenario_row_columns())


def _validate_capacity_scenario_mapping_preflight(
    rows_df: pd.DataFrame,
    base_type: str,
) -> None:
    if rows_df is None or rows_df.empty:
        return
    train_keys = rows_df.get("train_key", pd.Series(None, index=rows_df.index))
    unresolved_mask = train_keys.isna() | train_keys.astype(str).str.strip().eq("")
    unresolved_df = rows_df.loc[unresolved_mask].copy()
    if unresolved_df.empty:
        return

    examples = []
    for _, row in unresolved_df.head(5).iterrows():
        effective_date = pd.to_datetime(
            row.get("base_first_date"), errors="coerce"
        )
        date_text = (
            effective_date.strftime("%Y-%m-%d")
            if pd.notna(effective_date)
            else "unknown date"
        )
        examples.append(
            " / ".join(
                [
                    str(row.get("base_provider") or base_type or "provider"),
                    str(row.get("country_name") or "unknown country"),
                    str(row.get("plant_name") or "unknown terminal"),
                    str(row.get("train_label") or "unknown train"),
                    date_text,
                ]
            )
        )
    raise ValueError(
        f"Scenario creation is blocked by {len(unresolved_df)} provider event(s) "
        "without a canonical train mapping: "
        + "; ".join(examples)
        + ". Resolve these rows in Unmapped Provider Trains and retry."
    )


def _attach_and_validate_capacity_scenario_keys(
    rows_df: pd.DataFrame,
    base_type: str,
) -> pd.DataFrame:
    normalized_base_type = str(base_type or "").strip().casefold()
    if normalized_base_type in {"woodmac", "energy_aspects"}:
        _validate_capacity_scenario_mapping_preflight(rows_df, normalized_base_type)
    resolved_rows_df = attach_registry_keys_to_capacity_rows(rows_df, engine)
    _validate_capacity_scenario_mapping_preflight(
        resolved_rows_df,
        normalized_base_type,
    )
    return resolved_rows_df


def _build_internal_scenario_monthly_schedule(
    scenario_rows_df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    empty_df = pd.DataFrame(columns=["month", "country_name", "total_mmtpa"])
    scenario_rows_df = _prepare_capacity_scenario_rows_df(scenario_rows_df)
    if scenario_rows_df.empty:
        return empty_df

    schedule_rows_df = scenario_rows_df.copy()
    schedule_rows_df = schedule_rows_df[
        schedule_rows_df["scenario_first_date"].notna()
        & schedule_rows_df["scenario_capacity_mtpa"].fillna(0.0).round(6).ne(0)
    ].copy()
    if schedule_rows_df.empty:
        return empty_df

    start_month = _normalize_month_date(start_date) or schedule_rows_df["scenario_first_date"].min()
    end_month = _normalize_month_date(end_date) or schedule_rows_df["scenario_first_date"].max()
    if start_month is None or end_month is None or start_month > end_month:
        return empty_df

    expanded_frames = []
    for row in schedule_rows_df.itertuples(index=False):
        active_start = max(pd.Timestamp(row.scenario_first_date), start_month)
        if active_start > end_month:
            continue
        expanded_months = pd.date_range(active_start, end_month, freq="MS")
        if expanded_months.empty:
            continue
        expanded_frames.append(
            pd.DataFrame(
                {
                    "month": expanded_months,
                    "country_name": row.country_name,
                    "total_mmtpa": float(row.scenario_capacity_mtpa),
                }
            )
        )

    if not expanded_frames:
        return empty_df

    schedule_df = pd.concat(expanded_frames, ignore_index=True)
    schedule_df = (
        schedule_df.groupby(["month", "country_name"], as_index=False)["total_mmtpa"]
        .sum()
        .sort_values(["month", "country_name"])
        .reset_index(drop=True)
    )
    return schedule_df


def _get_cached_internal_scenario_monthly_schedule(
    scenario_rows_df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    prepared_rows_df = _prepare_capacity_scenario_rows_df(scenario_rows_df)
    scenario_payload = _serialize_dataframe(prepared_rows_df)
    return _get_cached_capacity_view_payload(
        "internal_scenario_schedule",
        {
            "scenario": hashlib.sha256(
                str(scenario_payload or "").encode("utf-8")
            ).hexdigest(),
            "start_date": start_date,
            "end_date": end_date,
        },
        lambda: _build_internal_scenario_monthly_schedule(
            prepared_rows_df,
            start_date,
            end_date,
        ),
    )


def _extract_total_yearly_capacity_series(
    raw_df: pd.DataFrame,
    series_label: str,
) -> pd.DataFrame:
    columns = ["Year", series_label]
    if raw_df.empty:
        return pd.DataFrame(columns=columns)

    yearly_matrix = _apply_capacity_time_view(
        _rename_total_column(
            build_export_flow_matrix(
                raw_df,
                None,
                "rest_of_world",
            )
        ),
        "yearly",
    )
    if yearly_matrix.empty or "Total MTPA" not in yearly_matrix.columns:
        return pd.DataFrame(columns=columns)

    series_df = yearly_matrix[["Month", "Total MTPA"]].copy()
    series_df["Year"] = series_df["Month"].fillna("").astype(str).str.strip()
    series_df = (
        series_df[series_df["Year"].ne("")]
        .drop_duplicates(subset=["Year"], keep="last")
        .rename(columns={"Total MTPA": series_label})
    )
    series_df[series_label] = pd.to_numeric(series_df[series_label], errors="coerce").round(2)
    year_sort = pd.to_numeric(series_df["Year"], errors="coerce")
    series_df = (
        series_df.assign(__year_sort=year_sort)
        .sort_values(["__year_sort", "Year"], ascending=[True, True], na_position="last")
        .drop(columns=["Month", "__year_sort"], errors="ignore")
        .reset_index(drop=True)
    )
    return series_df[columns]


def _build_yearly_capacity_comparison_df(
    woodmac_raw_df: pd.DataFrame,
    ea_raw_df: pd.DataFrame,
    scenario_rows_df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    columns = [
        "Year",
        "Internal Scenario",
        "Woodmac",
        "Energy Aspects",
        "Delta vs Woodmac",
        "Delta vs Energy Aspects",
    ]

    woodmac_yearly_df = _extract_total_yearly_capacity_series(
        _filter_by_date_range(woodmac_raw_df, start_date, end_date),
        "Woodmac",
    )
    ea_yearly_df = _extract_total_yearly_capacity_series(
        _build_ea_capacity_schedule(ea_raw_df, start_date, end_date),
        "Energy Aspects",
    )
    internal_yearly_df = _extract_total_yearly_capacity_series(
        _get_cached_internal_scenario_monthly_schedule(
            scenario_rows_df,
            start_date,
            end_date,
        ),
        "Internal Scenario",
    )

    if woodmac_yearly_df.empty or ea_yearly_df.empty:
        return pd.DataFrame(columns=columns)

    woodmac_years = set(woodmac_yearly_df["Year"].tolist())
    ea_years = set(ea_yearly_df["Year"].tolist())
    common_years = sorted(
        woodmac_years.intersection(ea_years),
        key=lambda value: (pd.to_numeric([value], errors="coerce")[0], str(value)),
    )
    if not common_years:
        return pd.DataFrame(columns=columns)

    comparison_df = pd.DataFrame({"Year": common_years})
    comparison_df = comparison_df.merge(woodmac_yearly_df, on="Year", how="left")
    comparison_df = comparison_df.merge(ea_yearly_df, on="Year", how="left")
    comparison_df = comparison_df.merge(internal_yearly_df, on="Year", how="left")

    internal_values = pd.to_numeric(comparison_df["Internal Scenario"], errors="coerce")
    woodmac_values = pd.to_numeric(comparison_df["Woodmac"], errors="coerce")
    ea_values = pd.to_numeric(comparison_df["Energy Aspects"], errors="coerce")
    comparison_df["Delta vs Woodmac"] = (
        internal_values - woodmac_values
    ).where(internal_values.notna()).round(2)
    comparison_df["Delta vs Energy Aspects"] = (
        internal_values - ea_values
    ).where(internal_values.notna()).round(2)

    return comparison_df[columns]


def _build_train_timeline_grid_df_for_scope(
    train_raw_df: pd.DataFrame,
    ea_raw_df: pd.DataFrame,
    scenario_rows_df: pd.DataFrame,
    selected_countries: list[str] | None,
    other_countries_mode: str,
    start_date: str | None,
    end_date: str | None,
    aggregate_from_date: str | None = None,
    prepared_provider_data: dict[str, object] | None = None,
) -> pd.DataFrame:
    scenario_rows_df = _prepare_capacity_scenario_rows_df(scenario_rows_df)
    if prepared_provider_data is None:
        resolved_countries = _resolve_capacity_country_scope(
            train_raw_df,
            ea_raw_df,
            selected_countries,
            scenario_rows_df=scenario_rows_df,
            start_date=start_date,
            end_date=end_date,
        )
        woodmac_change_df = _build_train_change_log(
            train_raw_df,
            resolved_countries,
            other_countries_mode,
            start_date,
            end_date,
        )
        ea_change_df = _build_ea_change_log(
            ea_raw_df,
            resolved_countries,
            other_countries_mode,
            start_date,
            end_date,
        )
        woodmac_lookup_df = _build_provider_timeline_lookup_snapshot(
            _build_train_change_log(
                train_raw_df,
                resolved_countries,
                other_countries_mode,
                None,
                None,
            ),
            "woodmac",
            start_date,
            end_date,
        )
        ea_lookup_df = _build_provider_timeline_lookup_snapshot(
            _build_ea_change_log(
                ea_raw_df,
                resolved_countries,
                other_countries_mode,
                None,
                None,
            ),
            "energy_aspects",
            start_date,
            end_date,
        )
    else:
        resolved_countries = list(prepared_provider_data["resolved_countries"])
        woodmac_change_df = prepared_provider_data["woodmac_change_df"]
        ea_change_df = prepared_provider_data["ea_change_df"]
        woodmac_lookup_df = prepared_provider_data["woodmac_lookup_df"]
        ea_lookup_df = prepared_provider_data["ea_lookup_df"]

    timeline_df = (
        prepared_provider_data["timeline_df"]
        if prepared_provider_data is not None
        and _normalize_month_date(aggregate_from_date) == _normalize_month_date(start_date)
        else _build_train_timeline_df(
            woodmac_change_df,
            ea_change_df,
            aggregate_from_date=aggregate_from_date,
        )
    )
    visible_scenario_rows_df = (
        _filter_visible_capacity_scenario_rows(
            scenario_rows_df,
            resolved_countries,
            other_countries_mode,
            start_date,
            end_date,
        )
        if not scenario_rows_df.empty
        else pd.DataFrame()
    )
    scenario_lookup_df = (
        _build_internal_scenario_lookup_snapshot(
            scenario_rows_df,
            resolved_countries,
            other_countries_mode,
            start_date,
            end_date,
        )
        if not scenario_rows_df.empty
        else pd.DataFrame()
    )
    return _build_train_timeline_grid_rows(
        timeline_df,
        visible_scenario_rows_df,
        aggregate_from_date=aggregate_from_date,
        woodmac_lookup_df=woodmac_lookup_df,
        ea_lookup_df=ea_lookup_df,
        scenario_lookup_df=scenario_lookup_df,
    )


def _build_provider_capacity_discrepancy_df_from_timeline_grid(
    provider: str,
    grid_df: pd.DataFrame,
    top_n: int | None = None,
) -> pd.DataFrame:
    discrepancy_df, provider_config = _prepare_provider_discrepancy_grid_df(
        provider,
        grid_df,
    )
    if discrepancy_df.empty:
        return pd.DataFrame(columns=provider_config["capacity_columns"])

    provider_capacity_column = str(provider_config["provider_capacity_change_column"])
    entity_input_column = str(provider_config["entity_input_column"])
    entity_output_column = str(provider_config["entity_output_column"])
    provider_date_column = str(provider_config["provider_date_column"])
    provider_capacity_display_column = str(provider_config["provider_capacity_display_column"])

    provider_capacity_values = pd.to_numeric(
        discrepancy_df.get(provider_capacity_column),
        errors="coerce",
    )
    scenario_capacity_values = pd.to_numeric(
        discrepancy_df.get("Scenario Capacity"),
        errors="coerce",
    )
    discrepancy_df = discrepancy_df[
        provider_capacity_values.notna() & scenario_capacity_values.notna()
    ].copy()
    if discrepancy_df.empty:
        return pd.DataFrame(columns=provider_config["capacity_columns"])

    provider_capacity_values = pd.to_numeric(
        discrepancy_df.get(provider_capacity_column),
        errors="coerce",
    )
    scenario_capacity_values = pd.to_numeric(
        discrepancy_df.get("Scenario Capacity"),
        errors="coerce",
    )
    discrepancy_df["Abs Capacity Delta"] = (
        scenario_capacity_values.fillna(0.0) - provider_capacity_values.fillna(0.0)
    ).abs().round(2)
    discrepancy_df = discrepancy_df[
        discrepancy_df["Abs Capacity Delta"].fillna(0.0).round(6).gt(0)
    ].copy()
    if discrepancy_df.empty:
        return pd.DataFrame(columns=provider_config["capacity_columns"])

    discrepancy_df = discrepancy_df.sort_values(
        [
            "Abs Capacity Delta",
            "Country",
            entity_input_column,
            "__train_sort",
            "Train",
            "__direction_sort",
            "__provider_date_sort",
            "__scenario_date_sort",
        ],
        ascending=[False, True, True, True, True, True, True, True],
        na_position="last",
    )
    if top_n is not None:
        discrepancy_df = discrepancy_df.head(top_n)
    discrepancy_df = discrepancy_df.reset_index(drop=True)

    discrepancy_df[provider_date_column] = discrepancy_df[provider_date_column].dt.strftime(
        "%Y-%m-%d"
    )
    discrepancy_df[provider_date_column] = discrepancy_df[provider_date_column].where(
        discrepancy_df[provider_date_column].notna(),
        None,
    )
    discrepancy_df["Scenario First Date"] = discrepancy_df["Scenario First Date"].dt.strftime(
        "%Y-%m-%d"
    )
    discrepancy_df["Scenario First Date"] = discrepancy_df["Scenario First Date"].where(
        discrepancy_df["Scenario First Date"].notna(),
        None,
    )
    discrepancy_df["Scenario Capacity"] = pd.to_numeric(
        discrepancy_df.get("Scenario Capacity"),
        errors="coerce",
    ).round(2)
    discrepancy_df["Scenario Capacity"] = discrepancy_df["Scenario Capacity"].where(
        discrepancy_df["Scenario Capacity"].notna(),
        None,
    )
    discrepancy_df[provider_capacity_display_column] = pd.to_numeric(
        discrepancy_df.get(provider_capacity_column),
        errors="coerce",
    ).round(2)
    discrepancy_df[provider_capacity_display_column] = discrepancy_df[
        provider_capacity_display_column
    ].where(
        discrepancy_df[provider_capacity_display_column].notna(),
        None,
    )
    if entity_output_column != entity_input_column:
        discrepancy_df[entity_output_column] = discrepancy_df[entity_input_column]

    discrepancy_df = discrepancy_df.drop(
        columns=[
            "__train_sort",
            "__direction_sort",
            "__provider_date_sort",
            "__scenario_date_sort",
        ],
        errors="ignore",
    )
    return discrepancy_df[list(provider_config["capacity_columns"])]


def _prepare_provider_discrepancy_grid_df(
    provider: str,
    grid_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    provider_config = _get_provider_discrepancy_config(provider)
    if grid_df.empty:
        return pd.DataFrame(), provider_config

    discrepancy_df = grid_df.copy()
    entity_input_column = str(provider_config["entity_input_column"])
    provider_date_column = str(provider_config["provider_date_column"])
    provider_capacity_column = str(provider_config["provider_capacity_change_column"])

    discrepancy_df["Country"] = (
        discrepancy_df.get("Country", pd.Series("", index=discrepancy_df.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    discrepancy_df[entity_input_column] = (
        discrepancy_df.get(
            entity_input_column,
            pd.Series("", index=discrepancy_df.index),
        )
        .fillna("")
        .astype(str)
        .str.strip()
    )
    discrepancy_df["Train"] = discrepancy_df.get(
        "Train",
        pd.Series("", index=discrepancy_df.index),
    ).map(_format_train_label)
    discrepancy_df[provider_date_column] = pd.to_datetime(
        discrepancy_df.get(provider_date_column),
        errors="coerce",
    )
    discrepancy_df["Scenario First Date"] = pd.to_datetime(
        discrepancy_df.get("Scenario First Date"),
        errors="coerce",
    )
    discrepancy_df[provider_capacity_column] = pd.to_numeric(
        discrepancy_df.get(provider_capacity_column),
        errors="coerce",
    )
    discrepancy_df["Scenario Capacity"] = pd.to_numeric(
        discrepancy_df.get("Scenario Capacity"),
        errors="coerce",
    )

    discrepancy_df["__train_sort"] = pd.to_numeric(
        discrepancy_df["Train"],
        errors="coerce",
    )
    discrepancy_df["__direction_sort"] = discrepancy_df.get("timeline_direction").map(
        {"reduction": 0, "addition": 1}
    ).fillna(2)
    discrepancy_df["__provider_date_sort"] = pd.to_datetime(
        discrepancy_df[provider_date_column],
        errors="coerce",
    )
    discrepancy_df["__scenario_date_sort"] = pd.to_datetime(
        discrepancy_df["Scenario First Date"],
        errors="coerce",
    )
    return discrepancy_df, provider_config


def _build_provider_timeline_discrepancy_df_from_timeline_grid(
    provider: str,
    grid_df: pd.DataFrame,
    top_n: int | None = None,
) -> pd.DataFrame:
    discrepancy_df, provider_config = _prepare_provider_discrepancy_grid_df(
        provider,
        grid_df,
    )
    if discrepancy_df.empty:
        return pd.DataFrame(columns=provider_config["timeline_columns"])

    entity_input_column = str(provider_config["entity_input_column"])
    entity_output_column = str(provider_config["entity_output_column"])
    provider_date_column = str(provider_config["provider_date_column"])

    provider_months = discrepancy_df[provider_date_column].map(_normalize_month_date)
    scenario_months = discrepancy_df["Scenario First Date"].map(_normalize_month_date)
    provider_date_present = provider_months.notna()
    scenario_date_present = scenario_months.notna()

    timeline_df = discrepancy_df[provider_date_present & scenario_date_present].copy()
    if timeline_df.empty:
        return pd.DataFrame(columns=provider_config["timeline_columns"])

    timeline_df["__provider_month"] = provider_months[provider_date_present & scenario_date_present]
    timeline_df["__scenario_month"] = scenario_months[provider_date_present & scenario_date_present]
    timeline_df["Abs Timeline Delta (Months)"] = [
        _month_distance_to_boundary(provider_month, scenario_month)
        for provider_month, scenario_month in zip(
            timeline_df["__provider_month"],
            timeline_df["__scenario_month"],
        )
    ]
    timeline_df = timeline_df[
        pd.to_numeric(timeline_df["Abs Timeline Delta (Months)"], errors="coerce")
        .fillna(0)
        .astype(int)
        .gt(0)
    ].copy()
    if timeline_df.empty:
        return pd.DataFrame(columns=provider_config["timeline_columns"])

    timeline_df = timeline_df.sort_values(
        [
            "Abs Timeline Delta (Months)",
            "Country",
            entity_input_column,
            "__train_sort",
            "Train",
            "__direction_sort",
            "__provider_month",
            "__scenario_month",
        ],
        ascending=[False, True, True, True, True, True, True, True],
        na_position="last",
    )
    if top_n is not None:
        timeline_df = timeline_df.head(top_n)
    timeline_df = timeline_df.reset_index(drop=True)

    timeline_df[provider_date_column] = timeline_df[provider_date_column].dt.strftime("%Y-%m-%d")
    timeline_df[provider_date_column] = timeline_df[provider_date_column].where(
        timeline_df[provider_date_column].notna(),
        None,
    )
    timeline_df["Scenario First Date"] = timeline_df["Scenario First Date"].dt.strftime("%Y-%m-%d")
    timeline_df["Scenario First Date"] = timeline_df["Scenario First Date"].where(
        timeline_df["Scenario First Date"].notna(),
        None,
    )
    timeline_df["Abs Timeline Delta (Months)"] = pd.to_numeric(
        timeline_df["Abs Timeline Delta (Months)"],
        errors="coerce",
    ).astype("Int64")
    if entity_output_column != entity_input_column:
        timeline_df[entity_output_column] = timeline_df[entity_input_column]

    timeline_df = timeline_df.drop(
        columns=[
            "__train_sort",
            "__direction_sort",
            "__provider_date_sort",
            "__scenario_date_sort",
            "__provider_month",
            "__scenario_month",
        ],
        errors="ignore",
    )
    return timeline_df[list(provider_config["timeline_columns"])]


def _build_provider_missing_internal_scenario_df_from_timeline_grid(
    provider: str,
    grid_df: pd.DataFrame,
) -> pd.DataFrame:
    discrepancy_df, provider_config = _prepare_provider_discrepancy_grid_df(
        provider,
        grid_df,
    )
    if discrepancy_df.empty:
        return pd.DataFrame(columns=provider_config["missing_columns"])

    entity_input_column = str(provider_config["entity_input_column"])
    entity_output_column = str(provider_config["entity_output_column"])
    provider_date_column = str(provider_config["provider_date_column"])
    provider_capacity_column = str(provider_config["provider_capacity_change_column"])
    provider_capacity_display_column = str(provider_config["provider_capacity_display_column"])

    provider_row_present = (
        discrepancy_df[provider_date_column].notna()
        | pd.to_numeric(discrepancy_df.get(provider_capacity_column), errors="coerce").notna()
    )
    scenario_row_missing = (
        discrepancy_df["Scenario First Date"].isna()
        & pd.to_numeric(discrepancy_df.get("Scenario Capacity"), errors="coerce").isna()
    )
    missing_df = discrepancy_df[provider_row_present & scenario_row_missing].copy()
    if missing_df.empty:
        return pd.DataFrame(columns=provider_config["missing_columns"])

    missing_df = missing_df.sort_values(
        [
            "Country",
            entity_input_column,
            "__train_sort",
            "Train",
            "__direction_sort",
            "__provider_date_sort",
        ],
        ascending=[True, True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    missing_df[provider_date_column] = missing_df[provider_date_column].dt.strftime("%Y-%m-%d")
    missing_df[provider_date_column] = missing_df[provider_date_column].where(
        missing_df[provider_date_column].notna(),
        None,
    )
    missing_df[provider_capacity_display_column] = pd.to_numeric(
        missing_df.get(provider_capacity_column),
        errors="coerce",
    ).round(2)
    missing_df[provider_capacity_display_column] = missing_df[
        provider_capacity_display_column
    ].where(
        missing_df[provider_capacity_display_column].notna(),
        None,
    )
    if entity_output_column != entity_input_column:
        missing_df[entity_output_column] = missing_df[entity_input_column]

    missing_df = missing_df.drop(
        columns=[
            "__train_sort",
            "__direction_sort",
            "__provider_date_sort",
            "__scenario_date_sort",
        ],
        errors="ignore",
    )
    return missing_df[list(provider_config["missing_columns"])]


def _build_yearly_provider_discrepancy_payloads(
    woodmac_raw_df: pd.DataFrame,
    ea_raw_df: pd.DataFrame,
    scenario_rows_df: pd.DataFrame,
    selected_countries: list[str] | None,
    other_countries_mode: str,
    start_date: str | None,
    end_date: str | None,
    prepared_provider_data: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    timeline_grid_df = _build_train_timeline_grid_df_for_scope(
        woodmac_raw_df,
        ea_raw_df,
        scenario_rows_df,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
        aggregate_from_date=start_date,
        prepared_provider_data=prepared_provider_data,
    )
    payloads: dict[str, dict[str, object]] = {}
    for provider in ["woodmac", "energy_aspects"]:
        payloads[provider] = {
            "capacity_df": _build_provider_capacity_discrepancy_df_from_timeline_grid(
                provider,
                timeline_grid_df,
            ),
            "timeline_df": _build_provider_timeline_discrepancy_df_from_timeline_grid(
                provider,
                timeline_grid_df,
            ),
            "missing_df": _build_provider_missing_internal_scenario_df_from_timeline_grid(
                provider,
                timeline_grid_df,
            ),
        }

    return payloads


def _build_internal_scenario_change_log(
    scenario_rows_df: pd.DataFrame,
    selected_countries: list[str] | None,
    other_countries_mode: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    columns = [
        "terminal_key",
        "train_key",
        "Canonical Train Label",
        "Effective Date",
        "Country",
        "Plant",
        "Train",
        INTERNAL_SCENARIO_ADDS_COLUMN,
        INTERNAL_SCENARIO_REDUCTIONS_COLUMN,
        INTERNAL_SCENARIO_NET_COLUMN,
        "Internal Scenario Activity Abs",
    ]
    scenario_rows_df = _prepare_capacity_scenario_rows_df(scenario_rows_df)
    if scenario_rows_df.empty:
        return pd.DataFrame(columns=columns)

    change_df = scenario_rows_df.copy()
    change_df = change_df[
        change_df["scenario_first_date"].notna()
        & change_df["scenario_capacity_mtpa"].fillna(0.0).round(6).ne(0)
    ].copy()
    if change_df.empty:
        return pd.DataFrame(columns=columns)

    if selected_countries:
        if other_countries_mode == "exclude":
            change_df = change_df[change_df["country_name"].isin(selected_countries)].copy()
    elif other_countries_mode == "exclude":
        return pd.DataFrame(columns=columns)

    start_month = _normalize_month_date(start_date)
    end_month = _normalize_month_date(end_date)
    if start_month is not None:
        change_df = change_df[change_df["scenario_first_date"] >= start_month].copy()
    if end_month is not None:
        change_df = change_df[change_df["scenario_first_date"] <= end_month].copy()
    if change_df.empty:
        return pd.DataFrame(columns=columns)

    change_df["Effective Date"] = change_df["scenario_first_date"].dt.strftime("%Y-%m-%d")
    change_df["Canonical Train Label"] = change_df["train_label"]
    change_df["Country"] = change_df["country_name"]
    change_df["Plant"] = change_df["plant_name"]
    change_df["Train"] = change_df["train_label"].map(_canonical_train_number).astype("Int64")
    change_df[INTERNAL_SCENARIO_ADDS_COLUMN] = change_df["scenario_capacity_mtpa"].clip(lower=0).round(2)
    change_df[INTERNAL_SCENARIO_REDUCTIONS_COLUMN] = change_df["scenario_capacity_mtpa"].where(
        change_df["scenario_capacity_mtpa"] < 0,
        0.0,
    ).round(2)
    change_df[INTERNAL_SCENARIO_NET_COLUMN] = change_df["scenario_capacity_mtpa"].round(2)
    change_df["Internal Scenario Activity Abs"] = change_df["scenario_capacity_mtpa"].abs().round(2)

    change_df = change_df.sort_values(
        ["scenario_first_date", "Country", "Plant", "Train"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return change_df[columns]


def _prepare_internal_period_change_df(
    change_df: pd.DataFrame,
    time_view: str,
) -> pd.DataFrame:
    columns = [
        "__plant_identity",
        "__train_identity",
        "__period_start",
        "Effective Date",
        "Country",
        "Plant",
        "Train",
        INTERNAL_SCENARIO_ADDS_COLUMN,
        INTERNAL_SCENARIO_REDUCTIONS_COLUMN,
        INTERNAL_SCENARIO_NET_COLUMN,
        "Internal Scenario Activity Abs",
    ]
    if change_df.empty:
        return pd.DataFrame(columns=columns)

    internal_df = _apply_train_change_time_view(change_df, time_view)
    internal_df = (
        internal_df.groupby(
            [
                "__plant_identity",
                "__train_identity",
                "__period_start",
                "Effective Date",
                "Country",
                "Plant",
                "Train",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            {
                INTERNAL_SCENARIO_ADDS_COLUMN: "sum",
                INTERNAL_SCENARIO_REDUCTIONS_COLUMN: "sum",
                INTERNAL_SCENARIO_NET_COLUMN: "sum",
                "Internal Scenario Activity Abs": "sum",
            }
        )
    )
    internal_df["__train_sort"] = internal_df["Train"].map(_canonical_train_number)
    internal_df = internal_df.sort_values(
        ["__period_start", "Country", "Plant", "__train_sort", "Internal Scenario Activity Abs"],
        ascending=[True, True, True, True, False],
    ).drop(columns=["__train_sort"], errors="ignore").reset_index(drop=True)
    return internal_df[columns]


def _serialize_capacity_scenario_options(options_df: pd.DataFrame) -> list[dict]:
    if options_df is None or options_df.empty:
        return []

    serialized_df = options_df.copy()
    for column_name in [
        "created_at",
        "updated_at",
        "current_snapshot_timestamp_utc",
    ]:
        if column_name in serialized_df.columns:
            serialized_df[column_name] = serialized_df[column_name].map(_serialize_timestamp)

    return serialized_df.to_dict("records")


def _get_capacity_scenario_option_map(options_data: list[dict] | None) -> dict[int, dict]:
    option_map: dict[int, dict] = {}
    for option in options_data or []:
        scenario_id = pd.to_numeric(option.get("scenario_id"), errors="coerce")
        if pd.isna(scenario_id):
            continue
        option_map[int(scenario_id)] = option
    return option_map


def _get_capacity_scenario_valid_ids_and_base_case_id(
    options_data: list[dict] | None,
) -> tuple[list[int], int | None]:
    valid_ids: list[int] = []
    base_case_id: int | None = None

    for option in options_data or []:
        scenario_id = pd.to_numeric(option.get("scenario_id"), errors="coerce")
        if pd.isna(scenario_id):
            continue

        normalized_id = int(scenario_id)
        valid_ids.append(normalized_id)
        scenario_name = " ".join(str(option.get("scenario_name") or "").strip().split())
        if scenario_name.casefold() == "base case":
            base_case_id = normalized_id

    return valid_ids, base_case_id


def _build_capacity_scenario_badge_text(
    selected_scenario_id: int | None,
    options_data: list[dict] | None,
) -> str:
    if selected_scenario_id is None:
        return "No internal scenario selected"

    option_map = _get_capacity_scenario_option_map(options_data)
    option = option_map.get(int(selected_scenario_id))
    if not option:
        return "No internal scenario selected"

    return str(option.get("scenario_name") or "Scenario")


def _build_capacity_scenario_dirty_label(
    dirty_data: dict | None,
) -> str:
    if (
        dirty_data
        and dirty_data.get("dirty")
        and str(dirty_data.get("source") or "").casefold() in {"grid", "working"}
    ):
        return "Unsaved edits"
    return "Saved"


def _build_capacity_scenario_message(
    text_value: str,
    tone: str = "neutral",
) -> html.Div:
    tone_styles = {
        "success": {"color": "#166534", "backgroundColor": "#f0fdf4", "borderColor": "#bbf7d0"},
        "warning": {"color": "#9a3412", "backgroundColor": "#fff7ed", "borderColor": "#fed7aa"},
        "error": {"color": "#991b1b", "backgroundColor": "#fef2f2", "borderColor": "#fecaca"},
        "neutral": {"color": "#334155", "backgroundColor": "#f8fafc", "borderColor": "#e2e8f0"},
    }
    style = tone_styles.get(tone, tone_styles["neutral"])
    return html.Div(
        text_value,
        style={
            "padding": "10px 12px",
            "borderRadius": "12px",
            "border": f"1px solid {style['borderColor']}",
            "backgroundColor": style["backgroundColor"],
            "color": style["color"],
            "fontSize": "12px",
            "fontWeight": "600",
        },
    )


def _run_capacity_ramp_forecast(scenario_id: int | None) -> dict:
    if scenario_id is None:
        return {
            "outcome": "failed",
            "run_status": None,
            "message": "No internal scenario selected for ramp generation.",
        }
    try:
        return generate_ramp_forecast_for_capacity_scenario(int(scenario_id), engine)
    except Exception as exc:
        return {
            "outcome": "failed",
            "run_status": None,
            "message": str(exc),
            "selector_token": f"capacity_ramp_{int(scenario_id)}",
        }


def _format_capacity_ramp_result_message(result: dict | None) -> tuple[str, str]:
    result = result or {}
    selector_token = result.get("selector_token") or "capacity_ramp_<scenario_id>"
    outcome = str(result.get("outcome") or "").casefold()
    run_id = result.get("run_id")
    run_suffix = f" Run {run_id}." if run_id is not None else ""

    if outcome == "published":
        records = pd.to_numeric(result.get("records_published"), errors="coerce")
        records_text = f" with {int(records):,} row(s)" if pd.notna(records) else ""
        return (
            f"Ramp forecast run published for {selector_token}{records_text}.{run_suffix}",
            "success",
        )

    if outcome == "completed":
        rows = pd.to_numeric(result.get("monthly_row_count"), errors="coerce")
        rows_text = f" with {int(rows):,} monthly row(s)" if pd.notna(rows) else ""
        blocking_value = pd.to_numeric(
            result.get("blocking_qa_count"), errors="coerce"
        )
        blocking_count = int(blocking_value) if pd.notna(blocking_value) else 0
        if blocking_count:
            return (
                f"Ramp forecast completed{rows_text} with {blocking_count} blocking QA issue(s). "
                f"The analytical snapshot was saved but cannot be published.{run_suffix}",
                "warning",
            )
        return (
            f"Ramp forecast completed{rows_text} and is ready for review and publishing.{run_suffix}",
            "success",
        )

    if outcome == "blocked":
        blocking_count = pd.to_numeric(result.get("blocking_qa_count"), errors="coerce")
        count_text = int(blocking_count) if pd.notna(blocking_count) else "some"
        return (
            f"Ramp forecast blocked by {count_text} QA issue(s); no run was published.{run_suffix}",
            "warning",
        )

    if outcome == "busy":
        return (
            result.get("message") or "Another ramp generation is already in progress.",
            "warning",
        )

    message = result.get("message") or result.get("error_message") or "Ramp forecast generation failed."
    return (
        f"Ramp forecast failed: {message}",
        "warning",
    )


def _append_capacity_ramp_result_message(
    prefix: str,
    result: dict | None,
    default_tone: str = "success",
) -> html.Div:
    ramp_text, ramp_tone = _format_capacity_ramp_result_message(result)
    tone = ramp_tone if ramp_tone != "success" else default_tone
    return _build_capacity_scenario_message(f"{prefix} {ramp_text}", tone)


def _render_capacity_ramp_status_card(status: dict | None) -> html.Div:
    if not status:
        return html.Div(
            "Ramp forecast: no generated run yet.",
            className="balance-metadata-row",
            style={"fontSize": "11px"},
        )

    run_status = str(status.get("run_status") or "unknown")
    is_stale = bool(status.get("is_stale"))
    is_current_published = bool(status.get("is_current_published"))
    run_id = status.get("run_id")
    generator_version = status.get("generator_version")
    generated_at = status.get("generated_at")
    scenario_id = status.get("capacity_scenario_id")
    selector_token = f"capacity_ramp_{scenario_id}" if scenario_id is not None else "Capacity ramp"
    blocking_count = pd.to_numeric(status.get("blocking_qa_count"), errors="coerce")
    blocking_text = int(blocking_count) if pd.notna(blocking_count) else 0
    if run_status == "failed":
        display_status = "failed"
    elif is_stale:
        display_status = "stale"
    elif blocking_text:
        display_status = "blocked"
    elif is_current_published:
        display_status = "published"
    else:
        display_status = "ready"
    warning_count = pd.to_numeric(status.get("warning_qa_count"), errors="coerce")
    warning_text = int(warning_count) if pd.notna(warning_count) else 0
    deferred_count = pd.to_numeric(status.get("deferred_qa_count"), errors="coerce")
    deferred_text = int(deferred_count) if pd.notna(deferred_count) else 0
    monthly_count = pd.to_numeric(status.get("monthly_row_count"), errors="coerce")
    monthly_text = int(monthly_count) if pd.notna(monthly_count) else 0
    horizon_start = _serialize_timestamp(status.get("horizon_start_month"))
    horizon_end = _serialize_timestamp(status.get("horizon_end_month"))
    horizon_text = (
        f"{horizon_start[:10]} to {horizon_end[:10]}"
        if horizon_start and horizon_end
        else "No horizon"
    )
    qa_messages = [
        str(issue.get("message") or "").strip()
        for issue in (status.get("qa_issues") or [])
        if str(issue.get("message") or "").strip()
    ][:3]

    status_color = {
        "published": "#166534",
        "ready": "#1d4ed8",
        "stale": "#9a3412",
        "blocked": "#9a3412",
        "failed": "#991b1b",
    }.get(display_status.casefold(), "#334155")

    children = [
        html.Div(
            [
                html.Span("Ramp forecast", style={"fontWeight": "800", "color": "#334155"}),
                html.Span(f"Run {run_id}" if run_id is not None else "No run"),
                html.Span(
                    f"Generator {generator_version}"
                    if generator_version
                    else "No generator version"
                ),
                html.Span(
                    display_status.replace("_", " ").title(),
                    style={"color": status_color, "fontWeight": "800"},
                ),
                html.Span(horizon_text),
                html.Span(f"{blocking_text} blocking QA"),
                html.Span(f"{warning_text} warning(s)"),
                html.Span(f"{deferred_text} deferred"),
                html.Span(f"{monthly_text:,} monthly rows"),
                html.Span(selector_token),
                html.Span(_serialize_timestamp(generated_at) or ""),
            ],
            className="balance-metadata-row",
            style={
                "display": "flex",
                "gap": "8px",
                "flexWrap": "wrap",
                "fontSize": "11px",
            },
        )
    ]
    if qa_messages:
        children.append(
            html.Details(
                [
                    html.Summary(
                        f"Review {min(len(qa_messages), 3)} QA message(s)",
                        style={"cursor": "pointer", "fontWeight": "700"},
                    ),
                    html.Ul(
                        [html.Li(message) for message in qa_messages],
                        style={"margin": "6px 0 0 18px", "padding": "0"},
                    ),
                ],
                style={"fontSize": "11px", "color": "#475569"},
            )
        )
    return html.Div(children)


def _serialize_capacity_ramp_action_status(status: dict | None) -> dict[str, object] | None:
    if not status:
        return None
    return {
        "capacity_scenario_id": status.get("capacity_scenario_id"),
        "run_id": status.get("run_id"),
        "run_status": status.get("run_status"),
        "is_stale": bool(status.get("is_stale")),
        "is_current_published": bool(status.get("is_current_published")),
        "blocking_qa_count": int(status.get("blocking_qa_count") or 0),
        "monthly_row_count": int(status.get("monthly_row_count") or 0),
        "error_message": status.get("error_message"),
    }


def _build_capacity_scenario_action_state(
    selected_scenario_id,
    dirty_data,
    pending_selection,
    ramp_status,
    profile_train,
    ramp_profile,
    seasonal_profile,
) -> dict[str, bool]:
    selected_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    has_scenario = pd.notna(selected_value)
    is_dirty = bool(isinstance(dirty_data, dict) and dirty_data.get("dirty"))
    has_pending_switch = pd.notna(pd.to_numeric(pending_selection, errors="coerce"))
    mutation_locked = not has_scenario or has_pending_switch
    stable_scenario = has_scenario and not has_pending_switch
    clean_scenario = stable_scenario and not is_dirty

    status = ramp_status if isinstance(ramp_status, dict) else {}
    status_scenario_value = pd.to_numeric(
        status.get("capacity_scenario_id"), errors="coerce"
    )
    status_matches_scenario = (
        has_scenario
        and pd.notna(status_scenario_value)
        and int(status_scenario_value) == int(selected_value)
    )
    publish_ready = (
        clean_scenario
        and status_matches_scenario
        and str(status.get("run_status") or "") == "completed"
        and int(status.get("monthly_row_count") or 0) > 0
        and int(status.get("blocking_qa_count") or 0) == 0
        and not bool(status.get("is_stale"))
        and not bool(status.get("is_current_published"))
    )
    profile_ready = (
        clean_scenario
        and bool(profile_train)
        and bool(ramp_profile or seasonal_profile)
    )
    return {
        "save_disabled": not (stable_scenario and is_dirty),
        "revert_disabled": not (stable_scenario and is_dirty),
        "delete_disabled": mutation_locked,
        "edit_disabled": mutation_locked,
        "generate_disabled": not clean_scenario,
        "publish_disabled": not publish_ready,
        "profile_disabled": not profile_ready,
        "reconcile_disabled": not clean_scenario,
    }


def _normalize_grid_event_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    numeric_value = pd.to_numeric([value], errors="coerce")[0]
    if pd.notna(numeric_value):
        return round(float(numeric_value), 6)
    return value


def _is_user_scenario_cell_edit(cell_event) -> bool:
    if not isinstance(cell_event, dict):
        return False

    column_id = cell_event.get("colId") or cell_event.get("columnId") or cell_event.get("field")
    if column_id not in {"Scenario First Date", "Scenario Capacity", "Scenario Note"}:
        return False

    old_value = _normalize_grid_event_value(cell_event.get("oldValue"))
    new_value = _normalize_grid_event_value(
        cell_event.get("value", cell_event.get("newValue"))
    )
    return old_value != new_value


def _coerce_timeline_row_date(value: object) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.to_period("M").to_timestamp()


def _coerce_timeline_row_capacity(value: object) -> float | None:
    numeric_value = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(numeric_value):
        return None
    return round(float(numeric_value), 6)


def _normalize_timeline_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _coerce_timeline_row_note(value: object) -> str:
    return _normalize_timeline_text(value)


def _resolve_grid_row_base_provider(row: dict) -> str:
    woodmac_capacity = _coerce_timeline_row_capacity(row.get("Woodmac Capacity Change"))
    woodmac_date = _coerce_timeline_row_date(row.get("Woodmac First Date"))
    if woodmac_capacity is not None or woodmac_date is not None:
        return "woodmac"

    ea_capacity = _coerce_timeline_row_capacity(row.get("Energy Aspects Capacity Change"))
    ea_date = _coerce_timeline_row_date(row.get("Energy Aspects First Date"))
    if ea_capacity is not None or ea_date is not None:
        return "energy_aspects"

    return "internal_scenario"


def _resolve_grid_row_base_date(row: dict) -> pd.Timestamp | None:
    return (
        _coerce_timeline_row_date(row.get("Woodmac First Date"))
        or _coerce_timeline_row_date(row.get("Energy Aspects First Date"))
    )


def _resolve_grid_row_base_capacity(row: dict) -> float | None:
    return (
        _coerce_timeline_row_capacity(row.get("Woodmac Capacity Change"))
        if _coerce_timeline_row_capacity(row.get("Woodmac Capacity Change")) is not None
        else _coerce_timeline_row_capacity(row.get("Energy Aspects Capacity Change"))
    )


def _update_working_scenario_rows_from_grid(
    working_rows_df: pd.DataFrame,
    grid_row_data: list[dict] | None,
) -> pd.DataFrame:
    working_rows_df = _prepare_capacity_scenario_rows_df(working_rows_df)
    grid_df = pd.DataFrame(grid_row_data or [])
    if grid_df.empty:
        return working_rows_df

    if "scenario_row_key" not in grid_df.columns:
        return working_rows_df

    edited_df = grid_df.copy()
    edited_df["scenario_row_key"] = edited_df["scenario_row_key"].fillna("").astype(str).str.strip()
    edited_df = edited_df[edited_df["scenario_row_key"] != ""].copy()
    if edited_df.empty:
        return working_rows_df

    edited_df["__scenario_first_date"] = edited_df["Scenario First Date"].map(_coerce_timeline_row_date)
    edited_df["__scenario_capacity"] = edited_df["Scenario Capacity"].map(_coerce_timeline_row_capacity)
    if "Scenario Note" not in edited_df.columns:
        edited_df["Scenario Note"] = ""
    edited_df["__scenario_note"] = edited_df["Scenario Note"].map(_coerce_timeline_row_note)

    if working_rows_df.empty:
        base_df = pd.DataFrame(columns=_get_capacity_scenario_row_columns())
    else:
        base_df = working_rows_df.copy()

    if not base_df.empty:
        base_df = base_df.set_index("scenario_row_key", drop=False)

    for row in edited_df.to_dict("records"):
        key = row["scenario_row_key"]
        if key in base_df.index:
            existing_base_date = base_df.at[key, "base_first_date"]
            existing_base_capacity = base_df.at[key, "base_capacity_mtpa"]
            if (
                row["__scenario_first_date"] is None
                and row["__scenario_capacity"] is None
                and row["__scenario_note"] == ""
                and pd.isna(existing_base_date)
                and pd.isna(existing_base_capacity)
            ):
                base_df = base_df.drop(index=key)
                continue
            base_df.at[key, "scenario_first_date"] = row["__scenario_first_date"]
            base_df.at[key, "scenario_capacity_mtpa"] = row["__scenario_capacity"]
            base_df.at[key, "scenario_note"] = row["__scenario_note"]
            continue

        if (
            row["__scenario_first_date"] is None
            and row["__scenario_capacity"] is None
            and row["__scenario_note"] == ""
        ):
            continue

        normalized_train_label = _format_train_label(row.get("Train"))

        new_record = {
            "scenario_row_key": key,
            "country_name": " ".join(str(row.get("Country") or "").strip().split()),
            "plant_name": " ".join(str(row.get("Plant") or "").strip().split()),
            "train_label": normalized_train_label,
            "base_provider": _resolve_grid_row_base_provider(row),
            "base_first_date": _resolve_grid_row_base_date(row),
            "base_capacity_mtpa": _resolve_grid_row_base_capacity(row),
            "scenario_first_date": row["__scenario_first_date"],
            "scenario_capacity_mtpa": row["__scenario_capacity"],
            "scenario_note": row["__scenario_note"],
            "display_sort_plant": " ".join(str(row.get("Plant") or "").strip().split()),
            "display_sort_train": pd.to_numeric([normalized_train_label], errors="coerce")[0],
        }
        if base_df.empty:
            base_df = pd.DataFrame([new_record]).set_index("scenario_row_key", drop=False)
        else:
            base_df.loc[key] = new_record

    return _prepare_capacity_scenario_rows_df(base_df.reset_index(drop=True))


def _get_capacity_scenario_base_revision(
    scenario_id: int,
    options_data: list[dict] | None = None,
) -> str:
    option = _get_capacity_scenario_option_map(options_data).get(int(scenario_id), {})
    snapshot_timestamp = option.get("current_snapshot_timestamp_utc")
    if not snapshot_timestamp:
        raise ValueError(
            f"Capacity scenario {int(scenario_id)} has no current snapshot timestamp."
        )
    return str(snapshot_timestamp)


def _cache_saved_capacity_scenario_rows(
    scenario_id: int,
    base_revision: str,
    rows_df: pd.DataFrame,
) -> None:
    cache_key = f"{int(scenario_id)}:{base_revision}"
    with _CAPACITY_SCENARIO_CACHE_LOCK:
        _CAPACITY_SCENARIO_CACHE[cache_key] = _prepare_capacity_scenario_rows_df(rows_df)
        _CAPACITY_SCENARIO_CACHE.move_to_end(cache_key)
        while len(_CAPACITY_SCENARIO_CACHE) > 16:
            _CAPACITY_SCENARIO_CACHE.popitem(last=False)


def _get_saved_capacity_scenario_rows(
    scenario_id: int,
    base_revision: str,
) -> pd.DataFrame:
    cache_key = f"{int(scenario_id)}:{base_revision}"
    with _CAPACITY_SCENARIO_CACHE_LOCK:
        cached_rows_df = _CAPACITY_SCENARIO_CACHE.get(cache_key)
        if cached_rows_df is not None:
            _CAPACITY_SCENARIO_CACHE.move_to_end(cache_key)
            return cached_rows_df.copy()

        rows_df = _prepare_capacity_scenario_rows_df(
            fetch_capacity_scenario_rows(
                int(scenario_id),
                engine,
                snapshot_timestamp_utc=base_revision,
            )
        )
        _CAPACITY_SCENARIO_CACHE[cache_key] = rows_df
        _CAPACITY_SCENARIO_CACHE.move_to_end(cache_key)
        while len(_CAPACITY_SCENARIO_CACHE) > 16:
            _CAPACITY_SCENARIO_CACHE.popitem(last=False)
        return rows_df.copy()


def _build_capacity_scenario_working_store(
    scenario_id: int,
    options_data: list[dict] | None = None,
    *,
    base_revision: str | None = None,
    upserts_df: pd.DataFrame | None = None,
    deleted_keys: list[str] | None = None,
    row_order: list[str] | None = None,
    full_override_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    normalized_revision = base_revision or _get_capacity_scenario_base_revision(
        int(scenario_id),
        options_data,
    )
    payload: dict[str, object] = {
        "format": CAPACITY_SCENARIO_WORKING_FORMAT,
        "scenario_id": int(scenario_id),
        "base_revision": normalized_revision,
        "mode": "clean",
        "upserts": None,
        "deleted_keys": sorted(set(deleted_keys or [])),
        "row_order": list(row_order or []),
        "full_override": None,
    }
    if full_override_df is not None:
        payload["mode"] = "full_override"
        payload["full_override"] = _serialize_dataframe(
            _prepare_capacity_scenario_rows_df(full_override_df)
        )
        return payload

    prepared_upserts_df = _prepare_capacity_scenario_rows_df(upserts_df)
    if not prepared_upserts_df.empty or payload["deleted_keys"]:
        payload["mode"] = "delta"
        payload["upserts"] = _serialize_dataframe(prepared_upserts_df)
    return payload


def _normalize_capacity_scenario_signature_value(value: object):
    if isinstance(value, pd.Timestamp):
        return _serialize_timestamp(value)
    if isinstance(value, dict):
        return {
            str(key): _normalize_capacity_scenario_signature_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_capacity_scenario_signature_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _build_capacity_scenario_row_signatures(rows_df: pd.DataFrame) -> dict[str, str]:
    signatures: dict[str, str] = {}
    for row in _prepare_capacity_scenario_rows_df(rows_df).to_dict("records"):
        row_key = str(row.get("scenario_row_key") or "")
        if not row_key:
            continue
        normalized_row = {
            column_name: _normalize_capacity_scenario_signature_value(row.get(column_name))
            for column_name in _get_capacity_scenario_row_columns()
        }
        signatures[row_key] = json.dumps(
            normalized_row,
            sort_keys=True,
            separators=(",", ":"),
        )
    return signatures


def _build_capacity_scenario_delta_store(
    scenario_id: int,
    active_rows_df: pd.DataFrame,
    options_data: list[dict] | None = None,
    *,
    base_revision: str | None = None,
) -> dict[str, object]:
    normalized_revision = base_revision or _get_capacity_scenario_base_revision(
        int(scenario_id),
        options_data,
    )
    baseline_df = _get_saved_capacity_scenario_rows(
        int(scenario_id),
        normalized_revision,
    )
    active_rows_df = _prepare_capacity_scenario_rows_df(active_rows_df)
    baseline_signatures = _build_capacity_scenario_row_signatures(baseline_df)
    active_signatures = _build_capacity_scenario_row_signatures(active_rows_df)
    deleted_keys = sorted(set(baseline_signatures) - set(active_signatures))
    upsert_keys = {
        key
        for key, signature in active_signatures.items()
        if baseline_signatures.get(key) != signature
    }
    upserts_df = active_rows_df[
        active_rows_df["scenario_row_key"].astype(str).isin(upsert_keys)
    ].copy()
    return _build_capacity_scenario_working_store(
        int(scenario_id),
        options_data,
        base_revision=normalized_revision,
        upserts_df=upserts_df,
        deleted_keys=deleted_keys,
        row_order=active_rows_df["scenario_row_key"].astype(str).tolist(),
    )


def _resolve_active_capacity_scenario_rows(
    working_store_data,
    dirty_store: dict | None = None,
    timeline_row_data: list[dict] | None = None,
) -> pd.DataFrame:
    if (
        isinstance(working_store_data, dict)
        and working_store_data.get("format") == CAPACITY_SCENARIO_WORKING_FORMAT
    ):
        scenario_value = pd.to_numeric(
            working_store_data.get("scenario_id"),
            errors="coerce",
        )
        if pd.isna(scenario_value):
            return pd.DataFrame(columns=_get_capacity_scenario_row_columns())
        if working_store_data.get("mode") == "full_override":
            return _prepare_capacity_scenario_rows_df(
                _deserialize_dataframe(working_store_data.get("full_override"))
            )

        rows_df = _get_saved_capacity_scenario_rows(
            int(scenario_value),
            str(working_store_data.get("base_revision") or "saved"),
        )
        deleted_keys = {
            str(key)
            for key in (working_store_data.get("deleted_keys") or [])
            if str(key)
        }
        if deleted_keys and not rows_df.empty:
            rows_df = rows_df[
                ~rows_df["scenario_row_key"].astype(str).isin(deleted_keys)
            ].copy()
        upserts_df = _prepare_capacity_scenario_rows_df(
            _deserialize_dataframe(working_store_data.get("upserts"))
        )
        if not upserts_df.empty:
            upsert_keys = set(upserts_df["scenario_row_key"].astype(str))
            if not rows_df.empty:
                rows_df = rows_df[
                    ~rows_df["scenario_row_key"].astype(str).isin(upsert_keys)
                ].copy()
            rows_df = pd.concat([rows_df, upserts_df], ignore_index=True, sort=False)
        row_order = {
            str(row_key): index
            for index, row_key in enumerate(working_store_data.get("row_order") or [])
        }
        if row_order and not rows_df.empty:
            rows_df["__working_order"] = (
                rows_df["scenario_row_key"].astype(str).map(row_order).fillna(len(row_order))
            )
            rows_df = rows_df.sort_values("__working_order", kind="mergesort").drop(
                columns=["__working_order"]
            )
        return _prepare_capacity_scenario_rows_df(rows_df)

    rows_df = _prepare_capacity_scenario_rows_df(_deserialize_dataframe(working_store_data))
    dirty_payload = dirty_store if isinstance(dirty_store, dict) else {"dirty": False}
    if (
        not dirty_payload.get("dirty")
        or dirty_payload.get("source") in {"rebuild", "working"}
        or not timeline_row_data
    ):
        return rows_df

    try:
        return _update_working_scenario_rows_from_grid(rows_df, timeline_row_data)
    except Exception:
        return rows_df


def _append_manual_capacity_scenario_row(
    rows_df: pd.DataFrame,
    terminal_key: str,
    train_key: str,
    country_name: object,
    plant_name: object,
    train_label: object,
    first_date: object,
    capacity_value: object,
) -> tuple[pd.DataFrame, str]:
    normalized_country = " ".join(str(country_name or "").strip().split())
    normalized_plant = " ".join(str(plant_name or "").strip().split())
    normalized_train = normalize_train_label(train_label)
    scenario_first_date = _coerce_timeline_row_date(first_date)
    scenario_capacity = _coerce_timeline_row_capacity(capacity_value)

    if not normalized_country:
        raise ValueError("Country is required.")
    if not normalized_plant:
        raise ValueError("Plant is required.")
    if not normalized_train:
        raise ValueError("Train label is required.")
    if scenario_first_date is None:
        raise ValueError("First Date must be a valid month like 2026-01.")
    if scenario_capacity is None or scenario_capacity <= 0:
        raise ValueError("Capacity must be greater than 0.")

    scenario_row_key = _build_capacity_scenario_row_key(
        normalized_country,
        normalized_plant,
        normalized_train,
        effective_date=scenario_first_date,
        capacity_value=scenario_capacity,
        provider="manual",
    )

    rows_df = _prepare_capacity_scenario_rows_df(rows_df)
    if not rows_df.empty and rows_df["scenario_row_key"].eq(scenario_row_key).any():
        raise ValueError("This internal scenario row already exists.")

    new_row = pd.DataFrame(
        [
            {
                "scenario_row_key": scenario_row_key,
                "terminal_key": terminal_key,
                "train_key": train_key,
                "country_name": normalized_country,
                "plant_name": normalized_plant,
                "train_label": normalized_train,
                "base_provider": "internal_scenario",
                "base_first_date": None,
                "base_capacity_mtpa": None,
                "scenario_first_date": scenario_first_date,
                "scenario_capacity_mtpa": scenario_capacity,
                "scenario_note": "",
                "display_sort_plant": normalized_plant,
                "display_sort_train": pd.to_numeric([normalized_train], errors="coerce")[0],
            }
        ]
    )

    return (
        _prepare_capacity_scenario_rows_df(pd.concat([rows_df, new_row], ignore_index=True, sort=False)),
        scenario_row_key,
    )


def _create_source_section(
    title: str,
    subtitle: str,
    title_note: str | None,
    summary_id: str,
    chart_id: str,
    table_container_id: str,
    export_button_id: str,
) -> html.Div:
    header_children = [
        html.Div(
            [
                html.Div(
                    html.H3(
                        title,
                        className="balance-section-title",
                        title=title_note or title,
                        style={
                            "cursor": "help" if title_note else None,
                            "whiteSpace": "nowrap",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                        },
                    ),
                    style={"flex": "1 1 auto", "minWidth": "0"},
                ),
                html.Button(
                    "Export to Excel",
                    id=export_button_id,
                    n_clicks=0,
                    style={**EXPORT_BUTTON_STYLE, "flexShrink": "0"},
                ),
            ],
            className="inline-section-header",
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "gap": "12px",
                "flexWrap": "nowrap",
            },
        )
    ]

    if subtitle:
        header_children.append(html.P(subtitle, className="balance-section-subtitle"))

    return html.Div(
        [
            html.Div(
                header_children,
                className="balance-section-header",
                style={
                    "padding": "0",
                    "gap": "0",
                    "borderBottom": "none",
                    "background": "transparent",
                },
            ),
            dcc.Graph(
                id=chart_id,
                figure=_create_empty_capacity_figure("Loading capacity chart..."),
                config={"displayModeBar": True, "displaylogo": False},
                style={"height": "100%", "marginBottom": "16px"},
            ),
            html.Div(id=table_container_id, className="balance-table-container"),
            html.Div(
                id=summary_id,
                style={
                    "padding": "12px 16px 16px",
                    "borderTop": "1px solid #e5e7eb",
                },
            ),
        ],
        className="balance-section-card",
    )


def _create_internal_scenario_section(
    title: str,
    summary_id: str,
    chart_id: str,
    table_container_id: str,
    export_button_id: str,
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.H3(
                            title,
                            className="balance-section-title",
                            title=title,
                            style={
                                "whiteSpace": "nowrap",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                            },
                        ),
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "12px",
                            "flexWrap": "nowrap",
                            "flex": "1 1 auto",
                            "minWidth": "0",
                        },
                    ),
                    html.Button(
                        "Export to Excel",
                        id=export_button_id,
                        n_clicks=0,
                        style={**EXPORT_BUTTON_STYLE, "flexShrink": "0"},
                    ),
                ],
                className="inline-section-header",
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "gap": "12px",
                    "flexWrap": "nowrap",
                },
            ),
            dcc.Graph(
                id=chart_id,
                figure=_create_empty_capacity_figure(INTERNAL_SCENARIO_EMPTY_MESSAGE),
                config={"displayModeBar": True, "displaylogo": False},
                style={"height": "100%", "marginBottom": "16px"},
            ),
            html.Div(id=table_container_id, className="balance-table-container"),
            html.Div(
                id=summary_id,
                style={
                    "padding": "12px 16px 16px",
                    "borderTop": "1px solid #e5e7eb",
                },
            ),
        ],
        className="balance-section-card",
    )


def _create_top_capacity_selector_region() -> html.Div:
    control_label_style = {
        "fontSize": "12px",
        "fontWeight": "700",
        "letterSpacing": "0.04em",
        "textTransform": "uppercase",
        "color": "#64748b",
        "marginBottom": "8px",
    }
    radio_label_style = {
        "display": "inline-flex",
        "alignItems": "center",
        "marginRight": "12px",
        "fontSize": "12px",
        "fontWeight": "600",
        "color": "#334155",
    }

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "Capacity Controls",
                        style={
                            "fontSize": "12px",
                            "fontWeight": "800",
                            "letterSpacing": "0.08em",
                            "textTransform": "uppercase",
                            "color": "#1d4ed8",
                        },
                    ),
                ],
                style={"display": "grid", "gap": "4px", "marginBottom": "14px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Country Columns", style=control_label_style),
                            dcc.Dropdown(
                                id="capacity-page-country-dropdown",
                                options=[],
                                value=None,
                                multi=True,
                                persistence=True,
                                persistence_type="session",
                                placeholder="Select countries to keep as separate columns",
                                className="filter-dropdown",
                                style={"minWidth": "320px"},
                            ),
                        ],
                        style={"minWidth": "0"},
                    ),
                    html.Div(
                        [
                            html.Div("Other Countries", style=control_label_style),
                            html.Div(
                                [
                                    dcc.RadioItems(
                                        id="capacity-page-other-country-mode",
                                        options=[
                                            {"label": "Include", "value": "rest_of_world"},
                                            {"label": "Exclude", "value": "exclude"},
                                        ],
                                        value="rest_of_world",
                                        inline=True,
                                        persistence=True,
                                        persistence_type="session",
                                        labelStyle=radio_label_style,
                                        inputStyle={"marginRight": "6px"},
                                        style={"display": "flex", "alignItems": "center"},
                                    )
                                ],
                                style=TRAIN_CHANGE_CONTROL_SHELL_STYLE,
                            ),
                        ],
                        style={"minWidth": "0"},
                    ),
                    html.Div(
                        [
                            html.Div("Table Values", style=control_label_style),
                            html.Div(
                                [
                                    dcc.RadioItems(
                                        id="capacity-page-top-table-view",
                                        options=[
                                            {"label": "Absolute", "value": "absolute"},
                                            {"label": "Change", "value": "change"},
                                        ],
                                        value="absolute",
                                        inline=True,
                                        persistence=True,
                                        persistence_type="session",
                                        labelStyle=radio_label_style,
                                        inputStyle={"marginRight": "6px"},
                                        style={"display": "flex", "alignItems": "center"},
                                    )
                                ],
                                style=TRAIN_CHANGE_CONTROL_SHELL_STYLE,
                            ),
                        ],
                        style={"minWidth": "0"},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "16px",
                    "alignItems": "end",
                },
            ),
        ],
        style={
            "padding": "16px 18px",
            "border": "1px solid #dbe7f4",
            "borderRadius": "16px",
            "background": "linear-gradient(180deg, #f8fbff 0%, #f1f7ff 100%)",
            "boxShadow": "0 8px 24px rgba(15, 23, 42, 0.05)",
        },
    )


def _create_yearly_capacity_comparison_section(
    title: str,
    subtitle: str,
    chart_id: str,
    table_container_id: str,
) -> html.Div:
    header_children = [html.H3(title, className="balance-section-title")]
    if subtitle:
        header_children.append(html.P(subtitle, className="balance-section-subtitle"))

    return html.Div(
        [
            html.Div(header_children, className="balance-section-header"),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id=chart_id,
                            figure=_create_empty_capacity_figure(
                                YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE
                            ),
                            config={"displayModeBar": False},
                            style={"height": "100%"},
                        ),
                        style={"minWidth": "0"},
                    ),
                    html.Div(
                        id=table_container_id,
                        children=_create_empty_state(
                            YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE
                        ),
                        className="balance-table-container",
                        style={"minWidth": "0"},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                    "gap": "24px",
                    "alignItems": "start",
                },
            ),
        ],
        className="balance-section-card",
    )


def _create_yearly_discrepancy_subcard(
    title: str,
    table_container_id: str,
) -> html.Div:
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "fontSize": "12px",
                    "fontWeight": "800",
                    "letterSpacing": "0.06em",
                    "textTransform": "uppercase",
                    "color": "#334155",
                },
            ),
            html.Div(
                id=table_container_id,
                children=_create_empty_state(YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE),
                className="balance-table-container",
                style={"minWidth": "0"},
            ),
        ],
        style={
            "display": "grid",
            "gap": "10px",
            "padding": "14px 16px",
            "border": "1px solid #e2e8f0",
            "borderRadius": "14px",
            "background": "#fbfdff",
            "boxShadow": "0 6px 20px rgba(15, 23, 42, 0.03)",
            "minWidth": "0",
        },
    )


def _create_yearly_provider_discrepancy_row(
    provider_title: str,
    cards: list[html.Div],
) -> html.Div:
    return html.Div(
        [
            html.Div(
                provider_title,
                style={
                    "fontSize": "12px",
                    "fontWeight": "800",
                    "letterSpacing": "0.08em",
                    "textTransform": "uppercase",
                    "color": "#1e3a5f",
                },
            ),
            html.Div(
                cards,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                    "gap": "16px",
                    "alignItems": "start",
                },
            ),
        ],
        style={"display": "grid", "gap": "12px"},
    )


def _create_yearly_provider_discrepancy_section(
    title: str,
    subtitle: str,
) -> html.Div:
    header_children = [html.H3(title, className="balance-section-title")]
    if subtitle:
        header_children.append(html.P(subtitle, className="balance-section-subtitle"))

    return html.Div(
        [
            html.Div(header_children, className="balance-section-header"),
            _create_yearly_provider_discrepancy_row(
                "Woodmac",
                [
                    _create_yearly_discrepancy_subcard(
                        "Woodmac Capacity Discrepancies",
                        "capacity-page-yearly-woodmac-capacity-discrepancy-table-container",
                    ),
                    _create_yearly_discrepancy_subcard(
                        "Woodmac Timeline Discrepancies",
                        "capacity-page-yearly-woodmac-timeline-discrepancy-table-container",
                    ),
                    _create_yearly_discrepancy_subcard(
                        "Plants And Trains Missing In Internal Scenario",
                        "capacity-page-yearly-woodmac-missing-internal-table-container",
                    ),
                ],
            ),
            _create_yearly_provider_discrepancy_row(
                "Energy Aspects",
                [
                    _create_yearly_discrepancy_subcard(
                        "Energy Aspects Capacity Discrepancies",
                        "capacity-page-yearly-ea-capacity-discrepancy-table-container",
                    ),
                    _create_yearly_discrepancy_subcard(
                        "Energy Aspects Timeline Discrepancies",
                        "capacity-page-yearly-ea-timeline-discrepancy-table-container",
                    ),
                    _create_yearly_discrepancy_subcard(
                        "Plants And Trains Missing In Internal Scenario",
                        "capacity-page-yearly-ea-missing-internal-table-container",
                    ),
                ],
            ),
        ],
        className="balance-section-card",
    )


def _create_train_change_section(
    title: str,
    subtitle: str,
    summary_id: str,
    table_container_id: str,
    export_button_id: str = "",
) -> html.Div:
    header_children = [
        html.Div(
            [
                html.H3(title, className="balance-section-title"),
                html.Button(
                    "Export to Excel",
                    id=export_button_id,
                    n_clicks=0,
                    style=EXPORT_BUTTON_STYLE,
                ),
                html.Div(
                    [
                        html.Span("Detail view", style=TRAIN_CHANGE_CONTROL_LABEL_STYLE),
                        html.Div(
                            [
                                dcc.RadioItems(
                                    id="capacity-page-train-change-view-mode",
                                    options=[
                                        {"label": "Total", "value": "total"},
                                        {"label": "Country", "value": "country"},
                                        {"label": "Plants View", "value": "plants"},
                                        {
                                            "label": "Plants + Trains View",
                                            "value": "plants_trains",
                                        },
                                    ],
                                    value="country",
                                    inline=True,
                                    persistence=True,
                                    persistence_type="session",
                                    labelStyle={
                                        "display": "inline-flex",
                                        "alignItems": "center",
                                        "marginRight": "10px",
                                        "fontSize": "12px",
                                        "fontWeight": "600",
                                        "color": "#334155",
                                    },
                                    inputStyle={"marginRight": "6px"},
                                    style={"display": "flex", "alignItems": "center"},
                                ),
                            ],
                            style={"display": "flex", "gap": "8px", "alignItems": "center"},
                        ),
                    ],
                    style=TRAIN_CHANGE_CONTROL_SHELL_STYLE,
                ),
            ],
            className="inline-section-header",
            style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
        )
    ]

    if subtitle:
        header_children.append(html.P(subtitle, className="balance-section-subtitle"))

    header_children.append(html.Div(id=summary_id))

    return html.Div(
        [
            html.Div(header_children, className="balance-section-header"),
            html.Div(
                _create_train_timeline_table(
                    "capacity-page-train-timeline-table",
                    pd.DataFrame(),
                    scenario_rows_df=pd.DataFrame(),
                    show_original_names=False,
                    enable_editing=False,
                ),
                id=table_container_id,
                className="balance-table-container",
            ),
        ],
        className="balance-section-card",
    )


def _create_train_timeline_section(
    title: str,
    subtitle: str,
    table_container_id: str,
    export_button_id: str = "",
) -> html.Div:
    action_button_style = {
        "padding": "7px 12px",
        "borderRadius": "999px",
        "border": "1px solid #cbd5e1",
        "backgroundColor": "#ffffff",
        "color": "#0f172a",
        "fontSize": "12px",
        "fontWeight": "700",
        "cursor": "pointer",
    }
    control_label_style = {
        "fontSize": "11px",
        "fontWeight": "700",
        "letterSpacing": "0.06em",
        "textTransform": "uppercase",
        "color": "#64748b",
        "marginBottom": "6px",
    }
    control_card_title_style = {
        "fontSize": "12px",
        "fontWeight": "800",
        "letterSpacing": "0.08em",
        "textTransform": "uppercase",
        "marginBottom": "12px",
    }
    create_card_style = {
        "display": "grid",
        "gap": "12px",
        "padding": "18px 20px",
        "border": "1px solid #cfe0f6",
        "borderRadius": "16px",
        "background": "linear-gradient(180deg, #f8fbff 0%, #f4f8ff 100%)",
        "boxShadow": "0 8px 24px rgba(15, 23, 42, 0.04)",
    }
    selected_card_style = {
        "display": "grid",
        "gap": "14px",
        "padding": "18px 20px",
        "border": "1px solid #d7eadc",
        "borderRadius": "16px",
        "background": "linear-gradient(180deg, #f8fcf8 0%, #f3faf4 100%)",
        "boxShadow": "0 8px 24px rgba(15, 23, 42, 0.04)",
    }
    add_row_card_style = {
        "display": "grid",
        "gap": "12px",
        "padding": "18px 20px",
        "border": "1px solid #dbe4ee",
        "borderRadius": "16px",
        "background": "#fbfcfe",
        "boxShadow": "0 8px 24px rgba(15, 23, 42, 0.03)",
    }
    header_children = [
        html.Div(
            [
                html.Div(
                    [
                        html.H3(title, className="balance-section-title"),
                        html.Div(
                            [
                                html.Span("Original names", style=TRAIN_CHANGE_CONTROL_LABEL_STYLE),
                                html.Div(
                                    [
                                        dcc.RadioItems(
                                            id="capacity-page-train-timeline-original-name-visibility",
                                            options=[
                                                {"label": "Hide", "value": "hide"},
                                                {"label": "Show", "value": "show"},
                                            ],
                                            value="hide",
                                            inline=True,
                                            persistence=True,
                                            persistence_type="session",
                                            labelStyle={
                                                "display": "inline-flex",
                                                "alignItems": "center",
                                                "marginRight": "10px",
                                                "fontSize": "12px",
                                                "fontWeight": "600",
                                                "color": "#334155",
                                            },
                                            inputStyle={"marginRight": "6px"},
                                            style={"display": "flex", "alignItems": "center"},
                                        ),
                                    ],
                                    style={"display": "flex", "gap": "8px", "alignItems": "center"},
                                ),
                            ],
                            style=TRAIN_CHANGE_CONTROL_SHELL_STYLE,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "12px",
                        "flexWrap": "wrap",
                    },
                ),
                html.Button(
                    "Export to Excel",
                    id=export_button_id,
                    n_clicks=0,
                    style=EXPORT_BUTTON_STYLE,
                ),
            ],
            className="inline-section-header",
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "gap": "12px",
                "flexWrap": "wrap",
            },
        )
    ]

    if subtitle:
        header_children.append(html.P(subtitle, className="balance-section-subtitle"))

    header_children.extend(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "Create New Scenario",
                                        style={**control_card_title_style, "color": "#1d5fa7", "marginBottom": "0"},
                                    ),
                                    html.Button(
                                        "Create Scenario",
                                        id="capacity-page-capacity-scenario-create-button",
                                        n_clicks=0,
                                        style={
                                            **action_button_style,
                                            "backgroundColor": "#1d4ed8",
                                            "borderColor": "#1d4ed8",
                                            "color": "white",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "space-between",
                                    "gap": "12px",
                                    "flexWrap": "wrap",
                                },
                            ),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="capacity-page-capacity-scenario-create-name",
                                        type="text",
                                        placeholder="Scenario name",
                                        style={
                                            "minWidth": "240px",
                                            "flex": "1 1 auto",
                                            "padding": "8px 10px",
                                            "border": "1px solid #cbd5e1",
                                            "borderRadius": "10px",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "10px",
                                    "flexWrap": "wrap",
                                    "alignItems": "center",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Create from",
                                        style={**control_label_style, "marginBottom": "0"},
                                    ),
                                ],
                            ),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="capacity-page-capacity-scenario-create-base-type",
                                        options=[
                                            {"label": "Woodmac", "value": "woodmac"},
                                            {"label": "Energy Aspects", "value": "energy_aspects"},
                                            {"label": "Current Scenario", "value": "current_scenario"},
                                            {"label": "Existing Internal Scenario", "value": "internal_scenario"},
                                        ],
                                        value="woodmac",
                                        clearable=False,
                                        persistence=True,
                                        persistence_type="session",
                                        style={"width": "190px"},
                                    ),
                                    dcc.Dropdown(
                                        id="capacity-page-capacity-scenario-create-source-dropdown",
                                        options=[],
                                        value=None,
                                        placeholder="Source internal scenario",
                                        style={"width": "220px", "opacity": "0.55"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "10px",
                                    "flexWrap": "wrap",
                                    "alignItems": "center",
                                },
                            ),
                        ],
                        style=create_card_style,
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "Selected Scenario",
                                        style={
                                            **control_card_title_style,
                                            "color": "#2f7a48",
                                            "marginBottom": "0",
                                        },
                                    ),
                                    html.Div(
                                        id="capacity-page-train-timeline-current-scenario-label",
                                        children="No internal scenario selected",
                                        style={
                                            "display": "inline-flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                            "padding": "6px 10px",
                                            "borderRadius": "999px",
                                            "backgroundColor": "#e2e8f0",
                                            "color": "#334155",
                                            "fontSize": "12px",
                                            "fontWeight": "700",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "10px",
                                    "flexWrap": "wrap",
                                    "alignItems": "center",
                                    "justifyContent": "space-between",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        id="capacity-page-capacity-scenario-dirty-indicator",
                                        style={
                                            "display": "inline-flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                            "padding": "8px 12px",
                                            "borderRadius": "999px",
                                            "backgroundColor": "#ecfeff",
                                            "color": "#155e75",
                                            "fontSize": "12px",
                                            "fontWeight": "700",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "10px",
                                    "flexWrap": "wrap",
                                    "alignItems": "center",
                                },
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Save",
                                        id="capacity-page-capacity-scenario-save-button",
                                        n_clicks=0,
                                        disabled=True,
                                        style={
                                            **action_button_style,
                                            "backgroundColor": "#166534",
                                            "borderColor": "#166534",
                                            "color": "white",
                                        },
                                    ),
                                    html.Button(
                                        "Generate Ramp",
                                        id="capacity-page-ramp-generate-button",
                                        n_clicks=0,
                                        disabled=True,
                                        style={
                                            **action_button_style,
                                            "backgroundColor": "#0f766e",
                                            "borderColor": "#0f766e",
                                            "color": "white",
                                        },
                                    ),
                                    html.Button(
                                        "Publish Ramp",
                                        id="capacity-page-ramp-publish-button",
                                        n_clicks=0,
                                        disabled=True,
                                        style={
                                            **action_button_style,
                                            "backgroundColor": "#1d4ed8",
                                            "borderColor": "#1d4ed8",
                                            "color": "white",
                                        },
                                    ),
                                    html.Button(
                                        "Revert",
                                        id="capacity-page-capacity-scenario-revert-button",
                                        n_clicks=0,
                                        disabled=True,
                                        style=action_button_style,
                                    ),
                                    html.Button(
                                        "Delete",
                                        id="capacity-page-capacity-scenario-delete-button",
                                        n_clicks=0,
                                        disabled=True,
                                        style={
                                            **action_button_style,
                                            "backgroundColor": "#fff1f2",
                                            "borderColor": "#fecdd3",
                                            "color": "#be123c",
                                        },
                                    ),
                                    dcc.Upload(
                                        id="capacity-page-capacity-scenario-upload",
                                        children=html.Button(
                                            "Upload Train Timeline Excel",
                                            id="capacity-page-capacity-scenario-upload-button",
                                            type="button",
                                            disabled=True,
                                            style={
                                                **action_button_style,
                                                "backgroundColor": "#fff7ed",
                                                "borderColor": "#fdba74",
                                                "color": "#9a3412",
                                                "opacity": "0.55",
                                                "cursor": "not-allowed",
                                            },
                                        ),
                                        multiple=False,
                                        disabled=True,
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "10px",
                                    "flexWrap": "wrap",
                                    "alignItems": "center",
                                },
                            ),
                            html.Div(
                                "Upload only a Train Timeline workbook exported from this section. Existing scenario rows retain their canonical train link; newly added trains must be registered above before publishing.",
                                className="balance-metadata-row",
                                style={"fontSize": "11px"},
                            ),
                            html.Div(
                                id="capacity-page-ramp-forecast-status",
                                className="balance-metadata-row",
                                style={"fontSize": "11px"},
                            ),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="capacity-page-ramp-profile-train",
                                        options=[],
                                        placeholder="Train requiring ramp profile",
                                        clearable=True,
                                        style={"minWidth": "260px", "flex": "1 1 300px"},
                                    ),
                                    dcc.Dropdown(
                                        id="capacity-page-ramp-profile-choice",
                                        options=[],
                                        placeholder="SQL ramp profile",
                                        clearable=True,
                                        style={"minWidth": "300px", "flex": "1 1 360px"},
                                    ),
                                    dcc.Dropdown(
                                        id="capacity-page-seasonal-profile-choice",
                                        options=[],
                                        placeholder="SQL seasonal profile",
                                        clearable=True,
                                        style={"minWidth": "220px", "flex": "1 1 260px"},
                                    ),
                                    html.Button(
                                        "Assign Profiles",
                                        id="capacity-page-ramp-profile-assign-button",
                                        n_clicks=0,
                                        disabled=True,
                                        style={
                                            **action_button_style,
                                            "backgroundColor": "#0f766e",
                                            "borderColor": "#0f766e",
                                            "color": "white",
                                        },
                                    ),
                                    html.Button(
                                        "Reconcile Identities",
                                        id="capacity-page-ramp-identity-reconcile-button",
                                        n_clicks=0,
                                        disabled=True,
                                        style=action_button_style,
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "8px",
                                    "flexWrap": "wrap",
                                    "alignItems": "center",
                                },
                            ),
                        ],
                        style=selected_card_style,
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))",
                    "gap": "16px",
                    "padding": "0 16px 16px",
                },
            ),
            html.Div(
                [
                    html.Div(
                        "Add Row to Selected Scenario",
                        style={**control_card_title_style, "color": "#5a6d88", "marginBottom": "0"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="capacity-page-scenario-add-country",
                                options=[],
                                placeholder="Country",
                                searchable=True,
                                clearable=True,
                                style={
                                    "width": "180px",
                                },
                            ),
                            dcc.Input(
                                id="capacity-page-scenario-add-plant",
                                type="text",
                                placeholder="Plant",
                                style={
                                    "width": "210px",
                                    "padding": "8px 10px",
                                    "border": "1px solid #cbd5e1",
                                    "borderRadius": "10px",
                                },
                            ),
                            dcc.Input(
                                id="capacity-page-scenario-add-train",
                                type="text",
                                placeholder="Train label",
                                style={
                                    "width": "140px",
                                    "padding": "8px 10px",
                                    "border": "1px solid #cbd5e1",
                                    "borderRadius": "10px",
                                },
                            ),
                            dcc.Input(
                                id="capacity-page-scenario-add-first-date",
                                type="text",
                                placeholder="First Date (YYYY-MM)",
                                style={
                                    "width": "150px",
                                    "padding": "8px 10px",
                                    "border": "1px solid #cbd5e1",
                                    "borderRadius": "10px",
                                },
                            ),
                            dcc.Input(
                                id="capacity-page-scenario-add-capacity",
                                type="number",
                                placeholder="Capacity",
                                style={
                                    "width": "120px",
                                    "padding": "8px 10px",
                                    "border": "1px solid #cbd5e1",
                                    "borderRadius": "10px",
                                },
                            ),
                            dcc.Dropdown(
                                id="capacity-page-scenario-add-seasonal-profile",
                                options=[],
                                placeholder="Seasonal profile",
                                clearable=True,
                                style={"width": "190px"},
                            ),
                            dcc.Dropdown(
                                id="capacity-page-scenario-add-ramp-category",
                                options=[],
                                placeholder="Ramp category",
                                clearable=True,
                                style={"width": "250px"},
                            ),
                            html.Button(
                                "Add Row",
                                id="capacity-page-scenario-add-row-button",
                                n_clicks=0,
                                style={
                                    **action_button_style,
                                    "backgroundColor": "#0f766e",
                                    "borderColor": "#0f766e",
                                    "color": "white",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "10px",
                            "flexWrap": "wrap",
                            "alignItems": "center",
                        },
                    ),
                ],
                style={
                    **add_row_card_style,
                    "margin": "0 16px 12px",
                },
            ),
            html.Div(
                id="capacity-page-capacity-scenario-message",
                style={"padding": "0 16px 12px"},
            ),
        ]
    )

    return html.Div(
        [
            html.Div(header_children, className="balance-section-header"),
            html.Div(id=table_container_id, className="balance-table-container"),
        ],
        className="balance-section-card",
    )


def _create_train_timeline_comparison_chart_section() -> html.Div:
    control_label_style = {
        "fontSize": "9px",
        "fontWeight": "700",
        "letterSpacing": "0.06em",
        "textTransform": "uppercase",
        "color": "#475569",
        "marginBottom": "4px",
    }
    dropdown_style = {"minWidth": "220px", "width": "100%"}
    header_shell_style = {
        "display": "flex",
        "flexDirection": "row",
        "alignItems": "flex-start",
        "justifyContent": "flex-start",
        "gap": "14px",
        "flexWrap": "nowrap",
        "padding": "12px 18px 10px",
        "border": "1px solid #dbe7f4",
        "borderRadius": "18px",
        "background": "linear-gradient(180deg, #fbfdff 0%, #f3f8ff 100%)",
        "boxShadow": "0 10px 26px rgba(15, 23, 42, 0.05)",
    }
    header_controls_style = {
        "display": "flex",
        "gap": "12px",
        "flexWrap": "nowrap",
        "alignItems": "flex-start",
        "minWidth": "460px",
        "marginLeft": "auto",
        "flex": "0 0 auto",
    }

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.H3(
                            "LNG Train Timeline",
                            className="balance-section-title",
                            style={
                                "margin": "0",
                                "fontSize": "24px",
                                "lineHeight": "1.15",
                                "textAlign": "left",
                            },
                        ),
                        style={
                            "minWidth": "0",
                            "flex": "1 1 auto",
                            "display": "flex",
                            "justifyContent": "flex-start",
                            "alignSelf": "flex-start",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Source", style=control_label_style),
                                    dcc.Dropdown(
                                        id="capacity-page-train-timeline-chart-source",
                                        options=[
                                            {"label": "Woodmac", "value": "woodmac"},
                                            {"label": "Energy Aspects", "value": "energy_aspects"},
                                        ],
                                        value="woodmac",
                                        clearable=False,
                                        persistence=True,
                                        persistence_type="session",
                                        style=dropdown_style,
                                    ),
                                ],
                                style={"minWidth": "0"},
                            ),
                            html.Div(
                                [
                                    html.Div("Compare To", style=control_label_style),
                                    dcc.Dropdown(
                                        id="capacity-page-train-timeline-chart-compare",
                                        options=[
                                            {"label": "Woodmac", "value": "woodmac"},
                                            {"label": "Energy Aspects", "value": "energy_aspects"},
                                        ],
                                        value="energy_aspects",
                                        clearable=False,
                                        persistence=True,
                                        persistence_type="session",
                                        style=dropdown_style,
                                    ),
                                ],
                                style={"minWidth": "0"},
                            ),
                        ],
                        style={**header_controls_style, "alignSelf": "flex-start"},
                    ),
                ],
                className="balance-section-header",
                style={
                    **header_shell_style,
                    "borderBottom": "none",
                },
            ),
            dcc.Graph(
                id="capacity-page-train-timeline-comparison-graph",
                figure=_create_empty_capacity_figure("Loading LNG train timeline..."),
                config={"displayModeBar": True, "displaylogo": False},
                style={"height": "100%", "marginTop": "6px"},
            ),
        ],
        className="balance-section-card",
    )


def _create_capacity_country_area_chart(
    matrix_df: pd.DataFrame,
    y_axis_title: str = "MTPA",
    time_view: str = "monthly",
) -> go.Figure:
    total_column_label = "Total MTPA"
    if matrix_df.empty:
        return _create_empty_capacity_figure("No capacity data available")

    country_columns = [
        column_name
        for column_name in matrix_df.columns
        if column_name not in {"Month", total_column_label, "__axis_date"}
    ]

    if not country_columns:
        return _create_empty_capacity_figure(
            "Select at least one country or switch to Rest of the World mode."
        )

    plot_df = matrix_df.copy()
    if "__axis_date" in plot_df.columns:
        plot_df["date"] = pd.to_datetime(plot_df["__axis_date"], errors="coerce")
    else:
        plot_df["date"] = pd.to_datetime(plot_df["Month"].astype(str) + "-01", errors="coerce")
    plot_df = plot_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    pivot_df = plot_df.set_index("date")[country_columns]
    pivot_df = pivot_df[pivot_df.sum(axis=1) > 0]

    if pivot_df.empty:
        return _create_empty_capacity_figure("No capacity data available")

    column_totals = pivot_df.sum().sort_values(ascending=False)
    pivot_df = pivot_df[column_totals.index]
    total_series = plot_df.set_index("date")[total_column_label].reindex(pivot_df.index).fillna(0.0)
    period_labels = (
        plot_df.drop_duplicates(subset=["date"])
        .set_index("date")["Month"]
        .reindex(pivot_df.index)
        .fillna("")
        .astype(str)
    )
    max_total = float(total_series.max()) if not total_series.empty else 0.0
    annotation_headroom = max(max_total * 0.12, 8.0)
    use_period_labels = time_view in {"quarterly", "seasonally", "yearly"}
    annotations = (
        _build_december_yoy_annotations(total_series)
        if time_view == "monthly"
        else []
    )

    fig = go.Figure()
    for color_index, group_name in enumerate(pivot_df.columns):
        color = _resolve_country_color(group_name, color_index)
        rgb = _hex_to_rgb(color)

        fig.add_trace(
            go.Scatter(
                x=period_labels.tolist() if use_period_labels else pivot_df.index,
                y=pivot_df[group_name],
                mode="lines",
                name=group_name,
                line=dict(width=0.4, color=color),
                fill="tonexty",
                fillcolor=f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.78)",
                hovertemplate=(
                    f"<b>{group_name}</b><br>Date: %{{x}}"
                    "<br>Capacity: %{y:.1f} MTPA<extra></extra>"
                    if use_period_labels
                    else f"<b>{group_name}</b><br>Date: %{{x|%b %Y}}"
                    "<br>Capacity: %{y:.1f} MTPA<extra></extra>"
                ),
                stackgroup="one",
            )
        )

    start_date = pivot_df.index.min()
    end_date = pivot_df.index.max()

    fig.update_layout(
        xaxis=(
            dict(
                title="",
                type="category",
                categoryorder="array",
                categoryarray=period_labels.tolist(),
                tickfont=dict(size=10, family="Arial", color="#475569"),
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.18)",
                showline=False,
                zeroline=False,
            )
            if use_period_labels
            else dict(
                title="",
                range=[start_date, end_date],
                type="date",
                tickformat="%b\n%Y",
                dtick="M3",
                tickfont=dict(size=10, family="Arial", color="#475569"),
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.18)",
                showline=False,
                zeroline=False,
            )
        ),
        yaxis=dict(
            title=dict(
                text=y_axis_title,
                font=dict(size=12, family="Arial", color="#334155"),
            ),
            range=[0, max_total + annotation_headroom],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.18)",
            tickfont=dict(size=10, family="Arial", color="#475569"),
            tickformat=",.0f",
            showline=False,
            zeroline=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        margin=dict(l=80, r=40, t=32, b=135),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.96)",
            bordercolor="rgba(203, 213, 225, 0.9)",
            font=dict(size=11, family="Arial", color="#0f172a"),
        ),
        annotations=annotations,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=11, family="Arial", color="#334155"),
            bgcolor="rgba(0, 0, 0, 0)",
            borderwidth=0,
            itemwidth=70,
            itemsizing="constant",
        ),
    )

    return fig


def _create_yearly_capacity_comparison_chart(
    comparison_df: pd.DataFrame,
) -> go.Figure:
    if comparison_df.empty:
        return _create_empty_capacity_figure(
            "No overlapping yearly Woodmac and Energy Aspects values are available for the current selection."
        )

    plot_df = comparison_df.copy()
    years = plot_df["Year"].fillna("").astype(str).tolist()
    series_config = [
        ("Woodmac", "#1d4ed8"),
        ("Energy Aspects", "#f59e0b"),
        ("Internal Scenario", "#0f766e"),
    ]
    numeric_df = plot_df[[label for label, _color in series_config]].apply(pd.to_numeric, errors="coerce")
    max_value = float(numeric_df.max().max()) if numeric_df.notna().any().any() else 0.0
    min_value = float(numeric_df.min().min()) if numeric_df.notna().any().any() else 0.0
    tick_step = 50
    lower_bound = min_value
    upper_padding = max(float(tick_step), max(abs(max_value), 1.0) * 0.12)
    upper_bound = max_value + upper_padding
    if upper_bound <= lower_bound:
        upper_bound = lower_bound + float(tick_step)
    tick_start = math.floor(lower_bound / tick_step) * tick_step

    fig = go.Figure()
    for label, color in series_config:
        y_values = pd.to_numeric(plot_df[label], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=years,
                y=[None if pd.isna(value) else float(value) for value in y_values],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=color),
                connectgaps=False,
                hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Capacity: %{{y:.1f}} MTPA<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis=dict(
            title="",
            type="category",
            categoryorder="array",
            categoryarray=years,
            tickfont=dict(size=11, family="Arial", color="#475569"),
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="MTPA",
                font=dict(size=12, family="Arial", color="#334155"),
            ),
            range=[lower_bound, upper_bound],
            tickmode="linear",
            tick0=tick_start,
            dtick=tick_step,
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.18)",
            tickfont=dict(size=10, family="Arial", color="#475569"),
            tickformat=",.0f",
            showline=False,
            zeroline=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=460,
        margin=dict(l=72, r=28, t=28, b=76),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.96)",
            bordercolor="rgba(203, 213, 225, 0.9)",
            font=dict(size=11, family="Arial", color="#0f172a"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11, family="Arial", color="#334155"),
            bgcolor="rgba(0, 0, 0, 0)",
            borderwidth=0,
        ),
    )

    return fig


def _get_train_timeline_chart_options(
    selected_scenario_id,
    scenario_options_data,
) -> list[dict]:
    options = [
        {"label": "Woodmac", "value": "woodmac"},
        {"label": "Energy Aspects", "value": "energy_aspects"},
    ]
    scenario_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.notna(scenario_value):
        selected_option = _get_capacity_scenario_option_map(scenario_options_data).get(
            int(scenario_value),
            {},
        )
        options.append(
            {
                "label": selected_option.get("scenario_name", "Internal Scenario"),
                "value": "internal_scenario",
            }
        )
    return options


def _coerce_train_timeline_chart_value(
    current_value: object,
    valid_values: set[str],
    fallback_value: str,
) -> str:
    normalized_value = str(current_value or "").strip().casefold()
    if normalized_value in valid_values:
        return normalized_value
    return fallback_value


def _get_train_timeline_chart_option_label(
    options: list[dict] | None,
    option_value: str,
) -> str:
    normalized_value = str(option_value or "").strip().casefold()
    for option in options or []:
        if str(option.get("value") or "").strip().casefold() == normalized_value:
            return str(option.get("label") or option_value)

    return TRAIN_TIMELINE_CHART_SOURCE_CONFIG.get(
        normalized_value,
        {"label": str(option_value or "").strip() or "Source"},
    )["label"]


def _build_train_timeline_chart_rows(
    timeline_row_data: list[dict] | None,
    source_key: str,
    compare_key: str | None = None,
) -> pd.DataFrame:
    columns = [
        "Country",
        "Plant",
        "Train",
        "timeline_direction",
        "source_date",
        "source_capacity",
        "source_month_start",
        "source_out_of_range",
        "compare_date",
        "compare_out_of_range",
        "compare_missing",
        "train_display",
        "missing_train_label",
        "display_sort_plant",
        "display_sort_train",
    ]
    if not timeline_row_data:
        return pd.DataFrame(columns=columns)

    source_config = TRAIN_TIMELINE_CHART_SOURCE_CONFIG.get(str(source_key or "").strip().casefold())
    if not source_config:
        return pd.DataFrame(columns=columns)

    compare_config = TRAIN_TIMELINE_CHART_SOURCE_CONFIG.get(str(compare_key or "").strip().casefold())
    rows_df = pd.DataFrame(timeline_row_data).copy()
    if rows_df.empty:
        return pd.DataFrame(columns=columns)

    if "Country" not in rows_df.columns:
        rows_df["Country"] = ""
    if "Plant" not in rows_df.columns:
        rows_df["Plant"] = ""
    if "Train" not in rows_df.columns:
        rows_df["Train"] = ""
    rows_df["Country"] = rows_df["Country"].fillna("").astype(str).str.strip()
    rows_df["Plant"] = rows_df["Plant"].fillna("").astype(str).str.strip()
    rows_df["Train"] = rows_df["Train"].fillna("").astype(str).str.strip()
    rows_df["train_display"] = rows_df["Train"].where(rows_df["Train"].ne(""), "(No Train)")
    rows_df["display_sort_plant"] = rows_df.get("display_sort_plant", rows_df["Plant"]).fillna("")
    rows_df["display_sort_plant"] = rows_df["display_sort_plant"].where(
        rows_df["display_sort_plant"].astype(str).str.strip().ne(""),
        rows_df["Plant"],
    )
    explicit_train_sort = pd.to_numeric(
        rows_df.get("display_sort_train"),
        errors="coerce",
    )
    inferred_train_sort = pd.to_numeric(rows_df["Train"], errors="coerce")
    rows_df["display_sort_train"] = explicit_train_sort.where(
        explicit_train_sort.notna(), inferred_train_sort
    )
    rows_df["source_date"] = pd.to_datetime(
        rows_df.get(source_config["date_column"]),
        errors="coerce",
    )
    rows_df["source_capacity"] = pd.to_numeric(
        rows_df.get(source_config["capacity_column"]),
        errors="coerce",
    ).round(6)
    if source_config["out_of_range_flag"] in rows_df.columns:
        rows_df["source_out_of_range"] = rows_df[source_config["out_of_range_flag"]].map(
            _coerce_train_timeline_import_bool
        )
    else:
        rows_df["source_out_of_range"] = False

    source_direction = rows_df["source_capacity"].map(_normalize_capacity_change_direction)
    if "timeline_direction" in rows_df.columns:
        rows_df["timeline_direction"] = rows_df["timeline_direction"].where(
            rows_df["timeline_direction"].notna(),
            source_direction,
        )
    else:
        rows_df["timeline_direction"] = source_direction
    rows_df["timeline_direction"] = rows_df["timeline_direction"].where(
        source_direction.isna(),
        source_direction,
    )

    if compare_config:
        rows_df["compare_date"] = pd.to_datetime(
            rows_df.get(compare_config["date_column"]),
            errors="coerce",
        )
        if compare_config["out_of_range_flag"] in rows_df.columns:
            rows_df["compare_out_of_range"] = rows_df[compare_config["out_of_range_flag"]].map(
                _coerce_train_timeline_import_bool
            )
        else:
            rows_df["compare_out_of_range"] = False
    else:
        rows_df["compare_date"] = pd.NaT
        rows_df["compare_out_of_range"] = False
    rows_df["compare_missing"] = rows_df["compare_date"].isna()
    rows_df["missing_train_label"] = rows_df["train_display"].where(
        rows_df["compare_missing"],
        "",
    )

    visible_mask = (
        rows_df["Country"].ne("")
        & rows_df["Plant"].ne("")
        & rows_df["source_date"].notna()
        & rows_df["source_capacity"].notna()
        & rows_df["source_capacity"].round(6).ne(0)
        & rows_df["timeline_direction"].notna()
        & ~rows_df["source_out_of_range"]
    )
    rows_df = rows_df[visible_mask].copy()
    if rows_df.empty:
        return pd.DataFrame(columns=columns)

    rows_df["source_month_start"] = rows_df["source_date"].dt.to_period("M").dt.to_timestamp()
    rows_df["__train_blank_sort"] = rows_df["Train"].eq("").astype(int)
    rows_df = rows_df.sort_values(
        [
            "Country",
            "display_sort_plant",
            "__train_blank_sort",
            "display_sort_train",
            "Train",
            "source_date",
        ],
        ascending=[True, True, False, True, True, True],
        na_position="last",
    ).drop(columns=["__train_blank_sort"], errors="ignore").reset_index(drop=True)

    return rows_df[columns]


def _create_train_timeline_comparison_figure(
    timeline_row_data: list[dict] | None,
    source_key: str,
    compare_key: str,
    start_date: str | None,
    end_date: str | None,
    source_label: str,
    compare_label: str,
) -> go.Figure:
    event_df = _build_train_timeline_chart_rows(
        timeline_row_data,
        source_key=source_key,
        compare_key=compare_key,
    )
    if event_df.empty:
        return _create_empty_capacity_figure(
            f"No valid {source_label} train timeline rows are available for the current selection."
        )

    start_month = _normalize_month_date(start_date) or event_df["source_month_start"].min()
    end_month = _normalize_month_date(end_date) or event_df["source_month_start"].max()
    if start_month is None or end_month is None:
        return _create_empty_capacity_figure("No valid train timeline rows are available.")

    chart_start_date = start_month - pd.DateOffset(days=15)
    chart_end_date = end_month + pd.offsets.MonthEnd(1) + pd.DateOffset(days=15)
    compare_differs = str(source_key or "").strip().casefold() != str(compare_key or "").strip().casefold()

    monthly_df = (
        event_df.groupby(
            ["Country", "Plant", "source_month_start", "timeline_direction"],
            as_index=False,
            dropna=False,
        )
        .agg(
            {
                "source_capacity": "sum",
                "source_date": "count",
                "compare_missing": "sum",
                "missing_train_label": _combine_distinct_text_values,
                "display_sort_plant": "last",
                "display_sort_train": "last",
            }
        )
        .rename(
            columns={
                "source_month_start": "month_start",
                "source_capacity": "capacity",
                "source_date": "train_count",
            }
        )
    )
    monthly_df = monthly_df[monthly_df["capacity"].round(6).ne(0)].copy()
    if monthly_df.empty:
        return _create_empty_capacity_figure(
            f"No valid {source_label} train timeline rows are available for the current selection."
        )
    monthly_df["missing_count"] = pd.to_numeric(
        monthly_df["compare_missing"],
        errors="coerce",
    ).fillna(0).astype(int)
    monthly_df["train_count"] = pd.to_numeric(
        monthly_df["train_count"],
        errors="coerce",
    ).fillna(0).astype(int)
    monthly_df["missing_train_label"] = monthly_df["missing_train_label"].fillna("").astype(str).str.strip()
    monthly_df["__all_missing_in_compare"] = (
        monthly_df["missing_count"].gt(0)
        & monthly_df["missing_count"].eq(monthly_df["train_count"])
    )
    monthly_df["__mixed_missing_in_compare"] = (
        monthly_df["missing_count"].gt(0)
        & monthly_df["missing_count"].lt(monthly_df["train_count"])
    )

    plants_df = (
        monthly_df[
            [
                "Country",
                "Plant",
                "display_sort_plant",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["Country", "display_sort_plant", "Plant"],
            ascending=[True, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    max_abs_capacity = float(monthly_df["capacity"].abs().max()) if not monthly_df.empty else 0.0
    lane_half_height = max(max_abs_capacity * 1.15, 4.0)
    row_spacing = max(lane_half_height * 2.8, 14.0)
    annotation_padding = max(lane_half_height * 0.12, 0.8)
    count_offset = lane_half_height * 0.62
    arrow_offset = lane_half_height * 0.78
    missing_badge_offset = lane_half_height * 0.34
    missing_outline_color = "#d97706"
    has_missing_compare = bool(monthly_df["missing_count"].gt(0).any())

    fig = go.Figure()
    plant_to_y: dict[tuple[str, str], float] = {}
    y_ticks: list[float] = []
    y_labels: list[str] = []
    y_position = 0.0

    for _, plant_info in plants_df.iterrows():
        plant_key = (plant_info["Country"], plant_info["Plant"])
        plant_to_y[plant_key] = y_position
        y_ticks.append(y_position)
        y_labels.append(plant_info["Plant"])

        plant_months = monthly_df[
            (monthly_df["Country"] == plant_info["Country"])
            & (monthly_df["Plant"] == plant_info["Plant"])
        ]
        color = _resolve_country_color(plant_info["Country"], len(plant_to_y) - 1)

        for _, month_data in plant_months.iterrows():
            capacity = float(month_data["capacity"])
            month_date = pd.Timestamp(month_data["month_start"])
            missing_count = int(month_data["missing_count"])
            train_count = int(month_data["train_count"])
            missing_train_summary = str(month_data["missing_train_label"] or "").strip()
            hover_missing_line = f"<br>Missing in {compare_label}: No"
            if missing_count > 0 and compare_differs:
                hover_missing_line = (
                    f"<br>Missing in {compare_label}: Yes ({missing_count}/{train_count})"
                )
                if missing_train_summary:
                    hover_missing_line += f"<br>Missing trains: {missing_train_summary}"
            fig.add_trace(
                go.Bar(
                    x=[month_date],
                    y=[capacity],
                    base=y_position,
                    width=10 * 24 * 60 * 60 * 1000,
                    marker=dict(
                        color=color,
                        line=dict(
                            color=missing_outline_color if bool(month_data["__all_missing_in_compare"]) and compare_differs else color,
                            width=2.0 if bool(month_data["__all_missing_in_compare"]) and compare_differs else 0,
                        ),
                        opacity=0.85,
                    ),
                    orientation="v",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{plant_info['Plant']}</b><br>{plant_info['Country']}"
                        f"<br>Source: {source_label}"
                        f"<br>Direction: {str(month_data['timeline_direction']).title()}"
                        "<br>Capacity: %{y:.1f} MTPA"
                        "<br>Date: %{x|%b %Y}"
                        + hover_missing_line
                        + "<extra></extra>"
                    ),
                )
            )

            if abs(capacity) >= 3.0:
                label_y = y_position + capacity + (annotation_padding if capacity >= 0 else -annotation_padding)
                fig.add_annotation(
                    x=month_date,
                    y=label_y,
                    text=f"{capacity:.1f}",
                    showarrow=False,
                    font=dict(size=9, color=color, family="Arial", weight="bold"),
                    xanchor="center",
                    yanchor="bottom" if capacity >= 0 else "top",
                )

            if int(month_data["train_count"]) > 1:
                fig.add_annotation(
                    x=month_date,
                    y=y_position - count_offset if capacity >= 0 else y_position + count_offset,
                    text=str(int(month_data["train_count"])),
                    showarrow=False,
                    font=dict(size=7, color=color, family="Arial"),
                    xanchor="center",
                    yanchor="top" if capacity >= 0 else "bottom",
                    opacity=0.9,
                )

            if bool(month_data["__mixed_missing_in_compare"]) and compare_differs:
                missing_badge_y = (
                    y_position + capacity + (annotation_padding + missing_badge_offset)
                    if capacity >= 0
                    else y_position + capacity - (annotation_padding + missing_badge_offset)
                )
                fig.add_annotation(
                    x=month_date,
                    y=missing_badge_y,
                    text=str(missing_count),
                    showarrow=False,
                    font=dict(size=8, color=missing_outline_color, family="Arial", weight="bold"),
                    bgcolor="rgba(251, 191, 36, 0.18)",
                    bordercolor=missing_outline_color,
                    borderwidth=0.8,
                    borderpad=2,
                    xanchor="center",
                    yanchor="bottom" if capacity >= 0 else "top",
                )

        y_position += row_spacing

    for idx, (_, plant_info) in enumerate(plants_df.iterrows()):
        color = _resolve_country_color(plant_info["Country"], idx)
        y_center = y_ticks[idx]
        y_top = y_center - row_spacing / 2 if idx == 0 else (y_ticks[idx - 1] + y_ticks[idx]) / 2
        y_bottom = y_center + row_spacing / 2 if idx == len(y_ticks) - 1 else (y_ticks[idx] + y_ticks[idx + 1]) / 2
        fig.add_shape(
            type="rect",
            x0=chart_start_date,
            x1=chart_end_date,
            y0=y_top,
            y1=y_bottom,
            fillcolor=color,
            opacity=0.05,
            line_width=0,
            layer="below",
        )
        fig.add_shape(
            type="line",
            x0=chart_start_date,
            x1=chart_end_date,
            y0=y_center,
            y1=y_center,
            line=dict(color="rgba(148, 163, 184, 0.25)", width=0.6),
            layer="below",
        )

    for year in range(start_month.year, end_month.year + 1):
        year_start = pd.Timestamp(f"{year}-01-01")
        if chart_start_date <= year_start <= chart_end_date:
            fig.add_shape(
                type="line",
                x0=year_start,
                x1=year_start,
                y0=(y_ticks[0] - row_spacing / 2) if y_ticks else 0,
                y1=(y_ticks[-1] + row_spacing / 2) if y_ticks else 1,
                line=dict(color="#999999", width=1, dash="solid"),
                opacity=0.35,
                layer="below",
            )

    first_of_month = pd.Timestamp.now().to_period("M").to_timestamp()
    if chart_start_date <= first_of_month <= chart_end_date and y_ticks:
        fig.add_shape(
            type="line",
            x0=first_of_month,
            x1=first_of_month,
            y0=y_ticks[0] - row_spacing / 2,
            y1=y_ticks[-1] + row_spacing / 2,
            line=dict(color="#E74C3C", width=2, dash="dash"),
            layer="above",
        )
        fig.add_annotation(
            x=first_of_month,
            y=y_ticks[-1] + row_spacing * 0.42,
            text="Today",
            showarrow=False,
            font=dict(size=10, color="#E74C3C", family="Arial", weight="bold"),
            bgcolor="white",
            bordercolor="#E74C3C",
            borderwidth=1,
            borderpad=3,
        )

    if compare_differs:
        arrow_df = event_df[
            event_df["compare_date"].notna()
            & ~event_df["compare_out_of_range"]
            & event_df["source_date"].ne(event_df["compare_date"])
        ].copy()
        if not arrow_df.empty:
            visible_arrow_mask = (
                (arrow_df["source_date"] >= chart_start_date)
                & (arrow_df["source_date"] <= chart_end_date)
                & (arrow_df["compare_date"] >= chart_start_date)
                & (arrow_df["compare_date"] <= chart_end_date)
            )
            arrow_df = arrow_df[visible_arrow_mask].copy()

        for _, row in arrow_df.iterrows():
            y_pos = plant_to_y.get((row["Country"], row["Plant"]))
            if y_pos is None:
                continue

            source_date = pd.Timestamp(row["source_date"])
            compare_date = pd.Timestamp(row["compare_date"])
            moved_earlier = source_date < compare_date
            arrow_color = "#2E8B57" if moved_earlier else "#B22222"
            arrow_y = y_pos - arrow_offset if row["timeline_direction"] == "addition" else y_pos + arrow_offset
            label_y = arrow_y + annotation_padding if row["timeline_direction"] == "addition" else arrow_y - annotation_padding
            days_diff = abs((source_date - compare_date).days)
            months_diff = max(1, round(days_diff / 30.44))

            fig.add_annotation(
                x=source_date,
                y=arrow_y,
                ax=compare_date,
                ay=arrow_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=arrow_color,
                opacity=0.82,
            )
            fig.add_annotation(
                x=compare_date + (source_date - compare_date) / 2,
                y=label_y,
                text=f"{months_diff:.0f}m",
                showarrow=False,
                font=dict(size=8, color=arrow_color, family="Arial", weight="bold"),
                bgcolor="white",
                bordercolor=arrow_color,
                borderwidth=0.5,
                borderpad=2,
                opacity=0.9,
                yanchor="bottom" if row["timeline_direction"] == "addition" else "top",
            )

    month_span = (_month_difference(start_month, end_month) or 0) + 1
    if month_span <= 18:
        dtick = "M3"
        tickformat = "%b\n%Y"
    elif month_span <= 48:
        dtick = "M6"
        tickformat = "%b\n%Y"
    else:
        dtick = "M12"
        tickformat = "%Y"

    y_min = y_ticks[0] - row_spacing / 2 if y_ticks else -10
    y_max = y_ticks[-1] + row_spacing / 2 if y_ticks else 10
    note_text = (
        "Bars show the selected source in MTPA • Reductions are drawn downward • "
        "Numbers near bars indicate multiple train rows in the same plant/month/sign bucket"
    )
    if compare_differs:
        note_text += f" • Arrows show date shifts from {compare_label} to {source_label}"
        if has_missing_compare:
            note_text += f" • Amber outline / badge = present in {source_label} but missing in {compare_label}"

    fig.update_layout(
        xaxis=dict(
            title="",
            range=[chart_start_date, chart_end_date],
            type="date",
            tickformat=tickformat,
            dtick=dtick,
            tickfont=dict(size=10, family="Arial", color="#333333"),
            tickangle=0,
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="#CCCCCC",
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            range=[y_min, y_max],
            tickmode="array",
            tickvals=y_ticks,
            ticktext=y_labels,
            tickfont=dict(size=11, family="Arial", color="#333333"),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="#CCCCCC",
            zeroline=False,
            side="left",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(820, len(y_ticks) * 56),
        margin=dict(l=220, r=180, t=24, b=95),
        hovermode="closest",
        showlegend=False,
        bargap=0,
        barmode="overlay",
        uirevision="capacity-train-timeline-chart",
        dragmode="pan",
    )

    current_country = None
    country_start_idx = 0
    for idx, (_, plant_info) in enumerate(plants_df.iterrows()):
        country = plant_info["Country"]
        if country != current_country:
            if current_country is not None:
                mid_y = (y_ticks[country_start_idx] + y_ticks[idx - 1]) / 2
                fig.add_annotation(
                    x=1.005,
                    y=mid_y,
                    xref="paper",
                    yref="y",
                    text=current_country,
                    showarrow=False,
                    font=dict(
                        size=12,
                        color=_resolve_country_color(current_country, country_start_idx),
                        family="Arial",
                        weight="bold",
                    ),
                    xanchor="left",
                    yanchor="middle",
                )
            current_country = country
            country_start_idx = idx

    if current_country is not None and y_ticks:
        mid_y = (y_ticks[country_start_idx] + y_ticks[-1]) / 2
        fig.add_annotation(
            x=1.005,
            y=mid_y,
            xref="paper",
            yref="y",
            text=current_country,
            showarrow=False,
            font=dict(
                size=12,
                color=_resolve_country_color(current_country, country_start_idx),
                family="Arial",
                weight="bold",
            ),
            xanchor="left",
            yanchor="middle",
        )

    fig.add_annotation(
        x=0,
        y=-0.08,
        xref="paper",
        yref="paper",
        text=note_text,
        showarrow=False,
        font=dict(size=9, color="#666666", family="Arial", style="italic"),
        xanchor="left",
        yanchor="top",
    )

    return fig


def _build_train_change_log(
    raw_df: pd.DataFrame,
    selected_countries: list[str] | None,
    other_countries_mode: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    empty_columns = [
        "series_key",
        "terminal_key",
        "train_key",
        "Canonical Train Label",
        "Provider Plant ID",
        "Provider Train ID",
        "Provider Sources",
        "Effective Date",
        "Country",
        "Plant",
        "Train",
        "Delta MTPA",
        "Abs Delta",
        "Source Field",
        "Source Name",
        "Parent Source Field",
        "Parent Source Name",
        "Train Source Field",
        "Train Source Name",
        "Train Display Source Name",
        "Woodmac FID Date",
        "Mapping Applied",
        "Train Mapping Applied",
    ]
    if raw_df.empty:
        return pd.DataFrame(columns=empty_columns)

    identity_columns = {
        "terminal_key",
        "train_key",
        "Canonical Train Label",
        "Provider Plant ID",
        "Provider Train ID",
        "Provider Sources",
    }
    legacy_event_columns = [
        column_name for column_name in empty_columns if column_name not in identity_columns
    ]
    if set(legacy_event_columns).issubset(raw_df.columns):
        change_df = raw_df.copy()
        for column_name in identity_columns:
            if column_name not in change_df.columns:
                change_df[column_name] = None
        change_df = change_df[empty_columns]
        effective_dates = pd.to_datetime(
            change_df["Effective Date"],
            errors="coerce",
        ).dt.to_period("M").dt.to_timestamp()
        start_month = _normalize_month_date(start_date)
        end_month = _normalize_month_date(end_date)
        visible_mask = effective_dates.notna()
        if start_month is not None:
            visible_mask &= effective_dates.ge(start_month)
        if end_month is not None:
            visible_mask &= effective_dates.le(end_month)
        change_df = change_df[visible_mask].copy()

        if selected_countries:
            if other_countries_mode == "exclude":
                change_df = change_df[
                    change_df["Country"].isin(selected_countries)
                ].copy()
        elif other_countries_mode == "exclude":
            return pd.DataFrame(columns=empty_columns)

        if change_df.empty:
            return pd.DataFrame(columns=empty_columns)

        change_df["Effective Date"] = pd.to_datetime(
            change_df["Effective Date"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
        change_df["Train"] = pd.to_numeric(
            change_df["Train"],
            errors="coerce",
        ).astype("Int64")
        for column_name in ["Delta MTPA", "Abs Delta"]:
            change_df[column_name] = pd.to_numeric(
                change_df[column_name],
                errors="coerce",
            ).astype(float)
        for column_name in ["Mapping Applied", "Train Mapping Applied"]:
            change_df[column_name] = change_df[column_name].fillna(False).astype(bool)
        for column_name in identity_columns - {"Provider Sources"}:
            change_df[column_name] = (
                change_df[column_name].fillna("").astype(str)
            )
        change_df["Provider Sources"] = change_df["Provider Sources"].map(
            lambda value: value if isinstance(value, list) else []
        )
        return change_df.reset_index(drop=True)[empty_columns]

    required_columns = {
        "month",
        "country_name",
        "plant_name",
        "capacity_mtpa",
        "id_plant",
        "id_lng_train",
    }
    if not required_columns.issubset(set(raw_df.columns)):
        return pd.DataFrame(columns=empty_columns)

    train_df = raw_df.copy()
    train_df["month"] = pd.to_datetime(train_df["month"]).dt.to_period("M").dt.to_timestamp()

    def _get_first_available_series(
        column_names: list[str],
        default_value: object = None,
    ) -> pd.Series:
        for column_name in column_names:
            if column_name in train_df.columns:
                return train_df[column_name]
        return pd.Series(default_value, index=train_df.index)

    start_month = _normalize_month_date(start_date)
    end_month = _normalize_month_date(end_date)
    if end_month is None:
        end_month = train_df["month"].max()
    if start_month is None:
        start_month = train_df["month"].min()

    lookback_month = start_month - pd.DateOffset(months=1)
    train_df = train_df[
        (train_df["month"] >= lookback_month) & (train_df["month"] <= end_month)
    ].copy()

    if train_df.empty:
        return pd.DataFrame(columns=empty_columns)

    if selected_countries:
        if other_countries_mode == "exclude":
            train_df = train_df[train_df["country_name"].isin(selected_countries)].copy()
    elif other_countries_mode == "exclude":
        return pd.DataFrame(columns=empty_columns)

    if train_df.empty:
        return pd.DataFrame(columns=empty_columns)

    train_df["raw_plant_name"] = (
        _get_first_available_series(["raw_plant_name", "plant_name"], "")
        .fillna("")
        .astype(str)
        .str.strip()
    )
    train_df["raw_train_name"] = (
        _get_first_available_series(["raw_train_name", "lng_train_name_short"], "")
        .fillna("")
        .astype(str)
        .str.strip()
    )
    train_df["raw_train_display_name"] = (
        _get_first_available_series(
            ["raw_train_display_name", "lng_train_name", "raw_train_name", "lng_train_name_short"],
            "",
        )
        .fillna("")
        .astype(str)
        .str.strip()
    )
    train_df["woodmac_fid_date"] = (
        _get_first_available_series(["woodmac_fid_date"], "")
        .fillna("")
        .astype(str)
        .str.strip()
    )
    train_df["train"] = pd.to_numeric(
        _get_first_available_series(["train"]),
        errors="coerce",
    ).astype("Int64")
    train_df["train_mapping_applied"] = (
        _get_first_available_series(["train_mapping_applied"], False).fillna(False).astype(bool)
    )
    train_df["allocation_share"] = pd.to_numeric(
        _get_first_available_series(["allocation_share"], 1.0),
        errors="coerce",
    ).fillna(1.0)
    train_df["provider_sources"] = train_df.apply(
        lambda row: [
            {
                "provider_name": "woodmac",
                "provider_plant_id": str(int(row["id_plant"])),
                "provider_train_id": str(int(row["id_lng_train"])),
                "allocation_share": float(row["allocation_share"]),
            }
        ],
        axis=1,
    )
    train_df["capacity_mtpa"] = pd.to_numeric(train_df["capacity_mtpa"], errors="coerce").fillna(0.0)
    for column_name in [
        "terminal_key",
        "train_key",
        "canonical_terminal_name",
        "canonical_train_label",
    ]:
        train_df[column_name] = (
            _get_first_available_series([column_name], "")
            .fillna("")
            .astype(str)
            .str.strip()
        )

    plant_key = (
        train_df["country_name"].fillna("").astype(str).str.strip()
        + "|"
        + train_df["plant_name"].fillna("").astype(str).str.strip()
    )
    raw_series_key = (
        plant_key
        + "|RAW|"
        + train_df["raw_plant_name"]
        + "|"
        + train_df["raw_train_name"]
        + "|"
        + train_df["id_plant"].fillna(-1).astype(int).astype(str)
        + "|"
        + train_df["id_lng_train"].fillna(-1).astype(int).astype(str)
    )
    mapped_series_key = (
        plant_key
        + "|TRAIN|"
        + train_df["train"].astype("Int64").astype(str)
    )
    train_df["train_series_key"] = train_df["train"].map(
        lambda value: None if pd.isna(value) else str(int(value))
    )
    inferred_series_key = mapped_series_key.where(
        train_df["train_mapping_applied"] & train_df["train"].notna(),
        raw_series_key,
    )
    canonical_series_key = "CANONICAL|" + train_df["train_key"]
    train_df["series_key"] = canonical_series_key.where(
        train_df["train_key"].ne(""),
        inferred_series_key,
    )

    train_df = (
        train_df.groupby(["series_key", "month"], as_index=False, dropna=False)
        .agg(
            {
                "country_name": "last",
                "plant_name": "last",
                "raw_plant_name": "last",
                "raw_train_name": "last",
                "raw_train_display_name": "last",
                "woodmac_fid_date": _combine_distinct_text_values,
                "id_plant": "last",
                "id_lng_train": "last",
                "terminal_key": "last",
                "train_key": "last",
                "canonical_train_label": "last",
                "train": "last",
                "train_mapping_applied": "max",
                "train_series_key": "last",
                "capacity_mtpa": "sum",
                "provider_sources": _combine_provider_sources,
            }
        )
        .rename(columns={"plant_name": "plant_name"})
    )

    metadata_df = (
        train_df.sort_values("month")
        .groupby(["series_key"], as_index=False, dropna=False)
        .agg(
            {
                "country_name": "last",
                "plant_name": "last",
                "raw_plant_name": "last",
                "raw_train_name": "last",
                "raw_train_display_name": "last",
                "woodmac_fid_date": "last",
                "id_plant": "last",
                "id_lng_train": "last",
                "terminal_key": "last",
                "train_key": "last",
                "canonical_train_label": "last",
                "train": "last",
                "train_mapping_applied": "last",
                "provider_sources": _combine_provider_sources,
            }
        )
    )

    month_index = pd.date_range(start=lookback_month, end=end_month, freq="MS")
    train_keys_df = train_df[["series_key"]].drop_duplicates()
    train_keys_df["__join_key"] = 1
    month_frame = pd.DataFrame({"month": month_index, "__join_key": 1})
    expanded_df = train_keys_df.merge(month_frame, on="__join_key", how="inner").drop(
        columns="__join_key"
    )

    expanded_df = expanded_df.merge(
        train_df[
            [
                "series_key",
                "month",
                "capacity_mtpa",
            ]
        ],
        on=["series_key", "month"],
        how="left",
    )
    expanded_df = expanded_df.merge(
        metadata_df,
        on=["series_key"],
        how="left",
    )
    expanded_df["capacity_mtpa"] = expanded_df["capacity_mtpa"].fillna(0.0)
    expanded_df = expanded_df.sort_values(["series_key", "month"])
    expanded_df["previous_capacity_mtpa"] = (
        expanded_df.groupby(["series_key"])["capacity_mtpa"]
        .shift(1)
        .fillna(0.0)
    )
    expanded_df["delta_mtpa"] = (
        expanded_df["capacity_mtpa"] - expanded_df["previous_capacity_mtpa"]
    )

    change_df = expanded_df[
        (expanded_df["month"] >= start_month)
        & (expanded_df["month"] <= end_month)
        & (expanded_df["delta_mtpa"].round(6) != 0)
    ].copy()

    if change_df.empty:
        return pd.DataFrame(columns=empty_columns)

    change_df["Effective Date"] = change_df["month"].dt.strftime("%Y-%m-%d")
    change_df["Country"] = change_df["country_name"]
    change_df["Plant"] = change_df["plant_name"]
    change_df["Train"] = change_df["train"]
    change_df["Canonical Train Label"] = change_df["canonical_train_label"]
    change_df["Provider Plant ID"] = pd.to_numeric(
        change_df["id_plant"], errors="coerce"
    ).astype("Int64").astype(str)
    change_df["Provider Train ID"] = pd.to_numeric(
        change_df["id_lng_train"], errors="coerce"
    ).astype("Int64").astype(str)
    change_df["Provider Sources"] = change_df["provider_sources"]
    change_df["Delta MTPA"] = change_df["delta_mtpa"].round(6)
    change_df["Abs Delta"] = change_df["Delta MTPA"].abs()
    change_df["Source Field"] = "plant_name"
    change_df["Source Name"] = change_df["raw_plant_name"].where(
        change_df["raw_plant_name"].notna() & change_df["raw_plant_name"].ne(""),
        change_df["Plant"],
    )
    change_df["Parent Source Field"] = "plant_name"
    change_df["Parent Source Name"] = change_df["raw_plant_name"].where(
        change_df["raw_plant_name"].notna() & change_df["raw_plant_name"].ne(""),
        change_df["Plant"],
    )
    change_df["Train Source Field"] = "lng_train_name_short"
    change_df["Train Source Name"] = change_df["raw_train_name"]
    change_df["Train Display Source Name"] = change_df["raw_train_display_name"].where(
        change_df["raw_train_display_name"].notna() & change_df["raw_train_display_name"].ne(""),
        change_df["Train Source Name"],
    )
    change_df["Woodmac FID Date"] = change_df["woodmac_fid_date"].fillna("").astype(str).str.strip()
    change_df["Mapping Applied"] = change_df["train_key"].ne("")
    change_df["Train Mapping Applied"] = (
        change_df["train_mapping_applied"].fillna(False).astype(bool)
    )

    change_df = change_df.sort_values(
        ["month", "Abs Delta", "Country", "Plant", "Train"],
        ascending=[True, False, True, True, True],
    )

    return change_df[empty_columns]


def _build_ea_change_log(
    raw_df: pd.DataFrame,
    selected_countries: list[str] | None,
    other_countries_mode: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    empty_columns = [
        "terminal_key",
        "train_key",
        "Canonical Train Label",
        "Provider Plant ID",
        "Provider Train ID",
        "Provider Sources",
        "Effective Date",
        "Country",
        "Plant",
        "Train",
        "project_name",
        "train_name",
        "status",
        "EA Adds (MTPA)",
        "EA Reductions (MTPA)",
        "EA Net Delta (MTPA)",
        "EA Abs Delta",
        "Source Field",
        "Source Name",
        "Parent Source Field",
        "Parent Source Name",
        "Train Source Field",
        "Train Source Name",
        "Mapping Applied",
        "Train Mapping Applied",
    ]
    if raw_df.empty:
        return pd.DataFrame(columns=empty_columns)

    if set(empty_columns).issubset(raw_df.columns):
        change_df = raw_df[empty_columns].copy()
        effective_dates = pd.to_datetime(
            change_df["Effective Date"],
            errors="coerce",
        ).dt.to_period("M").dt.to_timestamp()
        start_month = _normalize_month_date(start_date)
        end_month = _normalize_month_date(end_date)
        visible_mask = effective_dates.notna()
        if start_month is not None:
            visible_mask &= effective_dates.ge(start_month)
        if end_month is not None:
            visible_mask &= effective_dates.le(end_month)
        change_df = change_df[visible_mask].copy()

        if selected_countries:
            if other_countries_mode == "exclude":
                change_df = change_df[
                    change_df["Country"].isin(selected_countries)
                ].copy()
        elif other_countries_mode == "exclude":
            return pd.DataFrame(columns=empty_columns)

        if change_df.empty:
            return pd.DataFrame(columns=empty_columns)

        change_df["Effective Date"] = pd.to_datetime(
            change_df["Effective Date"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
        change_df["Train"] = pd.to_numeric(
            change_df["Train"],
            errors="coerce",
        ).astype("Int64")
        for column_name in [
            "EA Adds (MTPA)",
            "EA Reductions (MTPA)",
            "EA Net Delta (MTPA)",
            "EA Abs Delta",
        ]:
            change_df[column_name] = pd.to_numeric(
                change_df[column_name],
                errors="coerce",
            ).astype(float)
        for column_name in ["Mapping Applied", "Train Mapping Applied"]:
            change_df[column_name] = change_df[column_name].fillna(False).astype(bool)
        for column_name in [
            "terminal_key",
            "train_key",
            "Canonical Train Label",
            "Provider Plant ID",
            "Provider Train ID",
        ]:
            change_df[column_name] = change_df[column_name].fillna("").astype(str)
        change_df["Provider Sources"] = change_df["Provider Sources"].map(
            lambda value: value if isinstance(value, list) else []
        )
        return change_df.reset_index(drop=True)[empty_columns]

    ea_df = raw_df.copy()
    ea_df["month"] = pd.to_datetime(ea_df["month"]).dt.to_period("M").dt.to_timestamp()

    start_month = _normalize_month_date(start_date)
    end_month = _normalize_month_date(end_date)
    if end_month is None:
        end_month = ea_df["month"].max()
    if start_month is None:
        start_month = ea_df["month"].min()

    ea_df = ea_df[
        (ea_df["month"] >= start_month) & (ea_df["month"] <= end_month)
    ].copy()

    if ea_df.empty:
        return pd.DataFrame(columns=empty_columns)

    if selected_countries:
        if other_countries_mode == "exclude":
            ea_df = ea_df[ea_df["country_name"].isin(selected_countries)].copy()
    elif other_countries_mode == "exclude":
        return pd.DataFrame(columns=empty_columns)

    if ea_df.empty:
        return pd.DataFrame(columns=empty_columns)

    ea_df["status"] = (
        ea_df["status"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )
    ea_df["status_key"] = ea_df["status"].str.casefold()
    ea_df = ea_df[~ea_df["status_key"].isin(EA_EXCLUDED_STATUSES)].copy()

    if ea_df.empty:
        return pd.DataFrame(columns=empty_columns)

    negative_mask = ea_df["status_key"].isin(EA_NEGATIVE_STATUSES)
    ea_df["EA Adds (MTPA)"] = ea_df["capacity_mtpa"].where(~negative_mask, 0.0).round(6)
    ea_df["EA Reductions (MTPA)"] = (-ea_df["capacity_mtpa"]).where(negative_mask, 0.0).round(6)
    ea_df["EA Net Delta (MTPA)"] = (
        ea_df["EA Adds (MTPA)"] + ea_df["EA Reductions (MTPA)"]
    ).round(6)
    ea_df["EA Abs Delta"] = ea_df["EA Net Delta (MTPA)"].abs()
    ea_df["Effective Date"] = ea_df["month"].dt.strftime("%Y-%m-%d")
    ea_df["Country"] = ea_df["country_name"]
    ea_df["Plant"] = ea_df["plant_name"].where(
        ea_df["plant_name"].notna() & ea_df["plant_name"].ne(""),
        ea_df["project_name"],
    )
    ea_df["Train"] = pd.to_numeric(ea_df.get("train"), errors="coerce").astype("Int64")
    for source_column, output_column in [
        ("terminal_key", "terminal_key"),
        ("train_key", "train_key"),
        ("canonical_train_label", "Canonical Train Label"),
        ("provider_plant_id", "Provider Plant ID"),
        ("provider_train_id", "Provider Train ID"),
    ]:
        ea_df[output_column] = (
            ea_df.get(source_column, pd.Series("", index=ea_df.index))
            .fillna("")
            .astype(str)
            .str.strip()
        )
    ea_df["Provider Sources"] = ea_df.apply(
        lambda row: [
            {
                "provider_name": "energy_aspects",
                "provider_plant_id": row["Provider Plant ID"],
                "provider_train_id": row["Provider Train ID"],
                "allocation_share": float(row.get("allocation_share") or 1.0),
            }
        ],
        axis=1,
    )
    ea_df["Source Field"] = "project_name"
    ea_df["Source Name"] = ea_df["project_name"]
    ea_df["Parent Source Field"] = "project_name"
    ea_df["Parent Source Name"] = ea_df["project_name"]
    ea_df["Train Source Field"] = "train_name"
    ea_df["Train Source Name"] = ea_df["train_name"]
    ea_df["Mapping Applied"] = ea_df.get("train_key", "").fillna("").astype(str).str.strip().ne("")
    ea_df["Train Mapping Applied"] = ea_df.get("train_mapping_applied", False).fillna(False).astype(bool)
    ea_df = ea_df.drop(columns=["status_key"])

    ea_df = (
        ea_df.groupby(
            [
                "month",
                "Effective Date",
                "Country",
                "Plant",
                "Train",
                "terminal_key",
                "train_key",
                "Canonical Train Label",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            {
                "project_name": "last",
                "train_name": "last",
                "status": "last",
                "EA Adds (MTPA)": "sum",
                "EA Reductions (MTPA)": "sum",
                "EA Net Delta (MTPA)": "sum",
                "EA Abs Delta": "sum",
                "Source Field": "last",
                "Source Name": "last",
                "Parent Source Field": "last",
                "Parent Source Name": "last",
                "Train Source Field": "last",
                "Train Source Name": "last",
                "Mapping Applied": "max",
                "Train Mapping Applied": "max",
                "Provider Plant ID": _combine_distinct_text_values,
                "Provider Train ID": _combine_distinct_text_values,
                "Provider Sources": _combine_provider_sources,
            }
        )
    )

    ea_df = ea_df.sort_values(
        ["month", "Country", "EA Abs Delta", "Plant", "Train", "project_name", "train_name"],
        ascending=[True, True, False, True, True, True, True],
    )

    return ea_df[empty_columns]


def _build_unmapped_train_alias_df(
    woodmac_change_df: pd.DataFrame,
    ea_change_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "country_name",
        "plant_name",
        "provider_name",
        "provider_plant_id",
        "provider_train_id",
        "original_plant",
        "original_train",
        "first_effective_date",
        "total_capacity_mtpa",
        "train",
        "allocation_share",
    ]

    alias_frames = []
    if not woodmac_change_df.empty:
        woodmac_alias_df = woodmac_change_df.copy()
        woodmac_alias_df["provider_name"] = "woodmac"
        woodmac_alias_df["country_name"] = woodmac_alias_df["Country"]
        woodmac_alias_df["plant_name"] = woodmac_alias_df["Plant"]
        woodmac_alias_df["provider_plant_id"] = woodmac_alias_df["Provider Plant ID"]
        woodmac_alias_df["provider_train_id"] = woodmac_alias_df["Provider Train ID"]
        woodmac_alias_df["original_plant"] = woodmac_alias_df["Source Name"]
        woodmac_alias_df["original_train"] = woodmac_alias_df["Train Source Name"]
        woodmac_alias_df["first_effective_date"] = woodmac_alias_df["Effective Date"]
        woodmac_alias_df["total_capacity_mtpa"] = woodmac_alias_df["Abs Delta"].fillna(0.0)
        alias_frames.append(
            woodmac_alias_df.loc[
                woodmac_alias_df["train_key"].fillna("").astype(str).str.strip().eq(""),
                columns[:9],
            ]
        )

    if not ea_change_df.empty:
        ea_alias_df = ea_change_df.copy()
        ea_alias_df["provider_name"] = "energy_aspects"
        ea_alias_df["country_name"] = ea_alias_df["Country"]
        ea_alias_df["plant_name"] = ea_alias_df["Plant"]
        ea_alias_df["provider_plant_id"] = ea_alias_df["Provider Plant ID"]
        ea_alias_df["provider_train_id"] = ea_alias_df["Provider Train ID"]
        ea_alias_df["original_plant"] = ea_alias_df["Source Name"]
        ea_alias_df["original_train"] = ea_alias_df["Train Source Name"]
        ea_alias_df["first_effective_date"] = ea_alias_df["Effective Date"]
        ea_alias_df["total_capacity_mtpa"] = ea_alias_df["EA Abs Delta"].fillna(0.0)
        alias_frames.append(
            ea_alias_df.loc[
                ea_alias_df["train_key"].fillna("").astype(str).str.strip().eq(""),
                columns[:9],
            ]
        )

    if not alias_frames:
        return pd.DataFrame(columns=columns)

    alias_df = pd.concat(alias_frames, ignore_index=True)
    for column_name in [
        "country_name",
        "plant_name",
        "provider_name",
        "provider_plant_id",
        "provider_train_id",
        "original_plant",
        "original_train",
    ]:
        alias_df[column_name] = alias_df[column_name].fillna("").astype(str).str.strip()

    alias_df = alias_df[
        alias_df["provider_plant_id"].ne("")
        & alias_df["provider_train_id"].ne("")
        & alias_df["plant_name"].ne("")
    ].copy()
    if alias_df.empty:
        return pd.DataFrame(columns=columns)

    alias_df = (
        alias_df.groupby(
            [
                "country_name",
                "plant_name",
                "provider_name",
                "provider_plant_id",
                "provider_train_id",
                "original_plant",
                "original_train",
            ],
            as_index=False,
        )
        .agg(
            {
                "first_effective_date": "min",
                "total_capacity_mtpa": "sum",
            }
        )
    )
    alias_df["total_capacity_mtpa"] = alias_df["total_capacity_mtpa"].round(2)
    alias_df["train"] = None
    alias_df["allocation_share"] = 1.0
    alias_df = alias_df.sort_values(
        [
            "first_effective_date",
            "country_name",
            "plant_name",
            "provider_name",
            "total_capacity_mtpa",
            "provider_plant_id",
            "provider_train_id",
        ],
        ascending=[True, True, True, True, False, True, True],
    ).reset_index(drop=True)

    return alias_df[columns]


def _build_unmapped_train_summary(alias_df: pd.DataFrame) -> html.Div:
    if alias_df.empty:
        return html.Div(
            [
                html.Div(
                    "No unmapped train aliases in the complete source snapshot.",
                    className="balance-summary-row",
                ),
                html.Div(
                    [
                        html.Span(
                            "Only rows with a resolved plant but no canonical train assignment appear here for review."
                        )
                    ],
                    className="balance-metadata-row",
                ),
            ]
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"{len(alias_df):,} unmapped train aliases"),
                    html.Span(
                        f"{alias_df['plant_name'].nunique():,} plants in the source snapshot"
                    ),
                    html.Span(
                        f"{alias_df['country_name'].nunique():,} countries represented"
                    ),
                ],
                className="balance-summary-row",
            ),
            html.Div(
                [
                    html.Span(
                        "Assign the canonical train and allocation share, then save the complete provider-source allocation."
                    )
                ],
                className="balance-metadata-row",
            ),
        ]
    )


def _create_unmapped_train_mapping_table(
    table_id: str,
    alias_df: pd.DataFrame | None = None,
) -> dag.AgGrid:
    alias_df = pd.DataFrame() if alias_df is None else alias_df.copy()
    columns = [
        {"name": "Country", "id": "country_name", "editable": False},
        {"name": "Plant", "id": "plant_name", "editable": False},
        {"name": "Provider", "id": "provider_name", "editable": False},
        {"name": "Provider Plant ID", "id": "provider_plant_id", "editable": False},
        {"name": "Provider Train ID", "id": "provider_train_id", "editable": False},
        {"name": "Raw Plant", "id": "original_plant", "editable": False},
        {"name": "Raw Train", "id": "original_train", "editable": False},
        {"name": "First Date", "id": "first_effective_date", "editable": False},
        {
            "name": "Visible Capacity (MTPA)",
            "id": "total_capacity_mtpa",
            "editable": False,
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": "Canonical Train",
            "id": "train",
            "editable": True,
        },
        {
            "name": "Allocation Share",
            "id": "allocation_share",
            "editable": True,
            "type": "numeric",
            "format": Format(precision=4, scheme=Scheme.fixed),
        },
    ]

    return create_ag_grid_from_datatable(
        id=table_id,
        columns=columns,
        data=alias_df.to_dict("records"),
        editable=True,
        sort_action="native",
        filter_action="native",
        page_action="none",
        style_table={
            "overflowX": "auto",
            "borderRadius": "4px",
            "border": "1px solid #e2e8f0",
            "maxHeight": "420px",
        },
        style_cell_conditional=[
            {"if": {"column_id": "country_name"}, "minWidth": "120px", "maxWidth": "160px"},
            {"if": {"column_id": "plant_name"}, "minWidth": "180px", "maxWidth": "240px"},
            {"if": {"column_id": "provider_name"}, "minWidth": "120px", "maxWidth": "150px"},
            {"if": {"column_id": "provider_plant_id"}, "minWidth": "170px", "maxWidth": "260px"},
            {"if": {"column_id": "provider_train_id"}, "minWidth": "170px", "maxWidth": "260px"},
            {"if": {"column_id": "original_plant"}, "minWidth": "180px", "maxWidth": "260px"},
            {"if": {"column_id": "original_train"}, "minWidth": "160px", "maxWidth": "240px"},
        ],
        style_data_conditional=[
            {
                "if": {"column_id": "plant_name"},
                "backgroundColor": "#f8fafc",
                "fontWeight": "600",
                "color": "#1e3a5f",
            },
            {
                "if": {"column_id": "original_train"},
                "backgroundColor": "#f8fafc",
                "fontWeight": "600",
                "color": "#1e3a5f",
            },
            {
                "if": {"column_id": "train"},
                "backgroundColor": "rgba(34, 197, 94, 0.08)",
                "fontWeight": "600",
            },
            {
                "if": {"column_id": "allocation_share"},
                "backgroundColor": "rgba(34, 197, 94, 0.08)",
            },
        ],
        fill_width=False,
    )


def _create_unmapped_train_mapping_section(
    title: str,
    subtitle: str,
    summary_id: str,
    message_id: str,
    save_button_id: str,
    table_id: str,
) -> html.Div:
    header_children = [
        html.Div(
            [
                html.H3(title, className="balance-section-title"),
                html.Button(
                    "Save Train Mappings",
                    id=save_button_id,
                    n_clicks=0,
                    style=PRIMARY_CONTROL_BUTTON_STYLE,
                ),
            ],
            className="inline-section-header",
            style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
        )
    ]

    if subtitle:
        header_children.append(html.P(subtitle, className="balance-section-subtitle"))

    header_children.extend(
        [
            html.Div(id=summary_id),
            html.Div(id=message_id, style={"paddingBottom": "10px"}),
            _create_unmapped_train_mapping_table(table_id),
        ]
    )

    return html.Div(
        [html.Div(header_children, className="balance-section-header")],
        className="balance-section-card",
    )


def _build_capacity_table_delta_view(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    delta_df = df.copy()
    numeric_columns = [column for column in delta_df.columns if column not in {"Month", "__axis_date"}]
    if not numeric_columns:
        return delta_df

    delta_df[numeric_columns] = (
        delta_df[numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .diff()
    )
    delta_df.loc[delta_df.index[0], numeric_columns] = pd.NA
    return delta_df


def _apply_capacity_table_view(df: pd.DataFrame, table_view: str) -> pd.DataFrame:
    if table_view == "change":
        return _build_capacity_table_delta_view(df)

    return df.copy()


def _create_capacity_table(
    table_id: str,
    df: pd.DataFrame,
    table_view: str = "absolute",
) -> dag.AgGrid | html.Div:
    if df.empty:
        return _create_empty_state("No data available for the current selection.")

    display_df = df.drop(columns=["__axis_date"], errors="ignore").copy()
    base_config = StandardTableStyleManager.get_base_datatable_config()
    numeric_columns = [column for column in display_df.columns if column != "Month"]

    columns = [{"name": "Month", "id": "Month"}]
    columns.extend(
        {
            "name": column_name,
            "id": column_name,
            "type": "numeric",
            "format": Format(precision=1, scheme=Scheme.fixed),
        }
        for column_name in numeric_columns
    )

    style_data_conditional = list(base_config["style_data_conditional"])
    style_data_conditional.append(
        {
            "if": {"column_id": "Month"},
            "backgroundColor": "#f8fafc",
            "fontWeight": "600",
            "color": TABLE_COLORS["text_primary"],
        }
    )

    for column_name in numeric_columns:
        if str(column_name).startswith("Total "):
            style_data_conditional.append(
                {
                    "if": {"column_id": column_name},
                    "backgroundColor": "#edf6fd",
                    "fontWeight": "700",
                    "color": TABLE_COLORS["primary_dark"],
                }
            )
        elif column_name == "Rest of the World":
            style_data_conditional.append(
                {
                    "if": {"column_id": column_name},
                    "backgroundColor": "#f8f9fa",
                    "color": TABLE_COLORS["text_secondary"],
                }
            )

    if table_view == "change":
        for column_name in numeric_columns:
            style_data_conditional.extend(
                [
                    {
                        "if": {
                            "column_id": column_name,
                            "filter_query": f"{{{column_name}}} > 0",
                        },
                        "color": "#166534",
                        "fontWeight": "700",
                    },
                    {
                        "if": {
                            "column_id": column_name,
                            "filter_query": f"{{{column_name}}} < 0",
                        },
                        "color": "#991b1b",
                        "fontWeight": "700",
                    },
                    {
                        "if": {
                            "column_id": column_name,
                            "filter_query": f"{{{column_name}}} = 0",
                        },
                        "color": "#64748b",
                    },
                ]
            )

    return create_ag_grid_from_datatable(
        id=table_id,
        columns=columns,
        data=display_df.to_dict("records"),
        sort_action="native",
        page_action="none",
        fill_width=False,
        fixed_columns={"headers": True, "data": 1},
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": "560px",
            "width": "100%",
            "minWidth": "100%",
        },
        style_cell_conditional=_build_responsive_column_styles(display_df),
        style_data_conditional=style_data_conditional,
    )


def _create_yearly_capacity_comparison_table(
    table_id: str,
    comparison_df: pd.DataFrame,
) -> dag.AgGrid | html.Div:
    if comparison_df.empty:
        return _create_empty_state(
            "No overlapping yearly Woodmac and Energy Aspects values are available for the current selection."
        )

    display_df = comparison_df.copy()
    display_df = display_df.where(pd.notna(display_df), None)
    base_config = StandardTableStyleManager.get_base_datatable_config()
    numeric_columns = [
        "Internal Scenario",
        "Woodmac",
        "Energy Aspects",
        "Delta vs Woodmac",
        "Delta vs Energy Aspects",
    ]
    columns = [
        {"name": "Year", "id": "Year"},
        *[
            {
                "name": column_name,
                "id": column_name,
                "type": "numeric",
                "format": Format(precision=1, scheme=Scheme.fixed),
            }
            for column_name in numeric_columns
        ],
    ]

    style_data_conditional = list(base_config["style_data_conditional"])
    style_data_conditional.extend(
        [
            {
                "if": {"column_id": "Year"},
                "backgroundColor": "#f8fafc",
                "fontWeight": "700",
                "color": TABLE_COLORS["text_primary"],
            },
            {
                "if": {"column_id": "Internal Scenario"},
                "backgroundColor": "rgba(15, 118, 110, 0.08)",
                "fontWeight": "600",
                "color": "#0f766e",
            },
            {
                "if": {"column_id": "Woodmac"},
                "backgroundColor": "rgba(29, 78, 216, 0.08)",
                "fontWeight": "600",
                "color": "#1d4ed8",
            },
            {
                "if": {"column_id": "Energy Aspects"},
                "backgroundColor": "rgba(245, 158, 11, 0.08)",
                "fontWeight": "600",
                "color": "#b45309",
            },
        ]
    )

    for column_name in ["Delta vs Woodmac", "Delta vs Energy Aspects"]:
        style_data_conditional.extend(
            [
                {
                    "if": {
                        "column_id": column_name,
                        "filter_query": f"{{{column_name}}} > 0",
                    },
                    "color": "#166534",
                    "fontWeight": "700",
                },
                {
                    "if": {
                        "column_id": column_name,
                        "filter_query": f"{{{column_name}}} < 0",
                    },
                    "color": "#991b1b",
                    "fontWeight": "700",
                },
                {
                    "if": {
                        "column_id": column_name,
                        "filter_query": f"{{{column_name}}} = 0",
                    },
                    "color": "#64748b",
                    "fontWeight": "600",
                },
            ]
        )

    return create_ag_grid_from_datatable(
        id=table_id,
        columns=columns,
        data=display_df.to_dict("records"),
        sort_action="native",
        page_action="none",
        fill_width=True,
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": "460px",
            "width": "100%",
            "minWidth": "100%",
        },
        style_cell_conditional=_build_yearly_capacity_comparison_column_styles(),
        style_data_conditional=style_data_conditional,
    )


def _create_provider_capacity_discrepancy_table(
    table_id: str,
    discrepancy_df: pd.DataFrame,
    provider: str,
    empty_message: str | None = None,
) -> dag.AgGrid | html.Div:
    provider_config = _get_provider_discrepancy_config(provider)
    provider_key = str(provider_config["provider_key"])
    provider_capacity_column = str(provider_config["provider_capacity_display_column"])
    entity_column = str(provider_config["entity_output_column"])
    provider_color = str(provider_config["provider_color"])

    if discrepancy_df.empty:
        return _create_empty_state(empty_message or str(provider_config["capacity_empty_message"]))

    display_df = discrepancy_df.copy().where(pd.notna(discrepancy_df), None)
    base_config = StandardTableStyleManager.get_base_datatable_config()
    numeric_columns = [
        provider_capacity_column,
        "Scenario Capacity",
        "Abs Capacity Delta",
    ]
    columns = []
    for column_name in display_df.columns:
        column_config = {"name": column_name, "id": column_name}
        if column_name in numeric_columns:
            column_config |= {
                "type": "numeric",
                "format": Format(precision=1, scheme=Scheme.fixed),
            }
        columns.append(column_config)

    style_data_conditional = list(base_config["style_data_conditional"])
    style_data_conditional.extend(
        [
            {
                "if": {"column_id": entity_column},
                "backgroundColor": "#f8fafc",
                "fontWeight": "700",
                "color": TABLE_COLORS["text_primary"],
            },
            {
                "if": {"column_id": provider_capacity_column},
                "backgroundColor": (
                    "rgba(29, 78, 216, 0.08)"
                    if provider_key == "woodmac"
                    else "rgba(245, 158, 11, 0.08)"
                ),
                "fontWeight": "600",
                "color": provider_color,
            },
            {
                "if": {"column_id": "Scenario Capacity"},
                "backgroundColor": "rgba(15, 118, 110, 0.08)",
                "fontWeight": "600",
                "color": "#0f766e",
            },
            {
                "if": {
                    "column_id": "Abs Capacity Delta",
                    "filter_query": "{Abs Capacity Delta} > 0",
                },
                "backgroundColor": "rgba(190, 24, 93, 0.08)",
                "color": "#be123c",
                "fontWeight": "700",
            },
            {
                "if": {
                    "column_id": "Abs Capacity Delta",
                    "filter_query": "{Abs Capacity Delta} = 0",
                },
                "color": "#64748b",
            },
        ]
    )

    return create_ag_grid_from_datatable(
        id=table_id,
        columns=columns,
        data=display_df.to_dict("records"),
        sort_action="native",
        sort_by=[{"column_id": "Abs Capacity Delta", "direction": "desc"}],
        page_action="none",
        fill_width=True,
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": YEARLY_PROVIDER_DISCREPANCY_TABLE_MAX_HEIGHT,
            "width": "100%",
            "minWidth": "100%",
        },
        style_cell_conditional=_build_yearly_discrepancy_column_styles(provider),
        style_data_conditional=style_data_conditional,
    )


def _create_provider_timeline_discrepancy_table(
    table_id: str,
    discrepancy_df: pd.DataFrame,
    provider: str,
    empty_message: str | None = None,
) -> dag.AgGrid | html.Div:
    provider_config = _get_provider_discrepancy_config(provider)
    provider_key = str(provider_config["provider_key"])
    provider_date_column = str(provider_config["provider_date_column"])
    entity_column = str(provider_config["entity_output_column"])
    provider_color = str(provider_config["provider_color"])

    if discrepancy_df.empty:
        return _create_empty_state(empty_message or str(provider_config["timeline_empty_message"]))

    display_df = discrepancy_df.copy().where(pd.notna(discrepancy_df), None)
    base_config = StandardTableStyleManager.get_base_datatable_config()
    columns = []
    for column_name in display_df.columns:
        column_config = {"name": column_name, "id": column_name}
        if column_name == "Abs Timeline Delta (Months)":
            column_config |= {
                "type": "numeric",
                "format": Format(precision=0, scheme=Scheme.fixed),
            }
        columns.append(column_config)

    style_data_conditional = list(base_config["style_data_conditional"])
    style_data_conditional.extend(
        [
            {
                "if": {"column_id": entity_column},
                "backgroundColor": "#f8fafc",
                "fontWeight": "700",
                "color": TABLE_COLORS["text_primary"],
            },
            {
                "if": {"column_id": provider_date_column},
                "backgroundColor": (
                    "rgba(29, 78, 216, 0.08)"
                    if provider_key == "woodmac"
                    else "rgba(245, 158, 11, 0.08)"
                ),
                "fontWeight": "600",
                "color": provider_color,
            },
            {
                "if": {"column_id": "Scenario First Date"},
                "backgroundColor": "rgba(15, 118, 110, 0.08)",
                "fontWeight": "600",
                "color": "#0f766e",
            },
            {
                "if": {
                    "column_id": "Abs Timeline Delta (Months)",
                    "filter_query": "{Abs Timeline Delta (Months)} > 0",
                },
                "backgroundColor": "rgba(190, 24, 93, 0.08)",
                "color": "#be123c",
                "fontWeight": "700",
            },
        ]
    )

    return create_ag_grid_from_datatable(
        id=table_id,
        columns=columns,
        data=display_df.to_dict("records"),
        sort_action="native",
        sort_by=[{"column_id": "Abs Timeline Delta (Months)", "direction": "desc"}],
        page_action="none",
        fill_width=True,
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": YEARLY_PROVIDER_DISCREPANCY_TABLE_MAX_HEIGHT,
            "width": "100%",
            "minWidth": "100%",
        },
        style_cell_conditional=_build_yearly_timeline_discrepancy_column_styles(provider),
        style_data_conditional=style_data_conditional,
    )


def _create_provider_missing_internal_scenario_table(
    table_id: str,
    missing_df: pd.DataFrame,
    provider: str,
    empty_message: str | None = None,
) -> dag.AgGrid | html.Div:
    provider_config = _get_provider_discrepancy_config(provider)
    provider_key = str(provider_config["provider_key"])
    provider_date_column = str(provider_config["provider_date_column"])
    entity_column = str(provider_config["entity_output_column"])
    provider_capacity_column = str(provider_config["provider_capacity_display_column"])
    provider_color = str(provider_config["provider_color"])

    if missing_df.empty:
        return _create_empty_state(empty_message or str(provider_config["missing_empty_message"]))

    display_df = missing_df.copy().where(pd.notna(missing_df), None)
    base_config = StandardTableStyleManager.get_base_datatable_config()
    columns = []
    for column_name in display_df.columns:
        column_config = {"name": column_name, "id": column_name}
        if column_name == provider_capacity_column:
            column_config |= {
                "type": "numeric",
                "format": Format(precision=1, scheme=Scheme.fixed),
            }
        columns.append(column_config)

    style_data_conditional = list(base_config["style_data_conditional"])
    style_data_conditional.extend(
        [
            {
                "if": {"column_id": entity_column},
                "backgroundColor": "#f8fafc",
                "fontWeight": "700",
                "color": TABLE_COLORS["text_primary"],
            },
            {
                "if": {"column_id": provider_date_column},
                "backgroundColor": (
                    "rgba(29, 78, 216, 0.08)"
                    if provider_key == "woodmac"
                    else "rgba(245, 158, 11, 0.08)"
                ),
                "fontWeight": "600",
                "color": provider_color,
            },
            {
                "if": {"column_id": provider_capacity_column},
                "backgroundColor": (
                    "rgba(29, 78, 216, 0.08)"
                    if provider_key == "woodmac"
                    else "rgba(245, 158, 11, 0.08)"
                ),
                "fontWeight": "600",
                "color": provider_color,
            },
        ]
    )

    return create_ag_grid_from_datatable(
        id=table_id,
        columns=columns,
        data=display_df.to_dict("records"),
        sort_action="native",
        page_action="none",
        fill_width=True,
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": YEARLY_PROVIDER_DISCREPANCY_TABLE_MAX_HEIGHT,
            "width": "100%",
            "minWidth": "100%",
        },
        style_cell_conditional=_build_yearly_missing_internal_column_styles(provider),
        style_data_conditional=style_data_conditional,
    )


def _numeric_or_blank(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None

    value = float(value)
    if abs(value) < 1e-9:
        return None

    return round(value, 2)


def _combine_distinct_text_values(values: pd.Series | list[object]) -> str:
    unique_values: list[str] = []
    for value in values:
        if pd.isna(value):
            continue

        text_value = " ".join(str(value).strip().split())
        if not text_value or text_value in unique_values:
            continue

        unique_values.append(text_value)

    return "; ".join(unique_values)


def _combine_provider_sources(values: pd.Series | list[object]) -> list[dict]:
    combined: list[dict] = []
    seen: set[tuple[str, str, str, float]] = set()
    for value in values:
        source_rows = value if isinstance(value, list) else []
        for source in source_rows:
            if not isinstance(source, dict):
                continue
            provider_name = str(source.get("provider_name") or "").strip()
            provider_plant_id = str(source.get("provider_plant_id") or "").strip()
            provider_train_id = str(source.get("provider_train_id") or "").strip()
            allocation_share = pd.to_numeric(
                source.get("allocation_share"), errors="coerce"
            )
            if (
                not provider_name
                or not provider_plant_id
                or not provider_train_id
                or pd.isna(allocation_share)
            ):
                continue
            key = (
                provider_name,
                provider_plant_id,
                provider_train_id,
                round(float(allocation_share), 8),
            )
            if key in seen:
                continue
            seen.add(key)
            combined.append(
                {
                    "provider_name": provider_name,
                    "provider_plant_id": provider_plant_id,
                    "provider_train_id": provider_train_id,
                    "allocation_share": key[3],
                }
            )
    return sorted(
        combined,
        key=lambda source: (
            source["provider_name"],
            source["provider_plant_id"],
            source["provider_train_id"],
            source["allocation_share"],
        ),
    )


def _normalized_capacity_change_identity(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).strip().casefold().split())


def _capacity_change_train_display_value(
    canonical_label: object,
    fallback_value: object,
) -> object:
    label = "" if canonical_label is None else str(canonical_label).strip()
    train_number = _canonical_train_number(label)
    if not label or train_number is None:
        return fallback_value
    if re.fullmatch(r"(?:train\s+)?\d+", label, flags=re.IGNORECASE):
        return train_number
    return label


def _prepare_capacity_change_identity(
    change_df: pd.DataFrame,
    scenario_train_names: dict[str, tuple[str, str, object]],
    terminal_key_by_train: dict[str, str],
) -> pd.DataFrame:
    """Align display labels while retaining canonical keys as comparison grain."""
    prepared_df = change_df.copy()
    if prepared_df.empty:
        prepared_df["__plant_identity"] = pd.Series(dtype="object")
        prepared_df["__train_identity"] = pd.Series(dtype="object")
        return prepared_df

    for column_name in ["terminal_key", "train_key"]:
        if column_name not in prepared_df.columns:
            prepared_df[column_name] = None

    train_keys = prepared_df["train_key"].map(_normalized_capacity_change_identity)
    terminal_keys = prepared_df["terminal_key"].map(_normalized_capacity_change_identity)
    terminal_keys = pd.Series(
        [
            terminal_key or terminal_key_by_train.get(train_key, "")
            for terminal_key, train_key in zip(terminal_keys, train_keys)
        ],
        index=prepared_df.index,
        dtype="object",
    )

    if "Canonical Train Label" in prepared_df.columns:
        prepared_df["Train"] = [
            _capacity_change_train_display_value(canonical_label, current_value)
            for canonical_label, current_value in zip(
                prepared_df["Canonical Train Label"],
                prepared_df["Train"],
            )
        ]

    for row_index, train_key in train_keys.items():
        scenario_identity = scenario_train_names.get(train_key)
        if scenario_identity is None:
            continue
        country_name, plant_name, train_number = scenario_identity
        if country_name:
            prepared_df.at[row_index, "Country"] = country_name
        if plant_name:
            prepared_df.at[row_index, "Plant"] = plant_name
        if train_number is not None:
            prepared_df.at[row_index, "Train"] = train_number

    country_identity = prepared_df["Country"].map(_normalized_capacity_change_identity)
    plant_identity = prepared_df["Plant"].map(_normalized_capacity_change_identity)
    train_identity = prepared_df["Train"].map(_normalized_capacity_change_identity)
    prepared_df["__plant_identity"] = [
        f"CANONICAL|{terminal_key}"
        if terminal_key
        else f"DISPLAY|{country}|{plant}"
        for terminal_key, country, plant in zip(
            terminal_keys,
            country_identity,
            plant_identity,
        )
    ]
    prepared_df["__train_identity"] = [
        f"CANONICAL|{train_key}"
        if train_key
        else f"DISPLAY|{country}|{plant}|{train}"
        for train_key, country, plant, train in zip(
            train_keys,
            country_identity,
            plant_identity,
            train_identity,
        )
    ]
    return prepared_df


def _align_capacity_change_comparison_identity(
    woodmac_change_df: pd.DataFrame,
    ea_change_df: pd.DataFrame,
    internal_change_df: pd.DataFrame,
    woodmac_detail_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_train_names: dict[str, tuple[str, str, object]] = {}
    terminal_key_by_train: dict[str, str] = {}
    for change_df in (
        internal_change_df,
        woodmac_change_df,
        ea_change_df,
        woodmac_detail_df,
    ):
        if change_df.empty or not {"terminal_key", "train_key"}.issubset(change_df.columns):
            continue
        for terminal_key, train_key in zip(
            change_df["terminal_key"],
            change_df["train_key"],
        ):
            clean_train_key = _normalized_capacity_change_identity(train_key)
            clean_terminal_key = _normalized_capacity_change_identity(terminal_key)
            if clean_train_key and clean_terminal_key:
                terminal_key_by_train.setdefault(clean_train_key, clean_terminal_key)

    if not internal_change_df.empty and "train_key" in internal_change_df.columns:
        scenario_rows = internal_change_df.copy()
        scenario_rows["__train_key"] = scenario_rows["train_key"].map(
            _normalized_capacity_change_identity
        )
        scenario_rows = scenario_rows[scenario_rows["__train_key"].ne("")]
        for _, row in scenario_rows.iterrows():
            train_key = row["__train_key"]
            if train_key in scenario_train_names:
                continue
            scenario_train_names[train_key] = (
                "" if pd.isna(row.get("Country")) else str(row.get("Country")).strip(),
                "" if pd.isna(row.get("Plant")) else str(row.get("Plant")).strip(),
                _capacity_change_train_display_value(
                    row.get("Canonical Train Label"),
                    row.get("Train"),
                ),
            )

    return tuple(
        _prepare_capacity_change_identity(
            change_df,
            scenario_train_names,
            terminal_key_by_train,
        )
        for change_df in (
            woodmac_change_df,
            ea_change_df,
            internal_change_df,
            woodmac_detail_df,
        )
    )


def _apply_train_change_time_view(
    df: pd.DataFrame,
    time_view: str,
    source_column: str = "Effective Date",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    period_df = df.copy()
    period_dates = pd.to_datetime(period_df[source_column], errors="coerce")
    period_df["__period_start"] = period_dates.dt.to_period("M").dt.to_timestamp()

    if time_view == "quarterly":
        period_df["__period_start"] = period_dates.dt.to_period("Q").dt.start_time
        period_df["Effective Date"] = (
            period_df["__period_start"].dt.year.astype(str)
            + "-Q"
            + period_df["__period_start"].dt.quarter.astype(str)
        )
    elif time_view == "seasonally":
        period_df["__period_start"], period_df["Effective Date"] = _build_lng_season_periods(
            period_dates
        )
    elif time_view == "yearly":
        period_df["__period_start"] = period_dates.dt.to_period("Y").dt.start_time
        period_df["Effective Date"] = period_df["__period_start"].dt.year.astype(str)
    else:
        period_df["Effective Date"] = period_df["__period_start"].dt.strftime("%Y-%m-%d")

    return period_df


def _prepare_woodmac_period_change_df(
    change_df: pd.DataFrame,
    time_view: str,
) -> pd.DataFrame:
    columns = [
        "__plant_identity",
        "__train_identity",
        "__period_start",
        "Effective Date",
        "Country",
        "Plant",
        "Train",
        "Woodmac Adds (MTPA)",
        "Woodmac Reductions (MTPA)",
        "Woodmac Net Delta (MTPA)",
        "Woodmac Activity Abs",
        "Source Field",
        "Source Name",
        "Train Source Name",
        "Mapping Applied",
        "Train Mapping Applied",
    ]
    if change_df.empty:
        return pd.DataFrame(columns=columns)

    woodmac_df = _apply_train_change_time_view(change_df, time_view)
    woodmac_df["Woodmac Adds (MTPA)"] = woodmac_df["Delta MTPA"].where(
        woodmac_df["Delta MTPA"] > 0,
        0.0,
    )
    woodmac_df["Woodmac Reductions (MTPA)"] = woodmac_df["Delta MTPA"].where(
        woodmac_df["Delta MTPA"] < 0,
        0.0,
    )
    woodmac_df["Woodmac Net Delta (MTPA)"] = woodmac_df["Delta MTPA"]
    woodmac_df["Woodmac Activity Abs"] = (
        woodmac_df["Woodmac Adds (MTPA)"].abs()
        + woodmac_df["Woodmac Reductions (MTPA)"].abs()
    )

    woodmac_df = (
        woodmac_df.groupby(
            [
                "__plant_identity",
                "__train_identity",
                "__period_start",
                "Effective Date",
                "Country",
                "Plant",
                "Train",
            ],
            as_index=False,
            dropna=False,
        )
            .agg(
            {
                "Woodmac Adds (MTPA)": "sum",
                "Woodmac Reductions (MTPA)": "sum",
                "Woodmac Net Delta (MTPA)": "sum",
                "Woodmac Activity Abs": "sum",
                "Source Field": "last",
                "Source Name": _combine_distinct_text_values,
                "Train Source Name": _combine_distinct_text_values,
                "Mapping Applied": "max",
                "Train Mapping Applied": "max",
            }
        )
    )

    woodmac_df["__train_sort"] = woodmac_df["Train"].map(_canonical_train_number)
    woodmac_df = woodmac_df.sort_values(
        ["__period_start", "Country", "Plant", "__train_sort", "Woodmac Activity Abs"],
        ascending=[True, True, True, True, False],
    ).drop(columns=["__train_sort"], errors="ignore").reset_index(drop=True)

    return woodmac_df[columns]


def _prepare_ea_period_change_df(
    change_df: pd.DataFrame,
    time_view: str,
) -> pd.DataFrame:
    columns = [
        "__plant_identity",
        "__train_identity",
        "__period_start",
        "Effective Date",
        "Country",
        "Plant",
        "Train",
        "project_name",
        "train_name",
        "EA Adds (MTPA)",
        "EA Reductions (MTPA)",
        "EA Net Delta (MTPA)",
        "EA Activity Abs",
        "Source Field",
        "Source Name",
        "Train Source Name",
        "Mapping Applied",
        "Train Mapping Applied",
    ]
    if change_df.empty:
        return pd.DataFrame(columns=columns)

    ea_df = _apply_train_change_time_view(change_df, time_view)
    ea_df["EA Activity Abs"] = (
        ea_df["EA Adds (MTPA)"].abs()
        + ea_df["EA Reductions (MTPA)"].abs()
    )

    ea_df = (
        ea_df.groupby(
            [
                "__plant_identity",
                "__train_identity",
                "__period_start",
                "Effective Date",
                "Country",
                "Plant",
                "Train",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            {
                "project_name": "last",
                "train_name": "last",
                "EA Adds (MTPA)": "sum",
                "EA Reductions (MTPA)": "sum",
                "EA Net Delta (MTPA)": "sum",
                "EA Activity Abs": "sum",
                "Source Field": "last",
                "Source Name": _combine_distinct_text_values,
                "Train Source Name": _combine_distinct_text_values,
                "Mapping Applied": "max",
                "Train Mapping Applied": "max",
            }
        )
    )

    ea_df["__train_sort"] = ea_df["Train"].map(_canonical_train_number)
    ea_df = ea_df.sort_values(
        ["__period_start", "Country", "Plant", "__train_sort", "EA Activity Abs", "project_name", "train_name"],
        ascending=[True, True, True, True, False, True, True],
    ).drop(columns=["__train_sort"], errors="ignore").reset_index(drop=True)

    return ea_df[columns]


def _collect_unresolved_train_keys(*period_dfs: pd.DataFrame) -> set[tuple[str, str]]:
    unresolved_keys: set[tuple[str, str]] = set()
    for period_df in period_dfs:
        if period_df is None or period_df.empty:
            continue
        unresolved_df = period_df[period_df["Train"].isna()].copy()
        if unresolved_df.empty:
            continue
        unresolved_keys.update(
            {
                (
                    str(row["Country"]).strip(),
                    str(row["Plant"]).strip(),
                )
                for _, row in unresolved_df.iterrows()
            }
        )
    return unresolved_keys


def _build_flat_plants_trains_rows(
    plant_comparison_df: pd.DataFrame,
    train_comparison_df: pd.DataFrame,
    unresolved_plant_keys: set[tuple[str, str]],
) -> list[dict]:
    hierarchical_rows = []
    train_group_map: dict[tuple[str, str, str], pd.DataFrame] = {}

    if not train_comparison_df.empty:
        normalized_train_df = train_comparison_df.copy()
        for column_name in ["Effective Date", "Country", "Plant"]:
            normalized_train_df[column_name] = (
                normalized_train_df[column_name].fillna("").astype(str).str.strip()
            )
        normalized_train_df["__train_sort"] = normalized_train_df["Train"].map(
            _canonical_train_number
        )
        normalized_train_df = normalized_train_df.sort_values(
            ["Effective Date", "Country", "Plant", "__train_sort"],
            ascending=[True, True, True, True],
        )
        train_group_map = {
            key: group.drop(columns=["__train_sort"], errors="ignore").reset_index(drop=True)
            for key, group in normalized_train_df.groupby(
                ["Effective Date", "Country", "Plant"],
                sort=False,
            )
        }

    for _, plant_row in plant_comparison_df.iterrows():
        plant_name = plant_row.get("Plant")
        if pd.isna(plant_name) or str(plant_name).strip() == "":
            continue

        effective_date = (
            ""
            if pd.isna(plant_row.get("Effective Date"))
            else str(plant_row.get("Effective Date")).strip()
        )
        country = (
            ""
            if pd.isna(plant_row.get("Country"))
            else str(plant_row.get("Country")).strip()
        )
        plant_name = str(plant_name).strip()
        plant_key = (effective_date, country, plant_name)
        unresolved_plant_key = (country, plant_name)
        plant_train_df = train_group_map.get(plant_key)
        is_resolved_train_plant = (
            unresolved_plant_key not in unresolved_plant_keys
            and plant_train_df is not None
            and not plant_train_df.empty
        )

        if not is_resolved_train_plant:
            hierarchical_rows.append(
                {
                    "Effective Date": effective_date,
                    "Country": country,
                    "Plant": plant_name,
                    "Train": "",
                    "Woodmac Original Plant": plant_row.get("Woodmac Original Plant"),
                    "Woodmac Original Train": plant_row.get("Woodmac Original Train"),
                    "Woodmac Adds (MTPA)": _numeric_or_blank(plant_row.get("Woodmac Adds (MTPA)")),
                    "Woodmac Reductions (MTPA)": _numeric_or_blank(
                        plant_row.get("Woodmac Reductions (MTPA)")
                    ),
                    "Woodmac Net Delta (MTPA)": _numeric_or_blank(
                        plant_row.get("Woodmac Net Delta (MTPA)")
                    ),
                    "Energy Aspects Original Plant": plant_row.get(
                        "Energy Aspects Original Plant"
                    ),
                    "Energy Aspects Original Train": plant_row.get(
                        "Energy Aspects Original Train"
                    ),
                    "EA Adds (MTPA)": _numeric_or_blank(plant_row.get("EA Adds (MTPA)")),
                    "EA Reductions (MTPA)": _numeric_or_blank(plant_row.get("EA Reductions (MTPA)")),
                    "EA Net Delta (MTPA)": _numeric_or_blank(plant_row.get("EA Net Delta (MTPA)")),
                    INTERNAL_SCENARIO_ADDS_COLUMN: _numeric_or_blank(
                        plant_row.get(INTERNAL_SCENARIO_ADDS_COLUMN)
                    ),
                    INTERNAL_SCENARIO_REDUCTIONS_COLUMN: _numeric_or_blank(
                        plant_row.get(INTERNAL_SCENARIO_REDUCTIONS_COLUMN)
                    ),
                    INTERNAL_SCENARIO_NET_COLUMN: _numeric_or_blank(
                        plant_row.get(INTERNAL_SCENARIO_NET_COLUMN)
                    ),
                    "Type": "plant",
                    "country_group_key": "",
                    "plant_group_key": "",
                    "month_group_key": effective_date,
                    "month_group_end": "",
                }
            )
            continue

        for _, train_row in plant_train_df.iterrows():
            hierarchical_rows.append(
                {
                    "Effective Date": effective_date,
                    "Country": country,
                    "Plant": plant_name,
                    "Train": _format_train_label(train_row.get("Train")),
                    "Woodmac Original Plant": train_row.get("Woodmac Original Plant"),
                    "Woodmac Original Train": train_row.get("Woodmac Original Train"),
                    "Woodmac Adds (MTPA)": _numeric_or_blank(
                        train_row.get("Woodmac Adds (MTPA)")
                    ),
                    "Woodmac Reductions (MTPA)": _numeric_or_blank(
                        train_row.get("Woodmac Reductions (MTPA)")
                    ),
                    "Woodmac Net Delta (MTPA)": _numeric_or_blank(
                        train_row.get("Woodmac Net Delta (MTPA)")
                    ),
                    "Energy Aspects Original Plant": train_row.get(
                        "Energy Aspects Original Plant"
                    ),
                    "Energy Aspects Original Train": train_row.get(
                        "Energy Aspects Original Train"
                    ),
                    "EA Adds (MTPA)": _numeric_or_blank(train_row.get("EA Adds (MTPA)")),
                    "EA Reductions (MTPA)": _numeric_or_blank(
                        train_row.get("EA Reductions (MTPA)")
                    ),
                    "EA Net Delta (MTPA)": _numeric_or_blank(
                        train_row.get("EA Net Delta (MTPA)")
                    ),
                    INTERNAL_SCENARIO_ADDS_COLUMN: _numeric_or_blank(
                        train_row.get(INTERNAL_SCENARIO_ADDS_COLUMN)
                    ),
                    INTERNAL_SCENARIO_REDUCTIONS_COLUMN: _numeric_or_blank(
                        train_row.get(INTERNAL_SCENARIO_REDUCTIONS_COLUMN)
                    ),
                    INTERNAL_SCENARIO_NET_COLUMN: _numeric_or_blank(
                        train_row.get(INTERNAL_SCENARIO_NET_COLUMN)
                    ),
                    "Type": "train",
                    "country_group_key": "",
                    "plant_group_key": "",
                    "month_group_key": effective_date,
                    "month_group_end": "",
                }
            )

    return hierarchical_rows


def _build_train_change_hierarchical_rows(
    woodmac_change_df: pd.DataFrame,
    ea_change_df: pd.DataFrame,
    internal_change_df: pd.DataFrame | None = None,
    woodmac_detail_df: pd.DataFrame | None = None,
    time_view: str = "monthly",
    detail_view: str = "country",
) -> pd.DataFrame:
    columns = [
        "Effective Date",
        "Country",
        "Plant",
        "Train",
        "Woodmac Original Plant",
        "Woodmac Original Train",
        "Woodmac Adds (MTPA)",
        "Woodmac Reductions (MTPA)",
        "Woodmac Net Delta (MTPA)",
        "Energy Aspects Original Plant",
        "Energy Aspects Original Train",
        "EA Adds (MTPA)",
        "EA Reductions (MTPA)",
        "EA Net Delta (MTPA)",
        INTERNAL_SCENARIO_ADDS_COLUMN,
        INTERNAL_SCENARIO_REDUCTIONS_COLUMN,
        INTERNAL_SCENARIO_NET_COLUMN,
        "Type",
        "country_group_key",
        "plant_group_key",
        "month_group_key",
        "month_group_end",
    ]
    internal_change_df = pd.DataFrame() if internal_change_df is None else internal_change_df.copy()
    if woodmac_change_df.empty and ea_change_df.empty and internal_change_df.empty:
        return pd.DataFrame(columns=columns)

    woodmac_detail_df = woodmac_change_df.copy() if woodmac_detail_df is None else woodmac_detail_df.copy()

    (
        woodmac_change_df,
        ea_change_df,
        internal_change_df,
        woodmac_detail_df,
    ) = _align_capacity_change_comparison_identity(
        woodmac_change_df,
        ea_change_df,
        internal_change_df,
        woodmac_detail_df,
    )

    woodmac_period_df = _prepare_woodmac_period_change_df(woodmac_change_df, time_view)
    woodmac_detail_period_df = _prepare_woodmac_period_change_df(woodmac_detail_df, time_view)
    ea_period_df = _prepare_ea_period_change_df(ea_change_df, time_view)
    internal_period_df = _prepare_internal_period_change_df(internal_change_df, time_view)
    unresolved_plant_keys = _collect_unresolved_train_keys(
        woodmac_period_df,
        ea_period_df,
        internal_period_df,
    )

    woodmac_total_df = pd.DataFrame(columns=["__period_start", "Effective Date"])
    if not woodmac_period_df.empty:
        woodmac_total_df = (
            woodmac_period_df.groupby(["__period_start", "Effective Date"], as_index=False)
            .agg(
                {
                    "Woodmac Adds (MTPA)": "sum",
                    "Woodmac Reductions (MTPA)": "sum",
                    "Woodmac Net Delta (MTPA)": "sum",
                }
            )
        )

    ea_total_df = pd.DataFrame(columns=["__period_start", "Effective Date"])
    if not ea_period_df.empty:
        ea_total_df = (
            ea_period_df.groupby(["__period_start", "Effective Date"], as_index=False)
            .agg(
                {
                    "EA Adds (MTPA)": "sum",
                    "EA Reductions (MTPA)": "sum",
                    "EA Net Delta (MTPA)": "sum",
                }
            )
        )

    internal_total_df = pd.DataFrame(columns=["__period_start", "Effective Date"])
    if not internal_period_df.empty:
        internal_total_df = (
            internal_period_df.groupby(["__period_start", "Effective Date"], as_index=False)
            .agg(
                {
                    INTERNAL_SCENARIO_ADDS_COLUMN: "sum",
                    INTERNAL_SCENARIO_REDUCTIONS_COLUMN: "sum",
                    INTERNAL_SCENARIO_NET_COLUMN: "sum",
                }
            )
        )

    total_comparison_df = pd.merge(
        woodmac_total_df,
        ea_total_df,
        on=["__period_start", "Effective Date"],
        how="outer",
    )
    total_comparison_df = pd.merge(
        total_comparison_df,
        internal_total_df,
        on=["__period_start", "Effective Date"],
        how="outer",
    ).sort_values(["__period_start"]).reset_index(drop=True)

    woodmac_country_df = pd.DataFrame(columns=["__period_start", "Effective Date", "Country"])
    if not woodmac_period_df.empty:
        woodmac_country_df = (
            woodmac_period_df.groupby(["__period_start", "Effective Date", "Country"], as_index=False)
            .agg(
                {
                    "Woodmac Adds (MTPA)": "sum",
                    "Woodmac Reductions (MTPA)": "sum",
                    "Woodmac Net Delta (MTPA)": "sum",
                }
            )
        )

    ea_country_df = pd.DataFrame(columns=["__period_start", "Effective Date", "Country"])
    if not ea_period_df.empty:
        ea_country_df = (
            ea_period_df.groupby(["__period_start", "Effective Date", "Country"], as_index=False)
            .agg(
                {
                    "EA Adds (MTPA)": "sum",
                    "EA Reductions (MTPA)": "sum",
                    "EA Net Delta (MTPA)": "sum",
                }
            )
        )

    internal_country_df = pd.DataFrame(columns=["__period_start", "Effective Date", "Country"])
    if not internal_period_df.empty:
        internal_country_df = (
            internal_period_df.groupby(["__period_start", "Effective Date", "Country"], as_index=False)
            .agg(
                {
                    INTERNAL_SCENARIO_ADDS_COLUMN: "sum",
                    INTERNAL_SCENARIO_REDUCTIONS_COLUMN: "sum",
                    INTERNAL_SCENARIO_NET_COLUMN: "sum",
                }
            )
        )

    country_comparison_df = pd.merge(
        woodmac_country_df,
        ea_country_df,
        on=["__period_start", "Effective Date", "Country"],
        how="outer",
    )
    country_comparison_df = pd.merge(
        country_comparison_df,
        internal_country_df,
        on=["__period_start", "Effective Date", "Country"],
        how="outer",
    ).sort_values(["__period_start", "Country"]).reset_index(drop=True)

    plant_keys = [
        "__plant_identity",
        "__period_start",
        "Effective Date",
        "Country",
        "Plant",
    ]
    woodmac_plant_df = pd.DataFrame(columns=plant_keys)
    if not woodmac_period_df.empty:
        woodmac_plant_df = (
            woodmac_period_df.groupby(plant_keys, as_index=False)
            .agg(
                {
                    "Woodmac Adds (MTPA)": "sum",
                    "Woodmac Reductions (MTPA)": "sum",
                    "Woodmac Net Delta (MTPA)": "sum",
                    "Woodmac Activity Abs": "sum",
                    "Source Name": _combine_distinct_text_values,
                    "Train Source Name": _combine_distinct_text_values,
                }
            )
            .rename(
                columns={
                    "Source Name": "Woodmac Original Plant",
                    "Train Source Name": "Woodmac Original Train",
                }
            )
        )

    ea_plant_df = pd.DataFrame(columns=plant_keys)
    if not ea_period_df.empty:
        ea_plant_df = (
            ea_period_df.groupby(plant_keys, as_index=False)
            .agg(
                {
                    "EA Adds (MTPA)": "sum",
                    "EA Reductions (MTPA)": "sum",
                    "EA Net Delta (MTPA)": "sum",
                    "EA Activity Abs": "sum",
                    "Source Name": _combine_distinct_text_values,
                    "Train Source Name": _combine_distinct_text_values,
                }
            )
            .rename(
                columns={
                    "Source Name": "Energy Aspects Original Plant",
                    "Train Source Name": "Energy Aspects Original Train",
                }
            )
        )

    internal_plant_df = pd.DataFrame(columns=plant_keys)
    if not internal_period_df.empty:
        internal_plant_df = (
            internal_period_df.groupby(plant_keys, as_index=False)
            .agg(
                {
                    INTERNAL_SCENARIO_ADDS_COLUMN: "sum",
                    INTERNAL_SCENARIO_REDUCTIONS_COLUMN: "sum",
                    INTERNAL_SCENARIO_NET_COLUMN: "sum",
                    "Internal Scenario Activity Abs": "sum",
                }
            )
        )

    plant_comparison_df = pd.merge(
        woodmac_plant_df,
        ea_plant_df,
        on=plant_keys,
        how="outer",
    )
    plant_comparison_df = pd.merge(
        plant_comparison_df,
        internal_plant_df,
        on=plant_keys,
        how="outer",
    )
    if not plant_comparison_df.empty:
        plant_comparison_df["__sort_abs"] = (
            plant_comparison_df.get("Woodmac Activity Abs", pd.Series(dtype=float)).fillna(0.0)
            + plant_comparison_df.get("EA Activity Abs", pd.Series(dtype=float)).fillna(0.0)
            + plant_comparison_df.get("Internal Scenario Activity Abs", pd.Series(dtype=float)).fillna(0.0)
        )
        plant_comparison_df = plant_comparison_df.sort_values(
            ["__period_start", "Country", "__sort_abs", "Plant"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)

    woodmac_train_df = woodmac_detail_period_df[
        woodmac_detail_period_df["Train"].notna()
    ].copy()
    ea_train_df = ea_period_df[ea_period_df["Train"].notna()].copy()
    internal_train_df = internal_period_df[internal_period_df["Train"].notna()].copy()
    train_keys = [
        "__plant_identity",
        "__train_identity",
        "__period_start",
        "Effective Date",
        "Country",
        "Plant",
        "Train",
    ]
    train_comparison_df = pd.merge(
        woodmac_train_df[
            train_keys
            + [
                "Source Name",
                "Train Source Name",
                "Woodmac Adds (MTPA)",
                "Woodmac Reductions (MTPA)",
                "Woodmac Net Delta (MTPA)",
                "Woodmac Activity Abs",
            ]
        ],
        ea_train_df[
            train_keys
            + [
                "Source Name",
                "Train Source Name",
                "EA Adds (MTPA)",
                "EA Reductions (MTPA)",
                "EA Net Delta (MTPA)",
                "EA Activity Abs",
            ]
        ],
        on=train_keys,
        how="outer",
    )
    train_comparison_df = pd.merge(
        train_comparison_df,
        internal_train_df[
            train_keys
            + [
                INTERNAL_SCENARIO_ADDS_COLUMN,
                INTERNAL_SCENARIO_REDUCTIONS_COLUMN,
                INTERNAL_SCENARIO_NET_COLUMN,
                "Internal Scenario Activity Abs",
            ]
        ],
        on=train_keys,
        how="outer",
    )
    train_comparison_df = train_comparison_df.rename(
        columns={
            "Source Name_x": "Woodmac Original Plant",
            "Train Source Name_x": "Woodmac Original Train",
            "Source Name_y": "Energy Aspects Original Plant",
            "Train Source Name_y": "Energy Aspects Original Train",
        }
    )
    if not train_comparison_df.empty:
        train_comparison_df["__train_sort"] = train_comparison_df["Train"].map(
            _canonical_train_number
        )
        train_comparison_df = train_comparison_df.sort_values(
            ["__period_start", "Country", "Plant", "__train_sort"],
            ascending=[True, True, True, True],
        ).drop(columns=["__train_sort"], errors="ignore").reset_index(drop=True)

    if detail_view == "total":
        hierarchical_rows = [
            {
                "Effective Date": row["Effective Date"],
                "Country": "",
                "Plant": "",
                "Train": "",
                "Woodmac Adds (MTPA)": _numeric_or_blank(row.get("Woodmac Adds (MTPA)")),
                "Woodmac Reductions (MTPA)": _numeric_or_blank(row.get("Woodmac Reductions (MTPA)")),
                "Woodmac Net Delta (MTPA)": _numeric_or_blank(row.get("Woodmac Net Delta (MTPA)")),
                "EA Adds (MTPA)": _numeric_or_blank(row.get("EA Adds (MTPA)")),
                "EA Reductions (MTPA)": _numeric_or_blank(row.get("EA Reductions (MTPA)")),
                "EA Net Delta (MTPA)": _numeric_or_blank(row.get("EA Net Delta (MTPA)")),
                INTERNAL_SCENARIO_ADDS_COLUMN: _numeric_or_blank(row.get(INTERNAL_SCENARIO_ADDS_COLUMN)),
                INTERNAL_SCENARIO_REDUCTIONS_COLUMN: _numeric_or_blank(row.get(INTERNAL_SCENARIO_REDUCTIONS_COLUMN)),
                INTERNAL_SCENARIO_NET_COLUMN: _numeric_or_blank(row.get(INTERNAL_SCENARIO_NET_COLUMN)),
                "Type": "total",
                "country_group_key": "",
                "plant_group_key": "",
                "month_group_key": row["Effective Date"],
                "month_group_end": "",
            }
            for _, row in total_comparison_df.iterrows()
        ]
    elif detail_view == "country":
        hierarchical_rows = [
            {
                "Effective Date": row["Effective Date"],
                "Country": row.get("Country", ""),
                "Plant": "",
                "Train": "",
                "Woodmac Adds (MTPA)": _numeric_or_blank(row.get("Woodmac Adds (MTPA)")),
                "Woodmac Reductions (MTPA)": _numeric_or_blank(row.get("Woodmac Reductions (MTPA)")),
                "Woodmac Net Delta (MTPA)": _numeric_or_blank(row.get("Woodmac Net Delta (MTPA)")),
                "EA Adds (MTPA)": _numeric_or_blank(row.get("EA Adds (MTPA)")),
                "EA Reductions (MTPA)": _numeric_or_blank(row.get("EA Reductions (MTPA)")),
                "EA Net Delta (MTPA)": _numeric_or_blank(row.get("EA Net Delta (MTPA)")),
                INTERNAL_SCENARIO_ADDS_COLUMN: _numeric_or_blank(row.get(INTERNAL_SCENARIO_ADDS_COLUMN)),
                INTERNAL_SCENARIO_REDUCTIONS_COLUMN: _numeric_or_blank(row.get(INTERNAL_SCENARIO_REDUCTIONS_COLUMN)),
                INTERNAL_SCENARIO_NET_COLUMN: _numeric_or_blank(row.get(INTERNAL_SCENARIO_NET_COLUMN)),
                "Type": "country",
                "country_group_key": "",
                "plant_group_key": "",
                "month_group_key": row["Effective Date"],
                "month_group_end": "",
            }
            for _, row in country_comparison_df.iterrows()
        ]
    elif detail_view == "plants_trains":
        hierarchical_rows = _build_flat_plants_trains_rows(
            plant_comparison_df,
            train_comparison_df,
            unresolved_plant_keys,
        )
    else:
        hierarchical_rows = []
        for _, plant_row in plant_comparison_df.iterrows():
            plant_name = plant_row.get("Plant")
            if pd.isna(plant_name) or str(plant_name).strip() == "":
                continue

            effective_date = plant_row["Effective Date"]
            country = plant_row["Country"]

            hierarchical_rows.append(
                {
                    "Effective Date": effective_date,
                    "Country": country,
                    "Plant": plant_name,
                    "Train": "",
                    "Woodmac Adds (MTPA)": _numeric_or_blank(plant_row.get("Woodmac Adds (MTPA)")),
                    "Woodmac Reductions (MTPA)": _numeric_or_blank(plant_row.get("Woodmac Reductions (MTPA)")),
                    "Woodmac Net Delta (MTPA)": _numeric_or_blank(plant_row.get("Woodmac Net Delta (MTPA)")),
                    "EA Adds (MTPA)": _numeric_or_blank(plant_row.get("EA Adds (MTPA)")),
                    "EA Reductions (MTPA)": _numeric_or_blank(plant_row.get("EA Reductions (MTPA)")),
                    "EA Net Delta (MTPA)": _numeric_or_blank(plant_row.get("EA Net Delta (MTPA)")),
                    INTERNAL_SCENARIO_ADDS_COLUMN: _numeric_or_blank(plant_row.get(INTERNAL_SCENARIO_ADDS_COLUMN)),
                    INTERNAL_SCENARIO_REDUCTIONS_COLUMN: _numeric_or_blank(plant_row.get(INTERNAL_SCENARIO_REDUCTIONS_COLUMN)),
                    INTERNAL_SCENARIO_NET_COLUMN: _numeric_or_blank(plant_row.get(INTERNAL_SCENARIO_NET_COLUMN)),
                    "Type": "plant",
                    "country_group_key": "",
                    "plant_group_key": "",
                    "month_group_key": effective_date,
                    "month_group_end": "",
                }
            )

    hierarchical_df = pd.DataFrame(hierarchical_rows, columns=columns)
    if hierarchical_df.empty:
        return hierarchical_df

    next_month_group_key = hierarchical_df["month_group_key"].shift(-1).fillna("")
    hierarchical_df["month_group_end"] = (
        hierarchical_df["month_group_key"].ne("")
        & hierarchical_df["month_group_key"].ne(next_month_group_key)
    ).map({True: "yes", False: ""})

    return hierarchical_df


def _create_train_change_table(
    change_df: pd.DataFrame,
    internal_scenario_label: str | None = None,
) -> dag.AgGrid | html.Div:
    if change_df.empty:
        return html.Div(
            [
                _create_empty_state("No provider capacity changes in the current selection."),
                _build_train_change_footer_notes(),
            ]
        )

    columns = [
        {"name": ["Date", ""], "id": "Effective Date", "type": "text"},
        {"name": ["Country", ""], "id": "Country", "type": "text"},
        {"name": ["Plant", ""], "id": "Plant", "type": "text"},
        {"name": ["Train", ""], "id": "Train", "type": "text"},
        {
            "name": ["Woodmac", "Net Delta"],
            "id": "Woodmac Net Delta (MTPA)",
            "type": "numeric",
            "format": {"specifier": ".2f"},
        },
        {
            "name": ["Energy Aspects", "Net Delta"],
            "id": "EA Net Delta (MTPA)",
            "type": "numeric",
            "format": {"specifier": ".2f"},
        },
    ]

    if internal_scenario_label:
        columns.extend(
            [
                {
                    "name": [internal_scenario_label, "Net Delta"],
                    "id": INTERNAL_SCENARIO_NET_COLUMN,
                    "type": "numeric",
                    "format": {"specifier": ".2f"},
                },
            ]
        )

    columns.extend(
        [
        {"name": ["Meta", "Type"], "id": "Type", "type": "text"},
        {"name": ["Meta", "country_group_key"], "id": "country_group_key", "type": "text"},
        {"name": ["Meta", "plant_group_key"], "id": "plant_group_key", "type": "text"},
        {"name": ["Meta", "month_group_key"], "id": "month_group_key", "type": "text"},
        {"name": ["Meta", "month_group_end"], "id": "month_group_end", "type": "text"},
        ]
    )

    style_data_conditional = [
        {
            "if": {"filter_query": "{Type} = \"total\""},
            "backgroundColor": "#e2e8f0",
            "fontWeight": "700",
            "color": "#0f172a",
            "borderLeft": "4px solid #475569",
        },
        {
            "if": {"filter_query": "{Type} = \"country\""},
            "backgroundColor": "#f0f4f8",
            "fontWeight": "700",
            "color": "#1e3a5f",
            "borderLeft": "4px solid #1e3a5f",
        },
        {
            "if": {"filter_query": "{Type} = \"plant\""},
            "backgroundColor": "#f8fafc",
            "fontWeight": "600",
            "color": "#334155",
            "borderLeft": "3px solid #94a3b8",
        },
        {
            "if": {"filter_query": "{Type} = \"train\""},
            "backgroundColor": "#ffffff",
            "fontWeight": "400",
            "color": "#475569",
        },
        {
            "if": {"filter_query": "{month_group_end} = \"yes\""},
            "borderBottom": "2px solid rgba(148, 163, 184, 0.65)",
        },
        {"if": {"column_id": "Effective Date"}, "textAlign": "left"},
        {"if": {"column_id": "Country"}, "textAlign": "left"},
        {"if": {"column_id": "Plant"}, "textAlign": "left"},
        {"if": {"column_id": "Train"}, "textAlign": "center"},
        {
            "if": {"column_id": "Plant", "filter_query": "{plant_group_key} != \"\""},
            "cursor": "pointer",
        },
        {
            "if": {"column_id": "Woodmac Net Delta (MTPA)", "filter_query": "{Woodmac Net Delta (MTPA)} > 0"},
            "color": "#166534",
            "fontWeight": "700",
        },
        {
            "if": {"column_id": "Woodmac Net Delta (MTPA)", "filter_query": "{Woodmac Net Delta (MTPA)} < 0"},
            "color": "#991b1b",
            "fontWeight": "700",
        },
        {
            "if": {"column_id": "EA Net Delta (MTPA)", "filter_query": "{EA Net Delta (MTPA)} > 0"},
            "color": "#166534",
            "fontWeight": "700",
        },
        {
            "if": {"column_id": "EA Net Delta (MTPA)", "filter_query": "{EA Net Delta (MTPA)} < 0"},
            "color": "#991b1b",
            "fontWeight": "700",
        },
    ]

    if internal_scenario_label:
        style_data_conditional.extend(
            [
                {
                    "if": {
                        "column_id": INTERNAL_SCENARIO_NET_COLUMN,
                        "filter_query": f"{{{INTERNAL_SCENARIO_NET_COLUMN}}} > 0",
                    },
                    "color": "#166534",
                    "fontWeight": "700",
                },
                {
                    "if": {
                        "column_id": INTERNAL_SCENARIO_NET_COLUMN,
                        "filter_query": f"{{{INTERNAL_SCENARIO_NET_COLUMN}}} < 0",
                    },
                    "color": "#991b1b",
                    "fontWeight": "700",
                },
            ]
        )

    table = create_ag_grid_from_datatable(
        id="capacity-page-train-change-table",
        columns=columns,
        data=change_df.to_dict("records"),
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "borderRadius": "4px",
            "border": "1px solid #e2e8f0",
            "maxHeight": "620px",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Effective Date"}, "minWidth": "130px", "maxWidth": "140px"},
            {"if": {"column_id": "Country"}, "minWidth": "140px", "maxWidth": "180px"},
            {"if": {"column_id": "Plant"}, "minWidth": "220px", "maxWidth": "300px"},
            {"if": {"column_id": "Train"}, "minWidth": "90px", "maxWidth": "100px"},
        ],
        style_data_conditional=style_data_conditional,
        hidden_columns=[
            "Type",
            "country_group_key",
            "plant_group_key",
            "month_group_key",
            "month_group_end",
        ],
        sort_action="none",
        page_action="none",
        fill_width=False,
    )

    return html.Div([table, _build_train_change_footer_notes()])


def _filter_capacity_scenario_rows_by_scope(
    scenario_rows_df: pd.DataFrame,
    selected_countries: list[str] | None,
    other_countries_mode: str,
) -> pd.DataFrame:
    scenario_rows_df = _prepare_capacity_scenario_rows_df(scenario_rows_df)
    if scenario_rows_df.empty:
        return scenario_rows_df

    visible_df = scenario_rows_df.copy()
    if selected_countries:
        if other_countries_mode == "exclude":
            visible_df = visible_df[visible_df["country_name"].isin(selected_countries)].copy()
    elif other_countries_mode == "exclude":
        return pd.DataFrame(columns=visible_df.columns)

    return visible_df.reset_index(drop=True)


def _build_internal_scenario_timeline_event_rows(
    scenario_rows_df: pd.DataFrame,
    selected_countries: list[str] | None,
    other_countries_mode: str,
) -> pd.DataFrame:
    columns = [
        "scenario_row_key",
        "Country",
        "Plant",
        "Train",
        "train_key",
        "timeline_identity_key",
        "reference_effective_date",
        "reference_capacity_mtpa",
        "Effective Date",
        "Capacity Change",
        "Scenario Note",
        "timeline_direction",
        "base_provider",
        "base_first_date",
        "base_capacity_mtpa",
        "display_sort_plant",
        "display_sort_train",
    ]
    scenario_rows_df = _filter_capacity_scenario_rows_by_scope(
        scenario_rows_df,
        selected_countries,
        other_countries_mode,
    )
    if scenario_rows_df.empty:
        return pd.DataFrame(columns=columns)

    event_df = scenario_rows_df.copy()
    event_df["Country"] = event_df["country_name"].fillna("").astype(str).str.strip()
    event_df["Plant"] = event_df["plant_name"].fillna("").astype(str).str.strip()
    event_df["Train"] = event_df["train_label"].fillna("").astype(str).str.strip()
    event_df["reference_effective_date"] = pd.to_datetime(
        event_df["scenario_first_date"],
        errors="coerce",
    ).combine_first(
        pd.to_datetime(event_df["base_first_date"], errors="coerce")
    )
    event_df["reference_capacity_mtpa"] = pd.to_numeric(
        event_df["scenario_capacity_mtpa"],
        errors="coerce",
    ).combine_first(
        pd.to_numeric(event_df["base_capacity_mtpa"], errors="coerce")
    ).round(6)
    event_df["Effective Date"] = pd.to_datetime(
        event_df["scenario_first_date"],
        errors="coerce",
    )
    event_df["Capacity Change"] = pd.to_numeric(
        event_df["scenario_capacity_mtpa"],
        errors="coerce",
    ).round(6)
    event_df["Scenario Note"] = event_df["scenario_note"].fillna("").astype(str).str.strip()
    event_df["timeline_direction"] = event_df["reference_capacity_mtpa"].map(
        _normalize_capacity_change_direction
    )
    event_df = event_df[
        event_df["reference_effective_date"].notna()
        & event_df["timeline_direction"].notna()
    ].copy()
    if event_df.empty:
        return pd.DataFrame(columns=columns)

    event_df["display_sort_plant"] = event_df["display_sort_plant"].where(
        event_df["display_sort_plant"].notna() & event_df["display_sort_plant"].ne(""),
        event_df["Plant"],
    )
    event_df["display_sort_train"] = pd.to_numeric(
        event_df["display_sort_train"],
        errors="coerce",
    ).combine_first(pd.to_numeric(event_df["Train"], errors="coerce"))
    event_df = _ensure_train_timeline_identity_key(event_df)
    return event_df[columns]


def _build_internal_scenario_lookup_snapshot(
    scenario_rows_df: pd.DataFrame,
    selected_countries: list[str] | None,
    other_countries_mode: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    columns = [
        "scenario_row_key",
        "Country",
        "Plant",
        "Train",
        "train_key",
        "timeline_identity_key",
        "timeline_direction",
        "Scenario First Date",
        "Scenario Capacity",
        "Scenario Note",
        "__scenario_overridden",
        "base_provider",
        "base_first_date",
        "base_capacity_mtpa",
        "lookup_bucket",
        "lookup_is_out_of_range",
        "display_sort_plant",
        "display_sort_train",
        "display_sort_effective_date",
        "display_sort_direction",
    ]
    event_df = _build_internal_scenario_timeline_event_rows(
        scenario_rows_df,
        selected_countries,
        other_countries_mode,
    )
    if event_df.empty:
        return pd.DataFrame(columns=columns)

    start_month = _normalize_month_date(start_date)
    end_month = _normalize_month_date(end_date)
    if start_month is None:
        start_month = event_df["reference_effective_date"].min()
    if end_month is None:
        end_month = event_df["reference_effective_date"].max()

    identity_columns = ["timeline_identity_key"]
    in_range_df = event_df[
        (event_df["reference_effective_date"] >= start_month)
        & (event_df["reference_effective_date"] <= end_month)
    ].copy()
    if not in_range_df.empty:
        in_range_df = in_range_df.sort_values(
            identity_columns + ["reference_effective_date"],
            ascending=[True] * len(identity_columns) + [True],
            na_position="last",
        ).drop_duplicates(identity_columns, keep="first")
        in_range_df["lookup_bucket"] = "in_range"
        in_range_df["lookup_is_out_of_range"] = False
    else:
        in_range_df = pd.DataFrame(columns=identity_columns + ["lookup_bucket", "lookup_is_out_of_range"])

    outside_df = _select_train_timeline_out_of_range_candidates(
        event_df,
        identity_columns,
        start_month,
        end_month,
        effective_date_column="reference_effective_date",
    )
    if not outside_df.empty:
        outside_df["lookup_is_out_of_range"] = True
        outside_df = outside_df[
            ~outside_df.set_index(identity_columns).index.isin(
                in_range_df.set_index(identity_columns).index
            )
        ].copy()

    lookup_df = pd.concat([in_range_df, outside_df], ignore_index=True, sort=False)
    if lookup_df.empty:
        return pd.DataFrame(columns=columns)

    lookup_df["Scenario First Date"] = pd.to_datetime(
        lookup_df["Effective Date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    lookup_df["Scenario First Date"] = lookup_df["Scenario First Date"].where(
        pd.to_datetime(lookup_df["Effective Date"], errors="coerce").notna(),
        None,
    )
    lookup_df["Scenario Capacity"] = pd.to_numeric(
        lookup_df["Capacity Change"],
        errors="coerce",
    ).round(2)
    lookup_df["Scenario Capacity"] = lookup_df["Scenario Capacity"].where(
        pd.to_numeric(lookup_df["Capacity Change"], errors="coerce").notna(),
        None,
    )
    lookup_df["__scenario_overridden"] = (
        pd.to_datetime(lookup_df["base_first_date"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
        != pd.to_datetime(lookup_df["Effective Date"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    ) | (
        pd.to_numeric(lookup_df["base_capacity_mtpa"], errors="coerce").fillna(-999999.0).round(6)
        != pd.to_numeric(lookup_df["Capacity Change"], errors="coerce").fillna(-999999.0).round(6)
    )
    lookup_df["display_sort_effective_date"] = pd.to_datetime(
        lookup_df["reference_effective_date"],
        errors="coerce",
    )
    lookup_df["display_sort_direction"] = lookup_df["timeline_direction"].map(
        {"reduction": 0, "addition": 1}
    ).fillna(2)
    return lookup_df[columns]


def _filter_visible_capacity_scenario_rows(
    scenario_rows_df: pd.DataFrame,
    selected_countries: list[str] | None,
    other_countries_mode: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    visible_df = _filter_capacity_scenario_rows_by_scope(
        scenario_rows_df,
        selected_countries,
        other_countries_mode,
    )
    if visible_df.empty:
        return visible_df

    visibility_date = visible_df["scenario_first_date"].fillna(visible_df["base_first_date"])
    visibility_mask = pd.Series(True, index=visible_df.index)
    start_month = _normalize_month_date(start_date)
    end_month = _normalize_month_date(end_date)
    if start_month is not None:
        visibility_mask &= visibility_date >= start_month
    if end_month is not None:
        visibility_mask &= visibility_date <= end_month
    visible_df = visible_df[visibility_mask].copy()

    return visible_df.reset_index(drop=True)


def _get_train_timeline_columns(
    show_original_names: bool,
    include_scenario: bool = False,
) -> list[dict]:
    columns = [
        {"name": ["", "Country"], "id": "Country"},
        {"name": ["", "Plant"], "id": "Plant"},
        {"name": ["", "Train"], "id": "Train"},
    ]

    if show_original_names:
        columns.extend(
            [
                {"name": ["Woodmac", "Original Name"], "id": "Woodmac Original Name"},
            ]
        )

    columns.extend(
        [
            {"name": ["Woodmac", "FID Date"], "id": "Woodmac FID Date"},
            {"name": ["Woodmac", "First Date"], "id": "Woodmac First Date"},
            {
                "name": ["Woodmac", "Capacity Change"],
                "id": "Woodmac Capacity Change",
                "type": "numeric",
                "format": Format(precision=2, scheme=Scheme.fixed),
            },
        ]
    )

    if show_original_names:
        columns.extend(
            [
                {
                    "name": ["Energy Aspects", "Original Plant"],
                    "id": "Energy Aspects Original Plant",
                },
                {
                    "name": ["Energy Aspects", "Original Train"],
                    "id": "Energy Aspects Original Train",
                },
            ]
        )

    columns.extend(
        [
            {
                "name": ["Energy Aspects", "First Date"],
                "id": "Energy Aspects First Date",
            },
            {
                "name": ["Energy Aspects", "Capacity Change"],
                "id": "Energy Aspects Capacity Change",
                "type": "numeric",
                "format": Format(precision=2, scheme=Scheme.fixed),
            },
        ]
    )

    if include_scenario:
        columns.extend(
            [
                {
                    "name": ["Internal Scenario", "First Date"],
                    "id": "Scenario First Date",
                },
                {
                    "name": ["Internal Scenario", "Capacity"],
                    "id": "Scenario Capacity",
                    "type": "numeric",
                    "format": Format(precision=2, scheme=Scheme.fixed),
                },
                {
                    "name": ["Internal Scenario", "Note"],
                    "id": "Scenario Note",
                },
            ]
        )

    return columns


def _month_difference(start_value, end_value) -> int | None:
    start_ts = pd.to_datetime(start_value, errors="coerce")
    end_ts = pd.to_datetime(end_value, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return None
    return int((end_ts.year - start_ts.year) * 12 + (end_ts.month - start_ts.month))


def _build_scenario_timeline_reference_key(
    row: pd.Series,
    aggregate_from_date: str | None,
) -> str | None:
    base_provider = str(row.get("base_provider") or "").strip().casefold()
    if base_provider not in {"woodmac", "energy_aspects"}:
        return None

    base_first_date = pd.to_datetime(row.get("base_first_date"), errors="coerce")
    base_capacity = pd.to_numeric([row.get("base_capacity_mtpa")], errors="coerce")[0]
    if pd.isna(base_first_date) or pd.isna(base_capacity) or round(float(base_capacity), 6) == 0:
        return None

    return _build_train_timeline_reference_key(
        row.get("country_name"),
        row.get("plant_name"),
        row.get("train_label"),
        base_first_date,
        base_capacity,
        aggregate_from_date,
    )


def _is_blank_timeline_value(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _blank_timeline_value_mask(series: pd.Series) -> pd.Series:
    return series.map(_is_blank_timeline_value)


def _timeline_bool_mask(values: pd.Series | bool, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.reindex(index)
    else:
        series = pd.Series(values, index=index)
    return series.astype("boolean").fillna(False).astype(bool)


def _apply_train_timeline_provider_backfill(
    grid_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    provider_prefix: str,
    flag_column: str,
    original_name_column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    if grid_df.empty:
        return grid_df

    grid_df[flag_column] = _timeline_bool_mask(grid_df.get(flag_column, False), grid_df.index)
    if lookup_df is None or lookup_df.empty:
        return grid_df

    grid_df = _ensure_train_timeline_identity_key(grid_df)
    lookup_df = _ensure_train_timeline_identity_key(lookup_df)

    if original_name_column_map is None:
        original_name_column_map = {
            "Original Plant": f"{provider_prefix} Original Plant",
            "Original Train": f"{provider_prefix} Original Train",
        }

    identity_columns = ["timeline_identity_key"]
    provider_lookup_df = lookup_df[
        identity_columns
        + [
            "First Date",
            "Capacity Change",
            *original_name_column_map.keys(),
            "lookup_is_out_of_range",
        ]
    ].drop_duplicates(identity_columns, keep="first").rename(
        columns=(
            {
                "First Date": f"__lookup_{provider_prefix}_first_date",
                "Capacity Change": f"__lookup_{provider_prefix}_capacity_change",
                "lookup_is_out_of_range": f"__lookup_{provider_prefix}_out_of_range",
            }
            | {
                source_column: f"__lookup_{provider_prefix}_{target_column.casefold().replace(' ', '_')}"
                for source_column, target_column in original_name_column_map.items()
            }
        )
    )
    merged_df = pd.merge(
        grid_df,
        provider_lookup_df,
        on=identity_columns,
        how="left",
    )

    fill_masks = []
    column_map = {
        f"{provider_prefix} First Date": f"__lookup_{provider_prefix}_first_date",
        f"{provider_prefix} Capacity Change": f"__lookup_{provider_prefix}_capacity_change",
    }
    column_map.update(
        {
            target_column: f"__lookup_{provider_prefix}_{target_column.casefold().replace(' ', '_')}"
            for target_column in original_name_column_map.values()
        }
    )
    for target_column, lookup_column in column_map.items():
        if target_column not in merged_df.columns or lookup_column not in merged_df.columns:
            continue
        blank_mask = _blank_timeline_value_mask(merged_df[target_column])
        lookup_present_mask = ~_blank_timeline_value_mask(merged_df[lookup_column])
        fill_mask = blank_mask & lookup_present_mask
        merged_df[target_column] = merged_df[target_column].where(~fill_mask, merged_df[lookup_column])
        fill_masks.append(fill_mask)

    if fill_masks:
        any_fill_mask = fill_masks[0].copy()
        for mask in fill_masks[1:]:
            any_fill_mask |= mask
        merged_df[flag_column] = merged_df[flag_column] | (
            any_fill_mask
            & _timeline_bool_mask(
                merged_df[f"__lookup_{provider_prefix}_out_of_range"],
                merged_df.index,
            )
        )

    return merged_df.drop(
        columns=[
            f"__lookup_{provider_prefix}_first_date",
            f"__lookup_{provider_prefix}_capacity_change",
            f"__lookup_{provider_prefix}_original_plant",
            f"__lookup_{provider_prefix}_original_train",
            f"__lookup_{provider_prefix}_out_of_range",
        ],
        errors="ignore",
    )


def _apply_train_timeline_scenario_backfill(
    grid_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    if grid_df.empty:
        return grid_df

    grid_df["__scenario_out_of_range"] = _timeline_bool_mask(
        grid_df.get("__scenario_out_of_range", False),
        grid_df.index,
    )
    if lookup_df is None or lookup_df.empty:
        return grid_df

    grid_df = _ensure_train_timeline_identity_key(grid_df)
    lookup_df = _ensure_train_timeline_identity_key(lookup_df)

    identity_columns = ["timeline_identity_key"]
    scenario_lookup_df = lookup_df[
        identity_columns
        + [
            "scenario_row_key",
            "Scenario First Date",
            "Scenario Capacity",
            "Scenario Note",
            "__scenario_overridden",
            "lookup_is_out_of_range",
        ]
    ].drop_duplicates(identity_columns, keep="first").rename(
        columns={
            "scenario_row_key": "__lookup_scenario_row_key",
            "Scenario First Date": "__lookup_scenario_first_date",
            "Scenario Capacity": "__lookup_scenario_capacity",
            "Scenario Note": "__lookup_scenario_note",
            "__scenario_overridden": "__lookup_scenario_overridden",
            "lookup_is_out_of_range": "__lookup_scenario_out_of_range",
        }
    )
    merged_df = pd.merge(
        grid_df,
        scenario_lookup_df,
        on=identity_columns,
        how="left",
    )
    persisted_scenario_row_mask = _timeline_bool_mask(
        merged_df.get("__scenario_row_persisted", False),
        merged_df.index,
    )
    provider_only_row_mask = ~persisted_scenario_row_mask

    scenario_value_blank_mask = (
        _blank_timeline_value_mask(merged_df["Scenario First Date"])
        & _blank_timeline_value_mask(merged_df["Scenario Capacity"])
    )
    fill_masks = []
    column_map = {
        "Scenario First Date": "__lookup_scenario_first_date",
        "Scenario Capacity": "__lookup_scenario_capacity",
        "Scenario Note": "__lookup_scenario_note",
    }
    for target_column, lookup_column in column_map.items():
        blank_mask = _blank_timeline_value_mask(merged_df[target_column])
        lookup_present_mask = ~_blank_timeline_value_mask(merged_df[lookup_column])
        fill_mask = provider_only_row_mask & blank_mask & lookup_present_mask
        merged_df[target_column] = merged_df[target_column].where(~fill_mask, merged_df[lookup_column])
        fill_masks.append(fill_mask)

    scenario_identity_fill_mask = (
        provider_only_row_mask
        & scenario_value_blank_mask
        & ~_blank_timeline_value_mask(merged_df["__lookup_scenario_row_key"])
    )
    merged_df["scenario_row_key"] = merged_df["scenario_row_key"].where(
        ~scenario_identity_fill_mask,
        merged_df["__lookup_scenario_row_key"],
    )

    if fill_masks:
        any_fill_mask = fill_masks[0].copy()
        for mask in fill_masks[1:]:
            any_fill_mask |= mask
        merged_df["__scenario_overridden"] = merged_df["__scenario_overridden"].where(
            ~any_fill_mask,
            _timeline_bool_mask(merged_df["__lookup_scenario_overridden"], merged_df.index),
        )
        merged_df["__scenario_out_of_range"] = merged_df["__scenario_out_of_range"] | (
            any_fill_mask
            & _timeline_bool_mask(merged_df["__lookup_scenario_out_of_range"], merged_df.index)
        )

    return merged_df.drop(
        columns=[
            "__lookup_scenario_row_key",
            "__lookup_scenario_first_date",
            "__lookup_scenario_capacity",
            "__lookup_scenario_note",
            "__lookup_scenario_overridden",
            "__lookup_scenario_out_of_range",
        ],
        errors="ignore",
    )


def _validate_train_timeline_grid_row_ids(grid_df: pd.DataFrame) -> None:
    if grid_df.empty:
        return

    row_ids = grid_df.get(
        "scenario_row_key",
        pd.Series(index=grid_df.index, dtype="object"),
    ).fillna("").astype(str).str.strip()
    blank_count = int(row_ids.eq("").sum())
    duplicate_ids = sorted(row_ids[row_ids.ne("") & row_ids.duplicated(keep=False)].unique())
    if blank_count or duplicate_ids:
        details = []
        if blank_count:
            details.append(f"{blank_count} blank row id(s)")
        if duplicate_ids:
            details.append(f"duplicate row id(s): {duplicate_ids[:5]}")
        raise ValueError(
            "Train Timeline requires one unique scenario_row_key per grid row; "
            + "; ".join(details)
        )


def _build_train_timeline_grid_rows(
    timeline_df: pd.DataFrame,
    scenario_rows_df: pd.DataFrame | None = None,
    aggregate_from_date: str | None = None,
    woodmac_lookup_df: pd.DataFrame | None = None,
    ea_lookup_df: pd.DataFrame | None = None,
    scenario_lookup_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    reference_identity_columns = ["timeline_identity_key"]
    reference_display_columns = ["Country", "Plant", "Train", "train_key"]
    reference_value_columns = [
        "Woodmac Original Name",
        "Woodmac FID Date",
        "Woodmac First Date",
        "Woodmac Capacity Change",
        "Energy Aspects Original Plant",
        "Energy Aspects Original Train",
        "Energy Aspects First Date",
        "Energy Aspects Capacity Change",
        "timeline_reference_key",
        "display_sort_plant",
        "display_sort_train",
        "display_sort_effective_date",
        "display_sort_direction",
    ]
    reference_df = timeline_df.copy()
    scenario_rows_df = _prepare_capacity_scenario_rows_df(scenario_rows_df)

    if reference_df.empty and scenario_rows_df.empty:
        return pd.DataFrame()

    if not reference_df.empty:
        reference_df = _ensure_train_timeline_identity_key(reference_df)
        reference_df["Country"] = reference_df["Country"].fillna("").astype(str).str.strip()
        reference_df["Plant"] = reference_df["Plant"].fillna("").astype(str).str.strip()
        reference_df["Train"] = reference_df["Train"].fillna("").astype(str).str.strip()
        reference_df["timeline_reference_key"] = reference_df.get("timeline_reference_key").fillna("").astype(str)
        reference_df["display_sort_plant"] = reference_df.get("display_sort_plant", reference_df["Plant"])
        existing_direction = reference_df.get(
            "timeline_direction",
            pd.Series(index=reference_df.index, dtype="object"),
        )
        woodmac_capacity = pd.to_numeric(
            reference_df.get(
                "Woodmac Capacity Change",
                pd.Series(index=reference_df.index, dtype="float64"),
            ),
            errors="coerce",
        )
        ea_capacity = pd.to_numeric(
            reference_df.get(
                "Energy Aspects Capacity Change",
                pd.Series(index=reference_df.index, dtype="float64"),
            ),
            errors="coerce",
        )
        reference_df["timeline_direction"] = existing_direction.combine_first(
            woodmac_capacity.map(
                _normalize_capacity_change_direction
            )
        ).combine_first(
            ea_capacity.map(
                _normalize_capacity_change_direction
            )
        )
        reference_df["display_sort_train"] = pd.to_numeric(
            reference_df.get("display_sort_train"),
            errors="coerce",
        )
        reference_df["display_sort_effective_date"] = pd.to_datetime(
            reference_df.get("display_sort_effective_date"),
            errors="coerce",
        )
        reference_df["display_sort_direction"] = pd.to_numeric(
            reference_df.get("display_sort_direction"),
            errors="coerce",
        )
        for column_name in reference_value_columns:
            if column_name not in reference_df.columns:
                reference_df[column_name] = None

    if not scenario_rows_df.empty:
        scenario_display_df = scenario_rows_df.rename(
            columns={
                "country_name": "Country",
                "plant_name": "Plant",
                "train_label": "Train",
            }
        ).copy()
        scenario_direction_source = pd.to_numeric(
            scenario_display_df["scenario_capacity_mtpa"],
            errors="coerce",
        ).combine_first(
            pd.to_numeric(scenario_display_df["base_capacity_mtpa"], errors="coerce")
        )
        scenario_display_df["timeline_direction"] = scenario_direction_source.map(
            _normalize_capacity_change_direction
        )
        scenario_display_df = _ensure_train_timeline_identity_key(scenario_display_df)
        scenario_display_df["timeline_reference_key"] = scenario_rows_df.apply(
            lambda row: _build_scenario_timeline_reference_key(row, aggregate_from_date),
            axis=1,
        )
        scenario_display_df["Scenario First Date"] = scenario_display_df["scenario_first_date"].dt.strftime("%Y-%m-%d")
        scenario_display_df["Scenario First Date"] = scenario_display_df["Scenario First Date"].where(
            scenario_display_df["scenario_first_date"].notna(),
            None,
        )
        scenario_display_df["Scenario Capacity"] = scenario_display_df["scenario_capacity_mtpa"].round(2)
        scenario_display_df["Scenario Note"] = scenario_display_df["scenario_note"].fillna("").astype(str).str.strip()
        scenario_base_provider = (
            scenario_display_df["base_provider"].fillna("").astype(str).str.strip().str.casefold()
        )
        scenario_base_first_date = pd.to_datetime(
            scenario_display_df["base_first_date"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
        scenario_base_capacity = pd.to_numeric(
            scenario_display_df["base_capacity_mtpa"],
            errors="coerce",
        ).round(6)
        scenario_display_df["Woodmac First Date"] = scenario_base_first_date.where(
            scenario_base_provider.eq("woodmac"),
            None,
        )
        scenario_display_df["Woodmac Capacity Change"] = scenario_base_capacity.where(
            scenario_base_provider.eq("woodmac"),
            None,
        )
        scenario_display_df["Energy Aspects First Date"] = scenario_base_first_date.where(
            scenario_base_provider.eq("energy_aspects"),
            None,
        )
        scenario_display_df["Energy Aspects Capacity Change"] = scenario_base_capacity.where(
            scenario_base_provider.eq("energy_aspects"),
            None,
        )
        scenario_display_df["__scenario_overridden"] = (
            scenario_display_df["base_first_date"].fillna(pd.Timestamp("1900-01-01"))
            != scenario_display_df["scenario_first_date"].fillna(pd.Timestamp("1900-01-01"))
        ) | (
            scenario_display_df["base_capacity_mtpa"].fillna(-999999.0).round(6)
            != scenario_display_df["scenario_capacity_mtpa"].fillna(-999999.0).round(6)
        )
        scenario_display_df["display_sort_effective_date"] = pd.to_datetime(
            scenario_display_df["Scenario First Date"],
            errors="coerce",
        ).combine_first(pd.to_datetime(scenario_display_df["base_first_date"], errors="coerce"))
        scenario_display_df["display_sort_direction"] = scenario_display_df["timeline_direction"].map(
            {"reduction": 0, "addition": 1}
        ).fillna(2)
        scenario_display_df = scenario_display_df[
            [
                "scenario_row_key",
                "timeline_reference_key",
                "Country",
                "Plant",
                "Train",
                "train_key",
                "timeline_identity_key",
                "timeline_direction",
                "Scenario First Date",
                "Scenario Capacity",
                "Scenario Note",
                "Woodmac First Date",
                "Woodmac Capacity Change",
                "Energy Aspects First Date",
                "Energy Aspects Capacity Change",
                "__scenario_overridden",
                "display_sort_plant",
                "display_sort_train",
                "display_sort_effective_date",
                "display_sort_direction",
            ]
        ]
    else:
        scenario_display_df = pd.DataFrame(
            columns=[
                "scenario_row_key",
                "timeline_reference_key",
                "Country",
                "Plant",
                "Train",
                "train_key",
                "timeline_identity_key",
                "timeline_direction",
                "Scenario First Date",
                "Scenario Capacity",
                "Scenario Note",
                "Woodmac First Date",
                "Woodmac Capacity Change",
                "Energy Aspects First Date",
                "Energy Aspects Capacity Change",
                "__scenario_overridden",
                "display_sort_plant",
                "display_sort_train",
                "display_sort_effective_date",
                "display_sort_direction",
            ]
        )

    if reference_df.empty:
        grid_df = scenario_display_df.copy()
        grid_df["__scenario_row_persisted"] = True
        for column_name in [
            "Woodmac Original Name",
            "Woodmac FID Date",
            "Woodmac First Date",
            "Woodmac Capacity Change",
            "Energy Aspects Original Plant",
            "Energy Aspects Original Train",
            "Energy Aspects First Date",
            "Energy Aspects Capacity Change",
            "timeline_reference_key",
        ]:
            if column_name not in grid_df.columns:
                grid_df[column_name] = None
    elif scenario_display_df.empty:
        grid_df = reference_df.copy()
        grid_df["__scenario_row_persisted"] = False
        grid_df["Scenario First Date"] = None
        grid_df["Scenario Capacity"] = None
        grid_df["Scenario Note"] = ""
        grid_df["__scenario_overridden"] = False
    else:
        reference_meta_df = reference_df[
            ["scenario_row_key"]
            + reference_identity_columns
            + reference_display_columns
            + reference_value_columns
        ].drop_duplicates(reference_identity_columns).rename(
            columns={
                "scenario_row_key": "reference_row_key",
                "display_sort_plant": "reference_display_sort_plant",
                "display_sort_train": "reference_display_sort_train",
                "display_sort_effective_date": "reference_display_sort_effective_date",
                "display_sort_direction": "reference_display_sort_direction",
            }
        )
        scenario_with_reference_df = pd.merge(
            scenario_display_df,
            reference_meta_df,
            on=reference_identity_columns,
            how="left",
            suffixes=("", "_reference"),
        )
        scenario_with_reference_df["__scenario_row_persisted"] = True
        for column_name in reference_display_columns:
            reference_column = f"{column_name}_reference"
            if reference_column not in scenario_with_reference_df.columns:
                continue
            blank_mask = _blank_timeline_value_mask(scenario_with_reference_df[column_name])
            scenario_with_reference_df[column_name] = scenario_with_reference_df[column_name].where(
                ~blank_mask,
                scenario_with_reference_df[reference_column],
            )
        for column_name in [
            "Woodmac First Date",
            "Woodmac Capacity Change",
            "Energy Aspects First Date",
            "Energy Aspects Capacity Change",
            "timeline_reference_key",
        ]:
            reference_column = f"{column_name}_reference"
            if reference_column not in scenario_with_reference_df.columns:
                continue
            blank_mask = _blank_timeline_value_mask(scenario_with_reference_df[column_name])
            scenario_with_reference_df[column_name] = scenario_with_reference_df[column_name].where(
                ~blank_mask,
                scenario_with_reference_df[reference_column],
            )
        for column_name in [
            "display_sort_plant",
            "display_sort_train",
            "display_sort_effective_date",
            "display_sort_direction",
        ]:
            scenario_with_reference_df[column_name] = scenario_with_reference_df[column_name].where(
                scenario_with_reference_df[column_name].notna(),
                scenario_with_reference_df[f"reference_{column_name}"],
            )
        scenario_with_reference_df["scenario_row_key"] = scenario_with_reference_df["scenario_row_key"].fillna(
            scenario_with_reference_df["reference_row_key"]
        )
        scenario_with_reference_df = scenario_with_reference_df.drop(
            columns=[
                "reference_row_key",
                "reference_display_sort_plant",
                "reference_display_sort_train",
                "reference_display_sort_effective_date",
                "reference_display_sort_direction",
                *[f"{column_name}_reference" for column_name in reference_display_columns],
                "Woodmac First Date_reference",
                "Woodmac Capacity Change_reference",
                "Energy Aspects First Date_reference",
                "Energy Aspects Capacity Change_reference",
                "timeline_reference_key_reference",
            ],
            errors="ignore",
        )

        scenario_identity_df = scenario_display_df[reference_identity_columns].drop_duplicates()
        reference_only_df = pd.merge(
            reference_meta_df,
            scenario_identity_df.assign(__scenario_present=True),
            on=reference_identity_columns,
            how="left",
        )
        reference_only_df = reference_only_df[
            reference_only_df["__scenario_present"].isna()
        ].copy()
        reference_only_df = reference_only_df.drop(
            columns=["__scenario_present"],
            errors="ignore",
        )
        reference_only_df["Scenario First Date"] = pd.Series(
            [None] * len(reference_only_df),
            index=reference_only_df.index,
            dtype="object",
        )
        reference_only_df["Scenario Capacity"] = pd.Series(
            [float("nan")] * len(reference_only_df),
            index=reference_only_df.index,
            dtype="float64",
        )
        reference_only_df["Scenario Note"] = pd.Series(
            [""] * len(reference_only_df),
            index=reference_only_df.index,
            dtype="object",
        )
        reference_only_df["__scenario_overridden"] = pd.Series(
            False,
            index=reference_only_df.index,
            dtype="bool",
        )
        reference_only_df["__scenario_row_persisted"] = False
        reference_only_df = reference_only_df.rename(
            columns={
                "reference_row_key": "scenario_row_key",
                "reference_display_sort_plant": "display_sort_plant",
                "reference_display_sort_train": "display_sort_train",
                "reference_display_sort_effective_date": "display_sort_effective_date",
                "reference_display_sort_direction": "display_sort_direction",
            }
        )

        grid_df = pd.concat(
            [scenario_with_reference_df, reference_only_df],
            ignore_index=True,
            sort=False,
        )
        grid_df["__scenario_overridden"] = _timeline_bool_mask(
            grid_df["__scenario_overridden"],
            grid_df.index,
        )

    grid_df["__woodmac_out_of_range"] = False
    grid_df["__ea_out_of_range"] = False
    grid_df["__scenario_out_of_range"] = False
    grid_df = _apply_train_timeline_provider_backfill(
        grid_df,
        woodmac_lookup_df if woodmac_lookup_df is not None else pd.DataFrame(),
        "Woodmac",
        "__woodmac_out_of_range",
        {"Original Train": "Woodmac Original Name"},
    )
    grid_df = _apply_train_timeline_provider_backfill(
        grid_df,
        ea_lookup_df if ea_lookup_df is not None else pd.DataFrame(),
        "Energy Aspects",
        "__ea_out_of_range",
    )
    grid_df = _apply_train_timeline_scenario_backfill(
        grid_df,
        scenario_lookup_df if scenario_lookup_df is not None else pd.DataFrame(),
    )

    grid_df["__train_blank_sort"] = grid_df["Train"].fillna("").astype(str).eq("").astype(int)
    grid_df["__row_sort_date"] = pd.to_datetime(
        grid_df.get("Scenario First Date"),
        errors="coerce",
    ).combine_first(
        pd.to_datetime(grid_df.get("display_sort_effective_date"), errors="coerce")
    )
    grid_df = grid_df.sort_values(
        [
            "Country",
            "display_sort_plant",
            "__train_blank_sort",
            "display_sort_train",
            "Train",
            "__row_sort_date",
            "display_sort_direction",
        ],
        ascending=[True, True, False, True, True, True, True],
        na_position="last",
    ).drop(
        columns=["__train_blank_sort", "__row_sort_date"],
        errors="ignore",
    ).reset_index(drop=True)

    _validate_train_timeline_grid_row_ids(grid_df)
    return grid_df.where(pd.notna(grid_df), None)


def _get_train_timeline_grid_column_defs(
    show_original_names: bool,
    enable_editing: bool,
) -> list[dict]:
    value_formatter = {
        "function": "params.value != null && params.value !== '' ? d3.format(',.2f')(params.value) : ''"
    }
    out_of_range_cell_style = lambda flag_column: {
        "styleConditions": [
            {
                "condition": f"params.data && params.data.{flag_column} === true",
                "style": {
                    "backgroundColor": "rgba(245, 158, 11, 0.14)",
                    "color": "#92400e",
                    "fontWeight": "600",
                },
            }
        ]
    }

    column_defs = [
        {"field": "Country", "headerName": "Country", "pinned": "left", "minWidth": 140, "editable": False},
        {"field": "Plant", "headerName": "Plant", "pinned": "left", "minWidth": 220, "editable": False},
        {"field": "Train", "headerName": "Train", "pinned": "left", "width": 90, "editable": False},
        {
            "headerName": "Woodmac",
            "children": [
                {
                    "field": "Woodmac FID Date",
                    "headerName": "FID Date",
                    "minWidth": 105,
                    "maxWidth": 125,
                    "editable": False,
                    "cellStyle": out_of_range_cell_style("__woodmac_out_of_range"),
                },
                {
                    "field": "Woodmac First Date",
                    "headerName": "First Date",
                    "minWidth": 105,
                    "maxWidth": 125,
                    "editable": False,
                    "cellStyle": out_of_range_cell_style("__woodmac_out_of_range"),
                },
                {
                    "field": "Woodmac Capacity Change",
                    "headerName": "Capacity Change",
                    "minWidth": 105,
                    "maxWidth": 140,
                    "editable": False,
                    "type": "numericColumn",
                    "valueFormatter": value_formatter,
                    "cellStyle": out_of_range_cell_style("__woodmac_out_of_range"),
                },
            ],
        },
        {
            "headerName": "Energy Aspects",
            "children": [
                {
                    "field": "Energy Aspects First Date",
                    "headerName": "First Date",
                    "minWidth": 105,
                    "maxWidth": 125,
                    "editable": False,
                    "cellStyle": out_of_range_cell_style("__ea_out_of_range"),
                },
                {
                    "field": "Energy Aspects Capacity Change",
                    "headerName": "Capacity Change",
                    "minWidth": 105,
                    "maxWidth": 140,
                    "editable": False,
                    "type": "numericColumn",
                    "valueFormatter": value_formatter,
                    "cellStyle": out_of_range_cell_style("__ea_out_of_range"),
                },
            ],
        },
    ]

    if show_original_names:
        column_defs[3]["children"] = [
            {
                "field": "Woodmac Original Name",
                "headerName": "Original Name",
                "minWidth": 130,
                "maxWidth": 320,
                "editable": False,
            },
            *column_defs[3]["children"],
        ]
        column_defs[4]["children"] = [
            {
                "field": "Energy Aspects Original Plant",
                "headerName": "Original Plant",
                "minWidth": 130,
                "maxWidth": 280,
                "editable": False,
            },
            {
                "field": "Energy Aspects Original Train",
                "headerName": "Original Train",
                "minWidth": 120,
                "maxWidth": 220,
                "editable": False,
            },
            *column_defs[4]["children"],
        ]

    if enable_editing:
        column_defs.append(
            {
                "headerName": "Internal Scenario",
                "children": [
                    {
                        "field": "Scenario First Date",
                        "headerName": "First Date",
                        "minWidth": 105,
                        "maxWidth": 125,
                        "editable": True,
                        "cellDataType": "dateString",
                        "cellEditor": "agDateStringCellEditor",
                        "cellStyle": out_of_range_cell_style("__scenario_out_of_range"),
                    },
                    {
                        "field": "Scenario Capacity",
                        "headerName": "Capacity",
                        "minWidth": 100,
                        "maxWidth": 135,
                        "editable": True,
                        "cellEditor": "agNumberCellEditor",
                        "cellEditorParams": {"precision": 2, "step": 0.1},
                        "type": "numericColumn",
                        "valueFormatter": value_formatter,
                        "cellStyle": out_of_range_cell_style("__scenario_out_of_range"),
                    },
                    {
                        "field": "Scenario Note",
                        "headerName": "Note",
                        "minWidth": 140,
                        "maxWidth": 360,
                        "editable": True,
                        "cellEditor": "agLargeTextCellEditor",
                        "cellEditorPopup": True,
                        "cellEditorParams": {"maxLength": 500, "rows": 6, "cols": 42},
                        "tooltipField": "Scenario Note",
                        "cellStyle": out_of_range_cell_style("__scenario_out_of_range"),
                    },
                ],
            }
        )

    return column_defs


def _create_train_timeline_table(
    table_id: str,
    timeline_df: pd.DataFrame,
    scenario_rows_df: pd.DataFrame | None = None,
    show_original_names: bool = False,
    enable_editing: bool = False,
    aggregate_from_date: str | None = None,
    woodmac_lookup_df: pd.DataFrame | None = None,
    ea_lookup_df: pd.DataFrame | None = None,
    scenario_lookup_df: pd.DataFrame | None = None,
) -> dag.AgGrid | html.Div:
    grid_df = _build_train_timeline_grid_rows(
        timeline_df,
        scenario_rows_df,
        aggregate_from_date=aggregate_from_date,
        woodmac_lookup_df=woodmac_lookup_df,
        ea_lookup_df=ea_lookup_df,
        scenario_lookup_df=scenario_lookup_df,
    )
    empty_state = (
        _create_empty_state("No train timeline rows available for the current selection.")
        if grid_df.empty
        else None
    )

    note = html.Div(
        [
            html.Span(
                "Rows still come from signed in-range activity. Highlighted source cells were backfilled from outside the selected Date Range using the nearest same-sign row for that train, while internal scenario rows remain editable signed event rows."
            )
        ],
        className="balance-metadata-row",
        style={"paddingTop": "12px"},
    )

    grid = dag.AgGrid(
        id=table_id,
        rowData=grid_df.to_dict("records") if not grid_df.empty else [],
        columnDefs=_get_train_timeline_grid_column_defs(show_original_names, enable_editing),
        defaultColDef={
            "sortable": True,
            "resizable": True,
            "filter": True,
            "wrapHeaderText": True,
            "autoHeaderHeight": True,
        },
        dashGridOptions={
            "animateRows": False,
            "rowSelection": {
                "mode": "multiRow",
                "checkboxes": False,
                "headerCheckbox": False,
                "enableClickSelection": True,
                "enableSelectionWithoutKeys": True,
            },
            "undoRedoCellEditing": True,
            "undoRedoCellEditingLimit": 30,
            "stopEditingWhenCellsLoseFocus": True,
            "ensureDomOrder": True,
            "enableCellTextSelection": True,
            "rowHeight": 38,
        },
        getRowId="params.data.scenario_row_key",
        getRowStyle={
            "styleConditions": [
                {
                    "condition": (
                        "params.data && params.data['Woodmac First Date'] && params.data['Scenario First Date'] "
                        "&& params.data['Woodmac First Date'] !== params.data['Scenario First Date']"
                    ),
                    "style": {"boxShadow": "inset 0 0 0 1px #dc2626"},
                },
                {
                    "condition": "params.data.__scenario_overridden === true",
                    "style": {"backgroundColor": "rgba(59, 130, 246, 0.08)"},
                },
                {
                    "condition": "params.data.Train == null || params.data.Train === ''",
                    "style": {"backgroundColor": "#f8fafc"},
                },
            ]
        },
        className="ag-theme-alpine capacity-train-timeline-grid",
        style={"width": "100%", "height": "620px"},
    )

    children = [grid, note]
    if empty_state is not None:
        children.insert(0, empty_state)
    return html.Div(children)


def _autosize_excel_worksheet(worksheet) -> None:
    for column_cells in worksheet.columns:
        max_length = 0
        column_letter = column_cells[0].column_letter
        for cell in column_cells:
            cell_value = "" if cell.value is None else str(cell.value)
            if len(cell_value) > max_length:
                max_length = len(cell_value)
        worksheet.column_dimensions[column_letter].width = min(max_length + 2, 24)


def _apply_train_timeline_excel_mismatch_borders(
    worksheet,
    export_df: pd.DataFrame,
) -> None:
    required_columns = {"Woodmac First Date", "Scenario First Date"}
    if export_df.empty or not required_columns.issubset(export_df.columns):
        return

    woodmac_dates = export_df["Woodmac First Date"].fillna("").astype(str).str.strip()
    scenario_dates = export_df["Scenario First Date"].fillna("").astype(str).str.strip()
    mismatch_rows = export_df.index[
        woodmac_dates.ne("")
        & scenario_dates.ne("")
        & woodmac_dates.ne(scenario_dates)
    ]
    if len(mismatch_rows) == 0:
        return

    visible_column_indexes = [
        export_df.columns.get_loc(column_name) + 1
        for column_name in export_df.columns
        if column_name != TRAIN_TIMELINE_IMPORT_KEY_COLUMN
    ]
    if not visible_column_indexes:
        return

    first_visible_column = min(visible_column_indexes)
    last_visible_column = max(visible_column_indexes)
    red_side = Side(style="thin", color="FFDC2626")

    for row_index in mismatch_rows:
        excel_row_number = int(row_index) + 2
        for column_index in visible_column_indexes:
            cell = worksheet.cell(row=excel_row_number, column=column_index)
            cell.border = Border(
                left=red_side if column_index == first_visible_column else cell.border.left,
                right=red_side if column_index == last_visible_column else cell.border.right,
                top=red_side,
                bottom=red_side,
            )


def _export_matrix_to_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        _autosize_excel_worksheet(writer.sheets[sheet_name])

    output.seek(0)
    return output.getvalue()


def _build_train_timeline_upload_metadata_df(
    export_df: pd.DataFrame,
    selected_scenario_id: int,
    scenario_name: str,
    original_name_visibility: str,
    current_rows_df: pd.DataFrame,
) -> pd.DataFrame:
    metadata_df = export_df.copy()
    prepared_current_rows = _prepare_capacity_scenario_rows_df(current_rows_df)
    current_row_keys = set(
        prepared_current_rows["scenario_row_key"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )
    saved_rows_by_key = {
        str(row.get("scenario_row_key") or "").strip(): json.dumps(
            {
                column_name: _normalize_capacity_scenario_signature_value(
                    row.get(column_name)
                )
                for column_name in _get_capacity_scenario_row_columns()
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in prepared_current_rows.to_dict("records")
        if str(row.get("scenario_row_key") or "").strip()
    }
    metadata_df.insert(
        0,
        TRAIN_TIMELINE_IMPORT_SAVED_ROW_COLUMN,
        metadata_df[TRAIN_TIMELINE_IMPORT_KEY_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .map(saved_rows_by_key)
        .fillna(""),
    )
    metadata_df.insert(
        0,
        TRAIN_TIMELINE_IMPORT_HAS_SCENARIO_ROW_COLUMN,
        metadata_df[TRAIN_TIMELINE_IMPORT_KEY_COLUMN].fillna("").astype(str).str.strip().isin(current_row_keys),
    )
    metadata_df.insert(
        0,
        TRAIN_TIMELINE_IMPORT_EXPORT_ROW_COLUMN,
        range(2, len(metadata_df) + 2),
    )
    metadata_df.insert(0, "__original_name_visibility", str(original_name_visibility or "").strip())
    metadata_df.insert(0, "__scenario_name", str(scenario_name or "").strip())
    metadata_df.insert(0, "__scenario_id", int(selected_scenario_id))
    metadata_df.insert(0, "__template_version", TRAIN_TIMELINE_IMPORT_TEMPLATE_VERSION)
    return metadata_df


def _export_train_timeline_workbook_bytes(
    export_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name=TRAIN_TIMELINE_SHEET_NAME, index=False)
        metadata_df.to_excel(writer, sheet_name=TRAIN_TIMELINE_IMPORT_META_SHEET_NAME, index=False)

        main_sheet = writer.sheets[TRAIN_TIMELINE_SHEET_NAME]
        metadata_sheet = writer.sheets[TRAIN_TIMELINE_IMPORT_META_SHEET_NAME]
        _apply_train_timeline_excel_mismatch_borders(main_sheet, export_df)
        _autosize_excel_worksheet(main_sheet)
        _autosize_excel_worksheet(metadata_sheet)

        if TRAIN_TIMELINE_IMPORT_KEY_COLUMN in export_df.columns:
            hidden_column_index = export_df.columns.get_loc(TRAIN_TIMELINE_IMPORT_KEY_COLUMN) + 1
            hidden_column_letter = main_sheet.cell(row=1, column=hidden_column_index).column_letter
            main_sheet.column_dimensions[hidden_column_letter].hidden = True

        metadata_sheet.sheet_state = "hidden"

    output.seek(0)
    return output.getvalue()


def _normalize_train_timeline_import_text(value: object) -> str:
    return _normalize_timeline_text(value)


def _normalize_train_timeline_import_value(
    column_name: str,
    value: object,
):
    if column_name == "Train":
        return _normalize_train_timeline_import_text(value)
    if column_name in TRAIN_TIMELINE_DATE_COLUMNS:
        normalized_date = _coerce_timeline_row_date(value)
        return None if normalized_date is None else normalized_date.strftime("%Y-%m-%d")
    if column_name in TRAIN_TIMELINE_NUMERIC_COLUMNS:
        numeric_value = _coerce_timeline_row_capacity(value)
        return None if numeric_value is None else round(float(numeric_value), 6)
    if column_name == "Scenario Note":
        return _coerce_timeline_row_note(value)
    return _normalize_train_timeline_import_text(value)


def _coerce_train_timeline_import_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    return False


def _compose_train_timeline_upload_error(errors: list[str]) -> str:
    if not errors:
        return "Train Timeline upload failed."
    visible_errors = errors[:8]
    if len(errors) > 8:
        visible_errors.append(f"{len(errors) - 8} more issue(s) found.")
    return "Train Timeline upload failed: " + " | ".join(visible_errors)


def _decode_saved_train_timeline_scenario_row(
    metadata_row: dict,
    row_key: str,
) -> dict:
    payload = metadata_row.get(TRAIN_TIMELINE_IMPORT_SAVED_ROW_COLUMN)
    try:
        saved_row = json.loads(str(payload or ""))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Upload metadata does not contain a valid saved scenario row. "
            "Export a fresh Train Timeline workbook and try again."
        ) from exc
    if not isinstance(saved_row, dict):
        raise ValueError(
            "Upload metadata does not contain a valid saved scenario row. "
            "Export a fresh Train Timeline workbook and try again."
        )
    saved_key = _normalize_train_timeline_import_text(
        saved_row.get("scenario_row_key")
    )
    if saved_key != row_key or not _normalize_train_timeline_import_text(
        saved_row.get("train_key")
    ):
        raise ValueError(
            "Upload metadata cannot restore the canonical identity for a deleted row. "
            "Export a fresh Train Timeline workbook and try again."
        )
    return {
        column_name: saved_row.get(column_name)
        for column_name in _get_capacity_scenario_row_columns()
    }


def _build_train_timeline_upload_rows_df(
    main_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    selected_scenario_id: int,
    current_rows_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if TRAIN_TIMELINE_IMPORT_KEY_COLUMN not in main_df.columns:
        raise ValueError(
            f"Missing required hidden column '{TRAIN_TIMELINE_IMPORT_KEY_COLUMN}'. Export a fresh Train Timeline workbook and try again."
        )
    required_main_columns = {
        "Country",
        "Plant",
        "Train",
        "Scenario First Date",
        "Scenario Capacity",
        "Scenario Note",
    }
    missing_main_columns = [
        column_name
        for column_name in required_main_columns
        if column_name not in main_df.columns
    ]
    if missing_main_columns:
        raise ValueError(
            "Upload sheet is missing required column(s): "
            + ", ".join(sorted(missing_main_columns))
            + ". Export a fresh Train Timeline workbook and try again."
        )

    required_meta_columns = TRAIN_TIMELINE_IMPORT_META_COLUMNS | set(main_df.columns)
    missing_meta_columns = [
        column_name
        for column_name in required_meta_columns
        if column_name not in metadata_df.columns
    ]
    if missing_meta_columns:
        raise ValueError(
            "Upload metadata is incomplete. Missing column(s): "
            + ", ".join(sorted(missing_meta_columns))
            + ". Export a fresh Train Timeline workbook and try again."
        )

    if metadata_df.empty:
        raise ValueError("Upload metadata is empty. Export a fresh Train Timeline workbook and try again.")

    template_values = {
        _normalize_train_timeline_import_text(value)
        for value in metadata_df["__template_version"].tolist()
        if _normalize_train_timeline_import_text(value)
    }
    if template_values != {TRAIN_TIMELINE_IMPORT_TEMPLATE_VERSION}:
        raise ValueError("Unsupported Train Timeline upload template. Export a fresh workbook from the page and try again.")

    scenario_id_values = {
        int(pd.to_numeric(value, errors="coerce"))
        for value in metadata_df["__scenario_id"].tolist()
        if pd.notna(pd.to_numeric(value, errors="coerce"))
    }
    if scenario_id_values != {int(selected_scenario_id)}:
        raise ValueError("This workbook belongs to a different internal scenario than the one currently selected.")

    metadata_records: dict[str, dict] = {}
    duplicate_metadata_keys: set[str] = set()
    for row in metadata_df.to_dict("records"):
        row_key = _normalize_train_timeline_import_text(row.get(TRAIN_TIMELINE_IMPORT_KEY_COLUMN))
        if not row_key:
            continue
        if row_key in metadata_records:
            duplicate_metadata_keys.add(row_key)
            continue
        metadata_records[row_key] = row
    if duplicate_metadata_keys:
        raise ValueError("Upload metadata contains duplicate row keys. Export a fresh Train Timeline workbook and try again.")

    current_rows_df = _prepare_capacity_scenario_rows_df(current_rows_df)
    current_row_keys = set(
        current_rows_df["scenario_row_key"].fillna("").astype(str).str.strip().tolist()
    )

    allowed_new_row_columns = {"Country", "Plant", "Train"} | TRAIN_TIMELINE_EDITABLE_COLUMNS
    immutable_columns = [
        column_name
        for column_name in main_df.columns
        if column_name not in TRAIN_TIMELINE_EDITABLE_COLUMNS | {TRAIN_TIMELINE_IMPORT_KEY_COLUMN}
    ]

    errors: list[str] = []
    seen_existing_keys: set[str] = set()
    deleted_existing_keys: list[str] = []
    upload_grid_rows: list[dict] = []
    update_count = 0
    addition_count = 0
    restored_saved_rows: list[dict] = []

    normalized_uploaded_keys: set[str] = set()
    prepared_main_records = main_df.to_dict("records")
    for excel_row_number, row in enumerate(prepared_main_records, start=2):
        row_key = _normalize_train_timeline_import_text(row.get(TRAIN_TIMELINE_IMPORT_KEY_COLUMN))
        visible_values = {
            column_name: row.get(column_name)
            for column_name in main_df.columns
            if column_name != TRAIN_TIMELINE_IMPORT_KEY_COLUMN
        }
        if (
            not row_key
            and all(
                _normalize_train_timeline_import_text(value) == ""
                for value in visible_values.values()
            )
        ):
            continue

        if row_key:
            normalized_train = _normalize_train_timeline_import_text(
                row.get("Train")
            )
            if not normalized_train:
                errors.append(
                    f"Row {excel_row_number} has an empty canonical Train label."
                )
                continue
            if row_key not in metadata_records:
                errors.append(
                    f"Row {excel_row_number} has an unknown hidden key. Export a fresh Train Timeline workbook and try again."
                )
                continue
            if row_key in seen_existing_keys:
                errors.append(f"Row {excel_row_number} duplicates an existing exported row key.")
                continue

            seen_existing_keys.add(row_key)
            normalized_uploaded_keys.add(row_key)
            metadata_row = metadata_records[row_key]
            for column_name in immutable_columns:
                try:
                    uploaded_value = _normalize_train_timeline_import_value(column_name, row.get(column_name))
                    original_value = _normalize_train_timeline_import_value(column_name, metadata_row.get(column_name))
                except ValueError as exc:
                    errors.append(f"Row {excel_row_number} has invalid {column_name}: {exc}")
                    break
                if uploaded_value != original_value:
                    errors.append(f"Row {excel_row_number} changed read-only column '{column_name}'.")
                    break
            else:
                scenario_changed = any(
                    _normalize_train_timeline_import_value(column_name, row.get(column_name))
                    != _normalize_train_timeline_import_value(column_name, metadata_row.get(column_name))
                    for column_name in TRAIN_TIMELINE_EDITABLE_COLUMNS
                    if column_name in main_df.columns
                )
                if row_key not in current_row_keys and _coerce_train_timeline_import_bool(
                    metadata_row.get(TRAIN_TIMELINE_IMPORT_HAS_SCENARIO_ROW_COLUMN)
                ):
                    restored_row = _decode_saved_train_timeline_scenario_row(
                        metadata_row,
                        row_key,
                    )
                    restored_row["scenario_first_date"] = _coerce_timeline_row_date(
                        row.get("Scenario First Date")
                    )
                    restored_row["scenario_capacity_mtpa"] = _coerce_timeline_row_capacity(
                        row.get("Scenario Capacity")
                    )
                    restored_row["scenario_note"] = _coerce_timeline_row_note(
                        row.get("Scenario Note")
                    )
                    restored_saved_rows.append(restored_row)
                    addition_count += 1
                    continue
                if scenario_changed:
                    update_count += 1

                prepared_row = {column_name: row.get(column_name) for column_name in main_df.columns}
                prepared_row["Train"] = normalized_train
                prepared_row[TRAIN_TIMELINE_IMPORT_KEY_COLUMN] = row_key
                prepared_row["scenario_row_key"] = row_key
                upload_grid_rows.append(prepared_row)
            continue

        row_errors = []
        try:
            normalized_train = _coerce_positive_train_label(row.get("Train"))
        except ValueError as exc:
            errors.append(
                f"Row {excel_row_number} has invalid Train '{row.get('Train')}'. {exc}"
            )
            continue
        normalized_country = _normalize_train_timeline_import_text(row.get("Country"))
        normalized_plant = _normalize_train_timeline_import_text(row.get("Plant"))
        if not normalized_country:
            row_errors.append("Country is required.")
        if not normalized_plant:
            row_errors.append("Plant is required.")

        scenario_first_date = _coerce_timeline_row_date(row.get("Scenario First Date"))
        if scenario_first_date is None:
            row_errors.append("Scenario First Date must be a valid month.")

        scenario_capacity = _coerce_timeline_row_capacity(row.get("Scenario Capacity"))
        if scenario_capacity is None or round(float(scenario_capacity), 6) == 0:
            row_errors.append("Scenario Capacity must be a non-zero number.")

        disallowed_columns = [
            column_name
            for column_name in TRAIN_TIMELINE_PROVIDER_COLUMNS
            if column_name in main_df.columns
            and _normalize_train_timeline_import_text(row.get(column_name)) != ""
        ]
        if disallowed_columns:
            row_errors.append(
                "new rows must keep provider columns blank: " + ", ".join(disallowed_columns)
            )

        unexpected_columns = [
            column_name
            for column_name in main_df.columns
            if column_name not in allowed_new_row_columns | {TRAIN_TIMELINE_IMPORT_KEY_COLUMN}
            and _normalize_train_timeline_import_text(row.get(column_name)) != ""
        ]
        if unexpected_columns:
            row_errors.append(
                "new rows contain unsupported values in: " + ", ".join(unexpected_columns)
            )

        if row_errors:
            errors.append(f"Row {excel_row_number} is invalid: {' '.join(row_errors)}")
            continue

        new_row_key = _build_capacity_scenario_row_key(
            normalized_country,
            normalized_plant,
            normalized_train,
            effective_date=scenario_first_date,
            capacity_value=scenario_capacity,
            provider="manual",
        )
        if new_row_key in current_row_keys or any(
            existing_row.get("scenario_row_key") == new_row_key
            for existing_row in upload_grid_rows
        ):
            errors.append(f"Row {excel_row_number} already exists in the selected scenario.")
            continue

        upload_grid_rows.append(
            {
                "scenario_row_key": new_row_key,
                TRAIN_TIMELINE_IMPORT_KEY_COLUMN: new_row_key,
                "Country": normalized_country,
                "Plant": normalized_plant,
                "Train": normalized_train,
                "Woodmac Original Name": None,
                "Woodmac First Date": None,
                "Woodmac Capacity Change": None,
                "Energy Aspects Original Plant": None,
                "Energy Aspects Original Train": None,
                "Energy Aspects First Date": None,
                "Energy Aspects Capacity Change": None,
                "Scenario First Date": scenario_first_date.strftime("%Y-%m-%d"),
                "Scenario Capacity": round(float(scenario_capacity), 6),
                "Scenario Note": _coerce_timeline_row_note(row.get("Scenario Note")),
            }
        )
        addition_count += 1

    if errors:
        raise ValueError(_compose_train_timeline_upload_error(errors))

    for row_key, metadata_row in metadata_records.items():
        if row_key in normalized_uploaded_keys:
            continue
        export_row_numeric = pd.to_numeric(
            metadata_row.get(TRAIN_TIMELINE_IMPORT_EXPORT_ROW_COLUMN),
            errors="coerce",
        )
        export_row_number = int(export_row_numeric) if pd.notna(export_row_numeric) else "unknown"
        if not _coerce_train_timeline_import_bool(
            metadata_row.get(TRAIN_TIMELINE_IMPORT_HAS_SCENARIO_ROW_COLUMN)
        ):
            errors.append(
                f"Deleted row {export_row_number} is provider-only and cannot be removed via upload."
            )
            continue
        deleted_existing_keys.append(row_key)

    if errors:
        raise ValueError(_compose_train_timeline_upload_error(errors))

    base_rows_df = current_rows_df[
        ~current_rows_df["scenario_row_key"].fillna("").astype(str).str.strip().isin(deleted_existing_keys)
    ].copy()
    if restored_saved_rows:
        base_rows_df = pd.concat(
            [base_rows_df, pd.DataFrame(restored_saved_rows)],
            ignore_index=True,
            sort=False,
        )
    updated_rows_df = _update_working_scenario_rows_from_grid(base_rows_df, upload_grid_rows)
    summary = {
        "updated": update_count,
        "added": addition_count,
        "deleted": len(deleted_existing_keys),
    }
    return updated_rows_df, summary


layout = html.Div(
    [
        dcc.Store(
            id="capacity-page-initial-load-trigger",
            data={"load": 1},
            storage_type="memory",
        ),
        dcc.Store(id="capacity-page-woodmac-data-store", storage_type="memory"),
        dcc.Store(id="capacity-page-train-capacity-data-store", storage_type="memory"),
        dcc.Store(id="capacity-page-ea-capacity-data-store", storage_type="memory"),
        dcc.Store(id="capacity-page-country-options-store", storage_type="memory"),
        dcc.Store(id="capacity-page-refresh-timestamp-store", storage_type="memory"),
        dcc.Store(id="capacity-page-load-error-store", storage_type="memory"),
        dcc.Store(id="capacity-page-metadata-store", storage_type="memory"),
        dcc.Store(
            id="capacity-page-source-render-revision-store",
            storage_type="memory",
        ),
        dcc.Store(
            id="capacity-page-scenario-render-revision-store",
            storage_type="memory",
        ),
        dcc.Store(id="capacity-page-train-mapping-save-trigger", storage_type="memory"),
        dcc.Store(id="capacity-page-capacity-scenario-options-store", storage_type="memory"),
        dcc.Store(id="capacity-page-capacity-scenario-selected-store", storage_type="memory"),
        dcc.Store(id="capacity-page-capacity-scenario-working-store", storage_type="memory"),
        dcc.Store(id="capacity-page-capacity-scenario-dirty-store", storage_type="memory"),
        dcc.Store(id="capacity-page-capacity-scenario-pending-selection-store", storage_type="memory"),
        dcc.Store(id="capacity-page-capacity-scenario-dropdown-target-store", storage_type="memory"),
        dcc.Store(id="capacity-page-capacity-scenario-refresh-store", storage_type="memory"),
        dcc.Store(id="capacity-page-ramp-forecast-status-store", storage_type="memory"),
        dcc.Store(
            id="capacity-page-scenario-command-running-store",
            data=False,
            storage_type="memory",
        ),
        dcc.Store(id="capacity-page-source-refresh-job-store", storage_type="memory"),
        dcc.Interval(
            id="capacity-page-source-refresh-poll",
            interval=5000,
            n_intervals=0,
            disabled=True,
        ),
        dcc.Download(id="capacity-page-download-woodmac-excel"),
        dcc.Download(id="capacity-page-download-ea-excel"),
        dcc.Download(id="capacity-page-download-internal-scenario-excel"),
        dcc.Download(id="capacity-page-download-train-change-excel"),
        dcc.Download(id="capacity-page-download-train-timeline-excel"),
        dcc.ConfirmDialog(
            id="capacity-page-capacity-scenario-switch-confirm",
            message="You have unsaved scenario edits. Switch scenarios and discard the current working copy?",
        ),
        dcc.ConfirmDialog(
            id="capacity-page-capacity-scenario-delete-confirm",
            message=(
                "This permanently deletes the selected scenario and every saved Capacity snapshot. "
                "Historical ramp runs remain detached for audit. This cannot be reverted."
            ),
        ),
        html.Div(
            [
                        html.Div(
                            [
                                html.Div("Date Range", className="filter-group-header"),
                                html.Div(
                                    [
                                        dcc.DatePickerRange(
                                            id="capacity-page-date-range",
                                            start_date=None,
                                            end_date=None,
                                            min_date_allowed=None,
                                            max_date_allowed=None,
                                            minimum_nights=0,
                                            display_format="YYYY-MM",
                                            month_format="YYYY-MM",
                                            start_date_placeholder_text="Start month",
                                            end_date_placeholder_text="End month",
                                            clearable=False,
                                            number_of_months_shown=2,
                                            persistence=True,
                                            persistence_type="session",
                                        )
                                    ],
                                    className="professional-date-picker",
                                ),
                            ],
                            className="filter-group",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Time View",
                                    className="filter-group-header",
                                    title=SEASONAL_TIME_VIEW_TOOLTIP,
                                    style={"cursor": "help"},
                                ),
                                html.Div(
                                    [
                                        dcc.RadioItems(
                                            id="capacity-page-train-change-time-view",
                                            options=[
                                                {"label": "Monthly", "value": "monthly"},
                                                {"label": "Quarterly", "value": "quarterly"},
                                                {"label": "Seasonally", "value": "seasonally"},
                                                {"label": "Yearly", "value": "yearly"},
                                            ],
                                            value="monthly",
                                            inline=True,
                                            persistence=True,
                                            persistence_type="session",
                                            labelStyle={
                                                "display": "inline-flex",
                                                "alignItems": "center",
                                                "marginRight": "10px",
                                                "fontSize": "12px",
                                                "fontWeight": "600",
                                                "color": "#334155",
                                            },
                                            inputStyle={"marginRight": "6px"},
                                            style={"display": "flex", "alignItems": "center"},
                                        )
                                    ],
                                    style={**TRAIN_CHANGE_CONTROL_SHELL_STYLE, "padding": "3px 8px 3px 12px"},
                                ),
                            ],
                            className="filter-group",
                        ),
                        html.Div(
                            [
                                html.Div("Internal Scenario", className="filter-group-header"),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="capacity-page-internal-scenario-dropdown",
                                            options=[],
                                            value=None,
                                            placeholder="Select internal scenario",
                                            clearable=True,
                                            persistence=True,
                                            persistence_type="session",
                                            className="capacity-scenario-sticky-dropdown",
                                            style={"minWidth": "260px", "width": "100%"},
                                        )
                                    ],
                                    className="capacity-scenario-sticky-shell",
                                    style={"width": "100%"},
                                ),
                            ],
                            className="filter-group",
                        ),
            ],
            className="professional-section-header",
            style={
                "display": "flex",
                "gap": "12px",
                "alignItems": "flex-end",
                "flexWrap": "wrap",
            },
        ),
        html.Div(
            [
                html.Div(
                    id="capacity-page-source-metadata-status",
                    className="balance-metadata-row",
                    style={
                        "fontSize": "11px",
                        "fontWeight": "700",
                        "color": "#475569",
                        "marginBottom": "2px",
                    },
                ),
                html.Div(
                    id="capacity-page-source-refresh-status",
                    className="balance-metadata-row",
                    style={
                        "fontSize": "11px",
                        "fontWeight": "700",
                        "color": "#475569",
                        "marginBottom": "8px",
                    },
                ),
                html.Div(id="capacity-page-load-error-banner"),
                html.Div(
                            [
                                dcc.Loading(
                                    children=_create_yearly_capacity_comparison_section(
                                        "Yearly Capacity Comparison",
                                        (
                                            "December year-end snapshots only. Years shown only where Woodmac and Energy Aspects both have a yearly value within the selected Date Range."
                                        ),
                                        "capacity-page-yearly-capacity-comparison-chart",
                                        "capacity-page-yearly-capacity-comparison-table-container",
                                    ),
                                    type="default",
                                ),
                                dcc.Loading(
                                    children=_create_yearly_provider_discrepancy_section(
                                        "Provider Discrepancies",
                                        (
                                            "Compared against the selected Internal Scenario using the Train Timeline row logic. Capacity tables rank absolute MTPA gaps; timeline tables rank absolute first-date gaps in months."
                                        ),
                                    ),
                                    type="default",
                                ),
                                _create_top_capacity_selector_region(),
                                dcc.Loading(
                                    children=html.Div(
                                        [
                                            _create_source_section(
                                                "WoodMac Nominal LNG Capacity",
                                                None,
                                                (
                                                    f"Monthly country matrix built from {CAPACITY_SOURCE_TABLE} "
                                                    f"with annual-output fallback from {WOODMAC_ANNUAL_OUTPUT_SOURCE_TABLE} "
                                                    f"using last monthly capacity for monthly-backed trains and max annual output only for annual-only trains, with annual-only legacy capacity carried forward from {WOODMAC_ANNUAL_CARRY_FORWARD_START[:4]} onward to avoid source-timing reductions."
                                                ),
                                                "capacity-page-woodmac-summary",
                                                "capacity-page-woodmac-chart",
                                                "capacity-page-woodmac-table-container",
                                                "capacity-page-export-woodmac-button",
                                            ),
                                            _create_source_section(
                                                "Energy Aspects Scheduled LNG Capacity",
                                                None,
                                                (
                                                    f"Monthly country schedule derived from the latest snapshot in {EA_CAPACITY_SOURCE_TABLE} "
                                                    "using project start_date month and carrying active project capacity forward through the selected horizon."
                                                ),
                                                "capacity-page-ea-summary",
                                                "capacity-page-ea-chart",
                                                "capacity-page-ea-table-container",
                                                "capacity-page-export-ea-button",
                                            ),
                                            _create_internal_scenario_section(
                                                "Internal Scenario Capacity",
                                                "capacity-page-internal-scenario-summary",
                                                "capacity-page-internal-scenario-chart",
                                                "capacity-page-internal-scenario-table-container",
                                                "capacity-page-export-internal-scenario-button",
                                            ),
                                        ],
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))",
                                            "gap": "24px",
                                            "alignItems": "start",
                                        },
                                    ),
                                    type="default",
                                ),
                                dcc.Loading(
                                    children=html.Div(
                                        [
                                            _create_train_change_section(
                                                "Capacity Change Comparison in Selected Range",
                                                None,
                                                "capacity-page-train-change-summary",
                                                "capacity-page-train-change-table-container",
                                                "capacity-page-export-train-change-button",
                                            ),
                                            _create_train_timeline_section(
                                                "Train Timeline - Monthly",
                                                None,
                                                "capacity-page-train-timeline-table-container",
                                                "capacity-page-export-train-timeline-button",
                                            ),
                                        ],
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))",
                                            "gap": "24px",
                                            "alignItems": "start",
                                        },
                                    ),
                                    type="default",
                                ),
                                dcc.Loading(
                                    children=html.Div(
                                        [
                                            _create_unmapped_train_mapping_section(
                                                "Unmapped Provider Trains in Current Selection",
                                                "Review unresolved provider sources, assign a canonical business train, and save the complete allocation through the authoritative provider-link helper.",
                                                "capacity-page-unmapped-train-summary",
                                                "capacity-page-unmapped-train-message",
                                                "capacity-page-save-train-mappings-button",
                                                "capacity-page-unmapped-train-table",
                                            ),
                                        ],
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))",
                                            "gap": "24px",
                                            "alignItems": "start",
                                        },
                                    ),
                                    type="default",
                                ),
                                dcc.Loading(
                                    children=html.Div(
                                        [
                                            _create_train_timeline_comparison_chart_section(),
                                        ],
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "minmax(0, 1fr)",
                                            "gap": "24px",
                                            "alignItems": "start",
                                        },
                                    ),
                                    type="default",
                                ),
                            ],
                            className="balance-results-stack",
                ),
            ],
            className="balance-page-shell",
        ),
    ]
)


clientside_callback(
    """
    function(woodmacRef, trainRef, eaRef, metadata, currentRevision) {
        const noUpdate = window.dash_clientside.no_update;
        const expected = [
            [woodmacRef, "woodmac_store"],
            [trainRef, "train_store"],
            [eaRef, "ea_store"]
        ];
        let sourceKey = null;
        for (const [reference, component] of expected) {
            if (
                !reference ||
                reference.format !== "capacity_source_ref_v3" ||
                reference.component !== component ||
                !reference.source_key ||
                reference.snapshot_key !== reference.source_key
            ) {
                return noUpdate;
            }
            if (sourceKey === null) {
                sourceKey = String(reference.source_key);
            } else if (sourceKey !== String(reference.source_key)) {
                return noUpdate;
            }
        }
        if (
            !metadata ||
            String(metadata.snapshot_key || "") !== sourceKey
        ) {
            return noUpdate;
        }
        const revision = {
            format: "capacity_source_render_revision_v1",
            source_key: sourceKey
        };
        if (JSON.stringify(currentRevision) === JSON.stringify(revision)) {
            return noUpdate;
        }
        return revision;
    }
    """,
    Output("capacity-page-source-render-revision-store", "data"),
    Input("capacity-page-woodmac-data-store", "data"),
    Input("capacity-page-train-capacity-data-store", "data"),
    Input("capacity-page-ea-capacity-data-store", "data"),
    Input("capacity-page-metadata-store", "data"),
    State("capacity-page-source-render-revision-store", "data"),
)


clientside_callback(
    """
    function(
        sourceRevision,
        selectedScenario,
        workingScenario,
        dirtyState,
        refreshRevision,
        scenarioOptions,
        currentRevision
    ) {
        const noUpdate = window.dash_clientside.no_update;
        if (
            !sourceRevision ||
            sourceRevision.format !== "capacity_source_render_revision_v1" ||
            !sourceRevision.source_key
        ) {
            return noUpdate;
        }
        const noSelection =
            selectedScenario === null || selectedScenario === undefined;
        if (noSelection && !workingScenario) {
            const emptyRevision = {
                format: "capacity_scenario_render_revision_v1",
                source_key: String(sourceRevision.source_key),
                scenario_id: null,
                empty: true,
                dirty: Boolean(dirtyState && dirtyState.dirty),
                dirty_source: String(
                    (dirtyState && dirtyState.source) || ""
                ),
                dirty_updated_at: String(
                    (dirtyState && dirtyState.updated_at) || ""
                ),
                refresh_revision: String(refreshRevision || "")
            };
            if (
                JSON.stringify(currentRevision) ===
                JSON.stringify(emptyRevision)
            ) {
                return noUpdate;
            }
            return emptyRevision;
        }
        if (
            noSelection ||
            !workingScenario ||
            workingScenario.format !== "capacity_scenario_working_v3" ||
            String(workingScenario.scenario_id) !== String(selectedScenario) ||
            !workingScenario.base_revision ||
            !Array.isArray(scenarioOptions)
        ) {
            return noUpdate;
        }
        const selectedOption = scenarioOptions.find(
            option => option &&
                String(option.scenario_id) === String(selectedScenario)
        );
        if (
            !selectedOption ||
            String(selectedOption.current_snapshot_timestamp_utc || "") !==
                String(workingScenario.base_revision)
        ) {
            return noUpdate;
        }
        const revision = {
            format: "capacity_scenario_render_revision_v1",
            source_key: String(sourceRevision.source_key),
            scenario_id: String(selectedScenario),
            base_revision: String(workingScenario.base_revision),
            working_mode: String(workingScenario.mode || ""),
            dirty: Boolean(dirtyState && dirtyState.dirty),
            dirty_source: String((dirtyState && dirtyState.source) || ""),
            dirty_updated_at: String(
                (dirtyState && dirtyState.updated_at) || ""
            ),
            refresh_revision: String(refreshRevision || ""),
            option_name: String(selectedOption.scenario_name || ""),
            option_revision: String(
                selectedOption.current_snapshot_timestamp_utc || ""
            )
        };
        if (JSON.stringify(currentRevision) === JSON.stringify(revision)) {
            return noUpdate;
        }
        return revision;
    }
    """,
    Output("capacity-page-scenario-render-revision-store", "data"),
    Input("capacity-page-source-render-revision-store", "data"),
    Input("capacity-page-capacity-scenario-selected-store", "data"),
    Input("capacity-page-capacity-scenario-working-store", "data"),
    Input("capacity-page-capacity-scenario-dirty-store", "data"),
    Input("capacity-page-capacity-scenario-refresh-store", "data"),
    Input("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-scenario-render-revision-store", "data"),
)


@callback(
    Output("capacity-page-woodmac-data-store", "data"),
    Output("capacity-page-train-capacity-data-store", "data"),
    Output("capacity-page-ea-capacity-data-store", "data"),
    Output("capacity-page-country-options-store", "data"),
    Output("capacity-page-refresh-timestamp-store", "data"),
    Output("capacity-page-load-error-store", "data"),
    Output("capacity-page-metadata-store", "data"),
    Output("capacity-page-country-dropdown", "options"),
    Output("capacity-page-country-dropdown", "value"),
    Output("capacity-page-date-range", "min_date_allowed"),
    Output("capacity-page-date-range", "max_date_allowed"),
    Output("capacity-page-date-range", "start_date"),
    Output("capacity-page-date-range", "end_date"),
    Output("capacity-page-capacity-scenario-options-store", "data", allow_duplicate=True),
    Output("capacity-page-source-refresh-job-store", "data"),
    Output("capacity-page-source-refresh-poll", "disabled"),
    Output("capacity-page-source-refresh-status", "children"),
    Input("capacity-page-initial-load-trigger", "data"),
    State("capacity-page-country-dropdown", "value"),
    State("capacity-page-date-range", "start_date"),
    State("capacity-page-date-range", "end_date"),
    prevent_initial_call="initial_duplicate",
)
def load_capacity_source_data(
    _initial_trigger,
    current_countries,
    current_start_date,
    current_end_date,
):
    source_load_errors = []
    scenario_load_errors = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        source_state_future = executor.submit(
            fetch_capacity_source_state,
            engine,
            "__initial_capacity_display__",
        )
        scenario_future = executor.submit(get_available_capacity_scenarios, engine)
        try:
            source_state = source_state_future.result()
        except Exception as exc:
            source_state = {}
            source_load_errors.append(
                f"The completed Capacity source cache could not be loaded: {exc}"
            )
        source_state["source_key"] = ""
        source_state["watermarks"] = {}
        display_source_key = str(source_state.get("display_source_key") or "")
        snapshot = None
        if display_source_key:
            try:
                snapshot = _fetch_relational_capacity_source_snapshot(
                    display_source_key,
                    source_state.get("watermarks") or {},
                    source_state=source_state,
                )
            except Exception as exc:
                source_load_errors.append(f"Capacity source cache loading failed: {exc}")
        try:
            scenario_options = _serialize_capacity_scenario_options(
                scenario_future.result()
            )
        except Exception as exc:
            scenario_options = no_update
            scenario_load_errors.append(
                f"Internal scenario options could not be loaded: {exc}"
            )

    current_source_key = ""
    current_status = ""
    load_errors = [*source_load_errors, *scenario_load_errors]
    job = (
        None
        if source_load_errors and snapshot is None
        else {"status": "running", "discover_current": True, "source_key": ""}
    )

    if snapshot is None:
        error_message = " ".join(load_errors)
        status = str(
            (job or {}).get("status")
            or ("failed" if error_message else "running")
        )
        if status == "failed" and job is None:
            job = {
                "source_key": current_source_key,
                "status": "failed",
                "error_message": error_message,
            }
        return (
            None, None, None, [], None, error_message or None, {}, [], no_update,
            None, None, no_update, no_update, scenario_options, job,
            status != "running" or job is None,
            _build_capacity_refresh_status(
                "loading" if status == "running" else status,
                error_message=str((job or {}).get("error_message") or error_message),
            ),
        )

    if load_errors:
        existing_warning = str(snapshot.get("error_message") or "").strip()
        snapshot["error_message"] = " ".join(
            value for value in [existing_warning, *load_errors] if value
        )
    activation_values = _build_capacity_source_activation_values(
        snapshot,
        scenario_options,
        current_countries,
        current_start_date,
        current_end_date,
    )
    display_source_key = str(snapshot.get("snapshot_key") or "")
    status = str((job or {}).get("status") or current_status or "completed")
    if (
        current_source_key
        and display_source_key != current_source_key
        and status == "completed"
    ):
        status = "running"
    return activation_values + (
        job,
        status != "running",
        _build_capacity_refresh_status(
            status,
            snapshot_key=display_source_key,
            error_message=str((job or {}).get("error_message") or ""),
            last_success_at=snapshot.get("last_success_at"),
        ),
    )


@callback(
    Output("capacity-page-source-refresh-job-store", "data", allow_duplicate=True),
    Output("capacity-page-source-refresh-poll", "disabled", allow_duplicate=True),
    Output("capacity-page-source-refresh-status", "children", allow_duplicate=True),
    Input("global-refresh-button", "n_clicks"),
    Input("capacity-page-train-mapping-save-trigger", "data"),
    State("capacity-page-train-capacity-data-store", "data"),
    prevent_initial_call=True,
)
def request_capacity_source_refresh(
    _refresh_clicks,
    _train_save_trigger,
    current_train_ref,
):
    if ctx.triggered_id not in {
        "global-refresh-button",
        "capacity-page-train-mapping-save-trigger",
    }:
        raise PreventUpdate
    snapshot_key = (
        str(current_train_ref.get("snapshot_key") or "")
        if _is_capacity_source_ref(current_train_ref)
        else _get_train_capacity_snapshot_key(current_train_ref)
    )
    last_success_at = (
        current_train_ref.get("last_success_at")
        if _is_capacity_source_ref(current_train_ref)
        else None
    )
    try:
        job = _queue_capacity_source_refresh()
        job.pop("_created", None)
        status = str(job.get("status") or "running")
        return (
            job,
            status != "running",
            _build_capacity_refresh_status(
                status,
                snapshot_key=snapshot_key,
                last_success_at=last_success_at,
            ),
        )
    except Exception as exc:
        return (
            {
                "status": "failed",
                "error_message": str(exc),
            },
            True,
            _build_capacity_refresh_status(
                "failed",
                snapshot_key=snapshot_key,
                error_message=str(exc),
                last_success_at=last_success_at,
            ),
        )


def _activate_completed_capacity_source_refresh(
    source_key,
    job,
    current_countries,
    current_start_date,
    current_end_date,
    current_snapshot_key,
    current_last_success_at,
):
    try:
        snapshot = _fetch_relational_capacity_source_snapshot(
            source_key,
            source_state=job,
        )
    except Exception as exc:
        message = f"Completed Capacity source cache loading failed: {exc}"
        values = [no_update] * 14
        values[5] = message
        return tuple(values) + (
            {**job, "status": "failed", "error_message": message},
            True,
            _build_capacity_refresh_status(
                "failed",
                snapshot_key=current_snapshot_key,
                error_message=message,
                last_success_at=current_last_success_at,
            ),
        )
    if snapshot is None:
        message = "The completed source cache has no readable events."
        values = [no_update] * 14
        values[5] = message
        return tuple(values) + (
            {**job, "status": "failed", "error_message": message},
            True,
            _build_capacity_refresh_status(
                "failed",
                snapshot_key=current_snapshot_key,
                error_message=message,
                last_success_at=current_last_success_at,
            ),
        )
    try:
        scenario_options = _serialize_capacity_scenario_options(
            get_available_capacity_scenarios(engine)
        )
    except Exception as exc:
        scenario_options = no_update
        snapshot["error_message"] = (
            f"Internal scenario options could not be loaded: {exc}"
        )
    activation_values = _build_capacity_source_activation_values(
        snapshot,
        scenario_options,
        current_countries,
        current_start_date,
        current_end_date,
    )
    return activation_values + (
        job,
        True,
        _build_capacity_refresh_status(
            "completed",
            snapshot_key=str(snapshot.get("snapshot_key") or source_key),
            last_success_at=job.get("finished_at") or snapshot.get("last_success_at"),
        ),
    )


@callback(
    Output("capacity-page-woodmac-data-store", "data", allow_duplicate=True),
    Output("capacity-page-train-capacity-data-store", "data", allow_duplicate=True),
    Output("capacity-page-ea-capacity-data-store", "data", allow_duplicate=True),
    Output("capacity-page-country-options-store", "data", allow_duplicate=True),
    Output("capacity-page-refresh-timestamp-store", "data", allow_duplicate=True),
    Output("capacity-page-load-error-store", "data", allow_duplicate=True),
    Output("capacity-page-metadata-store", "data", allow_duplicate=True),
    Output("capacity-page-country-dropdown", "options", allow_duplicate=True),
    Output("capacity-page-country-dropdown", "value", allow_duplicate=True),
    Output("capacity-page-date-range", "min_date_allowed", allow_duplicate=True),
    Output("capacity-page-date-range", "max_date_allowed", allow_duplicate=True),
    Output("capacity-page-date-range", "start_date", allow_duplicate=True),
    Output("capacity-page-date-range", "end_date", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-options-store", "data", allow_duplicate=True),
    Output("capacity-page-source-refresh-job-store", "data", allow_duplicate=True),
    Output("capacity-page-source-refresh-poll", "disabled", allow_duplicate=True),
    Output("capacity-page-source-refresh-status", "children", allow_duplicate=True),
    Input("capacity-page-source-refresh-poll", "n_intervals"),
    State("capacity-page-source-refresh-job-store", "data"),
    State("capacity-page-country-dropdown", "value"),
    State("capacity-page-date-range", "start_date"),
    State("capacity-page-date-range", "end_date"),
    State("capacity-page-train-capacity-data-store", "data"),
    prevent_initial_call=True,
)
def poll_capacity_source_refresh(
    _n_intervals,
    refresh_job,
    current_countries,
    current_start_date,
    current_end_date,
    current_train_ref,
):
    current_snapshot_key = (
        str(current_train_ref.get("snapshot_key") or "")
        if _is_capacity_source_ref(current_train_ref)
        else _get_train_capacity_snapshot_key(current_train_ref)
    )
    current_last_success_at = (
        current_train_ref.get("last_success_at")
        if _is_capacity_source_ref(current_train_ref)
        else None
    )
    if (refresh_job or {}).get("discover_current"):
        try:
            current_state = _fetch_capacity_source_state()
            current_source_key = str(current_state.get("source_key") or "")
            current_status = str(current_state.get("current_status") or "")
            display_source_key = str(current_state.get("display_source_key") or "")
            if not current_source_key:
                raise RuntimeError("Current Capacity source watermarks are unavailable.")

            if current_status == "completed" and display_source_key == current_source_key:
                snapshot = _fetch_relational_capacity_source_snapshot(
                    current_source_key,
                    current_state.get("watermarks") or {},
                    source_state=current_state,
                )
                if snapshot is None:
                    raise RuntimeError(
                        "The current completed Capacity source has no readable events."
                    )
                try:
                    scenario_options = _serialize_capacity_scenario_options(
                        get_available_capacity_scenarios(engine)
                    )
                except Exception as exc:
                    scenario_options = no_update
                    snapshot["error_message"] = (
                        f"Internal scenario options could not be loaded: {exc}"
                    )
                activation_values = _build_capacity_source_activation_values(
                    snapshot,
                    scenario_options,
                    current_countries,
                    current_start_date,
                    current_end_date,
                )
                completed_job = {
                    "source_key": current_source_key,
                    "status": "completed",
                    "finished_at": _serialize_timestamp(
                        current_state.get("current_finished_at")
                        or current_state.get("display_finished_at")
                    ),
                }
                return activation_values + (
                    completed_job,
                    True,
                    _build_capacity_refresh_status(
                        "completed",
                        snapshot_key=current_source_key,
                        last_success_at=snapshot.get("last_success_at"),
                    ),
                )

            next_job = _queue_capacity_source_refresh(current_source_key)
            next_job.pop("_created", None)
            next_status = str(next_job.get("status") or "running")
            return (no_update,) * 14 + (
                next_job,
                next_status != "running",
                _build_capacity_refresh_status(
                    next_status,
                    snapshot_key=current_snapshot_key,
                    last_success_at=current_last_success_at,
                ),
            )
        except Exception as exc:
            message = f"Capacity source freshness check failed: {exc}"
            values = [no_update] * 14
            values[5] = message
            return tuple(values) + (
                {
                    "source_key": "",
                    "status": "failed",
                    "error_message": message,
                },
                True,
                _build_capacity_refresh_status(
                    "failed",
                    snapshot_key=current_snapshot_key,
                    error_message=message,
                    last_success_at=current_last_success_at,
                ),
            )

    source_key = str((refresh_job or {}).get("source_key") or "")
    if not source_key:
        message = str(
            (refresh_job or {}).get("error_message")
            or "Capacity source refresh stopped because no source key is available."
        )
        values = [no_update] * 14
        values[5] = message
        failed_job = {
            **(refresh_job or {}),
            "status": "failed",
            "error_message": message,
        }
        return tuple(values) + (
            failed_job,
            True,
            _build_capacity_refresh_status("failed", error_message=message),
        )

    try:
        job = _read_capacity_refresh_job(source_key)
    except Exception as exc:
        message = f"Capacity source refresh status could not be read: {exc}"
        values = [no_update] * 14
        values[5] = message
        return tuple(values) + (
            {
                "source_key": source_key,
                "status": "failed",
                "error_message": message,
            },
            True,
            _build_capacity_refresh_status(
                "failed",
                snapshot_key=current_snapshot_key,
                error_message=message,
                last_success_at=current_last_success_at,
            ),
        )
    if job is None:
        message = (
            "Capacity source refresh stopped because its database job row is missing."
        )
        values = [no_update] * 14
        values[5] = message
        failed_job = {
            "source_key": source_key,
            "status": "failed",
            "error_message": message,
        }
        return tuple(values) + (
            failed_job,
            True,
            _build_capacity_refresh_status(
                "failed",
                snapshot_key=current_snapshot_key,
                error_message=message,
                last_success_at=current_last_success_at,
            ),
        )
    status = str(job.get("status") or "running")
    if status == "running":
        requested_at = pd.to_datetime(
            job.get("requested_at"), errors="coerce", utc=True
        )
        if (
            pd.notna(requested_at)
            and requested_at
            < pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=5)
        ):
            message = "Refresh worker did not complete within five minutes."
            try:
                mark_capacity_source_failed(engine, source_key, message)
            except Exception as exc:
                message = f"{message} The failed state could not be persisted: {exc}"
            values = [no_update] * 14
            values[5] = message
            return tuple(values) + (
                {**job, "status": "failed", "error_message": message},
                True,
                _build_capacity_refresh_status(
                    "failed",
                    snapshot_key=current_snapshot_key,
                    error_message=message,
                    last_success_at=current_last_success_at,
                ),
            )
        return (no_update,) * 14 + (
            job,
            False,
            _build_capacity_refresh_status(
                status,
                snapshot_key=current_snapshot_key,
                last_success_at=current_last_success_at,
            ),
        )

    if status == "failed":
        try:
            current_state = _fetch_capacity_source_state()
        except Exception as exc:
            message = f"Capacity source freshness check failed: {exc}"
            values = [no_update] * 14
            values[5] = message
            return tuple(values) + (
                {**job, "error_message": message},
                True,
                _build_capacity_refresh_status(
                    "failed",
                    snapshot_key=current_snapshot_key,
                    error_message=message,
                    last_success_at=current_last_success_at,
                ),
            )
        current_source_key = str(current_state.get("source_key") or "")
        if current_source_key and current_source_key != source_key:
            try:
                next_job = _queue_capacity_source_refresh(current_source_key)
            except Exception as exc:
                message = f"Capacity source replacement refresh could not start: {exc}"
                values = [no_update] * 14
                values[5] = message
                return tuple(values) + (
                    {
                        "source_key": current_source_key,
                        "status": "failed",
                        "error_message": message,
                    },
                    True,
                    _build_capacity_refresh_status(
                        "failed",
                        snapshot_key=current_snapshot_key,
                        error_message=message,
                        last_success_at=current_last_success_at,
                    ),
                )
            next_job.pop("_created", None)
            next_status = str(next_job.get("status") or "running")
            if next_status == "completed":
                return _activate_completed_capacity_source_refresh(
                    current_source_key,
                    next_job,
                    current_countries,
                    current_start_date,
                    current_end_date,
                    current_snapshot_key,
                    current_last_success_at,
                )
            return (no_update,) * 14 + (
                next_job,
                next_status != "running",
                _build_capacity_refresh_status(
                    next_status,
                    snapshot_key=current_snapshot_key,
                    error_message=str(next_job.get("error_message") or ""),
                    last_success_at=current_last_success_at,
                ),
            )
        values = [no_update] * 14
        values[5] = str(job.get("error_message") or "") or no_update
        return tuple(values) + (
            job,
            True,
            _build_capacity_refresh_status(
                status,
                snapshot_key=current_snapshot_key,
                error_message=str(job.get("error_message") or ""),
                last_success_at=current_last_success_at,
            ),
        )

    return _activate_completed_capacity_source_refresh(
        source_key,
        job,
        current_countries,
        current_start_date,
        current_end_date,
        current_snapshot_key,
        current_last_success_at,
    )


def update_capacity_country_options(available_countries, current_selection):
    available_countries = available_countries or []
    options = [{"label": country, "value": country} for country in available_countries]

    if current_selection is None:
        selected_values = default_selected_countries(available_countries)
    else:
        selected_values = _resolve_selected_countries(
            available_countries,
            current_selection,
        )
        if current_selection and not selected_values:
            selected_values = default_selected_countries(available_countries)

    return options, selected_values


def update_capacity_date_range(woodmac_data, ea_capacity_data, current_start_date, current_end_date):
    if _is_capacity_source_ref(woodmac_data):
        source_events = _resolve_capacity_source_ref(woodmac_data, "woodmac_store")
        min_date, max_date = capacity_source_event_bounds(source_events)
        return _resolve_capacity_date_values(
            min_date,
            max_date,
            current_start_date,
            current_end_date,
        )
    woodmac_raw_df = _deserialize_woodmac_capacity_store(woodmac_data)
    ea_raw_df = _deserialize_ea_capacity_store(ea_capacity_data)
    ea_scope_df = pd.DataFrame(columns=["month", "country_name", "total_mmtpa"])
    if not ea_raw_df.empty:
        ea_scope_df = ea_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
            ["month", "country_name", "total_mmtpa"]
        ]

    min_date, max_date = _get_date_bounds([woodmac_raw_df, ea_scope_df])
    return _resolve_capacity_date_values(
        min_date,
        max_date,
        current_start_date,
        current_end_date,
    )


@callback(
    Output("capacity-page-load-error-banner", "children"),
    Input("capacity-page-load-error-store", "data"),
)
def update_capacity_error_banner(error_message):
    if not error_message:
        return html.Div()

    return html.Div(error_message, className="balance-error-banner")


@callback(
    Output("capacity-page-capacity-scenario-options-store", "data"),
    Output("capacity-page-load-error-store", "data", allow_duplicate=True),
    Input("capacity-page-capacity-scenario-refresh-store", "data"),
    State("capacity-page-load-error-store", "data"),
    prevent_initial_call=True,
)
def load_capacity_scenario_options(_scenario_refresh_timestamp, current_error):
    try:
        clear_recovered_error = (
            None
            if str(current_error or "").startswith(
                "Internal scenario options could not be refreshed:"
            )
            else no_update
        )
        return (
            _serialize_capacity_scenario_options(
                get_available_capacity_scenarios(engine)
            ),
            clear_recovered_error,
        )
    except Exception as exc:
        return no_update, f"Internal scenario options could not be refreshed: {exc}"


@callback(
    Output("capacity-page-scenario-add-country", "options"),
    Input("capacity-page-capacity-scenario-refresh-store", "data"),
)
def load_capacity_scenario_country_options(_scenario_refresh_timestamp):
    try:
        return [
            {"label": country_name, "value": country_name}
            for country_name in list_canonical_country_names(engine)
        ]
    except Exception:
        return []


@callback(
    Output("capacity-page-source-metadata-status", "children"),
    Input("capacity-page-metadata-store", "data"),
    Input("capacity-page-capacity-scenario-selected-store", "data"),
    Input("capacity-page-capacity-scenario-options-store", "data"),
)
def render_capacity_source_metadata_status(
    metadata,
    selected_scenario_id,
    scenario_options_data,
):
    return _build_capacity_source_metadata_text(
        metadata,
        selected_scenario_id,
        scenario_options_data,
    )


@callback(
    Output("capacity-page-internal-scenario-dropdown", "options"),
    Output("capacity-page-capacity-scenario-create-source-dropdown", "options"),
    Output("capacity-page-train-timeline-current-scenario-label", "children"),
    Output("capacity-page-capacity-scenario-dirty-indicator", "children"),
    Output("capacity-page-capacity-scenario-save-button", "disabled"),
    Output("capacity-page-capacity-scenario-revert-button", "disabled"),
    Output("capacity-page-capacity-scenario-delete-button", "disabled"),
    Output("capacity-page-scenario-add-row-button", "disabled"),
    Output("capacity-page-capacity-scenario-upload", "disabled"),
    Output("capacity-page-capacity-scenario-upload-button", "disabled"),
    Output("capacity-page-capacity-scenario-upload-button", "style"),
    Output("capacity-page-ramp-generate-button", "disabled"),
    Output("capacity-page-ramp-publish-button", "disabled"),
    Output("capacity-page-ramp-profile-assign-button", "disabled"),
    Output("capacity-page-ramp-identity-reconcile-button", "disabled"),
    Output("capacity-page-capacity-scenario-create-button", "disabled"),
    Output("capacity-page-internal-scenario-dropdown", "disabled"),
    Input("capacity-page-capacity-scenario-options-store", "data"),
    Input("capacity-page-capacity-scenario-selected-store", "data"),
    Input("capacity-page-capacity-scenario-dirty-store", "data"),
    Input("capacity-page-capacity-scenario-pending-selection-store", "data"),
    Input("capacity-page-ramp-forecast-status-store", "data"),
    Input("capacity-page-ramp-profile-train", "value"),
    Input("capacity-page-ramp-profile-choice", "value"),
    Input("capacity-page-seasonal-profile-choice", "value"),
    Input("capacity-page-scenario-command-running-store", "data"),
)
def sync_capacity_scenario_controls(
    options_data,
    selected_scenario_id,
    dirty_data,
    pending_selection,
    ramp_status,
    profile_train,
    ramp_profile,
    seasonal_profile,
    command_running,
):
    options = [
        {
            "label": option.get("scenario_name"),
            "value": int(option["scenario_id"]),
        }
        for option in (options_data or [])
        if pd.notna(pd.to_numeric(option.get("scenario_id"), errors="coerce"))
    ]
    normalized_selected = pd.to_numeric(selected_scenario_id, errors="coerce")
    selected_value = int(normalized_selected) if pd.notna(normalized_selected) else None

    action_state = _build_capacity_scenario_action_state(
        selected_value,
        dirty_data,
        pending_selection,
        ramp_status,
        profile_train,
        ramp_profile,
        seasonal_profile,
    )
    if command_running:
        action_state = {key: True for key in action_state}
    scenario_disabled = action_state["edit_disabled"]
    upload_button_style = {
        "padding": "7px 12px",
        "borderRadius": "999px",
        "border": "1px solid #fdba74",
        "backgroundColor": "#fff7ed",
        "color": "#9a3412",
        "fontSize": "12px",
        "fontWeight": "700",
        "cursor": "pointer" if not scenario_disabled else "not-allowed",
        "opacity": "1" if not scenario_disabled else "0.55",
    }
    return (
        options,
        options,
        _build_capacity_scenario_badge_text(selected_value, options_data),
        _build_capacity_scenario_dirty_label(dirty_data),
        action_state["save_disabled"],
        action_state["revert_disabled"],
        action_state["delete_disabled"],
        action_state["edit_disabled"],
        action_state["edit_disabled"],
        action_state["edit_disabled"],
        upload_button_style,
        action_state["generate_disabled"],
        action_state["publish_disabled"],
        action_state["profile_disabled"],
        action_state["reconcile_disabled"],
        bool(command_running),
        bool(command_running),
    )


@callback(
    Output("capacity-page-ramp-forecast-status", "children"),
    Output("capacity-page-ramp-forecast-status-store", "data"),
    Input("capacity-page-capacity-scenario-selected-store", "data"),
    Input("capacity-page-capacity-scenario-refresh-store", "data"),
)
def render_capacity_ramp_forecast_status(selected_scenario_id, _scenario_refresh_timestamp):
    selected_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_value):
        return _render_capacity_ramp_status_card(None), None
    try:
        status = get_latest_ramp_forecast_status(int(selected_value), engine)
        if status:
            status["qa_issues"] = get_latest_ramp_forecast_qa_issues(
                int(selected_value),
                engine,
            )[:3]
    except Exception as exc:
        message = f"Ramp forecast status unavailable: {exc}"
        return (
            html.Div(
                message,
                className="balance-metadata-row",
                style={"fontSize": "11px", "color": "#9a3412", "fontWeight": "700"},
            ),
            {"error_message": message},
        )
    return (
        _render_capacity_ramp_status_card(status),
        _serialize_capacity_ramp_action_status(status),
    )


@callback(
    Output("capacity-page-ramp-profile-train", "options"),
    Output("capacity-page-ramp-profile-choice", "options"),
    Output("capacity-page-scenario-add-ramp-category", "options"),
    Output("capacity-page-scenario-add-seasonal-profile", "options"),
    Output("capacity-page-seasonal-profile-choice", "options"),
    Output("capacity-page-ramp-profile-train", "value"),
    Output("capacity-page-ramp-profile-choice", "value"),
    Output("capacity-page-seasonal-profile-choice", "value"),
    Input("capacity-page-capacity-scenario-selected-store", "data"),
    Input("capacity-page-capacity-scenario-refresh-store", "data"),
)
def load_capacity_ramp_review_options(selected_scenario_id, _refresh_timestamp):
    profile_catalog = get_active_ramp_profile_catalog(engine)
    profile_choices = profile_catalog["choices"]
    ramp_category_options = profile_catalog["category_options"]
    seasonal_profile_options = get_active_seasonal_profile_options(engine)
    selected_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_value):
        return (
            [], profile_choices, ramp_category_options,
            seasonal_profile_options, seasonal_profile_options,
            None, None, None,
        )
    try:
        issues = get_latest_ramp_forecast_qa_issues(
            int(selected_value),
            engine,
            qa_type=None,
        )
    except Exception:
        return (
            [], profile_choices, ramp_category_options,
            seasonal_profile_options, seasonal_profile_options,
            None, None, None,
        )
    options = []
    seen = set()
    for issue in issues:
        if issue.get("qa_type") not in {
            "missing_approved_ramp_profile",
            "missing_approved_seasonal_profile",
        }:
            continue
        train_key = issue.get("train_key")
        if not train_key or str(train_key) in seen:
            continue
        seen.add(str(train_key))
        label = " / ".join(
            part
            for part in [
                str(issue.get("country_name") or "").strip(),
                str(issue.get("plant_name") or "").strip(),
                normalize_train_label(issue.get("train_label")),
            ]
            if part
        )
        options.append({"label": label or str(train_key), "value": str(train_key)})
    return (
        options, profile_choices, ramp_category_options,
        seasonal_profile_options, seasonal_profile_options,
        None, None, None,
    )


@callback(
    Output("capacity-page-capacity-scenario-refresh-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-message", "children", allow_duplicate=True),
    Input("capacity-page-ramp-profile-assign-button", "n_clicks"),
    State("capacity-page-ramp-profile-train", "value"),
    State("capacity-page-ramp-profile-choice", "value"),
    State("capacity-page-seasonal-profile-choice", "value"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    running=[
        (
            Output(
                "capacity-page-scenario-command-running-store",
                "data",
                allow_duplicate=True,
            ),
            True,
            False,
        ),
    ],
    prevent_initial_call=True,
)
def assign_capacity_ramp_profiles(
    n_clicks,
    train_key,
    profile_choice,
    seasonal_profile_choice,
    selected_scenario_id,
    dirty_store,
):
    if not n_clicks:
        raise PreventUpdate
    selected_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_value):
        return no_update, _build_capacity_scenario_message(
            "Select an internal scenario first.", "warning"
        )
    if isinstance(dirty_store, dict) and dirty_store.get("dirty"):
        return no_update, _build_capacity_scenario_message(
            "Save or Revert scenario edits before assigning forecast profiles.",
            "warning",
        )
    if not train_key or not (profile_choice or seasonal_profile_choice):
        return no_update, _build_capacity_scenario_message(
            "Select a train and at least one SQL profile.", "warning"
        )
    try:
        review_issues = get_latest_ramp_forecast_qa_issues(
            int(selected_value),
            engine,
            qa_type=None,
        )
        eligible_train_keys = {
            str(issue.get("train_key"))
            for issue in review_issues
            if issue.get("train_key")
            and issue.get("qa_type")
            in {
                "missing_approved_ramp_profile",
                "missing_approved_seasonal_profile",
            }
        }
        if str(train_key) not in eligible_train_keys:
            return no_update, _build_capacity_scenario_message(
                "The selected train is not a current profile QA issue for this scenario. Refresh the selection and try again.",
                "warning",
            )
        result = assign_terminal_train_profiles(
            engine,
            train_key=str(train_key),
            ramp_profile_id=str(profile_choice) if profile_choice else None,
            seasonal_profile_id=(
                str(seasonal_profile_choice) if seasonal_profile_choice else None
            ),
        )
        assigned_names = [
            value
            for value in (
                result.get("profile_name"),
                result.get("seasonal_profile_name"),
            )
            if value
        ]
    except Exception as exc:
        return no_update, _build_capacity_scenario_message(
            f"Profile assignment failed: {exc}", "error"
        )
    action = "Assigned" if result.get("changed") else "Already assigned"
    return dt.datetime.utcnow().isoformat(), _build_capacity_scenario_message(
        f"{action} {', '.join(map(str, assigned_names))}. Generate a new ramp run to revalidate the scenario.",
        "success",
    )


@callback(
    Output("capacity-page-capacity-scenario-refresh-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-message", "children", allow_duplicate=True),
    Input("capacity-page-ramp-identity-reconcile-button", "n_clicks"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    running=[
        (
            Output(
                "capacity-page-scenario-command-running-store",
                "data",
                allow_duplicate=True,
            ),
            True,
            False,
        ),
    ],
    prevent_initial_call=True,
)
def reconcile_capacity_ramp_identities(n_clicks, selected_scenario_id, dirty_store):
    if not n_clicks:
        raise PreventUpdate
    selected_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_value):
        return no_update, _build_capacity_scenario_message(
            "Select an internal scenario first.", "warning"
        )
    if isinstance(dirty_store, dict) and dirty_store.get("dirty"):
        return no_update, _build_capacity_scenario_message(
            "Save or Revert scenario edits before reconciling identities.",
            "warning",
        )
    try:
        provider_backfill_result = backfill_terminal_keys(
            engine,
            created_by="capacity_page",
            scenario_id=int(selected_value),
        )
        draft_result = create_draft_registry_for_unresolved_non_woodmac_rows(
            engine,
            created_by="capacity_page",
            scenario_id=int(selected_value),
        )
        backfill_result = backfill_terminal_keys(
            engine,
            created_by="capacity_page",
            scenario_id=int(selected_value),
        )
    except Exception as exc:
        return no_update, _build_capacity_scenario_message(
            f"Identity reconciliation failed: {exc}", "error"
        )
    resolved = int(provider_backfill_result.get("scenario_rows_resolved") or 0) + int(
        backfill_result.get("scenario_rows_resolved") or 0
    ) + int(
        backfill_result.get("aggregate_scenario_rows_resolved") or 0
    )
    drafted = int(draft_result.get("draft_trains_created") or 0)
    ambiguous = int(backfill_result.get("scenario_rows_ambiguous") or 0)
    return dt.datetime.utcnow().isoformat(), _build_capacity_scenario_message(
        f"Identity reconciliation linked {resolved:,} row(s), created {drafted:,} draft train(s), and left {ambiguous:,} ambiguous row(s) for manual mapping.",
        "success" if ambiguous == 0 else "warning",
    )


@callback(
    Output("capacity-page-capacity-scenario-refresh-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-message", "children", allow_duplicate=True),
    Input("capacity-page-ramp-generate-button", "n_clicks"),
    Input("capacity-page-ramp-publish-button", "n_clicks"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    running=[
        (
            Output(
                "capacity-page-scenario-command-running-store",
                "data",
                allow_duplicate=True,
            ),
            True,
            False,
        ),
    ],
    prevent_initial_call=True,
)
def manage_capacity_ramp_forecast(
    _generate_clicks,
    _publish_clicks,
    selected_scenario_id,
    dirty_store,
):
    trigger_id = ctx.triggered_id
    selected_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_value):
        return no_update, _build_capacity_scenario_message(
            "Select an internal scenario first.", "warning"
        )
    if isinstance(dirty_store, dict) and dirty_store.get("dirty"):
        return no_update, _build_capacity_scenario_message(
            "Save or Revert the scenario edits before generating or publishing a ramp forecast.",
            "warning",
        )

    scenario_id = int(selected_value)
    try:
        if trigger_id == "capacity-page-ramp-generate-button":
            result = generate_ramp_forecast_for_capacity_scenario(
                scenario_id,
                engine,
            )
        elif trigger_id == "capacity-page-ramp-publish-button":
            latest = get_latest_ramp_forecast_status(scenario_id, engine)
            if not latest:
                return no_update, _build_capacity_scenario_message(
                    "Generate a ramp forecast before publishing.", "warning"
                )
            result = publish_ramp_forecast_run(int(latest["run_id"]), engine)
        else:
            raise PreventUpdate
    except Exception as exc:
        return dt.datetime.utcnow().isoformat(), _build_capacity_scenario_message(
            f"Ramp forecast action failed: {exc}", "error"
        )

    ramp_text, ramp_tone = _format_capacity_ramp_result_message(result)
    return dt.datetime.utcnow().isoformat(), _build_capacity_scenario_message(
        ramp_text,
        ramp_tone,
    )


@callback(
    Output("capacity-page-internal-scenario-dropdown", "value"),
    Input("capacity-page-capacity-scenario-options-store", "data"),
    Input("capacity-page-capacity-scenario-dropdown-target-store", "data"),
    State("capacity-page-internal-scenario-dropdown", "value"),
)
def sync_capacity_scenario_dropdown_value(
    options_data,
    target_value,
    current_value,
):
    valid_ids, base_case_id = _get_capacity_scenario_valid_ids_and_base_case_id(
        options_data
    )

    clear_target = isinstance(target_value, dict) and target_value.get("clear")
    normalized_target = pd.to_numeric(
        None if clear_target else target_value,
        errors="coerce",
    )
    normalized_current = pd.to_numeric(current_value, errors="coerce")
    target_id = int(normalized_target) if pd.notna(normalized_target) else None
    current_id = int(normalized_current) if pd.notna(normalized_current) else None

    if clear_target:
        return None
    if target_id in valid_ids:
        return no_update if target_id == current_id else target_id
    if current_id in valid_ids:
        return no_update
    if base_case_id in valid_ids:
        return base_case_id
    return None if current_id is not None else no_update


@callback(
    Output("capacity-page-capacity-scenario-create-source-dropdown", "disabled"),
    Output("capacity-page-capacity-scenario-create-source-dropdown", "value"),
    Output("capacity-page-capacity-scenario-create-source-dropdown", "style"),
    Output("capacity-page-capacity-scenario-create-source-dropdown", "placeholder"),
    Input("capacity-page-capacity-scenario-create-base-type", "value"),
    State("capacity-page-capacity-scenario-create-source-dropdown", "value"),
)
def toggle_capacity_scenario_source_dropdown(base_type, current_source_value):
    base_style = {"width": "220px"}
    if base_type == "internal_scenario":
        return False, current_source_value, {**base_style, "opacity": "1"}, "Source internal scenario"
    return True, None, {**base_style, "opacity": "0.55"}, "Source internal scenario"


@callback(
    Output("capacity-page-capacity-scenario-selected-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-working-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-dirty-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-pending-selection-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-message", "children", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-switch-confirm", "displayed", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-dropdown-target-store", "data", allow_duplicate=True),
    Input("capacity-page-internal-scenario-dropdown", "value"),
    Input("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def handle_capacity_scenario_selection(
    dropdown_value,
    scenario_options_data,
    current_selected_scenario_id,
    current_dirty_store,
):
    trigger_id = ctx.triggered_id
    current_selected_value = pd.to_numeric(current_selected_scenario_id, errors="coerce")
    current_selected = int(current_selected_value) if pd.notna(current_selected_value) else None
    dirty_payload = (
        current_dirty_store
        if isinstance(current_dirty_store, dict)
        else {"dirty": False}
    )

    valid_ids, base_case_id = _get_capacity_scenario_valid_ids_and_base_case_id(
        scenario_options_data
    )
    next_selected_value = pd.to_numeric(dropdown_value, errors="coerce")
    next_selected = int(next_selected_value) if pd.notna(next_selected_value) else None

    if trigger_id == "capacity-page-capacity-scenario-options-store":
        if current_selected in valid_ids:
            raise PreventUpdate
        if next_selected not in valid_ids:
            next_selected = base_case_id if base_case_id in valid_ids else None
    elif trigger_id != "capacity-page-internal-scenario-dropdown":
        raise PreventUpdate

    if trigger_id in {
        "capacity-page-internal-scenario-dropdown",
        "capacity-page-capacity-scenario-options-store",
    }:
        if next_selected is not None and next_selected not in valid_ids:
            return (
                None,
                None,
                {"dirty": False},
                None,
                _build_capacity_scenario_message(
                    "The selected internal scenario is no longer available. Please choose another one from the dropdown.",
                    "warning",
                ),
                False,
                None,
            )
        if next_selected == current_selected:
            raise PreventUpdate

        if dirty_payload.get("dirty") and current_selected is not None:
            return (
                no_update,
                no_update,
                no_update,
                next_selected,
                _build_capacity_scenario_message(
                    "Unsaved edits detected. Confirm the scenario switch to discard the working copy.",
                    "warning",
                ),
                True,
                no_update,
            )

        if next_selected is None:
            return (
                None,
                None,
                {"dirty": False},
                None,
                html.Div(),
                False,
                None,
            )

        try:
            base_revision = _get_capacity_scenario_base_revision(
                next_selected,
                scenario_options_data,
            )
            selected_rows_df = fetch_capacity_scenario_rows(
                next_selected,
                engine,
                snapshot_timestamp_utc=base_revision,
            )
            _cache_saved_capacity_scenario_rows(
                next_selected,
                base_revision,
                selected_rows_df,
            )
        except Exception as exc:
            message = _build_capacity_scenario_message(
                f"The selected scenario could not be loaded: {exc}",
                "error",
            )
            if current_selected is None:
                return (
                    None,
                    None,
                    {"dirty": False},
                    None,
                    message,
                    False,
                    {"clear": True},
                )
            return (
                no_update,
                no_update,
                no_update,
                None,
                message,
                False,
                current_selected,
            )
        return (
            next_selected,
            _build_capacity_scenario_working_store(
                next_selected,
                scenario_options_data,
                base_revision=base_revision,
            ),
            {"dirty": False},
            None,
            html.Div(),
            False,
            next_selected,
        )

    raise PreventUpdate


@callback(
    Output("capacity-page-capacity-scenario-working-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-dirty-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-message", "children", allow_duplicate=True),
    Input("capacity-page-refresh-timestamp-store", "data"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    State("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    prevent_initial_call=True,
)
def refresh_selected_capacity_scenario_working_copy(
    _refresh_timestamp,
    selected_scenario_id,
    dirty_store,
    scenario_options_data,
    train_capacity_data,
    ea_capacity_data,
):
    selected_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_value):
        raise PreventUpdate

    dirty_payload = dirty_store if isinstance(dirty_store, dict) else {"dirty": False}
    if dirty_payload.get("dirty"):
        raise PreventUpdate

    try:
        selected_rows_df = fetch_capacity_scenario_rows(
            int(selected_value),
            engine,
        )
        if (
            selected_rows_df.empty
            or "upload_timestamp_utc" not in selected_rows_df.columns
        ):
            raise RuntimeError("The selected scenario has no current saved snapshot.")
        base_revision = _serialize_timestamp(
            selected_rows_df["upload_timestamp_utc"].iloc[0]
        )
        _cache_saved_capacity_scenario_rows(
            int(selected_value),
            base_revision,
            selected_rows_df,
        )
    except Exception as exc:
        return (
            no_update,
            no_update,
            _build_capacity_scenario_message(
                f"The selected scenario could not be refreshed: {exc}",
                "error",
            ),
        )
    return (
        _build_capacity_scenario_working_store(
            int(selected_value),
            scenario_options_data,
            base_revision=base_revision,
        ),
        {"dirty": False},
        no_update,
    )


@callback(
    Output("capacity-page-capacity-scenario-selected-store", "data"),
    Output("capacity-page-capacity-scenario-working-store", "data"),
    Output("capacity-page-capacity-scenario-dirty-store", "data"),
    Output("capacity-page-capacity-scenario-refresh-store", "data"),
    Output("capacity-page-capacity-scenario-pending-selection-store", "data"),
    Output("capacity-page-capacity-scenario-dropdown-target-store", "data"),
    Output("capacity-page-capacity-scenario-message", "children"),
    Output("capacity-page-capacity-scenario-switch-confirm", "displayed"),
    Output("capacity-page-capacity-scenario-delete-confirm", "displayed"),
    Input("capacity-page-capacity-scenario-create-button", "n_clicks"),
    Input("capacity-page-capacity-scenario-save-button", "n_clicks"),
    Input("capacity-page-capacity-scenario-revert-button", "n_clicks"),
    Input("capacity-page-capacity-scenario-delete-button", "n_clicks"),
    Input("capacity-page-capacity-scenario-switch-confirm", "submit_n_clicks"),
    Input("capacity-page-capacity-scenario-switch-confirm", "cancel_n_clicks"),
    Input("capacity-page-capacity-scenario-delete-confirm", "submit_n_clicks"),
    Input("capacity-page-capacity-scenario-delete-confirm", "cancel_n_clicks"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-working-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    State("capacity-page-capacity-scenario-pending-selection-store", "data"),
    State("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-capacity-scenario-create-name", "value"),
    State("capacity-page-capacity-scenario-create-base-type", "value"),
    State("capacity-page-capacity-scenario-create-source-dropdown", "value"),
    State("capacity-page-date-range", "start_date"),
    running=[
        (
            Output(
                "capacity-page-scenario-command-running-store",
                "data",
                allow_duplicate=True,
            ),
            True,
            False,
        ),
    ],
    prevent_initial_call=True,
)
def manage_capacity_scenario_state(
    _create_clicks,
    _save_clicks,
    _revert_clicks,
    _delete_clicks,
    _switch_submit_clicks,
    _switch_cancel_clicks,
    _delete_submit_clicks,
    _delete_cancel_clicks,
    current_selected_scenario_id,
    current_working_store,
    current_dirty_store,
    pending_selection_store,
    scenario_options_data,
    train_capacity_data,
    ea_capacity_data,
    create_name,
    create_base_type,
    create_source_scenario_id,
    selected_start_date,
    timeline_row_data=None,
):
    trigger_prop = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if trigger_prop in {"", "."}:
        raise PreventUpdate

    current_selected_value = pd.to_numeric(current_selected_scenario_id, errors="coerce")
    current_selected = int(current_selected_value) if pd.notna(current_selected_value) else None
    pending_selection_value = pd.to_numeric(pending_selection_store, errors="coerce")
    pending_selection = int(pending_selection_value) if pd.notna(pending_selection_value) else None
    working_rows_df = _resolve_active_capacity_scenario_rows(
        current_working_store,
        current_dirty_store,
        timeline_row_data,
    )

    if trigger_prop == "capacity-page-capacity-scenario-switch-confirm.submit_n_clicks":
        if pending_selection is None:
            raise PreventUpdate

        try:
            base_revision = _get_capacity_scenario_base_revision(
                pending_selection,
                scenario_options_data,
            )
            selected_rows_df = fetch_capacity_scenario_rows(
                pending_selection,
                engine,
                snapshot_timestamp_utc=base_revision,
            )
            _cache_saved_capacity_scenario_rows(
                pending_selection,
                base_revision,
                selected_rows_df,
            )
        except Exception as exc:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                None,
                current_selected,
                _build_capacity_scenario_message(
                    f"Scenario switch failed: {exc}",
                    "error",
                ),
                False,
                False,
            )
        return (
            pending_selection,
            _build_capacity_scenario_working_store(
                pending_selection,
                scenario_options_data,
                base_revision=base_revision,
            ),
            {"dirty": False},
            no_update,
            None,
            pending_selection,
            _build_capacity_scenario_message(
                "Scenario switched. Unsaved edits from the previous working copy were discarded.",
                "warning",
            ),
            False,
            False,
        )

    if trigger_prop == "capacity-page-capacity-scenario-switch-confirm.cancel_n_clicks":
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            None,
            current_selected,
            _build_capacity_scenario_message("Scenario switch cancelled.", "neutral"),
            False,
            False,
        )

    if trigger_prop == "capacity-page-capacity-scenario-create-button.n_clicks":
        try:
            normalized_base_type = (create_base_type or "").strip().casefold()
            normalized_source_value = pd.to_numeric(create_source_scenario_id, errors="coerce")
            source_scenario_id = int(normalized_source_value) if pd.notna(normalized_source_value) else None

            if normalized_base_type == "current_scenario":
                if current_selected is None:
                    raise ValueError("Select a current scenario before creating from Current Scenario.")
                source_rows_df = working_rows_df
                create_base_type_value = "internal_scenario"
                create_source_scenario_id_value = current_selected
            else:
                source_rows_df = _build_new_capacity_scenario_rows(
                    normalized_base_type,
                    _deserialize_train_capacity_store(train_capacity_data),
                    _deserialize_ea_capacity_store(ea_capacity_data),
                    source_scenario_id,
                    aggregate_from_date=selected_start_date,
                )
                create_base_type_value = normalized_base_type
                create_source_scenario_id_value = source_scenario_id

            source_rows_df = _attach_and_validate_capacity_scenario_keys(
                source_rows_df,
                normalized_base_type,
            )

            create_result = create_capacity_scenario_from_source(
                base_type=create_base_type_value,
                source_scenario_id=create_source_scenario_id_value,
                scenario_name=create_name,
                engine=engine,
                source_rows_df=source_rows_df,
            )
            created_rows_df = fetch_capacity_scenario_rows(create_result["scenario_id"], engine)
            refresh_value = dt.datetime.utcnow().isoformat()
            snapshot_revision = _serialize_timestamp(
                create_result["upload_timestamp_utc"]
            )
            _cache_saved_capacity_scenario_rows(
                create_result["scenario_id"],
                snapshot_revision,
                created_rows_df,
            )
            return (
                create_result["scenario_id"],
                _build_capacity_scenario_working_store(
                    create_result["scenario_id"],
                    scenario_options_data,
                    base_revision=snapshot_revision,
                ),
                {"dirty": False},
                refresh_value,
                None,
                create_result["scenario_id"],
                _build_capacity_scenario_message(
                    f"Created internal scenario '{create_result['scenario_name']}' with {create_result['row_count']:,} rows. Generate the ramp forecast when the scenario is ready.",
                    "success",
                ),
                False,
                False,
            )
        except Exception as exc:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                _build_capacity_scenario_message(f"Scenario creation failed: {exc}", "error"),
                False,
                False,
            )

    if trigger_prop == "capacity-page-capacity-scenario-save-button.n_clicks":
        if current_selected is None:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                _build_capacity_scenario_message(
                    "Select an internal scenario before saving.",
                    "warning",
                ),
                False,
                False,
            )
        try:
            updated_rows_df = working_rows_df
            updated_rows_df = attach_registry_keys_to_capacity_rows(updated_rows_df, engine)
            expected_snapshot = (
                current_working_store.get("base_revision")
                if isinstance(current_working_store, dict)
                else None
            )
            save_result = save_capacity_scenario_rows(
                current_selected,
                updated_rows_df,
                engine,
                updated_by="capacity_page",
                expected_snapshot_timestamp_utc=expected_snapshot,
            )
            refresh_value = dt.datetime.utcnow().isoformat()
            snapshot_revision = _serialize_timestamp(
                save_result["upload_timestamp_utc"]
            )
            _cache_saved_capacity_scenario_rows(
                current_selected,
                snapshot_revision,
                updated_rows_df,
            )
            return (
                current_selected,
                _build_capacity_scenario_working_store(
                    current_selected,
                    scenario_options_data,
                    base_revision=snapshot_revision,
                ),
                {"dirty": False},
                refresh_value,
                None,
                no_update,
                _build_capacity_scenario_message(
                    "Scenario saved successfully. The ramp forecast is now stale; generate a new run before publishing.",
                    "success",
                ),
                False,
                False,
            )
        except Exception as exc:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                _build_capacity_scenario_message(f"Scenario save failed: {exc}", "error"),
                False,
                False,
            )

    if trigger_prop == "capacity-page-capacity-scenario-revert-button.n_clicks":
        if current_selected is None:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                _build_capacity_scenario_message(
                    "Select an internal scenario before reverting.",
                    "warning",
                ),
                False,
                False,
            )
        try:
            reverted_rows_df = fetch_capacity_scenario_rows(
                current_selected,
                engine,
            )
            if (
                reverted_rows_df.empty
                or "upload_timestamp_utc" not in reverted_rows_df.columns
            ):
                raise RuntimeError("The selected scenario has no current saved snapshot.")
            base_revision = _serialize_timestamp(
                reverted_rows_df["upload_timestamp_utc"].iloc[0]
            )
            _cache_saved_capacity_scenario_rows(
                current_selected,
                base_revision,
                reverted_rows_df,
            )
        except Exception as exc:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                current_selected,
                _build_capacity_scenario_message(
                    f"Scenario revert failed: {exc}",
                    "error",
                ),
                False,
                False,
            )
        return (
            current_selected,
            _build_capacity_scenario_working_store(
                current_selected,
                scenario_options_data,
                base_revision=base_revision,
            ),
            {"dirty": False},
            dt.datetime.utcnow().isoformat(),
            None,
            no_update,
            _build_capacity_scenario_message(
                "Working copy reverted to the last saved scenario state.",
                "success",
            ),
            False,
            False,
        )

    if trigger_prop == "capacity-page-capacity-scenario-delete-button.n_clicks":
        if current_selected is None:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                _build_capacity_scenario_message(
                    "Select an internal scenario before deleting it.",
                    "warning",
                ),
                False,
                False,
            )
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            False,
            True,
        )

    if trigger_prop == "capacity-page-capacity-scenario-delete-confirm.submit_n_clicks":
        if current_selected is None:
            raise PreventUpdate
        try:
            delete_result = delete_capacity_scenario(current_selected, engine)
            delete_message = (
                f"Deleted internal scenario '{delete_result['scenario_name']}'. "
                f"Removed {delete_result['snapshots_deleted']:,} saved Capacity snapshot(s). "
                "Historical ramp runs remain detached for audit."
            )
            delete_tone = "success"
            refresh_value = dt.datetime.utcnow().isoformat()
            return (
                None,
                None,
                {"dirty": False},
                refresh_value,
                None,
                None,
                _build_capacity_scenario_message(
                    delete_message,
                    delete_tone,
                ),
                False,
                False,
            )
        except Exception as exc:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                _build_capacity_scenario_message(f"Scenario deletion failed: {exc}", "error"),
                False,
                False,
            )

    if trigger_prop == "capacity-page-capacity-scenario-delete-confirm.cancel_n_clicks":
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            False,
            False,
        )

    raise PreventUpdate


@callback(
    Output("capacity-page-capacity-scenario-working-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-dirty-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-refresh-store", "data", allow_duplicate=True),
    Input("capacity-page-train-timeline-table", "cellValueChanged"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-working-store", "data"),
    State("capacity-page-capacity-scenario-options-store", "data"),
    prevent_initial_call=True,
)
def mark_capacity_scenario_dirty(
    _cell_value_changed,
    selected_scenario_id,
    current_working_store,
    scenario_options_data,
):
    selected_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_value):
        raise PreventUpdate

    events = _cell_value_changed if isinstance(_cell_value_changed, list) else [_cell_value_changed]
    edited_rows = [
        event.get("data")
        for event in events
        if _is_user_scenario_cell_edit(event) and isinstance(event.get("data"), dict)
    ]
    if not edited_rows:
        raise PreventUpdate

    active_rows_df = _resolve_active_capacity_scenario_rows(current_working_store)
    updated_rows_df = _update_working_scenario_rows_from_grid(
        active_rows_df,
        edited_rows,
    )
    refresh_value = dt.datetime.utcnow().isoformat()
    return (
        _build_capacity_scenario_delta_store(
            int(selected_value),
            updated_rows_df,
            scenario_options_data,
        ),
        {"dirty": True, "source": "working", "updated_at": refresh_value},
        refresh_value,
    )


@callback(
    Output("capacity-page-capacity-scenario-working-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-dirty-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-refresh-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-message", "children", allow_duplicate=True),
    Output("capacity-page-scenario-add-country", "value"),
    Output("capacity-page-scenario-add-plant", "value"),
    Output("capacity-page-scenario-add-train", "value"),
    Output("capacity-page-scenario-add-first-date", "value"),
    Output("capacity-page-scenario-add-capacity", "value"),
    Output("capacity-page-scenario-add-seasonal-profile", "value"),
    Output("capacity-page-scenario-add-ramp-category", "value"),
    Input("capacity-page-scenario-add-row-button", "n_clicks"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-working-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    State("capacity-page-scenario-add-country", "value"),
    State("capacity-page-scenario-add-plant", "value"),
    State("capacity-page-scenario-add-train", "value"),
    State("capacity-page-scenario-add-first-date", "value"),
    State("capacity-page-scenario-add-capacity", "value"),
    State("capacity-page-scenario-add-seasonal-profile", "value"),
    State("capacity-page-scenario-add-ramp-category", "value"),
    running=[
        (
            Output(
                "capacity-page-scenario-command-running-store",
                "data",
                allow_duplicate=True,
            ),
            True,
            False,
        ),
    ],
    prevent_initial_call=True,
)
def add_capacity_scenario_row(
    n_clicks,
    selected_scenario_id,
    current_working_store,
    current_dirty_store,
    country_name,
    plant_name,
    train_label,
    first_date,
    capacity_value,
    seasonal_profile,
    ramp_category,
):
    if not n_clicks:
        raise PreventUpdate

    selected_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_value):
        return (
            no_update,
            no_update,
            no_update,
            _build_capacity_scenario_message(
                "Select an internal scenario before adding a row.",
                "warning",
            ),
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    active_rows_df = _resolve_active_capacity_scenario_rows(
        current_working_store,
        current_dirty_store,
    )

    try:
        pending_rows_df, pending_row_key = _append_manual_capacity_scenario_row(
            active_rows_df,
            "pending-terminal-key",
            "pending-train-key",
            country_name,
            plant_name,
            train_label,
            first_date,
            capacity_value,
        )
        candidates = find_terminal_train_candidates(
            engine,
            country_name,
            plant_name,
            train_label,
        )
        if len(candidates) > 1:
            raise ValueError(
                "This terminal/train matches multiple canonical records; resolve the registry mapping first."
            )
        if candidates:
            registry_train = candidates[0]
            registry_action = "Linked the row to the existing canonical train."
        else:
            if not ramp_category:
                raise ValueError("Ramp category is required when creating a new canonical train.")
            if not seasonal_profile:
                raise ValueError(
                    "Seasonal profile is required when creating a new canonical train."
                )
            registry_train = register_terminal_train(
                engine,
                country_name=str(country_name or ""),
                terminal_name=str(plant_name or ""),
                train_label=str(train_label or ""),
                ramp_profile_id=str(ramp_category),
                seasonal_profile_id=str(seasonal_profile),
            )
            registry_action = "Created a new canonical train without WoodMac identifiers."
        pending_mask = pending_rows_df["scenario_row_key"].eq(pending_row_key)
        pending_rows_df.loc[pending_mask, "terminal_key"] = str(registry_train["terminal_key"])
        pending_rows_df.loc[pending_mask, "train_key"] = str(registry_train["train_key"])
        updated_rows_df = _prepare_capacity_scenario_rows_df(pending_rows_df)
    except Exception as exc:
        return (
            no_update,
            no_update,
            no_update,
            _build_capacity_scenario_message(f"Unable to add row: {exc}", "error"),
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    return (
        _build_capacity_scenario_delta_store(
            int(selected_value),
            updated_rows_df,
            base_revision=(
                str(current_working_store.get("base_revision"))
                if isinstance(current_working_store, dict)
                and current_working_store.get("base_revision")
                else None
            ),
        ),
        {"dirty": True, "source": "working", "updated_at": dt.datetime.utcnow().isoformat()},
        dt.datetime.utcnow().isoformat(),
        _build_capacity_scenario_message(
            f"Added a new internal scenario row to the working copy. {registry_action}",
            "success",
        ),
        None,
        "",
        None,
        "",
        None,
        None,
        None,
    )


@callback(
    Output("capacity-page-capacity-scenario-working-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-dirty-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-refresh-store", "data", allow_duplicate=True),
    Output("capacity-page-capacity-scenario-message", "children", allow_duplicate=True),
    Input("capacity-page-capacity-scenario-upload", "contents"),
    State("capacity-page-capacity-scenario-upload", "filename"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-working-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    running=[
        (
            Output(
                "capacity-page-scenario-command-running-store",
                "data",
                allow_duplicate=True,
            ),
            True,
            False,
        ),
    ],
    prevent_initial_call=True,
)
def upload_train_timeline_scenario_workbook(
    upload_contents,
    filename,
    selected_scenario_id,
    current_working_store,
    current_dirty_store,
):
    if not upload_contents:
        raise PreventUpdate

    selected_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_value):
        return (
            no_update,
            no_update,
            no_update,
            _build_capacity_scenario_message(
                "Select an internal scenario before uploading a Train Timeline workbook.",
                "warning",
            ),
        )

    dirty_payload = current_dirty_store if isinstance(current_dirty_store, dict) else {"dirty": False}
    if dirty_payload.get("dirty"):
        return (
            no_update,
            no_update,
            no_update,
            _build_capacity_scenario_message(
                "Save or Revert the current unsaved edits before uploading a Train Timeline workbook.",
                "warning",
            ),
        )

    if not filename or not str(filename).lower().endswith(".xlsx"):
        return (
            no_update,
            no_update,
            no_update,
            _build_capacity_scenario_message(
                "Upload a .xlsx workbook exported from Train Timeline - Monthly.",
                "error",
            ),
        )

    try:
        _content_type, content_string = upload_contents.split(",", 1)
        workbook_bytes = base64.b64decode(content_string)
        excel_file = pd.ExcelFile(BytesIO(workbook_bytes))
        if TRAIN_TIMELINE_SHEET_NAME not in excel_file.sheet_names:
            raise ValueError(
                f"Missing worksheet '{TRAIN_TIMELINE_SHEET_NAME}'. Export a fresh Train Timeline workbook and try again."
            )
        if TRAIN_TIMELINE_IMPORT_META_SHEET_NAME not in excel_file.sheet_names:
            raise ValueError(
                f"Missing hidden metadata sheet '{TRAIN_TIMELINE_IMPORT_META_SHEET_NAME}'. Export a fresh Train Timeline workbook and try again."
            )

        main_df = pd.read_excel(
            excel_file,
            sheet_name=TRAIN_TIMELINE_SHEET_NAME,
            dtype=object,
        )
        metadata_df = pd.read_excel(
            excel_file,
            sheet_name=TRAIN_TIMELINE_IMPORT_META_SHEET_NAME,
            dtype=object,
        )
        updated_rows_df, summary = _build_train_timeline_upload_rows_df(
            main_df,
            metadata_df,
            int(selected_value),
            _resolve_active_capacity_scenario_rows(current_working_store),
        )
    except Exception as exc:
        return (
            no_update,
            no_update,
            no_update,
            _build_capacity_scenario_message(str(exc), "error"),
        )

    if summary["updated"] == 0 and summary["added"] == 0 and summary["deleted"] == 0:
        return (
            no_update,
            no_update,
            no_update,
            _build_capacity_scenario_message(
                "The uploaded workbook matches the current working copy. No changes were staged.",
                "neutral",
            ),
        )

    refresh_value = dt.datetime.utcnow().isoformat()
    return (
        _build_capacity_scenario_delta_store(
            int(selected_value),
            updated_rows_df,
            base_revision=(
                str(current_working_store.get("base_revision"))
                if isinstance(current_working_store, dict)
                and current_working_store.get("base_revision")
                else None
            ),
        ),
        {"dirty": True, "source": "working", "updated_at": refresh_value},
        refresh_value,
        _build_capacity_scenario_message(
            f"Imported {summary['updated']:,} update(s), {summary['added']:,} addition(s), and {summary['deleted']:,} deletion(s) into the working copy. Review the grid and click Save to persist.",
            "success",
        ),
    )


@callback(
    Output("capacity-page-yearly-capacity-comparison-chart", "figure"),
    Output("capacity-page-yearly-capacity-comparison-table-container", "children"),
    Input("capacity-page-scenario-render-revision-store", "data"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
    State("capacity-page-woodmac-data-store", "data"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-metadata-store", "data"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-capacity-scenario-working-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    State("capacity-page-capacity-scenario-refresh-store", "data"),
)
def render_yearly_capacity_comparison_section(
    scenario_revision,
    start_date,
    end_date,
    woodmac_data,
    train_capacity_data,
    ea_capacity_data,
    metadata,
    selected_scenario_id,
    scenario_options_data,
    working_store_data,
    dirty_store,
    scenario_refresh_revision,
    timeline_row_data=None,
):
    if not _capacity_scenario_revision_is_coherent(
        scenario_revision,
        selected_scenario_id,
        working_store_data,
        dirty_store,
        scenario_refresh_revision,
        scenario_options_data,
    ) or not _capacity_scenario_source_refs_are_coherent(
        scenario_revision,
        woodmac_data,
        train_capacity_data,
        ea_capacity_data,
        metadata,
    ):
        raise PreventUpdate
    selected_scenario_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_scenario_value):
        return (
            _create_empty_capacity_figure(YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE),
            _create_empty_state(YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE),
        )

    woodmac_total_df = _deserialize_woodmac_capacity_store(
        woodmac_data,
        start_date=start_date,
        end_date=end_date,
    )
    ea_raw_df = _deserialize_ea_capacity_store(ea_capacity_data)
    active_rows_df = _resolve_active_capacity_scenario_rows(
        working_store_data,
        dirty_store,
        timeline_row_data,
    )
    comparison_df = _get_cached_capacity_view_payload(
        "yearly_capacity_comparison",
        {
            "woodmac": _get_capacity_payload_signature(woodmac_data),
            "ea": _get_capacity_payload_signature(ea_capacity_data),
            "scenario": hashlib.sha256(
                str(_serialize_dataframe(active_rows_df) or "").encode("utf-8")
            ).hexdigest(),
            "start_date": start_date,
            "end_date": end_date,
        },
        lambda: _build_yearly_capacity_comparison_df(
            woodmac_total_df,
            ea_raw_df,
            active_rows_df,
            start_date,
            end_date,
        ),
    )
    empty_message = (
        "No overlapping yearly Woodmac and Energy Aspects values are available for the current selection."
    )

    return (
        (
            _create_yearly_capacity_comparison_chart(comparison_df)
            if not comparison_df.empty
            else _create_empty_capacity_figure(empty_message)
        ),
        (
            _create_yearly_capacity_comparison_table(
                "capacity-page-yearly-capacity-comparison-table",
                comparison_df,
            )
            if not comparison_df.empty
            else _create_empty_state(empty_message)
        ),
    )


@callback(
    Output("capacity-page-yearly-woodmac-capacity-discrepancy-table-container", "children"),
    Output("capacity-page-yearly-woodmac-timeline-discrepancy-table-container", "children"),
    Output("capacity-page-yearly-woodmac-missing-internal-table-container", "children"),
    Output("capacity-page-yearly-ea-capacity-discrepancy-table-container", "children"),
    Output("capacity-page-yearly-ea-timeline-discrepancy-table-container", "children"),
    Output("capacity-page-yearly-ea-missing-internal-table-container", "children"),
    Input("capacity-page-scenario-render-revision-store", "data"),
    Input("capacity-page-country-dropdown", "value"),
    Input("capacity-page-other-country-mode", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
    State("capacity-page-woodmac-data-store", "data"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-metadata-store", "data"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-capacity-scenario-working-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    State("capacity-page-capacity-scenario-refresh-store", "data"),
)
def render_yearly_capacity_discrepancy_section(
    scenario_revision,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    woodmac_data,
    train_capacity_data,
    ea_capacity_data,
    metadata,
    selected_scenario_id,
    scenario_options_data,
    working_store_data,
    dirty_store,
    scenario_refresh_revision,
    timeline_row_data=None,
):
    if not _capacity_scenario_revision_is_coherent(
        scenario_revision,
        selected_scenario_id,
        working_store_data,
        dirty_store,
        scenario_refresh_revision,
        scenario_options_data,
    ) or not _capacity_scenario_source_refs_are_coherent(
        scenario_revision,
        woodmac_data,
        train_capacity_data,
        ea_capacity_data,
        metadata,
    ):
        raise PreventUpdate
    selected_scenario_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_scenario_value):
        return (
            _create_empty_state(YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE),
            _create_empty_state(YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE),
            _create_empty_state(YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE),
            _create_empty_state(YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE),
            _create_empty_state(YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE),
            _create_empty_state(YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE),
        )

    active_rows_df = _resolve_active_capacity_scenario_rows(
        working_store_data,
        dirty_store,
        timeline_row_data,
    )
    prepared_provider_data = _get_prepared_capacity_provider_data(
        train_capacity_data,
        ea_capacity_data,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
        scenario_rows_df=active_rows_df,
    )
    discrepancy_payloads = _get_cached_capacity_view_payload(
        "yearly_provider_discrepancies",
        {
            "prepared": _build_capacity_derived_cache_key(
                train_capacity_data,
                ea_capacity_data,
                selected_countries,
                other_countries_mode,
                start_date,
                end_date,
                active_rows_df,
            )
        },
        lambda: _build_yearly_provider_discrepancy_payloads(
            prepared_provider_data["train_raw_df"],
            prepared_provider_data["ea_raw_df"],
            active_rows_df,
            selected_countries,
            other_countries_mode,
            start_date,
            end_date,
            prepared_provider_data=prepared_provider_data,
        ),
    )
    woodmac_payload = discrepancy_payloads["woodmac"]
    ea_payload = discrepancy_payloads["energy_aspects"]

    return (
        _create_provider_capacity_discrepancy_table(
            "capacity-page-yearly-woodmac-capacity-discrepancy-table",
            woodmac_payload["capacity_df"],
            "woodmac",
        ),
        _create_provider_timeline_discrepancy_table(
            "capacity-page-yearly-woodmac-timeline-discrepancy-table",
            woodmac_payload["timeline_df"],
            "woodmac",
        ),
        _create_provider_missing_internal_scenario_table(
            "capacity-page-yearly-woodmac-missing-internal-table",
            woodmac_payload["missing_df"],
            "woodmac",
        ),
        _create_provider_capacity_discrepancy_table(
            "capacity-page-yearly-ea-capacity-discrepancy-table",
            ea_payload["capacity_df"],
            "energy_aspects",
        ),
        _create_provider_timeline_discrepancy_table(
            "capacity-page-yearly-ea-timeline-discrepancy-table",
            ea_payload["timeline_df"],
            "energy_aspects",
        ),
        _create_provider_missing_internal_scenario_table(
            "capacity-page-yearly-ea-missing-internal-table",
            ea_payload["missing_df"],
            "energy_aspects",
        ),
    )


@callback(
    Output("capacity-page-woodmac-summary", "children"),
    Output("capacity-page-woodmac-chart", "figure"),
    Output("capacity-page-woodmac-table-container", "children"),
    Input("capacity-page-source-render-revision-store", "data"),
    Input("capacity-page-country-dropdown", "value"),
    Input("capacity-page-other-country-mode", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
    Input("capacity-page-train-change-time-view", "value"),
    Input("capacity-page-top-table-view", "value"),
    State("capacity-page-woodmac-data-store", "data"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-metadata-store", "data"),
)
def render_capacity_table(
    source_revision,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    table_view,
    woodmac_data,
    train_capacity_data,
    ea_capacity_data,
    metadata,
):
    if not _capacity_source_revision_is_coherent(
        source_revision,
        woodmac_data,
        train_capacity_data,
        ea_capacity_data,
        metadata,
    ):
        raise PreventUpdate
    prepared_view = _get_prepared_country_capacity_view(
        "woodmac",
        woodmac_data,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
        time_view,
    )
    woodmac_raw_df = prepared_view["source_df"]
    resolved_countries = prepared_view["resolved_countries"]
    woodmac_matrix = prepared_view["matrix_df"]

    woodmac_summary = _build_section_summary(
        woodmac_raw_df,
        _build_capacity_metadata_lines(metadata),
    )

    if resolved_countries == [] and other_countries_mode == "exclude":
        empty_message = _create_empty_state(
            "Select at least one country or switch to Rest of the World mode."
        )
        return (
            woodmac_summary,
            _create_empty_capacity_figure(empty_message.children),
            empty_message,
        )

    woodmac_chart = _create_capacity_country_area_chart(
        woodmac_matrix,
        time_view=time_view,
    )
    woodmac_table = _create_capacity_table(
        "capacity-page-woodmac-table",
        _apply_capacity_table_view(woodmac_matrix, table_view),
        table_view=table_view,
    )
    return woodmac_summary, woodmac_chart, woodmac_table


@callback(
    Output("capacity-page-ea-summary", "children"),
    Output("capacity-page-ea-chart", "figure"),
    Output("capacity-page-ea-table-container", "children"),
    Input("capacity-page-source-render-revision-store", "data"),
    Input("capacity-page-country-dropdown", "value"),
    Input("capacity-page-other-country-mode", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
    Input("capacity-page-train-change-time-view", "value"),
    Input("capacity-page-top-table-view", "value"),
    State("capacity-page-woodmac-data-store", "data"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-metadata-store", "data"),
)
def render_ea_capacity_table(
    source_revision,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    table_view,
    woodmac_data,
    train_capacity_data,
    ea_capacity_data,
    metadata,
):
    if not _capacity_source_revision_is_coherent(
        source_revision,
        woodmac_data,
        train_capacity_data,
        ea_capacity_data,
        metadata,
    ):
        raise PreventUpdate
    prepared_view = _get_prepared_country_capacity_view(
        "energy_aspects",
        ea_capacity_data,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
        time_view,
    )
    ea_schedule_df = prepared_view["source_df"]
    resolved_countries = prepared_view["resolved_countries"]
    ea_matrix = prepared_view["matrix_df"]

    ea_summary = _build_section_summary(
        ea_schedule_df,
        _build_ea_capacity_metadata_lines(metadata),
    )

    if resolved_countries == [] and other_countries_mode == "exclude":
        empty_message = _create_empty_state(
            "Select at least one country or switch to Rest of the World mode."
        )
        return (
            ea_summary,
            _create_empty_capacity_figure(empty_message.children),
            empty_message,
        )

    ea_chart = _create_capacity_country_area_chart(
        ea_matrix,
        y_axis_title="MTPA",
        time_view=time_view,
    )
    ea_table = _create_capacity_table(
        "capacity-page-ea-table",
        _apply_capacity_table_view(ea_matrix, table_view),
        table_view=table_view,
    )
    return ea_summary, ea_chart, ea_table


@callback(
    Output("capacity-page-internal-scenario-summary", "children"),
    Output("capacity-page-internal-scenario-chart", "figure"),
    Output("capacity-page-internal-scenario-table-container", "children"),
    Input("capacity-page-scenario-render-revision-store", "data"),
    Input("capacity-page-country-dropdown", "value"),
    Input("capacity-page-other-country-mode", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
    Input("capacity-page-train-change-time-view", "value"),
    Input("capacity-page-top-table-view", "value"),
    State("capacity-page-woodmac-data-store", "data"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-metadata-store", "data"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-capacity-scenario-working-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    State("capacity-page-capacity-scenario-refresh-store", "data"),
)
def render_internal_capacity_table(
    scenario_revision,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    table_view,
    woodmac_data,
    train_capacity_data,
    ea_capacity_data,
    metadata,
    selected_scenario_id,
    scenario_options_data,
    working_store_data,
    dirty_store,
    scenario_refresh_revision,
    timeline_row_data=None,
):
    if not _capacity_scenario_revision_is_coherent(
        scenario_revision,
        selected_scenario_id,
        working_store_data,
        dirty_store,
        scenario_refresh_revision,
        scenario_options_data,
    ) or not _capacity_scenario_source_refs_are_coherent(
        scenario_revision,
        woodmac_data,
        train_capacity_data,
        ea_capacity_data,
        metadata,
    ):
        raise PreventUpdate
    selected_scenario_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_scenario_value):
        empty_message = _create_empty_state(INTERNAL_SCENARIO_EMPTY_MESSAGE)
        return (
            _build_section_summary(pd.DataFrame(), [INTERNAL_SCENARIO_EMPTY_MESSAGE]),
            _create_empty_capacity_figure(INTERNAL_SCENARIO_EMPTY_MESSAGE),
            empty_message,
        )

    prepared_view = _get_prepared_internal_capacity_view(
        working_store_data,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
        time_view,
    )
    internal_schedule_df = prepared_view["schedule_df"]
    option_map = _get_capacity_scenario_option_map(scenario_options_data)
    selected_option = option_map.get(int(selected_scenario_value), {})
    metadata_lines = [
        f"Scenario: {selected_option.get('scenario_name', 'Internal Scenario')}. Monthly schedule is derived from scenario first dates and scenario capacities."
    ]
    internal_summary = _build_section_summary(internal_schedule_df, metadata_lines)

    if internal_schedule_df.empty:
        empty_message = _create_empty_state("No internal scenario capacity falls within the current selection.")
        return (
            internal_summary,
            _create_empty_capacity_figure("No internal scenario capacity falls within the current selection."),
            empty_message,
        )

    resolved_countries = prepared_view["resolved_countries"]
    if resolved_countries == [] and other_countries_mode == "exclude":
        empty_message = _create_empty_state(
            "Select at least one country or switch to Rest of the World mode."
        )
        return (
            internal_summary,
            _create_empty_capacity_figure(empty_message.children),
            empty_message,
        )

    internal_matrix = prepared_view["matrix_df"]
    internal_chart = _create_capacity_country_area_chart(
        internal_matrix,
        y_axis_title="MTPA",
        time_view=time_view,
    )
    internal_table = _create_capacity_table(
        "capacity-page-internal-scenario-table",
        _apply_capacity_table_view(internal_matrix, table_view),
        table_view=table_view,
    )
    return internal_summary, internal_chart, internal_table


@callback(
    Output("capacity-page-train-change-summary", "children"),
    Output("capacity-page-train-change-table-container", "children"),
    Input("capacity-page-scenario-render-revision-store", "data"),
    Input("capacity-page-country-dropdown", "value"),
    Input("capacity-page-other-country-mode", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
    Input("capacity-page-train-change-time-view", "value"),
    Input("capacity-page-train-change-view-mode", "value"),
    State("capacity-page-woodmac-data-store", "data"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-metadata-store", "data"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-capacity-scenario-working-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    State("capacity-page-capacity-scenario-refresh-store", "data"),
)
def render_train_capacity_change_table(
    scenario_revision,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    detail_view,
    woodmac_data,
    train_capacity_data,
    ea_capacity_data,
    metadata,
    selected_scenario_id,
    scenario_options_data,
    working_scenario_data,
    dirty_store,
    scenario_refresh_revision,
    timeline_row_data=None,
):
    if not _capacity_scenario_revision_is_coherent(
        scenario_revision,
        selected_scenario_id,
        working_scenario_data,
        dirty_store,
        scenario_refresh_revision,
        scenario_options_data,
    ) or not _capacity_scenario_source_refs_are_coherent(
        scenario_revision,
        woodmac_data,
        train_capacity_data,
        ea_capacity_data,
        metadata,
    ):
        raise PreventUpdate
    working_rows_df = _resolve_active_capacity_scenario_rows(
        working_scenario_data,
        dirty_store,
        timeline_row_data,
    )
    prepared_provider_data = _get_prepared_capacity_provider_data(
        train_capacity_data,
        ea_capacity_data,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
        scenario_rows_df=working_rows_df,
    )
    train_raw_df = prepared_provider_data["train_raw_df"]
    ea_raw_df = prepared_provider_data["ea_raw_df"]

    if train_raw_df.empty and ea_raw_df.empty and working_rows_df.empty:
        empty_df = pd.DataFrame()
        return (
            _build_train_change_summary(
                empty_df,
                empty_df,
                time_view=time_view,
                detail_view=detail_view,
            ),
            _create_train_change_table(empty_df),
        )

    resolved_countries = prepared_provider_data["resolved_countries"]
    woodmac_change_df = prepared_provider_data["woodmac_change_df"]
    ea_change_df = prepared_provider_data["ea_change_df"]
    internal_change_df = _build_internal_scenario_change_log(
        working_rows_df,
        resolved_countries,
        other_countries_mode,
        start_date,
        end_date,
    )
    hierarchical_df = _build_train_change_hierarchical_rows(
        woodmac_change_df,
        ea_change_df,
        internal_change_df=internal_change_df,
        time_view=time_view,
        detail_view=detail_view,
    )
    selected_option = _get_capacity_scenario_option_map(scenario_options_data).get(
        int(selected_scenario_id)
    ) if pd.notna(pd.to_numeric(selected_scenario_id, errors="coerce")) else None
    internal_label = selected_option.get("scenario_name") if selected_option else None

    return (
        _build_train_change_summary(
            woodmac_change_df,
            ea_change_df,
            time_view=time_view,
            detail_view=detail_view,
        ),
        _create_train_change_table(
            hierarchical_df,
            internal_scenario_label=internal_label,
        ),
    )


@callback(
    Output("capacity-page-train-timeline-table-container", "children"),
    Input("capacity-page-scenario-render-revision-store", "data"),
    Input("capacity-page-country-dropdown", "value"),
    Input("capacity-page-other-country-mode", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
    Input("capacity-page-train-timeline-original-name-visibility", "value"),
    State("capacity-page-woodmac-data-store", "data"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-metadata-store", "data"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-capacity-scenario-working-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    State("capacity-page-capacity-scenario-refresh-store", "data"),
)
def render_train_timeline_table(
    scenario_revision,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    original_name_visibility,
    woodmac_data,
    train_capacity_data,
    ea_capacity_data,
    metadata,
    selected_scenario_id,
    scenario_options_data,
    working_scenario_data,
    dirty_store,
    scenario_refresh_revision,
    current_timeline_row_data=None,
):
    if not _capacity_scenario_revision_is_coherent(
        scenario_revision,
        selected_scenario_id,
        working_scenario_data,
        dirty_store,
        scenario_refresh_revision,
        scenario_options_data,
    ) or not _capacity_scenario_source_refs_are_coherent(
        scenario_revision,
        woodmac_data,
        train_capacity_data,
        ea_capacity_data,
        metadata,
    ):
        raise PreventUpdate
    show_original_names = original_name_visibility == "show"
    selected_scenario_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    selected_scenario = int(selected_scenario_value) if pd.notna(selected_scenario_value) else None
    scenario_rows_df = _resolve_active_capacity_scenario_rows(
        working_scenario_data,
        dirty_store,
        current_timeline_row_data,
    )
    prepared_provider_data = _get_prepared_capacity_provider_data(
        train_capacity_data,
        ea_capacity_data,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
        scenario_rows_df=scenario_rows_df,
    )
    resolved_countries = prepared_provider_data["resolved_countries"]
    timeline_df = prepared_provider_data["timeline_df"]
    visible_scenario_rows_df = _filter_visible_capacity_scenario_rows(
        scenario_rows_df,
        resolved_countries,
        other_countries_mode,
        start_date,
        end_date,
    ) if selected_scenario is not None else pd.DataFrame()
    woodmac_lookup_df = prepared_provider_data["woodmac_lookup_df"]
    ea_lookup_df = prepared_provider_data["ea_lookup_df"]
    scenario_lookup_df = (
        _build_internal_scenario_lookup_snapshot(
            scenario_rows_df,
            resolved_countries,
            other_countries_mode,
            start_date,
            end_date,
        )
        if selected_scenario is not None
        else pd.DataFrame()
    )

    return _create_train_timeline_table(
        "capacity-page-train-timeline-table",
        timeline_df,
        scenario_rows_df=visible_scenario_rows_df,
        show_original_names=show_original_names,
        enable_editing=selected_scenario is not None,
        aggregate_from_date=start_date,
        woodmac_lookup_df=woodmac_lookup_df,
        ea_lookup_df=ea_lookup_df,
        scenario_lookup_df=scenario_lookup_df,
    )


@callback(
    Output("capacity-page-train-timeline-chart-source", "options"),
    Output("capacity-page-train-timeline-chart-source", "value"),
    Output("capacity-page-train-timeline-chart-compare", "options"),
    Output("capacity-page-train-timeline-chart-compare", "value"),
    Input("capacity-page-capacity-scenario-selected-store", "data"),
    Input("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-train-timeline-chart-source", "value"),
    State("capacity-page-train-timeline-chart-compare", "value"),
)
def sync_train_timeline_chart_controls(
    selected_scenario_id,
    scenario_options_data,
    current_source_value,
    current_compare_value,
):
    options = _get_train_timeline_chart_options(
        selected_scenario_id,
        scenario_options_data,
    )
    valid_values = {str(option.get("value") or "").strip().casefold() for option in options}
    if "internal_scenario" in valid_values:
        return options, "internal_scenario", options, "woodmac"

    source_value = _coerce_train_timeline_chart_value(
        current_source_value,
        valid_values,
        "woodmac",
    )
    compare_value = _coerce_train_timeline_chart_value(
        current_compare_value,
        valid_values,
        "energy_aspects",
    )
    return options, source_value, options, compare_value


@callback(
    Output("capacity-page-train-timeline-comparison-graph", "figure"),
    Input("capacity-page-train-timeline-chart-source", "value"),
    Input("capacity-page-train-timeline-chart-compare", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
    Input("capacity-page-train-timeline-table", "rowData"),
    State("capacity-page-train-timeline-chart-source", "options"),
)
def render_train_timeline_comparison_chart(
    source_value,
    compare_value,
    start_date,
    end_date,
    timeline_row_data,
    source_options,
):
    options = source_options or [
        {"label": "Woodmac", "value": "woodmac"},
        {"label": "Energy Aspects", "value": "energy_aspects"},
    ]
    valid_values = {str(option.get("value") or "").strip().casefold() for option in options}
    normalized_source = _coerce_train_timeline_chart_value(
        source_value,
        valid_values,
        "woodmac",
    )
    compare_fallback = "internal_scenario" if "internal_scenario" in valid_values else "energy_aspects"
    normalized_compare = _coerce_train_timeline_chart_value(
        compare_value,
        valid_values,
        compare_fallback,
    )
    source_label = _get_train_timeline_chart_option_label(options, normalized_source)
    compare_label = _get_train_timeline_chart_option_label(options, normalized_compare)

    signature_columns = {
        "Country",
        "Plant",
        "Train",
        "timeline_direction",
        "display_sort_plant",
        "display_sort_train",
    }
    for config_key in {normalized_source, normalized_compare}:
        config = TRAIN_TIMELINE_CHART_SOURCE_CONFIG.get(config_key) or {}
        signature_columns.update(
            value
            for value in (
                config.get("date_column"),
                config.get("capacity_column"),
                config.get("out_of_range_flag"),
            )
            if value
        )
    signature_rows = [
        {column: row.get(column) for column in sorted(signature_columns)}
        for row in (timeline_row_data or [])
    ]
    row_signature = hashlib.sha256(
        json.dumps(
            signature_rows,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return _get_cached_capacity_view_payload(
        "train_timeline_comparison_figure",
        {
            "rows": row_signature,
            "source": normalized_source,
            "compare": normalized_compare,
            "start": start_date,
            "end": end_date,
            "source_label": source_label,
            "compare_label": compare_label,
        },
        lambda: _create_train_timeline_comparison_figure(
            timeline_row_data,
            source_key=normalized_source,
            compare_key=normalized_compare,
            start_date=start_date,
            end_date=end_date,
            source_label=source_label,
            compare_label=compare_label,
        ),
    )


@callback(
    Output("capacity-page-unmapped-train-summary", "children"),
    Output("capacity-page-unmapped-train-table", "rowData"),
    Input("capacity-page-source-render-revision-store", "data"),
    State("capacity-page-woodmac-data-store", "data"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-metadata-store", "data"),
)
def render_unmapped_train_mapping_table(
    source_revision,
    woodmac_data,
    train_capacity_data,
    ea_capacity_data,
    metadata,
):
    if not _capacity_source_revision_is_coherent(
        source_revision,
        woodmac_data,
        train_capacity_data,
        ea_capacity_data,
        metadata,
    ):
        raise PreventUpdate
    train_raw_df = _deserialize_train_capacity_store(train_capacity_data)
    ea_raw_df = _deserialize_ea_capacity_store(ea_capacity_data)

    if train_raw_df.empty and ea_raw_df.empty:
        empty_df = pd.DataFrame()
        return (
            _build_unmapped_train_summary(empty_df),
            [],
        )

    train_alias_df = _build_unmapped_train_alias_df(train_raw_df, ea_raw_df)

    return (
        _build_unmapped_train_summary(train_alias_df),
        train_alias_df.to_dict("records"),
    )


def _clean_mapping_text_value(value: object) -> str | None:
    if pd.isna(value):
        return None

    text_value = str(value).strip()
    if not text_value:
        return None

    return " ".join(text_value.split())


def _clean_positive_train_value(value: object) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None

    numeric_value = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(numeric_value):
        return None

    numeric_value = float(numeric_value)
    if numeric_value <= 0 or not numeric_value.is_integer():
        return None

    return int(numeric_value)


def _clean_allocation_share_value(value: object, default: float = 1.0) -> float | None:
    if pd.isna(value) or str(value).strip() == "":
        return float(default)

    numeric_value = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(numeric_value) or float(numeric_value) <= 0:
        return None

    return round(float(numeric_value), 8)


@callback(
    Output("capacity-page-train-mapping-save-trigger", "data"),
    Output("capacity-page-unmapped-train-message", "children"),
    Input("capacity-page-save-train-mappings-button", "n_clicks"),
    State("capacity-page-unmapped-train-table", "rowData"),
    running=[
        (Output("capacity-page-save-train-mappings-button", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def save_unmapped_train_mappings(n_clicks, table_data):
    if not n_clicks:
        raise PreventUpdate

    alias_df = pd.DataFrame(table_data or [])
    if alias_df.empty:
        return no_update, html.Div(
            "No unmapped train aliases are currently available to save.",
            className="balance-metadata-row",
        )

    save_columns = [
        "country_name",
        "plant_name",
        "provider_name",
        "provider_plant_id",
        "provider_train_id",
        "original_plant",
        "original_train",
        "train",
        "allocation_share",
    ]
    for column_name in save_columns:
        if column_name not in alias_df.columns:
            alias_df[column_name] = None

    text_columns = [
        "country_name",
        "plant_name",
        "provider_name",
        "provider_plant_id",
        "provider_train_id",
        "original_plant",
        "original_train",
    ]
    for column_name in text_columns:
        alias_df[column_name] = alias_df[column_name].map(_clean_mapping_text_value)

    alias_df["train"] = alias_df["train"].map(_clean_mapping_text_value)
    alias_df["allocation_share"] = alias_df["allocation_share"].map(_clean_allocation_share_value)

    save_df = alias_df.dropna(
        subset=[
            "country_name",
            "plant_name",
            "provider_name",
            "provider_plant_id",
            "provider_train_id",
            "train",
            "allocation_share",
        ]
    ).copy()
    save_df = save_df[save_columns].drop_duplicates(
        subset=[
            "provider_name",
            "provider_plant_id",
            "provider_train_id",
            "train",
        ],
        keep="last",
    )

    if save_df.empty:
        return no_update, html.Div(
            "Fill at least one canonical train before saving.",
            style={"color": "#9a3412", "fontSize": "12px", "fontWeight": "600"},
        )

    try:
        resolved_rows = []
        unresolved = []
        for row in save_df.to_dict("records"):
            matches = find_terminal_train_candidates(
                engine,
                row["country_name"],
                row["plant_name"],
                row["train"],
            )
            if len(matches) != 1:
                unresolved.append(
                    f"{row['country_name']} / {row['plant_name']} / {row['train']}"
                )
                continue
            resolved_rows.append({**row, "train_key": str(matches[0]["train_key"])})
        if unresolved:
            raise ValueError(
                "Canonical train must resolve exactly once: " + "; ".join(unresolved[:5])
            )

        resolved_df = pd.DataFrame(resolved_rows)
        source_columns = ["provider_name", "provider_plant_id", "provider_train_id"]
        for source_key, source_rows in resolved_df.groupby(source_columns, sort=True):
            replace_provider_source_allocations(
                engine,
                provider_name=source_key[0],
                provider_plant_id=source_key[1],
                provider_train_id=source_key[2],
                allocations=source_rows[["train_key", "allocation_share"]].to_dict("records"),
            )

        save_timestamp = dt.datetime.utcnow().isoformat()
        source_count = resolved_df[source_columns].drop_duplicates().shape[0]
        return save_timestamp, html.Div(
            f"Saved {source_count:,} provider-source allocation(s) across {len(resolved_df):,} canonical train link(s). The comparison data has been reloaded.",
            style={"color": "#166534", "fontSize": "12px", "fontWeight": "600"},
        )
    except Exception as exc:
        return no_update, html.Div(
            f"Train mapping save failed: {exc}",
            style={"color": "#991b1b", "fontSize": "12px", "fontWeight": "600"},
        )


def _build_filtered_matrix_for_export(
    source_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view: str,
) -> pd.DataFrame:
    raw_df = _deserialize_woodmac_capacity_store(
        source_data,
        start_date=start_date,
        end_date=end_date,
        selected_countries=selected_countries,
    )
    return _apply_capacity_time_view(
        _rename_total_column(
            build_export_flow_matrix(
                raw_df,
                selected_countries,
                other_countries_mode,
            )
        ),
        time_view,
    )


def _build_filtered_ea_matrix_for_export(
    source_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view: str,
) -> pd.DataFrame:
    schedule_df = _build_ea_capacity_schedule(
        _deserialize_ea_capacity_store(source_data),
        start_date,
        end_date,
    )
    return _apply_capacity_time_view(
        _rename_total_column(
            build_export_flow_matrix(
                schedule_df,
                selected_countries,
                other_countries_mode,
            )
        ),
        time_view,
    )


def _build_filtered_internal_scenario_matrix_for_export(
    working_store_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view: str,
) -> pd.DataFrame:
    schedule_df = _get_cached_internal_scenario_monthly_schedule(
        _resolve_active_capacity_scenario_rows(working_store_data),
        start_date,
        end_date,
    )
    return _apply_capacity_time_view(
        _rename_total_column(
            build_export_flow_matrix(
                schedule_df,
                selected_countries,
                other_countries_mode,
            )
        ),
        time_view,
    )


@callback(
    Output("capacity-page-download-woodmac-excel", "data"),
    Input("capacity-page-export-woodmac-button", "n_clicks"),
    State("capacity-page-woodmac-data-store", "data"),
    State("capacity-page-country-dropdown", "value"),
    State("capacity-page-other-country-mode", "value"),
    State("capacity-page-date-range", "start_date"),
    State("capacity-page-date-range", "end_date"),
    State("capacity-page-train-change-time-view", "value"),
    State("capacity-page-top-table-view", "value"),
    running=[
        (Output("capacity-page-export-woodmac-button", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def export_woodmac_capacity_excel(
    n_clicks,
    woodmac_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    table_view,
):
    if not n_clicks:
        raise PreventUpdate

    export_df = _apply_capacity_table_view(
        _build_filtered_matrix_for_export(
            woodmac_data,
            selected_countries,
            other_countries_mode,
            start_date,
            end_date,
            time_view,
        ),
        table_view,
    ).drop(columns=["__axis_date"], errors="ignore")
    if export_df.empty:
        raise PreventUpdate

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"WoodMac_Nominal_Capacity_{timestamp}.xlsx"
    return dcc.send_bytes(
        _export_matrix_to_excel_bytes(export_df, "Nominal Capacity"),
        filename,
    )


@callback(
    Output("capacity-page-download-ea-excel", "data"),
    Input("capacity-page-export-ea-button", "n_clicks"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-country-dropdown", "value"),
    State("capacity-page-other-country-mode", "value"),
    State("capacity-page-date-range", "start_date"),
    State("capacity-page-date-range", "end_date"),
    State("capacity-page-train-change-time-view", "value"),
    State("capacity-page-top-table-view", "value"),
    running=[
        (Output("capacity-page-export-ea-button", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def export_ea_capacity_excel(
    n_clicks,
    ea_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    table_view,
):
    if not n_clicks:
        raise PreventUpdate

    export_df = _apply_capacity_table_view(
        _build_filtered_ea_matrix_for_export(
            ea_data,
            selected_countries,
            other_countries_mode,
            start_date,
            end_date,
            time_view,
        ),
        table_view,
    ).drop(columns=["__axis_date"], errors="ignore")
    if export_df.empty:
        raise PreventUpdate

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Energy_Aspects_Scheduled_Capacity_{timestamp}.xlsx"
    return dcc.send_bytes(
        _export_matrix_to_excel_bytes(export_df, "Scheduled Capacity"),
        filename,
    )


@callback(
    Output("capacity-page-download-internal-scenario-excel", "data"),
    Input("capacity-page-export-internal-scenario-button", "n_clicks"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-working-store", "data"),
    State("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-country-dropdown", "value"),
    State("capacity-page-other-country-mode", "value"),
    State("capacity-page-date-range", "start_date"),
    State("capacity-page-date-range", "end_date"),
    State("capacity-page-train-change-time-view", "value"),
    State("capacity-page-top-table-view", "value"),
    running=[
        (Output("capacity-page-export-internal-scenario-button", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def export_internal_scenario_capacity_excel(
    n_clicks,
    selected_scenario_id,
    working_scenario_data,
    scenario_options_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    table_view,
):
    if not n_clicks:
        raise PreventUpdate

    selected_scenario_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    if pd.isna(selected_scenario_value):
        raise PreventUpdate

    export_df = _apply_capacity_table_view(
        _build_filtered_internal_scenario_matrix_for_export(
            working_scenario_data,
            selected_countries,
            other_countries_mode,
            start_date,
            end_date,
            time_view,
        ),
        table_view,
    ).drop(columns=["__axis_date"], errors="ignore")
    if export_df.empty:
        raise PreventUpdate

    selected_option = _get_capacity_scenario_option_map(scenario_options_data).get(
        int(selected_scenario_value)
    )
    scenario_name = (selected_option or {}).get("scenario_name", "Internal_Scenario")
    safe_scenario_name = re.sub(r"[^A-Za-z0-9_]+", "_", scenario_name).strip("_") or "Internal_Scenario"
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_scenario_name}_Capacity_{timestamp}.xlsx"
    return dcc.send_bytes(
        _export_matrix_to_excel_bytes(export_df, "Internal Scenario"),
        filename,
    )


@callback(
    Output("capacity-page-download-train-change-excel", "data"),
    Input("capacity-page-export-train-change-button", "n_clicks"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-working-store", "data"),
    State("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-country-dropdown", "value"),
    State("capacity-page-other-country-mode", "value"),
    State("capacity-page-date-range", "start_date"),
    State("capacity-page-date-range", "end_date"),
    State("capacity-page-train-change-time-view", "value"),
    State("capacity-page-train-change-view-mode", "value"),
    running=[
        (Output("capacity-page-export-train-change-button", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def export_train_change_excel(
    n_clicks,
    train_capacity_data,
    ea_capacity_data,
    selected_scenario_id,
    working_scenario_data,
    scenario_options_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    detail_view,
):
    if not n_clicks:
        raise PreventUpdate

    train_raw_df = _deserialize_train_capacity_store(train_capacity_data)
    ea_raw_df = _deserialize_ea_capacity_store(ea_capacity_data)
    working_rows_df = _resolve_active_capacity_scenario_rows(working_scenario_data)

    resolved_countries = _resolve_capacity_country_scope(
        train_raw_df,
        ea_raw_df,
        selected_countries,
        scenario_rows_df=working_rows_df,
        start_date=start_date,
        end_date=end_date,
    )

    woodmac_change_df = _build_train_change_log(
        train_raw_df, resolved_countries, other_countries_mode, start_date, end_date
    )
    ea_change_df = _build_ea_change_log(
        ea_raw_df, resolved_countries, other_countries_mode, start_date, end_date
    )
    internal_change_df = _build_internal_scenario_change_log(
        working_rows_df,
        resolved_countries,
        other_countries_mode,
        start_date,
        end_date,
    )
    hierarchical_df = _build_train_change_hierarchical_rows(
        woodmac_change_df,
        ea_change_df,
        internal_change_df=internal_change_df,
        time_view=time_view,
        detail_view=detail_view,
    )

    if hierarchical_df.empty:
        raise PreventUpdate

    internal_cols = {"Type", "country_group_key", "plant_group_key", "month_group_key", "month_group_end"}
    export_df = hierarchical_df.drop(columns=[c for c in internal_cols if c in hierarchical_df.columns])
    export_df = export_df.drop(
        columns=[
            "Woodmac Adds (MTPA)",
            "Woodmac Reductions (MTPA)",
            "EA Adds (MTPA)",
            "EA Reductions (MTPA)",
            INTERNAL_SCENARIO_ADDS_COLUMN,
            INTERNAL_SCENARIO_REDUCTIONS_COLUMN,
        ],
        errors="ignore",
    )
    selected_option = _get_capacity_scenario_option_map(scenario_options_data).get(
        int(selected_scenario_id)
    ) if pd.notna(pd.to_numeric(selected_scenario_id, errors="coerce")) else None
    if not selected_option:
        export_df = export_df.drop(
            columns=[
                INTERNAL_SCENARIO_NET_COLUMN,
            ],
            errors="ignore",
        )
    else:
        scenario_name = selected_option.get("scenario_name", "Internal Scenario")
        export_df = export_df.rename(
            columns={
                INTERNAL_SCENARIO_NET_COLUMN: f"{scenario_name} Net Delta (MTPA)",
            }
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Capacity_Change_Comparison_{timestamp}.xlsx"
    return dcc.send_bytes(
        _export_matrix_to_excel_bytes(export_df, "Capacity Change"),
        filename,
    )


@callback(
    Output("capacity-page-download-train-timeline-excel", "data"),
    Input("capacity-page-export-train-timeline-button", "n_clicks"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-capacity-scenario-selected-store", "data"),
    State("capacity-page-capacity-scenario-working-store", "data"),
    State("capacity-page-capacity-scenario-options-store", "data"),
    State("capacity-page-capacity-scenario-dirty-store", "data"),
    State("capacity-page-country-dropdown", "value"),
    State("capacity-page-other-country-mode", "value"),
    State("capacity-page-date-range", "start_date"),
    State("capacity-page-date-range", "end_date"),
    State("capacity-page-train-timeline-original-name-visibility", "value"),
    running=[
        (Output("capacity-page-export-train-timeline-button", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def export_train_timeline_excel(
    n_clicks,
    train_capacity_data,
    ea_capacity_data,
    selected_scenario_id,
    working_scenario_data,
    scenario_options_data,
    dirty_store,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    original_name_visibility,
    timeline_row_data=None,
):
    if not n_clicks:
        raise PreventUpdate

    train_raw_df = _deserialize_train_capacity_store(train_capacity_data)
    ea_raw_df = _deserialize_ea_capacity_store(ea_capacity_data)
    active_rows_df = _resolve_active_capacity_scenario_rows(
        working_scenario_data,
        dirty_store,
        timeline_row_data,
    )

    resolved_countries = _resolve_capacity_country_scope(
        train_raw_df,
        ea_raw_df,
        selected_countries,
        scenario_rows_df=active_rows_df,
        start_date=start_date,
        end_date=end_date,
    )

    woodmac_change_df = _build_train_change_log(
        train_raw_df,
        resolved_countries,
        other_countries_mode,
        start_date,
        end_date,
    )
    ea_change_df = _build_ea_change_log(
        ea_raw_df,
        resolved_countries,
        other_countries_mode,
        start_date,
        end_date,
    )
    export_df = _build_train_timeline_df(
        woodmac_change_df,
        ea_change_df,
        aggregate_from_date=start_date,
    )
    woodmac_lookup_df = _build_provider_timeline_lookup_snapshot(
        _build_train_change_log(
            train_raw_df,
            resolved_countries,
            other_countries_mode,
            None,
            None,
        ),
        "woodmac",
        start_date,
        end_date,
    )
    ea_lookup_df = _build_provider_timeline_lookup_snapshot(
        _build_ea_change_log(
            ea_raw_df,
            resolved_countries,
            other_countries_mode,
            None,
            None,
        ),
        "energy_aspects",
        start_date,
        end_date,
    )
    selected_scenario_value = pd.to_numeric(selected_scenario_id, errors="coerce")
    export_row_keys = pd.Series(dtype="object")
    if pd.notna(selected_scenario_value):
        visible_scenario_rows_df = _filter_visible_capacity_scenario_rows(
            active_rows_df,
            resolved_countries,
            other_countries_mode,
            start_date,
            end_date,
        )
        scenario_lookup_df = _build_internal_scenario_lookup_snapshot(
            active_rows_df,
            resolved_countries,
            other_countries_mode,
            start_date,
            end_date,
        )
        export_df = _build_train_timeline_grid_rows(
            export_df,
            visible_scenario_rows_df,
            aggregate_from_date=start_date,
            woodmac_lookup_df=woodmac_lookup_df,
            ea_lookup_df=ea_lookup_df,
            scenario_lookup_df=scenario_lookup_df,
        )
        export_row_keys = export_df["scenario_row_key"].copy()
        export_df = export_df.drop(
            columns=[
                "__scenario_overridden",
                "__woodmac_out_of_range",
                "__ea_out_of_range",
                "__scenario_out_of_range",
                "timeline_direction",
                "timeline_reference_key",
            ],
            errors="ignore",
        )
        export_columns = [
            column["id"]
            for column in _get_train_timeline_columns(
                show_original_names=(original_name_visibility == "show"),
                include_scenario=True,
            )
        ]
    else:
        export_df = _build_train_timeline_grid_rows(
            export_df,
            pd.DataFrame(),
            aggregate_from_date=start_date,
            woodmac_lookup_df=woodmac_lookup_df,
            ea_lookup_df=ea_lookup_df,
        )
        export_row_keys = export_df["scenario_row_key"].copy()
        export_df = export_df.drop(
            columns=[
                "__woodmac_out_of_range",
                "__ea_out_of_range",
                "__scenario_out_of_range",
                "timeline_direction",
                "timeline_reference_key",
                "__scenario_overridden",
            ],
            errors="ignore",
        )
        export_columns = [
            column["id"]
            for column in _get_train_timeline_columns(
                show_original_names=(original_name_visibility == "show"),
                include_scenario=False,
            )
        ]

    export_df = export_df[[column_name for column_name in export_columns if column_name in export_df.columns]]

    if export_df.empty:
        raise PreventUpdate

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Train_Timeline_{timestamp}.xlsx"
    if pd.notna(selected_scenario_value):
        selected_option = _get_capacity_scenario_option_map(scenario_options_data).get(
            int(selected_scenario_value)
        )
        scenario_name = (selected_option or {}).get("scenario_name", "Internal Scenario")
        export_workbook_df = export_df.copy()
        export_workbook_df.insert(0, TRAIN_TIMELINE_IMPORT_KEY_COLUMN, export_row_keys.tolist())
        metadata_df = _build_train_timeline_upload_metadata_df(
            export_workbook_df,
            int(selected_scenario_value),
            scenario_name,
            original_name_visibility,
            active_rows_df,
        )
        return dcc.send_bytes(
            _export_train_timeline_workbook_bytes(export_workbook_df, metadata_df),
            filename,
        )

    return dcc.send_bytes(
        _export_matrix_to_excel_bytes(export_df, TRAIN_TIMELINE_SHEET_NAME),
        filename,
    )
