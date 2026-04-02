from io import BytesIO, StringIO
import datetime as dt
import re

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, dash_table, callback, Input, Output, State, ALL, ctx, no_update
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate
from sqlalchemy import text

from utils.export_flow_data import (
    COUNTRY_MAPPING_CTE,
    DB_SCHEMA,
    build_export_flow_matrix,
    default_selected_countries,
    engine,
    get_available_countries,
)
from utils.table_styles import StandardTableStyleManager, TABLE_COLORS


EXPORT_BUTTON_STYLE = {
    "marginLeft": "20px",
    "padding": "5px 15px",
    "backgroundColor": "#28a745",
    "color": "white",
    "border": "none",
    "borderRadius": "4px",
    "cursor": "pointer",
    "fontWeight": "bold",
    "fontSize": "12px",
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

SECONDARY_CONTROL_BUTTON_STYLE = {
    "padding": "7px 14px",
    "backgroundColor": "#ffffff",
    "color": "#334155",
    "border": "1px solid #cbd5e1",
    "borderRadius": "999px",
    "cursor": "pointer",
    "fontWeight": "600",
    "fontSize": "12px",
}

CAPACITY_SOURCE_TABLE = f"{DB_SCHEMA}.woodmac_lng_plant_monthly_capacity_nominal_mta"
WOODMAC_ANNUAL_OUTPUT_SOURCE_TABLE = f"{DB_SCHEMA}.woodmac_lng_plant_train_annual_output_mta"
EA_CAPACITY_SOURCE_TABLE = f"{DB_SCHEMA}.ea_lng_liquefaction_projects"
EA_EXCLUDED_STATUSES = {"cancelled"}
EA_NEGATIVE_STATUSES = {"retired"}
EA_TOP_PANEL_EXCLUDED_STATUSES = EA_EXCLUDED_STATUSES | EA_NEGATIVE_STATUSES
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

TRAIN_CHANGE_TIME_VIEW_LABELS = {
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "yearly": "Yearly",
}

TRAIN_CHANGE_DETAIL_VIEW_LABELS = {
    "total": "Total",
    "country": "Country",
    "plants": "Plants View",
    "plants_trains": "Plants + Trains View",
}

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
latest_train_metadata AS (
    SELECT DISTINCT ON (train_row.id_plant, train_row.id_lng_train)
        train_row.id_plant,
        train_row.id_lng_train,
        to_jsonb(train_row) AS train_json
    FROM {DB_SCHEMA}.woodmac_lng_plant_train AS train_row
    ORDER BY train_row.id_plant, train_row.id_lng_train, train_row.upload_timestamp_utc DESC
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
            NULLIF(BTRIM(train_metadata.train_json ->> 'lng_train_name_short'), ''),
            NULLIF(BTRIM(train_metadata.train_json ->> 'lng_train_name'), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'lng_train_name_short'), ''),
            NULLIF(BTRIM(to_jsonb(source_row) ->> 'lng_train_name'), ''),
            CONCAT('Train ', source_row.id_lng_train::text)
        ) AS lng_train_name_short,
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
            NULLIF(BTRIM(train_metadata.train_json ->> 'lng_train_name_short'), ''),
            NULLIF(BTRIM(train_metadata.train_json ->> 'lng_train_name'), ''),
            NULLIF(BTRIM(annual_row.lng_train_name_short), ''),
            CONCAT('Train ', annual_row.id_lng_train::text)
        ) AS lng_train_name_short,
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
        MAX(annual_row.lng_train_name_short) AS lng_train_name_short,
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
        monthly_row.lng_train_name_short,
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
        monthly_row.lng_train_name_short,
        monthly_row.capacity_mtpa
    FROM monthly_carry_forward_base AS monthly_row
    CROSS JOIN coverage_horizon AS horizon
    CROSS JOIN LATERAL generate_series(
        (monthly_row.last_monthly_month + INTERVAL '1 month')::date,
        horizon.max_month,
        INTERVAL '1 month'
    ) AS month_series(month)
    LEFT JOIN monthly_coverage_map AS monthly_map
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
        annual_row.plant_name,
        annual_row.lng_train_name_short,
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
        annual_row.lng_train_name_short,
        annual_row.capacity_mtpa
    FROM annual_only_fallback_base AS annual_row
    CROSS JOIN LATERAL generate_series(
        annual_row.first_active_month,
        annual_row.last_annual_month,
        INTERVAL '1 month'
    ) AS month_series(month)
    LEFT JOIN monthly_coverage_map AS monthly_map
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
        annual_row.plant_name,
        annual_row.lng_train_name_short,
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
    LEFT JOIN monthly_coverage_map AS monthly_map
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
        plant_name,
        lng_train_name_short,
        capacity_mtpa
    FROM latest_capacity

    UNION ALL

    SELECT
        id_plant,
        id_lng_train,
        month,
        country_name,
        plant_name,
        lng_train_name_short,
        capacity_mtpa
    FROM monthly_carry_forward

    UNION ALL

    SELECT
        id_plant,
        id_lng_train,
        month,
        country_name,
        plant_name,
        lng_train_name_short,
        capacity_mtpa
    FROM annual_only_fallback

    UNION ALL

    SELECT
        id_plant,
        id_lng_train,
        month,
        country_name,
        plant_name,
        lng_train_name_short,
        capacity_mtpa
    FROM annual_only_carry_forward
)
SELECT
    month,
    country_name,
    plant_name,
    lng_train_name_short,
    id_plant,
    id_lng_train,
    capacity_mtpa
FROM combined_capacity
ORDER BY month, country_name, plant_name, lng_train_name_short
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
{COUNTRY_MAPPING_CTE}
SELECT raw_country_key, country_name
FROM country_mapping
ORDER BY country_name, raw_country_key
"""

PLANT_MAPPING_QUERY = f"""
SELECT
    country_name,
    provider,
    source_field,
    source_name,
    scope_hint,
    component_hint,
    plant_name
FROM {DB_SCHEMA}.mapping_plant_name
ORDER BY country_name, provider, source_field, source_name
"""

TRAIN_MAPPING_QUERY = f"""
SELECT
    country_name,
    plant_name,
    provider,
    parent_source_field,
    parent_source_name,
    source_field,
    source_name,
    scope_hint,
    component_hint,
    train,
    allocation_share,
    notes
FROM {DB_SCHEMA}.mapping_plant_train_name
ORDER BY country_name, plant_name, provider, train, parent_source_name, source_name
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


def fetch_country_mapping_df() -> pd.DataFrame:
    with engine.connect() as connection:
        mapping_df = pd.read_sql_query(COUNTRY_MAPPING_QUERY, connection)

    if mapping_df.empty:
        return pd.DataFrame(columns=["raw_country_key", "country_name"])

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
    mapping_df = mapping_df[
        (mapping_df["raw_country_key"] != "") & (mapping_df["country_name"] != "")
    ].drop_duplicates(subset=["raw_country_key"], keep="first")

    return mapping_df


def fetch_plant_mapping_df() -> pd.DataFrame:
    with engine.connect() as connection:
        mapping_df = pd.read_sql_query(PLANT_MAPPING_QUERY, connection)

    if mapping_df.empty:
        return pd.DataFrame(
            columns=[
                "country_name",
                "provider",
                "source_field",
                "source_name",
                "scope_hint",
                "component_hint",
                "plant_name",
            ]
        )

    for column_name in [
        "country_name",
        "provider",
        "source_field",
        "source_name",
        "scope_hint",
        "component_hint",
        "plant_name",
    ]:
        mapping_df[column_name] = (
            mapping_df[column_name]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    mapping_df = mapping_df[
        (mapping_df["country_name"] != "")
        & (mapping_df["provider"] != "")
        & (mapping_df["source_field"] != "")
        & (mapping_df["source_name"] != "")
        & (mapping_df["plant_name"] != "")
    ].copy()
    mapping_df["__source_name_key"] = mapping_df["source_name"].str.upper()
    mapping_df = mapping_df.drop_duplicates(
        subset=["country_name", "provider", "source_field", "__source_name_key"],
        keep="first",
    )

    return mapping_df.reset_index(drop=True)


def fetch_train_mapping_df() -> pd.DataFrame:
    with engine.connect() as connection:
        mapping_df = pd.read_sql_query(TRAIN_MAPPING_QUERY, connection)

    if mapping_df.empty:
        return pd.DataFrame(
            columns=[
                "country_name",
                "plant_name",
                "provider",
                "parent_source_field",
                "parent_source_name",
                "source_field",
                "source_name",
                "scope_hint",
                "component_hint",
                "train",
                "allocation_share",
                "notes",
            ]
        )

    text_columns = [
        "country_name",
        "plant_name",
        "provider",
        "parent_source_field",
        "parent_source_name",
        "source_field",
        "source_name",
        "scope_hint",
        "component_hint",
        "notes",
    ]
    for column_name in text_columns:
        mapping_df[column_name] = (
            mapping_df[column_name]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    mapping_df["train"] = pd.to_numeric(mapping_df["train"], errors="coerce").astype("Int64")
    mapping_df["allocation_share"] = (
        pd.to_numeric(mapping_df["allocation_share"], errors="coerce")
        .fillna(1.0)
    )
    mapping_df = mapping_df.dropna(
        subset=[
            "country_name",
            "plant_name",
            "provider",
            "parent_source_field",
            "parent_source_name",
            "source_field",
            "source_name",
            "train",
        ]
    ).copy()
    mapping_df["__parent_source_name_key"] = mapping_df["parent_source_name"].str.upper()
    mapping_df["__source_name_key"] = mapping_df["source_name"].str.upper()
    mapping_df = mapping_df.drop_duplicates(
        subset=[
            "country_name",
            "plant_name",
            "provider",
            "parent_source_field",
            "__parent_source_name_key",
            "source_field",
            "__source_name_key",
            "train",
        ],
        keep="first",
    )

    return mapping_df.reset_index(drop=True)


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


def _standardize_plant_names(
    raw_df: pd.DataFrame,
    plant_mapping_df: pd.DataFrame | None,
    provider: str,
    source_field: str,
    source_column: str,
    output_column: str = "plant_name",
    mapping_applied_column: str | None = None,
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

    if plant_mapping_df is None or plant_mapping_df.empty:
        if output_column != source_column or output_column not in standardized_df.columns:
            standardized_df[output_column] = standardized_df[source_column]
        if mapping_applied_column:
            standardized_df[mapping_applied_column] = False
        return standardized_df

    relevant_mapping_df = plant_mapping_df[
        (plant_mapping_df["provider"] == provider)
        & (plant_mapping_df["source_field"] == source_field)
    ][["country_name", "__source_name_key", "plant_name"]].rename(
        columns={"plant_name": "__mapped_plant_name"}
    )

    if relevant_mapping_df.empty:
        if output_column != source_column or output_column not in standardized_df.columns:
            standardized_df[output_column] = standardized_df[source_column]
        if mapping_applied_column:
            standardized_df[mapping_applied_column] = False
        return standardized_df

    standardized_df["__source_name_key"] = standardized_df[source_column].str.upper()
    standardized_df = standardized_df.merge(
        relevant_mapping_df,
        how="left",
        on=["country_name", "__source_name_key"],
    )
    standardized_df[output_column] = standardized_df["__mapped_plant_name"].where(
        standardized_df["__mapped_plant_name"].notna()
        & standardized_df["__mapped_plant_name"].ne(""),
        standardized_df[source_column],
    )
    if mapping_applied_column:
        standardized_df[mapping_applied_column] = (
            standardized_df["__mapped_plant_name"].notna()
            & standardized_df["__mapped_plant_name"].ne("")
        )
    standardized_df = standardized_df.drop(
        columns=["__source_name_key", "__mapped_plant_name"],
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
    plant_mapping_df: pd.DataFrame | None = None,
    train_mapping_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    with engine.connect() as connection:
        raw_df = pd.read_sql_query(WOODMAC_TRAIN_CAPACITY_QUERY, connection)

    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "country_name",
                "raw_plant_name",
                "plant_name",
                "plant_mapping_applied",
                "raw_train_name",
                "train",
                "allocation_share",
                "train_mapping_applied",
                "lng_train_name_short",
                "id_plant",
                "id_lng_train",
                "capacity_mtpa",
            ]
        )

    raw_df["month"] = pd.to_datetime(raw_df["month"])
    for column_name in ["country_name", "plant_name", "lng_train_name_short"]:
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
    raw_df["raw_plant_name"] = raw_df["plant_name"]
    raw_df["raw_train_name"] = raw_df["lng_train_name_short"]

    raw_df = _standardize_country_names(raw_df, country_mapping_df)
    raw_df = _standardize_plant_names(
        raw_df,
        plant_mapping_df,
        provider="woodmac",
        source_field="plant_name",
        source_column="plant_name",
        output_column="plant_name",
        mapping_applied_column="plant_mapping_applied",
    )
    raw_df = _apply_train_mapping(
        raw_df,
        train_mapping_df,
        provider="woodmac",
        parent_source_field="plant_name",
        parent_source_column="raw_plant_name",
        source_field="lng_train_name_short",
        source_column="raw_train_name",
        capacity_column="capacity_mtpa",
        output_column="train",
        mapping_applied_column="train_mapping_applied",
    )

    return raw_df


def fetch_ea_capacity_raw_data(
    country_mapping_df: pd.DataFrame | None = None,
    plant_mapping_df: pd.DataFrame | None = None,
    train_mapping_df: pd.DataFrame | None = None,
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
                "plant_mapping_applied",
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
    raw_df = _standardize_plant_names(
        raw_df,
        plant_mapping_df,
        provider="energy_aspects",
        source_field="project_name",
        source_column="project_name",
        output_column="plant_name",
        mapping_applied_column="plant_mapping_applied",
    )
    raw_df = _apply_train_mapping(
        raw_df,
        train_mapping_df,
        provider="energy_aspects",
        parent_source_field="project_name",
        parent_source_column="project_name",
        source_field="train_name",
        source_column="train_name",
        capacity_column="capacity_mtpa",
        output_column="train",
        mapping_applied_column="train_mapping_applied",
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


def _serialize_dataframe(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None

    return df.to_json(date_format="iso", orient="split")


def _deserialize_dataframe(data: str | None) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()

    return pd.read_json(StringIO(data), orient="split")


def _serialize_timestamp(value) -> str | None:
    if value is None or pd.isna(value):
        return None

    return pd.Timestamp(value).isoformat()


def _normalize_month_date(value) -> pd.Timestamp | None:
    if not value:
        return None

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None

    return timestamp.to_period("M").to_timestamp()


def _get_date_bounds(dataframes: list[pd.DataFrame]) -> tuple[str | None, str | None]:
    non_empty_frames = [df for df in dataframes if df is not None and not df.empty]
    if not non_empty_frames:
        return None, None

    combined_df = pd.concat(non_empty_frames, ignore_index=True)
    min_month = pd.to_datetime(combined_df["month"]).min()
    max_month = pd.to_datetime(combined_df["month"]).max()

    return min_month.strftime("%Y-%m-%d"), max_month.strftime("%Y-%m-%d")


def _get_default_interval_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    current_year = pd.Timestamp.now().year
    default_start = pd.Timestamp(year=current_year, month=1, day=1)
    default_end = pd.Timestamp(year=current_year + 5, month=12, day=1)
    return default_start, default_end


def _filter_by_date_range(
    raw_df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    filtered_df = raw_df.copy()
    filtered_df["month"] = pd.to_datetime(filtered_df["month"]).dt.to_period("M").dt.to_timestamp()

    start_month = _normalize_month_date(start_date)
    end_month = _normalize_month_date(end_date)

    if start_month is not None:
        filtered_df = filtered_df[filtered_df["month"] >= start_month]
    if end_month is not None:
        filtered_df = filtered_df[filtered_df["month"] <= end_month]

    return filtered_df


def _create_empty_state(message: str) -> html.Div:
    return html.Div(message, className="balance-empty-state")


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
    return f"{value:+.1f} MTPA"


def _format_total_capacity(value: float) -> str:
    return f"{value:.1f} MTPA total"


def _format_yoy_percent(current_value: float, previous_value: float) -> str:
    if abs(previous_value) < 1e-9:
        return "n/a vs 12m ago"

    percent_change = ((current_value - previous_value) / previous_value) * 100
    return f"{percent_change:+.1f}% vs 12m ago"


def _build_january_yoy_annotations(total_series: pd.Series) -> list[dict]:
    if total_series.empty:
        return []

    january_points = total_series[total_series.index.month == 1]
    if january_points.empty:
        return []

    annotations = []
    offset_pattern = [-34, -48, -34, -48]

    for annotation_index, (current_date, current_value) in enumerate(january_points.items()):
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
                    f"{_format_total_capacity(current_value)}"
                    f"<br>{_format_yoy_delta(delta_value)}"
                    f"<br>{_format_yoy_percent(current_value, previous_value)}"
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
                font=dict(size=10, family="Arial", color=accent_color),
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


def _format_table_cell_value(value) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"

    return str(value)


def _build_responsive_column_styles(df: pd.DataFrame) -> list[dict]:
    column_styles = []

    for column_name in df.columns:
        header_length = len(str(column_name))
        value_lengths = df[column_name].map(_format_table_cell_value).map(len)
        max_length = max([header_length] + value_lengths.tolist()) if not df.empty else header_length

        if column_name == "Month":
            width_px = max(92, min((max_length * 8) + 24, 120))
        elif str(column_name).startswith("Total "):
            width_px = max(96, min((max_length * 9) + 28, 140))
        else:
            width_px = max(72, min((max_length * 9) + 28, 220))

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


def _format_metadata_timestamp(value) -> str | None:
    if not value:
        return None

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return str(value)

    return timestamp.strftime("%Y-%m-%d %H:%M")


def _rename_total_column(matrix_df: pd.DataFrame) -> pd.DataFrame:
    if matrix_df.empty:
        return pd.DataFrame(columns=["Month", "Total MTPA"])

    renamed_df = matrix_df.rename(columns={"Total MMTPA": "Total MTPA"})
    if "Total MTPA" not in renamed_df.columns:
        renamed_df["Total MTPA"] = 0.0

    return renamed_df


def _build_capacity_metadata_lines(metadata: dict | None) -> list[str]:
    if not metadata:
        return [
            f"Primary source table: {CAPACITY_SOURCE_TABLE}",
            f"Fallback source table: {WOODMAC_ANNUAL_OUTPUT_SOURCE_TABLE}",
        ]

    metadata = metadata.get("woodmac", metadata)
    metadata_lines = [
        f"Primary source table: {CAPACITY_SOURCE_TABLE}",
        f"Fallback source table: {WOODMAC_ANNUAL_OUTPUT_SOURCE_TABLE}",
    ]

    upload_timestamp = _format_metadata_timestamp(metadata.get("upload_timestamp_utc"))
    if upload_timestamp:
        metadata_lines.append(f"latest upload_timestamp_utc: {upload_timestamp}")

    source_rows = metadata.get("source_rows")
    monthly_source_rows = metadata.get("monthly_source_rows")
    monthly_carry_forward_train_months = metadata.get("monthly_carry_forward_train_months")
    annual_fallback_train_months = metadata.get("annual_fallback_train_months")
    source_trains = metadata.get("source_trains")
    if (
        source_rows is not None
        or monthly_source_rows is not None
        or monthly_carry_forward_train_months is not None
        or annual_fallback_train_months is not None
        or source_trains is not None
    ):
        metadata_lines.append(
            "Combined monthly rows: "
            f"{source_rows or 0:,} | "
            f"monthly nominal rows: {monthly_source_rows or 0:,} | "
            f"monthly carry-forward train-months: {monthly_carry_forward_train_months or 0:,} | "
            f"annual-only proxy/carry-forward train-months: {annual_fallback_train_months or 0:,} | "
            f"plant/train series: {source_trains or 0:,}"
        )

    min_month = _format_metadata_timestamp(metadata.get("min_month"))
    max_month = _format_metadata_timestamp(metadata.get("max_month"))
    if min_month or max_month:
        metadata_lines.append(
            f"Combined source coverage: {(min_month or 'n/a')[:7]} to {(max_month or 'n/a')[:7]}"
        )

    metadata_lines.append(WOODMAC_LEGACY_CAPACITY_NOTE)

    return metadata_lines


def _build_ea_capacity_metadata_lines(
    metadata: dict | None,
    schedule_df: pd.DataFrame,
) -> list[str]:
    if not metadata:
        return [f"Source table: {EA_CAPACITY_SOURCE_TABLE}", EA_SCHEDULE_CAPACITY_NOTE]

    metadata = metadata.get("ea", metadata)
    metadata_lines = [f"Source table: {EA_CAPACITY_SOURCE_TABLE}"]

    publication_date = _format_metadata_timestamp(metadata.get("publication_date"))
    if publication_date:
        metadata_lines.append(f"latest publication_date: {publication_date[:10]}")

    source_rows = metadata.get("source_rows")
    dated_rows = metadata.get("dated_rows")
    source_countries = metadata.get("source_countries")
    if source_rows is not None or dated_rows is not None or source_countries is not None:
        metadata_lines.append(
            "Latest snapshot rows: "
            f"{source_rows or 0:,} | "
            f"dated rows: {dated_rows or 0:,} | "
            f"source countries: {source_countries or 0:,}"
        )

    if not schedule_df.empty:
        min_month = pd.to_datetime(schedule_df["month"]).min().strftime("%Y-%m")
        max_month = pd.to_datetime(schedule_df["month"]).max().strftime("%Y-%m")
        metadata_lines.append(
            f"Derived schedule coverage in current view: {min_month} to {max_month}"
        )

    metadata_lines.append(EA_SCHEDULE_CAPACITY_NOTE)
    metadata_lines.append(
        "Cancelled and retired projects are excluded from this top schedule view."
    )

    return metadata_lines


def _build_section_summary(
    raw_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    other_countries_mode: str,
    metadata_lines: list[str] | None = None,
) -> html.Div:
    summary_children = []

    if raw_df.empty:
        summary_children.append(
            html.Div("No source data returned.", className="balance-summary-row")
        )
    else:
        month_start = (
            matrix_df["Month"].iloc[0]
            if not matrix_df.empty
            else raw_df["month"].min().strftime("%Y-%m")
        )
        month_end = (
            matrix_df["Month"].iloc[-1]
            if not matrix_df.empty
            else raw_df["month"].max().strftime("%Y-%m")
        )
        visible_country_count = max(len(matrix_df.columns) - 2, 0)
        source_country_count = raw_df["country_name"].nunique()
        visibility_note = (
            "Other countries grouped into Rest of the World."
            if other_countries_mode == "rest_of_world"
            else "Only selected countries are included in the totals."
        )

        summary_children.append(
            html.Div(
                [
                    html.Span(f"{len(matrix_df):,} months"),
                    html.Span(f"{month_start} to {month_end}"),
                    html.Span(f"{visible_country_count} visible country columns"),
                    html.Span(f"{source_country_count} source countries"),
                    html.Span(visibility_note),
                ],
                className="balance-summary-row",
            )
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
    metadata: dict | None = None,
    time_view: str = "monthly",
    detail_view: str = "country",
    visible_row_count: int | None = None,
) -> html.Div:
    metadata = metadata or {}
    ea_metadata = metadata.get("ea", metadata if "publication_date" in metadata else {})
    time_view_label = TRAIN_CHANGE_TIME_VIEW_LABELS.get(time_view, "Monthly")
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
                            f"Woodmac Effective Date reflects the first day of each monthly series point. Monthly nominal capacity is used first. For trains with monthly history, the last monthly capacity is carried forward after monthly coverage ends. Annual proxy is only used for trains with no monthly capacity history, and annual-only legacy trains are carried forward from {WOODMAC_ANNUAL_CARRY_FORWARD_START[:4]} onward."
                        )
                    ],
                    className="balance-metadata-row",
                ),
                html.Div(
                    [html.Span(WOODMAC_LEGACY_CAPACITY_NOTE)],
                    className="balance-metadata-row",
                ),
                html.Div(
                    [
                        html.Span(
                            "Energy Aspects Effective Date uses the month of start_date from the latest liquefaction publication snapshot. Cancelled projects are excluded from this comparison table."
                        )
                    ],
                    className="balance-metadata-row",
                ),
                html.Div(
                    [
                        html.Span(
                            f"Time View aggregates changes into {time_view_label.lower()} periods. Detail View is set to {detail_view_label}. Total groups all visible countries together, Country shows one row per period-country, Plants View keeps one row per period-plant, and Plants + Trains View adds shared canonical train rows only when the visible provider changes are fully resolved at train level."
                        )
                    ],
                    className="balance-metadata-row",
                ),
                html.Div(
                    [
                        html.Span(
                            "Country names are standardized with at_lng.mappings_country, plant names are standardized with at_lng.mapping_plant_name, and train numbers are inferred directly from simple raw Train N labels before exception mappings from at_lng.mapping_plant_train_name are applied."
                        )
                    ],
                    className="balance-metadata-row",
                ),
            ]
        )

    woodmac_total_added = woodmac_change_df.loc[
        woodmac_change_df["Delta MTPA"] > 0, "Delta MTPA"
    ].sum()
    woodmac_total_reduced = woodmac_change_df.loc[
        woodmac_change_df["Delta MTPA"] < 0, "Delta MTPA"
    ].sum()
    woodmac_affected_trains = woodmac_change_df["series_key"].dropna().nunique()
    ea_total_added = ea_change_df.get("EA Adds (MTPA)", pd.Series(dtype=float)).sum()
    ea_total_reduced = ea_change_df.get("EA Reductions (MTPA)", pd.Series(dtype=float)).sum()
    comparison_rows = int(visible_row_count or 0)

    ea_publication_label = _format_metadata_timestamp(ea_metadata.get("publication_date"))

    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"{comparison_rows:,} visible table rows"),
                    html.Span(f"Time view: {time_view_label}"),
                    html.Span(f"Detail view: {detail_view_label}"),
                    html.Span(f"Woodmac: {woodmac_affected_trains:,} trains"),
                    html.Span(f"Woodmac +{woodmac_total_added:,.1f} MTPA"),
                    html.Span(f"Woodmac {woodmac_total_reduced:,.1f} MTPA"),
                    html.Span(f"EA +{ea_total_added:,.1f} MTPA"),
                    html.Span(f"EA {ea_total_reduced:,.1f} MTPA"),
                ],
                className="balance-summary-row",
            ),
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
            html.Div(
                [
                    html.Span(
                        "Energy Aspects Effective Date uses the month of start_date from the latest liquefaction publication snapshot. Cancelled projects are excluded from this comparison table."
                    )
                ],
                className="balance-metadata-row",
            ),
            html.Div(
                [
                    html.Span(
                        (
                            f"Latest EA publication_date: {ea_publication_label}. "
                            if ea_publication_label
                            else ""
                        )
                        + "Monthly keeps the existing month-by-month effective dates, while Quarterly and Yearly aggregate monthly changes into the selected period labels."
                    )
                ],
                className="balance-metadata-row",
            ),
            html.Div(
                [
                    html.Span(
                        "Country names are standardized with at_lng.mappings_country, plant names are standardized with at_lng.mapping_plant_name, and train numbers are inferred directly from simple raw Train N labels before exception mappings from at_lng.mapping_plant_train_name are applied."
                    )
                ],
                className="balance-metadata-row",
            ),
        ]
    )


def _build_train_timeline_df(
    woodmac_change_df: pd.DataFrame,
    ea_change_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "Country",
        "Plant",
        "Train",
        "Woodmac First Effective Date",
        "Woodmac Total Capacity Added",
        "Energy Aspects First Effective Date",
        "Energy Aspects Total Capacity Added",
    ]
    if woodmac_change_df.empty and ea_change_df.empty:
        return pd.DataFrame(columns=columns)

    hierarchical_df = _build_train_change_hierarchical_rows(
        woodmac_change_df,
        ea_change_df,
        time_view="monthly",
        detail_view="plants_trains",
    )
    if hierarchical_df.empty:
        return pd.DataFrame(columns=columns)

    timeline_df = hierarchical_df.copy()
    timeline_df["Effective Date"] = pd.to_datetime(
        timeline_df["Effective Date"],
        errors="coerce",
    )
    for column_name in ["Country", "Plant", "Train"]:
        timeline_df[column_name] = (
            timeline_df[column_name].fillna("").astype(str).str.strip()
        )

    def _summarize_provider(group_df: pd.DataFrame, value_column: str) -> tuple[str | None, float | None]:
        positive_series = pd.to_numeric(
            group_df.get(value_column),
            errors="coerce",
        ).fillna(0.0)
        positive_mask = positive_series > 0
        if not positive_mask.any():
            return None, None

        first_effective_date = group_df.loc[positive_mask, "Effective Date"].min()
        if pd.isna(first_effective_date):
            formatted_date = None
        else:
            formatted_date = pd.Timestamp(first_effective_date).strftime("%Y-%m-%d")

        return formatted_date, round(float(positive_series[positive_mask].sum()), 2)

    summary_rows = []
    grouped_df = timeline_df.groupby(["Country", "Plant", "Train"], dropna=False, sort=False)
    for (country, plant, train), group_df in grouped_df:
        woodmac_first_date, woodmac_total_added = _summarize_provider(
            group_df,
            "Woodmac Adds (MTPA)",
        )
        ea_first_date, ea_total_added = _summarize_provider(
            group_df,
            "EA Adds (MTPA)",
        )
        summary_rows.append(
            {
                "Country": country,
                "Plant": plant,
                "Train": train,
                "Woodmac First Effective Date": woodmac_first_date,
                "Woodmac Total Capacity Added": _numeric_or_blank(woodmac_total_added),
                "Energy Aspects First Effective Date": ea_first_date,
                "Energy Aspects Total Capacity Added": _numeric_or_blank(ea_total_added),
            }
        )

    if not summary_rows:
        return pd.DataFrame(columns=columns)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["__train_numeric_sort"] = pd.to_numeric(
        summary_df["Train"],
        errors="coerce",
    )
    summary_df["__train_blank_sort"] = summary_df["Train"].eq("").astype(int)
    summary_df = summary_df.sort_values(
        ["Country", "Plant", "__train_blank_sort", "__train_numeric_sort", "Train"],
        ascending=[True, True, False, True, True],
        na_position="last",
    ).drop(
        columns=["__train_numeric_sort", "__train_blank_sort"],
        errors="ignore",
    ).reset_index(drop=True)

    return summary_df[columns]


def _build_train_timeline_summary(timeline_df: pd.DataFrame) -> html.Div:
    if timeline_df.empty:
        return html.Div(
            [
                html.Div(
                    "No train timeline rows available for the current selection.",
                    className="balance-summary-row",
                ),
                html.Div(
                    [
                        html.Span(
                            "This table follows the same Plants + Trains resolution logic as the comparison table, but summarizes positive additions only inside the selected date range."
                        )
                    ],
                    className="balance-metadata-row",
                ),
            ]
        )

    fallback_row_count = int(timeline_df["Train"].fillna("").astype(str).str.strip().eq("").sum())
    resolved_row_count = int(len(timeline_df) - fallback_row_count)
    woodmac_populated = int(
        timeline_df["Woodmac First Effective Date"].fillna("").astype(str).str.strip().ne("").sum()
    )
    ea_populated = int(
        timeline_df["Energy Aspects First Effective Date"].fillna("").astype(str).str.strip().ne("").sum()
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"{len(timeline_df):,} timeline rows"),
                    html.Span(f"{resolved_row_count:,} resolved train rows"),
                    html.Span(f"{fallback_row_count:,} fallback plant rows"),
                    html.Span(f"{woodmac_populated:,} Woodmac rows with additions"),
                    html.Span(f"{ea_populated:,} EA rows with additions"),
                ],
                className="balance-summary-row",
            ),
            html.Div(
                [
                    html.Span(
                        "First Effective Date and Total Capacity Added use positive additions only within the current selected date range."
                    )
                ],
                className="balance-metadata-row",
            ),
            html.Div(
                [
                    html.Span(
                        "Fallback rows with blank Train mirror the Plants + Trains comparison logic whenever visible provider changes are not fully resolved at train level."
                    )
                ],
                className="balance-metadata-row",
            ),
        ]
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
                html.H3(
                    title,
                    className="balance-section-title",
                    title=title_note,
                    style={"cursor": "help"} if title_note else None,
                ),
                html.Button(
                    "Export to Excel",
                    id=export_button_id,
                    n_clicks=0,
                    style=EXPORT_BUTTON_STYLE,
                ),
            ],
            className="inline-section-header",
            style={"display": "flex", "alignItems": "center"},
        )
    ]

    if subtitle:
        header_children.append(html.P(subtitle, className="balance-section-subtitle"))

    return html.Div(
        [
            html.Div(header_children, className="balance-section-header"),
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
                        html.Span("Time view", style=TRAIN_CHANGE_CONTROL_LABEL_STYLE),
                        html.Div(
                            [
                                dcc.RadioItems(
                                    id="capacity-page-train-change-time-view",
                                    options=[
                                        {"label": "Monthly", "value": "monthly"},
                                        {"label": "Quarterly", "value": "quarterly"},
                                        {"label": "Yearly", "value": "yearly"},
                                    ],
                                    value="monthly",
                                    inline=True,
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
            html.Div(id=table_container_id, className="balance-table-container"),
        ],
        className="balance-section-card",
    )


def _create_train_timeline_section(
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
            html.Div(id=table_container_id, className="balance-table-container"),
        ],
        className="balance-section-card",
    )


def _create_capacity_country_area_chart(
    matrix_df: pd.DataFrame,
    title_prefix: str = "Cumulative Monthly LNG Capacity by Country (MTPA)",
    y_axis_title: str = "Monthly Nominal Capacity (MTPA)",
) -> go.Figure:
    total_column_label = "Total MTPA"
    if matrix_df.empty:
        return _create_empty_capacity_figure("No capacity data available")

    country_columns = [
        column_name
        for column_name in matrix_df.columns
        if column_name not in {"Month", total_column_label}
    ]

    if not country_columns:
        return _create_empty_capacity_figure(
            "Select at least one country or switch to Rest of the World mode."
        )

    plot_df = matrix_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["Month"] + "-01")
    pivot_df = plot_df.set_index("date")[country_columns]
    pivot_df = pivot_df[pivot_df.sum(axis=1) > 0]

    if pivot_df.empty:
        return _create_empty_capacity_figure("No capacity data available")

    column_totals = pivot_df.sum().sort_values(ascending=False)
    pivot_df = pivot_df[column_totals.index]
    total_series = plot_df.set_index("date")[total_column_label].reindex(pivot_df.index).fillna(0.0)
    max_total = float(total_series.max()) if not total_series.empty else 0.0
    annotation_headroom = max(max_total * 0.12, 8.0)
    january_annotations = _build_january_yoy_annotations(total_series)

    fig = go.Figure()
    for color_index, group_name in enumerate(pivot_df.columns):
        color = _resolve_country_color(group_name, color_index)
        rgb = _hex_to_rgb(color)

        fig.add_trace(
            go.Scatter(
                x=pivot_df.index,
                y=pivot_df[group_name],
                mode="lines",
                name=group_name,
                line=dict(width=0.4, color=color),
                fill="tonexty",
                fillcolor=f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.78)",
                hovertemplate=(
                    f"<b>{group_name}</b><br>Date: %{{x|%b %Y}}"
                    "<br>Capacity: %{y:.1f} MTPA<extra></extra>"
                ),
                stackgroup="one",
            )
        )

    start_date = pivot_df.index.min()
    end_date = pivot_df.index.max()

    fig.update_layout(
        title={
            "text": (
                f"{title_prefix} "
                f"| {start_date.year}-{end_date.year}"
            ),
            "font": {"size": 20, "family": "Arial", "color": "#1f2937"},
            "x": 0.5,
            "xanchor": "center",
            "y": 0.95,
            "yanchor": "top",
        },
        xaxis=dict(
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
        margin=dict(l=80, r=40, t=110, b=135),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.96)",
            bordercolor="rgba(203, 213, 225, 0.9)",
            font=dict(size=11, family="Arial", color="#0f172a"),
        ),
        annotations=january_annotations,
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


def _build_train_change_log(
    raw_df: pd.DataFrame,
    selected_countries: list[str] | None,
    other_countries_mode: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    empty_columns = [
        "series_key",
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
        "Mapping Applied",
        "Train Mapping Applied",
    ]
    if raw_df.empty:
        return pd.DataFrame(columns=empty_columns)

    train_df = raw_df.copy()
    train_df["month"] = pd.to_datetime(train_df["month"]).dt.to_period("M").dt.to_timestamp()

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
        train_df.get("raw_plant_name", train_df.get("plant_name"))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    train_df["raw_train_name"] = (
        train_df.get("raw_train_name", train_df.get("lng_train_name_short"))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    train_df["train"] = pd.to_numeric(train_df.get("train"), errors="coerce").astype("Int64")
    train_df["train_mapping_applied"] = (
        train_df.get("train_mapping_applied", False).fillna(False).astype(bool)
    )
    train_df["plant_mapping_applied"] = (
        train_df.get("plant_mapping_applied", False).fillna(False).astype(bool)
    )
    train_df["allocation_share"] = pd.to_numeric(
        train_df.get("allocation_share", 1.0),
        errors="coerce",
    ).fillna(1.0)
    train_df["capacity_mtpa"] = pd.to_numeric(train_df["capacity_mtpa"], errors="coerce").fillna(0.0)

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
    train_df["series_key"] = mapped_series_key.where(train_df["train"].notna(), raw_series_key)

    train_df = (
        train_df.groupby(["series_key", "month"], as_index=False, dropna=False)
        .agg(
            {
                "country_name": "last",
                "plant_name": "last",
                "raw_plant_name": "last",
                "plant_mapping_applied": "max",
                "raw_train_name": "last",
                "train": "last",
                "train_mapping_applied": "max",
                "train_series_key": "last",
                "capacity_mtpa": "sum",
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
                "plant_mapping_applied": "last",
                "raw_train_name": "last",
                "train": "last",
                "train_mapping_applied": "last",
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
    ).round(2)

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
    change_df["Delta MTPA"] = change_df["delta_mtpa"].round(2)
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
    change_df["Mapping Applied"] = (
        change_df["plant_mapping_applied"].fillna(False).astype(bool)
    )
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
    ea_df["EA Adds (MTPA)"] = ea_df["capacity_mtpa"].where(~negative_mask, 0.0).round(2)
    ea_df["EA Reductions (MTPA)"] = (-ea_df["capacity_mtpa"]).where(negative_mask, 0.0).round(2)
    ea_df["EA Net Delta (MTPA)"] = (
        ea_df["EA Adds (MTPA)"] + ea_df["EA Reductions (MTPA)"]
    ).round(2)
    ea_df["EA Abs Delta"] = ea_df["EA Net Delta (MTPA)"].abs()
    ea_df["Effective Date"] = ea_df["month"].dt.strftime("%Y-%m-%d")
    ea_df["Country"] = ea_df["country_name"]
    ea_df["Plant"] = ea_df["plant_name"].where(
        ea_df["plant_name"].notna() & ea_df["plant_name"].ne(""),
        ea_df["project_name"],
    )
    ea_df["Train"] = pd.to_numeric(ea_df.get("train"), errors="coerce").astype("Int64")
    ea_df["Source Field"] = "project_name"
    ea_df["Source Name"] = ea_df["project_name"]
    ea_df["Parent Source Field"] = "project_name"
    ea_df["Parent Source Name"] = ea_df["project_name"]
    ea_df["Train Source Field"] = "train_name"
    ea_df["Train Source Name"] = ea_df["train_name"]
    ea_df["Mapping Applied"] = ea_df["plant_mapping_applied"].fillna(False).astype(bool)
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
            }
        )
    )

    ea_df = ea_df.sort_values(
        ["month", "Country", "EA Abs Delta", "Plant", "Train", "project_name", "train_name"],
        ascending=[True, True, False, True, True, True, True],
    )

    return ea_df[empty_columns]


def _infer_scope_component_from_name(source_name: str) -> tuple[str | None, str | None]:
    normalized_name = str(source_name or "").strip()
    if not normalized_name:
        return None, None

    pattern_map = [
        ("phase", r"\b(phase\s+\d+[a-z]?)\b"),
        ("phase", r"\b(stage\s+\d+[a-z]?)\b"),
        ("train", r"\b(train\s+\d+[a-z]?)\b"),
        ("train", r"\b(trains?\s+[0-9][0-9&,\-\s]*)\b"),
    ]
    for scope_hint, pattern in pattern_map:
        match = re.search(pattern, normalized_name, flags=re.IGNORECASE)
        if match:
            return scope_hint, match.group(1).strip()

    return None, None


def _build_unmapped_plant_alias_df(
    woodmac_change_df: pd.DataFrame,
    ea_change_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "country_name",
        "provider",
        "source_field",
        "source_name",
        "first_effective_date",
        "total_capacity_mtpa",
        "scope_hint",
        "component_hint",
        "plant_name",
    ]

    alias_frames = []
    if not woodmac_change_df.empty:
        woodmac_alias_df = woodmac_change_df.copy()
        woodmac_alias_df["provider"] = "woodmac"
        woodmac_alias_df["country_name"] = woodmac_alias_df["Country"]
        woodmac_alias_df["source_field"] = woodmac_alias_df["Source Field"]
        woodmac_alias_df["source_name"] = woodmac_alias_df["Source Name"]
        woodmac_alias_df["first_effective_date"] = woodmac_alias_df["Effective Date"]
        woodmac_alias_df["total_capacity_mtpa"] = woodmac_alias_df["Abs Delta"].fillna(0.0)
        alias_frames.append(
            woodmac_alias_df.loc[
                ~woodmac_alias_df["Mapping Applied"].fillna(False).astype(bool),
                [
                    "country_name",
                    "provider",
                    "source_field",
                    "source_name",
                    "first_effective_date",
                    "total_capacity_mtpa",
                ],
            ]
        )

    if not ea_change_df.empty:
        ea_alias_df = ea_change_df.copy()
        ea_alias_df["provider"] = "energy_aspects"
        ea_alias_df["country_name"] = ea_alias_df["Country"]
        ea_alias_df["source_field"] = ea_alias_df["Source Field"]
        ea_alias_df["source_name"] = ea_alias_df["Source Name"]
        ea_alias_df["first_effective_date"] = ea_alias_df["Effective Date"]
        ea_alias_df["total_capacity_mtpa"] = ea_alias_df["EA Abs Delta"].fillna(0.0)
        alias_frames.append(
            ea_alias_df.loc[
                ~ea_alias_df["Mapping Applied"].fillna(False).astype(bool),
                [
                    "country_name",
                    "provider",
                    "source_field",
                    "source_name",
                    "first_effective_date",
                    "total_capacity_mtpa",
                ],
            ]
        )

    if not alias_frames:
        return pd.DataFrame(columns=columns)

    alias_df = pd.concat(alias_frames, ignore_index=True)
    alias_df["source_name"] = (
        alias_df["source_name"].fillna("").astype(str).str.strip()
    )
    alias_df = alias_df[alias_df["source_name"] != ""].copy()
    if alias_df.empty:
        return pd.DataFrame(columns=columns)

    alias_df = (
        alias_df.groupby(
            ["country_name", "provider", "source_field", "source_name"],
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
    scope_component = alias_df["source_name"].map(_infer_scope_component_from_name)
    alias_df["scope_hint"] = scope_component.map(lambda value: value[0] or "")
    alias_df["component_hint"] = scope_component.map(lambda value: value[1] or "")
    alias_df["plant_name"] = ""
    alias_df = alias_df.sort_values(
        ["first_effective_date", "country_name", "provider", "total_capacity_mtpa", "source_name"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)

    return alias_df[columns]


def _build_unmapped_plant_summary(alias_df: pd.DataFrame) -> html.Div:
    if alias_df.empty:
        return html.Div(
            [
                html.Div(
                    "No unmapped plant aliases in the current selection.",
                    className="balance-summary-row",
                ),
                html.Div(
                    [
                        html.Span(
                            "When a raw Woodmac plant_name or Energy Aspects project_name is still visible in the comparison table, it will appear here for review."
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
                    html.Span(f"{len(alias_df):,} unmapped aliases"),
                    html.Span(
                        f"{alias_df['provider'].nunique():,} providers in current selection"
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
                        "Fill scope_hint, component_hint, and plant_name as needed, then save to upload the new mappings into at_lng.mapping_plant_name."
                    )
                ],
                className="balance-metadata-row",
            ),
        ]
    )


def _build_unmapped_train_alias_df(
    woodmac_change_df: pd.DataFrame,
    ea_change_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "country_name",
        "plant_name",
        "provider",
        "parent_source_field",
        "parent_source_name",
        "source_field",
        "source_name",
        "first_effective_date",
        "total_capacity_mtpa",
        "scope_hint",
        "component_hint",
        "train",
        "allocation_share",
        "notes",
    ]

    alias_frames = []
    if not woodmac_change_df.empty:
        woodmac_alias_df = woodmac_change_df.copy()
        woodmac_alias_df["provider"] = "woodmac"
        woodmac_alias_df["country_name"] = woodmac_alias_df["Country"]
        woodmac_alias_df["plant_name"] = woodmac_alias_df["Plant"]
        woodmac_alias_df["parent_source_field"] = woodmac_alias_df["Parent Source Field"]
        woodmac_alias_df["parent_source_name"] = woodmac_alias_df["Parent Source Name"]
        woodmac_alias_df["source_field"] = woodmac_alias_df["Train Source Field"]
        woodmac_alias_df["source_name"] = woodmac_alias_df["Train Source Name"]
        woodmac_alias_df["first_effective_date"] = woodmac_alias_df["Effective Date"]
        woodmac_alias_df["total_capacity_mtpa"] = woodmac_alias_df["Abs Delta"].fillna(0.0)
        alias_frames.append(
            woodmac_alias_df.loc[
                woodmac_alias_df["Mapping Applied"].fillna(False).astype(bool)
                & woodmac_alias_df["Train"].isna(),
                [
                    "country_name",
                    "plant_name",
                    "provider",
                    "parent_source_field",
                    "parent_source_name",
                    "source_field",
                    "source_name",
                    "first_effective_date",
                    "total_capacity_mtpa",
                ],
            ]
        )

    if not ea_change_df.empty:
        ea_alias_df = ea_change_df.copy()
        ea_alias_df["provider"] = "energy_aspects"
        ea_alias_df["country_name"] = ea_alias_df["Country"]
        ea_alias_df["plant_name"] = ea_alias_df["Plant"]
        ea_alias_df["parent_source_field"] = ea_alias_df["Parent Source Field"]
        ea_alias_df["parent_source_name"] = ea_alias_df["Parent Source Name"]
        ea_alias_df["source_field"] = ea_alias_df["Train Source Field"]
        ea_alias_df["source_name"] = ea_alias_df["Train Source Name"]
        ea_alias_df["first_effective_date"] = ea_alias_df["Effective Date"]
        ea_alias_df["total_capacity_mtpa"] = ea_alias_df["EA Abs Delta"].fillna(0.0)
        alias_frames.append(
            ea_alias_df.loc[
                ea_alias_df["Mapping Applied"].fillna(False).astype(bool)
                & ea_alias_df["Train"].isna(),
                [
                    "country_name",
                    "plant_name",
                    "provider",
                    "parent_source_field",
                    "parent_source_name",
                    "source_field",
                    "source_name",
                    "first_effective_date",
                    "total_capacity_mtpa",
                ],
            ]
        )

    if not alias_frames:
        return pd.DataFrame(columns=columns)

    alias_df = pd.concat(alias_frames, ignore_index=True)
    for column_name in [
        "country_name",
        "plant_name",
        "provider",
        "parent_source_field",
        "parent_source_name",
        "source_field",
        "source_name",
    ]:
        alias_df[column_name] = alias_df[column_name].fillna("").astype(str).str.strip()

    alias_df = alias_df[
        alias_df["source_name"].ne("")
        & alias_df["plant_name"].ne("")
        & alias_df["parent_source_name"].ne("")
    ].copy()
    if alias_df.empty:
        return pd.DataFrame(columns=columns)

    alias_df = (
        alias_df.groupby(
            [
                "country_name",
                "plant_name",
                "provider",
                "parent_source_field",
                "parent_source_name",
                "source_field",
                "source_name",
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
    inferred_scope_component = alias_df.apply(
        lambda row: (
            _infer_scope_component_from_name(row["source_name"])
            if any(_infer_scope_component_from_name(row["source_name"]))
            else _infer_scope_component_from_name(row["parent_source_name"])
        ),
        axis=1,
    )
    alias_df["scope_hint"] = inferred_scope_component.map(lambda value: value[0] or "")
    alias_df["component_hint"] = inferred_scope_component.map(lambda value: value[1] or "")
    alias_df["train"] = None
    alias_df["allocation_share"] = 1.0
    alias_df["notes"] = ""
    alias_df = alias_df.sort_values(
        [
            "first_effective_date",
            "country_name",
            "plant_name",
            "provider",
            "total_capacity_mtpa",
            "source_name",
        ],
        ascending=[True, True, True, True, False, True],
    ).reset_index(drop=True)

    return alias_df[columns]


def _build_unmapped_train_summary(alias_df: pd.DataFrame) -> html.Div:
    if alias_df.empty:
        return html.Div(
            [
                html.Div(
                    "No unmapped train aliases in the current selection.",
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
                        f"{alias_df['plant_name'].nunique():,} plants in current selection"
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
                        "Fill train, allocation_share, and notes as needed, then save to upload the new mappings into at_lng.mapping_plant_train_name."
                    )
                ],
                className="balance-metadata-row",
            ),
        ]
    )


def _create_unmapped_plant_mapping_table(
    table_id: str,
    alias_df: pd.DataFrame | None = None,
) -> dash_table.DataTable:
    alias_df = pd.DataFrame() if alias_df is None else alias_df.copy()
    columns = [
        {"name": "Country", "id": "country_name", "editable": False},
        {"name": "Provider", "id": "provider", "editable": False},
        {"name": "Source Field", "id": "source_field", "editable": False},
        {"name": "Source Name", "id": "source_name", "editable": False},
        {"name": "First Effective Date", "id": "first_effective_date", "editable": False},
        {
            "name": "Visible Capacity (MTPA)",
            "id": "total_capacity_mtpa",
            "editable": False,
            "type": "numeric",
            "format": {"specifier": ".2f"},
        },
        {"name": "Scope Hint", "id": "scope_hint", "editable": True},
        {"name": "Component Hint", "id": "component_hint", "editable": True},
        {"name": "Plant Name", "id": "plant_name", "editable": True},
    ]

    return dash_table.DataTable(
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
        style_header={
            "backgroundColor": "#1e293b",
            "color": "white",
            "fontWeight": "700",
            "fontSize": "11px",
            "textAlign": "center",
            "textTransform": "uppercase",
            "letterSpacing": "0.05em",
            "padding": "10px 8px",
        },
        style_cell={
            "textAlign": "left",
            "fontSize": "12px",
            "padding": "7px 10px",
            "minWidth": "90px",
            "maxWidth": "260px",
            "border": "1px solid #f1f5f9",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_cell_conditional=[
            {"if": {"column_id": "country_name"}, "minWidth": "130px", "maxWidth": "170px"},
            {"if": {"column_id": "provider"}, "minWidth": "120px", "maxWidth": "140px"},
            {"if": {"column_id": "source_field"}, "minWidth": "110px", "maxWidth": "120px"},
            {"if": {"column_id": "source_name"}, "minWidth": "220px", "maxWidth": "320px"},
            {"if": {"column_id": "plant_name"}, "minWidth": "200px", "maxWidth": "260px"},
        ],
        style_data_conditional=[
            {
                "if": {"column_id": "source_name"},
                "backgroundColor": "#f8fafc",
                "fontWeight": "600",
                "color": "#1e3a5f",
            },
            {
                "if": {"column_id": "scope_hint"},
                "backgroundColor": "rgba(59, 130, 246, 0.06)",
            },
            {
                "if": {"column_id": "component_hint"},
                "backgroundColor": "rgba(59, 130, 246, 0.06)",
            },
            {
                "if": {"column_id": "plant_name"},
                "backgroundColor": "rgba(34, 197, 94, 0.08)",
                "fontWeight": "600",
            },
        ],
        fill_width=False,
    )


def _create_unmapped_train_mapping_table(
    table_id: str,
    alias_df: pd.DataFrame | None = None,
) -> dash_table.DataTable:
    alias_df = pd.DataFrame() if alias_df is None else alias_df.copy()
    columns = [
        {"name": "Country", "id": "country_name", "editable": False},
        {"name": "Plant", "id": "plant_name", "editable": False},
        {"name": "Provider", "id": "provider", "editable": False},
        {"name": "Parent Source Field", "id": "parent_source_field", "editable": False},
        {"name": "Parent Source Name", "id": "parent_source_name", "editable": False},
        {"name": "Source Field", "id": "source_field", "editable": False},
        {"name": "Source Name", "id": "source_name", "editable": False},
        {"name": "First Effective Date", "id": "first_effective_date", "editable": False},
        {
            "name": "Visible Capacity (MTPA)",
            "id": "total_capacity_mtpa",
            "editable": False,
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {"name": "Scope Hint", "id": "scope_hint", "editable": True},
        {"name": "Component Hint", "id": "component_hint", "editable": True},
        {
            "name": "Train",
            "id": "train",
            "editable": True,
            "type": "numeric",
            "format": Format(precision=0, scheme=Scheme.fixed),
        },
        {
            "name": "Allocation Share",
            "id": "allocation_share",
            "editable": True,
            "type": "numeric",
            "format": Format(precision=4, scheme=Scheme.fixed),
        },
        {"name": "Notes", "id": "notes", "editable": True},
    ]

    return dash_table.DataTable(
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
        style_header={
            "backgroundColor": "#1e293b",
            "color": "white",
            "fontWeight": "700",
            "fontSize": "11px",
            "textAlign": "center",
            "textTransform": "uppercase",
            "letterSpacing": "0.05em",
            "padding": "10px 8px",
        },
        style_cell={
            "textAlign": "left",
            "fontSize": "12px",
            "padding": "7px 10px",
            "minWidth": "90px",
            "maxWidth": "260px",
            "border": "1px solid #f1f5f9",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_cell_conditional=[
            {"if": {"column_id": "country_name"}, "minWidth": "120px", "maxWidth": "160px"},
            {"if": {"column_id": "plant_name"}, "minWidth": "180px", "maxWidth": "240px"},
            {"if": {"column_id": "provider"}, "minWidth": "120px", "maxWidth": "140px"},
            {"if": {"column_id": "parent_source_field"}, "minWidth": "130px", "maxWidth": "140px"},
            {"if": {"column_id": "parent_source_name"}, "minWidth": "220px", "maxWidth": "320px"},
            {"if": {"column_id": "source_field"}, "minWidth": "110px", "maxWidth": "120px"},
            {"if": {"column_id": "source_name"}, "minWidth": "180px", "maxWidth": "260px"},
            {"if": {"column_id": "notes"}, "minWidth": "200px", "maxWidth": "320px"},
        ],
        style_data_conditional=[
            {
                "if": {"column_id": "plant_name"},
                "backgroundColor": "#f8fafc",
                "fontWeight": "600",
                "color": "#1e3a5f",
            },
            {
                "if": {"column_id": "source_name"},
                "backgroundColor": "#f8fafc",
                "fontWeight": "600",
                "color": "#1e3a5f",
            },
            {
                "if": {"column_id": "scope_hint"},
                "backgroundColor": "rgba(59, 130, 246, 0.06)",
            },
            {
                "if": {"column_id": "component_hint"},
                "backgroundColor": "rgba(59, 130, 246, 0.06)",
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
            {
                "if": {"column_id": "notes"},
                "backgroundColor": "rgba(15, 23, 42, 0.03)",
            },
        ],
        fill_width=False,
    )


def _create_unmapped_plant_mapping_section(
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
                    "Save Plant Mappings",
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
            _create_unmapped_plant_mapping_table(table_id),
        ]
    )

    return html.Div(
        [html.Div(header_children, className="balance-section-header")],
        className="balance-section-card",
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


def _create_capacity_table(table_id: str, df: pd.DataFrame) -> dash_table.DataTable | html.Div:
    if df.empty:
        return _create_empty_state("No data available for the current selection.")

    base_config = StandardTableStyleManager.get_base_datatable_config()
    numeric_columns = [column for column in df.columns if column != "Month"]

    columns = [{"name": "Month", "id": "Month"}]
    columns.extend(
        {
            "name": column_name,
            "id": column_name,
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
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

    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=df.to_dict("records"),
        sort_action="native",
        page_action="none",
        fill_width=False,
        fixed_rows={"headers": True},
        fixed_columns={"headers": True, "data": 1},
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": "560px",
            "width": "100%",
            "minWidth": "100%",
        },
        style_header=base_config["style_header"],
        style_cell={
            **base_config["style_cell"],
            "minWidth": "auto",
            "width": "auto",
            "maxWidth": "none",
            "border": f"1px solid {TABLE_COLORS['border_light']}",
            "padding": "6px 8px",
        },
        style_cell_conditional=_build_responsive_column_styles(df),
        style_data_conditional=style_data_conditional,
    )


def _format_capacity_value(value: float, signed: bool = False) -> str:
    if signed:
        return f"{value:+,.2f}"
    return f"{value:,.2f}"


def _numeric_or_blank(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None

    value = float(value)
    if abs(value) < 1e-9:
        return None

    return round(value, 2)


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
            ["__period_start", "Effective Date", "Country", "Plant", "Train"],
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
                "Source Name": "last",
                "Mapping Applied": "max",
                "Train Mapping Applied": "max",
            }
        )
    )

    woodmac_df["__train_sort"] = pd.to_numeric(woodmac_df["Train"], errors="coerce")
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
                "Source Name": "last",
                "Mapping Applied": "max",
                "Train Mapping Applied": "max",
            }
        )
    )

    ea_df["__train_sort"] = pd.to_numeric(ea_df["Train"], errors="coerce")
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
        normalized_train_df["Train"] = pd.to_numeric(
            normalized_train_df["Train"],
            errors="coerce",
        ).astype("Int64")
        normalized_train_df["__train_sort"] = pd.to_numeric(
            normalized_train_df["Train"],
            errors="coerce",
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
                    "Woodmac Adds (MTPA)": _numeric_or_blank(plant_row.get("Woodmac Adds (MTPA)")),
                    "Woodmac Reductions (MTPA)": _numeric_or_blank(
                        plant_row.get("Woodmac Reductions (MTPA)")
                    ),
                    "Woodmac Net Delta (MTPA)": _numeric_or_blank(
                        plant_row.get("Woodmac Net Delta (MTPA)")
                    ),
                    "EA Adds (MTPA)": _numeric_or_blank(plant_row.get("EA Adds (MTPA)")),
                    "EA Reductions (MTPA)": _numeric_or_blank(plant_row.get("EA Reductions (MTPA)")),
                    "EA Net Delta (MTPA)": _numeric_or_blank(plant_row.get("EA Net Delta (MTPA)")),
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
                    "Woodmac Adds (MTPA)": _numeric_or_blank(
                        train_row.get("Woodmac Adds (MTPA)")
                    ),
                    "Woodmac Reductions (MTPA)": _numeric_or_blank(
                        train_row.get("Woodmac Reductions (MTPA)")
                    ),
                    "Woodmac Net Delta (MTPA)": _numeric_or_blank(
                        train_row.get("Woodmac Net Delta (MTPA)")
                    ),
                    "EA Adds (MTPA)": _numeric_or_blank(train_row.get("EA Adds (MTPA)")),
                    "EA Reductions (MTPA)": _numeric_or_blank(
                        train_row.get("EA Reductions (MTPA)")
                    ),
                    "EA Net Delta (MTPA)": _numeric_or_blank(
                        train_row.get("EA Net Delta (MTPA)")
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
    expanded_country_groups: list[str] | None = None,
    expanded_plant_groups: list[str] | None = None,
    woodmac_detail_df: pd.DataFrame | None = None,
    time_view: str = "monthly",
    detail_view: str = "country",
    view_mode: str | None = None,
) -> pd.DataFrame:
    columns = [
        "Effective Date",
        "Country",
        "Plant",
        "Train",
        "Woodmac Adds (MTPA)",
        "Woodmac Reductions (MTPA)",
        "Woodmac Net Delta (MTPA)",
        "EA Adds (MTPA)",
        "EA Reductions (MTPA)",
        "EA Net Delta (MTPA)",
        "Type",
        "country_group_key",
        "plant_group_key",
        "month_group_key",
        "month_group_end",
    ]
    if woodmac_change_df.empty and ea_change_df.empty:
        return pd.DataFrame(columns=columns)

    # Backward compatibility for any still-live callback path that passes the
    # old `view_mode` keyword from the pre-refactor table controls.
    if view_mode is not None:
        detail_view = {
            "summary": "country",
            "country": "country",
            "total": "total",
            "plants": "plants",
            "plants_trains": "plants_trains",
        }.get(view_mode, detail_view)

    woodmac_detail_df = woodmac_change_df.copy() if woodmac_detail_df is None else woodmac_detail_df.copy()

    woodmac_period_df = _prepare_woodmac_period_change_df(woodmac_change_df, time_view)
    woodmac_detail_period_df = _prepare_woodmac_period_change_df(woodmac_detail_df, time_view)
    ea_period_df = _prepare_ea_period_change_df(ea_change_df, time_view)
    unresolved_plant_keys = _collect_unresolved_train_keys(woodmac_period_df, ea_period_df)

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

    total_comparison_df = pd.merge(
        woodmac_total_df,
        ea_total_df,
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

    country_comparison_df = pd.merge(
        woodmac_country_df,
        ea_country_df,
        on=["__period_start", "Effective Date", "Country"],
        how="outer",
    ).sort_values(["__period_start", "Country"]).reset_index(drop=True)

    woodmac_plant_df = pd.DataFrame(columns=["__period_start", "Effective Date", "Country", "Plant"])
    if not woodmac_period_df.empty:
        woodmac_plant_df = (
            woodmac_period_df.groupby(["__period_start", "Effective Date", "Country", "Plant"], as_index=False)
            .agg(
                {
                    "Woodmac Adds (MTPA)": "sum",
                    "Woodmac Reductions (MTPA)": "sum",
                    "Woodmac Net Delta (MTPA)": "sum",
                    "Woodmac Activity Abs": "sum",
                }
            )
        )

    ea_plant_df = pd.DataFrame(columns=["__period_start", "Effective Date", "Country", "Plant"])
    if not ea_period_df.empty:
        ea_plant_df = (
            ea_period_df.groupby(["__period_start", "Effective Date", "Country", "Plant"], as_index=False)
            .agg(
                {
                    "EA Adds (MTPA)": "sum",
                    "EA Reductions (MTPA)": "sum",
                    "EA Net Delta (MTPA)": "sum",
                    "EA Activity Abs": "sum",
                }
            )
        )

    plant_comparison_df = pd.merge(
        woodmac_plant_df,
        ea_plant_df,
        on=["__period_start", "Effective Date", "Country", "Plant"],
        how="outer",
    )
    if not plant_comparison_df.empty:
        plant_comparison_df["__sort_abs"] = (
            plant_comparison_df.get("Woodmac Activity Abs", pd.Series(dtype=float)).fillna(0.0)
            + plant_comparison_df.get("EA Activity Abs", pd.Series(dtype=float)).fillna(0.0)
        )
        plant_comparison_df = plant_comparison_df.sort_values(
            ["__period_start", "Country", "__sort_abs", "Plant"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)

    woodmac_train_df = woodmac_detail_period_df[
        woodmac_detail_period_df["Train"].notna()
    ].copy()
    ea_train_df = ea_period_df[ea_period_df["Train"].notna()].copy()
    train_comparison_df = pd.merge(
        woodmac_train_df[
            [
                "__period_start",
                "Effective Date",
                "Country",
                "Plant",
                "Train",
                "Woodmac Adds (MTPA)",
                "Woodmac Reductions (MTPA)",
                "Woodmac Net Delta (MTPA)",
                "Woodmac Activity Abs",
            ]
        ],
        ea_train_df[
            [
                "__period_start",
                "Effective Date",
                "Country",
                "Plant",
                "Train",
                "EA Adds (MTPA)",
                "EA Reductions (MTPA)",
                "EA Net Delta (MTPA)",
                "EA Activity Abs",
            ]
        ],
        on=["__period_start", "Effective Date", "Country", "Plant", "Train"],
        how="outer",
    )
    if not train_comparison_df.empty:
        train_comparison_df["__train_sort"] = pd.to_numeric(
            train_comparison_df["Train"],
            errors="coerce",
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


def _create_train_change_table(table_id: str, change_df: pd.DataFrame) -> dash_table.DataTable | html.Div:
    if change_df.empty:
        return _create_empty_state("No provider capacity changes in the current selection.")

    columns = [
        {"name": ["Effective Date", ""], "id": "Effective Date", "type": "text"},
        {"name": ["Country", ""], "id": "Country", "type": "text"},
        {"name": ["Plant", ""], "id": "Plant", "type": "text"},
        {"name": ["Train", ""], "id": "Train", "type": "text"},
        {
            "name": ["Woodmac", "Adds"],
            "id": "Woodmac Adds (MTPA)",
            "type": "numeric",
            "format": {"specifier": ".2f"},
        },
        {
            "name": ["Woodmac", "Reductions"],
            "id": "Woodmac Reductions (MTPA)",
            "type": "numeric",
            "format": {"specifier": ".2f"},
        },
        {
            "name": ["Woodmac", "Net Delta"],
            "id": "Woodmac Net Delta (MTPA)",
            "type": "numeric",
            "format": {"specifier": ".2f"},
        },
        {
            "name": ["Energy Aspects", "Adds"],
            "id": "EA Adds (MTPA)",
            "type": "numeric",
            "format": {"specifier": ".2f"},
        },
        {
            "name": ["Energy Aspects", "Reductions"],
            "id": "EA Reductions (MTPA)",
            "type": "numeric",
            "format": {"specifier": ".2f"},
        },
        {
            "name": ["Energy Aspects", "Net Delta"],
            "id": "EA Net Delta (MTPA)",
            "type": "numeric",
            "format": {"specifier": ".2f"},
        },
        {"name": ["Meta", "Type"], "id": "Type", "type": "text"},
        {"name": ["Meta", "country_group_key"], "id": "country_group_key", "type": "text"},
        {"name": ["Meta", "plant_group_key"], "id": "plant_group_key", "type": "text"},
        {"name": ["Meta", "month_group_key"], "id": "month_group_key", "type": "text"},
        {"name": ["Meta", "month_group_end"], "id": "month_group_end", "type": "text"},
    ]

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
            "if": {"column_id": "Woodmac Adds (MTPA)", "filter_query": "{Woodmac Adds (MTPA)} > 0"},
            "backgroundColor": "rgba(22, 101, 52, 0.08)",
            "color": "#166534",
            "fontWeight": "700",
        },
        {
            "if": {"column_id": "Woodmac Reductions (MTPA)", "filter_query": "{Woodmac Reductions (MTPA)} < 0"},
            "backgroundColor": "rgba(153, 27, 27, 0.08)",
            "color": "#991b1b",
            "fontWeight": "700",
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
            "if": {"column_id": "EA Adds (MTPA)", "filter_query": "{EA Adds (MTPA)} > 0"},
            "backgroundColor": "rgba(22, 101, 52, 0.08)",
            "color": "#166534",
            "fontWeight": "700",
        },
        {
            "if": {"column_id": "EA Reductions (MTPA)", "filter_query": "{EA Reductions (MTPA)} < 0"},
            "backgroundColor": "rgba(153, 27, 27, 0.08)",
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

    legend = html.Div(
        [
            html.Span(
                "Time View: Monthly, Quarterly, or Yearly",
                style={"color": "#475569", "marginRight": "18px", "fontSize": "11px"},
            ),
            html.Span(
                "Total groups all visible countries together",
                style={"color": "#475569", "marginRight": "18px", "fontSize": "11px"},
            ),
            html.Span(
                "Country shows one row per period-country",
                style={"color": "#475569", "marginRight": "18px", "fontSize": "11px"},
            ),
            html.Span(
                "Plants View keeps one row per plant; Plants + Trains adds child rows by canonical train when both providers are safely resolved",
                style={"color": "#475569", "marginRight": "18px", "fontSize": "11px"},
            ),
            html.Span(
                "Unresolved train conflicts stay at plant level only",
                style={"color": "#475569", "marginRight": "18px", "fontSize": "11px"},
            ),
            html.Span(
                "Green = additions",
                style={"color": "#166534", "marginRight": "18px", "fontSize": "11px", "fontWeight": "600"},
            ),
            html.Span(
                "Red = reductions",
                style={"color": "#991b1b", "fontSize": "11px", "fontWeight": "600"},
            ),
        ],
        style={"padding": "4px 0 12px 2px"},
    )

    table = dash_table.DataTable(
        id={"type": "capacity-train-change-expandable-table", "index": 0},
        columns=columns,
        data=change_df.to_dict("records"),
        style_table={
            "overflowX": "auto",
            "borderRadius": "4px",
            "border": "1px solid #e2e8f0",
            "maxHeight": "620px",
        },
        style_header={
            "backgroundColor": "#1e293b",
            "color": "white",
            "fontWeight": "700",
            "fontSize": "11px",
            "textAlign": "center",
            "textTransform": "uppercase",
            "letterSpacing": "0.05em",
            "padding": "10px 8px",
        },
        style_cell={
            "textAlign": "center",
            "fontSize": "12px",
            "padding": "7px 10px",
            "minWidth": "72px",
            "maxWidth": "160px",
            "border": "1px solid #f1f5f9",
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
        merge_duplicate_headers=True,
        sort_action="none",
        page_action="none",
        fill_width=False,
    )

    return html.Div([legend, table])


def _create_train_timeline_table(
    table_id: str,
    timeline_df: pd.DataFrame,
) -> dash_table.DataTable | html.Div:
    if timeline_df.empty:
        return _create_empty_state("No train timeline rows available for the current selection.")

    display_df = timeline_df.copy().where(pd.notna(timeline_df), None)

    columns = [
        {"name": ["", "Country"], "id": "Country"},
        {"name": ["", "Plant"], "id": "Plant"},
        {"name": ["", "Train"], "id": "Train"},
        {"name": ["Woodmac", "First Effective Date"], "id": "Woodmac First Effective Date"},
        {
            "name": ["Woodmac", "Total Capacity Added"],
            "id": "Woodmac Total Capacity Added",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": ["Energy Aspects", "First Effective Date"],
            "id": "Energy Aspects First Effective Date",
        },
        {
            "name": ["Energy Aspects", "Total Capacity Added"],
            "id": "Energy Aspects Total Capacity Added",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
    ]

    style_data_conditional = [
        {
            "if": {"filter_query": "{Train} = \"\""},
            "backgroundColor": "#f8fafc",
        },
        {
            "if": {"column_id": "Woodmac Total Capacity Added", "filter_query": "{Woodmac Total Capacity Added} > 0"},
            "backgroundColor": "rgba(22, 101, 52, 0.08)",
            "color": "#166534",
            "fontWeight": "700",
        },
        {
            "if": {"column_id": "Energy Aspects Total Capacity Added", "filter_query": "{Energy Aspects Total Capacity Added} > 0"},
            "backgroundColor": "rgba(22, 101, 52, 0.08)",
            "color": "#166534",
            "fontWeight": "700",
        },
    ]

    legend = html.Div(
        [
            html.Span(
                "Rows follow the same Plants + Trains resolution logic as the comparison table",
                style={"color": "#475569", "marginRight": "18px", "fontSize": "11px"},
            ),
            html.Span(
                "Blank Train = unresolved provider changes kept at plant level",
                style={"color": "#475569", "marginRight": "18px", "fontSize": "11px"},
            ),
            html.Span(
                "Totals reflect positive additions only inside the selected range",
                style={"color": "#475569", "fontSize": "11px"},
            ),
        ],
        style={"padding": "4px 0 12px 2px"},
    )

    table = dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=display_df.to_dict("records"),
        style_table={
            "overflowX": "auto",
            "borderRadius": "4px",
            "border": "1px solid #e2e8f0",
            "maxHeight": "620px",
        },
        style_header={
            "backgroundColor": "#1e293b",
            "color": "white",
            "fontWeight": "700",
            "fontSize": "11px",
            "textAlign": "center",
            "textTransform": "uppercase",
            "letterSpacing": "0.05em",
            "padding": "10px 8px",
        },
        style_cell={
            "textAlign": "center",
            "fontSize": "12px",
            "padding": "7px 10px",
            "minWidth": "90px",
            "maxWidth": "180px",
            "border": "1px solid #f1f5f9",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Country"}, "textAlign": "left", "minWidth": "140px", "maxWidth": "180px"},
            {"if": {"column_id": "Plant"}, "textAlign": "left", "minWidth": "220px", "maxWidth": "300px"},
            {"if": {"column_id": "Train"}, "minWidth": "90px", "maxWidth": "100px"},
            {"if": {"column_id": "Woodmac First Effective Date"}, "minWidth": "150px", "maxWidth": "160px"},
            {"if": {"column_id": "Energy Aspects First Effective Date"}, "minWidth": "170px", "maxWidth": "180px"},
            {"if": {"column_id": "Woodmac Total Capacity Added"}, "minWidth": "150px", "maxWidth": "160px"},
            {"if": {"column_id": "Energy Aspects Total Capacity Added"}, "minWidth": "170px", "maxWidth": "180px"},
        ],
        style_data_conditional=style_data_conditional,
        merge_duplicate_headers=True,
        sort_action="native",
        filter_action="native",
        page_action="none",
        fill_width=False,
    )

    return html.Div([legend, table])


def _export_matrix_to_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        worksheet = writer.sheets[sheet_name]
        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter
            for cell in column_cells:
                cell_value = "" if cell.value is None else str(cell.value)
                if len(cell_value) > max_length:
                    max_length = len(cell_value)
            worksheet.column_dimensions[column_letter].width = min(max_length + 2, 24)

    output.seek(0)
    return output.getvalue()


layout = html.Div(
    [
        dcc.Store(id="capacity-page-woodmac-data-store", storage_type="memory"),
        dcc.Store(id="capacity-page-train-capacity-data-store", storage_type="memory"),
        dcc.Store(id="capacity-page-ea-capacity-data-store", storage_type="memory"),
        dcc.Store(id="capacity-page-train-change-expanded-country-store", storage_type="memory", data=[]),
        dcc.Store(id="capacity-page-train-change-expanded-plant-store", storage_type="memory", data=[]),
        dcc.Store(id="capacity-page-country-options-store", storage_type="memory"),
        dcc.Store(id="capacity-page-refresh-timestamp-store", storage_type="memory"),
        dcc.Store(id="capacity-page-load-error-store", storage_type="memory"),
        dcc.Store(id="capacity-page-metadata-store", storage_type="memory"),
        dcc.Store(id="capacity-page-plant-mapping-save-trigger", storage_type="memory"),
        dcc.Store(id="capacity-page-train-mapping-save-trigger", storage_type="memory"),
        dcc.Download(id="capacity-page-download-woodmac-excel"),
        dcc.Download(id="capacity-page-download-ea-excel"),
        dcc.Download(id="capacity-page-download-train-change-excel"),
        dcc.Download(id="capacity-page-download-train-timeline-excel"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Date Range", className="filter-group-header"),
                                html.Label("Month interval:", className="filter-label"),
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
                                        )
                                    ],
                                    className="professional-date-picker",
                                ),
                            ],
                            className="filter-section filter-section-destination",
                        ),
                        html.Div(
                            [
                                html.Div("Country Columns", className="filter-group-header"),
                                html.Label("Countries:", className="filter-label"),
                                dcc.Dropdown(
                                    id="capacity-page-country-dropdown",
                                    options=[],
                                    value=None,
                                    multi=True,
                                    placeholder="Select countries to keep as separate columns",
                                    className="filter-dropdown",
                                    style={"minWidth": "380px"},
                                ),
                            ],
                            className="filter-section filter-section-origin-exp",
                        ),
                        html.Div(
                            [
                                html.Div("Other Countries", className="filter-group-header"),
                                html.Label("Handling:", className="filter-label"),
                                dcc.RadioItems(
                                    id="capacity-page-other-country-mode",
                                    options=[
                                        {
                                            "label": "Include as Rest of the World",
                                            "value": "rest_of_world",
                                        },
                                        {
                                            "label": "Exclude from the table",
                                            "value": "exclude",
                                        },
                                    ],
                                    value="rest_of_world",
                                    className="balance-radio-group",
                                    labelStyle={"display": "inline-flex", "alignItems": "center"},
                                    inputStyle={"marginRight": "6px"},
                                ),
                            ],
                            className="filter-section filter-section-volume",
                        ),
                        html.Div(
                            [
                                html.Div("Status", className="filter-group-header"),
                                html.Div(
                                    id="capacity-page-refresh-indicator",
                                    className="text-tertiary",
                                    style={"fontSize": "12px", "whiteSpace": "nowrap"},
                                ),
                                html.Div(
                                    id="capacity-page-meta-indicator",
                                    className="text-tertiary",
                                    style={"fontSize": "12px", "maxWidth": "260px"},
                                ),
                            ],
                            className="filter-section filter-section-analysis",
                        ),
                    ],
                    className="filter-bar-grouped",
                )
            ],
            className="professional-section-header",
        ),
        html.Div(
            [
                html.Div(id="capacity-page-load-error-banner"),
                dcc.Loading(
                    children=[
                        html.Div(
                            [
                                html.Div(
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
                                    ],
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(auto-fit, minmax(520px, 1fr))",
                                        "gap": "24px",
                                        "alignItems": "start",
                                    },
                                ),
                                _create_train_change_section(
                                    "Capacity Change Comparison in Selected Range",
                                    "Compare monthly, quarterly, or yearly capacity changes across Woodmac and Energy Aspects at total, country, plant, and train detail. Woodmac annual-only legacy capacity is carried forward from 2029 onward to avoid source-timing noise in year-over-year changes.",
                                    "capacity-page-train-change-summary",
                                    "capacity-page-train-change-table-container",
                                    "capacity-page-export-train-change-button",
                                ),
                                _create_train_timeline_section(
                                    "Train Timeline",
                                    "Summarize the first visible addition date and total capacity added by provider using the same Plants + Trains resolution logic as the comparison table, within the current selected date range.",
                                    "capacity-page-train-timeline-summary",
                                    "capacity-page-train-timeline-table-container",
                                    "capacity-page-export-train-timeline-button",
                                ),
                                html.Div(
                                    [
                                        _create_unmapped_plant_mapping_section(
                                            "Unmapped Plant Names in Current Selection",
                                            "Review raw provider aliases still visible in the comparison table, fill the mapping fields you want to save, and upload them directly into at_lng.mapping_plant_name.",
                                            "capacity-page-unmapped-plant-summary",
                                            "capacity-page-unmapped-plant-message",
                                            "capacity-page-save-plant-mappings-button",
                                            "capacity-page-unmapped-plant-table",
                                        ),
                                        _create_unmapped_train_mapping_section(
                                            "Unmapped Train Names in Current Selection",
                                            "Review unresolved raw train aliases for rows whose plant is already standardized, fill the train mapping fields you want to save, and upload them directly into at_lng.mapping_plant_train_name.",
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
                            ],
                            className="balance-results-stack",
                        )
                    ],
                    type="default",
                ),
            ],
            className="balance-page-shell",
        ),
    ]
)


@callback(
    Output("capacity-page-woodmac-data-store", "data"),
    Output("capacity-page-train-capacity-data-store", "data"),
    Output("capacity-page-ea-capacity-data-store", "data"),
    Output("capacity-page-country-options-store", "data"),
    Output("capacity-page-refresh-timestamp-store", "data"),
    Output("capacity-page-load-error-store", "data"),
    Output("capacity-page-metadata-store", "data"),
    Input("global-refresh-button", "n_clicks"),
    Input("capacity-page-plant-mapping-save-trigger", "data"),
    Input("capacity-page-train-mapping-save-trigger", "data"),
)
def load_capacity_source_data(_, _plant_save_trigger, _train_save_trigger):
    woodmac_df = pd.DataFrame()
    train_capacity_df = pd.DataFrame()
    ea_capacity_df = pd.DataFrame()
    metadata = {"woodmac": {}, "ea": {}}
    errors = []
    plant_mapping_df = pd.DataFrame()
    train_mapping_df = pd.DataFrame()

    try:
        country_mapping_df = fetch_country_mapping_df()
    except Exception as exc:
        country_mapping_df = pd.DataFrame()
        errors.append(f"Country mapping load failed: {exc}")

    try:
        plant_mapping_df = fetch_plant_mapping_df()
    except Exception as exc:
        errors.append(f"Plant mapping load failed: {exc}")

    try:
        train_mapping_df = fetch_train_mapping_df()
    except Exception as exc:
        errors.append(f"Train mapping load failed: {exc}")

    try:
        woodmac_df = fetch_woodmac_capacity_raw_data(country_mapping_df)
    except Exception as exc:
        errors.append(f"WoodMac nominal capacity load failed: {exc}")

    try:
        train_capacity_df = fetch_woodmac_train_capacity_raw_data(
            country_mapping_df,
            plant_mapping_df,
            train_mapping_df,
        )
    except Exception as exc:
        errors.append(f"WoodMac train capacity load failed: {exc}")

    try:
        ea_capacity_df = fetch_ea_capacity_raw_data(
            country_mapping_df,
            plant_mapping_df,
            train_mapping_df,
        )
    except Exception as exc:
        errors.append(f"Energy Aspects liquefaction capacity load failed: {exc}")

    try:
        metadata["woodmac"] = fetch_woodmac_capacity_metadata()
    except Exception as exc:
        errors.append(f"WoodMac nominal capacity metadata load failed: {exc}")

    try:
        metadata["ea"] = fetch_ea_capacity_metadata()
    except Exception as exc:
        errors.append(f"Energy Aspects liquefaction metadata load failed: {exc}")

    ea_country_scope_df = pd.DataFrame(columns=["month", "country_name", "total_mmtpa"])
    if not ea_capacity_df.empty:
        ea_country_scope_df = ea_capacity_df.rename(
            columns={"capacity_mtpa": "total_mmtpa"}
        )[["month", "country_name", "total_mmtpa"]]

    available_countries = get_available_countries([woodmac_df, ea_country_scope_df])
    refresh_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = " | ".join(errors) if errors else None

    return (
        _serialize_dataframe(woodmac_df),
        _serialize_dataframe(train_capacity_df),
        _serialize_dataframe(ea_capacity_df),
        available_countries,
        refresh_timestamp,
        error_message,
        metadata,
    )


@callback(
    Output("capacity-page-country-dropdown", "options"),
    Output("capacity-page-country-dropdown", "value"),
    Input("capacity-page-country-options-store", "data"),
    State("capacity-page-country-dropdown", "value"),
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


@callback(
    Output("capacity-page-date-range", "min_date_allowed"),
    Output("capacity-page-date-range", "max_date_allowed"),
    Output("capacity-page-date-range", "start_date"),
    Output("capacity-page-date-range", "end_date"),
    Input("capacity-page-woodmac-data-store", "data"),
    Input("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-date-range", "start_date"),
    State("capacity-page-date-range", "end_date"),
)
def update_capacity_date_range(woodmac_data, ea_capacity_data, current_start_date, current_end_date):
    woodmac_raw_df = _deserialize_dataframe(woodmac_data)
    ea_raw_df = _deserialize_dataframe(ea_capacity_data)
    ea_scope_df = pd.DataFrame(columns=["month", "country_name", "total_mmtpa"])
    if not ea_raw_df.empty:
        ea_scope_df = ea_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
            ["month", "country_name", "total_mmtpa"]
        ]

    min_date, max_date = _get_date_bounds([woodmac_raw_df, ea_scope_df])
    if min_date is None or max_date is None:
        return None, None, None, None

    normalized_min = _normalize_month_date(min_date)
    normalized_max = _normalize_month_date(max_date)
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


@callback(
    Output("capacity-page-refresh-indicator", "children"),
    Output("capacity-page-meta-indicator", "children"),
    Input("capacity-page-refresh-timestamp-store", "data"),
    Input("capacity-page-country-options-store", "data"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
)
def update_capacity_status(refresh_timestamp, available_countries, start_date, end_date):
    refresh_text = (
        f"Last refreshed: {refresh_timestamp}"
        if refresh_timestamp
        else "Last refreshed: waiting for data"
    )
    if start_date and end_date:
        start_label = _normalize_month_date(start_date).strftime("%Y-%m")
        end_label = _normalize_month_date(end_date).strftime("%Y-%m")
        range_text = f"Showing {start_label} to {end_label}."
    else:
        range_text = "Using the latest Woodmac and Energy Aspects comparison snapshots."

    meta_text = (
        f"{len(available_countries or []):,} source countries available. {range_text}"
        if available_countries
        else range_text
    )
    return refresh_text, meta_text


@callback(
    Output("capacity-page-load-error-banner", "children"),
    Input("capacity-page-load-error-store", "data"),
)
def update_capacity_error_banner(error_message):
    if not error_message:
        return html.Div()

    return html.Div(error_message, className="balance-error-banner")


@callback(
    Output("capacity-page-woodmac-summary", "children"),
    Output("capacity-page-woodmac-chart", "figure"),
    Output("capacity-page-woodmac-table-container", "children"),
    Input("capacity-page-woodmac-data-store", "data"),
    Input("capacity-page-metadata-store", "data"),
    Input("capacity-page-country-dropdown", "value"),
    Input("capacity-page-other-country-mode", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
)
def render_capacity_table(
    woodmac_data,
    metadata,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
):
    woodmac_raw_df = _filter_by_date_range(
        _deserialize_dataframe(woodmac_data),
        start_date,
        end_date,
    )

    available_countries = get_available_countries([woodmac_raw_df])
    resolved_countries = _resolve_selected_countries(
        available_countries,
        selected_countries,
    )

    woodmac_matrix = _rename_total_column(
        build_export_flow_matrix(
            woodmac_raw_df,
            resolved_countries,
            other_countries_mode,
        )
    )

    woodmac_summary = _build_section_summary(
        woodmac_raw_df,
        woodmac_matrix,
        other_countries_mode,
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

    woodmac_chart = _create_capacity_country_area_chart(woodmac_matrix)
    woodmac_table = _create_capacity_table("capacity-page-woodmac-table", woodmac_matrix)
    return woodmac_summary, woodmac_chart, woodmac_table


@callback(
    Output("capacity-page-ea-summary", "children"),
    Output("capacity-page-ea-chart", "figure"),
    Output("capacity-page-ea-table-container", "children"),
    Input("capacity-page-ea-capacity-data-store", "data"),
    Input("capacity-page-metadata-store", "data"),
    Input("capacity-page-country-dropdown", "value"),
    Input("capacity-page-other-country-mode", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
)
def render_ea_capacity_table(
    ea_capacity_data,
    metadata,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
):
    ea_schedule_df = _build_ea_capacity_schedule(
        _deserialize_dataframe(ea_capacity_data),
        start_date,
        end_date,
    )

    available_countries = get_available_countries([ea_schedule_df])
    resolved_countries = _resolve_selected_countries(
        available_countries,
        selected_countries,
    )

    ea_matrix = _rename_total_column(
        build_export_flow_matrix(
            ea_schedule_df,
            resolved_countries,
            other_countries_mode,
        )
    )

    ea_summary = _build_section_summary(
        ea_schedule_df,
        ea_matrix,
        other_countries_mode,
        _build_ea_capacity_metadata_lines(metadata, ea_schedule_df),
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
        title_prefix="Cumulative Scheduled LNG Capacity by Country (MTPA)",
        y_axis_title="Scheduled Capacity (MTPA)",
    )
    ea_table = _create_capacity_table("capacity-page-ea-table", ea_matrix)
    return ea_summary, ea_chart, ea_table


@callback(
    Output("capacity-page-train-change-summary", "children"),
    Output("capacity-page-train-change-table-container", "children"),
    Input("capacity-page-train-capacity-data-store", "data"),
    Input("capacity-page-ea-capacity-data-store", "data"),
    Input("capacity-page-metadata-store", "data"),
    Input("capacity-page-country-dropdown", "value"),
    Input("capacity-page-other-country-mode", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
    Input("capacity-page-train-change-time-view", "value"),
    Input("capacity-page-train-change-view-mode", "value"),
    Input("capacity-page-train-change-expanded-country-store", "data"),
    Input("capacity-page-train-change-expanded-plant-store", "data"),
)
def render_train_capacity_change_table(
    train_capacity_data,
    ea_capacity_data,
    metadata,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    detail_view,
    expanded_country_groups,
    expanded_plant_groups,
):
    train_raw_df = _deserialize_dataframe(train_capacity_data)
    ea_raw_df = _deserialize_dataframe(ea_capacity_data)

    if train_raw_df.empty and ea_raw_df.empty:
        empty_df = pd.DataFrame()
        return (
            _build_train_change_summary(
                empty_df,
                empty_df,
                metadata,
                time_view=time_view,
                detail_view=detail_view,
                visible_row_count=0,
            ),
            _create_train_change_table("capacity-page-train-change-table", empty_df),
        )

    country_scope_frames = []
    if not train_raw_df.empty:
        country_scope_frames.append(
            train_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
                ["month", "country_name", "total_mmtpa"]
            ]
        )
    if not ea_raw_df.empty:
        country_scope_frames.append(
            ea_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
                ["month", "country_name", "total_mmtpa"]
            ]
        )
    available_countries = get_available_countries(country_scope_frames)
    resolved_countries = _resolve_selected_countries(
        available_countries,
        selected_countries,
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
    hierarchical_df = _build_train_change_hierarchical_rows(
        woodmac_change_df,
        ea_change_df,
        expanded_country_groups,
        expanded_plant_groups,
        time_view=time_view,
        detail_view=detail_view,
    )

    return (
        _build_train_change_summary(
            woodmac_change_df,
            ea_change_df,
            metadata,
            time_view=time_view,
            detail_view=detail_view,
            visible_row_count=len(hierarchical_df),
        ),
        _create_train_change_table("capacity-page-train-change-table", hierarchical_df),
    )


@callback(
    Output("capacity-page-train-timeline-summary", "children"),
    Output("capacity-page-train-timeline-table-container", "children"),
    Input("capacity-page-train-capacity-data-store", "data"),
    Input("capacity-page-ea-capacity-data-store", "data"),
    Input("capacity-page-country-dropdown", "value"),
    Input("capacity-page-other-country-mode", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
)
def render_train_timeline_table(
    train_capacity_data,
    ea_capacity_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
):
    train_raw_df = _deserialize_dataframe(train_capacity_data)
    ea_raw_df = _deserialize_dataframe(ea_capacity_data)

    if train_raw_df.empty and ea_raw_df.empty:
        empty_df = pd.DataFrame()
        return (
            _build_train_timeline_summary(empty_df),
            _create_train_timeline_table("capacity-page-train-timeline-table", empty_df),
        )

    country_scope_frames = []
    if not train_raw_df.empty:
        country_scope_frames.append(
            train_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
                ["month", "country_name", "total_mmtpa"]
            ]
        )
    if not ea_raw_df.empty:
        country_scope_frames.append(
            ea_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
                ["month", "country_name", "total_mmtpa"]
            ]
        )
    available_countries = get_available_countries(country_scope_frames)
    resolved_countries = _resolve_selected_countries(
        available_countries,
        selected_countries,
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
    timeline_df = _build_train_timeline_df(woodmac_change_df, ea_change_df)

    return (
        _build_train_timeline_summary(timeline_df),
        _create_train_timeline_table("capacity-page-train-timeline-table", timeline_df),
    )


@callback(
    Output("capacity-page-unmapped-plant-summary", "children"),
    Output("capacity-page-unmapped-plant-table", "data"),
    Output("capacity-page-unmapped-train-summary", "children"),
    Output("capacity-page-unmapped-train-table", "data"),
    Input("capacity-page-train-capacity-data-store", "data"),
    Input("capacity-page-ea-capacity-data-store", "data"),
    Input("capacity-page-country-dropdown", "value"),
    Input("capacity-page-other-country-mode", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
)
def render_unmapped_mapping_tables(
    train_capacity_data,
    ea_capacity_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
):
    train_raw_df = _deserialize_dataframe(train_capacity_data)
    ea_raw_df = _deserialize_dataframe(ea_capacity_data)

    if train_raw_df.empty and ea_raw_df.empty:
        empty_df = pd.DataFrame()
        return (
            _build_unmapped_plant_summary(empty_df),
            [],
            _build_unmapped_train_summary(empty_df),
            [],
        )

    country_scope_frames = []
    if not train_raw_df.empty:
        country_scope_frames.append(
            train_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
                ["month", "country_name", "total_mmtpa"]
            ]
        )
    if not ea_raw_df.empty:
        country_scope_frames.append(
            ea_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
                ["month", "country_name", "total_mmtpa"]
            ]
        )

    available_countries = get_available_countries(country_scope_frames)
    resolved_countries = _resolve_selected_countries(
        available_countries,
        selected_countries,
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
    alias_df = _build_unmapped_plant_alias_df(woodmac_change_df, ea_change_df)
    train_alias_df = _build_unmapped_train_alias_df(woodmac_change_df, ea_change_df)

    return (
        _build_unmapped_plant_summary(alias_df),
        alias_df.to_dict("records"),
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
    Output("capacity-page-plant-mapping-save-trigger", "data"),
    Output("capacity-page-unmapped-plant-message", "children"),
    Input("capacity-page-save-plant-mappings-button", "n_clicks"),
    State("capacity-page-unmapped-plant-table", "data"),
    prevent_initial_call=True,
)
def save_unmapped_plant_mappings(n_clicks, table_data):
    if not n_clicks:
        raise PreventUpdate

    alias_df = pd.DataFrame(table_data or [])
    if alias_df.empty:
        return no_update, html.Div(
            "No unmapped aliases are currently available to save.",
            className="balance-metadata-row",
        )

    save_columns = [
        "country_name",
        "provider",
        "source_field",
        "source_name",
        "scope_hint",
        "component_hint",
        "plant_name",
    ]
    for column_name in save_columns:
        if column_name not in alias_df.columns:
            alias_df[column_name] = None

    for column_name in save_columns:
        alias_df[column_name] = alias_df[column_name].map(
            lambda value: (
                None
                if pd.isna(value) or str(value).strip() == ""
                else " ".join(str(value).strip().split())
            )
        )

    save_df = alias_df.dropna(
        subset=["country_name", "provider", "source_field", "source_name", "plant_name"]
    ).copy()
    save_df = save_df[save_columns].drop_duplicates(
        subset=["country_name", "provider", "source_field", "source_name"],
        keep="last",
    )

    if save_df.empty:
        return no_update, html.Div(
            "Fill at least one plant_name before saving.",
            style={"color": "#9a3412", "fontSize": "12px", "fontWeight": "600"},
        )

    table_ref = f'"{DB_SCHEMA}"."mapping_plant_name"'
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_ref} (
                        country_name TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        source_field TEXT NOT NULL,
                        source_name TEXT NOT NULL,
                        scope_hint TEXT,
                        component_hint TEXT,
                        plant_name TEXT NOT NULL
                    )
                    """
                )
            )

            for row in save_df.to_dict("records"):
                connection.execute(
                    text(
                        f"""
                        DELETE FROM {table_ref}
                        WHERE country_name = :country_name
                          AND provider = :provider
                          AND source_field = :source_field
                          AND source_name = :source_name
                        """
                    ),
                    row,
                )
                connection.execute(
                    text(
                        f"""
                        INSERT INTO {table_ref} (
                            country_name,
                            provider,
                            source_field,
                            source_name,
                            scope_hint,
                            component_hint,
                            plant_name
                        ) VALUES (
                            :country_name,
                            :provider,
                            :source_field,
                            :source_name,
                            :scope_hint,
                            :component_hint,
                            :plant_name
                        )
                        """
                    ),
                    row,
                )

        save_timestamp = dt.datetime.utcnow().isoformat()
        return save_timestamp, html.Div(
            f"Saved {len(save_df):,} plant mappings to at_lng.mapping_plant_name. The comparison data has been reloaded.",
            style={"color": "#166534", "fontSize": "12px", "fontWeight": "600"},
        )
    except Exception as exc:
        return no_update, html.Div(
            f"Plant mapping save failed: {exc}",
            style={"color": "#991b1b", "fontSize": "12px", "fontWeight": "600"},
        )


@callback(
    Output("capacity-page-train-mapping-save-trigger", "data"),
    Output("capacity-page-unmapped-train-message", "children"),
    Input("capacity-page-save-train-mappings-button", "n_clicks"),
    State("capacity-page-unmapped-train-table", "data"),
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
        "provider",
        "parent_source_field",
        "parent_source_name",
        "source_field",
        "source_name",
        "scope_hint",
        "component_hint",
        "train",
        "allocation_share",
        "notes",
    ]
    for column_name in save_columns:
        if column_name not in alias_df.columns:
            alias_df[column_name] = None

    text_columns = [
        "country_name",
        "plant_name",
        "provider",
        "parent_source_field",
        "parent_source_name",
        "source_field",
        "source_name",
        "scope_hint",
        "component_hint",
        "notes",
    ]
    for column_name in text_columns:
        alias_df[column_name] = alias_df[column_name].map(_clean_mapping_text_value)

    alias_df["train"] = alias_df["train"].map(_clean_positive_train_value)
    alias_df["allocation_share"] = alias_df["allocation_share"].map(_clean_allocation_share_value)

    save_df = alias_df.dropna(
        subset=[
            "country_name",
            "plant_name",
            "provider",
            "parent_source_field",
            "parent_source_name",
            "source_field",
            "source_name",
            "train",
            "allocation_share",
        ]
    ).copy()
    save_df["train"] = save_df["train"].astype(int)
    save_df = save_df[save_columns].drop_duplicates(
        subset=[
            "country_name",
            "plant_name",
            "provider",
            "parent_source_field",
            "parent_source_name",
            "source_field",
            "source_name",
            "train",
        ],
        keep="last",
    )

    if save_df.empty:
        return no_update, html.Div(
            "Fill at least one canonical train before saving.",
            style={"color": "#9a3412", "fontSize": "12px", "fontWeight": "600"},
        )

    table_ref = f'"{DB_SCHEMA}"."mapping_plant_train_name"'
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_ref} (
                        country_name TEXT NOT NULL,
                        plant_name TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        parent_source_field TEXT NOT NULL,
                        parent_source_name TEXT NOT NULL,
                        source_field TEXT NOT NULL,
                        source_name TEXT NOT NULL,
                        scope_hint TEXT,
                        component_hint TEXT,
                        train INTEGER NOT NULL,
                        allocation_share DOUBLE PRECISION,
                        notes TEXT
                    )
                    """
                )
            )

            unique_raw_keys = (
                save_df[
                    [
                        "country_name",
                        "plant_name",
                        "provider",
                        "parent_source_field",
                        "parent_source_name",
                        "source_field",
                        "source_name",
                    ]
                ]
                .drop_duplicates()
                .to_dict("records")
            )
            for row in unique_raw_keys:
                connection.execute(
                    text(
                        f"""
                        DELETE FROM {table_ref}
                        WHERE country_name = :country_name
                          AND plant_name = :plant_name
                          AND provider = :provider
                          AND parent_source_field = :parent_source_field
                          AND parent_source_name = :parent_source_name
                          AND source_field = :source_field
                          AND source_name = :source_name
                        """
                    ),
                    row,
                )

            for row in save_df.to_dict("records"):
                connection.execute(
                    text(
                        f"""
                        INSERT INTO {table_ref} (
                            country_name,
                            plant_name,
                            provider,
                            parent_source_field,
                            parent_source_name,
                            source_field,
                            source_name,
                            scope_hint,
                            component_hint,
                            train,
                            allocation_share,
                            notes
                        ) VALUES (
                            :country_name,
                            :plant_name,
                            :provider,
                            :parent_source_field,
                            :parent_source_name,
                            :source_field,
                            :source_name,
                            :scope_hint,
                            :component_hint,
                            :train,
                            :allocation_share,
                            :notes
                        )
                        """
                    ),
                    row,
                )

        save_timestamp = dt.datetime.utcnow().isoformat()
        return save_timestamp, html.Div(
            f"Saved {len(save_df):,} train mappings to at_lng.mapping_plant_train_name. The comparison data has been reloaded.",
            style={"color": "#166534", "fontSize": "12px", "fontWeight": "600"},
        )
    except Exception as exc:
        return no_update, html.Div(
            f"Train mapping save failed: {exc}",
            style={"color": "#991b1b", "fontSize": "12px", "fontWeight": "600"},
        )


@callback(
    Output("capacity-page-train-change-expanded-country-store", "data"),
    Output("capacity-page-train-change-expanded-plant-store", "data"),
    Input({"type": "capacity-train-change-expandable-table", "index": ALL}, "active_cell"),
    State({"type": "capacity-train-change-expandable-table", "index": ALL}, "data"),
    State("capacity-page-train-change-expanded-country-store", "data"),
    State("capacity-page-train-change-expanded-plant-store", "data"),
    prevent_initial_call=True,
)
def toggle_train_change_expansion(
    active_cells,
    table_data_list,
    expanded_country_groups,
    expanded_plant_groups,
):
    if not active_cells or not any(active_cells):
        raise PreventUpdate

    active_cell = next((cell for cell in active_cells if cell), None)
    table_data = next((data for data in table_data_list if data), None)
    if not active_cell or not table_data:
        raise PreventUpdate

    clicked_row = table_data[active_cell["row"]]
    row_type = clicked_row.get("Type")
    column_id = active_cell.get("column_id")
    expanded_country_groups = list(expanded_country_groups or [])
    expanded_plant_groups = list(expanded_plant_groups or [])

    if row_type == "country" and column_id in {"Effective Date", "Country"}:
        country_key = clicked_row.get("country_group_key")
        if not country_key:
            raise PreventUpdate

        if country_key in expanded_country_groups:
            expanded_country_groups.remove(country_key)
            expanded_plant_groups = [
                plant_key
                for plant_key in expanded_plant_groups
                if not plant_key.startswith(f"{country_key}|")
            ]
        else:
            expanded_country_groups.append(country_key)

        return expanded_country_groups, expanded_plant_groups

    if row_type == "plant" and column_id == "Plant":
        plant_key = clicked_row.get("plant_group_key")
        if not plant_key:
            raise PreventUpdate

        if plant_key in expanded_plant_groups:
            expanded_plant_groups.remove(plant_key)
        else:
            expanded_plant_groups.append(plant_key)

        return expanded_country_groups, expanded_plant_groups

    raise PreventUpdate


@callback(
    Output("capacity-page-train-change-expanded-country-store", "data", allow_duplicate=True),
    Output("capacity-page-train-change-expanded-plant-store", "data", allow_duplicate=True),
    Input("capacity-page-train-change-time-view", "value"),
    Input("capacity-page-train-change-view-mode", "value"),
    Input("capacity-page-train-capacity-data-store", "data"),
    Input("capacity-page-ea-capacity-data-store", "data"),
    Input("capacity-page-country-dropdown", "value"),
    Input("capacity-page-other-country-mode", "value"),
    Input("capacity-page-date-range", "start_date"),
    Input("capacity-page-date-range", "end_date"),
    prevent_initial_call=True,
)
def set_train_change_view_mode(
    time_view,
    view_mode,
    train_capacity_data,
    ea_capacity_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
):
    return [], []


def _build_filtered_matrix_for_export(
    source_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
) -> pd.DataFrame:
    raw_df = _filter_by_date_range(
        _deserialize_dataframe(source_data),
        start_date,
        end_date,
    )
    return _rename_total_column(
        build_export_flow_matrix(
            raw_df,
            selected_countries,
            other_countries_mode,
        )
    )


def _build_filtered_ea_matrix_for_export(
    source_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
) -> pd.DataFrame:
    schedule_df = _build_ea_capacity_schedule(
        _deserialize_dataframe(source_data),
        start_date,
        end_date,
    )
    return _rename_total_column(
        build_export_flow_matrix(
            schedule_df,
            selected_countries,
            other_countries_mode,
        )
    )


@callback(
    Output("capacity-page-download-woodmac-excel", "data"),
    Input("capacity-page-export-woodmac-button", "n_clicks"),
    State("capacity-page-woodmac-data-store", "data"),
    State("capacity-page-country-dropdown", "value"),
    State("capacity-page-other-country-mode", "value"),
    State("capacity-page-date-range", "start_date"),
    State("capacity-page-date-range", "end_date"),
    prevent_initial_call=True,
)
def export_woodmac_capacity_excel(
    n_clicks,
    woodmac_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
):
    if not n_clicks:
        raise PreventUpdate

    export_df = _build_filtered_matrix_for_export(
        woodmac_data,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
    )
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
    prevent_initial_call=True,
)
def export_ea_capacity_excel(
    n_clicks,
    ea_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
):
    if not n_clicks:
        raise PreventUpdate

    export_df = _build_filtered_ea_matrix_for_export(
        ea_data,
        selected_countries,
        other_countries_mode,
        start_date,
        end_date,
    )
    if export_df.empty:
        raise PreventUpdate

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Energy_Aspects_Scheduled_Capacity_{timestamp}.xlsx"
    return dcc.send_bytes(
        _export_matrix_to_excel_bytes(export_df, "Scheduled Capacity"),
        filename,
    )


@callback(
    Output("capacity-page-download-train-change-excel", "data"),
    Input("capacity-page-export-train-change-button", "n_clicks"),
    State("capacity-page-train-capacity-data-store", "data"),
    State("capacity-page-ea-capacity-data-store", "data"),
    State("capacity-page-country-dropdown", "value"),
    State("capacity-page-other-country-mode", "value"),
    State("capacity-page-date-range", "start_date"),
    State("capacity-page-date-range", "end_date"),
    State("capacity-page-train-change-time-view", "value"),
    State("capacity-page-train-change-view-mode", "value"),
    State("capacity-page-train-change-expanded-country-store", "data"),
    State("capacity-page-train-change-expanded-plant-store", "data"),
    prevent_initial_call=True,
)
def export_train_change_excel(
    n_clicks,
    train_capacity_data,
    ea_capacity_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
    time_view,
    detail_view,
    expanded_country_groups,
    expanded_plant_groups,
):
    if not n_clicks:
        raise PreventUpdate

    train_raw_df = _deserialize_dataframe(train_capacity_data)
    ea_raw_df = _deserialize_dataframe(ea_capacity_data)

    country_scope_frames = []
    if not train_raw_df.empty:
        country_scope_frames.append(
            train_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
                ["month", "country_name", "total_mmtpa"]
            ]
        )
    if not ea_raw_df.empty:
        country_scope_frames.append(
            ea_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
                ["month", "country_name", "total_mmtpa"]
            ]
        )
    available_countries = get_available_countries(country_scope_frames)
    resolved_countries = _resolve_selected_countries(available_countries, selected_countries)

    woodmac_change_df = _build_train_change_log(
        train_raw_df, resolved_countries, other_countries_mode, start_date, end_date
    )
    ea_change_df = _build_ea_change_log(
        ea_raw_df, resolved_countries, other_countries_mode, start_date, end_date
    )
    hierarchical_df = _build_train_change_hierarchical_rows(
        woodmac_change_df,
        ea_change_df,
        expanded_country_groups,
        expanded_plant_groups,
        time_view=time_view,
        detail_view=detail_view,
    )

    if hierarchical_df.empty:
        raise PreventUpdate

    internal_cols = {"Type", "country_group_key", "plant_group_key", "month_group_key", "month_group_end"}
    export_df = hierarchical_df.drop(columns=[c for c in internal_cols if c in hierarchical_df.columns])

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
    State("capacity-page-country-dropdown", "value"),
    State("capacity-page-other-country-mode", "value"),
    State("capacity-page-date-range", "start_date"),
    State("capacity-page-date-range", "end_date"),
    prevent_initial_call=True,
)
def export_train_timeline_excel(
    n_clicks,
    train_capacity_data,
    ea_capacity_data,
    selected_countries,
    other_countries_mode,
    start_date,
    end_date,
):
    if not n_clicks:
        raise PreventUpdate

    train_raw_df = _deserialize_dataframe(train_capacity_data)
    ea_raw_df = _deserialize_dataframe(ea_capacity_data)

    country_scope_frames = []
    if not train_raw_df.empty:
        country_scope_frames.append(
            train_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
                ["month", "country_name", "total_mmtpa"]
            ]
        )
    if not ea_raw_df.empty:
        country_scope_frames.append(
            ea_raw_df.rename(columns={"capacity_mtpa": "total_mmtpa"})[
                ["month", "country_name", "total_mmtpa"]
            ]
        )
    available_countries = get_available_countries(country_scope_frames)
    resolved_countries = _resolve_selected_countries(available_countries, selected_countries)

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
    export_df = _build_train_timeline_df(woodmac_change_df, ea_change_df)

    if export_df.empty:
        raise PreventUpdate

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Train_Timeline_{timestamp}.xlsx"
    return dcc.send_bytes(
        _export_matrix_to_excel_bytes(export_df, "Train Timeline"),
        filename,
    )
