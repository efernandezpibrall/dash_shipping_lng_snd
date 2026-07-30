from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from threading import Lock

from dash import html, dcc, callback, Output, Input, State, ALL, MATCH
import dash_ag_grid as dag
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
from dash.exceptions import PreventUpdate
from sqlalchemy import text

from utils.database import engine

logger = logging.getLogger(__name__)

############################################ Style Constants ###################################################

CONTRACTS_SECTION_STYLE = {
    "background": "#ffffff",
    "border": "1px solid #e5e7eb",
    "borderRadius": "8px",
    "padding": "12px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.05)",
}

CONTRACTS_GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}
AG_GRID_THEME = "ag-theme-alpine"

CONTRACTS_AG_GRID_DEFAULT_COL_DEF = {
    "sortable": True,
    "filter": False,
    "resizable": True,
    "wrapHeaderText": True,
    "autoHeaderHeight": True,
    "suppressHeaderMenuButton": True,
    "suppressHeaderFilterButton": True,
}

CONTRACTS_AG_GRID_OPTIONS = {
    "animateRows": False,
    "ensureDomOrder": True,
    "enableCellTextSelection": True,
    "suppressDragLeaveHidesColumns": True,
    "rowHeight": 30,
    "headerHeight": 32,
}

############################################ Data Loading Functions ###################################################

def normalize_flex_flag(value):
    """Normalize WoodMac flexibility flags to canonical Y/N/Unknown values."""
    if pd.isna(value):
        return 'Unknown'

    normalized = str(value).strip().upper()
    if normalized in {'Y', 'YES', 'TRUE', 'T', '1'}:
        return 'Y'
    if normalized in {'N', 'NO', 'FALSE', 'F', '0'}:
        return 'N'
    return 'Unknown'


CONTRACTS_DATA_COLUMNS = [
    'id_contract', 'contract_name', 'id_contract_primary', 'contract_name_primary',
    'contract_type', 'cargo_basis', 'contract_pricing_type', 'contract_date_signed',
    'contract_year_signed', 'contract_date_start', 'contract_date_end',
    'company_name_seller', 'country_name_hq_company_seller', 'company_category_seller',
    'company_name_buyer', 'country_name_hq_company_buyer', 'company_category_buyer',
    'country_name_source', 'id_lng_plant_source', 'lng_plant_name_source',
    'id_lng_project', 'lng_project_name', 'flexibility_source', 'is_source_flexible',
    'country_name_delivery', 'flexibility_delivery', 'is_destination_flexible',
    'max_acq_volume', 'max_acq_volume_unit', 'contract_note', 'equity_third_party',
    'destination_flexible_vs_end_users', 'indexation_category', 'indexation_point'
]

CONTRACT_DETAIL_DATE_COLUMNS = {
    'contract_date_signed',
    'contract_date_start',
    'contract_date_end',
}

ANNUAL_DEMAND_DATA_COLUMNS = [
    'id_contract', 'contract_name', 'year', 'acq_volume__mmtpa', 'metric_name',
    'metric_value', 'unit', 'company_name_seller', 'company_name_buyer',
    'country_name_source', 'country_name_delivery', 'cargo_basis',
    'contract_type', 'contract_pricing_type'
]

PRICE_ASSUMPTIONS_COLUMNS = [
    'id_contract', 'contract_name', 'indexation_category', 'indexation_point',
    'oil_pricing_structure', 'slope', 'intercept', 'lower_inflection',
    'slope_lower', 'intercept_lower', 'upper_inflection', 'slope_upper',
    'intercept_upper', 'weighting', 'gas_pricing_structure', 'fixed_fee',
    'transport_tariff', 'regas_tariff', 'linkage_percent',
    'oil_price_in_signed_year', 'normalized_slope',
    'oil_indexed_shipping_cost', 'gas_indexed_shipping_cost', 'other_costs'
]

PRICE_FORMULA_COLUMNS = [
    'id_contract', 'contract_name', 'indexation_point', 'pricing_structure',
    'indexation_category', 'index_pricing_point', 'lower_bound', 'upper_bound',
    'coefficient_type', 'coefficient_value', 'lag_months', 'average_months',
    'weighting'
]


def load_contracts_data():
    """Load main contracts data from WoodMac tables"""
    query = """
    SELECT 
        id_contract,
        contract_name,
        id_contract_primary,
        contract_name_primary,
        contract_type,
        cargo_basis,
        contract_pricing_type,
        contract_date_signed,
        contract_year_signed,
        contract_date_start,
        contract_date_end,
        COALESCE(company_name_seller, 'Unknown') as company_name_seller,
        COALESCE(country_name_hq_company_seller, 'Unknown') as country_name_hq_company_seller,
        COALESCE(company_category_seller, 'Unknown') as company_category_seller,
        COALESCE(company_name_buyer, 'Unknown') as company_name_buyer,
        COALESCE(country_name_hq_company_buyer, 'Unknown') as country_name_hq_company_buyer,
        COALESCE(company_category_buyer, 'Unknown') as company_category_buyer,
        COALESCE(country_name_source, 'Unknown') as country_name_source,
        id_lng_plant_source,
        COALESCE(lng_plant_name_source, 'Unknown') as lng_plant_name_source,
        id_lng_project,
        COALESCE(lng_project_name, 'Unknown') as lng_project_name,
        flexibility_source,
        COALESCE(is_source_flexible, 'Unknown') as is_source_flexible,
        COALESCE(country_name_delivery, 'Unknown') as country_name_delivery,
        flexibility_delivery,
        COALESCE(is_destination_flexible, 'Unknown') as is_destination_flexible,
        max_acq_volume,
        max_acq_volume_unit,
        contract_note,
        equity_third_party,
        destination_flexible_vs_end_users,
        indexation_category,
        indexation_point
    FROM at_lng.woodmac_lng_contract
    WHERE id_contract IS NOT NULL
    """
    try:
        df = pd.read_sql(query, engine)
        
        # Fill NA values for most columns, but preserve indexation_category
        cols_to_fill = [col for col in df.columns if col not in ['indexation_category', 'indexation_point']]
        df[cols_to_fill] = df[cols_to_fill].fillna('Unknown')
        
        # Extra safety: ensure empty strings are also mapped to 'Unknown' (except indexation columns)
        for col in ['company_name_seller', 'company_name_buyer', 'country_name_source', 'country_name_delivery', 'cargo_basis', 'contract_type', 'contract_pricing_type']:
            if col in df.columns:
                df.loc[df[col] == '', col] = 'Unknown'
                df.loc[df[col].isna(), col] = 'Unknown'

        for col in ['is_source_flexible', 'is_destination_flexible']:
            if col in df.columns:
                df[col] = df[col].apply(normalize_flex_flag)
        
        return df
    except Exception as e:
        print(f"Error loading contracts data: {e}")
        return pd.DataFrame(columns=CONTRACTS_DATA_COLUMNS)

def load_annual_demand_data():
    """Load annual contracted demand data with contract details"""
    query = """
    SELECT 
        d.id_contract,
        d.contract_name,
        d.year,
        d.acq_volume__mmtpa,
        d.metric_name,
        d.metric_value,
        d.unit,
        COALESCE(c.company_name_seller, 'Unknown') as company_name_seller,
        COALESCE(c.company_name_buyer, 'Unknown') as company_name_buyer,
        COALESCE(c.country_name_source, 'Unknown') as country_name_source,
        COALESCE(c.country_name_delivery, 'Unknown') as country_name_delivery,
        COALESCE(c.cargo_basis, 'Unknown') as cargo_basis,
        COALESCE(c.contract_type, 'Unknown') as contract_type,
        COALESCE(c.contract_pricing_type, 'Unknown') as contract_pricing_type
    FROM at_lng.woodmac_lng_contract_annual_contracted_demand_mta d
    LEFT JOIN at_lng.woodmac_lng_contract c ON d.id_contract = c.id_contract
    WHERE d.id_contract IS NOT NULL
    """
    try:
        df = pd.read_sql(query, engine)
        # Additional null safety - ensure no NaN values remain
        df = df.fillna('Unknown')
        
        # Extra safety: ensure empty strings are also mapped to 'Unknown'
        for col in ['company_name_seller', 'company_name_buyer', 'country_name_source', 'country_name_delivery', 'cargo_basis', 'contract_type', 'contract_pricing_type']:
            if col in df.columns:
                df.loc[df[col] == '', col] = 'Unknown'
                df.loc[df[col].isna(), col] = 'Unknown'
        
        return df
    except Exception as e:
        print(f"Error loading annual demand data: {e}")
        return pd.DataFrame(columns=ANNUAL_DEMAND_DATA_COLUMNS)

def load_price_assumptions_data():
    """Load price assumptions data"""
    query = """
    SELECT 
        id_contract,
        contract_name,
        indexation_category,
        indexation_point,
        oil_pricing_structure,
        slope,
        intercept,
        lower_inflection,
        slope_lower,
        intercept_lower,
        upper_inflection,
        slope_upper,
        intercept_upper,
        weighting,
        gas_pricing_structure,
        fixed_fee,
        transport_tariff,
        regas_tariff,
        linkage_percent,
        oil_price_in_signed_year,
        normalized_slope,
        oil_indexed_shipping_cost,
        gas_indexed_shipping_cost,
        other_costs
    FROM at_lng.woodmac_lng_contract_price_assumptions
    WHERE id_contract IS NOT NULL
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error loading price assumptions data: {e}")
        return pd.DataFrame(columns=PRICE_ASSUMPTIONS_COLUMNS)

def load_price_formula_data():
    """Load price formula data"""
    query = """
    SELECT 
        id_contract,
        contract_name,
        indexation_point,
        pricing_structure,
        indexation_category,
        index_pricing_point,
        lower_bound,
        upper_bound,
        coefficient_type,
        coefficient_value,
        lag_months,
        average_months,
        weighting
    FROM at_lng.woodmac_lng_contract_price_formula
    WHERE id_contract IS NOT NULL
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error loading price formula data: {e}")
        return pd.DataFrame(columns=PRICE_FORMULA_COLUMNS)

############################################ Helper Functions ###################################################

def extract_index_detail(structure, indexation_cat, type='oil', indexation_point=None):
    """Extract specific index detail from pricing structure or indexation category"""
    if not structure and not indexation_cat and not indexation_point:
        return None
    
    # Convert to string and lower case for checking
    structure_str = str(structure).lower() if structure else ""
    cat_str = str(indexation_cat).lower() if indexation_cat else ""
    point_str = str(indexation_point).lower() if indexation_point else ""
    combined = f"{structure_str} {cat_str} {point_str}"
    
    if type == 'oil':
        if 'brent' in combined:
            return 'Brent'
        elif 'jcc' in combined or 'japan crude cocktail' in combined:
            return 'JCC'
        elif 'wti' in combined or 'west texas' in combined:
            return 'WTI'
        elif 'dubai' in combined:
            return 'Dubai'
        elif 'crude' in combined or 'oil' in combined:
            return 'Oil'
    elif type == 'gas':
        if 'henry hub' in combined or 'hhub' in combined or 'hh' in combined:
            return 'Henry Hub'
        elif 'nbp' in combined:
            return 'NBP'
        elif 'ttf' in combined:
            return 'TTF'
        elif 'jkm' in combined:
            return 'JKM'
        elif 'slope' in combined:
            return 'Slope'
        elif 'gas' in combined:
            return 'Others'
    
    return None

def prepare_volume_table_for_display(demand_df, table_type, available_years, expanded_entities=None):
    """Prepare volume data for AG Grid display with expandable source-destination breakdown.
    
    Args:
        demand_df: DataFrame with volume data
        table_type: 'country', 'seller', or 'destination'
        available_years: List of years to display as columns
        expanded_entities: List of expanded entity names
    """
    if demand_df.empty:
        return [], []
    
    expanded_entities = expanded_entities or []
    
    # Define grouping columns based on table type
    if table_type == 'country':
        main_entity_col = 'country_name_source'
        detail_entity_col = 'country_name_delivery'
        entity_name = 'Source Country'
        detail_name = 'Destination'
    elif table_type == 'destination':
        main_entity_col = 'country_name_delivery'
        detail_entity_col = 'country_name_source'
        entity_name = 'Destination Country'
        detail_name = 'Source'
    elif table_type == 'buyer':
        main_entity_col = 'company_name_buyer'
        detail_entity_col = 'country_name_delivery'
        entity_name = 'Buyer Company'
        detail_name = 'Destination'
    else:  # seller
        main_entity_col = 'company_name_seller'
        detail_entity_col = 'country_name_delivery'
        entity_name = 'Seller Company'
        detail_name = 'Destination'
    
    # Check if required columns exist
    required_cols = [main_entity_col, 'year', 'acq_volume__mmtpa', detail_entity_col]
    missing_cols = [col for col in required_cols if col not in demand_df.columns]
    if missing_cols:
        return [], []
    
    
    # Create pivot table for main entities
    main_pivot = demand_df.groupby([main_entity_col, 'year'])['acq_volume__mmtpa'].sum().unstack(fill_value=0).round(2)
    
    # Sort by total volume and take top 14 (leaving room for TOTAL row = 15 rows max)
    total_volumes = main_pivot.sum(axis=1)
    main_pivot = main_pivot.loc[total_volumes.sort_values(ascending=False).index]
    
    # Prepare display data
    filtered_rows = []
    
    for entity in main_pivot.index:
        # Add main entity row with expand/collapse indicator
        entity_row = {main_entity_col: f"▼ {entity}" if entity in expanded_entities else f"▶ {entity}"}
        entity_row[detail_entity_col] = ""  # Empty detail column for main entity
        
        # Add year columns - use the actual year as column name, not string
        for year in available_years:
            if year in main_pivot.columns:
                entity_row[str(year)] = main_pivot.loc[entity, year]
            else:
                entity_row[str(year)] = 0.0
        
        filtered_rows.append(entity_row)
        
        # If entity is expanded, show destination breakdown
        if entity in expanded_entities:
            entity_data = demand_df[demand_df[main_entity_col] == entity]
            
            # Create destination breakdown pivot
            dest_pivot = entity_data.groupby([detail_entity_col, 'year'])['acq_volume__mmtpa'].sum().unstack(fill_value=0).round(2)
            
            # Sort destinations by total volume
            if not dest_pivot.empty:
                dest_totals = dest_pivot.sum(axis=1)
                dest_pivot = dest_pivot.loc[dest_totals.sort_values(ascending=False).index]
                
                for destination in dest_pivot.index:
                    dest_row = {main_entity_col: ""}  # Empty main entity for destinations
                    dest_row[detail_entity_col] = f"    → {destination}"  # Indented destination
                    
                    # Add year columns - use the actual year as column name, not string
                    for year in available_years:
                        if year in dest_pivot.columns:
                            dest_row[str(year)] = dest_pivot.loc[destination, year]
                        else:
                            dest_row[str(year)] = 0.0
                    
                    filtered_rows.append(dest_row)
    
    # Add TOTAL row for both country and seller views
    # Calculate total from original unfiltered data to ensure consistency
    if not demand_df.empty:
        total_row = {main_entity_col: "TOTAL", detail_entity_col: ""}
        
        # Calculate totals for each year from original data (not just top entities)
        original_totals = demand_df.groupby('year')['acq_volume__mmtpa'].sum()
        for year in available_years:
            if year in original_totals.index:
                total_row[str(year)] = round(original_totals[year], 2)
            else:
                total_row[str(year)] = 0.0
        
        filtered_rows.append(total_row)
    
    # Create column definitions
    columns = [
        {'name': entity_name, 'id': main_entity_col, 'type': 'text'},
        {'name': detail_name, 'id': detail_entity_col, 'type': 'text'}
    ]
    columns.extend([
        {'name': str(year), 'id': str(year), 'type': 'numeric', 'precision': 2, 'width': 86, 'minWidth': 70}
        for year in available_years
    ])
    
    return filtered_rows, columns


############################################ Layout Components ###################################################


def _contracts_number_formatter(precision=2):
    return {
        "function": (
            "params.value == null || params.value === '' || params.value === 'N/A' "
            "? (params.value || '') "
            f": Number(params.value).toLocaleString(undefined, {{minimumFractionDigits: {precision}, maximumFractionDigits: {precision}}})"
        )
    }


def _contracts_records(records):
    if not records:
        return []
    return pd.DataFrame(records).where(pd.notna, None).to_dict("records")


def _contracts_date_only(value):
    if pd.isna(value):
        return value

    parsed_value = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed_value):
        return value

    return parsed_value.strftime("%Y-%m-%d")


def _contracts_ag_column_defs(columns, filterable=False, pinned_fields=None, default_width=118):
    pinned_fields = set(pinned_fields or [])
    column_defs = []

    for column in columns:
        field = column.get("id")
        if not field:
            continue

        column_type = column.get("type")
        is_numeric = column_type == "numeric"
        header_name = column.get("name", field)
        precision = column.get("precision", 2)
        width = column.get("width") or (98 if is_numeric else default_width)
        min_width = column.get("minWidth") or (72 if is_numeric else 96)

        ag_column = {
            "headerName": header_name,
            "field": field,
            "sortable": True,
            "filter": "agNumberColumnFilter" if filterable and is_numeric else "agTextColumnFilter" if filterable else False,
            "resizable": True,
            "width": width,
            "minWidth": min_width,
            "cellClass": "fleet-metrics-number-cell" if is_numeric else "fleet-metrics-left-cell",
        }

        if field in pinned_fields:
            ag_column.update({"pinned": "left", "lockPinned": True, "suppressMovable": True})

        if is_numeric:
            ag_column["type"] = "rightAligned"
            ag_column["valueFormatter"] = _contracts_number_formatter(precision)

        column_defs.append(ag_column)

    return column_defs


def _contracts_ag_grid(
    id_value,
    row_data=None,
    column_defs=None,
    height=480,
    filterable=False,
    total_field=None,
    total_color="#2E86C1",
    total_border="#1B4F72",
    extra_class="",
    filename="contracts_table.csv",
):
    default_col_def = {
        **CONTRACTS_AG_GRID_DEFAULT_COL_DEF,
        "filter": filterable,
        "suppressHeaderMenuButton": not filterable,
        "suppressHeaderFilterButton": not filterable,
    }
    grid_options = {
        **CONTRACTS_AG_GRID_OPTIONS,
        "pagination": False,
    }
    grid_kwargs = {
        "id": id_value,
        "rowData": row_data or [],
        "columnDefs": column_defs or [],
        "defaultColDef": default_col_def,
        "dashGridOptions": grid_options,
        "csvExportParams": {"fileName": filename},
        "className": f"{AG_GRID_THEME} fleet-metrics-grid contracts-ag-grid {extra_class}".strip(),
        "style": {"width": "100%", "height": f"{height}px"},
        "dangerously_allow_code": True,
        "exportDataAsCsv": False,
    }

    if total_field:
        grid_kwargs["getRowStyle"] = {
            "styleConditions": [
                {
                    "condition": f"params.data && params.data['{total_field}'] === 'TOTAL'",
                    "style": {
                        "backgroundColor": total_color,
                        "color": "white",
                        "fontWeight": "800",
                        "borderTop": f"2px solid {total_border}",
                    },
                },
            ],
            "defaultStyle": {},
        }
        grid_kwargs["rowClassRules"] = {
            "contracts-total-row": f"params.data && params.data['{total_field}'] === 'TOTAL'",
        }

    return dag.AgGrid(**grid_kwargs)


def _contracts_volume_grid(table_id, display_data, columns, entity_field, filename, total_color="#2E86C1", total_border="#1B4F72"):
    return _contracts_ag_grid(
        id_value=table_id,
        row_data=_contracts_records(display_data),
        column_defs=_contracts_ag_column_defs(columns, pinned_fields=[entity_field], default_width=112),
        height=480,
        filterable=False,
        total_field=entity_field,
        total_color=total_color,
        total_border=total_border,
        extra_class="contracts-ag-grid--volume",
        filename=filename,
    )


def _contracts_export_button(button_id):
    return html.Button(
        "Export CSV",
        id=button_id,
        n_clicks=0,
        className="contracts-export-button",
        type="button",
    )


def _contracts_section_heading(title, subtitle=None, control=None):
    title_block = [html.H3(title, className="contracts-section-title")]
    if subtitle:
        title_block.append(html.Div(subtitle, className="fleet-metrics-table-subtitle"))

    children = [html.Div(title_block, className="contracts-section-title-block")]
    if control:
        children.append(control)

    return html.Div(children, className="fleet-metrics-table-heading contracts-section-heading")


def _contracts_inline_control(label, control):
    return html.Div(
        [
            html.Div(label, className="filter-group-header"),
            control,
        ],
        className="contracts-inline-control",
    )


def _year_slider_marks(min_year, max_year):
    span = max_year - min_year
    tick_step = 20 if span > 60 else 10 if span > 35 else 5
    first_tick = ((min_year + tick_step - 1) // tick_step) * tick_step

    marked_years = {min_year, max_year}
    marked_years.update(range(first_tick, max_year + 1, tick_step))

    return {year: str(year) for year in sorted(marked_years) if min_year <= year <= max_year}


def _contracts_graph(figure=None, graph_id=None, height=450):
    graph_kwargs = {
        "style": {"height": f"{height}px"},
        "config": CONTRACTS_GRAPH_CONFIG,
    }
    if figure is not None:
        graph_kwargs["figure"] = figure
    if graph_id is not None:
        graph_kwargs["id"] = graph_id

    return html.Div(
        dcc.Graph(**graph_kwargs),
        className="contracts-chart-panel",
    )


def _contracts_table_panel(title, table):
    return html.Div(
        [
            html.Div(
                [
                    html.H5(title, className="contracts-table-title"),
                    _contracts_export_button(
                        {'type': f'{table.id["type"]}-export-button', 'index': table.id.get('index', 'volume')}
                        if isinstance(table.id, dict)
                        else f"{table.id}-export-button"
                    ),
                ],
                className="contracts-table-panel-heading",
            ),
            table,
        ],
        className="contracts-table-panel",
    )


def style_contracts_figure(fig, height=None):
    if fig is None:
        return fig

    title_text = fig.layout.title.text if fig.layout and fig.layout.title else None
    layout_update = {
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif", "size": 12, "color": "#1f2937"},
        "title": {
            "text": title_text,
            "font": {"size": 14, "color": "#0f172a"},
            "x": 0.01,
            "xanchor": "left",
        },
        "hoverlabel": {
            "bgcolor": "#0f172a",
            "bordercolor": "#0f172a",
            "font": {"color": "#ffffff", "size": 11},
        },
        "legend": {
            "font": {"size": 10, "color": "#334155"},
            "title_font": {"size": 10, "color": "#475569"},
            "bgcolor": "rgba(255, 255, 255, 0.88)",
            "borderwidth": 0,
        },
    }
    if height is not None:
        layout_update["height"] = height

    fig.update_layout(**layout_update)
    fig.update_xaxes(
        gridcolor="#eef2f7",
        zeroline=False,
        linecolor="#cbd5e1",
        tickfont={"size": 11, "color": "#475569"},
        title_font={"size": 11, "color": "#475569"},
    )
    fig.update_yaxes(
        gridcolor="#eef2f7",
        zeroline=False,
        linecolor="#cbd5e1",
        tickfont={"size": 11, "color": "#475569"},
        title_font={"size": 11, "color": "#475569"},
    )
    return fig


def create_filter_controls(min_year=2000, max_year=2030, default_start=2015, default_end=2025):
    """Create filter controls panel"""
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Contracts", className="filter-group-header"),
                    html.Div("WoodMac LNG contract book", className="contracts-filter-title"),
                ],
                className="filter-group contracts-filter-title-group",
                style={"flex": "0 0 200px", "maxWidth": "100%"},
            ),
            html.Div(
                [
                    html.Div("Destination Country", className="filter-group-header"),
                    dcc.Dropdown(
                        id='destination-country-dropdown',
                        placeholder="Select destination countries...",
                        multi=True,
                        className="filter-dropdown",
                    ),
                ],
                className="filter-group",
                style={"flex": "1 1 220px", "maxWidth": "100%"},
            ),
            html.Div(
                [
                    html.Div("Contract Type", className="filter-group-header"),
                    dcc.Dropdown(
                        id='contract-type-dropdown',
                        placeholder="Select contract types...",
                        multi=True,
                        className="filter-dropdown",
                    ),
                ],
                className="filter-group",
                style={"flex": "1 1 190px", "maxWidth": "100%"},
            ),
            html.Div(
                [
                    html.Div("Pricing Type", className="filter-group-header"),
                    dcc.Dropdown(
                        id='pricing-type-dropdown',
                        placeholder="Select pricing types...",
                        multi=True,
                        className="filter-dropdown",
                    ),
                ],
                className="filter-group",
                style={"flex": "1 1 190px", "maxWidth": "100%"},
            ),
            html.Div(
                [
                    html.Div("Seller Company", className="filter-group-header"),
                    dcc.Dropdown(
                        id='seller-company-dropdown',
                        placeholder="Select sellers...",
                        multi=True,
                        className="filter-dropdown",
                    ),
                ],
                className="filter-group",
                style={"flex": "1 1 220px", "maxWidth": "100%"},
            ),
            html.Div(
                [
                    html.Div("Year Range", className="filter-group-header"),
                    dcc.RangeSlider(
                        id='year-range-slider',
                        min=min_year,
                        max=max_year,
                        step=1,
                        value=[default_start, default_end],
                        marks=_year_slider_marks(min_year, max_year),
                        tooltip={"placement": "bottom", "always_visible": False},
                        className="contracts-year-range-slider",
                    ),
                ],
                className="filter-group",
                style={"flex": "1.4 1 360px", "maxWidth": "100%", "padding": "0 8px"},
            ),
            html.Div(
                [
                    html.Div("Cargo Basis", className="filter-group-header"),
                    dcc.Dropdown(
                        id='cargo-basis-dropdown',
                        placeholder="Select cargo basis...",
                        multi=True,
                        className="filter-dropdown",
                    ),
                ],
                className="filter-group",
                style={"flex": "1 1 170px", "maxWidth": "100%"},
            ),
            html.Div(
                [
                    html.Div("Source Flexible", className="filter-group-header"),
                    dcc.Dropdown(
                        id='source-flexible-dropdown',
                        options=[
                            {'label': 'Flexible', 'value': 'Y'},
                            {'label': 'Not Flexible', 'value': 'N'},
                            {'label': 'Unknown', 'value': 'Unknown'}
                        ],
                        placeholder="Select flexibility...",
                        multi=True,
                        className="filter-dropdown",
                    ),
                ],
                className="filter-group",
                style={"flex": "1 1 190px", "maxWidth": "100%"},
            ),
            html.Div(
                [
                    html.Div("Destination Flexible", className="filter-group-header"),
                    dcc.Dropdown(
                        id='dest-flexible-dropdown',
                        options=[
                            {'label': 'Flexible', 'value': 'Y'},
                            {'label': 'Not Flexible', 'value': 'N'},
                            {'label': 'Unknown', 'value': 'Unknown'}
                        ],
                        placeholder="Select flexibility...",
                        multi=True,
                        className="filter-dropdown",
                    ),
                ],
                className="filter-group",
                style={"flex": "1 1 210px", "maxWidth": "100%"},
            ),
        ],
        className="professional-section-header",
        style={
            "display": "flex",
            "gap": "12px",
            "alignItems": "flex-end",
            "flexWrap": "wrap",
        },
    )

def create_contracts_sections_layout():
    """Create the main sections layout replacing tabs"""
    return html.Div(
        [
            create_timeline_section(),
            create_volume_analysis_section(),
        ],
        className="contracts-section-stack",
    )

def create_timeline_section():
    """Create Contract Signing Timeline section at the top of the page"""
    metric_control = _contracts_inline_control(
        "Y-axis metric",
        dcc.Dropdown(
            id='timeline-metric-dropdown',
            options=[
                {'label': 'Number of Contracts', 'value': 'count'},
                {'label': 'Volume (MTPA)', 'value': 'volume'}
            ],
            value='count',
            className="inline-dropdown",
            placeholder='Select metric...',
            clearable=False,
            style={'width': '230px'}
        ),
    )

    return html.Div(
        [
            _contracts_section_heading(
                "Contract signing timeline",
                "Historical signing cadence and pricing mix across the filtered contract book.",
                metric_control,
            ),
            html.Div(
                [
                    _contracts_graph(graph_id='timeline-chart', height=420),
                    _contracts_graph(graph_id='pricing-timeline-chart', height=420),
                ],
                className="contracts-chart-grid",
            ),
        ],
        className="contracts-section contracts-timeline-section",
        style=CONTRACTS_SECTION_STYLE,
    )

def create_volume_analysis_section():
    """Create Volume Analysis section following Enterprise Standard"""
    view_control = _contracts_inline_control(
        "View mode",
        dcc.Dropdown(
            id='volume-view-section-dropdown',
            options=[
                {'label': 'By Country', 'value': 'country'},
                {'label': 'By Company', 'value': 'company'}
            ],
            value='country',
            className="inline-dropdown",
            placeholder='Select view...',
            clearable=False,
            style={'width': '230px'}
        ),
    )

    return html.Div(
        [
            _contracts_section_heading(
                "Volume analysis",
                "Contracted quantity, cargo basis, pricing type, and entity breakdowns for the selected delivery years.",
                view_control,
            ),
            html.Div(
                [
                    html.Div(id="volume-analysis-charts"),
                    html.Div(id="volume-analysis-trends"),
                    html.Div(id="volume-analysis-tables"),
                ],
                className="contracts-volume-content",
            ),
            html.P(
                "Volume data represents annual contracted quantities. Expand source, destination, seller, or buyer rows for route detail.",
                className="contracts-footnote",
            ),
        ],
        className="contracts-section contracts-volume-section",
        style=CONTRACTS_SECTION_STYLE,
    )



def create_contracts_table():
    """Create interactive contracts data table"""
    return html.Div(
        [
            _contracts_section_heading(
                "Contract details",
                "Filtered contract-level records with pricing, flexibility, counterparties, and source/delivery metadata.",
                _contracts_export_button("contracts-table-export-button"),
            ),
            _contracts_ag_grid(
                id_value='contracts-table',
                row_data=[],
                column_defs=[],
                height=520,
                filterable=True,
                extra_class="contracts-ag-grid--details",
                filename="contracts_detail.csv",
            ),
        ],
        className="contracts-section contracts-detail-table-section",
        style=CONTRACTS_SECTION_STYLE,
    )

############################################ Tab Content Functions ###################################################

def create_volume_analysis_content(contracts_df, demand_df, volume_view='both', expanded_countries=None, expanded_sellers=None, expanded_destinations=None, expanded_buyers=None, year_range=None):
    """Create volume analysis tab content with country and seller breakdowns"""
    if demand_df.empty:
        return html.Div("No volume data available", className="text-center p-4")
    
    expanded_countries = expanded_countries or []
    expanded_sellers = expanded_sellers or []
    expanded_destinations = expanded_destinations or []
    expanded_buyers = expanded_buyers or []
    
    # Ensure any remaining NaN or empty values are mapped to 'Unknown'
    # This is critical for charts to show matching totals
    demand_df = demand_df.copy()  # Make a copy to avoid warnings
    
    # Map any NaN or empty string values to 'Unknown' for country columns
    for col in ['country_name_source', 'country_name_delivery']:
        if col in demand_df.columns:
            demand_df[col] = demand_df[col].fillna('Unknown')
            demand_df.loc[demand_df[col] == '', col] = 'Unknown'
            demand_df.loc[demand_df[col].isna(), col] = 'Unknown'
    
    # Use year range filter if provided, otherwise use recent years
    if year_range and len(year_range) == 2:
        filter_years = list(range(year_range[0], year_range[1] + 1))
        year_label = f"({year_range[0]}-{year_range[1]})"
    else:
        current_year = datetime.now().year
        filter_years = [current_year - 1, current_year, current_year + 1]
        year_label = f"({filter_years[0]}-{filter_years[-1]})"
    
    # Volume by source country - using filtered demand data
    # Ensure no NaN values in groupby
    demand_df['country_name_source'] = demand_df['country_name_source'].fillna('Unknown')
    country_volume = demand_df.groupby(['country_name_source', 'year'])['acq_volume__mmtpa'].sum().reset_index()
    
    # Get source countries by total volume
    country_totals = country_volume.groupby('country_name_source')['acq_volume__mmtpa'].sum().sort_values(ascending=False)
    
    # Show more countries if needed for consistency with destination chart
    num_source_countries = len(country_totals)
    
    # Match the destination chart approach - show up to 20 countries
    if num_source_countries <= 20:
        top_countries = country_totals.index.tolist()
    else:
        top_countries = country_totals.head(20).index.tolist()
        
        # Always include 'Unknown' if it exists and not in top 20
        if 'Unknown' in country_totals.index and 'Unknown' not in top_countries:
            top_countries.append('Unknown')
    
    country_volume_filtered = country_volume[country_volume['country_name_source'].isin(top_countries)]
    
    if not country_volume_filtered.empty:
        try:
            country_fig = px.bar(
                country_volume_filtered,
                x='year',
                y='acq_volume__mmtpa',
                color='country_name_source',
                title=f"Contracted Volume by Source Country {year_label}",
                labels={'acq_volume__mmtpa': 'Volume (MMTPA)', 'year': 'Year', 'country_name_source': 'Source Country'},
                template="plotly_white"  # Use explicit template to avoid conflicts
            )
            country_fig.update_layout(
                height=500,  # Increased from 400
                barmode='stack',  # This creates the cumulative effect
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=11)  # Slightly larger font
                ),
                margin=dict(l=60, r=150, t=80, b=60)  # Better margins
            )
        except Exception as e:
            print(f"Error creating country chart: {e}")
            country_fig = go.Figure()
            country_fig.add_annotation(text="Chart rendering error", x=0.5, y=0.5, showarrow=False)
            country_fig.update_layout(height=500, title="Contracted Volume by Source Country")
    else:
        country_fig = go.Figure()
        country_fig.add_annotation(text="No volume data available", x=0.5, y=0.5, showarrow=False)
        country_fig.update_layout(height=500, title="Contracted Volume by Source Country")
    
    # Volume by seller company - using filtered demand data
    seller_volume = demand_df.groupby(['company_name_seller', 'year'])['acq_volume__mmtpa'].sum().reset_index()
    
    # Use all sellers
    seller_volume_filtered = seller_volume
    
    if not seller_volume_filtered.empty:
        try:
            seller_fig = px.bar(
                seller_volume_filtered,
                x='year',
                y='acq_volume__mmtpa',
                color='company_name_seller',
                title=f"Contracted Volume by Seller Company {year_label}",
                labels={'acq_volume__mmtpa': 'Volume (MMTPA)', 'year': 'Year', 'company_name_seller': 'Seller Company'},
                template="plotly_white"  # Use explicit template to avoid conflicts
            )
            seller_fig.update_layout(
                height=500,  # Increased from 400
                barmode='stack',  # This creates the cumulative effect
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=11)  # Consistent font size
                ),
                margin=dict(l=60, r=150, t=80, b=60)  # Better margins
            )
        except Exception as e:
            print(f"Error creating seller chart: {e}")
            seller_fig = go.Figure()
            seller_fig.add_annotation(text="Chart rendering error", x=0.5, y=0.5, showarrow=False)
            seller_fig.update_layout(height=500, title="Contracted Volume by Seller Company")
    else:
        seller_fig = go.Figure()
        seller_fig.add_annotation(text="No seller volume data available", x=0.5, y=0.5, showarrow=False)
        seller_fig.update_layout(height=500, title="Contracted Volume by Seller Company")
    
    # Volume by buyer company - using filtered demand data
    if 'company_name_buyer' in demand_df.columns:
        buyer_volume = demand_df.groupby(['company_name_buyer', 'year'])['acq_volume__mmtpa'].sum().reset_index()
        
        # Use all buyers
        buyer_volume_filtered = buyer_volume
        
        if not buyer_volume_filtered.empty:
            try:
                buyer_fig = px.bar(
                    buyer_volume_filtered,
                    x='year',
                    y='acq_volume__mmtpa',
                    color='company_name_buyer',
                    title=f"Contracted Volume by Buyer Company {year_label}",
                    labels={'acq_volume__mmtpa': 'Volume (MMTPA)', 'year': 'Year', 'company_name_buyer': 'Buyer Company'},
                    template="plotly_white"
                )
                buyer_fig.update_layout(
                    height=500,
                    barmode='stack',
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02,
                        font=dict(size=11)
                    ),
                    margin=dict(l=60, r=150, t=80, b=60)
                )
            except Exception as e:
                print(f"Error creating buyer chart: {e}")
                buyer_fig = go.Figure()
                buyer_fig.add_annotation(text="Chart rendering error", x=0.5, y=0.5, showarrow=False)
                buyer_fig.update_layout(height=500, title="Contracted Volume by Buyer Company")
        else:
            buyer_fig = go.Figure()
            buyer_fig.add_annotation(text="No buyer volume data available", x=0.5, y=0.5, showarrow=False)
            buyer_fig.update_layout(height=500, title="Contracted Volume by Buyer Company")
    else:
        buyer_fig = go.Figure()
        buyer_fig.add_annotation(text="Buyer data not available", x=0.5, y=0.5, showarrow=False)
        buyer_fig.update_layout(height=500, title="Contracted Volume by Buyer Company")
    
    # Volume by destination country
    # Ensure no NaN values in groupby
    demand_df['country_name_delivery'] = demand_df['country_name_delivery'].fillna('Unknown')
    dest_volume = demand_df.groupby(['country_name_delivery', 'year'])['acq_volume__mmtpa'].sum().reset_index()
    
    # Get destination countries by total volume
    dest_totals = dest_volume.groupby('country_name_delivery')['acq_volume__mmtpa'].sum().sort_values(ascending=False)
    
    # Show more countries in destination chart (top 20 or all if less than 20)
    num_destinations = len(dest_totals)
    
    # If there are 20 or fewer destinations, show all; otherwise show top 20
    if num_destinations <= 20:
        top_destinations = dest_totals.index.tolist()
    else:
        top_destinations = dest_totals.head(20).index.tolist()
        
        # Always include 'Unknown' if it exists and not in top 20
        if 'Unknown' in dest_totals.index and 'Unknown' not in top_destinations:
            top_destinations.append('Unknown')
    
    dest_volume_filtered = dest_volume[dest_volume['country_name_delivery'].isin(top_destinations)]
    
    if not dest_volume_filtered.empty:
        try:
            dest_fig = px.bar(
                dest_volume_filtered,
                x='year',
                y='acq_volume__mmtpa',
                color='country_name_delivery',
                title=f"Contracted Volume by Destination Country {year_label}",
                labels={'acq_volume__mmtpa': 'Volume (MMTPA)', 'year': 'Year', 'country_name_delivery': 'Destination Country'},
                template="plotly_white"
            )
            dest_fig.update_layout(
                height=500,  # Increased from 400
                barmode='stack',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=11)  # Slightly larger font
                ),
                margin=dict(l=60, r=150, t=80, b=60)  # Better margins
            )
        except Exception as e:
            print(f"Error creating destination chart: {e}")
            dest_fig = go.Figure()
            dest_fig.add_annotation(text="Chart rendering error", x=0.5, y=0.5, showarrow=False)
            dest_fig.update_layout(height=500, title="Contracted Volume by Destination Country")
    else:
        dest_fig = go.Figure()
        dest_fig.add_annotation(text="No destination data available", x=0.5, y=0.5, showarrow=False)
        dest_fig.update_layout(height=500, title="Contracted Volume by Destination Country")
    
    # FOB vs DES breakdown
    fob_des_data = demand_df.groupby(['cargo_basis', 'year'])['acq_volume__mmtpa'].sum().reset_index()
    
    if not fob_des_data.empty and 'cargo_basis' in fob_des_data.columns:
        try:
            fob_des_fig = px.bar(
                fob_des_data,
                x='year',
                y='acq_volume__mmtpa',
                color='cargo_basis',
                title=f"Contracted Volume by Cargo Basis {year_label}",
                labels={'acq_volume__mmtpa': 'Volume (MMTPA)', 'year': 'Year'},
                barmode='stack',
                template="plotly_white"
            )
            fob_des_fig.update_layout(
                height=400,  # Increased from 300
                margin=dict(l=60, r=60, t=80, b=60)  # Better margins
            )
        except Exception as e:
            print(f"Error creating cargo basis chart: {e}")
            fob_des_fig = go.Figure()
            fob_des_fig.add_annotation(text="Chart rendering error", x=0.5, y=0.5, showarrow=False)
            fob_des_fig.update_layout(height=400, title="Contracted Volume by Cargo Basis")
    else:
        fob_des_fig = go.Figure()
        fob_des_fig.add_annotation(text="No cargo basis data available", x=0.5, y=0.5, showarrow=False)
        fob_des_fig.update_layout(height=400, title="Contracted Volume by Cargo Basis")
    
    # Annual volume trend for all years
    annual_volume = demand_df.groupby('year')['acq_volume__mmtpa'].sum().reset_index()
    if not annual_volume.empty:
        try:
            trend_fig = px.line(
                annual_volume,
                x='year',
                y='acq_volume__mmtpa',
                title="Total Contracted Volume Trend",
                labels={'acq_volume__mmtpa': 'Volume (MMTPA)', 'year': 'Year'},
                markers=True,
                template="plotly_white"
            )
            trend_fig.update_layout(
                height=400,  # Increased from 300
                margin=dict(l=60, r=60, t=80, b=60)  # Better margins
            )
        except Exception as e:
            print(f"Error creating trend chart: {e}")
            trend_fig = go.Figure()
            trend_fig.add_annotation(text="Chart rendering error", x=0.5, y=0.5, showarrow=False)
            trend_fig.update_layout(height=400, title="Total Contracted Volume Trend")
    else:
        trend_fig = go.Figure()
        trend_fig.add_annotation(text="No annual trend data available", x=0.5, y=0.5, showarrow=False)
        trend_fig.update_layout(height=400, title="Total Contracted Volume Trend")
    
    # Volume summary tables with years as columns and expandable functionality
    if not demand_df.empty:
        # Get all years in data
        available_years = sorted(demand_df['year'].unique())
        
        # Prepare expandable country table
        country_display_data, country_columns = prepare_volume_table_for_display(
            demand_df, 'country', available_years, expanded_countries
        )
        
        country_table = _contracts_volume_grid(
            {'type': 'volume-country-expandable-table', 'index': 'volume'},
            country_display_data,
            country_columns,
            'country_name_source',
            "volume_by_source_country.csv",
        )
        
        # Prepare expandable seller table
        seller_display_data, seller_columns = prepare_volume_table_for_display(
            demand_df, 'seller', available_years, expanded_sellers
        )
        
        seller_table = _contracts_volume_grid(
            {'type': 'volume-seller-expandable-table', 'index': 'volume'},
            seller_display_data,
            seller_columns,
            'company_name_seller',
            "volume_by_seller_company.csv",
        )
    else:
        country_table = html.Div("No country data available")
        seller_table = html.Div("No seller data available")

    for fig, height in [
        (country_fig, 550),
        (seller_fig, 550),
        (buyer_fig, 550),
        (dest_fig, 550),
        (fob_des_fig, 450),
        (trend_fig, 450),
    ]:
        style_contracts_figure(fig, height=height)
    
    # Determine layout based on volume_view selection
    if volume_view == 'country':
        main_charts_row = html.Div([
            _contracts_graph(figure=country_fig, height=550),
            _contracts_graph(figure=dest_fig, height=550),
        ], className="contracts-chart-grid")
        
        # Prepare destination country table
        dest_display_data, dest_columns = prepare_volume_table_for_display(
            demand_df, 'destination', available_years, expanded_destinations
        )
        
        dest_table = _contracts_volume_grid(
            {'type': 'volume-destination-expandable-table', 'index': 'volume'},
            dest_display_data,
            dest_columns,
            'country_name_delivery',
            "volume_by_destination_country.csv",
        )
        
        tables_row = html.Div([
            _contracts_table_panel("Volume by source country", country_table),
            _contracts_table_panel("Volume by destination country", dest_table),
        ], className="contracts-table-grid")
        
    elif volume_view == 'company':
        main_charts_row = html.Div([
            _contracts_graph(figure=seller_fig, height=550),
            _contracts_graph(figure=buyer_fig, height=550),
        ], className="contracts-chart-grid")
        
        # Prepare destination table for seller view too
        dest_display_data, dest_columns = prepare_volume_table_for_display(
            demand_df, 'destination', available_years, expanded_destinations
        )
        
        dest_table = _contracts_volume_grid(
            {'type': 'volume-destination-expandable-table', 'index': 'volume'},
            dest_display_data,
            dest_columns,
            'country_name_delivery',
            "volume_by_destination_country.csv",
        )
        
        # For company view, show seller and buyer tables
        # Prepare buyer table similar to seller table
        buyer_display_data, buyer_columns = prepare_volume_table_for_display(
            demand_df, 'buyer', available_years, expanded_buyers
        )
        
        buyer_table = _contracts_volume_grid(
            {'type': 'volume-buyer-expandable-table', 'index': 'volume'},
            buyer_display_data,
            buyer_columns,
            'company_name_buyer',
            "volume_by_buyer_company.csv",
            total_color="#27AE60",
            total_border="#145A32",
        )
        
        tables_row = html.Div([
            _contracts_table_panel("Volume by seller company", seller_table),
            _contracts_table_panel("Volume by buyer company", buyer_table),
        ], className="contracts-table-grid")
        
    else:  # fallback to country view
        main_charts_row = html.Div([
            _contracts_graph(figure=country_fig, height=550),
            _contracts_graph(figure=dest_fig, height=550),
        ], className="contracts-chart-grid")
        
        # Prepare destination table
        dest_display_data, dest_columns = prepare_volume_table_for_display(
            demand_df, 'destination', available_years, expanded_destinations
        )
        
        dest_table = _contracts_volume_grid(
            {'type': 'volume-destination-expandable-table', 'index': 'volume'},
            dest_display_data,
            dest_columns,
            'country_name_delivery',
            "volume_by_destination_country.csv",
        )
        
        tables_row = html.Div([
            _contracts_table_panel("Volume by source country", country_table),
            _contracts_table_panel("Volume by destination country", dest_table),
        ], className="contracts-table-grid")
    
    # Create Pricing Type and Contract Type Distribution charts by delivery year
    # Merge contracts with demand to get delivery years
    if not demand_df.empty:
        # Get unique contract-year combinations from demand data
        contract_years = demand_df[['id_contract', 'year']].drop_duplicates()
        
        # Merge with contracts to get pricing and contract types for each delivery year
        analysis_data = contract_years.merge(
            contracts_df[['id_contract', 'detailed_pricing_type', 'contract_type', 'contract_pricing_type']], 
            on='id_contract', 
            how='left'
        )
        
        # Apply year range filter if provided
        if year_range and len(year_range) == 2:
            analysis_data = analysis_data[
                (analysis_data['year'] >= year_range[0]) & 
                (analysis_data['year'] <= year_range[1])
            ]
        
        # Create Pricing Type distribution by delivery year
        if 'detailed_pricing_type' in analysis_data.columns:
            pricing_by_year = analysis_data.groupby(['year', 'detailed_pricing_type']).size().reset_index(name='count')
            pricing_by_year.columns = ['Year', 'Pricing Type', 'Count']
        else:
            pricing_by_year = analysis_data.groupby(['year', 'contract_pricing_type']).size().reset_index(name='count')
            pricing_by_year.columns = ['Year', 'Pricing Type', 'Count']
    else:
        # Fallback if no demand data
        pricing_by_year = pd.DataFrame(columns=['Year', 'Pricing Type', 'Count'])
    
    # Use the same color mapping as the timeline charts
    pricing_color_map = {
        'Fixed': '#2E86C1',                      # Blue
        'Spot': '#27AE60',                       # Green
        'Index - Oil': '#E74C3C',                # Red
        'Index - Oil (Brent)': '#C0392B',        # Dark Red
        'Index - Oil (JCC)': '#E74C3C',          # Red
        'Index - Oil (WTI)': '#EC7063',          # Light Red
        'Index - Oil (Dubai)': '#F1948A',        # Lighter Red
        'Index - Oil (Oil)': '#E74C3C',          # Red
        'Index - Gas': '#F39C12',                # Orange
        'Index - Gas (Henry Hub)': '#E67E22',    # Dark Orange
        'Index - Gas (NBP)': '#F39C12',          # Orange
        'Index - Gas (TTF)': '#F5B041',          # Light Orange
        'Index - Gas (JKM)': '#FAD7A0',          # Light Yellow
        'Index - Gas (Slope)': '#F8C471',        # Yellow-Orange
        'Index - Gas (Others)': '#F39C12',       # Orange
        'Index - Hybrid': '#9B59B6',             # Purple
        'Index': '#95A5A6',                      # Gray
        'Indexed': '#95A5A6',                    # Gray
        'Unknown': '#BDC3C7'                     # Light Gray
    }
    
    # Create stacked bar chart for pricing type
    if not pricing_by_year.empty:
        pricing_type_fig = px.bar(
            pricing_by_year,
            x='Year',
            y='Count',
            color='Pricing Type',
            title="Pricing Type Distribution by Year",
            color_discrete_map=pricing_color_map,
            text_auto=False
        )
        pricing_type_fig.update_layout(
            height=450,
            barmode='stack',
            xaxis_title="Year",
            yaxis_title="Number of Contracts",
            template="plotly_white",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=10)
            ),
            margin=dict(r=180)  # More space for legend
        )
        pricing_type_fig.update_xaxes(tickformat='d')  # Display years as integers
    else:
        pricing_type_fig = go.Figure()
        pricing_type_fig.add_annotation(text="No pricing data available", x=0.5, y=0.5, showarrow=False)
        pricing_type_fig.update_layout(height=450, title="Pricing Type Distribution by Year")
    
    # Create Contract Type distribution by delivery year
    if not demand_df.empty and 'year' in analysis_data.columns:
        contract_by_year = analysis_data.groupby(['year', 'contract_type']).size().reset_index(name='count')
        contract_by_year.columns = ['Year', 'Contract Type', 'Count']
    else:
        # Fallback if no demand data
        contract_by_year = pd.DataFrame(columns=['Year', 'Contract Type', 'Count'])
    
    # Define colors for contract types
    contract_color_map = {
        'SPA': '#3498DB',         # Bright Blue
        'HOA': '#9B59B6',         # Purple
        'MOU': '#E74C3C',         # Red
        'MSPA': '#F39C12',        # Orange
        'Equity': '#27AE60',      # Green
        'Unknown': '#95A5A6'      # Gray
    }
    
    # Create stacked bar chart for contract type
    if not contract_by_year.empty:
        contract_type_fig = px.bar(
            contract_by_year,
            x='Year',
            y='Count',
            color='Contract Type',
            title="Contract Type Distribution by Year",
            color_discrete_map=contract_color_map,
            text_auto=False
        )
        contract_type_fig.update_layout(
            height=450,
            barmode='stack',
            xaxis_title="Year",
            yaxis_title="Number of Contracts",
            template="plotly_white",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=11)
            ),
            margin=dict(r=150)  # Space for legend
        )
        contract_type_fig.update_xaxes(tickformat='d')  # Display years as integers
    else:
        contract_type_fig = go.Figure()
        contract_type_fig.add_annotation(text="No contract type data available", x=0.5, y=0.5, showarrow=False)
        contract_type_fig.update_layout(height=450, title="Contract Type Distribution by Year")

    style_contracts_figure(pricing_type_fig, height=450)
    style_contracts_figure(contract_type_fig, height=450)
    
    # Create distribution charts row
    distribution_charts_row = html.Div([
        _contracts_graph(figure=pricing_type_fig, height=450),
        _contracts_graph(figure=contract_type_fig, height=450),
    ], className="contracts-chart-grid")
    
    # Create the overview charts row (FOB/DES and Trend)
    overview_charts_row = html.Div([
        _contracts_graph(figure=fob_des_fig, height=450),
        _contracts_graph(figure=trend_fig, height=450),
    ], className="contracts-chart-grid")
    
    return html.Div([
        # Distribution Charts Row (Pricing Type and Contract Type) - First Row
        distribution_charts_row,
        
        # Overview Charts Row (FOB/DES and Trend) - Second Row
        overview_charts_row,
        
        # Main Charts Row (Contracted Volume by Source/Seller and Destination) - Third Row
        main_charts_row,
        
        # Summary Tables Row - Fourth Row
        tables_row
    ])


############################################ Main Layout ###################################################

# Add detailed pricing type to contracts_df for filtering
def get_detailed_pricing_type(row):
    """Generate detailed pricing type label for a contract"""
    pricing_type = row.get('contract_pricing_type', None)
    indexation_cat = row.get('indexation_category', None)
    indexation_point = row.get('indexation_point', None)
    oil_structure = row.get('oil_pricing_structure', None)
    gas_structure = row.get('gas_pricing_structure', None)
    
    # Check if it's indexed - either by pricing_type OR by having indexation data
    is_indexed = (pricing_type in ['Index', 'Indexed'] or 
                 (indexation_cat and pd.notna(indexation_cat) and str(indexation_cat).strip() not in ['', 'None', 'Unknown', 'nan']) or
                 (indexation_point and pd.notna(indexation_point) and str(indexation_point).strip() not in ['', 'None', 'Unknown', 'nan']) or
                 (oil_structure and pd.notna(oil_structure) and str(oil_structure).strip() not in ['', 'None', 'Unknown']) or
                 (gas_structure and pd.notna(gas_structure) and str(gas_structure).strip() not in ['', 'None', 'Unknown']))
    
    if is_indexed:
        # FIRST: Check if explicitly marked as Hybrid in indexation fields
        combined_text = f"{str(indexation_cat).lower() if indexation_cat else ''} {str(indexation_point).lower() if indexation_point else ''}"
        
        if 'hybrid' in combined_text or 'mixed' in combined_text:
            # It's explicitly hybrid - try to get details
            has_oil = oil_structure and pd.notna(oil_structure) and str(oil_structure).strip() not in ['', 'None', 'Unknown']
            has_gas = gas_structure and pd.notna(gas_structure) and str(gas_structure).strip() not in ['', 'None', 'Unknown']
            
            if has_oil and has_gas:
                oil_detail = extract_index_detail(oil_structure, indexation_cat, 'oil', indexation_point)
                gas_detail = extract_index_detail(gas_structure, indexation_cat, 'gas', indexation_point)
                if oil_detail and gas_detail:
                    return f"Index - Hybrid ({oil_detail}/{gas_detail})"
            return "Index - Hybrid"
        
        # SECOND: Check pricing structures
        has_oil = oil_structure and pd.notna(oil_structure) and str(oil_structure).strip() not in ['', 'None', 'Unknown']
        has_gas = gas_structure and pd.notna(gas_structure) and str(gas_structure).strip() not in ['', 'None', 'Unknown']
        
        if has_oil and has_gas:
            # Both structures present means hybrid
            oil_detail = extract_index_detail(oil_structure, indexation_cat, 'oil', indexation_point)
            gas_detail = extract_index_detail(gas_structure, indexation_cat, 'gas', indexation_point)
            if oil_detail and gas_detail:
                return f"Index - Hybrid ({oil_detail}/{gas_detail})"
            return "Index - Hybrid"
        elif has_oil:
            oil_detail = extract_index_detail(oil_structure, indexation_cat, 'oil', indexation_point)
            return f"Index - Oil ({oil_detail})" if oil_detail else "Index - Oil"
        elif has_gas:
            gas_detail = extract_index_detail(gas_structure, indexation_cat, 'gas', indexation_point)
            return f"Index - Gas ({gas_detail})" if gas_detail else "Index - Gas"
        
        # THIRD: Categorize based on specific index mentions
        if 'brent' in combined_text:
            return "Index - Oil (Brent)"
        elif 'jcc' in combined_text or 'japan crude cocktail' in combined_text:
            return "Index - Oil (JCC)"
        elif 'wti' in combined_text:
            return "Index - Oil (WTI)"
        elif 'dubai' in combined_text:
            return "Index - Oil (Dubai)"
        elif 'henry hub' in combined_text or 'hhub' in combined_text:
            return "Index - Gas (Henry Hub)"
        elif 'nbp' in combined_text:
            return "Index - Gas (NBP)"
        elif 'ttf' in combined_text:
            return "Index - Gas (TTF)"
        elif 'jkm' in combined_text:
            return "Index - Gas (JKM)"
        elif 'oil' in combined_text or 'crude' in combined_text:
            return "Index - Oil"
        elif 'gas' in combined_text or 'slope' in combined_text:
            return "Index - Gas"
        else:
            return "Unknown"
    elif pricing_type == 'Spot':
        return 'Spot'
    elif pricing_type == 'Fixed':
        # Double-check it's not actually indexed
        if indexation_cat and pd.notna(indexation_cat) and str(indexation_cat).strip() not in ['', 'None', 'Unknown', 'nan']:
            return "Unknown"
        return 'Fixed'
    elif pricing_type in ['Unknown', None, '', 'nan'] or pd.isna(pricing_type):
        return "Unknown"
    else:
        return str(pricing_type)


CONTRACTS_RUNTIME_COLUMNS = CONTRACTS_DATA_COLUMNS + ['detailed_pricing_type']
CONTRACTS_SOURCE_NAMES = (
    'contracts',
    'demand',
    'price_assumptions',
    'price_formula',
)
CONTRACTS_SOURCE_LABELS = {
    'contracts': 'Contracts',
    'demand': 'Demand',
    'price_assumptions': 'Assumptions',
    'price_formula': 'Formula',
}
CONTRACTS_REVISION_QUERY = text("""
    SELECT 'contracts' AS source_name,
           COUNT(*) AS row_count,
           MAX(upload_timestamp_utc) AS watermark
    FROM at_lng.woodmac_lng_contract
    UNION ALL
    SELECT 'demand',
           COUNT(*),
           MAX(upload_timestamp_utc)
    FROM at_lng.woodmac_lng_contract_annual_contracted_demand_mta
    UNION ALL
    SELECT 'price_assumptions',
           COUNT(*),
           MAX(upload_timestamp_utc)
    FROM at_lng.woodmac_lng_contract_price_assumptions
    UNION ALL
    SELECT 'price_formula',
           COUNT(*),
           MAX(upload_timestamp_utc)
    FROM at_lng.woodmac_lng_contract_price_formula
""")


@dataclass(frozen=True)
class ContractsSnapshot:
    revision_key: tuple[tuple[str, int, str | None], ...]
    contracts: pd.DataFrame
    demand: pd.DataFrame
    price_assumptions: pd.DataFrame
    price_formula: pd.DataFrame
    year_settings: dict


_contracts_snapshot: ContractsSnapshot | None = None
_contracts_snapshot_status = 'unavailable'
_contracts_snapshot_message = 'Contracts data has not been loaded.'
_contracts_snapshot_lock = Lock()


def _default_year_settings():
    current_year = datetime.now().year
    min_year = 2000
    max_year = current_year + 30
    return {
        'min_year': min_year,
        'max_year': max_year,
        'default_start': max(min_year, current_year - 1),
        'default_end': max_year,
    }


def _calculate_contract_year_settings(loaded_contracts_df):
    current_year = datetime.now().year
    valid_sign_years = loaded_contracts_df['contract_year_signed'].dropna()
    if not valid_sign_years.empty:
        min_year = int(valid_sign_years.min())
    else:
        min_year = 2000

    end_years = []
    if 'contract_date_end' in loaded_contracts_df.columns:
        end_dates = loaded_contracts_df['contract_date_end'].dropna()
        for date_str in end_dates:
            try:
                date_str = str(date_str).strip()
                if date_str and date_str not in ['', 'None', 'nan', 'NaT']:
                    import re
                    year_match = re.search(r'20\d{2}|19\d{2}', date_str)
                    if year_match:
                        end_years.append(int(year_match.group()))
            except Exception:
                continue

        if end_years:
            max_year = max(end_years)
        else:
            max_year = int(valid_sign_years.max()) if not valid_sign_years.empty else current_year
    else:
        max_year = int(valid_sign_years.max()) if not valid_sign_years.empty else current_year

    max_year = min(max_year, current_year + 30)
    return {
        'min_year': min_year,
        'max_year': max_year,
        'default_start': max(min_year, current_year - 1),
        'default_end': max_year,
    }


def _enhance_contracts_data(loaded_contracts_df, loaded_price_assumptions_df):
    enhanced_contracts_df = loaded_contracts_df.copy()

    if not loaded_price_assumptions_df.empty:
        pricing_cols = [
            'id_contract', 'indexation_category', 'indexation_point',
            'oil_pricing_structure', 'gas_pricing_structure'
        ]
        available_pricing_cols = [
            col for col in pricing_cols
            if col in loaded_price_assumptions_df.columns
        ]
        if available_pricing_cols:
            pricing_data = loaded_price_assumptions_df[available_pricing_cols].drop_duplicates('id_contract')
            enhanced_contracts_df = enhanced_contracts_df.merge(
                pricing_data,
                on='id_contract',
                how='left',
                suffixes=('', '_pa'),
            )

            if 'indexation_category_pa' in enhanced_contracts_df.columns:
                enhanced_contracts_df['indexation_category'] = (
                    enhanced_contracts_df['indexation_category']
                    .combine_first(enhanced_contracts_df['indexation_category_pa'])
                )
            if 'indexation_point_pa' in enhanced_contracts_df.columns:
                enhanced_contracts_df['indexation_point'] = (
                    enhanced_contracts_df['indexation_point']
                    .combine_first(enhanced_contracts_df['indexation_point_pa'])
                )

    enhanced_contracts_df['detailed_pricing_type'] = enhanced_contracts_df.apply(
        get_detailed_pricing_type,
        axis=1,
    )
    return enhanced_contracts_df


def _empty_contracts_snapshot():
    return ContractsSnapshot(
        revision_key=tuple(),
        contracts=pd.DataFrame(columns=CONTRACTS_RUNTIME_COLUMNS),
        demand=pd.DataFrame(columns=ANNUAL_DEMAND_DATA_COLUMNS),
        price_assumptions=pd.DataFrame(
            columns=PRICE_ASSUMPTIONS_COLUMNS
        ),
        price_formula=pd.DataFrame(columns=PRICE_FORMULA_COLUMNS),
        year_settings=_default_year_settings(),
    )


def _normalize_contracts_revision_value(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return str(value)


def fetch_contracts_revision_key():
    with engine.connect() as connection:
        rows = connection.execute(
            CONTRACTS_REVISION_QUERY
        ).mappings().all()
    revisions = {
        str(row['source_name']): (
            int(row['row_count'] or 0),
            _normalize_contracts_revision_value(row['watermark']),
        )
        for row in rows
    }
    if set(revisions) != set(CONTRACTS_SOURCE_NAMES):
        raise RuntimeError('Contracts source revision query was incomplete')
    return tuple(
        (
            source_name,
            revisions[source_name][0],
            revisions[source_name][1],
        )
        for source_name in CONTRACTS_SOURCE_NAMES
    )


def _build_contracts_snapshot(revision_key):
    loaders = {
        'contracts': load_contracts_data,
        'demand': load_annual_demand_data,
        'price_assumptions': load_price_assumptions_data,
        'price_formula': load_price_formula_data,
    }
    with ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix='contracts-load',
    ) as executor:
        futures = {
            name: executor.submit(loader)
            for name, loader in loaders.items()
        }
        loaded_frames = {
            name: futures[name].result()
            for name in loaders
        }

    expected_counts = {
        source_name: row_count
        for source_name, row_count, _watermark in revision_key
    }
    for source_name, frame in loaded_frames.items():
        if expected_counts.get(source_name, 0) > 0 and frame.empty:
            raise RuntimeError(
                f'Contracts source {source_name} returned no rows'
            )

    enhanced_contracts_df = _enhance_contracts_data(
        loaded_frames['contracts'],
        loaded_frames['price_assumptions'],
    )
    year_settings = _calculate_contract_year_settings(
        enhanced_contracts_df
    )
    return ContractsSnapshot(
        revision_key=revision_key,
        contracts=enhanced_contracts_df,
        demand=loaded_frames['demand'],
        price_assumptions=loaded_frames['price_assumptions'],
        price_formula=loaded_frames['price_formula'],
        year_settings=year_settings,
    )


def _ensure_contracts_snapshot(*, force=False):
    global _contracts_snapshot
    global _contracts_snapshot_status
    global _contracts_snapshot_message

    try:
        revision_key = fetch_contracts_revision_key()
    except Exception:
        logger.warning(
            'Contracts source revision check failed',
            exc_info=True,
        )
        with _contracts_snapshot_lock:
            if _contracts_snapshot is None:
                _contracts_snapshot = _empty_contracts_snapshot()
                _contracts_snapshot_status = 'unavailable'
                _contracts_snapshot_message = (
                    'Contracts data is unavailable because source revisions '
                    'could not be verified.'
                )
            else:
                _contracts_snapshot_status = 'stale'
                _contracts_snapshot_message = (
                    'Source refresh failed. Showing the last verified '
                    'contracts snapshot.'
                )
            return _contracts_snapshot

    with _contracts_snapshot_lock:
        if (
            not force
            and _contracts_snapshot is not None
            and _contracts_snapshot.revision_key == revision_key
        ):
            _contracts_snapshot_status = 'fresh'
            _contracts_snapshot_message = ''
            return _contracts_snapshot

        try:
            candidate = _build_contracts_snapshot(revision_key)
        except Exception:
            logger.warning(
                'Contracts snapshot refresh failed',
                exc_info=True,
            )
            if _contracts_snapshot is None:
                _contracts_snapshot = _empty_contracts_snapshot()
                _contracts_snapshot_status = 'unavailable'
                _contracts_snapshot_message = (
                    'Contracts data is unavailable because a complete '
                    'snapshot could not be loaded.'
                )
            else:
                _contracts_snapshot_status = 'stale'
                _contracts_snapshot_message = (
                    'Source refresh failed. Showing the last verified '
                    'contracts snapshot.'
                )
            return _contracts_snapshot

        _contracts_snapshot = candidate
        _contracts_snapshot_status = 'fresh'
        _contracts_snapshot_message = ''
        return _contracts_snapshot


def _contracts_snapshot_token(snapshot, refresh_generation=None):
    token = repr(snapshot.revision_key)
    if refresh_generation is not None:
        token = f"{token}|refresh:{refresh_generation}"
    return token


def _contracts_source_status(snapshot):
    if _contracts_snapshot_status == 'unavailable':
        return (
            _contracts_snapshot_message,
            'contracts-source-status contracts-source-status-unavailable',
        )

    revision_labels = []
    for source_name, row_count, watermark in snapshot.revision_key:
        label = CONTRACTS_SOURCE_LABELS[source_name]
        revision_labels.append(
            f"{label} {watermark or '—'} ({row_count:,} rows)"
        )
    revision_text = ' | '.join(revision_labels)
    if _contracts_snapshot_status == 'stale':
        return (
            f"{_contracts_snapshot_message} Source revisions (UTC): "
            f"{revision_text}",
            'contracts-source-status contracts-source-status-stale',
        )
    return (
        f"Source revisions (UTC): {revision_text}",
        'contracts-source-status contracts-source-status-fresh',
    )


def _current_contracts_snapshot():
    return _contracts_snapshot or _empty_contracts_snapshot()


def _ensure_contracts_data_loaded():
    """Compatibility wrapper for callers that previously initialized globals."""
    return _ensure_contracts_snapshot()


def layout():
    snapshot = _ensure_contracts_snapshot()
    year_settings = snapshot.year_settings
    source_status_text, source_status_class = _contracts_source_status(
        snapshot
    )

    return html.Div(
        [
            html.Div(
                source_status_text,
                id='contracts-source-status',
                className=source_status_class,
                role='status',
                **{'aria-live': 'polite'},
            ),
            create_filter_controls(
                year_settings['min_year'],
                year_settings['max_year'],
                year_settings['default_start'],
                year_settings['default_end'],
            ),
            html.Div(
                [
                    create_contracts_sections_layout(),
                    create_contracts_table(),
                ],
                className="contracts-content-stack",
            ),
            html.Div(
                _contracts_snapshot_token(snapshot),
                id='contracts-data-store',
                style={'display': 'none'},
            ),
            dcc.Store(id='volume-country-expanded-store', data=[]),
            dcc.Store(id='volume-seller-expanded-store', data=[]),
            dcc.Store(id='volume-destination-expanded-store', data=[]),
            dcc.Store(id='volume-buyer-expanded-store', data=[]),
        ],
        className="contracts-page-shell",
    )

############################################ Callbacks ###################################################


@callback(
    Output('contracts-data-store', 'children'),
    Output('contracts-source-status', 'children'),
    Output('contracts-source-status', 'className'),
    Input('global-refresh-button', 'n_clicks'),
    prevent_initial_call=True,
)
def refresh_contracts_snapshot(n_clicks):
    snapshot = _ensure_contracts_snapshot(force=True)
    source_status_text, source_status_class = _contracts_source_status(
        snapshot
    )
    return (
        _contracts_snapshot_token(snapshot, n_clicks),
        source_status_text,
        source_status_class,
    )


@callback(
    [Output('destination-country-dropdown', 'options'),
     Output('contract-type-dropdown', 'options'),
     Output('pricing-type-dropdown', 'options'),
     Output('seller-company-dropdown', 'options'),
     Output('cargo-basis-dropdown', 'options')],
    [Input('contracts-data-store', 'children')]
)
def update_filter_options(_):
    """Update filter dropdown options based on available data"""
    snapshot = _current_contracts_snapshot()
    contracts_df = snapshot.contracts
    demand_df = snapshot.demand
    if contracts_df.empty and demand_df.empty:
        empty_options = []
        return [empty_options] * 5
    
    dest_countries = [{'label': country, 'value': country} 
                     for country in sorted(contracts_df['country_name_delivery'].dropna().unique())]
    
    contract_types = [{'label': ct, 'value': ct} 
                     for ct in sorted(contracts_df['contract_type'].dropna().unique())]
    
    # Use detailed pricing types for the filter
    pricing_types = [{'label': pt, 'value': pt} 
                    for pt in sorted(contracts_df['detailed_pricing_type'].dropna().unique())]
    
    sellers = [{'label': seller, 'value': seller} 
              for seller in sorted(contracts_df['company_name_seller'].dropna().unique())]
    
    # Cargo basis from demand data (contains cargo_basis from joined contracts)
    cargo_basis_options = []
    if not demand_df.empty and 'cargo_basis' in demand_df.columns:
        cargo_basis_options = [{'label': basis, 'value': basis} 
                              for basis in sorted(demand_df['cargo_basis'].dropna().unique())]
    
    return dest_countries, contract_types, pricing_types, sellers, cargo_basis_options

# Individual section update callbacks
@callback(
    [Output('volume-analysis-charts', 'children'),
     Output('volume-analysis-tables', 'children'),
     Output('volume-analysis-trends', 'children')],
    [Input('destination-country-dropdown', 'value'),
     Input('contract-type-dropdown', 'value'),
     Input('pricing-type-dropdown', 'value'),
     Input('seller-company-dropdown', 'value'),
     Input('year-range-slider', 'value'),
     Input('cargo-basis-dropdown', 'value'),
     Input('source-flexible-dropdown', 'value'),
     Input('dest-flexible-dropdown', 'value'),
     Input('volume-view-section-dropdown', 'value'),
     Input('volume-country-expanded-store', 'data'),
     Input('volume-seller-expanded-store', 'data'),
     Input('volume-destination-expanded-store', 'data'),
     Input('volume-buyer-expanded-store', 'data'),
     Input('contracts-data-store', 'children')]
)
def update_volume_analysis_section(dest_countries, contract_types, 
                                  pricing_types, sellers, year_range, cargo_basis, 
                                  source_flexible, dest_flexible, volume_view_section,
                                  expanded_countries, expanded_sellers, expanded_destinations, expanded_buyers,
                                  _snapshot_token):
    """Update volume analysis section content"""
    snapshot = _current_contracts_snapshot()
    contracts_df = snapshot.contracts
    demand_df = snapshot.demand
    
    # Apply filters to dataframes
    filtered_contracts = contracts_df.copy()
    filtered_demand = demand_df.copy()
    
    if dest_countries:
        filtered_contracts = filtered_contracts[filtered_contracts['country_name_delivery'].isin(dest_countries)]
    if contract_types:
        filtered_contracts = filtered_contracts[filtered_contracts['contract_type'].isin(contract_types)]
    if pricing_types:
        # Filter by detailed pricing type
        filtered_contracts = filtered_contracts[filtered_contracts['detailed_pricing_type'].isin(pricing_types)]
    if sellers:
        filtered_contracts = filtered_contracts[filtered_contracts['company_name_seller'].isin(sellers)]
    if year_range:
        filtered_demand = filtered_demand[
            (filtered_demand['year'] >= year_range[0]) & 
            (filtered_demand['year'] <= year_range[1])
        ]
    if cargo_basis and 'cargo_basis' in filtered_demand.columns:
        filtered_demand = filtered_demand[filtered_demand['cargo_basis'].isin(cargo_basis)]
    
    if source_flexible:
        filtered_contracts = filtered_contracts[filtered_contracts['is_source_flexible'].isin(source_flexible)]
    if dest_flexible:
        filtered_contracts = filtered_contracts[filtered_contracts['is_destination_flexible'].isin(dest_flexible)]
    
    # Filter demand data to match filtered contracts
    if not filtered_contracts.empty:
        filtered_demand = filtered_demand[filtered_demand['id_contract'].isin(filtered_contracts['id_contract'])]
    
    # Generate volume analysis content using the section dropdown value
    content = create_volume_analysis_content(filtered_contracts, filtered_demand, volume_view_section, expanded_countries, expanded_sellers, expanded_destinations, expanded_buyers, year_range)
    
    # Extract charts, tables, and trends from the content
    if isinstance(content, html.Div):
        children = content.children if hasattr(content, 'children') else []
        
        # Parse content structure to separate components
        # We now have 4 rows: distribution charts, overview charts, main charts, tables
        # We'll combine the first three as "charts" and keep tables separate
        if len(children) >= 4:
            # Combine distribution and main charts
            charts = html.Div([children[0], children[2]])  # Distribution charts + Main charts
            trends = children[1]  # Overview charts (FOB/DES and trends)
            tables = children[3]  # Tables
        elif len(children) >= 3:
            charts = children[0]
            trends = children[1] 
            tables = children[2]
        else:
            charts = children[0] if len(children) > 0 else html.Div("No chart data available")
            trends = children[1] if len(children) > 1 else html.Div("No trend data available")
            tables = children[2] if len(children) > 2 else html.Div("No table data available")
        
        return charts, tables, trends
    
    return html.Div("No data available"), html.Div("No data available"), html.Div("No data available")

@callback(
    [Output('timeline-chart', 'figure'),
     Output('pricing-timeline-chart', 'figure')],
    [Input('destination-country-dropdown', 'value'),
     Input('contract-type-dropdown', 'value'),
     Input('pricing-type-dropdown', 'value'),
     Input('seller-company-dropdown', 'value'),
     Input('source-flexible-dropdown', 'value'),
     Input('dest-flexible-dropdown', 'value'),
     Input('timeline-metric-dropdown', 'value'),
     Input('contracts-data-store', 'children')]
)
def update_timeline_charts(dest_countries, contract_types, pricing_types, sellers, source_flexible, dest_flexible, timeline_metric, _snapshot_token):
    """Update contract signing timeline charts"""
    snapshot = _current_contracts_snapshot()
    contracts_df = snapshot.contracts
    demand_df = snapshot.demand
    price_assumptions_df = snapshot.price_assumptions
    price_formula_df = snapshot.price_formula
    
    # Timeline charts intentionally show the full historical range.
    filtered_contracts = contracts_df.copy()
    
    if dest_countries:
        filtered_contracts = filtered_contracts[filtered_contracts['country_name_delivery'].isin(dest_countries)]
    if contract_types:
        filtered_contracts = filtered_contracts[filtered_contracts['contract_type'].isin(contract_types)]
    if pricing_types:
        # Filter by detailed pricing type
        filtered_contracts = filtered_contracts[filtered_contracts['detailed_pricing_type'].isin(pricing_types)]
    if sellers:
        filtered_contracts = filtered_contracts[filtered_contracts['company_name_seller'].isin(sellers)]
    # Note: Year range filter is NOT applied to timeline charts - they show all years
    # Timeline charts should display the complete historical view
    if source_flexible:
        filtered_contracts = filtered_contracts[filtered_contracts['is_source_flexible'].isin(source_flexible)]
    if dest_flexible:
        filtered_contracts = filtered_contracts[filtered_contracts['is_destination_flexible'].isin(dest_flexible)]
    
    # Merge with price assumptions and formula to get better indexation data
    if not price_assumptions_df.empty:
        # Get indexation data from price assumptions (which has better coverage)
        # Only select columns that aren't already in contracts (except id_contract)
        indexation_cols = ['id_contract']
        
        # Add indexation columns with different names to avoid duplicates
        if 'indexation_category' in price_assumptions_df.columns:
            indexation_cols.append('indexation_category')
        if 'indexation_point' in price_assumptions_df.columns:
            indexation_cols.append('indexation_point')
        if 'oil_pricing_structure' in price_assumptions_df.columns:
            indexation_cols.append('oil_pricing_structure')
        if 'gas_pricing_structure' in price_assumptions_df.columns:
            indexation_cols.append('gas_pricing_structure')
        
        indexation_data = price_assumptions_df[indexation_cols].drop_duplicates('id_contract')
        
        # Rename columns to avoid conflicts
        rename_map = {col: f"{col}_pa" for col in indexation_cols if col != 'id_contract'}
        indexation_data = indexation_data.rename(columns=rename_map)
        
        # Merge with contracts to enhance indexation information
        filtered_contracts = filtered_contracts.merge(
            indexation_data, 
            on='id_contract', 
            how='left'
        )
        
        # Use price assumptions indexation_category if main one is missing
        if 'indexation_category_pa' in filtered_contracts.columns and 'indexation_category' in filtered_contracts.columns:
            filtered_contracts['indexation_category_enhanced'] = filtered_contracts['indexation_category'].combine_first(filtered_contracts['indexation_category_pa'])
        elif 'indexation_category_pa' in filtered_contracts.columns:
            filtered_contracts['indexation_category_enhanced'] = filtered_contracts['indexation_category_pa']
        elif 'indexation_category' in filtered_contracts.columns:
            filtered_contracts['indexation_category_enhanced'] = filtered_contracts['indexation_category']
        else:
            filtered_contracts['indexation_category_enhanced'] = pd.Series(index=filtered_contracts.index)
            
        # Similarly for indexation_point
        if 'indexation_point_pa' in filtered_contracts.columns and 'indexation_point' in filtered_contracts.columns:
            filtered_contracts['indexation_point_enhanced'] = filtered_contracts['indexation_point'].combine_first(filtered_contracts['indexation_point_pa'])
        elif 'indexation_point_pa' in filtered_contracts.columns:
            filtered_contracts['indexation_point_enhanced'] = filtered_contracts['indexation_point_pa']
        elif 'indexation_point' in filtered_contracts.columns:
            filtered_contracts['indexation_point_enhanced'] = filtered_contracts['indexation_point']
        else:
            filtered_contracts['indexation_point_enhanced'] = pd.Series(index=filtered_contracts.index)
    else:
        filtered_contracts['indexation_category_enhanced'] = filtered_contracts.get('indexation_category', pd.Series())
        filtered_contracts['indexation_point_enhanced'] = filtered_contracts.get('indexation_point', pd.Series())
    
    # Also merge with price formula for additional indexation data
    if not price_formula_df.empty:
        # Get unique indexation categories per contract from formula table
        formula_indexation = price_formula_df.groupby('id_contract')['indexation_category'].apply(
            lambda x: ', '.join(x.dropna().unique()) if x.notna().any() else None
        ).reset_index()
        formula_indexation.columns = ['id_contract', 'indexation_category_formula']
        
        # Merge with contracts
        filtered_contracts = filtered_contracts.merge(
            formula_indexation, 
            on='id_contract', 
            how='left'
        )
        
        # Further enhance indexation category with formula data
        filtered_contracts['indexation_category_enhanced'] = filtered_contracts['indexation_category_enhanced'].fillna(
            filtered_contracts.get('indexation_category_formula', pd.Series())
        )
    
    # Check if data is empty
    if filtered_contracts.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available for selected filters", x=0.5, y=0.5, showarrow=False)
        empty_fig.update_layout(height=400, title="Contract Signing Timeline")
        style_contracts_figure(empty_fig, height=420)
        return empty_fig, empty_fig
    
    # Create timeline chart (first chart)
    # Default to 'count' if timeline_metric is None
    if timeline_metric is None:
        timeline_metric = 'count'
    
    if timeline_metric == 'volume':
        # For volume, we need to merge with annual demand data to get volumes
        if not demand_df.empty:
            # Get total volume per contract per year
            volume_by_contract_year = demand_df.groupby(['id_contract', 'year'])['acq_volume__mmtpa'].sum().reset_index()
            
            # Merge with filtered contracts to get signing year
            volume_with_signing = volume_by_contract_year.merge(
                filtered_contracts[['id_contract', 'contract_year_signed']].drop_duplicates(),
                on='id_contract',
                how='inner'
            )
            
            # Group by signing year and sum volumes
            contracts_by_year = volume_with_signing.groupby('contract_year_signed')['acq_volume__mmtpa'].sum().reset_index()
            contracts_by_year.columns = ['Year', 'Volume']
            
            y_col = 'Volume'
            y_title = "Total Volume (MTPA)"
            hover_template = '<b>Year:</b> %{x}<br><b>Volume:</b> %{y:.2f} MTPA<extra></extra>'
        else:
            # If no demand data, fall back to contract count
            contracts_by_year = filtered_contracts.groupby('contract_year_signed').size().reset_index()
            contracts_by_year.columns = ['Year', 'Contracts']
            y_col = 'Contracts'
            y_title = "Number of Contracts"
            hover_template = '<b>Year:</b> %{x}<br><b>Contracts:</b> %{y}<extra></extra>'
    else:
        # Count contracts
        contracts_by_year = filtered_contracts.groupby('contract_year_signed').size().reset_index()
        contracts_by_year.columns = ['Year', 'Contracts']
        y_col = 'Contracts'
        y_title = "Number of Contracts"
        hover_template = '<b>Year:</b> %{x}<br><b>Contracts:</b> %{y}<extra></extra>'
    
    timeline_fig = px.line(
        contracts_by_year,
        x='Year',
        y=y_col,
        title="Contract Signing Timeline",
        markers=True
    )
    
    timeline_fig.update_layout(
        height=400,
        xaxis_title="Year",
        yaxis_title=y_title,
        template="plotly_white",
        hovermode='x unified'
    )
    
    timeline_fig.update_traces(
        line_color='#2E86C1',
        line_width=3,
        marker_size=8,
        marker_color='#1B4F72',
        hovertemplate=hover_template
    )
    
    # Create pricing distribution by year chart (second chart)
    # Prepare data with detailed pricing categories
    yearly_pricing_data = []
    
    # If volume metric is selected and demand data is available, prepare volume data
    volume_by_contract = {}
    if timeline_metric == 'volume' and not demand_df.empty:
        # Get total volume per contract
        volume_by_contract = demand_df.groupby('id_contract')['acq_volume__mmtpa'].sum().to_dict()
    
    for _, row in filtered_contracts.iterrows():
        year = row['contract_year_signed']
        pricing_type = row['contract_pricing_type']
        indexation_cat = row.get('indexation_category_enhanced', row.get('indexation_category', None))
        indexation_point = row.get('indexation_point_enhanced', row.get('indexation_point', None))
        oil_structure = row.get('oil_pricing_structure', None)
        gas_structure = row.get('gas_pricing_structure', None)
        
        # Create detailed pricing label with specific indexation details
        # Check if it's indexed - either by pricing_type OR by having indexation data
        is_indexed = (pricing_type in ['Index', 'Indexed'] or 
                     (indexation_cat and pd.notna(indexation_cat) and str(indexation_cat).strip() not in ['', 'None', 'Unknown', 'nan']) or
                     (indexation_point and pd.notna(indexation_point) and str(indexation_point).strip() not in ['', 'None', 'Unknown', 'nan']) or
                     (oil_structure and pd.notna(oil_structure) and str(oil_structure).strip() not in ['', 'None', 'Unknown']) or
                     (gas_structure and pd.notna(gas_structure) and str(gas_structure).strip() not in ['', 'None', 'Unknown']))
        
        if is_indexed:
            # FIRST: Check if explicitly marked as Hybrid in indexation fields
            combined_text = f"{str(indexation_cat).lower() if indexation_cat else ''} {str(indexation_point).lower() if indexation_point else ''}"
            
            if 'hybrid' in combined_text or 'mixed' in combined_text:
                # It's explicitly hybrid - try to get details from pricing structures
                has_oil = oil_structure and pd.notna(oil_structure) and str(oil_structure).strip() not in ['', 'None', 'Unknown']
                has_gas = gas_structure and pd.notna(gas_structure) and str(gas_structure).strip() not in ['', 'None', 'Unknown']
                
                if has_oil and has_gas:
                    oil_detail = extract_index_detail(oil_structure, indexation_cat, 'oil', indexation_point)
                    gas_detail = extract_index_detail(gas_structure, indexation_cat, 'gas', indexation_point)
                    if oil_detail and gas_detail:
                        pricing_label = f"Index - Hybrid ({oil_detail}/{gas_detail})"
                    else:
                        pricing_label = "Index - Hybrid"
                else:
                    pricing_label = "Index - Hybrid"
            else:
                # SECOND: Check pricing structures
                has_oil = oil_structure and pd.notna(oil_structure) and str(oil_structure).strip() not in ['', 'None', 'Unknown']
                has_gas = gas_structure and pd.notna(gas_structure) and str(gas_structure).strip() not in ['', 'None', 'Unknown']
                
                if has_oil and has_gas:
                    # Both structures present means hybrid
                    oil_detail = extract_index_detail(oil_structure, indexation_cat, 'oil', indexation_point)
                    gas_detail = extract_index_detail(gas_structure, indexation_cat, 'gas', indexation_point)
                    if oil_detail and gas_detail:
                        pricing_label = f"Index - Hybrid ({oil_detail}/{gas_detail})"
                    else:
                        pricing_label = "Index - Hybrid"
                elif has_oil:
                    # Oil indexed - show specific oil index
                    oil_detail = extract_index_detail(oil_structure, indexation_cat, 'oil', indexation_point)
                    pricing_label = f"Index - Oil ({oil_detail})" if oil_detail else "Index - Oil"
                elif has_gas:
                    # Gas indexed - show specific gas index
                    gas_detail = extract_index_detail(gas_structure, indexation_cat, 'gas', indexation_point)
                    pricing_label = f"Index - Gas ({gas_detail})" if gas_detail else "Index - Gas"
                else:
                    # THIRD: Categorize based on specific index mentions
                    if 'brent' in combined_text:
                        pricing_label = "Index - Oil (Brent)"
                    elif 'jcc' in combined_text or 'japan crude cocktail' in combined_text:
                        pricing_label = "Index - Oil (JCC)"
                    elif 'wti' in combined_text:
                        pricing_label = "Index - Oil (WTI)"
                    elif 'dubai' in combined_text:
                        pricing_label = "Index - Oil (Dubai)"
                    elif 'henry hub' in combined_text or 'hhub' in combined_text:
                        pricing_label = "Index - Gas (Henry Hub)"
                    elif 'nbp' in combined_text:
                        pricing_label = "Index - Gas (NBP)"
                    elif 'ttf' in combined_text:
                        pricing_label = "Index - Gas (TTF)"
                    elif 'jkm' in combined_text:
                        pricing_label = "Index - Gas (JKM)"
                    elif 'oil' in combined_text or 'crude' in combined_text:
                        pricing_label = "Index - Oil"
                    elif 'gas' in combined_text or 'slope' in combined_text:
                        pricing_label = "Index - Gas"
                    else:
                        # Use the actual indexation category if it's meaningful
                        if indexation_cat and len(str(indexation_cat)) < 30 and str(indexation_cat) not in ['None', 'Unknown', '', 'nan']:
                            pricing_label = f"Index - {indexation_cat}"
                        else:
                            pricing_label = "Unknown"
        elif pricing_type == 'Spot':
            pricing_label = 'Spot'
        elif pricing_type == 'Fixed':
            # Double-check it's not actually indexed
            if (indexation_cat and pd.notna(indexation_cat) and str(indexation_cat).strip() not in ['', 'None', 'Unknown', 'nan']):
                # It has indexation data, so treat as indexed despite 'Fixed' label
                pricing_label = "Unknown"
            else:
                pricing_label = 'Fixed'
        elif pricing_type in ['Unknown', None, '', 'nan'] or pd.isna(pricing_type):
            pricing_label = "Unknown"
        else:
            # Handle any other pricing types - keep original name
            pricing_label = str(pricing_type)
        
        # Extra safety check - never allow just "Index" or "Indexed"
        if pricing_label in ['Index', 'Indexed']:
            pricing_label = "Unknown"
        
        # Add volume if available
        contract_id = row.get('id_contract')
        volume = volume_by_contract.get(contract_id, 0) if timeline_metric == 'volume' else 1
        
        yearly_pricing_data.append({
            'Year': year,
            'Pricing Type': pricing_label,
            'Value': volume  # Either volume in MTPA or 1 for count
        })
    
    # Convert to DataFrame and aggregate
    pricing_df = pd.DataFrame(yearly_pricing_data)
    if timeline_metric == 'volume':
        # Sum volumes by year and pricing type
        pricing_counts = pricing_df.groupby(['Year', 'Pricing Type'])['Value'].sum().reset_index(name='Count')
    else:
        # Count contracts by year and pricing type
        pricing_counts = pricing_df.groupby(['Year', 'Pricing Type']).size().reset_index(name='Count')
    
    # Create comprehensive color map for all pricing types
    color_map = {
        'Fixed': '#2E86C1',                      # Blue
        'Spot': '#27AE60',                       # Green
        
        # Oil indices - Red shades
        'Index - Oil': '#E74C3C',                # Red
        'Index - Oil (Brent)': '#C0392B',        # Dark Red
        'Index - Oil (JCC)': '#E74C3C',          # Red
        'Index - Oil (WTI)': '#EC7063',          # Light Red
        'Index - Oil (Dubai)': '#F1948A',        # Lighter Red
        'Index - Oil (Oil)': '#E74C3C',          # Red
        
        # Gas indices - Orange/Yellow shades
        'Index - Gas': '#F39C12',                # Orange
        'Index - Gas (Henry Hub)': '#E67E22',    # Dark Orange
        'Index - Gas (NBP)': '#F39C12',          # Orange
        'Index - Gas (TTF)': '#F5B041',          # Light Orange
        'Index - Gas (JKM)': '#FAD7A0',          # Light Yellow
        'Index - Gas (Slope)': '#F8C471',        # Yellow-Orange
        'Index - Gas (Others)': '#F39C12',       # Orange
        
        # Hybrid indices - Purple shades
        'Index - Hybrid': '#9B59B6',             # Purple
        
        # Others
        'Index': '#95A5A6',                      # Gray (fallback)
        'Indexed': '#95A5A6',                    # Gray (fallback for raw 'Indexed')
        'Unknown': '#BDC3C7'                     # Light Gray
    }
    
    # Add any dynamic index types not in the map
    for pricing_type in pricing_counts['Pricing Type'].unique():
        if pricing_type not in color_map:
            if 'Oil' in pricing_type:
                color_map[pricing_type] = '#E74C3C'  # Default red for oil
            elif 'Gas' in pricing_type:
                color_map[pricing_type] = '#F39C12'  # Default orange for gas
            elif 'Hybrid' in pricing_type:
                color_map[pricing_type] = '#9B59B6'  # Default purple for hybrid
            else:
                color_map[pricing_type] = '#95A5A6'  # Default gray for others
    
    # Create stacked bar chart
    y_title_pricing = "Total Volume (MTPA)" if timeline_metric == 'volume' else "Number of Contracts"
    
    pricing_fig = px.bar(
        pricing_counts,
        x='Year',
        y='Count',
        color='Pricing Type',
        title="Pricing Type Distribution by Year (with Indexation Details)",
        text_auto=False,
        color_discrete_map=color_map
    )
    
    # Update hover template based on metric
    if timeline_metric == 'volume':
        pricing_fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Pricing: %{fullData.name}<br>Volume: %{y:.2f} MTPA<extra></extra>'
        )
    else:
        pricing_fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Pricing: %{fullData.name}<br>Contracts: %{y}<extra></extra>'
        )
    
    pricing_fig.update_layout(
        height=400,
        xaxis_title="Year",
        yaxis_title=y_title_pricing,
        template="plotly_white",
        barmode='stack',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=9)
        ),
        margin=dict(l=60, r=250, t=80, b=60),
        hovermode='x unified'
    )
    
    style_contracts_figure(timeline_fig, height=420)
    style_contracts_figure(pricing_fig, height=420)

    return timeline_fig, pricing_fig


@callback(
    [Output('contracts-table', 'columnDefs'),
     Output('contracts-table', 'rowData')],
    [Input('destination-country-dropdown', 'value'),
     Input('contract-type-dropdown', 'value'),
     Input('pricing-type-dropdown', 'value'),
     Input('seller-company-dropdown', 'value'),
     Input('year-range-slider', 'value'),
     Input('source-flexible-dropdown', 'value'),
     Input('dest-flexible-dropdown', 'value'),
     Input('contracts-data-store', 'children')]
)
def update_contracts_table(dest_countries, contract_types, 
                          pricing_types, sellers, year_range, source_flexible,
                          dest_flexible, _snapshot_token):
    """Update contracts table based on filters"""
    snapshot = _current_contracts_snapshot()
    contracts_df = snapshot.contracts
    price_assumptions_df = snapshot.price_assumptions
    price_formula_df = snapshot.price_formula
    
    # Apply same filters as tab content
    filtered_contracts = contracts_df.copy()
    
    if dest_countries:
        filtered_contracts = filtered_contracts[filtered_contracts['country_name_delivery'].isin(dest_countries)]
    if contract_types:
        filtered_contracts = filtered_contracts[filtered_contracts['contract_type'].isin(contract_types)]
    if pricing_types:
        # Filter by detailed pricing type
        filtered_contracts = filtered_contracts[filtered_contracts['detailed_pricing_type'].isin(pricing_types)]
    if sellers:
        filtered_contracts = filtered_contracts[filtered_contracts['company_name_seller'].isin(sellers)]
    if year_range:
        filtered_contracts = filtered_contracts[
            (filtered_contracts['contract_year_signed'] >= year_range[0]) & 
            (filtered_contracts['contract_year_signed'] <= year_range[1])
        ]
    if source_flexible:
        filtered_contracts = filtered_contracts[filtered_contracts['is_source_flexible'].isin(source_flexible)]
    if dest_flexible:
        filtered_contracts = filtered_contracts[filtered_contracts['is_destination_flexible'].isin(dest_flexible)]
    
    # Add detailed pricing type if not already present
    if 'detailed_pricing_type' not in filtered_contracts.columns:
        filtered_contracts['detailed_pricing_type'] = filtered_contracts.apply(get_detailed_pricing_type, axis=1)
    
    # Merge with price assumptions to get all pricing details
    if not price_assumptions_df.empty:
        # Get all pricing data from price assumptions
        pricing_data = price_assumptions_df.drop_duplicates('id_contract')
        
        # Merge with contracts
        filtered_contracts = filtered_contracts.merge(
            pricing_data,
            on='id_contract',
            how='left',
            suffixes=('', '_pricing')
        )
    
    # Merge with price formula for formula details
    if not price_formula_df.empty:
        # Aggregate formula data per contract
        formula_agg = price_formula_df.groupby('id_contract').agg({
            'pricing_structure': lambda x: ', '.join(x.dropna().unique()) if x.notna().any() else None,
            'index_pricing_point': lambda x: ', '.join(x.dropna().unique()) if x.notna().any() else None,
            'coefficient_type': lambda x: ', '.join(x.dropna().unique()) if x.notna().any() else None,
            'coefficient_value': 'mean',
            'lower_bound': 'min',
            'upper_bound': 'max',
            'lag_months': 'mean',
            'average_months': 'mean',
            'weighting': 'mean'
        }).reset_index()
        formula_agg.columns = ['id_contract', 'formula_structures', 'formula_index_points', 
                               'formula_coeff_types', 'avg_coefficient', 'formula_lower_bound', 
                               'formula_upper_bound', 'avg_lag_months', 'avg_average_months', 'formula_avg_weighting']
        
        # Merge with contracts
        filtered_contracts = filtered_contracts.merge(
            formula_agg,
            on='id_contract',
            how='left'
        )
    
    # Prepare comprehensive table columns with all contract details
    table_columns = [
        {'name': 'Contract ID', 'id': 'id_contract', 'type': 'numeric'},
        {'name': 'Contract Name', 'id': 'contract_name', 'type': 'text'},
        {'name': 'Primary Contract ID', 'id': 'id_contract_primary', 'type': 'numeric'},
        {'name': 'Primary Contract Name', 'id': 'contract_name_primary', 'type': 'text'},
        {'name': 'Type', 'id': 'contract_type', 'type': 'text'},
        {'name': 'Cargo Basis', 'id': 'cargo_basis', 'type': 'text'},
        {'name': 'Pricing Type (Original)', 'id': 'contract_pricing_type', 'type': 'text'},
        {'name': 'Detailed Pricing Type', 'id': 'detailed_pricing_type', 'type': 'text'},
        {'name': 'Date Signed', 'id': 'contract_date_signed', 'type': 'text'},
        {'name': 'Year Signed', 'id': 'contract_year_signed', 'type': 'numeric'},
        {'name': 'Start Date', 'id': 'contract_date_start', 'type': 'text'},
        {'name': 'End Date', 'id': 'contract_date_end', 'type': 'text'},
        {'name': 'Seller Company', 'id': 'company_name_seller', 'type': 'text'},
        {'name': 'Seller HQ Country', 'id': 'country_name_hq_company_seller', 'type': 'text'},
        {'name': 'Seller Category', 'id': 'company_category_seller', 'type': 'text'},
        {'name': 'Buyer Company', 'id': 'company_name_buyer', 'type': 'text'},
        {'name': 'Buyer HQ Country', 'id': 'country_name_hq_company_buyer', 'type': 'text'},
        {'name': 'Buyer Category', 'id': 'company_category_buyer', 'type': 'text'},
        {'name': 'Source Country', 'id': 'country_name_source', 'type': 'text'},
        {'name': 'LNG Plant ID', 'id': 'id_lng_plant_source', 'type': 'numeric'},
        {'name': 'LNG Plant', 'id': 'lng_plant_name_source', 'type': 'text'},
        {'name': 'LNG Project ID', 'id': 'id_lng_project', 'type': 'numeric'},
        {'name': 'LNG Project', 'id': 'lng_project_name', 'type': 'text'},
        {'name': 'Source Flexibility', 'id': 'flexibility_source', 'type': 'text'},
        {'name': 'Source Flexible', 'id': 'is_source_flexible', 'type': 'text'},
        {'name': 'Delivery Country', 'id': 'country_name_delivery', 'type': 'text'},
        {'name': 'Delivery Flexibility', 'id': 'flexibility_delivery', 'type': 'text'},
        {'name': 'Dest Flexible', 'id': 'is_destination_flexible', 'type': 'text'},
        {'name': 'Volume (MMTPA)', 'id': 'max_acq_volume', 'type': 'numeric', 'precision': 2},
        {'name': 'Volume Unit', 'id': 'max_acq_volume_unit', 'type': 'text'},
        {'name': 'Contract Note', 'id': 'contract_note', 'type': 'text'},
        {'name': 'Equity/Third Party', 'id': 'equity_third_party', 'type': 'text'},
        {'name': 'Dest Flexible vs End Users', 'id': 'destination_flexible_vs_end_users', 'type': 'text'},
        {'name': 'Indexation Category', 'id': 'indexation_category', 'type': 'text'},
        {'name': 'Indexation Point', 'id': 'indexation_point', 'type': 'text'}
    ]
    
    # Add pricing columns from price_assumptions if they exist
    pricing_columns = [
        {'name': 'Oil Pricing Structure', 'id': 'oil_pricing_structure', 'type': 'text'},
        {'name': 'Gas Pricing Structure', 'id': 'gas_pricing_structure', 'type': 'text'},
        {'name': 'Slope', 'id': 'slope', 'type': 'numeric', 'precision': 4},
        {'name': 'Intercept', 'id': 'intercept', 'type': 'numeric', 'precision': 2},
        {'name': 'Lower Inflection', 'id': 'lower_inflection', 'type': 'numeric', 'precision': 2},
        {'name': 'Slope Lower', 'id': 'slope_lower', 'type': 'numeric', 'precision': 4},
        {'name': 'Intercept Lower', 'id': 'intercept_lower', 'type': 'numeric', 'precision': 2},
        {'name': 'Upper Inflection', 'id': 'upper_inflection', 'type': 'numeric', 'precision': 2},
        {'name': 'Slope Upper', 'id': 'slope_upper', 'type': 'numeric', 'precision': 4},
        {'name': 'Intercept Upper', 'id': 'intercept_upper', 'type': 'numeric', 'precision': 2},
        {'name': 'Weighting', 'id': 'weighting', 'type': 'numeric', 'precision': 2},
        {'name': 'Fixed Fee', 'id': 'fixed_fee', 'type': 'numeric', 'precision': 2},
        {'name': 'Transport Tariff', 'id': 'transport_tariff', 'type': 'numeric', 'precision': 2},
        {'name': 'Regas Tariff', 'id': 'regas_tariff', 'type': 'numeric', 'precision': 2},
        {'name': 'Linkage %', 'id': 'linkage_percent', 'type': 'numeric', 'precision': 1},
        {'name': 'Oil Price at Signing', 'id': 'oil_price_in_signed_year', 'type': 'numeric', 'precision': 2},
        {'name': 'Normalized Slope', 'id': 'normalized_slope', 'type': 'numeric', 'precision': 4},
        {'name': 'Oil Indexed Ship Cost', 'id': 'oil_indexed_shipping_cost', 'type': 'numeric', 'precision': 2},
        {'name': 'Gas Indexed Ship Cost', 'id': 'gas_indexed_shipping_cost', 'type': 'numeric', 'precision': 2},
        {'name': 'Other Costs', 'id': 'other_costs', 'type': 'numeric', 'precision': 2}
    ]
    
    # Add formula columns if they exist
    formula_columns = [
        {'name': 'Formula Structures', 'id': 'formula_structures', 'type': 'text'},
        {'name': 'Formula Index Points', 'id': 'formula_index_points', 'type': 'text'},
        {'name': 'Formula Coeff Types', 'id': 'formula_coeff_types', 'type': 'text'},
        {'name': 'Avg Coefficient', 'id': 'avg_coefficient', 'type': 'numeric', 'precision': 4},
        {'name': 'Formula Lower Bound', 'id': 'formula_lower_bound', 'type': 'numeric', 'precision': 2},
        {'name': 'Formula Upper Bound', 'id': 'formula_upper_bound', 'type': 'numeric', 'precision': 2},
        {'name': 'Avg Lag Months', 'id': 'avg_lag_months', 'type': 'numeric', 'precision': 1},
        {'name': 'Avg Average Months', 'id': 'avg_average_months', 'type': 'numeric', 'precision': 1},
        {'name': 'Formula Avg Weighting', 'id': 'formula_avg_weighting', 'type': 'numeric', 'precision': 2}
    ]
    
    # Add enhanced indexation column if it exists
    if 'indexation_category_enhanced' in filtered_contracts.columns:
        table_columns.append({'name': 'Enhanced Index Cat', 'id': 'indexation_category_enhanced', 'type': 'text'})
    
    # Add all pricing columns that exist in the dataframe
    for col in pricing_columns + formula_columns:
        if col['id'] in filtered_contracts.columns:
            table_columns.append(col)
    
    # Only select columns that exist in the dataframe
    available_columns = [col['id'] for col in table_columns if col['id'] in filtered_contracts.columns]
    available_table_columns = [col for col in table_columns if col['id'] in filtered_contracts.columns]
    table_df = filtered_contracts[available_columns].copy()
    for date_column in CONTRACT_DETAIL_DATE_COLUMNS.intersection(table_df.columns):
        table_df[date_column] = table_df[date_column].map(_contracts_date_only)

    table_data = table_df.fillna('N/A').to_dict('records')
    column_defs = _contracts_ag_column_defs(
        available_table_columns,
        filterable=True,
        pinned_fields=['contract_name'],
        default_width=150,
    )
    
    return column_defs, _contracts_records(table_data)


def _ag_grid_clicked_row(clicked_events, table_data_list):
    if not clicked_events:
        raise PreventUpdate

    event = next((candidate for candidate in clicked_events if candidate), None)
    if not event:
        raise PreventUpdate

    clicked_row = event.get('data') if isinstance(event, dict) else None
    if clicked_row:
        return clicked_row

    row_index = event.get('rowIndex') if isinstance(event, dict) else None
    if row_index is None:
        row_index = event.get('row') if isinstance(event, dict) else None

    if row_index is None or not table_data_list:
        raise PreventUpdate

    table_data = next((data for data in table_data_list if data), None)
    if table_data and row_index < len(table_data):
        return table_data[row_index]

    raise PreventUpdate


def _toggle_expanded_entity(clicked_row, field, expanded_entities):
    expanded_entities = list(expanded_entities or [])
    entity_name = clicked_row.get(field, '')

    if entity_name.startswith('▼ '):
        clean_entity = entity_name[2:]
        if clean_entity in expanded_entities:
            expanded_entities.remove(clean_entity)
    elif entity_name.startswith('▶ '):
        clean_entity = entity_name[2:]
        if clean_entity not in expanded_entities:
            expanded_entities.append(clean_entity)

    return expanded_entities


@callback(
    Output('contracts-table', 'exportDataAsCsv'),
    Input('contracts-table-export-button', 'n_clicks'),
    prevent_initial_call=True,
)
def export_contracts_detail_table(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return True


@callback(
    Output({'type': 'volume-country-expandable-table', 'index': MATCH}, 'exportDataAsCsv'),
    Input({'type': 'volume-country-expandable-table-export-button', 'index': MATCH}, 'n_clicks'),
    prevent_initial_call=True,
)
def export_volume_country_table(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return True


@callback(
    Output({'type': 'volume-destination-expandable-table', 'index': MATCH}, 'exportDataAsCsv'),
    Input({'type': 'volume-destination-expandable-table-export-button', 'index': MATCH}, 'n_clicks'),
    prevent_initial_call=True,
)
def export_volume_destination_table(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return True


@callback(
    Output({'type': 'volume-seller-expandable-table', 'index': MATCH}, 'exportDataAsCsv'),
    Input({'type': 'volume-seller-expandable-table-export-button', 'index': MATCH}, 'n_clicks'),
    prevent_initial_call=True,
)
def export_volume_seller_table(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return True


@callback(
    Output({'type': 'volume-buyer-expandable-table', 'index': MATCH}, 'exportDataAsCsv'),
    Input({'type': 'volume-buyer-expandable-table-export-button', 'index': MATCH}, 'n_clicks'),
    prevent_initial_call=True,
)
def export_volume_buyer_table(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return True


# Callback to handle expanding/collapsing rows for volume country table
@callback(
    Output('volume-country-expanded-store', 'data'),
    [Input({'type': 'volume-country-expandable-table', 'index': ALL}, 'cellClicked')],
    [State('volume-country-expanded-store', 'data'),
     State({'type': 'volume-country-expandable-table', 'index': ALL}, 'rowData')]
)
def handle_volume_country_expansion(clicked_events, expanded_countries, table_data_list):
    """Handle clicking on rows to expand/collapse in volume country table"""
    clicked_row = _ag_grid_clicked_row(clicked_events, table_data_list)
    return _toggle_expanded_entity(clicked_row, 'country_name_source', expanded_countries)

# Callback to handle expanding/collapsing rows for volume seller table
@callback(
    Output('volume-seller-expanded-store', 'data'),
    [Input({'type': 'volume-seller-expandable-table', 'index': ALL}, 'cellClicked')],
    [State('volume-seller-expanded-store', 'data'),
     State({'type': 'volume-seller-expandable-table', 'index': ALL}, 'rowData')]
)
def handle_volume_seller_expansion(clicked_events, expanded_sellers, table_data_list):
    """Handle clicking on rows to expand/collapse in volume seller table"""
    clicked_row = _ag_grid_clicked_row(clicked_events, table_data_list)
    return _toggle_expanded_entity(clicked_row, 'company_name_seller', expanded_sellers)

# Callback to handle expanding/collapsing rows for volume destination table
@callback(
    Output('volume-destination-expanded-store', 'data'),
    [Input({'type': 'volume-destination-expandable-table', 'index': ALL}, 'cellClicked')],
    [State('volume-destination-expanded-store', 'data'),
     State({'type': 'volume-destination-expandable-table', 'index': ALL}, 'rowData')]
)
def handle_volume_destination_expansion(clicked_events, expanded_destinations, table_data_list):
    """Handle clicking on rows to expand/collapse in volume destination table"""
    clicked_row = _ag_grid_clicked_row(clicked_events, table_data_list)
    return _toggle_expanded_entity(clicked_row, 'country_name_delivery', expanded_destinations)

# Callback to handle expanding/collapsing rows for volume buyer table
@callback(
    Output('volume-buyer-expanded-store', 'data'),
    [Input({'type': 'volume-buyer-expandable-table', 'index': ALL}, 'cellClicked')],
    [State('volume-buyer-expanded-store', 'data'),
     State({'type': 'volume-buyer-expandable-table', 'index': ALL}, 'rowData')]
)
def handle_volume_buyer_expansion(clicked_events, expanded_buyers, table_data_list):
    """Handle clicking on rows to expand/collapse in volume buyer table"""
    clicked_row = _ag_grid_clicked_row(clicked_events, table_data_list)
    return _toggle_expanded_entity(clicked_row, 'company_name_buyer', expanded_buyers)
