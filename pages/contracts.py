from dash import html, dcc, dash_table, callback, Output, Input, State, Dash, ALL, callback_context
from dash.dash_table.Format import Format, Group, Scheme
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from io import StringIO
from dash.exceptions import PreventUpdate
import configparser
import os
from sqlalchemy import create_engine, text
import calendar

from utils.table_styles import StandardTableStyleManager, TABLE_COLORS

############################################ Style Constants ###################################################

# Enterprise Standard Styles following dash_style.md guidelines
CONTRACTS_SECTION_STYLES = {
    # Chart container styles - for content areas
    'chart_container': {
        'margin-bottom': '20px',
        'padding': '10px',
        'background-color': 'white',
        'border-radius': '6px',
        'border': '1px solid #e5e7eb',
        'width': '100%'
    },
    
    # Table container styles
    'table_container': {
        'margin-bottom': '20px',
        'padding': '10px',
        'background-color': 'white',
        'border-radius': '6px',
        'border': '1px solid #e5e7eb',
        'overflow-x': 'auto',
        'width': '100%'
    },
    
    # Layout panels
    'left_panel': {'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'},
    'right_panel': {'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'},
    'section_container': {
        'width': '100%',
        'marginBottom': '30px'
    },
    
    # Content wrapper
    'content_wrapper': {
        'background-color': 'white',
        'border-radius': '6px',
        'padding': '20px',
        'margin-top': '10px',
        'margin-bottom': '20px',
        'box-shadow': '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
        'width': '100%',
        'maxWidth': '100%'
    },
    
    # Footnote styles
    'footnote_style': {
        'font-size': '11px',
        'color': '#6b7280',
        'margin-top': '8px',
        'font-style': 'italic',
        'padding': '0 20px'
    }
}

############################################ postgres sql connection ###################################################
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    CONFIG_FILE_PATH = os.path.join(config_dir, 'config.ini')
except:
    CONFIG_FILE_PATH = 'config.ini'

config_reader = configparser.ConfigParser(interpolation=None)
config_reader.read(CONFIG_FILE_PATH)

DB_CONNECTION_STRING = config_reader.get('DATABASE', 'CONNECTION_STRING', fallback=None)
DB_SCHEMA = config_reader.get('DATABASE', 'SCHEMA', fallback=None)

engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)

def setup_database_connection():
    """Setup database connection using existing configuration"""
    return engine, DB_SCHEMA

############################################ Data Loading Functions ###################################################

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
        
        return df
    except Exception as e:
        print(f"Error loading contracts data: {e}")
        return pd.DataFrame()

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
        return pd.DataFrame()

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
        return pd.DataFrame()

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
        return pd.DataFrame()

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

def prepare_volume_table_for_display(demand_df, table_type, available_years, volume_view, expanded_entities=None):
    """Prepare volume data for display in DataTable with expandable rows showing source-destination breakdown
    
    Args:
        demand_df: DataFrame with volume data
        table_type: 'country', 'seller', or 'destination'
        available_years: List of years to display as columns
        volume_view: View selection from dropdown
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
        {'name': str(year), 'id': str(year), 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)} 
        for year in available_years
    ])
    
    return filtered_rows, columns


############################################ Layout Components ###################################################


def create_filter_controls(min_year=2000, max_year=2030, default_start=2015, default_end=2025):
    """Create filter controls panel"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Destination Country", className="form-label"),
                dcc.Dropdown(
                    id='destination-country-dropdown',
                    placeholder="Select destination countries...",
                    multi=True
                )
            ], width=3),
            dbc.Col([
                html.Label("Contract Type", className="form-label"),
                dcc.Dropdown(
                    id='contract-type-dropdown',
                    placeholder="Select contract types...",
                    multi=True
                )
            ], width=3),
            dbc.Col([
                html.Label("Pricing Type", className="form-label"),
                dcc.Dropdown(
                    id='pricing-type-dropdown',
                    placeholder="Select pricing types...",
                    multi=True
                )
            ], width=3),
            dbc.Col([
                html.Label("Seller Company", className="form-label"),
                dcc.Dropdown(
                    id='seller-company-dropdown',
                    placeholder="Select sellers...",
                    multi=True
                )
            ], width=3)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Year Range", className="form-label"),
                dcc.RangeSlider(
                    id='year-range-slider',
                    min=min_year,
                    max=max_year,
                    step=1,
                    value=[default_start, default_end],
                    marks={i: str(i) for i in range(min_year, max_year + 1, 5) if i % 5 == 0 or i in [min_year, max_year, default_start, default_end]},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=4),
            dbc.Col([
                html.Label("Cargo Basis", className="form-label"),
                dcc.Dropdown(
                    id='cargo-basis-dropdown',
                    placeholder="Select cargo basis...",
                    multi=True
                )
            ], width=2),
            dbc.Col([
                html.Label("Source Flexible", className="form-label"),
                dcc.Dropdown(
                    id='source-flexible-dropdown',
                    options=[
                        {'label': 'Flexible', 'value': 'Y'},
                        {'label': 'Not Flexible', 'value': 'N'},
                        {'label': 'Unknown', 'value': 'Unknown'}
                    ],
                    placeholder="Select flexibility...",
                    multi=True
                )
            ], width=3),
            dbc.Col([
                html.Label("Destination Flexible", className="form-label"),
                dcc.Dropdown(
                    id='dest-flexible-dropdown',
                    options=[
                        {'label': 'Flexible', 'value': 'Y'},
                        {'label': 'Not Flexible', 'value': 'N'},
                        {'label': 'Unknown', 'value': 'Unknown'}
                    ],
                    placeholder="Select flexibility...",
                    multi=True
                )
            ], width=3)
        ], className="mb-3"),
        html.Hr(className="mb-4")
    ], style={'padding': '20px', 'background-color': '#f8f9fa', 'border-radius': '8px', 'margin-bottom': '20px'})

def create_contracts_sections_layout():
    """Create the main sections layout replacing tabs"""
    return html.Div([
        # Contract Signing Timeline Section at the top
        create_timeline_section(),
        
        # Volume Analysis Section
        create_volume_analysis_section()
    ])

def create_timeline_section():
    """Create Contract Signing Timeline section at the top of the page"""
    return html.Div([
        # Enterprise Standard Inline Section Header
        html.Div([
            html.H3("Contract Signing Timeline", className="section-title-inline"),
            html.Label("Y-Axis Metric:", className="inline-filter-label"),
            dcc.Dropdown(
                id='timeline-metric-dropdown',
                options=[
                    {'label': 'Number of Contracts', 'value': 'count'},
                    {'label': 'Volume (MTPA)', 'value': 'volume'}
                ],
                value='count',
                className="inline-dropdown",
                placeholder='Select metric...',
                style={'width': '200px'}
            ),
        ], className="inline-section-header"),
        
        # Section content wrapper with two charts side by side
        html.Div([
            html.Div([
                dcc.Graph(
                    id='timeline-chart',
                    style={'height': '400px'}
                )
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'}),
            html.Div([
                dcc.Graph(
                    id='pricing-timeline-chart',
                    style={'height': '400px'}
                )
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'})
        ], style=CONTRACTS_SECTION_STYLES['content_wrapper']),
        
        html.Hr(className="section-divider")
    ], className="mb-4")

def create_volume_analysis_section():
    """Create Volume Analysis section following Enterprise Standard"""
    return html.Div([
        # Enterprise Standard Inline Section Header
        html.Div([
            html.H3("Volume Analysis", className="section-title-inline"),
            html.Label("View Mode:", className="inline-filter-label"),
            dcc.Dropdown(
                id='volume-view-section-dropdown',
                options=[
                    {'label': 'By Country', 'value': 'country'},
                    {'label': 'By Company', 'value': 'company'}
                ],
                value='country',
                className="inline-dropdown",
                placeholder='Select view...'
            ),
        ], className="inline-section-header"),
        
        # Section Content
        html.Div([
            # Two-panel layout for charts and controls
            html.Div([
                # Charts and tables will be inserted directly here
                html.Div(id="volume-analysis-charts"),
                html.Div(id="volume-analysis-tables"),
                html.Div(id="volume-analysis-trends"),
                
            ], style=CONTRACTS_SECTION_STYLES['section_container']),
            
            # Footnote
            html.P([
                html.I(className="fas fa-info-circle", style={'margin-right': '5px', 'color': '#2E86C1'}),
                "Volume data represents annual contracted quantities. Click on country/seller names to expand destination breakdown."
            ], style=CONTRACTS_SECTION_STYLES['footnote_style'])
            
        ], style=CONTRACTS_SECTION_STYLES['content_wrapper'])
    ])



def create_contracts_table():
    """Create interactive contracts data table"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Contracts Summary - All Details", className="mb-0"),
            dbc.Button("Export Data", id="export-button", className="btn-sm", color="primary")
        ]),
        dbc.CardBody([
            dash_table.DataTable(
                id='contracts-table',
                columns=[],
                data=[],
                fill_width=False,
                style_table={'height': '480px', 'overflowY': 'auto', 'overflowX': 'auto'},
                style_header=StandardTableStyleManager.get_base_datatable_config()['style_header'],
                style_cell={**StandardTableStyleManager.get_base_datatable_config()['style_cell'], 
                           'minWidth': '100px', 'maxWidth': '300px'},
                style_data_conditional=StandardTableStyleManager.get_base_datatable_config()['style_data_conditional'],
                sort_action="native",
                filter_action="native",
                page_action="none",
                fixed_rows={'headers': True},
                export_format="xlsx",
                export_headers="display"
            )
        ])
    ])

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
    
    # Debug output (can be commented out in production)
    print(f"\n=== Volume Analysis Debug ===")
    print(f"Total demand records: {len(demand_df)}")
    print(f"Unique source countries: {sorted(demand_df['country_name_source'].unique())}")
    print(f"Unique destination countries: {sorted(demand_df['country_name_delivery'].unique())}")
    
    # The total volume should be the same whether grouped by source or destination
    total_volume = demand_df['acq_volume__mmtpa'].sum()
    print(f"Total volume in dataset: {total_volume:.2f} MMTPA")
    
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
    
    # Debug source totals
    source_chart_total = country_volume['acq_volume__mmtpa'].sum()
    print(f"Source chart will show total: {source_chart_total:.2f} MMTPA")
    
    # Get source countries by total volume
    country_totals = country_volume.groupby('country_name_source')['acq_volume__mmtpa'].sum().sort_values(ascending=False)
    
    # Show more countries if needed for consistency with destination chart
    num_source_countries = len(country_totals)
    print(f"Total unique source countries: {num_source_countries}")
    
    # Match the destination chart approach - show up to 20 countries
    if num_source_countries <= 20:
        top_countries = country_totals.index.tolist()
        print(f"Showing all {num_source_countries} source countries")
    else:
        top_countries = country_totals.head(20).index.tolist()
        print(f"Showing top 20 of {num_source_countries} source countries")
        
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
    
    # Get all sellers sorted by total volume (no limit)
    seller_totals = seller_volume.groupby('company_name_seller')['acq_volume__mmtpa'].sum().sort_values(ascending=False)
    
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
        
        # Get all buyers sorted by total volume (no limit)
        buyer_totals = buyer_volume.groupby('company_name_buyer')['acq_volume__mmtpa'].sum().sort_values(ascending=False)
        
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
    
    # Debug destination totals
    dest_chart_total = dest_volume['acq_volume__mmtpa'].sum()
    print(f"Destination chart will show total: {dest_chart_total:.2f} MMTPA")
    
    # Check the difference
    if abs(source_chart_total - dest_chart_total) > 0.01:
        print(f"\nWARNING: Chart totals don't match!")
        print(f"Difference: {source_chart_total - dest_chart_total:.2f} MMTPA")
    else:
        print(f"✓ Chart totals match: {source_chart_total:.2f} MMTPA")
    
    print("=== End Debug ===")
    
    # Get destination countries by total volume
    dest_totals = dest_volume.groupby('country_name_delivery')['acq_volume__mmtpa'].sum().sort_values(ascending=False)
    
    # Show more countries in destination chart (top 20 or all if less than 20)
    num_destinations = len(dest_totals)
    print(f"Total unique destination countries: {num_destinations}")
    
    # If there are 20 or fewer destinations, show all; otherwise show top 20
    if num_destinations <= 20:
        top_destinations = dest_totals.index.tolist()
        print(f"Showing all {num_destinations} destination countries")
    else:
        top_destinations = dest_totals.head(20).index.tolist()
        print(f"Showing top 20 of {num_destinations} destination countries")
        
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
            demand_df, 'country', available_years, volume_view, expanded_countries
        )
        
        # Get base config and add special styling for TOTAL row
        country_config = StandardTableStyleManager.get_base_datatable_config()
        country_config['style_data_conditional'].extend([
            {
                'if': {'filter_query': '{country_name_source} = "TOTAL"'},
                'backgroundColor': '#2E86C1',
                'color': 'white',
                'fontWeight': 'bold',
                'border': '2px solid #1B4F72'
            }
        ])
        # Update style_table with fixed height
        country_config['style_table'].update({'height': '480px', 'overflowY': 'auto'})
        
        country_table = dash_table.DataTable(
            id={'type': 'volume-country-expandable-table', 'index': 'volume'},
            data=country_display_data,
            columns=country_columns,
            page_action="none",
            sort_action="native",
            fixed_rows={'headers': True},
            export_format="xlsx",
            export_headers="display",
            **country_config
        )
        
        # Prepare expandable seller table
        seller_display_data, seller_columns = prepare_volume_table_for_display(
            demand_df, 'seller', available_years, volume_view, expanded_sellers
        )
        
        # Get base config and add special styling for TOTAL row
        seller_config = StandardTableStyleManager.get_base_datatable_config()
        seller_config['style_data_conditional'].extend([
            {
                'if': {'filter_query': '{company_name_seller} = "TOTAL"'},
                'backgroundColor': '#2E86C1',
                'color': 'white',
                'fontWeight': 'bold',
                'border': '2px solid #1B4F72'
            }
        ])
        # Update style_table with fixed height
        seller_config['style_table'].update({'height': '480px', 'overflowY': 'auto'})
        
        seller_table = dash_table.DataTable(
            id={'type': 'volume-seller-expandable-table', 'index': 'volume'},
            data=seller_display_data,
            columns=seller_columns,
            page_action="none",
            sort_action="native",
            fixed_rows={'headers': True},
            export_format="xlsx",
            export_headers="display",
            **seller_config
        )
    else:
        country_table = html.Div("No country data available")
        seller_table = html.Div("No seller data available")
    
    # Determine layout based on volume_view selection
    if volume_view == 'country':
        main_charts_row = html.Div([
            html.Div([
                dcc.Graph(figure=country_fig, style={'height': '550px'})
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'}),
            html.Div([
                dcc.Graph(figure=dest_fig, style={'height': '550px'})
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'})
        ], style={'width': '100%', 'marginBottom': '20px'})
        
        # Prepare destination country table
        dest_display_data, dest_columns = prepare_volume_table_for_display(
            demand_df, 'destination', available_years, volume_view, expanded_destinations
        )
        
        # Get base config and add special styling for TOTAL row
        dest_config = StandardTableStyleManager.get_base_datatable_config()
        dest_config['style_data_conditional'].extend([
            {
                'if': {'filter_query': '{country_name_delivery} = "TOTAL"'},
                'backgroundColor': '#2E86C1',
                'color': 'white',
                'fontWeight': 'bold',
                'border': '2px solid #1B4F72'
            }
        ])
        # Update style_table with fixed height
        dest_config['style_table'].update({'height': '480px', 'overflowY': 'auto'})
        
        dest_table = dash_table.DataTable(
            id={'type': 'volume-destination-expandable-table', 'index': 'volume'},
            data=dest_display_data,
            columns=dest_columns,
            page_action="none",
            sort_action="native",
            fixed_rows={'headers': True},
            export_format="xlsx",
            export_headers="display",
            **dest_config
        )
        
        tables_row = html.Div([
            html.Div([
                html.Div([
                    html.H5("Volume by Source Country", style={'display': 'inline-block', 'marginBottom': '15px', 'fontSize': '16px', 'fontWeight': 'bold'}),
                    html.Span(" (Click ⬇ to export)", style={'fontSize': '12px', 'color': '#6c757d', 'marginLeft': '10px'})
                ]),
                country_table
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'}),
            html.Div([
                html.Div([
                    html.H5("Volume by Destination Country", style={'display': 'inline-block', 'marginBottom': '15px', 'fontSize': '16px', 'fontWeight': 'bold'}),
                    html.Span(" (Click ⬇ to export)", style={'fontSize': '12px', 'color': '#6c757d', 'marginLeft': '10px'})
                ]),
                dest_table
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'})
        ], style={'width': '100%', 'marginTop': '20px'})
        
    elif volume_view == 'company':
        main_charts_row = html.Div([
            html.Div([
                dcc.Graph(figure=seller_fig, style={'height': '550px'})
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'}),
            html.Div([
                dcc.Graph(figure=buyer_fig, style={'height': '550px'})
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'})
        ], style={'width': '100%', 'marginBottom': '20px'})
        
        # Prepare destination table for seller view too
        dest_display_data, dest_columns = prepare_volume_table_for_display(
            demand_df, 'destination', available_years, volume_view, expanded_destinations
        )
        
        dest_config = StandardTableStyleManager.get_base_datatable_config()
        dest_config['style_data_conditional'].extend([
            {
                'if': {'filter_query': '{country_name_delivery} = "TOTAL"'},
                'backgroundColor': '#2E86C1',
                'color': 'white',
                'fontWeight': 'bold',
                'border': '2px solid #1B4F72'
            }
        ])
        # Update style_table with fixed height
        dest_config['style_table'].update({'height': '480px', 'overflowY': 'auto'})
        
        dest_table = dash_table.DataTable(
            id={'type': 'volume-destination-expandable-table', 'index': 'volume'},
            data=dest_display_data,
            columns=dest_columns,
            page_action="none",
            sort_action="native",
            fixed_rows={'headers': True},
            export_format="xlsx",
            export_headers="display",
            **dest_config
        )
        
        # For company view, show seller and buyer tables
        # Prepare buyer table similar to seller table
        buyer_display_data, buyer_columns = prepare_volume_table_for_display(
            demand_df, 'buyer', available_years, volume_view, expanded_buyers
        )
        
        buyer_config = StandardTableStyleManager.get_base_datatable_config()
        buyer_config['style_data_conditional'].extend([
            {
                'if': {'filter_query': '{company_name_buyer} = "TOTAL"'},
                'backgroundColor': '#27AE60',
                'color': 'white',
                'fontWeight': 'bold',
                'border': '2px solid #145A32'
            }
        ])
        # Update style_table with fixed height
        buyer_config['style_table'].update({'height': '480px', 'overflowY': 'auto'})
        
        buyer_table = dash_table.DataTable(
            id={'type': 'volume-buyer-expandable-table', 'index': 'volume'},
            data=buyer_display_data,
            columns=buyer_columns,
            page_action="none",
            sort_action="native",
            fixed_rows={'headers': True},
            export_format="xlsx",
            export_headers="display",
            **buyer_config
        )
        
        tables_row = html.Div([
            html.Div([
                html.Div([
                    html.H5("Volume by Seller Company", style={'display': 'inline-block', 'marginBottom': '15px', 'fontSize': '16px', 'fontWeight': 'bold'}),
                    html.Span(" (Click ⬇ to export)", style={'fontSize': '12px', 'color': '#6c757d', 'marginLeft': '10px'})
                ]),
                seller_table
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'}),
            html.Div([
                html.Div([
                    html.H5("Volume by Buyer Company", style={'display': 'inline-block', 'marginBottom': '15px', 'fontSize': '16px', 'fontWeight': 'bold'}),
                    html.Span(" (Click ⬇ to export)", style={'fontSize': '12px', 'color': '#6c757d', 'marginLeft': '10px'})
                ]),
                buyer_table
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'})
        ], style={'width': '100%', 'marginTop': '20px'})
        
    else:  # fallback to country view
        main_charts_row = html.Div([
            html.Div([
                dcc.Graph(figure=country_fig, style={'height': '550px'})
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'}),
            html.Div([
                dcc.Graph(figure=dest_fig, style={'height': '550px'})
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'})
        ], style={'width': '100%', 'marginBottom': '20px'})
        
        # Prepare destination table
        dest_display_data, dest_columns = prepare_volume_table_for_display(
            demand_df, 'destination', available_years, volume_view, expanded_destinations
        )
        
        dest_config = StandardTableStyleManager.get_base_datatable_config()
        dest_config['style_data_conditional'].extend([
            {
                'if': {'filter_query': '{country_name_delivery} = "TOTAL"'},
                'backgroundColor': '#2E86C1',
                'color': 'white',
                'fontWeight': 'bold',
                'border': '2px solid #1B4F72'
            }
        ])
        # Update style_table with fixed height
        dest_config['style_table'].update({'height': '480px', 'overflowY': 'auto'})
        
        dest_table = dash_table.DataTable(
            id={'type': 'volume-destination-expandable-table', 'index': 'volume'},
            data=dest_display_data,
            columns=dest_columns,
            page_action="none",
            sort_action="native",
            fixed_rows={'headers': True},
            export_format="xlsx",
            export_headers="display",
            **dest_config
        )
        
        tables_row = html.Div([
            html.Div([
                html.Div([
                    html.H5("Volume by Source Country", style={'display': 'inline-block', 'marginBottom': '15px', 'fontSize': '16px', 'fontWeight': 'bold'}),
                    html.Span(" (Click ⬇ to export)", style={'fontSize': '12px', 'color': '#6c757d', 'marginLeft': '10px'})
                ]),
                country_table
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'}),
            html.Div([
                html.Div([
                    html.H5("Volume by Destination Country", style={'display': 'inline-block', 'marginBottom': '15px', 'fontSize': '16px', 'fontWeight': 'bold'}),
                    html.Span(" (Click ⬇ to export)", style={'fontSize': '12px', 'color': '#6c757d', 'marginLeft': '10px'})
                ]),
                dest_table
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'})
        ], style={'width': '100%', 'marginTop': '20px'})
    
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
    
    # Create distribution charts row
    distribution_charts_row = html.Div([
        html.Div([
            dcc.Graph(figure=pricing_type_fig, style={'height': '450px'})
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'}),
        html.Div([
            dcc.Graph(figure=contract_type_fig, style={'height': '450px'})
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'})
    ], style={'width': '100%', 'marginBottom': '20px'})
    
    # Create the overview charts row (FOB/DES and Trend)
    overview_charts_row = html.Div([
        html.Div([
            dcc.Graph(figure=fob_des_fig, style={'height': '450px'})
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'}),
        html.Div([
            dcc.Graph(figure=trend_fig, style={'height': '450px'})
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 10px'})
    ], style={'width': '100%', 'marginBottom': '20px'})
    
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


def create_contract_analytics_content(contracts_df, demand_df):
    """Create contract analytics tab content"""
    if contracts_df.empty:
        return html.Div("No data available", className="text-center p-4")
    
    # Contract type distribution
    contract_types = contracts_df['contract_type'].value_counts()
    type_fig = px.pie(
        values=contract_types.values,
        names=contract_types.index,
        title="Contract Type Distribution"
    )
    type_fig.update_layout(height=400)
    
    # Create merged pricing type chart with indexation categories
    # For indexed contracts, show oil/gas/hybrid categorization
    pricing_data = []
    
    # Debug: Check what indexation categories we have
    index_contracts = contracts_df[contracts_df['contract_pricing_type'] == 'Index']
    if not index_contracts.empty:
        print(f"Found {len(index_contracts)} indexed contracts")
        print(f"Unique indexation categories: {index_contracts['indexation_category'].unique()}")
    
    for _, row in contracts_df.iterrows():
        pricing_type = row['contract_pricing_type']
        indexation_cat = row.get('indexation_category', None)
        
        # Create detailed pricing label with oil/gas/hybrid categorization
        if pricing_type == 'Index':
            if indexation_cat and pd.notna(indexation_cat) and indexation_cat not in ['Unknown', '', None]:
                # Categorize based on indexation category content
                index_lower = str(indexation_cat).lower()
                if 'oil' in index_lower or 'brent' in index_lower or 'crude' in index_lower or 'wti' in index_lower:
                    pricing_label = "Index - Oil"
                elif 'gas' in index_lower or 'henry hub' in index_lower or 'hhub' in index_lower or 'nbp' in index_lower or 'ttf' in index_lower or 'jkm' in index_lower:
                    pricing_label = "Index - Gas"
                elif 'hybrid' in index_lower or ('oil' in index_lower and 'gas' in index_lower):
                    pricing_label = "Index - Hybrid"
                else:
                    pricing_label = "Unknown"  # Changed from f"Index - {indexation_cat}" to always use Unknown
            else:
                pricing_label = "Unknown"
        else:
            pricing_label = pricing_type
        
        # Extra safety check - never allow just "Index" or "Indexed"
        if pricing_label in ['Index', 'Indexed']:
            pricing_label = "Unknown"
        
        pricing_data.append(pricing_label)
    
    # Count the pricing types with categories
    pricing_counts = pd.Series(pricing_data).value_counts()
    
    # Create the merged pricing chart
    pricing_fig = px.pie(
        values=pricing_counts.values,
        names=pricing_counts.index,
        title="Pricing Type & Indexation Distribution"
    )
    pricing_fig.update_layout(height=400)
    
    return dbc.Row([
        dbc.Col([
            dcc.Graph(figure=type_fig)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=pricing_fig)
        ], width=6)
    ])

def create_pricing_analysis_content(contracts_df, price_assumptions_df):
    """Create pricing analysis tab content"""
    if contracts_df.empty:
        return html.Div("No data available", className="text-center p-4")
    
    # Create merged pricing type chart with indexation categories
    pricing_data = []
    
    # Debug: Check what indexation categories we have
    index_contracts = contracts_df[contracts_df['contract_pricing_type'] == 'Index']
    if not index_contracts.empty:
        print(f"Pricing Analysis - Found {len(index_contracts)} indexed contracts")
        print(f"Pricing Analysis - Unique indexation categories: {index_contracts['indexation_category'].unique()}")
    
    for _, row in contracts_df.iterrows():
        pricing_type = row['contract_pricing_type']
        indexation_cat = row.get('indexation_category', None)
        
        # Create detailed pricing label with oil/gas/hybrid categorization
        if pricing_type == 'Index':
            if indexation_cat and pd.notna(indexation_cat) and indexation_cat not in ['Unknown', '', None]:
                # Categorize based on indexation category content
                index_lower = str(indexation_cat).lower()
                if 'oil' in index_lower or 'brent' in index_lower or 'crude' in index_lower or 'wti' in index_lower:
                    pricing_label = "Index - Oil"
                elif 'gas' in index_lower or 'henry hub' in index_lower or 'hhub' in index_lower or 'nbp' in index_lower or 'ttf' in index_lower or 'jkm' in index_lower:
                    pricing_label = "Index - Gas"
                elif 'hybrid' in index_lower or ('oil' in index_lower and 'gas' in index_lower):
                    pricing_label = "Index - Hybrid"
                else:
                    pricing_label = "Unknown"  # Changed from f"Index - {indexation_cat}" to always use Unknown
            else:
                pricing_label = "Unknown"
        else:
            pricing_label = pricing_type
        
        # Extra safety check - never allow just "Index" or "Indexed"
        if pricing_label in ['Index', 'Indexed']:
            pricing_label = "Unknown"
        
        pricing_data.append(pricing_label)
    
    # Count the pricing types with categories
    pricing_counts = pd.Series(pricing_data).value_counts()
    
    # Create the merged pricing/indexation chart
    merged_pricing_fig = px.pie(
        values=pricing_counts.values,
        names=pricing_counts.index,
        title="Pricing Type & Indexation Distribution"
    )
    merged_pricing_fig.update_layout(height=400)
    
    # Flexibility analysis
    flexibility_data = {
        'Source Flexible': contracts_df['is_source_flexible'].value_counts().get('True', 0),
        'Source Not Flexible': contracts_df['is_source_flexible'].value_counts().get('False', 0),
        'Destination Flexible': contracts_df['is_destination_flexible'].value_counts().get('True', 0),
        'Destination Not Flexible': contracts_df['is_destination_flexible'].value_counts().get('False', 0)
    }
    
    flexibility_fig = px.bar(
        x=list(flexibility_data.keys()),
        y=list(flexibility_data.values()),
        title="Contract Flexibility Analysis"
    )
    flexibility_fig.update_layout(height=400)
    
    return dbc.Row([
        dbc.Col([
            dcc.Graph(figure=merged_pricing_fig)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=flexibility_fig)
        ], width=6)
    ])


############################################ Main Layout ###################################################

# Load initial data
contracts_df = load_contracts_data()
demand_df = load_annual_demand_data()
price_assumptions_df = load_price_assumptions_data()
price_formula_df = load_price_formula_data()

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

# Enhance contracts with oil/gas pricing structures if available
if not price_assumptions_df.empty:
    # Merge pricing structures
    pricing_cols = ['id_contract', 'indexation_category', 'indexation_point', 'oil_pricing_structure', 'gas_pricing_structure']
    available_pricing_cols = [col for col in pricing_cols if col in price_assumptions_df.columns]
    if available_pricing_cols:
        pricing_data = price_assumptions_df[available_pricing_cols].drop_duplicates('id_contract')
        contracts_df = contracts_df.merge(pricing_data, on='id_contract', how='left', suffixes=('', '_pa'))
        
        # Use enhanced indexation if available
        if 'indexation_category_pa' in contracts_df.columns:
            contracts_df['indexation_category'] = contracts_df['indexation_category'].combine_first(contracts_df['indexation_category_pa'])
        if 'indexation_point_pa' in contracts_df.columns:
            contracts_df['indexation_point'] = contracts_df['indexation_point'].combine_first(contracts_df['indexation_point_pa'])

# Add detailed pricing type column
contracts_df['detailed_pricing_type'] = contracts_df.apply(get_detailed_pricing_type, axis=1)

# Calculate year range from data
from datetime import datetime
current_year = datetime.now().year

# Get min year from contract signing dates
valid_sign_years = contracts_df['contract_year_signed'].dropna()
if not valid_sign_years.empty:
    min_year = int(valid_sign_years.min())
else:
    min_year = 2000

# Get max year from contract end dates
# Parse contract_date_end to extract years (format might be YYYY-MM-DD or similar)
end_years = []
if 'contract_date_end' in contracts_df.columns:
    end_dates = contracts_df['contract_date_end'].dropna()
    for date_str in end_dates:
        try:
            # Try to extract year from string - handle various formats
            date_str = str(date_str).strip()
            if date_str and date_str not in ['', 'None', 'nan', 'NaT']:
                # Extract year (assuming it's 4 digits)
                import re
                year_match = re.search(r'20\d{2}|19\d{2}', date_str)
                if year_match:
                    end_years.append(int(year_match.group()))
        except:
            continue
    
    if end_years:
        max_year = max(end_years)
    else:
        # Fallback to max signing year if no valid end dates
        max_year = int(valid_sign_years.max()) if not valid_sign_years.empty else current_year
else:
    max_year = int(valid_sign_years.max()) if not valid_sign_years.empty else current_year

# Ensure max_year is reasonable (not too far in future)
max_year = min(max_year, current_year + 30)  # Cap at 30 years from now

# Default range: previous year to last available year
default_start = max(min_year, current_year - 1)  # Ensure start is not before min_year
default_end = max_year


layout = html.Div([
    # Filter Controls
    create_filter_controls(min_year, max_year, default_start, default_end),
    
    # Main Sections (replace tabs)
    create_contracts_sections_layout(),
    
    # Interactive Data Table
    create_contracts_table(),
    
    # Hidden div to store data
    html.Div(id='contracts-data-store', style={'display': 'none'}),
    
    # Stores for expanded states in volume tables
    dcc.Store(id='volume-country-expanded-store', data=[]),
    dcc.Store(id='volume-seller-expanded-store', data=[]),
    dcc.Store(id='volume-destination-expanded-store', data=[]),
    dcc.Store(id='volume-buyer-expanded-store', data=[])
])

############################################ Callbacks ###################################################

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
     Input('volume-buyer-expanded-store', 'data')]
)
def update_volume_analysis_section(dest_countries, contract_types, 
                                  pricing_types, sellers, year_range, cargo_basis, 
                                  source_flexible, dest_flexible, volume_view_section,
                                  expanded_countries, expanded_sellers, expanded_destinations, expanded_buyers):
    """Update volume analysis section content"""
    
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
     Input('year-range-slider', 'value'),
     Input('source-flexible-dropdown', 'value'),
     Input('dest-flexible-dropdown', 'value'),
     Input('timeline-metric-dropdown', 'value')]
)
def update_timeline_charts(dest_countries, contract_types, pricing_types, sellers, year_range, source_flexible, dest_flexible, timeline_metric):
    """Update contract signing timeline charts"""
    
    # Apply filters (except year_range - timeline shows all years)
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
    
    # Debug indexation categories - check both 'Index' and 'Indexed'
    print(f"\n=== Timeline Pricing Debug ===")
    print(f"Contract pricing types in data: {filtered_contracts['contract_pricing_type'].value_counts().to_dict()}")
    
    # Check for contracts with Unknown pricing but indexation data
    unknown_pricing = filtered_contracts[filtered_contracts['contract_pricing_type'].isin(['Unknown', None, ''])]
    if not unknown_pricing.empty:
        has_index_cat = unknown_pricing['indexation_category'].notna().sum() if 'indexation_category' in unknown_pricing.columns else 0
        has_index_point = unknown_pricing['indexation_point'].notna().sum() if 'indexation_point' in unknown_pricing.columns else 0
        print(f"\nContracts with Unknown pricing type: {len(unknown_pricing)}")
        print(f"  - With indexation_category: {has_index_cat}")
        print(f"  - With indexation_point: {has_index_point}")
        
        if 'indexation_category_enhanced' in unknown_pricing.columns:
            sample_cats = unknown_pricing[unknown_pricing['indexation_category_enhanced'].notna()]['indexation_category_enhanced'].head(5)
            if not sample_cats.empty:
                print(f"  Sample indexation categories in Unknown contracts:")
                for cat in sample_cats:
                    print(f"    - {cat}")
    
    index_contracts = filtered_contracts[filtered_contracts['contract_pricing_type'].isin(['Index', 'Indexed'])]
    if not index_contracts.empty:
        print(f"Total indexed contracts: {len(index_contracts)}")
        
        # Check original indexation category
        if 'indexation_category' in index_contracts.columns:
            orig_cats = index_contracts['indexation_category'].value_counts().head(10)
            print(f"\nOriginal indexation categories (top 10):")
            for cat, count in orig_cats.items():
                print(f"  {cat}: {count}")
        
        # Check enhanced indexation category
        if 'indexation_category_enhanced' in index_contracts.columns:
            enh_cats = index_contracts['indexation_category_enhanced'].value_counts().head(10)
            print(f"\nEnhanced indexation categories (top 10):")
            for cat, count in enh_cats.items():
                print(f"  {cat}: {count}")
        
        # Check pricing structures
        if 'oil_pricing_structure' in index_contracts.columns:
            oil_structs = index_contracts['oil_pricing_structure'].dropna()
            if not oil_structs.empty:
                print(f"\nOil pricing structures (sample):")
                for struct in oil_structs.head(5):
                    print(f"  {struct}")
            oil_count = oil_structs.shape[0]
            print(f"Total contracts with oil pricing structure: {oil_count}")
        
        if 'gas_pricing_structure' in index_contracts.columns:
            gas_structs = index_contracts['gas_pricing_structure'].dropna()
            if not gas_structs.empty:
                print(f"\nGas pricing structures (sample):")
                for struct in gas_structs.head(5):
                    print(f"  {struct}")
            gas_count = gas_structs.shape[0]
            print(f"Total contracts with gas pricing structure: {gas_count}")
        
        # Check both oil and gas
        if 'oil_pricing_structure' in index_contracts.columns and 'gas_pricing_structure' in index_contracts.columns:
            both = index_contracts[(index_contracts['oil_pricing_structure'].notna()) & (index_contracts['gas_pricing_structure'].notna())]
            print(f"\nContracts with BOTH oil and gas structures: {len(both)}")
    
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
                print(f"WARNING: Contract {row.get('id_contract', 'unknown')} labeled as Fixed but has indexation: {indexation_cat}")
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
    
    # Debug what pricing types we ended up with
    print(f"\n=== Final Pricing Labels ===")
    print(f"Unique pricing types in chart: {pricing_counts['Pricing Type'].unique()}")
    label_counts = pricing_counts.groupby('Pricing Type')['Count'].sum().sort_values(ascending=False)
    print(f"\nTotal contracts by pricing type:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    print(f"=== End Debug ===\n")
    
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
    hover_name = "Volume" if timeline_metric == 'volume' else "Contracts"
    
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
    
    return timeline_fig, pricing_fig


@callback(
    [Output('contracts-table', 'columns'),
     Output('contracts-table', 'data')],
    [Input('destination-country-dropdown', 'value'),
     Input('contract-type-dropdown', 'value'),
     Input('pricing-type-dropdown', 'value'),
     Input('seller-company-dropdown', 'value'),
     Input('year-range-slider', 'value'),
     Input('cargo-basis-dropdown', 'value'),
     Input('source-flexible-dropdown', 'value'),
     Input('dest-flexible-dropdown', 'value')]
)
def update_contracts_table(dest_countries, contract_types, 
                          pricing_types, sellers, year_range, cargo_basis,
                          source_flexible, dest_flexible):
    """Update contracts table based on filters"""
    
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
        {'name': 'Volume (MMTPA)', 'id': 'max_acq_volume', 'type': 'numeric', 'format': Format(precision=2)},
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
        {'name': 'Slope', 'id': 'slope', 'type': 'numeric', 'format': Format(precision=4)},
        {'name': 'Intercept', 'id': 'intercept', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Lower Inflection', 'id': 'lower_inflection', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Slope Lower', 'id': 'slope_lower', 'type': 'numeric', 'format': Format(precision=4)},
        {'name': 'Intercept Lower', 'id': 'intercept_lower', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Upper Inflection', 'id': 'upper_inflection', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Slope Upper', 'id': 'slope_upper', 'type': 'numeric', 'format': Format(precision=4)},
        {'name': 'Intercept Upper', 'id': 'intercept_upper', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Weighting', 'id': 'weighting', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Fixed Fee', 'id': 'fixed_fee', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Transport Tariff', 'id': 'transport_tariff', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Regas Tariff', 'id': 'regas_tariff', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Linkage %', 'id': 'linkage_percent', 'type': 'numeric', 'format': Format(precision=1)},
        {'name': 'Oil Price at Signing', 'id': 'oil_price_in_signed_year', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Normalized Slope', 'id': 'normalized_slope', 'type': 'numeric', 'format': Format(precision=4)},
        {'name': 'Oil Indexed Ship Cost', 'id': 'oil_indexed_shipping_cost', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Gas Indexed Ship Cost', 'id': 'gas_indexed_shipping_cost', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Other Costs', 'id': 'other_costs', 'type': 'numeric', 'format': Format(precision=2)}
    ]
    
    # Add formula columns if they exist
    formula_columns = [
        {'name': 'Formula Structures', 'id': 'formula_structures', 'type': 'text'},
        {'name': 'Formula Index Points', 'id': 'formula_index_points', 'type': 'text'},
        {'name': 'Formula Coeff Types', 'id': 'formula_coeff_types', 'type': 'text'},
        {'name': 'Avg Coefficient', 'id': 'avg_coefficient', 'type': 'numeric', 'format': Format(precision=4)},
        {'name': 'Formula Lower Bound', 'id': 'formula_lower_bound', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Formula Upper Bound', 'id': 'formula_upper_bound', 'type': 'numeric', 'format': Format(precision=2)},
        {'name': 'Avg Lag Months', 'id': 'avg_lag_months', 'type': 'numeric', 'format': Format(precision=1)},
        {'name': 'Avg Average Months', 'id': 'avg_average_months', 'type': 'numeric', 'format': Format(precision=1)},
        {'name': 'Formula Avg Weighting', 'id': 'formula_avg_weighting', 'type': 'numeric', 'format': Format(precision=2)}
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
    table_data = filtered_contracts[available_columns].fillna('N/A').to_dict('records')
    
    return available_table_columns, table_data

# Callback to handle expanding/collapsing rows for volume country table
@callback(
    Output('volume-country-expanded-store', 'data'),
    [Input({'type': 'volume-country-expandable-table', 'index': ALL}, 'active_cell')],
    [State('volume-country-expanded-store', 'data'),
     State({'type': 'volume-country-expandable-table', 'index': ALL}, 'data')]
)
def handle_volume_country_expansion(active_cells, expanded_countries, table_data_list):
    """Handle clicking on rows to expand/collapse in volume country table"""
    if not active_cells or not table_data_list:
        raise PreventUpdate
    
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    prop_id = ctx.triggered[0]['prop_id']
    
    if 'volume-country-expandable-table' in prop_id and '.active_cell' in prop_id:
        active_cell = active_cells[0]
        if active_cell and 'row' in active_cell:
            table_data = table_data_list[0]
            if active_cell['row'] < len(table_data):
                clicked_row = table_data[active_cell['row']]
                
                # Extract country name from the clicked row
                country_name = clicked_row.get('country_name_source', '')
                
                # Remove expand/collapse indicators to get clean name
                if country_name.startswith('▼ '):
                    clean_country = country_name[2:]
                    # Collapse: remove from expanded list
                    if clean_country in expanded_countries:
                        expanded_countries.remove(clean_country)
                elif country_name.startswith('▶ '):
                    clean_country = country_name[2:]
                    # Expand: add to expanded list
                    if clean_country not in expanded_countries:
                        expanded_countries.append(clean_country)
    
    return expanded_countries

# Callback to handle expanding/collapsing rows for volume seller table
@callback(
    Output('volume-seller-expanded-store', 'data'),
    [Input({'type': 'volume-seller-expandable-table', 'index': ALL}, 'active_cell')],
    [State('volume-seller-expanded-store', 'data'),
     State({'type': 'volume-seller-expandable-table', 'index': ALL}, 'data')]
)
def handle_volume_seller_expansion(active_cells, expanded_sellers, table_data_list):
    """Handle clicking on rows to expand/collapse in volume seller table"""
    if not active_cells or not table_data_list:
        raise PreventUpdate
    
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    prop_id = ctx.triggered[0]['prop_id']
    
    if 'volume-seller-expandable-table' in prop_id and '.active_cell' in prop_id:
        active_cell = active_cells[0]
        if active_cell and 'row' in active_cell:
            table_data = table_data_list[0]
            if active_cell['row'] < len(table_data):
                clicked_row = table_data[active_cell['row']]
                
                # Extract seller name from the clicked row
                seller_name = clicked_row.get('company_name_seller', '')
                
                # Remove expand/collapse indicators to get clean name
                if seller_name.startswith('▼ '):
                    clean_seller = seller_name[2:]
                    # Collapse: remove from expanded list
                    if clean_seller in expanded_sellers:
                        expanded_sellers.remove(clean_seller)
                elif seller_name.startswith('▶ '):
                    clean_seller = seller_name[2:]
                    # Expand: add to expanded list
                    if clean_seller not in expanded_sellers:
                        expanded_sellers.append(clean_seller)
    
    return expanded_sellers

# Callback to handle expanding/collapsing rows for volume destination table
@callback(
    Output('volume-destination-expanded-store', 'data'),
    [Input({'type': 'volume-destination-expandable-table', 'index': ALL}, 'active_cell')],
    [State('volume-destination-expanded-store', 'data'),
     State({'type': 'volume-destination-expandable-table', 'index': ALL}, 'data')]
)
def handle_volume_destination_expansion(active_cells, expanded_destinations, table_data_list):
    """Handle clicking on rows to expand/collapse in volume destination table"""
    if not active_cells or not table_data_list:
        raise PreventUpdate
    
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    prop_id = ctx.triggered[0]['prop_id']
    
    if 'volume-destination-expandable-table' in prop_id and '.active_cell' in prop_id:
        active_cell = active_cells[0]
        if active_cell and 'row' in active_cell:
            table_data = table_data_list[0]
            if active_cell['row'] < len(table_data):
                clicked_row = table_data[active_cell['row']]
                
                # Extract destination name from the clicked row
                dest_name = clicked_row.get('country_name_delivery', '')
                
                # Remove expand/collapse indicators to get clean name
                if dest_name.startswith('▼ '):
                    clean_dest = dest_name[2:]
                    # Collapse: remove from expanded list
                    if clean_dest in expanded_destinations:
                        expanded_destinations.remove(clean_dest)
                elif dest_name.startswith('▶ '):
                    clean_dest = dest_name[2:]
                    # Expand: add to expanded list
                    if clean_dest not in expanded_destinations:
                        expanded_destinations.append(clean_dest)
    
    return expanded_destinations

# Callback to handle expanding/collapsing rows for volume buyer table
@callback(
    Output('volume-buyer-expanded-store', 'data'),
    [Input({'type': 'volume-buyer-expandable-table', 'index': ALL}, 'active_cell')],
    [State('volume-buyer-expanded-store', 'data'),
     State({'type': 'volume-buyer-expandable-table', 'index': ALL}, 'data')]
)
def handle_volume_buyer_expansion(active_cells, expanded_buyers, table_data_list):
    """Handle clicking on rows to expand/collapse in volume buyer table"""
    if not active_cells or not table_data_list:
        raise PreventUpdate
    
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    prop_id = ctx.triggered[0]['prop_id']
    
    if 'volume-buyer-expandable-table' in prop_id and '.active_cell' in prop_id:
        active_cell = active_cells[0]
        if active_cell and 'row' in active_cell:
            table_data = table_data_list[0]
            if active_cell['row'] < len(table_data):
                clicked_row = table_data[active_cell['row']]
                buyer_name = clicked_row.get('company_name_buyer', '')
                
                if expanded_buyers is None:
                    expanded_buyers = []
                
                if buyer_name.startswith('▼ '):
                    # Already expanded, collapse it
                    clean_buyer = buyer_name[2:]
                    if clean_buyer in expanded_buyers:
                        expanded_buyers.remove(clean_buyer)
                elif buyer_name.startswith('▶ '):
                    clean_buyer = buyer_name[2:]
                    # Expand: add to expanded list
                    if clean_buyer not in expanded_buyers:
                        expanded_buyers.append(clean_buyer)
    
    return expanded_buyers