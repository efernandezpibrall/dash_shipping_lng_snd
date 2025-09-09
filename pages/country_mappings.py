from dash import html, dcc, dash_table, callback, Output, Input, State
from dash.dash_table.Format import Format
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import configparser
import os
from sqlalchemy import create_engine, text
from utils.table_styles import StandardTableStyleManager, TABLE_COLORS
from app import app
import dash_leaflet as dl
import json
import requests

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

def fetch_country_mappings_data(engine, schema='at_lng'):
    """Fetch distinct country mappings data from the database"""
    try:
        with engine.connect() as conn:
            query = text(f"""
            SELECT DISTINCT
                country_name,
                continent,
                subcontinent,
                basin,
                country_classification_level1,
                country_classification,
                shipping_region,
                iso3
            FROM {schema}.mappings_country
            WHERE country_name IS NOT NULL
            ORDER BY country_name
            """)
            
            df = pd.read_sql(query, conn)
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.fillna('')
            
            return df
            
    except Exception as e:
        print(f"Error fetching country mappings data: {e}")
        return pd.DataFrame()

def create_summary_cards(df):
    """Create summary cards showing key statistics"""
    if df.empty:
        return html.Div("No data available")
    
    total_countries = len(df['country_name'].unique())
    
    classification_counts = df.groupby('country_classification_level1')['country_name'].nunique()
    continent_counts = df.groupby('continent')['country_name'].nunique()
    
    cards = []
    
    cards.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Countries", className="text-secondary", style={'marginBottom': '8px'}),
                    html.H3(f"{total_countries:,}", className="text-primary font-bold"),
                ])
            ], className="shadow-sm h-100")
        ], width=3)
    )
    
    cards.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Continents", className="text-secondary", style={'marginBottom': '8px'}),
                    html.H3(f"{len(continent_counts)}", className="text-primary font-bold"),
                ])
            ], className="shadow-sm h-100")
        ], width=3)
    )
    
    cards.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Classification Levels", className="text-secondary", style={'marginBottom': '8px'}),
                    html.H3(f"{len(classification_counts)}", className="text-primary font-bold"),
                ])
            ], className="shadow-sm h-100")
        ], width=3)
    )
    
    cards.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Shipping Regions", className="text-secondary", style={'marginBottom': '8px'}),
                    html.H3(f"{df['shipping_region'].nunique()}", className="text-primary font-bold"),
                ])
            ], className="shadow-sm h-100")
        ], width=3)
    )
    
    return dbc.Row(cards, className="mb-4")

def create_world_map(df, category='continent'):
    """Create a choropleth world map using dash-leaflet with iso3 codes"""
    if df.empty:
        return html.Div("No data available for map", style={'padding': '20px', 'textAlign': 'center'})
    
    # Map category column names to display names
    category_map = {
        'continent': 'Continent',
        'subcontinent': 'Subcontinent', 
        'basin': 'Basin',
        'country_classification_level1': 'Classification Level 1',
        'country_classification': 'Classification',
        'shipping_region': 'Shipping Region'
    }
    
    # Handle empty values
    df[category] = df[category].fillna('Not Assigned')
    df[category] = df[category].replace('', 'Not Assigned')
    
    # Get unique values for color mapping
    unique_values = sorted(df[category].unique())
    
    # Extended color palette with more distinct colors
    # Using colors that are visually distinct and colorblind-friendly where possible
    colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue  
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#ffff33',  # Yellow
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#999999',  # Grey
        '#66c2a5',  # Teal
        '#fc8d62',  # Light Orange
        '#8da0cb',  # Light Blue
        '#006400',  # Dark Green (replaced Light Pink)
        '#a6d854',  # Light Green
        '#ffd92f',  # Gold
        '#e5c494',  # Tan
        '#b3b3b3',  # Light Grey
        '#1b9e77',  # Dark Teal
        '#d95f02',  # Dark Orange
        '#7570b3',  # Dark Purple
        '#8b008b',  # Dark Magenta (replaced Magenta)
        '#66a61e',  # Olive Green
        '#e6ab02',  # Dark Gold
        '#a6761d',  # Dark Brown
        '#666666',  # Dark Grey
        '#1f78b4',  # Medium Blue
        '#b2df8a',  # Pale Green
        '#33a02c',  # Forest Green
        '#fb9a99',  # Pale Red
        '#fdbf6f',  # Pale Orange
        '#4b0082',  # Indigo (replaced Fuchsia)
        '#00ffff',  # Cyan
        '#800080',  # Deep Purple
        '#008080',  # Deep Teal
        '#800000',  # Maroon
        '#808000',  # Olive
        '#000080',  # Navy
        '#ff4500',  # Orange Red (replaced Deep Pink)
        '#00ff00',  # Lime
        '#00ff7f',  # Spring Green
    ]
    
    if len(unique_values) > len(colors):
        import random
        # Generate additional distinct colors if needed
        for i in range(len(unique_values) - len(colors)):
            # Generate colors with good contrast
            hue = (i * 137.5) % 360  # Use golden angle for better distribution
            colors.append(f'hsl({hue}, 70%, 50%)')
    
    color_map = dict(zip(unique_values, colors[:len(unique_values)]))
    
    try:
        # Use Natural Earth GeoJSON which has proper country boundaries
        geojson_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
        response = requests.get(geojson_url, timeout=10)
        geojson_data = response.json()
        
        # Create ISO3 to data mapping
        iso3_to_data = {}
        for _, row in df.iterrows():
            iso3_code = row.get('iso3', '').upper() if row.get('iso3') else ''
            if iso3_code:
                iso3_to_data[iso3_code] = {
                    'name': row['country_name'],
                    'value': row[category],
                    'continent': row.get('continent', ''),
                    'subcontinent': row.get('subcontinent', ''),
                    'basin': row.get('basin', ''),
                    'classification_l1': row.get('country_classification_level1', ''),
                    'classification': row.get('country_classification', ''),
                    'shipping_region': row.get('shipping_region', '')
                }
        
        # Process GeoJSON features
        for feature in geojson_data['features']:
            props = feature.get('properties', {})
            # Get ISO3 code from GeoJSON
            iso3_geojson = (props.get('ISO_A3') or props.get('ADM0_A3', '')).upper()
            country_name = props.get('NAME') or props.get('ADMIN', 'Unknown')
            
            # Match using ISO3 code
            if iso3_geojson in iso3_to_data:
                data = iso3_to_data[iso3_geojson]
                feature['properties']['__value'] = data['value']  # Store the value for grouping
                feature['properties']['__color'] = color_map[data['value']]
                feature['properties']['tooltip'] = data['name']  # Just the country name
            else:
                feature['properties']['__value'] = 'Not Assigned'  # Store as Not Assigned
                feature['properties']['__color'] = '#e5e7eb'
                feature['properties']['tooltip'] = country_name  # Just the country name
        
        # Create individual GeoJSON layers for each unique value/color
        # This approach ensures proper styling without JavaScript dependencies
        geojson_layers = []
        
        for value in unique_values:
            # Filter features for this value
            features_for_value = []
            for feature in geojson_data['features']:
                if feature['properties'].get('__value') == value or \
                   (value == 'Not Assigned' and '__color' in feature['properties'] and 
                    feature['properties']['__color'] == '#e5e7eb'):
                    features_for_value.append(feature)
            
            if features_for_value:
                # Create a FeatureCollection for this value
                feature_collection = {
                    "type": "FeatureCollection",
                    "features": features_for_value
                }
                
                # Add a GeoJSON layer for this value with its specific color
                geojson_layer = dl.GeoJSON(
                    data=feature_collection,
                    options=dict(
                        style=dict(
                            fillColor=color_map.get(value, '#e5e7eb'),
                            weight=0.5,
                            color='white',
                            fillOpacity=0.7,
                            fill=True
                        )
                    ),
                    hoverStyle=dict(
                        weight=2,
                        color='#333',
                        fillOpacity=0.9
                    ),
                    zoomToBoundsOnClick=False,
                    children=[dl.Tooltip(direction='top', permanent=False, offset=[0, 20], className='leaflet-tooltip-no-arrow')]  # Tooltip without arrow
                )
                geojson_layers.append(geojson_layer)
        
        # Combine all layers into a LayerGroup
        geojson = dl.LayerGroup(children=geojson_layers, id="geojson-layer")
        
        # Create legend
        legend_items = []
        for value in unique_values:
            legend_items.append(
                html.Div([
                    html.Div(style={
                        'backgroundColor': color_map[value],
                        'width': '20px',
                        'height': '20px',
                        'display': 'inline-block',
                        'marginRight': '8px',
                        'border': '1px solid #ddd',
                        'borderRadius': '3px',
                        'verticalAlign': 'middle'
                    }),
                    html.Span(str(value), style={'verticalAlign': 'middle', 'fontSize': '12px'})
                ], style={'marginBottom': '4px'})
            )
        
        legend = html.Div([
            html.H6(category_map.get(category, category), 
                   style={'marginBottom': '10px', 'fontWeight': 'bold', 'fontSize': '14px'}),
            *legend_items
        ], style={
            'position': 'absolute',
            'top': '10px',
            'right': '10px',
            'backgroundColor': 'rgba(255,255,255,0.95)',
            'padding': '12px',
            'border': '1px solid #ccc',
            'borderRadius': '5px',
            'zIndex': 1000,
            'maxHeight': '500px',
            'overflowY': 'auto',
            'minWidth': '180px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        })
        
        # Create the map
        map_component = dl.Map(
            children=[
                dl.TileLayer(url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"),
                geojson,
                dl.GeoJSON(id="info", children=[dl.Tooltip(id="tooltip")])
            ],
            style={'width': '100%', 'height': '650px'},
            center=[20, 0],
            zoom=2,
            worldCopyJump=True,
            preferCanvas=True
        )
        
        # Add custom CSS for tooltip styling
        custom_css = html.Div([
            map_component,
            legend
        ], style={'position': 'relative'})
        
        # Wrap with a div that includes inline styles for the tooltip
        return html.Div([
            dcc.Markdown("""
                <style>
                .leaflet-tooltip:before {
                    display: none !important;
                }
                .leaflet-tooltip {
                    background: rgba(255, 255, 255, 0.95) !important;
                    border: 1px solid #ccc !important;
                    border-radius: 4px !important;
                    padding: 4px 8px !important;
                    font-size: 12px !important;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
                }
                </style>
            """, dangerously_allow_html=True),
            custom_css
        ])
        
    except Exception as e:
        print(f"Error creating leaflet map: {e}")
        return html.Div(f"Error creating map: {str(e)}", style={'padding': '20px', 'textAlign': 'center'})


layout = html.Div([
    dcc.Store(id='country-mappings-data-store', storage_type='memory'),
    
    dcc.Interval(id='country-mappings-load-trigger', interval=1000*60*60*24, n_intervals=0, max_intervals=1),
    
    html.Div([
        html.H2('Country Mappings', className="page-title"),
        html.P('Geographic and classification mapping for all countries in the system', 
               className="text-secondary", style={'marginBottom': '20px'})
    ], style={'marginBottom': '30px', 'padding': '20px 20px 0px 20px'}),
    
    html.Div(id='summary-cards-container', style={'padding': '0px 20px'}),
    
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H5('Filters', className="text-primary font-bold", style={'marginBottom': '15px'}),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label('Continent', className="text-secondary font-semibold", 
                                     style={'fontSize': '13px', 'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='continent-filter',
                                multi=True,
                                placeholder="Select continent(s)...",
                                style={'fontSize': '13px'}
                            )
                        ], width=3),
                        
                        dbc.Col([
                            html.Label('Classification Level 1', className="text-secondary font-semibold",
                                     style={'fontSize': '13px', 'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='classification-filter',
                                multi=True,
                                placeholder="Select classification(s)...",
                                style={'fontSize': '13px'}
                            )
                        ], width=3),
                        
                        dbc.Col([
                            html.Label('Basin', className="text-secondary font-semibold",
                                     style={'fontSize': '13px', 'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='basin-filter',
                                multi=True,
                                placeholder="Select basin(s)...",
                                style={'fontSize': '13px'}
                            )
                        ], width=3),
                        
                        dbc.Col([
                            html.Div([
                                html.Label(' ', style={'display': 'block', 'marginBottom': '5px'}),
                                dbc.Button(
                                    'Clear Filters',
                                    id='clear-filters-btn',
                                    color='secondary',
                                    outline=True,
                                    size='sm',
                                    style={'width': '100%', 'marginTop': '20px'}
                                )
                            ])
                        ], width=3)
                    ])
                ])
            ])
        ], className="shadow-sm mb-4")
    ], style={'padding': '0px 20px'}),
    
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H5('Country Mapping Data', className="text-primary font-bold", 
                           style={'marginBottom': '15px'})
                ], style={'marginBottom': '15px'}),
                
                html.Div(id='country-mappings-table-container')
            ])
        ], className="shadow-sm mb-4")
    ], style={'padding': '0px 20px'}),
    
    # World Map Section
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H5('Geographic Visualization', className="text-primary font-bold", 
                           style={'marginBottom': '15px'})
                ], style={'marginBottom': '15px'}),
                
                dbc.Row([
                    dbc.Col([
                        html.Label('Select Category', className="text-secondary font-semibold", 
                                 style={'fontSize': '13px', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='map-category-dropdown',
                            options=[
                                {'label': 'Continent', 'value': 'continent'},
                                {'label': 'Subcontinent', 'value': 'subcontinent'},
                                {'label': 'Basin', 'value': 'basin'},
                                {'label': 'Classification Level 1', 'value': 'country_classification_level1'},
                                {'label': 'Classification', 'value': 'country_classification'},
                                {'label': 'Shipping Region', 'value': 'shipping_region'}
                            ],
                            value='continent',
                            style={'fontSize': '13px'},
                            clearable=False
                        )
                    ], width=3)
                ], className="mb-3"),
                
                dcc.Loading(
                    id="map-loading",
                    children=[
                        html.Div(id='world-map-visualization')
                    ],
                    type="default"
                )
            ])
        ], className="shadow-sm mb-4")
    ], style={'padding': '0px 20px', 'marginBottom': '30px'}),
])

@callback(
    Output('country-mappings-data-store', 'data'),
    [Input('country-mappings-load-trigger', 'n_intervals'),
     Input('global-refresh-button', 'n_clicks')]
)
def load_data(_, n_clicks):
    """Load country mappings data on page load or when refresh button is clicked"""
    df = fetch_country_mappings_data(engine)
    if not df.empty:
        return df.to_dict('records')
    return []

@callback(
    [Output('continent-filter', 'options'),
     Output('classification-filter', 'options'),
     Output('basin-filter', 'options'),
     Output('summary-cards-container', 'children')],
    Input('country-mappings-data-store', 'data')
)
def update_filters_and_visuals(data):
    """Update filter options and visual elements based on loaded data"""
    if not data:
        return [], [], [], html.Div("Loading...")
    
    df = pd.DataFrame(data)
    
    continent_options = [{'label': c, 'value': c} for c in sorted(df['continent'].unique()) if c]
    classification_options = [{'label': c, 'value': c} for c in sorted(df['country_classification_level1'].unique()) if c]
    basin_options = [{'label': b, 'value': b} for b in sorted(df['basin'].unique()) if b]
    
    summary_cards = create_summary_cards(df)
    
    return continent_options, classification_options, basin_options, summary_cards

@callback(
    Output('country-mappings-table-container', 'children'),
    [Input('country-mappings-data-store', 'data'),
     Input('continent-filter', 'value'),
     Input('classification-filter', 'value'),
     Input('basin-filter', 'value')]
)
def update_table(data, continents, classifications, basins):
    """Update the data table based on filters"""
    if not data:
        return html.Div("No data available")
    
    df = pd.DataFrame(data)
    
    if continents:
        df = df[df['continent'].isin(continents)]
    if classifications:
        df = df[df['country_classification_level1'].isin(classifications)]
    if basins:
        df = df[df['basin'].isin(basins)]
    
    # Don't show ISO3 column in the table
    display_df = df.drop(columns=['iso3'], errors='ignore')
    
    table_config = StandardTableStyleManager.get_base_datatable_config()
    
    return dash_table.DataTable(
        id='country-mappings-table',
        columns=[
            {"name": "Country", "id": "country_name"},
            {"name": "Continent", "id": "continent"},
            {"name": "Subcontinent", "id": "subcontinent"},
            {"name": "Basin", "id": "basin"},
            {"name": "Classification Level 1", "id": "country_classification_level1"},
            {"name": "Classification", "id": "country_classification"},
            {"name": "Shipping Region", "id": "shipping_region"}
        ],
        data=display_df.to_dict('records'),
        sort_action='native',
        filter_action='native',
        page_action='native' if len(df) > 50 else 'none',
        page_size=50,
        export_format='xlsx',
        export_headers='display',
        style_table=table_config['style_table'],
        style_header=table_config['style_header'],
        style_cell=table_config['style_cell'],
        style_data_conditional=table_config['style_data_conditional'],
        style_cell_conditional=[
            {
                'if': {'column_id': 'country_name'},
                'textAlign': 'left',
                'fontWeight': 'bold'
            }
        ]
    )

@callback(
    [Output('continent-filter', 'value'),
     Output('classification-filter', 'value'),
     Output('basin-filter', 'value')],
    Input('clear-filters-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_filters(_):
    """Clear all filters"""
    return None, None, None

# Callback to handle tooltips for the leaflet map
@app.callback(
    Output("tooltip", "children"),
    [Input("geojson-layer", "hover_feature")],
    prevent_initial_call=True
)
def update_tooltip(feature):
    """Display country information on hover"""
    if not feature:
        return None
    
    if feature and 'properties' in feature:
        # Get tooltip text from properties
        tooltip_text = feature['properties'].get('tooltip', 'No data')
        # Return formatted tooltip
        return tooltip_text

@callback(
    Output('world-map-visualization', 'children'),
    [Input('country-mappings-data-store', 'data'),
     Input('map-category-dropdown', 'value'),
     Input('continent-filter', 'value'),
     Input('classification-filter', 'value'),
     Input('basin-filter', 'value')]
)
def update_world_map(data, category, continents, classifications, basins):
    """Update the world map based on selected category and filters"""
    if not data or not category:
        return html.Div("No data available for map", style={'padding': '20px', 'textAlign': 'center'})
    
    df = pd.DataFrame(data)
    
    # Apply filters
    if continents:
        df = df[df['continent'].isin(continents)]
    if classifications:
        df = df[df['country_classification_level1'].isin(classifications)]
    if basins:
        df = df[df['basin'].isin(basins)]
    
    # Create the leaflet map
    return create_world_map(df, category)