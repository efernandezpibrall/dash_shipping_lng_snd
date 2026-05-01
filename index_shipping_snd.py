# index.py
from dash import html, dcc, clientside_callback, ClientsideFunction, callback, callback_context
from dash.dependencies import Input, Output, State
from app import app
import pages.shipping_balance
import pages.supply
import pages.demand
import pages.market_balance
import pages.exporter_detail
import pages.importer_detail
import pages.exporters
import pages.importers
import pages.country_mappings
import pages.plant_names_mapping
import pages.train_names_mapping
import pages.contracts
import pages.capacity
import pages.production
import pages.terminal_adjustments

import pandas as pd
import configparser
import os
from sqlalchemy import create_engine
import subprocess
import threading
import dash_bootstrap_components as dbc
import sys

############################################ postgres sql connection ###################################################
try:
    # Get the directory where your script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the directory containing config.ini
    config_dir = os.path.abspath(os.path.join(script_dir, '..'))  # Go up one level
    CONFIG_FILE_PATH = os.path.join(config_dir, 'config.ini')
except:
    CONFIG_FILE_PATH = 'config.ini'  # Assumes it's in the same directory or the path it is detected

# --- Load Configuration from INI File ---
config_reader = configparser.ConfigParser(interpolation=None)
config_reader.read(CONFIG_FILE_PATH)

# Read values from the ini file sections
DB_CONNECTION_STRING = config_reader.get('DATABASE', 'CONNECTION_STRING', fallback=None)
DB_SCHEMA = config_reader.get('DATABASE', 'SCHEMA', fallback=None)

# --- Essential Variable Checks ---
if not DB_CONNECTION_STRING:
    raise ValueError(f"Missing DATABASE CONNECTION_STRING in {CONFIG_FILE_PATH}")

# create engine
engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)

# Professional Navigation Bar with Option B Blue System
nav_links = html.Header([
    html.Div([
        # Primary navigation section
        html.Nav([
            # Navigation group
            html.Div([
                dcc.Link('Shipping Balance', href='/shipping_balance', id='nav-shipping-balance', className='nav-link-secondary'),
                dcc.Link('Supply', href='/supply', id='nav-supply', className='nav-link-secondary'),
                dcc.Link('Demand', href='/demand', id='nav-demand', className='nav-link-secondary'),
                dcc.Link('Market Balance', href='/market_balance', id='nav-market-balance', className='nav-link-secondary'),
                dcc.Link('Exporters', href='/exporters', id='nav-exporters', className='nav-link-secondary'),
                dcc.Link('Importers', href='/importers', id='nav-importers', className='nav-link-secondary'),
                dcc.Link('Exporter Detail', href='/exporter_detail', id='nav-exporter-detail', className='nav-link-secondary'),
                dcc.Link('Importer Detail', href='/importer_detail', id='nav-importer-detail', className='nav-link-secondary'),
                dcc.Link('Contracts', href='/contracts', id='nav-contracts', className='nav-link-secondary'),
                dcc.Link('Production', href='/production', id='nav-production', className='nav-link-secondary'),
                dcc.Link('Capacity', href='/capacity', id='nav-capacity', className='nav-link-secondary'),
                dcc.Link('Mappings', href='/mappings', id='nav-mappings', className='nav-link-secondary'),
            ], className='nav-group-secondary')
        ], className='main-navigation'),
        
        # Professional Controls Section
        html.Div([
            # Professional refresh button
            html.Button(
                'Refresh Data', 
                id='global-refresh-button', 
                n_clicks=0,
                className='btn-refresh'
            ),
        ], className='top-bar-controls')
    ], className='top-bar-content')
], className='top-bar-header')

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    nav_links,
    html.Div(id='page-content'),
])


# Callback to handle page routing
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/' or pathname == '/shipping_balance':
        return pages.shipping_balance.layout
    elif pathname == '/balance':
        return dcc.Location(pathname='/supply', id='redirect-supply-from-balance')
    elif pathname == '/supply':
        return pages.supply.layout
    elif pathname == '/demand':
        return pages.demand.layout
    elif pathname == '/market_balance':
        return pages.market_balance.layout
    elif pathname == '/exporters':
        return pages.exporters.layout
    elif pathname == '/importers':
        return pages.importers.layout
    elif pathname == '/exporter_detail':
        return pages.exporter_detail.layout
    elif pathname == '/importer_detail':
        return pages.importer_detail.layout
    elif pathname == '/contracts':
        return pages.contracts.layout()
    elif pathname == '/terminals':
        return dcc.Location(pathname='/capacity', id='redirect-capacity-from-terminals')
    elif pathname == '/production':
        return pages.production.layout
    elif pathname == '/capacity':
        return pages.capacity.layout
    elif pathname == '/terminal_adjustments':
        return pages.terminal_adjustments.layout
    elif pathname == '/mappings':
        return pages.country_mappings.layout
    elif pathname == '/country_mappings':
        return pages.country_mappings.layout
    elif pathname == '/plant_names_mapping':
        return pages.plant_names_mapping.layout
    elif pathname == '/train_names_mapping':
        return pages.train_names_mapping.layout
    else:
        return '404 - Page not found'

# Enhanced clientside callback to update page title and navigation active states
app.clientside_callback(
    """
    function(pathname) {
        // Update page title
        if (pathname === '/' || pathname === '/shipping_balance') {
            document.title = 'LNG Shipping - Shipping Balance';
        } else if (pathname === '/balance' || pathname === '/supply') {
            document.title = 'LNG Shipping - Supply';
        } else if (pathname === '/demand') {
            document.title = 'LNG Shipping - Demand';
        } else if (pathname === '/market_balance') {
            document.title = 'LNG Shipping - Market Balance';
        } else if (pathname === '/exporters') {
            document.title = 'LNG Shipping - Exporters';
        } else if (pathname === '/importers') {
            document.title = 'LNG Shipping - Importers';
        } else if (pathname === '/exporter_detail') {
            document.title = 'LNG Shipping - Exporter Detail';
        } else if (pathname === '/importer_detail') {
            document.title = 'LNG Shipping - Importer Detail';
        } else if (pathname === '/contracts') {
            document.title = 'LNG Shipping - Contracts';
        } else if (pathname === '/terminals') {
            document.title = 'LNG Shipping - Capacity';
        } else if (pathname === '/production') {
            document.title = 'LNG Shipping - Production';
        } else if (pathname === '/capacity') {
            document.title = 'LNG Shipping - Capacity';
        } else if (pathname === '/terminal_adjustments') {
            document.title = 'LNG Shipping - Terminal Adjustments';
        } else if (pathname === '/mappings') {
            document.title = 'LNG Shipping - Country Mappings';
        } else if (pathname === '/country_mappings') {
            document.title = 'LNG Shipping - Country Mappings';
        } else if (pathname === '/plant_names_mapping') {
            document.title = 'LNG Shipping - Plant Mapping';
        } else if (pathname === '/train_names_mapping') {
            document.title = 'LNG Shipping - Train Mapping';
        } else {
            document.title = 'LNG Shipping - Page Not Found';
        }
        
        // Update navigation active states
        const navLinks = document.querySelectorAll('.nav-link-primary, .nav-link-secondary');
        navLinks.forEach(link => {
            link.classList.remove('active');
        });
        
        // Add active class to current page link
        let activeNavId = '';
        if (pathname === '/' || pathname === '/shipping_balance') {
            activeNavId = 'nav-shipping-balance';
        } else if (pathname === '/balance' || pathname === '/supply') {
            activeNavId = 'nav-supply';
        } else if (pathname === '/demand') {
            activeNavId = 'nav-demand';
        } else if (pathname === '/market_balance') {
            activeNavId = 'nav-market-balance';
        } else if (pathname === '/exporters') {
            activeNavId = 'nav-exporters';
        } else if (pathname === '/importers') {
            activeNavId = 'nav-importers';
        } else if (pathname === '/exporter_detail') {
            activeNavId = 'nav-exporter-detail';
        } else if (pathname === '/importer_detail') {
            activeNavId = 'nav-importer-detail';
        } else if (pathname === '/contracts') {
            activeNavId = 'nav-contracts';
        } else if (pathname === '/terminals') {
            activeNavId = 'nav-capacity';
        } else if (pathname === '/production') {
            activeNavId = 'nav-production';
        } else if (pathname === '/capacity') {
            activeNavId = 'nav-capacity';
        } else if (
            pathname === '/mappings' ||
            pathname === '/country_mappings' ||
            pathname === '/plant_names_mapping' ||
            pathname === '/train_names_mapping'
        ) {
            activeNavId = 'nav-mappings';
        }
        
        if (activeNavId) {
            const activeLink = document.getElementById(activeNavId);
            if (activeLink) {
                activeLink.classList.add('active');
            }
        }
        
        return {};
    }
    """,
    Output('page-content', 'style'),  # Dummy output
    Input('url', 'pathname')
)


if __name__ == '__main__':
    app.run(debug=True, port=8067)
