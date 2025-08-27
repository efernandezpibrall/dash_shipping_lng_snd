# utils/table_styles.py
"""
Unified table styling system for standardized DataTable appearance across all pages.
Based on supply.py styling standards with McKinsey blue (#2E86C1) headers.
"""

# ========================================
# STANDARD TABLE STYLES
# ========================================

# Standard McKinsey table header style - matching supply.py reference
STANDARD_TABLE_HEADER = {
    'backgroundColor': '#2E86C1',     # McKinsey blue - primary brand color
    'color': 'white',                 # White text for contrast
    'fontWeight': 'bold',             # Bold headers for prominence
    'fontSize': '12px',               # Consistent font size
    'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',  # Updated to Inter per dash_style.md
    'padding': '8px',                 # Standard padding
    'whiteSpace': 'pre-wrap',
    'lineHeight': '1.2',
    'textAlign': 'center',
    'border': '1px solid #1B4F72'    # Darker border for definition
}

# Standard cell style for consistency
STANDARD_TABLE_CELL = {
    'textAlign': 'center',
    'padding': '8px',
    'fontSize': '12px',
    'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',  # Updated to Inter per dash_style.md
    'backgroundColor': 'white',
    'minWidth': '60px',
    'maxWidth': '180px',
    'whiteSpace': 'normal',
    'height': 'auto'
}

# Common conditional styling patterns
STANDARD_CONDITIONAL_STYLES = {
    'alternating_rows': {
        'if': {'row_index': 'odd'}, 
        'backgroundColor': '#f8f9fa'
    },
    'weekend_holiday': {
        'if': {'filter_query': '{is_weekend_holiday} = True'},
        'backgroundColor': '#f3f4f6',
        'fontStyle': 'italic'
    },
    'header_row': {
        'if': {'row_index': 0},
        'fontWeight': 'bold',
        'backgroundColor': '#f3f4f6',
        'color': '#1f2937'
    }
}

# ========================================
# STANDARDIZED TABLE STYLE MANAGER
# ========================================

class StandardTableStyleManager:
    """Centralized table styling system for consistent appearance across all pages"""
    
    @staticmethod
    def get_base_datatable_config():
        """Base DataTable configuration used across all pages"""
        return {
            'style_table': {
                'overflowX': 'auto',
                'overflowY': 'auto',
                'margin': '0 auto'
            },
            'style_header': STANDARD_TABLE_HEADER,
            'style_cell': STANDARD_TABLE_CELL,
            'style_data_conditional': [
                STANDARD_CONDITIONAL_STYLES['alternating_rows']
            ]
        }
    
    @staticmethod
    def get_mckinsey_delta_table_config():
        """Configuration for McKinsey-style delta tables with heat map colors"""
        base_config = StandardTableStyleManager.get_base_datatable_config()
        
        # Enhanced cell style for delta tables
        base_config['style_cell'].update({
            'fontSize': '12px',
            'padding': '6px 4px',  # More compact for data-heavy tables
            'textAlign': 'center',
            'border': '1px solid #e5e7eb'
        })
        
        return base_config

# ========================================
# LEGACY COMPATIBILITY FUNCTIONS
# ========================================

def get_standard_table_style():
    """
    Returns a complete standard table style configuration.
    Use this as the base for all DataTables for consistency.
    """
    return StandardTableStyleManager.get_base_datatable_config()

def get_compact_table_style():
    """
    Returns a compact version of the standard table style.
    Use this for tables with many columns or limited space.
    """
    compact_style = get_standard_table_style()
    compact_style['style_cell']['fontSize'] = '11px'
    compact_style['style_cell']['padding'] = '6px'
    compact_style['style_cell']['minWidth'] = '50px'
    compact_style['style_header']['fontSize'] = '11px'
    compact_style['style_header']['padding'] = '6px'
    return compact_style

# ========================================
# STANDARD COLOR PALETTE
# ========================================

# Export commonly used color values for consistency
TABLE_COLORS = {
    'primary': '#2E86C1',           # McKinsey blue
    'primary_dark': '#1B4F72',      # Darker McKinsey blue
    'primary_light': '#5DADE2',     # Lighter McKinsey blue
    'text_primary': '#1f2937',      # Dark gray for text
    'text_secondary': '#374151',    # Medium gray for secondary text
    'text_white': '#ffffff',        # White text
    'bg_light': '#f8f9fa',          # Light background
    'bg_lighter': '#f3f4f6',        # Lighter background
    'border_light': '#e5e7eb',      # Light borders
    'success': '#28a745',           # Success green
    'warning': '#ffc107',           # Warning yellow
    'danger': '#dc3545',            # Danger red
    'info': '#17a2b8'               # Info cyan
}