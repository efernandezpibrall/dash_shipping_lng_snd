# utils/table_styles.py
"""
Unified table styling system for standardized DataTable appearance across all pages.
Based on supply.py styling standards with McKinsey blue (#2E86C1) headers.
"""

import pandas as pd

# ========================================
# STANDARD TABLE STYLES
# ========================================

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
            'style_data_conditional': [
                STANDARD_CONDITIONAL_STYLES['alternating_rows']
            ]
        }
    

def format_numeric_table_cell(value, decimals: int = 2) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, float)):
        return f"{float(value):.{decimals}f}"

    return str(value)


def format_table_cell_value_1dp(value) -> str:
    return format_numeric_table_cell(value, decimals=1)


def format_table_cell_value_2dp(value) -> str:
    return format_numeric_table_cell(value, decimals=2)


def build_responsive_column_styles(
    df: pd.DataFrame,
    *,
    value_formatter=format_table_cell_value_2dp,
) -> list[dict]:
    column_styles = []
    column_weights = {}
    column_min_widths = {}

    for column_name in df.columns:
        header_length = len(str(column_name))
        value_lengths = df[column_name].map(value_formatter).map(len)
        max_length = max([header_length] + value_lengths.tolist()) if not df.empty else header_length

        if column_name == "Month":
            column_weights[column_name] = max(8, min(max_length, 12))
            column_min_widths[column_name] = 92
        elif column_name == "Total MMTPA":
            column_weights[column_name] = max(8, min(max_length, 14))
            column_min_widths[column_name] = 96
        else:
            column_weights[column_name] = max(6, min(max_length, 18))
            column_min_widths[column_name] = 72

    total_weight = sum(column_weights.values()) or 1

    for column_name in df.columns:
        width_pct = column_weights[column_name] / total_weight * 100
        style_entry = {
            "if": {"column_id": column_name},
            "minWidth": f"{column_min_widths[column_name]}px",
            "width": f"{width_pct:.2f}%",
        }

        if column_name == "Month":
            style_entry["textAlign"] = "left"

        column_styles.append(style_entry)

    return column_styles


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
