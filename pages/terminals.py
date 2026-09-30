from utils.database import engine

# Professional color palette (McKinsey-style) - matching original
PRIMARY_COLORS = {
    'US (Gulf Coast)': '#003A6C',  # Navy Blue
    'Canada (British Columbia)': '#00A3E0',  # Light Blue
    'Mexico': '#6BCABA',  # Teal
    'Argentina': '#90B23C',  # Green
    'Mauritania': '#FFC72C',  # Yellow
    'Senegal': '#FFC72C',  # Yellow
    'Gabon': '#F58220',  # Orange
    'Congo': '#E03C31',  # Red
    'Nigeria': '#7F3F98',  # Purple
    'Qatar': '#A21F5A',  # Burgundy
    'Malaysia': '#005EB8',  # Royal Blue
    'Indonesia': '#00B5E2',  # Sky Blue
    'Australia': '#78BE20',  # Lime Green
    'United States': '#003A6C',  # Navy Blue
    'Canada': '#00A3E0',  # Light Blue
}


def convert_to_mcmd(capacity_mtpa):
    """Convert MTPA to Mcm/d using formula: MTPA * 1.36 / 365 * 1000"""
    return capacity_mtpa * 1.36 / 365 * 1000


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple for rgba() formatting."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
