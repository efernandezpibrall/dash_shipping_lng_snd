# Alternative Map Solution to Fix France/French Guiana Issue

## The Problem
Plotly's choropleth has a known bug where French overseas territories (French Guiana, Martinique, etc.) are mapped as France when using country-level data. This is a fundamental issue with Plotly's built-in country geometries.

## Recommended Solution: Dash-Leaflet

### Installation
```bash
pip install dash-leaflet
```

### Implementation
Replace the current map implementation with dash-leaflet:

```python
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash import html
import json

def create_world_map_leaflet(df, category='continent'):
    """Create a world map using dash-leaflet to avoid Plotly issues"""
    
    # Get unique values for color mapping
    unique_values = df[category].unique()
    
    # Create color mapping
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#F8B739', '#C0392B',
              '#8E44AD', '#3498DB', '#1ABC9C', '#F39C12', '#D35400']
    
    color_map = dict(zip(unique_values, colors[:len(unique_values)]))
    
    # Create markers for each country
    markers = []
    for _, row in df.iterrows():
        # Country coordinates (you'll need to add these to your database or use a geocoding service)
        coords = get_country_coords(row['country_name'])
        if coords:
            lat, lon = coords
            markers.append(
                dl.CircleMarker(
                    center=[lat, lon],
                    radius=5,
                    children=[
                        dl.Tooltip(f"{row['country_name']}: {row[category]}")
                    ],
                    color=color_map.get(row[category], 'gray'),
                    fill=True,
                    fillOpacity=0.7
                )
            )
    
    # Create the map
    return dl.Map(
        children=[
            dl.TileLayer(),  # Base map tiles
            *markers  # Country markers
        ],
        style={'width': '100%', 'height': '600px'},
        center=[20, 0],
        zoom=2
    )
```

Then in your layout, replace:
```python
dcc.Graph(id='world-map-visualization')
```

With:
```python
html.Div(id='world-map-visualization')
```

And update the callback to return the leaflet map instead of a plotly figure.

## Alternative Solution 2: Use GeoJSON with Plotly

Download a custom GeoJSON file that has France and French Guiana as separate entities:

```python
import json
import requests

def create_world_map_geojson(df, category='continent'):
    """Use custom GeoJSON to properly separate territories"""
    
    # Download or load custom GeoJSON
    # This GeoJSON should have France and French Guiana as separate features
    with open('custom_world.geojson', 'r') as f:
        geojson = json.load(f)
    
    # Create the map using the custom GeoJSON
    fig = px.choropleth(
        df,
        geojson=geojson,
        locations='country_name',
        featureidkey='properties.name',  # Match with GeoJSON property
        color=category,
        projection="natural earth"
    )
    
    return fig
```

## Alternative Solution 3: Use Plotly's Scattergeo (Current Workaround)

This is what we implemented earlier - using markers instead of filled regions. While not ideal, it completely avoids the France/French Guiana confusion.

## Recommended Action

1. **Best Solution**: Install and use `dash-leaflet` for proper country mapping
2. **Good Solution**: Obtain a custom GeoJSON file with properly separated territories
3. **Current Workaround**: Continue with the scattergeo approach if you can't install new packages

The dash-leaflet solution is the most robust and doesn't have any of Plotly's choropleth issues.